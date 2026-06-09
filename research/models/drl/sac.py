"""
Soft Actor-Critic (Discrete) for trading (ADR-038, ADR-039).

Implements the discrete action SAC variant from Christodoulou (2019),
adapted to the {SELL=0, HOLD=1, BUY=2} action space.

Components
----------
SACConfig            : Frozen hyperparameter dataclass.
TradingQNetwork      : Single Q-network Q(s, a) for all actions simultaneously.
TradingDiscreteActor : Policy network π(a|s) returning a probability distribution.
SACTrainer           : Full SAC training loop — twin critics, soft target updates,
                       automatic entropy tuning, off-policy replay buffer.

Architecture
------------
Actor  : TradingResMLP → Linear → softmax → Categorical
Q1, Q2 : TradingResMLP → Linear → Q(s, ·)  (separate networks)
Q1_tgt, Q2_tgt : soft-updated copies of Q1, Q2 (no grad)

Key design decisions
--------------------
- **Twin critics** (Fujimoto et al. 2018): take min(Q1, Q2) for the Bellman
  target, reducing overestimation bias without requiring a separate value net.
- **Soft target update** (τ = 0.005): smoother than DQN hard copy; θ_tgt ← τθ + (1-τ)θ_tgt.
- **Automatic entropy tuning** (Haarnoja et al. 2018): learns α so that
  E[H(π)] ≈ target_entropy. Target entropy defaults to -log(1/n_actions)·0.98,
  keeping the policy close to uniform in early training.
- **Discrete value backup** (Christodoulou 2019):
    V(s) = Σ_a π(a|s) · [min(Q1,Q2)(s,a) − α·log π(a|s)]
  No reparameterization trick needed — entropy is computed analytically.
- **Shared ReplayBuffer with DQN**: same Transition namedtuple and interface,
  making it easy to swap DQN ↔ SAC in the experiment config.
- **Separate backbones** for actor and both critics: gradient interference
  between policy and value updates is avoided at the cost of more parameters.
  This is the standard SAC practice (vs. A3C-style shared backbone for PPO).

Promotion gate (ADR-039)
------------------------
DSR(SAC-agent) > DSR(PPO-agent) > DSR(XGBoost-baseline) to promote to production.

References
----------
Haarnoja et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy DRL.
Haarnoja et al. (2018). Soft Actor-Critic Algorithms and Applications.
Christodoulou, P. (2019). Soft Actor-Critic for Discrete Action Settings.
Fujimoto et al. (2018). Addressing Function Approximation Error in Actor-Critic. (TD3)
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from models.drl.backbone import TradingResMLP, init_weights
from models.drl.dqn import ReplayBuffer

if TYPE_CHECKING:
    import gymnasium as gym

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SACConfig:
    """
    Hyperparameters for Discrete SAC.

    Parameters
    ----------
    lr_actor : float
        Actor Adam learning rate.
    lr_critic : float
        Critic Adam learning rate.
    lr_alpha : float
        Entropy coefficient Adam learning rate.
    gamma : float
        Discount factor.
    tau : float
        Soft target update coefficient (0.005 is standard).
    alpha_init : float
        Initial entropy coefficient. Overridden quickly by auto-tuning.
    target_entropy_ratio : float
        target_entropy = -log(1/n_actions) * target_entropy_ratio.
        0.98 keeps policy close to uniform initially.
    batch_size : int
        Minibatch size drawn from replay buffer.
    buffer_capacity : int
        Replay buffer maximum size.
    min_buffer : int
        Minimum transitions before training starts.
    grad_clip : float
        Max gradient norm for actor + critics (0 = disabled).
    update_every : int
        Perform one gradient update every N environment steps.
    n_updates : int
        Number of gradient updates per ``update_every`` trigger.
    n_actions : int
        Number of discrete actions.
    obs_dim : int
        Observation dimension.
    hidden_dim : int
        Backbone hidden size.
    n_blocks : int
        Number of ResBlocks in each backbone.
    device : str
        Torch device string.
    """

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha_init: float = 1.0
    target_entropy_ratio: float = 0.98
    batch_size: int = 256
    buffer_capacity: int = 100_000
    min_buffer: int = 1_000
    grad_clip: float = 1.0
    update_every: int = 1
    n_updates: int = 1
    n_actions: int = 3
    obs_dim: int = 42
    hidden_dim: int = 256
    n_blocks: int = 3
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Network components
# ---------------------------------------------------------------------------


class TradingQNetwork(nn.Module):
    """
    Q-network Q(s, a) for all discrete actions simultaneously.

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    hidden_dim : int
        Backbone hidden size.
    n_blocks : int
        Number of residual blocks.
    n_actions : int
        Number of discrete actions.

    Examples
    --------
    >>> q = TradingQNetwork()
    >>> q(torch.randn(4, 42)).shape
    torch.Size([4, 3])
    """

    def __init__(
        self,
        obs_dim: int = 42,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        n_actions: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = TradingResMLP(obs_dim, hidden_dim, n_blocks)
        self.q_head = nn.Linear(hidden_dim, n_actions)
        # Small gain on q_head: Q-values should be near zero initially
        nn.init.orthogonal_(self.q_head.weight, gain=0.01)
        nn.init.zeros_(self.q_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : torch.Tensor
            Shape ``(batch, obs_dim)``.

        Returns
        -------
        torch.Tensor
            Q-values shape ``(batch, n_actions)``.
        """
        return self.q_head(self.backbone(obs))


class TradingDiscreteActor(nn.Module):
    """
    Stochastic actor π(a|s) for discrete action spaces.

    Outputs action probabilities (softmax) and their log-probabilities,
    with a small epsilon to prevent log(0) for unvisited actions.

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    hidden_dim : int
        Backbone hidden size.
    n_blocks : int
        Number of residual blocks.
    n_actions : int
        Number of discrete actions.

    Examples
    --------
    >>> actor = TradingDiscreteActor()
    >>> obs = torch.randn(4, 42)
    >>> probs, log_probs = actor.get_probs(obs)
    >>> probs.shape, log_probs.shape
    (torch.Size([4, 3]), torch.Size([4, 3]))
    """

    _LOG_EPS: float = 1e-8  # prevent log(0)

    def __init__(
        self,
        obs_dim: int = 42,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        n_actions: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = TradingResMLP(obs_dim, hidden_dim, n_blocks)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        # Very small gain: nearly-uniform initial policy
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return raw logits ``(batch, n_actions)``."""
        return self.policy_head(self.backbone(obs))

    def get_probs(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action probabilities and their log-probabilities.

        Parameters
        ----------
        obs : torch.Tensor
            Shape ``(batch, obs_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(probs, log_probs)`` — both shape ``(batch, n_actions)``.
            log_probs are clamped to avoid -inf for zero-probability actions.
        """
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + self._LOG_EPS)
        return probs, log_probs

    def sample(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the current policy.

        Parameters
        ----------
        obs : torch.Tensor
            Shape ``(batch, obs_dim)`` or ``(obs_dim,)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(action, log_prob_action, entropy)`` — scalars or ``(batch,)``.
            entropy = -Σ_a π(a|s) log π(a|s) (analytic, full distribution).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        probs, log_probs = self.get_probs(obs)
        dist = Categorical(probs=probs)
        action = dist.sample()
        log_prob_action = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return action, log_prob_action, entropy

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> int:
        """
        Select a single action for environment interaction.

        Parameters
        ----------
        obs : torch.Tensor
            Single observation ``(obs_dim,)`` or ``(1, obs_dim)``.
        deterministic : bool
            If True, return argmax action (evaluation mode).

        Returns
        -------
        int
            Selected action index.
        """
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            probs, _ = self.get_probs(obs)
            if deterministic:
                return int(probs.argmax(dim=-1).item())
            return int(Categorical(probs=probs).sample().item())


# ---------------------------------------------------------------------------
# SAC Trainer
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SACUpdateStats:
    """Metrics from one SAC gradient update step."""

    step: int
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    mean_entropy: float


class SACTrainer:
    """
    Discrete Soft Actor-Critic training loop.

    Uses twin Q-networks with soft target updates and automatic entropy
    coefficient tuning.

    Parameters
    ----------
    actor : TradingDiscreteActor
        The policy network being optimised.
    config : SACConfig
        Training hyperparameters.

    Examples
    --------
    >>> actor = TradingDiscreteActor()
    >>> trainer = SACTrainer(actor)
    >>> # history = trainer.train(env, n_steps=100_000)
    """

    def __init__(
        self,
        actor: TradingDiscreteActor,
        config: SACConfig | None = None,
    ) -> None:
        self.config = config or SACConfig()
        self.device = torch.device(self.config.device)

        self.actor = actor.to(self.device)

        # Twin critics + their soft-updated targets
        self.q1 = TradingQNetwork(
            self.config.obs_dim, self.config.hidden_dim,
            self.config.n_blocks, self.config.n_actions
        ).to(self.device)
        self.q2 = TradingQNetwork(
            self.config.obs_dim, self.config.hidden_dim,
            self.config.n_blocks, self.config.n_actions
        ).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        for net in (self.q1_target, self.q2_target):
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)

        # Optimisers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_opt = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.config.lr_critic,
        )

        # Automatic entropy tuning (Haarnoja et al. 2018 Appendix H)
        self.target_entropy = float(
            -np.log(1.0 / self.config.n_actions) * self.config.target_entropy_ratio
        )
        self.log_alpha = nn.Parameter(
            torch.tensor([np.log(self.config.alpha_init)], dtype=torch.float32, device=self.device)
        )
        self.alpha_opt = optim.Adam([self.log_alpha], lr=self.config.lr_alpha)

        # Replay buffer (shared interface with DQN)
        self.buffer = ReplayBuffer(self.config.buffer_capacity)
        self._total_steps = 0

    @property
    def alpha(self) -> float:
        """Current entropy coefficient (exp of learned log_alpha)."""
        return float(self.log_alpha.exp().item())

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def train_step(self) -> SACUpdateStats | None:
        """
        Sample a minibatch and perform one SAC gradient step.

        Updates critics → actor → alpha in sequence.

        Returns
        -------
        SACUpdateStats | None
            Update metrics, or None if buffer is below min_buffer.
        """
        if len(self.buffer) < self.config.min_buffer:
            return None

        transitions = self.buffer.sample(self.config.batch_size)
        states = torch.stack([t.state for t in transitions]).to(self.device)
        actions = torch.tensor(
            [t.action for t in transitions], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [t.reward for t in transitions], dtype=torch.float32, device=self.device
        )
        next_states = torch.stack([t.next_state for t in transitions]).to(self.device)
        dones = torch.tensor(
            [t.done for t in transitions], dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            # Discrete SAC target (Christodoulou 2019):
            # V(s') = Σ_a π(a|s') [min(Q1,Q2)(s',a) - α log π(a|s')]
            next_probs, next_log_probs = self.actor.get_probs(next_states)
            q1_next = self.q1_target(next_states)
            q2_next = self.q2_target(next_states)
            min_q_next = torch.min(q1_next, q2_next)
            v_next = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + self.config.gamma * v_next * (1.0 - dones)

        # Critic update
        q1_pred = self.q1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_pred = self.q2(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        critic_loss = F.smooth_l1_loss(q1_pred, target_q) + F.smooth_l1_loss(q2_pred, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                self.config.grad_clip,
            )
        self.critic_opt.step()

        # Actor update
        # J(π) = Σ_a π(a|s)[α log π(a|s) - min(Q1,Q2)(s,a)]  minimise
        probs, log_probs = self.actor.get_probs(states)
        with torch.no_grad():
            min_q = torch.min(self.q1(states), self.q2(states))
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=-1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_opt.step()

        # Entropy coefficient update
        # J(α) = E[-α log π(a|s) - α * target_entropy]
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1).mean()
        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy.detach()))

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft target update: θ_tgt ← τθ + (1-τ)θ_tgt
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return SACUpdateStats(
            step=self._total_steps,
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            alpha=self.alpha,
            mean_entropy=float(entropy.item()),
        )

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """θ_tgt ← τ·θ_src + (1-τ)·θ_tgt"""
        tau = self.config.tau
        with torch.no_grad():
            for p_src, p_tgt in zip(source.parameters(), target.parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(tau * p_src.data)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        env: "gym.Env",
        n_steps: int = 100_000,
        checkpoint_dir: Path | None = None,
        checkpoint_every: int = 10_000,
        log_every: int = 1_000,
    ) -> list[SACUpdateStats]:
        """
        Full SAC training loop (step-based, off-policy).

        Parameters
        ----------
        env : gym.Env
            Gymnasium environment.
        n_steps : int
            Total environment steps to collect.
        checkpoint_dir : Path | None
            If set, saves checkpoints here.
        checkpoint_every : int
            Save checkpoint every N steps.
        log_every : int
            Log summary every N steps.

        Returns
        -------
        list[SACUpdateStats]
            History of update stats (one entry per gradient step taken).
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history: list[SACUpdateStats] = []
        obs, _ = env.reset()
        current_obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        episode_reward = 0.0
        episode_rewards: list[float] = []

        for step in range(n_steps):
            self._total_steps = step

            # Collect one step
            self.actor.eval()
            with torch.no_grad():
                action = self.actor.select_action(current_obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)

            next_state = torch.tensor(next_obs, dtype=torch.float32)
            self.buffer.push(current_obs.cpu(), action, float(reward), next_state, done)

            current_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs, _ = env.reset()
                current_obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

            # Gradient updates
            if step % self.config.update_every == 0:
                self.actor.train()
                for _ in range(self.config.n_updates):
                    stats = self.train_step()
                    if stats is not None:
                        history.append(stats)

            # Logging
            if log_every > 0 and (step + 1) % log_every == 0 and history:
                last = history[-1]
                mean_ep = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0
                logger.info(
                    "step=%d critic=%.4f actor=%.4f alpha=%.4f ent=%.4f ep_reward=%.4f",
                    step + 1,
                    last.critic_loss,
                    last.actor_loss,
                    last.alpha,
                    last.mean_entropy,
                    mean_ep,
                )

            # Checkpoint
            if (
                checkpoint_dir is not None
                and checkpoint_every > 0
                and (step + 1) % checkpoint_every == 0
            ):
                self._save_checkpoint(checkpoint_dir, step + 1)

        return history

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, directory: Path, step: int) -> Path:
        path = directory / f"sac_step{step:07d}.pt"
        torch.save(
            {
                "step": step,
                "actor_state": self.actor.state_dict(),
                "q1_state": self.q1.state_dict(),
                "q2_state": self.q2.state_dict(),
                "q1_target_state": self.q1_target.state_dict(),
                "q2_target_state": self.q2_target.state_dict(),
                "actor_opt_state": self.actor_opt.state_dict(),
                "critic_opt_state": self.critic_opt.state_dict(),
                "alpha_opt_state": self.alpha_opt.state_dict(),
                "log_alpha": self.log_alpha.data,
                "config": dataclasses.asdict(self.config),
            },
            path,
        )
        logger.info("SAC checkpoint saved -> %s", path)
        return path

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        actor: TradingDiscreteActor | None = None,
        config: SACConfig | None = None,
    ) -> "SACTrainer":
        """
        Restore a SACTrainer from a checkpoint file.

        Parameters
        ----------
        path : Path
            Checkpoint file produced by ``_save_checkpoint``.
        actor : TradingDiscreteActor | None
            If None, a default actor is instantiated from config.
        config : SACConfig | None
            If None, config is loaded from the checkpoint.

        Returns
        -------
        SACTrainer
            Fully restored trainer.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if config is None:
            config = SACConfig(**ckpt["config"])
        if actor is None:
            actor = TradingDiscreteActor(
                obs_dim=config.obs_dim,
                hidden_dim=config.hidden_dim,
                n_blocks=config.n_blocks,
                n_actions=config.n_actions,
            )
        trainer = cls(actor, config)
        trainer.actor.load_state_dict(ckpt["actor_state"])
        trainer.q1.load_state_dict(ckpt["q1_state"])
        trainer.q2.load_state_dict(ckpt["q2_state"])
        trainer.q1_target.load_state_dict(ckpt["q1_target_state"])
        trainer.q2_target.load_state_dict(ckpt["q2_target_state"])
        trainer.actor_opt.load_state_dict(ckpt["actor_opt_state"])
        trainer.critic_opt.load_state_dict(ckpt["critic_opt_state"])
        trainer.alpha_opt.load_state_dict(ckpt["alpha_opt_state"])
        trainer.log_alpha.data.copy_(ckpt["log_alpha"])
        trainer._total_steps = ckpt["step"]
        return trainer
