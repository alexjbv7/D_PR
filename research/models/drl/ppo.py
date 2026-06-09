"""
Proximal Policy Optimization for trading (ADR-038, ADR-039).

Components
----------
RolloutBuffer    : On-policy storage for states, actions, rewards, values, log-probs.
PPOConfig        : Training hyperparameters as a frozen dataclass.
TradingActorCritic : Shared TradingResMLP backbone → actor (Categorical) + critic (V).
PPOTrainer       : GAE advantage estimation, PPO-clip objective, multi-epoch updates.

Design decisions
----------------
- Shared backbone between actor and critic reduces total parameters and
  encourages useful representations (Mnih et al. 2016 A3C).
- Separate learning rates are NOT used (single Adam over all params) —
  vf_coef controls the relative scale of value loss instead.
- GAE (Schulman et al. 2015) is used for advantage estimation.
  lambda_ = 0.95 is the empirically recommended value.
- Entropy bonus (ent_coef) prevents premature policy collapse.
- Advantages are normalised per minibatch (not globally) for stability.
- Value function loss uses Huber (smooth_l1) — consistent with DQNTrainer.
- No separate actor/critic gradient clipping — single clip_grad_norm.
- Policy ratio clip_eps = 0.2 is the standard PPO default.

Action space
------------
MVP: Discrete(3) — {SELL=0, HOLD=1, BUY=2}.
The actor head outputs logits → torch.distributions.Categorical.
Upgrading to continuous (Box) requires swapping the head for a
DiagGaussian head; backbone and trainer are unchanged.

References
----------
Schulman et al. (2017). Proximal Policy Optimization Algorithms.
Schulman et al. (2015). High-Dimensional Continuous Control Using GAE.
Mnih et al. (2016). Asynchronous Methods for DRL (A3C, shared backbone).
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from models.drl.backbone import TradingResMLP, init_weights

if TYPE_CHECKING:
    import gymnasium as gym

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    """
    Hyperparameters for PPO training.

    Parameters
    ----------
    lr : float
        Adam learning rate (shared actor + critic).
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE smoothing parameter (0 = TD(0), 1 = MC).
    clip_eps : float
        PPO clip ratio ε.
    vf_coef : float
        Relative weight of value function loss.
    ent_coef : float
        Entropy bonus coefficient (encourages exploration).
    n_epochs : int
        Number of optimisation epochs per rollout.
    n_steps : int
        Steps collected per rollout before an update.
    batch_size : int
        Minibatch size during optimisation epochs.
    grad_clip : float
        Max gradient norm (0 = disabled).
    target_kl : float | None
        Early-stop epoch if approx KL exceeds this value (None = disabled).
    n_actions : int
        Number of discrete actions.
    obs_dim : int
        Observation dimension.
    hidden_dim : int
        Backbone hidden size.
    n_blocks : int
        Number of ResBlocks in the backbone.
    device : str
        Torch device string.
    """

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    n_epochs: int = 10
    n_steps: int = 2_048
    batch_size: int = 64
    grad_clip: float = 0.5
    target_kl: float | None = 0.015
    n_actions: int = 3
    obs_dim: int = 42
    hidden_dim: int = 256
    n_blocks: int = 3
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class UpdateStats:
    """Metrics from a single PPO update cycle."""

    update: int
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    explained_variance: float
    n_episodes: int
    mean_episode_reward: float


class RolloutBuffer:
    """
    Fixed-size on-policy buffer for PPO rollout data.

    Stores exactly ``n_steps`` transitions, then is consumed by
    ``PPOTrainer.train_epoch``.

    Parameters
    ----------
    n_steps : int
        Number of steps per rollout.
    obs_dim : int
        Observation dimension.
    device : torch.device
        Where tensors live after ``finalize()``.
    """

    def __init__(self, n_steps: int, obs_dim: int, device: torch.device) -> None:
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.device = device
        self._ptr = 0
        self._full = False

        self.observations = torch.zeros(n_steps, obs_dim)
        self.actions = torch.zeros(n_steps, dtype=torch.long)
        self.rewards = torch.zeros(n_steps)
        self.values = torch.zeros(n_steps)
        self.log_probs = torch.zeros(n_steps)
        self.dones = torch.zeros(n_steps)
        # Filled in finalize()
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def push(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Store one transition."""
        i = self._ptr
        self.observations[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.dones[i] = float(done)
        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True

    @property
    def is_full(self) -> bool:
        return self._full

    def finalize(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """
        Compute GAE advantages and discounted returns in-place.

        Parameters
        ----------
        last_value : float
            Bootstrap value V(s_{T+1}) from the critic.
        gamma : float
            Discount factor.
        gae_lambda : float
            GAE lambda.
        """
        advantages = torch.zeros(self.n_steps)
        last_gae = 0.0
        next_val = float(last_value)

        for t in reversed(range(self.n_steps)):
            mask = 1.0 - float(self.dones[t])
            delta = float(self.rewards[t]) + gamma * next_val * mask - float(self.values[t])
            last_gae = delta + gamma * gae_lambda * mask * last_gae
            advantages[t] = last_gae
            next_val = float(self.values[t])

        self.advantages = advantages
        self.returns = advantages + self.values

        # Move to device
        self.observations = self.observations.to(self.device)
        self.actions = self.actions.to(self.device)
        self.log_probs = self.log_probs.to(self.device)
        self.advantages = self.advantages.to(self.device)
        self.returns = self.returns.to(self.device)

    def reset(self) -> None:
        """Clear buffer for the next rollout."""
        self._ptr = 0
        self._full = False
        self.advantages = None
        self.returns = None

    def get_minibatches(self, batch_size: int) -> list[dict[str, torch.Tensor]]:
        """
        Shuffle and split the buffer into minibatches.

        Parameters
        ----------
        batch_size : int
            Minibatch size.

        Returns
        -------
        list[dict[str, Tensor]]
            List of minibatch dicts with keys:
            obs, actions, log_probs_old, advantages, returns.
        """
        assert self.advantages is not None, "Call finalize() before get_minibatches()"
        indices = torch.randperm(self.n_steps, device=self.device)
        batches = []
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start : start + batch_size]
            batches.append(
                {
                    "obs": self.observations[idx],
                    "actions": self.actions[idx],
                    "log_probs_old": self.log_probs[idx],
                    "advantages": self.advantages[idx],
                    "returns": self.returns[idx],
                }
            )
        return batches


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------


class TradingActorCritic(nn.Module):
    """
    Shared-backbone actor-critic for discrete trading actions.

    Architecture
    ------------
    obs → TradingResMLP → embedding (256)
              ├─ actor_head  → logits (n_actions) → Categorical
              └─ critic_head → scalar V(s)

    Parameters
    ----------
    obs_dim : int
        Observation dimension (default 42, ADR-037).
    hidden_dim : int
        Backbone hidden size.
    n_blocks : int
        Number of ResBlocks in TradingResMLP.
    n_actions : int
        Number of discrete actions (default 3).

    Examples
    --------
    >>> ac = TradingActorCritic()
    >>> obs = torch.randn(4, 42)
    >>> actions, log_probs, values = ac.act(obs)
    >>> actions.shape, log_probs.shape, values.shape
    (torch.Size([4]), torch.Size([4]), torch.Size([4]))
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

        # Actor head: orthogonal init with small gain for stable initial policy
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

        # Critic head: standard gain
        self.critic_head = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning logits and value estimate.

        Parameters
        ----------
        obs : torch.Tensor
            Batch of observations ``(batch, obs_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(logits, values)`` — shapes ``(batch, n_actions)`` and ``(batch,)``.
        """
        emb = self.backbone(obs)
        logits = self.actor_head(emb)
        value = self.critic_head(emb).squeeze(-1)
        return logits, value

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or greedily select) actions and return log-probs + values.

        Parameters
        ----------
        obs : torch.Tensor
            Observation batch ``(batch, obs_dim)`` or single ``(obs_dim,)``.
        deterministic : bool
            If True, select argmax action (used at evaluation time).

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(actions, log_probs, values)`` — all shape ``(batch,)``.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probs, values, and entropy for given obs-action pairs.

        Used inside the PPO update to compute the probability ratio.

        Parameters
        ----------
        obs : torch.Tensor
            Observation batch ``(batch, obs_dim)``.
        actions : torch.Tensor
            Action indices ``(batch,)`` — long tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(log_probs, values, entropy)`` — all shape ``(batch,)``.
        """
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """
    Proximal Policy Optimization training loop.

    Parameters
    ----------
    actor_critic : TradingActorCritic
        The network being optimised.
    config : PPOConfig
        Training hyperparameters.

    Examples
    --------
    >>> ac = TradingActorCritic()
    >>> trainer = PPOTrainer(ac)
    >>> # stats_list = trainer.train(env, n_updates=100)
    """

    def __init__(
        self,
        actor_critic: TradingActorCritic,
        config: PPOConfig | None = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device)
        self.actor_critic = actor_critic.to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.config.lr)
        self.buffer = RolloutBuffer(
            n_steps=self.config.n_steps,
            obs_dim=self.config.obs_dim,
            device=self.device,
        )
        self._update_count = 0

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self, env: "gym.Env") -> tuple[float, int, float]:
        """
        Run the current policy in the environment for ``n_steps`` steps.

        Handles episode boundaries automatically (auto-resets).

        Parameters
        ----------
        env : gym.Env
            Gymnasium environment.

        Returns
        -------
        tuple[float, int, float]
            ``(last_value, n_episodes, mean_episode_reward)``
        """
        self.buffer.reset()
        self.actor_critic.eval()

        obs, _ = env.reset()
        current_obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        episode_rewards: list[float] = []
        ep_reward = 0.0
        n_episodes = 0

        with torch.no_grad():
            for _ in range(self.config.n_steps):
                actions, log_probs, values = self.actor_critic.act(current_obs)
                action = int(actions.item())
                log_prob = float(log_probs.item())
                value = float(values.item())

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += float(reward)

                self.buffer.push(
                    obs=current_obs.cpu(),
                    action=action,
                    reward=float(reward),
                    value=value,
                    log_prob=log_prob,
                    done=done,
                )

                if done:
                    episode_rewards.append(ep_reward)
                    ep_reward = 0.0
                    n_episodes += 1
                    next_obs, _ = env.reset()

                current_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

            # Bootstrap value for last state
            _, _, last_val = self.actor_critic.act(current_obs)
            last_value = float(last_val.item())

        self.buffer.finalize(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        mean_ep_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        return last_value, n_episodes, mean_ep_reward

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        """
        Run ``n_epochs`` of minibatch PPO updates on the current rollout buffer.

        Returns
        -------
        dict[str, float]
            Aggregated metrics: policy_loss, value_loss, entropy, approx_kl,
            explained_variance.
        """
        self.actor_critic.train()
        assert self.buffer.advantages is not None, "Call collect_rollout() first"

        all_policy_loss: list[float] = []
        all_value_loss: list[float] = []
        all_entropy: list[float] = []
        all_kl: list[float] = []

        for _ in range(self.config.n_epochs):
            for batch in self.buffer.get_minibatches(self.config.batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                log_probs_old = batch["log_probs_old"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Normalise advantages per minibatch
                if advantages.std() > 1e-8:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                log_probs_new, values, entropy = self.actor_critic.evaluate(obs, actions)

                # Policy loss (PPO-clip)
                ratio = torch.exp(log_probs_new - log_probs_old)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.smooth_l1_loss(values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.config.vf_coef * value_loss + self.config.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.grad_clip)
                self.optimizer.step()

                # Approx KL for early stopping
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (log_probs_new - log_probs_old)).mean().item()

                all_policy_loss.append(float(policy_loss.item()))
                all_value_loss.append(float(value_loss.item()))
                all_entropy.append(float(-entropy_loss.item()))
                all_kl.append(float(approx_kl))

            # Early stopping on KL divergence
            if self.config.target_kl is not None:
                mean_kl = float(np.mean(all_kl[-len(self.buffer.get_minibatches(self.config.batch_size)):]))
                if mean_kl > self.config.target_kl:
                    logger.debug("Early stopping at KL=%.4f (target=%.4f)", mean_kl, self.config.target_kl)
                    break

        # Explained variance
        with torch.no_grad():
            returns_np = self.buffer.returns.cpu().numpy()  # type: ignore[union-attr]
            values_np = self.buffer.values.cpu().numpy()
            var_y = np.var(returns_np)
            explained_var = 1.0 - np.var(returns_np - values_np) / (var_y + 1e-8)

        return {
            "policy_loss": float(np.mean(all_policy_loss)),
            "value_loss": float(np.mean(all_value_loss)),
            "entropy": float(np.mean(all_entropy)),
            "approx_kl": float(np.mean(all_kl)),
            "explained_variance": float(explained_var),
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        env: "gym.Env",
        n_updates: int = 200,
        checkpoint_dir: Path | None = None,
        checkpoint_every: int = 50,
        log_every: int = 10,
    ) -> list[UpdateStats]:
        """
        Full PPO training loop: collect rollout → update → repeat.

        Parameters
        ----------
        env : gym.Env
            Environment to train on.
        n_updates : int
            Number of collect+update cycles.
        checkpoint_dir : Path | None
            If set, saves checkpoints here.
        checkpoint_every : int
            Save checkpoint every N updates.
        log_every : int
            Log every N updates.

        Returns
        -------
        list[UpdateStats]
            History of per-update metrics.
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history: list[UpdateStats] = []

        for upd in range(n_updates):
            _, n_eps, mean_ep_reward = self.collect_rollout(env)
            metrics = self.train_epoch()
            self._update_count += 1

            stats = UpdateStats(
                update=upd + 1,
                policy_loss=metrics["policy_loss"],
                value_loss=metrics["value_loss"],
                entropy=metrics["entropy"],
                approx_kl=metrics["approx_kl"],
                explained_variance=metrics["explained_variance"],
                n_episodes=n_eps,
                mean_episode_reward=mean_ep_reward,
            )
            history.append(stats)

            if log_every > 0 and (upd + 1) % log_every == 0:
                logger.info(
                    "update=%d pi_loss=%.4f v_loss=%.4f ent=%.4f kl=%.4f ev=%.3f "
                    "ep_reward=%.4f n_eps=%d",
                    upd + 1,
                    stats.policy_loss,
                    stats.value_loss,
                    stats.entropy,
                    stats.approx_kl,
                    stats.explained_variance,
                    stats.mean_episode_reward,
                    stats.n_episodes,
                )

            if (
                checkpoint_dir is not None
                and checkpoint_every > 0
                and (upd + 1) % checkpoint_every == 0
            ):
                self._save_checkpoint(checkpoint_dir, upd + 1)

        return history

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, directory: Path, update: int) -> Path:
        path = directory / f"ppo_upd{update:05d}.pt"
        torch.save(
            {
                "update": update,
                "actor_critic_state": self.actor_critic.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": dataclasses.asdict(self.config),
            },
            path,
        )
        logger.info("PPO checkpoint saved → %s", path)
        return path

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        actor_critic: TradingActorCritic | None = None,
        config: PPOConfig | None = None,
    ) -> "PPOTrainer":
        """
        Restore a PPOTrainer from a checkpoint file.

        Parameters
        ----------
        path : Path
            Checkpoint file produced by ``_save_checkpoint``.
        actor_critic : TradingActorCritic | None
            If None, a default TradingActorCritic is instantiated.
        config : PPOConfig | None
            If None, config is loaded from the checkpoint.

        Returns
        -------
        PPOTrainer
            Fully restored trainer.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if config is None:
            config = PPOConfig(**ckpt["config"])
        if actor_critic is None:
            actor_critic = TradingActorCritic(
                obs_dim=config.obs_dim,
                hidden_dim=config.hidden_dim,
                n_blocks=config.n_blocks,
                n_actions=config.n_actions,
            )
        trainer = cls(actor_critic, config)
        trainer.actor_critic.load_state_dict(ckpt["actor_critic_state"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        trainer._update_count = ckpt["update"]
        return trainer
