"""
DQN training loop for discrete trading actions (ADR-038, ADR-039).

Components
----------
EpisodeStats     : Dataclass with per-episode metrics (reward, steps, epsilon, loss).
DQNConfig        : Training hyperparameters as a frozen dataclass.
DQNTrainer       : Target network, epsilon-greedy schedule, TD loss, train/eval loops.

Design decisions
----------------
- Double DQN: online net selects action, target net evaluates Q-value.
  Reduces overestimation bias (van Hasselt et al. 2015).
- Target net updated via hard copy every ``target_update`` steps (not soft update)
  — simpler, sufficient for discrete action spaces of this size.
- Huber loss (smooth_l1) instead of MSE: clips gradients for large TD errors,
  improving stability in early training when Q-estimates are poor.
- Epsilon decay is multiplicative per episode (not per step) to keep exploration
  proportional to episode count, independent of episode length.
- No per-step gradient updates during episode collection — batch updates only
  (avoids correlation between consecutive samples).
- train() saves a checkpoint dict (not just state_dict) so run metadata is preserved.

References
----------
Mnih et al. (2015). Human-level control through DRL. Nature.
van Hasselt et al. (2015). Deep Reinforcement Learning with Double Q-learning.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim

from models.drl.dqn import ReplayBuffer, TradingDQN

if TYPE_CHECKING:
    import gymnasium as gym

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + stats
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DQNConfig:
    """
    Hyperparameters for DQN training.

    Parameters
    ----------
    lr : float
        Adam learning rate.
    gamma : float
        Discount factor.
    epsilon_start : float
        Initial exploration probability.
    epsilon_end : float
        Minimum exploration probability.
    epsilon_decay : float
        Multiplicative decay applied per episode.
    target_update : int
        Number of gradient steps between hard target-net copies.
    batch_size : int
        Minibatch size drawn from replay buffer.
    buffer_capacity : int
        Maximum replay buffer size.
    min_buffer : int
        Minimum transitions before training starts.
    grad_clip : float
        Max gradient norm (0 = disabled).
    device : str
        Torch device string ("cpu" or "cuda").
    """

    lr: float = 3e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update: int = 100
    batch_size: int = 64
    buffer_capacity: int = 50_000
    min_buffer: int = 1_000
    grad_clip: float = 10.0
    device: str = "cpu"


@dataclasses.dataclass
class EpisodeStats:
    """Metrics collected over a single training episode."""

    episode: int
    total_reward: float
    steps: int
    epsilon: float
    mean_loss: float
    final_equity: float


# ---------------------------------------------------------------------------
# DQNTrainer
# ---------------------------------------------------------------------------


class DQNTrainer:
    """
    Training loop for TradingDQN with double-DQN updates.

    Parameters
    ----------
    online_net : TradingDQN
        The network being optimised.
    config : DQNConfig
        Training hyperparameters.

    Examples
    --------
    >>> net = TradingDQN()
    >>> trainer = DQNTrainer(net)
    >>> # trainer.train(env, n_episodes=500, checkpoint_dir=Path("artifacts/drl"))
    """

    def __init__(
        self,
        online_net: TradingDQN,
        config: DQNConfig | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.device = torch.device(self.config.device)

        self.online_net = online_net.to(self.device)
        self.target_net = copy.deepcopy(online_net).to(self.device)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.buffer_capacity)
        self.epsilon = self.config.epsilon_start
        self._grad_steps = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(self) -> float | None:
        """
        Sample a minibatch and perform one gradient step.

        Returns
        -------
        float | None
            TD loss value, or None if buffer is too small.
        """
        if len(self.buffer) < self.config.min_buffer:
            return None

        transitions = self.buffer.sample(self.config.batch_size)
        states = torch.stack([t.state for t in transitions]).to(self.device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        next_states = torch.stack([t.next_state for t in transitions]).to(self.device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=self.device)

        # Double DQN: online net selects best action, target net evaluates it
        with torch.no_grad():
            best_actions = self.online_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.config.gamma * next_q * (1.0 - dones)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self._grad_steps += 1
        if self._grad_steps % self.config.target_update == 0:
            self._sync_target()

        return float(loss.item())

    def run_episode(self, env: "gym.Env") -> EpisodeStats:
        """
        Collect one episode of experience and interleave gradient updates.

        Parameters
        ----------
        env : gym.Env
            A reset-able Gymnasium environment (TradingEnvironment).

        Returns
        -------
        EpisodeStats
            Per-episode metrics.
        """
        obs, _ = env.reset()
        state = torch.tensor(obs, dtype=torch.float32)

        total_reward = 0.0
        losses: list[float] = []
        step = 0
        final_equity = 1.0

        while True:
            action = self.online_net.select_action(state.to(self.device), epsilon=self.epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = torch.tensor(next_obs, dtype=torch.float32)
            self.buffer.push(state, action, float(reward), next_state, done)

            state = next_state
            total_reward += float(reward)
            step += 1
            final_equity = info.get("equity", 1.0)

            loss = self.train_step()
            if loss is not None:
                losses.append(loss)

            if done:
                break

        return EpisodeStats(
            episode=0,  # caller sets this
            total_reward=total_reward,
            steps=step,
            epsilon=self.epsilon,
            mean_loss=float(sum(losses) / len(losses)) if losses else 0.0,
            final_equity=float(final_equity),
        )

    def train(
        self,
        env: "gym.Env",
        n_episodes: int = 500,
        checkpoint_dir: Path | None = None,
        checkpoint_every: int = 100,
        log_every: int = 10,
    ) -> list[EpisodeStats]:
        """
        Full training loop.

        Parameters
        ----------
        env : gym.Env
            Environment to train on.
        n_episodes : int
            Number of episodes to run.
        checkpoint_dir : Path | None
            If set, saves checkpoints to this directory.
        checkpoint_every : int
            Save a checkpoint every N episodes.
        log_every : int
            Log summary every N episodes.

        Returns
        -------
        list[EpisodeStats]
            History of per-episode metrics.
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history: list[EpisodeStats] = []

        for ep in range(n_episodes):
            stats = self.run_episode(env)
            stats.episode = ep
            history.append(stats)

            # Epsilon decay — per episode
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay,
            )

            if log_every > 0 and (ep + 1) % log_every == 0:
                logger.info(
                    "ep=%d reward=%.4f equity=%.4f eps=%.3f loss=%.5f steps=%d",
                    ep + 1,
                    stats.total_reward,
                    stats.final_equity,
                    stats.epsilon,
                    stats.mean_loss,
                    stats.steps,
                )

            if (
                checkpoint_dir is not None
                and checkpoint_every > 0
                and (ep + 1) % checkpoint_every == 0
            ):
                self._save_checkpoint(checkpoint_dir, ep + 1)

        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_target(self) -> None:
        """Hard copy of online net weights to target net."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        logger.debug("Target net synced at grad_step=%d", self._grad_steps)

    def _save_checkpoint(self, directory: Path, episode: int) -> Path:
        """
        Persist online net + trainer state.

        Parameters
        ----------
        directory : Path
            Directory to write the checkpoint file.
        episode : int
            Current episode number (used in filename).

        Returns
        -------
        Path
            Path to the saved checkpoint file.
        """
        path = directory / f"dqn_ep{episode:05d}.pt"
        torch.save(
            {
                "episode": episode,
                "epsilon": self.epsilon,
                "grad_steps": self._grad_steps,
                "online_net_state": self.online_net.state_dict(),
                "target_net_state": self.target_net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": dataclasses.asdict(self.config),
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)
        return path

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        online_net: TradingDQN | None = None,
        config: DQNConfig | None = None,
    ) -> "DQNTrainer":
        """
        Restore a DQNTrainer from a checkpoint file.

        Parameters
        ----------
        path : Path
            Checkpoint file produced by ``_save_checkpoint``.
        online_net : TradingDQN | None
            If None, a default TradingDQN is instantiated.
        config : DQNConfig | None
            If None, config is loaded from the checkpoint.

        Returns
        -------
        DQNTrainer
            Fully restored trainer ready to continue training.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if config is None:
            config = DQNConfig(**ckpt["config"])
        if online_net is None:
            online_net = TradingDQN()
        trainer = cls(online_net, config)
        trainer.online_net.load_state_dict(ckpt["online_net_state"])
        trainer.target_net.load_state_dict(ckpt["target_net_state"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        trainer.epsilon = ckpt["epsilon"]
        trainer._grad_steps = ckpt["grad_steps"]
        return trainer
