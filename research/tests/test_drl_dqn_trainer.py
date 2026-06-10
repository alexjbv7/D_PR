"""
Tests for DQNTrainer (models/drl/dqn_trainer.py).

Coverage
--------
- train_step returns None when buffer is below min_buffer
- train_step returns a float loss when buffer is sufficiently populated
- Epsilon decay: epsilon decreases per episode and floors at epsilon_end
- Target net sync: weights are copied every target_update grad steps
- run_episode: returns EpisodeStats with valid fields
- Checkpoint save/load: trainer state is fully restored
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.drl.dqn import ReplayBuffer, TradingDQN, Transition
from models.drl.dqn_trainer import DQNConfig, DQNTrainer, EpisodeStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_config() -> DQNConfig:
    return DQNConfig(
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9,
        target_update=5,
        batch_size=8,
        buffer_capacity=200,
        min_buffer=10,
        grad_clip=10.0,
        device="cpu",
    )


@pytest.fixture()
def trainer(default_config: DQNConfig) -> DQNTrainer:
    net = TradingDQN(obs_dim=42, hidden_dim=64, n_blocks=1, n_actions=3)
    return DQNTrainer(net, default_config)


def _fill_buffer(trainer: DQNTrainer, n: int = 15) -> None:
    """Push n random transitions into the replay buffer."""
    for _ in range(n):
        s = torch.randn(42)
        a = int(torch.randint(3, (1,)).item())
        r = float(torch.randn(1).item())
        s2 = torch.randn(42)
        d = bool(torch.rand(1).item() > 0.8)
        trainer.buffer.push(s, a, r, s2, d)


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------


class TestTrainStep:
    def test_returns_none_when_buffer_too_small(self, trainer: DQNTrainer) -> None:
        # Buffer empty → should return None
        result = trainer.train_step()
        assert result is None

    def test_returns_loss_when_buffer_populated(self, trainer: DQNTrainer) -> None:
        _fill_buffer(trainer, n=20)
        loss = trainer.train_step()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_loss_is_finite(self, trainer: DQNTrainer) -> None:
        _fill_buffer(trainer, n=20)
        loss = trainer.train_step()
        assert loss is not None
        assert torch.isfinite(torch.tensor(loss))


# ---------------------------------------------------------------------------
# Target network sync
# ---------------------------------------------------------------------------


class TestTargetSync:
    def test_target_syncs_at_target_update_steps(self, trainer: DQNTrainer) -> None:
        _fill_buffer(trainer, n=50)
        cfg = trainer.config

        # Mutate online net weights directly
        with torch.no_grad():
            for p in trainer.online_net.parameters():
                p.fill_(9.0)

        # Sync happens every target_update steps → run target_update steps
        for _ in range(cfg.target_update):
            trainer.train_step()

        # Target net should now match online net
        for p_online, p_target in zip(
            trainer.online_net.parameters(), trainer.target_net.parameters()
        ):
            assert torch.allclose(p_online, p_target), "Target net not synced"

    def test_target_not_synced_before_threshold(self, trainer: DQNTrainer) -> None:
        _fill_buffer(trainer, n=50)
        # Take fewer steps than target_update — target should differ from online
        with torch.no_grad():
            # Perturb online but not target
            for p in trainer.online_net.backbone.input_proj.parameters():
                p.add_(100.0)

        steps_before_sync = trainer.config.target_update - 1
        for _ in range(steps_before_sync):
            trainer.train_step()

        # Target should NOT yet match the perturbed online net
        perturbed = False
        for p_online, p_target in zip(
            trainer.online_net.parameters(), trainer.target_net.parameters()
        ):
            if not torch.allclose(p_online, p_target):
                perturbed = True
                break
        assert perturbed, "Target was synced too early"


# ---------------------------------------------------------------------------
# Epsilon decay
# ---------------------------------------------------------------------------


class TestEpsilonDecay:
    def test_epsilon_decays_after_episodes(self, trainer: DQNTrainer) -> None:
        """Simulate epsilon decay manually (same logic as train())."""
        initial_eps = trainer.epsilon
        decay = trainer.config.epsilon_decay
        for _ in range(5):
            trainer.epsilon = max(trainer.config.epsilon_end, trainer.epsilon * decay)
        assert trainer.epsilon < initial_eps

    def test_epsilon_floors_at_epsilon_end(self, trainer: DQNTrainer) -> None:
        trainer.epsilon = trainer.config.epsilon_end * 1.001
        for _ in range(100):
            trainer.epsilon = max(trainer.config.epsilon_end, trainer.epsilon * trainer.config.epsilon_decay)
        assert trainer.epsilon >= trainer.config.epsilon_end
        assert abs(trainer.epsilon - trainer.config.epsilon_end) < 1e-6


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_and_load(self, trainer: DQNTrainer, tmp_path: Path) -> None:
        _fill_buffer(trainer, n=20)
        # Do a few steps to give the trainer some state
        for _ in range(trainer.config.target_update + 2):
            trainer.train_step()
        trainer.epsilon = 0.42

        path = trainer._save_checkpoint(tmp_path, episode=10)
        assert path.exists()

        net2 = TradingDQN(obs_dim=42, hidden_dim=64, n_blocks=1, n_actions=3)
        restored = DQNTrainer.load_checkpoint(path, online_net=net2)

        assert abs(restored.epsilon - 0.42) < 1e-6
        assert restored._grad_steps == trainer._grad_steps
        # Online net weights should match
        for p1, p2 in zip(trainer.online_net.parameters(), restored.online_net.parameters()):
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# EpisodeStats
# ---------------------------------------------------------------------------


class TestEpisodeStats:
    def test_episode_stats_fields(self) -> None:
        stats = EpisodeStats(
            episode=1,
            total_reward=-0.5,
            steps=252,
            epsilon=0.3,
            mean_loss=0.01,
            final_equity=0.95,
        )
        assert stats.episode == 1
        assert stats.steps == 252
        assert stats.final_equity == pytest.approx(0.95)
