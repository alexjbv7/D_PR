"""
Tests for PPO actor-critic and trainer (models/drl/ppo.py).

Coverage
--------
- TradingActorCritic: forward shapes, act() shapes, evaluate() shapes
- RolloutBuffer: push, finalize (GAE), get_minibatches
- PPOTrainer: collect_rollout populates buffer, train_epoch returns valid metrics
- GAE sanity: advantages finite, returns = advantages + values
- Checkpoint save/load
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from envs.trading_env import EnvironmentConfig, TradingEnvironment
from models.drl.ppo import (
    PPOConfig,
    PPOTrainer,
    RolloutBuffer,
    TradingActorCritic,
    UpdateStats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_env(n_bars: int = 400) -> TradingEnvironment:
    """Minimal TradingEnvironment with synthetic data."""
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")
    price = 100.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
    df = pd.DataFrame(index=idx)
    df["close"] = price
    for col in [
        "ret_1", "ret_5", "ret_20", "vol_realized_20", "vol_z_60",
        "rsi_14", "macd_signal", "atr_14", "bb_pct", "volume_z_20",
        "ob_imbalance", "spread_bps", "funding_z_60", "session_rth",
        "regime_prob_0", "regime_prob_1", "regime_prob_2",
        "regime_prob_3", "regime_prob_4", "regime_stability", "vol_regime",
    ]:
        df[col] = np.random.randn(n_bars) * 0.1
    cfg = EnvironmentConfig(episode_length=100)
    return TradingEnvironment(df, cfg, seed=42)


@pytest.fixture()
def env() -> TradingEnvironment:
    return _make_env()


@pytest.fixture()
def small_config() -> PPOConfig:
    return PPOConfig(
        lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        n_epochs=2,
        n_steps=64,
        batch_size=16,
        grad_clip=0.5,
        target_kl=None,  # disable for speed
        n_actions=3,
        obs_dim=42,
        hidden_dim=64,
        n_blocks=1,
        device="cpu",
    )


@pytest.fixture()
def actor_critic(small_config: PPOConfig) -> TradingActorCritic:
    return TradingActorCritic(
        obs_dim=small_config.obs_dim,
        hidden_dim=small_config.hidden_dim,
        n_blocks=small_config.n_blocks,
        n_actions=small_config.n_actions,
    )


@pytest.fixture()
def ppo_trainer(actor_critic: TradingActorCritic, small_config: PPOConfig) -> PPOTrainer:
    return PPOTrainer(actor_critic, small_config)


# ---------------------------------------------------------------------------
# TradingActorCritic
# ---------------------------------------------------------------------------


class TestTradingActorCritic:
    def test_forward_output_shapes(self, actor_critic: TradingActorCritic) -> None:
        obs = torch.randn(8, 42)
        logits, values = actor_critic.forward(obs)
        assert logits.shape == (8, 3)
        assert values.shape == (8,)

    def test_act_batch_shapes(self, actor_critic: TradingActorCritic) -> None:
        obs = torch.randn(4, 42)
        actions, log_probs, values = actor_critic.act(obs)
        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)

    def test_act_single_obs(self, actor_critic: TradingActorCritic) -> None:
        obs = torch.randn(42)
        actions, log_probs, values = actor_critic.act(obs)
        assert actions.shape == (1,)

    def test_act_actions_in_valid_range(self, actor_critic: TradingActorCritic) -> None:
        obs = torch.randn(32, 42)
        actions, _, _ = actor_critic.act(obs)
        assert (actions >= 0).all() and (actions < 3).all()

    def test_act_deterministic_consistent(self, actor_critic: TradingActorCritic) -> None:
        obs = torch.randn(8, 42)
        a1, _, _ = actor_critic.act(obs, deterministic=True)
        a2, _, _ = actor_critic.act(obs, deterministic=True)
        assert (a1 == a2).all()

    def test_evaluate_shapes(self, actor_critic: TradingActorCritic) -> None:
        obs = torch.randn(16, 42)
        actions = torch.randint(0, 3, (16,))
        log_probs, values, entropy = actor_critic.evaluate(obs, actions)
        assert log_probs.shape == (16,)
        assert values.shape == (16,)
        assert entropy.shape == (16,)

    def test_evaluate_log_probs_finite(self, actor_critic: TradingActorCritic) -> None:
        obs = torch.randn(16, 42)
        actions = torch.randint(0, 3, (16,))
        log_probs, _, _ = actor_critic.evaluate(obs, actions)
        assert torch.isfinite(log_probs).all()

    def test_entropy_positive(self, actor_critic: TradingActorCritic) -> None:
        obs = torch.randn(16, 42)
        actions = torch.randint(0, 3, (16,))
        _, _, entropy = actor_critic.evaluate(obs, actions)
        assert (entropy >= 0).all()


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------


class TestRolloutBuffer:
    def test_push_and_is_full(self) -> None:
        buf = RolloutBuffer(n_steps=10, obs_dim=42, device=torch.device("cpu"))
        for _ in range(10):
            buf.push(torch.randn(42), 1, 0.5, 0.1, -0.3, False)
        assert buf.is_full

    def test_finalize_computes_advantages(self) -> None:
        buf = RolloutBuffer(n_steps=10, obs_dim=42, device=torch.device("cpu"))
        for _ in range(10):
            buf.push(torch.randn(42), 1, 0.1, 0.5, -0.3, False)
        buf.finalize(last_value=0.5, gamma=0.99, gae_lambda=0.95)
        assert buf.advantages is not None
        assert buf.advantages.shape == (10,)
        assert torch.isfinite(buf.advantages).all()

    def test_returns_equals_advantages_plus_values(self) -> None:
        buf = RolloutBuffer(n_steps=8, obs_dim=42, device=torch.device("cpu"))
        for _ in range(8):
            buf.push(torch.randn(42), 0, 0.05, 0.3, -0.1, False)
        buf.finalize(last_value=0.3, gamma=0.99, gae_lambda=0.95)
        assert buf.returns is not None
        expected = buf.advantages + buf.values.to(buf.device)
        assert torch.allclose(buf.returns, expected, atol=1e-5)

    def test_get_minibatches_coverage(self) -> None:
        buf = RolloutBuffer(n_steps=16, obs_dim=42, device=torch.device("cpu"))
        for _ in range(16):
            buf.push(torch.randn(42), 1, 0.1, 0.2, -0.1, False)
        buf.finalize(last_value=0.0, gamma=0.99, gae_lambda=0.95)
        batches = buf.get_minibatches(batch_size=4)
        assert len(batches) == 4  # 16 / 4
        total_samples = sum(b["obs"].shape[0] for b in batches)
        assert total_samples == 16

    def test_minibatch_keys(self) -> None:
        buf = RolloutBuffer(n_steps=8, obs_dim=42, device=torch.device("cpu"))
        for _ in range(8):
            buf.push(torch.randn(42), 2, 0.0, 0.1, -0.2, False)
        buf.finalize(last_value=0.0, gamma=0.99, gae_lambda=0.95)
        batch = buf.get_minibatches(batch_size=8)[0]
        for key in ("obs", "actions", "log_probs_old", "advantages", "returns"):
            assert key in batch

    def test_reset_clears_buffer(self) -> None:
        buf = RolloutBuffer(n_steps=4, obs_dim=42, device=torch.device("cpu"))
        for _ in range(4):
            buf.push(torch.randn(42), 0, 0.1, 0.2, -0.1, False)
        buf.finalize(last_value=0.0, gamma=0.99, gae_lambda=0.95)
        buf.reset()
        assert not buf.is_full
        assert buf.advantages is None


# ---------------------------------------------------------------------------
# GAE correctness
# ---------------------------------------------------------------------------


class TestGAE:
    def test_gae_terminal_episode(self) -> None:
        """At done=True, future rewards should not propagate."""
        buf = RolloutBuffer(n_steps=4, obs_dim=1, device=torch.device("cpu"))
        # Steps: r=1,1,1 then done, r=5 after
        buf.push(torch.zeros(1), 0, 1.0, 0.0, 0.0, False)
        buf.push(torch.zeros(1), 0, 1.0, 0.0, 0.0, False)
        buf.push(torch.zeros(1), 0, 1.0, 0.0, 0.0, True)   # terminal
        buf.push(torch.zeros(1), 0, 5.0, 0.0, 0.0, False)
        buf.finalize(last_value=0.0, gamma=1.0, gae_lambda=1.0)
        assert buf.advantages is not None
        # After terminal at t=2, advantage at t=3 should not include t=0,1,2 rewards
        # adv[3] = r[3] = 5.0 (no future)
        assert abs(float(buf.advantages[3].item()) - 5.0) < 1e-4

    def test_gae_all_zeros_rewards(self) -> None:
        """Zero rewards + zero values → advantages = 0."""
        buf = RolloutBuffer(n_steps=8, obs_dim=1, device=torch.device("cpu"))
        for _ in range(8):
            buf.push(torch.zeros(1), 0, 0.0, 0.0, 0.0, False)
        buf.finalize(last_value=0.0, gamma=0.99, gae_lambda=0.95)
        assert buf.advantages is not None
        assert torch.allclose(buf.advantages, torch.zeros(8), atol=1e-6)


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------


class TestPPOTrainer:
    def test_collect_rollout_fills_buffer(
        self, ppo_trainer: PPOTrainer, env: TradingEnvironment
    ) -> None:
        ppo_trainer.collect_rollout(env)
        assert ppo_trainer.buffer.is_full

    def test_train_epoch_returns_metrics(
        self, ppo_trainer: PPOTrainer, env: TradingEnvironment
    ) -> None:
        ppo_trainer.collect_rollout(env)
        metrics = ppo_trainer.train_epoch()
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "explained_variance"):
            assert key in metrics
            assert np.isfinite(metrics[key]), f"{key} is not finite"

    def test_policy_loss_is_scalar(
        self, ppo_trainer: PPOTrainer, env: TradingEnvironment
    ) -> None:
        ppo_trainer.collect_rollout(env)
        metrics = ppo_trainer.train_epoch()
        assert isinstance(metrics["policy_loss"], float)

    def test_entropy_positive(
        self, ppo_trainer: PPOTrainer, env: TradingEnvironment
    ) -> None:
        ppo_trainer.collect_rollout(env)
        metrics = ppo_trainer.train_epoch()
        assert metrics["entropy"] > 0.0

    def test_update_count_increments(
        self, ppo_trainer: PPOTrainer, env: TradingEnvironment
    ) -> None:
        history = ppo_trainer.train(env, n_updates=3, log_every=0)
        assert ppo_trainer._update_count == 3
        assert len(history) == 3

    def test_update_stats_type(
        self, ppo_trainer: PPOTrainer, env: TradingEnvironment
    ) -> None:
        history = ppo_trainer.train(env, n_updates=2, log_every=0)
        assert all(isinstance(s, UpdateStats) for s in history)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class TestPPOCheckpoint:
    def test_save_and_load(
        self,
        ppo_trainer: PPOTrainer,
        env: TradingEnvironment,
        tmp_path: Path,
        small_config: PPOConfig,
    ) -> None:
        ppo_trainer.train(env, n_updates=2, log_every=0)
        path = ppo_trainer._save_checkpoint(tmp_path, update=2)
        assert path.exists()

        ac2 = TradingActorCritic(
            obs_dim=small_config.obs_dim,
            hidden_dim=small_config.hidden_dim,
            n_blocks=small_config.n_blocks,
            n_actions=small_config.n_actions,
        )
        restored = PPOTrainer.load_checkpoint(path, actor_critic=ac2, config=small_config)
        assert restored._update_count == 2

        for p1, p2 in zip(
            ppo_trainer.actor_critic.parameters(),
            restored.actor_critic.parameters(),
        ):
            assert torch.allclose(p1, p2)
