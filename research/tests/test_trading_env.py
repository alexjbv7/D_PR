"""Tests for TradingEnvironment (ADR-037)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from envs.trading_env import (
    ACTION_BUY,
    ACTION_HOLD,
    EnvironmentConfig,
    TradingEnvironment,
    compute_reward,
)

_REQUIRED_COLS = [
    "close",
    "ret_1",
    "ret_5",
    "ret_20",
    "vol_realized_20",
    "vol_z_60",
    "rsi_14",
    "macd_signal",
    "atr_14",
    "bb_pct",
    "volume_z_20",
    "ob_imbalance",
    "spread_bps",
    "funding_z_60",
    "session_rth",
    "regime_prob_0",
    "regime_prob_1",
    "regime_prob_2",
    "regime_prob_3",
    "regime_prob_4",
    "regime_stability",
    "vol_regime",
]


def _make_synthetic_data(n_bars: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1D", tz="UTC")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n_bars))
    df = pd.DataFrame(index=idx)
    df["close"] = close
    for col in _REQUIRED_COLS:
        if col == "close":
            continue
        if col.startswith("regime_prob"):
            df[col] = 0.2
        elif col == "session_rth":
            df[col] = 1.0
        else:
            df[col] = rng.normal(0.0, 0.5, size=n_bars)
    return df


@pytest.fixture
def env() -> TradingEnvironment:
    data = _make_synthetic_data()
    config = EnvironmentConfig(episode_length=50)
    return TradingEnvironment(data, config=config, seed=0)


class TestTradingEnvironment:
    def test_reset_returns_valid_obs(self, env: TradingEnvironment) -> None:
        obs, info = env.reset(seed=1)
        assert obs.shape == (42,)
        assert obs.dtype == np.float32
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))
        assert info == {}

    def test_step_hold_no_pnl(self, env: TradingEnvironment) -> None:
        env.reset(seed=2)
        total_pnl = 0.0
        for _ in range(10):
            _, reward, _, _, info = env.step(ACTION_HOLD)
            total_pnl += info["pnl_realized"]
            assert env._position == 0
        assert abs(total_pnl) < 1e-6
        assert abs(reward) < 0.1  # last step reward bounded

    def test_step_buy_changes_position(self, env: TradingEnvironment) -> None:
        env.reset(seed=3)
        env.step(ACTION_BUY)
        assert env._position == 1

    def test_reward_no_nan(self, env: TradingEnvironment) -> None:
        env.reset(seed=4)
        rng = np.random.default_rng(4)
        for _ in range(100):
            action = int(rng.integers(0, 3))
            _, reward, terminated, _, _ = env.step(action)
            assert not np.isnan(reward)
            if terminated:
                break

    def test_episode_length(self) -> None:
        data = _make_synthetic_data()
        ep_len = 30
        config = EnvironmentConfig(episode_length=ep_len)
        env = TradingEnvironment(data, config=config, seed=5)
        env.reset(seed=5)
        done_at: int | None = None
        for step in range(1, ep_len + 5):
            _, _, terminated, truncated, _ = env.step(ACTION_HOLD)
            if terminated or truncated:
                done_at = step
                break
        assert done_at == ep_len

    def test_obs_clipped(self, env: TradingEnvironment) -> None:
        obs, _ = env.reset(seed=6)
        assert np.all(obs >= -3.0)
        assert np.all(obs <= 3.0)
        for _ in range(20):
            obs, _, terminated, _, _ = env.step(ACTION_HOLD)
            assert np.all(obs >= -3.0)
            assert np.all(obs <= 3.0)
            if terminated:
                break


class TestComputeReward:
    def test_pure_function_finite(self) -> None:
        r = compute_reward(
            pnl_realized=0.0,
            equity=1.0,
            drawdown=0.0,
            vol_realized=0.1,
            vol_target=0.1,
            delta_position=0.0,
            price=100.0,
            fee_bps=5.0,
            holding_bars=0,
        )
        assert not np.isnan(r)
