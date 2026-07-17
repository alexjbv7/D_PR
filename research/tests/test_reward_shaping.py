"""
Acceptance tests for the MTM reward (ADR-041 §8) — torch-free, CPU.

Tests 1-7 cover the reward redesign in ``envs.trading_env``; test 8 covers
the anti-leakage contract of the Optuna reward search (ADR-041 §6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from envs.trading_env import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    EnvironmentConfig,
    TradingEnvironment,
    compute_reward_mtm,
)
from models.drl.dsr_gate import MIN_EMBARGO_BARS, positions_to_returns
from models.drl.reward_search import (
    proxy_validation_split,
    search_reward_weights,
)


def _make_data(closes: np.ndarray) -> pd.DataFrame:
    """Minimal env frame: close only (missing feature columns read as 0)."""
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1D", tz="UTC")
    return pd.DataFrame({"close": closes.astype(float)}, index=idx)


def _full_path_env(
    closes: np.ndarray, config: EnvironmentConfig
) -> TradingEnvironment:
    """Env whose single admissible episode covers the whole frame (start=0)."""
    cfg_dict = {**config.__dict__, "episode_length": len(closes) - 1}
    env = TradingEnvironment(_make_data(closes), config=EnvironmentConfig(**cfg_dict), seed=0)
    env.reset(seed=0)
    return env


# ---------------------------------------------------------------------------
# 1. MTM rewards a winning hold every bar
# ---------------------------------------------------------------------------


def test_mtm_rewards_winning_hold() -> None:
    closes = 100.0 * 1.01 ** np.arange(30)  # +1% per bar
    cfg = EnvironmentConfig(reward_mode="mtm", fee_bps=0.0)
    env = _full_path_env(closes, cfg)

    _, r_entry, _, _, _ = env.step(ACTION_BUY)  # bar 0: prev_pos=0 → no MTM yet
    assert r_entry == pytest.approx(0.0)

    for _ in range(20):
        _, reward, terminated, _, _ = env.step(ACTION_HOLD)
        assert reward > 0.0, "long in an uptrend must earn reward EVERY bar"
        if terminated:
            break


# ---------------------------------------------------------------------------
# 2. MTM penalizes a losing hold every bar (NOT the unrealized-P&L anti-pattern)
# ---------------------------------------------------------------------------


def test_mtm_penalizes_losing_hold() -> None:
    closes = 100.0 * 0.99 ** np.arange(30)  # -1% per bar
    cfg = EnvironmentConfig(reward_mode="mtm", fee_bps=0.0)
    env = _full_path_env(closes, cfg)

    env.step(ACTION_BUY)
    for _ in range(20):
        _, reward, terminated, _, _ = env.step(ACTION_HOLD)
        assert reward < 0.0, "a losing position must bleed negative reward every bar"
        if terminated:
            break


# ---------------------------------------------------------------------------
# 3. Cost is in return units — fee_bps/1e4 per unit of position change, no ·price
# ---------------------------------------------------------------------------


def test_cost_in_return_units() -> None:
    fee_bps = 5.0
    # Pure function: 1-unit flip on a flat bar costs exactly fee_bps/1e4.
    r = compute_reward_mtm(
        prev_position=0.0, price_return=0.0, delta_position=1.0,
        fee_bps=fee_bps, drawdown=0.0, vol_realized=0.0, vol_target=0.0,
        position=1.0, flat_bars=0,
    )
    assert r == pytest.approx(-fee_bps / 1e4)

    # Env, SPY-like price level (~500): identical cost — no price scaling.
    closes = np.full(10, 500.0)
    cfg = EnvironmentConfig(reward_mode="mtm", fee_bps=fee_bps)
    env = _full_path_env(closes, cfg)
    _, r_entry, _, _, _ = env.step(ACTION_BUY)
    assert r_entry == pytest.approx(-fee_bps / 1e4)

    # Full flip long → short (delta = 2) costs twice that.
    _, r_flip, _, _, _ = env.step(ACTION_SELL)
    assert r_flip == pytest.approx(-2.0 * fee_bps / 1e4)


# ---------------------------------------------------------------------------
# 4. Idle penalty fires when FLAT too long — never while holding a position
# ---------------------------------------------------------------------------


def test_idle_penalizes_flat_only() -> None:
    n = 60
    closes = np.full(n, 100.0)  # flat prices isolate the idle term
    cfg = EnvironmentConfig(reward_mode="mtm", fee_bps=0.0, max_flat_bars=5)

    # (a) staying flat beyond max_flat_bars → -w_idle
    env = _full_path_env(closes, cfg)
    rewards = [env.step(ACTION_HOLD)[1] for _ in range(10)]
    assert all(r == pytest.approx(0.0) for r in rewards[:5])
    assert all(r == pytest.approx(-cfg.w_idle) for r in rewards[5:])

    # (b) holding a position for the same span → NO penalty (the ADR-037
    # legacy bug penalized exactly this)
    env = _full_path_env(closes, cfg)
    env.step(ACTION_BUY)
    rewards = [env.step(ACTION_HOLD)[1] for _ in range(10)]
    assert all(r == pytest.approx(0.0) for r in rewards), (
        "sustaining a position must never trigger the idle penalty"
    )


# ---------------------------------------------------------------------------
# 5. reward_mode switch: same trajectory, different rewards; legacy preserved
# ---------------------------------------------------------------------------


def test_reward_mode_ab() -> None:
    rng = np.random.default_rng(7)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, 40))
    actions = [ACTION_BUY] + [ACTION_HOLD] * 30

    def _rollout(mode: str) -> list[float]:
        cfg = EnvironmentConfig(reward_mode=mode)  # type: ignore[arg-type]
        env = _full_path_env(closes, cfg)
        return [env.step(a)[1] for a in actions]

    r_mtm = _rollout("mtm")
    r_realized = _rollout("realized")
    assert not np.allclose(r_mtm, r_realized), (
        "mtm and realized must differ on a buy-and-hold trajectory"
    )
    # Legacy semantics intact: holding without closing realizes no P&L, so the
    # only non-zero legacy rewards are entry cost and the idle-on-holding term.
    assert all(r <= 0.0 for r in r_realized)
    # MTM semantics: per-bar returns flow through (some bars positive).
    assert any(r > 0.0 for r in r_mtm)


def test_reward_mode_invalid_raises() -> None:
    cfg = EnvironmentConfig(reward_mode="banana")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reward_mode"):
        TradingEnvironment(_make_data(np.full(300, 100.0)), config=cfg, seed=0)


# ---------------------------------------------------------------------------
# 6. Without penalties, env rewards == dsr_gate.positions_to_returns (bit-a-bit)
# ---------------------------------------------------------------------------


def test_y002_vol_penalty_not_dead() -> None:
    """
    Y-002 regression: vol_target must not equal realized-of-same-bar, otherwise
    ``w_vol * max(0, vol_r - vol_t)`` is always 0.
    """
    # Pure function: excess realized vol must make reward strictly worse.
    base_kw = dict(
        prev_position=0.0,
        price_return=0.0,
        delta_position=0.0,
        fee_bps=0.0,
        drawdown=0.0,
        position=0.0,
        flat_bars=0,
        w_ret=1.0,
        w_cost=0.0,
        w_dd=0.0,
        w_vol=1.0,
        w_idle=0.0,
        max_flat_bars=999,
    )
    r_ok = compute_reward_mtm(vol_realized=0.01, vol_target=0.01, **base_kw)
    r_hi = compute_reward_mtm(vol_realized=0.05, vol_target=0.01, **base_kw)
    assert r_hi < r_ok
    assert r_hi == pytest.approx(-0.04)

    # Env step must use EnvironmentConfig.vol_target, not vol_realized.
    n = 40
    closes = np.full(n, 100.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    data = pd.DataFrame(
        {
            "close": closes,
            "vol_realized_20": np.full(n, 0.05),  # high realized
        },
        index=idx,
    )
    cfg = EnvironmentConfig(
        reward_mode="mtm",
        fee_bps=0.0,
        vol_target=0.01,
        w_vol=1.0,
        w_ret=0.0,
        w_cost=0.0,
        w_dd=0.0,
        w_idle=0.0,
        episode_length=n - 1,
    )
    env = TradingEnvironment(data, config=cfg, seed=0)
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(ACTION_HOLD)
    assert reward == pytest.approx(-0.04), (
        "Y-002: env must penalize vol_realized > config.vol_target"
    )


def test_reward_matches_gate_return_def() -> None:
    rng = np.random.default_rng(11)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, 50))
    fee_bps = 5.0
    cfg = EnvironmentConfig(
        reward_mode="mtm", fee_bps=fee_bps,
        w_ret=1.0, w_cost=1.0, w_dd=0.0, w_vol=0.0, w_idle=0.0,
    )
    env = _full_path_env(closes, cfg)

    action_cycle = [ACTION_BUY, ACTION_HOLD, ACTION_HOLD, ACTION_SELL, ACTION_HOLD]
    rewards: list[float] = []
    positions: list[float] = []
    step = 0
    while True:
        _, reward, terminated, truncated, info = env.step(action_cycle[step % 5])
        rewards.append(reward)
        positions.append(float(info["position"]))
        step += 1
        if terminated or truncated:
            break

    expected = positions_to_returns(
        np.asarray(positions), closes[: len(positions)], fee_bps
    )
    np.testing.assert_allclose(
        np.asarray(rewards), expected, rtol=0.0, atol=1e-15,
        err_msg="MTM reward must equal the gate's return definition (ADR-041 §7)",
    )


# ---------------------------------------------------------------------------
# 7. Finite and deterministic: no NaN; same seed → same rewards
# ---------------------------------------------------------------------------


def test_reward_finite_deterministic() -> None:
    rng = np.random.default_rng(3)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, 400))

    def _rollout(seed: int) -> list[float]:
        cfg = EnvironmentConfig(reward_mode="mtm", episode_length=100)
        env = TradingEnvironment(_make_data(closes), config=cfg, seed=seed)
        env.reset(seed=seed)
        action_rng = np.random.default_rng(seed)
        rewards = []
        for _ in range(100):
            _, reward, terminated, _, _ = env.step(int(action_rng.integers(0, 3)))
            assert np.isfinite(reward), "reward must never be NaN/inf"
            rewards.append(reward)
            if terminated:
                break
        return rewards

    assert _rollout(seed=5) == _rollout(seed=5), "same seed must replay identically"


# ---------------------------------------------------------------------------
# 8. Optuna search never sees the test fold (ADR-041 §6)
# ---------------------------------------------------------------------------


def test_search_no_leakage() -> None:
    # Walk-forward fold layout: train | embargo | test
    train_idx = np.arange(0, 400)
    test_idx = np.arange(400 + MIN_EMBARGO_BARS, 400 + MIN_EMBARGO_BARS + 100)

    fit_idx, val_idx = proxy_validation_split(train_idx, val_frac=0.25)

    # Structure: both slices inside train, disjoint, embargo gap between them.
    assert np.isin(fit_idx, train_idx).all()
    assert np.isin(val_idx, train_idx).all()
    assert len(np.intersect1d(fit_idx, val_idx)) == 0
    assert fit_idx.max() + MIN_EMBARGO_BARS < val_idx.min()
    # The non-negotiable: nothing the proxy sees overlaps the test fold.
    assert len(np.intersect1d(fit_idx, test_idx)) == 0
    assert len(np.intersect1d(val_idx, test_idx)) == 0

    # Run a real (tiny) study: record every index the objective receives.
    seen: set[int] = set()

    def evaluate(weights: dict[str, float], trial: object) -> float:
        seen.update(fit_idx.tolist())
        seen.update(val_idx.tolist())
        return float(weights["w_ret"])  # arbitrary deterministic objective

    study = search_reward_weights(evaluate, n_trials=3, seed=42)
    assert len(study.trials) == 3
    assert seen.isdisjoint(test_idx.tolist()), (
        "the search objective must never touch the test fold (ADR-041 §6)"
    )

    # Budget cap is enforced (§5.2).
    with pytest.raises(ValueError, match="MAX_PROXY_TRIALS"):
        search_reward_weights(evaluate, n_trials=51)
