"""
Acceptance tests for the ADR-040 walk-forward DSR promotion gate.

CPU-only and fast: synthetic OHLCV, no network, no Alpaca credentials.
Tests 1-6 are torch-free (``models.drl`` resolves exports lazily); the heavy
end-to-end test trains a tiny DQN and is skipped where torch is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

_RESEARCH = Path(__file__).parents[1]
for _p in (str(_RESEARCH), str(_RESEARCH.parent / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.drl_dataset import clean_close_series, n_clean_bars  # noqa: E402
from envs import EnvironmentConfig  # noqa: E402
from models.drl import dsr_gate  # noqa: E402
from models.validation import WalkForwardSplitter  # noqa: E402
from models.walk_forward_runner import (  # noqa: E402
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _synthetic_ohlcv(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, n))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))
    open_ = np.clip(close * (1.0 + rng.normal(0.0, 0.003, n)), low, high)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _splitter(embargo: int = 60) -> WalkForwardSplitter:
    return WalkForwardSplitter(
        train_size=170, test_size=60, embargo=embargo, expanding=True
    )


def _returns_with_sharpe(
    sharpe_annual: float, n: int, seed: int, sigma: float = 0.01
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = sharpe_annual / np.sqrt(252.0) * sigma
    r = rng.normal(0.0, sigma, n)
    # Recenter so the empirical Sharpe is exactly the requested one.
    r = r - r.mean() + mu * (r.std(ddof=1) / sigma)
    return r


# ---------------------------------------------------------------------------
# 1. DSR sanity vs an independent reference computation
# ---------------------------------------------------------------------------


def test_dsr_matches_reference():
    ppy, T = 252, 1500
    r = _returns_with_sharpe(1.2, T, seed=0)

    psr = probabilistic_sharpe_ratio(r, 0.0, ppy)

    # Independent reference: Mertens SE + normal CDF, computed from scratch.
    sr_1 = r.mean() / r.std(ddof=1)
    skew = float(scipy_stats.skew(r))
    kurt = float(scipy_stats.kurtosis(r, fisher=False))
    se = np.sqrt((1.0 - skew * sr_1 + (kurt - 1.0) / 4.0 * sr_1**2) / (T - 1))
    ref = float(scipy_stats.norm.cdf(sr_1 / se))
    assert psr == pytest.approx(ref, abs=1e-9)
    assert 0.7 < psr <= 1.0  # SR 1.2 over 1500 bars must look credible

    # DSR contract: n_trials=1 degenerates to PSR; deflation lowers it.
    assert deflated_sharpe_ratio(r, 1, ppy) == pytest.approx(psr)
    dsr_50 = deflated_sharpe_ratio(r, 50, ppy)
    assert dsr_50 < psr
    assert 0.0 <= dsr_50 <= 1.0


# ---------------------------------------------------------------------------
# 2. Anti-leakage: GMM sees ONLY each fold's train bars (ADR-040 §4.1)
# ---------------------------------------------------------------------------


def test_no_leakage_gmm_per_fold(monkeypatch):
    import features.regime_gmm as rg

    ohlcv = _synthetic_ohlcv(500)
    splitter = _splitter()
    n_clean = n_clean_bars(ohlcv)
    expected = [
        len(train_idx)
        for train_idx, _ in splitter.split(pd.DataFrame(index=np.arange(n_clean)))
    ]
    assert len(expected) >= 2, "fixture must produce multiple folds"

    seen: list[int] = []
    orig_fit = rg.GMMRegimeDetector.fit

    def spy_fit(self, close, atr):
        seen.append(len(close))
        return orig_fit(self, close, atr)

    monkeypatch.setattr(rg.GMMRegimeDetector, "fit", spy_fit)

    dsr_gate.xgb_oos_returns(ohlcv, splitter, xgb_params={"n_estimators": 25})

    # One fit per fold, each seeing EXACTLY that fold's train bars — never
    # the full series (the old single train_frac behaviour was leakage).
    assert seen == expected
    assert all(s < n_clean for s in seen)


# ---------------------------------------------------------------------------
# 3. Buy-and-hold baseline reproduces close-to-close returns minus entry fee
# ---------------------------------------------------------------------------


def test_buyhold_returns_equal_price_returns():
    ohlcv = _synthetic_ohlcv(500)
    splitter = _splitter()
    fee_bps = 5.0

    got = dsr_gate.buyhold_oos_returns(ohlcv, splitter, fee_bps=fee_bps)

    closes = clean_close_series(ohlcv).to_numpy()
    n_clean = len(closes)
    expected_parts: list[np.ndarray] = []
    for _, test_idx in splitter.split(pd.DataFrame(index=np.arange(n_clean))):
        n_t = len(test_idx) - 1
        px = closes[test_idx[:n_t]]
        exp = np.zeros(n_t)
        exp[1:] = px[1:] / px[:-1] - 1.0  # pure price returns
        exp[0] -= fee_bps / 1e4           # one-off entry fee per fold
        expected_parts.append(exp)
    expected = np.concatenate(expected_parts)

    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 4-5. Gate verdicts
# ---------------------------------------------------------------------------


def test_gate_fails_when_below_threshold():
    n = 1200
    agent = _returns_with_sharpe(-0.5, n, seed=1)    # no edge
    buyhold = _returns_with_sharpe(0.8, n, seed=2)
    xgb = _returns_with_sharpe(0.0, n, seed=3)

    res = dsr_gate.evaluate_drl_gate(agent, buyhold, xgb, n_trials=1)

    assert res.passed is False
    assert res.dsr_agent <= 0.4
    assert "dsr_threshold" in res.reason and "FAIL" in res.reason
    assert res.n_oos_bars == n


def test_gate_fails_when_below_buyhold():
    n = 2500
    agent = _returns_with_sharpe(1.0, n, seed=4)     # decent, clears 0.4 DSR
    buyhold = _returns_with_sharpe(3.0, n, seed=5)   # ... but market is better
    xgb = _returns_with_sharpe(-1.0, n, seed=6)      # supervised baseline weak

    res = dsr_gate.evaluate_drl_gate(agent, buyhold, xgb, n_trials=1)

    assert res.passed is False
    assert res.dsr_agent > 0.4, "fixture must isolate the buy-hold condition"
    assert res.dsr_agent > res.dsr_xgb
    assert "sharpe_buyhold" in res.reason
    assert "dsr_threshold" not in res.reason


# ---------------------------------------------------------------------------
# 6. Embargo contract (ADR-040 §4.3)
# ---------------------------------------------------------------------------


def test_embargo_enforced():
    ohlcv = _synthetic_ohlcv(500)

    # Embargo below the 60-bar floor (vol_z_60 window) must be rejected
    # by every OOS-return path.
    bad = _splitter(embargo=10)
    with pytest.raises(ValueError, match="embargo"):
        dsr_gate.buyhold_oos_returns(ohlcv, bad)
    with pytest.raises(ValueError, match="embargo"):
        dsr_gate.xgb_oos_returns(ohlcv, bad)

    good = _splitter(embargo=60)
    assert good.embargo >= dsr_gate.MIN_EMBARGO_BARS

    n_clean = n_clean_bars(ohlcv)
    n_folds = 0
    for train_idx, test_idx in good.split(pd.DataFrame(index=np.arange(n_clean))):
        n_folds += 1
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert test_idx.min() - train_idx.max() > 60  # embargo gap respected
    assert n_folds >= 2


# ---------------------------------------------------------------------------
# 7. End-to-end stub (tiny DQN; skipped without torch)
# ---------------------------------------------------------------------------


def test_end_to_end_stub():
    pytest.importorskip("torch")

    ohlcv = _synthetic_ohlcv(520, seed=3)
    env_cfg = EnvironmentConfig(episode_length=40)
    splitter = WalkForwardSplitter(
        train_size=200, test_size=80, embargo=60, expanding=True
    )
    spec = dsr_gate.AgentSpec(algo="dqn", episodes=2, seed=0)

    agent_r = dsr_gate.walk_forward_oos_returns(
        spec, ohlcv, splitter, env_cfg, seed=0
    )
    buyhold_r = dsr_gate.buyhold_oos_returns(ohlcv, splitter, fee_bps=env_cfg.fee_bps)
    xgb_r = dsr_gate.xgb_oos_returns(
        ohlcv, splitter, fee_bps=env_cfg.fee_bps, xgb_params={"n_estimators": 25}
    )

    assert len(agent_r) == len(buyhold_r) == len(xgb_r) > 0
    assert not np.isnan(agent_r).any()

    res = dsr_gate.evaluate_drl_gate(agent_r, buyhold_r, xgb_r, n_trials=1)

    assert isinstance(res, dsr_gate.GateResult)
    for field in ("dsr_agent", "psr_agent", "sharpe_agent",
                  "sharpe_buyhold", "dsr_xgb"):
        assert np.isfinite(getattr(res, field)), f"{field} must be finite"
    assert res.n_trials == 1
    assert res.n_oos_bars == len(agent_r)
    assert isinstance(res.passed, bool)
    assert res.reason.startswith(("PASS", "FAIL"))
