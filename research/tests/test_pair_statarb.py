"""
Acceptance tests for PairStatArb + gate vs ZERO (ADR-043 §8) — CPU.

El test crítico es el anti-leakage (§4): espía sobre los helpers de fit
verifica que cointegración/β/half-life ven SOLO índices de train por fold,
con embargo >= 60 entre train y test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).parents[2]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import alpha.statarb.pairs as pairs_mod
from alpha.statarb.pairs import (
    PairParams,
    PairStatArb,
    PairStatArbConfig,
    walk_forward_pair_returns,
    _half_life,
)
from models.drl.dsr_gate import MIN_EMBARGO_BARS, evaluate_zero_gate
from models.validation import WalkForwardSplitter


# ---------------------------------------------------------------------------
# Generadores sintéticos
# ---------------------------------------------------------------------------


def _ou_series(n: int, theta: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] - theta * s[t - 1] + rng.normal(0.0, sigma)
    return s


def _cointegrated_pair(
    n: int = 1500,
    seed: int = 7,
    theta: float = 0.2,
    sigma_ou: float = 0.01,
    beta: float = 1.0,
) -> pd.DataFrame:
    """log_y = β·log_x + OU(θ, σ) + c — par cointegrado por construcción."""
    rng = np.random.default_rng(seed)
    log_x = np.log(100.0) + np.cumsum(rng.normal(0.0, 0.01, n))
    spread = _ou_series(n, theta, sigma_ou, rng)
    log_y = beta * log_x + spread + 0.5
    idx = pd.date_range("2018-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"SPY": np.exp(log_y), "QQQ": np.exp(log_x)}, index=idx)


def _independent_walks(n: int = 1500, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_y = np.log(100.0) + np.cumsum(rng.normal(0.0, 0.01, n))
    log_x = np.log(300.0) + np.cumsum(rng.normal(0.0, 0.01, n))
    idx = pd.date_range("2018-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"SPY": np.exp(log_y), "QQQ": np.exp(log_x)}, index=idx)


def _splitter() -> WalkForwardSplitter:
    return WalkForwardSplitter(
        train_size=600, test_size=200, expanding=True, embargo=MIN_EMBARGO_BARS
    )


# ---------------------------------------------------------------------------
# 1. Anti-leakage (§4): los helpers de fit ven SOLO train por fold
# ---------------------------------------------------------------------------


def test_antileakage_fit_sees_train_only(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = _cointegrated_pair()
    strategy = PairStatArb("SPY", "QQQ")

    helper_indexes: list[pd.DatetimeIndex] = []
    orig_coint = pairs_mod._coint_pvalue
    orig_beta = pairs_mod._fit_beta
    orig_hl = pairs_mod._half_life

    def spy_coint(ly: pd.Series, lx: pd.Series) -> float:
        helper_indexes.append(ly.index)
        helper_indexes.append(lx.index)
        return orig_coint(ly, lx)

    def spy_beta(ly: pd.Series, lx: pd.Series) -> float:
        helper_indexes.append(ly.index)
        helper_indexes.append(lx.index)
        return orig_beta(ly, lx)

    def spy_hl(spread: pd.Series) -> float:
        helper_indexes.append(spread.index)
        return orig_hl(spread)

    monkeypatch.setattr(pairs_mod, "_coint_pvalue", spy_coint)
    monkeypatch.setattr(pairs_mod, "_fit_beta", spy_beta)
    monkeypatch.setattr(pairs_mod, "_half_life", spy_hl)

    fold_train: list[pd.DatetimeIndex] = []
    fold_test: list[pd.DatetimeIndex] = []
    orig_fit = PairStatArb.fit
    orig_signals = PairStatArb.signals

    def spy_fit(self: PairStatArb, train: pd.DataFrame) -> PairParams:
        fold_train.append(train.index)
        return orig_fit(self, train)

    def spy_signals(self: PairStatArb, test: pd.DataFrame, params: PairParams):
        fold_test.append(test.index)
        return orig_signals(self, test, params)

    monkeypatch.setattr(PairStatArb, "fit", spy_fit)
    monkeypatch.setattr(PairStatArb, "signals", spy_signals)

    walk_forward_pair_returns(prices, _splitter(), strategy)

    assert fold_train and fold_test
    train_sets = [set(idx) for idx in fold_train]

    # Cada llamada a coint/OLS/half-life vio EXACTAMENTE un train de fold
    for idx in helper_indexes:
        assert set(idx) in train_sets, (
            "un helper de fit vio índices fuera del train del fold (leakage)"
        )

    # Train y test disjuntos + embargo >= 60 barras posicionales por fold
    for tr_idx, te_idx in zip(fold_train, fold_test):
        assert set(tr_idx).isdisjoint(set(te_idx))
        gap = prices.index.get_loc(te_idx[0]) - prices.index.get_loc(tr_idx[-1]) - 1
        assert gap >= MIN_EMBARGO_BARS


# ---------------------------------------------------------------------------
# 2. Cointegración sintética
# ---------------------------------------------------------------------------


def test_cointegrated_pair_accepted_and_z_centered() -> None:
    prices = _cointegrated_pair()
    strategy = PairStatArb("SPY", "QQQ")
    params = strategy.fit(prices)

    assert params.tradeable, params.reject_reason
    assert params.coint_pvalue <= strategy.config.coint_alpha
    assert params.beta == pytest.approx(1.0, abs=0.1)

    # El z-score del train tiene media 0 por construcción de mean/std
    log_y = np.log(prices["SPY"])
    log_x = np.log(prices["QQQ"])
    z = (log_y - params.beta * log_x - params.spread_mean) / params.spread_std
    assert abs(float(z.mean())) < 1e-10


def test_independent_walks_rejected() -> None:
    prices = _independent_walks()
    params = PairStatArb("SPY", "QQQ").fit(prices)
    assert params.tradeable is False
    assert "not_cointegrated" in params.reject_reason


# ---------------------------------------------------------------------------
# 3. Half-life
# ---------------------------------------------------------------------------


def test_half_life_fast_reversion_passes() -> None:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=2000, freq="B", tz="UTC")
    ou = pd.Series(_ou_series(2000, theta=0.2, sigma=0.01, rng=rng), index=idx)
    hl = _half_life(ou)
    # θ=0.2 → half-life teórica ln(2)/0.2 ≈ 3.5 barras
    assert 1.0 < hl < 10.0


def test_half_life_random_walk_rejected() -> None:
    rng = np.random.default_rng(5)
    idx = pd.date_range("2020-01-01", periods=2000, freq="B", tz="UTC")
    rw = pd.Series(np.cumsum(rng.normal(0.0, 0.01, 2000)), index=idx)
    hl = _half_life(rw)
    assert not (0.0 < hl <= 30.0), f"un random walk no debe pasar el filtro (hl={hl})"


def test_fit_rejects_slow_half_life() -> None:
    prices = _cointegrated_pair()  # hl real ~3.5 barras
    strategy = PairStatArb("SPY", "QQQ", PairStatArbConfig(max_half_life=1.0))
    params = strategy.fit(prices)
    assert params.tradeable is False
    assert "half_life" in params.reject_reason


# ---------------------------------------------------------------------------
# 4. Long-short: usa ambos lados
# ---------------------------------------------------------------------------


def test_signals_use_both_sides() -> None:
    prices = _cointegrated_pair()
    strategy = PairStatArb("SPY", "QQQ")
    train, test = prices.iloc[:700], prices.iloc[760:]
    params = strategy.fit(train)
    assert params.tradeable, params.reject_reason

    positions = strategy.signals(test, params)
    assert (positions == 1.0).any(), "nunca va long el spread"
    assert (positions == -1.0).any(), "nunca va short el spread — no es long-short"


# ---------------------------------------------------------------------------
# 5. Doble fee: el costo escala con |Δpos_y| + |Δpos_x| = |Δpos|·(1+|β|)
# ---------------------------------------------------------------------------


def test_double_fee_two_legs() -> None:
    fee_bps = 5.0
    idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
    flat = pd.DataFrame({"SPY": [100.0] * 5, "QQQ": [200.0] * 5}, index=idx)
    positions = np.array([0.0, 1.0, 1.0, 0.0, 0.0])

    def _params(beta: float) -> PairParams:
        return PairParams(
            beta=beta, spread_mean=0.0, spread_std=1.0,
            half_life=5.0, coint_pvalue=0.01, tradeable=True,
        )

    strategy = PairStatArb("SPY", "QQQ", PairStatArbConfig(fee_bps=fee_bps))
    fee = fee_bps / 1e4

    # β=1: una entrada mueve AMBAS patas → costo 2·fee (no 1·fee)
    r1 = strategy.returns(flat, positions, _params(beta=1.0))
    np.testing.assert_allclose(r1, [0.0, -2 * fee, 0.0, -2 * fee, 0.0], atol=1e-15)

    # β=2: |Δpos_y|+|Δpos_x| = (1+2)·|Δpos| → costo 3·fee
    r2 = strategy.returns(flat, positions, _params(beta=2.0))
    np.testing.assert_allclose(r2, [0.0, -3 * fee, 0.0, -3 * fee, 0.0], atol=1e-15)


# ---------------------------------------------------------------------------
# 6. Gate vs ZERO
# ---------------------------------------------------------------------------


def test_gate_passes_on_profitable_cointegrated_pair() -> None:
    prices = _cointegrated_pair()
    strategy = PairStatArb("SPY", "QQQ", PairStatArbConfig(fee_bps=1.0))
    r = walk_forward_pair_returns(prices, _splitter(), strategy)

    result = evaluate_zero_gate(r, n_trials=1)
    assert result.passed, result.reason
    assert result.sharpe_agent > 0.0
    assert result.sharpe_buyhold == 0.0  # el benchmark ES cero


def test_gate_fails_on_noise() -> None:
    prices = _independent_walks()
    strategy = PairStatArb("SPY", "QQQ")
    r = walk_forward_pair_returns(prices, _splitter(), strategy)

    result = evaluate_zero_gate(r, n_trials=1)
    assert result.passed is False
    assert "FAIL" in result.reason


def test_gate_control_baseline_is_diagnostic_only() -> None:
    """El control (z barajado) se reporta pero NO es condición del gate."""
    rng = np.random.default_rng(0)
    strong = rng.normal(0.002, 0.005, 800)   # edge claro
    control = rng.normal(0.0, 0.005, 800)    # ruido
    result = evaluate_zero_gate(strong, n_trials=1, control_returns=control)
    assert result.passed, result.reason
    assert "dsr_control" in result.reason


def test_gate_deflation_by_n_trials() -> None:
    """Más pares buscados OOS → DSR deflactado menor (sesgo de selección)."""
    rng = np.random.default_rng(1)
    r = rng.normal(0.0006, 0.005, 800)  # edge marginal
    dsr_1 = evaluate_zero_gate(r, n_trials=1).dsr_agent
    dsr_20 = evaluate_zero_gate(r, n_trials=20).dsr_agent
    assert dsr_20 < dsr_1


# ---------------------------------------------------------------------------
# 7. Embargo >= 60
# ---------------------------------------------------------------------------


def test_embargo_enforced() -> None:
    prices = _cointegrated_pair()
    bad = WalkForwardSplitter(
        train_size=600, test_size=200, expanding=True, embargo=10
    )
    with pytest.raises(ValueError, match="embargo"):
        walk_forward_pair_returns(prices, bad, PairStatArb("SPY", "QQQ"))
