"""
DSR n_trials correction tests.

ADR-029: With 3 horizons × 50 Optuna trials = 150 total hypotheses.
The deflated_sharpe_ratio must be called with n_trials=150, not 50.

Reference: Bailey & López de Prado (2014). The Deflated Sharpe Ratio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).parents[2]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.walk_forward_runner import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from models.multi_horizon.horizon_config import ALL_HORIZONS, TOTAL_OPTUNA_TRIALS


def _make_positive_returns(seed: int = 0, n: int = 252) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0015, 0.018, n)


def test_total_trials_constant() -> None:
    """TOTAL_OPTUNA_TRIALS must equal sum of per-horizon trials."""
    total = sum(h.n_optuna_trials for h in ALL_HORIZONS)
    assert TOTAL_OPTUNA_TRIALS == total
    assert TOTAL_OPTUNA_TRIALS == 150


def test_dsr_n150_lt_dsr_n50_positive_returns() -> None:
    """With positive-Sharpe returns, more trials lowers DSR."""
    r = _make_positive_returns()
    dsr50  = deflated_sharpe_ratio(r, n_trials=50)
    dsr150 = deflated_sharpe_ratio(r, n_trials=150)
    assert dsr150 < dsr50, (
        f"DSR(150)={dsr150:.6f} must be < DSR(50)={dsr50:.6f}. "
        "Higher N raises the expected max SR benchmark."
    )


def test_dsr_decreases_monotonically_with_n() -> None:
    r = _make_positive_returns(seed=7)
    prev = float("inf")
    for n in [1, 5, 10, 50, 100, 150, 300]:
        dsr = deflated_sharpe_ratio(r, n_trials=n)
        assert dsr <= prev + 1e-10, f"DSR not monotone at n={n}: {dsr:.6f} > {prev:.6f}"
        prev = dsr


def test_dsr_bounded_0_1() -> None:
    """DSR is a probability, must be in [0, 1]."""
    for seed in range(5):
        r = np.random.default_rng(seed).normal(0, 0.02, 100)
        dsr = deflated_sharpe_ratio(r, n_trials=150)
        assert 0.0 <= dsr <= 1.0, f"DSR out of bounds: {dsr}"


def test_dsr_n1_equals_psr() -> None:
    """DSR(n_trials=1) == PSR(SR>0) by definition."""
    r = _make_positive_returns(seed=3)
    dsr1 = deflated_sharpe_ratio(r, n_trials=1)
    psr  = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    assert abs(dsr1 - psr) < 1e-10, f"DSR(n=1)={dsr1:.8f} != PSR={psr:.8f}"


def test_cross_horizon_correction_in_trainer() -> None:
    """Verify the trainer uses TOTAL_OPTUNA_TRIALS=150 for cross-horizon DSR."""
    import inspect
    from models.multi_horizon.trainer import MultiHorizonTrainer

    source = inspect.getsource(MultiHorizonTrainer._compute_metrics)
    assert "TOTAL_OPTUNA_TRIALS" in source, (
        "MultiHorizonTrainer._compute_metrics must use TOTAL_OPTUNA_TRIALS for DSR, "
        "not a hard-coded 50 or per-horizon value."
    )
