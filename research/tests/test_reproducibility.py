"""
Reproducibility tests for multi-horizon trainer.

Same seed + same data → same artifact binary hash.

Note: XGBoost with n_jobs=1 and fixed seed is deterministic.
      DeepMLP with fixed torch seed is deterministic on CPU.
"""
from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).parents[2]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.multi_horizon.trainer import MultiHorizonTrainer, _bar_size_minutes
from models.multi_horizon.horizon_config import INTRADAY


# ============================================================================
# Helpers
# ============================================================================


def _make_synth_dataset(seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    n = 300
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    X = pd.DataFrame(
        rng.standard_normal((n, 5)),
        index=idx,
        columns=[f"feat_{i}" for i in range(5)],
    )
    y = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx)
    prices = pd.Series(100.0 * (1 + rng.normal(0, 0.01, n)).cumprod(), index=idx)
    return X, y, prices


def _artifact_hash(model_object: object) -> str:
    blob = pickle.dumps(model_object, protocol=5)
    return hashlib.sha256(blob).hexdigest()


# ============================================================================
# Tests
# ============================================================================


def test_same_seed_same_final_model_hash() -> None:
    """
    Two MultiHorizonTrainer instances with the same seed must produce
    identical final model artifacts (same pickle hash).
    Uses the INTRADAY horizon on a small synthetic dataset.
    """
    X, y, prices = _make_synth_dataset(seed=42)

    # Minimal horizon config (strip down to avoid long run time in tests)
    from models.multi_horizon.horizon_config import HorizonConfig
    from decimal import Decimal

    cfg_small = HorizonConfig(
        name="intraday",
        bar_size="5min",
        tp_pct=Decimal("0.005"),
        sl_pct=Decimal("0.005"),
        timeout_bars=3,
        embargo=INTRADAY.embargo,
        train_lookback=INTRADAY.train_lookback,
        feature_set="INTRADAY_FEATURES",
        model_name="xgb",
        n_optuna_trials=2,  # minimal for speed
    )

    def _run_once(seed: int) -> object:
        trainer = MultiHorizonTrainer(seed=seed, ablate=False, n_wf_splits=3)
        result = trainer.run_horizon(cfg_small, X, y, prices)
        return result.model_object

    model_a = _run_once(42)
    model_b = _run_once(42)

    hash_a = _artifact_hash(model_a)
    hash_b = _artifact_hash(model_b)

    assert hash_a == hash_b, (
        f"Same seed (42) produced different artifact hashes:\n"
        f"  run 1: {hash_a}\n  run 2: {hash_b}\n"
        "Check that XGBoost uses n_jobs=1 and seed=42, "
        "and that Optuna sampler is seeded."
    )


def test_different_seed_different_model() -> None:
    """Different seeds should (almost always) produce different models."""
    X, y, prices = _make_synth_dataset(seed=0)

    from models.multi_horizon.horizon_config import HorizonConfig
    from decimal import Decimal

    cfg_small = HorizonConfig(
        name="intraday",
        bar_size="5min",
        tp_pct=Decimal("0.005"),
        sl_pct=Decimal("0.005"),
        timeout_bars=3,
        embargo=INTRADAY.embargo,
        train_lookback=INTRADAY.train_lookback,
        feature_set="INTRADAY_FEATURES",
        model_name="xgb",
        n_optuna_trials=2,
    )

    def _run(seed: int) -> str:
        trainer = MultiHorizonTrainer(seed=seed, ablate=False, n_wf_splits=3)
        result = trainer.run_horizon(cfg_small, X, y, prices)
        return _artifact_hash(result.model_object)

    hash_42  = _run(42)
    hash_123 = _run(123)

    # This could theoretically be equal by chance, but is extremely unlikely
    assert hash_42 != hash_123, (
        "Seeds 42 and 123 produced identical artifacts — seeding may be broken."
    )


def test_bar_size_minutes_known_values() -> None:
    assert _bar_size_minutes("5min") == 5.0
    assert _bar_size_minutes("4H")   == 240.0
    assert _bar_size_minutes("1D")   == 390.0
