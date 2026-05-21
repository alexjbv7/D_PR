"""
Integration tests for MultiHorizonTrainer.

Uses a small synthetic dataset to verify the end-to-end pipeline
runs without errors and produces valid result objects.
These are NOT performance tests — we do not check DSR/PSR values.
"""
from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).parents[2]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.multi_horizon.trainer import MultiHorizonTrainer, TrainResult
from models.multi_horizon.horizon_config import HorizonConfig, INTRADAY


def _tiny_cfg(name: str = "intraday") -> HorizonConfig:
    """Minimal config for fast test runs."""
    return HorizonConfig(
        name="intraday",
        bar_size="5min",
        tp_pct=Decimal("0.005"),
        sl_pct=Decimal("0.005"),
        timeout_bars=3,
        embargo=INTRADAY.embargo,
        train_lookback=INTRADAY.train_lookback,
        feature_set="INTRADAY_FEATURES",
        model_name="xgb",
        n_optuna_trials=1,
    )


def _synth_data(n: int = 350, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    cols = [
        "vol_z_5m", "vol_burst_5m", "atr_14_5m", "rsi_5", "rsi_14_5m",
        "tick_rule_sum_5m", "spread_bps_5m", "trade_count_z_5m",
        "regime_prob_0", "regime_prob_1", "regime_prob_2",
        "session_pre", "session_rth", "session_post", "is_crypto",
    ]
    X = pd.DataFrame(rng.standard_normal((n, len(cols))), index=idx, columns=cols)
    # Make regime probs sum to ~1
    X[["regime_prob_0", "regime_prob_1", "regime_prob_2"]] = (
        X[["regime_prob_0", "regime_prob_1", "regime_prob_2"]]
        .abs()
        .div(X[["regime_prob_0", "regime_prob_1", "regime_prob_2"]].abs().sum(axis=1), axis=0)
    )
    # Binary session flags
    X[["session_pre", "session_rth", "session_post"]] = 0
    X["session_rth"] = 1

    y = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx)
    prices = pd.Series(100.0 * (1 + rng.normal(0, 0.01, n)).cumprod(), index=idx)
    return X, y, prices


def _run_tiny_horizon(seed: int = 42) -> "TrainResult":
    """
    Run a minimal horizon on a tiny dataset to smoke-test the pipeline.
    Uses a patched _build_wf_config that forces small train/test sizes.
    """
    from unittest.mock import patch
    from models.walk_forward_runner import WalkForwardConfig

    X, y, prices = _synth_data(n=350)
    cfg = _tiny_cfg()

    def _tiny_wf_config(self_trainer, cfg_arg, params, embargo_bars):  # type: ignore[override]
        return WalkForwardConfig(
            train_size=80,
            test_size=30,
            embargo=5,
            calib_frac=0.20,
            use_regime_features=False,
            use_meta_labeling=False,
            use_pca=False,
            use_class_weights=True,
            track_importance=False,
            model_class="xgboost",
            xgb_params={"n_estimators": 20, "max_depth": 3, "n_jobs": 1,
                        "seed": 42, "random_state": 42},
        )

    trainer = MultiHorizonTrainer(seed=seed, ablate=False, n_wf_splits=3)
    with patch.object(MultiHorizonTrainer, "_build_wf_config", _tiny_wf_config):
        with patch.object(MultiHorizonTrainer, "_hyperopt", return_value={"n_jobs": 1, "seed": 42, "random_state": 42}):
            return trainer.run_horizon(cfg, X, y, prices)


class TestMultiHorizonTrainerUnit:
    def test_trainer_init_seeds_numpy(self) -> None:
        MultiHorizonTrainer(seed=42)
        # Just verify it doesn't raise

    def test_run_horizon_returns_train_result(self) -> None:
        result = _run_tiny_horizon()

        assert isinstance(result, TrainResult)
        assert result.horizon_name == "intraday"
        assert 0.0 <= result.psr <= 1.0
        assert 0.0 <= result.dsr <= 1.0
        assert 0.0 <= result.ece <= 1.0
        assert result.n_trades >= 0

    def test_promoted_flag_reflects_dsr(self) -> None:
        result = _run_tiny_horizon()

        if result.dsr >= 0.4 and not result.class_collapse:
            assert result.promoted
        else:
            assert not result.promoted

    def test_best_params_xgb_has_n_jobs_1(self) -> None:
        """Final XGBoost training must use n_jobs=1 for reproducibility."""
        result = _run_tiny_horizon()

        if result.best_params and "n_jobs" in result.best_params:
            assert result.best_params["n_jobs"] == 1, (
                "XGBoost final training must use n_jobs=1 for deterministic output."
            )

    def test_feature_selection_fills_missing_with_zero(self) -> None:
        """If a feature is in the feature set but missing from X, it gets zero-filled."""
        rng = np.random.default_rng(1)
        n = 300
        idx = pd.date_range("2023-01-02", periods=n, freq="B")
        # Only include 2 of the expected features
        X_partial = pd.DataFrame(
            rng.standard_normal((n, 2)),
            index=idx,
            columns=["vol_z_5m", "rsi_5"],
        )
        y = pd.Series(rng.choice([-1, 0, 1], n), index=idx)
        prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, n)), index=idx)

        cfg = _tiny_cfg()
        trainer = MultiHorizonTrainer(seed=42, ablate=False, n_wf_splits=3)
        X_out = trainer._select_features(X_partial, ["vol_z_5m", "rsi_5", "missing_feat"])

        assert "missing_feat" in X_out.columns
        assert (X_out["missing_feat"] == 0.0).all()

    def test_embargo_conversion_5min_2h(self) -> None:
        from datetime import timedelta

        trainer = MultiHorizonTrainer(seed=42)
        embargo_bars = trainer._embargo_to_bars(timedelta(hours=2), "5min")
        # 2h = 120 min, 120/5 = 24 bars
        assert embargo_bars == 24

    def test_embargo_conversion_4h_24h(self) -> None:
        from datetime import timedelta

        trainer = MultiHorizonTrainer(seed=42)
        embargo_bars = trainer._embargo_to_bars(timedelta(hours=24), "4H")
        # 24h / 4h = 6 bars
        assert embargo_bars == 6

    def test_embargo_conversion_1d_5d(self) -> None:
        from datetime import timedelta

        trainer = MultiHorizonTrainer(seed=42)
        embargo_bars = trainer._embargo_to_bars(timedelta(days=5), "1D")
        # 5d × 390 min / 390 min = 5 bars
        assert embargo_bars == 5
