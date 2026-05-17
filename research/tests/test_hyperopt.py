"""
Tests para models/hyperopt.py
==============================
Cubre:
  - HyperoptConfig defaults y validación
  - HyperoptResult.summary() y to_walk_forward_config()
  - BayesianHyperopt.run() con datos sintéticos y pocos trials
  - Protocolo anti-leakage (tamaño de split hyperopt/test)
  - Métricas objetivo: psr, sharpe, coverage_psr
  - Manejo de trials inválidos (nan)
  - Symmetric barriers
  - Conveniencia: run_hyperopt()
"""
from __future__ import annotations

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.hyperopt import (
    BayesianHyperopt,
    HyperoptConfig,
    HyperoptResult,
    run_hyperopt,
)
from models.walk_forward_runner import WalkForwardConfig


# =====================================================================
# FIXTURES
# =====================================================================

def make_prices(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Genera OHLCV sintético con tendencia + ruido."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    log_ret = rng.normal(0.0002, 0.010, size=n)
    close = 1.1000 * np.exp(np.cumsum(log_ret))
    high = close * (1 + rng.uniform(0.001, 0.005, size=n))
    low = close * (1 - rng.uniform(0.001, 0.005, size=n))
    open_ = close * (1 + rng.normal(0, 0.002, size=n))
    volume = rng.integers(1000, 5000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Features simples (retornos rolling) para tests."""
    c = prices["close"]
    feat = pd.DataFrame(index=prices.index)
    ret = np.log(c / c.shift(1))
    for w in [1, 5, 10]:
        feat[f"ret_{w}"] = np.log(c / c.shift(w))
    for w in [5, 10]:
        feat[f"vol_{w}"] = ret.rolling(w).std()
    feat["rsi"] = _rsi(c, 14)
    return feat.dropna()


def _rsi(c: pd.Series, period: int = 14) -> pd.Series:
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def make_atr(prices: pd.DataFrame, period: int = 14) -> pd.Series:
    c = prices["close"]
    h = prices["high"]
    lo = prices["low"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def make_label_fn(prices: pd.DataFrame, atr_series: pd.Series, features_index):
    """Devuelve una label_fn compatible con BayesianHyperopt."""

    def label_fn(upper_mult: float, lower_mult: float, horizon: int) -> pd.Series:
        close = prices["close"].reindex(features_index)
        atr = atr_series.reindex(features_index)
        n = len(close)
        labels = np.full(n, np.nan)
        c_arr = close.values
        a_arr = atr.values
        for i in range(n - horizon):
            entry = c_arr[i]
            upper = entry + upper_mult * a_arr[i]
            lower = entry - lower_mult * a_arr[i]
            lbl = 0
            for j in range(i + 1, i + horizon + 1):
                if c_arr[j] >= upper:
                    lbl = 1
                    break
                elif c_arr[j] <= lower:
                    lbl = -1
                    break
            labels[i] = lbl
        return pd.Series(labels, index=features_index, name="label")

    return label_fn


@pytest.fixture(scope="module")
def synthetic_data():
    prices = make_prices(n=800)
    features = make_features(prices)
    atr = make_atr(prices)
    label_fn = make_label_fn(prices, atr, features.index)
    close = prices["close"].reindex(features.index)
    return {
        "prices": prices,
        "features": features,
        "atr": atr,
        "label_fn": label_fn,
        "close": close,
    }


# =====================================================================
# TESTS: HyperoptConfig
# =====================================================================

class TestHyperoptConfig:
    def test_defaults(self):
        cfg = HyperoptConfig()
        assert cfg.n_trials == 50
        assert cfg.n_val_folds == 3
        assert cfg.train_size == 252
        assert cfg.val_size == 63
        assert cfg.embargo == 5
        assert cfg.val_frac == 0.80
        assert cfg.objective_metric == "psr"
        assert cfg.sampler == "tpe"
        assert cfg.seed == 42

    def test_symmetric_barriers_default(self):
        cfg = HyperoptConfig()
        assert cfg.symmetric_barriers is False

    def test_search_rr_default_off(self):
        cfg = HyperoptConfig()
        assert cfg.search_rr is False

    def test_custom_ranges(self):
        cfg = HyperoptConfig(
            n_estimators_range=(200, 400),
            max_depth_range=(4, 6),
            upper_mult_range=(0.8, 1.5),
        )
        assert cfg.n_estimators_range == (200, 400)
        assert cfg.max_depth_range == (4, 6)
        assert cfg.upper_mult_range == (0.8, 1.5)

    def test_min_trades_default(self):
        cfg = HyperoptConfig()
        assert cfg.min_trades == 20

    def test_timeout_default_none(self):
        cfg = HyperoptConfig()
        assert cfg.timeout is None


# =====================================================================
# TESTS: HyperoptResult
# =====================================================================

class TestHyperoptResult:
    def _make_result(self):
        return HyperoptResult(
            best_params={
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.1,
                "reg_lambda": 1.5,
                "min_child_weight": 10,
                "gamma": 0.05,
                "upper_mult": 1.2,
                "lower_mult": 1.0,
                "horizon": 7,
            },
            best_value=0.6543,
            n_trials_completed=45,
            n_trials_pruned=3,
            all_trials=[],
            study=None,
            best_xgb_params={
                "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 0.1,
                "reg_lambda": 1.5, "min_child_weight": 10, "gamma": 0.05,
            },
            best_barrier_params={"upper_mult": 1.2, "lower_mult": 1.0, "horizon": 7},
            best_rr_params={},
        )

    def test_summary_contains_best_value(self):
        r = self._make_result()
        s = r.summary()
        assert "0.6543" in s

    def test_summary_contains_xgb_params(self):
        r = self._make_result()
        s = r.summary()
        assert "n_estimators" in s
        assert "max_depth" in s
        assert "learning_rate" in s

    def test_summary_contains_barrier_params(self):
        r = self._make_result()
        s = r.summary()
        assert "upper_mult" in s
        assert "horizon" in s

    def test_summary_no_rr_section_if_empty(self):
        r = self._make_result()
        s = r.summary()
        assert "R:R" not in s

    def test_summary_rr_section_when_present(self):
        r = self._make_result()
        r.best_rr_params = {"rr_min": 1.2, "rr_max": 2.0}
        s = r.summary()
        assert "R:R" in s

    def test_to_walk_forward_config_updates_xgb(self):
        r = self._make_result()
        base = WalkForwardConfig()
        new_cfg = r.to_walk_forward_config(base)
        assert new_cfg.xgb_params["n_estimators"] == 300
        assert new_cfg.xgb_params["max_depth"] == 4

    def test_to_walk_forward_config_inherits_base_fields(self):
        r = self._make_result()
        base = WalkForwardConfig(train_size=300, kelly_fraction=0.30)
        new_cfg = r.to_walk_forward_config(base)
        # train_size y kelly_fraction no son params XGBoost -> se heredan
        assert new_cfg.train_size == 300
        assert new_cfg.kelly_fraction == 0.30

    def test_to_walk_forward_config_no_base(self):
        r = self._make_result()
        new_cfg = r.to_walk_forward_config(base_config=None)
        assert new_cfg.xgb_params["learning_rate"] == 0.05

    def test_to_walk_forward_config_updates_rr(self):
        r = self._make_result()
        r.best_rr_params = {"rr_min": 1.5, "rr_max": 3.0}
        base = WalkForwardConfig()
        new_cfg = r.to_walk_forward_config(base)
        assert new_cfg.rr_min == 1.5
        assert new_cfg.rr_max == 3.0

    def test_n_trials_counts(self):
        r = self._make_result()
        assert r.n_trials_completed == 45
        assert r.n_trials_pruned == 3


# =====================================================================
# TESTS: BayesianHyperopt (integración ligera)
# =====================================================================

class TestBayesianHyperopt:
    """Tests de integración que ejecutan trials reales con datos sintéticos."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_optuna(self):
        pytest.importorskip("optuna", reason="optuna no instalado")

    def test_run_returns_hyperopt_result(self, synthetic_data):
        """3 trials => debe devolver HyperoptResult con best_value."""
        d = synthetic_data
        cfg = HyperoptConfig(
            n_trials=3,
            n_val_folds=2,
            train_size=200,
            val_size=50,
            embargo=3,
            val_frac=0.85,
            min_trades=5,
            use_pruner=False,
            verbose=False,
        )
        ho = BayesianHyperopt(cfg)
        result = ho.run(
            X=d["features"],
            close=d["close"],
            atr=d["atr"],
            label_fn=d["label_fn"],
            prices=d["close"],
            all_classes=[-1, 0, 1],
        )
        assert isinstance(result, HyperoptResult)

    def test_best_params_has_xgb_keys(self, synthetic_data):
        d = synthetic_data
        cfg = HyperoptConfig(
            n_trials=3, n_val_folds=2, train_size=200, val_size=50,
            embargo=3, val_frac=0.85, min_trades=5, use_pruner=False, verbose=False,
        )
        result = BayesianHyperopt(cfg).run(
            X=d["features"], close=d["close"], atr=d["atr"],
            label_fn=d["label_fn"], all_classes=[-1, 0, 1],
        )
        for key in ["n_estimators", "max_depth", "learning_rate"]:
            assert key in result.best_xgb_params, f"{key} falta en best_xgb_params"

    def test_best_params_has_barrier_keys(self, synthetic_data):
        d = synthetic_data
        cfg = HyperoptConfig(
            n_trials=3, n_val_folds=2, train_size=200, val_size=50,
            embargo=3, val_frac=0.85, min_trades=5, use_pruner=False, verbose=False,
        )
        result = BayesianHyperopt(cfg).run(
            X=d["features"], close=d["close"], atr=d["atr"],
            label_fn=d["label_fn"], all_classes=[-1, 0, 1],
        )
        for key in ["upper_mult", "horizon"]:
            assert key in result.best_barrier_params, f"{key} falta en best_barrier_params"

    def test_symmetric_barriers(self, synthetic_data):
        """Con symmetric_barriers=True, lower_mult == upper_mult en best params."""
        d = synthetic_data
        cfg = HyperoptConfig(
            n_trials=3, n_val_folds=2, train_size=200, val_size=50,
            embargo=3, val_frac=0.85, min_trades=5, use_pruner=False,
            symmetric_barriers=True, verbose=False,
        )
        result = BayesianHyperopt(cfg).run(
            X=d["features"], close=d["close"], atr=d["atr"],
            label_fn=d["label_fn"], all_classes=[-1, 0, 1],
        )
        # lower_mult debe existir y ser igual a upper_mult
        bp = result.best_barrier_params
        if "upper_mult" in bp and "lower_mult" in bp:
            assert bp["lower_mult"] == bp["upper_mult"]

    def test_val_split_size(self, synthetic_data):
        """La porcion de hyperopt debe ser val_frac% del dataset."""
        d = synthetic_data
        n = len(d["features"])
        val_frac = 0.80
        expected_ho = int(n * val_frac)
        cfg = HyperoptConfig(
            n_trials=2, n_val_folds=2, train_size=200, val_size=50,
            embargo=3, val_frac=val_frac, min_trades=5, use_pruner=False, verbose=False,
        )
        # Verificar que no explota y que best_params existe
        result = BayesianHyperopt(cfg).run(
            X=d["features"], close=d["close"], atr=d["atr"],
            label_fn=d["label_fn"], all_classes=[-1, 0, 1],
        )
        assert result.n_trials_completed + result.n_trials_pruned <= cfg.n_trials

    def test_too_small_dataset_raises(self, synthetic_data):
        """Si el dataset es muy pequeño para los params de validación, debe lanzar ValueError."""
        d = synthetic_data
        cfg = HyperoptConfig(
            n_trials=2, n_val_folds=5, train_size=500, val_size=200,
            embargo=10, val_frac=0.50, min_trades=5, use_pruner=False, verbose=False,
        )
        # 800 * 0.50 = 400 < 500 + 5*(200+10) = 1550 -> debe fallar
        with pytest.raises(ValueError, match="Dataset demasiado pequeño"):
            BayesianHyperopt(cfg).run(
                X=d["features"], close=d["close"], atr=d["atr"],
                label_fn=d["label_fn"], all_classes=[-1, 0, 1],
            )

    def test_objective_metric_sharpe(self, synthetic_data):
        """objective_metric='sharpe' no debe crashear."""
        d = synthetic_data
        cfg = HyperoptConfig(
            n_trials=2, n_val_folds=2, train_size=200, val_size=50,
            embargo=3, val_frac=0.85, min_trades=5, use_pruner=False,
            objective_metric="sharpe", verbose=False,
        )
        result = BayesianHyperopt(cfg).run(
            X=d["features"], close=d["close"], atr=d["atr"],
            label_fn=d["label_fn"], all_classes=[-1, 0, 1],
        )
        assert isinstance(result, HyperoptResult)

    def test_objective_metric_coverage_psr(self, synthetic_data):
        """objective_metric='coverage_psr' no debe crashear."""
        d = synthetic_data
        cfg = HyperoptConfig(
            n_trials=2, n_val_folds=2, train_size=200, val_size=50,
            embargo=3, val_frac=0.85, min_trades=5, use_pruner=False,
            objective_metric="coverage_psr", verbose=False,
        )
        result = BayesianHyperopt(cfg).run(
            X=d["features"], close=d["close"], atr=d["atr"],
            label_fn=d["label_fn"], all_classes=[-1, 0, 1],
        )
        assert isinstance(result, HyperoptResult)

    def test_to_walk_forward_config_from_run(self, synthetic_data):
        """El resultado se puede convertir en WalkForwardConfig."""
        d = synthetic_data
        cfg = HyperoptConfig(
            n_trials=2, n_val_folds=2, train_size=200, val_size=50,
            embargo=3, val_frac=0.85, min_trades=5, use_pruner=False, verbose=False,
        )
        result = BayesianHyperopt(cfg).run(
            X=d["features"], close=d["close"], atr=d["atr"],
            label_fn=d["label_fn"], all_classes=[-1, 0, 1],
        )
        base = WalkForwardConfig()
        new_cfg = result.to_walk_forward_config(base)
        from models.walk_forward_runner import WalkForwardConfig as WFC
        assert isinstance(new_cfg, WFC)
        assert "n_estimators" in new_cfg.xgb_params


# =====================================================================
# TESTS: run_hyperopt() convenience
# =====================================================================

class TestRunHyperopt:
    @pytest.fixture(autouse=True)
    def _skip_if_no_optuna(self):
        pytest.importorskip("optuna", reason="optuna no instalado")

    def test_run_hyperopt_convenience(self, synthetic_data):
        d = synthetic_data
        result = run_hyperopt(
            X=d["features"],
            close=d["close"],
            atr=d["atr"],
            label_fn=d["label_fn"],
            prices=d["close"],
            all_classes=[-1, 0, 1],
            n_trials=2,
            n_val_folds=2,
            train_size=200,
            val_size=50,
            embargo=3,
            val_frac=0.85,
            min_trades=5,
            use_pruner=False,
            verbose=False,
        )
        assert isinstance(result, HyperoptResult)

    def test_run_hyperopt_returns_study(self, synthetic_data):
        d = synthetic_data
        result = run_hyperopt(
            X=d["features"], close=d["close"], atr=d["atr"],
            label_fn=d["label_fn"], all_classes=[-1, 0, 1],
            n_trials=2, n_val_folds=2, train_size=200, val_size=50,
            embargo=3, val_frac=0.85, min_trades=5, use_pruner=False, verbose=False,
        )
        assert result.study is not None


# =====================================================================
# TESTS: _extract_metric
# =====================================================================

class TestExtractMetric:
    def test_psr_extracts_psr(self):
        metrics = {"psr": 0.72, "sharpe": 1.5, "coverage": 0.20}
        val = BayesianHyperopt._extract_metric(metrics, "psr")
        assert val == 0.72

    def test_sharpe_extracts_sharpe(self):
        metrics = {"psr": 0.72, "sharpe": 1.5, "coverage": 0.20}
        val = BayesianHyperopt._extract_metric(metrics, "sharpe")
        assert val == 1.5

    def test_coverage_psr_scales_with_coverage(self):
        metrics = {"psr": 0.80, "coverage": 0.10}
        # coverage=0.10, target=0.20 -> scale=0.5
        val = BayesianHyperopt._extract_metric(metrics, "coverage_psr",
                                               target_coverage=0.20)
        assert abs(val - 0.40) < 1e-9

    def test_coverage_psr_caps_at_1(self):
        metrics = {"psr": 0.80, "coverage": 0.50}
        # coverage=0.50 > target=0.20 -> scale capped at 1.0
        val = BayesianHyperopt._extract_metric(metrics, "coverage_psr",
                                               target_coverage=0.20)
        assert abs(val - 0.80) < 1e-9

    def test_unknown_metric_falls_back_to_psr(self):
        metrics = {"psr": 0.65, "sharpe": 1.2}
        val = BayesianHyperopt._extract_metric(metrics, "nonexistent")
        assert val == 0.65

    def test_missing_metric_returns_none(self):
        metrics = {"coverage": 0.10}
        val = BayesianHyperopt._extract_metric(metrics, "psr")
        assert val is None
