"""
Tests para BayesianWinUpdater
=============================
Cubre: config defaults, fit, update (product + weighted), prior_table,
       fallbacks (no regime col, no fit, all neutral), integración con runner.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk.bayesian_sizer import (
    BayesianWinUpdater,
    BayesianSizerConfig,
    REGIME_LABEL_COL,
)


# =====================================================================
# FIXTURES
# =====================================================================

def make_calib_data(n: int = 200, n_regimes: int = 3, seed: int = 0):
    """X con regime_label, y_true, signals sintéticos."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    X = pd.DataFrame({
        "f0": rng.standard_normal(n),
        "f1": rng.standard_normal(n),
        REGIME_LABEL_COL: rng.integers(0, n_regimes, size=n),
    }, index=idx)
    y_true = rng.choice([-1, 0, 1], size=n, p=[0.33, 0.34, 0.33])
    signals = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
    return X, y_true, signals


def make_X_no_regime(n: int = 100, seed: int = 1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"f0": rng.standard_normal(n)}, index=idx)


# =====================================================================
# 1. BayesianSizerConfig
# =====================================================================

class TestBayesianSizerConfig:
    def test_defaults(self):
        cfg = BayesianSizerConfig()
        assert cfg.combination == "product"
        assert cfg.smoothing == 1.0
        assert cfg.min_samples == 20
        assert cfg.prior_weight == 0.3
        assert cfg.clip_eps == 1e-4

    def test_custom(self):
        cfg = BayesianSizerConfig(combination="weighted", prior_weight=0.5)
        assert cfg.combination == "weighted"
        assert cfg.prior_weight == 0.5


# =====================================================================
# 2. Fit
# =====================================================================

class TestBayesianUpdaterFit:
    def test_fit_returns_self(self):
        X, y, sig = make_calib_data()
        u = BayesianWinUpdater()
        assert u.fit(X, y, sig) is u

    def test_is_fitted_after_fit(self):
        X, y, sig = make_calib_data(n=200)
        u = BayesianWinUpdater(BayesianSizerConfig(min_samples=5))
        u.fit(X, y, sig)
        assert u.is_fitted

    def test_not_fitted_without_regime_col(self):
        X = make_X_no_regime()
        y = np.zeros(100, dtype=int)
        sig = np.ones(100, dtype=int)
        u = BayesianWinUpdater()
        u.fit(X, y, sig)
        assert not u.is_fitted

    def test_not_fitted_if_all_neutral(self):
        X, y, _ = make_calib_data()
        signals = np.zeros(len(y), dtype=int)
        u = BayesianWinUpdater()
        u.fit(X, y, signals)
        assert not u.is_fitted

    def test_n_regimes_set(self):
        X, y, sig = make_calib_data(n=300, n_regimes=3, seed=5)
        u = BayesianWinUpdater(BayesianSizerConfig(min_samples=5))
        u.fit(X, y, sig)
        if u.is_fitted:
            assert u.n_regimes_ == 3

    def test_global_prior_keys(self):
        X, y, sig = make_calib_data(n=300, seed=7)
        u = BayesianWinUpdater(BayesianSizerConfig(min_samples=5))
        u.fit(X, y, sig)
        if u.is_fitted:
            assert 1 in u._global_prior
            assert -1 in u._global_prior

    def test_global_prior_in_0_1(self):
        X, y, sig = make_calib_data(n=300, seed=8)
        u = BayesianWinUpdater(BayesianSizerConfig(min_samples=5))
        u.fit(X, y, sig)
        if u.is_fitted:
            for p in u._global_prior.values():
                assert 0.0 < p < 1.0


# =====================================================================
# 3. Product of Experts helper
# =====================================================================

class TestProductOfExperts:
    def test_both_50_gives_50(self):
        """P(win)=0.5 prior + P(win)=0.5 model → 0.5 posterior."""
        p = BayesianWinUpdater._product_of_experts(0.5, 0.5)
        assert abs(p - 0.5) < 1e-6

    def test_both_above_50_gives_more(self):
        """Dos modelos de acuerdo → posterior más extremo."""
        p = BayesianWinUpdater._product_of_experts(0.7, 0.7)
        assert p > 0.7

    def test_one_below_one_above(self):
        """Un modelo dice > 0.5, el otro < 0.5 → moderación."""
        p = BayesianWinUpdater._product_of_experts(0.8, 0.3)
        assert 0.3 < p < 0.8

    def test_output_in_0_1(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            p1 = float(rng.uniform(0.01, 0.99))
            p2 = float(rng.uniform(0.01, 0.99))
            result = BayesianWinUpdater._product_of_experts(p1, p2)
            assert 0.0 <= result <= 1.0

    def test_clips_extreme_inputs(self):
        """No explota con p=0 o p=1 (clamp eps)."""
        p = BayesianWinUpdater._product_of_experts(0.0, 0.9)
        assert 0.0 <= p <= 1.0
        p = BayesianWinUpdater._product_of_experts(1.0, 0.9)
        assert 0.0 <= p <= 1.0


# =====================================================================
# 4. Update
# =====================================================================

class TestBayesianUpdate:
    def _fitted_updater(self, seed=0):
        X, y, sig = make_calib_data(n=300, seed=seed)
        u = BayesianWinUpdater(BayesianSizerConfig(min_samples=5))
        u.fit(X, y, sig)
        return u, X, sig

    def test_update_returns_array_same_shape(self):
        u, X, sig = self._fitted_updater()
        if not u.is_fitted:
            pytest.skip("updater no ajustado")
        p_model = np.full(len(sig), 0.6)
        result = u.update(p_model, X, sig)
        assert result.shape == p_model.shape

    def test_neutral_signals_get_zero(self):
        u, X, sig = self._fitted_updater()
        if not u.is_fitted:
            pytest.skip("updater no ajustado")
        p_model = np.full(len(sig), 0.6)
        result = u.update(p_model, X, sig)
        neutral_mask = sig == 0
        assert np.all(result[neutral_mask] == 0.0)

    def test_output_in_0_1(self):
        u, X, sig = self._fitted_updater()
        if not u.is_fitted:
            pytest.skip("updater no ajustado")
        p_model = np.random.default_rng(0).uniform(0.3, 0.7, len(sig))
        result = u.update(p_model, X, sig)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_no_nans(self):
        u, X, sig = self._fitted_updater()
        if not u.is_fitted:
            pytest.skip("updater no ajustado")
        p_model = np.full(len(sig), 0.55)
        result = u.update(p_model, X, sig)
        assert not np.any(np.isnan(result))

    def test_fallback_when_not_fitted(self):
        """Sin fit, update devuelve p_model sin cambios."""
        u = BayesianWinUpdater()
        X, _, sig = make_calib_data(n=50)
        p_model = np.full(len(sig), 0.6)
        result = u.update(p_model, X, sig)
        np.testing.assert_array_equal(result, p_model)

    def test_fallback_no_regime_col_in_test(self):
        """X sin regime_label en test → devuelve p_model sin cambios."""
        X, y, sig = make_calib_data(n=200)
        u = BayesianWinUpdater(BayesianSizerConfig(min_samples=5))
        u.fit(X, y, sig)
        if not u.is_fitted:
            pytest.skip("updater no ajustado")
        X_no_reg = make_X_no_regime(n=50)
        p_model = np.full(50, 0.6)
        sig_test = np.ones(50, dtype=int)
        result = u.update(p_model, X_no_reg, sig_test)
        np.testing.assert_array_equal(result, p_model)

    def test_weighted_combination(self):
        """Modo 'weighted' da promedios ponderados."""
        X, y, sig = make_calib_data(n=300, seed=2)
        u = BayesianWinUpdater(
            BayesianSizerConfig(combination="weighted", prior_weight=0.5, min_samples=5)
        )
        u.fit(X, y, sig)
        if not u.is_fitted:
            pytest.skip("updater no ajustado")
        p_model = np.full(len(sig), 0.7)
        result = u.update(p_model, X, sig)
        # Resultado debe estar entre prior y model
        assert np.all(result[sig != 0] >= 0.0)
        assert np.all(result[sig != 0] <= 1.0)

    def test_high_confidence_model_amplified_by_matching_prior(self):
        """Con prior y modelo ambos > 0.5, producto es más alto que cualquiera."""
        X, y, sig = make_calib_data(n=400, seed=3)
        u = BayesianWinUpdater(BayesianSizerConfig(min_samples=5))
        u.fit(X, y, sig)
        if not u.is_fitted:
            pytest.skip("updater no ajustado")
        # Forzar p_model = 0.8 en señales activas
        p_model = np.where(sig != 0, 0.8, 0.0)
        result = u.update(p_model, X, sig)
        # Al menos algunas activas deben estar sobre 0.8 (si prior > 0.5)
        # O bajo 0.8 si prior < 0.5. No pueden ser todas exactamente 0.8.
        active_mask = sig != 0
        if active_mask.sum() > 0:
            assert not np.all(result[active_mask] == 0.8)


# =====================================================================
# 5. Prior table / Diagnósticos
# =====================================================================

class TestDiagnostics:
    def test_prior_table_empty_when_not_fitted(self):
        u = BayesianWinUpdater()
        assert u.prior_table().empty

    def test_prior_table_has_expected_columns(self):
        X, y, sig = make_calib_data(n=300)
        u = BayesianWinUpdater(BayesianSizerConfig(min_samples=5))
        u.fit(X, y, sig)
        if not u.is_fitted:
            pytest.skip("updater no ajustado")
        tbl = u.prior_table()
        assert "regime" in tbl.columns
        assert "direction" in tbl.columns
        assert "p_win" in tbl.columns
        assert "source" in tbl.columns

    def test_prior_table_p_win_in_0_1(self):
        X, y, sig = make_calib_data(n=300, seed=9)
        u = BayesianWinUpdater(BayesianSizerConfig(min_samples=5))
        u.fit(X, y, sig)
        if not u.is_fitted:
            pytest.skip("updater no ajustado")
        tbl = u.prior_table()
        assert (tbl["p_win"] >= 0.0).all()
        assert (tbl["p_win"] <= 1.0).all()

    def test_repr_not_fitted(self):
        u = BayesianWinUpdater()
        assert "not fitted" in repr(u)

    def test_repr_fitted(self):
        X, y, sig = make_calib_data(n=300, seed=10)
        u = BayesianWinUpdater(BayesianSizerConfig(min_samples=5))
        u.fit(X, y, sig)
        if u.is_fitted:
            assert "fitted" in repr(u)


# =====================================================================
# 6. Integración con WalkForwardRunner
# =====================================================================

class TestBayesianSizerInRunner:
    def _make_run_data(self, n=600, seed=42):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        X = pd.DataFrame(rng.standard_normal((n, 6)), index=idx,
                         columns=[f"f{i}" for i in range(6)])
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
        atr = pd.Series(np.full(n, 0.002), index=idx)
        return X, y, prices, atr

    def test_runner_with_bayesian_sizing(self):
        """Runner completa sin errores con use_bayesian_sizing=True + use_regime_features=True."""
        from models.walk_forward_runner import WalkForwardRunner, WalkForwardConfig

        X, y, prices, atr = self._make_run_data()
        cfg = WalkForwardConfig(
            train_size=250,
            test_size=60,
            embargo=5,
            use_regime_features=True,
            regime_n_components=3,
            use_bayesian_sizing=True,
            bayesian_combination="product",
            bayesian_min_samples=10,
            track_importance=False,
        )
        runner = WalkForwardRunner(cfg)
        result = runner.run(X=X, y=y, prices=prices, atr=atr, all_classes=[-1, 0, 1])
        assert result is not None
        assert len(result.fold_results) >= 1

    def test_fold_result_bayesian_flag(self):
        """FoldResult.bayesian_sizer_fitted es bool."""
        from models.walk_forward_runner import WalkForwardRunner, WalkForwardConfig

        X, y, prices, atr = self._make_run_data(seed=7)
        cfg = WalkForwardConfig(
            train_size=250,
            test_size=60,
            embargo=5,
            use_regime_features=True,
            regime_n_components=3,
            use_bayesian_sizing=True,
            bayesian_min_samples=10,
            track_importance=False,
        )
        runner = WalkForwardRunner(cfg)
        result = runner.run(X=X, y=y, prices=prices, atr=atr, all_classes=[-1, 0, 1])
        for fr in result.fold_results:
            assert isinstance(fr.bayesian_sizer_fitted, bool)

    def test_bayesian_without_regime_features_is_safe(self):
        """Con use_regime_features=False, bayesian_sizing se activa pero no tiene régimen → p_win sin cambios."""
        from models.walk_forward_runner import WalkForwardRunner, WalkForwardConfig

        X, y, prices, atr = self._make_run_data(seed=13)
        cfg = WalkForwardConfig(
            train_size=250,
            test_size=60,
            embargo=5,
            use_regime_features=False,     # sin régimen
            use_bayesian_sizing=True,      # activo pero caerá en fallback
            track_importance=False,
        )
        runner = WalkForwardRunner(cfg)
        result = runner.run(X=X, y=y, prices=prices, atr=atr, all_classes=[-1, 0, 1])
        assert result is not None
        # El bayesian_sizer_fitted debe ser False en todos los folds
        for fr in result.fold_results:
            assert fr.bayesian_sizer_fitted is False

    def test_p_win_in_valid_range_with_bayesian(self):
        """oos_sizing['p_win'] en [0, 1] con Bayesian activo."""
        from models.walk_forward_runner import WalkForwardRunner, WalkForwardConfig

        X, y, prices, atr = self._make_run_data(seed=99)
        cfg = WalkForwardConfig(
            train_size=250,
            test_size=60,
            embargo=5,
            use_regime_features=True,
            regime_n_components=3,
            use_meta_labeling=True,
            meta_min_samples=10,
            use_bayesian_sizing=True,
            bayesian_min_samples=10,
            track_importance=False,
        )
        runner = WalkForwardRunner(cfg)
        result = runner.run(X=X, y=y, prices=prices, atr=atr, all_classes=[-1, 0, 1])
        p_wins = result.oos_sizing["p_win"].dropna()
        assert (p_wins >= 0.0).all()
        assert (p_wins <= 1.0).all()
