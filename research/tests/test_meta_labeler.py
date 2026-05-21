"""
Tests para MetaLabeler
======================
Cubre: create_meta_labels, MetaLabelConfig, fit, predict_p_correct,
       fallback sin fit, integración con WalkForwardRunner.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.meta_labeler import MetaLabeler, MetaLabelConfig, create_meta_labels


# =====================================================================
# FIXTURES
# =====================================================================

def make_signals_and_labels(n: int = 100, seed: int = 0):
    """Genera señales primarias y labels verdaderos."""
    rng = np.random.default_rng(seed)
    signals = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
    y_true = rng.choice([-1, 0, 1], size=n, p=[0.33, 0.34, 0.33])
    return signals, y_true


def make_X_and_proba(n: int = 100, n_feat: int = 6, seed: int = 1):
    """Genera features y probabilidades sintéticas."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    X = pd.DataFrame(rng.standard_normal((n, n_feat)), index=idx,
                     columns=[f"f{i}" for i in range(n_feat)])
    # probabilidades de 3 clases que suman 1
    raw = rng.dirichlet(alpha=[1, 1, 1], size=n)
    return X, raw


# =====================================================================
# 1. create_meta_labels
# =====================================================================

class TestCreateMetaLabels:
    def test_active_mask_shape(self):
        signals, y = make_signals_and_labels(100)
        mask, meta_y = create_meta_labels(signals, y)
        assert mask.shape == (100,)
        assert meta_y.shape == (int(mask.sum()),)

    def test_active_mask_excludes_zeros(self):
        signals = np.array([1, 0, -1, 0, 1])
        y_true  = np.array([1, 1,  1, 0, 0])
        mask, meta_y = create_meta_labels(signals, y_true)
        assert mask.tolist() == [True, False, True, False, True]

    def test_meta_y_binary(self):
        signals, y = make_signals_and_labels(200)
        _, meta_y = create_meta_labels(signals, y)
        assert set(meta_y).issubset({0, 1})

    def test_meta_y_correct_where_match(self):
        signals = np.array([1, -1, 1])
        y_true  = np.array([1, -1, -1])
        mask, meta_y = create_meta_labels(signals, y_true)
        # todos activos; correcto en idx 0 y 1
        assert meta_y.tolist() == [1, 1, 0]

    def test_empty_if_all_neutral(self):
        signals = np.zeros(10, dtype=int)
        y_true  = np.ones(10, dtype=int)
        mask, meta_y = create_meta_labels(signals, y_true)
        assert int(mask.sum()) == 0
        assert len(meta_y) == 0


# =====================================================================
# 2. MetaLabelConfig
# =====================================================================

class TestMetaLabelConfig:
    def test_defaults(self):
        cfg = MetaLabelConfig()
        assert cfg.min_samples == 20
        assert cfg.use_primary_proba is True
        assert cfg.use_original_features is True
        assert "n_estimators" in cfg.xgb_params

    def test_custom_min_samples(self):
        cfg = MetaLabelConfig(min_samples=5)
        assert cfg.min_samples == 5


# =====================================================================
# 3. MetaLabeler.fit
# =====================================================================

class TestMetaLabelerFit:
    def test_fit_returns_self(self):
        X, proba = make_X_and_proba(80)
        signals, y = make_signals_and_labels(80)
        m = MetaLabeler(MetaLabelConfig(min_samples=5))
        assert m.fit(X, proba, signals, y) is m

    def test_is_fitted_when_enough_samples(self):
        X, proba = make_X_and_proba(200)
        signals, y = make_signals_and_labels(200)
        m = MetaLabeler(MetaLabelConfig(min_samples=5))
        m.fit(X, proba, signals, y)
        assert m.is_fitted

    def test_not_fitted_too_few_samples(self):
        X, proba = make_X_and_proba(5)
        signals = np.array([1, 0, 0, 0, 0])
        y       = np.array([1, 0, 1, 0, 0])
        m = MetaLabeler(MetaLabelConfig(min_samples=10))
        m.fit(X, proba, signals, y, class_labels=[-1, 0, 1])
        assert not m.is_fitted

    def test_not_fitted_all_neutral_signals(self):
        X, proba = make_X_and_proba(50)
        signals = np.zeros(50, dtype=int)
        y = np.ones(50, dtype=int)
        m = MetaLabeler(MetaLabelConfig(min_samples=5))
        m.fit(X, proba, signals, y)
        assert not m.is_fitted

    def test_not_fitted_single_meta_class(self):
        """Si todos los meta-labels son 1 (o todos 0), no hay que entrenar."""
        X, proba = make_X_and_proba(50)
        # Señales siempre correctas → meta_y todo 1
        signals = np.ones(50, dtype=int)
        y = np.ones(50, dtype=int)
        m = MetaLabeler(MetaLabelConfig(min_samples=5))
        m.fit(X, proba, signals, y)
        assert not m.is_fitted

    def test_n_active_train_set(self):
        X, proba = make_X_and_proba(200)
        signals, y = make_signals_and_labels(200, seed=5)
        m = MetaLabeler(MetaLabelConfig(min_samples=5))
        m.fit(X, proba, signals, y)
        if m.is_fitted:
            assert m.n_active_train > 0
            n_active_expected = int((signals != 0).sum())
            assert m.n_active_train == n_active_expected


# =====================================================================
# 4. MetaLabeler.predict_p_correct
# =====================================================================

class TestPredictPCorrect:
    def _fit_meta(self, seed=0):
        X, proba = make_X_and_proba(200, seed=seed)
        signals, y = make_signals_and_labels(200, seed=seed)
        m = MetaLabeler(MetaLabelConfig(min_samples=5))
        m.fit(X, proba, signals, y, class_labels=[-1, 0, 1])
        return m, X, proba, signals

    def test_returns_none_when_not_fitted(self):
        X, proba = make_X_and_proba(20)
        signals = np.zeros(20, dtype=int)
        m = MetaLabeler()
        result = m.predict_p_correct(X, proba, signals)
        assert result is None

    def test_output_shape(self):
        m, X, proba, signals = self._fit_meta()
        if not m.is_fitted:
            pytest.skip("meta-labeler no entrenado con estos datos")
        result = m.predict_p_correct(X, proba, signals)
        assert result is not None
        assert result.shape == (len(X),)

    def test_neutral_signals_get_zero(self):
        m, X, proba, signals = self._fit_meta()
        if not m.is_fitted:
            pytest.skip("meta-labeler no entrenado con estos datos")
        result = m.predict_p_correct(X, proba, signals)
        neutral_mask = signals == 0
        assert np.all(result[neutral_mask] == 0.0)

    def test_active_signals_in_0_1(self):
        m, X, proba, signals = self._fit_meta()
        if not m.is_fitted:
            pytest.skip("meta-labeler no entrenado con estos datos")
        result = m.predict_p_correct(X, proba, signals)
        active_mask = signals != 0
        if active_mask.sum() > 0:
            assert np.all(result[active_mask] >= 0.0)
            assert np.all(result[active_mask] <= 1.0)

    def test_no_nans(self):
        m, X, proba, signals = self._fit_meta()
        if not m.is_fitted:
            pytest.skip("meta-labeler no entrenado con estos datos")
        result = m.predict_p_correct(X, proba, signals)
        assert not np.any(np.isnan(result))

    def test_all_neutral_returns_zeros(self):
        m, X, proba, signals = self._fit_meta()
        if not m.is_fitted:
            pytest.skip("meta-labeler no entrenado con estos datos")
        neutral_signals = np.zeros(len(X), dtype=int)
        result = m.predict_p_correct(X, proba, neutral_signals)
        assert np.all(result == 0.0)


# =====================================================================
# 5. Diagnósticos y repr
# =====================================================================

class TestDiagnostics:
    def test_repr_not_fitted(self):
        m = MetaLabeler()
        r = repr(m)
        assert "not fitted" in r

    def test_repr_fitted(self):
        X, proba = make_X_and_proba(200)
        signals, y = make_signals_and_labels(200)
        m = MetaLabeler(MetaLabelConfig(min_samples=5))
        m.fit(X, proba, signals, y)
        if m.is_fitted:
            r = repr(m)
            assert "fitted" in r

    def test_feature_importance_none_when_not_fitted(self):
        m = MetaLabeler()
        assert m.feature_importance() is None

    def test_feature_importance_series_when_fitted(self):
        X, proba = make_X_and_proba(200)
        signals, y = make_signals_and_labels(200)
        m = MetaLabeler(MetaLabelConfig(min_samples=5))
        m.fit(X, proba, signals, y)
        if m.is_fitted:
            imp = m.feature_importance()
            assert isinstance(imp, pd.Series)
            assert len(imp) > 0


# =====================================================================
# 6. Integración con WalkForwardRunner
# =====================================================================

class TestMetaLabelInRunner:
    def test_runner_with_meta_labeling_true(self):
        """El runner completa sin errores con use_meta_labeling=True."""
        from models.walk_forward_runner import WalkForwardRunner, WalkForwardConfig

        rng = np.random.default_rng(99)
        n = 500
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        X = pd.DataFrame(rng.standard_normal((n, 6)), index=idx,
                         columns=[f"f{i}" for i in range(6)])
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)

        cfg = WalkForwardConfig(
            train_size=200,
            test_size=60,
            embargo=5,
            use_meta_labeling=True,
            meta_min_samples=10,
            track_importance=False,
        )
        runner = WalkForwardRunner(cfg)
        result = runner.run(X=X, y=y, prices=prices, all_classes=[-1, 0, 1])
        assert result is not None
        assert len(result.fold_results) >= 1

    def test_fold_result_has_meta_trained_flag(self):
        """FoldResult.meta_labeler_trained existe y es bool."""
        from models.walk_forward_runner import WalkForwardRunner, WalkForwardConfig

        rng = np.random.default_rng(77)
        n = 500
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        X = pd.DataFrame(rng.standard_normal((n, 6)), index=idx,
                         columns=[f"f{i}" for i in range(6)])
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)

        cfg = WalkForwardConfig(
            train_size=200,
            test_size=60,
            embargo=5,
            use_meta_labeling=True,
            meta_min_samples=10,
            track_importance=False,
        )
        runner = WalkForwardRunner(cfg)
        result = runner.run(X=X, y=y, prices=prices, all_classes=[-1, 0, 1])
        for fr in result.fold_results:
            assert isinstance(fr.meta_labeler_trained, bool)

    def test_runner_oos_sizing_p_win_in_range(self):
        """Con meta-labeling, p_win en oos_sizing debe estar en [0, 1]."""
        from models.walk_forward_runner import WalkForwardRunner, WalkForwardConfig

        rng = np.random.default_rng(55)
        n = 500
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        X = pd.DataFrame(rng.standard_normal((n, 6)), index=idx,
                         columns=[f"f{i}" for i in range(6)])
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)

        cfg = WalkForwardConfig(
            train_size=200,
            test_size=60,
            embargo=5,
            use_meta_labeling=True,
            meta_min_samples=10,
            track_importance=False,
        )
        runner = WalkForwardRunner(cfg)
        result = runner.run(X=X, y=y, prices=prices, all_classes=[-1, 0, 1])
        p_win_vals = result.oos_sizing["p_win"].dropna()
        assert (p_win_vals >= 0.0).all()
        assert (p_win_vals <= 1.0).all()
