"""
Tests para PCADenoiser
======================
Cubre: config defaults, fit, transform, fit_transform, propiedades,
       diagnósticos (loadings, scree), anti-leakage, integración con runner.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.pca_denoiser import PCADenoiser, PCAConfig


# =====================================================================
# FIXTURES
# =====================================================================

def make_df(n: int = 200, n_features: int = 10, seed: int = 42) -> pd.DataFrame:
    """DataFrame de features sin columnas de régimen."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    cols = [f"feat_{i}" for i in range(n_features)]
    data = rng.standard_normal((n, n_features))
    # añadir correlación para que PCA sea no trivial
    data[:, 1] = data[:, 0] * 0.8 + rng.standard_normal(n) * 0.2
    data[:, 2] = data[:, 0] * 0.6 + rng.standard_normal(n) * 0.4
    return pd.DataFrame(data, index=idx, columns=cols)


def make_df_with_regime(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """DataFrame con features normales + columnas regime_prob_k."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    feat_cols = {f"feat_{i}": rng.standard_normal(n) for i in range(8)}
    # columnas de régimen: probabilidades que suman 1
    raw = rng.dirichlet(alpha=[1, 1, 1], size=n)
    regime_cols = {
        "regime_prob_0": raw[:, 0],
        "regime_prob_1": raw[:, 1],
        "regime_prob_2": raw[:, 2],
    }
    return pd.DataFrame({**feat_cols, **regime_cols}, index=idx)


# =====================================================================
# 1. CONFIGURACIÓN
# =====================================================================

class TestPCAConfig:
    def test_defaults(self):
        cfg = PCAConfig()
        assert cfg.n_components == 0.95
        assert cfg.scale is True
        assert cfg.whiten is False
        assert cfg.exclude_prefix == "regime_"
        assert cfg.min_components == 3

    def test_custom(self):
        cfg = PCAConfig(n_components=5, scale=False, min_components=2)
        assert cfg.n_components == 5
        assert cfg.scale is False
        assert cfg.min_components == 2


# =====================================================================
# 2. FIT
# =====================================================================

class TestPCADenoiserFit:
    def test_fit_returns_self(self):
        d = PCADenoiser()
        X = make_df()
        assert d.fit(X) is d

    def test_is_fitted_after_fit(self):
        d = PCADenoiser()
        d.fit(make_df())
        assert d.is_fitted

    def test_not_fitted_initially(self):
        d = PCADenoiser()
        assert not d.is_fitted

    def test_n_components_positive_after_fit(self):
        d = PCADenoiser()
        d.fit(make_df())
        assert d.n_components_ >= 1

    def test_min_components_respected(self):
        cfg = PCAConfig(n_components=0.50, min_components=5)
        d = PCADenoiser(cfg)
        d.fit(make_df(n=200, n_features=10))
        assert d.n_components_ >= 5

    def test_fixed_n_components(self):
        cfg = PCAConfig(n_components=4)
        d = PCADenoiser(cfg)
        d.fit(make_df(n=200, n_features=10))
        assert d.n_components_ == 4

    def test_mle_n_components(self):
        cfg = PCAConfig(n_components="mle")
        d = PCADenoiser(cfg)
        d.fit(make_df(n=200, n_features=10))
        assert d.n_components_ >= 1

    def test_fit_excludes_regime_cols(self):
        d = PCADenoiser()
        X = make_df_with_regime()
        d.fit(X)
        assert "regime_prob_0" not in d._feature_cols
        assert "regime_prob_0" in d._excluded_cols

    def test_scaler_fitted_when_scale_true(self):
        d = PCADenoiser(PCAConfig(scale=True))
        d.fit(make_df())
        assert d._scaler is not None

    def test_no_scaler_when_scale_false(self):
        d = PCADenoiser(PCAConfig(scale=False))
        d.fit(make_df())
        assert d._scaler is None


# =====================================================================
# 3. TRANSFORM
# =====================================================================

class TestPCADenoiserTransform:
    def test_output_shape(self):
        X = make_df(n=200, n_features=10)
        d = PCADenoiser()
        d.fit(X)
        out = d.transform(X)
        assert out.shape[0] == len(X)
        assert out.shape[1] == d.n_components_

    def test_output_index_preserved(self):
        X = make_df()
        d = PCADenoiser()
        d.fit(X)
        out = d.transform(X)
        pd.testing.assert_index_equal(out.index, X.index)

    def test_pca_column_names(self):
        X = make_df(n_features=10)
        d = PCADenoiser()
        d.fit(X)
        out = d.transform(X)
        assert all(c.startswith("pca_") for c in out.columns)

    def test_regime_cols_preserved_in_output(self):
        X = make_df_with_regime()
        d = PCADenoiser()
        d.fit(X)
        out = d.transform(X)
        assert "regime_prob_0" in out.columns
        assert "regime_prob_1" in out.columns
        assert "regime_prob_2" in out.columns

    def test_regime_cols_values_unchanged(self):
        X = make_df_with_regime()
        d = PCADenoiser()
        d.fit(X)
        out = d.transform(X)
        pd.testing.assert_series_equal(
            out["regime_prob_0"].reset_index(drop=True),
            X["regime_prob_0"].reset_index(drop=True),
        )

    def test_no_nans_in_output(self):
        X = make_df()
        d = PCADenoiser()
        d.fit(X)
        out = d.transform(X)
        assert not out.isnull().any().any()

    def test_transform_without_fit_returns_X(self):
        X = make_df()
        d = PCADenoiser()
        out = d.transform(X)  # no fit
        pd.testing.assert_frame_equal(out, X)

    def test_transform_different_subset(self):
        """Anti-leakage: transform en subset distinto al fit."""
        X = make_df(n=300)
        X_fit = X.iloc[:200]
        X_test = X.iloc[200:]
        d = PCADenoiser()
        d.fit(X_fit)
        out = d.transform(X_test)
        assert out.shape[0] == len(X_test)
        assert not out.isnull().any().any()


# =====================================================================
# 4. FIT_TRANSFORM
# =====================================================================

class TestFitTransform:
    def test_fit_transform_equivalent_to_fit_then_transform(self):
        X = make_df()
        d1 = PCADenoiser()
        out1 = d1.fit_transform(X)

        d2 = PCADenoiser()
        d2.fit(X)
        out2 = d2.transform(X)

        pd.testing.assert_frame_equal(out1, out2)

    def test_fit_transform_marks_fitted(self):
        X = make_df()
        d = PCADenoiser()
        d.fit_transform(X)
        assert d.is_fitted


# =====================================================================
# 5. PROPIEDADES
# =====================================================================

class TestProperties:
    def test_explained_variance_ratio_length(self):
        X = make_df()
        d = PCADenoiser()
        d.fit(X)
        assert len(d.explained_variance_ratio_) == d.n_components_

    def test_explained_variance_ratio_sum_le_1(self):
        X = make_df()
        d = PCADenoiser()
        d.fit(X)
        assert d.explained_variance_ratio_.sum() <= 1.0 + 1e-6

    def test_cumulative_variance_95(self):
        """Con n_components=0.95 la varianza acumulada debe ser >= 95%."""
        X = make_df(n=300)
        d = PCADenoiser(PCAConfig(n_components=0.95))
        d.fit(X)
        assert d.cumulative_variance_ >= 0.95 - 1e-6

    def test_n_components_zero_when_not_fitted(self):
        d = PCADenoiser()
        assert d.n_components_ == 0

    def test_explained_variance_empty_when_not_fitted(self):
        d = PCADenoiser()
        assert len(d.explained_variance_ratio_) == 0


# =====================================================================
# 6. DIAGNÓSTICOS
# =====================================================================

class TestDiagnostics:
    def test_loadings_shape(self):
        X = make_df(n_features=10)
        d = PCADenoiser()
        d.fit(X)
        ld = d.loadings()
        assert ld.shape == (len(d._feature_cols), d.n_components_)

    def test_loadings_columns_named_pc(self):
        X = make_df()
        d = PCADenoiser()
        d.fit(X)
        assert all(c.startswith("PC") for c in d.loadings().columns)

    def test_top_loadings_returns_series(self):
        X = make_df()
        d = PCADenoiser()
        d.fit(X)
        top = d.top_loadings(pc=1, top_n=3)
        assert isinstance(top, pd.Series)
        assert len(top) == 3

    def test_scree_summary_columns(self):
        X = make_df()
        d = PCADenoiser()
        d.fit(X)
        scree = d.scree_summary()
        assert "Componente" in scree.columns
        assert "Var. explicada" in scree.columns
        assert "Var. acumulada" in scree.columns

    def test_scree_summary_length(self):
        X = make_df()
        d = PCADenoiser()
        d.fit(X)
        scree = d.scree_summary()
        assert len(scree) == d.n_components_

    def test_repr_before_fit(self):
        d = PCADenoiser()
        r = repr(d)
        assert "not fitted" in r

    def test_repr_after_fit(self):
        d = PCADenoiser()
        d.fit(make_df())
        r = repr(d)
        assert "PCADenoiser(" in r
        assert "var_explicada" in r


# =====================================================================
# 7. INTEGRACIÓN CON WALK-FORWARD RUNNER
# =====================================================================

class TestPCAInRunner:
    def test_runner_with_use_pca_true(self):
        """El runner completa sin errores con use_pca=True."""
        from models.walk_forward_runner import WalkForwardRunner, WalkForwardConfig

        rng = np.random.default_rng(7)
        n = 400
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        X = pd.DataFrame(rng.standard_normal((n, 6)), index=idx,
                         columns=[f"f{i}" for i in range(6)])
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx, name="label")
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)

        cfg = WalkForwardConfig(
            train_size=200,
            test_size=50,
            embargo=5,
            use_pca=True,
            pca_n_components=0.95,
            track_importance=False,
        )
        runner = WalkForwardRunner(cfg)
        result = runner.run(X=X, y=y, prices=prices, all_classes=[-1, 0, 1])
        assert result is not None
        assert len(result.fold_results) >= 1

    def test_runner_pca_colnames_are_pca_k(self):
        """Tras PCA las features de importancia deben llamarse pca_k."""
        from models.walk_forward_runner import WalkForwardRunner, WalkForwardConfig

        rng = np.random.default_rng(8)
        n = 400
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        X = pd.DataFrame(rng.standard_normal((n, 6)), index=idx,
                         columns=[f"f{i}" for i in range(6)])
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)

        cfg = WalkForwardConfig(
            train_size=200,
            test_size=50,
            embargo=5,
            use_pca=True,
            pca_n_components=3,
            track_importance=True,
        )
        runner = WalkForwardRunner(cfg)
        result = runner.run(X=X, y=y, prices=prices, all_classes=[-1, 0, 1])

        if not result.feature_importance_agg.empty:
            cols = result.feature_importance_agg.index.tolist()
            assert any(c.startswith("pca_") for c in cols), \
                f"Esperaba columnas pca_k, encontradas: {cols}"
