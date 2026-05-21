"""
Tests para features/regime_gmm.py
====================================
Cubre:
  - GMMRegimeConfig defaults
  - _build_regime_features: shape, sin NaN, columnas
  - GMMRegimeDetector.fit: is_fitted, sort_order, min_bars
  - GMMRegimeDetector.transform: shape, índice, columnas, rango [0,1]
  - Consistencia cross-fold: regime_0 siempre = menor vol
  - Fallback (no ajustado / pocos datos): features neutras
  - fit_transform end-to-end
  - regime_summary: DataFrame con columnas esperadas
  - select_n_components: devuelve int en [2, max_n]
  - Integración con WalkForwardRunner (use_regime_features=True)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.regime_gmm import (
    GMMRegimeConfig,
    GMMRegimeDetector,
    select_n_components,
    _auto_label,
)


# =====================================================================
# FIXTURES
# =====================================================================

def make_prices(n: int = 600, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Genera close + ATR sintéticos con tendencia y ruido."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    log_ret = rng.normal(0.0002, 0.010, size=n)
    close = pd.Series(
        1.1000 * np.exp(np.cumsum(log_ret)), index=dates, name="close"
    )
    # ATR como volatilidad realizada suavizada
    tr = close.pct_change().abs().ewm(alpha=1/14, adjust=False).mean() * close
    atr = tr.bfill()
    return close, atr


@pytest.fixture(scope="module")
def prices_600():
    return make_prices(n=600)


@pytest.fixture(scope="module")
def prices_150():
    return make_prices(n=150)


# =====================================================================
# TESTS: GMMRegimeConfig
# =====================================================================

class TestGMMRegimeConfig:
    def test_defaults(self):
        cfg = GMMRegimeConfig()
        assert cfg.n_components == 3
        assert cfg.covariance_type == "diag"
        assert cfg.n_init == 5
        assert cfg.sort_by == "vol"
        assert cfg.min_fit_bars == 80

    def test_custom_params(self):
        cfg = GMMRegimeConfig(n_components=4, covariance_type="full", n_init=3)
        assert cfg.n_components == 4
        assert cfg.covariance_type == "full"


# =====================================================================
# TESTS: _build_regime_features
# =====================================================================

class TestBuildRegimeFeatures:
    def test_output_columns(self, prices_600):
        close, atr = prices_600
        df = GMMRegimeDetector._build_regime_features(close, atr)
        expected = {"vol_20", "vol_ratio", "sharpe_roll", "autocorr_1", "atr_z"}
        assert set(df.columns) == expected

    def test_no_nan(self, prices_600):
        close, atr = prices_600
        df = GMMRegimeDetector._build_regime_features(close, atr)
        assert not df.isnull().any().any()

    def test_fewer_rows_than_input(self, prices_600):
        close, atr = prices_600
        df = GMMRegimeDetector._build_regime_features(close, atr)
        assert len(df) < len(close)   # primeras ~60 filas eliminadas por rolling

    def test_index_is_subset_of_input(self, prices_600):
        close, atr = prices_600
        df = GMMRegimeDetector._build_regime_features(close, atr)
        assert df.index.isin(close.index).all()

    def test_vol_20_positive(self, prices_600):
        close, atr = prices_600
        df = GMMRegimeDetector._build_regime_features(close, atr)
        assert (df["vol_20"] > 0).all()

    def test_autocorr_bounded(self, prices_600):
        close, atr = prices_600
        df = GMMRegimeDetector._build_regime_features(close, atr)
        assert df["autocorr_1"].between(-1.05, 1.05).all()


# =====================================================================
# TESTS: GMMRegimeDetector.fit
# =====================================================================

class TestGMMRegimeDetectorFit:
    def test_is_fitted_after_fit(self, prices_600):
        close, atr = prices_600
        det = GMMRegimeDetector()
        det.fit(close, atr)
        assert det.is_fitted is True

    def test_not_fitted_with_few_bars(self):
        close, atr = make_prices(n=50)   # muy pocas barras
        det = GMMRegimeDetector(GMMRegimeConfig(min_fit_bars=80))
        det.fit(close, atr)
        assert det.is_fitted is False

    def test_sort_order_length(self, prices_600):
        close, atr = prices_600
        det = GMMRegimeDetector(GMMRegimeConfig(n_components=3))
        det.fit(close, atr)
        assert len(det._sort_order) == 3

    def test_sort_order_is_permutation(self, prices_600):
        close, atr = prices_600
        det = GMMRegimeDetector(GMMRegimeConfig(n_components=3))
        det.fit(close, atr)
        assert set(det._sort_order) == {0, 1, 2}

    def test_n_components_property(self, prices_600):
        close, atr = prices_600
        det = GMMRegimeDetector(GMMRegimeConfig(n_components=4))
        det.fit(close, atr)
        assert det.n_components_ == 4

    def test_fit_returns_self(self, prices_600):
        close, atr = prices_600
        det = GMMRegimeDetector()
        result = det.fit(close, atr)
        assert result is det


# =====================================================================
# TESTS: GMMRegimeDetector.transform
# =====================================================================

class TestGMMRegimeDetectorTransform:
    @pytest.fixture(autouse=True)
    def fitted_det(self, prices_600):
        close, atr = prices_600
        self.det = GMMRegimeDetector(GMMRegimeConfig(n_components=3))
        self.det.fit(close, atr)
        self.close = close
        self.atr = atr

    def test_output_index_matches_input(self):
        result = self.det.transform(self.close, self.atr)
        assert (result.index == self.close.index).all()

    def test_output_columns_present(self):
        result = self.det.transform(self.close, self.atr)
        for k in range(3):
            assert f"regime_prob_{k}" in result.columns
        assert "regime_label" in result.columns
        assert "regime_entropy" in result.columns

    def test_proba_sum_to_one(self):
        result = self.det.transform(self.close, self.atr)
        prob_cols = [f"regime_prob_{k}" for k in range(3)]
        row_sums = result[prob_cols].sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-6

    def test_proba_in_unit_interval(self):
        result = self.det.transform(self.close, self.atr)
        for k in range(3):
            col = result[f"regime_prob_{k}"]
            assert col.between(0.0, 1.0).all(), f"regime_prob_{k} fuera de [0,1]"

    def test_regime_label_valid(self):
        result = self.det.transform(self.close, self.atr)
        assert result["regime_label"].isin([0, 1, 2]).all()

    def test_entropy_nonneg(self):
        result = self.det.transform(self.close, self.atr)
        assert (result["regime_entropy"] >= -1e-9).all()

    def test_no_nan_in_output(self):
        result = self.det.transform(self.close, self.atr)
        assert not result.isnull().any().any()

    def test_transform_on_subset(self):
        """transform() sobre una ventana más chica (ej. test fold)."""
        sub_close = self.close.iloc[-63:]
        sub_atr = self.atr.iloc[-63:]
        result = self.det.transform(sub_close, sub_atr)
        assert len(result) == 63
        assert not result.isnull().any().any()


# =====================================================================
# TESTS: Fallback (detector no ajustado)
# =====================================================================

class TestNeutralFallback:
    def test_transform_returns_neutral_when_not_fitted(self, prices_600):
        close, atr = prices_600
        det = GMMRegimeDetector(GMMRegimeConfig(n_components=3))
        # Sin llamar fit()
        result = det.transform(close, atr)
        # Probabilidades uniformes = 1/3
        for k in range(3):
            assert (result[f"regime_prob_{k}"] - 1/3).abs().max() < 1e-9

    def test_transform_returns_neutral_with_few_bars(self):
        close, atr = make_prices(n=50)
        det = GMMRegimeDetector(GMMRegimeConfig(min_fit_bars=80))
        det.fit(close, atr)
        assert not det.is_fitted
        result = det.transform(close, atr)
        assert not result.isnull().any().any()


# =====================================================================
# TESTS: Consistencia cross-fold (ordenamiento por vol)
# =====================================================================

class TestCrossFoldConsistency:
    def test_regime0_has_lower_vol_than_regime2(self, prices_600):
        """regime_0 debe ser siempre el de menor volatilidad."""
        close, atr = prices_600
        det = GMMRegimeDetector(GMMRegimeConfig(n_components=3))
        det.fit(close, atr)
        result = det.transform(close, atr)

        # Calcular vol realizada 20d
        log_ret = np.log(close / close.shift(1))
        vol_20 = log_ret.rolling(20).std().reindex(result.index)

        # Vol media en cada régimen
        labels = result["regime_label"]
        vol_0 = vol_20[labels == 0].mean()
        vol_2 = vol_20[labels == 2].mean()

        # regime_0 debe tener menor o igual vol que regime_2
        assert vol_0 <= vol_2 + 1e-5, (
            f"regime_0 vol={vol_0:.5f} > regime_2 vol={vol_2:.5f}: "
            "sort_by='vol' no funcionó"
        )

    def test_two_independent_fits_same_semantic_ordering(self):
        """Dos datasets con vol diferente deben tener el mismo orden semántico."""
        close1, atr1 = make_prices(n=500, seed=1)
        close2, atr2 = make_prices(n=500, seed=2)

        det1 = GMMRegimeDetector(GMMRegimeConfig(n_components=3))
        det2 = GMMRegimeDetector(GMMRegimeConfig(n_components=3))
        det1.fit(close1, atr1)
        det2.fit(close2, atr2)

        r1 = det1.transform(close1, atr1)
        r2 = det2.transform(close2, atr2)

        log_ret1 = np.log(close1 / close1.shift(1))
        log_ret2 = np.log(close2 / close2.shift(1))
        vol1 = log_ret1.rolling(20).std().reindex(r1.index)
        vol2 = log_ret2.rolling(20).std().reindex(r2.index)

        # En ambos casos regime_0 debe tener menor vol que regime_2
        for r, vol in [(r1, vol1), (r2, vol2)]:
            labels = r["regime_label"]
            v0 = vol[labels == 0].mean()
            v2 = vol[labels == 2].mean()
            assert v0 <= v2 + 1e-5


# =====================================================================
# TESTS: fit_transform y regime_summary
# =====================================================================

class TestFitTransformAndSummary:
    def test_fit_transform_same_as_fit_then_transform(self, prices_600):
        close, atr = prices_600
        det1 = GMMRegimeDetector(GMMRegimeConfig(n_components=3, random_state=0))
        det2 = GMMRegimeDetector(GMMRegimeConfig(n_components=3, random_state=0))
        r1 = det1.fit_transform(close, atr)
        det2.fit(close, atr)
        r2 = det2.transform(close, atr)
        pd.testing.assert_frame_equal(r1, r2)

    def test_regime_summary_columns(self, prices_600):
        close, atr = prices_600
        det = GMMRegimeDetector()
        det.fit(close, atr)
        summary = det.regime_summary(close, atr)
        assert not summary.empty
        for col in ["Regime", "N_bars", "Pct_time", "Vol_mean", "Label"]:
            assert col in summary.columns

    def test_regime_summary_n_rows(self, prices_600):
        close, atr = prices_600
        det = GMMRegimeDetector(GMMRegimeConfig(n_components=3))
        det.fit(close, atr)
        summary = det.regime_summary(close, atr)
        assert len(summary) <= 3


# =====================================================================
# TESTS: select_n_components
# =====================================================================

class TestSelectNComponents:
    def test_returns_int(self, prices_600):
        close, atr = prices_600
        n = select_n_components(close, atr, max_n=4)
        assert isinstance(n, int)

    def test_within_range(self, prices_600):
        close, atr = prices_600
        n = select_n_components(close, atr, max_n=5)
        assert 2 <= n <= 5

    def test_few_data_returns_3(self):
        close, atr = make_prices(n=30)
        n = select_n_components(close, atr, max_n=4)
        assert n == 3


# =====================================================================
# TESTS: Integración con WalkForwardRunner
# =====================================================================

class TestWalkForwardIntegration:
    def test_run_with_regime_features(self, prices_600):
        """WalkForwardRunner con use_regime_features=True debe completar sin errores."""
        from models.walk_forward_runner import WalkForwardConfig, WalkForwardRunner
        from examples.pipeline_ml_real_data import build_features, triple_barrier_labels

        close, atr = prices_600
        prices_df = pd.DataFrame({
            "open": close, "high": close * 1.001,
            "low": close * 0.999, "close": close, "volume": 1000.0,
        })
        features = build_features(prices_df)
        atr_al = atr.reindex(features.index)
        close_al = close.reindex(features.index)
        labels = triple_barrier_labels(close_al, atr_al, horizon=5, upper_mult=1.0, lower_mult=1.0)
        valid = labels.dropna().index
        X = features.loc[valid]
        y = labels.loc[valid].astype(int)

        cfg = WalkForwardConfig(
            train_size=200, test_size=50, embargo=3,
            use_regime_features=True, regime_n_components=3,
            use_class_weights=True,
            track_importance=False,
        )
        runner = WalkForwardRunner(cfg)
        result = runner.run(X=X, y=y, prices=close_al.reindex(valid),
                            atr=atr_al.reindex(valid), all_classes=[-1, 0, 1])

        assert result is not None
        assert result.oos_regimes is not None
        assert len(result.oos_regimes) == len(result.oos_signals)

    def test_regime_features_in_feature_names(self, prices_600):
        """Las features de régimen deben aparecer en la importancia cross-fold."""
        from models.walk_forward_runner import WalkForwardConfig, WalkForwardRunner
        from examples.pipeline_ml_real_data import build_features, triple_barrier_labels

        close, atr = prices_600
        prices_df = pd.DataFrame({
            "open": close, "high": close * 1.001,
            "low": close * 0.999, "close": close, "volume": 1000.0,
        })
        features = build_features(prices_df)
        atr_al = atr.reindex(features.index)
        close_al = close.reindex(features.index)
        labels = triple_barrier_labels(close_al, atr_al, horizon=5, upper_mult=1.0, lower_mult=1.0)
        valid = labels.dropna().index
        X = features.loc[valid]
        y = labels.loc[valid].astype(int)

        cfg = WalkForwardConfig(
            train_size=200, test_size=50, embargo=3,
            use_regime_features=True, regime_n_components=3,
            use_class_weights=True,
            track_importance=True,
        )
        runner = WalkForwardRunner(cfg)
        result = runner.run(X=X, y=y, prices=close_al.reindex(valid),
                            atr=atr_al.reindex(valid), all_classes=[-1, 0, 1])

        if not result.feature_importance_agg.empty:
            feat_names = result.feature_importance_agg.index.tolist()
            regime_feats = [f for f in feat_names if f.startswith("regime_")]
            assert len(regime_feats) > 0, "regime_prob_k no aparece en feature importance"


# =====================================================================
# TESTS: _auto_label
# =====================================================================

class TestAutoLabel:
    def test_volatile_label(self):
        assert "Volatil" in _auto_label(0.012, 0.1, 0.1)

    def test_trending_label(self):
        label = _auto_label(0.005, 0.5, 0.1)
        assert "Tendencia" in label

    def test_mean_reversion_label(self):
        assert "reversion" in _auto_label(0.005, 0.0, -0.15).lower()

    def test_lateral_label(self):
        assert "Lateral" in _auto_label(0.004, 0.1, 0.0)
