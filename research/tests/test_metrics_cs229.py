"""
Tests para models/metrics.py, models/error_analysis.py, models/ablative_analysis.py
(CS229 Tips & Tricks implementation)
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.metrics import (
    compute_classification_metrics,
    compute_bias_variance_verdict,
    format_confusion_matrix,
    aggregate_classification_metrics,
    bias_variance_summary,
    FoldClassificationMetrics,
)


# =====================================================================
# FIXTURES
# =====================================================================

def make_predictions(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
    y_pred = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
    raw = rng.dirichlet([1, 1, 1], size=n)
    signals = np.where(np.abs(y_pred) > 0, y_pred, 0)
    return y_true, y_pred, raw, signals


# =====================================================================
# 1. compute_classification_metrics
# =====================================================================

class TestComputeClassificationMetrics:
    def test_returns_dataclass(self):
        y_true, y_pred, proba, sig = make_predictions()
        m = compute_classification_metrics(y_true, y_pred, proba, [-1, 0, 1])
        assert isinstance(m, FoldClassificationMetrics)

    def test_accuracy_in_0_1(self):
        y_true, y_pred, proba, _ = make_predictions()
        m = compute_classification_metrics(y_true, y_pred, proba, [-1, 0, 1])
        assert 0.0 <= m.accuracy_all <= 1.0

    def test_perfect_predictions(self):
        y = np.array([1, -1, 0, 1, -1])
        m = compute_classification_metrics(y, y, None, [-1, 0, 1])
        assert m.accuracy_all == 1.0
        assert m.f1_macro == 1.0

    def test_f1_macro_in_0_1(self):
        y_true, y_pred, proba, _ = make_predictions()
        m = compute_classification_metrics(y_true, y_pred, proba, [-1, 0, 1])
        assert m.f1_macro is not None
        assert 0.0 <= m.f1_macro <= 1.0

    def test_f1_weighted_in_0_1(self):
        y_true, y_pred, proba, _ = make_predictions()
        m = compute_classification_metrics(y_true, y_pred, proba, [-1, 0, 1])
        assert m.f1_weighted is not None
        assert 0.0 <= m.f1_weighted <= 1.0

    def test_auc_computed_with_proba(self):
        y_true, y_pred, proba, _ = make_predictions(n=200)
        m = compute_classification_metrics(y_true, y_pred, proba, [-1, 0, 1])
        if m.auc_macro is not None:
            assert 0.0 <= m.auc_macro <= 1.0

    def test_auc_none_without_proba(self):
        y_true, y_pred, _, _ = make_predictions()
        m = compute_classification_metrics(y_true, y_pred, None, [-1, 0, 1])
        assert m.auc_macro is None

    def test_per_class_metrics_exist(self):
        y_true, y_pred, proba, _ = make_predictions()
        m = compute_classification_metrics(y_true, y_pred, proba, [-1, 0, 1])
        assert m.precision_long is not None
        assert m.precision_short is not None
        assert m.f1_long is not None

    def test_confusion_matrix_keys(self):
        y_true, y_pred, proba, _ = make_predictions()
        m = compute_classification_metrics(y_true, y_pred, proba, [-1, 0, 1])
        assert m.confusion_matrix is not None
        assert -1 in m.confusion_matrix
        assert 1  in m.confusion_matrix

    def test_confusion_matrix_sums_to_n(self):
        y_true, y_pred, proba, _ = make_predictions(n=100)
        m = compute_classification_metrics(y_true, y_pred, proba, [-1, 0, 1])
        total = sum(
            cnt
            for pred_dict in m.confusion_matrix.values()
            for cnt in pred_dict.values()
        )
        assert total == 100

    def test_accuracy_active_only_on_signals(self):
        y_true = np.array([1, -1, 0, 1, -1, 0])
        y_pred = np.array([1, -1, 0, 0,  1, 0])
        sig    = np.array([1, -1, 0, 1, -1, 0])  # neutral excluded
        m = compute_classification_metrics(y_true, y_pred, None, [-1, 0, 1], signals=sig)
        # active: indices 0,1,3,4 → pred correct: 0 (1==1) and 1 (-1==-1) → acc=2/4=0.5
        assert m.accuracy_active == 0.5

    def test_empty_input_returns_empty_metrics(self):
        m = compute_classification_metrics(
            np.array([]), np.array([]), None, [-1, 0, 1]
        )
        assert m.accuracy_all is None


# =====================================================================
# 2. compute_bias_variance_verdict
# =====================================================================

class TestBiasVarianceVerdict:
    def test_overfitting_large_gap(self):
        # train=0.90, test=0.50 → gap=0.40 > 0.12
        verdict = compute_bias_variance_verdict(0.90, 0.70, 0.50)
        assert verdict == "overfitting"

    def test_underfitting_both_low(self):
        # train=0.38, test=0.36 → both < 0.42
        verdict = compute_bias_variance_verdict(0.38, 0.37, 0.36)
        assert verdict == "underfitting"

    def test_just_right(self):
        # train=0.60, test=0.57 → gap=0.03 < 0.12, both > 0.42
        verdict = compute_bias_variance_verdict(0.60, 0.58, 0.57)
        assert verdict == "just_right"

    def test_unknown_when_none(self):
        assert compute_bias_variance_verdict(None, None, None) == "unknown"
        assert compute_bias_variance_verdict(0.6, 0.5, None)  == "unknown"

    def test_custom_threshold(self):
        # With threshold=0.05, gap=0.08 → overfit
        verdict = compute_bias_variance_verdict(0.60, 0.55, 0.52, overfit_gap=0.05)
        assert verdict == "overfitting"


# =====================================================================
# 3. format_confusion_matrix
# =====================================================================

class TestFormatConfusionMatrix:
    def test_shape(self):
        cm = {-1: {-1: 10, 0: 2, 1: 3}, 0: {-1: 1, 0: 20, 1: 2}, 1: {-1: 2, 0: 1, 1: 15}}
        df = format_confusion_matrix(cm)
        assert df.shape == (3, 3)

    def test_empty_dict(self):
        df = format_confusion_matrix({})
        assert df.empty

    def test_diagonal_dominance_perfect(self):
        cm = {-1: {-1: 10, 0: 0, 1: 0}, 0: {-1: 0, 0: 20, 1: 0}, 1: {-1: 0, 0: 0, 1: 15}}
        df = format_confusion_matrix(cm)
        # Diagonal values: first row col0=10, second row col1=20, third row col2=15
        assert df.iloc[0, 0] == 10
        assert df.iloc[1, 1] == 20
        assert df.iloc[2, 2] == 15


# =====================================================================
# 4. aggregate_classification_metrics
# =====================================================================

class TestAggregateMetrics:
    def test_returns_dataframe(self):
        metrics = []
        for seed in range(3):
            y_true, y_pred, proba, _ = make_predictions(seed=seed)
            m = compute_classification_metrics(y_true, y_pred, proba, [-1, 0, 1])
            metrics.append(m)
        agg = aggregate_classification_metrics(metrics)
        assert isinstance(agg, pd.DataFrame)
        assert "mean" in agg.columns
        assert "std"  in agg.columns

    def test_empty_list(self):
        agg = aggregate_classification_metrics([])
        assert agg.empty


# =====================================================================
# 5. Integration: runner computes extended metrics
# =====================================================================

class TestRunnerExtendedMetrics:
    def _make_run_data(self, n=500, seed=42):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        X = pd.DataFrame(rng.standard_normal((n, 6)), index=idx,
                         columns=[f"f{i}" for i in range(6)])
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
        return X, y, prices

    def test_fold_has_classification_metrics(self):
        from models.walk_forward_runner import WalkForwardRunner, WalkForwardConfig
        X, y, prices = self._make_run_data()
        cfg = WalkForwardConfig(
            train_size=200, test_size=60, embargo=5,
            track_extended_metrics=True, track_importance=False,
        )
        result = WalkForwardRunner(cfg).run(X=X, y=y, prices=prices, all_classes=[-1, 0, 1])
        for fr in result.fold_results:
            assert fr.classification_metrics is not None
            assert isinstance(fr.classification_metrics.f1_macro, float)

    def test_fold_has_bias_variance(self):
        from models.walk_forward_runner import WalkForwardRunner, WalkForwardConfig
        X, y, prices = self._make_run_data(seed=7)
        cfg = WalkForwardConfig(
            train_size=200, test_size=60, embargo=5,
            track_extended_metrics=True, track_importance=False,
        )
        result = WalkForwardRunner(cfg).run(X=X, y=y, prices=prices, all_classes=[-1, 0, 1])
        for fr in result.fold_results:
            assert fr.bias_variance is not None
            assert "verdict" in fr.bias_variance
            assert fr.bias_variance["verdict"] in (
                "just_right", "overfitting", "underfitting", "unknown"
            )

    def test_no_extended_metrics_when_disabled(self):
        from models.walk_forward_runner import WalkForwardRunner, WalkForwardConfig
        X, y, prices = self._make_run_data(seed=3)
        cfg = WalkForwardConfig(
            train_size=200, test_size=60, embargo=5,
            track_extended_metrics=False, track_importance=False,
        )
        result = WalkForwardRunner(cfg).run(X=X, y=y, prices=prices, all_classes=[-1, 0, 1])
        for fr in result.fold_results:
            assert fr.classification_metrics is None
            assert fr.bias_variance is None


# =====================================================================
# 6. ErrorAnalyzer
# =====================================================================

class TestErrorAnalyzer:
    def _make_result(self, seed=0):
        from models.walk_forward_runner import WalkForwardRunner, WalkForwardConfig
        rng = np.random.default_rng(seed)
        n = 500
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        X = pd.DataFrame(rng.standard_normal((n, 6)), index=idx,
                         columns=[f"f{i}" for i in range(6)])
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
        cfg = WalkForwardConfig(
            train_size=200, test_size=60, embargo=5,
            track_extended_metrics=True, track_importance=False,
        )
        result = WalkForwardRunner(cfg).run(X=X, y=y, prices=prices, all_classes=[-1, 0, 1])
        return result, y

    def test_analyze_returns_report(self):
        from models.error_analysis import ErrorAnalyzer
        result, y = self._make_result()
        ea = ErrorAnalyzer()
        report = ea.analyze(result.fold_results, result.oos_sizing, y)
        assert report is not None

    def test_fold_table_not_empty(self):
        from models.error_analysis import ErrorAnalyzer
        result, y = self._make_result()
        report = ErrorAnalyzer().analyze(result.fold_results, result.oos_sizing, y)
        assert not report.fold_table.empty
        assert "fold" in report.fold_table.columns

    def test_confusion_agg_not_empty(self):
        from models.error_analysis import ErrorAnalyzer
        result, y = self._make_result()
        report = ErrorAnalyzer().analyze(result.fold_results, result.oos_sizing, y)
        assert not report.confusion_agg.empty

    def test_top_issues_is_list(self):
        from models.error_analysis import ErrorAnalyzer
        result, y = self._make_result()
        report = ErrorAnalyzer().analyze(result.fold_results, result.oos_sizing, y)
        assert isinstance(report.top_issues, list)
        assert len(report.top_issues) >= 1

    def test_summary_is_string(self):
        from models.error_analysis import ErrorAnalyzer
        result, y = self._make_result()
        report = ErrorAnalyzer().analyze(result.fold_results, result.oos_sizing, y)
        s = report.summary()
        assert isinstance(s, str)
        assert "ERROR ANALYSIS" in s


# =====================================================================
# 7. AblativeAnalyzer (light test — only baseline + one ablation)
# =====================================================================

class TestAblativeAnalyzer:
    def _make_run_data(self, n=600, seed=42):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        X = pd.DataFrame(rng.standard_normal((n, 6)), index=idx,
                         columns=[f"f{i}" for i in range(6)])
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
        atr = pd.Series(np.full(n, 0.002), index=idx)
        return X, y, prices, atr

    def test_ablative_returns_result(self):
        from models.walk_forward_runner import WalkForwardConfig
        from models.ablative_analysis import AblativeAnalyzer, AblativeConfig

        X, y, prices, atr = self._make_run_data()
        base_cfg = WalkForwardConfig(
            train_size=250, test_size=60, embargo=5,
            track_importance=False, track_extended_metrics=True,
        )
        abl_cfg = AblativeConfig(
            ablations=["baseline", "+gmm"],
            verbose=False,
        )
        analyzer = AblativeAnalyzer(base_cfg, abl_cfg)
        result = analyzer.run(X, y, prices, atr, all_classes=[-1, 0, 1])
        assert result is not None
        assert not result.metrics_table.empty

    def test_metrics_table_has_expected_columns(self):
        from models.walk_forward_runner import WalkForwardConfig
        from models.ablative_analysis import AblativeAnalyzer, AblativeConfig

        X, y, prices, atr = self._make_run_data()
        base_cfg = WalkForwardConfig(
            train_size=250, test_size=60, embargo=5,
            track_importance=False,
        )
        abl_cfg = AblativeConfig(ablations=["baseline"], verbose=False)
        result = AblativeAnalyzer(base_cfg, abl_cfg).run(X, y, prices, atr,
                                                          all_classes=[-1, 0, 1])
        cols = result.metrics_table.columns.tolist()
        assert "config" in cols
        assert "sharpe" in cols

    def test_delta_table_baseline_zeros(self):
        from models.walk_forward_runner import WalkForwardConfig
        from models.ablative_analysis import AblativeAnalyzer, AblativeConfig

        X, y, prices, atr = self._make_run_data()
        base_cfg = WalkForwardConfig(
            train_size=250, test_size=60, embargo=5,
            track_importance=False,
        )
        abl_cfg = AblativeConfig(ablations=["baseline"], verbose=False)
        result = AblativeAnalyzer(base_cfg, abl_cfg).run(X, y, prices, atr,
                                                          all_classes=[-1, 0, 1])
        delta = result.delta_table()
        # baseline row deltas should be 0
        baseline_row = delta[delta["config"] == "baseline"]
        if not baseline_row.empty:
            numeric_delta_cols = [c for c in delta.columns if c.startswith("Δ_")]
            for col in numeric_delta_cols:
                val = baseline_row[col].iloc[0]
                if val is not None and not pd.isna(val):
                    assert abs(float(val)) < 1e-9
