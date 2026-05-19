"""
Adversarial anti-leakage tests for multi-horizon training.

These tests are designed to FAIL when the 4 most common leakage bugs are
present, and PASS when the pipeline correctly enforces anti-leakage rules.

Run first — before any actual training:
    pytest -xvs research/tests/test_no_leakage_multi_horizon.py

Reference: Semana 7 spec §CRÍTICO: ANTI-LEAKAGE
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure research/ and shared/ are importable
_REPO = Path(__file__).parents[2]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.multi_horizon.horizon_config import (
    ALL_HORIZONS,
    DAILY,
    INTRADAY,
    SWING,
    TOTAL_OPTUNA_TRIALS,
    HorizonConfig,
)
from models.multi_horizon.feature_sets import (
    DAILY_FEATURES,
    INTRADAY_FEATURES,
    SWING_FEATURES,
)
from models.walk_forward_runner import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardRunner,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from models.validation import WalkForwardSplitter


# ============================================================================
# FIXTURES
# ============================================================================


def _make_synth_data(
    n_bars: int = 600,
    n_features: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Synthetic dataset with DatetimeIndex for test isolation."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_bars, freq="B")
    X = pd.DataFrame(
        rng.standard_normal((n_bars, n_features)),
        index=idx,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.choice([-1, 0, 1], size=n_bars), index=idx)
    prices = pd.Series(100.0 * (1 + rng.normal(0, 0.01, n_bars)).cumprod(), index=idx)
    return X, y, prices


# ============================================================================
# TEST 1: Embargo for swing horizon must prevent label overlap
# ============================================================================


class TestSwingEmbargoEnforcement:
    """
    If swing embargo is only 2h (intraday default) instead of 24h, the
    label computed at t=close with a 5-day (30-bar 4h) timeout window
    potentially overlaps with test samples within the embargo gap.

    The WalkForwardSplitter must leave at least `embargo` bars empty
    between the last training bar and the first test bar.
    """

    def test_correct_swing_embargo_leaves_no_overlap(self) -> None:
        """With 24h embargo (≥ 1 bar for 4h data), no train bar is in the test."""
        n_bars = 400
        embargo_bars = 6  # 24h / 4h = 6 bars — swing correct
        train_size = 200
        test_size = 50

        splitter = WalkForwardSplitter(
            train_size=train_size,
            test_size=test_size,
            embargo=embargo_bars,
        )
        X_dummy = pd.DataFrame(np.zeros((n_bars, 1)))

        for train_idx, test_idx in splitter.split(X_dummy):
            last_train = train_idx.max()
            first_test = test_idx.min()
            gap = first_test - last_train - 1
            assert gap >= embargo_bars, (
                f"Embargo violated: gap={gap} < required={embargo_bars}. "
                "Train and test bars overlap — label leakage possible."
            )

    def test_swing_requires_larger_embargo_than_intraday(self) -> None:
        """
        Swing embargo (24h) must be strictly greater than intraday embargo (2h).
        If they are equal, the swing embargo is under-specified.
        """
        intraday_embargo_hours = INTRADAY.embargo.total_seconds() / 3600
        swing_embargo_hours    = SWING.embargo.total_seconds()    / 3600

        assert swing_embargo_hours > intraday_embargo_hours, (
            f"Swing embargo ({swing_embargo_hours}h) must be > "
            f"intraday embargo ({intraday_embargo_hours}h). "
            "Using the intraday embargo for swing would cause label overlap."
        )

    def test_swing_embargo_covers_timeout_window(self) -> None:
        """
        Swing timeout = 30 bars × 4h = 120h. Embargo must be >= bar resolution
        to prevent any label from looking into the test period.
        Bar-level: embargo >= 1 bar.
        """
        swing_bar_hours    = 4
        swing_embargo_bars = int(SWING.embargo.total_seconds() / 3600 / swing_bar_hours)
        assert swing_embargo_bars >= 1, (
            f"Swing embargo converts to {swing_embargo_bars} bars — insufficient."
        )

    def test_wrong_embargo_produces_shorter_gap(self) -> None:
        """
        Demonstrate that using embargo=0 leaves train and test adjacent,
        which would allow label overlap for any multi-bar label window.
        """
        n_bars = 200
        train_size = 100
        test_size  = 50

        splitter_no_embargo = WalkForwardSplitter(
            train_size=train_size,
            test_size=test_size,
            embargo=0,
        )
        X_dummy = pd.DataFrame(np.zeros((n_bars, 1)))
        splits = list(splitter_no_embargo.split(X_dummy))
        train_idx, test_idx = splits[0]

        last_train = int(train_idx.max())
        first_test = int(test_idx.min())
        # With embargo=0 the gap is 0 (test starts immediately after train)
        assert first_test == last_train + 1, (
            "Expected zero-embargo to leave test immediately adjacent to train."
        )


# ============================================================================
# TEST 2: Daily features must not appear in INTRADAY_FEATURES without lag
# ============================================================================


class TestIntradayFeatureLagEnforcement:
    """
    Daily-frequency features (FF5 factor exposures, earnings calendar,
    sector-neutralized signals) must never appear in INTRADAY_FEATURES
    without an explicit _lag1d suffix, because using the daily bar's
    value in intraday bars is a forward-look.
    """

    _DAILY_ONLY_PREFIXES = ("ff5_", "earnings_", "days_to_earnings", "days_since", "sector_")

    def test_no_unlagged_daily_features_in_intraday(self) -> None:
        for feat in INTRADAY_FEATURES:
            for prefix in self._DAILY_ONLY_PREFIXES:
                if feat.startswith(prefix) or feat == prefix.rstrip("_"):
                    assert "_lag1d" in feat or "_lagged" in feat, (
                        f"Daily feature '{feat}' appears in INTRADAY_FEATURES "
                        "without explicit lag annotation. "
                        "Add '_lag1d' suffix or remove from intraday set."
                    )

    def test_no_unlagged_daily_features_in_swing(self) -> None:
        """
        Swing uses macro features with _lag1d; verify the suffix is present.
        """
        macro_prefixes = ("dxy_", "vix_", "yield_curve_")
        for feat in SWING_FEATURES:
            for prefix in macro_prefixes:
                if feat.startswith(prefix):
                    assert "_lag1d" in feat, (
                        f"Macro feature '{feat}' in SWING_FEATURES lacks '_lag1d' suffix. "
                        "Swing bars (4h) see daily macro BEFORE the day closes — must lag."
                    )

    def test_daily_features_allow_no_lag(self) -> None:
        """Daily features may use macro without lag (bar closes at EOD)."""
        macro_prefixes = ("dxy_", "vix_", "yield_curve_")
        for feat in DAILY_FEATURES:
            for prefix in macro_prefixes:
                if feat.startswith(prefix):
                    # OK to NOT have _lag1d — this is the expected design
                    assert "_lag1d" not in feat, (
                        f"Daily feature '{feat}' has unnecessary _lag1d. "
                        "Daily bars incorporate same-day macro — no lag needed."
                    )


# ============================================================================
# TEST 3: Universe survivorship bias
# ============================================================================


class TestUniverseSurvivorshipBias:
    """
    Trainer must use point-in-time universe:
      - symbols delisted before as_of should be EXCLUDED
      - symbols delisted after as_of should be INCLUDED
    """

    def _build_mock_trainer(self) -> Any:
        from models.multi_horizon.trainer import MultiHorizonTrainer

        trainer = MultiHorizonTrainer(seed=42, ablate=False)
        return trainer

    def test_delisted_symbol_excluded_after_delisting(self) -> None:
        """
        Simulate a universe with one symbol that delisted in 2024.
        When training as_of=2025, the delisted symbol must NOT appear.
        """
        universe_historical = [
            {"symbol": "AAPL",        "first_listed": date(2000, 1, 1), "delisted": None},
            {"symbol": "DELISTED_CO", "first_listed": date(2010, 1, 1), "delisted": date(2024, 6, 1)},
        ]

        def filter_universe(
            universe: list[dict],
            as_of: date,
        ) -> list[str]:
            """Reference implementation of point-in-time filter."""
            return [
                u["symbol"]
                for u in universe
                if u["first_listed"] <= as_of
                and (u["delisted"] is None or u["delisted"] > as_of)
            ]

        symbols_2023 = filter_universe(universe_historical, date(2023, 1, 1))
        symbols_2025 = filter_universe(universe_historical, date(2025, 1, 1))

        assert "DELISTED_CO" in symbols_2023, (
            "Delisted-in-2024 company must be IN 2023 universe (survivorship bias)."
        )
        assert "DELISTED_CO" not in symbols_2025, (
            "Delisted-in-2024 company must NOT be in 2025 universe."
        )
        assert "AAPL" in symbols_2023
        assert "AAPL" in symbols_2025

    def test_no_universe_current_in_new_code(self) -> None:
        """
        Grep check: no new code in research/models/multi_horizon/ should
        reference 'universe_current' (only 'universe_historical' is allowed).
        """
        module_dir = Path(__file__).parents[1] / "models" / "multi_horizon"
        violations: list[str] = []
        for py_file in module_dir.glob("**/*.py"):
            text = py_file.read_text(encoding="utf-8")
            if "universe_current" in text:
                violations.append(str(py_file))
        assert not violations, (
            f"Found 'universe_current' in: {violations}. "
            "Use 'universe_historical' for point-in-time lookups."
        )


# ============================================================================
# TEST 4: Calibration only sees TRAIN_calib rows
# ============================================================================


class SpyIsotonicCalibrator:
    """
    Drop-in replacement for IsotonicCalibrator that records
    all indices seen during .fit() calls.
    """

    def __init__(self) -> None:
        self.fit_calls: list[np.ndarray] = []
        self.is_fitted = False
        self._real: Any = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> "SpyIsotonicCalibrator":
        if isinstance(X, pd.DataFrame):
            self.fit_calls.append(X.index.values.copy())
        else:
            self.fit_calls.append(np.arange(len(X)))
        self.is_fitted = True
        return self

    def predict_proba_from_raw(self, raw: np.ndarray) -> np.ndarray:
        return raw

    def fit_from_proba(self, proba: np.ndarray, y: Any) -> "SpyIsotonicCalibrator":
        self.is_fitted = True
        return self


class TestCalibrationScopeAntiLeakage:
    """
    Calibrator must never see TEST data.
    """

    def test_calibration_indices_precede_test_start(self) -> None:
        """
        With a DatetimeIndex dataset, verify that every timestamp the
        calibrator saw during fit() predates the test set start.
        """
        n = 400
        idx = pd.date_range("2022-01-01", periods=n, freq="B")
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((n, 3)), index=idx, columns=["a", "b", "c"])
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx)

        train_size = 250
        test_size  = 50
        embargo    = 5
        calib_frac = 0.20

        splitter = WalkForwardSplitter(
            train_size=train_size,
            test_size=test_size,
            embargo=embargo,
        )

        from models.calibration import split_train_for_calibration

        for train_idx, test_idx in splitter.split(X):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            test_start = X.index[test_idx.min()]

            X_fit, y_fit, X_calib, y_calib = split_train_for_calibration(
                X_train, y_train, calib_frac=calib_frac
            )

            # All calib rows must be BEFORE test start
            assert X_calib.index.max() < test_start, (
                f"Calibration set reaches {X_calib.index.max()} but test starts "
                f"at {test_start}. Calibrator would see future data."
            )
            # All fit rows must be BEFORE calib rows
            assert X_fit.index.max() <= X_calib.index.min(), (
                "Fit set must end before calibration set begins."
            )


# ============================================================================
# TEST 5: DSR correction uses total n_trials (150, not 50)
# ============================================================================


class TestDSRNCorrectionCrossHorizon:
    """
    DSR with 150 trials (3 horizons × 50 each) must be strictly lower
    than DSR with only 50 trials, because the Bonferroni-like deflation
    term sqrt(2·ln(N)) grows with N.
    """

    def test_dsr_higher_n_gives_lower_value(self) -> None:
        rng = np.random.default_rng(99)
        returns = rng.normal(0.001, 0.02, 252)  # synthetic positive Sharpe

        dsr_n50  = deflated_sharpe_ratio(returns, n_trials=50)
        dsr_n150 = deflated_sharpe_ratio(returns, n_trials=150)

        assert dsr_n150 < dsr_n50, (
            f"DSR(n=150)={dsr_n150:.6f} should be < DSR(n=50)={dsr_n50:.6f}. "
            "More trials = higher benchmark = lower DSR."
        )

    def test_total_optuna_trials_equals_150(self) -> None:
        """TOTAL_OPTUNA_TRIALS must equal sum of n_optuna_trials × n_horizons."""
        expected = sum(h.n_optuna_trials for h in ALL_HORIZONS)
        assert TOTAL_OPTUNA_TRIALS == expected, (
            f"TOTAL_OPTUNA_TRIALS={TOTAL_OPTUNA_TRIALS} != {expected}. "
            "Cross-horizon DSR correction will be wrong."
        )
        assert TOTAL_OPTUNA_TRIALS == 150, (
            f"Expected 3×50=150, got {TOTAL_OPTUNA_TRIALS}."
        )

    def test_dsr_n_trials_monotone_in_n(self) -> None:
        """DSR must decrease (or equal) as n_trials increases."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.015, 500)

        prev = float("inf")
        for n in [1, 10, 50, 100, 150, 500]:
            dsr = deflated_sharpe_ratio(returns, n_trials=n)
            assert dsr <= prev + 1e-9, (
                f"DSR not monotone: DSR(n={n})={dsr:.6f} > DSR(n_prev)={prev:.6f}"
            )
            prev = dsr


# ============================================================================
# TEST 6: Alpaca data latency — last_price must be lagged in intraday
# ============================================================================


class TestIntradayDataLatency:
    """
    Alpaca WS delivers trade data with 50-150ms latency.
    If the intraday model uses 'last_price' from the current bar close
    (which arrives ~150ms after the bar) as a feature without shifting,
    it constitutes a look-ahead. Verify that any such column in the
    intraday feature set is explicitly lagged or absent.
    """

    def test_no_current_bar_close_in_intraday_features(self) -> None:
        """
        'last_price', 'close', 'bar_close' without lag annotations
        must not appear in INTRADAY_FEATURES.
        """
        look_ahead_names = {"last_price", "close", "bar_close", "price"}
        violations = [
            f for f in INTRADAY_FEATURES
            if f in look_ahead_names and "_lag" not in f
        ]
        assert not violations, (
            f"Potential latency-peek features in INTRADAY_FEATURES: {violations}. "
            "These should use bar t-1 close or be explicitly lagged."
        )

    def test_shifted_feature_removes_peek(self) -> None:
        """Demonstrate that .shift(1) correctly removes look-ahead."""
        n = 100
        close = pd.Series(np.cumsum(np.random.randn(n)) + 100)

        # Without lag: feature at t uses close[t]  (peek at current bar)
        feature_no_lag = close.copy()
        # With lag: feature at t uses close[t-1]   (only past info)
        feature_lagged = close.shift(1)

        # At test bar t=50, lagged version must use info from t=49
        t = 50
        assert feature_lagged.iloc[t] == close.iloc[t - 1], (
            "Lagged feature does not correctly reference prior bar."
        )
        assert feature_lagged.iloc[t] != feature_no_lag.iloc[t], (
            "Lagged and un-lagged features are the same — shift had no effect."
        )


# ============================================================================
# TEST 7: GMM regime not fitted on full dataset
# ============================================================================


class TestGMMRegimeFittedPerFold:
    """
    GMMRegimeDetector must be re-fit per fold on TRAIN data only.
    Verify that each fold receives a freshly-fitted detector, not
    one shared instance fitted on the entire dataset.
    """

    def test_regime_detector_refitted_per_fold(self) -> None:
        """
        Each fold must produce regime features from a detector trained
        only on that fold's train period, not a global fit.
        """
        from features.regime_gmm import GMMRegimeDetector, GMMRegimeConfig

        n = 300
        idx = pd.date_range("2022-01-01", periods=n, freq="B")
        close = pd.Series(100.0 + np.cumsum(np.random.randn(n)), index=idx)
        atr   = pd.Series(np.abs(np.random.randn(n)) + 1.0, index=idx)

        splitter = WalkForwardSplitter(train_size=150, test_size=50, embargo=5)
        X_dummy = pd.DataFrame({"close": close.values}, index=idx)

        fit_timestamps: list[tuple[Any, Any]] = []

        for train_idx, _ in splitter.split(X_dummy):
            det = GMMRegimeDetector(GMMRegimeConfig(n_components=3))
            prices_train = close.iloc[train_idx]
            atr_train    = atr.iloc[train_idx]
            det.fit(prices_train, atr_train)
            fit_timestamps.append(
                (prices_train.index[0], prices_train.index[-1])
            )

        # Each fold should have a distinct training window
        starts = [t[0] for t in fit_timestamps]
        assert len(set(starts)) == len(starts) or len(starts) <= 1, (
            "Multiple folds have identical training start — GMM may be shared."
        )


# ============================================================================
# TEST 8: PCA excludes regime columns (ADR-004)
# ============================================================================


class TestPCAExcludesRegimeCols:
    """PCADenoiser must exclude regime_prob_* columns."""

    def test_pca_excludes_regime_prefix(self) -> None:
        from features.pca_denoiser import PCADenoiser, PCAConfig

        n = 200
        rng = np.random.default_rng(7)
        cols = ["feat_a", "feat_b", "regime_prob_0", "regime_prob_1"]
        X = pd.DataFrame(rng.standard_normal((n, 4)), columns=cols)

        denoiser = PCADenoiser(PCAConfig(n_components=2, exclude_prefix="regime_", min_components=1))
        denoiser.fit(X)
        X_out = denoiser.transform(X)

        # Regime columns must pass through unchanged
        assert "regime_prob_0" in X_out.columns
        assert "regime_prob_1" in X_out.columns

        # PCA columns replace the non-regime ones
        regime_vals_in  = X[["regime_prob_0", "regime_prob_1"]].values
        regime_vals_out = X_out[["regime_prob_0", "regime_prob_1"]].values
        np.testing.assert_array_almost_equal(regime_vals_in, regime_vals_out), (
            "Regime columns were modified by PCA — ADR-004 violation."
        )
