"""
Tests — NanValidator & NanReport.
==================================

All tests are pure unit tests (no network, no filesystem).

Covers
------
* NanReport.summary() formatting.
* NanValidator.validate():
  - empty DataFrame passes.
  - clean DataFrame passes (no NaN).
  - column with NaN only in warmup window passes.
  - column with NaN after warmup is flagged as alert.
  - 100%-NaN column appears in empty_columns.
  - raise_on_alert=True raises ValueError.
  - post-warmup calculation respects warmup_bars boundary.
  - pct values are rounded to 4 decimal places.
* NanValidator.clean():
  - drops warmup rows.
  - forward-fills remaining NaN.
  - back-fills leading NaN (after warmup drop).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.nan_validator import NanReport, NanValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 300, n_cols: int = 5, seed: int = 0) -> pd.DataFrame:
    """Create a clean DataFrame with no NaN values."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    data = {f"feat_{i}": rng.standard_normal(n) for i in range(n_cols)}
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# NanReport
# ---------------------------------------------------------------------------

class TestNanReport:
    def test_summary_passed(self):
        r = NanReport(
            total_bars=100, total_features=5,
            warmup_bars=20, alert_threshold=0.05,
            passed=True,
        )
        s = r.summary()
        assert "PASSED" in s
        assert "100" in s

    def test_summary_alerts(self):
        r = NanReport(
            total_bars=100, total_features=3,
            warmup_bars=20, alert_threshold=0.05,
            alerts={"feat_A": 0.15, "feat_B": 0.08},
            passed=False,
        )
        s = r.summary()
        assert "ALERTS" in s
        assert "feat_A" in s
        assert "feat_B" in s

    def test_summary_empty_columns(self):
        r = NanReport(
            total_bars=50, total_features=2,
            warmup_bars=10, alert_threshold=0.05,
            empty_columns=["dead_feature"],
            passed=False,
        )
        assert "dead_feature" in r.summary()

    def test_frozen(self):
        r = NanReport(
            total_bars=10, total_features=1,
            warmup_bars=5, alert_threshold=0.05,
        )
        with pytest.raises(Exception):
            r.total_bars = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NanValidator.validate — basic cases
# ---------------------------------------------------------------------------

class TestValidateBasic:
    def test_empty_df_passes(self):
        v = NanValidator()
        r = v.validate(pd.DataFrame())
        assert r.passed
        assert r.total_bars == 0
        assert r.total_features == 0

    def test_clean_df_passes(self):
        df = _make_df(n=300)
        v = NanValidator(warmup_bars=50)
        r = v.validate(df)
        assert r.passed
        assert all(pct == 0.0 for pct in r.nan_pct.values())

    def test_nan_only_in_warmup_passes(self):
        """NaN only in the first warmup_bars rows → post-warmup NaN% = 0 → passes."""
        df = _make_df(n=300)
        df.iloc[:50, 0] = np.nan   # first 50 rows (= warmup) set to NaN
        v = NanValidator(warmup_bars=50)
        r = v.validate(df)
        assert r.passed
        assert r.nan_pct[df.columns[0]] == 0.0

    def test_nan_after_warmup_triggers_alert(self):
        df = _make_df(n=200)
        # Introduce NaN in rows 100-150 (well after warmup=50)
        df.iloc[100:150, 0] = np.nan
        v = NanValidator(warmup_bars=50, alert_threshold=0.05)
        r = v.validate(df)
        assert not r.passed
        assert df.columns[0] in r.alerts

    def test_100pct_nan_column_in_empty_columns(self):
        df = _make_df(n=200)
        df["dead"] = np.nan
        v = NanValidator(warmup_bars=50)
        r = v.validate(df)
        assert "dead" in r.empty_columns
        assert not r.passed

    def test_raise_on_alert(self):
        df = _make_df(n=200)
        df.iloc[100:150, 0] = np.nan
        v = NanValidator(warmup_bars=50, alert_threshold=0.05, raise_on_alert=True)
        with pytest.raises(ValueError, match="NaN"):
            v.validate(df)

    def test_no_raise_when_raise_false(self):
        df = _make_df(n=200)
        df.iloc[100:150, 0] = np.nan
        v = NanValidator(warmup_bars=50, alert_threshold=0.05, raise_on_alert=False)
        r = v.validate(df)  # should not raise
        assert not r.passed


# ---------------------------------------------------------------------------
# NanValidator.validate — boundary conditions
# ---------------------------------------------------------------------------

class TestValidateBoundary:
    def test_fewer_bars_than_warmup_uses_all_bars(self):
        """If df has ≤ warmup_bars rows the entire frame acts as post-warmup."""
        df = _make_df(n=30)
        df.iloc[5:10, 0] = np.nan   # 5 NaN in 30 rows
        v = NanValidator(warmup_bars=100)   # warmup > n_bars
        r = v.validate(df)
        # post-warmup window = entire frame because n_bars <= warmup_bars
        # NaN fraction = 5/30 ≈ 0.1667 > 0.05 → alert
        assert df.columns[0] in r.nan_pct

    def test_pct_rounds_to_4_places(self):
        df = _make_df(n=200)
        df.iloc[51:54, 0] = np.nan   # 3 NaN in 150 post-warmup bars
        v = NanValidator(warmup_bars=50)
        r = v.validate(df)
        pct = r.nan_pct[df.columns[0]]
        assert pct == round(pct, 4)

    def test_nan_counts_full_frame(self):
        df = _make_df(n=200)
        df.iloc[:10, 0] = np.nan   # 10 NaN in warmup → not in alert but counted
        v = NanValidator(warmup_bars=50)
        r = v.validate(df)
        assert r.nan_counts[df.columns[0]] == 10

    def test_multiple_columns_partial_alerts(self):
        df = _make_df(n=200, n_cols=4)
        df.iloc[100:160, 0] = np.nan   # col 0 → alert
        # cols 1-3 clean
        v = NanValidator(warmup_bars=50, alert_threshold=0.05)
        r = v.validate(df)
        assert df.columns[0] in r.alerts
        for col in df.columns[1:]:
            assert col not in r.alerts

    def test_passed_false_when_any_alert(self):
        df = _make_df(n=200)
        df.iloc[100:160, 0] = np.nan
        v = NanValidator(warmup_bars=50, alert_threshold=0.05)
        r = v.validate(df)
        assert r.passed is False

    def test_total_features_matches_columns(self):
        df = _make_df(n=200, n_cols=7)
        v = NanValidator(warmup_bars=50)
        r = v.validate(df)
        assert r.total_features == 7

    def test_alert_threshold_zero_flags_any_nan(self):
        df = _make_df(n=200)
        df.iloc[100, 0] = np.nan   # single NaN after warmup
        v = NanValidator(warmup_bars=50, alert_threshold=0.0)
        r = v.validate(df)
        assert df.columns[0] in r.alerts


# ---------------------------------------------------------------------------
# NanValidator.clean
# ---------------------------------------------------------------------------

class TestClean:
    def test_drops_warmup_rows(self):
        df = _make_df(n=200)
        v = NanValidator(warmup_bars=50)
        clean = v.clean(df)
        assert len(clean) == 150

    def test_ffill_fills_nan(self):
        df = _make_df(n=200)
        df.iloc[100:105, 0] = np.nan
        v = NanValidator(warmup_bars=50)
        clean = v.clean(df)
        assert clean.iloc[:, 0].isna().sum() == 0

    def test_bfill_fills_leading_nan_after_warmup_drop(self):
        df = _make_df(n=200)
        # NaN at exactly the first row post-warmup
        df.iloc[50, 0] = np.nan
        v = NanValidator(warmup_bars=50)
        clean = v.clean(df)
        assert clean.iloc[:, 0].isna().sum() == 0

    def test_returns_copy_not_view(self):
        df = _make_df(n=200)
        v = NanValidator(warmup_bars=50)
        clean = v.clean(df)
        clean.iloc[0, 0] = 99999.0
        assert df.iloc[50, 0] != 99999.0   # original unchanged

    def test_index_starts_after_warmup(self):
        df = _make_df(n=200)
        v = NanValidator(warmup_bars=50)
        clean = v.clean(df)
        assert clean.index[0] == df.index[50]
