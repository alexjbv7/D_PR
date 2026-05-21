"""
Verify documented (partial) disjointness of feature sets.

Regime features appear in all 3 sets — this is intentional.
Daily-frequency features must NOT appear in intraday without a lag.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parents[2]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.multi_horizon.feature_sets import (
    DAILY_FEATURES,
    INTRADAY_FEATURES,
    SWING_FEATURES,
    get_feature_set,
)


def test_get_feature_set_returns_correct_list() -> None:
    assert get_feature_set("INTRADAY_FEATURES") == INTRADAY_FEATURES
    assert get_feature_set("SWING_FEATURES")    == SWING_FEATURES
    assert get_feature_set("DAILY_FEATURES")    == DAILY_FEATURES


def test_get_feature_set_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError, match="Unknown feature set"):
        get_feature_set("NONEXISTENT_FEATURES")


def test_no_duplicate_features_within_set() -> None:
    for name, fs in [
        ("INTRADAY", INTRADAY_FEATURES),
        ("SWING",    SWING_FEATURES),
        ("DAILY",    DAILY_FEATURES),
    ]:
        assert len(fs) == len(set(fs)), (
            f"Duplicate features detected in {name}_FEATURES: "
            + str([f for f in fs if fs.count(f) > 1])
        )


def test_regime_features_present_in_all_three_sets() -> None:
    """Regime probs are intentionally shared across horizons."""
    for k in range(3):
        feat = f"regime_prob_{k}"
        assert feat in INTRADAY_FEATURES, f"{feat} missing from INTRADAY"
        assert feat in SWING_FEATURES,    f"{feat} missing from SWING"
        assert feat in DAILY_FEATURES,    f"{feat} missing from DAILY"


def test_ff5_factors_only_in_daily() -> None:
    """FF5 factor exposures are daily-only features."""
    ff5 = [f for f in INTRADAY_FEATURES + SWING_FEATURES if "ff5_" in f]
    assert not ff5, (
        f"FF5 features found in non-daily feature sets: {ff5}. "
        "These are daily-frequency and must not appear in intraday/swing "
        "without explicit _lag1d annotation."
    )


def test_earnings_features_only_in_daily() -> None:
    non_daily = [
        f for f in INTRADAY_FEATURES + SWING_FEATURES
        if "earnings" in f or "days_to" in f or "days_since" in f
    ]
    assert not non_daily, (
        f"Earnings features found outside DAILY_FEATURES: {non_daily}"
    )


def test_sector_demean_only_in_daily() -> None:
    non_daily = [
        f for f in INTRADAY_FEATURES + SWING_FEATURES
        if "sector_demean" in f
    ]
    assert not non_daily, (
        f"Sector-demean features found outside DAILY_FEATURES: {non_daily}"
    )


def test_macro_features_lagged_in_swing() -> None:
    """Swing macro features carry _lag1d to prevent daily-in-4h peek."""
    macro_in_swing = [
        f for f in SWING_FEATURES
        if any(f.startswith(p) for p in ("dxy_", "vix_", "yield_curve_"))
    ]
    assert macro_in_swing, "No macro features found in SWING_FEATURES."
    for feat in macro_in_swing:
        assert "_lag1d" in feat, (
            f"Macro feature '{feat}' in SWING_FEATURES lacks '_lag1d' suffix."
        )


def test_macro_features_not_lagged_in_daily() -> None:
    """Daily macro features should NOT have unnecessary _lag1d."""
    macro_in_daily = [
        f for f in DAILY_FEATURES
        if any(f.startswith(p) for p in ("dxy_", "vix_", "yield_curve_"))
    ]
    assert macro_in_daily, "No macro features found in DAILY_FEATURES."
    for feat in macro_in_daily:
        assert "_lag1d" not in feat, (
            f"Daily macro feature '{feat}' has unnecessary _lag1d."
        )


def test_is_crypto_flag_in_all_sets() -> None:
    """Cross-asset flag present in all three sets."""
    assert "is_crypto" in INTRADAY_FEATURES
    assert "is_crypto" in SWING_FEATURES
    assert "is_crypto" in DAILY_FEATURES
