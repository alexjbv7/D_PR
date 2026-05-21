"""
Feature set constants for each trading horizon.

Design rules:
  - INTRADAY_FEATURES: only microstructure + regime. No daily-frequency
    features without an explicit _lag1d suffix (anti-leakage rule).
  - SWING_FEATURES: macro features carry _lag1d suffix because the swing bar
    closes before the daily macro release is available.
  - DAILY_FEATURES: macro features have no lag — the daily bar's close
    already incorporates that day's macro close.
  - regime_prob_* appears in all three sets: it is a stable cross-horizon
    signal computed per-fold with zero look-ahead.
  - ff5_*, earnings_*, sector_* features must never appear in INTRADAY_FEATURES
    without an explicit lag annotation.
"""
from __future__ import annotations


INTRADAY_FEATURES: list[str] = [
    # Microstructure
    "vol_z_5m",
    "vol_burst_5m",
    "atr_14_5m",
    "rsi_5",
    "rsi_14_5m",
    "tick_rule_sum_5m",
    "spread_bps_5m",
    "trade_count_z_5m",
    # Regime (GMM, computed per-fold on train)
    "regime_prob_0",
    "regime_prob_1",
    "regime_prob_2",
    # Session phase — one-hot encoded from quant_shared.calendar.SessionPhase
    "session_pre",
    "session_rth",
    "session_post",
    # Cross-asset flag
    "is_crypto",
]

SWING_FEATURES: list[str] = [
    # Technical (4h)
    "rsi_14_4h",
    "macd_signal_4h",
    "bb_pct_4h",
    "atr_14_4h",
    "momentum_z_20_4h",
    "momentum_z_60_4h",
    # Regime
    "regime_prob_0",
    "regime_prob_1",
    "regime_prob_2",
    "regime_stability_60",
    # Macro — daily features lagged 1 calendar day to prevent daily-in-4h peek
    "dxy_z_60_lag1d",
    "vix_level_lag1d",
    "yield_curve_slope_lag1d",
    # On-chain — only valid for BTC/ETH; other symbols → 0-fill + flag
    "btc_funding_z_lag1d",
    "btc_exchange_netflow_z_lag1d",
    # Cross-asset flag
    "is_crypto",
]

DAILY_FEATURES: list[str] = [
    # Fama-French 5 factor exposures (rolling 60d OLS)
    "ff5_mkt_rf",
    "ff5_smb",
    "ff5_hml",
    "ff5_rmw",
    "ff5_cma",
    # Earnings calendar
    "days_to_earnings",
    "days_since_earnings",
    "is_earnings_blackout",
    # Sector-neutralized signals
    "sector_demean_momentum_60d",
    "sector_demean_value",
    # Macro — no lag needed at daily frequency
    "dxy_z_60",
    "vix_level",
    "yield_curve_slope",
    # Regime
    "regime_prob_0",
    "regime_prob_1",
    "regime_prob_2",
    # Cross-asset flag
    "is_crypto",
]

_REGISTRY: dict[str, list[str]] = {
    "INTRADAY_FEATURES": INTRADAY_FEATURES,
    "SWING_FEATURES": SWING_FEATURES,
    "DAILY_FEATURES": DAILY_FEATURES,
}


def get_feature_set(name: str) -> list[str]:
    """
    Return the feature list for a given constant name.

    Parameters
    ----------
    name : str
        One of "INTRADAY_FEATURES", "SWING_FEATURES", "DAILY_FEATURES".

    Raises
    ------
    KeyError if the name is unknown.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown feature set '{name}'. "
            f"Valid names: {sorted(_REGISTRY)}"
        )
    return list(_REGISTRY[name])
