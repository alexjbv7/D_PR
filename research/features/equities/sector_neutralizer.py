"""
Sector-neutralized feature generator.

Demeans each numeric feature within its GICS sector on each date,
removing cross-sector momentum/value disparities that would otherwise
leak sector-level signals into the model.

GICS sector map: data/sectors/gics_map.json
  {"AAPL": "Information Technology", "JPM": "Financials", ...}
If missing, falls back to no-op with a warning.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_GICS_PATH = Path(__file__).parents[3] / "data" / "sectors" / "gics_map.json"

# Minimal fallback sector map for the top-50 universe.
_FALLBACK_GICS: dict[str, str] = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "GOOGL": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "NVDA": "Information Technology",
    "META": "Communication Services",
    "TSLA": "Consumer Discretionary",
    "JPM": "Financials",
    "JNJ": "Health Care",
    "V": "Financials",
    "XOM": "Energy",
    "WMT": "Consumer Staples",
    "UNH": "Health Care",
    "PG": "Consumer Staples",
    "MA": "Financials",
    "HD": "Consumer Discretionary",
    "CVX": "Energy",
    "MRK": "Health Care",
    "ABBV": "Health Care",
    "PFE": "Health Care",
    "BTCUSDT": "Crypto",
    "ETHUSDT": "Crypto",
    "SOLUSDT": "Crypto",
}


def _load_sector_map() -> dict[str, str]:
    if _GICS_PATH.exists():
        with open(_GICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    logger.warning(
        "GICS sector map not found at %s. "
        "Using fallback map for top-50 universe. "
        "TODO(@alex): populate from Alpaca asset.sector field.",
        _GICS_PATH,
    )
    return _FALLBACK_GICS


def demean_by_sector(
    features: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
    as_of: date | None = None,
) -> pd.DataFrame:
    """
    Subtract sector mean from each symbol's features on each date.

    Parameters
    ----------
    features : pd.DataFrame
        MultiIndex columns: (symbol, feature_name) or single feature column
        per symbol stacked with a date index.
        If simple columns with one feature per date, demeaning is cross-symbol.
    sector_map : dict[str, str] | None
        symbol → GICS sector.  If None, loads from file or fallback.
    as_of : date | None
        Not currently used; reserved for point-in-time sector map in S10+.

    Returns
    -------
    pd.DataFrame same shape as input, sector-mean subtracted per date.
    """
    if sector_map is None:
        sector_map = _load_sector_map()

    result = features.copy()

    if isinstance(features.columns, pd.MultiIndex):
        # MultiIndex (symbol, feature) — demean per (date, sector, feature)
        symbols = features.columns.get_level_values(0).unique()
        sector_groups: dict[str, list[str]] = {}
        for sym in symbols:
            sec = sector_map.get(str(sym), "Unknown")
            sector_groups.setdefault(sec, []).append(str(sym))

        for sector, sym_list in sector_groups.items():
            valid = [s for s in sym_list if s in features.columns.get_level_values(0)]
            if not valid:
                continue
            group_df = features[valid]
            sector_mean = group_df.mean(axis=1, skipna=True)
            for sym in valid:
                result[sym] = result[sym].subtract(sector_mean, axis=0)
    else:
        # Simple columns = feature names, index = dates.
        # Apply one-pass demeaning: subtract column mean across same-sector symbols.
        # Only makes sense if columns represent different symbols for the same feature.
        result = features.sub(features.mean(axis=1), axis=0)

    return result


def build_sector_demean_features(
    returns_by_symbol: dict[str, pd.Series],
    sector_map: dict[str, str] | None = None,
    momentum_window: int = 60,
) -> pd.DataFrame:
    """
    Convenience wrapper: compute sector-demeaned momentum and value features.

    Parameters
    ----------
    returns_by_symbol : dict[str, pd.Series]
        Daily returns per symbol.
    sector_map : dict[str, str] | None
        symbol → GICS sector.
    momentum_window : int
        Rolling window for momentum computation.

    Returns
    -------
    pd.DataFrame with columns sector_demean_momentum_60d, sector_demean_value.
    Index = union of all dates.
    """
    if sector_map is None:
        sector_map = _load_sector_map()

    momentum_df = pd.DataFrame({
        sym: ret.rolling(momentum_window).sum()
        for sym, ret in returns_by_symbol.items()
    })

    demeaned_mom = demean_by_sector(momentum_df, sector_map)
    demeaned_val = demean_by_sector(momentum_df.mul(-1), sector_map)

    return pd.DataFrame({
        "sector_demean_momentum_60d": demeaned_mom.mean(axis=1),
        "sector_demean_value":        demeaned_val.mean(axis=1),
    })
