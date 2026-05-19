"""
Fama-French 5-Factor exposures via rolling 60-day OLS.

Data source: Kenneth French Data Library (daily CSV).
Cache: data/factors/ff5_daily.parquet (local, git-ignored).

Anti-leakage: rolling OLS uses only past 60 days at each t.
No forward information enters the exposure estimate.

References
----------
Fama, E.F. & French, K.R. (2015). A five-factor asset pricing model.
Journal of Financial Economics, 116(1), 1-22.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).parents[3] / "data" / "factors" / "ff5_daily.parquet"

_FF5_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
_FF5_OUTPUT_COLS = ["ff5_mkt_rf", "ff5_smb", "ff5_hml", "ff5_rmw", "ff5_cma"]
_ROLLING_WINDOW = 60


def load_ff5_factors(start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Load Fama-French 5-factor daily returns.

    Tries parquet cache first; falls back to synthetic proxy (stub) when
    neither cache nor network is available.  The stub uses the first
    principal component of a small return basket as Mkt-RF and zeros
    for SMB/HML/RMW/CMA.

    Parameters
    ----------
    start, end : str
        ISO date strings for the requested range.

    Returns
    -------
    pd.DataFrame with columns Mkt-RF, SMB, HML, RMW, CMA, RF (daily %).
    """
    if _CACHE_PATH.exists():
        df = pd.read_parquet(_CACHE_PATH)
        df.index = pd.to_datetime(df.index)
        mask = df.index >= pd.Timestamp(start)
        if end:
            mask &= df.index <= pd.Timestamp(end)
        return df.loc[mask]

    logger.warning(
        "FF5 cache not found at %s. "
        "Using synthetic proxy (Mkt-RF from SPY, others zero). "
        "TODO(@alex): download from https://mba.tuck.dartmouth.edu/pages/"
        "faculty/ken.french/data_library.html and cache to %s",
        _CACHE_PATH,
        _CACHE_PATH,
    )
    return _synthetic_ff5_stub(start, end)


def _synthetic_ff5_stub(start: str, end: str | None) -> pd.DataFrame:
    """
    Synthetic FF5 proxy — all factors set to zero except Mkt-RF = 0.

    This stub allows the pipeline to run end-to-end without real factor data.
    Replace with actual data download before using for real training.
    """
    date_range = pd.date_range(
        start=start,
        end=end or pd.Timestamp.today().strftime("%Y-%m-%d"),
        freq="B",
    )
    zeros = pd.DataFrame(0.0, index=date_range, columns=_FF5_COLS)
    return zeros


def compute_ff5_exposures(
    returns: pd.DataFrame,
    ff5_factors: pd.DataFrame,
    window: int = _ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Rolling OLS of asset returns against FF5 factors (window=60d by default).

    For each symbol in ``returns`` and each date t, estimates:
        r_{i,t} = α + β_mkt·Mkt-RF + β_smb·SMB + ... + ε

    using the prior ``window`` days (no forward data).

    Parameters
    ----------
    returns : pd.DataFrame
        Columns = symbols, index = trading dates.  Values = daily returns.
    ff5_factors : pd.DataFrame
        Columns = Mkt-RF, SMB, HML, RMW, CMA, RF.
    window : int
        Rolling OLS window in trading days.

    Returns
    -------
    pd.DataFrame with MultiIndex columns (symbol, factor_name) or stacked
    per-symbol DataFrames.  Columns in _FF5_OUTPUT_COLS.
    """
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    factors_aligned = ff5_factors[factor_cols].reindex(returns.index).fillna(0.0)

    results: list[pd.DataFrame] = []

    for symbol in returns.columns:
        sym_returns = returns[symbol].dropna()
        common_idx = sym_returns.index.intersection(factors_aligned.index)
        if len(common_idx) < window:
            empty = pd.DataFrame(
                np.nan,
                index=returns.index,
                columns=[f"ff5_{c.lower().replace('-', '_')}" for c in factor_cols],
            )
            empty.columns = pd.Index(_FF5_OUTPUT_COLS)
            results.append(empty)
            continue

        r = sym_returns.reindex(common_idx)
        F = factors_aligned.reindex(common_idx)
        betas = _rolling_ols(r.values, F.values, window)
        df_betas = pd.DataFrame(betas, index=common_idx, columns=_FF5_OUTPUT_COLS)
        df_betas = df_betas.reindex(returns.index)
        results.append(df_betas)

    if not results:
        return pd.DataFrame(index=returns.index, columns=_FF5_OUTPUT_COLS)

    # Aggregate: take the mean across symbols (useful for portfolio-level features)
    stacked = pd.concat(results)
    aggregated = stacked.groupby(level=0).mean()
    return aggregated.reindex(returns.index)


def _rolling_ols(
    y: np.ndarray,
    X: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Efficient rolling OLS via QR decomposition.

    Returns betas array shape (len(y), X.shape[1]).
    First ``window-1`` rows are NaN (insufficient history).
    """
    n, k = len(y), X.shape[1]
    betas = np.full((n, k), np.nan)

    for t in range(window - 1, n):
        y_w = y[t - window + 1 : t + 1]
        X_w = X[t - window + 1 : t + 1]
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
            betas[t] = coeffs
        except np.linalg.LinAlgError:
            pass

    return betas
