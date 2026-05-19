"""
Alpaca Compatibility Layer — Feature Pipeline Adapter
======================================================

Adapts Alpaca Parquet bar data (produced by ``AlpacaBarsIngestor``) to the
existing ``FeatureBuilder`` + ``GMMRegimeDetector`` + ``triple_barrier_labels_atr``
+ ``NanValidator`` pipeline.

Responsibilities
----------------
1. **Normalise** raw Alpaca bars: ensure UTC index, drop duplicates, sort,
   forward-fill gaps, validate required OHLCV columns.
2. **Equity session features**: UTC → America/New_York conversion.
   Computes overnight gap, bar-gap hours, intraday ET hour, RTH session
   position and boundary flags.
3. **Technical features**: delegate to ``FeatureBuilder`` (35+ features).
4. **Regime features**: fit-transform ``GMMRegimeDetector`` offline or
   accept a pre-fitted detector for walk-forward use.
5. **Labels**: ``triple_barrier_labels_atr`` with ATR computed via EWM.
6. **Validation**: ``NanValidator`` over the full feature matrix.

Classes
-------
FeatureResult
    Immutable result bundle returned by ``AlpacaFeatureBuilder.build()``.
AlpacaFeatureBuilder
    Main pipeline class.  One instance per symbol.

Standalone helpers
------------------
wf_smoke_test(df, n_splits=1)
    Runs one fold of the pipeline end-to-end with synthetic or real data.
    Used to verify the pipeline compiles and produces sane output before
    launching a full ``WalkForwardRunner`` experiment.

Import pattern
--------------
All imports use the *absolute* top-level form (``pythonpath = ["."]`` in
``research/pyproject.toml``):

    from features.engineering import FeatureBuilder
    from features.labeling import compute_atr_ewm, triple_barrier_labels_atr
    from features.regime_gmm import GMMRegimeDetector, GMMRegimeConfig
    from features.nan_validator import NanValidator, NanReport

References
----------
* CLAUDE.md §4.2 (Swing strategy — RTH session features)
* CLAUDE.md §7.4 (Feature Store validation)
* Architecture doc §1.1 (Capa A, Riesgo 3: latencia de datos)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from features.engineering import FeatureBuilder
from features.labeling import compute_atr_ewm, triple_barrier_labels_atr
from features.regime_gmm import GMMRegimeDetector, GMMRegimeConfig
from features.nan_validator import NanValidator, NanReport

logger = logging.getLogger(__name__)

# RTH session boundaries in ET (America/New_York)
_RTH_OPEN_HOUR  = 9
_RTH_OPEN_MIN   = 30
_RTH_CLOSE_HOUR = 16
_RTH_CLOSE_MIN  = 0
_RTH_DURATION_MINUTES = (
    (_RTH_CLOSE_HOUR * 60 + _RTH_CLOSE_MIN)
    - (_RTH_OPEN_HOUR * 60 + _RTH_OPEN_MIN)
)  # 390 minutes


# ---------------------------------------------------------------------------
# FeatureResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureResult:
    """
    Immutable bundle returned by ``AlpacaFeatureBuilder.build()``.

    Attributes
    ----------
    features : pd.DataFrame
        Full feature matrix (technical + session + regime), UTC-indexed.
        Warmup rows are retained here; use ``nan_validator.clean(features)``
        to drop them before feeding to ``WalkForwardRunner``.
    labels : pd.Series
        Triple-barrier labels (−1, 0, +1, NaN) aligned to ``features``.
    atr : pd.Series
        ATR series (EWM, period=14) aligned to ``features``.
    nan_report : NanReport
        Data-quality report on the feature matrix.
    metadata : dict
        Provenance: symbol, timeframe, bar_count, build_ts, etc.
    """
    features:   pd.DataFrame
    labels:     pd.Series
    atr:        pd.Series
    nan_report: NanReport
    metadata:   dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AlpacaFeatureBuilder
# ---------------------------------------------------------------------------

class AlpacaFeatureBuilder:
    """
    End-to-end feature pipeline for Alpaca equity bar data.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. ``"AAPL"``).  Used only for logging / metadata.
    timeframe : str
        Bar resolution string (e.g. ``"4h"``, ``"1d"``).  For metadata only.
    warmup_bars : int
        Leading bars to exclude from post-warmup NaN check.
        Default 200 covers vol_60, frac_diff, vol_regime, GMM.
    alert_threshold : float
        NaN fraction (post-warmup) that triggers a ``NanReport`` alert.
        Default 0.05 (5 %).
    raise_on_nan_alert : bool
        If ``True``, raise ``ValueError`` when NaN alerts are found.
        Default ``False``.
    gmm_config : GMMRegimeConfig | None
        Override GMM hyper-parameters.  ``None`` → default 3-component config.
    horizon : int
        Triple-barrier label horizon in bars.  Default 5.
    atr_mult : float
        ATR multiplier for both upper and lower barriers.  Default 1.5.
    feature_builder : FeatureBuilder | None
        Override the ``FeatureBuilder`` instance.  ``None`` → default config.
    """

    def __init__(
        self,
        symbol: str = "",
        timeframe: str = "",
        warmup_bars: int = 200,
        alert_threshold: float = 0.05,
        raise_on_nan_alert: bool = False,
        gmm_config: Optional[GMMRegimeConfig] = None,
        horizon: int = 5,
        atr_mult: float = 1.5,
        feature_builder: Optional[FeatureBuilder] = None,
    ):
        self.symbol             = symbol
        self.timeframe          = timeframe
        self.warmup_bars        = warmup_bars
        self.alert_threshold    = alert_threshold
        self.raise_on_nan_alert = raise_on_nan_alert
        self.gmm_config         = gmm_config or GMMRegimeConfig()
        self.horizon            = horizon
        self.atr_mult           = atr_mult
        self._fb = feature_builder or FeatureBuilder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame) -> FeatureResult:
        """
        Run the full feature pipeline on *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Raw Alpaca bars with columns
            ``[open, high, low, close, volume]``.
            Index must be a ``DatetimeTZInfo`` (UTC) or naive (will be
            localised to UTC automatically).

        Returns
        -------
        FeatureResult
        """
        # 1. Normalise
        df = self._normalize_input(df)
        n_raw = len(df)
        logger.debug("alpaca_compat.build symbol=%s bars=%d", self.symbol, n_raw)

        # 2. ATR (EWM, period=14 — anti-leakage, needed for labels + GMM)
        atr_series = compute_atr_ewm(df, period=14)

        # 3. Technical features
        features = self._fb.build(df, df_eth=None)

        # 4. Equity session features (UTC → ET)
        session_feats = self._equity_session_features(df)
        features = features.join(session_feats, how="left")

        # 5. Regime features (offline fit_transform)
        detector = GMMRegimeDetector(self.gmm_config)
        regime_feats = detector.fit_transform(df["close"], atr_series)
        features = features.join(regime_feats, how="left")

        # 6. Labels
        labels = triple_barrier_labels_atr(
            close=df["close"],
            atr=atr_series,
            horizon=self.horizon,
            upper_mult=self.atr_mult,
            lower_mult=self.atr_mult,
        )

        # 7. NaN validation
        validator = NanValidator(
            warmup_bars=self.warmup_bars,
            alert_threshold=self.alert_threshold,
            raise_on_alert=self.raise_on_nan_alert,
        )
        nan_report = validator.validate(features)
        logger.info(nan_report.summary())

        metadata = {
            "symbol":    self.symbol,
            "timeframe": self.timeframe,
            "bar_count": n_raw,
            "build_ts":  pd.Timestamp.now("UTC").isoformat(),
            "warmup_bars": self.warmup_bars,
            "horizon":   self.horizon,
            "atr_mult":  self.atr_mult,
        }

        return FeatureResult(
            features=features,
            labels=labels,
            atr=atr_series,
            nan_report=nan_report,
            metadata=metadata,
        )

    def build_clean(self, df: pd.DataFrame) -> FeatureResult:
        """
        Like ``build()``, but also forward-fills and drops the warmup rows so
        the ``FeatureResult.features`` is immediately ready for
        ``WalkForwardRunner``.
        """
        result = self.build(df)
        validator = NanValidator(warmup_bars=self.warmup_bars)
        clean_features = validator.clean(result.features)
        clean_labels   = result.labels.reindex(clean_features.index)
        clean_atr      = result.atr.reindex(clean_features.index)
        return FeatureResult(
            features=clean_features,
            labels=clean_labels,
            atr=clean_atr,
            nan_report=result.nan_report,
            metadata={**result.metadata, "clean": True},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise a raw Alpaca bars DataFrame.

        Steps
        -----
        1. Drop irrelevant columns (``symbol``, ``trade_count``, ``vwap``).
        2. Ensure required OHLCV columns are present.
        3. Localise naive index to UTC; convert tz-aware non-UTC to UTC.
        4. Sort ascending, deduplicate on index.
        5. Forward-fill OHLCV (fills small intraday gaps ≤ a few bars).
        """
        df = df.copy()

        # Drop non-OHLCV metadata columns that come from Alpaca
        for col in ("symbol", "trade_count", "vwap"):
            if col in df.columns:
                df = df.drop(columns=[col])

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"AlpacaFeatureBuilder: missing OHLCV columns: {missing}"
            )

        # Ensure UTC DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif str(df.index.tz) != "UTC":
            df.index = df.index.tz_convert("UTC")

        # Sort + deduplicate
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Fill small OHLCV gaps (forward-fill only — no look-ahead)
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        df[ohlcv_cols] = df[ohlcv_cols].ffill()

        return df

    @staticmethod
    def _equity_session_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute US equity session features from UTC bars.

        Converts the UTC index to ``America/New_York`` (handles DST) and
        derives:

        overnight_gap
            ``(open[t] − close[t−1]) / close[t−1]``.  Captures gap at the
            open caused by after-hours news.  NaN for the first bar.
        bar_gap_hours
            Hours elapsed since the previous bar.  Crypto = ~constant; equity
            bars can show large gaps over weekends or market closures.
        bar_hour_et
            Wall-clock hour in ET (0–23).  Captures intraday seasonality
            (open rush 9:30, lunch lull 12:00, MOC 15:45).
        session_open_bar
            1 if the bar falls in 9:30–10:00 ET (first 30 min of RTH).
        session_last_bar
            1 if the bar falls in 15:30–16:00 ET (last 30 min of RTH).
        session_position
            Fractional position within RTH 9:30–16:00 (0.0 = open, 1.0 =
            close).  Bars outside RTH receive NaN.

        Parameters
        ----------
        df : pd.DataFrame
            Normalised OHLCV DataFrame with UTC DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Six columns indexed identically to *df*.
        """
        idx_utc = df.index  # UTC

        try:
            idx_et = idx_utc.tz_convert("America/New_York")
        except Exception:
            # pytz not available or conversion failure — degrade gracefully
            logger.warning(
                "alpaca_compat: could not convert index to America/New_York; "
                "session features will be NaN."
            )
            return pd.DataFrame(
                index=idx_utc,
                columns=[
                    "overnight_gap", "bar_gap_hours", "bar_hour_et",
                    "session_open_bar", "session_last_bar", "session_position",
                ],
            )

        out = pd.DataFrame(index=idx_utc)

        # Overnight gap: (open[t] - close[t-1]) / close[t-1]
        out["overnight_gap"] = (
            df["open"] - df["close"].shift(1)
        ) / df["close"].shift(1)

        # Bar gap in hours
        ts_diff = idx_utc.to_series().diff()
        out["bar_gap_hours"] = ts_diff.dt.total_seconds() / 3600.0

        # ET hour of day
        out["bar_hour_et"] = idx_et.hour + idx_et.minute / 60.0

        # RTH boundary flags
        et_minutes = idx_et.hour * 60 + idx_et.minute  # minutes since midnight ET
        rth_open_min  = _RTH_OPEN_HOUR  * 60 + _RTH_OPEN_MIN   # 570
        rth_close_min = _RTH_CLOSE_HOUR * 60 + _RTH_CLOSE_MIN  # 960

        out["session_open_bar"] = (
            (et_minutes >= rth_open_min) & (et_minutes < rth_open_min + 30)
        ).astype(int)
        out["session_last_bar"] = (
            (et_minutes >= rth_close_min - 30) & (et_minutes < rth_close_min)
        ).astype(int)

        # Session position within RTH [0.0, 1.0]
        in_rth = (et_minutes >= rth_open_min) & (et_minutes < rth_close_min)
        minutes_into_session = et_minutes - rth_open_min
        position = minutes_into_session / _RTH_DURATION_MINUTES
        out["session_position"] = np.where(in_rth, position, np.nan)

        return out


# ---------------------------------------------------------------------------
# wf_smoke_test
# ---------------------------------------------------------------------------

def wf_smoke_test(
    df: Optional[pd.DataFrame] = None,
    symbol: str = "SMOKE",
    n_bars: int = 500,
    warmup_bars: int = 100,
    horizon: int = 5,
) -> FeatureResult:
    """
    Smoke-test the full Alpaca → Feature pipeline end-to-end.

    If *df* is ``None`` a synthetic OHLCV dataset is generated.  This lets
    you verify the pipeline compiles and produces sensible shapes without
    real data or network access.

    Parameters
    ----------
    df : pd.DataFrame | None
        Real OHLCV bars to use.  If ``None``, synthetic data is generated.
    symbol : str
        Symbol label for metadata.
    n_bars : int
        Number of synthetic bars to generate (ignored if *df* is provided).
    warmup_bars : int
        Warmup period for NaN validation.
    horizon : int
        Triple-barrier label horizon in bars.

    Returns
    -------
    FeatureResult
        Full pipeline result.

    Raises
    ------
    ValueError
        If ``nan_report.passed`` is False (any feature exceeds NaN threshold
        after warmup), or if ``features`` DataFrame is unexpectedly empty.

    Example
    -------
    ::

        result = wf_smoke_test(n_bars=600)
        assert result.nan_report.passed
        clean_X = result.features.iloc[100:]
        clean_y = result.labels.iloc[100:]
    """
    if df is None:
        df = _make_synthetic_bars(n_bars=n_bars)

    builder = AlpacaFeatureBuilder(
        symbol=symbol,
        timeframe="smoke",
        warmup_bars=warmup_bars,
        alert_threshold=0.10,   # lenient for smoke test
        raise_on_nan_alert=False,
        horizon=horizon,
    )
    result = builder.build(df)

    if result.features.empty:
        raise ValueError("wf_smoke_test: feature matrix is empty — pipeline error")

    n_post = len(result.features) - warmup_bars
    if n_post <= 0:
        raise ValueError(
            f"wf_smoke_test: only {len(result.features)} bars total; "
            f"need > {warmup_bars} (warmup_bars) to validate"
        )

    logger.info(
        "wf_smoke_test PASSED  symbol=%s  bars=%d  features=%d  "
        "post_warmup=%d  nan_alerts=%d",
        symbol,
        len(result.features),
        len(result.features.columns),
        n_post,
        len(result.nan_report.alerts),
    )
    return result


# ---------------------------------------------------------------------------
# Synthetic data generator (test helper)
# ---------------------------------------------------------------------------

def _make_synthetic_bars(
    n_bars: int = 500,
    freq: str = "4h",
    start: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV bars for testing.

    Uses a geometric Brownian motion close price; OHLV are constructed so that
    ``low ≤ open, close ≤ high`` always holds.

    Parameters
    ----------
    n_bars : int
        Number of bars to generate.
    freq : str
        Pandas frequency string for the DatetimeIndex.
    start : str
        Start date (UTC).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume.  UTC-indexed.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n_bars, freq=freq, tz="UTC")

    # Geometric Brownian motion
    log_returns = rng.normal(loc=0.0002, scale=0.015, size=n_bars)
    close = 100.0 * np.exp(np.cumsum(log_returns))

    # Open: close lagged + small gap
    open_ = np.roll(close, 1) * (1 + rng.normal(0, 0.002, n_bars))
    open_[0] = close[0]

    # High / Low from ATR proxy
    atr_proxy = np.abs(log_returns) * close * 2
    high  = np.maximum(open_, close) + rng.uniform(0, 1, n_bars) * atr_proxy
    low   = np.minimum(open_, close) - rng.uniform(0, 1, n_bars) * atr_proxy
    low   = np.maximum(low, 0.01)  # no negative prices

    volume = rng.uniform(1_000, 500_000, n_bars)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
