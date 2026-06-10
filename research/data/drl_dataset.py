"""
Real-data loader for the DRL TradingEnvironment (ADR-037 feature contract).

Fetches OHLCV via the canonical AlpacaBarsIngestor, then computes the *exact*
21-column observation contract the env expects (14 market + 7 regime) plus
``close``. The env's feature names (``atr_14``, ``bb_pct``, ``vol_realized_20``,
``volume_z_20`` ...) do NOT match ``features.engineering.FeatureEngineer.build``
output names, so this module builds them explicitly from the engineering
primitives to avoid silently-zero features.

Anti-leakage
------------
- Technical features are causal (rolling windows over past data only).
- The GMM regime model is fitted ONLY on a caller-supplied training slice,
  then applied to the full series. The evaluation slice never participates in
  the regime fit.
- Single split (``build_drl_dataset`` + ``train_frac``): the GMM sees the
  first ``train_frac`` of the clean bars. Mirrors the driver's train/eval cut.
- Walk-forward (``build_env_frame`` + ``gmm_train_idx``, ADR-040): the GMM is
  re-fitted PER FOLD on exactly the train bars of that fold. Fitting it once
  on a global ``train_frac`` would leak future folds into earlier regimes.

The Alpaca fetch is isolated in ``_fetch_ohlcv`` so tests can monkeypatch it
(no network / no credentials required for unit tests).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from features.engineering import atr as _atr
from features.engineering import bollinger_pct_b, macd, rsi, zscore
from features.regime_gmm import GMMRegimeConfig, GMMRegimeDetector

logger = logging.getLogger(__name__)

# Exact env contract (mirror of envs.trading_env._MARKET_COLS without the None).
_MARKET_FEATURES = (
    "ret_1", "ret_5", "ret_20", "vol_realized_20", "vol_z_60",
    "rsi_14", "macd_signal", "atr_14", "bb_pct", "volume_z_20",
    "ob_imbalance", "spread_bps", "funding_z_60", "session_rth",
)
_REGIME_FEATURES = (
    "regime_prob_0", "regime_prob_1", "regime_prob_2", "regime_prob_3",
    "regime_prob_4", "regime_stability", "vol_regime",
)
_N_REGIMES = 3  # GMM components; env reserves 5 prob slots, extras stay 0.


def _fetch_ohlcv(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
    feed: str,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Alpaca and return a UTC-indexed DataFrame with
    columns [open, high, low, close, volume].

    Reads ALPACA_API_KEY / ALPACA_API_SECRET from the environment. Isolated
    here so unit tests can monkeypatch it.
    """
    from data.alpaca_bars import AlpacaBarsIngestor

    ingestor = AlpacaBarsIngestor.from_env(feed=feed)
    with ingestor:
        ingestor.ingest([symbol], timeframe=timeframe, start=start, end=end)
        df = ingestor._load_df(symbol, timeframe)
        # Incremental ingest only moves the watermark FORWARD. If the cached
        # parquet starts after the requested start (e.g. cache built recently
        # but a multi-year walk-forward is requested), backfill once from
        # `start` with incremental=False so history is actually available.
        if df is not None and not df.empty and start is not None:
            first = df.index.min()
            if first.tz is None:
                first = first.tz_localize("UTC")
            if first > pd.Timestamp(start) + pd.Timedelta(days=7):
                logger.info(
                    "drl_dataset backfill: cache starts %s > requested %s — "
                    "re-ingesting full history (incremental=False)",
                    first.date(), start.date(),
                )
                ingestor.ingest(
                    [symbol], timeframe=timeframe, start=start, end=end,
                    incremental=False,
                )
                df = ingestor._load_df(symbol, timeframe)
    if df is None or df.empty:
        raise RuntimeError(f"Alpaca returned no bars for {symbol} [{start}..{end}]")
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    # The parquet stores ALL cached history; honor the requested window.
    return df.loc[pd.Timestamp(start):pd.Timestamp(end)]


def _market_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute the env's 14 market columns by exact name from primitives."""
    o, h, l, c, v = (ohlcv[k] for k in ("open", "high", "low", "close", "volume"))
    f = pd.DataFrame(index=ohlcv.index)

    f["ret_1"] = np.log(c / c.shift(1))
    f["ret_5"] = np.log(c / c.shift(5))
    f["ret_20"] = np.log(c / c.shift(20))
    f["vol_realized_20"] = f["ret_1"].rolling(20).std()
    f["vol_z_60"] = zscore(f["vol_realized_20"], 60)
    f["rsi_14"] = rsi(c, 14)
    f["macd_signal"] = macd(c)["macd_signal"]
    f["atr_14"] = _atr(h, l, c, 14) / c            # normalized, env clips to [-3,3]
    f["bb_pct"] = bollinger_pct_b(c, 20)
    f["volume_z_20"] = zscore(v, 20)
    # Not available from equity bars — honest zeros (env defaults missing to 0 too):
    f["ob_imbalance"] = 0.0
    f["spread_bps"] = 0.0
    f["funding_z_60"] = 0.0
    f["session_rth"] = 1.0                         # Alpaca RTH bars
    return f


def _regime_features(
    ohlcv: pd.DataFrame,
    train_idx: "int | np.ndarray",
) -> pd.DataFrame:
    """
    Fit the GMM ONLY on the train bars, then transform the full series.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Clean (post-warmup) OHLCV frame.
    train_idx : int or np.ndarray
        Positional rows of ``ohlcv`` the GMM may see during ``fit``. An int
        ``k`` is shorthand for ``arange(k)`` (single-split case). For
        walk-forward (ADR-040) pass the exact train indices of the fold.

    Returns
    -------
    pd.DataFrame
        The 7 regime columns the env expects. regime_prob_3/4 are zero
        (only 3 components); regime_stability is derived from entropy;
        vol_regime is a bounded realized-vol indicator. The transform is
        causal per-bar (rolling lookback only), so computing it over the
        full series does not leak test data into the GMM fit.
    """
    c, h, l = ohlcv["close"], ohlcv["high"], ohlcv["low"]
    atr_raw = _atr(h, l, c, 14)

    if isinstance(train_idx, (int, np.integer)):
        train_idx = np.arange(int(train_idx))
    train_idx = np.asarray(train_idx, dtype=int)

    det = GMMRegimeDetector(GMMRegimeConfig(n_components=_N_REGIMES))
    det.fit(c.iloc[train_idx], atr_raw.iloc[train_idx])   # TRAIN ONLY — anti-leakage
    gmm = det.transform(c, atr_raw)

    r = pd.DataFrame(index=ohlcv.index)
    for k in range(5):
        col = f"regime_prob_{k}"
        r[col] = gmm[col] if col in gmm.columns else 0.0
    entropy = gmm.get("regime_entropy", pd.Series(0.0, index=ohlcv.index))
    r["regime_stability"] = (1.0 - entropy / float(np.log(_N_REGIMES))).clip(0.0, 1.0)
    vol = np.log(c / c.shift(1)).rolling(20).std()
    r["vol_regime"] = np.tanh(zscore(vol, 60).fillna(0.0))
    return r


def fetch_ohlcv_frame(
    symbol: str,
    start: "str | datetime",
    end: "str | datetime",
    *,
    timeframe: str = "1d",
    feed: str = "iex",
) -> pd.DataFrame:
    """
    Fetch raw OHLCV bars for ``symbol`` as a UTC-indexed DataFrame.

    Thin public wrapper over ``_fetch_ohlcv`` (which tests monkeypatch) that
    normalizes timestamps. This is the raw input expected by
    ``models.drl.dsr_gate`` (ADR-040), which builds features per fold.

    Parameters
    ----------
    symbol : str
        e.g. "SPY", "AAPL", or "BTC/USD".
    start, end : str or datetime
        ISO date strings or datetimes (interpreted as UTC).
    timeframe : str
        Alpaca timeframe ("1d", "4h", ...). Default daily.
    feed : str
        Alpaca data feed ("iex" free, "sip" paid).

    Returns
    -------
    pd.DataFrame
        UTC DatetimeIndex, columns = [open, high, low, close, volume].
    """
    start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime()
    end_dt = pd.Timestamp(end, tz="UTC").to_pydatetime()
    ohlcv = _fetch_ohlcv(symbol, start_dt, end_dt, timeframe, feed)
    if str(ohlcv.index.tz) != "UTC":
        ohlcv.index = ohlcv.index.tz_convert("UTC")
    return ohlcv


def clean_close_series(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Close prices restricted to the clean (post-warmup) bars.

    Positionally aligned with ``build_env_frame`` output — baselines that
    only need prices (e.g. buy-and-hold in the ADR-040 gate) can index this
    with the same fold indices without fitting any regime model.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Raw OHLCV frame (columns open/high/low/close/volume).

    Returns
    -------
    pd.Series
        ``close`` over the clean bars (length == ``n_clean_bars(ohlcv)``).
    """
    market = _market_features(ohlcv)
    return ohlcv.loc[market.notna().all(axis=1), "close"]


def n_clean_bars(ohlcv: pd.DataFrame) -> int:
    """
    Number of bars that survive the feature warmup trim.

    Walk-forward splitters (ADR-040) must be sized on the CLEAN bar count,
    because ``build_env_frame`` drops the first ~80 bars of rolling-window
    warmup before fold indices apply.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Raw OHLCV frame (columns open/high/low/close/volume).

    Returns
    -------
    int
        Rows of the env frame ``build_env_frame`` will return.
    """
    market = _market_features(ohlcv)
    return int(market.notna().all(axis=1).sum())


def build_env_frame(
    ohlcv: pd.DataFrame,
    gmm_train_idx: "int | np.ndarray",
) -> pd.DataFrame:
    """
    Build the env-ready frame (close + 21 features) from raw OHLCV.

    Market features are computed over the FULL raw series (so rolling windows
    keep their history), then warmup-NaN rows are trimmed. The regime GMM is
    fitted ONLY on ``gmm_train_idx`` — positional rows of the *clean* frame —
    and transformed causally over the whole series.

    This is the per-fold entry point of the ADR-040 walk-forward gate: call it
    once per fold with that fold's train indices, then slice train/test rows
    from the result. Never fit the GMM on a global split when folding.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Raw OHLCV (columns open/high/low/close/volume, DatetimeIndex UTC).
    gmm_train_idx : int or np.ndarray
        Positional indices into the returned (clean) frame that the GMM may
        see in ``fit``. An int ``k`` means the first ``k`` clean bars.

    Returns
    -------
    pd.DataFrame
        UTC DatetimeIndex, columns = ["close", *14 market, *7 regime], no NaN.
        Length == ``n_clean_bars(ohlcv)``.
    """
    market = _market_features(ohlcv)
    clean_mask = market.notna().all(axis=1)
    ohlcv_clean = ohlcv.loc[clean_mask]
    market_clean = market.loc[clean_mask]
    if len(ohlcv_clean) < 50:
        raise RuntimeError(
            f"only {len(ohlcv_clean)} clean bars; widen the date range"
        )

    regime = _regime_features(ohlcv_clean, gmm_train_idx)

    out = pd.concat(
        [ohlcv_clean["close"].rename("close"), market_clean, regime], axis=1
    )
    out = out.reindex(columns=["close", *_MARKET_FEATURES, *_REGIME_FEATURES])
    return out.dropna()


def build_drl_dataset(
    symbol: str,
    start: str | datetime,
    end: str | datetime,
    *,
    timeframe: str = "1d",
    train_frac: float = 0.7,
    feed: str = "iex",
) -> pd.DataFrame:
    """
    Build an env-ready dataset (close + 21 features) from real Alpaca OHLCV.

    Parameters
    ----------
    symbol : e.g. "SPY", "AAPL", or "BTC/USD".
    start, end : ISO date strings or datetimes.
    timeframe : Alpaca timeframe ("1d", "4h", ...). Default daily.
    train_frac : fraction used to fit the regime GMM (anti-leakage boundary).
    feed : Alpaca data feed ("iex" free, "sip" paid).

    Returns
    -------
    pd.DataFrame
        UTC DatetimeIndex, columns = ["close", *14 market, *7 regime], no NaN.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0,1), got {train_frac}")

    ohlcv = fetch_ohlcv_frame(symbol, start, end, timeframe=timeframe, feed=feed)

    # Trim warmup NaN BEFORE computing the split so the regime boundary aligns
    # with the clean dataset the driver will split.
    split_idx = int(n_clean_bars(ohlcv) * train_frac)
    out = build_env_frame(ohlcv, split_idx)
    logger.info(
        "drl_dataset built: %s rows=%d cols=%d train_split=%d (%.0f%%)",
        symbol, len(out), out.shape[1], split_idx, 100 * train_frac,
    )
    return out
