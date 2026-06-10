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
- The GMM regime model is fitted ONLY on the training slice (``train_frac``),
  then applied to the full series. The evaluation slice never participates in
  the regime fit. This mirrors the train/eval split the driver applies later,
  so the regime boundary is consistent and leakage-free.

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
    if df is None or df.empty:
        raise RuntimeError(f"Alpaca returned no bars for {symbol} [{start}..{end}]")
    return df[["open", "high", "low", "close", "volume"]].sort_index()


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
    split_idx: int,
) -> pd.DataFrame:
    """
    Fit the GMM on the TRAIN slice [:split_idx] only, transform the full series.

    Returns the 7 regime columns the env expects. regime_prob_3/4 are zero
    (only 3 components); regime_stability is derived from entropy; vol_regime
    is a bounded realized-vol indicator.
    """
    c, h, l = ohlcv["close"], ohlcv["high"], ohlcv["low"]
    atr_raw = _atr(h, l, c, 14)

    det = GMMRegimeDetector(GMMRegimeConfig(n_components=_N_REGIMES))
    det.fit(c.iloc[:split_idx], atr_raw.iloc[:split_idx])   # TRAIN ONLY — anti-leakage
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
    start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime()
    end_dt = pd.Timestamp(end, tz="UTC").to_pydatetime()
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0,1), got {train_frac}")

    ohlcv = _fetch_ohlcv(symbol, start_dt, end_dt, timeframe, feed)
    if str(ohlcv.index.tz) != "UTC":
        ohlcv.index = ohlcv.index.tz_convert("UTC")

    market = _market_features(ohlcv)

    # Trim warmup NaN BEFORE computing the split so the regime boundary aligns
    # with the clean dataset the driver will split.
    clean_mask = market.notna().all(axis=1)
    ohlcv_clean = ohlcv.loc[clean_mask]
    market_clean = market.loc[clean_mask]
    if len(ohlcv_clean) < 50:
        raise RuntimeError(
            f"only {len(ohlcv_clean)} clean bars for {symbol}; widen the date range"
        )

    split_idx = int(len(ohlcv_clean) * train_frac)
    regime = _regime_features(ohlcv_clean, split_idx)

    out = pd.concat(
        [ohlcv_clean["close"].rename("close"), market_clean, regime], axis=1
    )
    out = out.reindex(columns=["close", *_MARKET_FEATURES, *_REGIME_FEATURES])
    out = out.dropna()
    logger.info(
        "drl_dataset built: %s rows=%d cols=%d train_split=%d (%.0f%%)",
        symbol, len(out), out.shape[1], split_idx, 100 * train_frac,
    )
    return out
