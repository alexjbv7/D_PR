"""Unit tests for the BTC/ETH/EUR 4H experiment data layer.

Pure pandas/numpy: no network, no Alpaca/yfinance/torch. The yfinance call is
isolated in ``data_sources._yf_download`` and monkeypatched, mirroring the
``data.drl_dataset._fetch_ohlcv`` testing pattern.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.btc_eth_eur_4h import config as cfgmod
from experiments.btc_eth_eur_4h import data_sources as ds


# --------------------------------------------------------------------------- #
# Routing + annualization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "symbol,expected",
    [("EUR/USD", True), ("GBP/USD", True), ("USD/JPY", True),
     ("BTC/USD", False), ("ETH/USD", False), ("SOL/USD", False)],
)
def test_is_fx_routing(symbol: str, expected: bool) -> None:
    assert cfgmod.is_fx(symbol) is expected


def test_periods_per_year_4h() -> None:
    assert cfgmod.periods_per_year("BTC/USD") == cfgmod.CRYPTO_PPY_4H == 2190
    assert cfgmod.periods_per_year("ETH/USD") == 2190
    assert cfgmod.periods_per_year("EUR/USD") == cfgmod.FX_PPY_4H == 1560


def test_periods_per_year_rejects_non_4h() -> None:
    with pytest.raises(NotImplementedError):
        cfgmod.periods_per_year("BTC/USD", timeframe="1d")


def test_fx_to_yahoo() -> None:
    assert ds._fx_to_yahoo("EUR/USD") == "EURUSD=X"
    assert ds._fx_to_yahoo("usd/jpy") == "USDJPY=X"


def test_config_start_for_windows_fx() -> None:
    cfg = cfgmod.ExperimentConfig(end="2026-01-01", fx_lookback_days=700)
    # crypto keeps the long history (2021); FX is windowed to the recent
    # lookback (~2024), so the FX start is MORE recent than crypto's.
    assert cfg.start_for("BTC/USD") == "2021-01-01"
    assert cfg.start_for("EUR/USD") > cfg.start_for("BTC/USD")
    assert cfg.start_for("EUR/USD").startswith("2024")


# --------------------------------------------------------------------------- #
# 1H -> 4H resampling (OHLC aggregation correctness)
# --------------------------------------------------------------------------- #


def _synthetic_1h(n: int = 8) -> pd.DataFrame:
    """8 hourly bars from 2024-01-01 00:00 with identifiable OHLC extremes."""
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="1h")  # tz-naive
    return pd.DataFrame(
        {
            "open": np.arange(1, n + 1, dtype=float),
            "high": np.arange(10, 10 + n, dtype=float),
            "low": np.arange(1, n + 1, dtype=float) / 10.0,
            "close": np.arange(2, n + 2, dtype=float),
            "volume": np.full(n, 100.0),
        },
        index=idx,
    )


def test_resample_1h_to_4h(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ds, "_yf_download", lambda *a, **k: _synthetic_1h())
    out = ds.load_fx_yfinance("EUR/USD", "2024-01-01", "2024-01-02")

    assert str(out.index.tz) == "UTC"
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 2  # 8 hourly bars -> two 4H bars

    first = out.iloc[0]
    assert first["open"] == 1.0          # first open of bars 0..3
    assert first["high"] == 13.0         # max(10,11,12,13)
    assert first["low"] == pytest.approx(0.1)   # min(0.1..0.4)
    assert first["close"] == 5.0         # last close of bars 0..3
    assert first["volume"] == 400.0      # sum of 4 * 100

    second = out.iloc[1]
    assert second["open"] == 5.0
    assert second["close"] == 9.0
    assert second["volume"] == 400.0


def test_resample_drops_empty_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    # A weekend-style gap: 4 bars, then nothing for 8h, then 4 bars.
    a = _synthetic_1h(4)
    b = _synthetic_1h(4)
    b.index = pd.date_range("2024-01-01 12:00", periods=4, freq="1h")
    gapped = pd.concat([a, b])
    monkeypatch.setattr(ds, "_yf_download", lambda *a_, **k_: gapped)

    out = ds.load_fx_yfinance("EUR/USD", "2024-01-01", "2024-01-02")
    # Only the two populated 4H buckets survive (00:00 and 12:00) — no NaN bars.
    assert len(out) == 2
    assert not out.isna().any().any()


def test_load_raw_ohlcv_routes_fx_to_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ds, "_yf_download", lambda *a, **k: _synthetic_1h())
    out = ds.load_raw_ohlcv("EUR/USD", "2024-01-01", "2024-01-02", timeframe="4h")
    assert len(out) == 2
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
