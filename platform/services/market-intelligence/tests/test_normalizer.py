"""Tests for DataNormalizer — cross-source market data normalization."""
import pytest
from app.normalizer import DataNormalizer, NormalizedTick


@pytest.fixture
def norm():
    return DataNormalizer(kafka_servers="")


class TestNormalizeTick:
    def test_basic_binance_agg_trade(self, norm):
        raw = {"s": "BTCUSDT", "p": "65000.0", "q": "0.5", "b": "64990.0", "a": "65010.0"}
        tick = norm.normalize_tick(raw, source="binance")
        assert tick.symbol == "BTCUSDT"
        assert tick.price == pytest.approx(65000.0)
        assert tick.volume == pytest.approx(0.5)
        assert tick.source == "binance"

    def test_spread_computed(self, norm):
        raw = {"s": "BTCUSDT", "p": "65000", "b": "64990", "a": "65010"}
        tick = norm.normalize_tick(raw)
        # spread = (65010 - 64990) / 65000 * 10000 ≈ 3.08 bps
        assert tick.spread_bps == pytest.approx((65010 - 64990) / 65000 * 10_000, rel=1e-2)

    def test_usdc_alias_to_usdt(self, norm):
        raw = {"s": "BTCUSDC", "p": "65000"}
        tick = norm.normalize_tick(raw)
        assert tick.symbol == "BTCUSDT"

    def test_spread_zero_if_no_bid_ask(self, norm):
        raw = {"s": "ETHUSDT", "p": "3500"}
        tick = norm.normalize_tick(raw)
        assert tick.spread_bps == 0.0

    def test_ob_imbalance_from_top_of_book(self, norm):
        # bid_qty=80, ask_qty=20 → imbalance = (80-20)/(80+20) = 0.6
        raw = {"s": "SOLUSDT", "p": "150", "B": "80", "A": "20"}
        tick = norm.normalize_tick(raw)
        assert tick.ob_imbalance == pytest.approx(0.6, rel=1e-2)

    def test_ob_imbalance_from_l2(self, norm):
        raw = {
            "s": "ETHUSDT", "p": "3500",
            "bids": [["3499", "10"], ["3498", "5"]],
            "asks": [["3501", "3"], ["3502", "2"]],
        }
        tick = norm.normalize_tick(raw)
        # bids=15, asks=5 → imbalance = 10/20 = 0.5
        assert tick.ob_imbalance == pytest.approx(0.5, rel=1e-2)

    def test_funding_rate_normalized_bitmex(self, norm):
        # BitMEX gives daily rate = 3× 8h rate → we divide by 3
        raw = {"s": "XBTUSD", "r": "0.0003"}
        tick = norm.normalize_tick(raw, source="bitmex")
        assert tick.funding_rate == pytest.approx(0.0001, rel=1e-6)

    def test_funding_rate_binance_unchanged(self, norm):
        raw = {"s": "BTCUSDT", "r": "0.0001"}
        tick = norm.normalize_tick(raw, source="binance")
        assert tick.funding_rate == pytest.approx(0.0001, rel=1e-6)

    def test_missing_fields_default_to_zero(self, norm):
        tick = norm.normalize_tick({"s": "BTCUSDT"})
        assert tick.price == 0.0
        assert tick.volume == 0.0
        assert tick.spread_bps == 0.0

    def test_open_interest_field(self, norm):
        raw = {"s": "BTCUSDT", "p": "65000", "openInterest": "1500000000"}
        tick = norm.normalize_tick(raw)
        assert tick.open_interest == pytest.approx(1_500_000_000, rel=1e-3)


class TestNormalizeOHLCV:
    def test_binance_kline_format(self, norm):
        raw = {
            "k": {
                "s": "BTCUSDT", "t": 1700000000000,
                "o": "64500", "h": "65000", "l": "64400",
                "c": "64900", "v": "10.5", "q": "680000",
            }
        }
        candle = norm.normalize_ohlcv(raw)
        assert candle.symbol == "BTCUSDT"
        assert candle.open == pytest.approx(64500.0)
        assert candle.close == pytest.approx(64900.0)
        assert candle.volume == pytest.approx(10.5)

    def test_vwap_computed(self, norm):
        raw = {
            "k": {
                "s": "BTCUSDT", "t": 0,
                "o": "100", "h": "110", "l": "90", "c": "105",
                "v": "20", "q": "2100",   # vwap = 2100/20 = 105
            }
        }
        candle = norm.normalize_ohlcv(raw)
        assert candle.vwap == pytest.approx(105.0)

    def test_canonical_symbol_applied(self, norm):
        raw = {"symbol": "BTCUSDC", "open": "65000", "high": "65000",
               "low": "65000", "close": "65000", "volume": "1", "ts": 0}
        candle = norm.normalize_ohlcv(raw)
        assert candle.symbol == "BTCUSDT"


class TestHelpers:
    def test_canonical_symbol_slash_removed(self, norm):
        assert norm._canonical_symbol("BTC/USDT") == "BTCUSDT"

    def test_canonical_symbol_dash_removed(self, norm):
        assert norm._canonical_symbol("ETH-USDT") == "ETHUSDT"

    def test_canonical_symbol_lowercase(self, norm):
        assert norm._canonical_symbol("btcusdt") == "BTCUSDT"
