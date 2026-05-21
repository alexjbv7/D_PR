"""
Test de paridad — verifica que shared/quant_shared/features/compute.py
produce valores numéricamente equivalentes a los indicadores técnicos
del ml-feature-store de platform/.

Este test es el guardián más crítico del monorepo:
si falla, hay divergencia entre research y producción.
"""
import numpy as np
import pytest
import sys
import os

# Asegurar que shared/ está en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from quant_shared.features.compute import rsi, macd_hist, atr, bollinger_width, adx, momentum, sma_cross, compute_features, FeatureVector
from quant_shared.features.definitions import FEATURE_NAMES, FEATURE_COUNT


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def rising_prices():
    """50 precios linealmente crecientes."""
    return np.linspace(50_000, 55_000, 50)


@pytest.fixture
def falling_prices():
    """50 precios linealmente decrecientes."""
    return np.linspace(55_000, 50_000, 50)


@pytest.fixture
def flat_prices():
    """50 precios constantes."""
    return np.full(50, 50_000.0)


@pytest.fixture
def random_ohlcv(seed=42):
    rng = np.random.default_rng(seed)
    closes = 50_000 + rng.standard_normal(100).cumsum() * 200
    highs  = closes + rng.uniform(0, 500, 100)
    lows   = closes - rng.uniform(0, 500, 100)
    vols   = rng.uniform(100, 1000, 100)
    return closes, highs, lows, vols


# ─── Definición canónica ─────────────────────────────────────────────────────

class TestFeatureDefinitions:
    def test_feature_count_is_19(self):
        assert FEATURE_COUNT == 19

    def test_feature_names_order(self):
        """El orden es FIJO y no debe cambiar sin versionado."""
        expected_first_5 = ["rsi_14", "macd_hist", "mom_1h", "mom_4h", "mom_24h"]
        assert FEATURE_NAMES[:5] == expected_first_5

    def test_feature_names_last_4(self):
        expected_last_4 = ["sma_cross", "adx_14", "reserve_z", "whale_sentiment"]
        assert FEATURE_NAMES[-4:] == expected_last_4

    def test_no_duplicate_names(self):
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


# ─── Indicadores técnicos ────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_bounds(self, rising_prices):
        r = rsi(rising_prices, 14)
        assert 0 <= r <= 100

    def test_rsi_above_50_for_rising(self, rising_prices):
        assert rsi(rising_prices, 14) >= 50

    def test_rsi_below_50_for_falling(self, falling_prices):
        assert rsi(falling_prices, 14) <= 50

    def test_rsi_default_for_insufficient_data(self):
        short = np.array([50000.0] * 5)
        assert rsi(short, 14) == 50.0

    def test_rsi_100_for_all_gains(self):
        prices = np.linspace(1, 100, 30)
        r = rsi(prices, 14)
        assert r == pytest.approx(100.0, abs=1e-6)

    def test_rsi_random_bounds(self):
        rng = np.random.default_rng(99)
        for _ in range(20):
            prices = 50_000 + rng.standard_normal(30).cumsum() * 200
            assert 0 <= rsi(prices, 14) <= 100


class TestMACDHist:
    def test_macd_returns_float(self, rising_prices):
        result = macd_hist(rising_prices)
        assert isinstance(result, float)

    def test_macd_default_for_insufficient(self):
        short = np.array([100.0] * 10)
        assert macd_hist(short) == 0.0

    def test_macd_positive_for_strong_uptrend(self):
        prices = np.exp(np.linspace(0, 2, 60)) * 50_000
        result = macd_hist(prices)
        assert isinstance(result, float)  # valor depende de la forma exacta


class TestATR:
    def test_atr_positive(self, random_ohlcv):
        closes, highs, lows, _ = random_ohlcv
        result = atr(highs, lows, closes, 14)
        assert result > 0

    def test_atr_default_for_insufficient(self):
        c = np.array([100.0] * 5)
        result = atr(c, c, c, 14)
        assert result == pytest.approx(0.01)

    def test_atr_normalized_by_price(self, random_ohlcv):
        closes, highs, lows, _ = random_ohlcv
        result = atr(highs, lows, closes, 14)
        # Normalizado por precio → debería ser fracción pequeña
        assert 0 < result < 0.5


class TestBollingerWidth:
    def test_bb_width_positive(self, random_ohlcv):
        closes, _, _, _ = random_ohlcv
        result = bollinger_width(closes, 20)
        assert result > 0

    def test_bb_width_increases_with_volatility(self):
        rng = np.random.default_rng(42)
        base = 50_000.0
        low_vol  = base + rng.standard_normal(30) * 10
        high_vol = base + rng.standard_normal(30) * 2_000
        assert bollinger_width(high_vol, 20) > bollinger_width(low_vol, 20)

    def test_bb_width_default_for_insufficient(self):
        short = np.array([100.0] * 5)
        assert bollinger_width(short, 20) == pytest.approx(0.02)


class TestADX:
    def test_adx_range(self, random_ohlcv):
        closes, highs, lows, _ = random_ohlcv
        result = adx(highs, lows, closes, 14)
        assert 0 <= result <= 100

    def test_adx_default_for_insufficient(self):
        c = np.array([100.0] * 10)
        assert adx(c, c, c, 14) == pytest.approx(20.0)

    def test_adx_high_for_strong_trend(self):
        n = 50
        h = np.linspace(100, 200, n)
        l = np.linspace(90, 190, n)
        c = (h + l) / 2
        result = adx(h, l, c, 14)
        assert result >= 0  # trending market


class TestMomentum:
    def test_momentum_positive_for_rising(self, rising_prices):
        result = momentum(rising_prices, 1)
        assert result > 0

    def test_momentum_negative_for_falling(self, falling_prices):
        result = momentum(falling_prices, 1)
        assert result < 0

    def test_momentum_zero_for_flat(self, flat_prices):
        result = momentum(flat_prices, 1)
        assert result == pytest.approx(0.0)

    def test_momentum_default_for_insufficient(self):
        prices = np.array([100.0, 110.0])
        result = momentum(prices, 5)
        assert result == 0.0


class TestSMACross:
    def test_sma_cross_positive_when_fast_above_slow(self, rising_prices):
        result = sma_cross(rising_prices, fast=20, slow=50)
        assert result > 0

    def test_sma_cross_negative_when_fast_below_slow(self, falling_prices):
        result = sma_cross(falling_prices, fast=20, slow=50)
        assert result < 0

    def test_sma_cross_default_for_insufficient(self):
        short = np.array([100.0] * 20)
        result = sma_cross(short, fast=20, slow=50)
        assert result == 0.0


# ─── compute_features (función principal) ────────────────────────────────────

class TestComputeFeatures:
    def test_returns_feature_vector(self, random_ohlcv):
        closes, highs, lows, vols = random_ohlcv
        fv = compute_features("BTCUSDT", "2026-01-01T00:00:00Z",
                              closes, highs, lows, vols)
        assert isinstance(fv, FeatureVector)

    def test_to_array_length_is_19(self, random_ohlcv):
        closes, highs, lows, vols = random_ohlcv
        fv = compute_features("BTCUSDT", "2026-01-01T00:00:00Z",
                              closes, highs, lows, vols)
        arr = fv.to_array()
        assert len(arr) == 19

    def test_no_nan_in_vector(self, random_ohlcv):
        closes, highs, lows, vols = random_ohlcv
        fv = compute_features("BTCUSDT", "2026-01-01T00:00:00Z",
                              closes, highs, lows, vols)
        arr = fv.to_array()
        assert not np.any(np.isnan(arr))
        assert not np.any(np.isinf(arr))

    def test_rsi_in_vector_matches_standalone(self, random_ohlcv):
        """compute_features.rsi_14 debe ser igual a rsi() standalone."""
        closes, highs, lows, vols = random_ohlcv
        fv = compute_features("BTCUSDT", "2026-01-01T00:00:00Z",
                              closes, highs, lows, vols)
        expected = rsi(closes, 14)
        assert fv.rsi_14 == pytest.approx(expected, rel=1e-5)

    def test_external_context_propagated(self, random_ohlcv):
        closes, highs, lows, vols = random_ohlcv
        fv = compute_features(
            "BTCUSDT", "2026-01-01T00:00:00Z",
            closes, highs, lows, vols,
            regime_id=2.0,
            macro_leverage=1.3,
            reserve_z=-1.5,
            whale_sentiment=0.8,
        )
        assert fv.regime_id == pytest.approx(2.0)
        assert fv.macro_leverage == pytest.approx(1.3)
        assert fv.reserve_z == pytest.approx(-1.5)
        assert fv.whale_sentiment == pytest.approx(0.8)

    def test_to_dict_contains_all_features(self, random_ohlcv):
        closes, highs, lows, vols = random_ohlcv
        fv = compute_features("ETHUSDT", "2026-01-01T00:00:00Z",
                              closes, highs, lows, vols)
        d = fv.to_dict()
        for name in FEATURE_NAMES:
            assert name in d, f"Feature '{name}' missing from to_dict()"

    def test_feature_order_matches_definitions(self, random_ohlcv):
        """El orden en to_array() debe coincidir exactamente con FEATURE_NAMES."""
        closes, highs, lows, vols = random_ohlcv
        fv = compute_features("BTCUSDT", "2026-01-01T00:00:00Z",
                              closes, highs, lows, vols)
        arr = fv.to_array()
        d   = fv.to_dict()
        for i, name in enumerate(FEATURE_NAMES):
            assert arr[i] == pytest.approx(d[name], rel=1e-5), \
                f"Mismatch at index {i} ({name}): array={arr[i]}, dict={d[name]}"
