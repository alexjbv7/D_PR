"""
Tests para FeatureStreaming — buffers, integración con quant_shared y comportamiento async.

NOTA: Los tests de indicadores técnicos individuales (RSI, ATR, MACD, etc.)
ya NO están aquí. Viven en:

    shared/tests/test_features_parity.py  (36 tests, guardianes del monorepo)

Este archivo testea lo que es responsabilidad de FeatureStreaming:
  - Gestión de buffers por símbolo
  - Delegación correcta a compute_features()
  - Comportamiento con datos insuficientes
  - Propiedades del vector producido
  - Contexto externo propagado correctamente
"""
import numpy as np
import pytest
from app.feature_streaming import FeatureStreaming, FeatureVector
from quant_shared.features.definitions import FEATURE_NAMES, FEATURE_COUNT


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def fs():
    """FeatureStreaming sin Kafka ni Redis — solo cómputo puro."""
    return FeatureStreaming(kafka_servers="", redis_url="")


def _feed_prices(fs, symbol, n=50, start=50_000.0, step=100.0):
    """Alimenta n ticks sintéticos al buffer del símbolo."""
    for i in range(n):
        price = start + i * step
        fs._update_buffers(symbol, {
            "price":         price,
            "high":          price * 1.001,
            "low":           price * 0.999,
            "volume":        100.0,
            "vwap":          price,
            "open_interest": 1_000_000,
        })


# ── Buffer management ─────────────────────────────────────────────────────────

class TestBuffers:
    def test_buffer_initializes_on_first_tick(self, fs):
        fs._update_buffers("BTCUSDT", {"price": 50_000.0})
        assert "BTCUSDT" in fs._closes
        assert len(fs._closes["BTCUSDT"]) == 1

    def test_buffer_accumulates_ticks(self, fs):
        _feed_prices(fs, "BTCUSDT", n=30)
        assert len(fs._closes["BTCUSDT"]) == 30

    def test_buffer_max_size(self, fs):
        _feed_prices(fs, "BTCUSDT", n=fs._CLOSE_WINDOW + 50)
        assert len(fs._closes["BTCUSDT"]) == fs._CLOSE_WINDOW

    def test_zero_price_not_appended(self, fs):
        fs._update_buffers("BTCUSDT", {"price": 0.0})
        assert len(fs._closes.get("BTCUSDT", [])) == 0

    def test_negative_price_not_appended(self, fs):
        fs._update_buffers("BTCUSDT", {"price": -100.0})
        assert len(fs._closes.get("BTCUSDT", [])) == 0

    def test_independent_buffers_per_symbol(self, fs):
        _feed_prices(fs, "BTCUSDT", n=30)
        _feed_prices(fs, "ETHUSDT", n=20, start=3_000.0, step=5.0)
        assert len(fs._closes["BTCUSDT"]) == 30
        assert len(fs._closes["ETHUSDT"]) == 20

    def test_fallback_fields_from_price(self, fs):
        """Si no se pasan high/low, deben usar el precio como fallback."""
        fs._update_buffers("SOLUSDT", {"price": 200.0})
        assert fs._highs["SOLUSDT"][-1] == 200.0
        assert fs._lows["SOLUSDT"][-1]  == 200.0

    def test_vwap_fallback_to_close(self, fs):
        fs._update_buffers("BTCUSDT", {"price": 50_000.0})
        assert fs._vwaps["BTCUSDT"][-1] == 50_000.0


# ── Vector building ───────────────────────────────────────────────────────────

class TestBuildVector:
    async def test_returns_feature_vector_after_warmup(self, fs):
        symbol = "BTCUSDT"
        _feed_prices(fs, symbol, n=50)
        fv = await fs._build_vector(symbol, {"price": 55_000.0, "volume": 120.0})
        assert fv is not None
        assert isinstance(fv, FeatureVector)

    async def test_returns_none_with_insufficient_data(self, fs):
        symbol = "SOLUSDT"
        _feed_prices(fs, symbol, n=5)  # < MIN_BARS
        fv = await fs._build_vector(symbol, {"price": 100.0})
        assert fv is None

    async def test_symbol_in_vector(self, fs):
        _feed_prices(fs, "ETHUSDT", n=50, start=3_000.0, step=5.0)
        fv = await fs._build_vector("ETHUSDT", {"price": 3_250.0})
        assert fv is not None
        assert fv.symbol == "ETHUSDT"

    async def test_no_nan_in_produced_vector(self, fs):
        _feed_prices(fs, "BTCUSDT", n=60)
        fv = await fs._build_vector("BTCUSDT", {"price": 56_000.0, "volume": 80.0})
        assert fv is not None
        arr = fv.to_array()
        assert not np.any(np.isnan(arr)), f"NaN found: {fv.to_dict()}"
        assert not np.any(np.isinf(arr)), f"Inf found: {fv.to_dict()}"

    async def test_vector_length_is_19(self, fs):
        """El array producido debe tener exactamente FEATURE_COUNT elementos."""
        _feed_prices(fs, "BTCUSDT", n=60)
        fv = await fs._build_vector("BTCUSDT", {"price": 56_000.0})
        assert fv is not None
        assert len(fv.to_array()) == FEATURE_COUNT  # 19

    async def test_to_dict_contains_all_canonical_features(self, fs):
        """El dict producido contiene exactamente los 19 features canónicos."""
        _feed_prices(fs, "BTCUSDT", n=60)
        fv = await fs._build_vector("BTCUSDT", {"price": 56_000.0})
        assert fv is not None
        d = fv.to_dict()
        for name in FEATURE_NAMES:
            assert name in d, f"Feature canónica '{name}' ausente del vector"

    async def test_momentum_positive_for_rising_prices(self, fs):
        """Precios estrictamente crecientes → mom_1h > 0."""
        _feed_prices(fs, "BNBUSDT", n=50, start=400.0, step=2.0)
        fv = await fs._build_vector("BNBUSDT", {"price": 500.0})
        if fv:
            assert fv.mom_1h > 0

    async def test_microstructure_propagated(self, fs):
        """ob_imbalance y spread_bps del tick se propagan al vector."""
        _feed_prices(fs, "BTCUSDT", n=50)
        tick = {"price": 55_000.0, "ob_imbalance": 0.42, "spread_bps": 5.0}
        fs._update_buffers("BTCUSDT", tick)
        fv = await fs._build_vector("BTCUSDT", tick)
        assert fv is not None
        assert fv.ob_imbalance == pytest.approx(0.42)
        assert fv.spread_bps   == pytest.approx(5.0)

    async def test_funding_rate_propagated(self, fs):
        _feed_prices(fs, "BTCUSDT", n=50)
        tick = {"price": 55_000.0, "funding_rate": 0.0003}
        fs._update_buffers("BTCUSDT", tick)
        fv = await fs._build_vector("BTCUSDT", tick)
        assert fv is not None
        assert fv.funding_rate == pytest.approx(0.0003)


# ── Context injection ─────────────────────────────────────────────────────────

class TestContextInjection:
    async def test_default_context_without_redis(self, fs):
        """Sin Redis, regime_id=0, macro_leverage=1.0, reserve_z=0, whale=0."""
        _feed_prices(fs, "BTCUSDT", n=50)
        fv = await fs._build_vector("BTCUSDT", {"price": 55_000.0})
        assert fv is not None
        assert fv.regime_id      == pytest.approx(0.0)
        assert fv.macro_leverage == pytest.approx(1.0)
        assert fv.reserve_z      == pytest.approx(0.0)
        assert fv.whale_sentiment == pytest.approx(0.0)

    async def test_get_context_returns_defaults_without_redis(self, fs):
        regime, lev, rz, ws = await fs._get_context("BTC")
        assert regime == 0.0
        assert lev    == 1.0
        assert rz     == 0.0
        assert ws     == 0.0


# ── FeatureVector (importado de quant_shared) ─────────────────────────────────

class TestFeatureVectorCompat:
    def test_feature_vector_importable_from_app(self):
        """Garantiza que los imports existentes de app.feature_streaming siguen funcionando."""
        from app.feature_streaming import FeatureVector as FV
        fv = FV(symbol="X", ts="2026-01-01")
        assert fv.symbol == "X"

    def test_to_array_length_is_19(self):
        fv = FeatureVector(symbol="X", ts="2026-01-01")
        assert len(fv.to_array()) == 19

    def test_to_array_dtype_float32(self):
        fv = FeatureVector(symbol="X", ts="2026-01-01")
        assert fv.to_array().dtype == np.float32

    def test_default_macro_leverage_is_one(self):
        fv = FeatureVector(symbol="X", ts="2026-01-01")
        assert fv.macro_leverage == pytest.approx(1.0)

    def test_feature_vector_is_quant_shared_type(self):
        """FeatureVector importada de app.feature_streaming ES la de quant_shared."""
        from app.feature_streaming import FeatureVector as AppFV
        from quant_shared.features.compute import FeatureVector as SharedFV
        assert AppFV is SharedFV  # mismo objeto, no copia
