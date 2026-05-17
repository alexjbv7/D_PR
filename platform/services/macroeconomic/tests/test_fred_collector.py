"""
Tests para FredCollector (versión OpenBB).

Cubre:
  - _fetch_from_openbb produce lista correcta
  - _rolling_z cálculo correcto
  - collect_all maneja errores de series individuales sin abortar
  - get_cached retorna formato esperado
  - get_yield_curve_features calcula slope correctamente
  - _persist no propaga excepciones de DB
  - Sin OpenBB (obb=None) retorna lista vacía
"""
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Mocks de dependencias externas antes de importar app ─────────────────────
for _mod in ["openbb", "libs", "libs.shared", "libs.shared.events",
             "libs.shared.kafka_client", "libs.shared.redis_client",
             "libs.shared.db"]:
    sys.modules.setdefault(_mod, MagicMock())

# Configurar KafkaTopics mock
_topics = MagicMock()
_topics.MACRO_DATA = "los_ojos.macro.data"
sys.modules["libs.shared.events"].KafkaTopics  = _topics
sys.modules["libs.shared.events"].MacroDataEvent = MagicMock(return_value=MagicMock())

from app.fred_collector import FredCollector, FRED_SERIES


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_producer():
    p = AsyncMock()
    p.send = AsyncMock()
    return p


@pytest.fixture
def mock_cache():
    c = AsyncMock()
    c.get = AsyncMock(return_value=None)
    c.set = AsyncMock()
    return c


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def mock_obb():
    """Mock del SDK de OpenBB para FRED."""
    import pandas as pd
    obb = MagicMock()
    result = MagicMock()
    result.to_df.return_value = pd.DataFrame({
        "date":  ["2024-01-01", "2024-02-01", "2024-03-01",
                  "2024-04-01", "2024-05-01"],
        "value": [4.0, 4.1, 4.2, 4.0, 3.9],
    })
    obb.economy.fred_series.return_value = result
    return obb


@pytest.fixture
def collector(mock_producer, mock_cache, mock_db, mock_obb):
    """FredCollector con OpenBB mockeado — sin calls reales."""
    fc = FredCollector(
        producer=mock_producer,
        cache=mock_cache,
        db=mock_db,
        api_key="",
    )
    fc._obb = mock_obb
    return fc


@pytest.fixture
def collector_no_obb(mock_producer, mock_cache, mock_db):
    """FredCollector sin OpenBB disponible."""
    fc = FredCollector(
        producer=mock_producer,
        cache=mock_cache,
        db=mock_db,
        api_key="",
    )
    fc._obb = None
    return fc


# ── Tests: _fetch_from_openbb ─────────────────────────────────────────────────

class TestFetchFromOpenBB:
    async def test_returns_list_of_dicts(self, collector):
        data = await collector._fetch_from_openbb("UNRATE")
        assert isinstance(data, list)
        assert len(data) == 5

    async def test_each_record_has_date_and_value(self, collector):
        data = await collector._fetch_from_openbb("DGS10")
        for record in data:
            assert "date"  in record
            assert "value" in record
            assert isinstance(record["value"], float)

    async def test_sorted_chronologically(self, collector):
        data = await collector._fetch_from_openbb("CPIAUCSL")
        dates = [d["date"] for d in data]
        assert dates == sorted(dates)

    async def test_no_obb_returns_empty(self, collector_no_obb):
        data = await collector_no_obb._fetch_from_openbb("UNRATE")
        assert data == []

    async def test_openbb_error_returns_empty(self, collector, mock_obb):
        mock_obb.economy.fred_series.side_effect = RuntimeError("FRED error")
        data = await collector._fetch_from_openbb("UNRATE")
        assert data == []

    async def test_filters_nan_values(self, collector, mock_obb):
        """Valores NaN no deben aparecer en el resultado."""
        import pandas as pd
        import numpy as np
        result = MagicMock()
        result.to_df.return_value = pd.DataFrame({
            "date":  ["2024-01-01", "2024-02-01", "2024-03-01"],
            "value": [4.0, float("nan"), 4.2],
        })
        mock_obb.economy.fred_series.return_value = result
        data = await collector._fetch_from_openbb("UNRATE")
        assert all(str(d["value"]) not in ("nan", "NaN") for d in data)
        assert len(data) == 2  # NaN filtrado


# ── Tests: _rolling_z ────────────────────────────────────────────────────────

class TestRollingZ:
    def test_positive_z_for_above_average(self, collector):
        values = [4.0, 4.0, 4.0, 4.0, 5.0]  # último > media
        z = collector._rolling_z(values, 5)
        assert z > 0

    def test_negative_z_for_below_average(self, collector):
        values = [4.0, 4.0, 4.0, 4.0, 3.0]  # último < media
        z = collector._rolling_z(values, 5)
        assert z < 0

    def test_zero_for_constant_series(self, collector):
        """Serie constante → std=0 → z=0 (sin división por cero)."""
        values = [4.0, 4.0, 4.0, 4.0, 4.0]
        z = collector._rolling_z(values, 5)
        assert z == pytest.approx(0.0)

    def test_too_few_values_returns_zero(self, collector):
        z = collector._rolling_z([4.0, 4.1], 5)
        assert z == pytest.approx(0.0)

    def test_window_limits_lookback(self, collector):
        """Window de 3 usa solo los últimos 3 valores, no todos."""
        values = list(range(100)) + [1000]  # spike al final
        z_narrow = collector._rolling_z(values, 3)
        z_wide   = collector._rolling_z(values, 50)
        # Ventana más estrecha → más impacto del spike
        assert z_narrow > z_wide


# ── Tests: collect_all ────────────────────────────────────────────────────────

class TestCollectAll:
    async def test_collect_all_runs_without_exception(self, collector):
        """collect_all no debe lanzar aunque algunas series fallen."""
        await collector.collect_all()

    async def test_partial_failure_doesnt_stop_loop(self, collector, mock_obb):
        """Si una serie falla, las demás deben procesarse igual."""
        call_count = 0
        orig_side = mock_obb.economy.fred_series.side_effect

        def sometimes_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fallo deliberado")
            return mock_obb.economy.fred_series.return_value

        mock_obb.economy.fred_series.side_effect = sometimes_fail
        # No debe lanzar
        await collector.collect_all()

    async def test_kafka_send_called_per_series(self, collector, mock_producer):
        await collector.collect_all()
        assert mock_producer.send.call_count > 0

    async def test_redis_set_called_per_series(self, collector, mock_cache):
        await collector.collect_all()
        assert mock_cache.set.call_count > 0


# ── Tests: get_cached ─────────────────────────────────────────────────────────

class TestGetCached:
    async def test_empty_before_collect(self, collector):
        result = collector.get_cached()
        assert result == {}

    async def test_contains_series_after_collect(self, collector):
        await collector._collect_series("UNRATE")
        cached = collector.get_cached()
        assert "UNRATE" in cached

    async def test_cached_has_required_keys(self, collector):
        await collector._collect_series("DGS10")
        entry = collector.get_cached().get("DGS10", {})
        for key in ("value", "z_score", "date", "name", "category"):
            assert key in entry, f"Missing key: {key}"


# ── Tests: get_yield_curve_features ──────────────────────────────────────────

class TestYieldCurveFeatures:
    async def test_slope_computed_from_t10_t2(self, collector, mock_cache):
        mock_cache.get.side_effect = lambda key: {
            "macro:fred:DGS10": {"value": 4.5, "z_score": 0.0},
            "macro:fred:DGS2":  {"value": 4.8, "z_score": 0.0},
            "macro:fred:T10Y2Y": {"value": -0.3, "z_score": -1.5},
        }.get(key)

        features = await collector.get_yield_curve_features()
        assert features["yield_curve_slope"] == pytest.approx(4.5 - 4.8)

    async def test_inversion_flag_when_slope_negative(self, collector, mock_cache):
        mock_cache.get.side_effect = lambda key: {
            "macro:fred:DGS10": {"value": 4.0, "z_score": 0.0},
            "macro:fred:DGS2":  {"value": 4.8, "z_score": 0.0},
            "macro:fred:T10Y2Y": None,
        }.get(key)
        features = await collector.get_yield_curve_features()
        assert features.get("yield_curve_inverted") == pytest.approx(1.0)

    async def test_no_data_returns_empty(self, collector, mock_cache):
        mock_cache.get.return_value = None
        features = await collector.get_yield_curve_features()
        assert features == {}


# ── Tests: _persist ───────────────────────────────────────────────────────────

class TestPersist:
    async def test_persist_calls_db_execute(self, collector, mock_db):
        await collector._persist("UNRATE", "2026-01-01", 4.2)
        mock_db.execute.assert_called_once()

    async def test_persist_db_error_is_silent(self, collector, mock_db):
        mock_db.execute.side_effect = Exception("DB down")
        # No debe lanzar excepción
        await collector._persist("UNRATE", "2026-01-01", 4.2)


# ── Tests: catálogo de series ─────────────────────────────────────────────────

class TestSeriesCatalog:
    def test_contains_at_least_22_series(self):
        assert len(FRED_SERIES) >= 22

    def test_all_series_have_required_fields(self):
        for sid, meta in FRED_SERIES.items():
            assert "name"     in meta, f"{sid} missing 'name'"
            assert "freq"     in meta, f"{sid} missing 'freq'"
            assert "category" in meta, f"{sid} missing 'category'"

    def test_new_series_present(self):
        """JTSJOL (JOLTS) y DTWEXBGS (DXY broad) deben estar en el catálogo."""
        assert "JTSJOL"     in FRED_SERIES
        assert "DTWEXBGS"   in FRED_SERIES
        assert "SAHMREALTIME" in FRED_SERIES

    def test_no_duplicate_ids(self):
        ids = list(FRED_SERIES.keys())
        assert len(ids) == len(set(ids))
