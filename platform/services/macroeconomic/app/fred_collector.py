"""
FRED Collector — Federal Reserve Economic Data via OpenBB Platform.
==================================================================
Descarga y procesa las series macro más relevantes para trading.

MIGRACIÓN 2026-05-14:
  fredapi (cliente directo de FRED) reemplazado por OpenBB FRED provider.
  OpenBB normaliza la respuesta, gestiona rate limiting y soporta
  múltiples providers como fallback (OECD, IMF).

  API pública de esta clase: sin cambios — mismos métodos, mismos
  Redis keys, mismos Kafka topics.

Series por categoría:

  INFLACIÓN      : CPIAUCSL, PCEPI, CPILFESL (core CPI), PPIFIS
  TIPO DE INTERÉS: DFF (fed funds), T10Y2Y (yield curve), T10YIE (breakeven)
  EMPLEO         : UNRATE, PAYEMS (NFP), ICSA (initial claims), JTSJOL (JOLTS)
  CRECIMIENTO    : GDP (trimestral), INDPRO, RSXFS (retail ex-auto)
  DINERO/CRÉDITO : M2SL, TOTCI (total credit)
  SENTIMIENTO    : UMCSENT (U Mich), CSCICP03 (Conf Board LEI)
  HOUSING        : HOUST (housing starts), CS20RPSNSA (Case-Shiller)
  GLOBAL         : DTWEXBGS (DXY broad), DEXUSEU, DEXJPUS
  RECESIÓN       : SAHMREALTIME, USREC

Para cada serie se calcula:
  - Cambio MoM
  - Z-score rolling 36 períodos
  - Estado del ciclo (por encima / por debajo de tendencia)
"""
from __future__ import annotations

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Optional

import pandas as pd

from libs.shared.events import MacroDataEvent, KafkaTopics
from libs.shared.kafka_client import KafkaProducerClient
from libs.shared.redis_client import RedisCache, TTL
from libs.shared.db import PostgresPool

logger = logging.getLogger(__name__)

POLL_INTERVAL_H = float(4)  # cada 4 horas por defecto

# Catálogo completo de series — ampliado con JOLTS y CONFBoard
FRED_SERIES: dict[str, dict] = {
    "CPIAUCSL":   {"name": "CPI All Urban",        "freq": "monthly",  "category": "inflation"},
    "CPILFESL":   {"name": "CPI Core (ex F&E)",    "freq": "monthly",  "category": "inflation"},
    "PCEPI":      {"name": "PCE Price Index",       "freq": "monthly",  "category": "inflation"},
    "PPIFIS":     {"name": "PPI Final Demand",      "freq": "monthly",  "category": "inflation"},
    "DFF":        {"name": "Fed Funds Rate",        "freq": "daily",    "category": "rates"},
    "T10Y2Y":     {"name": "10Y-2Y Yield Curve",   "freq": "daily",    "category": "rates"},
    "T10YIE":     {"name": "10Y Breakeven Inf",    "freq": "daily",    "category": "rates"},
    "DGS10":      {"name": "10Y Treasury",          "freq": "daily",    "category": "rates"},
    "DGS2":       {"name": "2Y Treasury",           "freq": "daily",    "category": "rates"},
    "UNRATE":     {"name": "Unemployment Rate",     "freq": "monthly",  "category": "labor"},
    "PAYEMS":     {"name": "Nonfarm Payrolls",      "freq": "monthly",  "category": "labor"},
    "ICSA":       {"name": "Initial Claims",        "freq": "weekly",   "category": "labor"},
    "JTSJOL":     {"name": "Job Openings (JOLTS)",  "freq": "monthly",  "category": "labor"},
    "GDP":        {"name": "GDP (real, annlzd)",    "freq": "quarterly","category": "growth"},
    "INDPRO":     {"name": "Industrial Production", "freq": "monthly",  "category": "growth"},
    "RSXFS":      {"name": "Retail Sales (ex auto)","freq": "monthly",  "category": "growth"},
    "M2SL":       {"name": "M2 Money Supply",       "freq": "monthly",  "category": "monetary"},
    "TOTCI":      {"name": "Total Credit",          "freq": "monthly",  "category": "monetary"},
    "UMCSENT":    {"name": "U Mich Consumer Sent",  "freq": "monthly",  "category": "sentiment"},
    "HOUST":      {"name": "Housing Starts",        "freq": "monthly",  "category": "housing"},
    "VIXCLS":     {"name": "VIX (CBOE)",            "freq": "daily",    "category": "risk"},
    "DEXUSEU":    {"name": "USD/EUR Exchange Rate", "freq": "daily",    "category": "fx"},
    "DEXJPUS":    {"name": "JPY/USD Exchange Rate", "freq": "daily",    "category": "fx"},
    "DTWEXBGS":   {"name": "USD Broad Index (DXY)", "freq": "daily",    "category": "fx"},
    "SAHMREALTIME":{"name": "Sahm Rule (RT)",       "freq": "monthly",  "category": "recession"},
    "USREC":      {"name": "Recession Indicator",   "freq": "monthly",  "category": "recession"},
}


class FredCollector:
    """
    Descarga series FRED via OpenBB Platform, calcula estadísticas y emite eventos.

    OpenBB es síncrono → las llamadas se ejecutan en threadpool.
    Compatible drop-in con la versión anterior basada en fredapi:
      mismos métodos públicos, mismos Redis keys, mismos Kafka topics.
    """

    def __init__(
        self,
        producer: KafkaProducerClient,
        cache: RedisCache,
        db: PostgresPool,
        api_key: str = "",          # FRED API key — inyectada en OpenBB
        kafka_servers: str = "",    # legacy — mantenido para compatibilidad
        redis_url: str = "",        # legacy — mantenido para compatibilidad
        postgres_dsn: str = "",     # legacy — mantenido para compatibilidad
    ):
        self._producer = producer
        self._cache    = cache
        self._db       = db
        self._api_key  = api_key
        self._obb      = None           # lazy import
        self._series_cache: dict[str, list[dict]] = {}

        if api_key:
            self._configure_openbb(api_key)

    def _configure_openbb(self, api_key: str) -> None:
        """Inyectar FRED API key en OpenBB al instanciar."""
        try:
            from openbb import obb
            obb.user.credentials.fred_api_key = api_key
            self._obb = obb
            logger.info("FredCollector: OpenBB FRED provider configurado")
        except ImportError:
            logger.warning(
                "FredCollector: openbb no instalado — "
                "instalar con: pip install openbb openbb-fred"
            )

    async def connect(self) -> None:
        """Compatibilidad con versión anterior — configura OpenBB si no se hizo en __init__."""
        if self._obb is None and self._api_key:
            self._configure_openbb(self._api_key)

    async def close(self) -> None:
        """No hay conexiones persistentes — OpenBB es stateless."""
        pass

    # ── Interfaz pública ─────────────────────────────────────────────

    async def run(self) -> None:
        """Polling loop: actualiza cada POLL_INTERVAL_H horas."""
        logger.info("FredCollector started. Polling every %.1fh", POLL_INTERVAL_H)
        while True:
            try:
                await self.collect_all()
            except Exception as exc:
                logger.error("FredCollector error: %s", exc)
            await asyncio.sleep(POLL_INTERVAL_H * 3600)

    async def collect_all(self) -> None:
        """Descarga y procesa todas las series configuradas."""
        tasks = [self._collect_series(sid) for sid in FRED_SERIES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        ok = sum(1 for r in results if not isinstance(r, Exception))
        logger.info("FredCollector: %d/%d series OK", ok, len(FRED_SERIES))

    async def fetch_all(self) -> dict[str, dict]:
        """
        Descarga todas las series y retorna snapshot.
        Compatible con la interfaz que usa macroeconomic/app/main.py.
        """
        await self.collect_all()
        return self.get_cached()

    def get_cached(self) -> dict[str, dict]:
        """
        Retorna el último snapshot en memoria.
        Formato: {series_id: {value, z_score, mom_pct, date, name, category}}
        """
        result = {}
        for sid, data in self._series_cache.items():
            if not data:
                continue
            meta     = FRED_SERIES.get(sid, {})
            values   = [d["value"] for d in data]
            last_val = values[-1]
            prior    = values[-2] if len(values) >= 2 else None

            mom_pct = None
            if prior and prior != 0:
                mom_pct = round((last_val - prior) / abs(prior) * 100, 4)

            z = self._rolling_z(values, min(36, len(values)))

            result[sid] = {
                "value":    last_val,
                "prior":    prior,
                "mom_pct":  mom_pct,
                "z_score":  z,
                "date":     data[-1].get("date", ""),
                "name":     meta.get("name", sid),
                "category": meta.get("category", ""),
            }
        return result

    async def get_series(self, series_id: str, limit: int = 100) -> list[dict]:
        """Retorna últimos `limit` valores de una serie."""
        if series_id not in self._series_cache:
            await self._collect_series(series_id)
        data = self._series_cache.get(series_id, [])
        return data[-limit:]

    async def get_current_snapshot(self) -> dict[str, dict]:
        """
        Snapshot actual de todas las series.
        Intenta Redis primero (para evitar calls redundantes).
        """
        snapshot = {}
        for sid in FRED_SERIES:
            cache_key = f"macro:fred:{sid}"
            cached = await self._cache.get(cache_key)
            if cached:
                snapshot[sid] = cached
        return snapshot

    async def get_yield_curve_features(self) -> dict[str, float]:
        """Features clave de la curva de rendimientos (slope, inversion, z-score)."""
        t10 = await self._cache.get("macro:fred:DGS10")
        t2  = await self._cache.get("macro:fred:DGS2")
        t10y2y = await self._cache.get("macro:fred:T10Y2Y")

        features: dict[str, float] = {}
        if t10 and t2:
            slope = t10["value"] - t2["value"]
            features["yield_curve_slope"]    = slope
            features["yield_curve_inverted"] = float(slope < 0)
        if t10y2y:
            features["yield_curve_t10y2y"]   = t10y2y["value"]
            features["yield_curve_t10y2y_z"] = t10y2y.get("z_score", 0.0)
        return features

    # ── Internals ─────────────────────────────────────────────────────

    async def _collect_series(self, series_id: str) -> None:
        """Descarga una serie via OpenBB y emite evento si hay dato nuevo."""
        meta = FRED_SERIES.get(series_id, {})
        try:
            data = await self._fetch_from_openbb(series_id)
            if not data:
                return

            self._series_cache[series_id] = data

            values     = [d["value"] for d in data]
            last_date  = data[-1]["date"]
            last_value = values[-1]
            prior_val  = values[-2] if len(values) >= 2 else None

            mom_pct = None
            if prior_val and prior_val != 0:
                mom_pct = round((last_value - prior_val) / abs(prior_val) * 100, 4)

            window  = min(36, len(values))
            z_score = self._rolling_z(values, window)

            # ── Kafka event ───────────────────────────────────────────
            event = MacroDataEvent(
                source="fred-collector",
                series_id=series_id,
                series_name=meta.get("name", series_id),
                value=last_value,
                frequency=meta.get("freq", ""),
                release_date=datetime.now(timezone.utc),
                prior_value=prior_val,
            )
            await self._producer.send(KafkaTopics.MACRO_DATA, event)

            # ── Redis cache ───────────────────────────────────────────
            cache_payload = {
                "value":    last_value,
                "prior":    prior_val,
                "mom_pct":  mom_pct,
                "z_score":  z_score,
                "date":     last_date,
                "name":     meta.get("name", series_id),
                "category": meta.get("category", ""),
            }
            await self._cache.set(
                f"macro:fred:{series_id}",
                cache_payload,
                ttl=TTL.get("macro", 86400),
            )

            # ── TimescaleDB ───────────────────────────────────────────
            await self._persist(series_id, last_date, last_value)

        except Exception as exc:
            logger.warning("FredCollector series %s error: %s", series_id, exc)
            raise

    async def _fetch_from_openbb(self, series_id: str) -> list[dict]:
        """
        Descarga una serie FRED via OpenBB en threadpool.

        Retorna lista de dicts [{date: str, value: float}] ordenada
        cronológicamente.
        """
        if self._obb is None:
            logger.debug("FredCollector: OpenBB no disponible, skipping %s", series_id)
            return []

        start = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                partial(
                    self._obb.economy.fred_series,
                    symbol=series_id,
                    start_date=start,
                    provider="fred",
                ),
            )
            df = result.to_df().reset_index()
            df.columns = [c.lower() for c in df.columns]

            # Normalizar nombre de columnas (OpenBB puede variar)
            date_col  = next((c for c in df.columns if "date" in c or "index" in c), df.columns[0])
            value_col = next((c for c in df.columns if c not in (date_col,)), "value")

            records = [
                {"date": str(row[date_col]), "value": float(row[value_col])}
                for _, row in df.iterrows()
                if row[value_col] is not None and str(row[value_col]) not in ("nan", "NaN")
            ]
            records.sort(key=lambda x: x["date"])
            return records

        except Exception as exc:
            logger.warning("OpenBB FRED fetch failed series=%s: %s", series_id, exc)
            return []

    def _rolling_z(self, values: list[float], window: int) -> float:
        """Z-score del último valor respecto a ventana rolling."""
        if len(values) < 3:
            return 0.0
        tail = values[-window:]
        try:
            mu = statistics.mean(tail)
            sd = statistics.stdev(tail) if len(tail) > 1 else 1e-9
            return round((tail[-1] - mu) / max(sd, 1e-9), 4)
        except Exception:
            return 0.0

    async def _persist(self, series_id: str, date: str, value: float) -> None:
        """Upsert en TimescaleDB tabla macro.fred_series."""
        try:
            dt = datetime.fromisoformat(date.replace("Z", "+00:00")) if "T" in date \
                 else datetime.strptime(date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            await self._db.execute(
                """
                INSERT INTO macro.fred_series (series_id, date, value, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (series_id, date)
                DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """,
                series_id, dt, value,
            )
        except Exception as exc:
            logger.debug("FRED persist error %s: %s", series_id, exc)
