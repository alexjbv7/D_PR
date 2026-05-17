"""
Macroeconomic Service — FastAPI entrypoint.

Responsabilidades:
  - Polling FRED API para series macroeconómicas
  - Detección de recesión (Sahm Rule + Yield Curve + Leading Indicators)
  - Clasificación de régimen macro (expansion/slowdown/recession/recovery)
  - Publica a Kafka: MacroRegimeEvent, RecessionAlertEvent, MacroIndicatorEvent
"""
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import structlog

from .fred_collector import FredCollector
from .recession_detector import RecessionDetector
from libs.shared.kafka_client import KafkaProducerClient
from libs.shared.redis_client import RedisCache
from libs.shared.db import PostgresPool

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL     = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POSTGRES_DSN  = os.getenv("POSTGRES_DSN", "postgresql://trading:trading@localhost:5432/trading_db")
FRED_API_KEY  = os.getenv("FRED_API_KEY", "")
POLL_INTERVAL = int(os.getenv("FRED_POLL_INTERVAL", "3600"))

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
fred_collector:  FredCollector | None = None
recession_det:   RecessionDetector | None = None
_current_regime: dict | None = None
_bg_tasks: list[asyncio.Task] = []

# Shared infra clients (kept for shutdown)
_producer: KafkaProducerClient | None = None
_cache:    RedisCache | None = None
_db:       PostgresPool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global fred_collector, recession_det, _producer, _cache, _db

    logger.info("macroeconomic.startup")

    _producer = KafkaProducerClient(bootstrap_servers=KAFKA_SERVERS)
    _cache    = RedisCache(url=REDIS_URL)
    _db       = PostgresPool(dsn=POSTGRES_DSN)

    await _producer.start()
    await _cache.connect()
    await _db.connect()

    fred_collector = FredCollector(
        producer=_producer,
        cache=_cache,
        db=_db,
        api_key=FRED_API_KEY,
    )
    await fred_collector.connect()

    recession_det = RecessionDetector(
        producer=_producer,
        cache=_cache,
    )

    _bg_tasks.append(asyncio.create_task(_poll_fred_loop(), name="fred-poller"))
    _bg_tasks.append(asyncio.create_task(_detect_regime_loop(), name="regime-detector"))

    logger.info("macroeconomic.ready")
    yield

    for task in _bg_tasks:
        task.cancel()
    await asyncio.gather(*_bg_tasks, return_exceptions=True)

    if fred_collector:
        await fred_collector.close()
    if _producer:
        await _producer.stop()
    if _cache:
        await _cache.close()
    if _db:
        await _db.close()

    logger.info("macroeconomic.shutdown")


async def _poll_fred_loop():
    """Fetch all FRED series at configured interval."""
    global _current_regime
    while True:
        try:
            logger.info("fred.poll.start")
            indicators = await fred_collector.fetch_all()
            if indicators and recession_det:
                result = await recession_det.evaluate(indicators)
                _current_regime = result
                logger.info("macro.regime",
                            regime=result.get("regime"),
                            prob=result.get("recession_probability"))
            await asyncio.sleep(POLL_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("fred.poll.error", error=str(e))
            await asyncio.sleep(300)


async def _detect_regime_loop():
    """Re-run recession detection every 10 min using cached data."""
    global _current_regime
    await asyncio.sleep(120)  # Give fred_collector time to fetch initial data
    while True:
        try:
            if fred_collector and recession_det:
                indicators = fred_collector.get_cached()
                if indicators:
                    result = await recession_det.evaluate(indicators)
                    _current_regime = result
            await asyncio.sleep(600)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("regime.detect.error", error=str(e))
            await asyncio.sleep(120)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Los Ojos — Macroeconomic",
    version="1.0.0",
    description="FRED macro indicators, recession detection, rate environment",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "service": "macroeconomic"}


@app.get("/regime", tags=["macro"])
async def get_regime():
    if not recession_det:
        raise HTTPException(503, "Recession detector not initialized")
    if not _current_regime:
        raise HTTPException(404, "No macro regime computed yet")
    return _current_regime


@app.get("/indicators", tags=["macro"])
async def get_indicators():
    if not fred_collector:
        raise HTTPException(503, "FRED collector not initialized")
    cached = fred_collector.get_cached()
    if not cached:
        raise HTTPException(404, "No indicators cached yet")
    return cached


@app.get("/indicators/{series_id}", tags=["macro"])
async def get_series(series_id: str, limit: int = 100):
    if not fred_collector:
        raise HTTPException(503, "FRED collector not initialized")
    data = await fred_collector.get_series(series_id.upper(), limit)
    if not data:
        raise HTTPException(404, f"Series {series_id} not found")
    return {"series_id": series_id.upper(), "data": data}
