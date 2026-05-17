"""
ML Feature Store Service — FastAPI entrypoint.

Responsabilidades:
  - Computar features técnicos en tiempo real vía FeatureStreaming
  - Servir feature vectors para modelo ML (baja latencia desde Redis)
  - Validar features (drift PSI, NaN, out-of-range)
  - Consumir de Kafka: los_ojos.market.normalized
  - Publicar a Kafka: los_ojos.features.vector
"""
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import structlog

from .feature_store import FeatureStore
from .feature_streaming import FeatureStreaming
from .feature_validator import FeatureValidator

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAFKA_SERVERS       = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL           = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POSTGRES_DSN        = os.getenv("POSTGRES_DSN", "postgresql://trading:trading@localhost:5432/trading_db")
SYMBOLS             = os.getenv("OHLCV_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
feature_store:    FeatureStore | None = None
feature_stream:   FeatureStreaming | None = None
feature_validator: FeatureValidator | None = None
_bg_tasks:        list[asyncio.Task] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global feature_store, feature_stream, feature_validator

    logger.info("ml-feature-store.startup")

    feature_store = FeatureStore(
        redis_url=REDIS_URL,
        postgres_dsn=POSTGRES_DSN,
        flush_interval=60,
    )
    await feature_store.connect()

    feature_stream = FeatureStreaming(
        kafka_servers=KAFKA_SERVERS,
        redis_url=REDIS_URL,
        topic_in="los_ojos.market.normalized",
        topic_out="los_ojos.features.vector",
        group_id="feature-streaming",
    )
    await feature_stream.connect()

    feature_validator = FeatureValidator()

    _bg_tasks.append(asyncio.create_task(feature_stream.run(), name="feature-streaming"))
    _bg_tasks.append(asyncio.create_task(_store_sync_loop(), name="store-sync"))

    logger.info("ml-feature-store.ready")
    yield

    for task in _bg_tasks:
        task.cancel()
    await asyncio.gather(*_bg_tasks, return_exceptions=True)
    if feature_store:
        await feature_store.close()
    if feature_stream:
        await feature_stream.close()
    logger.info("ml-feature-store.shutdown")


async def _store_sync_loop():
    """
    Every 30s: sync latest feature vectors from FeatureStreaming into FeatureStore.
    Bridges the streaming computation layer to the persistent store.
    """
    if not feature_stream or not feature_store:
        return
    while True:
        try:
            await asyncio.sleep(30)
            for symbol in SYMBOLS:
                closes = feature_stream._closes.get(symbol)
                if not closes or len(closes) < 20:
                    continue
                # Get latest feature vector computed by streaming
                fv = await feature_stream.compute_for(symbol, {
                    "price": list(closes)[-1],
                })
                if fv:
                    await feature_store.update(symbol, fv.to_dict(), fv.ts)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("store_sync.error", error=str(e))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Los Ojos — ML Feature Store",
    version="1.0.0",
    description="Real-time feature computation and serving for ML models",
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
    symbols_live = [s for s in SYMBOLS if s in (feature_stream._closes if feature_stream else {})]
    return {
        "status":       "ok",
        "service":      "ml-feature-store",
        "symbols_live": symbols_live,
        "symbols_total": len(SYMBOLS),
    }


@app.get("/features/{symbol}", tags=["features"])
async def get_features(symbol: str):
    """Return latest feature vector for a symbol (from Redis fast-path)."""
    if not feature_store:
        raise HTTPException(503, "Feature store not initialized")
    feats = await feature_store.get(symbol.upper())
    if not feats:
        raise HTTPException(404, f"No features cached for {symbol.upper()}")
    return {"symbol": symbol.upper(), "features": feats}


@app.get("/features", tags=["features"])
async def get_all_features():
    """Return latest features for all tracked symbols."""
    if not feature_store:
        raise HTTPException(503, "Feature store not initialized")
    result = {}
    for sym in SYMBOLS:
        feats = await feature_store.get(sym)
        if feats:
            result[sym] = feats
    return result


@app.get("/features/{symbol}/vector", tags=["features"])
async def get_feature_vector(
    symbol: str,
    features: str = Query(
        "rsi_14,macd_hist,ob_imbalance,funding_rate,regime_id,macro_leverage,"
        "mom_1h,mom_4h,atr_14,whale_sentiment",
        description="Comma-separated feature names in inference order",
    ),
):
    """Return an ordered float vector for direct model inference."""
    if not feature_store:
        raise HTTPException(503, "Feature store not initialized")
    feats = await feature_store.get(symbol.upper()) or {}
    feature_names = [f.strip() for f in features.split(",")]
    vector = [feats.get(f) for f in feature_names]
    has_nulls = any(v is None for v in vector)
    return {
        "symbol":       symbol.upper(),
        "feature_names": feature_names,
        "vector":       vector,
        "has_nulls":    has_nulls,
        "complete":     not has_nulls,
    }


@app.get("/features/{symbol}/validate", tags=["features"])
async def validate_features(symbol: str):
    """Validate latest feature vector for NaN, OOR, and drift."""
    if not feature_store or not feature_validator:
        raise HTTPException(503, "Store or validator not initialized")
    feats = await feature_store.get(symbol.upper())
    if not feats:
        raise HTTPException(404, f"No features for {symbol.upper()}")
    result = feature_validator.validate(feats)
    return result.__dict__ if hasattr(result, "__dict__") else result


@app.get("/features/{symbol}/history", tags=["features"])
async def get_feature_history(
    symbol: str,
    feature: str = Query("rsi_14"),
    limit: int = Query(100, ge=10, le=1000),
):
    """Return historical values of a single feature from TimescaleDB."""
    if not feature_store:
        raise HTTPException(503, "Feature store not initialized")
    rows = await feature_store.get_history(symbol.upper(), [feature], limit)
    return {"symbol": symbol.upper(), "feature": feature, "rows": rows}
