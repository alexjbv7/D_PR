"""
On-Chain Analysis Service — FastAPI entrypoint.

Responsabilidades:
  - Detectar transacciones whale via Crucix API
  - Clasificar dirección de flujo (exchange inflow/outflow/wallet-to-wallet)
  - Calcular net flow y sentiment on-chain
  - Publicar WhaleAlertEvent y SmartMoneyFlowEvent a Kafka
"""
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import structlog

from .whale_detector import WhaleDetector
from libs.shared.kafka_client import KafkaProducerClient
from libs.shared.redis_client import RedisCache

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAFKA_SERVERS     = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL         = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ONCHAIN_ASSETS    = os.getenv("ONCHAIN_ASSETS", "BTC,ETH,SOL").split(",")
CHECK_INTERVAL    = int(os.getenv("WHALE_CHECK_INTERVAL", "60"))

# Map ticker symbols → blockchain names used by the detector
_BLOCKCHAIN_MAP = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
whale_detector: WhaleDetector | None = None
_producer: KafkaProducerClient | None = None
_cache:    RedisCache | None = None
_bg_tasks: list[asyncio.Task] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whale_detector, _producer, _cache

    logger.info("onchain-analysis.startup")

    _producer = KafkaProducerClient(bootstrap_servers=KAFKA_SERVERS)
    _cache    = RedisCache(url=REDIS_URL)

    await _producer.start()
    await _cache.connect()

    blockchains = [_BLOCKCHAIN_MAP.get(a, a.lower()) for a in ONCHAIN_ASSETS]

    whale_detector = WhaleDetector(
        producer=_producer,
        cache=_cache,
        blockchains=blockchains,
    )

    _bg_tasks.append(asyncio.create_task(
        whale_detector.run(), name="whale-detector"
    ))

    logger.info("onchain-analysis.ready", assets=ONCHAIN_ASSETS)
    yield

    for task in _bg_tasks:
        task.cancel()
    await asyncio.gather(*_bg_tasks, return_exceptions=True)

    if _producer:
        await _producer.stop()
    if _cache:
        await _cache.close()

    logger.info("onchain-analysis.shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Los Ojos — On-Chain Analysis",
    version="1.0.0",
    description="Whale detection, exchange flows, smart money tracking",
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
    return {"status": "ok", "service": "onchain-analysis", "assets": ONCHAIN_ASSETS}


@app.get("/whales/recent", tags=["onchain"])
async def get_recent_whales(
    asset: str = Query("BTC"),
    limit: int = Query(20, ge=1, le=100),
):
    if not whale_detector:
        raise HTTPException(503, "Whale detector not initialized")
    # Return cached whale data from Redis
    if not _cache or not _cache._client:
        raise HTTPException(503, "Cache not available")
    raw = await _cache._client.get(f"onchain:whales:recent:{asset.upper()}")
    import json
    txns = json.loads(raw) if raw else []
    return {"asset": asset.upper(), "transactions": txns[:limit]}


@app.get("/whales/net-flow/{asset}", tags=["onchain"])
async def get_net_flow(asset: str, window_hours: int = Query(24, ge=1, le=168)):
    if not _cache or not _cache._client:
        raise HTTPException(503, "Cache not available")
    import json
    raw = await _cache._client.get(f"onchain:net_flow:{asset.upper()}")
    if not raw:
        raise HTTPException(404, f"No flow data for {asset.upper()}")
    return json.loads(raw)


@app.get("/whales/sentiment", tags=["onchain"])
async def get_sentiment():
    if not _cache or not _cache._client:
        raise HTTPException(503, "Cache not available")
    result = {}
    for asset in ONCHAIN_ASSETS:
        sent = await _cache._client.get(f"onchain:whale_sentiment:{asset}")
        conf = await _cache._client.get(f"onchain:whale_confidence:{asset}")
        result[asset] = {
            "sentiment": float(sent) if sent else 0.0,
            "confidence": float(conf) if conf else 0.0,
        }
    return result
