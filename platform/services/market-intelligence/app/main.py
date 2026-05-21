"""
Market Intelligence Service — FastAPI entrypoint.

Responsabilidades:
  - REST API para OHLCV, orderbook, funding rates, news, VIX, dark pool data
  - Background tasks: polling OpenBB, streaming Binance WS
  - Publica a Kafka: MarketDataEvent, OrderbookEvent, FundingRateEvent
  - Cache en Redis (TTL corto)
"""
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import structlog

from .normalizer import DataNormalizer
from .openbb_client import OpenBBClient
from .binance.orderbook_engine import OrderbookEngine
from .binance.funding_monitor import FundingMonitor
from .polymarket.polymarket_parser import PolymarketParser
from .polymarket.probability_analyzer import ProbabilityAnalyzer
from libs.shared.kafka_client import KafkaProducerClient
from libs.shared.redis_client import RedisCache

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
KAFKA_SERVERS  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POSTGRES_DSN   = os.getenv("POSTGRES_DSN", "postgresql://trading:trading@localhost:5432/trading_db")

OHLCV_SYMBOLS  = os.getenv("OHLCV_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
OHLCV_TFS      = os.getenv("OHLCV_TIMEFRAMES", "1h,4h").split(",")
OB_SYMBOLS     = os.getenv("ORDERBOOK_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
FUNDING_SYMS   = os.getenv("FUNDING_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")

POLYMARKET_API_URL  = os.getenv("POLYMARKET_API_URL", "https://clob.polymarket.com")
POLYMARKET_INTERVAL = int(os.getenv("POLYMARKET_INTERVAL", "900"))  # 15 min

# ---------------------------------------------------------------------------
# Singleton clients (initialized in lifespan)
# ---------------------------------------------------------------------------
openbb_client:  OpenBBClient | None = None
ob_engine:      OrderbookEngine | None = None
funding_mon:    FundingMonitor | None = None
normalizer:     DataNormalizer | None = None
poly_parser:    PolymarketParser | None = None
poly_analyzer:  ProbabilityAnalyzer | None = None
_producer:      KafkaProducerClient | None = None
_cache:         RedisCache | None = None
_bg_tasks:      list[asyncio.Task] = []
_poly_signals:  list[dict] = []   # cached Polymarket signals


@asynccontextmanager
async def lifespan(app: FastAPI):
    global openbb_client, ob_engine, funding_mon, normalizer, poly_parser, poly_analyzer
    global _producer, _cache

    logger.info("market-intelligence.startup")

    # Shared infra clients
    _producer = KafkaProducerClient(bootstrap_servers=KAFKA_SERVERS)
    _cache    = RedisCache(url=REDIS_URL)
    await _producer.start()
    await _cache.connect()

    # Components using DI
    openbb_client = OpenBBClient(cache=_cache)

    ob_engine = OrderbookEngine(
        symbols=OB_SYMBOLS,
        producer=_producer,
        cache=_cache,
    )

    funding_mon = FundingMonitor(
        symbols=FUNDING_SYMS,
        producer=_producer,
        cache=_cache,
    )

    # DataNormalizer still uses URL-based init
    normalizer = DataNormalizer(
        kafka_servers=KAFKA_SERVERS,
        topic_out="los_ojos.market.normalized",
    )
    await normalizer.connect()

    poly_parser   = PolymarketParser()
    poly_analyzer = ProbabilityAnalyzer()

    # Launch background tasks
    _bg_tasks.append(asyncio.create_task(ob_engine.run(), name="orderbook-engine"))
    _bg_tasks.append(asyncio.create_task(funding_mon.run(), name="funding-monitor"))
    _bg_tasks.append(asyncio.create_task(_poll_ohlcv_loop(), name="ohlcv-poller"))
    _bg_tasks.append(asyncio.create_task(_poll_polymarket_loop(), name="polymarket-poller"))

    logger.info("market-intelligence.ready", symbols=OHLCV_SYMBOLS)

    yield

    # Shutdown
    for task in _bg_tasks:
        task.cancel()
    await asyncio.gather(*_bg_tasks, return_exceptions=True)

    if normalizer:
        await normalizer.close()
    if _producer:
        await _producer.stop()
    if _cache:
        await _cache.close()

    logger.info("market-intelligence.shutdown")


async def _poll_ohlcv_loop():
    """Poll OpenBB for OHLCV data and publish NormalizedTick to Kafka."""
    while True:
        try:
            for symbol in OHLCV_SYMBOLS:
                for tf in OHLCV_TFS:
                    candles = await openbb_client.get_ohlcv(
                        symbol, interval=tf, limit=500
                    )
                    if candles is not None and normalizer:
                        # get_ohlcv returns a DataFrame; convert last row
                        if hasattr(candles, "to_dict"):
                            rows = candles.to_dict("records")
                        elif isinstance(candles, list):
                            rows = candles
                        else:
                            rows = []
                        if rows:
                            latest = rows[-1]
                            latest["symbol"] = symbol
                            tick = normalizer.normalize_tick(latest, source="openbb")
                            await normalizer.publish(tick)
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("ohlcv_poll.error", error=str(e))
            await asyncio.sleep(30)


async def _poll_polymarket_loop():
    """Poll Polymarket every POLYMARKET_INTERVAL seconds and cache signals in Redis."""
    import aiohttp

    # Redis for caching results
    _redis = None
    try:
        import aioredis
        _redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        pass

    while True:
        try:
            if not poly_parser or not poly_analyzer:
                await asyncio.sleep(60)
                continue

            async with aiohttp.ClientSession() as session:
                url = f"{POLYMARKET_API_URL}/markets"
                params = {"active": "true", "closed": "false", "limit": 200}
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        raw = await resp.json()
                        events = poly_parser.parse_markets(raw)
                        signals = poly_analyzer.analyze(events)

                        _poly_signals.clear()
                        _poly_signals.extend([s.to_dict() for s in signals])

                        # Cache key signals in Redis for other services
                        if _redis:
                            rec_prob = poly_analyzer.get_recession_prob(events)
                            if rec_prob is not None:
                                await _redis.setex("polymarket:recession_prob", 1800, str(rec_prob))

                            btc_sig = next(
                                (s for s in signals if s.asset == "BTC" and s.direction == "bullish"),
                                None
                            )
                            if btc_sig:
                                await _redis.setex("polymarket:btc_up_prob", 1800, str(btc_sig.probability))

                        logger.info("polymarket.refreshed", signals=len(signals))

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("polymarket.poll.error", error=str(e))

        await asyncio.sleep(POLYMARKET_INTERVAL)

    if _redis:
        await _redis.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Los Ojos — Market Intelligence",
    version="1.0.0",
    description="OHLCV, orderbook, funding rates, news, VIX, dark pool data",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "service": "market-intelligence"}


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------
@app.get("/ohlcv/{symbol}", tags=["market"])
async def get_ohlcv(
    symbol: str,
    timeframe: str = Query("1h", pattern=r"^(1m|5m|15m|30m|1h|4h|1d)$"),
    limit: int = Query(200, ge=10, le=1000),
):
    if not openbb_client:
        raise HTTPException(503, "OpenBB client not initialized")
    data = await openbb_client.get_ohlcv(symbol.upper(), interval=timeframe, limit=limit)
    if hasattr(data, "to_dict"):
        data = data.to_dict("records")
    return {"symbol": symbol.upper(), "timeframe": timeframe, "data": data or []}


# ---------------------------------------------------------------------------
# Orderbook
# ---------------------------------------------------------------------------
@app.get("/orderbook/{symbol}", tags=["market"])
async def get_orderbook(symbol: str):
    if not ob_engine:
        raise HTTPException(503, "Orderbook engine not initialized")
    snap = ob_engine.latest.get(symbol.upper())
    if not snap:
        raise HTTPException(404, f"No orderbook snapshot for {symbol}")
    return snap.to_dict()


# ---------------------------------------------------------------------------
# Funding Rate
# ---------------------------------------------------------------------------
@app.get("/funding/{symbol}", tags=["market"])
async def get_funding(symbol: str):
    if not funding_mon:
        raise HTTPException(503, "Funding monitor not initialized")
    data = funding_mon.latest.get(symbol.upper())
    if not data:
        raise HTTPException(404, f"No funding data for {symbol}")
    return data


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------
@app.get("/news/{symbol}", tags=["market"])
async def get_news(symbol: str, limit: int = Query(20, ge=1, le=100)):
    if not openbb_client:
        raise HTTPException(503, "OpenBB client not initialized")
    news = await openbb_client.get_news(symbol.upper(), limit)
    return {"symbol": symbol.upper(), "news": news}


# ---------------------------------------------------------------------------
# VIX / Macro indicators
# ---------------------------------------------------------------------------
@app.get("/vix", tags=["market"])
async def get_vix():
    if not openbb_client:
        raise HTTPException(503, "OpenBB client not initialized")
    vix = await openbb_client.get_vix()
    return {"vix": vix}


@app.get("/realized-vol/{symbol}", tags=["market"])
async def get_realized_vol(symbol: str, window: int = Query(30, ge=7, le=365)):
    if not openbb_client:
        raise HTTPException(503, "OpenBB client not initialized")
    rv = await openbb_client.get_realized_vol(symbol.upper(), window)
    return {"symbol": symbol.upper(), "window": window, "realized_vol": rv}


# ---------------------------------------------------------------------------
# Symbols catalog
# ---------------------------------------------------------------------------
@app.get("/symbols", tags=["market"])
async def list_symbols():
    return {"symbols": OHLCV_SYMBOLS, "orderbook": OB_SYMBOLS, "funding": FUNDING_SYMS}


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------

@app.get("/polymarket/signals", tags=["prediction-markets"])
async def get_polymarket_signals():
    return {
        "signals": _poly_signals,
        "count":   len(_poly_signals),
    }


@app.get("/polymarket/signals/{asset}", tags=["prediction-markets"])
async def get_polymarket_signal(asset: str):
    asset_upper = asset.upper()
    sig = next((s for s in _poly_signals if s.get("asset") == asset_upper), None)
    if not sig:
        raise HTTPException(404, f"No Polymarket signal for {asset_upper}")
    return sig
