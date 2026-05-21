"""
Context Engine Service — FastAPI entrypoint.

Responsabilidades:
  - Clasificar régimen de mercado (GMM 5 componentes)
  - Detectar anomalías (precio, volumen, spread, funding)
  - Construir y publicar MarketState consolidado
  - Consumir de Kafka: los_ojos.market.normalized
  - Publicar a Kafka: los_ojos.context.regime, los_ojos.context.anomaly,
                      los_ojos.context.state
  - Cache en Redis: market:state, context:regime_id
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
import structlog

from .anomaly_detector import AnomalyDetector
from .market_state_engine import MarketStateEngine
from .regime_classifier import RegimeClassifier
from libs.shared.kafka_client import KafkaProducerClient
from libs.shared.redis_client import RedisCache

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAFKA_SERVERS   = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SYMBOLS         = os.getenv("OHLCV_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
STATE_INTERVAL  = int(os.getenv("MARKET_STATE_INTERVAL", "30"))  # seconds

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
regime_clf:    RegimeClassifier | None = None
anomaly_det:   AnomalyDetector | None = None
state_engine:  MarketStateEngine | None = None
_producer:     KafkaProducerClient | None = None
_cache:        RedisCache | None = None
_bg_tasks:     list[asyncio.Task] = []

# In-memory latest state for REST
_latest_state:    dict = {}
_latest_anomalies: list[dict] = []
_current_regimes:  dict = {}   # symbol → regime dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    global regime_clf, anomaly_det, state_engine, _producer, _cache

    logger.info("context-engine.startup")

    # Shared infra clients
    _producer = KafkaProducerClient(bootstrap_servers=KAFKA_SERVERS)
    _cache    = RedisCache(url=REDIS_URL)
    await _producer.start()
    await _cache.connect()

    # RegimeClassifier uses DI
    regime_clf = RegimeClassifier(
        n_components=5,
        producer=_producer,
        cache=_cache,
    )

    # AnomalyDetector and MarketStateEngine use URL-based init (legacy API)
    anomaly_det = AnomalyDetector(
        kafka_servers=KAFKA_SERVERS,
        redis_url=REDIS_URL,
        price_z_thresh=4.0,
        volume_mult=5.0,
        spread_mult=4.0,
        funding_z_thresh=3.0,
        window=20,
    )
    await anomaly_det.connect()

    state_engine = MarketStateEngine(
        kafka_servers=KAFKA_SERVERS,
        redis_url=REDIS_URL,
    )
    await state_engine.connect()

    _bg_tasks.append(asyncio.create_task(_regime_loop(), name="regime-classifier"))
    _bg_tasks.append(asyncio.create_task(_anomaly_consumer_loop(), name="anomaly-consumer"))
    _bg_tasks.append(asyncio.create_task(_state_publisher_loop(), name="state-publisher"))

    logger.info("context-engine.ready", symbols=SYMBOLS)
    yield

    for task in _bg_tasks:
        task.cancel()
    await asyncio.gather(*_bg_tasks, return_exceptions=True)
    await anomaly_det.close()
    await state_engine.close()
    if _producer:
        await _producer.stop()
    if _cache:
        await _cache.close()
    logger.info("context-engine.shutdown")


# ---------------------------------------------------------------------------
# Background loops
# ---------------------------------------------------------------------------

def _compute_regime_features(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Derive GMM-ready feature vector from a raw OHLCV DataFrame.

    Input columns expected: close, (open, high, low, volume optional).
    Output columns match RegimeClassifier.GMM_FEATURES:
        vol_realized_20d, vol_realized_5d, momentum_20d, momentum_5d,
        trend_strength, macro_recession_p, vix_z.
    """
    import numpy as np
    import pandas as pd

    # Handle both OHLCV (close) and tick (price) formats
    if "close" in df.columns:
        closes = pd.to_numeric(df["close"], errors="coerce")
    elif "price" in df.columns:
        closes = pd.to_numeric(df["price"], errors="coerce")
    else:
        return pd.DataFrame()
    closes = closes.dropna()
    if len(closes) < 6:
        return pd.DataFrame()

    log_ret = np.log(closes / closes.shift(1)).dropna()

    features = pd.DataFrame(index=log_ret.index)

    # Realized volatility (annualised) — min_periods=3 allows sparse data
    features["vol_realized_20d"] = log_ret.rolling(20, min_periods=3).std() * (252 ** 0.5)
    features["vol_realized_5d"]  = log_ret.rolling(5,  min_periods=2).std() * (252 ** 0.5)

    # Momentum z-scores — min_periods allows short histories
    ret_20 = closes.pct_change(min(20, max(1, len(closes) - 1))).reindex(log_ret.index)
    ret_5  = closes.pct_change(min(5,  max(1, len(closes) - 1))).reindex(log_ret.index)

    features["momentum_20d"] = (
        (ret_20 - ret_20.rolling(60, min_periods=5).mean()) /
        (ret_20.rolling(60, min_periods=5).std() + 1e-9)
    )
    features["momentum_5d"] = (
        (ret_5 - ret_5.rolling(20, min_periods=3).mean()) /
        (ret_5.rolling(20, min_periods=3).std() + 1e-9)
    )

    # Trend strength proxy: |momentum| / vol
    features["trend_strength"] = (
        features["momentum_20d"].abs() /
        (features["vol_realized_20d"].abs() + 1e-9)
    )

    # Macro overlays: unavailable from OHLCV alone — default 0
    features["macro_recession_p"] = 0.0
    features["vix_z"]             = 0.0

    return features.dropna()


async def _regime_loop():
    """
    Consume market data from Kafka, accumulate features, periodically
    fit RegimeClassifier and classify each symbol.
    """
    global _current_regimes
    import pandas as pd
    import numpy as np

    _buffers: dict[str, list[dict]] = {sym: [] for sym in SYMBOLS}

    try:
        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            "los_ojos.market.normalized",
            bootstrap_servers=KAFKA_SERVERS,
            group_id="context-engine-regime-v2",   # new group → starts from earliest
            auto_offset_reset="earliest",           # consume historical data on first start
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        await consumer.start()
        try:
            fit_counter = 0
            async for msg in consumer:
                try:
                    data = msg.value
                    symbol = data.get("symbol", "")
                    if symbol not in _buffers:
                        _buffers[symbol] = []

                    # Accumulate OHLCV data
                    _buffers[symbol].append(data)
                    if len(_buffers[symbol]) > 200:
                        _buffers[symbol] = _buffers[symbol][-200:]

                    fit_counter += 1

                    # Refit every 50 messages if we have enough data
                    if fit_counter >= 50 and regime_clf:
                        fit_counter = 0
                        for sym, buf in _buffers.items():
                            if len(buf) < 20:
                                continue
                            try:
                                df = pd.DataFrame(buf)
                                # Derive GMM features from raw OHLCV
                                feature_df = _compute_regime_features(df)
                                if len(feature_df) < 5:
                                    continue
                                # Train the GMM on the feature history
                                regime_clf.fit(feature_df)
                                # classify(features: dict, symbol: str)
                                current_features = feature_df.iloc[-1].to_dict()
                                result = await regime_clf.classify(
                                    features=current_features, symbol=sym,
                                )
                                _current_regimes[sym] = result
                            except Exception as ce:
                                logger.warning("regime_classify.error",
                                               symbol=sym, error=str(ce))

                except Exception as e:
                    logger.error("regime_loop.msg_error", error=str(e))
        finally:
            await consumer.stop()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("regime_loop.error", error=str(e))


async def _anomaly_consumer_loop():
    """
    Consume from los_ojos.market.normalized and run AnomalyDetector.
    Keeps _latest_anomalies updated for the REST API.
    """
    global _latest_anomalies
    if not anomaly_det:
        return
    try:
        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            "los_ojos.market.normalized",
            bootstrap_servers=KAFKA_SERVERS,
            group_id="context-engine-anomaly",
            auto_offset_reset="latest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        await consumer.start()
        try:
            async for msg in consumer:
                anomalies = await anomaly_det.process(msg.value)
                if anomalies:
                    _latest_anomalies = (
                        [a.to_dict() for a in anomalies] + _latest_anomalies
                    )[:50]
        finally:
            await consumer.stop()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("anomaly_consumer.error", error=str(e))


async def _state_publisher_loop():
    """
    Every STATE_INTERVAL seconds, build and publish MarketState.
    """
    global _latest_state
    if not state_engine:
        return

    _redis = None
    try:
        import aioredis
        _redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        pass

    while True:
        try:
            await asyncio.sleep(STATE_INTERVAL)

            # Current regime (use BTC as representative)
            regime = _current_regimes.get("BTCUSDT")

            from .anomaly_detector import AnomalyEvent, Severity
            recent_anomaly_dicts = _latest_anomalies[:20]

            class _AnomalyProxy:
                def __init__(self, d):
                    self.severity = d.get("severity", "low")
                    self.anomaly_type = d.get("anomaly_type", "")

            proxies = [_AnomalyProxy(d) for d in recent_anomaly_dicts]

            macro = await _read_macro_from_redis(_redis)

            whale_sent = 0.0
            whale_conf = 0.0
            if _redis:
                try:
                    ws = await _redis.get("onchain:whale_sentiment:BTC")
                    wc = await _redis.get("onchain:whale_confidence:BTC")
                    whale_sent = float(ws or 0)
                    whale_conf = float(wc or 0)
                except Exception:
                    pass

            btc_up_prob = None
            rec_mkt_prob = None
            if _redis:
                try:
                    b = await _redis.get("polymarket:btc_up_prob")
                    r = await _redis.get("polymarket:recession_prob")
                    btc_up_prob  = float(b) if b else None
                    rec_mkt_prob = float(r) if r else None
                except Exception:
                    pass

            state = state_engine.build(
                regime=regime,
                liquidity=None,
                anomalies=proxies,
                macro=macro,
                whale_sentiment=whale_sent,
                whale_confidence=whale_conf,
                btc_up_prob=btc_up_prob,
                recession_market_prob=rec_mkt_prob,
            )
            await state_engine.publish(state)
            _latest_state = state.to_dict()

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("state_publisher.error", error=str(e))

    if _redis:
        await _redis.close()


async def _read_macro_from_redis(redis) -> object | None:
    """Read latest macro signal from Redis and return a lightweight proxy."""
    if not redis:
        return None
    try:
        raw = await redis.get("macro:signal:latest")
        if not raw:
            return None
        data = json.loads(raw)

        class _MacroProxy:
            def __init__(self, d):
                self.bias             = d.get("bias", "neutral")
                self.leverage_adj     = float(d.get("leverage_adj", 1.0))
                self.recession_prob   = float(d.get("recession_prob", 0.0))
                self.rate_environment = d.get("rate_env", "neutral")
                self.yield_curve_inversion = d.get("yield_inv", False)

        return _MacroProxy(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Los Ojos — Context Engine",
    version="1.0.0",
    description="Market regime classification, anomaly detection, consolidated market state",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/metrics", make_asgi_app())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health():
    return {
        "status": "ok",
        "service": "context-engine",
        "symbols": SYMBOLS,
        "anomalies_active": len(_latest_anomalies),
    }


@app.get("/state", tags=["context"])
async def get_market_state():
    """Return latest consolidated MarketState."""
    if not _latest_state:
        raise HTTPException(503, "Market state not yet built")
    return _latest_state


@app.get("/anomalies", tags=["context"])
async def get_anomalies(limit: int = Query(20, ge=1, le=50)):
    """Return recent anomaly events."""
    return {"anomalies": _latest_anomalies[:limit], "total": len(_latest_anomalies)}


@app.get("/regime/{symbol}", tags=["context"])
async def get_regime(symbol: str):
    if not regime_clf:
        raise HTTPException(503, "Regime classifier not initialized")
    result = _current_regimes.get(symbol.upper())
    if not result:
        raise HTTPException(404, f"No regime for {symbol} (classifier still warming up)")
    return result


@app.get("/regime", tags=["context"])
async def get_all_regimes():
    if not regime_clf:
        raise HTTPException(503, "Regime classifier not initialized")
    return _current_regimes
