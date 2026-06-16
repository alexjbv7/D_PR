"""
FastAPI entrypoint for the execution-engine service.
=====================================================

Lifecycle (managed by FastAPI's ``lifespan``):
  1. Connect asyncpg pool → :class:`PostgresRepository`.
  2. Build :class:`AlpacaAdapter` and/or :class:`CCXTAdapter` per settings.
  3. Register them in :class:`Router`.
  4. Build :class:`RiskGate` and :class:`Reconciler`.
  5. Wire :class:`ExecutionService`.
  6. Start the Kafka consumer task (consumes ``los_ojos.signals.trading``).
  7. On shutdown: cancel consumer, stop reconciler, close adapters + pool.

REST endpoints
--------------
* ``GET /health``                          — liveness + counters
* ``GET /api/positions``                   — internal positions snapshot
* ``GET /api/orders/recent?limit=50``      — recent OrderResult log
* ``GET /api/account/{venue}``             — broker account snapshot
* ``POST /api/kill_switch/{action}``       — manual trip / reset
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from prometheus_client import make_asgi_app

from .brokers import (
    AlpacaAdapter,
    AlpacaConfig,
    BrokerError,
    CCXTAdapter,
    CCXTConfig,
)
from .brokers._alpaca.market_data import AlpacaMarketData
from .reconciler import ReconcileReport, Reconciler
from .repository import MemoryRepository, PostgresRepository, Repository
from .risk_gate import RiskConfig, RiskGate
from .routing import Router
from .service import ExecutionService
from .settings import Settings, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:
    """Holds references to every runtime singleton.  Attached to ``app.state``."""

    settings:     Settings
    repository:   Repository
    router:       Router
    risk_gate:    RiskGate
    reconciler:   Reconciler
    service:      ExecutionService
    market_data:  Optional[AlpacaMarketData]
    pg_pool:      Any                  # asyncpg pool
    kafka_consumer_task: Optional[asyncio.Task]
    kafka_consumer:      Any
    kafka_producer:      Any
    kill_switch_tripped: bool

    def __init__(self):
        self.kafka_consumer_task = None
        self.kafka_consumer      = None
        self.kafka_producer      = None
        self.kill_switch_tripped = False
        self.market_data         = None


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

async def _build_repository(settings: Settings) -> tuple[Repository, Any]:
    """
    Build the repository.  Returns ``(repo, pg_pool_or_None)``.
    Falls back to :class:`MemoryRepository` if asyncpg cannot connect — useful
    for local smoke tests without Postgres.
    """
    try:
        import asyncpg
        pool = await asyncpg.create_pool(
            settings.postgres_dsn,
            min_size=settings.postgres_min_size,
            max_size=settings.postgres_max_size,
        )
        logger.info("repository.postgres_connected")
        return PostgresRepository(pool), pool
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("repository.fallback_to_memory: %s", exc)
        return MemoryRepository(), None


def _build_market_data(settings: Settings) -> Optional[AlpacaMarketData]:
    """Build market data client if Alpaca credentials are present."""
    if not settings.alpaca_enabled or not settings.alpaca_api_key:
        return None
    try:
        md = AlpacaMarketData(
            api_key=settings.alpaca_api_key,
            api_secret=settings.alpaca_api_secret,
            feed=settings.alpaca_data_feed,
        )
        md.connect()
        logger.info("alpaca.market_data.ready feed=%s", settings.alpaca_data_feed)
        return md
    except Exception as exc:  # noqa: BLE001
        logger.warning("alpaca.market_data.init_failed: %s", exc)
        return None


def _build_router(
    settings: Settings,
    market_data: Optional[AlpacaMarketData] = None,
) -> Router:
    router = Router(default_equity="alpaca", default_crypto=settings.ccxt_exchange)
    if settings.alpaca_enabled and settings.alpaca_api_key:
        router.register(AlpacaAdapter(
            config=AlpacaConfig(
                api_key=settings.alpaca_api_key,
                api_secret=settings.alpaca_api_secret,
                paper=settings.alpaca_paper,
            ),
            market_data=market_data,
        ))
    if settings.ccxt_enabled and settings.ccxt_api_key:
        router.register(CCXTAdapter(config=CCXTConfig(
            exchange=settings.ccxt_exchange,
            api_key=settings.ccxt_api_key,
            api_secret=settings.ccxt_api_secret,
            testnet=settings.ccxt_testnet,
            market_type=settings.ccxt_market_type,
        )))
    return router


async def _connect_router(router: Router) -> None:
    for venue in router.venues():
        try:
            await router.get(venue).connect()
        except BrokerError as exc:
            logger.error("router.connect_failed venue=%s err=%s", venue, exc)


# ---------------------------------------------------------------------------
# Kafka loop
# ---------------------------------------------------------------------------

async def _consume_signals(
    state:    AppState,
    settings: Settings,
) -> None:
    """Background task: consume FinalSignalEvent → service.handle_signal."""
    try:
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    except ImportError:
        logger.error("aiokafka not installed — Kafka loop disabled")
        return

    consumer = AIOKafkaConsumer(
        settings.kafka_signal_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: v.decode("utf-8"),
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: v.encode("utf-8"),
    )
    state.kafka_consumer = consumer
    state.kafka_producer = producer

    await consumer.start()
    await producer.start()

    # Late-bind emitters now that the producer exists
    state.service.result_emitter = _make_result_emitter(producer, settings)
    if state.reconciler is not None:
        state.reconciler.on_discrepancy = _make_anomaly_emitter(producer, settings)
        state.reconciler.kill_switch_callback = _make_kill_switch_callback(state)

    logger.info("kafka.consumer_started topic=%s", settings.kafka_signal_topic)

    try:
        async for msg in consumer:
            if state.kill_switch_tripped:
                continue
            try:
                payload = json.loads(msg.value)
                # FinalSignalEvent shape from strategy-orchestrator
                await state.service.handle_signal(payload)
            except Exception as exc:                                # noqa: BLE001
                logger.exception("kafka.handle_signal_error: %s", exc)
    except asyncio.CancelledError:
        pass
    finally:
        await consumer.stop()
        await producer.stop()


def _make_result_emitter(producer: Any, settings: Settings):
    async def emit(result):
        envelope = {
            "type": "ExecutionResult",
            "data": json.loads(result.model_dump_json()),
            "ts":   datetime.now(tz=timezone.utc).isoformat(),
        }
        await producer.send_and_wait(settings.kafka_result_topic, json.dumps(envelope))
    return emit


def _make_anomaly_emitter(producer: Any, settings: Settings):
    async def emit(report: ReconcileReport):
        envelope = {
            "type": "AnomalyEvent",
            "data": {
                "source": "execution-engine.reconciler",
                "ts":     report.ts.isoformat(),
                "discrepancies": [
                    {"kind": d.kind, "venue": d.venue,
                     "symbol": d.symbol, "detail": d.detail}
                    for d in report.discrepancies
                ],
            },
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        }
        await producer.send_and_wait(settings.kafka_anomaly_topic, json.dumps(envelope))
    return emit


def _make_kill_switch_callback(state: AppState):
    async def trip(reason: str):
        state.kill_switch_tripped = True
        # Also propagate to RiskGate so REST-submitted intents are blocked,
        # not just the Kafka consumer loop (fixes P1-002).
        if state.risk_gate is not None:
            state.risk_gate.trip_kill_switch()
        logger.critical("kill_switch.tripped reason=%s", reason)
    return trip


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state = AppState()
    state.settings = settings
    app.state.app_state = state

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 1. Repository
    state.repository, state.pg_pool = await _build_repository(settings)

    # 2. Market data + Router + adapters
    state.market_data = _build_market_data(settings)
    state.router = _build_router(settings, market_data=state.market_data)
    await _connect_router(state.router)

    # 3. Risk gate
    state.risk_gate = RiskGate(
        config=RiskConfig(
            per_symbol_cap_pct=settings.risk_per_symbol_cap_pct,
            per_venue_cap_pct=settings.risk_per_venue_cap_pct,
            daily_dd_kill_pct=settings.risk_daily_dd_kill_pct,
            min_cash_buffer_pct=settings.risk_min_cash_buffer_pct,
            require_paper=settings.risk_require_paper,
        ),
        repository=state.repository,
    )

    # 4. Reconciler (callbacks attached when Kafka loop starts)
    state.reconciler = Reconciler(
        router=state.router,
        repository=state.repository,
        interval_sec=settings.reconciler_interval_sec,
        failure_threshold=settings.reconciler_failure_threshold,
    )

    # 5. Service
    state.service = ExecutionService(
        router=state.router,
        risk_gate=state.risk_gate,
        repository=state.repository,
        reconciler=state.reconciler,
        account_refresh_sec=settings.account_refresh_sec,
    )
    await state.service.start()

    # 6. Kafka loop — always start so signals are consumed even before adapters connect.
    # FIX(kafka-gate): the original guard `if state.router.venues()` silently dropped
    # all signals when API keys were absent (router empty → Kafka loop never started).
    # The consumer now always runs; handle_signal() logs a clear error per-signal
    # when no adapter is available, making the failure visible in metrics/logs.
    if not state.router.venues():
        logger.warning(
            "no broker adapters registered — signals will be consumed but rejected. "
            "Set ALPACA_API_KEY / CCXT_API_KEY in .env to enable execution."
        )
    state.kafka_consumer_task = asyncio.create_task(
        _consume_signals(state, settings), name="kafka-consumer",
    )

    logger.info("execution-engine.started port=%d", settings.port)
    yield

    # ---- shutdown ----
    if state.kafka_consumer_task is not None:
        state.kafka_consumer_task.cancel()
        try:
            await state.kafka_consumer_task
        except (asyncio.CancelledError, Exception):                # noqa: BLE001
            pass
    await state.service.stop()
    if state.market_data is not None:
        state.market_data.close()
    if state.pg_pool is not None:
        await state.pg_pool.close()
    logger.info("execution-engine.stopped")


# ---------------------------------------------------------------------------
# App + endpoints
# ---------------------------------------------------------------------------

app = FastAPI(
    title="quant_bot — Execution Engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/metrics", make_asgi_app())


def _state(app: FastAPI = Depends()) -> AppState:                    # noqa: B008
    if not hasattr(app.state, "app_state"):
        raise HTTPException(503, "service not ready")
    return app.state.app_state                                       # type: ignore[no-any-return]


@app.get("/health")
async def health():
    state: AppState = app.state.app_state
    return {
        "status":             "ok",
        "service":            "execution-engine",
        "ts":                 datetime.now(tz=timezone.utc).isoformat(),
        "venues":             state.router.venues(),
        "kill_switch":        state.kill_switch_tripped,
        "counters":           state.service.stats(),
    }


@app.get("/api/positions")
async def list_positions(venue: Optional[str] = None):
    state: AppState = app.state.app_state
    positions = await state.repository.get_open_positions(venue=venue)
    return {
        "count": len(positions),
        "positions": [json.loads(p.model_dump_json()) for p in positions],
    }


@app.get("/api/orders/recent")
async def list_recent_orders(limit: int = 50):
    state: AppState = app.state.app_state
    results = await state.repository.list_recent_results(limit=limit)
    return {
        "count": len(results),
        "orders": [json.loads(r.model_dump_json()) for r in results],
    }


@app.get("/api/account/{venue}")
async def get_account(venue: str):
    state: AppState = app.state.app_state
    try:
        adapter = state.router.get(venue)
        info = await adapter.get_account()
    except BrokerError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "venue":       info.venue,
        "account_id":  info.account_id,
        "equity":      str(info.equity),
        "cash":        str(info.cash),
        "margin_used": str(info.margin_used),
        "pnl_day":     str(info.pnl_day),
        "currency":    info.currency,
        "is_paper":    info.is_paper,
    }


@app.post("/api/kill_switch/{action}")
async def kill_switch(action: str):
    state: AppState = app.state.app_state
    if action == "trip":
        state.kill_switch_tripped = True
        logger.critical("kill_switch.manual_trip")
    elif action == "reset":
        state.kill_switch_tripped = False
        if state.reconciler is not None:
            state.reconciler._kill_switch_tripped = False
            state.reconciler._consecutive_failures = 0
        if state.risk_gate is not None:
            state.risk_gate.reset_kill_switch()
        logger.warning("kill_switch.manual_reset")
    else:
        raise HTTPException(400, f"unknown action: {action}")
    return {"kill_switch": state.kill_switch_tripped}


# ---------------------------------------------------------------------------
# Market data endpoints
# ---------------------------------------------------------------------------

def _require_market_data() -> AlpacaMarketData:
    state: AppState = app.state.app_state
    if state.market_data is None:
        raise HTTPException(
            503,
            "Market data not available.  "
            "Set ALPACA_API_KEY and ALPACA_API_SECRET in .env.",
        )
    return state.market_data


@app.get("/api/market-data/bars/{symbol:path}")
async def market_data_bars(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 200,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """
    Historical OHLCV bars.

    Query params
    ------------
    * ``symbol``    — Alpaca-format (``AAPL``, ``BTC/USD``).
    * ``timeframe`` — ``1min|5min|15min|30min|1h|4h|1d|1w|1m``.
    * ``limit``     — max bars (default 200).
    * ``start``, ``end`` — ISO-8601 UTC timestamps.
    """
    md = _require_market_data()
    start_dt = (
        datetime.fromisoformat(start.replace("Z", "+00:00"))
        if start else None
    )
    end_dt = (
        datetime.fromisoformat(end.replace("Z", "+00:00"))
        if end else None
    )
    try:
        bars = await md.get_bars(symbol, timeframe, start_dt, end_dt, limit)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Alpaca data error: {exc}")
    return {"symbol": symbol, "timeframe": timeframe, "count": len(bars), "bars": bars}


@app.get("/api/market-data/quote/{symbol:path}")
async def market_data_quote(symbol: str):
    """Latest bid/ask quote for *symbol*."""
    md = _require_market_data()
    quote = await md.get_latest_quote(symbol)
    if quote is None:
        raise HTTPException(404, f"No quote available for {symbol}")
    return quote


@app.get("/api/market-data/snapshot/{symbol:path}")
async def market_data_snapshot(symbol: str):
    """Full snapshot: latest trade + quote + minute bar + daily bar."""
    md = _require_market_data()
    snap = await md.get_snapshot(symbol)
    if snap is None:
        raise HTTPException(404, f"No snapshot available for {symbol}")
    return snap


@app.get("/api/market-data/price/{symbol:path}")
async def market_data_price(symbol: str):
    """Single last-traded price as Decimal string (for risk gate / UI)."""
    md = _require_market_data()
    price = await md.get_last_price(symbol)
    if price is None:
        raise HTTPException(404, f"No price available for {symbol}")
    return {"symbol": symbol, "price": str(price)}
