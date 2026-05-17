"""
Strategy Orchestrator Service — FastAPI entrypoint.

Responsabilidades:
  - Seleccionar estrategia activa según régimen y condiciones macro
  - Generar señales de trading combinando ML P(win) + Bayesian sizing
  - Gestionar bot configurations (paper / live) desde Postgres
  - Aplicar kill-switch y circuit breakers
  - Consumir de Kafka: FeatureVectorEvent, RegimeUpdateEvent, MacroRegimeEvent, WhaleAlertEvent
  - Publicar a Kafka: TradingSignalEvent, KillSwitchEvent
"""
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAFKA_SERVERS  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POSTGRES_DSN   = os.getenv("POSTGRES_DSN", "postgresql://trading:trading@localhost:5432/trading_db")
DEFAULT_CAPITAL = float(os.getenv("DEFAULT_CAPITAL", "10000"))
MAX_POSITIONS   = int(os.getenv("MAX_POSITIONS", "5"))
RISK_PER_TRADE  = float(os.getenv("RISK_PER_TRADE", "0.02"))
MAX_LEVERAGE    = float(os.getenv("MAX_LEVERAGE", "2.0"))
MAX_DRAWDOWN    = float(os.getenv("MAX_DRAWDOWN_PCT", "0.10"))

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class BotConfig(BaseModel):
    name:              str
    description:       Optional[str] = None
    mode:              str = Field("paper", pattern=r"^(paper|live)$")
    max_positions:     int = Field(3, ge=1, le=20)
    max_leverage:      float = Field(1.5, ge=0.1, le=10.0)
    risk_per_trade:    float = Field(0.02, ge=0.001, le=0.10)
    max_drawdown:      float = Field(0.10, ge=0.01, le=0.50)
    total_capital:     Optional[float] = None
    active_strategies: list[str] = []
    exchange:          str = "binance"


class StrategyToggle(BaseModel):
    strategy_name: str
    active: bool


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_current_config: Optional[BotConfig] = None
_kill_switch: bool = False
_active_positions: list[dict] = []
_pnl_cache: dict = {}
_bg_tasks: list[asyncio.Task] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _current_config

    logger.info("strategy-orchestrator.startup")

    # Load active config from Postgres
    _current_config = BotConfig(
        name="Default Paper Config",
        mode="paper",
        max_positions=MAX_POSITIONS,
        max_leverage=MAX_LEVERAGE,
        risk_per_trade=RISK_PER_TRADE,
        max_drawdown=MAX_DRAWDOWN,
        total_capital=DEFAULT_CAPITAL,
        active_strategies=["momentum_ml", "mean_reversion_funding"],
    )

    _bg_tasks.append(asyncio.create_task(
        _orchestrate_loop(), name="orchestrator"
    ))

    logger.info("strategy-orchestrator.ready", mode=_current_config.mode)
    yield

    for task in _bg_tasks:
        task.cancel()
    await asyncio.gather(*_bg_tasks, return_exceptions=True)
    logger.info("strategy-orchestrator.shutdown")


async def _orchestrate_loop():
    """Main signal generation loop — consumes feature vectors and emits signals."""
    try:
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        import json, uuid
        from datetime import datetime, timezone

        consumer = AIOKafkaConsumer(
            "los_ojos.features.vector",   # feature store output
            "los_ojos.context.regime",
            "los_ojos.macro.regime",
            bootstrap_servers=KAFKA_SERVERS,
            group_id="strategy-orchestrator",
            auto_offset_reset="latest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await consumer.start()
        await producer.start()
        try:
            async for msg in consumer:
                if _kill_switch:
                    continue
                if not _current_config or not _current_config.active_strategies:
                    continue

                payload = msg.value
                # Only generate signals from feature vector messages
                if "features" not in msg.topic:
                    continue

                signal = _generate_signal(payload)
                if signal:
                    await producer.send(
                        "los_ojos.signals.trading",
                        value=signal,
                    )
                    logger.info("signal.emitted",
                                symbol=signal["symbol"],
                                direction=signal["direction"],
                                p_win=signal["p_win"])
        finally:
            await consumer.stop()
            await producer.stop()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("orchestrate.error", error=str(e))


def _generate_signal(payload: dict) -> Optional[dict]:
    """
    Signal generation from FeatureVector (19 canonical features).

    FeatureVector.to_dict() produces top-level keys (not nested under "features"):
      rsi_14, macd_hist, mom_1h, mom_4h, mom_24h, atr_14, bb_width,
      vol_ratio_1h, ob_imbalance, spread_bps, vwap_deviation, funding_rate,
      oi_change_1h, regime_id, macro_leverage, sma_cross, adx_14,
      reserve_z, whale_sentiment
    """
    import uuid
    from datetime import datetime, timezone

    symbol = payload.get("symbol", "BTCUSDT")

    # Feature vector fields are top-level keys in the payload.
    # Use dict.get() with explicit default — avoid `or default` which treats 0.0 as falsy.
    def _f(key: str, default: float) -> float:
        v = payload.get(key)
        return float(v) if v is not None else default

    rsi          = _f("rsi_14", 50.0)
    macd         = _f("macd_hist", 0.0)
    mom_24h      = _f("mom_24h", 0.0)
    mom_4h       = _f("mom_4h", 0.0)
    whale_sent   = _f("whale_sentiment", 0.0)
    ob_imbal     = _f("ob_imbalance", 0.0)
    adx          = _f("adx_14", 20.0)
    sma_cross    = _f("sma_cross", 0.0)
    regime_id    = _f("regime_id", 0.0)
    macro_lev    = _f("macro_leverage", 1.0)
    funding_rate = _f("funding_rate", 0.0)
    reserve_z    = _f("reserve_z", 0.0)

    # LOB / orderflow events have a nested "features" dict — skip them
    if "features" in payload and "rsi_14" not in payload:
        return None
    # All at exact defaults → feature vector not yet populated, skip
    if rsi == 50.0 and macd == 0.0 and mom_24h == 0.0 and mom_4h == 0.0:
        return None

    # ── Composite bull/bear score from canonical features ──────────────────

    bull = 0.0
    bear = 0.0

    # RSI signal (overbought/oversold)
    if rsi > 60:
        bull += 0.30
    elif rsi > 55:
        bull += 0.10
    elif rsi < 40:
        bear += 0.30
    elif rsi < 45:
        bear += 0.10

    # MACD histogram
    if macd > 0:
        bull += 0.20
    else:
        bear += 0.20

    # 24h momentum
    if mom_24h > 0.01:
        bull += 0.25
    elif mom_24h < -0.01:
        bear += 0.25
    elif mom_24h > 0.005:
        bull += 0.10
    elif mom_24h < -0.005:
        bear += 0.10

    # 4h momentum (short-term confirmation)
    if mom_4h > 0.005:
        bull += 0.10
    elif mom_4h < -0.005:
        bear += 0.10

    # SMA cross (trend alignment)
    if sma_cross > 0.002:
        bull += 0.10
    elif sma_cross < -0.002:
        bear += 0.10

    # Whale sentiment
    bull += max(0.0, whale_sent) * 0.15
    bear += max(0.0, -whale_sent) * 0.15

    # Orderbook imbalance
    bull += max(0.0, ob_imbal) * 0.10
    bear += max(0.0, -ob_imbal) * 0.10

    # Funding rate: extreme positive → overheated longs → bearish
    if funding_rate > 0.0008:
        bear += 0.10
    elif funding_rate < -0.0003:
        bull += 0.10

    # Reserve z-score: outflows from exchanges → bullish
    if reserve_z < -1.0:
        bull += 0.10
    elif reserve_z > 1.0:
        bear += 0.10

    # Regime filter (regime_id: 0=ranging, 1=trending_up, 2=trending_down, 3=volatile)
    if regime_id == 1.0:     # trending up → boost bull
        bull *= 1.15
    elif regime_id == 2.0:   # trending down → boost bear
        bear *= 1.15
    elif regime_id == 3.0:   # volatile → halve both (avoid in chaos)
        bull *= 0.5
        bear *= 0.5

    # ADX filter: only trade when market is trending
    if adx < 15:
        bull *= 0.5
        bear *= 0.5

    # Macro leverage adjustment
    if macro_lev < 0.7:
        bull *= 0.6

    score = bull - bear

    direction = 0
    if score > 0.35:
        direction = 1
    elif score < -0.35:
        direction = -1

    if direction == 0:
        return None

    # Drawdown gate
    if _pnl_cache.get("drawdown", 0) > (_current_config.max_drawdown if _current_config else 0.10):
        logger.warning("signal.blocked.drawdown_exceeded")
        return None

    p_win = round(0.5 + min(abs(score) * 0.20, 0.20), 4)  # cap at 0.70
    regime_names = {0: "ranging", 1: "trending_up", 2: "trending_down", 3: "volatile"}

    return {
        "event_id":      str(uuid.uuid4()),
        "event_type":    "TradingSignalEvent",
        "ts":            datetime.now(timezone.utc).isoformat(),
        "symbol":        symbol,
        "strategy":      "regime_adaptive",
        "direction":     direction,
        "p_win":         p_win,
        "confidence":    round(abs(score), 4),
        "regime":        regime_names.get(int(regime_id), "unknown"),
        "position_size": _current_config.risk_per_trade if _current_config else 0.02,
        "source_service": "strategy-orchestrator",
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Los Ojos — Strategy Orchestrator",
    version="1.0.0",
    description="Bot configuration, strategy selection, signal generation, kill-switch",
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
    return {
        "status": "ok",
        "service": "strategy-orchestrator",
        "mode": _current_config.mode if _current_config else "uninitialized",
        "kill_switch": _kill_switch,
    }


# ---------------------------------------------------------------------------
# Bot Config
# ---------------------------------------------------------------------------
@app.get("/config", tags=["bot"])
async def get_config():
    if not _current_config:
        raise HTTPException(404, "No active configuration")
    return _current_config.model_dump()


@app.put("/config", tags=["bot"])
async def update_config(config: BotConfig):
    global _current_config
    _current_config = config
    logger.info("config.updated", name=config.name, mode=config.mode)
    return {"status": "updated", "config": config.model_dump()}


# ---------------------------------------------------------------------------
# Kill Switch
# ---------------------------------------------------------------------------
@app.post("/kill-switch/{state}", tags=["bot"])
async def toggle_kill_switch(state: str):
    global _kill_switch
    if state not in ("on", "off"):
        raise HTTPException(400, "State must be 'on' or 'off'")
    _kill_switch = (state == "on")
    logger.warning("kill_switch.toggled", state=state)
    return {"kill_switch": _kill_switch, "message": f"Kill switch {'ACTIVATED' if _kill_switch else 'deactivated'}"}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
@app.get("/strategies", tags=["bot"])
async def list_strategies():
    strategies = [
        {
            "name": "momentum_ml",
            "display_name": "ML Momentum",
            "type": "ml",
            "timeframe": "1h",
            "is_active": "momentum_ml" in (_current_config.active_strategies if _current_config else []),
        },
        {
            "name": "mean_reversion_funding",
            "display_name": "Funding Rate Mean Reversion",
            "type": "mean_reversion",
            "timeframe": "4h",
            "is_active": "mean_reversion_funding" in (_current_config.active_strategies if _current_config else []),
        },
        {
            "name": "regime_adaptive",
            "display_name": "Regime Adaptive",
            "type": "ml",
            "timeframe": "1h",
            "is_active": "regime_adaptive" in (_current_config.active_strategies if _current_config else []),
        },
        {
            "name": "whale_follow",
            "display_name": "Whale Smart Money",
            "type": "trend",
            "timeframe": "4h",
            "is_active": "whale_follow" in (_current_config.active_strategies if _current_config else []),
        },
    ]
    return {"strategies": strategies}


@app.post("/strategies/toggle", tags=["bot"])
async def toggle_strategy(body: StrategyToggle):
    global _current_config
    if not _current_config:
        raise HTTPException(404, "No active configuration")

    active = list(_current_config.active_strategies)
    if body.active and body.strategy_name not in active:
        active.append(body.strategy_name)
    elif not body.active and body.strategy_name in active:
        active.remove(body.strategy_name)

    _current_config = _current_config.model_copy(update={"active_strategies": active})
    return {"status": "ok", "active_strategies": active}


# ---------------------------------------------------------------------------
# Positions & PnL
# ---------------------------------------------------------------------------
@app.get("/positions", tags=["bot"])
async def get_positions():
    return {"positions": _active_positions, "count": len(_active_positions)}


@app.get("/pnl", tags=["bot"])
async def get_pnl():
    return _pnl_cache or {
        "total_pnl": 0.0,
        "daily_pnl": 0.0,
        "drawdown": 0.0,
        "win_rate_7d": 0.0,
        "portfolio_value": DEFAULT_CAPITAL,
    }
