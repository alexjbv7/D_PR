"""
Realtime Signal Service — FastAPI + WebSocket + Kafka consumer.
==============================================================
Arquitectura:
  Kafka (signals.final, whale alerts, macro events)
    → KafkaConsumer
      → RedisPubSub publish
        → WebSocket broadcast to all connected clients

Endpoints:
  GET  /health
  WS   /ws                — stream completo de eventos del bot
  WS   /ws/signals        — solo señales de trading
  WS   /ws/whale          — solo whale alerts
  WS   /ws/macro          — solo macro events
  GET  /api/signals/recent — últimas N señales (REST)
  GET  /api/positions      — posiciones actuales
  GET  /api/regime/{symbol} — régimen actual

El servicio no tiene estado propio: lee de Redis + emite via WS.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware

from libs.shared.events import (
    FinalSignalEvent, WhaleAlertEvent, MacroDataEvent,
    RegimeUpdateEvent, AnomalyEvent, KafkaTopics,
)
from libs.shared.kafka_client import KafkaConsumerClient
from libs.shared.redis_client import RedisCache, RedisPubSub, CHANNELS

logger = logging.getLogger(__name__)

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# ===========================================================================
# WebSocket Connection Manager
# ===========================================================================

class ConnectionManager:
    """
    Gestiona todas las conexiones WebSocket activas.

    Soporta múltiples canales (rooms):
      - "all"     : todo
      - "signals" : solo señales
      - "whale"   : solo on-chain
      - "macro"   : solo macro
      - "system"  : alerts y kills
    """

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {
            "all":     [],
            "signals": [],
            "whale":   [],
            "macro":   [],
            "system":  [],
        }

    async def connect(self, ws: WebSocket, channel: str = "all"):
        await ws.accept()
        if channel not in self._connections:
            channel = "all"
        self._connections[channel].append(ws)
        logger.info("WS connect channel=%s total=%d", channel,
                    self._total_connections())

    def disconnect(self, ws: WebSocket, channel: str = "all"):
        if channel in self._connections:
            self._connections[channel] = [
                c for c in self._connections[channel] if c != ws
            ]
        logger.debug("WS disconnect channel=%s total=%d", channel,
                     self._total_connections())

    async def broadcast(self, message: str, channel: str = "all"):
        """Envía mensaje a todos los clientes del canal."""
        dead = []
        targets = (
            self._connections.get(channel, []) +
            (self._connections["all"] if channel != "all" else [])
        )
        # Dedup
        targets = list({id(c): c for c in targets}.values())

        for ws in targets:
            try:
                await ws.send_text(message)
            except (WebSocketDisconnect, RuntimeError):
                dead.append((ws, channel))

        # Cleanup dead connections
        for ws, ch in dead:
            self.disconnect(ws, ch)

    def _total_connections(self) -> int:
        return sum(len(v) for v in self._connections.values())

    def stats(self) -> dict:
        return {ch: len(conns) for ch, conns in self._connections.items()}


manager = ConnectionManager()
cache   = RedisCache(REDIS_URL)
pubsub  = RedisPubSub(REDIS_URL)


# ===========================================================================
# Kafka → Redis PubSub bridge
# ===========================================================================

async def kafka_to_pubsub():
    """
    Consume topics Kafka relevantes y los republica en Redis PubSub.
    Los WebSocket handlers luego leen de Redis PubSub.
    """
    topics = [
        KafkaTopics.SIGNAL_FINAL,
        KafkaTopics.WHALE_ALERT,
        KafkaTopics.MACRO_DATA,
        KafkaTopics.REGIME_UPDATE,
        KafkaTopics.ANOMALY,
        KafkaTopics.RECESSION_ALERT,
    ]

    channel_map = {
        KafkaTopics.SIGNAL_FINAL:   CHANNELS["signals"],
        KafkaTopics.WHALE_ALERT:    CHANNELS["whale"],
        KafkaTopics.MACRO_DATA:     CHANNELS["macro"],
        KafkaTopics.MACRO_REGIME:   CHANNELS["macro"],
        KafkaTopics.REGIME_UPDATE:  CHANNELS["regime"],
        KafkaTopics.ANOMALY:        CHANNELS["anomaly"],
        KafkaTopics.RECESSION_ALERT: CHANNELS["macro"],
    }

    async with KafkaConsumerClient(
        topics=topics,
        group_id="realtime-signal-ws",
        auto_offset_reset="latest",
    ) as consumer:
        async for event in consumer.consume():
            try:
                topic = event.source  # aproximado
                redis_channel = CHANNELS["signals"]  # default

                # Buscar el canal correcto por tipo de evento
                event_class = type(event).__name__
                if "Whale" in event_class or "SmartMoney" in event_class:
                    redis_channel = CHANNELS["whale"]
                elif "Macro" in event_class or "Recession" in event_class:
                    redis_channel = CHANNELS["macro"]
                elif "Regime" in event_class:
                    redis_channel = CHANNELS["regime"]
                elif "Anomaly" in event_class or "KillSwitch" in event_class:
                    redis_channel = CHANNELS["anomaly"]

                payload = json.dumps({
                    "type":    type(event).__name__,
                    "channel": redis_channel,
                    "data":    json.loads(event.model_dump_json()),
                    "ts":      datetime.now(tz=timezone.utc).isoformat(),
                })
                await pubsub.publish(redis_channel, payload)

            except Exception as exc:
                logger.error("Kafka→PubSub bridge error: %s", exc)


async def pubsub_to_websocket():
    """
    Consume Redis PubSub y hace broadcast a los WebSocket clients.
    Separa por canal.
    """
    channels_to_ws = {
        CHANNELS["signals"]: "signals",
        CHANNELS["whale"]:   "whale",
        CHANNELS["macro"]:   "macro",
        CHANNELS["regime"]:  "all",
        CHANNELS["anomaly"]: "system",
    }
    all_channels = list(channels_to_ws.keys())

    async with RedisPubSub(REDIS_URL) as ps:
        async for raw_message in ps.subscribe(*all_channels):
            try:
                message = json.loads(raw_message)
                ws_channel = channels_to_ws.get(
                    message.get("channel", ""), "all"
                )
                await manager.broadcast(raw_message, ws_channel)
            except Exception as exc:
                logger.debug("PubSub→WS error: %s", exc)


# ===========================================================================
# App lifecycle
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    await cache.connect()
    # Background tasks
    asyncio.create_task(kafka_to_pubsub())
    asyncio.create_task(pubsub_to_websocket())
    logger.info("Realtime Signal Service started")
    yield
    await cache.close()


app = FastAPI(
    title="Los Ojos — Realtime Signal Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# WebSocket endpoints
# ===========================================================================

@app.websocket("/ws")
async def ws_all(websocket: WebSocket):
    """Stream completo de todos los eventos."""
    await manager.connect(websocket, "all")
    try:
        while True:
            # Mantener alive; el cliente puede enviar ping
            data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
            if data == "ping":
                await websocket.send_text("pong")
    except (WebSocketDisconnect, asyncio.TimeoutError):
        manager.disconnect(websocket, "all")


@app.websocket("/ws/signals")
async def ws_signals(websocket: WebSocket):
    """Stream solo de señales de trading."""
    await manager.connect(websocket, "signals")
    try:
        while True:
            await asyncio.wait_for(websocket.receive_text(), timeout=30)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        manager.disconnect(websocket, "signals")


@app.websocket("/ws/whale")
async def ws_whale(websocket: WebSocket):
    """Stream de whale alerts on-chain."""
    await manager.connect(websocket, "whale")
    try:
        while True:
            await asyncio.wait_for(websocket.receive_text(), timeout=30)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        manager.disconnect(websocket, "whale")


@app.websocket("/ws/macro")
async def ws_macro(websocket: WebSocket):
    """Stream de eventos macro."""
    await manager.connect(websocket, "macro")
    try:
        while True:
            await asyncio.wait_for(websocket.receive_text(), timeout=30)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        manager.disconnect(websocket, "macro")


# ===========================================================================
# REST endpoints
# ===========================================================================

@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "service": "realtime-signal",
        "ws_connections": manager.stats(),
        "ts":      datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/api/signals/recent")
async def get_recent_signals(limit: int = 20):
    """Últimas N señales emitidas (desde Redis sorted set)."""
    raw = await cache.zrange_ts("signals:history", start="-inf", end="+inf")
    signals = [json.loads(r) for r in raw[-limit:]]
    return {"signals": signals, "count": len(signals)}


@app.get("/api/regime/{symbol}")
async def get_regime(symbol: str):
    """Régimen de mercado actual para un símbolo."""
    regime = await cache.get(f"regime:{symbol.upper()}")
    if not regime:
        return {"symbol": symbol, "regime": "unknown", "ts": None}
    return {"symbol": symbol, **regime}


@app.get("/api/positions")
async def get_positions():
    """Posiciones abiertas actuales."""
    positions = await cache.get("positions:current") or []
    return {"positions": positions}


@app.get("/api/whale/latest/{token}")
async def get_whale_latest(token: str):
    """Última alerta whale para un token."""
    data = await cache.get(f"whale:latest:{token.upper()}")
    return {"token": token, "data": data}


@app.get("/api/macro/snapshot")
async def get_macro_snapshot():
    """Snapshot macro actual (todas las series FRED)."""
    recession = await cache.get("macro:recession_signal") or {}
    return {"recession": recession}


@app.get("/api/stats")
async def get_stats():
    """Estadísticas del servicio."""
    return {
        "ws_connections": manager.stats(),
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }
