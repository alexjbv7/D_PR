"""
Whale Detector — Large on-chain transaction detection + classification.
======================================================================
Consume el feed de Crucix (o APIs públicas de blockchain) y detecta:

  1. Large transfers (> threshold USD)
  2. Exchange inflows / outflows
  3. Smart money wallet accumulation
  4. Cross-chain bridge activity
  5. DEX large swaps

Clasificación de wallets:
  - known_exchange   : Binance, Coinbase, Kraken, etc.
  - known_miner      : Mining pools
  - known_whale      : Wallets históricamente grandes
  - defi_protocol    : Uniswap, Aave, Compound
  - unknown          : Dirección sin label

Señal resultante:
  - exchange_inflow  (bearish): whale mueve BTC a exchange → posible venta
  - exchange_outflow (bullish): whale saca BTC de exchange → hodl
  - whale_to_whale   (neutral): movimiento entre wallets frías
  - defi_activity    (context): interacción con DeFi

Anti-leakage: los thresholds se calibran sobre historial, no sobre OOS.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp
from pydantic import BaseModel

from libs.shared.events import WhaleAlertEvent, SmartMoneyFlowEvent, KafkaTopics
from libs.shared.kafka_client import KafkaProducerClient
from libs.shared.redis_client import RedisCache, TTL

logger = logging.getLogger(__name__)

CRUCIX_API_KEY      = os.getenv("CRUCIX_API_KEY", "")
CRUCIX_BASE_URL     = os.getenv("CRUCIX_BASE_URL", "https://api.crucix.io/v1")
WHALE_THRESHOLD_USD = float(os.getenv("WHALE_THRESHOLD_USD", "1_000_000"))
POLL_INTERVAL_S     = int(os.getenv("WHALE_POLL_INTERVAL_S", "30"))

# Label database (simplificado — en producción usar Arkham/Nansen)
KNOWN_WALLETS: dict[str, dict] = {
    # Bitcoin exchanges (addresses ilustrativas)
    "1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ": {"label": "Binance Cold", "type": "known_exchange"},
    "3E5JtFnRCHmmEBMZiV5FpXCmfCyXGMqGDt": {"label": "Coinbase",    "type": "known_exchange"},
    # ETH exchanges
    "0x28c6c06298d514db089934071355e5743bf21d60": {"label": "Binance Hot",  "type": "known_exchange"},
    "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43": {"label": "Coinbase Pro", "type": "known_exchange"},
    # DeFi
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": {"label": "Uniswap V2 Router", "type": "defi_protocol"},
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": {"label": "Uniswap V3 Router", "type": "defi_protocol"},
}


class WhaleTransaction(BaseModel):
    tx_hash:        str
    blockchain:     str
    from_address:   str
    to_address:     str
    amount_usd:     float
    amount_native:  float
    token:          str
    timestamp:      datetime
    from_label:     Optional[str] = None
    to_label:       Optional[str] = None
    from_type:      str = "unknown"
    to_type:        str = "unknown"

    @property
    def direction(self) -> str:
        """Clasifica la dirección del flujo."""
        from_is_exch = self.from_type == "known_exchange"
        to_is_exch   = self.to_type   == "known_exchange"

        if to_is_exch and not from_is_exch:
            return "exchange_inflow"
        elif from_is_exch and not to_is_exch:
            return "exchange_outflow"
        elif self.to_type == "defi_protocol":
            return "defi_deposit"
        elif self.from_type == "defi_protocol":
            return "defi_withdrawal"
        else:
            return "wallet_to_wallet"

    @property
    def sentiment(self) -> str:
        d = self.direction
        if d == "exchange_inflow":
            return "bearish"
        elif d == "exchange_outflow":
            return "bullish"
        return "neutral"


class WhaleDetector:
    """
    Detecta y clasifica transacciones whale en tiempo real.

    Flujo:
      Crucix API → WhaleTransaction → clasificación → WhaleAlertEvent → Kafka
    """

    def __init__(
        self,
        producer: KafkaProducerClient,
        cache: RedisCache,
        blockchains: list[str] = None,
    ):
        self._producer    = producer
        self._cache       = cache
        self._blockchains = blockchains or ["bitcoin", "ethereum"]
        self._seen_txs: set[str] = set()
        self._flow_buffer: dict[str, list[float]] = {}  # symbol → últimos flujos

    async def run(self):
        """Polling loop de detección de whales."""
        logger.info("WhaleDetector started. Threshold: $%,.0f", WHALE_THRESHOLD_USD)
        while True:
            try:
                await self._poll_transactions()
                await self._compute_net_flows()
            except Exception as exc:
                logger.error("WhaleDetector error: %s", exc)
            await asyncio.sleep(POLL_INTERVAL_S)

    # ── Polling ────────────────────────────────────────────────────────

    async def _poll_transactions(self):
        """Consulta Crucix por transacciones grandes recientes."""
        headers = {"Authorization": f"Bearer {CRUCIX_API_KEY}"}

        async with aiohttp.ClientSession(headers=headers) as session:
            for blockchain in self._blockchains:
                await self._poll_blockchain(session, blockchain)

    async def _poll_blockchain(
        self, session: aiohttp.ClientSession, blockchain: str
    ):
        """Fetch transacciones whale para un blockchain."""
        url = f"{CRUCIX_BASE_URL}/large-transactions"
        params = {
            "blockchain":  blockchain,
            "min_usd":     WHALE_THRESHOLD_USD,
            "limit":       50,
            "since":       (
                datetime.now(tz=timezone.utc) - timedelta(seconds=POLL_INTERVAL_S * 2)
            ).isoformat(),
        }

        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    # Fallback a datos mock si no hay API key
                    if not CRUCIX_API_KEY:
                        return
                    logger.warning("Crucix API %d for %s", resp.status, blockchain)
                    return

                data = await resp.json()
                txs = data.get("transactions", [])

            for tx_data in txs:
                tx = self._parse_transaction(tx_data, blockchain)
                if tx and tx.tx_hash not in self._seen_txs:
                    self._seen_txs.add(tx.tx_hash)
                    await self._process_transaction(tx)

        except Exception as exc:
            logger.debug("Crucix poll %s: %s", blockchain, exc)

    def _parse_transaction(
        self, data: dict, blockchain: str
    ) -> Optional[WhaleTransaction]:
        """Parsea y enriquece una transacción con labels."""
        try:
            from_addr = data.get("from", "")
            to_addr   = data.get("to", "")

            from_info = KNOWN_WALLETS.get(from_addr.lower(), {})
            to_info   = KNOWN_WALLETS.get(to_addr.lower(), {})

            return WhaleTransaction(
                tx_hash       = data.get("hash", ""),
                blockchain    = blockchain,
                from_address  = from_addr,
                to_address    = to_addr,
                amount_usd    = float(data.get("usd_value", 0)),
                amount_native = float(data.get("amount", 0)),
                token         = data.get("token", blockchain[:3].upper()),
                timestamp     = datetime.fromisoformat(
                    data.get("timestamp", datetime.now().isoformat())
                ).replace(tzinfo=timezone.utc),
                from_label    = from_info.get("label"),
                to_label      = to_info.get("label"),
                from_type     = from_info.get("type", "unknown"),
                to_type       = to_info.get("type", "unknown"),
            )
        except Exception as exc:
            logger.debug("Transaction parse error: %s", exc)
            return None

    async def _process_transaction(self, tx: WhaleTransaction):
        """Procesa transacción: emite evento y actualiza buffers."""
        # Emitir WhaleAlertEvent
        event = WhaleAlertEvent(
            source       = "whale-detector",
            blockchain   = tx.blockchain,
            tx_hash      = tx.tx_hash,
            from_address = tx.from_address,
            to_address   = tx.to_address,
            amount_usd   = tx.amount_usd,
            amount_native= tx.amount_native,
            token        = tx.token,
            direction    = tx.direction,
            from_label   = tx.from_label,
            to_label     = tx.to_label,
        )
        await self._producer.send(KafkaTopics.WHALE_ALERT, event)

        # Actualizar buffer de flujos
        symbol = f"{tx.token}USDT"
        if symbol not in self._flow_buffer:
            self._flow_buffer[symbol] = []

        flow = tx.amount_usd if tx.direction == "exchange_outflow" else -tx.amount_usd
        self._flow_buffer[symbol].append(flow)

        logger.info(
            "WHALE %s: %s $%.0fM → %s (%s)",
            tx.blockchain,
            tx.from_label or tx.from_address[:8],
            tx.amount_usd / 1_000_000,
            tx.to_label or tx.to_address[:8],
            tx.direction.upper(),
        )

        # Cache para dashboard
        await self._cache.set(
            f"whale:latest:{tx.token}",
            {
                "direction":  tx.direction,
                "amount_usd": tx.amount_usd,
                "sentiment":  tx.sentiment,
                "ts":         tx.timestamp.isoformat(),
                "tx_hash":    tx.tx_hash,
            },
            ttl=TTL["whale"],
        )

    # ── Net flow computation ───────────────────────────────────────────

    async def _compute_net_flows(self):
        """
        Calcula el flujo neto de las últimas 24h por símbolo y emite
        SmartMoneyFlowEvent.
        """
        for symbol, flows in self._flow_buffer.items():
            if not flows:
                continue

            net_flow = sum(flows[-50:])   # últimas 50 transacciones
            token = symbol.replace("USDT", "")

            # Score de acumulación: cuánto % del flujo es positivo (outflow = bullish)
            n_positive = sum(1 for f in flows[-50:] if f > 0)
            accum_score = (n_positive / len(flows[-50:])) * 100

            signal_label = (
                "accumulation" if accum_score > 60 else
                "distribution" if accum_score < 40 else
                "neutral"
            )

            event = SmartMoneyFlowEvent(
                source                 = "whale-detector",
                blockchain             = "unknown",
                token                  = token,
                net_flow_exchange_24h  = net_flow,
                whale_accumulation_score = accum_score,
                dex_volume_24h         = 0.0,
                signal                 = signal_label,
                confidence             = abs(accum_score - 50) / 50,
            )
            await self._producer.send(KafkaTopics.SMART_MONEY, event, key=token)

            # Actualizar Redis
            await self._cache.set(
                f"onchain:flow:{token}",
                {
                    "net_flow":    net_flow,
                    "accum_score": accum_score,
                    "signal":      signal_label,
                },
                ttl=TTL["whale"],
            )
