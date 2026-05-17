"""
WalletTracker — Rastreo de wallets de smart money / whales conocidos.

Mantiene un registro local de wallets etiquetadas y permite:
  - Lookup de label por dirección (exchange, fund, miner, whale)
  - Clasificación automática por heurísticas (tamaño, patrón de tx)
  - Actualización periódica desde fuentes externas (Etherscan labels, etc.)

El WalletTracker es usado por WhaleDetector para enriquecer transacciones
con contexto semántico (e.g. "Binance hot wallet" vs "unknown").

Decisiones de diseño:
  - Cache in-memory con TTL; Postgres como store persistente.
  - Clasificación heurística es best-effort — confianza explícita (0-1).
  - No se exponen datos privados de wallets de usuarios.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class WalletLabel:
    address:     str
    label:       str
    entity_type: str     # exchange | fund | whale | miner | defi | unknown
    exchange:    Optional[str] = None
    is_cex:      bool = False
    confidence:  float = 1.0   # 0-1
    source:      str = "manual"


class WalletTracker:
    """
    In-memory wallet label store with optional Postgres persistence.

    Usage
    -----
    tracker = WalletTracker(postgres_dsn=...)
    await tracker.connect()
    label = await tracker.lookup("0xabc...")
    """

    # Seed data — well-known exchange hot/cold wallets
    KNOWN_WALLETS: dict[str, WalletLabel] = {
        # Binance
        "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": WalletLabel(
            "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",
            "Binance Hot Wallet", "exchange", "binance", is_cex=True),
        "0xd551234ae421e3bcba99a0da6d736074f22192ff": WalletLabel(
            "0xd551234ae421e3bcba99a0da6d736074f22192ff",
            "Binance Hot Wallet 2", "exchange", "binance", is_cex=True),
        "0x564286362092d8e7936f0549571a803b203aaced": WalletLabel(
            "0x564286362092d8e7936f0549571a803b203aaced",
            "Binance Cold Wallet", "exchange", "binance", is_cex=True),
        # Coinbase
        "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": WalletLabel(
            "0x71660c4005ba85c37ccec55d0c4493e66fe775d3",
            "Coinbase Hot Wallet", "exchange", "coinbase", is_cex=True),
        "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43": WalletLabel(
            "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43",
            "Coinbase Cold Wallet", "exchange", "coinbase", is_cex=True),
        # Kraken
        "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": WalletLabel(
            "0x2910543af39aba0cd09dbb2d50200b3e800a63d2",
            "Kraken Wallet", "exchange", "kraken", is_cex=True),
        # OKX
        "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": WalletLabel(
            "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b",
            "OKX Wallet", "exchange", "okx", is_cex=True),
        # Known funds
        "0x3bfc20f0b9afcace800d73d2191166ff16540258": WalletLabel(
            "0x3bfc20f0b9afcace800d73d2191166ff16540258",
            "Jump Trading", "fund", confidence=0.85),
        "0xf977814e90da44bfa03b6295a0616a897441acec": WalletLabel(
            "0xf977814e90da44bfa03b6295a0616a897441acec",
            "Binance: Fund 2", "exchange", "binance", is_cex=True),
    }

    def __init__(
        self,
        postgres_dsn: str = "",
        ttl_seconds:  int = 3600,
    ):
        self._dsn    = postgres_dsn
        self._ttl    = ttl_seconds
        self._cache: dict[str, WalletLabel] = dict(self.KNOWN_WALLETS)
        self._pg     = None

    async def connect(self) -> None:
        if self._dsn:
            try:
                import asyncpg
                self._pg = await asyncpg.create_pool(self._dsn, min_size=1, max_size=3)
                await self._load_from_db()
                logger.info("wallet_tracker.connected", cached=len(self._cache))
            except Exception as e:
                logger.warning("wallet_tracker.db_unavailable", error=str(e))
        else:
            logger.info("wallet_tracker.in_memory_only", known=len(self._cache))

    async def close(self) -> None:
        if self._pg:
            await self._pg.close()

    async def lookup(self, address: str) -> Optional[WalletLabel]:
        """Return label for address or None if unknown."""
        addr = address.lower()
        return self._cache.get(addr)

    async def label(self, address: str) -> str:
        """Return human-readable label or truncated address."""
        lbl = await self.lookup(address)
        if lbl:
            return lbl.label
        short = address[:6] + "…" + address[-4:] if len(address) > 10 else address
        return f"Unknown ({short})"

    def is_exchange(self, address: str) -> bool:
        lbl = self._cache.get(address.lower())
        return lbl is not None and lbl.is_cex

    async def add(self, label: WalletLabel) -> None:
        """Add or update a wallet label."""
        self._cache[label.address.lower()] = label
        if self._pg:
            await self._persist(label)

    async def _load_from_db(self) -> None:
        if not self._pg:
            return
        try:
            rows = await self._pg.fetch(
                "SELECT address, label, entity_type, exchange_name, is_cex "
                "FROM onchain.wallet_labels"
            )
            for row in rows:
                self._cache[row["address"].lower()] = WalletLabel(
                    address=row["address"],
                    label=row["label"],
                    entity_type=row["entity_type"],
                    exchange=row["exchange_name"],
                    is_cex=row["is_cex"],
                    source="db",
                )
            logger.info("wallet_tracker.loaded_from_db", count=len(rows))
        except Exception as e:
            logger.error("wallet_tracker.load_error", error=str(e))

    async def _persist(self, lbl: WalletLabel) -> None:
        if not self._pg:
            return
        try:
            await self._pg.execute(
                """
                INSERT INTO onchain.wallet_labels
                    (address, label, entity_type, exchange_name, is_cex)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (address) DO UPDATE
                    SET label = EXCLUDED.label,
                        entity_type = EXCLUDED.entity_type,
                        updated_at = NOW()
                """,
                lbl.address, lbl.label, lbl.entity_type, lbl.exchange, lbl.is_cex,
            )
        except Exception as e:
            logger.error("wallet_tracker.persist_error", error=str(e))
