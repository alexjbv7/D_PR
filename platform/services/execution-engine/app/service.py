"""
ExecutionService — the orchestration layer.
============================================

Wires together :class:`Router`, :class:`RiskGate`, :class:`Repository`, and
:class:`Reconciler`, and exposes a single :meth:`handle_signal` entry-point
for the Kafka consumer (and any future REST submission endpoint).

This class is FastAPI-agnostic and Kafka-agnostic so it can be unit-tested
without spinning up either.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Awaitable, Callable, Optional

from quant_shared.schemas.orders import OrderIntent, OrderResult

from .brokers.base import AccountInfo, BrokerError
from .reconciler import ReconcileReport, Reconciler
from .repository import Repository, RiskDecision
from .risk_gate import RiskGate
from .routing import Router
from .signal_translator import translate_signal

logger = logging.getLogger(__name__)


ResultEmitter   = Callable[[OrderResult], Awaitable[None]]
AnomalyEmitter  = Callable[[ReconcileReport], Awaitable[None]]


class ExecutionService:
    """
    Coordinates the full intent-to-fill pipeline.

    Parameters
    ----------
    router : Router
    risk_gate : RiskGate
    repository : Repository
    reconciler : Reconciler, optional
        If provided, ``start()`` will start its background loop.
    result_emitter : callable, optional
        Awaitable called with every :class:`OrderResult` (used by the Kafka
        producer to publish ``los_ojos.execution.result``).
    account_refresh_sec : int
        Seconds between automatic account-snapshot refreshes.
    """

    def __init__(
        self,
        router:         Router,
        risk_gate:      RiskGate,
        repository:     Repository,
        reconciler:     Optional[Reconciler] = None,
        result_emitter: Optional[ResultEmitter] = None,
        account_refresh_sec: int = 30,
    ):
        self.router         = router
        self.risk_gate      = risk_gate
        self.repo           = repository
        self.reconciler     = reconciler
        self.result_emitter = result_emitter
        self.account_refresh_sec = account_refresh_sec

        # Per-venue cached account; the kafka loop refreshes opportunistically.
        self._account_cache: dict[str, tuple[AccountInfo, datetime]] = {}

        # Statistics for /health
        self._counters = {
            "signals_seen":  0,
            "intents_built": 0,
            "approved":      0,
            "rejected":      0,
            "submitted":     0,
            "submit_errors": 0,
        }

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        if self.reconciler is not None:
            self.reconciler.start()

    async def stop(self) -> None:
        if self.reconciler is not None:
            await self.reconciler.stop()
        await self.router.close_all()

    # -----------------------------------------------------------------------
    # Public entry-point — called once per FinalSignalEvent
    # -----------------------------------------------------------------------

    async def handle_signal(self, signal: dict) -> Optional[OrderResult]:
        """
        Process a FinalSignalEvent end-to-end.

        Steps:
          1. Resolve venue (signal or default).
          2. Get account snapshot (cached).
          3. Get current price (best-effort).
          4. Translate signal → :class:`OrderIntent`.
          5. Run risk-gate.
          6. Submit via router.
          7. Persist + emit result.

        Returns ``None`` at any step that legitimately stops execution
        (flat signal, missing price, risk rejection, etc.).
        """
        self._counters["signals_seen"] += 1

        venue = signal.get("venue") or self._default_venue_for(signal["symbol"])

        try:
            account = await self._account_for(venue)
        except BrokerError as exc:
            logger.error("service.account_unreachable venue=%s err=%s", venue, exc)
            return None

        last_price = await self._last_price_for(venue, signal["symbol"])

        intent = translate_signal(
            signal,
            equity=account.equity,
            current_price=last_price,
            default_venue=venue,
        )
        if intent is None:
            return None
        self._counters["intents_built"] += 1

        decision = await self.risk_gate.evaluate(intent, account)
        if not decision.approved:
            self._counters["rejected"] += 1
            return None
        self._counters["approved"] += 1

        try:
            result = await self.router.submit(intent)
        except BrokerError as exc:
            self._counters["submit_errors"] += 1
            logger.error(
                "service.submit_failed intent=%s err=%s",
                intent.intent_id[:8], exc,
            )
            return None

        self._counters["submitted"] += 1
        await self.repo.save_result(result)

        for fill in result.fills:
            tagged = fill.model_copy(
                update={"account_id": account.account_id, "venue": venue},
            )
            await self.repo.save_fill(tagged)

        if self.result_emitter is not None:
            try:
                await self.result_emitter(result)
            except Exception as exc:                                # noqa: BLE001
                logger.exception("service.result_emit_error: %s", exc)
        return result

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _default_venue_for(self, symbol: str) -> str:
        from .routing import is_equity  # local import to avoid cycle
        return self.router.default_equity if is_equity(symbol) else self.router.default_crypto

    async def _account_for(self, venue: str) -> AccountInfo:
        now = datetime.now(tz=timezone.utc)
        cached = self._account_cache.get(venue)
        if cached and (now - cached[1]).total_seconds() < self.account_refresh_sec:
            return cached[0]

        adapter = self.router.get(venue)
        account = await adapter.get_account()
        self._account_cache[venue] = (account, now)
        return account

    async def _last_price_for(self, venue: str, symbol: str) -> Optional[Decimal]:
        try:
            adapter = self.router.get(venue)
        except BrokerError:
            return None
        try:
            return await adapter.get_last_price(symbol)
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("service.last_price_unavailable %s/%s: %s", venue, symbol, exc)
            return None

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def stats(self) -> dict:
        return dict(self._counters)
