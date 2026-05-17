"""
Reconciler — periodic position reconciliation.
==============================================

Compares the execution-engine's internal position state (from
:class:`Repository`) with the brokers' reported positions (via
:class:`Router`).  On discrepancy:

* logs structured warnings
* invokes an optional ``on_discrepancy`` callback (for emitting
  ``AnomalyEvent`` to Kafka and / or paging)
* after ``failure_threshold`` consecutive cycles with discrepancies, trips
  an optional kill switch via the ``kill_switch_callback``

The reconciler intentionally does NOT auto-correct state — autonomous
correction of position records would mask real bugs.  Discrepancies require
human review.

Design
------
* Single background asyncio task started via :meth:`start`; cancellable.
* :meth:`reconcile_once` is the unit of work and is independently testable.
* Tolerant to broker errors: a failure in one venue does not stop checking
  the others (handled by :meth:`Router.get_positions_all`).

References
----------
* CLAUDE.md §11.4 — Reconciliation loop runs every 60 s
* CLAUDE.md §12.7 — Kill switch triggers include broker/internal mismatch
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Awaitable, Callable, Optional

from quant_shared.schemas.orders import OrderSide, Position

from .repository import Repository
from .routing import Router

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discrepancy report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Discrepancy:
    """One mismatch between internal and broker state."""
    kind:     str      # "PHANTOM" | "MISSING" | "QTY_MISMATCH" | "SIDE_MISMATCH"
    venue:    str
    symbol:   str
    detail:   str

    def __str__(self) -> str:           # pragma: no cover (cosmetic)
        return f"[{self.kind} {self.venue}:{self.symbol}] {self.detail}"


@dataclass
class ReconcileReport:
    """Result of one reconciliation cycle."""
    ts:             datetime
    discrepancies:  list[Discrepancy] = field(default_factory=list)
    broker_count:   int = 0
    internal_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.discrepancies


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

DiscrepancyCallback = Callable[[ReconcileReport], Awaitable[None]]
KillSwitchCallback  = Callable[[str], Awaitable[None]]


class Reconciler:
    """
    Periodic position reconciler.

    Parameters
    ----------
    router : Router
        Provides the aggregated broker view via :meth:`Router.get_positions_all`.
    repository : Repository
        Provides the execution-engine's internal view via
        :meth:`Repository.get_open_positions`.
    interval_sec : int
        Seconds between cycles (default 60).
    failure_threshold : int
        Consecutive cycles with discrepancies before the kill switch trips.
    qty_tolerance : Decimal
        Absolute qty difference below which we consider two positions equal.
        Useful when broker reports float-rounded balances vs internal Decimal.
    on_discrepancy : callable, optional
        Awaitable invoked with the :class:`ReconcileReport` whenever the
        report has discrepancies.  Receives the full report.
    kill_switch_callback : callable, optional
        Awaitable invoked with a reason string the first time
        ``failure_threshold`` is crossed.
    """

    def __init__(
        self,
        router: Router,
        repository: Repository,
        *,
        interval_sec:           int            = 60,
        failure_threshold:      int            = 3,
        qty_tolerance:          Decimal        = Decimal("1e-9"),
        on_discrepancy:         Optional[DiscrepancyCallback] = None,
        kill_switch_callback:   Optional[KillSwitchCallback]  = None,
    ):
        if interval_sec <= 0:
            raise ValueError("interval_sec must be > 0")
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be > 0")

        self.router               = router
        self.repo                 = repository
        self.interval_sec         = interval_sec
        self.failure_threshold    = failure_threshold
        self.qty_tolerance        = qty_tolerance
        self.on_discrepancy       = on_discrepancy
        self.kill_switch_callback = kill_switch_callback

        self._task:    Optional[asyncio.Task] = None
        self._consecutive_failures = 0
        self._kill_switch_tripped  = False

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background reconciliation task (idempotent)."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(
            self._run_forever(), name="execution-engine.reconciler"
        )
        logger.info("reconciler.started interval=%ds", self.interval_sec)

    async def stop(self) -> None:
        """Cancel and wait for the background task."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):           # noqa: BLE001
            pass
        self._task = None
        logger.info("reconciler.stopped")

    async def reconcile_once(self) -> ReconcileReport:
        """
        Execute one reconciliation cycle.  Pure function, no side effects on
        the consecutive-failures counter — call :meth:`_handle_report` for
        the full effect.
        """
        try:
            broker_positions = await self.router.get_positions_all()
        except Exception as exc:                              # noqa: BLE001
            logger.error("reconciler.broker_query_failed error=%s", exc)
            broker_positions = []

        internal_positions = await self.repo.get_open_positions()

        report = ReconcileReport(
            ts=datetime.now(tz=timezone.utc),
            broker_count=len(broker_positions),
            internal_count=len(internal_positions),
            discrepancies=self._diff(broker_positions, internal_positions),
        )
        return report

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    async def _run_forever(self) -> None:
        while True:
            try:
                report = await self.reconcile_once()
                await self._handle_report(report)
            except asyncio.CancelledError:
                raise
            except Exception as exc:                         # noqa: BLE001
                logger.exception("reconciler.cycle_error: %s", exc)
            await asyncio.sleep(self.interval_sec)

    async def _handle_report(self, report: ReconcileReport) -> None:
        if report.ok:
            if self._consecutive_failures:
                logger.info(
                    "reconciler.recovered after=%d consecutive failures",
                    self._consecutive_failures,
                )
            self._consecutive_failures = 0
            return

        self._consecutive_failures += 1
        logger.warning(
            "reconciler.discrepancies count=%d streak=%d",
            len(report.discrepancies), self._consecutive_failures,
        )
        for d in report.discrepancies:
            logger.warning("  %s", d)

        if self.on_discrepancy is not None:
            try:
                await self.on_discrepancy(report)
            except Exception as exc:                         # noqa: BLE001
                logger.exception("reconciler.callback_error: %s", exc)

        if (
            self._consecutive_failures >= self.failure_threshold
            and not self._kill_switch_tripped
            and self.kill_switch_callback is not None
        ):
            self._kill_switch_tripped = True
            reason = (
                f"reconciler: {self._consecutive_failures} consecutive cycles "
                f"with discrepancies (latest count={len(report.discrepancies)})"
            )
            try:
                await self.kill_switch_callback(reason)
                logger.critical("reconciler.kill_switch_tripped reason=%s", reason)
            except Exception as exc:                         # noqa: BLE001
                logger.exception("reconciler.kill_switch_callback_error: %s", exc)

    # -----------------------------------------------------------------------
    # Diff
    # -----------------------------------------------------------------------

    def _diff(
        self,
        broker_positions:   list[Position],
        internal_positions: list[Position],
    ) -> list[Discrepancy]:
        """Identify every mismatch, grouped by ``(venue, symbol)``."""
        broker_by   = {(p.venue, p.symbol): p for p in broker_positions}
        internal_by = {(p.venue, p.symbol): p for p in internal_positions}

        keys = set(broker_by) | set(internal_by)
        out: list[Discrepancy] = []

        for key in sorted(keys):
            venue, symbol = key
            b = broker_by.get(key)
            i = internal_by.get(key)

            if b is None:
                out.append(Discrepancy(
                    kind="MISSING", venue=venue, symbol=symbol,
                    detail=f"internal has qty={i.qty} side={i.side.value}, "
                           f"broker has no position",
                ))
                continue
            if i is None:
                out.append(Discrepancy(
                    kind="PHANTOM", venue=venue, symbol=symbol,
                    detail=f"broker has qty={b.qty} side={b.side.value}, "
                           f"internal has no position",
                ))
                continue
            if b.side != i.side:
                out.append(Discrepancy(
                    kind="SIDE_MISMATCH", venue=venue, symbol=symbol,
                    detail=f"broker={b.side.value} internal={i.side.value}",
                ))
                continue
            if abs(b.qty - i.qty) > self.qty_tolerance:
                out.append(Discrepancy(
                    kind="QTY_MISMATCH", venue=venue, symbol=symbol,
                    detail=f"broker={b.qty} internal={i.qty} "
                           f"diff={b.qty - i.qty}",
                ))

        return out
