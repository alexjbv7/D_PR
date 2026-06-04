"""
RiskGate — pre-submit risk validation for OrderIntents.
=========================================================

The gate consults:
  * current account snapshot (equity, cash, intraday P&L)
  * current open positions (per-symbol / per-venue exposure)
  * static config (caps, kill thresholds)

and either approves or rejects each intent.  Every decision is logged via
the :class:`Repository`.

Checks (evaluated in order; first breach short-circuits)
--------------------------------------------------------
1. **require_paper**    — block live submission when ``require_paper`` is set
2. **daily_dd**         — pnl_day ≤ -daily_dd_kill_pct × equity  →  reject
3. **per_symbol_cap**   — new notional in symbol > per_symbol_cap_pct × equity
4. **per_venue_cap**    — new notional in venue  > per_venue_cap_pct × equity
5. **extended_hours**   — Alpaca ETH requires LIMIT and a valid pre/post window
6. **market_open**      — US equities outside RTH/ETH → reject (``market_closed``)
7. **cash_buffer**      — would leave cash < min_cash_buffer_pct × equity

References
----------
* CLAUDE.md §12 (Risk Management): exposure caps, DD protection, kill switch
* §12.9 lists portfolio_caps and kill_switch as "pending" — this is them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time
from decimal import Decimal
from typing import Optional
from zoneinfo import ZoneInfo

from quant_shared.calendar import market_calendar
from quant_shared.calendar.session_phase import SessionPhase
from quant_shared.schemas.orders import OrderIntent, OrderSide, OrderType, Position

from ._pdt.pdt_tracker import PDTTracker
from .brokers.base import AccountInfo
from .repository import Repository, RiskDecision

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_ALPACA_EXTENDED_OPEN_ET = time(4, 0)
_ALPACA_EXTENDED_CLOSE_ET = time(20, 0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """
    Static risk limits.  All percentages are fractions of equity (0.05 = 5 %).

    Parameters
    ----------
    per_symbol_cap_pct : float
        Max notional in a single symbol, as fraction of equity.
    per_venue_cap_pct : float
        Max notional in a single venue, as fraction of equity.
    daily_dd_kill_pct : float
        If ``pnl_day / equity`` falls below ``-daily_dd_kill_pct``, the gate
        rejects all new intents (CLAUDE.md §12.2).
    min_cash_buffer_pct : float
        Minimum cash to keep free at all times (defensive against margin calls).
    require_paper : bool
        When True, reject any intent submitted via a live (non-paper) account.
        Useful as a global safety while in development.
    """
    per_symbol_cap_pct:  float = 0.05      # 5 %
    per_venue_cap_pct:   float = 0.50      # 50 %
    daily_dd_kill_pct:   float = 0.03      # 3 %
    min_cash_buffer_pct: float = 0.10      # 10 %
    require_paper:       bool  = True

    def __post_init__(self) -> None:
        for name, value in (
            ("per_symbol_cap_pct",  self.per_symbol_cap_pct),
            ("per_venue_cap_pct",   self.per_venue_cap_pct),
            ("daily_dd_kill_pct",   self.daily_dd_kill_pct),
            ("min_cash_buffer_pct", self.min_cash_buffer_pct),
        ):
            if not (0 <= value <= 1):
                raise ValueError(f"{name} must be in [0, 1], got {value}")


# ---------------------------------------------------------------------------
# RiskGate
# ---------------------------------------------------------------------------

class RiskGate:
    """
    Pre-submit validator.

    Parameters
    ----------
    config : RiskConfig
    repository : Repository
        Used to look up open positions and persist the decision.

    Examples
    --------
    >>> gate = RiskGate(RiskConfig(), repo)
    >>> decision = await gate.evaluate(intent, account)
    >>> if decision.approved:
    ...     result = await broker.submit(intent)

    Notes
    -----
    Kill switch: call :meth:`trip_kill_switch` to block all new intents
    immediately (checked before every other check).  Call
    :meth:`reset_kill_switch` to re-enable submission.  Wired from
    ``main.py._make_kill_switch_callback`` so that the reconciler can also
    block REST-submitted intents, not just the Kafka consumer loop.
    """

    def __init__(self, config: RiskConfig, repository: Repository):
        self.config             = config
        self.repo               = repository
        self.pdt_tracker        = PDTTracker(repository)
        self._kill_switch_active = False

    # -----------------------------------------------------------------------
    # Kill switch
    # -----------------------------------------------------------------------

    def trip_kill_switch(self) -> None:
        """Block all new intents.  Idempotent."""
        self._kill_switch_active = True
        logger.critical("risk_gate.kill_switch.tripped")

    def reset_kill_switch(self) -> None:
        """Re-enable intent submission.  Idempotent."""
        self._kill_switch_active = False
        logger.warning("risk_gate.kill_switch.reset")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def evaluate(
        self,
        intent: OrderIntent,
        account: AccountInfo,
    ) -> RiskDecision:
        """
        Run all checks and persist the decision alongside the intent.

        Returns
        -------
        RiskDecision
            ``approved=True`` if every check passed; otherwise ``approved=False``
            with ``reason`` and ``breach`` filled in.
        """
        decision = await self._run_checks(intent, account)
        await self.repo.save_intent(intent, decision)

        if decision.approved:
            logger.info(
                "risk_gate.approved intent=%s symbol=%s qty=%s",
                intent.intent_id[:8], intent.symbol, intent.qty,
            )
        else:
            logger.warning(
                "risk_gate.rejected intent=%s breach=%s reason=%s",
                intent.intent_id[:8], decision.breach, decision.reason,
            )
        return decision

    # -----------------------------------------------------------------------
    # Internal: check pipeline
    # -----------------------------------------------------------------------

    async def _run_checks(
        self,
        intent: OrderIntent,
        account: AccountInfo,
    ) -> RiskDecision:
        # ---- 0. kill switch (checked before everything else) ----
        if self._kill_switch_active:
            return RiskDecision(
                approved=False,
                breach="kill_switch",
                reason="Kill switch is active — all new intents blocked",
            )

        # ---- 1. require_paper (global safety) ----
        if self.config.require_paper and not account.is_paper:
            return RiskDecision(
                approved=False,
                breach="require_paper",
                reason="Live trading blocked by RiskConfig.require_paper=True",
            )

        # ---- 2. daily DD kill ----
        if account.equity > 0:
            dd_ratio = float(account.pnl_day / account.equity)
            if dd_ratio < -self.config.daily_dd_kill_pct:
                return RiskDecision(
                    approved=False,
                    breach="daily_dd",
                    reason=(
                        f"Daily DD {dd_ratio:.2%} breaches "
                        f"-{self.config.daily_dd_kill_pct:.0%} kill threshold"
                    ),
                )

        # Estimate notional of this new intent.  For market orders we don't
        # know the fill price ex-ante; use limit_price when present, else
        # treat as 0 (cannot pre-validate notional).
        intent_notional = self._estimate_notional(intent)

        # Fetch positions once; reuse across checks.
        positions = await self.repo.get_open_positions()

        # ---- 3. per-symbol cap ----
        symbol_notional = _notional_for_symbol(positions, intent.symbol) + intent_notional
        cap_symbol = Decimal(str(self.config.per_symbol_cap_pct)) * account.equity
        if symbol_notional > cap_symbol:
            return RiskDecision(
                approved=False,
                breach="per_symbol_cap",
                reason=(
                    f"symbol notional {symbol_notional} > "
                    f"{self.config.per_symbol_cap_pct:.0%} of equity "
                    f"({cap_symbol})"
                ),
            )

        # ---- 4. per-venue cap ----
        venue = intent.venue or "unknown"
        venue_notional = _notional_for_venue(positions, venue) + intent_notional
        cap_venue = Decimal(str(self.config.per_venue_cap_pct)) * account.equity
        if venue_notional > cap_venue:
            return RiskDecision(
                approved=False,
                breach="per_venue_cap",
                reason=(
                    f"venue {venue!r} notional {venue_notional} > "
                    f"{self.config.per_venue_cap_pct:.0%} of equity "
                    f"({cap_venue})"
                ),
            )

        # ---- 5. extended hours requires LIMIT (Alpaca ETH) ----
        if intent.extended_hours and intent.order_type != OrderType.LIMIT:
            return RiskDecision(
                approved=False,
                breach="extended_hours_requires_limit",
                reason=(
                    f"extended_hours=True requires order_type=LIMIT, "
                    f"got {intent.order_type.value}"
                ),
            )

        # ---- 6. market access (US equities RTH or Alpaca ETH; crypto 24/7) ----
        if not _market_access_allowed(intent):
            return RiskDecision(
                approved=False,
                breach="market_closed",
                reason=(
                    f"Market closed or outside supported extended-hours window "
                    f"for {intent.symbol} at "
                    f"{intent.ts.isoformat()}"
                ),
            )

        # ---- 7. PDT rule (equities, equity < $26k buffer) ----
        pdt = await self.pdt_tracker.check(
            account_id=account.account_id,
            symbol=intent.symbol,
            equity=account.equity,
            intent_ts=intent.ts,
        )
        if pdt.blocked:
            return RiskDecision(
                approved=False,
                breach="pdt_rule",
                reason=pdt.reason,
            )

        # ---- 8. cash buffer ----
        required_buffer = Decimal(str(self.config.min_cash_buffer_pct)) * account.equity
        if intent.side == OrderSide.BUY and intent_notional > 0:
            projected_cash = account.cash - intent_notional
            if projected_cash < required_buffer:
                return RiskDecision(
                    approved=False,
                    breach="cash_buffer",
                    reason=(
                        f"projected cash {projected_cash} would fall below "
                        f"{self.config.min_cash_buffer_pct:.0%} buffer "
                        f"({required_buffer})"
                    ),
                )

        return RiskDecision(approved=True, reason="all checks passed")

    @staticmethod
    def _estimate_notional(intent: OrderIntent) -> Decimal:
        if intent.notional is not None:
            return intent.notional
        if intent.qty is not None and intent.limit_price is not None:
            return intent.qty * intent.limit_price
        return Decimal("0")


# ---------------------------------------------------------------------------
# Module-level helpers (free functions, easy to test)
# ---------------------------------------------------------------------------

def _notional_for_symbol(positions: list[Position], symbol: str) -> Decimal:
    return sum(
        (p.notional for p in positions if p.symbol == symbol),
        Decimal("0"),
    )


def _notional_for_venue(positions: list[Position], venue: str) -> Decimal:
    return sum(
        (p.notional for p in positions if p.venue == venue),
        Decimal("0"),
    )


def _market_access_allowed(intent: OrderIntent) -> bool:
    """True for crypto, equity RTH, or Alpaca's supported extended window."""
    if market_calendar.is_open(intent.symbol, intent.ts):
        return True
    if not intent.extended_hours:
        return False

    phase = market_calendar.session_phase(intent.symbol, intent.ts)
    if phase not in (SessionPhase.PRE_MARKET, SessionPhase.POST_MARKET):
        return False

    local_time = intent.ts.astimezone(_ET).time()
    return _ALPACA_EXTENDED_OPEN_ET <= local_time < _ALPACA_EXTENDED_CLOSE_ET
