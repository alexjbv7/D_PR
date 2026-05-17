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
1. **daily_dd**         — pnl_day ≤ -daily_dd_kill_pct × equity  →  reject
2. **per_symbol_cap**   — new notional in symbol > per_symbol_cap_pct × equity
3. **per_venue_cap**    — new notional in venue  > per_venue_cap_pct × equity
4. **cash_buffer**      — would leave cash < min_cash_buffer_pct × equity
5. **broker_paper**     — block live submission when ``require_paper`` is set

References
----------
* CLAUDE.md §12 (Risk Management): exposure caps, DD protection, kill switch
* §12.9 lists portfolio_caps and kill_switch as "pending" — this is them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from quant_shared.schemas.orders import OrderIntent, OrderSide, Position

from .brokers.base import AccountInfo
from .repository import Repository, RiskDecision

logger = logging.getLogger(__name__)


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
    """

    def __init__(self, config: RiskConfig, repository: Repository):
        self.config = config
        self.repo   = repository

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

        # ---- 5. cash buffer ----
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
        if intent.limit_price is None:
            return Decimal("0")
        return intent.qty * intent.limit_price


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
