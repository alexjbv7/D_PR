"""
PDTTracker — FINRA 4210(f)(8) day-trade counter.

Rolling 5-trading-day window. Equities only. Buffer at $26k to avoid
accidental breach of the $25k legal minimum after a losing day-trade.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from quant_shared.calendar import market_calendar
from quant_shared.symbols import is_equity

from ..repository import Repository

logger = logging.getLogger(__name__)

# Buffer: block when equity < $26k (4% above FINRA $25k minimum).
_PDT_EQUITY_THRESHOLD = Decimal("26000")
_PDT_MAX_DAY_TRADES = 3


@dataclass(frozen=True)
class PDTDecision:
    blocked:       bool
    reason:        str
    count_last_5d: int
    equity:        Decimal


class PDTTracker:
    """
    Stateless query layer over ``risk.position_actions``.

    Uses ``account.equity`` as the equity snapshot (TODO: prior close when
    available from paper-trading history — Semana 10).
    """

    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    async def check(
        self,
        account_id: str,
        symbol:     str,
        equity:     Decimal,
        intent_ts:  datetime,
    ) -> PDTDecision:
        if not is_equity(symbol):
            return PDTDecision(
                blocked=False,
                reason="not_equity",
                count_last_5d=0,
                equity=equity,
            )

        if equity >= _PDT_EQUITY_THRESHOLD:
            return PDTDecision(
                blocked=False,
                reason="above_threshold",
                count_last_5d=0,
                equity=equity,
            )

        last_5 = market_calendar.last_n_trading_dates(symbol, intent_ts, n=5)
        if not last_5:
            return PDTDecision(
                blocked=False,
                reason="no_trading_history",
                count_last_5d=0,
                equity=equity,
            )

        count = await self.repo.count_day_trades(
            account_id=account_id,
            since_date_et=last_5[0],
            until_date_et=last_5[-1],
        )

        if count >= _PDT_MAX_DAY_TRADES:
            reason = (
                f"PDT: {count} day-trades in last 5 trading days, "
                f"equity ${equity} < ${_PDT_EQUITY_THRESHOLD} threshold"
            )
            logger.warning(
                "pdt.blocked account=%s symbol=%s count=%s equity=%s",
                account_id, symbol, count, equity,
            )
            return PDTDecision(
                blocked=True,
                reason=reason,
                count_last_5d=count,
                equity=equity,
            )

        return PDTDecision(
            blocked=False,
            reason="under_limit",
            count_last_5d=count,
            equity=equity,
        )
