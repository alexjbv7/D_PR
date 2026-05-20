# ADR-027 — PDT Buffer at $26k

**Status**: Accepted  
**Date**: 2026-05-20  
**Deciders**: Alex, Cursor  
**Implements**: Semana 6 of `docs/architecture/alpaca_integration.md §11`

---

## Context

FINRA 4210(f)(8) restricts accounts below $25,000 equity from executing more
than three day trades in a rolling five-trading-day window. In live brokerage
operations, equity can move intraday because of open P&L, fees, corporate
actions, and broker-side valuation timing.

The execution engine evaluates order intents before submit. If it waits until
the account is exactly below $25,000, a small adverse move can put the account
under the regulatory threshold after the risk check but before or after fill.

---

## Decision

The PDT controller blocks the next equity day trade when:

```text
equity < 26000 AND day_trade_count_last_5_trading_days >= 3
```

The $26,000 threshold is a protective buffer, roughly 4% above the legal
$25,000 threshold. It is deliberately conservative and only affects accounts
near the PDT boundary.

PDT remains equities-only. Crypto symbols bypass this check.

---

## Consequences

Positive:

- Reduces accidental PDT violations caused by small intraday equity changes.
- Keeps the rule deterministic and easy to audit.
- Preserves the FINRA definition of a day trade: one `(account, symbol,
  trade_date_et)` with at least one buy and one sell.

Negative:

- Accounts between $25,000 and $26,000 are more restricted than the legal
  minimum requires.
- The buffer can reject trades that a broker might technically accept.

---

## Follow-up

When prior-close equity history is available from paper/live reconciliation,
the controller should use equity at the previous close rather than the current
snapshot. Until then, `AccountInfo.equity` is the best available proxy.
