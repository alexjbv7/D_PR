# ADR-016: Market calendar module in `shared/quant_shared/calendar/`

## Status

Accepted (2026-05-18)

## Context

`docs/architecture/alpaca_integration.md` §4 originally listed calendar-related paths
under `platform/services/execution-engine/` (aligned with §9.2-style “new folders” for
the execution service). During **Semana 3** (market calendar + RTH sessions), the need
arose to answer “is this symbol tradable now?” and expose `session_phase` not only at
order submission but also in:

- `feature-engine` / `ml-feature-store` (session-aware features),
- `signal_translator` and walk-forward research jobs,
- future multi-strategy runners.

Placing the calendar inside `execution-engine` would force those components to depend on
the executor package or duplicate NYSE/NASDAQ schedule logic.

## Decision

The canonical **`MarketCalendar`** implementation lives in:

`shared/quant_shared/calendar/`

Exports include `market_calendar` (singleton), `SessionPhase`, `get_session_phase`,
`session_phase_value`, and `MarketClosedError` (domain error, broker-neutral).

`RiskGate` imports only `market_calendar` and rejects equities outside RTH via
`RiskDecision(breach="market_closed")` — it does **not** raise `MarketClosedError`
(that exception remains for broker-layer mapping, e.g. Alpaca adapter).

`is_equity()` heuristics live in `shared/quant_shared/symbols.py` and are reused by
`execution-engine` routing.

## Consequences

**Positive**

- Single source of truth for US equity session boundaries across services.
- No duplication of holiday / DST / half-day rules.
- In-memory schedule cache with explicit hit/miss counters for observability tests.

**Negative / trade-offs**

- `quant-shared` gains a runtime dependency on `pandas_market_calendars` (transitive
  for any consumer of the calendar package).
- Schedule cache is intentionally **±400 calendar days** from “today” when refreshed
  (see comment in `market_calendar.py`). This supports historical `is_open()` and
  walk-forward without hitting `pandas_market_calendars` on the hot path. Estimated
  memory cost ~20 MB for XNYS + NASDAQ frames and day indexes — acceptable vs per-call
  schedule builds.

**Neutral**

- ADR-016 in the architecture doc table (BaseExecutor alias) is a separate naming
  collision; this file is the calendar ADR for Semana 3.

## References

- `docs/architecture/alpaca_integration.md` — §4 (folder diff), §5 Semana 3
- `shared/quant_shared/calendar/market_calendar.py`
- `platform/services/execution-engine/app/risk_gate.py`
- `shared/tests/test_market_calendar.py`
