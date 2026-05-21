# ADR-024 — Corporate Actions Architecture

**Status**: Accepted  
**Date**: 2026-05-18  
**Deciders**: Alex, Claude Sonnet 4.6  
**Implements**: Semana 4 of `docs/architecture/alpaca_integration.md §11`

---

## Context

When a corporate action (split, dividend, merger…) occurs on an equity, two
separate concerns must be addressed:

1. **Historical bars** must be adjusted so that strategies see a continuous
   price series without artificial discontinuities.
2. **Open positions** must be adjusted so that the book's `qty` and `avg_entry`
   reflect the post-action reality (otherwise P&L calculations become wrong).

These two concerns live in different services:
- Historical bars are fetched and stored by `market-intelligence`.
- Open positions are tracked by `execution-engine`.

The key architectural question is: *where and how do we trigger each adjustment?*

---

## Decision

### D1 — Module in `market-intelligence`, not a new microservice

Corporate action fetching and bar adjustment are batch jobs running once per
day. They do not require dedicated ownership or independent deployment
cadence at this stage. Introducing a new microservice would add K8s overhead,
a new repo entry, and inter-service auth without proportional benefit.

**Chosen**: modules `app/corporate_actions/` and `app/universe/` inside the
existing `market-intelligence` service.

**Alternative rejected**: `corporate-actions-service` microservice (ADR to revisit
if team splits ownership in H2 2026).

### D2 — Positions adjusted by Kafka consumer in `execution-engine`

`execution-engine` is the single source of truth for open positions. Only it
should modify `Position` objects. `market-intelligence` must never import or
write to the positions store directly.

**Chosen**: `execution-engine/app/corporate_actions_consumer.py` listens to
`los_ojos.corporate_actions` (Kafka) and calls `repo.upsert_position()`.

**Alternative rejected**: DB trigger or direct HTTP call from `market-intelligence`
to `execution-engine` (violates service isolation; introduces tight coupling).

### D3 — Adjusted bars in a separate table

Raw bars (`market.ohlcv`) must remain untouched as an audit trail. Adjusted
bars are stored in `market.bars_1m_adjusted` (TimescaleDB hypertable). This
allows back-testing strategies against either adjusted or raw prices.

**Chosen**: new hypertable `market.bars_1m_adjusted`.

**Alternative rejected**: adding `adj_close`, `adj_volume` columns to
`market.ohlcv` (breaks TimescaleDB compression policies; complicates the
schema for consumers that only want raw data).

### D4 — `is_provisional` column for bars

Alpaca can publish or revise a CA announcement up to 48 hours after the
ex-date. During this window, the adjusted bars are marked `is_provisional=TRUE`.
A nightly re-run clears the flag once the CA is confirmed.

**Chosen**: `is_provisional BOOLEAN` column in `market.bars_1m_adjusted`.

**Alternative rejected**: separate `bars_provisional` table (same rows, more
join complexity).

### D5 — Point-in-time universe via SQL

Avoiding survivorship bias requires knowing which symbols were *listed* as of a
given historical date. This is achieved with:

```sql
SELECT symbol
  FROM data.universe_historical
 WHERE first_listed_ts <= $as_of_ts
   AND (delisted_ts IS NULL OR delisted_ts > $as_of_ts);
```

This view is cheap, no extra tables needed.

**Alternative rejected**: daily S3 snapshots (expensive, redundant, harder to
query point-in-time).

### D6 — Idempotence via `corporate_actions_applied`

The daily cron may be re-run (crash recovery, re-deployment). The
`market.corporate_actions_applied` table records each `(ca_id, target)` pair
after a successful application. Before any application, the system checks this
table.

**Chosen**: append-only `corporate_actions_applied` table with
`PRIMARY KEY (ca_id, target)`.

**Alternative rejected**: UPSERT-based idempotence (less auditable; does not
distinguish first application from re-runs).

### D7 — Trading halts vs. permanent delistings (3-day buffer)

When Alpaca marks a symbol as `inactive`, it may be a temporary trading halt,
not a permanent delisting. To avoid false `delisting` events:

1. Symbol is added to `data.delisting_candidates` with `first_seen_inactive_ts`.
2. If the symbol returns to `active` within 3 days, the candidate is removed
   (no delisting emitted).
3. If the symbol remains inactive for 3+ consecutive cron runs, the delisting
   is confirmed.

**Chosen**: `data.delisting_candidates` auxiliary table with 3-day buffer.

---

## Consequences

### Positive

- Clear service boundaries: `market-intelligence` handles data ingestion,
  `execution-engine` handles position state.
- Full audit trail: raw bars untouched, CA applications logged.
- Idempotent: safe to re-run nightly crons without side effects.
- Survivorship-bias-free universe via point-in-time SQL.

### Negative

- Additional Kafka consumer in `execution-engine` adds startup complexity.
- 48-hour provisional window means adjusted bars may be incorrect for a short
  time after a CA. Consumers should filter `is_provisional=FALSE` in production.

### Neutral

- Spinoffs are logged as `WARN` and marked for manual resolution
  (`TODO(@alex 2026-07-01)`). Two-symbol events require a future dedicated flow.
- Cash dividends do not adjust price (see ADR-025).
