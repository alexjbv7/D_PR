-- ============================================================================
-- Migration 009 — Allocator state (Thompson sampling per horizon)
-- ============================================================================
-- Semana 9: Beta(α, β) posteriors per horizon + append-only audit log.
--
-- Execution:
--   psql $POSTGRES_DSN -f platform/infra/sql/migrations/009_allocator_state.sql
--   # or: cd platform && make db-migrate
--
-- Pre-requisites: risk schema (created in earlier migrations).
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS risk;

-- ---------------------------------------------------------------------------
-- risk.allocator_state
-- ---------------------------------------------------------------------------
-- One row per horizon.  Warm-start priors Beta(20, 20) → mean=0.5, var≈6.1e-3
-- (see ADR-032).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS risk.allocator_state (
    horizon          TEXT          NOT NULL,                  -- 'intraday' | 'swing' | 'daily'
    alpha            NUMERIC(20,10) NOT NULL DEFAULT 20.0,    -- wins (incl. prior)
    beta             NUMERIC(20,10) NOT NULL DEFAULT 20.0,    -- losses (incl. prior)
    last_update_ts   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    PRIMARY KEY (horizon)
);

-- Seed with warm-start priors.  No-op if already present.
INSERT INTO risk.allocator_state (horizon, alpha, beta)
VALUES ('intraday', 20.0, 20.0),
       ('swing',    20.0, 20.0),
       ('daily',    20.0, 20.0)
ON CONFLICT (horizon) DO NOTHING;

-- ---------------------------------------------------------------------------
-- risk.allocator_updates
-- ---------------------------------------------------------------------------
-- Append-only audit log.  PK on update_id (== UUID v7 == trade_id by convention)
-- gives us idempotency for free: re-applying the same trade_id is a no-op
-- via ON CONFLICT DO NOTHING from the consumer.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS risk.allocator_updates (
    update_id        TEXT          PRIMARY KEY,               -- UUID v7 (idempotency key)
    horizon          TEXT          NOT NULL,
    trade_id         TEXT          NOT NULL,                  -- ref to OrderResult.result_id
    outcome          TEXT          NOT NULL,                  -- 'win' | 'loss'
    realized_pnl     NUMERIC(20,10) NOT NULL,
    alpha_delta      NUMERIC(20,10) NOT NULL,                 -- +1 if win, 0 if loss (post-decay)
    beta_delta       NUMERIC(20,10) NOT NULL,                 -- 0 if win, +1 if loss
    alpha_after      NUMERIC(20,10) NOT NULL,
    beta_after       NUMERIC(20,10) NOT NULL,
    ts_utc           TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_allocator_updates_trade_id
    ON risk.allocator_updates (trade_id);

CREATE INDEX IF NOT EXISTS ix_allocator_updates_horizon_ts
    ON risk.allocator_updates (horizon, ts_utc DESC);
