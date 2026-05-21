-- =============================================================================
-- Migration 001 — execution-engine
-- =============================================================================
-- Schema: orders
--   intents     : every OrderIntent the risk-engine produces (approved or not)
--   results     : broker responses (one row per submission)
--   fills       : individual fills (multiple per result possible)
--   positions   : execution-engine's internal view (reconciled vs broker)
--
-- All monetary values use NUMERIC(28, 12) — sufficient for crypto (1e-12 BTC)
-- and equities.  All timestamps are TIMESTAMPTZ (UTC).  All IDs are UUID v7
-- generated client-side by quant_shared.schemas.orders._uuid7.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS orders;

-- ---------------------------------------------------------------------------
-- orders.intents — the audit log of every risk-evaluated OrderIntent
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS orders.intents (
    intent_id        UUID            PRIMARY KEY,
    signal_id        UUID,
    strategy         VARCHAR(50),
    symbol           VARCHAR(20)     NOT NULL,
    side             VARCHAR(10)     NOT NULL CHECK (side IN ('buy', 'sell')),
    qty              NUMERIC(28, 12) NOT NULL CHECK (qty > 0),
    order_type       VARCHAR(20)     NOT NULL,
    limit_price      NUMERIC(28, 12),
    sl_price         NUMERIC(28, 12),
    tp_price         NUMERIC(28, 12),
    tif              VARCHAR(10)     NOT NULL,
    venue            VARCHAR(30),

    -- risk metadata carried for audit
    kelly_fraction   NUMERIC(8, 6),
    target_risk_pct  NUMERIC(8, 6),
    p_win            NUMERIC(6, 4),

    -- risk-gate decision
    risk_decision    VARCHAR(20)     NOT NULL DEFAULT 'pending'
                     CHECK (risk_decision IN ('approved', 'rejected', 'pending')),
    risk_reason      TEXT,
    risk_breach      VARCHAR(50),

    ts               TIMESTAMPTZ     NOT NULL,
    created_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_intents_symbol_ts
    ON orders.intents (symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_intents_strategy_ts
    ON orders.intents (strategy, ts DESC);
CREATE INDEX IF NOT EXISTS idx_intents_decision
    ON orders.intents (risk_decision, created_at DESC);


-- ---------------------------------------------------------------------------
-- orders.results — broker responses (one row per submit attempt)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS orders.results (
    result_id        UUID            PRIMARY KEY,
    intent_id        UUID            NOT NULL REFERENCES orders.intents(intent_id),
    broker_id        VARCHAR(100),
    symbol           VARCHAR(20)     NOT NULL,
    side             VARCHAR(10)     NOT NULL CHECK (side IN ('buy', 'sell')),
    status           VARCHAR(20)     NOT NULL,
    qty              NUMERIC(28, 12) NOT NULL,
    filled_qty       NUMERIC(28, 12) NOT NULL DEFAULT 0,
    avg_price        NUMERIC(28, 12),
    venue            VARCHAR(30),
    reject_reason    TEXT,
    ts_submitted     TIMESTAMPTZ     NOT NULL,
    ts_updated       TIMESTAMPTZ     NOT NULL,
    raw              JSONB
);

CREATE INDEX IF NOT EXISTS idx_results_intent
    ON orders.results (intent_id);
CREATE INDEX IF NOT EXISTS idx_results_broker
    ON orders.results (broker_id) WHERE broker_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_results_status_ts
    ON orders.results (status, ts_updated DESC);


-- ---------------------------------------------------------------------------
-- orders.fills — individual executions
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS orders.fills (
    fill_id          UUID            PRIMARY KEY,
    order_id         VARCHAR(100)    NOT NULL,        -- broker_id from results
    result_id        UUID            REFERENCES orders.results(result_id),
    symbol           VARCHAR(20)     NOT NULL,
    side             VARCHAR(10)     NOT NULL CHECK (side IN ('buy', 'sell')),
    qty              NUMERIC(28, 12) NOT NULL,
    price            NUMERIC(28, 12) NOT NULL,
    fee              NUMERIC(28, 12) NOT NULL DEFAULT 0,
    fee_asset        VARCHAR(10)     NOT NULL DEFAULT 'USD',
    venue            VARCHAR(30),
    ts               TIMESTAMPTZ     NOT NULL,
    raw              JSONB
);

CREATE INDEX IF NOT EXISTS idx_fills_order
    ON orders.fills (order_id);
CREATE INDEX IF NOT EXISTS idx_fills_symbol_ts
    ON orders.fills (symbol, ts DESC);


-- ---------------------------------------------------------------------------
-- orders.positions — internal view; reconciler compares vs broker
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS orders.positions (
    venue            VARCHAR(30)     NOT NULL,
    symbol           VARCHAR(20)     NOT NULL,
    side             VARCHAR(10)     NOT NULL CHECK (side IN ('buy', 'sell')),
    qty              NUMERIC(28, 12) NOT NULL CHECK (qty > 0),
    avg_entry        NUMERIC(28, 12) NOT NULL,
    current_price    NUMERIC(28, 12),
    unrealized_pnl   NUMERIC(28, 12),
    margin_used      NUMERIC(28, 12),
    ts_opened        TIMESTAMPTZ,
    ts_updated       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (venue, symbol)
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol
    ON orders.positions (symbol);


-- ---------------------------------------------------------------------------
-- audit trigger: enforce ts_updated on positions row updates
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION orders.touch_ts_updated() RETURNS TRIGGER AS $$
BEGIN
    NEW.ts_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_positions_touch ON orders.positions;
CREATE TRIGGER trg_positions_touch
    BEFORE UPDATE ON orders.positions
    FOR EACH ROW
    EXECUTE FUNCTION orders.touch_ts_updated();
