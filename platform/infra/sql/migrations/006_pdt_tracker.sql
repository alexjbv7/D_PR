-- =============================================================================
-- Migration 006 — PDT day-trade ledger (Semana 6)
-- =============================================================================
-- Append-only position actions for FINRA 4210(f)(8) rolling 5-day day-trade count.

CREATE SCHEMA IF NOT EXISTS risk;

CREATE TABLE IF NOT EXISTS risk.position_actions (
    action_id        TEXT          PRIMARY KEY,
    account_id       TEXT          NOT NULL,
    symbol           TEXT          NOT NULL,
    asset_class      TEXT          NOT NULL DEFAULT 'us_equity',
    side             TEXT          NOT NULL CHECK (side IN ('buy', 'sell')),
    qty              NUMERIC(20,10) NOT NULL,
    notional         NUMERIC(20,10),
    fill_id          TEXT,
    ts_utc           TIMESTAMPTZ   NOT NULL,
    trade_date_et    DATE          NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pa_account_symbol_date
    ON risk.position_actions (account_id, symbol, trade_date_et);

CREATE INDEX IF NOT EXISTS idx_pa_account_date
    ON risk.position_actions (account_id, trade_date_et);
