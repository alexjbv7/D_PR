-- =============================================================================
-- Migration 004 — Universe (no-survivorship) + Corporate Actions
-- =============================================================================
-- Implements Semana 4 of docs/architecture/alpaca_integration.md §11.
--
-- New schemas / tables:
--   data.equities.universe_historical  — point-in-time equity universe
--   data.equities.delisting_candidates — 3-day halt-vs-delisting buffer
--   market.corporate_actions           — audit-trail append-only CA log
--   market.corporate_actions_applied   — idempotency log per (ca_id, target)
--   market.bars_1m_adjusted            — adjusted bars hypertable (raw untouched)
--
-- Raw market.ohlcv remains UNCHANGED.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema: data (equity universe lives here, separate from market time-series)
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS data;

-- ---------------------------------------------------------------------------
-- 1. Equity universe — point-in-time, no survivorship bias
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS data.universe_historical (
    symbol               TEXT        NOT NULL,
    asset_class          TEXT        NOT NULL DEFAULT 'us_equity',
    exchange             TEXT        NOT NULL,       -- 'XNYS' | 'XNAS' | ...
    name                 TEXT,
    first_listed_ts      TIMESTAMPTZ,               -- NULL if outside backfill window
    delisted_ts          TIMESTAMPTZ,               -- NULL = still active
    is_tradable          BOOLEAN     NOT NULL DEFAULT TRUE,
    fractionable         BOOLEAN     NOT NULL DEFAULT FALSE,
    shortable            BOOLEAN     NOT NULL DEFAULT FALSE,
    last_updated_ts      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw                  JSONB,                     -- full Alpaca /v2/assets payload
    PRIMARY KEY (symbol, asset_class)
);

-- Fast filter: point-in-time active universe
-- Usage: WHERE delisted_ts IS NULL OR delisted_ts > $as_of_ts
CREATE INDEX IF NOT EXISTS idx_universe_active
    ON data.universe_historical (delisted_ts)
    WHERE delisted_ts IS NULL;

CREATE INDEX IF NOT EXISTS idx_universe_exchange
    ON data.universe_historical (exchange, delisted_ts);

-- ---------------------------------------------------------------------------
-- 2. Delisting candidates — buffer to avoid confusing trading halts with delists
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS data.delisting_candidates (
    symbol                  TEXT        PRIMARY KEY,
    first_seen_inactive_ts  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    confirmed               BOOLEAN     NOT NULL DEFAULT FALSE,
    last_checked_ts         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- 3. Corporate actions — append-only audit trail
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS market.corporate_actions (
    ca_id                TEXT        PRIMARY KEY,   -- UUID v7 (client-generated)
    alpaca_id            TEXT        UNIQUE,         -- Alpaca native ID (for upsert key)
    symbol               TEXT        NOT NULL,
    ca_type              TEXT        NOT NULL
                         CHECK (ca_type IN (
                             'forward_split', 'reverse_split',
                             'stock_dividend', 'cash_dividend',
                             'merger', 'spinoff', 'name_change'
                         )),
    declared_ts          TIMESTAMPTZ,
    ex_ts                TIMESTAMPTZ NOT NULL,      -- price adjusts from this date
    record_ts            TIMESTAMPTZ,
    payable_ts           TIMESTAMPTZ,
    -- Split / dividend fields
    split_from           NUMERIC(20, 10),           -- e.g. 1 in 4:1 split
    split_to             NUMERIC(20, 10),           -- e.g. 4 in 4:1 split
    cash_amount          NUMERIC(20, 10),           -- cash dividend per share
    stock_amount         NUMERIC(20, 10),           -- stock dividend ratio (0.10 = 10%)
    new_symbol           TEXT,                      -- mergers / name_changes / spinoffs
    -- Lifecycle
    is_provisional       BOOLEAN     NOT NULL DEFAULT TRUE,
    fetched_ts           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw                  JSONB
);

CREATE INDEX IF NOT EXISTS idx_ca_symbol_ex_ts
    ON market.corporate_actions (symbol, ex_ts);

CREATE INDEX IF NOT EXISTS idx_ca_provisional
    ON market.corporate_actions (is_provisional, fetched_ts)
    WHERE is_provisional = TRUE;

-- ---------------------------------------------------------------------------
-- 4. Corporate actions applied — idempotency table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS market.corporate_actions_applied (
    ca_id                TEXT        NOT NULL REFERENCES market.corporate_actions(ca_id),
    target               TEXT        NOT NULL
                         CHECK (target IN ('bars', 'positions')),
    ts_applied           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rows_affected        INTEGER     NOT NULL,
    success              BOOLEAN     NOT NULL,
    error_msg            TEXT,
    PRIMARY KEY (ca_id, target)
);

-- ---------------------------------------------------------------------------
-- 5. Adjusted bars — separate hypertable (raw market.ohlcv untouched)
-- ---------------------------------------------------------------------------
-- Note: market.ohlcv uses (time, symbol, timeframe) PK.
-- bars_1m_adjusted mirrors that structure + adj columns + CA metadata.
CREATE TABLE IF NOT EXISTS market.bars_1m_adjusted (
    time                 TIMESTAMPTZ NOT NULL,
    symbol               TEXT        NOT NULL,
    timeframe            TEXT        NOT NULL DEFAULT '1m',
    -- Raw-equivalent columns (adjusted values stored here)
    open                 NUMERIC(20, 8) NOT NULL,
    high                 NUMERIC(20, 8) NOT NULL,
    low                  NUMERIC(20, 8) NOT NULL,
    close                NUMERIC(20, 8) NOT NULL,
    volume               NUMERIC(30, 8) NOT NULL,
    quote_volume         NUMERIC(30, 4),
    trade_count          INTEGER,
    taker_buy_vol        NUMERIC(30, 8),
    source               TEXT          DEFAULT 'alpaca',
    -- Adjustment metadata
    is_provisional       BOOLEAN     NOT NULL DEFAULT FALSE,
    last_ca_id_applied   TEXT        REFERENCES market.corporate_actions(ca_id),
    PRIMARY KEY (time, symbol, timeframe)
);

SELECT create_hypertable(
    'market.bars_1m_adjusted', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

DO $$
BEGIN
    BEGIN
        PERFORM add_compression_policy(
            'market.bars_1m_adjusted',
            INTERVAL '7 days',
            if_not_exists => TRUE
        );
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Skipping compression policy for market.bars_1m_adjusted: %', SQLERRM;
    END;
END $$;

-- ---------------------------------------------------------------------------
-- 6. Kafka topics (comment-only — applied via infra/kafka/topics.yml)
-- ---------------------------------------------------------------------------
-- los_ojos.corporate_actions   partitions=4 compact  (key=ca_id)
-- los_ojos.universe.updates    partitions=2 delete 1y
