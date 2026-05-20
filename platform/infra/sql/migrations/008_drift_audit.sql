-- ============================================================================
-- Migration 008 — Drift detection audit tables
-- ============================================================================
-- Semana 8: PSI history, ECE history, and model predictions for audit.
--
-- Execution:
--   psql $POSTGRES_DSN -f platform/infra/sql/migrations/008_drift_audit.sql
--   # or: cd platform && make db-migrate
--
-- Requires: TimescaleDB extension already enabled (see earlier migrations).
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Schemas
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS drift;


-- ---------------------------------------------------------------------------
-- audit.predictions
-- ---------------------------------------------------------------------------
-- Model prediction log.  true_label is filled asynchronously once the
-- outcome bar closes (label join worker).  Used to compute rolling ECE.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS audit.predictions (
    ts               TIMESTAMPTZ  NOT NULL,
    horizon          TEXT         NOT NULL,   -- "intraday" | "swing" | "daily"
    model_version    TEXT         NOT NULL,
    symbol           TEXT         NOT NULL,
    direction        SMALLINT     NOT NULL,   -- -1 | 0 | 1
    probas           FLOAT[]      NOT NULL,   -- [p_short, p_neutral, p_long]
    true_label       SMALLINT,               -- NULL until outcome known
    feature_set_hash TEXT         NOT NULL DEFAULT ''
);

SELECT create_hypertable(
    'audit.predictions', 'ts',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '7 days'
);

CREATE INDEX IF NOT EXISTS ix_predictions_horizon_version
    ON audit.predictions (horizon, model_version, ts DESC);

CREATE INDEX IF NOT EXISTS ix_predictions_labelled
    ON audit.predictions (horizon, model_version, ts DESC)
    WHERE true_label IS NOT NULL;

-- ---------------------------------------------------------------------------
-- drift.psi_history
-- ---------------------------------------------------------------------------
-- Historical record of PSI readings per feature, per model version.
-- Retention: 90 days (same as warm tier in TimescaleDB policy).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS drift.psi_history (
    ts               TIMESTAMPTZ  NOT NULL,
    horizon          TEXT         NOT NULL,
    model_version    TEXT         NOT NULL,
    feature_name     TEXT         NOT NULL,
    psi              FLOAT        NOT NULL,
    severity         TEXT         NOT NULL,   -- "stable" | "moderate" | "severe"
    n_buckets        INT          NOT NULL DEFAULT 10,
    macro_suppressed BOOLEAN      NOT NULL DEFAULT FALSE
);

SELECT create_hypertable(
    'drift.psi_history', 'ts',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '30 days'
);

CREATE INDEX IF NOT EXISTS ix_psi_horizon_feature
    ON drift.psi_history (horizon, feature_name, ts DESC);

-- ---------------------------------------------------------------------------
-- drift.ece_history
-- ---------------------------------------------------------------------------
-- Historical record of Expected Calibration Error per horizon.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS drift.ece_history (
    ts              TIMESTAMPTZ  NOT NULL,
    horizon         TEXT         NOT NULL,
    model_version   TEXT         NOT NULL,
    ece             FLOAT        NOT NULL,
    brier           FLOAT        NOT NULL,
    n_samples       INT          NOT NULL,
    window_days     INT          NOT NULL DEFAULT 7
);

SELECT create_hypertable(
    'drift.ece_history', 'ts',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '30 days'
);

CREATE INDEX IF NOT EXISTS ix_ece_horizon
    ON drift.ece_history (horizon, ts DESC);

-- ---------------------------------------------------------------------------
-- drift.retrain_history — audit trail for retrain triggers (Grafana panel 4)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS drift.retrain_history (
    ts              TIMESTAMPTZ  NOT NULL,
    horizon         TEXT         NOT NULL,
    model_version   TEXT         NOT NULL,
    trigger_reason  TEXT         NOT NULL,
    suppressed      BOOLEAN      NOT NULL,
    psi_max         FLOAT        NOT NULL DEFAULT 0,
    ece             FLOAT        NOT NULL DEFAULT 0
);

SELECT create_hypertable(
    'drift.retrain_history', 'ts',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '30 days'
);

CREATE INDEX IF NOT EXISTS ix_retrain_horizon
    ON drift.retrain_history (horizon, ts DESC);

-- ---------------------------------------------------------------------------
-- Retention policies (90 days warm tier)
-- ---------------------------------------------------------------------------

SELECT add_retention_policy(
    'audit.predictions', INTERVAL '90 days', if_not_exists => TRUE
);

SELECT add_retention_policy(
    'drift.psi_history', INTERVAL '90 days', if_not_exists => TRUE
);

SELECT add_retention_policy(
    'drift.ece_history', INTERVAL '90 days', if_not_exists => TRUE
);

SELECT add_retention_policy(
    'drift.retrain_history', INTERVAL '90 days', if_not_exists => TRUE
);
