-- =============================================================================
-- Migration 007 — OrderIntent notional + extended-hours persistence (Semana 6)
-- =============================================================================
-- Supports fractional-share intents where qty is NULL and notional is set.

ALTER TABLE orders.intents
    ADD COLUMN IF NOT EXISTS notional NUMERIC(28, 12),
    ADD COLUMN IF NOT EXISTS extended_hours BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE orders.intents
    ALTER COLUMN qty DROP NOT NULL;

ALTER TABLE orders.intents
    DROP CONSTRAINT IF EXISTS intents_qty_check,
    DROP CONSTRAINT IF EXISTS chk_intents_qty_positive,
    DROP CONSTRAINT IF EXISTS chk_intents_notional_positive,
    DROP CONSTRAINT IF EXISTS chk_intents_qty_xor_notional;

ALTER TABLE orders.intents
    ADD CONSTRAINT chk_intents_qty_positive
        CHECK (qty IS NULL OR qty > 0),
    ADD CONSTRAINT chk_intents_notional_positive
        CHECK (notional IS NULL OR notional > 0),
    ADD CONSTRAINT chk_intents_qty_xor_notional
        CHECK ((qty IS NULL) <> (notional IS NULL));
