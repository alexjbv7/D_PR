-- =============================================================================
-- Los Ojos — PostgreSQL + TimescaleDB Schema
-- Version: 1.0.0
-- =============================================================================
-- Orden de creación:
--   1. Extensions
--   2. Tablas base (assets, strategies)
--   3. Hypertables TimescaleDB (series temporales)
--   4. Tablas de señales y posiciones
--   5. Tablas macro y on-chain
--   6. Feature store
--   7. Auditoría y logs
--   8. Índices
--   9. Compression policies
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- ---------------------------------------------------------------------------
-- Schemas
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS market;
CREATE SCHEMA IF NOT EXISTS signals;
CREATE SCHEMA IF NOT EXISTS macro;
CREATE SCHEMA IF NOT EXISTS onchain;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS bot;
CREATE SCHEMA IF NOT EXISTS audit;

-- =============================================================================
-- 1. MARKET DATA
-- =============================================================================

-- Catálogo de instrumentos
CREATE TABLE IF NOT EXISTS market.assets (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL UNIQUE,
    base_currency   VARCHAR(10) NOT NULL,
    quote_currency  VARCHAR(10) NOT NULL DEFAULT 'USDT',
    asset_class     VARCHAR(20) NOT NULL DEFAULT 'crypto', -- crypto | equity | fx | commodity
    exchange        VARCHAR(30) NOT NULL DEFAULT 'binance',
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    tick_size       NUMERIC(20, 10),
    lot_size        NUMERIC(20, 10),
    min_notional    NUMERIC(20, 4),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- OHLCV candles (hypertable)
CREATE TABLE IF NOT EXISTS market.ohlcv (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(5) NOT NULL DEFAULT '1h',  -- 1m, 5m, 15m, 1h, 4h, 1d
    open            NUMERIC(20, 8) NOT NULL,
    high            NUMERIC(20, 8) NOT NULL,
    low             NUMERIC(20, 8) NOT NULL,
    close           NUMERIC(20, 8) NOT NULL,
    volume          NUMERIC(30, 8) NOT NULL,
    quote_volume    NUMERIC(30, 4),
    trade_count     INTEGER,
    taker_buy_vol   NUMERIC(30, 8),
    source          VARCHAR(20) DEFAULT 'binance',
    PRIMARY KEY (time, symbol, timeframe)
);

SELECT create_hypertable('market.ohlcv', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Order book snapshots (hypertable)
CREATE TABLE IF NOT EXISTS market.orderbook_snapshots (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    bid_price       NUMERIC(20, 8) NOT NULL,
    ask_price       NUMERIC(20, 8) NOT NULL,
    spread_bps      NUMERIC(10, 4),
    imbalance       NUMERIC(6, 4),          -- -1 to +1
    weighted_mid    NUMERIC(20, 8),
    bid_depth_5pct  NUMERIC(30, 4),
    ask_depth_5pct  NUMERIC(30, 4),
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('market.orderbook_snapshots', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Funding rates (perpetuals)
CREATE TABLE IF NOT EXISTS market.funding_rates (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    funding_rate    NUMERIC(12, 8) NOT NULL,
    basis_bps       NUMERIC(10, 4),
    funding_z       NUMERIC(8, 4),          -- rolling z-score
    annual_pct      NUMERIC(8, 4),
    next_funding_ts TIMESTAMPTZ,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('market.funding_rates', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Trades / tick data (hypertable — high volume)
CREATE TABLE IF NOT EXISTS market.trades (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    trade_id        BIGINT,
    price           NUMERIC(20, 8) NOT NULL,
    quantity        NUMERIC(20, 8) NOT NULL,
    is_buyer_maker  BOOLEAN,
    PRIMARY KEY (time, symbol, trade_id)
);

SELECT create_hypertable('market.trades', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- =============================================================================
-- 2. MACRO DATA
-- =============================================================================

-- FRED time series (hypertable)
CREATE TABLE IF NOT EXISTS macro.fred_series (
    time            TIMESTAMPTZ NOT NULL,
    series_id       VARCHAR(30) NOT NULL,
    value           NUMERIC(20, 6) NOT NULL,
    source          VARCHAR(20) DEFAULT 'fred',
    PRIMARY KEY (time, series_id)
);

SELECT create_hypertable('macro.fred_series', 'time',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- Macro regimes (snapshot table)
CREATE TABLE IF NOT EXISTS macro.regimes (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    recession_prob  NUMERIC(5, 4) NOT NULL,   -- 0-1
    expansion_prob  NUMERIC(5, 4) NOT NULL,
    regime          VARCHAR(30) NOT NULL,      -- expansion | slowdown | recession | recovery | stagflation
    rate_env        VARCHAR(20) NOT NULL,      -- hiking | cutting | neutral | pause
    yield_curve_inv BOOLEAN NOT NULL DEFAULT FALSE,
    sahm_value      NUMERIC(6, 4),
    t10y2y          NUMERIC(6, 4),
    yield_curve_sig NUMERIC(5, 4),
    sahm_sig        NUMERIC(5, 4),
    leading_sig     NUMERIC(5, 4),
    is_current      BOOLEAN NOT NULL DEFAULT TRUE
);

-- =============================================================================
-- 3. ON-CHAIN DATA
-- =============================================================================

-- Whale transactions (hypertable)
CREATE TABLE IF NOT EXISTS onchain.whale_transactions (
    time            TIMESTAMPTZ NOT NULL,
    tx_hash         VARCHAR(100) NOT NULL,
    asset           VARCHAR(20) NOT NULL,
    amount_usd      NUMERIC(20, 2) NOT NULL,
    direction       SMALLINT NOT NULL,         -- -1 bearish | 0 neutral | 1 bullish
    tx_type         VARCHAR(30) NOT NULL,      -- exchange_inflow | exchange_outflow | wallet_to_wallet
    from_address    VARCHAR(100),
    to_address      VARCHAR(100),
    from_label      VARCHAR(100),
    to_label        VARCHAR(100),
    whale_tier      VARCHAR(20),               -- mega | large | medium
    source          VARCHAR(30) DEFAULT 'crucix',
    PRIMARY KEY (time, tx_hash)
);

SELECT create_hypertable('onchain.whale_transactions', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Net flow aggregates (hypertable)
CREATE TABLE IF NOT EXISTS onchain.net_flows (
    time            TIMESTAMPTZ NOT NULL,
    asset           VARCHAR(20) NOT NULL,
    window_hours    INTEGER NOT NULL DEFAULT 24,
    net_flow_usd    NUMERIC(20, 2),
    inflow_usd      NUMERIC(20, 2),
    outflow_usd     NUMERIC(20, 2),
    whale_count     INTEGER,
    sentiment_score NUMERIC(5, 4),             -- -1 to +1
    PRIMARY KEY (time, asset, window_hours)
);

SELECT create_hypertable('onchain.net_flows', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Known wallet labels
CREATE TABLE IF NOT EXISTS onchain.wallet_labels (
    address         VARCHAR(100) PRIMARY KEY,
    label           VARCHAR(100) NOT NULL,
    entity_type     VARCHAR(30) NOT NULL,      -- exchange | fund | whale | miner | defi
    exchange_name   VARCHAR(50),
    is_cex          BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- 4. SIGNALS AND POSITIONS
-- =============================================================================

-- Trading signals (hypertable)
CREATE TABLE IF NOT EXISTS signals.trading_signals (
    time            TIMESTAMPTZ NOT NULL,
    signal_id       UUID NOT NULL DEFAULT uuid_generate_v4(),
    symbol          VARCHAR(20) NOT NULL,
    strategy        VARCHAR(50) NOT NULL,
    direction       SMALLINT NOT NULL,         -- -1 | 0 | 1
    p_win           NUMERIC(5, 4) NOT NULL,
    confidence      NUMERIC(5, 4),
    regime          VARCHAR(30),
    entry_price     NUMERIC(20, 8),
    stop_loss       NUMERIC(20, 8),
    take_profit     NUMERIC(20, 8),
    position_size   NUMERIC(10, 4),            -- fraction of portfolio
    metadata        JSONB,
    source_service  VARCHAR(30) DEFAULT 'strategy-orchestrator',
    PRIMARY KEY (time, signal_id)
);

SELECT create_hypertable('signals.trading_signals', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Positions
CREATE TABLE IF NOT EXISTS bot.positions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(5) NOT NULL,       -- long | short
    entry_price     NUMERIC(20, 8) NOT NULL,
    current_price   NUMERIC(20, 8),
    quantity        NUMERIC(20, 8) NOT NULL,
    notional_usd    NUMERIC(20, 2),
    unrealized_pnl  NUMERIC(20, 4),
    realized_pnl    NUMERIC(20, 4),
    pnl_pct         NUMERIC(8, 4),
    stop_loss       NUMERIC(20, 8),
    take_profit     NUMERIC(20, 8),
    strategy        VARCHAR(50),
    signal_id       UUID,
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    status          VARCHAR(20) NOT NULL DEFAULT 'open',  -- open | closed | liquidated
    close_reason    VARCHAR(50),               -- tp_hit | sl_hit | manual | signal_reversal
    exchange_order_id VARCHAR(100),
    metadata        JSONB
);

-- PnL history (hypertable)
CREATE TABLE IF NOT EXISTS bot.pnl_snapshots (
    time            TIMESTAMPTZ NOT NULL,
    portfolio_value NUMERIC(20, 4) NOT NULL,
    cash_balance    NUMERIC(20, 4),
    total_pnl       NUMERIC(20, 4),
    daily_pnl       NUMERIC(20, 4),
    drawdown        NUMERIC(8, 4),
    max_drawdown    NUMERIC(8, 4),
    sharpe_1m       NUMERIC(8, 4),
    win_rate_7d     NUMERIC(5, 4),
    open_positions  INTEGER DEFAULT 0,
    PRIMARY KEY (time)
);

SELECT create_hypertable('bot.pnl_snapshots', 'time',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- Strategies catalog
CREATE TABLE IF NOT EXISTS bot.strategies (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(50) NOT NULL UNIQUE,
    display_name    VARCHAR(100),
    description     TEXT,
    strategy_type   VARCHAR(30) NOT NULL,      -- trend | mean_reversion | arbitrage | ml | rl
    timeframe       VARCHAR(5) NOT NULL DEFAULT '1h',
    symbols         JSONB NOT NULL DEFAULT '[]',
    params          JSONB NOT NULL DEFAULT '{}',
    risk_params     JSONB NOT NULL DEFAULT '{}',
    is_active       BOOLEAN NOT NULL DEFAULT FALSE,
    is_paper        BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Bot configurations
CREATE TABLE IF NOT EXISTS bot.configurations (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(100) NOT NULL,
    description     TEXT,
    mode            VARCHAR(20) NOT NULL DEFAULT 'paper',  -- paper | live
    max_positions   INTEGER NOT NULL DEFAULT 3,
    max_leverage    NUMERIC(5, 2) NOT NULL DEFAULT 1.0,
    risk_per_trade  NUMERIC(5, 4) NOT NULL DEFAULT 0.02,   -- 2%
    max_drawdown    NUMERIC(5, 4) NOT NULL DEFAULT 0.10,   -- 10%
    total_capital   NUMERIC(20, 4),
    active_strategies JSONB NOT NULL DEFAULT '[]',
    exchange        VARCHAR(30) NOT NULL DEFAULT 'binance',
    kill_switch     BOOLEAN NOT NULL DEFAULT FALSE,
    is_active       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- 5. FEATURE STORE
-- =============================================================================

-- Feature definitions
CREATE TABLE IF NOT EXISTS features.feature_definitions (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL UNIQUE,
    description     TEXT,
    feature_group   VARCHAR(50) NOT NULL,      -- market | macro | onchain | derived
    dtype           VARCHAR(20) NOT NULL DEFAULT 'float64',
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Feature values (hypertable)
CREATE TABLE IF NOT EXISTS features.feature_values (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    feature_name    VARCHAR(100) NOT NULL,
    value           DOUBLE PRECISION,
    version         INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (time, symbol, feature_name)
);

SELECT create_hypertable('features.feature_values', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Feature snapshots (latest values per symbol — for low-latency serving)
CREATE TABLE IF NOT EXISTS features.feature_snapshots (
    symbol          VARCHAR(20) NOT NULL,
    features        JSONB NOT NULL DEFAULT '{}',
    feature_version INTEGER NOT NULL DEFAULT 1,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol)
);

-- =============================================================================
-- 6. REGIMES
-- =============================================================================

-- Market regimes (hypertable)
CREATE TABLE IF NOT EXISTS signals.market_regimes (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    regime_name     VARCHAR(30) NOT NULL,      -- bull_trend | bear_trend | high_vol | low_vol | ranging
    regime_id       INTEGER NOT NULL,
    probabilities   JSONB NOT NULL DEFAULT '{}',
    volatility      NUMERIC(10, 6),
    trend_strength  NUMERIC(8, 4),
    volume_ratio    NUMERIC(8, 4),
    confidence      NUMERIC(5, 4),
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('signals.market_regimes', 'time',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- =============================================================================
-- 7. AUDIT LOGS
-- =============================================================================

-- Event log (all Kafka events persisted)
CREATE TABLE IF NOT EXISTS audit.event_log (
    id              BIGSERIAL PRIMARY KEY,
    event_id        UUID NOT NULL,
    event_type      VARCHAR(50) NOT NULL,
    topic           VARCHAR(100) NOT NULL,
    source_service  VARCHAR(50),
    payload         JSONB NOT NULL,
    received_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_event_log_type ON audit.event_log (event_type);
CREATE INDEX IF NOT EXISTS idx_event_log_received ON audit.event_log (received_at DESC);

-- System health log (hypertable)
CREATE TABLE IF NOT EXISTS audit.service_health (
    time            TIMESTAMPTZ NOT NULL,
    service_name    VARCHAR(50) NOT NULL,
    status          VARCHAR(20) NOT NULL,      -- healthy | degraded | down
    latency_ms      NUMERIC(8, 2),
    error_count     INTEGER DEFAULT 0,
    metadata        JSONB,
    PRIMARY KEY (time, service_name)
);

SELECT create_hypertable('audit.service_health', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- =============================================================================
-- 8. INDEXES
-- =============================================================================

-- market.ohlcv
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf   ON market.ohlcv (symbol, timeframe, time DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time  ON market.ohlcv (symbol, time DESC);

-- market.orderbook_snapshots
CREATE INDEX IF NOT EXISTS idx_ob_symbol ON market.orderbook_snapshots (symbol, time DESC);

-- market.funding_rates
CREATE INDEX IF NOT EXISTS idx_funding_symbol ON market.funding_rates (symbol, time DESC);

-- signals.trading_signals
CREATE INDEX IF NOT EXISTS idx_signals_symbol     ON signals.trading_signals (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_signals_strategy   ON signals.trading_signals (strategy, time DESC);
CREATE INDEX IF NOT EXISTS idx_signals_direction  ON signals.trading_signals (direction, time DESC);

-- bot.positions
CREATE INDEX IF NOT EXISTS idx_positions_symbol   ON bot.positions (symbol, status);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON bot.positions (strategy, status);
CREATE INDEX IF NOT EXISTS idx_positions_status   ON bot.positions (status, opened_at DESC);

-- onchain.whale_transactions
CREATE INDEX IF NOT EXISTS idx_whale_asset  ON onchain.whale_transactions (asset, time DESC);
CREATE INDEX IF NOT EXISTS idx_whale_dir    ON onchain.whale_transactions (direction, time DESC);

-- features.feature_values
CREATE INDEX IF NOT EXISTS idx_feat_symbol ON features.feature_values (symbol, feature_name, time DESC);

-- =============================================================================
-- 9. COMPRESSION POLICIES (TimescaleDB)
-- =============================================================================

-- Compress older OHLCV data
SELECT add_compression_policy('market.ohlcv',
    compress_after => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- Compress older trades data (high volume)
SELECT add_compression_policy('market.trades',
    compress_after => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Compress feature values after 14 days
SELECT add_compression_policy('features.feature_values',
    compress_after => INTERVAL '14 days',
    if_not_exists => TRUE
);

-- =============================================================================
-- 10. SEED DATA
-- =============================================================================

-- Default assets
INSERT INTO market.assets (symbol, base_currency, quote_currency, asset_class, exchange)
VALUES
    ('BTCUSDT',  'BTC',  'USDT', 'crypto', 'binance'),
    ('ETHUSDT',  'ETH',  'USDT', 'crypto', 'binance'),
    ('SOLUSDT',  'SOL',  'USDT', 'crypto', 'binance'),
    ('BNBUSDT',  'BNB',  'USDT', 'crypto', 'binance'),
    ('AVAXUSDT', 'AVAX', 'USDT', 'crypto', 'binance'),
    ('ARBUSDT',  'ARB',  'USDT', 'crypto', 'binance'),
    ('OPUSDT',   'OP',   'USDT', 'crypto', 'binance')
ON CONFLICT (symbol) DO NOTHING;

-- Default strategies
INSERT INTO bot.strategies (name, display_name, description, strategy_type, timeframe, symbols, params, risk_params)
VALUES
    (
        'momentum_ml',
        'ML Momentum',
        'Trend-following strategy powered by XGBoost classifier on multi-timeframe features',
        'ml',
        '1h',
        '["BTCUSDT","ETHUSDT","SOLUSDT"]',
        '{"model": "xgboost", "lookback": 200, "min_p_win": 0.58}',
        '{"risk_per_trade": 0.02, "max_leverage": 2.0, "stop_atr_mult": 2.0}'
    ),
    (
        'mean_reversion_funding',
        'Funding Rate Mean Reversion',
        'Short when funding z-score > 2.5, long when < -2.5. Perps only.',
        'mean_reversion',
        '4h',
        '["BTCUSDT","ETHUSDT"]',
        '{"entry_z": 2.5, "exit_z": 0.5, "lookback_hours": 168}',
        '{"risk_per_trade": 0.015, "max_leverage": 1.5, "hard_stop_pct": 0.03}'
    ),
    (
        'regime_adaptive',
        'Regime Adaptive',
        'Adjusts exposure based on detected market regime and macro conditions',
        'ml',
        '1h',
        '["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT"]',
        '{"model": "deep_mlp", "regime_filter": true, "min_confidence": 0.6}',
        '{"risk_per_trade": 0.025, "max_leverage": 1.0, "recession_halt": true}'
    ),
    (
        'whale_follow',
        'Whale Smart Money',
        'Follow large on-chain outflows from exchanges as accumulation signals',
        'trend',
        '4h',
        '["BTCUSDT","ETHUSDT"]',
        '{"min_whale_usd": 5000000, "confirm_candles": 2, "direction_window_h": 6}',
        '{"risk_per_trade": 0.02, "max_leverage": 1.5, "stop_atr_mult": 3.0}'
    )
ON CONFLICT (name) DO NOTHING;

-- Default bot configuration (paper trading)
INSERT INTO bot.configurations (name, description, mode, max_positions, max_leverage, risk_per_trade, max_drawdown, total_capital, active_strategies)
VALUES (
    'Default Paper Config',
    'Conservative paper trading configuration for testing',
    'paper',
    3,
    1.5,
    0.02,
    0.10,
    10000.00,
    '["momentum_ml", "mean_reversion_funding"]'
) ON CONFLICT DO NOTHING;

-- Feature definitions
INSERT INTO features.feature_definitions (name, description, feature_group, dtype)
VALUES
    ('rsi_14',            'RSI 14 periods',              'market',  'float64'),
    ('rsi_7',             'RSI 7 periods',               'market',  'float64'),
    ('ema_9',             'EMA 9 periods',               'market',  'float64'),
    ('ema_21',            'EMA 21 periods',              'market',  'float64'),
    ('ema_50',            'EMA 50 periods',              'market',  'float64'),
    ('bb_pct',            'Bollinger %B',                'market',  'float64'),
    ('atr_14',            'ATR 14 periods',              'market',  'float64'),
    ('volume_ratio',      'Volume vs 20d MA',            'market',  'float64'),
    ('spread_bps',        'Order book spread bps',       'market',  'float64'),
    ('ob_imbalance',      'Order book imbalance',        'market',  'float64'),
    ('funding_z',         'Funding rate z-score',        'market',  'float64'),
    ('recession_prob',    'Recession probability',       'macro',   'float64'),
    ('yield_inv',         'Yield curve inversion flag',  'macro',   'float64'),
    ('sahm_value',        'Sahm rule value',             'macro',   'float64'),
    ('whale_sentiment',   'Whale net flow sentiment',    'onchain', 'float64'),
    ('whale_net_flow',    'Whale net flow USD',          'onchain', 'float64'),
    ('regime_id',         'Market regime cluster id',    'derived', 'int32'),
    ('p_win_ml',          'ML model P(win)',             'derived', 'float64'),
    ('p_win_bayesian',    'Bayesian P(win)',             'derived', 'float64')
ON CONFLICT (name) DO NOTHING;
