// ============================================================================
// Types — Los Ojos Trading Bot
// ============================================================================

export type SignalDirection = -1 | 0 | 1;
export type RegimeName =
  | "bull_low_vol"
  | "bull_high_vol"
  | "range_bound"
  | "bear_low_vol"
  | "bear_high_vol";

export type MacroRegime = "expansion" | "slowdown" | "recession" | "recovery";
export type RateEnv = "hiking" | "cutting" | "hold";

// ── Market Data ─────────────────────────────────────────────────────────────

export interface OHLCV {
  time: number;   // unix timestamp
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Tick {
  symbol:    string;
  price:     number;
  qty:       number;
  side:      "buy" | "sell";
  ts:        string;
}

// ── Signals ─────────────────────────────────────────────────────────────────

export interface TradingSignal {
  event_id:    string;
  strategy:    string;
  symbol:      string;
  timeframe:   string;
  direction:   SignalDirection;
  p_win:       number;
  kelly_fraction:  number;
  rr_ratio:        number;
  target_risk_pct: number;
  meta_filter_passed: boolean;
  bayesian_updated:   boolean;
  ts:          string;
}

export interface SignalDirection2 {
  label:   string;
  color:   string;
  icon:    string;
}

export const DIRECTION_META: Record<SignalDirection, SignalDirection2> = {
  1:  { label: "LONG",  color: "#22c55e", icon: "▲" },
  0:  { label: "FLAT",  color: "#94a3b8", icon: "─" },
  "-1": { label: "SHORT", color: "#ef4444", icon: "▼" },
};

// ── Regime ───────────────────────────────────────────────────────────────────

export interface RegimeState {
  symbol:      string;
  regime_label: number;
  regime_name:  RegimeName;
  regime_probs: number[];
  stability:   number;
  ts:          string;
}

export const REGIME_META: Record<RegimeName, { color: string; label: string }> = {
  bull_low_vol:  { color: "#22c55e", label: "Bull (Low Vol)" },
  bull_high_vol: { color: "#84cc16", label: "Bull (High Vol)" },
  range_bound:   { color: "#f59e0b", label: "Range Bound" },
  bear_low_vol:  { color: "#f97316", label: "Bear (Low Vol)" },
  bear_high_vol: { color: "#ef4444", label: "Bear (High Vol)" },
};

// ── Macro ────────────────────────────────────────────────────────────────────

export interface MacroSnapshot {
  recession_probability: number;
  regime:               MacroRegime;
  rate_environment:     RateEnv;
  yield_curve_inverted: boolean;
  sahm_value:           number | null;
  yield_inversion_days: number;
  p_yield_curve:        number;
  p_sahm:               number;
}

export interface FredSeries {
  value:    number;
  prior:    number | null;
  mom_pct:  number | null;
  z_score:  number;
  date:     string;
  name:     string;
  category: string;
}

// ── On-chain ─────────────────────────────────────────────────────────────────

export interface WhaleAlert {
  event_id:     string;
  blockchain:   string;
  tx_hash:      string;
  from_label:   string | null;
  to_label:     string | null;
  amount_usd:   number;
  token:        string;
  direction:    string;
  ts:           string;
}

export interface OnChainFlow {
  token:       string;
  net_flow:    number;
  accum_score: number;
  signal:      "accumulation" | "distribution" | "neutral";
}

// ── Funding ───────────────────────────────────────────────────────────────────

export interface FundingData {
  symbol:          string;
  funding_rate:    number;
  funding_z_score: number;
  basis_bps:       number;
  annual_funding:  number;
}

// ── Position / P&L ───────────────────────────────────────────────────────────

export interface Position {
  symbol:       string;
  strategy:     string;
  direction:    SignalDirection;
  entry_price:  number;
  current_price: number;
  qty:          number;
  pnl_usd:      number;
  pnl_pct:      number;
  sl_price:     number;
  tp_price:     number;
  opened_at:    string;
  regime:       string;
}

export interface PnLSummary {
  total_pnl_usd:    number;
  daily_pnl_usd:    number;
  weekly_pnl_usd:   number;
  sharpe_30d:       number;
  max_drawdown:     number;
  win_rate:         number;
  n_trades_30d:     number;
}

// ── Bot Config ────────────────────────────────────────────────────────────────

export interface BotConfig {
  model_class:        "xgboost" | "deep_mlp" | "lstm";
  strategy:           string;
  symbols:            string[];
  use_meta_labeling:  boolean;
  use_bayesian_sizing: boolean;
  use_regime_features: boolean;
  kelly_fraction:     number;
  max_risk_pct:       number;
  active:             boolean;
}

// ── WebSocket Events ──────────────────────────────────────────────────────────

export interface WSMessage {
  type:    string;
  channel: string;
  data:    unknown;
  ts:      string;
}

export type WSEventType =
  | "FinalSignalEvent"
  | "WhaleAlertEvent"
  | "SmartMoneyFlowEvent"
  | "MacroDataEvent"
  | "MacroRegimeEvent"
  | "RecessionAlertEvent"
  | "RegimeUpdateEvent"
  | "AnomalyEvent"
  | "KillSwitchEvent"
  | "ExecutionResult";

// ============================================================================
// SEC Research — REST shapes
// Mirrors platform/services/sec-research/app/main.py endpoints.
// ============================================================================

export type FilingSentimentLabel = "positive" | "negative" | "neutral" | "mixed";

export interface FilingSignal {
  event_id:         string;
  event_type:       string;   // "SECFilingSignal"
  ts:               string;
  ticker:           string;
  form_type:        string;   // "8-K" | "10-K" | "10-Q" | "13F"
  sentiment:        FilingSentimentLabel;
  score:            number;   // -1 to +1
  confidence:       number;   // 0 to 1
  signals:          string[];
  is_market_moving: boolean;
  summary:          string;
  regulators:       string[];
  crypto_assets:    string[];
  amounts_usd:      number[];
}

export interface InstitutionalPosition {
  institution: string;
  ticker:      string;   // IBIT, FBTC, COIN, etc.
  shares:      number;
  value_usd:   number;
  period:      string;   // "2025-Q1"
  change_pct:  number;
}

export interface SECHealth {
  status:  string;
  service: string;
  ts:      string;
}

export interface RegulatorySignalsResponse {
  signals:                   FilingSignal[];
  count:                     number;
  has_active_regulatory_risk: boolean;
}

export interface InstitutionalPositionsResponse {
  positions: InstitutionalPosition[];
  count:     number;
  ts:        string;
}

// ============================================================================
// Execution Engine — REST shapes
// Mirrors platform/services/execution-engine/app/main.py endpoints.
// All numeric monetary values come over the wire as strings (Decimal → str)
// per quant_shared.schemas.orders convention.
// ============================================================================

export type OrderSideStr   = "buy" | "sell";
export type OrderStatusStr =
  | "pending"
  | "submitted"
  | "partial"
  | "filled"
  | "cancelled"
  | "rejected"
  | "expired";

export interface ExecutionCounters {
  signals_seen:  number;
  intents_built: number;
  approved:      number;
  rejected:      number;
  submitted:     number;
  submit_errors: number;
}

export interface ExecutionHealth {
  status:       string;
  service:      string;
  ts:           string;
  venues:       string[];
  kill_switch:  boolean;
  counters:     ExecutionCounters;
}

export interface ExecutionPosition {
  symbol:          string;
  side:            OrderSideStr;
  qty:             string;            // Decimal as string
  avg_entry:       string;
  current_price:   string | null;
  unrealized_pnl:  string | null;
  margin_used:     string | null;
  venue:           string;
  ts_opened:       string | null;
  ts_updated:      string;
}

export interface ExecutionOrderResult {
  result_id:      string;
  intent_id:      string;
  broker_id:      string;
  symbol:         string;
  side:           OrderSideStr;
  status:         OrderStatusStr;
  qty:            string;
  filled_qty:     string;
  avg_price:      string | null;
  fills:          unknown[];
  reject_reason:  string | null;
  ts_submitted:   string;
  ts_updated:     string;
  venue:          string;
}

export interface ExecutionAccount {
  venue:        string;
  account_id:   string;
  equity:       string;
  cash:         string;
  margin_used:  string;
  pnl_day:      string;
  currency:     string;
  is_paper:     boolean;
}
