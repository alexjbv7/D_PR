# ADR-025 — Cash Dividends: No Price Adjustment in `bars_1m_adjusted`

**Status**: Accepted  
**Date**: 2026-05-18  
**Deciders**: Alex, Claude Sonnet 4.6  
**Implements**: Semana 4 of `docs/architecture/alpaca_integration.md §11`

---

## Context

When a company pays a cash dividend, the stock price typically drops by
approximately the dividend amount on the ex-dividend date. In many financial
data providers (Yahoo Finance, Bloomberg) the "adjusted close" price series
**retroactively adjusts** pre-dividend prices downward to create a continuous
return series.

There are two valid approaches:

1. **Adjust for dividends** (common in research / buy-and-hold analysis):
   `adj_close_before = raw_close_before - dividend_per_share`

2. **Do not adjust for dividends** (common in short-term / signal-based
   strategies): let the price drop appear as-is in the adjusted series.

---

## Decision

**We do NOT adjust `bars_1m_adjusted` prices for cash dividends.**

The `CorporateActionEvent` with `ca_type="cash_dividend"` is stored in
`market.corporate_actions` and emitted to Kafka for P&L attribution
downstream, but the `BarsApplier` treats it as a no-op (returns 0 rows
adjusted, records the CA as applied).

---

## Rationale

### 1. Strategies react to the observable price drop

Our ML models are trained on raw price observations. On the ex-dividend date,
the price genuinely drops. A model trained on dividend-adjusted series would
learn a different (arguably wrong) response to that day's bar — it would
never see the drop. Since our strategies (swing, scalping) use real-time bars
to generate signals, the signal at open on the ex-date should reflect the true
observed price, not a retroactively adjusted one.

### 2. Dividend adjustment is asymmetric for short-term strategies

For strategies with a 1-minute to 4-hour horizon, the dividend drop is a
*signal* (potential mean-reversion opportunity, gap fill, etc.). Removing it
from the series eliminates a legitimate edge.

### 3. P&L attribution is handled separately

Cash dividends increase account equity when settled. This is recorded via
the `CorporateActionEvent` payload (`cash_amount`) and can be attributed to
P&L in the post-trade analytics layer without touching bar prices.

### 4. Industry precedent for shorter horizons

Providers such as Alpaca, Interactive Brokers' TWS, and most quantitative
frameworks (Zipline, bt, VectorBT) allow choosing whether to adjust for
dividends. For intraday and swing strategies, non-dividend-adjusted prices
are the standard.

---

## Consequences

### Positive

- Strategies see the actual historical price drop on ex-dividend dates.
- No retroactive re-writing of bars for cash dividends simplifies the pipeline.
- P&L attribution is explicit: cash dividends are tracked via CA events.

### Negative

- Long-term buy-and-hold backtests using `bars_1m_adjusted` will show
  artificial negative returns on ex-dividend dates. **This table is not
  intended for multi-year buy-and-hold analysis** — use a dividend-adjusted
  source (e.g. OpenBB with `adj_close=True`) for such research.
- Consumer engineers must be aware: `bars_1m_adjusted` adjusts for splits and
  stock dividends, but NOT for cash dividends.

---

## What IS adjusted

| CA Type         | Price adjusted | Volume adjusted | Notes                          |
|-----------------|:--------------:|:---------------:|--------------------------------|
| `forward_split` | ✅ / ratio     | ✅ × ratio      | ratio = split_to / split_from  |
| `reverse_split` | ✅ / ratio     | ✅ × ratio      | ratio < 1 → price increases    |
| `stock_dividend`| ✅ / ratio     | ✅ × ratio      | ratio = 1 + stock_amount       |
| `cash_dividend` | ❌             | ❌              | This ADR                       |
| `merger`        | ❌ (tagged)    | ❌              | `last_ca_id_applied` set       |
| `name_change`   | ❌ (tagged)    | ❌              | `last_ca_id_applied` set       |
| `spinoff`       | WARN (manual)  | WARN (manual)   | TODO ADR-2026-H2               |
