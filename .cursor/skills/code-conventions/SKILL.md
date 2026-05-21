---
name: code-conventions
description: >-
  PROJECT ML quant_bot naming (feat_/sig_/strat_), return type rules (log vs
  simple), and UTC timezone handling. Use when writing or reviewing Python/TS in
  research/, shared/, or platform/, or when the user invokes @code-conventions.
disable-model-invocation: true
---

# Code conventions — quant_bot

Static reference for agents. Source of truth for feature **names** in production:
`shared/quant_shared/features/definitions.py` (19 canonical features, fixed order).

## Naming

### Prefixes (research & backtest code)

| Prefix | Meaning | Type / values | Naming pattern | Examples |
|--------|---------|---------------|----------------|----------|
| *(semantic)* | ML / feature columns | `pd.DataFrame` columns, `float` | `{indicator}_{window}` or `{concept}_{horizon}` | `log_ret_5`, `rsi_14`, `regime_prob_0` |
| `feat_` | **Tests & placeholders only** | Synthetic columns | `feat_{i}` | `feat_0`, `feat_1` in `test_calibration.py` |
| `sig_` | Trade signal series | `pd.Series`, values ∈ **{-1, 0, 1}** | `sig_{scope}` | `sig_train`, `sig_test`, `sig_oos` |
| `strat_` | Strategy P&L / equity | `pd.Series` of **simple** period returns | `strat_{metric}` | `strat_ret`, `strat_cum` |

**Rules**

1. Do **not** rename canonical production features to `feat_*`. Online store and models depend on exact names in `FEATURE_NAMES` (`rsi_14`, `mom_1h`, …).
2. Regime outputs: `regime_prob_{k}`, `regime_label`, `regime_entropy` (see `research/features/regime_gmm.py`). PCA / feature selection must **exclude** `regime_*` from drops that treat them like noise (ADR-004).
3. Local DataFrame variables may use `feat_df`, `feat_agg` (aggregation), `feat_sel` (UI) — prefix describes role, not column names.
4. Signals: align index with `prices.index`; backtest applies `signals.shift(1)` before P&L (signal at `t` → position from `t+1`).
5. `strat_*` is only for **post-signal** return streams, never for raw price features.

### Layer naming (cross-language)

- Python: `snake_case` (`feature_set_hash`, `log_ret_1`)
- TypeScript: `camelCase` for fields (`featureSetHash`) when mirroring API payloads
- Kafka / shared schemas: match Python names in JSON payloads

---

## Returns: log vs simple

Use the right return definition per layer. Mixing them without conversion breaks Sharpe, vol, and cumprod.

### Decision table

| Use case | Return type | Formula | Notes |
|----------|-------------|---------|-------|
| Feature engineering, vol, GMM, labeling, slippage vol proxy | **Log** | `np.log(close / close.shift(p))` → columns `log_ret_{p}` | Default in `research/features/engineering.py` `log_returns()` |
| Walk-forward OOS diagnostic P&L | **Simple** | `prices.pct_change()` × signal | `walk_forward_runner.compute_oos_metrics` |
| Dashboard strategy curve | **Simple** | `price_ret = prices.pct_change()`; `strat_ret = sig * price_ret` | `research/dashboard/components/charts.py` |
| Equity / cum P&L | **Simple** | `(1 + strat_ret).cumprod()` | Never cumprod on log returns without `exp` |
| Shared momentum features (`mom_1h`, …) | **Simple** | `(close / close_n_ago) - 1` | Defined in `definitions.py` |
| Sim paths / GBM synthetic data | **Log** | `close = base * exp(cumsum(log_returns))` | `validation_demo*.py` |
| Backtest engine equity series | **Simple** | `equity.pct_change()` | Internal vol uses log for slippage scaling only |

### Invariants

```python
# Features & vol — log
log_ret_1 = np.log(close / close.shift(1))

# Strategy metrics & Sharpe on traded periods — simple
price_ret = close.pct_change()
strat_ret = (signals * price_ret).fillna(0.0)

# Annualization: set periods_per_year to bar frequency (252 daily, 8760 hourly, etc.)
```

- Annualized vol from log returns: `log_ret.std() * sqrt(periods_per_year)`.
- Do not feed `log_ret_*` columns into `(1+r).cumprod()` without `np.expm1(r)`.
- When joining simple and log series, **reindex to a common UTC index** first.

---

## Timezone (ADR-010)

**All timestamps are UTC, timezone-aware.** No naive `datetime` in events, orders, bars, or Kafka payloads.

### Python

```python
from datetime import datetime, timezone

UTC = timezone.utc

def utc_now() -> datetime:
    return datetime.now(tz=UTC)

# pandas index
df.index = pd.to_datetime(df.index, utc=True)
# or on ingest:
ts = pd.to_datetime(raw_ts, utc=True)
```

- Alpaca / bar ingest: `pd.to_datetime(..., utc=True)` then `DatetimeIndex` (`research/data/alpaca_bars.py`).
- Pydantic defaults: `datetime.now(tz=timezone.utc)` (`shared/quant_shared/schemas/orders.py`).
- ISO strings in APIs: include offset (`2026-05-18T14:00:00+00:00`) or suffix `Z`.
- If you receive naive timestamps, localize explicitly: `.tz_localize("UTC")` — never assume local machine TZ.

### TypeScript / frontend

- Serialize instants as ISO-8601 UTC.
- `Date` parsing: treat displayed times as UTC unless the UI layer documents a user locale.

### Forbidden

- `datetime.now()` without `tz=`
- `pd.Timestamp("2026-01-01")` without `tz="UTC"` on time-series indexes
- Stripping tz info before persistence or Kafka publish

---

## Agent checklist (before submitting code)

- [ ] Feature columns use semantic names; `feat_N` only in synthetic tests
- [ ] `sig_*` is {-1,0,1} and index-aligned; backtest shift(1) respected
- [ ] `strat_*` built from **simple** returns; cumprod only on simple
- [ ] Log returns only in feature/vol/label paths unless converting explicitly
- [ ] All new timestamps are UTC-aware; money in execution paths uses `Decimal` (ADR-010)
- [ ] Canonical 19-feature names unchanged unless versioned with migration note

## Canonical paths

| Topic | File |
|-------|------|
| 19 production features | `shared/quant_shared/features/definitions.py` |
| Research feature builder | `research/features/engineering.py` |
| OOS metrics (simple returns) | `research/models/walk_forward_runner.py` |
| Backtest engine | `research/backtesting/engine.py` |
| Bar UTC ingest | `research/data/alpaca_bars.py` |
| Order / time helpers | `shared/quant_shared/schemas/orders.py` |
| Repo-wide invariants | `CLAUDE.md` §19.9, ADR-010; `.cursor/BUGBOT.md` |
