# ADR-028 — Multi-Horizon Training Configuration

**Status:** Accepted  
**Date:** 2026-05-18  
**Authors:** Alex, Claude Sonnet 4.6

---

## Context

Semana 7 introduces simultaneous training of three ML models across different trading horizons.
Each horizon has fundamentally different label dynamics, data density, and feature availability.
Decisions must be codified as immutable constants to prevent accidental drift across experiments.

## Decisions

### Horizons and bar sizes

| Horizon  | Bar  | Rationale |
|----------|------|-----------|
| Intraday | 5min | Microstructure signal decays within one RTH session |
| Swing    | 4h   | Aligns with institutional session-level positions |
| Daily    | 1d   | Macro factor exposures meaningful only at EOD |

### Triple-barrier parameters

| Horizon  | TP%   | SL%   | Timeout  | Rationale |
|----------|-------|-------|----------|-----------|
| Intraday | 0.5%  | 0.5%  | 3 bars   | Tight barriers prevent holding through microstructure reversals |
| Swing    | 2.0%  | 2.0%  | 30 bars  | Matches ATR×2 stop used in swing strategy (§4.2) |
| Daily    | 4.0%  | 3.0%  | 20 bars  | Asymmetric: higher TP reflects positive drift expectation |

### Embargo (anti-leakage critical)

Embargo must satisfy: `embargo >= max(bar_resolution, label_horizon_duration)`

| Horizon  | Embargo | Bar timeout duration | Why sufficient |
|----------|---------|----------------------|----------------|
| Intraday | 2h      | 15 min (3 × 5min)    | 2h >> 15min; blocks all label overlap |
| Swing    | 24h     | 120h (30 × 4h)       | 1 bar ≥ bar resolution; prevents adjacent-fold contamination |
| Daily    | 5 days  | 20 days              | 5 trading days = minimum meaningful gap |

**CRITICAL:** Never apply the intraday embargo (2h) to the swing horizon. A swing label at
`t=Monday` with `timeout=5 trading days` looks forward to Friday. If the test starts at
`t + 2h`, the label and the test **overlap**, inflating DSR by 2-3×.

### Model selection

| Horizon  | Model | Rationale |
|----------|-------|-----------|
| Intraday | XGBoost | Sparse tabular, CPU-friendly, fast iteration |
| Swing    | XGBoost | Established pattern from §4.2; proven on similar data |
| Daily    | DeepMLP | Factor exposures benefit from non-linearities; ~750 bars/year/symbol |

### Train lookback

| Horizon  | Lookback | Rationale |
|----------|----------|-----------|
| Intraday | 12 months | Microstructure regime changes frequently; recency matters |
| Swing    | 24 months | Needs 2 full market cycles for regime diversity |
| Daily    | 36 months | FF5 rolling 60d OLS needs sufficient historical breadth |

## Consequences

- `HorizonConfig` is `frozen=True` — immutable after definition.
- Any change requires updating this ADR and a PR review.
- `TOTAL_OPTUNA_TRIALS = 150` (3 × 50) must be passed to `deflated_sharpe_ratio` — see ADR-029.
