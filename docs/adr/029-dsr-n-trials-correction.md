# ADR-029 — DSR N-Trials Cross-Horizon Correction

**Status:** Accepted  
**Date:** 2026-05-18  
**Authors:** Alex, Claude Sonnet 4.6

---

## Context

The Deflated Sharpe Ratio (Bailey & López de Prado, 2014) corrects for multiple testing by
estimating the expected maximum SR when `N` independent strategies are evaluated:

```
E[max SR | N] ≈ √(2·ln N) · (1 - γ_EM/(2·ln N) - ln(ln N)/(2·ln N))
```

where `γ_EM ≈ 0.5772` (Euler-Mascheroni constant).

## The Problem

In Semana 7 we run Optuna with 50 trials **per horizon** across 3 horizons.  
Total hypotheses evaluated: **150**.

If each horizon computes DSR with `n_trials=50`, the benchmark SR is too low,
and DSR values appear inflated. The correct correction accounts for all 150 trials.

## Decision

`deflated_sharpe_ratio(returns, n_trials=TOTAL_OPTUNA_TRIALS)` where:

```python
TOTAL_OPTUNA_TRIALS = sum(h.n_optuna_trials for h in ALL_HORIZONS)  # = 150
```

This constant lives in `research/models/multi_horizon/horizon_config.py` and is
imported by `MultiHorizonTrainer._compute_metrics`.

## Rationale

- The 150 trials are not truly independent (they share the same market regime),
  making this a conservative correction. Being conservative prevents false promotions.
- If `n_optuna_trials` is reduced to 25 (time budget override), `TOTAL_OPTUNA_TRIALS`
  automatically adjusts to 75 — no manual update needed.
- PSR uses `n_trials=1` (no multiple-testing correction) as a raw signal quality measure.
  DSR is the promotion gate.

## Consequences

- DSR threshold remains at 0.4 (as defined in `ModelCard.is_production_ready`).
- Test `test_dsr_n_trials_correction.py` verifies monotonicity and the 150-value.
- Any change to `n_optuna_trials` per horizon automatically propagates to the correction.
