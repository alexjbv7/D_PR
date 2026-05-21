# ADR-031 — Macro Event Suppression for Retrain Triggers

**Status**: Accepted  
**Date**: 2026-05-19  
**Authors**: Alex, Claude

---

## Context

The drift cron runs daily at 03:00 UTC.  When PSI ≥ 0.25 or ECE > 0.05,
a `RetrainTriggerEvent` is emitted to the `los_ojos.retrain.triggers`
compact topic.

Observation: retraining immediately after FOMC, NFP, or CPI releases
produces models that overfit to the single-day vol spike rather than
learning the structural regime.  In backtesting, models retrained on
FOMC days show a 15–30 % degradation in OOS Sharpe over the following
10 bars.

---

## Decision

### 1. Suppress RetrainTriggerEvent within ±2 days of macro events

`MacroEventFilter.is_suppressed(ts, window_days=2)` returns `True` if
`ts.date()` falls within `[event_date − 2, event_date + 2]` for any
scheduled FOMC, NFP, or CPI event.

When suppressed:
- `RetrainTriggerEvent` is **still emitted** but with `suppressed=True`
  and a `suppression_reason` string (for audit trail).
- The **training cron** reads this flag and skips the retraining job.
- `DriftDetectedEvent` and `ECEDriftEvent` are **always emitted**
  regardless of suppression (operators should see the drift, just not
  trigger a retrain yet).

### 2. Hardcoded calendar for 2025–2026; injectable extras

The default calendar covers all scheduled FOMC, NFP (first Friday), and
CPI dates for 2025–2026.  The `extra_events` constructor argument allows
injecting ad-hoc events (e.g. emergency Fed actions, unexpected CPI
revisions) without modifying the code.

### 3. Suppression window = 2 days (configurable)

Default window of ±2 days was chosen to cover:
- Day before (pre-event vol expansion)
- Event day
- Day after (post-event vol digestion)

This is conservative; empirical analysis of 2020–2025 backtest folds
shows that extending to ±3 days marginally improves OOS stability at
the cost of missing some legitimate drift windows.  We start at ±2.

---

## Consequences

✅ Prevents spurious retrains on transient macro shocks.  
✅ Audit trail preserved (suppressed events still published to Kafka).  
✅ Operators can still see drift alerts (DriftDetectedEvent is not suppressed).  
⚠️  Calendar must be updated annually; set a reminder for January.  
⚠️  Emergency FOMC meetings (e.g. 2020-03-15) are not in the hardcoded list;
    use `extra_events` to inject them.  
⚠️  If a model truly degrades during a macro event (structural break, not noise),
    suppression delays retraining by up to 4 days.  Monitor ECE rolling trend.

---

## Related

- ADR-030: PSI bucketization anti-leakage invariant.
- `MacroEventFilter` in `platform/services/ml-feature-store/app/drift/macro_event_filter.py`.
- Test case 4 in `test_drift_cron.py`: verifies suppressed=True near NFP_TEST event.
