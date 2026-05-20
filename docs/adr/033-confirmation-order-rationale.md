# ADR-033 — Multi-Factor Confirmation Order

**Status**: Accepted
**Date**: 2026-05-19
**Authors**: Alex, Claude

---

## Context

The S9 confirmation gate (`platform/services/strategy-orchestrator/app/confirmation/`)
sits between the raw signal stream and the Thompson allocator. It runs
four checks and only forwards a signal when **all** pass.

Two design choices needed an explicit decision:

1. **Aggregation**: AND vs weighted vote.
2. **Order**: which factor runs first, second, etc.

---

## Decision

### 1. AND aggregation (all factors must pass)

A signal is forwarded only if every factor returns `True`. A weighted
vote (e.g. "regime fails but meta-label is very strong, so still execute")
is explicitly rejected because:

* Risk-control failures (e.g. regime breakdown) should not be
  compensated for by model confidence. CLAUDE.md §12.7 ("kill switch")
  is structurally the same logic — any single critical failure halts
  execution.
* Calibrating per-factor weights requires offline OOS optimisation,
  which would tune confirmation against backtest outcomes — a leakage
  risk if the factors share signals with the primary model.
* Operational debugging is simpler: a rejection has a single,
  identifiable cause (`rejected_by`).

### 2. Order: cheapest first, dollars-saved-by-skipping

| Order | Factor | Cost (typical) | Cost (worst) |
|-------|--------|----------------|--------------|
| 1 | `has_primary_direction(signal)` | O(1) in-process | O(1) |
| 2 | `regime_stable(regime)` | O(1) Redis GET (cached) | ~1 ms Redis miss |
| 3 | `macro_coherent(direction, symbol_is_crypto, macro)` | O(1) Redis GET | ~1 ms |
| 4 | `meta_label_confident(p_correct)` | ~10 ms model inference | ~30 ms |

Rationale:

* **Why direction first**: a flat signal must never trigger any
  downstream work. Free filter.
* **Why regime before macro**: both are Redis lookups, so the
  ordering between them is essentially arbitrary, but
  regime is per-symbol (more likely to be unstable on a
  particular symbol than the macro state) so we expect it to
  reject more often per call.
* **Why meta-label last**: it is 10–100× more expensive than the
  other three and represents the heaviest compute. If we evaluate
  it first, the average cost of confirmation jumps from < 1 ms to
  > 10 ms even when the signal is rejected by regime instability.
  Across thousands of signals per day, the savings compound.

### 3. Suppression semantics

Each factor returns `(passed: bool, rejected_by: str | None)`. The
first failure short-circuits subsequent factors. The
`rejected_by` string is the **canonical** reason and is the only
value emitted in metrics and in the `AllocatorDecisionEvent`'s
`rejected_by` field — downstream consumers can rely on it for
attribution.

---

## Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| Weighted vote with learned weights | Requires offline calibration; leakage risk; harder to debug. |
| Parallel evaluation (asyncio.gather) | Wastes the expensive meta-label call when an earlier factor fails. Saves zero latency in the rejection case. |
| Meta-label first | Average confirmation cost balloons by ~10×; defeats the latency budget. |
| Two-stage pipeline (cheap factors as a Kafka topic, expensive on a separate consumer) | Adds 1 hop and serialisation overhead for negligible parallelism benefit at our QPS. |

---

## Consequences

✅ Confirmation latency is dominated by the lookup, not the inference,
   when signals are rejected — which is the common case.

✅ The `rejected_by` field is a single source of truth for attribution.

✅ Each factor is independently unit-testable
   (`confirmation/factors.py` is pure).

✅ The early-reject order is enforced by code, not by convention:
   tests `test_early_reject_regime_unstable` and
   `test_early_reject_macro_incoherent` verify the meta-labeler
   spy is never called when an earlier factor fails.

⚠️  If a factor's cost profile changes (e.g. regime moves to a remote
   feature store), this ADR must be revisited and the order possibly
   adjusted.

⚠️  A signal rejected by an early factor will not produce a
   meta-label score for downstream observability. That is by
   design — we are not paying for unused inference.
