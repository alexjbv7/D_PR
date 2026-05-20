# ADR-032 — Allocator Thompson Priors (Beta(20, 20))

**Status**: Accepted
**Date**: 2026-05-19
**Authors**: Alex, Claude

---

## Context

The S9 allocator (`platform/services/strategy-orchestrator/app/allocator/`)
uses Thompson sampling over Beta(α, β) posteriors — one per horizon
(intraday / swing / daily) — to decide which horizon executes when more
than one confirmed signal coincides on the same `(symbol, direction)`.

We need to pick:

1. The **warm-start prior** `Beta(α₀, β₀)`.
2. The **decay factor** applied to historical evidence each day.
3. The **floor** for α and β under aggressive decay.

A naïve uninformative prior Beta(1, 1) was rejected because (a) the
cold-start sample distribution is uniform on [0, 1] — extreme picks
dominate early decisions, and (b) the variance is large
(≈ 8.3 × 10⁻²), so Thompson samples swing wildly until enough trades
accumulate (typically 50–100).

---

## Decision

### 1. Warm-start prior: **Beta(20, 20)**

| Parameter | Value | Reason |
|-----------|-------|--------|
| α₀         | 20.0  | Pseudo-wins of a "neutral expert" |
| β₀         | 20.0  | Pseudo-losses, symmetric → mean=0.5 |
| mean       | 0.5   | No bias toward any horizon |
| variance   | ≈ 6.1 × 10⁻³ | Tight enough that early Thompson samples are not extreme |

Rationale:
* Mean = 0.5 — neutral. The allocator does not prefer any horizon at boot.
* Variance ≈ 6.1 × 10⁻³ → standard deviation ≈ 0.078. The 95 % CI of
  the cold-start sample is [0.34, 0.66], avoiding the [0.05, 0.95]
  spread of Beta(1, 1).
* Beta(20, 20) is equivalent to having seen **40 prior trades** in a
  vacuum. With 0.99/day decay, this prior persists for ~40 days
  before getting drowned out by real evidence — long enough to
  prevent thrashing during warm-up, short enough that real signal
  dominates within a quarter.

### 2. Decay factor: **0.99 per day**

| Property | Value |
|----------|-------|
| Half-life | ≈ 69 days (ln(0.5) / ln(0.99)) |
| Weight at 30 days | ≈ 0.74 |
| Weight at 60 days | ≈ 0.55 |
| Weight at 90 days | ≈ 0.40 |

Matches CLAUDE.md §4.10 / §13.3 "ventana 90d decayed". Old evidence
is not deleted — it fades. A win 100 days ago contributes
0.99¹⁰⁰ ≈ 0.366 to the current α; a win today contributes 1.0.

### 3. Floor: **Beta(1, 1)**

Under aggressive decay (e.g. a horizon that goes inactive for 6
months), α and β both drift toward 0. We floor them at 1.0 to
prevent:

* `numpy.random.Generator.beta(0, 0)` → NaN.
* Division by zero in `mean` and `variance`.
* Posterior collapse where a single new trade dominates entirely
  (sample becomes effectively deterministic).

The floor is **per-parameter**, not on the sum. A horizon at
Beta(1, 1) still produces uniformly-distributed samples — fully
exploratory.

### 4. Lazy decay (read-time, not write-time)

Decay is applied **on every read** (sample, mean, variance) — not
when persisting. This avoids the "posterior aged between writes" bug
where, if decay is applied only at save time, a posterior persisted
at T then read at T+30d would not include those 30 days of decay
unless explicitly refreshed.

The cost is small: each `decayed_to(ts)` is an O(1) Decimal exponent.
Hot-path measurements (`test_choose_p99_under_5ms`) show ~70 µs
median for the full `choose()` call including 3 lazy-decay
computations.

---

## Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| Beta(1, 1)   | Too uncertain at cold start; extreme samples dominate. |
| Beta(5, 5)   | Variance ≈ 2.3 × 10⁻² — still too noisy. |
| Beta(50, 50) | Variance ≈ 2.5 × 10⁻³ — too rigid; real evidence takes ~6 months to dominate. |
| Beta(2, 1) "optimistic" | Asymmetric mean = 0.67 introduces bias — UCB-style exploration without justification. |
| Hard reset on retrain | Throws away all evidence; punishes models that already proved themselves. |
| No decay | Static posteriors do not adapt to regime changes. |

---

## Consequences

✅ Cold start is well-behaved: tested by `test_cold_start_distribution`
   (mean ≈ 0.5, variance in [0.003, 0.012]).

✅ A real edge is captured within ~100 trades: tested by
   `test_allocator_concentrates_with_edge` (swing share > 60 %
   when win rate is 65 % vs 50 %).

✅ Without edge, picks are exploratory: tested by
   `test_allocator_explores_no_edge` (entropy > 0.85 bits).

⚠️  If a horizon is deprecated and stops trading, its posterior
   slowly drifts toward Beta(1, 1) over months. That is desired —
   re-introduction starts fresh exploration.

⚠️  Decay is applied per **day**, not per trade. Two losing days
   for a normally-good horizon don't immediately tank it — they
   first need to accumulate in the β.
