# ADR-030 — PSI Bucketization Strategy

**Status**: Accepted  
**Date**: 2026-05-19  
**Authors**: Alex, Claude

---

## Context

Population Stability Index (PSI) requires dividing a continuous feature
distribution into discrete buckets.  Two strategies exist:

1. **Fixed-width buckets**: equal-width intervals over the feature range.
2. **Quantile buckets**: equal-probability intervals derived from training data.

Additionally, a critical **anti-leakage** concern arises: if bucket edges
are recomputed from recent data, PSI will always be ≈ 0 (the recent data
fills each bucket proportionally by construction — a tautology that
suppresses all PSI signal).

---

## Decision

### 1. Quantile buckets from training data

We use **N=10 quantile buckets** derived from the training distribution
via `np.percentile(train_values, linspace(0,100,11))`.

Rationale:
- Quantile buckets are insensitive to scale and outliers.
- Equal-width buckets are dominated by tails; important regime shifts in
  the body of the distribution would be missed.
- N=10 is the industry standard for PSI (Yurdakul 2018); sufficient
  resolution for the 200–5000 sample ranges we expect.

### 2. Edges persisted in Redis with 30-day TTL

Bucket edges are computed **once** at model-promotion time and stored in
Redis under the key `drift:bucket_edges:{model_version}:{feature_name}`.

- TTL = 30 days (matches the warm-tier retention window).
- If a key expires before a cron run (unusual), the cron performs a
  cold-start: it recomputes edges from the historical train proxy
  (fetched from TimescaleDB) and re-caches.
- At model promotion time, the promoting workflow MUST call
  `BucketEdgesRepository.save(model_version, feature_name, edges)` for
  all top-20 features.

### 3. Anti-leakage invariant enforced by code contract

`compute_psi` in `research/models/drift/psi.py` accepts an optional
`bucket_edges` parameter.  In **production** the cron always passes
pre-computed edges loaded from Redis.  Computing from scratch (no edges
parameter) is restricted to **research / exploratory** use.

This invariant is documented in the module docstring, enforced by the
parameter design, and verified by test case 3 in `test_psi.py`.

---

## Epsilon guard

Empty buckets (a feature cluster absent from one distribution) are
handled with `eps = 1e-6`:

```
p_i = (count_i + eps) / (total + eps * N)
```

This prevents `log(0)` and `0/0` while keeping PSI numerically stable.
Verified by test case 6 in `test_psi.py`.

---

## Thresholds

| PSI value | Tier     | Action                        |
|-----------|----------|-------------------------------|
| < 0.10    | stable   | No action                     |
| 0.10–0.25 | moderate | P2 alert (DriftDetectedEvent) |
| ≥ 0.25    | severe   | P1 alert + RetrainTriggerEvent|

Source: Yurdakul (2018), Table 2; industry convention.

---

## Consequences

✅ Anti-leakage: bucket edges never adapt to recent data.  
✅ Robust to outliers (quantile buckets).  
✅ Cold-start handled transparently in drift_cron.  
⚠️  Model promotion workflow must be updated to persist bucket edges.  
⚠️  If Redis is unavailable at cron time, edges are recomputed per run  
    (cold-start path) — PSI is still valid but slightly less efficient.
