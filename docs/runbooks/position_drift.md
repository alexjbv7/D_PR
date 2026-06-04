# Runbook: Position Drift (ALERT-004)

**Alert**: ALERT-004 — Reconciliation Discrepancy  
**Severity**: P0 when ≥ 3 consecutive cycles; P1 for 1–2 cycles  
**Owner**: execution-engine  
**Last reviewed**: 2026-05-31 (auto-generated from drill Day 12)

---

## What is position drift?

The reconciler (`platform/services/execution-engine/app/reconciler.py`) runs every
`reconciler_interval_sec` seconds (default 60 s).  It compares the execution-engine's
internal position state (Postgres / MemoryRepository) against the broker's reported
positions (Alpaca, CCXT).

Discrepancy types:

| Kind | Description |
|------|-------------|
| `PHANTOM` | Broker has a position; internal state does not |
| `MISSING` | Internal state has a position; broker does not |
| `QTY_MISMATCH` | Both sides have the position but qty differs beyond tolerance |
| `SIDE_MISMATCH` | Same symbol/venue but opposite side |

After `failure_threshold` (default **3**) consecutive cycles with discrepancies the
reconciler trips the kill switch via `kill_switch_callback`.

---

## Detection

**Automatic**: `Reconciler._handle_report()` logs at `WARNING` level per cycle.
After threshold, logs `CRITICAL reconciler.kill_switch_tripped`.

**Prometheus metric** (TODO): no dedicated counter yet — grep logs for
`reconciler.discrepancies` or `kill_switch.tripped`.

**Kafka**: `AnomalyEvent` published to `settings.kafka_anomaly_topic`
(`los_ojos.anomalies` by default) on every discrepant cycle.

**ALERT-004**: Alertmanager rule (TODO — not yet implemented as of 2026-05-31;
see P1-001 from Day 5 drill).

---

## Kill switch behaviour

When tripped:
- `state.kill_switch_tripped = True` on the `AppState` singleton.
- **Kafka consumer** stops processing new signals (`main.py:198`).
- `GET /health` returns `"kill_switch": true`.

⚠️ **Known gap (P1-002)**: `RiskGate.evaluate()` does NOT check
`kill_switch_tripped`.  A signal injected via REST API (not Kafka) still passes
through risk evaluation.  Fix: add a kill-switch check as step 0 in
`RiskGate._run_checks()`.

---

## Immediate response (human operator)

1. **Acknowledge the alert** and check logs:
   ```
   kubectl logs -n prod deployment/execution-engine | grep reconciler
   ```

2. **Identify discrepancy type** from the log line:
   ```
   reconciler.discrepancies count=1 streak=3
     [PHANTOM alpaca:AAPL] broker has qty=50 side=buy, internal has no position
   ```

3. **Do NOT auto-correct** — the reconciler intentionally does not auto-fix.
   Human review is required.

4. **Inspect broker positions** via API:
   ```
   curl http://execution-engine:8000/api/positions
   curl http://execution-engine:8000/api/account/alpaca
   ```

5. **Determine root cause**:
   - PHANTOM: order filled at broker but fill event never reached the engine?
     Check `fills` Kafka topic for missing messages.
   - MISSING: order cancel failed silently?  Check `orders.placed` topic.
   - QTY_MISMATCH: partial fill not accounted for?  Check fill records.

6. **Correct state** manually after confirming root cause.  Update the internal
   position record or reconcile via the broker's position as truth (document
   the decision in the incident log).

7. **Reset kill switch** once state is confirmed clean:
   ```
   curl -X POST http://execution-engine:8000/api/kill_switch/reset
   ```
   This also resets `reconciler._consecutive_failures` to 0.

8. **Verify recovery**: next reconcile cycle should log
   `reconciler.recovered after=N consecutive failures`.

---

## Drill procedure (synthetic injection)

Used for periodic drills.  The easiest approach (Opción A) is to inject a
phantom position on the **broker side**, i.e. a position the broker would
report but internal state does not know about.  Since we cannot inject into
Alpaca's paper account, this is simulated by running the reconciler logic
against a mock Router in the drill script (`tools/drills/drill_reconciliation.py`).

For a **live integration drill** on the dev environment:

1. Temporarily add a fake position to `MemoryRepository` (internal state) while
   leaving broker state clean → produces a `MISSING` discrepancy.
2. Use `POST /api/kill_switch/trip` to manually trip and test `RiskGate` response.
3. Verify `/health` shows `"kill_switch": true`.
4. Issue `POST /api/kill_switch/reset` and confirm recovery.

---

## References

- Reconciler implementation: `platform/services/execution-engine/app/reconciler.py`
- Kill switch endpoint: `POST /api/kill_switch/{trip|reset}` (main.py:402–416)
- Tests: `platform/services/execution-engine/tests/test_reconciler.py`
- Related incidents: `docs/incidents/2026-05-31-drill-reconciliation-drift.md`
- Open bugs: P1-001 (no Alertmanager rules), P1-002 (RiskGate kill-switch gap)
