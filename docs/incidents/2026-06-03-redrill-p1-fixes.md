# Incident DRILL-004 — Re-drill P1-001 + P1-002 (Day 14)

**Severidad**: DRILL (sin impacto en capital)  
**Fecha**: 2026-06-03 / 2026-06-04 01:41 UTC  
**Detector**: Scheduled weekly review (W2) + decisión de cerrar P1-001/P1-002  
**Ejecutado por**: Claude (Cowork, sesión con usuario presente)  
**Duración**: < 5 min (simulación lógica)

---

## TL;DR

Re-drill de los gaps P1-001 (circuit breaker) y P1-002 (RiskGate kill switch)
identificados en DRILL-002 (2026-05-24) y DRILL-003 (2026-05-31).

**Resultado: ✅ 21/21 checks PASSED — ambos gaps cerrados.**

---

## Contexto

La semana W2 del paper trading run (2026-05-20 → 2026-06-19) reportó 0 trades
en 14 días. El weekly briefing identificó los P1 abiertos como causa probable
de que el kill switch pudiera estar activo y bloqueando la ejecución.

Implementados en esta sesión (2026-06-03):

| Fix | Archivo | Descripción |
|-----|---------|-------------|
| P1-001 circuit breaker | `app/brokers/_alpaca/circuit_breaker.py` | CLOSED→OPEN→HALF_OPEN, sliding window, Prometheus metrics |
| P1-001 alert rules | `platform/monitoring/rules/alpaca.yml` | ALERT-004/005/006/007/008 habilitadas |
| P1-002 RiskGate kill switch | `app/risk_gate.py` | `trip_kill_switch()`, `reset_kill_switch()`, check 0 en `_run_checks()` |
| P1-002 callback wiring | `app/main.py` | `_make_kill_switch_callback` y endpoint reset propagan a `RiskGate` |

---

## Escenarios del drill

### Scenario A — Circuit Breaker (P1-001)

| Check | Resultado | Detalle |
|-------|-----------|---------|
| A1 initial state CLOSED | ✅ PASS | — |
| A2 4 failures → still CLOSED | ✅ PASS | failure_times=4 |
| A3 5th failure → OPEN | ✅ PASS | is_open=True |
| A4 OPEN rejects calls con CircuitBreakerOpenError | ✅ PASS | mensaje incluye tiempo de recovery |
| A5 after recovery_timeout → probe allowed (HALF_OPEN) | ✅ PASS | — |
| A6 probe success → CLOSED | ✅ PASS | — |
| A7 probe failure → back to OPEN | ✅ PASS | HALF_OPEN probe failed → re-open |
| A8 old failures outside window not counted | ✅ PASS | 2+2 failures con window=0.05s |

**Subtotal A: 8/8**

### Scenario B — RiskGate Kill Switch (P1-002)

| Check | Resultado | Detalle |
|-------|-----------|---------|
| B1 kill switch inactive by default | ✅ PASS | — |
| B2 intent approved cuando switch inactive | ✅ PASS | breach=None |
| B3 `_kill_switch_active=True` after trip | ✅ PASS | — |
| B4 intent rechazado AFTER trip (breach=kill_switch) | ✅ PASS | approved=False |
| B5 REST-path intent bloqueado tras reconciler callback | ✅ PASS | **gap de DRILL-003 cerrado** |
| B6 trip idempotente | ✅ PASS | — |
| B7 `reset_kill_switch` → inactive | ✅ PASS | — |
| B8 intent aprobado after reset | ✅ PASS | breach=None |
| B9 kill_switch chequeado ANTES de daily_dd | ✅ PASS | breach=kill_switch (no daily_dd) |

**Subtotal B: 9/9**

### Scenario C — End-to-end integración

| Check | Resultado | Detalle |
|-------|-----------|---------|
| C1 circuit breaker OPEN after 3 failures | ✅ PASS | — |
| C2 kill switch triggered after circuit open | ✅ PASS | callback invocado |
| C3 `RiskGate._kill_switch_active=True` | ✅ PASS | — |
| C4 new intents bloqueados vía RiskGate (end-to-end) | ✅ PASS | breach=kill_switch |

**Subtotal C: 4/4**

---

## Resultado final

```
DRILL SUMMARY  21/21 checks passed  (0 failed)
DRILL RESULT: ✅ ALL CHECKS PASSED — P1-001 + P1-002 CLOSED
```

---

## Comparativa histórica de drills

| Componente | DRILL-002 (Day 5) | DRILL-003 (Day 12) | **DRILL-004 (Day 14)** |
|------------|-------------------|--------------------|------------------------|
| Circuit breaker | ❌ NO implementado | N/A | **✅ PASS** |
| Alert rules Prometheus | ❌ NO implementadas | ❌ NO implementadas | **✅ 5 reglas activas** |
| Reconciler logic (Kafka) | N/A | ✅ PASS | ✅ PASS |
| Kill switch (Kafka consumer) | ❌ NO implementado | ✅ PASS | ✅ PASS |
| Kill switch (RiskGate REST) | N/A | ❌ GAP (P1-002) | **✅ PASS** |
| End-to-end CB→KS→RiskGate | N/A | N/A | **✅ PASS** |

---

## Follow-ups abiertos

- [ ] **0 trades en 14 días** — investigar causa raíz (pipeline señal→executor, strategy-orchestrator conectado, kill switch activo en instancia real)
- [ ] **Re-drill en vivo** contra servicio corriendo (requiere Python 3.11 + stack up)
- [ ] **Commit** de todos los cambios de esta sesión con co-author Claude
- [ ] **S11** — DAG de entrenamiento nocturno (próxima semana del roadmap)

---

## Lecciones

1. Los drills automatizados como simulación lógica detectan gaps reales sin
   necesitar el stack en vivo — DRILL-003 encontró P1-002, DRILL-004 lo confirma cerrado.
2. El kill switch necesita ser un "gate universal": cualquier path que no pase
   por el consumer Kafka (REST, admin tools) debe pasar por `RiskGate.evaluate()`.
   El fix en `_make_kill_switch_callback` garantiza esto.
3. El circuit breaker con sliding window es más robusto que un contador simple:
   los fallos fuera de la ventana no impiden la recuperación.
4. La integración end-to-end (Scenario C) es el check más importante — verifica
   que el sistema como un todo responde correctamente, no solo sus partes aisladas.
