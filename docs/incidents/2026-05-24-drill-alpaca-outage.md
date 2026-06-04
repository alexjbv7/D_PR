# DRILL — Alpaca Outage Simulation (Día 5)

**Tipo**: Drill planificado (no incident real)  
**Severidad**: N/A — pero identificados **P1 bugs** que bloquean el drill  
**Fecha**: 2026-05-24  
**Detector**: Scheduled task automatizado (Cowork, S10 Parte 2)  
**Duración del outage simulado**: N/A — drill no pudo ejecutarse  
**Resultado**: ❌ DRILL FAILED — defenses no implementadas

---

## TL;DR

El drill del día 5 (run 2026-05-20 → 2026-06-19) no pudo ejecutarse.
La auditoría estática del código reveló que el circuit breaker cuyo comportamiento
se pretendía validar **no existe en el codebase**. Tampoco existen las Prometheus
alert rules que dispararían ALERT-002. El runbook de referencia tampoco estaba
creado. Se documentan 3 P1s que deben cerrarse antes de continuar el run
con exposición operativa real.

---

## Contexto del drill

**Objetivo original**: simular un outage de Alpaca paper API (docker pause o
firewall block a `paper-api.alpaca.markets`) y verificar:
1. Circuit breaker detecta > 5 errores 5xx/60 s → CLOSED → OPEN
2. Alertmanager dispara ALERT-002 P0
3. RiskGate rechaza nuevas intents con `CircuitOpenError`
4. Reconciler continúa sin marcar discrepancias falsas
5. Recovery: OPEN → HALF_OPEN → CLOSED tras TTL + request exitosa

**Resultado real**: ninguno de los puntos 1–2 es verificable porque la
infraestructura no existe.

---

## Línea de tiempo (UTC — 2026-05-24)

| Hora (UTC) | Evento |
|------------|--------|
| ~14:50 | Scheduled task activado. Inicio del drill. |
| ~14:52 | Auditoría de structure: `docs/runbooks/` no existe. |
| ~14:53 | Grep exhaustivo: cero matches para `CircuitBreaker`, `circuit_state`, `HALF_OPEN`, `CircuitOpenError` en todo el repo. |
| ~14:54 | Verificación `prometheus.yml`: `rule_files: # - "rules/*.yml"` comentado. No hay directorio `rules/`. |
| ~14:55 | Verificación Docker: sandbox sin acceso a Docker daemon — live simulation no posible en este entorno de ejecución. |
| ~14:56 | Conclusión: drill bloqueado por 3 P1s. Procedo a documentar y generar runbook. |
| ~14:58 | `docs/runbooks/alpaca_outage.md` creado (acción autónoma — usuario no presente). |
| ~14:59 | Este incident report generado. |

---

## Root cause (de los gaps)

La memoria del proyecto registra "S2 (circuit breaker + 8 errores tipados)" como
completado antes de la sesión del 2026-05-18. Sin embargo, solo la mitad de S2
llegó al codebase:

- ✅ **8 errores tipados**: `platform/services/execution-engine/app/brokers/alpaca_errors.py` — implementado y con tests.
- ❌ **Circuit breaker**: no hay ningún archivo con `CircuitBreaker`, `circuit_state`, `HALF_OPEN` o equivalente. La funcionalidad nunca se implementó o fue excluida del commit.

El docstring de `strategy-orchestrator/app/main.py` menciona "circuit breakers"
como feature planificada (línea 8), confirmando que la intención existía pero
nunca materializó en código.

---

## Impacto operativo

**En paper trading (estado actual)**:
- Trades afectados: 0 (bot flat, sin posiciones abiertas)
- P&L impacto: $0
- Riesgo real: bajo en paper. El retry (3 intentos tenacity) es la única
  defensa ante errores de Alpaca.

**Si esto fuera live**:
- Outage de Alpaca → execution-engine reintenta 3x con backoff → lanza excepción
  no capturada → **comportamiento indefinido** (crash del servicio, pérdida de
  intents sin log, o intents acumuladas para retry posterior)
- No hay notificación automática: ALERT-002 no puede dispararse
- No hay RiskGate check de circuit: intents seguirían llegando al adapter durante el outage

---

## P1 Bugs identificados — a corregir antes de continuar

### P1-001: Circuit breaker no implementado

**Componente**: `platform/services/execution-engine/app/brokers/`  
**Descripción**: No existe ninguna clase `CircuitBreaker` ni módulo `circuit_breaker.py`.
La defensa ante outages de Alpaca se limita a 3 reintentos con backoff.
Tras los 3 reintentos, `AlpacaAdapter.submit()` propaga la excepción sin estado
persistente. Si el execution-engine sigue recibiendo intents, seguirá intentando
llamar a Alpaca (y fallando) sin modo `read_only` ni rechazo inmediato.

**Fix requerido**:
- Implementar `_alpaca/circuit_breaker.py` con estados `CLOSED`, `OPEN`, `HALF_OPEN`
- TTL configurable (default 5 min en OPEN)
- Thresholds: > 5 errores 5xx en 60 s → OPEN
- Métrica Prometheus: `alpaca_circuit_state{state}` (gauge)
- `AlpacaAdapter.submit()` debe lanzar `CircuitOpenError` (subclase de `BrokerError`) cuando `state == OPEN`
- `RiskGate` debe verificar estado del circuit y retornar `breach="circuit_open"` antes de aprobar intents

**Tests mínimos**:
- `test_circuit_breaker.py`: transiciones CLOSED→OPEN→HALF_OPEN→CLOSED
- `test_risk_gate.py`: nuevo caso `circuit_open` en checks
- `test_alpaca_adapter.py`: submit lanza `CircuitOpenError` cuando state=OPEN

**Estimación**: 4–6h (Cursor Composer o Claude Code Sonnet)

---

### P1-002: Prometheus alert rules no existen

**Componente**: `platform/monitoring/`  
**Descripción**: `prometheus.yml` tiene `rule_files: # - "rules/*.yml"` comentado.
No existe el directorio `platform/monitoring/rules/`. Por tanto:
- ALERT-002 (circuit open) no puede dispararse
- ALERT-001 (daily DD kill) no puede dispararse
- ALERT-004 (reconciler discrepancies) no puede dispararse
- Alertmanager no tiene nada que evaluar

**Fix requerido**:
- Crear `platform/monitoring/rules/trading_alerts.yml` con:
  ```yaml
  groups:
    - name: trading
      rules:
        - alert: ALERT-001-DailyDDKill
          expr: risk_daily_drawdown_pct < -0.03
          for: 1m
          labels:
            severity: P0
          annotations:
            summary: "Daily drawdown kill threshold breached"

        - alert: ALERT-002-AlpacaCircuitOpen
          expr: alpaca_circuit_state{state="open"} == 1
          for: 30s
          labels:
            severity: P0
          annotations:
            summary: "Alpaca circuit breaker OPEN — no orders will be submitted"

        - alert: ALERT-004-ReconcilerDiscrepancy
          expr: reconciler_discrepancies_total > 0
          for: 2m
          labels:
            severity: P0
          annotations:
            summary: "Reconciler detected position mismatch"
  ```
- Descomentar `rule_files: - "rules/*.yml"` en `prometheus.yml`

**Estimación**: 1–2h

---

### P1-003: Runbook `docs/runbooks/alpaca_outage.md` no existía

**Status**: ✅ **Corregido en este drill** (acción autónoma — usuario no presente).  
El runbook ha sido creado en `docs/runbooks/alpaca_outage.md` con instrucciones
para diagnóstico, remediación y recovery. Incluye nota explícita del gap del
circuit breaker para que el operador lo complete antes del siguiente drill.

---

## Lo que funciona como esperado ✅

| Defensa | Estado | Evidencia |
|---------|--------|-----------|
| `retry_with_jitter` (3x, tenacity) | ✅ Implementado | `brokers/_alpaca/retry.py` — unit-tested |
| 8 errores tipados Alpaca | ✅ Implementado | `brokers/alpaca_errors.py` — 8 clases + mapping + tests |
| Reconciler 60s loop | ✅ Implementado | `app/reconciler.py` |
| RiskGate: DD kill, market_open, exposure caps | ✅ Implementado | `app/risk_gate.py` + tests |
| `alpaca_submit_attempts_total{result}` | ✅ Prometheus counter | `brokers/alpaca.py:75` |
| `alpaca_submit_latency_seconds` | ✅ Prometheus histogram | `brokers/alpaca.py` |
| Briefing diario (bot flat, 0 discrepancias) | ✅ | `tools/briefing/output/2026-05-23.md` |

---

## Gaps detectados ❌

| # | Gap | Severidad | ETA fix |
|---|-----|-----------|---------|
| P1-001 | Circuit breaker no implementado | P1 | Antes de S11 |
| P1-002 | Prometheus alert rules no existen | P1 | Antes de S11 |
| P1-003 | Runbook alpaca_outage.md | P1 | ✅ Corregido |

---

## Acciones inmediatas requeridas

- [ ] **Operador**: revisar este report y confirmar severidad de los P1s.
- [ ] **Implementar P1-001** (circuit breaker) — asignar a Cursor Composer o Claude Code.
- [ ] **Implementar P1-002** (alert rules) — puede hacerse en el mismo PR que P1-001.
- [ ] **Re-correr el drill** una vez ambos P1s estén implementados y en docker-compose up.
- [ ] **Política**: no aceptar próximas semanas del roadmap con estos P1s abiertos.

---

## Lecciones

1. **Memory ≠ código**: el roadmap memory registró S2 como "done" pero solo
   la mitad (typed errors) llegó al repo. Los drills son la única forma de
   verificar que lo documentado coincide con lo implementado.

2. **Alert rules deben testearse en smoke**: añadir a `tools/smoke/alerts_check.py`
   una verificación de que el endpoint `/api/v2/alerts` de Alertmanager conoce
   las reglas esperadas (ALERT-001, -002, -004).

3. **El runbook debe preceder al drill**: crear runbooks en la misma semana que
   se implementa la defensa (S2 en este caso), no descubrirlo en S10.

4. **Drill automatizado sin Docker = auditoría estática**: documentar explícitamente
   en el scheduled task que el drill requiere stack levantado (`make up` en
   `platform/`). Añadir check de salud de servicios como precondición.

---

*Generado automáticamente por scheduled task `alpaca-bot-drill-day-5-outage` @ 2026-05-24 UTC.*  
*Claude Sonnet 4.6 — Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>*
