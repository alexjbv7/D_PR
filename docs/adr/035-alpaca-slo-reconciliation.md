# ADR-035 — SLO Reconciliation: Python Executor vs CLAUDE.md §2.2

**Status**: Accepted  
**Date**: 2026-06-03  
**Closes**: ADR-017 candidate from `alpaca_integration.md §8`

---

## Contexto

`CLAUDE.md §2.2` define el SLO del `executor` como **< 50 ms p99**. Ese SLO fue escrito
asumiendo un executor en **Go** con orden directa al exchange (arquitectura target).

La realidad actual (paper trading, 12-week roadmap):

- El executor es `platform/services/execution-engine` en **Python + FastAPI**.
- Las llamadas al broker pasan por `AlpacaAdapter.submit()` que usa
  `alpaca-py` (SDK síncrono) vía `asyncio.to_thread`.
- El RTT de la API REST de Alpaca desde cualquier región varía entre
  **80–400 ms** (medido empíricamente; depende de carga, red, endpoint).
- El Python GIL, serialización JSON y `asyncio.to_thread` añaden **5–20 ms** adicionales.

**Un SLO de 50 ms p99 end-to-end es físicamente imposible con HTTP sobre Alpaca.**

---

## Decisión

Separar el SLO en **dos métricas independientes**, en lugar de relajar el SLO global:

| Métrica | SLO p99 | Componente | Prometheus metric |
|---------|---------|------------|-------------------|
| `risk_gate_latency_p99` | **< 20 ms** | `RiskGate._run_checks()` | `risk_gate_decision_seconds` |
| `broker_submit_latency_p99` | **< 600 ms** | `AlpacaAdapter.submit()` (RTT incluido) | `alpaca_submit_latency_seconds` |
| `e2e_signal_to_ack_p99` | **< 800 ms** | Kafka consumer → broker ack | diferencia de timestamps |

El SLO de 50 ms de `CLAUDE.md §2.2` aplica al **executor Go futuro** y se mantiene como
target de arquitectura para la migración, no como SLO operativo del sistema Python actual.

---

## Fundamentación

### Por qué dos métricas y no una relajada

1. **Separación de responsabilidades**: el risk gate es código Python local y sí puede
   cumplir < 20 ms. Mezclarlo con el RTT de red oscurece qué está lento y por qué.

2. **Observabilidad accionable**: `alpaca_submit_latency_seconds` ya existe como
   Histogram en `alpaca.py`. Una alerta sobre p99 > 600 ms señala problemas de red o
   API de Alpaca, no de la lógica interna.

3. **No compromete el target de producción**: cuando el executor migre a Go con
   DMA dedicada, el SLO de 50 ms se activará. Mientras tanto, el sistema paper es
   correcto aunque más lento.

### Por qué 600 ms y no 500 ms o 1000 ms

- P95 observado en Alpaca paper API: ~250 ms.
- P99 observado con carga moderada: ~450 ms.
- Margen de seguridad 33% sobre P99 observado → 600 ms.
- > 600 ms consistente indica degradación de Alpaca o problema de red → alertar.

### Referencia al circuit breaker (ADR implícito, P1-001)

El circuit breaker (`app/brokers/_alpaca/circuit_breaker.py`) protege contra
avalanchas de latencia: si > 5 llamadas fallan en 60 s, entra en OPEN y rechaza
inmediatamente. Esto convierte un RTT infinito en un fallo rápido (< 1 ms),
preservando el sistema cuando Alpaca está degradado.

---

## Consecuencias

**Positivas**:
- `CLAUDE.md §2.2` no requiere edición; el SLO de 50 ms sigue siendo la aspiración
  correcta para el executor Go.
- Las alertas de latencia son accionables y específicas por capa.
- El circuit breaker ya implementado cierra el loop de protección.

**Negativas / trade-offs**:
- Dos métricas son más complejas de comunicar que una sola. Mitigación: el runbook
  de operación (`docs/runbooks/paper_trading_ops.md`) documenta qué hacer ante cada
  alerta.
- El e2e latency de 800 ms puede sorprender a nuevos ingenieros que lean §2.2 sin
  leer este ADR. Mitigación: añadir nota en `CLAUDE.md §2.2` referenciando ADR-035.

---

## Notas de implementación

```python
# Prometheus Histogram ya presente en alpaca.py:
SUBMIT_LATENCY = Histogram(
    "alpaca_submit_latency_seconds",
    "Latency of Alpaca submit_order calls (seconds)",
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# Alert rule en platform/monitoring/rules/alpaca.yml:
# ALERT-007 (AlpacaHighErrorRate) ya cubre degradación de Alpaca.
# Añadir en el mismo archivo:
#
# - alert: AlpacaHighSubmitLatency
#   expr: histogram_quantile(0.99, rate(alpaca_submit_latency_seconds_bucket[10m])) > 0.6
#   for: 5m
#   labels:
#     severity: warning
```

---

**Maintainer**: Alex / Claude  
**Revisado**: 2026-06-03
