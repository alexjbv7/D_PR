# Runbook ALERT-002 — Alpaca Outage / Circuit Breaker Open

**Alerta**: `ALERT-002`  
**Severidad**: P0  
**Creado**: 2026-05-24 (drill día 5 — paper trading run 2026-05-20/06-19)  
**Mantenimiento**: actualizar tras cada incident real o drill

---

## Cuándo activa este runbook

- Prometheus dispara `ALERT-002` (regla: `alpaca_circuit_state == 1` durante > 30s)
- Circuit breaker del `execution-engine` transiciona `CLOSED → OPEN` tras detectar > 5 errores 5xx en 60 s
- Monitoreo manual detecta que `execution-engine` no procesa intents

---

## Estado del sistema cuando la alerta está activa

| Componente | Comportamiento esperado |
|------------|------------------------|
| `AlpacaAdapter` | Rechaza todos los `submit()` con `CircuitOpenError` sin llamar a Alpaca |
| `RiskGate` | Devuelve `breach="circuit_open"` en lugar de aprobar intents |
| `strategy-orchestrator` | Señales se encolan o descartan (según config `circuit_open_mode`) |
| `reconciler` | Continúa corriendo — solo lectura de posiciones, sin nuevos intentos |
| Métricas | `alpaca_circuit_state{state="open"} 1` visible en Prometheus |

---

## Diagnóstico (checklist en orden)

### Paso 1 — Confirmar que la alerta es real (no falso positivo)

```bash
# ¿El circuit breaker está realmente open?
curl -s http://localhost:9090/api/v1/query \
  -d 'query=alpaca_circuit_state' | jq .

# ¿Cuántos errores 5xx recientes?
curl -s http://localhost:9090/api/v1/query \
  -d 'query=rate(alpaca_submit_attempts_total{result="5xx"}[2m])' | jq .

# ¿Logs del execution-engine?
docker logs execution-engine --tail=100 --since=10m | grep -E "CircuitOpen|5xx|ERROR"
```

### Paso 2 — Verificar conectividad con Alpaca paper API

```bash
# Test básico de conectividad
curl -s -o /dev/null -w "%{http_code}" \
  https://paper-api.alpaca.markets/v2/account \
  -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
  -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY"

# Si el resultado es 000 → no hay conexión (red/DNS/firewall)
# Si el resultado es 503/502 → Alpaca down
# Si el resultado es 401 → credenciales caducadas
```

### Paso 3 — Verificar estado de posiciones (reconciler)

```bash
# ¿El reconciler sigue corriendo?
docker logs execution-engine --tail=50 | grep -i "reconcil"

# ¿Discrepancias internas vs Alpaca?
# (si reconciler detecta discrepancias, ALERT-004 debería dispararse también)
curl -s http://localhost:8010/health | jq .reconciler_last_run
```

### Paso 4 — Evaluar duración esperada del outage

| Causa | Duración típica | Acción |
|-------|----------------|--------|
| Alpaca maint. window (anunciada) | < 30 min | Esperar, monitorear |
| Alpaca incident no anunciado | Desconocida | Escalar a status.alpaca.markets |
| Red/DNS local | Minutos | Verificar VPN, DNS, firewall |
| Credenciales expiradas | Inmediato con fix | Rotar credenciales (ver §Secrets) |

---

## Acciones de remediación

### Si Alpaca está down y se estima < 30 min

1. No cerrar posiciones ni cambiar estado.
2. Confirmar que `strategy-orchestrator` no genera nuevas señales (`kill_switch` activado o `circuit_open_mode=discard`).
3. Esperar. El circuit breaker hace `HALF_OPEN` automáticamente tras TTL de 5 min.
4. Monitorear `alpaca_circuit_state` en Grafana (dashboard `trading.json`).

### Si el outage supera 30 min o es causa desconocida

```bash
# 1. Activar kill switch manual
curl -X POST http://localhost:8007/kill-switch \
  -H "Content-Type: application/json" \
  -d '{"reason": "ALERT-002 > 30min — operator manual halt"}'

# 2. Verificar posiciones actuales
curl http://localhost:8010/api/positions | jq .

# 3. Documentar en docs/incidents/ con template
cp tools/incidents/template.md docs/incidents/$(date +%Y-%m-%d)-alpaca-outage.md
```

### Recovery automático (circuit breaker TTL)

```
Tras TTL de 5 minutos en estado OPEN:
  → Circuit transiciona a HALF_OPEN
  → Se permite UNA request de prueba
  → Si la request tiene éxito: HALF_OPEN → CLOSED
  → Si falla: volver a OPEN con TTL reiniciado

Monitorear:
  alpaca_circuit_state{state="half_open"} == 1
  alpaca_circuit_state{state="closed"}    == 1  ← recovery confirmado
```

### Recovery manual forzado (solo si se está seguro del fix)

```bash
# Reiniciar solo el circuit breaker vía endpoint admin (si implementado)
curl -X POST http://localhost:8010/admin/circuit-breaker/reset \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# O reiniciar el servicio completo (última opción — perderás métricas in-memory)
docker-compose restart execution-engine
```

---

## Verificación post-recovery

```bash
# 1. Circuit CLOSED
curl -s http://localhost:9090/api/v1/query \
  -d 'query=alpaca_circuit_state{state="closed"}' | jq .data.result[0].value[1]
# Esperado: "1"

# 2. Submits exitosos
curl -s http://localhost:9090/api/v1/query \
  -d 'query=rate(alpaca_submit_attempts_total{result="success"}[5m])' | jq .

# 3. RiskGate aprueba intents (sin breach=circuit_open en logs)
docker logs execution-engine --tail=50 | grep -v circuit_open | grep -i "intent"

# 4. Reconciler sin discrepancias
curl -s http://localhost:8010/health | jq .reconciler_discrepancies
# Esperado: 0
```

---

## Escalación

| Condición | Acción |
|-----------|--------|
| Outage > 1h sin causa clara | Notificar operador por Discord + email |
| Discrepancias reconciler > 0 post-recovery | Escalado P0 inmediato — ver ALERT-004 |
| Credenciales inválidas | Rotar API keys en `.env` + `docker-compose restart execution-engine` |
| Alpaca confirma incident en status.alpaca.markets | Guardar screenshot en incident doc |

---

## Secrets — rotación de credenciales Alpaca

```bash
# 1. Generar nuevas keys en dashboard.alpaca.markets → Paper Account → API Keys
# 2. Actualizar .env (nunca commitear)
#    ALPACA_API_KEY=NUEVO_KEY
#    ALPACA_SECRET_KEY=NUEVO_SECRET
# 3. Recargar servicio
docker-compose up -d execution-engine
# 4. Verificar health
curl http://localhost:8010/health
```

---

## ⚠️ Estado actual (2026-05-24 — drill día 5)

> **GAP CRÍTICO identificado en drill:** El circuit breaker (`CLOSED → OPEN → HALF_OPEN`) 
> **NO está implementado** en `execution-engine` a la fecha de este runbook.  
> Solo existe `retry.py` (tenacity, max 3 intentos) pero no hay estado persistente de circuit.  
> Las Prometheus alert rules (`platform/monitoring/rules/`) tampoco existen.  
>  
> **ALERT-002 no puede dispararse actualmente.** Ver incident report  
> `docs/incidents/2026-05-24-drill-alpaca-outage.md` para P1s abiertos.
