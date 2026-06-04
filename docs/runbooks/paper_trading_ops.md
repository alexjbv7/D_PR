# Runbook: Operación Paper Trading

**Sistema**: PROJECT ML — execution-engine + strategy-orchestrator + los_ojos  
**Modo**: Paper trading (Alpaca paper API, `ALPACA_PAPER=true`)  
**Última actualización**: 2026-06-03 (S12 handoff)

---

## 1. Arranque del stack

```bash
# 1. Infraestructura base
cd platform && make up          # Kafka, Redis, Postgres, Prometheus, Grafana

# 2. Verificar topics Kafka
make kafka-create-topics

# 3. Servicios en orden
docker compose up -d market-intelligence macroeconomic onchain-analysis
docker compose up -d context-engine ml-feature-store
docker compose up -d strategy-orchestrator
docker compose up -d execution-engine

# 4. Smoke test
cd /quant_bot && python -m tools.smoke.test_post_deploy_smoke
# Esperado: 22 passed / 1 skipped
```

**Verificar que el kill switch NO está activo** antes de arrancar:

```bash
curl -s http://localhost:8080/health | python3 -m json.tool | grep kill_switch
# Esperado: "kill_switch": false
```

Si `kill_switch: true` → ver §4 (kill switch manual).

---

## 2. Health checks diarios

### 2.1 Briefing diario (automatizado)

```bash
# Genera markdown en tools/briefing/output/daily_YYYY-MM-DD.md
python -m tools.briefing.daily --date $(date +%Y-%m-%d)
```

Métricas a revisar:

| Métrica | Umbral OK | Acción si falla |
|---------|-----------|-----------------|
| Equity | > $95,000 | Ver §5 (DD protection) |
| DD intraday | < 3% | Kill switch automático debería disparar |
| Sharpe rolling 7d | > 0 | Flag para revisión de modelo |
| Trades en 24h | > 0 | Verificar pipeline señal→executor |
| Drift events | 0 o bajo | PSI > 0.25 → retrain forzado |
| Alerts P0 | 0 | Acción inmediata (ver §6) |

### 2.2 Pipeline check

```bash
# Verificar consumer lag de Kafka
docker exec kafka kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --group execution-engine-cg

# Lag > 1000 mensajes → execution-engine no está consumiendo
```

### 2.3 Prometheus / Grafana

- Dashboard principal: http://localhost:3001 → "Trading Dashboard"
- Alert rules activas: `platform/monitoring/rules/alpaca.yml`
- Alertas críticas: ALERT-004 (circuit breaker OPEN), ALERT-007 (5xx storm)

---

## 3. Retraining nocturno (DAG S11)

El DAG corre automáticamente vía scheduled task. Para ejecutar manualmente:

```bash
cd /quant_bot/research

# Dry-run (valida config, no entrena)
python -m cli.run_nightly_retrain --dry-run

# Run completo (entrena los 3 horizontes)
python -m cli.run_nightly_retrain --as-of $(date +%Y-%m-%d)

# Solo swing y daily (más rápido, 2h aprox.)
python -m cli.run_nightly_retrain --horizons swing,daily --n-trials 25
```

**Gates de promoción** (todos deben pasar para staging):
1. DSR nuevo ≥ 0.40 (floor absoluto)
2. ECE nuevo ≤ 0.05 (calibración)
3. Sin class collapse (> 5% por clase)
4. DSR nuevo ≥ DSR producción × 0.95 (no regresión)

**Run logs**: `research/artifacts/runs/{run_id}.json`

**Exit codes**:
- `0` → ≥ 1 horizonte promovido a staging
- `1` → error de configuración o runtime
- `2` → ningún horizonte promovió (sin edge en datos actuales — documentar, no alarmar)

Si exit code 2 por 3 días consecutivos → revisión manual del dataset + posible drift.

---

## 4. Kill switch manual

### Activar (emergencia)

```bash
# Vía REST (desde cualquier host con acceso al servicio)
curl -X POST http://localhost:8080/api/kill_switch/trip

# Efecto: bloquea TODOS los nuevos intents (Kafka + REST)
# RiskGate._kill_switch_active = True
# Kafka consumer: state.kill_switch_tripped = True
```

### Resetear (tras resolución del incidente)

```bash
# Solo después de verificar que el problema está resuelto
curl -X POST http://localhost:8080/api/kill_switch/reset

# Verificar que el sistema está sano antes de reset:
# 1. Posiciones internas == posiciones broker (reconciler)
# 2. Circuit breaker en CLOSED
# 3. No hay alertas P0 activas
curl -s http://localhost:8080/health | python3 -m json.tool
```

### Circuit breaker (automático)

El circuit breaker se abre automáticamente tras 5 errores 5xx en 60 s.
Se recupera solo (HALF_OPEN → CLOSED) tras 30 s si la siguiente llamada tiene éxito.

Para forzar recovery manual:

```bash
# No hay endpoint de reset para el CB; se recupera solo.
# Si el CB está OPEN por > 5 min, investigar Alpaca status:
# https://status.alpaca.markets
```

---

## 5. Drawdown protection

Thresholds configurados en `RiskConfig`:

| Nivel | Threshold | Acción automática |
|-------|-----------|-------------------|
| Intraday | -3% equity | Kill switch (RiskGate `daily_dd` breach) |
| Semanal | -7% equity | Kill switch + alerta P0 |
| Mensual | -12% equity | Freeze + revisión humana obligatoria |

Verificar thresholds actuales:

```bash
curl -s http://localhost:8080/health | python3 -m json.tool | grep -A5 config
```

---

## 6. Escalaciones por severidad

### P0 — Acción inmediata (< 15 min)

- Circuit breaker OPEN > 5 min
- Equity DD intraday > 5%
- Discrepancia posiciones interno vs broker (reconciler)
- NaN o error en inferencia ML propagado a señal

**Acción**: trip kill switch → investigar → resolver → reset.

### P1 — Acción en < 2h

- Circuit breaker HALF_OPEN repetido (inestabilidad de red)
- Sharpe rolling 7d < 0
- Drift PSI > 0.25 en feature top-5
- Exit code 2 del DAG de retraining por 3 días

**Acción**: investigar causa, escalar si no hay resolución en 2h.

### P2 — Revisión en el día

- 429 rate limiting de Alpaca elevado
- Latencia broker p99 > 600 ms (ALERT-005 de alpaca.yml)
- 0 trades en 24h (pipeline warning, no emergencia si el mercado estaba cerrado)

---

## 7. Reconciliation manual

```bash
# Ver posiciones actuales internas
curl -s http://localhost:8080/api/positions | python3 -m json.tool

# Ver posiciones en Alpaca (vía endpoint)
curl -s "http://localhost:8080/api/account/alpaca" | python3 -m json.tool

# El reconciler corre cada 60s automáticamente.
# Si hay discrepancia persisente: revisar logs del reconciler
docker logs execution-engine 2>&1 | grep reconciler | tail -20
```

---

## 8. Pre-checklist para live trading (post-S12)

> **NO activar live trading sin completar este checklist con verificación humana.**

- [ ] Paper trading funcionando 30 días sin P0 incidents
- [ ] DD máximo en paper < 5% en cualquier período de 7 días
- [ ] ≥ 1 horizonte con DSR ≥ 0.5 en producción (registry status=production)
- [ ] Kill switch manual testado exitosamente (DRILL-004 o posterior)
- [ ] Secrets rotados (Alpaca API keys, Postgres password)
- [ ] `risk_require_paper` cambiado a `False` solo en env de live
- [ ] Revisión humana de `RiskConfig` limits (per_symbol_cap_pct, daily_dd_kill_pct)
- [ ] SLO de latencia broker validado (p99 < 600 ms en paper)
- [ ] Backup de registry.json verificado antes del switch
- [ ] Approval explícito del PM/operador principal

---

## 9. Archivos clave

| Componente | Path |
|------------|------|
| Risk gate | `platform/services/execution-engine/app/risk_gate.py` |
| Circuit breaker | `platform/services/execution-engine/app/brokers/_alpaca/circuit_breaker.py` |
| Reconciler | `platform/services/execution-engine/app/reconciler.py` |
| Kill switch endpoint | `platform/services/execution-engine/app/main.py` (POST /api/kill_switch/{action}) |
| Nightly retrain DAG | `research/pipelines/nightly_retrain.py` |
| Retrain CLI | `research/cli/run_nightly_retrain.py` |
| Model registry | `shared/quant_shared/models/registry.py` |
| Alert rules | `platform/monitoring/rules/alpaca.yml` |
| Daily briefing | `tools/briefing/daily.py` |
| Weekly briefing | `tools/briefing/weekly.py` |

---

**Related**: `docs/runbooks/alpaca_outage.md`, `docs/runbooks/position_drift.md`  
**ADRs**: ADR-035 (SLO), ADR-010 (UTC+Decimal+UUID), ADR-009 (RL no decide risk limits)
