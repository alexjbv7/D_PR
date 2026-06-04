# PROJECT ML — Cursor Background Agents Setup

> Este archivo configura el entorno para los Background Agents de Cursor.
> Los agentes lo leen al iniciar para saber cómo preparar el workspace.

## Descripción del proyecto

Monorepo de trading algorítmico con tres capas:

| Carpeta | Contenido | Stack |
|---------|-----------|-------|
| `research/` | ML, backtesting, entrenamiento | Python 3.11, XGBoost, PyTorch |
| `platform/` | 8 microservicios FastAPI + frontend | Python 3.11, FastAPI, Kafka, Redis |
| `shared/` | `quant_shared` — schemas, features, registry | Python 3.11, Pydantic v2 |
| `tools/` | Briefings, smoke tests, MCP servers | Python 3.11 |

## Setup de entorno

### Python (requerido: 3.11+)

```bash
# 1. Shared library (siempre primero)
pip install -e shared/

# 2. Research (ML, backtesting)
cd research && pip install -e ".[dev]"

# 3. Platform services (instalar por servicio)
cd platform/services/execution-engine && pip install -r requirements.txt
```

### Variables de entorno

Crear `.env` en la raíz del repo (nunca commitear):

```bash
# Alpaca — paper trading
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_PAPER=true

# Bases de datos
POSTGRES_DSN=postgresql://user:pass@localhost:5432/quant_bot
REDIS_URL=redis://localhost:6379

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# FRED (macro data)
FRED_API_KEY=your_key

# Model registry
MODEL_REGISTRY_PATH=research/artifacts/registry.json
```

### Stack de infraestructura (Docker)

```bash
cd platform && make up
# Levanta: Kafka, Redis, Postgres/TimescaleDB, Prometheus, Grafana
# URLs: Dashboard http://localhost:3000 · Grafana http://localhost:3001 · Kafka UI http://localhost:8080
```

## Tests

```bash
# Research (ML core)
cd research && pytest tests/ -v --tb=short

# Execution engine (broker, risk gate, circuit breaker)
cd platform/services/execution-engine && pytest tests/ -v --tb=short

# Smoke test post-deploy
python -m tools.smoke.test_post_deploy_smoke
```

## Tareas frecuentes para agentes

### Entrenamiento multi-horizonte (dry-run)
```bash
cd research
python -m cli.run_nightly_retrain --dry-run
# Genera artifacts/runs/{run_id}.json
```

### Entrenamiento real (requiere GPU para DAILY/MLP)
```bash
cd research
python -m cli.run_nightly_retrain \
    --horizons swing,daily \
    --n-trials 50 \
    --as-of $(date +%Y-%m-%d)
# Exit 0 = ≥1 horizonte promovido a staging
# Exit 2 = sin edge detectado (no alarmar, documentar)
```

### Walk-forward completo (3 horizontes, ~4-8h con GPU)
```bash
cd research
python -m cli.train_multi_horizon \
    --as-of $(date +%Y-%m-%d) \
    --seed 42
```

### Briefing semanal
```bash
ISO_WEEK=$(python3 -c "from datetime import date; d=date.today(); print(f'{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}')")
python -m tools.briefing.weekly --week $ISO_WEEK
```

## Arquitectura clave

- **Circuit breaker**: `platform/services/execution-engine/app/brokers/_alpaca/circuit_breaker.py`
- **Risk gate** (kill switch step-0): `platform/services/execution-engine/app/risk_gate.py`
- **Nightly retrain DAG**: `research/pipelines/nightly_retrain.py`
- **Model registry**: `shared/quant_shared/models/registry.py`
- **Feature store**: `platform/services/ml-feature-store/`
- **Multi-horizon trainer**: `research/models/multi_horizon/trainer.py`

## Reglas para agentes (resumen de CLAUDE.md §20)

1. **Leer antes de escribir** — siempre leer el archivo completo antes de modificar.
2. **Walk-forward o nada** — métricas IS no se reportan; solo OOS con `WalkForwardRunner`.
3. **Anti-leakage** — `fit()` solo sobre datos de entrenamiento. Nunca sobre test.
4. **Tipado estricto** — `Decimal` para dinero, `UUID v7` para IDs, `UTC` para timestamps.
5. **Tests primero** — nunca commitear código público sin tests.
6. **No commitear secrets** — si detectas una clave en diff, abortar inmediatamente.
7. **No correr contra producción** sin doble confirmación humana.

## Estado del paper trading run

- **Run**: 2026-05-20 → 2026-06-19 (Alpaca paper, $100k inicial)
- **Kill switch**: Verificar `GET /health` antes de cualquier operación.
- **Roadmap 12 semanas**: COMPLETO (ver `docs/incidents/2026-06-03-s12-handoff.md`)
- **Pendiente**: diagnosticar 0 trades en W1-W2 (pipeline señal→executor)

## ADRs relevantes

| ADR | Decisión |
|-----|----------|
| 028 | Multi-horizon config (intraday 5min, swing 4H, daily 1D) |
| 034 | ResMLP reemplaza DeepMLP (post paper run 2026-06-19) |
| 035 | SLO: risk gate < 20ms, broker RTT < 600ms |
| 010 | UTC + Decimal + UUID v7 en toda la plataforma |

---
**Última actualización**: 2026-06-03 — S12 handoff completo
