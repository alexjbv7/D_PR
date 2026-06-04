# quant_bot — Monorepo de Trading Algorítmico ML

Sistema institucional de trading algorítmico construido sobre Machine Learning,
siguiendo las prácticas de López de Prado (*Advances in Financial Machine Learning*)
y operando en **paper trading sobre Alpaca** (run activo: 2026-05-20 → 2026-06-19).

> **Documento maestro**: [`CLAUDE.md`](CLAUDE.md) — arquitectura, ADRs, roadmap y reglas para agentes IA.  
> **Runbook de operación**: [`docs/runbooks/paper_trading_ops.md`](docs/runbooks/paper_trading_ops.md)

---

## Estado del sistema

| Componente | Estado | Notas |
|------------|--------|-------|
| Paper trading (Alpaca) | 🟢 Activo | Run 30d, $100k inicial |
| Execution engine | 🟢 Operativo | Circuit breaker + kill switch |
| Nightly retrain DAG | 🟢 Configurado | dry-run diario, gates DSR/ECE |
| Observabilidad | 🟢 Activo | Prometheus + Grafana + 5 alert rules |
| Multi-horizon ML | 🟡 Staging | intraday/swing/daily, esperando datos reales |
| Live trading | ⚪ No iniciado | Requiere 30d paper sin P0 + aprobación humana |

---

## Estructura del monorepo

```
quant_bot/
│
├── research/                   Núcleo ML: modelos, backtesting, validación
│   ├── models/
│   │   ├── zoo.py              LogReg, XGBoost, ResMLP, LSTM
│   │   ├── walk_forward_runner.py
│   │   ├── calibration.py      IsotonicCalibrator + TemperatureScaling
│   │   ├── meta_labeler.py
│   │   ├── multi_horizon/      trainer, horizon_config, registry_adapter
│   │   └── drift/              PSI, ECE, KS-test, macro event filter
│   ├── pipelines/
│   │   └── nightly_retrain.py  DAG nocturno (S11) — gates DSR/ECE/collapse
│   ├── cli/
│   │   ├── train_multi_horizon.py
│   │   └── run_nightly_retrain.py
│   ├── features/               engineering, regime_gmm, pca_denoiser, labeling
│   ├── risk/                   kelly, dynamic_rr, bayesian_sizer
│   ├── backtesting/            engine bar-level, multi_asset_engine
│   └── tests/                  anti-leakage, walk-forward, calibración, drift
│
├── platform/                   Microservicios event-driven (Kafka + Redis)
│   ├── services/
│   │   ├── execution-engine/   AlpacaAdapter + RiskGate + Reconciler
│   │   │   ├── app/brokers/_alpaca/
│   │   │   │   ├── circuit_breaker.py   ← CLOSED→OPEN→HALF_OPEN
│   │   │   │   ├── retry.py
│   │   │   │   └── rate_limiter.py
│   │   │   ├── app/risk_gate.py         ← kill switch step-0
│   │   │   └── app/reconciler.py        ← 60s loop + kill switch
│   │   ├── market-intelligence/  OpenBB + Binance WS
│   │   ├── macroeconomic/        FRED + Sahm Rule + yield curve
│   │   ├── onchain-analysis/     Whale detection + smart money
│   │   ├── context-engine/       GMM regime classifier (5 componentes)
│   │   ├── ml-feature-store/     Feature serving + drift detection
│   │   └── strategy-orchestrator/ Thompson sampling allocator
│   ├── monitoring/
│   │   ├── rules/alpaca.yml    ALERT-004/005/006/007/008
│   │   └── grafana/dashboards/
│   └── frontend/               React + TypeScript + Vite + Tailwind
│
├── shared/                     quant_shared — schemas, features, model registry
│   └── quant_shared/
│       ├── schemas/            OrderIntent, Fill, Signal (Pydantic v2)
│       ├── models/registry.py  ModelCard + ModelRegistry
│       ├── features/           19 features canónicos
│       └── calendar/           MarketCalendar (RTH/ETH/crypto 24/7)
│
├── docs/
│   ├── adr/                    35 ADRs (001–035)
│   ├── runbooks/               alpaca_outage, position_drift, paper_trading_ops
│   └── incidents/              DRILL-002/003/004, S12 handoff
│
└── tools/
    ├── briefing/               daily.py + weekly.py (CLI + Discord)
    └── smoke/                  post-deploy smoke test (22 checks)
```

---

## Arranque rápido

### Research (backtesting + ML)

```bash
# Dependencias
pip install -e shared/
cd research && pip install -e ".[dev]"

# Tests
pytest tests/ -v

# Dry-run del DAG nocturno (valida entorno sin entrenar)
python -m cli.run_nightly_retrain --dry-run
# Exit 0 + JSON en research/artifacts/runs/ = OK

# Training real (swing + daily, ~2h)
python -m cli.run_nightly_retrain --horizons swing,daily --n-trials 25
```

### Platform (microservicios)

```bash
cd platform
make up        # Kafka, Redis, Postgres, todos los servicios + frontend
make monitoring  # Prometheus + Grafana
```

URLs: dashboard `http://localhost:3000` · Grafana `http://localhost:3001` · Kafka UI `http://localhost:8080`

### Health check rápido

```bash
# Kill switch status (debe ser false)
curl -s http://localhost:8080/health | python3 -m json.tool | grep kill_switch

# Briefing semanal
python -m tools.briefing.weekly --week $(python3 -c "from datetime import date; d=date.today(); print(f'{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}')")
```

---

## Pipeline de inferencia (happy path)

```
Kafka signal (strategy-orchestrator)
        │
        ▼
execution-engine consumer
  ├── kill_switch? → rechazar
  ├── RiskGate (8 checks en orden):
  │     0. kill_switch_active      ← step-0, siempre primero
  │     1. require_paper
  │     2. daily_dd (-3% kill)
  │     3. per_symbol_cap (5%)
  │     4. per_venue_cap (50%)
  │     5. extended_hours (LIMIT only)
  │     6. market_closed
  │     7. PDT rule
  │     8. cash_buffer
  │
  ├── AlpacaAdapter.submit()
  │     └── CircuitBreaker (5 fallos/60s → OPEN → 30s → HALF_OPEN)
  │
  └── Reconciler (60s loop) → kill switch si drift > threshold
```

---

## Nightly Retrain DAG (S11)

```bash
# Gates que debe pasar cada horizonte para promover a "staging":
# 1. DSR nuevo ≥ 0.40 (floor absoluto)
# 2. ECE nuevo ≤ 0.05 (calibración)
# 3. Sin class collapse (> 5% por clase)
# 4. DSR nuevo ≥ DSR producción × 0.95 (no regresión)
```

El DAG corre automáticamente via scheduled task (22:00 local). Run logs en `research/artifacts/runs/`.

---

## Métricas OOS objetivo

| Métrica | Mínimo | Objetivo |
|---------|--------|----------|
| Sharpe anual OOS | > 0.8 | > 1.5 |
| DSR | > 0.4 | > 0.6 |
| ECE (calibración) | < 0.10 | < 0.05 |
| Max Drawdown | < 25% | < 15% |
| Broker latency p99 | — | < 600 ms (ADR-035) |
| Risk gate p99 | — | < 20 ms (ADR-035) |

---

## ADRs clave

| ADR | Decisión |
|-----|----------|
| 001 | Walk-forward como única métrica oficial |
| 003 | Kelly cuarto (0.25) como default |
| 010 | UTC + Decimal + UUID v7 en toda la plataforma |
| 028 | Multi-horizon: intraday 5min, swing 4H, daily 1D |
| 032 | Allocator Thompson: Beta(20,20) priors, decay 0.99/día |
| 034 | ResMLP reemplaza DeepMLP (post paper run 2026-06-19) |
| 035 | SLO: risk gate < 20ms, broker RTT < 600ms |

Lista completa en [`docs/adr/`](docs/adr/).

---

## Reglas no negociables

1. **Walk-forward o no existe** — métricas IS son irrelevantes. Solo OOS con `WalkForwardRunner`.
2. **Calibra antes de filtrar** — ECE < 0.05 verificado antes de usar `predict_proba()`.
3. **Purge + embargo obligatorio** — sin esto el backtest miente por leakage temporal.
4. **Kelly fraccional ≤ 0.25** — nunca Kelly completo en producción.
5. **Paper trading mínimo 30 días** antes de capital real, con aprobación humana explícita.
6. **`n_trials` honesto en DSR** — si optimizaste 100 combinaciones, `n_trials=100`.
7. **No commitear secrets** — API keys nunca en código ni `.env` commiteado.

---

## Referencias

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio". *JPM*.
- Guo, C. et al. (2017). "On Calibration of Modern Neural Networks". *ICML*.
- Shazeer, N. (2020). "GLU Variants Improve Transformer". *arXiv*.

---

## Disclaimer

Proyecto de investigación. Operar con dinero real conlleva riesgo de pérdida total del capital.
Las performances pasadas no garantizan resultados futuros.
