# quant_bot — Deep Reinforcement Learning Trading System

Sistema institucional de trading algorítmico basado en **Deep Reinforcement Learning (DRL)**.
El agente aprende directamente una política de trading óptima a partir de la interacción
con el mercado, sin necesidad de labelear manualmente la dirección del precio.

Opera en **paper trading sobre Alpaca** (run activo: 2026-05-20 → 2026-06-19).

> **Documento maestro**: [`CLAUDE.md`](CLAUDE.md) — arquitectura, ADRs, roadmap y reglas para agentes IA.  
> **Runbook de operación**: [`docs/runbooks/paper_trading_ops.md`](docs/runbooks/paper_trading_ops.md)

---

## Filosofía: por qué DRL y no ML supervisado

| | ML supervisado (XGBoost) | **DRL (objetivo)** |
|---|---|---|
| **Objetivo** | Predecir dirección del precio | Maximizar P&L ajustado por riesgo |
| **Labels** | Requiere Triple-Barrier labels manuales | No — aprende del reward directo |
| **Acción** | Señal binaria → ejecutar o no | Política continua: tamaño + dirección + timing |
| **Adaptación** | Estática entre retrainings | Se adapta online al régimen de mercado |
| **Limitación** | Asume que predecir ≈ ganar | Optimiza directamente lo que importa |

XGBoost se mantiene como **baseline de comparación** para medir si el agente DRL
realmente supera al enfoque clásico.

---

## Estado del sistema

| Componente | Estado | Notas |
|------------|--------|-------|
| Paper trading (Alpaca) | 🟢 Activo | Run 30d, $100k inicial |
| Execution engine | 🟢 Operativo | Circuit breaker + kill switch |
| Nightly retrain DAG | 🟢 Configurado | dry-run diario, gates DSR/ECE |
| Observabilidad | 🟢 Activo | Prometheus + Grafana + 5 alert rules |
| Q-learning tabular (MVP) | 🟡 Activo | Estado discretizado, acción {-1, 0, +1} |
| DQN (siguiente paso) | 🔵 En diseño | Red neuronal como función Q |
| PPO / SAC (objetivo) | ⚪ Roadmap | Política continua, sizing fraccional |
| Live trading | ⚪ No iniciado | Requiere 30d paper sin P0 + aprobación humana |

---

## Arquitectura DRL

### Formulación del problema

```
Estado (s):    vector de features de mercado + estado del portfolio
               ├── Features técnicos: RSI, MACD, ATR, z-scores, vol
               ├── Régimen: GMM probs (5 componentes)
               ├── Macro / on-chain: FRED, whale flows, funding rate
               └── Portfolio: posición actual, P&L no realizado, cash

Acción (a):    {-1 = vender, 0 = mantener, +1 = comprar}  ← MVP tabular
               → continua ∈ [-1, 1] = fracción de Kelly     ← PPO/SAC target

Reward (r):    P&L realizado ajustado por riesgo
               r_t = pnl_t - λ × volatility_t - c × |Δposition_t|
               donde λ = aversión al riesgo, c = costos de transacción

Política (π):  red neuronal → mapea estado a distribución sobre acciones
```

### Evolución de modelos (roadmap)

```
[ACTUAL]    Q-learning tabular
            Estado discretizado (regime_bin × p_win_bin × trend_bin)
            Q-table 3D: O(estados × acciones) parámetros
            research/models/rl_agent.py

[PRÓXIMO]   DQN (Deep Q-Network)
            Red neuronal sustituye la Q-table
            Replay buffer + target network (estabilidad)
            Arquitectura: ResMLP 3 bloques → Q(s, a)

[OBJETIVO]  PPO (Proximal Policy Optimization)
            Policy network + value network (actor-critic)
            Acción continua → sizing fraccional directo
            Clip ratio ε = 0.2, GAE λ = 0.95

[AVANZADO]  SAC (Soft Actor-Critic)
            Maximum entropy → exploración robusta
            Off-policy → sample efficiency superior a PPO
            Ideal para paper → live transition
```

### Arquitectura de red (policy / value function)

```
Input: [features mercado | portfolio state]   # dim ≈ 50–80
        │
        ▼
  ResMLP backbone (bloques residuales)
  ┌─────────────────────────────────────────┐
  │  Block 1: Linear → SwiGLU → LayerNorm  │
  │  Block 2: Linear → SwiGLU → LayerNorm  │  ← skip connections
  │  Block 3: Linear → SwiGLU → LayerNorm  │    gradientes estables
  └─────────────────────────────────────────┘
        │
        ├─── Policy head  → distribución sobre acciones (softmax / Normal)
        └─── Value head   → V(s) para reducir varianza (critic)
```

---

## Pipeline de entrenamiento DRL

```
Datos históricos (Alpaca bars / CCXT)
        │
        ▼
  Environment gym-compatible
  ├── reset() → estado inicial
  ├── step(action) → nuevo estado, reward, done
  └── render() → P&L curve, positions

        │  episodios (un episodio = un período histórico)
        ▼
  Agente DRL (PPO / SAC)
  ├── rollout: π(s) → a → r → s'
  ├── replay buffer (SAC) o batch on-policy (PPO)
  └── gradient update (policy + value network)

        │  cada N episodios
        ▼
  Validación walk-forward
  ├── WalkForwardRunner en modo DRL
  ├── Comparar vs XGBoost baseline (DSR, Sharpe)
  └── Gates: DSR ≥ 0.4, ECE (calibración de valor) ≤ 0.05

        │  si pasa gates
        ▼
  ModelRegistry → status "staging"
  → shadow trading 24h → canary 5% → producción
```

---

## Estructura del monorepo

```
quant_bot/
│
├── research/
│   ├── models/
│   │   ├── rl_agent.py          Q-learning tabular (MVP activo)
│   │   ├── zoo.py               ResMLP, LSTM (policy/value backbones)
│   │   ├── walk_forward_runner.py  Validación temporal anti-leakage
│   │   ├── calibration.py       Calibración de value estimates
│   │   └── multi_horizon/       trainer, horizon_config, registry_adapter
│   ├── pipelines/
│   │   └── nightly_retrain.py   DAG nocturno — gates DSR/ECE
│   ├── cli/
│   │   ├── run_nightly_retrain.py
│   │   └── train_multi_horizon.py
│   ├── features/                State space: engineering, regime_gmm, pca_denoiser
│   └── risk/                    kelly, dynamic_rr, bayesian_sizer (externos al agente)
│
├── platform/                    Infraestructura de ejecución
│   ├── services/
│   │   ├── execution-engine/
│   │   │   ├── app/brokers/_alpaca/circuit_breaker.py  ← CLOSED→OPEN→HALF_OPEN
│   │   │   ├── app/risk_gate.py                        ← kill switch step-0
│   │   │   └── app/reconciler.py                       ← 60s drift detection
│   │   ├── strategy-orchestrator/  Thompson sampling entre estrategias
│   │   ├── ml-feature-store/       State space en tiempo real (Redis)
│   │   └── context-engine/         Régimen de mercado (GMM 5 componentes)
│   └── monitoring/
│       └── rules/alpaca.yml    ALERT-004/005/006/007/008
│
├── shared/quant_shared/
│   ├── schemas/                OrderIntent, Signal (Pydantic v2)
│   ├── models/registry.py      ModelCard — registra agentes DRL
│   └── features/               19 features canónicos = state space base
│
└── docs/
    ├── adr/                    35 ADRs
    └── runbooks/               paper_trading_ops, alpaca_outage, position_drift
```

---

## Arranque rápido

```bash
# Dependencias
pip install -e shared/
cd research && pip install -e ".[dev]"

# Correr el agente Q-learning actual (simulación)
python -m research.models.rl_agent

# Dry-run del DAG nocturno
python -m cli.run_nightly_retrain --dry-run

# Platform (ejecución)
cd platform && make up
curl -s http://localhost:8080/health | python3 -m json.tool | grep kill_switch
```

---

## Reglas de riesgo (externas al agente DRL)

> El agente DRL **nunca decide** los límites de riesgo — eso es responsabilidad del `RiskGate`.

| Regla | Valor | Dónde |
|-------|-------|-------|
| Kill switch | Automático si DD > 3% intraday | `risk_gate.py` step-0 |
| Per-symbol cap | 5% del equity | `RiskGate` check 3 |
| Kelly máximo | 0.25 (quarter Kelly) | `bayesian_sizer.py` |
| Circuit breaker | 5 errores/60s → OPEN | `circuit_breaker.py` |
| Paper only | `ALPACA_PAPER=true` siempre | `AlpacaConfig` |

---

## Métricas objetivo

| Métrica | Mínimo | Objetivo |
|---------|--------|----------|
| Sharpe anual OOS | > 0.8 | > 1.5 |
| DSR (vs XGBoost baseline) | > 0.4 | > 0.6 |
| Max Drawdown | < 25% | < 15% |
| Broker latency p99 | — | < 600 ms (ADR-035) |
| Risk gate p99 | — | < 20 ms (ADR-035) |

---

## ADRs clave

| ADR | Decisión |
|-----|----------|
| 006 | Q-learning tabular antes de DQN/PPO |
| 009 | RL **no decide** risk limits — siempre externos |
| 010 | UTC + Decimal + UUID v7 |
| 034 | ResMLP como backbone de policy/value network |
| 035 | SLO: risk gate < 20ms, broker RTT < 600ms |

---

## Referencias

- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms". *arXiv*.
- Haarnoja et al. (2018). "Soft Actor-Critic". *ICML*.
- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio". *JPM*.

---

## Disclaimer

Proyecto de investigación. Operar con dinero real conlleva riesgo de pérdida total del capital.
Las performances pasadas no garantizan resultados futuros.
