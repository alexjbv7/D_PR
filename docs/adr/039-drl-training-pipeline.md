# ADR-039 — Pipeline de Entrenamiento DRL

**Status**: Accepted  
**Date**: 2026-06-03  
**Depende de**: ADR-037 (Environment), ADR-038 (Redes)  
**Implementa**: `research/pipelines/drl_train.py`

---

## Contexto

Con el environment (ADR-037) y las redes (ADR-038) definidos, este ADR especifica
**cómo se entrena el agente**: el loop de episodios, el régimen de datos, logging,
checkpointing, evaluación OOS y promoción al registry.

Un pipeline de entrenamiento DRL mal diseñado puede producir agentes que parecen
buenos en training pero fallan en OOS — el mismo problema que el ML supervisado,
pero más difícil de detectar porque el agente "parece aprender" durante los episodios.

---

## 1. RÉGIMEN DE DATOS

### Split temporal (no aleatorio)

```
Dataset completo: 2020-01-01 → hoy

Train:  2020-01-01 → 2024-12-31   (5 años — episodios de entrenamiento)
Val:    2025-01-01 → 2025-12-31   (1 año  — early stopping, no se toca para tuning)
Test:   2026-01-01 → hoy          (hold-out — se evalúa UNA SOLA VEZ al final)
```

**Por qué no shuffle aleatorio**: los mercados tienen dependencia temporal.
Un episodio de enero 2024 tiene información que filtraría a un episodio de
diciembre 2023 si los mezclas. La división temporal es el equivalente del
embargo en ML supervisado.

### Episodios de entrenamiento

```python
# Cada episodio = ventana deslizante de N barras del período Train
episode_length = 252   # barras (≈ 1 año diario, ≈ 1 semana 5-min)

# Muestreo de episodios: aleatorio DENTRO del período Train
# El agente ve el mismo dato histórico múltiples veces desde
# distintos puntos de inicio → generalización temporal
start_idx = random.randint(0, len(train_data) - episode_length)
episode_data = train_data[start_idx : start_idx + episode_length]
```

---

## 2. LOOP DE ENTRENAMIENTO

### Estructura general (DQN)

```python
def train_dqn(
    env_config:    EnvironmentConfig,
    dqn_config:    DQNConfig,
    train_data:    pd.DataFrame,
    val_data:      pd.DataFrame,
    registry:      ModelRegistry,
    n_episodes:    int = 2000,
    eval_every:    int = 50,     # evaluar en val cada N episodios
    patience:      int = 200,    # early stopping
) -> TrainResult:

    agent   = TradingDQN(...)
    buffer  = ReplayBuffer(capacity=100_000)
    best_val_sharpe = -inf
    no_improve = 0

    for episode in range(n_episodes):
        # 1. Muestrear ventana aleatoria del período train
        env  = TradingEnvironment(sample_window(train_data), env_config)
        obs, _ = env.reset()
        done = False

        while not done:
            # 2. Seleccionar acción (ε-greedy)
            epsilon = compute_epsilon(episode, dqn_config)
            action  = agent.select_action(obs, epsilon)

            # 3. Step en el environment
            next_obs, reward, done, _, info = env.step(action)

            # 4. Guardar en replay buffer
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs

            # 5. Actualizar si hay suficiente experiencia
            if len(buffer) >= dqn_config.batch_size:
                batch = buffer.sample(dqn_config.batch_size)
                loss  = agent.update(batch)

        # 6. Evaluación periódica en val set
        if episode % eval_every == 0:
            val_metrics = evaluate(agent, val_data, env_config)
            log_metrics(episode, val_metrics)

            if val_metrics.sharpe > best_val_sharpe:
                best_val_sharpe = val_metrics.sharpe
                save_checkpoint(agent, episode)
                no_improve = 0
            else:
                no_improve += eval_every
                if no_improve >= patience:
                    logger.info("Early stopping at episode %d", episode)
                    break

    # 7. Cargar mejor checkpoint y evaluar en val final
    agent = load_best_checkpoint()
    return compute_train_result(agent, val_data, env_config)
```

---

## 3. MÉTRICAS DE EVALUACIÓN

El agente se evalúa con las mismas métricas que el ML supervisado + métricas específicas de RL:

```python
@dataclass
class DRLEvalMetrics:
    # Métricas financieras (comparables con XGBoost baseline)
    sharpe_annual:    float   # Sharpe anualizado sobre el período de eval
    dsr:              float   # Deflated Sharpe Ratio (corregido por n_trials)
    max_drawdown:     float   # DD máximo en el período
    calmar:           float   # Sharpe / |max_drawdown|

    # Métricas de comportamiento del agente
    trade_frequency:  float   # trades / total_bars (ideal: 0.05 - 0.30)
    avg_holding_bars: float   # barras promedio por posición
    action_entropy:   float   # entropía de la distribución de acciones
    reward_mean:      float   # reward promedio por step
    reward_std:       float   # volatilidad del reward

    # Comparación con baseline
    sharpe_vs_xgb:   float   # sharpe_agente - sharpe_xgb (> 0 = supera baseline)
    dsr_vs_xgb:      float   # dsr_agente - dsr_xgb
```

### Gate de promoción a staging

```python
def passes_promotion_gate(metrics: DRLEvalMetrics, xgb_baseline: XGBMetrics) -> bool:
    return (
        metrics.dsr              >= 0.4                    # floor absoluto
        and metrics.sharpe_annual >= 0.8                   # sharpe mínimo
        and metrics.max_drawdown  <= 0.20                  # DD máximo
        and metrics.trade_frequency >= 0.02                # no idle policy
        and metrics.sharpe_vs_xgb  > 0.0                  # supera baseline
        and metrics.action_entropy  > 0.3                  # no política degenerada
    )
```

---

## 4. LOGGING Y CHECKPOINTING

### Log estructurado por episodio

```python
# Cada episodio → una línea JSON en el run log
{
    "episode":        1234,
    "train_reward":   0.0234,
    "val_sharpe":     1.23,      # solo cuando eval_every
    "val_dsr":        0.51,
    "epsilon":        0.15,
    "buffer_size":    45000,
    "loss":           0.0012,
    "timestamp_utc":  "2026-06-03T04:00:00+00:00"
}
```

### Checkpoints

```
research/artifacts/drl/
├── {run_id}/
│   ├── config.json          EnvironmentConfig + DQNConfig
│   ├── best_checkpoint.pt   state_dict del mejor val Sharpe
│   ├── final_checkpoint.pt  state_dict del último episodio
│   └── train_log.jsonl      log por episodio (append)
```

---

## 5. COMPARACIÓN CON XGBOOST BASELINE

Antes de registrar el agente DRL en el ModelRegistry, se corre automáticamente
el XGBoost baseline sobre el mismo período val y test:

```python
def compare_with_baseline(
    drl_agent:   TradingDQN,
    val_data:    pd.DataFrame,
    registry:    ModelRegistry,
) -> ComparisonReport:

    # Cargar mejor XGBoost de producción del registry
    xgb_card = registry.get_production("multi_horizon_swing")

    drl_metrics = evaluate(drl_agent, val_data)
    xgb_metrics = evaluate_supervised(xgb_card, val_data)

    return ComparisonReport(
        drl=drl_metrics,
        xgb=xgb_metrics,
        winner="drl" if drl_metrics.sharpe > xgb_metrics.sharpe else "xgb",
        delta_sharpe=drl_metrics.sharpe - xgb_metrics.sharpe,
        delta_dsr=drl_metrics.dsr - xgb_metrics.dsr,
    )
```

---

## 6. INTEGRACIÓN CON NIGHTLY RETRAIN DAG

El pipeline DRL se integra con el DAG existente (`research/pipelines/nightly_retrain.py`)
como un modo adicional:

```bash
# ML supervisado (ya implementado)
python -m cli.run_nightly_retrain --horizons swing,daily

# DRL (nuevo)
python -m cli.run_drl_train --algorithm dqn --episodes 500 --dry-run
python -m cli.run_drl_train --algorithm dqn --episodes 2000
```

Los gates son los mismos (DSR ≥ 0.4, superar baseline), el registry es el mismo
(`ModelCard` con `model_class="dqn"` o `"ppo"` o `"sac"`).

---

## 7. ESTRUCTURA DE ARCHIVOS

```
research/
├── pipelines/
│   ├── nightly_retrain.py      ya implementado (S11)
│   └── drl_train.py            nuevo — DRLTrainConfig, train_dqn(), train_ppo()
├── cli/
│   ├── run_nightly_retrain.py  ya implementado
│   └── run_drl_train.py        nuevo — CLI wrapper
└── tests/
    └── test_drl_pipeline.py    nuevo — train loop en miniatura (10 episodios)
```

---

## 8. HIPERPARÁMETROS A TUNEAR (Optuna)

```yaml
# DQN
learning_rate:    [1e-5, 1e-3]   log
batch_size:       [64, 512]      log
buffer_size:      [10000, 500000] log
gamma:            [0.95, 0.999]
epsilon_decay:    [5000, 100000]
target_update:    [500, 5000]

# Environment (compartido con ADR-037)
lambda_dd:        [0.5, 5.0]
lambda_vol:       [0.0, 2.0]
idle_penalty:     [0.0, 0.01]
episode_length:   [126, 504]

# Backbone (compartido con ADR-038)
hidden_dim:       [128, 512]
n_blocks:         [2, 5]
dropout:          [0.0, 0.3]

# Budget Optuna
n_trials:         50     # mismo que ML supervisado
pruner:           HyperbandPruner
```

---

## Consecuencias

**Positivas**:
- Loop de entrenamiento claro y auditable — mismo estándar que el walk-forward supervisado
- Early stopping basado en val Sharpe → evita overfitting silencioso
- Comparación automática con XGBoost baseline → nunca se promueve un agente peor
- Integración con ModelRegistry existente → un único lugar para todos los modelos

**Negativas / trade-offs**:
- 2000 episodios × 252 steps = 504,000 interacciones mínimas → horas de compute (GPU recomendada)
- El split temporal fijo (Train/Val/Test) es más conservador que shuffle → menos datos de train. Mitigación: usar datos desde 2018 si disponibles
- El agente puede tardar muchos episodios en "despegar" (reward ≈ 0 las primeras 200 épocas) → normal en DRL, no señal de error

---

**Maintainer**: Alex / Claude  
**Revisado**: 2026-06-03
