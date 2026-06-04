# ADR-038 — Arquitectura de Redes Policy/Value (DQN → PPO → SAC)

**Status**: Accepted  
**Date**: 2026-06-03  
**Depende de**: ADR-036 (DRL-First), ADR-037 (Environment)  
**Implementa**: `research/models/drl/`

---

## Contexto

ADR-037 definió el estado (42 dims), la acción y el reward. Este ADR especifica
las redes neuronales que mapean estado → política/valor en cada etapa de la escalera DRL.

El espacio de estado es **tabular estructurado** (no imagen, no texto): cada dimensión
tiene significado financiero preciso. Por eso ResMLP (bloques residuales sobre tabular)
es superior a CNN o Transformer puro para este dominio — confirmado por ADR-034.

---

## 1. BACKBONE COMPARTIDO — TradingResMLP

Todas las redes (Q-network, policy, value) comparten el mismo backbone.
Esto permite preentrenamiento y transferencia entre etapas de la escalera.

```python
class TradingResMLP(nn.Module):
    """
    Backbone residual para el espacio de estado de trading.

    Input:  s_t ∈ ℝ^{obs_dim}   (default 42)
    Output: embedding ∈ ℝ^{hidden_dim}

    Decisiones de diseño:
    - SwiGLU en lugar de ReLU/GELU: mejor gradiente en tabular financiero (ADR-034)
    - LayerNorm en lugar de BatchNorm: estable con batch size variable en RL
    - Skip connections: gradientes estables a 4+ bloques sin degradación
    - Dropout 0.1: regularización ligera, el mercado ya tiene ruido suficiente
    """

    def __init__(
        self,
        obs_dim:    int = 42,
        hidden_dim: int = 256,
        n_blocks:   int = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_norm(h)


class ResBlock(nn.Module):
    """Bloque residual con SwiGLU y LayerNorm pre-activación."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim * 2)   # SwiGLU necesita 2× dim
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gate, linear = self.gate(h).chunk(2, dim=-1)
        h = self.proj(torch.sigmoid(gate) * linear)   # SwiGLU
        return x + self.drop(h)                        # skip connection
```

### Hiperparámetros del backbone (Optuna search space)

```yaml
hidden_dim:  [128, 512]   step=64
n_blocks:    [2, 5]
dropout:     [0.0, 0.3]
```

---

## 2. ETAPA 1 — DQN (Deep Q-Network)

### Arquitectura

```
Estado s_t (42)
      │
      ▼
TradingResMLP backbone (42 → 256)
      │
      ▼
Q-head: Linear(256 → n_actions=3)
      │
      ▼
Q(s_t, a) para a ∈ {SELL, HOLD, BUY}
```

```python
class TradingDQN(nn.Module):
    """
    Deep Q-Network para trading.

    Predice Q(s, a) para las 3 acciones discretas.
    La política es ε-greedy sobre argmax Q(s, a).
    """

    def __init__(self, obs_dim: int = 42, hidden_dim: int = 256, n_blocks: int = 3):
        super().__init__()
        self.backbone = TradingResMLP(obs_dim, hidden_dim, n_blocks)
        self.q_head   = nn.Linear(hidden_dim, 3)   # 3 acciones

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.backbone(state))   # (batch, 3)

    def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        if torch.rand(1).item() < epsilon:
            return torch.randint(3, (1,)).item()   # exploración
        with torch.no_grad():
            return self.forward(state).argmax(dim=-1).item()
```

### Entrenamiento DQN

```python
# Componentes estándar DQN (Mnih et al. 2015)
replay_buffer_size:  100_000   # experience replay
batch_size:          256
target_network_update: 1000    # steps entre sincronizaciones
gamma:               0.99      # descuento
epsilon_start:       1.0
epsilon_end:         0.05
epsilon_decay:       50_000    # steps hasta epsilon_end
learning_rate:       3e-4
optimizer:           Adam
loss:                Huber     # más robusto que MSE ante outliers de reward
```

### Por qué Huber loss en lugar de MSE

Los rewards de trading tienen distribución fat-tail (eventos extremos frecuentes).
MSE amplifica estos outliers → gradientes explosivos. Huber loss es cuadrática en
`|δ| < 1` y lineal en `|δ| ≥ 1`, siendo robusta sin perder sensibilidad local.

---

## 3. ETAPA 2 — PPO (Proximal Policy Optimization)

### Arquitectura actor-critic

```
Estado s_t (42)
      │
      ├──► Actor (policy network)
      │         TradingResMLP(42 → 256)
      │         + policy_head: Linear(256 → n_actions) → Categorical
      │         → distribución π(a|s)
      │
      └──► Critic (value network)
                TradingResMLP(42 → 256)   ← backbone COMPARTIDO en MVP
                + value_head: Linear(256 → 1)
                → V(s) ∈ ℝ
```

```python
class TradingActorCritic(nn.Module):
    """
    Actor-Critic compartido para PPO.

    El backbone compartido reduce parámetros y acelera entrenamiento.
    En versiones avanzadas, backbones separados permiten especialización.
    """

    def __init__(
        self,
        obs_dim:     int = 42,
        hidden_dim:  int = 256,
        n_blocks:    int = 3,
        n_actions:   int = 3,        # discreta MVP
        shared_backbone: bool = True,
    ):
        super().__init__()
        self.shared = shared_backbone

        if shared_backbone:
            self.backbone     = TradingResMLP(obs_dim, hidden_dim, n_blocks)
            self.policy_head  = nn.Linear(hidden_dim, n_actions)
            self.value_head   = nn.Linear(hidden_dim, 1)
        else:
            self.actor_backbone  = TradingResMLP(obs_dim, hidden_dim, n_blocks)
            self.critic_backbone = TradingResMLP(obs_dim, hidden_dim, n_blocks)
            self.policy_head     = nn.Linear(hidden_dim, n_actions)
            self.value_head      = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        if self.shared:
            h = self.backbone(state)
            logits = self.policy_head(h)
            value  = self.value_head(h)
        else:
            logits = self.policy_head(self.actor_backbone(state))
            value  = self.value_head(self.critic_backbone(state))

        dist = torch.distributions.Categorical(logits=logits)
        return dist, value.squeeze(-1)

    def get_action(self, state: torch.Tensor):
        dist, value = self.forward(state)
        action      = dist.sample()
        log_prob    = dist.log_prob(action)
        return action, log_prob, value
```

### Hiperparámetros PPO

```python
# Clip ratio — el más crítico en PPO
clip_epsilon:    0.2       # rango conservador (Schulman et al. 2017)
n_epochs:        10        # actualizaciones por rollout
rollout_steps:   2048      # pasos por rollout antes de update
gae_lambda:      0.95      # GAE para reducir varianza del advantage
entropy_coef:    0.01      # exploración via entropía
value_loss_coef: 0.5
max_grad_norm:   0.5       # gradient clipping
learning_rate:   3e-4
lr_schedule:     linear_decay_to_zero
```

### Acción continua (PPO fase 2)

Cuando se pase a sizing continuo ∈ [-1, 1]:

```python
# Reemplazar Categorical por Normal distribution
self.policy_mean = nn.Linear(hidden_dim, 1)
self.policy_log_std = nn.Parameter(torch.zeros(1))   # aprendida

mean = torch.tanh(self.policy_mean(h))     # clip a [-1, 1]
std  = torch.exp(self.policy_log_std).clamp(0.01, 1.0)
dist = torch.distributions.Normal(mean, std)
```

---

## 4. ETAPA 3 — SAC (Soft Actor-Critic)

### Diferencias clave respecto a PPO

| | PPO | SAC |
|---|---|---|
| **On/Off-policy** | On-policy | Off-policy (replay buffer) |
| **Sample efficiency** | Menor | Mayor (~3-5×) |
| **Entropía** | Coeficiente fijo | Temperatura aprendida automáticamente |
| **Estabilidad** | Alta | Alta (twin Q-networks) |
| **Acción** | Discreta o continua | Diseñado para continua |

### Arquitectura SAC

```
Actor:    TradingResMLP → mean, log_std → reparameterization trick
Critic 1: TradingResMLP → Q_1(s, a)   ← twin critics para
Critic 2: TradingResMLP → Q_2(s, a)   ← reducir sobreestimación
Target 1: copia de Critic 1 (soft update τ=0.005)
Target 2: copia de Critic 2 (soft update τ=0.005)
```

```python
# Temperatura de entropía aprendida (clave de SAC)
log_alpha = nn.Parameter(torch.tensor(0.0))
alpha = log_alpha.exp()   # se ajusta para mantener entropía objetivo H_target

# H_target = -dim(action_space) = -1 para acción 1D
```

### Hiperparámetros SAC

```python
replay_buffer_size: 1_000_000
batch_size:         256
tau:                0.005      # soft update de target networks
gamma:              0.99
learning_rate:      3e-4
target_entropy:     -1.0       # para acción continua 1D
```

---

## 5. INICIALIZACIÓN DE PESOS

Crítica en DRL: mala inicialización → gradientes muertos o explosivos desde el primer episodio.

```python
def init_weights(module: nn.Module) -> None:
    """Inicialización ortogonal (recomendada para RL por Schulman et al.)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        nn.init.zeros_(module.bias)

# Para las cabezas de política y valor — ganancia menor
policy_head.apply(lambda m: nn.init.orthogonal_(m.weight, gain=0.01)
                  if isinstance(m, nn.Linear) else None)
value_head.apply(lambda m: nn.init.orthogonal_(m.weight, gain=1.0)
                 if isinstance(m, nn.Linear) else None)
```

---

## 6. ESTRUCTURA DE ARCHIVOS

```
research/
├── models/
│   └── drl/
│       ├── __init__.py
│       ├── backbone.py        TradingResMLP, ResBlock
│       ├── dqn.py             TradingDQN, ReplayBuffer, DQNTrainer
│       ├── ppo.py             TradingActorCritic, PPOTrainer
│       ├── sac.py             SACAgent, TwinCritic, SACTrainer
│       └── utils.py           init_weights, polyak_update, RolloutBuffer
├── envs/
│   ├── __init__.py
│   └── trading_env.py         TradingEnvironment (gym.Env)
└── tests/
    ├── test_drl_backbone.py   shapes, gradientes, no NaN
    ├── test_drl_dqn.py        Q-values, select_action, replay buffer
    └── test_trading_env.py    reset, step, reward no NaN, episode length
```

---

## 7. VALIDACIÓN ANTI-OVERFITTING

DRL en finanzas sufre de overfitting silencioso: el agente memoriza el período de
entrenamiento sin aprender la política subyacente.

```python
# Protocolo de validación (walk-forward para DRL)
train_episodes:  años 2021-2024  (simulación histórica)
val_episodes:    año 2025        (out-of-sample, nunca visto durante training)
test_episodes:   año 2026        (solo se toca una vez, al final)

# Señales de overfitting
- Sharpe train >> Sharpe val   → reducir capacidad del backbone o aumentar dropout
- Agent nunca cierra posición  → revisar reward (r_idle demasiado pequeño)
- Agent siempre en máximo size → revisar penalización de riesgo

# Early stopping
- Monitor: Sharpe rolling 30 episodios en val set
- Parar si no mejora en 100 episodios consecutivos
- Guardar checkpoint del mejor val Sharpe, no del último
```

---

## 8. LIBRERÍA DE ENTRENAMIENTO

Se usa **Stable-Baselines3** (SB3) como wrapper de entrenamiento para DQN y PPO,
con implementación custom para SAC (mayor control sobre el reward shaping).

```python
# DQN — SB3
from stable_baselines3 import DQN
model = DQN(
    policy="MlpPolicy",
    env=trading_env,
    policy_kwargs={"net_arch": [256, 256, 256]},  # sustituir por TradingResMLP
    ...
)

# PPO — SB3
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env=trading_env, ...)

# SAC — custom (mayor control sobre twin critics y temperatura)
```

> **Nota para agentes de Cursor**: SB3 usa `MlpPolicy` por defecto. Para usar
> `TradingResMLP` hay que registrar un custom policy. Ver:
> `research/models/drl/sb3_custom_policy.py` (a implementar en S13+).

---

## Consecuencias

**Positivas**:
- Backbone compartido entre DQN/PPO/SAC → preentrenamiento reutilizable
- SwiGLU + LayerNorm + skip connections → gradientes estables hasta 5 bloques
- Inicialización ortogonal → arranque estable en entornos financieros ruidosos
- Estructura de archivos clara → los agentes de Cursor saben exactamente dónde crear cada módulo

**Negativas / trade-offs**:
- Backbone compartido actor-critic en PPO puede saturarse si las tareas de policy y value divergen mucho → separar backbones en fase avanzada
- SAC custom implica más código a mantener vs SB3 → justificado por control sobre reward shaping
- Sin CNN ni Attention en MVP → adecuado para estado tabular de 42 dims; añadir en fases futuras si se incorporan datos de imagen (L2 order book)

---

**Maintainer**: Alex / Claude  
**Revisado**: 2026-06-03
