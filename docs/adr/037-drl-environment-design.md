# ADR-037 — Diseño del Environment DRL (Estado, Acción, Reward)

**Status**: Accepted  
**Date**: 2026-06-03  
**Depende de**: ADR-036 (DRL-First)  
**Próximo**: ADR-038 (Arquitectura de redes policy/value)

---

## Contexto

El ADR-036 estableció DRL como arquitectura primaria. Este ADR especifica formalmente
los tres componentes del environment gym-compatible que el agente necesita para aprender:

```
Agente ──► acción a_t ──► Environment ──► estado s_{t+1}, reward r_t ──► Agente
```

Estas decisiones son las más críticas del sistema DRL: un estado mal diseñado hace
que el agente no pueda aprender; un reward mal diseñado hace que aprenda lo incorrecto.

---

## 1. ESTADO (Observation Space)

### Diseño

El estado `s_t ∈ ℝ^n` es un vector numérico normalizado que el agente observa en cada
paso de tiempo. Se divide en cuatro bloques:

```
s_t = [ mercado | régimen | portfolio | contexto ]
```

### Bloque 1 — Features de mercado (dim ≈ 30)

Características del instrumento en la barra actual. Todos normalizados (z-score rolling).

| Feature | Descripción | Fuente |
|---------|-------------|--------|
| `ret_1`, `ret_5`, `ret_20` | Retornos logarítmicos a 1, 5, 20 barras | OHLCV |
| `vol_realized_20` | Volatilidad realizada 20 barras (Garman-Klass) | OHLCV |
| `vol_z_60` | Z-score de volatilidad vs 60 barras | OHLCV |
| `rsi_14` | RSI normalizado ∈ [-1, 1] | técnico |
| `macd_signal` | MACD - señal, z-score | técnico |
| `atr_14` | ATR normalizado por precio | técnico |
| `bb_pct` | Posición en Bollinger Bands ∈ [0, 1] | técnico |
| `volume_z_20` | Z-score de volumen | OHLCV |
| `ob_imbalance` | Order book imbalance (bid - ask) / total | microstructura |
| `spread_bps` | Spread bid-ask en basis points | microstructura |
| `funding_z_60` | Z-score funding rate (solo perpetuals) | Crucix |
| `session_rth` | 1 si mercado en RTH, 0 si pre/post | calendar |

### Bloque 2 — Régimen de mercado (dim = 7)

Estado probabilístico del régimen actual. El agente sabe "en qué tipo de mercado está".

| Feature | Descripción | Fuente |
|---------|-------------|--------|
| `regime_prob_0..4` | Probabilidades GMM (5 componentes, suman 1) | context-engine |
| `regime_stability` | Estabilidad del régimen (0=cambiando, 1=estable) | context-engine |
| `vol_regime` | Régimen de volatilidad: {low=0, mid=0.5, high=1} | derivado |

### Bloque 3 — Estado del portfolio (dim = 5)

El agente conoce su situación actual. Sin esto, no puede razonar sobre costos de
oportunidad ni sobre el riesgo de aumentar posición.

| Feature | Descripción | Normalización |
|---------|-------------|---------------|
| `position` | Posición actual ∈ {-1, 0, +1} (discreta) o ∈ [-1, 1] (continua) | sin cambio |
| `unrealized_pnl_pct` | P&L no realizado como % del equity | / 0.10 (clip ±1) |
| `holding_bars` | Barras desde la última entrada | / max_holding |
| `daily_pnl_pct` | P&L del día como % del equity | / 0.05 (clip ±1) |
| `cash_ratio` | Cash disponible / equity total | sin cambio |

### Bloque 4 — Contexto macro (dim ≈ 8, opcional en MVP)

Información de horizonte más largo. Se añade en pasos posteriores al MVP.

| Feature | Descripción | Fuente |
|---------|-------------|--------|
| `vix_z` | Z-score del VIX | FRED |
| `dxy_z` | Z-score del DXY | FRED |
| `yield_curve_slope` | 10Y - 2Y yield | FRED |
| `btc_dominance` | BTC dominance (solo crypto) | on-chain |
| `xgb_proba_long` | P(long) calibrado de XGBoost baseline | research/models |
| `xgb_proba_short` | P(short) calibrado de XGBoost baseline | research/models |

> **Nota**: incluir la probabilidad de XGBoost como feature del estado permite al agente
> DRL usar el baseline como "consultor" sin depender de él para la decisión final.

### Dimensión total

| Bloque | Dim | MVP |
|--------|-----|-----|
| Mercado | 30 | ✅ |
| Régimen | 7 | ✅ |
| Portfolio | 5 | ✅ |
| Macro | 8 | Paso 2 |
| **Total MVP** | **42** | |
| **Total completo** | **50** | |

### Normalización

- **Z-score rolling** para features de precio/volumen: `(x - μ_60) / σ_60`
- **Clip** a ±3σ para robustez ante outliers
- **Sin normalización** para features ya en [0,1] (probabilidades GMM, position)
- **Anti-leakage**: la normalización usa solo datos hasta `t-1`, nunca look-ahead

---

## 2. ACCIÓN (Action Space)

### MVP — Discreta (DQN)

```python
action ∈ {0, 1, 2}
  0 = SELL  (o cerrar long si hay posición)
  1 = HOLD  (mantener posición actual)
  2 = BUY   (o cerrar short si hay posición)
```

Simple, estable, fácil de debuggear. El sizing lo decide el RiskGate externo
(Kelly fraccional sobre la confianza del agente).

### Objetivo — Continua (PPO / SAC)

```python
action ∈ [-1.0, 1.0]
  -1.0 = máximo short  (fracción Kelly negativa)
   0.0 = flat          (sin posición)
  +1.0 = máximo long   (fracción Kelly positiva)
```

El valor absoluto `|action|` determina el sizing: `qty = |action| × kelly_max × equity / price`.
El signo determina la dirección.

**Por qué no discretizar el sizing**: discretizar crea bordes artificiales (e.g., el agente
aprende "nunca usar 0.4, siempre 0.5 o 0.3") que no reflejan la realidad continua del mercado.

### Restricción de RiskGate

La acción del agente es una **intención**, no una orden directa. Pasa siempre por el RiskGate:

```
agente → action_t → RiskGate → OrderIntent (o rechazo) → AlpacaAdapter
```

Si el RiskGate rechaza (kill switch activo, cap superado, etc.), el agente recibe
`reward = 0` para ese paso y el estado no cambia. El agente aprende implícitamente
a no proponer acciones que el RiskGate rechazará.

---

## 3. REWARD (Función de recompensa)

### Diseño general

```
r_t = r_pnl + r_risk + r_cost + r_idle
```

Cada componente tiene un peso configurable. El diseño evita los problemas más comunes
de reward hacking en trading.

### Componente 1 — P&L realizado (r_pnl)

```python
r_pnl = pnl_t / equity_t
```

P&L del paso normalizado por equity. Usar ratio (no valor absoluto) hace el reward
independiente del tamaño de la cuenta — el agente aprende la política, no el monto.

**Solo P&L realizado**: incluir P&L no realizado crea incentivos perversos (el agente
mantiene posiciones perdedoras esperando recuperación). El P&L se realiza cuando se cierra
la posición o al final del episodio.

### Componente 2 — Penalización por riesgo (r_risk)

```python
r_risk = -λ_dd × max(0, dd_t - dd_threshold)
       - λ_vol × max(0, vol_realized_t - vol_target)

# Valores default:
# λ_dd = 2.0  (penaliza fuertemente el drawdown)
# dd_threshold = 0.02  (2% de tolerancia)
# λ_vol = 0.5
# vol_target = volatilidad histórica del instrumento
```

El agente aprende a gestionar el riesgo sin necesidad de reglas externas adicionales.
Complementa (no reemplaza) el RiskGate.

### Componente 3 — Costos de transacción (r_cost)

```python
r_cost = -c × |Δposition_t| × price_t
```

Donde `c = fee_bps / 10000` (maker o taker según el tipo de orden).
Incluir costos reales hace que el agente aprenda a no sobre-operar (churning).

### Componente 4 — Penalización por inactividad excesiva (r_idle)

```python
r_idle = -ε × (holding_bars > max_idle_bars)

# ε = 0.001 (penalización pequeña)
# max_idle_bars = 20
```

Sin esta penalización, el agente puede aprender la política trivial "nunca operar"
para evitar pérdidas. Una penalización pequeña por inactividad excesiva rompe este
equilibrio degenerado.

### Función completa

```python
def compute_reward(
    pnl_realized: float,
    equity: float,
    drawdown: float,
    vol_realized: float,
    vol_target: float,
    delta_position: float,
    price: float,
    fee_bps: float,
    holding_bars: int,
    # Hiperparámetros
    lambda_dd: float = 2.0,
    dd_threshold: float = 0.02,
    lambda_vol: float = 0.5,
    max_idle_bars: int = 20,
    idle_penalty: float = 0.001,
) -> float:

    r_pnl  = pnl_realized / equity
    r_risk = (
        -lambda_dd * max(0, drawdown - dd_threshold)
        - lambda_vol * max(0, vol_realized - vol_target)
    )
    r_cost = -(fee_bps / 10_000) * abs(delta_position) * price
    r_idle = -idle_penalty if holding_bars > max_idle_bars else 0.0

    return r_pnl + r_risk + r_cost + r_idle
```

### Episodio y horizonte temporal

```
episodio = N barras consecutivas de datos históricos
  N = 252 barras (≈ 1 año de datos diarios, o 1 semana de datos 5-min)

Al final del episodio:
  - P&L no realizado se marca a mercado y se incluye en el reward final
  - El environment hace reset() a un período histórico aleatorio
```

---

## 4. VALIDACIÓN (cómo saber si el reward funciona)

El reward está bien diseñado si el agente entrenado exhibe estas propiedades:

| Check | Señal de reward sano | Señal de reward roto |
|-------|----------------------|----------------------|
| Operaciones | El agente opera con frecuencia razonable | Nunca opera (r_idle muy pequeño) o opera en cada barra (c muy pequeño) |
| Sizing | Sizes variables según confianza | Siempre máximo o siempre mínimo |
| Drawdown | Reduce posición en drawdown | Aumenta posición en drawdown (espera recuperación) |
| Regime | Diferente comportamiento por régimen | Comportamiento idéntico en todos los regímenes |
| vs XGBoost | DSR agente > DSR XGBoost en OOS | DSR agente ≤ DSR XGBoost (agente no aprende nada) |

---

## 5. IMPLEMENTACIÓN — Interfaz gym

```python
import gymnasium as gym
import numpy as np

class TradingEnvironment(gym.Env):
    """
    Environment DRL para trading algorítmico.

    Parámetros
    ----------
    data : pd.DataFrame
        Features + OHLCV del instrumento. Index = DatetimeIndex UTC.
    config : EnvironmentConfig
        Parámetros del reward, sizing, costos.
    mode : Literal["train", "eval"]
        En "eval" el reward no incluye r_idle y el logging es más detallado.
    """

    # MVP (DQN)
    action_space = gym.spaces.Discrete(3)       # {0=SELL, 1=HOLD, 2=BUY}

    # Target (PPO/SAC)
    # action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))

    observation_space = gym.spaces.Box(
        low=-3.0, high=3.0, shape=(42,), dtype=np.float32
    )

    def reset(self, seed=None) -> tuple[np.ndarray, dict]: ...
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]: ...
    def render(self) -> None: ...   # P&L curve + position overlay
```

Archivo target: `research/envs/trading_env.py`

---

## 6. HIPERPARÁMETROS A TUNEAR (Optuna)

```yaml
reward:
  lambda_dd:      [0.5, 5.0]   # penalización drawdown
  lambda_vol:     [0.0, 2.0]   # penalización volatilidad
  idle_penalty:   [0.0, 0.01]  # penalización inactividad
  dd_threshold:   [0.01, 0.05] # tolerancia de drawdown

environment:
  episode_length: [126, 504]   # barras por episodio
  max_idle_bars:  [10, 50]
```

Los hiperparámetros del reward se tunean junto con los de la red (ver ADR-038).

---

## Consecuencias

**Positivas**:
- El estado captura mercado + régimen + portfolio en un vector compacto (42 dims)
- El reward alinea el objetivo del agente con el objetivo real de trading
- La inclusión de XGBoost proba como feature permite transferencia de conocimiento
- La interfaz gym estándar permite usar cualquier librería DRL (Stable-Baselines3, CleanRL)

**Negativas / trade-offs**:
- El bloque macro (8 dims) añade latencia por queries a FRED → MVP lo excluye
- El reward tiene 4+ hiperparámetros → Optuna obligatorio, no tuning manual
- "Solo P&L realizado" puede crear episodios donde el agente nunca cierra → mitigado por r_idle y episode end mark-to-market

---

## Próximo ADR

**ADR-038**: Arquitectura de redes policy/value — capas, dimensiones, activaciones,
inicialización de pesos para el agente DRL.

---

**Maintainer**: Alex / Claude  
**Revisado**: 2026-06-03
