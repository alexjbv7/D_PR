# RISK_PARAMS — Parámetros de riesgo (valores actuales en código)

> **Generado desde el repositorio** (`quant_bot/`, 2026-05-18).  
> Fuente de verdad operativa: **defaults en código + env vars + seeds SQL**.  
> `CLAUDE.md` §12 describe objetivos de diseño; donde difiere, se marca **SPEC** vs **CODE**.

---

## Índice

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Research — sizing por trade](#2-research--sizing-por-trade)
3. [Research — walk-forward / backtest](#3-research--walk-forward--backtest)
4. [Platform — strategy-orchestrator](#4-platform--strategy-orchestrator)
5. [Platform — context-engine (market state)](#5-platform--context-engine-market-state)
6. [Platform — execution-engine](#6-platform--execution-engine)
7. [Base de datos (seeds)](#7-base-de-datos-seeds)
8. [Divergencias conocidas](#8-divergencias-conocidas)
9. [Variables de entorno](#9-variables-de-entorno)

---

## 1. Resumen ejecutivo

| Capa | Rol principal | Kelly / trade risk | DD diario | DD portfolio |
|------|---------------|-------------------|-----------|--------------|
| **Research** | Backtest / WF | 25% Kelly, cap 2% equity/trade | 3% pausa | 10% soft / 20% hard (`DrawdownGuard`) |
| **Orchestrator** | Bot + allocation | 25% Kelly (allocation), 2% `risk_per_trade` UI | — | 5% soft / 10% hard (allocation); 10% halt (orchestrator loop) |
| **Execution-engine** | Pre-submit gate | — | 3% kill | 5% per symbol, 50% per venue |
| **Context-engine** | Señales macro | — | — | Score 0–100; block longs >70, defensive >85 |

---

## 2. Research — sizing por trade

### 2.1 `KellyAtrSizer` — default producción backtest

Archivo: `research/risk/kelly.py` (`KellyAtrSizer` dataclass).

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| `kelly_fraction` | **0.25** | Quarter Kelly |
| `max_risk_pct` | **0.02** | Cap duro: ≤2% equity por trade |
| `min_risk_pct` | **0.001** | &lt;0.1% → skip (posición simbólica) |
| `min_edge` | **0.0** | EV &gt; 0 suficiente |
| `daily_loss_pct_pause` | **0.03** | Pérdida intradía ≥3% → no nuevas posiciones |
| `rr_ratio` | **1.5** | Default; si `atr_tp_mult` set → `tp/sl` |
| `atr_sl_mult` | **2.0** | Stop a 2×ATR |
| `atr_tp_mult` | **3.0** | TP a 3×ATR → R:R efectivo **1.5** |

Función pura `kelly_fraction_binary(..., kelly_fraction=0.25)` — mismo default.

### 2.2 `DynamicRRManager` — R:R dinámico por P(win)

Archivo: `research/risk/dynamic_rr.py`.

| Parámetro | Valor |
|-----------|-------|
| `atr_sl_mult` | **2.0** |
| `rr_min` | **1.2** |
| `rr_max` | **2.5** |
| `p_low` | **0.45** |
| `p_high` | **0.75** |
| `shape` | **`sigmoid`** |

`compute_dynamic_rr()` usa los mismos defaults.

`compute_full_sizing()` (integración Kelly+RR): `kelly_fraction=0.25`, `max_risk_pct=0.02`, `min_risk_pct=0.001`.

### 2.3 `ATRRiskSizer` — sizing fijo (legacy / multi-asset)

Archivo: `research/risk/sizing_multi_asset.py`.

| Parámetro | Valor |
|-----------|-------|
| `risk_pct` | **0.005** (0.5% equity/trade) |
| `atr_stop_mult` | **2.0** |
| `daily_loss_pct_pause` | **0.03** |

### 2.4 `StopLoss` / `StopManager`

Archivo: `research/risk/management.py`.

| Parámetro | Valor |
|-----------|-------|
| `initial_atr_mult` | **2.0** |
| `take_profit_atr_mult` | **4.0** (R:R ≈ 2:1 vs SL) |
| `trailing` | **True** |

### 2.5 `DrawdownGuard` — portfolio (research)

Archivo: `research/risk/management.py`.

| Parámetro | Valor | Efecto |
|-----------|-------|--------|
| `soft_dd_threshold` | **0.10** (10%) | Multiplicador sizing × **0.5** |
| `hard_dd_threshold` | **0.20** (20%) | Pausa total (hasta ~99% recuperación del peak) |
| `soft_dd_multiplier` | **0.5** | |

`IntegratedRiskManager`: `target_vol=0.15`, `max_position_pct=0.95`, `soft_dd=0.10`, `hard_dd=0.20`.

### 2.6 Helpers `management.py`

| Función | Default relevante |
|---------|-------------------|
| `kelly_fractional_size` | `kelly_fraction=0.25`, `cap=1.0` |
| `vol_target_size` | `target_volatility=0.15`, `leverage_cap=1.0` |
| `fixed_fraction_size` | `fraction=0.95` |

### 2.7 `BayesianSizerConfig`

Archivo: `research/risk/bayesian_sizer.py`.

| Parámetro | Valor |
|-----------|-------|
| `combination` | **`product`** |
| `smoothing` | **1.0** (Laplace) |
| `min_samples` | **20** |
| `prior_weight` | **0.3** (modo weighted) |
| `clip_eps` | **1e-4** |

### 2.8 `EntryFilter` — umbral de entrada ML

Archivo: `research/models/entry_filter.py`.

| Parámetro | Valor |
|-----------|-------|
| `min_coverage` | **0.05** |
| `n_thresholds` | **50** |
| `fallback_threshold` | **0.45** |
| `min_samples_to_optimize` | **60** |
| `min_trades` | **10** |
| `symmetric` | **True** |

Grid search: `t_min = 1/K + 0.05`, `t_max = 0.90`.

---

## 3. Research — walk-forward / backtest

### 3.1 `WalkForwardConfig`

Archivo: `research/models/walk_forward_runner.py`.

| Grupo | Parámetro | Valor |
|-------|-----------|-------|
| WF splits | `train_size` | **252** |
| | `test_size` | **63** |
| | `embargo` | **5** |
| | `expanding` | **False** |
| Calibración | `calib_frac` | **0.20** |
| | `calib_method` | **`sigmoid`** |
| Entry filter | `filter_symmetric` | **True** |
| | `filter_min_coverage` | **0.05** |
| | `filter_n_thresholds` | **50** |
| Kelly + RR | `kelly_fraction` | **0.25** |
| | `max_risk_pct` | **0.02** |
| | `rr_min` / `rr_max` | **1.2** / **2.5** |
| | `rr_p_low` / `rr_p_high` | **0.45** / **0.75** |
| | `rr_shape` | **`sigmoid`** |
| | `atr_sl_mult` | **2.0** |

Pipelines de ejemplo (`research/examples/pipeline_ml_real_data.py`, `pipeline_hyperopt.py`): `kelly_fraction=0.25`, `max_risk_pct=0.02`.

---

## 4. Platform — strategy-orchestrator

### 4.1 Env vars (startup)

Archivo: `platform/services/strategy-orchestrator/app/main.py`.

| Variable | Default | Uso |
|----------|---------|-----|
| `DEFAULT_CAPITAL` | **10000** | USD |
| `MAX_POSITIONS` | **5** | |
| `RISK_PER_TRADE` | **0.02** | 2% |
| `MAX_LEVERAGE` | **2.0** | |
| `MAX_DRAWDOWN_PCT` | **0.10** | 10% — halt en loop si DD cache &gt; umbral |

### 4.2 `BotConfig` (Pydantic API)

Mismos defaults que env al crear config en lifespan:

| Campo | Default | Rango validación |
|-------|---------|------------------|
| `mode` | **`paper`** | `paper` \| `live` |
| `max_positions` | **3** | 1–20 |
| `max_leverage` | **1.5** | 0.1–10 |
| `risk_per_trade` | **0.02** | 0.001–0.10 |
| `max_drawdown` | **0.10** | 0.01–0.50 |
| `total_capital` | **10000** (vía env) | opcional |

Kill switch: flag global `_kill_switch` (in-memory), endpoint `POST /kill-switch/{on|off}`.

### 4.3 `AllocationConfig` / `AllocationEngine`

Archivo: `platform/services/strategy-orchestrator/app/allocation_engine.py`.

| Parámetro | Valor |
|-----------|-------|
| `total_capital` | **10000** USD |
| `kelly_fraction` | **0.25** |
| `target_vol` | **0.15** (anualizada) |
| `max_leverage` | **2.0** |
| `max_single_alloc` | **0.40** (40% a una estrategia) |
| `min_single_alloc` | **0.05** (debajo → skip) |
| `dd_soft_brake` | **0.05** (5%) |
| `dd_hard_brake` | **0.10** (10%) → 0 notional |
| `corr_threshold` | **0.70** |
| `risk_free_rate` | **0.05** |

Vol scaling: `scale = min(target_vol / realized_vol, 2.0)`; floor `realized_vol ≥ 0.01`.

### 4.4 Frontend `BotConfigPanel` (límites UI, no defaults de servidor)

Archivo: `platform/frontend/src/components/BotConfigPanel.tsx`.

| Control | Min | Max | Step |
|---------|-----|-----|------|
| Max positions | 1 | 10 | 1 |
| Max leverage | 0.5 | 5.0 | 0.5 |
| Risk per trade | 0.5% | 5% | 0.5% |
| Max drawdown | 5% | 30% | 1% |
| Total capital | $1,000 | $1,000,000 | $1,000 |

Fallback display capital: **10000** si `total_capital` null.

---

## 5. Platform — context-engine

### 5.1 `MarketStateEngine` — composite risk score

Archivo: `platform/services/context-engine/app/market_state_engine.py`.

**Umbrales de acción**

| Condición | Efecto |
|-----------|--------|
| `composite_risk_score > 70` | `allow_new_longs = False` |
| `composite_risk_score > 85` | `defensive_mode = True` |

**Pesos `_WEIGHTS` (suma máx. teórica ~100)**

| Clave | Puntos |
|-------|--------|
| `squeeze_critical` | 30 |
| `squeeze_high` | 20 |
| `squeeze_medium` | 10 |
| `squeeze_low` | 5 |
| `anomaly_critical` | 20 |
| `anomaly_high` | 10 |
| `macro_strong_bearish` | 20 |
| `macro_bearish` | 10 |
| `recession_high` (rec_prob ≥ 0.65) | 15 |
| `recession_mid` (rec_prob ≥ 0.40) | 8 |
| `regime_bearish` (bear_trend / crisis) | 10 |

**Polymarket (hardcoded en `build`)**

| rec_prob mercado | Puntos |
|------------------|--------|
| ≥ 0.60 | +8 |
| ≥ 0.40 | +4 |

Redis TTL `market:state`: **60 s**.

### 5.2 `AnomalyDetector`

Archivo: `platform/services/context-engine/app/anomaly_detector.py`.

| Parámetro | Default |
|-----------|---------|
| `price_z_thresh` | **4.0** |
| `volume_mult` | **5.0** |
| `spread_mult` | **4.0** |
| `funding_z_thresh` | **3.0** |
| `window` | **20** barras |

---

## 6. Platform — execution-engine

### 6.1 `RiskConfig` / `RiskGate`

Archivo: `platform/services/execution-engine/app/risk_gate.py`.

| Parámetro | Default | Check order |
|-----------|---------|-------------|
| `per_symbol_cap_pct` | **0.05** (5%) | #3 |
| `per_venue_cap_pct` | **0.50** (50%) | #4 |
| `daily_dd_kill_pct` | **0.03** (3%) | #2 — `pnl_day/equity < -3%` |
| `min_cash_buffer_pct` | **0.10** (10%) | #6 — compras |
| `require_paper` | **True** | #1 — bloquea cuentas live |

También: **#5** `market_calendar.is_open()` para equities fuera de RTH.

### 6.2 `Settings` (env → `RiskConfig`)

Archivo: `platform/services/execution-engine/app/settings.py`.

| Env / field | Default |
|-------------|---------|
| `risk_per_symbol_cap_pct` | **0.05** |
| `risk_per_venue_cap_pct` | **0.50** |
| `risk_daily_dd_kill_pct` | **0.03** |
| `risk_min_cash_buffer_pct` | **0.10** |
| `risk_require_paper` | **True** |
| `redis_kill_switch_key` | **`execution:kill_switch`** |

### 6.3 `signal_translator`

Archivo: `platform/services/execution-engine/app/signal_translator.py`.

| Parámetro | Valor |
|-----------|-------|
| `_MIN_NOTIONAL` | **10** USD — skip si `kelly × equity / price` &lt; 10 |

Sizing: `notional = position_size × equity` (campo señal = fracción Kelly).

### 6.4 `Reconciler`

Archivo: `platform/services/execution-engine/app/settings.py` + `reconciler.py`.

| Parámetro | Default |
|-----------|---------|
| `reconciler_interval_sec` | **60** |
| `reconciler_failure_threshold` | **3** ciclos con discrepancia → trip kill switch callback |

### 6.5 Kill switch (execution-engine)

- REST: `POST /api/kill_switch/{trip|reset}`
- Estado: `AppState.kill_switch_tripped` (in-memory)
- Redis key configurada pero trip principal vía API/reconciler

---

## 7. Base de datos (seeds)

Archivo: `platform/infra/sql/schema.sql`.

### 7.1 `bot.configurations` (column defaults)

| Columna | DEFAULT |
|---------|---------|
| `max_positions` | **3** |
| `max_leverage` | **1.0** |
| `risk_per_trade` | **0.02** |
| `max_drawdown` | **0.10** |
| `kill_switch` | **FALSE** |

### 7.2 Seed «Default Paper Config»

| Campo | Valor |
|-------|-------|
| `max_positions` | 3 |
| `max_leverage` | 1.5 |
| `risk_per_trade` | 0.02 |
| `max_drawdown` | 0.10 |
| `total_capital` | 10000 |
| `active_strategies` | `momentum_ml`, `mean_reversion_funding` |

### 7.3 `bot.strategies.risk_params` (JSON por estrategia)

| Estrategia | `risk_per_trade` | `max_leverage` | Otros |
|------------|------------------|----------------|-------|
| `momentum_ml` | 0.02 | 2.0 | `stop_atr_mult: 2.0` |
| `mean_reversion_funding` | 0.015 | 1.5 | `hard_stop_pct: 0.03` |
| `regime_adaptive` | 0.025 | 1.0 | `recession_halt: true` |
| `whale_follow` | 0.02 | 1.5 | `stop_atr_mult: 3.0` |

Params ML relacionados: `min_p_win: 0.58` (momentum_ml), `min_confidence: 0.6` (regime_adaptive), funding `entry_z: 2.5`.

---

## 8. Divergencias conocidas

| Tema | CLAUDE.md §12 (SPEC) | Código actual |
|------|----------------------|---------------|
| DD diario kill | 3% | **3%** — alineado (`RiskGate`, `KellyAtrSizer`, `ATRRiskSizer`) |
| DD semanal / mensual | 7% / 12% | **No implementado** en servicios |
| Per-symbol cap | 5% | **5%** `RiskGate`; orchestrator usa `risk_per_trade` 2% por trade (distinto concepto) |
| Altcoin sector cap 30% | SPEC | **No implementado** |
| Kelly producción | 0.10–0.25 | **0.25** en todos los paths principales |
| Max leverage crypto | 3x SPEC | **2.0** orchestrator allocation; **1.5** bot config API default |
| Drawdown portfolio | — | **Tres escalas distintas**: research `DrawdownGuard` 10%/20%; allocation 5%/10%; orchestrator halt **10%** |
| Correlación | corr &gt; 0.8 → una posición SPEC | allocation **`corr_threshold=0.70`** |
| Kill switch latency 5s | SPEC | **No** en execution-engine (solo reconciler + manual API) |
| `research/risk/portfolio.py` | listado CLAUDE §12.9 | **Pendiente** — no existe en repo |

---

## 9. Variables de entorno

### Strategy-orchestrator

```
DEFAULT_CAPITAL=10000
MAX_POSITIONS=5
RISK_PER_TRADE=0.02
MAX_LEVERAGE=2.0
MAX_DRAWDOWN_PCT=0.10
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
REDIS_URL=redis://localhost:6379/0
POSTGRES_DSN=postgresql://trading:trading@localhost:5432/trading_db
```

### Execution-engine (riesgo)

```
RISK_PER_SYMBOL_CAP_PCT=0.05
RISK_PER_VENUE_CAP_PCT=0.50
RISK_DAILY_DD_KILL_PCT=0.03
RISK_MIN_CASH_BUFFER_PCT=0.10
RISK_REQUIRE_PAPER=true
REDIS_KILL_SWITCH_KEY=execution:kill_switch
RECONCILER_INTERVAL_SEC=60
RECONCILER_FAILURE_THRESHOLD=3
```

(Pydantic-settings: nombres de campo en mayúsculas con prefijo implícito del nombre del atributo.)

---

## Mantenimiento

Al cambiar un default en código:

1. Actualizar la fila correspondiente en este documento.
2. Si es decisión arquitectónica, considerar ADR en `docs/adr/`.
3. No duplicar valores en un segundo archivo de config hasta existir un loader único (objetivo futuro: `RiskConfig` unificado research ↔ platform).

**Referencias cruzadas:** `research/docs/RISK_ENGINE.md` (narrativa), `CLAUDE.md` §12 (especificación objetivo).
