# ADR-042 — Arquitectura objetivo: librerías, contratos y migración

> **Status: Proposed (documento de referencia / north star).**
> Formaliza el rediseño de `DIAGNOSTICO_FABLE5_ARQUITECTURA.md §2` en una
> especificación concreta: corte de librerías, interfaces, dependencias y un
> **mapa de migración por fases** desde el monorepo actual.
>
> **Regla de oro de este ADR:** es un *plano*, no una orden de construir. La
> implementación del núcleo (`portfolio`) **espera a que el Nivel 1 cruce** (un
> agente con DSR deflactado > 0.4 que bata buy-and-hold). Construir antes = el
> error de "escalar prematuramente" que el diagnóstico identifica como riesgo #1.

---

## 1. Re-corte de la propuesta de librerías (A–F → corte correcto)

La propuesta original (A Estrategias · B Modelos · C Activos · D Riesgo · E
Interpretabilidad · F ML/DL) mezcla ejes no ortogonales. Corte corregido:

| Propuesta original | Problema | Destino correcto |
|--------------------|----------|------------------|
| A. Estrategias + B. Modelos + F. ML/DL | Estrategia y modelo NO son separables; un agente = hipótesis + modelo + features | **Fusionar en `alpha/`** organizado por *hipótesis de alfa* |
| C. Activos | No es una librería de clases de activo; es un problema de **datos** | **`data/`** (point-in-time, ingesta, calidad) |
| D. Gestión de riesgo | Correcto, pero **externo al agente** | **`risk/`** (RiskGate) |
| E. Interpretabilidad | No es librería; es *concern transversal* | **`interpretability/`** como protocolo `Explainer`, cross-cutting |
| (ausente) | El núcleo de un fondo multiagente | **`portfolio/`** (ensemble + allocation + sizing) ← FALTA |
| (ausente) | Sin esto no hay producción seria | **`governance/`** (registry, drift, MLOps) |

---

## 1.1 Estilos de trading y edges por clase de activo (el contenido real de `alpha/`)

El bot **debe** soportar múltiples estilos (intraday, swing, position, arbitraje) y
**edges únicos por clase** (CRYPTO / STOCKS / FX). Esto NO es una librería aparte: es
exactamente lo que vive en `alpha/`. Cada combinación **(estilo × clase × edge)** es un
`AlphaAgent` distinto, con su propia hipótesis, `FeatureProvider`, `EnvironmentConfig`
(barra, fee model, calendario, apalancamiento) y **su propio gate de promoción**. La
capacidad multi-estilo es **arquitectónica** (el contrato la soporta), no un mega-agente
omnisciente.

### Estilos (definen el horizonte → barra y `episode_length`)

| Estilo | Barra típica | Horizonte | Qué cambia respecto a hoy |
|--------|--------------|-----------|---------------------------|
| **Intraday** | 1m–15m | minutos–horas | microestructura, order book; requiere **tick/LOB data** |
| **Swing** | 1h–4h | días | momentum / mean-reversion multi-día |
| **Position** | diario–semanal | semanas–meses | tendencia/carry — **≈ lo que corres hoy** (SPY diario) |
| **Arbitraje** | s–min | variable | market-neutral; requiere **multi-venue / pares cointegrados** |

### Edges únicos por clase de activo

| Clase | Edges característicos | Features clave | Dato que falta hoy |
|-------|----------------------|----------------|--------------------|
| **CRYPTO** | funding mean-reversion (perps), liquidation cascades, basis spot-perp, on-chain flows, 24/7 | `funding_z`, OI, liquidations, exchange netflow | funding/OI/LOB (Binance/Bybit WS) |
| **STOCKS** | earnings drift, rotación sectorial, gaps overnight, order-book imbalance, short-borrow | earnings, sector, `ob_imbalance`, gap | LOB/tick, fundamentals point-in-time |
| **FX** | liquidez por sesión (Asia/EU/US), carry, sensibilidad macro/banca central, 24/5 con gaps de fin de semana | flags de sesión, diferencial de tasas, DXY, vol por sesión | feed FX (OANDA/IB), calendario macro |

### La honestidad (qué soporta tu dato HOY)

- Hoy solo tienes **diario IEX (equities)** + **crypto diario Alpaca**. Eso únicamente
  soporta **Position / Swing direccional**. **Intraday y Arbitraje requieren datos que
  NO tienes** (tick/LOB, multi-venue, funding) → son **Nivel 3+**, gated por la capa
  `data/`. No los construyas antes de tener el dato y un agente con edge.
- El contrato del env **ya reserva** `ob_imbalance`, `spread_bps`, `funding_z_60` — hoy
  están en **0** (placeholders) precisamente porque faltan esos datos. El día que
  conectes el feed, el agente intraday/crypto los consume **sin cambiar la
  arquitectura**. Esto es el contrato haciendo su trabajo.
- Cada agente declara `asset_class` y `style` en su `alpha_hypothesis_id`; el
  `EnvironmentConfig` parametriza barra, fee model, **calendario** (RTH vs 24/7 vs 24/5)
  y apalancamiento por clase. **Un solo env, configs distintas** — no un env por activo.

### Anti-patrón (vinculante)
❌ Un mega-agente "que hace intraday + swing + arbitraje en crypto + stocks + FX".
✅ N agentes especializados por (estilo × clase × edge), cada uno con su gate y sus
datos; el `portfolio/` los combina. La diversificación de estilos/activos es justamente
de donde sale el Sharpe de cartera > Sharpe individual — pero solo si cada pata pasa su
propio gate deflactado primero.

---

## 2. Layout de paquetes objetivo

```
quant/
├── data/            # ingesta multi-venue, point-in-time, survivorship, calidad
│   ├── sources/         alpaca, ccxt, yfinance, fred…
│   ├── pit/             corrección point-in-time + corporate actions
│   └── quality/         data quality monitor, drift de datos
├── features/        # state space causal versionado (ya existe parcialmente)
├── alpha/           # AGENTES enchufables — uno por HIPÓTESIS de alfa
│   ├── base.py          AlphaAgent Protocol + Signal
│   ├── agents/          drl_dqn, drl_ppo, xgb_directional, …
│   └── models/          backbones reutilizables (ResMLP, GBM, TFT*…)
├── portfolio/       # ★ EL NÚCLEO QUE FALTA — Nivel 2
│   ├── ensemble.py      meta-labeling, stacking
│   ├── allocator.py     capital allocation + correlación + deflación de selección
│   └── sizing.py        vol-target · frac-Kelly · CVaR · regime (stack del diagnóstico)
├── risk/            # RiskGate: límites duros, kill switch (externo a alpha)
├── execution/       # smart routing, TWAP/VWAP, slippage, reconciliación
├── interpretability/# Explainer Protocol + impls por tipo de modelo (cross-cutting)
├── research/        # ★ GATEKEEPER — walk-forward, DSR deflactado, ablation
└── governance/      # model registry, drift, retraining gated, MLOps, audit
```
`(*)` = experimental, solo si los datos lo justifican (ver diagnóstico F.4).

---

## 3. Contratos (interfaces) — lo único construible HOY sin riesgo

Son **tipos**, no lógica. Construirlos ahora cuesta poco, no toca el path de
riesgo, y fuerza el desacoplamiento correcto. Es lo único del rediseño que vale
adelantar — y solo si no distrae de cerrar el Nivel 1.

```python
@dataclass(frozen=True)
class Signal:
    direction: int                 # {-1, 0, +1}  (o ∈[-1,1] cuando haya sizing continuo)
    p_win: float                   # probabilidad calibrada
    horizon: int
    confidence: float
    alpha_hypothesis_id: str       # la TESIS que el agente encarna
    model_version: str
    feature_set_hash: str

class AlphaAgent(Protocol):
    hypothesis_id: str
    def predict(self, state: FeatureVector) -> Signal: ...
    #   NO devuelve tamaño ni orden. Solo señal.

class FeatureProvider(Protocol):
    def build(self, raw: MarketData, train_idx: Index) -> FeatureVector: ...
    #   fit causal/anti-leakage (GMM por fold ya implementado, ADR-040)

class CapitalAllocator(Protocol):
    def allocate(self, signals: list[Signal], cov: Matrix,
                 track: dict[str, Performance]) -> dict[str, float]: ...
    #   consciente de correlación Y de deflación de selección (multiple testing)

class PositionSizer(Protocol):
    def size(self, weight: float, vol_forecast: float,
             regime: Regime, edge_posterior: Distribution) -> float: ...

class RiskGate(Protocol):
    def check(self, intent: OrderIntent) -> RiskDecision: ...  # allow/deny + caps

class Explainer(Protocol):
    def explain(self, model: Any, state: FeatureVector) -> Explanation: ...
    #   selección por tipo de modelo: tabular→SHAP, policy→XRL, transformer→attention

class PromotionGate(Protocol):
    def evaluate(self, agent: AlphaAgent, data: MarketData) -> GateResult: ...
    #   walk-forward + DSR deflactado (ADR-040 ya lo implementa)
```

**Contrato duro:** el flujo es `AlphaAgent → Signal → Ensemble → Allocator →
Sizer → RiskGate → Execution`. Un agente **nunca** toca capital, sizing ni riesgo.
Esto permite intercambiar DQN ↔ PPO ↔ XGBoost ↔ TFT sin tocar cartera/ejecución.

---

## 3.1 Modularidad — el agente es plug-and-play para CUALQUIER modelo

> Característica **central** de la arquitectura: hoy hay XGBoost y DQN; mañana
> habrá PPO, SAC, LSTM, TFT, GNN. **Agregar un modelo nuevo = UNA clase
> adaptadora; cero cambios en el resto del sistema.** (Patrón Strategy +
> principio abierto/cerrado + inversión de dependencia.)

### Cómo se integra cada familia de modelo

| Familia | Ejemplos | Qué consume de `MarketContext` |
|---------|----------|--------------------------------|
| **Tabular** | XGBoost, LightGBM, MLP | `features` (FeatureVector plano) |
| **RL** | DQN (hoy), PPO, SAC | `features` + `portfolio` (obs del env) |
| **Secuencial** | LSTM, TFT, TCN, N-BEATS | `window` (ventana de barras/features) |
| **Grafo** | GNN | construye el grafo desde `window` / fuentes externas |

### Receta para añadir un modelo (5 pasos, cero ramificación del sistema)

1. Entrena/carga el modelo en su propia capa (no toca contratos).
2. Escribe `MiModeloAlphaAgent` implementando `AlphaAgent`: atributo `hypothesis`
   + método `predict(context: MarketContext) -> TradeSignal`.
3. Dentro del adapter, extrae de `MarketContext` lo que tu modelo necesita
   (tabular→`features`, secuencial→`window`, grafo→construye) y mapea la salida a
   `TradeSignal`.
4. Valídalo con el **mismo gate** (ADR-040). Si pasa, el `portfolio/` lo enchufa.
5. **Cero cambios** en `portfolio/`, `risk/`, `execution/` ni el gate.

### Qué permanece constante vs qué varía

- **Constante (el contrato):** `MarketContext` (entrada), `TradeSignal` (salida),
  `AlphaHypothesis`. Toda la arquitectura depende de estos, no del modelo.
- **Varía (interior del agente):** arquitectura del modelo, qué features extrae,
  la librería (torch / sklearn / lightgbm / dgl…).

### Límite honesto y cómo se evoluciona

`MarketContext` ya es model-agnóstico (tabular/secuencial/grafo). Si un modelo
futuro necesita una **fuente que el contexto no expone** (p.ej. order book
completo, grafo de transacciones on-chain), se **extiende `MarketContext` de forma
aditiva** (campo opcional nuevo) — sin romper agentes existentes (backwards-compat
de schema, §1.4). Evolución controlada, no ruptura.

**Anti-patrón prohibido:** ❌ ramificar el resto del sistema por tipo de modelo
(`if model_type == "dqn": ...`). Toda la variabilidad vive **dentro del adapter**,
detrás del Protocol. Si ves un `if` por tipo de modelo fuera de `alpha/`, el diseño
se rompió.

### Módulos autocontenidos + las DOS capas de riesgo

Es una **librería**: cada agente/módulo es **autocontenido** — trae su propio
`StrategyConfig` (fees/rates de SU mercado + stops/targets intrínsecos + parámetros
de la estrategia). Importas un módulo y viene completo: estrategia, costos y riesgo
intrínseco propios.

Esto exige separar **dos capas de riesgo** que NO deben confundirse:

| Capa | Quién la posee | Qué contiene |
|------|----------------|--------------|
| **Riesgo intrínseco** (de la estrategia) | el **AGENTE** (`StrategyConfig`) | stop/target/holding que *definen* la estrategia (un breakout sin su stop no es la estrategia); el **fee model** de su clase/venue |
| **Riesgo de firma** (de cartera) | **EXTERNO** (`RiskGate` / `Allocator` / `Sizer`) | caps por símbolo/sector, kill switch, drawdown, leverage, **sizing de capital** |

**Regla:** el agente **PROPONE** (su stop/target viajan en la `TradeSignal`); la
firma **DISPONE** (el `RiskGate` puede reducir/denegar; el `Sizer` decide el
capital). El agente **nunca** puede subir su propio sizing ni saltarse un cap de
firma. Así se respeta ADR-009 *y* se cumple tu requisito de módulos autocontenidos.

> Por qué importa: un fee model equivocado = **edge fantasma** (lo viviste con el
> bug de `r_cost·price`). Que cada módulo traiga el fee correcto de SU mercado
> —crypto perp (maker/taker + funding) ≠ equity (comisión + borrow) ≠ FX (spread por
> sesión)— evita comparar peras con manzanas en el gate.

---

## 4. DAG de dependencias (sin ciclos)

```
data → features → alpha → portfolio → risk → execution
                   │          │
                   └──────────┴──→ research (gatekeeper, depende de todo, lo usa nadie en prod)
governance observa todo (registry/drift)         interpretability es transversal (depende de alpha/models)
```
Reglas: `alpha` no importa `portfolio` (inversión de dependencia vía `Signal`);
`research` y `governance` dependen hacia abajo pero **nada de producción depende de
ellos**; `interpretability` depende de `alpha/models` pero no al revés.

---

## 5. Mapa de migración desde el código actual

| Hoy | Va a | Cambio |
|-----|------|--------|
| `research/data/drl_dataset.py`, `data/alpaca_bars.py` | `data/sources/` + `features/` | separar fetch (data) de features |
| `research/features/*`, `regime_gmm.py` | `features/` | ya casi está |
| `research/models/drl/*`, `models/zoo.py` (XGBoost) | `alpha/agents/` + `alpha/models/` | envolver cada uno tras `AlphaAgent`/`Signal` |
| `research/models/dsr_gate.py`, `walk_forward_runner.py`, `validation.py` | `research/` | ya es el gatekeeper; renombrar paquete |
| `research/risk/{kelly,bayesian_sizer,dynamic_rr}.py` | `risk/` + `portfolio/sizing.py` | sizing→portfolio, límites→risk |
| `platform/services/execution-engine/*` | `execution/` | ya existe, integrar contrato |
| **(no existe)** `portfolio/{ensemble,allocator}.py` | `portfolio/` | **construcción nueva — Nivel 2** |
| **(no existe)** `data/pit/`, `data/quality/` | `data/` | **construcción nueva — Nivel 2/3** |

> El monorepo ya tiene ~70% de las piezas; lo que **falta de verdad** es
> `portfolio/` (allocation + sizing-stack + ensemble) y la capa `data/pit`.

---

## 6. Qué construir ahora vs qué esperar

| Trabajo | ¿Cuándo? | Por qué |
|---------|----------|---------|
| Contratos `Signal`/`AlphaAgent` + envolver DQN y XGBoost | **Opcional ahora** (bajo coste, bajo riesgo) | Hace los agentes enchufables; prerequisito de todo. Solo si NO distrae del Nivel 1 |
| Interpretabilidad del agente actual (policy behavior) | **Ahora** | Ya es útil — explica el "se queda flat" |
| `portfolio/` (allocator + sizing-stack + deflación de selección) | **Esperar Nivel 1** | Es el núcleo difícil; sin un agente con edge no hay qué asignar |
| `data/pit` + quality | **Nivel 2/3** | Necesario al ir multi-asset serio |
| Migración completa de paquetes | **Nivel 2** | Refactor grande; no antes de tener edge |
| TFT/GNN/AutoML/NAS | **Experimental / no** | Diagnóstico F.4/F.6 |

---

## 7. Asignación de trabajo (modelo · esfuerzo)

| Paquete de trabajo | Quién | Esfuerzo |
|--------------------|-------|----------|
| Diseño de criterios (deflación, allocation math, sizing-stack) | **Opus + revisión humana** | Muy alto — riesgo de capital, NO delegar |
| Contratos `Signal`/interfaces (esqueleto de tipos) | Composer / Sonnet | Bajo (mecánico) |
| Interpretabilidad del agente (XRL: posiciones por régimen) | Opus diseña → Sonnet/Composer implementa | Medio |
| **`portfolio/` núcleo** (allocator correlación+deflación, sizer stack) | **Fable 5** (build) tras diseño de Opus | Alto/Muy alto |
| `data/pit` + quality monitor | Fable 5 / Sonnet | Medio-alto |
| Hardening de `execution/` | Fable 5 | Alto |
| Wiring, tests, formato | Composer | Bajo |

Regla del diagnóstico §8: **el diseño del criterio de promoción/deflación y del
allocator no se delega a ningún modelo agéntico** — un error ahí es alfa fantasma
con dinero real. Eso vive en razonamiento (Opus) + tu revisión. Fable 5 construye
lo que ya está diseñado y blindado.

---

## 8. Anti-patrones (heredados del diagnóstico, vinculantes)

- ❌ Producto cartesiano Activo×Estrategia×Modelo de agentes → especializar por
  **hipótesis de alfa**.
- ❌ Interpretabilidad como librería monolítica → protocolo `Explainer` transversal.
- ❌ Allocator sin deflación de selección → meta-overfitting a escala industrial.
- ❌ Modelos data-hungry (TFT/GNN) sobre datos data-poor (diario, ~1.500 barras).
- ❌ AutoML/NAS sin disciplina de deflación → destruye valor.
- ❌ Construir `portfolio/` antes de tener UN agente con edge → escalar prematuro.

---

## 9. Definition of Done (de este ADR como documento)

- Plano de referencia acordado (este archivo).
- **NO implica implementación.** El siguiente paso de construcción es: cerrar
  Nivel 1 (gate del agente actual). Solo entonces se abre el work-package
  `portfolio/` con Fable 5, sobre el diseño que Opus produzca de la matemática de
  allocation + deflación.
