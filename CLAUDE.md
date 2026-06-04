# PROJECT ML — Institutional AI-Native Trading Platform

> **Documento técnico interno.** Memoria operativa persistente, manual de
> arquitectura, especificación funcional, fuente de verdad para agentes IA
> autónomos (Claude, Codex, asistentes internos) y referencia de onboarding
> para nuevos ingenieros cuantitativos.

> **Estado del repositorio (2026-05-14 — MONOREPO)**: `quant_bot/` es ahora un
> monorepo con tres capas claramente separadas:
>
> | Carpeta | Contenido | Arranque |
> |---------|-----------|---------|
> | `research/` | I+D, backtesting, entrenamiento ML | `cd research && pip install -e ../shared && pip install -e .` |
> | `platform/` | 8 microservicios FastAPI + frontend React + Kafka + Redis | `cd platform && make up` |
> | `shared/` | Librería `quant_shared` — 19 features canónicos, schemas Kafka, model registry | `pip install -e shared/` |
>
> Los Ojos (backup en `C:\Users\alexj\OneDrive\Desktop\los_ojos\`) fue integrado
> en `platform/`. Ver §15 para estructura detallada.

---

## ÍNDICE

1. [Visión general del proyecto](#1-visión-general-del-proyecto)
2. [Arquitectura general](#2-arquitectura-general)
3. [Mercados soportados](#3-mercados-soportados)
4. [Estrategias](#4-estrategias)
5. [Capa "Los Ojos" — Inteligencia financiera](#5-capa-los-ojos--inteligencia-financiera)
6. [Machine Learning](#6-machine-learning)
7. [Feature Store](#7-feature-store)
8. [Bases de datos](#8-bases-de-datos)
9. [Infraestructura](#9-infraestructura)
10. [Arquitectura de eventos](#10-arquitectura-de-eventos)
11. [Motor de ejecución](#11-motor-de-ejecución)
12. [Risk Management](#12-risk-management)
13. [IA autónoma](#13-ia-autónoma)
14. [Dashboard](#14-dashboard)
15. [Estructura del repositorio](#15-estructura-del-repositorio)
16. [Despliegue](#16-despliegue)
17. [Seguridad](#17-seguridad)
18. [Roadmap](#18-roadmap)
19. [Estándares de código](#19-estándares-de-código)
20. [Reglas para Claude](#20-reglas-para-claude)

---

# 1. VISIÓN GENERAL DEL PROYECTO

## 1.1 Objetivos estratégicos

PROJECT ML es una plataforma de trading algorítmico institucional construida sobre
tres pilares:

| Pilar | Descripción | KPI primario |
|-------|-------------|--------------|
| **Edge cuantitativo** | Modelos ML calibrados con verificación walk-forward y deflated Sharpe ratio | DSR > 0.6 OOS, anual |
| **Cobertura multi-mercado** | Crypto spot/perp, FX spot, futuros, prediction markets, on-chain | ≥ 5 mercados activos, ≥ 20 instrumentos |
| **Autonomía operativa** | Agentes IA que seleccionan estrategia, ajustan sizing, detectan régimen y desactivan riesgo | Mean time between human interventions > 7 días |

**No-objetivos** (declarados para evitar scope creep):
- No es un HFT de microsegundos. Latencia objetivo: 50–500 ms.
- No es un agregador de copy-trading.
- No es una plataforma SaaS multi-tenant para clientes externos (MVP).

## 1.2 Filosofía del sistema

1. **Probabilidad sobre certeza.** Toda decisión se traduce a `P(y=c|x)` calibrada
   (ECE < 0.05). Nada de scores ad-hoc.
2. **Walk-forward o no existe.** Métricas IS son irrelevantes. PSR/DSR sobre OOS
   concatenado es la única medida válida.
3. **Anti-leakage por contrato.** Todo transformador `fit()` ve solo datos
   anteriores al test. Auditable estáticamente.
4. **Modular y desacoplado.** Cada componente es reemplazable: cambiar `XGBoost`
   por `DeepMLP` es una línea de config.
5. **Failure-first design.** Cada servicio asume que los demás pueden caer.
   Idempotencia, retries con backoff, DLQ.
6. **Observabilidad como ciudadano de primera.** Sin trazas distribuidas no se
   despliega.
7. **Coste cero por defecto.** Ningún recurso "warm" sin justificación: GPUs
   son spot, bases secundarias se ponen en sleep, brokers pagados son opt-in.

## 1.3 Problemas que resuelve

- **Sobreajuste sistémico** de estrategias manuales: PSR/DSR + ablative analysis
  cuantifican el aporte real de cada módulo.
- **Fragmentación de señales**: la capa "Los Ojos" centraliza datos macro,
  on-chain, prediction markets, news en un Feature Store único.
- **Latencia operativa**: pipeline event-driven con Kafka reduce el ciclo
  "señal → orden" a < 500 ms.
- **Riesgo de modelo único**: el meta-labeler, Bayesian sizing y el ensemble
  multi-modelo garantizan que un modelo defectuoso no liquida la cuenta.
- **Inconsistencia entre research y producción**: el código de entrenamiento
  y el de inferencia comparten el mismo `models.zoo`.

## 1.4 Principios de diseño

- **SOLID** aplicado estrictamente en el dominio ML: `BaseModel`, `BaseFeatureGenerator`,
  `BaseRiskController`.
- **Strategy pattern** para modelos, regímenes, sizing.
- **Event sourcing** para órdenes y fills.
- **CQRS** ligero: lectura (dashboard) y escritura (executor) separadas.
- **12-factor app** para todos los servicios (config en env, statelessness, logs
  a stdout).
- **Backwards compatibility** en el `FeatureStore`: versionado semántico de
  schemas; nunca borrar columnas hasta deprecate window.

## 1.5 Capacidades institucionales

- Backtesting con fee model realista (maker/taker, funding, slippage).
- Riesgo agregado por portfolio (correlación, volatilidad, exposición sectorial).
- Audit trail completo: cada orden referencia el `model_version`, `feature_set_hash`,
  `signal_id` y `risk_decision_id`.
- Replay determinista: cualquier día histórico se puede reejecutar bit-a-bit.
- A/B testing en producción: nuevas estrategias se exponen al 5 % del capital
  con kill-switch automático.

## 1.6 Visión a largo plazo (24 meses)

1. **Q1**: monolito `quant_bot` → microservicios (`ml-service`, `executor`,
   `risk-engine`, `feature-store`).
2. **Q2**: integración completa de "Los Ojos" + Polymarket signals.
3. **Q3**: RL en producción (PPO/SAC) con shadow trading.
4. **Q4**: ensemble multi-mercado, agente meta-coordinador.
5. **Año 2**: replicación cross-region, broker DMA dedicado, DAO de modelos.

---

# 2. ARQUITECTURA GENERAL

## 2.1 Diagrama de alto nivel

```
                                  PROJECT ML — Production Topology
+----------------------------------------------------------------------------------------------+
|                                       INGESTION LAYER                                        |
|  +-----------------+  +------------------+  +-------------+  +-----------+  +---------------+|
|  | binance-stream  |  | bybit-stream     |  | fx-feed     |  | onchain   |  | macro (FRED)  ||
|  +-----------------+  +------------------+  +-------------+  +-----------+  +---------------+|
|           |                  |                    |               |                 |        |
+-----------v------------------v--------------------v---------------v-----------------v--------+
                                              |
                                Kafka topics (raw.*)
                                              |
+----------------------------------------------------------------------------------------------+
|                                   STREAM PROCESSING LAYER                                    |
|     +----------------+  +-------------+  +-------------------+  +--------------------+       |
|     | normalizer     |  | dedup       |  | rolling-features  |  | regime-detector    |       |
|     | (currencies,   |  | (idempotent |  | (sliding windows) |  | (GMM, HMM, change  |       |
|     |  schema)       |  |  by hash)   |  |                   |  |  point)            |       |
|     +----------------+  +-------------+  +-------------------+  +--------------------+       |
+----------------------------------------------------------------------------------------------+
                                              |
                                  Kafka topics (clean.*, features.*)
                                              |
                       +-----------------------+-------------------------+
                       |                                                 |
+--------------------v-----------------+    +--------------------v-----------------+
|        FEATURE STORE  (online)        |    |        FEATURE STORE  (offline)      |
|        Redis + RocksDB                |    |        TimescaleDB + Parquet/S3      |
+---------------------------------------+    +--------------------------------------+
            |                                            |
            v                                            v
+---------------------------------------+    +--------------------------------------+
|        ML INFERENCE SERVICE           |    |        ML TRAINING PIPELINE          |
|  - models.zoo (XGBoost, MLP, LSTM)    |    |  - walk_forward_runner               |
|  - calibrator (isotonic / sigmoid)    |    |  - ablative_analysis                 |
|  - meta_labeler                       |    |  - hyperparameter_search             |
|  - bayesian_updater                   |    |  - drift_detector                    |
+---------------------------------------+    +--------------------------------------+
            |                                            ^
            v                                            |
+---------------------------------------+    +--------------------------------------+
|        SIGNAL ROUTER                  |    |        MODEL REGISTRY                |
|  - dedup signals                      |--->|  - artifacts (pickle / ONNX)         |
|  - aggregate cross-strategy           |    |  - metadata (PSR, DSR, ECE)          |
+---------------------------------------+    +--------------------------------------+
            |
            v
+---------------------------------------+
|        RISK ENGINE                    |
|  - kelly sizing                       |
|  - dynamic R:R                        |
|  - portfolio exposure                 |
|  - kill switch                        |
+---------------------------------------+
            |
            v
+---------------------------------------+
|        EXECUTION ENGINE               |
|  - smart order routing                |
|  - TWAP / VWAP / Iceberg              |
|  - slippage tracker                   |
+---------------------------------------+
            |
            v
+---------------------------------------+      +-------------------------------+
|     EXCHANGES / BROKERS                |---->|  POST-TRADE ANALYTICS         |
+---------------------------------------+      |  - P&L attribution            |
                                               |  - fee analysis               |
                                               +-------------------------------+
```

## 2.2 Separación de responsabilidades

| Servicio | Lenguaje | Responsabilidad | SLO p99 |
|----------|----------|-----------------|---------|
| `ingestion-binance` | Rust | WS feed → Kafka raw | < 5 ms |
| `ingestion-fx` | Python | REST poll → Kafka raw | < 200 ms |
| `normalizer` | Python | Schema unification | < 10 ms |
| `feature-engine` | Python (numba) | Compute features | < 50 ms |
| `ml-inference` | Python (FastAPI) | predict_proba | < 100 ms |
| `signal-router` | Go | Dedup & aggregate | < 5 ms |
| `risk-engine` | Python | Sizing & limits | < 20 ms |
| `executor` | Go | Order placement | < 50 ms |
| `dashboard-api` | Python (FastAPI) | Queries | < 200 ms |
| `dashboard-web` | TypeScript (Next.js) | UI | n/a |

**Regla de oro**: cada servicio tiene **un único motivo de cambio**. Si dos
features tocan el mismo servicio en la misma sprint sin razón clara, refactor.

## 2.3 Flujo de datos (happy path)

```
t=0ms       Binance pushes trade @ BTCUSDT
t=2ms       ingestion-binance writes to kafka raw.trades.binance.btcusdt
t=8ms       normalizer consumes, writes clean.trades.btcusdt
t=15ms      feature-engine updates rolling features (rsi, atr, volume_imbalance)
            writes features.btcusdt
t=25ms      ml-inference consumes feature update, computes proba
            (cached model artifact in memory)
            writes signals.raw.btcusdt
t=30ms      signal-router aggregates with cross-strategy signals
            writes signals.final.btcusdt
t=50ms      risk-engine evaluates: kelly_size, current_exposure, vol_regime
            writes orders.intent.btcusdt (or rejects)
t=80ms      executor receives intent, decides routing (maker/taker, venue)
            submits to exchange
t=150ms     fill confirmation → fills.btcusdt → P&L attribution
```

## 2.4 Flujo de inferencia

```python
# Pseudocódigo del path crítico de inferencia
def on_feature_update(features: FeatureVector) -> Optional[Signal]:
    # 1. Validar feature freshness
    if features.timestamp < now() - FRESHNESS_THRESHOLD:
        metrics.stale_features.inc()
        return None

    # 2. Inferencia primaria
    proba = primary_model.predict_proba(features)   # calibrada
    signal = entry_filter.predict(proba)
    if signal == 0:
        return None

    # 3. Meta-labeler (filtro de calidad)
    p_correct = meta_labeler.predict_p_correct(features, proba, signal)
    if p_correct < META_THRESHOLD:
        return None

    # 4. Bayesian update con prior de régimen
    p_win_posterior = bayesian_updater.update(
        p_model=p_correct,
        regime=features.regime_label,
        direction=signal,
    )

    # 5. Emitir señal con metadata completa
    return Signal(
        direction=signal,
        p_win=p_win_posterior,
        model_version=primary_model.version,
        feature_set_hash=features.hash,
        timestamp=now(),
    )
```

## 2.5 Flujo de entrenamiento

```
Daily cron @ 02:00 UTC
  └─> Trigger training-pipeline DAG
       ├─> Fetch features from offline store (last 3 years)
       ├─> WalkForwardRunner.run(X, y, ...)
       │    ├─> Per fold:
       │    │    ├─> split → fit → calibrate → predict
       │    │    ├─> compute_classification_metrics
       │    │    └─> compute_bias_variance_verdict
       │    └─> aggregate metrics
       ├─> If PSR_new > PSR_prod + delta:
       │    ├─> AblativeAnalyzer.run()  # verifica que cada módulo aporta
       │    ├─> ErrorAnalyzer.analyze()  # diagnóstico
       │    └─> Promote to staging registry
       ├─> Shadow trading 24h
       └─> Promote to production (canary 5%) → full 24h sin alertas → 100%
```

## 2.6 Escalabilidad horizontal

| Componente | Estrategia | Sharding key |
|------------|-----------|--------------|
| Ingestion | 1 pod por venue+stream | venue_id |
| Feature-engine | Consumer group | instrument_id (hash) |
| ML-inference | Stateless replicas | round-robin |
| Risk-engine | Singleton por book | account_id |
| Executor | 1 pod por venue | venue_id |

## 2.7 Fault tolerance y alta disponibilidad

- **Kafka**: replication-factor 3, min.insync.replicas 2.
- **Postgres**: streaming replication + pgBouncer.
- **Redis**: Sentinel con 3 nodos.
- **Servicios stateless**: 2+ réplicas detrás de load balancer.
- **Risk-engine** (stateful): activo-pasivo con failover < 30 s.
- **Circuit breakers**: si `executor` recibe > 5 errores 5xx en 60s, pasa a
  modo `read-only`.

---

# 3. MERCADOS SOPORTADOS

## 3.1 Tabla de cobertura

| Mercado | Venues | Instrumentos | Frecuencia | Latencia ingestion | Estado |
|---------|--------|--------------|------------|--------------------|--------|
| Crypto spot | Binance, Coinbase, Kraken | 30 pares top-volumen | 1s, 1m, 5m, 1h, 1d | < 10 ms | Activo |
| Crypto perpetuals | Binance, Bybit, OKX | 20 pares + funding | 1s, 1m, 5m, 1h | < 10 ms | Activo |
| Forex spot | OANDA, IB | 10 pares G10 | 1m, 5m, 1h, 1d | < 200 ms | Planificado Q2 |
| Futuros | CME (via IB), ICE | ES, NQ, CL, GC | 1m, 5m, 1h, 1d | < 100 ms | Planificado Q3 |
| Prediction markets | Polymarket, Kalshi | Top 50 markets | 5m | < 1 s | Investigación |
| On-chain | Glassnode, Etherscan, Dune | BTC, ETH, top 20 ERC-20 | 1h, 1d | < 5 min | MVP |
| Macro | FRED, BLS, ECB | 50 series clave | 1d | n/a | Activo |

## 3.2 Crypto

- **Spot**: foco en BTC, ETH, SOL + 27 pares de alta liquidez.
- **Perpetuals**: usar funding rate como feature (mean-reversion en funding extremos).
- **Considerar**: latencia API, rate limits, withdrawal limits, fees por tier.

## 3.3 Forex

- 23/5 trading window. Atención a gaps de fin de semana.
- Liquidez asimétrica por sesión (Asia/EU/US).
- Spread variable: usar mid-price + spread adjustment en backtest.

## 3.4 Futuros

- Roll dates: lógica de rollover automática.
- Initial margin y maintenance margin diferenciados.
- Sesión RTH vs ETH: features distintas.

## 3.5 Prediction markets

- Probabilidades implícitas como feature regulatoria/de sentimiento.
- Liquidez baja: no operar directamente hasta tener evidencia de edge.

## 3.6 On-chain

- Métricas core: exchange netflow, miner activity, SOPR, MVRV, whale tx.
- Latencia inherente: confirmaciones blockchain → uso solo en horizontes > 4h.

## 3.7 Macro

- Series FRED: VIX, DXY, 10Y yield, M2, CPI, NFP.
- Régimen macro como feature global para todos los mercados.

---

# 4. ESTRATEGIAS

Cada estrategia se documenta en un archivo `strategies/<name>.md` con la misma
estructura. Aquí el resumen.

## 4.1 Scalping (crypto 1m)

| Campo | Valor |
|-------|-------|
| Horizonte | 1–15 min |
| Inputs | Order book imbalance, micro volatility, taker pressure |
| Features | `ob_imbalance_1`, `volume_burst`, `bid_ask_spread`, `tick_rule_sum` |
| Modelo | XGBoost shallow (`max_depth=3`) con calibración |
| Risk | Kelly fraccionado 0.10, max risk 0.5% / trade, hard stop ATR×1 |
| KPI | Sharpe > 2, win rate > 55%, avg holding < 10 min |
| Riesgos | Adverse selection, flash crashes, exchange outage |

## 4.2 Swing (crypto/FX 1h–4h)

| Campo | Valor |
|-------|-------|
| Horizonte | 4h – 5 días |
| Inputs | Triple barrier labels, ATR-adjusted returns, regime GMM |
| Features | Momentum (z-score 20/60), RSI, MACD, regime probs, on-chain (BTC) |
| Modelo | XGBoost + DeepMLP ensemble + meta-labeler |
| Risk | Quarter Kelly (0.25), dynamic R:R 1.2–2.5, ATR×2 SL |
| KPI | Sharpe > 1.0, DSR > 0.5, max DD < 15% |
| Riesgos | Regime shifts, weekend gaps (crypto sigue 24/7) |

> Esta es la estrategia **actualmente implementada** en `quant_bot`.

## 4.3 Market making (crypto)

| Campo | Valor |
|-------|-------|
| Horizonte | < 1 s |
| Inputs | LOB depth, microstructure, queue position |
| Features | Top-of-book imbalance, queue ETA, recent trade flow |
| Modelo | Stoikov + reinforcement learning fine-tuning |
| Risk | Inventory hard cap, kill switch en inventario > 2σ |
| KPI | Spread captured > 1.5× fees, inventory turnover |
| Riesgos | Adverse selection, latencia, exchange downtime |

## 4.4 Statistical arbitrage

| Campo | Valor |
|-------|-------|
| Horizonte | Días |
| Inputs | Spread series, cointegration test (Johansen) |
| Features | Z-score spread, half-life of mean reversion |
| Modelo | OLS / Kalman filter para hedge ratio dinámico |
| Risk | Per-pair risk cap, correlation cluster cap |
| KPI | Sharpe > 1.5, neutralidad direccional |
| Riesgos | Cointegración rompe (regime shift) |

## 4.5 BTC cross-exchange arbitrage

| Campo | Valor |
|-------|-------|
| Horizonte | Segundos |
| Inputs | Order books de N venues |
| Features | Spread bid/ask cross-venue, withdrawal time, fees |
| Modelo | Heurístico determinista (no ML) |
| Risk | Latencia mata el edge: bloquear si network > 100 ms |
| KPI | Bps capturados netos de fees+withdrawal |
| Riesgos | Withdrawal delays, KYC/limites |

## 4.6 Trend following

| Campo | Valor |
|-------|-------|
| Horizonte | Semanas–meses |
| Inputs | Long-term moving averages, breakout signals |
| Features | 50/200 SMA cross, Donchian channels, ADX |
| Modelo | Rule-based + DeepMLP filter |
| Risk | Trailing stop, position pyramiding |
| KPI | Profit factor > 1.5, recovery factor > 2 |
| Riesgos | Whipsaws en mercados laterales |

## 4.7 Mean reversion

| Campo | Valor |
|-------|-------|
| Horizonte | Horas–días |
| Inputs | Oversold/overbought oscillators, bollinger bands |
| Features | RSI extremos, BBand %B, vol regime |
| Modelo | XGBoost classifier en bordes |
| Risk | SL ajustado (mean reversion falla rápido) |
| KPI | Win rate > 60%, avg win/loss > 1 |
| Riesgos | Trending markets aniquilan |

## 4.8 Volatility trading

| Campo | Valor |
|-------|-------|
| Horizonte | 1 día – 2 semanas |
| Inputs | Implied vs realized volatility, VIX term structure |
| Features | IV/RV ratio, vol skew, VIX z-score |
| Modelo | Options-based, GARCH para realized |
| Risk | Vega/gamma caps |
| KPI | Vega-normalized P&L |
| Riesgos | Vol spikes (gap risk) |

## 4.9 RL trading

| Campo | Valor |
|-------|-------|
| Horizonte | Variable (aprendido) |
| Inputs | Estado de mercado discretizado/continuo |
| Features | Mismas que swing + p_win primario |
| Modelo | Q-learning tabular (MVP) → PPO/SAC (prod) |
| Risk | Hard caps externos al RL (RL no debe romper risk limits) |
| KPI | Sharpe vs baseline supervised |
| Riesgos | Distribución shift, training instability |

> MVP implementado en `models/rl_agent.py` (Q-learning tabular).

## 4.10 Multi-agent system

| Campo | Valor |
|-------|-------|
| Horizonte | Meta-coordinación |
| Inputs | Señales de todas las estrategias |
| Features | Performance reciente por estrategia, régimen actual |
| Modelo | Bandit (Thompson sampling) para asignar capital |
| Risk | Allocations suman 100%, no cortos en allocations |
| KPI | Portfolio Sharpe > max(Sharpe individual) |
| Riesgos | Correlación oculta entre estrategias |

---

# 5. CAPA "LOS OJOS" — INTELIGENCIA FINANCIERA

> **"Los Ojos"** es la capa de awareness contextual: agrega datos macro,
> on-chain, social, news y prediction markets en features unificadas que
> alimentan el ML core.
>
> **Estado actual (2026-05-14)**: implementada como repositorio independiente.
> Path: `C:\Users\alexj\OneDrive\Desktop\los_ojos\`
> Stack: FastAPI · Kafka · Redis · TimescaleDB · MongoDB · React/TypeScript
> Arranque: `cd los_ojos && make up`

## 5.0 Repositorio los_ojos — Servicios implementados

| Servicio | Puerto | Estado | Función |
|---------|--------|--------|---------|
| market-intelligence | 8001 | ✅ | OpenBB OHLCV, Binance orderbook+funding WS |
| macroeconomic | 8002 | ✅ | FRED 22 series, Sahm Rule, yield curve, macro regime |
| onchain-analysis | 8003 | ✅ | Crucix whale detection, exchange flows, smart money |
| context-engine | 8004 | ✅ | GMM 5-component regime classifier |
| realtime-signal | 8005 | ✅ | FastAPI WebSocket, ConnectionManager, Kafka→Redis→WS |
| ml-feature-store | 8006 | ✅ | Feature computation + serving (Redis, baja latencia) |
| strategy-orchestrator | 8007 | ✅ | Bot config, kill switch, strategy toggle, señales |

Frontend en `los_ojos/frontend/` (React + TypeScript + Vite + Tailwind):
- `TradingDashboard.tsx` — layout principal con PnL, señales, régimen, whale alerts
- `TradingChart.tsx` — lightweight-charts candlestick + signal markers
- `BotConfigPanel.tsx` — configuración del bot: mode paper/live, sliders de riesgo, strategy toggles, kill switch
- `useWebSocket.ts` — reconexión automática con backoff exponencial

## 5.1 OpenBB

- **Propósito**: terminal cuantitativo open-source con conectores a > 100 APIs
  (Yahoo, Quandl, Polygon, FMP, CoinGecko, etc.).
- **Arquitectura**: SDK Python `openbb`, se llama desde `los_ojos/openbb_collector.py`.
- **Flujo**: cron → `openbb.economy.gdp(...)` → normalizer → TimescaleDB (`macro.*`).
- **Casos de uso**: backfill histórico de series macro, fundamentals de equities.
- **Outputs**: pandas DataFrames → Parquet → feature store offline.
- **Integración ML**: features `macro_*` consumidas por todas las estrategias swing+.

## 5.2 Dexter

- **Propósito**: agente de búsqueda y análisis financiero (web + APIs) tipo
  Perplexity for finance.
- **Arquitectura**: cliente HTTP a Dexter Cloud o self-hosted; resultados a Redis.
- **Flujo**: trigger event (e.g. earnings release) → query Dexter → embeddings → feature.
- **Casos de uso**: sentiment analysis sobre noticias, análisis cualitativo de eventos.
- **Outputs**: structured JSON con sentiment score, key entities, summary.
- **Integración ML**: features `news_sentiment_24h`, `news_entity_btc_count`.

## 5.3 MCP Server

- **Propósito**: Model Context Protocol — bridge entre Claude y herramientas internas.
- **Arquitectura**: servidor MCP propio en `mcp_server/` expone tools:
  `get_portfolio`, `get_signals`, `query_features`, `backtest_strategy`.
- **Flujo**: Claude (operador) llama tool → MCP route → handler interno.
- **Casos de uso**: investigación interactiva, debugging de señales, ejecución
  manual con doble confirmación.
- **Outputs**: respuestas JSON estructuradas.
- **Integración ML**: tool `train_model(spec)` permite a Claude lanzar runs.

## 5.4 Crucix

- **Propósito**: plataforma de análisis cripto on-chain y de derivados (perpetuales,
  funding, OI, liquidations).
- **Arquitectura**: cliente REST/WS a Crucix API; cache en Redis.
- **Flujo**: WS subscribe → Kafka topic `crucix.events.*` → feature-engine.
- **Casos de uso**: detectar squeezes, niveles de liquidación, sentiment de derivados.
- **Outputs**: structured events (e.g. `large_liquidation`, `funding_spike`).
- **Integración ML**: features `liquidations_long_1h`, `funding_z_score`.

## 5.5 fredapi

- **Propósito**: cliente oficial de Federal Reserve Economic Data.
- **Arquitectura**: `fredapi.Fred(api_key)`, llamadas batched diarias.
- **Flujo**: cron @ 06:00 ET → fetch 50 series → TimescaleDB.
- **Casos de uso**: macro regime detection, FOMC event features.
- **Outputs**: pandas Series → tabla `macro.fred_series`.
- **Integración ML**: features `dxy_z`, `vix_level`, `yield_curve_slope`.

## 5.6 Binance Collector

- **Propósito**: stream de datos crypto de baja latencia.
- **Arquitectura**: Rust binary, conecta a `wss://stream.binance.com:9443/ws`,
  escribe a Kafka.
- **Flujo**: WS → deserialización → particionado por symbol → Kafka raw.
- **Casos de uso**: trades, klines, depth, liquidations en realtime.
- **Outputs**: protobuf messages a Kafka.
- **Integración ML**: feed primario para todas las estrategias crypto.

## 5.7 Polymarket Assistant Tool

- **Propósito**: scraper + analyzer de mercados de predicción.
- **Arquitectura**: cliente GraphQL contra subgraph de Polymarket; poll cada 5m.
- **Flujo**: query → parse → normalizer → features.
- **Casos de uso**: implied probability de eventos macro (e.g. recession 2026).
- **Outputs**: tabla `prediction_markets.events` con prob, volume, liquidity.
- **Integración ML**: feature `polymarket_recession_prob` como prior macro.

## 5.8 lightweight-charts

- **Propósito**: librería de gráficos financiera (TradingView OSS).
- **Arquitectura**: integrada en el dashboard Next.js.
- **Flujo**: API → WS realtime → chart rendering.
- **Casos de uso**: visualización OHLC, señales, P&L, indicadores custom.
- **Outputs**: UI interactiva.
- **Integración ML**: overlay de señales y predicciones del modelo.

## 5.9 Pipeline unificado de "Los Ojos"

```
+-------+   +---------+   +---------+   +-------+   +----------+   +------------+
|OpenBB |   |Dexter   |   |Crucix   |   |FRED   |   |Binance   |   |Polymarket  |
+---+---+   +----+----+   +----+----+   +---+---+   +-----+----+   +------+-----+
    |            |             |             |             |               |
    +-----+------+------+------+------+------+------+------+-------+-------+
          |                            |
          v                            v
   +-----------------+         +-----------------+
   |  ETL / Normaliz |         |  Realtime stream|
   |  (batch)        |         |  (Kafka)        |
   +--------+--------+         +--------+--------+
            |                           |
            +-----------+---------------+
                        |
                        v
              +-----------------+
              |  FEATURE STORE  |
              +-----------------+
                        |
                        v
              +-----------------+
              |  ML PIPELINES   |
              +-----------------+
```

---

# 6. MACHINE LEARNING

## 6.1 Arquitectura ML (estado actual + objetivo)

```
                    +---------------------------+
                    |     FEATURE STORE         |
                    +-------------+-------------+
                                  |
                  +---------------+-----------------+
                  |                                 |
                  v                                 v
        +------------------+              +------------------+
        | OFFLINE Training |              | ONLINE Inference |
        | walk_forward_    |              | ml-inference     |
        | runner.py        |              | service          |
        +--------+---------+              +---------+--------+
                 |                                  ^
                 v                                  |
        +------------------+              +------------------+
        | Model Registry   +-------------->  Model artifact  |
        | (pickle + meta)  |              |  loaded in mem   |
        +------------------+              +------------------+
```

## 6.2 Stack ML actual (implementado en `quant_bot`)

| Componente | Archivo | Función |
|------------|---------|---------|
| Modelos base | `models/zoo.py` | `LogisticBaseline`, `XGBoostClassifier`, `DeepMLPClassifier` (legacy/baseline), `ResMLPClassifier` (deep tabular — ADR-034), `LSTMClassifier` |
| NN layers | `research/models/nn_layers.py` | `SwiGLU`, `ResBlock`, `TemperatureScaling` (introducido en ADR-034) |
| Calibración | `models/calibration.py` | `IsotonicCalibrator` (isotonic + sigmoid). En cascada con `TemperatureScaling` para NNs. |
| Meta-labeling | `models/meta_labeler.py` | Segundo classifier binario |
| Bayesian sizing | `risk/bayesian_sizer.py` | Product of experts |
| Walk-forward | `models/walk_forward_runner.py` | Pipeline completo por fold |
| Métricas CS229 | `models/metrics.py` | F1/AUC/CM/bias-variance |
| Error analysis | `models/error_analysis.py` | Diagnóstico por dirección/régimen/confianza |
| Ablative analysis | `models/ablative_analysis.py` | Contribución por módulo |
| Validation | `models/validation.py` | WalkForwardSplitter |
| Entry filter | `models/entry_filter.py` | Threshold optimization |
| Feature selection | `models/feature_selection.py` | Gain + SHAP |
| PCA denoising | `features/pca_denoiser.py` | Reducción dimensional anti-leakage |
| Q-learning | `models/rl_agent.py` | RL agent tabular |

## 6.3 Modelos: cuándo usar cada uno

| Modelo | Cuándo | Ventajas | Limitaciones |
|--------|--------|----------|--------------|
| Logistic Regression | Baseline obligatorio | Rápido, interpretable | Solo señal lineal |
| XGBoost | Default para tabular | Robusto, gain importance | Overfitting con max_depth alto |
| DeepMLP (legacy) | Reemplazado por ResMLP en multi-horizon trainer (ADR-034) | Mantenido como baseline A/B durante shadow trading | Sin skip connections, calibración pobre out-of-the-box |
| ResMLP | > 10K muestras, MLP profundo con skip connections + SwiGLU + BatchNorm; reemplaza DeepMLP en multi-horizon trainer | Mejor bias-variance que MLP plano; calibrable vía temperature scaling + isotonic cascade; gradientes estables hasta 6 bloques | Coste training mayor (GPU recomendado); Optuna search más caro; latency p99 mayor que XGBoost en CPU |
| LSTM | > 100K muestras, señal secuencial real | Memoria temporal | Inestable, requiere GPU |
| PPO/SAC (futuro) | Optimal control problem | Optimiza directamente reward, learns position management | Sample-inefficient, hard to debug |

**Regla**: ascender en la jerarquía solo si el modelo más simple no captura
la señal (medido por bias/variance gap < 0.10).

## 6.4 Feature engineering

Pipeline de features:
1. **Raw** → bar OHLCV + microstructure
2. **Technical** → RSI, MACD, ATR, BBands, EMA cross
3. **Statistical** → z-scores rolling, vol estimators (Garman-Klass, Parkinson)
4. **Regime** → GMM probs (3 componentes default)
5. **Macro/On-chain** → de "Los Ojos"
6. **PCA denoising** → solo aplicado a features tecnicales (no a regime probs)

## 6.5 Backtesting

- **WalkForwardRunner**: única función de verdad.
- **Fees realistas**: maker/taker bps, slippage proporcional a ADV.
- **Triple barrier method** (López de Prado) para labels.
- **Embargo**: separación temporal train/test ≥ horizonte de predicción.

## 6.6 Retraining

```yaml
schedule: "0 2 * * *"   # diario @ 02:00 UTC
strategy:
  - full_retrain: domingo (todo el histórico)
  - incremental:  lun-sáb (últimas 4 semanas)
promotion:
  - shadow: 24h en producción sin ejecutar órdenes
  - canary: 24h al 5% del capital
  - full:   resto del capital
gates:
  - PSR_new > PSR_prod * 0.95   # tolerancia 5%
  - DSR_new > 0.4
  - ECE     < 0.05
  - no_class_collapse: min(per_class_predict) > 0.05
```

## 6.7 Hyperparameter tuning

- **Optuna** con pruning para XGBoost / MLP.
- **Budget**: 100 trials max por modelo, 4h wall-clock.
- **Search space** definido por estrategia.
- **CV interna**: walk-forward purgado dentro del train del outer fold (nested WF).

## 6.8 Drift detection

- **PSI** (Population Stability Index) sobre features clave, ventana 7d vs 30d.
- **KS test** sobre proba distributions.
- **Decay alert**: si Sharpe rolling 60d < Sharpe IS * 0.5 → alert.
- **Action**: retrain forzado + freeze del modelo si PSI > 0.25.

## 6.9 Ensemble systems

- **Weighted average** de proba (pesos = inverse CV loss).
- **Stacking**: meta-learner (logistic) sobre proba de N base models.
- **Bayesian Model Averaging**: pesos = posterior P(model | data).

## 6.10 Reinforcement Learning

Roadmap:
1. **MVP (actual)**: Q-learning tabular sobre estado discretizado
   `(regime_bin, p_win_bin, trend_bin)` → acción `{-1,0,+1}`.
2. **Q1+**: DQN con replay buffer y target network.
3. **Q2+**: PPO con policy continuous (position sizing fraccional).
4. **Q3+**: SAC para mejor sample efficiency.
5. **Multi-agent**: PPO por estrategia + meta-bandit para asignar capital.

Anti-pattern conocido: dejar al RL decidir risk limits.
**Risk limits son externos al agente RL, siempre.**

---

# 7. FEATURE STORE

## 7.1 Estructura general

```
+---------------------------------------------------------------+
|                       FEATURE STORE                            |
|                                                                |
|  +---------------------+        +---------------------------+ |
|  |  ONLINE (low-lat)   |        |  OFFLINE (training)       | |
|  |                     |        |                           | |
|  |  Redis cluster      |        |  TimescaleDB              | |
|  |  - 1 hash/symbol    |        |  + Parquet on S3          | |
|  |  - TTL 10 min       |        |  + DuckDB para queries    | |
|  +---------------------+        +---------------------------+ |
|              ^                              ^                  |
|              |                              |                  |
|         (consumed by                  (read by training        |
|          ml-inference)                 + research notebooks)   |
+---------------------------------------------------------------+
                              ^
                              |
                  +-----------+-----------+
                  | Feature-engine        |
                  | (stream + batch)      |
                  +-----------------------+
```

## 7.2 Versionado

- **SemVer en feature sets**: `feature_set_v1.2.3.yaml`.
- **Major** = breaking schema change.
- **Minor** = nueva feature backwards-compatible.
- **Patch** = bug fix en computación.
- **Hash**: cada vector tiene `feature_set_hash = sha256(spec + data)`.

## 7.3 Pipelines

```python
# Definición declarativa de feature
@feature(
    name="rsi_14",
    version="1.0.0",
    inputs=["close"],
    window=14,
    online=True,   # disponible en realtime
    offline=True,  # disponible en histórico
)
def compute_rsi(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

## 7.4 Validación

- **Schema validation**: pandera o pydantic en cada escritura.
- **Range checks**: RSI ∈ [0,100], proba ∈ [0,1].
- **Freshness check**: max staleness por feature.
- **NaN policy**: declarado por feature (forward-fill, drop, fill 0).

## 7.5 Almacenamiento

| Layer | Tech | Retention | Latency |
|-------|------|-----------|---------|
| Hot | Redis | 24h | < 1 ms |
| Warm | TimescaleDB | 90 días | < 100 ms |
| Cold | S3 Parquet (Iceberg) | 7 años | seconds |

## 7.6 Streaming sync

```
feature-engine compute → write to Redis (online)
                      └→ write to Kafka topic features.* (event sourcing)
Async consumer        ← reads Kafka, batches by hour
                      → writes to TimescaleDB (offline)
                      → flushes daily to S3 Parquet
```

## 7.7 Datasets compartidos vs independientes

- **Shared**: macro, regime, on-chain → un solo conjunto, consumido por todas las
  estrategias.
- **Independent**: features de microstructure son por símbolo+venue.

---

# 8. BASES DE DATOS

## 8.1 PostgreSQL

| Schema | Tablas | Uso |
|--------|--------|-----|
| `core` | `users`, `accounts`, `permissions` | RBAC, auth |
| `models` | `model_versions`, `training_runs`, `promotions` | Registry |
| `orders` | `intents`, `orders`, `fills` | Trading log |
| `risk` | `limits`, `breaches` | Risk audit |
| `audit` | `actions`, `decisions` | Audit trail |

- **Indexes**: `(account_id, ts)` en orders, `(model_id, run_id)` en training_runs.
- **Particionamiento**: orders por mes; audit por mes.
- **Replicación**: streaming async a réplica RO; pgBouncer transaction pooling.

## 8.2 TimescaleDB

| Hypertable | Particionado | Compresión |
|------------|--------------|------------|
| `market.ohlcv_1m` | (symbol, time) | después de 7 días |
| `market.trades` | (symbol, time) | después de 1 día |
| `features.online` | (symbol, time) | después de 24h |
| `macro.fred_series` | (series_id, time) | después de 30 días |

- **Continuous aggregates**: `ohlcv_5m`, `ohlcv_1h` materializadas desde `ohlcv_1m`.
- **Retention policies**: trades → 30 días raw, después agregado.

## 8.3 MongoDB

- **Uso**: documentos semi-estructurados (configs de modelo, resultados de
  hyperparameter search, JSONs de eventos de Dexter).
- **Collections**: `experiments`, `events`, `news_articles`.

## 8.4 Redis

- **Uso**: feature store online, session cache, rate limits, distributed locks.
- **Eviction policy**: `allkeys-lru` para feature store; `noeviction` para locks.
- **Persistence**: AOF (append-only file) + snapshots cada hora.

## 8.5 Decisiones arquitectónicas

- **Postgres vs Mongo**: schema-on-read SOLO para datos exploratorios; cualquier
  cosa con SLOs va a Postgres + Timescale.
- **No usar Timescale como OLTP**: solo time-series append-only.
- **Redis no es base de datos**: si necesitas persistencia garantizada, no Redis.

---

# 9. INFRAESTRUCTURA

## 9.1 Docker

- Cada servicio tiene `Dockerfile` multi-stage:
  1. `builder`: instala deps, compila.
  2. `runtime`: imagen slim (distroless o python:3.11-slim) con solo binarios.
- `docker-compose.yaml` para desarrollo local con dependencias (Postgres, Redis, Kafka).

## 9.2 Kubernetes

- Cluster managed (EKS / GKE) con nodepools:
  - `general`: app servicios (autoscale 2–10).
  - `ml-cpu`: pods de inferencia (autoscale 4–20).
  - `ml-gpu`: training, spot instances (autoscale 0–4).
- Namespace por entorno: `dev`, `staging`, `prod`.
- ResourceQuotas por namespace.

## 9.3 Helm

- Un chart por servicio en `infra/helm/<service>/`.
- Umbrella chart `infra/helm/platform` para deploy completo.
- Values por entorno: `values.dev.yaml`, `values.prod.yaml`.

## 9.4 CI/CD (GitHub Actions)

Pipeline estándar por servicio:
```yaml
on: [push, pull_request]
jobs:
  lint:        # ruff + mypy
  test:        # pytest + coverage > 80%
  build:       # docker build + push a ECR
  deploy-dev:  # helm upgrade --install (en push a main)
  deploy-prod: # manual approval gate
```

Pipeline ML:
```yaml
- nightly: training-pipeline
  steps:
    - fetch_features
    - walk_forward_run
    - if PSR > threshold: register_model_to_staging
    - shadow_trade_24h
    - canary_promote_5pct
    - full_promote (after 24h sin alertas)
```

## 9.5 Observabilidad

| Layer | Tech |
|-------|------|
| Logs | structlog → stdout → Vector → Loki |
| Metrics | Prometheus + exporters |
| Traces | OpenTelemetry → Tempo / Jaeger |
| Dashboards | Grafana |
| Alerts | Alertmanager → PagerDuty + Slack |

Dashboards mínimos:
- `Service health`: error rate, p99 latency, throughput.
- `ML inference`: predict latency, calibration ECE rolling, signal counts.
- `Risk`: exposure por símbolo, drawdown intraday.
- `P&L`: cumulative, by strategy, by symbol.

## 9.6 Autoscaling

- HPA por CPU/RAM en stateless services.
- KEDA por lag de Kafka consumer en feature-engine.
- Cluster autoscaler en nodepool `ml-gpu`.

## 9.7 GPU orchestration

- GPUs solo se levantan para training (cronjob nocturno).
- `nvidia-device-plugin` en nodepool dedicado.
- Spot instances con savings 60–80%; checkpointing cada 5 epochs.

---

# 10. ARQUITECTURA DE EVENTOS

## 10.1 Stack: Kafka

- Brokers: 3 nodos, replication-factor 3, min-ISR 2.
- Schemas: Avro con Schema Registry.
- Topics:

| Topic | Partitions | Retention |
|-------|------------|-----------|
| `raw.trades.{venue}.{symbol}` | 8 | 7 días |
| `clean.trades.{symbol}` | 8 | 30 días |
| `features.{symbol}` | 4 | 7 días |
| `signals.raw.{symbol}` | 2 | 30 días |
| `signals.final` | 2 | 90 días |
| `orders.intent` | 2 | 90 días |
| `orders.placed` | 2 | 90 días |
| `fills` | 2 | 7 años (audit) |

## 10.2 Producers

- Idempotente (`enable.idempotence=true`).
- Acks=all.
- Compression: lz4 (balance latencia/CPU).

## 10.3 Consumers

- Consumer groups por servicio (`feature-engine-cg`, `risk-engine-cg`).
- Manual commit después de procesar con éxito.
- Retry pattern: 3 reintentos con backoff exponencial → DLQ.

## 10.4 DLQ

- Topic `dlq.<original-topic>` con mismo schema + columna `error`.
- Alert en > 10 msgs/min en cualquier DLQ.
- Reprocessing manual desde UI admin.

## 10.5 Diseño low-latency

- **No sync I/O** en el path crítico.
- **Pre-warmed connections** (Kafka, Postgres pools).
- **Sticky partitions** para microstructure feeds.
- **Locality**: ingestion y feature-engine en mismo AZ que Kafka brokers.

---

# 11. MOTOR DE EJECUCIÓN

## 11.1 Order Routing

Decisión por venue basada en:
- Liquidez (top-of-book size vs orden objetivo)
- Fees (maker rebate vs taker fee)
- Latency (medida rolling p99)
- Withdrawal cost (para arbitraje)

## 11.2 Smart Execution

| Modo | Cuándo | Implementación |
|------|--------|----------------|
| `MARKET` | Señal urgente, vol baja | IOC order |
| `LIMIT_MAKER` | Sin urgencia | Post-only en mid + offset |
| `TWAP` | Orden grande > 5% ADV/hora | Slice en N chunks |
| `VWAP` | Orden grande con benchmark | Match curva histórica |
| `ICEBERG` | Esconder size | Visible size = N%, refill al fill |

## 11.3 Slippage Control

- Pre-trade: simulación contra orderbook actual.
- Hard limit: si slippage estimado > 30 bps → reject o subdividir.
- Post-trade: tracking de slippage vs benchmark (arrival price, VWAP).

## 11.4 Latency Arbitrage Defense

No somos HFT, pero:
- **Latency budget** documentado: 500 ms tick → fill.
- **Time-out aggressive**: si ack > 1 s, asumir orden perdida y cancelar.
- **Reconciliation loop** cada 60 s verifica positions reales vs internas.

## 11.5 Execution Engine — pseudocódigo

```python
def execute(intent: OrderIntent) -> OrderResult:
    venue = router.choose_venue(intent)
    mode = router.choose_mode(intent, venue)

    if not risk.allow(intent):
        return OrderResult(status="rejected", reason="risk")

    if mode == "LIMIT_MAKER":
        ob = orderbook.snapshot(venue, intent.symbol)
        price = compute_maker_price(ob, intent.side)
        order = exchange.place_limit(intent.symbol, intent.side,
                                     intent.qty, price, post_only=True)
    elif mode == "TWAP":
        slicer = TWAPSlicer(intent, duration=intent.urgency.duration)
        order = slicer.execute()
    # ...

    await order.confirmation(timeout=1000)
    return OrderResult(status="placed", venue=venue, order=order)
```

## 11.6 Orderbook Analysis

- Snapshot + diff updates persistidos por símbolo.
- Features derivadas: `imbalance_top5`, `depth_total`, `weighted_mid`.
- Anomaly detection: spread spike, depth collapse → señal `lob.anomaly`.

## 11.7 Funding Analysis

- Tracking de funding rate por perpetual.
- Feature `funding_z_60d` para mean-reversion.
- Strategy `funding_carry`: long el lado con funding pago.

## 11.8 Fair Value Estimation

- Mid-price con ajuste por imbalance (Stoikov).
- Cross-venue VWAP como ground truth.
- Discrepancia > 5 bps cross-venue → señal de arbitraje.

---

# 12. RISK MANAGEMENT

## 12.1 Exposure Control

- **Per-instrument cap**: max 5% del equity en un símbolo.
- **Per-sector cap** (crypto): max 30% en altcoins (ex-BTC, ex-ETH).
- **Per-strategy cap**: definido por allocation del meta-coordinador.

## 12.2 Drawdown Protection

- **Daily kill**: si DD intraday > 3% → pausar todas las nuevas órdenes.
- **Weekly kill**: si DD semanal > 7% → modo solo cierre.
- **Monthly kill**: si DD mensual > 12% → freeze + revisión humana.

## 12.3 Volatility Adaptation

- Sizing escala inverso a volatilidad realizada (`risk = target / vol`).
- Vol regime alto → fracción de Kelly se reduce (× 0.5).
- Vol regime extremo → modo defensivo.

## 12.4 Portfolio Allocation

- Mean-variance optimización con shrinkage de Ledoit-Wolf.
- Restricciones: long-only en sub-portfolios swing; long-short en stat-arb.
- Rebalance: diario para tactical, semanal para strategic.

## 12.5 Correlation Control

- Matrix de correlación rolling 60d.
- Si dos posiciones tienen corr > 0.8 → trata como una sola para sizing.
- Diversificación obligatoria: max correlación promedio del libro < 0.4.

## 12.6 Dynamic Leverage

- Leverage = f(volatility, drawdown, regime, available_margin).
- Cap absoluto: 3x para crypto, 2x para FX.
- Tightening automático ante regime shift.

## 12.7 Kill Switch

- **Trigger automático**:
  - DD > limits
  - Latency a exchange > 5 s sostenida
  - Discrepancia internal vs exchange positions
  - Errores de modelo (NaN probas, calibrator fail)
- **Trigger manual**: comando MCP `risk.kill_switch.activate()`.
- **Effect**: cancel all open orders, close al cierre del día, freeze new signals.

## 12.8 Anomaly Protection

- Outlier detection en P&L tick-by-tick (z-score > 5 → alert).
- Sanity checks en cada feature antes de inferencia.
- Predict probas que rompen invariantes (sum != 1, NaN) → reject signal.

## 12.9 Implementación actual

| Función | Archivo | Estado |
|---------|---------|--------|
| Kelly sizing | `risk/kelly.py` | Implementado |
| Dynamic R:R | `risk/dynamic_rr.py` | Implementado |
| Bayesian sizing | `risk/bayesian_sizer.py` | Implementado |
| Portfolio caps | `risk/portfolio.py` | Pendiente |
| Kill switch | `risk/kill_switch.py` | Pendiente |
| Drawdown protection | `risk/drawdown.py` | Pendiente |

---

# 13. IA AUTÓNOMA

## 13.1 Agentes inteligentes

Capa de agentes encima del stack ML+risk:

| Agente | Responsabilidad | Stack |
|--------|----------------|-------|
| `Researcher` | Backtesting, hypothesis testing, paper drafting | Claude + MCP tools |
| `Allocator` | Asigna capital entre estrategias | Bandit + RL |
| `Monitor` | Detecta anomalías, escala alertas | Rule-based + Claude |
| `Reactor` | Ejecuta acciones defensivas (reducir sizing) | Rule-based |
| `Reporter` | Genera summaries diarios/semanales | Claude |

## 13.2 Coordinación multiagente

- **Topic bus** (Kafka `agents.*`) para comunicación.
- **Shared state** en Redis (cooperación) y MongoDB (memoria).
- **Voting / consensus** en decisiones de alto impacto (e.g. cambio de allocation
  > 10%).

## 13.3 Selección automática de estrategias

```python
def select_strategies(regime: MarketRegime,
                     performance: Dict[str, StrategyPerf]) -> Dict[str, float]:
    """
    Thompson sampling sobre Beta(α, β) por estrategia.
    α = wins, β = losses en ventana 90d, decayed.
    Devuelve allocations que suman 1.
    """
    sampled = {s: np.random.beta(p.wins + 1, p.losses + 1)
               for s, p in performance.items()}
    if regime.is_high_vol():
        sampled = filter_low_risk_strategies(sampled)
    return softmax_normalize(sampled)
```

## 13.4 Detección de régimen

- GMM (3–4 componentes) sobre features `[vol_realized, trend_strength,
  return_skew]`.
- HMM como alternativa (modela transiciones).
- Change point detection (Bayesian online) para alertas tempranas.
- Outputs: `regime_probs[k]`, `regime_label`, `regime_stability`.

## 13.5 Razonamiento contextual

- Claude como interfaz de razonamiento sobre eventos del bot.
- Prompts con context completo: positions actuales, señales recientes,
  régimen, P&L, alerts.
- Tool use vía MCP server.

## 13.6 Memoria persistente

- **Episodic**: log de decisiones importantes en Postgres (`audit.decisions`).
- **Semantic**: embeddings de eventos en vector DB (pgvector).
- **Procedural**: políticas aprendidas (RL Q-tables, allocation history).

## 13.7 Autoevaluación

- Reflection diaria: el `Reporter` agent revisa decisiones del día y marca
  cada una como `good / neutral / bad` basado en outcome.
- Update de prior beliefs sobre estrategias.

## 13.8 Aprendizaje continuo

- Online learning controlado (no en risk-critical paths).
- River.py o similar para features no-stacionarias.
- A/B testing como mecanismo principal de validación.

---

# 14. DASHBOARD

## 14.1 Stack frontend

- Next.js 14 + TypeScript.
- Componentes shadcn/ui + Tailwind.
- Charts: `lightweight-charts` para precio, `recharts` para métricas.
- WS via SSE o Socket.IO.

## 14.2 Páginas

| Ruta | Vista | Audiencia |
|------|-------|-----------|
| `/` | Overview: P&L, exposure, alerts | Operador |
| `/strategies/:id` | Per-strategy deep dive | PM |
| `/signals` | Live signals con explanation | PM |
| `/positions` | Current book | Trader |
| `/risk` | Risk dashboard | Risk officer |
| `/ml/models` | Model registry + drift | ML eng |
| `/research/notebooks` | Embedded Jupyter | Researcher |
| `/admin` | RBAC, configs | Admin |

## 14.3 Realtime

- WS bridge: `dashboard-api` consume Kafka topics → forward al cliente.
- Updates: P&L cada 1s, signals al instante, positions al instante.

## 14.4 Alerts

- Whale alerts (on-chain) via WS.
- Macro alerts (FOMC, NFP) via cron.
- Risk breaches via PagerDuty integration.

---

# 15. ESTRUCTURA DEL REPOSITORIO

## 15.1 Estado actual — MONOREPO (2026-05-14)

### `quant_bot/` (núcleo ML — este repo)

```
quant_bot/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── data/                       # raw data, parquet/csv
├── features/
│   ├── __init__.py
│   ├── pca_denoiser.py
│   ├── regime.py               # GMM
│   ├── technical.py            # RSI, MACD, etc.
│   └── triple_barrier.py       # labeling
├── models/
│   ├── __init__.py
│   ├── zoo.py                  # BaseModel, LogReg, XGB, MLP, LSTM
│   ├── calibration.py          # IsotonicCalibrator
│   ├── meta_labeler.py
│   ├── validation.py           # WalkForwardSplitter
│   ├── walk_forward_runner.py
│   ├── entry_filter.py
│   ├── feature_selection.py
│   ├── metrics.py              # CS229 metrics
│   ├── error_analysis.py
│   ├── ablative_analysis.py
│   └── rl_agent.py             # Q-learning
├── risk/
│   ├── __init__.py
│   ├── kelly.py
│   ├── dynamic_rr.py
│   └── bayesian_sizer.py
├── execution/                  # MVP
├── tests/
│   ├── test_pca_denoiser.py
│   ├── test_meta_labeler.py
│   ├── test_bayesian_sizer.py
│   ├── test_metrics_cs229.py
│   └── test_deep_learning.py
├── examples/
│   └── pipeline_ml_real_data.py
└── notebooks/                  # research
```

### `los_ojos/` (plataforma de producción — repo separado)

```
los_ojos/                           C:\Users\alexj\OneDrive\Desktop\los_ojos\
├── docker-compose.yml              Stack completo (build context = root)
├── Makefile                        make up/infra/services/db-shell/kafka-create-topics
├── .env.example                    Template de variables de entorno
├── libs/shared/                    Shared library (Pydantic events, kafka, redis, db)
│   ├── events.py                   Todos los Kafka schemas + KafkaTopics registry
│   ├── kafka_client.py             Producer (retry+DLQ) + Consumer (async gen)
│   ├── redis_client.py             Cache + PubSub
│   └── db.py                       PostgresPool (asyncpg) + MongoClient (motor)
├── services/
│   ├── market-intelligence/        OpenBB + Binance orderbook + funding rate
│   ├── macroeconomic/              FRED + Sahm Rule + yield curve + macro regime
│   ├── onchain-analysis/           Crucix whale detection + exchange flows
│   ├── context-engine/             GMM regime classifier (5 componentes)
│   ├── realtime-signal/            FastAPI WebSocket server (Kafka→Redis→WS)
│   ├── ml-feature-store/           Feature computation + serving
│   └── strategy-orchestrator/      Bot config, kill switch, señal generation
├── frontend/                       React + TypeScript + Vite + Tailwind
│   └── src/
│       ├── components/
│       │   ├── TradingDashboard.tsx
│       │   ├── TradingChart.tsx    lightweight-charts + signal markers
│       │   └── BotConfigPanel.tsx  Bot config UI (mode, riesgo, strategies)
│       ├── hooks/useWebSocket.ts
│       └── types/index.ts
├── infra/
│   ├── sql/schema.sql              TimescaleDB: 18 tablas, 7 schemas, hypertables
│   └── kafka/topics.yml            14 topics (los_ojos.*)
└── monitoring/
    ├── prometheus.yml              Scrape de 7 servicios + infra
    └── grafana/dashboards/         trading.json pre-cargado
```

URLs locales: Dashboard `http://localhost:3000` · Kafka UI `http://localhost:8080` · Grafana `http://localhost:3001`

## 15.2 Estructura target (PROJECT ML monorepo)

```
project-ml/
├── CLAUDE.md
├── README.md
├── docs/
│   ├── architecture/
│   ├── strategies/
│   └── runbooks/
├── services/
│   ├── ingestion-binance/      # Rust
│   ├── ingestion-fx/           # Python
│   ├── normalizer/             # Python
│   ├── feature-engine/         # Python (numba)
│   ├── ml-inference/           # FastAPI
│   ├── signal-router/          # Go
│   ├── risk-engine/            # Python
│   ├── executor/               # Go
│   ├── dashboard-api/          # FastAPI
│   └── dashboard-web/          # Next.js
├── libs/
│   ├── py-models/              # antes models/ del monolito
│   ├── py-risk/                # antes risk/
│   ├── py-features/            # antes features/
│   ├── py-feature-store-sdk/
│   ├── py-mcp-tools/
│   └── shared-schemas/         # protobuf/avro
├── pipelines/
│   ├── training/               # Airflow DAGs
│   ├── data-ingestion/
│   └── retraining/
├── infra/
│   ├── docker/
│   ├── helm/
│   ├── terraform/
│   └── ansible/
├── ml/
│   ├── notebooks/              # exploración
│   ├── experiments/            # MLflow runs
│   └── registry/               # model artifacts metadata
├── ops/
│   ├── runbooks/
│   ├── playbooks/
│   └── dashboards/             # Grafana JSON
├── tools/
│   ├── mcp-server/
│   └── cli/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── .github/
    └── workflows/
```

## 15.3 Convenciones

- **Snake_case** Python, **PascalCase** TS/Go types, **kebab-case** dirs.
- **Tests siempre al lado o en `tests/`** con sufijo `_test.py` o `test_*.py`.
- **README por servicio** con: propósito, arranque local, env vars, SLOs.
- **OpenAPI/AsyncAPI** spec por servicio público.

---

# 16. DESPLIEGUE

## 16.1 Entornos

| Entorno | Propósito | Datos | Capital |
|---------|-----------|-------|---------|
| `local` | Desarrollo | Mocks o snapshot | N/A |
| `dev` | Integración | Datos reales delayed | Paper |
| `staging` | Pre-prod | Datos realtime | Paper |
| `prod-canary` | Canary | Datos realtime | 5% capital |
| `prod` | Full | Datos realtime | 100% capital |

## 16.2 Local

```bash
# 1. Setup
make setup
docker-compose up -d postgres redis kafka

# 2. Run services
make run-service SERVICE=ml-inference
make run-dashboard

# 3. Tests
pytest tests/ -v
```

## 16.3 Staging

- GitHub Actions push a `main` → build → deploy automático.
- Smoke tests post-deploy: heartbeat endpoints + sample inference.

## 16.4 Production

- Manual approval gate.
- Blue/green deploy via Helm rollback strategies.
- Health checks + rollback automático si error rate > umbral.

## 16.5 Cloud Deployment

- **Provider**: AWS preferred (Datadog ya integrado, ELB, EKS).
- **Multi-AZ**: 3 AZ en la región principal.
- **Multi-region** (futuro): replicación read-only para resiliencia.

## 16.6 GPU Nodes

- Spot instances `g4dn.xlarge` para training.
- On-demand `g5.xlarge` solo para inferencia de modelos pesados.
- Cluster autoscaler scale-to-zero fuera de horario.

## 16.7 Secrets

- AWS Secrets Manager via External Secrets Operator en K8s.
- Rotation policy: keys de exchange cada 30 días, DB passwords cada 90 días.
- Nunca en `.env` committeados; `.env.example` solo con placeholders.

## 16.8 Scaling

- HPA basado en CPU + custom metrics (Kafka lag).
- Predictive scaling para horarios conocidos (sesión NY open).

---

# 17. SEGURIDAD

## 17.1 Authentication

- **JWT** firmados con clave rotativa (RS256).
- **TTL**: 15 min access, 7 d refresh.
- **MFA** obligatorio para operadores con permiso de ejecución.

## 17.2 API Gateway

- **Kong** o **Envoy** delante de servicios públicos.
- **Rate limiting** por usuario y por endpoint.
- **WAF** para signature attacks comunes.

## 17.3 RBAC

| Rol | Permisos |
|-----|----------|
| `viewer` | Read positions, P&L, signals |
| `researcher` | + MCP tools de research (no ejecución) |
| `trader` | + cancel orders, adjust limits |
| `operator` | + start/stop services, trigger retraining |
| `admin` | Todo |

## 17.4 Secrets Management

- Vault + Auto-unseal.
- Secret injection via init containers, nunca en env de pod.
- Audit log de cada acceso.

## 17.5 Encryption

- **At rest**: AES-256 en discos EBS, S3 con SSE-KMS.
- **In transit**: TLS 1.3 en todos los endpoints, mTLS entre servicios internos.
- **End-to-end**: payloads sensibles (e.g. credenciales de broker) cifrados
  client-side.

## 17.6 Audit Logs

- Append-only en Postgres (`audit.*`).
- Cada acción incluye: `actor`, `timestamp`, `action`, `payload_hash`, `outcome`.
- Retention 7 años.

## 17.7 Network Isolation

- VPC privada.
- Public subnets solo para LB.
- DB en private subnets sin internet access.
- Egress filtering: solo IPs whitelisteadas para exchanges.

---

# 18. ROADMAP

## 18.1 Fase 1 — Núcleo cuantitativo (DONE — Q1 hipotético)

- [x] WalkForwardRunner
- [x] Modelos: Logistic, XGBoost, LSTM, DeepMLP
- [x] Calibración (isotonic + sigmoid)
- [x] Triple barrier labels
- [x] Kelly + Dynamic R:R
- [x] GMM regime detection
- [x] PCA denoising
- [x] Meta-labeling
- [x] Bayesian sizing
- [x] CS229 metrics (F1/AUC/CM/bias-variance)
- [x] Error analysis
- [x] Ablative analysis
- [x] Q-learning agent (tabular)

## 18.2 Fase 2 — Plataforma de datos (**EN PROGRESO** — Q2)

- [ ] Migración a monorepo `project-ml/`
- [x] Feature Store online (Redis) — ml-feature-store service en los_ojos
- [x] TimescaleDB schema — 18 tablas, hypertables, compression policies
- [x] Ingestion services — Binance WS (orderbook + funding) en market-intelligence
- [x] Kafka backbone — 14 topics en los_ojos.\* + shared kafka_client
- [ ] Schema Registry + Avro contracts (actualmente Pydantic JSON)
- [x] Los Ojos integraciones: OpenBB, fredapi, Crucix, Binance WS
- [x] Macro regime detection (Sahm + yield curve + leading indicators)
- [x] Whale detection + smart money flow (on-chain)
- [x] GMM regime classifier (context-engine, 5 componentes)
- [x] Realtime WebSocket service (Kafka → Redis PubSub → WS)
- [x] Dashboard React/TS con BotConfigPanel (paper/live, kill switch, risk sliders)
- [x] docker-compose stack completo (7 microservicios + infra + monitoring)
- [ ] Polymarket signals integration
- [ ] SEC research service (Dexter API)
- [ ] **ResMLPClassifier reemplaza DeepMLP en multi-horizon trainer** (ADR-034, post paper run 2026-06-19, shadow trading → A/B 30d → canary)

## 18.2b Alpaca paper trading — roadmap 12 semanas (**COMPLETO** — 2026-06-03)

- [x] S1–S9: Universe, features, market calendar, brackets, risk gate PDT, reconciler (ver `alpaca_integration.md §5`)
- [x] S10: Observabilidad — ALERT-004/005/006/007/008 (`platform/monitoring/rules/alpaca.yml`), DRILL-004 21/21 PASS
- [x] S11: DAG nocturno — `research/pipelines/nightly_retrain.py`, gates DSR/ECE, 17/17 tests, JSON run log
- [x] S12: Hardening — circuit breaker CLOSED→OPEN→HALF_OPEN, RiskGate kill switch step-0, ADR-035 (SLO), runbook ops, handoff doc
- [x] P1-001: Circuit breaker (`app/brokers/_alpaca/circuit_breaker.py`)
- [x] P1-002: RiskGate kill switch propagation (REST + Kafka, `risk_gate.py`)
- [ ] **Pendiente**: diagnosticar 0 trades en W1-W2 (pipeline señal→executor conectividad)

## 18.3 Fase 3 — Servicios productivos (Q3)

- [ ] ml-inference service dedicado (cargar artifacts de quant_bot)
- [ ] risk-engine service completo (portfolio caps, drawdown protection)
- [ ] executor service (paper trading → live)
- [x] dashboard-api + dashboard-web (los_ojos realtime-signal + frontend)
- [ ] CI/CD pipelines completos (GitHub Actions)
- [x] Observabilidad: Prometheus scrape + Grafana dashboards pre-cargados
- [ ] Loki + Tempo (logs + trazas distribuidas)

## 18.4 Fase 4 — Producción (Q4)

- [ ] K8s deployment con Helm umbrella
- [ ] Secrets management (Vault)
- [ ] Live trading canary 5% capital
- [ ] Multi-AZ HA
- [ ] Backup + disaster recovery drill

## 18.5 Fase 5 — Escalado y multi-mercado (Año 2 Q1)

- [ ] Forex live
- [ ] Futuros via IB
- [ ] Multi-strategy allocation con bandit
- [ ] Drift detection automatizado
- [ ] Retraining pipeline completo

## 18.6 Fase 6 — IA autónoma completa (Año 2 Q2–Q3)

- [ ] PPO/SAC en producción (shadow → canary)
- [ ] Multi-agent coordination
- [ ] MCP server con Claude como interfaz operativa
- [ ] Autoevaluación + memoria persistente
- [ ] Reporter agent (daily/weekly briefs)

## 18.7 Optimización y madurez (Año 2 Q4 → Año 3)

- [ ] Cross-region replication
- [ ] DMA dedicada
- [ ] FinOps optimization (spot, reserved instances)
- [ ] Compliance / audit prep
- [ ] Externalización opcional como SaaS

---

# 19. ESTÁNDARES DE CÓDIGO

## 19.1 Python

- **Versión**: 3.11+.
- **Estilo**: ruff (lint + format) + black.
- **Tipado**: mypy strict en libs/, gradual en services/.
- **Tests**: pytest + pytest-cov, coverage > 80% en libs/, > 60% en services/.
- **Docstrings**: Google style.
- **Imports**: isort, absolutos siempre.

```python
# Patrón canónico para una clase de modelo
class MyModel(BaseModel):
    """One-liner.

    Detalles, decisiones de diseño, referencias.

    Parameters
    ----------
    param1 : tipo
        Descripción.

    Examples
    --------
    >>> model = MyModel(param1=10)
    >>> model.fit(X, y)
    """
    name = "mymodel"

    def __init__(self, param1: int = 10):
        super().__init__(param1=param1)
        ...
```

## 19.2 Go

- gofmt + golangci-lint.
- Tests con `testing` + `testify`.

## 19.3 TypeScript

- ESLint + Prettier.
- Strict tsconfig.

## 19.4 Tests

- **Unit**: aislado, mocks de dependencias.
- **Integration**: con dependencias reales (Postgres, Redis en docker-compose).
- **E2E**: flujo completo end-to-end, en CI nocturno.

## 19.5 Patrones de diseño

- **Strategy** (modelos, calibradores, regimes).
- **Factory** (`get_model(name, **params)`).
- **Adapter** (exchanges, brokers).
- **Repository** (data access).
- **Observer** (event-driven).
- **Circuit Breaker** (resiliencia).

## 19.6 SOLID

- **S**: cada clase un motivo de cambio (`Calibrator` solo calibra).
- **O**: extensión via subclase, no modificación (añadir modelo nuevo no toca
  el runner).
- **L**: subclases respetan contratos (`BaseModel.predict_proba()` signature).
- **I**: interfaces pequeñas (`BaseRiskController` solo decide allow/deny + size).
- **D**: depender de abstracciones (`runner.model: BaseModel`).

## 19.7 Naming

- Funciones: verbo en infinitivo (`compute_metrics`, no `metrics()`).
- Booleanos: prefijo `is_`, `has_`, `should_`.
- Constantes: SCREAMING_SNAKE.
- Tipos genéricos: `T`, `K`, `V`; específicos: `TModel`, `TFeature`.

## 19.8 Documentación

- Cada módulo público tiene docstring de nivel módulo.
- Cada clase pública tiene docstring con `Parameters`, `Examples`.
- Decisiones arquitectónicas no triviales → ADR en `docs/adr/NNN-title.md`.

## 19.9 Anti-patterns prohibidos

- ❌ Modelos que aceptan `**kwargs` sin documentar.
- ❌ Magic numbers en código (constantes nombradas).
- ❌ Bloques try/except sin tipo específico.
- ❌ Print statements (usar logging).
- ❌ Estado global mutable.
- ❌ Side effects en imports.
- ❌ Mocks deep en tests unitarios (señal de mal diseño).

---

# 20. REGLAS PARA CLAUDE

Esta sección define cómo Claude (cualquier instancia, cualquier herramienta) debe
operar en este repositorio. Es **vinculante** y persiste entre sesiones.

## 20.1 Razonamiento

1. **Lee antes de escribir.** Antes de modificar un archivo, leerlo entero o las
   secciones relevantes. No asumir API basado en nombres.
2. **Verifica contratos.** Si un módulo declara invariantes (e.g. proba suma a 1,
   anti-leakage), respetarlos.
3. **Walk-forward o nada.** Cualquier nueva métrica o resultado debe pasar
   por `WalkForwardRunner`. Backtests in-sample no se reportan.
4. **Cita el archivo.** En respuestas técnicas, citar `models/walk_forward_runner.py:657`
   en lugar de "el runner".

## 20.2 Mantener contexto

1. **Consultar este `CLAUDE.md`** al inicio de cada sesión nueva o tras
   `/clear`.
2. **Resumir cambios** en commits con co-autoría
   `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.
3. **Actualizar `CLAUDE.md`** cuando se introducen cambios arquitectónicos
   (nuevo servicio, nueva integración, cambio de stack).
4. **No reinventar**: si un módulo ya existe (e.g. `IsotonicCalibrator`),
   extenderlo, no duplicar.

## 20.3 Documentar cambios

1. **Cada nuevo módulo público** lleva docstring de nivel módulo con: propósito,
   decisiones de diseño, referencias.
2. **ADR** para decisiones que cambien la arquitectura o stack.
3. **Tests** acompañan cada feature nuevo, no como anécdota.
4. **Changelog** entries en commits descriptivos (no `wip`, no `update`).

## 20.4 Generar código

1. **Tipado estricto** en libs/. Si no puedes tipar algo, repensar el diseño.
2. **Reutilizar abstracciones**: `BaseModel`, `BaseFeatureGenerator`, etc.
3. **Anti-leakage por defecto**: cada transformador hace `fit` solo sobre datos
   de entrenamiento. Si la API permite ambigüedad, documentarla.
4. **No introducir dependencias nuevas** sin justificación explícita y entrada
   en `pyproject.toml`.
5. **Tests primero o tests con código** — nunca commitear sin tests si el módulo
   es público.

## 20.5 Respetar la arquitectura

1. **Capas**: `features → models → risk → execution`. Nunca al revés.
2. **No imports cíclicos**.
3. **Servicios no comparten DBs** (en target architecture). Comunicación via
   Kafka o REST.
4. **Schemas en `shared-schemas/`**, nunca duplicar tipos entre servicios.

## 20.6 Evitar deuda técnica

1. **No TODO sin owner y fecha.** Formato: `# TODO(@alex 2026-06-01): ...`.
2. **No comments-as-docs.** Si necesita explicación larga, sacar a markdown.
3. **No optimizar prematuramente.** Profilar antes.
4. **Refactor en mismo PR si toca el código vecino.** "Boy Scout rule".

## 20.7 Trabajo modular

1. **Un cambio = un PR.** Mezclar refactor + feature + bugfix → reject.
2. **Commits atómicos** dentro del PR.
3. **Branch naming**: `feat/`, `fix/`, `refactor/`, `docs/`, `chore/`.

## 20.8 Diseñar componentes desacoplados

1. **Strategy pattern** para variantes.
2. **DI por constructor** (no globales).
3. **Config como datos** (dataclasses con defaults), no como código.
4. **Eventos sobre llamadas** cuando hay > 2 consumidores potenciales.

## 20.9 Consistencia enterprise-grade

1. **Naming consistente** entre lenguajes: `feature_set_hash` en Python ↔
   `featureSetHash` en TS ↔ `feature_set_hash` en Go.
2. **Time = UTC**, siempre. Nunca naive datetimes.
3. **Money = Decimal**, nunca float, en cualquier valor monetario.
4. **IDs = UUID v7** (sortable, time-prefixed).
5. **Errors = tipados**: `class ModelNotCalibratedError(QuantBotError)`.

## 20.10 Operacional

1. **Nunca commit secrets**. Si Claude detecta una clave en cambios, abortar.
2. **Nunca correr código contra producción** sin doble confirmación humana.
3. **Nunca borrar datos** sin backup verificado.
4. **Logs estructurados** (structlog), no `print`.
5. **Tests en CI deben pasar** antes de merge. Si fallan, investigar la causa,
   no skipear.

## 20.11 Workflow recomendado para Claude en este repo

```
1. Si la sesión es nueva → leer CLAUDE.md + MEMORY.md.
2. Si el usuario pide implementar algo:
   a. Identificar archivos relevantes (Glob/Grep).
   b. Leer esos archivos completos.
   c. Plantear plan corto (no >5 puntos).
   d. Confirmar con usuario si la decisión es no-trivial.
   e. Implementar con tests.
   f. Verificar con pytest (si autorizado).
   g. Resumir cambios con archivos tocados.
3. Si el usuario pide debugging:
   a. Reproducir antes de hipotetizar.
   b. Bisect entre commits si aplica.
   c. Fix mínimo + test de regresión.
4. Si el usuario pide research:
   a. Usar agentes (subagent_type=general-purpose) para búsquedas amplias.
   b. Citar archivos y líneas.
   c. Producir output estructurado.
```

## 20.12 Lo que Claude NO hace en este repo

- Crear archivos `*.md` o documentación a menos que se pida explícitamente.
- Añadir emojis en código o documentación a menos que se pida.
- Lanzar órdenes reales contra exchanges.
- Modificar configuraciones de risk-engine sin confirmación humana.
- Borrar tests existentes.
- Modificar `CLAUDE.md` sin pedir permiso (excepto si el cambio es directamente
  solicitado por el usuario).

---

## APÉNDICE A — Glosario

| Término | Definición |
|---------|------------|
| **PSR** | Probabilistic Sharpe Ratio. P(SR real > benchmark). |
| **DSR** | Deflated Sharpe Ratio. PSR ajustado por multiple testing. |
| **ECE** | Expected Calibration Error. < 0.05 = bien calibrado. |
| **Brier Score** | MSE de probabilidades. Lower = better. |
| **Triple barrier** | Labeling con barreras de TP/SL/timeout. |
| **Embargo** | Buffer temporal entre train y test para evitar leakage. |
| **Walk-forward** | Validación temporal con folds expansivos o rolling. |
| **Meta-labeling** | Segundo classifier que filtra señales primarias. |
| **Kelly fraction** | Sizing óptimo log-utility. Usar 0.10–0.25 en práctica. |
| **Funding rate** | Costo periódico de mantener perpetual; revierte spot-perp. |
| **DLQ** | Dead Letter Queue. Mensajes fallidos para inspección. |
| **PSI** | Population Stability Index para drift detection. |
| **SwiGLU** | Activation gated (Shazeer 2020): `SwiGLU(x) = (xW + b) ⊙ swish(xV + c)`. Empíricamente mejor que ReLU/GELU en tabular con baja SNR. |
| **Temperature Scaling** | Post-hoc calibration de NNs (Guo et al. 2017): divide logits por `T` aprendido en val set para reducir ECE sin afectar accuracy. |
| **ResBlock** | Bloque residual `y = x + f(LayerNorm(x))` (He et al. 2015). Estabiliza gradientes en NN profundas; permite > 4 capas sin degradación. |
| **ResMLP** | MLP profunda con bloques residuales, BatchNorm/LayerNorm, SwiGLU y dropout. Reemplaza DeepMLP en multi-horizon trainer (ADR-034). |

## APÉNDICE B — Referencias canónicas

- López de Prado, *Advances in Financial Machine Learning* (2018).
- Bailey & López de Prado, *The Deflated Sharpe Ratio* (2014).
- Mertens, *Comments on Variance of the IID estimator of Sharpe* (2002).
- Stanford CS229 Tips & Tricks cheatsheet (Amidi & Amidi, 2018).
- Stanford CS229 Deep Learning cheatsheet.
- Stoikov & Avellaneda, *High-frequency trading in a limit order book* (2008).

## APÉNDICE C — Decisiones arquitectónicas registradas

> Lista corta; ADRs completas en `docs/adr/`.

| ADR | Decisión | Status |
|-----|----------|--------|
| 001 | Walk-forward como única métrica oficial | Accepted |
| 002 | Calibración isotonic + sigmoid, no Platt puro | Accepted |
| 003 | Kelly cuarto (0.25) como default, nunca full | Accepted |
| 004 | PCA excluye columnas `regime_*` | Accepted |
| 005 | Bayesian update via product of experts | Accepted |
| 006 | Q-learning tabular antes de DQN/PPO | Accepted |
| 007 | Kafka + Pydantic JSON (MVP) → Avro Schema Registry (target) | Accepted |
| 008 | Postgres + Timescale; no MongoDB en path crítico | Accepted |
| 009 | RL no decide risk limits | Accepted |
| 010 | UTC + Decimal + UUID v7 en toda la plataforma | Accepted |
| 011 | Los Ojos como repositorio separado (`los_ojos/`) hasta merge al monorepo | Accepted |
| 012 | Build context de Docker en root (`.`) para acceso a `libs/shared/` | Accepted |
| 013 | Kafka → Redis PubSub → WebSocket (no Kafka directo a WS) | Accepted |
| 014 | GMM 5 componentes con semantic label mapping via centroides | Accepted |
| 034 | ResMLPClassifier reemplaza DeepMLPClassifier en multi-horizon trainer | Proposed |

---

**Última actualización**: 2026-06-03 (sesión S12 — roadmap Alpaca 12 semanas COMPLETO; P1-001/P1-002 cerrados; ADR-035 SLO; runbook ops; NightlyRetrainDAG 17/17 tests)
**Maintainers**: Alex (lead), Claude (AI assistant)
**Status**: Living document. Actualizar con cada cambio arquitectónico.
