# Integración Alpaca — Arquitectura y roadmap (PROJECT ML / `quant_bot`)

> **Supuestos (encuesta omitida ×2)**  
> - Universo: **(d) mixto** equities US + crypto vía Alpaca donde aplique.  
> - Horizonte primer release: **(b) swing 4h–5d** (alineado con estrategia swing en `CLAUDE.md` §4.2).  
> - Estado 12 semanas: **(a) paper en producción 100%** (sin live capital).  
> - Formato: **(a)** este `.md` (override explícito a `CLAUDE.md` §20.12 pedido por el operador).

> **Nota de paths**: en código real el detector GMM vive en `research/features/regime_gmm.py` (`GMMRegimeDetector`), no en `regime.py` (nombre histórico en §15.1 de `CLAUDE.md`).

> **Aviso**: documento de arquitectura — **no promete rentabilidad**. Métricas de aceptación solo **OOS** (PSR/DSR, ECE), nunca IS como KPI.

---

## 0. Alineación con `CLAUDE.md` (fuente de verdad)

| Referencia | Implicación para Alpaca |
|------------|------------------------|
| §1.2 (walk-forward o nada) | Todo modelo/evaluación Alpaca pasa por `WalkForwardRunner`; reportar solo concatenación OOS. |
| §2.1–2.2 (topología + SLO) | Ver **§10.1** de este doc: reconciliación con SLO **executor \<50 ms p99** de `CLAUDE.md` §2.2. |
| §2.5–2.6 (flujo inferencia / training) | Reuso de calibración, meta-label, Bayesian update, Kelly — sin duplicar módulos. |
| §6.2–6.3 (`models/zoo.py`, calibración) | `BaseModel`, `IsotonicCalibrator` extendidos, no reescritos. |
| §11–12 (ejecución + riesgo) | Orden: `features → models → risk → execution` (`CLAUDE.md` §20.5). |
| §15.2 (estructura target monorepo) | Diff de carpetas en **§8**; no reestructurar todo el repo en fase Alpaca. |
| §18.1 (Fase 1 DONE) | Roadmap **no** repite implementar WF/Kelly/calibración desde cero — solo integración y hardening. |
| §20.9 (UTC, Decimal, UUID v7) | Schemas internos: `shared/quant_shared/schemas/orders.py` (`_uuid7`, `Decimal`). |
| §20.12 | Este archivo existe **solo** por solicitud explícita del operador. |

---

## 1. ARQUITECTURA EN 6 CAPAS

> Para cada capa: **Resumen** · **Fundamentación** · **Arquitectura** · **Riesgos (≥3)** · **Mejoras (≥2)** · **Investigación futura**.

### 1.1 Capa A — Datos

**Resumen técnico (3–5 líneas)**  
Ingesta OHLCV y corporate actions para símbolos Alpaca (equities + crypto), almacenamiento offline alineado con Timescale/Parquet, y **feature freshness** antes de inferencia. Reutiliza patrones de `CLAUDE.md` §7 (Feature Store) sin duplicar pipelines crypto.

**Fundamentación**  
Como en un experimento de laboratorio: si los reactivos (datos) están contaminados, el resultado (señal) es basura — **GIGO**. Para swing 4h–5d, el edge está en coherencia temporal **train vs live** (mismo vendor, mismos ajustes).

**Arquitectura — tabla de componentes**

| Componente | Reusar (path) | Nuevo (path propuesto) | Responsabilidad | SLO p99 | Test asociado |
|------------|---------------|------------------------|-----------------|--------|---------------|
| Ingesta histórica multi-activo | `data/ingestion.py` (CCXT) — patrón; **nuevo conector Alpaca Data** | `research/data/alpaca_bars.py` o `platform/services/alpaca-ingest/` | Descarga barras Alpaca, Parquet particionado `symbol/date` | \< 2 s/bar batch agg | `tests/test_alpaca_bars.py` |
| Universo top-200 + filtros ADV | — | `research/universe/alpaca_equity_universe.py` | Lista rebalanceable, líquidos | \< 30 s/rebuild | pytest + snapshot |
| Corporate actions | — | `research/data/corporate_actions_alpaca.py` | splits/dividends → ajuste de qty/notional interno | \< 5 s/symbol/day | golden files |
| Feature offline | `research/features/*`, `features/regime_gmm.py` | envoltorio `research/features/alpaca_compat.py` | Misma API de columnas research + validadores | job batch | WF smoke |
| Feature online (opc.) | `platform/services/ml-feature-store/` | topic `features.alpaca.{symbol}` | Vector Redis alineado a bar close | \< 100 ms compute | integration |

**Riesgos y limitaciones (≥3)**  
1. **Survivorship**: universos “solo vivos” inflan backtest — documentar universo por fecha.  
2. **Sesgo de vendor**: Alpaca vs otros proveedores pueden diferir en splits ajustados.  
3. **Latencia de datos free vs paid**: throttling distinto; riesgo de stale bars en producción.  
4. **Crypto 24/7 vs equities RTH**: mezclar sin flags `session` contamina features.

**Mejoras (≥2, trade-offs)**  
1. **DuckDB sobre Parquet** (rápido, local) vs **Timescale** (operación, más coste).  
2. **Un solo job diario** de ingest (simple) vs **streaming** (complejo, menor lag).

**Próximas líneas de investigación**  
- Ponderación por **ADV** y **spread** para filtrar small-caps ruidosos.  
- Detección de **datos fantasma** (PSI entre histórico y live feed).

---

### 1.2 Capa B — ML

**Resumen**  
Mismos artefactos que crypto: `WalkForwardRunner` ensambla `fit → calibrate → entry_filter → meta/Bayesian` sin fork.

**Fundamentación**  
El modelo es un **termómetro calibrado**: `predict_proba` debe reflejar frecuencias reales (ECE) para que Kelly y umbrales tengan sentido — ver `models/calibration.py` y el protocolo en docstring de `IsotonicCalibrator` (`models/calibration.py:18–33`).

**Arquitectura**

| Componente | Reusar | Nuevo | Responsabilidad | SLO p99 | Test |
|------------|--------|-------|-----------------|--------|------|
| Model zoo | `research/models/zoo.py` (`BaseModel` `fit`/`predict_proba` `30:52`) | — | XGBoost / MLP según spec | inferencia \<100 ms/símbolo (servicio futuro) | `tests/test_*` existentes |
| Walk-forward | `research/models/walk_forward_runner.py` (`1:63` imports `kelly`, `dynamic_rr`) | configs YAML Alpaca | Solo métricas OOS | horas (batch) | WF tests |
| Calibración | `models/calibration.py` | — | ECE \< 0.05 target | — | calibration report |
| Régimen GMM | `research/features/regime_gmm.py` (`GMMRegimeDetector` `92:99`) | — | `regime_label` para `BayesianWinUpdater` | — | reuse |
| Bayesian | `research/risk/bayesian_sizer.py` (`REGIME_LABEL_COL` `48:49`) | — | Posterior `p_win` | — | existente |

**Riesgos (≥3)**  
1. **Label leakage** si barras US no respetan horario (ej. usar high del día siguiente sin timezone).  
2. **Class imbalance** en equities puede ser peor que crypto.  
3. **Drift** post-IPO / post-earnings no capturado por GMM estático intra-fold.  
4. **Correlation≠causalidad**: features macro que solo predicen régimen de mercado pueden desaparecer OOS.

**Mejoras**  
1. **Nested walk-forward** para Optuna (caro, más honesto).  
2. **Purged CV** ya en espíritu `validation.py` — endurecer labels multi-clase.

**Investigación**  
- Transfer learning desde modelo crypto a equities (si covarían).  
- **Meta-label** específico para “gap risk” overnight.

---

### 1.3 Capa C — Estrategia

**Resumen**  
Señal = dirección + `p_win` + SL/TP desde `risk/dynamic_rr.py` + tamaño vía `KellyAtrSizer` (`risk/kelly.py` `66:100` `kelly_fraction_binary`). Misma jerarquía que `CLAUDE.md` §2.4.

**Fundamentación**  
La estrategia es el **directorio de orquesta**: no “predice precio”, traduce distribución condicional a **acciones con restricciones** (riesgo acotado).

**Arquitectura**

| Componente | Reusar | Nuevo | Responsabilidad | SLO p99 | Test |
|------------|--------|-------|-----------------|--------|------|
| Entry filter | `models/entry_filter.py` | thresholds por universo | Filtra `predict_proba` | — | WF |
| Dynamic R:R | `risk/dynamic_rr.py` | — | `DynamicRRManager` | — | unit |
| Kelly+ATR | `risk/kelly.py`, `risk/sizing_multi_asset.py` | `InstrumentSpec` para acciones | `qty` entera/fractional | — | unit |
| Portfolio (research) | `risk/management.py` `IntegratedRiskManager` | — | vol targeting + DD | — | backtest |
| Orchestrator (platform) | `platform/services/strategy-orchestrator/` | adapter Alpaca en feature vector | paper signals | \<500 ms e2e target `CLAUDE.md` §1.1 | integration |

**Riesgos (≥3)**  
1. **PDT** y restricciones de cuenta cash vs margin no modeladas en backtest.  
2. **Overfitting de umbral** si se re-optimiza cada fold sin guardrail DSR.  
3. **Horario**: señales cerca del close pueden ejecutarse mal en paper vs live.  
4. **Fracciones vs enteros**: desalineación notional real.

**Mejoras**  
1. Simulador de **reglas PDT** en backtest (complejidad alta, más realismo).  
2. **Multi-strategy** con `AllocationEngine` ya en platform (`strategy-orchestrator`).

**Investigación**  
- Optimal **staging** de órdenes (TWAP) para tamaños \> X% ADV.

---

### 1.4 Capa D — Ejecución (AlpacaAdapter)

**Resumen**  
**Ya existe** `AlpacaAdapter` en `platform/services/execution-engine/app/brokers/alpaca.py` (`169:193`) implementado sobre `alpaca-py` con llamadas sync envueltas en `asyncio.to_thread` (`alpaca.py:10:12`). La interfaz canónica es `BrokerAdapter` (`brokers/base.py:101:131`), **no** `BaseExecutor` — **ADR propuesta (016)**: renombrar en docs o añadir alias `BaseExecutor = BrokerAdapter` para compatibilidad con especificaciones legacy.

**Fundamentación**  
La ejecución es el **puente levadizo**: hasta que baja (órdene aceptada), el castillo (estrategia) es irrelevante. Idempotencia `client_order_id` ↔ `intent_id` UUID v7 reduce duplicados en retries.

**Arquitectura — contrato y SLO**

| Componente | Reusar | Nuevo | Responsabilidad | SLO p99 | Test |
|------------|--------|-------|-----------------|--------|------|
| Schemas | `shared/quant_shared/schemas/orders.py` (`OrderIntent` `96:118`, `_uuid7` `40:54`) | extensiones `extended_hours`, `notional` | Contrato interno | validación \<1 ms | pydantic |
| Adapter | `AlpacaAdapter` `alpaca.py` | ampliar tipos orden (bracket, OCO, trailing) | `submit` / `cancel` / `get_positions` | ver abajo | `test_alpaca_adapter.py` |
| Router | `execution-engine/app/routing.py` | — | routing multi-venue | — | tests |
| Risk gate | `execution-engine/app/risk_gate.py` | límites Alpaca-specific | bloqueo PRE submit | \< 20 ms `CLAUDE.md` §2.2 risk | `test_risk_gate.py` |
| Reconciler | `execution-engine/app/reconciler.py` | — | diff interno vs broker 60 s | ciclo 60 s | `test_reconciler.py` |
| Service | `execution-engine/app/service.py` | consume Kafka signals | orquestación | --- | `test_service.py` |

**Reconciliación explícita con `CLAUDE.md` §2.2**  
- §2.2 lista **`executor` (Go)** con **\<50 ms p99**.  
- La ruta **actual** es **Python + `AlpacaAdapter` + HTTP a Alpaca** (`alpaca.py:10:12`). Eso **no** puede garantizar 50 ms de punta a punta: la API remota domina.  
- **Postura recomendada**:  
  - **SLO interno v1 (paper)**:`submit` ack **\<200 ms p99** proceso local + **\<600 ms** incluyendo RTT Alpaca (medido).  
  - **ADR-017**: cuando el executor migrará a Go o a cola dedicada, **contraer** el SLO hacia §2.2.  
  - **Desacoplar**: el *risk-gate* local debe seguir §2.2 (\<20 ms); el *broker RTT* es otra métrica (`broker_latency_p99`).

**Extensiones requeridas (sin reinventar)**  
- **Order types**: hoy `MarketOrderRequest`, `LimitOrderRequest`, `StopLimitOrderRequest` (`alpaca.py:67:71`); faltan **bracket**, **OCO**, **trailing_stop** — mapear a SDK Alpaca v2 y extender `OrderType` enum en `orders.py` si el tipo no existe (`orders.py:66:72`).  
- **Idempotencia**: `OrderIntent.intent_id` ya UUID v7 (`orders.py:106:109`) — asegurar `client_order_id=intent_id` en submit Alpaca.  
- **Rate limit**: token bucket **200+200**/min — nuevo `rate_limit.py` en execution-engine.  
- **Retries**: solo 429/5xx, max 3, exp backoff+jitter.  
- **Circuit breaker**: \>5 errores 5xx/60 s → `read_only` mode 5 min — alinear con `CLAUDE.md` §2.7 / §11.4 espíritu.  
- **Errores tipados**: subclases de `BrokerError` (`base.py:44:45`) mapeadas desde códigos Alpaca (`InsufficientBuyingPowerError`, `PDTRuleViolationError`, etc.) — nuevo `brokers/alpaca_errors.py`.  
- **extended_hours / fractional / PDT / corporate actions**: lógica en `RiskGate` + actualización de posiciones tras eventos (`service` + `repository`).

**Nota código actual**: `alpaca/docstring lines 19-21` — *SL/TP como metadatos; brackets pendientes lifecycle* — riesgo operativo hasta completar.

**Riesgos Alpaca-específicos (≥3)**  
1. **Cuenta cash**: no short; rechazos silenciosos si el modelo asume long/short.  
2. **PDT** \<25 k USD: bloqueo tras 4 day trades.  
3. **Fracciones**: no todos los tipos de orden aceptan notional fraccional.  
4. **Market closed** sin `extended_hours`.  
5. **Crypto vs equity** misma API pero reglas distintas (24/7, lot size).

**Mejoras**  
1. **Brackets atómicos** vs legs separados (mejor fill risk vs complejidad).  
2. **Smart routing** Alpaca-only MVP vs multi-broker (coste operativo).

**Investigación**  
- Colas de órdenes priorizadas por **imminent close**.

---

### 1.5 Capa E — Monitorización

**Resumen**  
Logs estructurados, métricas Prometheus (latency, reject rate, ECE rolling), alertas PagerDuty — eco de `CLAUDE.md` §9.5.

**Fundamentación**  
Observabilidad es el **radar**: sin ella solo “vuelas VFR” en niebla (outages invisibles hasta pérdida).

**Arquitectura**

| Componente | Reusar | Nuevo | Responsabilidad | SLO | Test |
|------------|--------|-------|-----------------|-----|------|
| structlog | servicios platform | correlation `signal_id` | Trazas | — | — |
| Prometheus | `platform/monitoring` | dashboard `alpaca_execution.json` | RED metrics | scrape 15 s | — |
| ECE online | — | job compare `proba` vs outcomes | drift | daily | statistical tests |
| Kill switch | `strategy-orchestrator` + `execution-engine` `main.py` | persistencia Redis `CLAUDE.md` gap conocido | halt | manual \<1 s | e2e |

**Riesgos (≥3)**  
1. Métricas **sin labels** de `paper vs live` mezclan métricas.  
2. **Falso negativo** en drift si solo miras P&L (confounding).  
3. Cardinalidad alta en labels `symbol`.  
4. **Sobrealerta** → fatiga operador.

**Mejoras**  
- SLO multimetra (`broker_latency` vs `internal_latency`).  
- **Synthetic probes** contra Alpaca paper cada N minutos.

**Investigación**  
- OpenTelemetry traces de extremo a extremo (Kafka → submit).

---

### 1.6 Capa F — Automatización

**Resumen**  
DAG nocturno: datos → WF (OOS) → ablative → registro modelo → **shadow** paper (sin órdenes) → promoción solo si gates — `CLAUDE.md` §2.5, §6.6.

**Fundamentación**  
Automatizar sin guardarraíles es como **autopiloto sin redundancia**: funciona hasta que no.

**Arquitectura**

| Componente | Reusar | Nuevo | Responsabilidad | SLO | Test |
|------------|--------|-------|-----------------|-----|------|
| Cron / Airflow | pipelines futuros `CLAUDE.md` §9.4 | `pipelines/alpaca_train_dag.py` | schedule | diario | dry-run |
| Model registry | `shared/quant_shared/models/registry.py` | metadata Alpaca | versionado | — | integration |
| Promotion gates | `metrics/objective.py` `HardConstraints` | — | DSR/ECE | — | unit |

**Riesgos (≥3)**  
1. **Promoción automática** sin human para primer live — aquí: **NO** en 12 semanas (paper 100%).  
2. **Secrets** en CI — rotación (`CLAUDE.md` §16.7).  
3. **Non-determinismo** GPU si se usa.  
4. DAG falla silenciosamente si alertas mal configuradas.

**Mejoras**  
- **Canary automático** solo post-semana 12+ con capital real (fuera de alcance).  
- **Feature flags** por símbolo.

**Investigación**  
- Bandit contextual para **programar** retrain (coste vs beneficio).

---

### 1.7 Diagrama — 6 capas + SLO (ASCII, formato documento .md)

```
[ A Datos ] --Parquet/TSDB--> [ B ML / WF+Calib ] --p_win,signal--> [ C Estrategia ]
     |                                                          |
     +---------------------- macro/on-chain (opc) --------------+
                                                                  v
                    [ D Ejecución / AlpacaAdapter ] <--- [ Risk local <20ms ]
                              |  (broker RTT métrica aparte, §10.1)
                              v
                    [ E Monitorización / SRE ] <-----> [ F Automatización / CI+DAG ]
```

---

## 2. Flujo de datos end-to-end (tick → fill)

> Cada hop: trigger · schema · checkpoint · fallback · SLO p99 interno.

```
Cron/Bar close (4h)
   trigger: bar timestamp UTC (close)
   schema: OHLCV row (+ adjusted flag) → Parquet
   checkpoint: watermark `ingest:{symbol}:last_ts` en Redis o file manifest
   fallback: retry ingest + DLQ archivo corrupto
   SLO p99: N/A batch

Feature job
   trigger: nuevo Parquet slice
   schema: `FeatureVector` hash `feature_set_hash` `CLAUDE.md` §7.2
   checkpoint: commit offset en job table
   fallback: skip bar + ALERT si NaNs > threshold
   SLO p99: \< 50ms/symbol en streaming (target futuro)

Inference (servicio futuro o batch)
   trigger: `features.alpaca.*` Kafka o RPC
   schema: `signals` / `TradingSignalEvent` family `shared/quant_shared/schemas/signals.py`
   checkpoint: consumer group Kafka
   fallback: DLQ `dlq.signals`
   SLO p99: \< 100ms `CLAUDE.md` §2.2 ml-inference

Risk gate (PRE trade)
   trigger: `OrderIntent` pending
   schema: `orders.OrderIntent` (`orders.py:96`)
   checkpoint: DB row `orders.intents` (Postgres) — cuando exista
   fallback: reject + metric `risk_reject_total`
   SLO p99: \< 20ms `CLAUDE.md` §2.2 risk-engine

Execution submit
   trigger: approved intent
   schema: Alpaca REST payload (SDK)
   checkpoint: `OrderResult.broker_id` + `intent_id` UUIDv7
   fallback: retry 429/5xx max3 → circuit breaker
   SLO p99: ver §10.1 (desglosado)

Reconcile
   trigger: each 60s + after fills
   schema: internal `Position` list vs Alpaca `get_positions`
   checkpoint: `reconciler` log table
   fallback: freeze new orders + alert `CLAUDE.md` §12.7 spirit
   SLO p99: \< 250ms + RTT
```

### 2.1 Diagrama — retraining loop (ASCII)

```
02:00 UTC cron
   -> fetch last N years features (offline store)
   -> WalkForwardRunner.run()  [solo OOS metrics]
   -> if DSR/ECE gates pass -> register artifact
   -> shadow: genera señales sin submit 24h (paper log)
   -> (post week 12+) canary capital -- NO en este roadmap
```

---

## 3. STACK tecnológico

| Categoría | Tecnología | Nivel | Justificación 1-línea | ¿En repo? | ¿Conflicto CLAUDE.md? |
|-----------|------------|-------|------------------------|-----------|------------------------|
| Broker SDK | `alpaca-py` | IMPRESCINDIBLE | Oficial, ya import opcional `alpaca.py:59:73` | sí `alpaca.py` | no |
| Execution service | FastAPI + asyncio | IMPRESCINDIBLE | `execution-engine/app/main.py` | sí | no |
| Schemas | Pydantic v2 + Decimal | IMPRESCINDIBLE | `orders.py` ADR-010 | sí | no |
| WF / ML | XGBoost + sklearn | IMPRESCINDIBLE | `zoo.py`, `walk_forward_runner.py` | sí | no |
| Calibración | Isotonic (+ sigmoid) | IMPRESCINDIBLE | `calibration.py` | sí | no |
| Orchestration prod | Kafka + Redis | RECOMENDADO | `CLAUDE.md` §10 vs `los_ojos.*` topics hoy | sí parcial | ADR-007 Avro target |
| Rate limiter | `aiolimiter` o similar | RECOMENDADO | cumplir 200/200 | no | no → nueva dep: justificar PR |
| DB | asyncpg / Postgres | RECOMENDADO | `execution-engine` repository | sí | no |
| Go executor | Go | OPCIONAL (futuro) | §2.2 \<50ms | no | tensions → ADR-017 |

---

## 4. Estructura de carpetas (diff vs `CLAUDE.md` §15.2)

| Carpeta nueva | Función exacta | Consumidores | Tests | ¿Por qué no reuso? |
|---------------|------------------|--------------|-------|---------------------|
| `research/data/alpaca_*` | ingest bars/universe | WF notebooks | pytest | CCXT paths distintos |
| `research/universe/` | filtros liquidez US | data + WF | golden | no existe aún |
| `platform/services/execution-engine/app/brokers/alpaca_errors.py` | mapa errores | adapter | unit | separación limpia |
| `platform/services/execution-engine/app/rate_limit.py` | token bucket | adapter | unit | no genérico en repo |
| `shared/quant_shared/calendar/` | NYSE/NASDAQ RTH, `session_phase`, `MarketClosedError` | risk-gate, feature-engine | `shared/tests/test_market_calendar.py` | cross-service (ADR calendar en `shared/`) |
| `shared/quant_shared/symbols.py` | `is_equity` heurística | routing, calendar | unit vía calendar tests | evita duplicar en execution-engine |
| `docs/adr/016-base-executor-alias.md` | alias BaseExecutor | humanos | n/a | documentación |
| `docs/adr/017-executor-slo-python-alpaca.md` | SLO vs §2.2 | humanos | n/a | doc |

No se duplica `libs/py-models/` aún — reusar `research/` hasta migración monorepo target.

---

## 5. Roadmap 12 semanas (Alpaca-first)

> **No repite** `CLAUDE.md` §18.1 (WF, Kelly, calibración core ya DONE). Se asume existente y se **conecta**.

| Semana | Objetivo | Tareas (3–6) | Entregable verificable | DoD | Riesgo principal | Métrica progreso | Errores comunes |
|--------|----------|--------------|------------------------|-----|------------------|------------------|-----------------|
| 1 | Contrato datos | Universe top-200 MVP; ingest 4h bars; tests | Parquet + manifest | 100% símbolos descargan sin fallo | rate limit data API | # símbolos | Endpoint histórico equity ≠ adjusted |
| 2 | Features parity | `regime_gmm` + técnicos sobre equities; validar NaNs | notebook + report | WF smoke 1 fold | timezone | % bars válidos | Mezclar UTC/ET |
| 3 | **Market calendar + RTH** ✅ | `quant_shared/calendar/`; `RiskGate` check `market_closed`; `session_phase_value` | 32+ pytest calendar + 5 risk-gate; `mypy --strict`; `is_open` p99 \<1 ms | equities fuera RTH rechazadas; crypto 24/7 | UTC/ET, half-days | tests verdes | Hardcodear holidays; confundir extended vs RTH |
| 3b | WF OOS baseline (paralelo) | Config YAML Alpaca; DL DSR\|PSR OOS | CSV métricas | sin mirar IS aggregated | overfit univ | DSR OOS | Optimizar en test |
| 4 | Intent SL/TP | `OrderIntent.sl_price` wired desde `DynamicRRManager` | demo backtest→intent | campos no null cuando signal | sizing float | intents válidos | SL decimal mal redondeado |
| 5 | Alpaca paper submit | Extender `AlpacaAdapter` limit/stop; `client_order_id` | order placed paper | idempotencia doble POST | creds | # orders OK | Olvidar `extended_hours=True` |
| 6 | Brackets / OCO | Diseño legs + tests mock SDK | plan + stubs | sin live | complejidad OCO | tests pass | Fracciones no permitidas en stop bracket |
| 7 | Risk gate Alpaca | PDT + cash + buying power | unit tests reglas | reject correcto | reglas mal interpretadas | % rejects esperados | Confundir BP margin vs cash |
| 8 | Reconciler hardening | 60s loop + freeze | alert on mismatch | §12.7 triggers | estados partial fill | MTTR | No simular partial fills |
| 9 | E2E paper | Kafka → execution-engine | runbook | 24h sin crash | Redis kill | uptime | Consumer group lag |
| 10 | Observabilidad | Dashboard + ALERT-8 rules | Grafana JSON | alert firing test | cardinalidad | dashboard | Mezclar paper/live metrics |
| 11 | Automation | DAG train nocturno **solo**如果 artifacts | log + artifact registry | gates DSR/ECE | secret leak | runs OK | Subir API keys a git |
| 12 | Hardening + doc | ADR 016–017; runbook operación | doc firmado | handoff PM | scope creep | checklist OK | Asumir live cuando es paper |

---

## 6. Uso óptimo de herramientas

| Tarea | Cowork (este chat) | Claude Code CLI | Cursor | Grok | Humano (no delegar) |
|-------|--------------------|-----------------|--------|------|---------------------|
| ADR + arquitectura Alpaca | ✓ principal | ✓ largos diffs | edición | research web SDK | aprobar ADR |
| Implement `alpaca_errors.py` | plan | ✓ implementación + pytest | ✓ | — | — |
| Run pytest monorepo | orquestar | ✓ | — | — | — |
| Comparar notas comunidad Alpaca | prompt | — | — | ✓ | — |
| Promotion canary / live | — | — | — | — | ✓ obligatorio |
| Cambiar risk limits prod | — | — | — | — | ✓ |

---

## 7. Tabla de reuso explícito (anti-duplicación)

| Módulo | Path | Acción |
|--------|------|--------|
| WalkForwardRunner | `research/models/walk_forward_runner.py` | REUSAR |
| IsotonicCalibrator | `research/models/calibration.py` | REUSAR |
| Kelly / extract_p_win | `research/risk/kelly.py` | REUSAR |
| BayesianWinUpdater | `research/risk/bayesian_sizer.py` | REUSAR |
| GMMRegimeDetector | `research/features/regime_gmm.py` | REUSAR |
| Triple barrier | `research/features/labeling.py` (canónico) | REUSAR |
| AlpacaAdapter | `platform/.../brokers/alpaca.py` | EXTENDER |
| OrderIntent | `shared/quant_shared/schemas/orders.py` | EXTENDER campos |

---

## 8. Conflictos / ADRs candidatos

| ID | Tema | Resolución |
|----|------|------------|
| ADR-016 | `BaseExecutor` vs `BrokerAdapter` | Alias o rename documental |
| ADR-017 | SLO §2.2 50 ms vs Python+HTTP Alpaca | Dos métricas: internal vs broker RTT |
| ADR-018 | Topics `los_ojos.*` vs `CLAUDE.md` `raw.*` | Convivencia hasta migración |

---

## 9. VERIFICACIÓN interna (checklist del prompt)

| Item | Estado |
|------|--------|
| CLAUDE.md citado múltiples veces | ✓ §0 tabla |
| ≥5 archivos `ruta:línea` | ✓ zoo 30:52, wf 1:63, orders 40:54, base 101, alpaca 169, risk gate archivo en tests |
| Sin duplicar núcleo ML | ✓ §7 |
| SLO §2.2 discutido | ✓ §1.4 “Reconciliación explícita” |
| UTC + Decimal + UUIDv7 | ✓ §0 + `orders.py` |
| 6 capas × 6 bloques | ✓ §1.1–1.6 |
| Roadmap sin repetir §18.1 | ✓ §5 |
| Riesgos Alpaca | ✓ §1.4 |
| Diagramas ASCII | ✓ §1.7, §2, §2.1 |
| Archivo correcto | ✓ esta ruta |
| No promesas rentabilidad / IS | ✓ disclaimer |
| ADRs | ✓ §8 |

---

## 10. Enlaces

- Código adapter: `platform/services/execution-engine/app/brokers/alpaca.py`
- Interfaz broker: `platform/services/execution-engine/app/brokers/base.py`
- Dominio órdenes: `shared/quant_shared/schemas/orders.py`

**Última actualización**: 2026-05-18 (Semana 3 — market calendar en `shared/`)  
**Mantenimiento**: actualizar al cerrar cada fase del roadmap §5.
