# quant_bot — Monorepo de Trading Algorítmico ML

Sistema institucional de trading basado en Machine Learning, diseñado siguiendo las
prácticas de López de Prado (*Advances in Financial Machine Learning*) con
estándares de validación estadística de hedge funds quant.

> **Documento maestro de arquitectura**: [`CLAUDE.md`](CLAUDE.md).
> Este README es una guía de arranque rápido; para decisiones de diseño,
> ADRs y roadmap consulta `CLAUDE.md`.

---

## Estructura del monorepo

```
quant_bot/
│
├── research/              Núcleo ML: modelos, backtesting, validación, features
│   ├── models/            WalkForwardRunner, zoo, calibration, meta_labeler,
│   │                      entry_filter, feature_selection, hyperopt, rl_agent
│   ├── features/          engineering, regime_gmm, pca_denoiser, volatility_target,
│   │                      labeling (triple-barrier, ATR)
│   ├── backtesting/       engine (bar-level, fees/slippage), multi_asset_engine
│   ├── risk/              kelly, dynamic_rr, bayesian_sizer, sizing_multi_asset
│   ├── metrics/           objective (DSR/PSR/constraints), advanced
│   ├── reporting/         report, FoldReport, compare_is_oos
│   ├── instruments/       catalog, specs (ForexSpec, FutureSpec)
│   ├── dashboard/         Streamlit app (app.py, pages/, components/)
│   ├── examples/          pipeline_ml_real_data.py, validation_demo*.py,
│   │                      pipeline_hyperopt.py
│   └── tests/             test_walk_forward_runner, test_calibration,
│                          test_meta_labeler, test_deep_learning, test_no_leakage…
│
├── platform/              Plataforma event-driven en producción (Los Ojos)
│   ├── services/
│   │   ├── market-intelligence/   OpenBB + Binance orderbook + funding rate
│   │   ├── macroeconomic/         FRED + Sahm Rule + yield curve + macro regime
│   │   ├── onchain-analysis/      Whale detection + smart money flow
│   │   ├── context-engine/        GMM regime classifier (5 componentes)
│   │   ├── realtime-signal/       FastAPI WebSocket (Kafka → Redis → WS)
│   │   ├── ml-feature-store/      Feature computation + serving
│   │   ├── strategy-orchestrator/ Bot config, kill switch, señal generation
│   │   └── sec-research/          SEC filings, NLP, sentiment
│   ├── libs/shared/       Kafka client, Redis client, Postgres pool,
│   │                      eventos Pydantic (fuente de verdad de contratos Kafka)
│   ├── frontend/          React + TypeScript + Vite + Tailwind
│   └── Makefile           make up / infra / services / monitoring / frontend
│
├── shared/                Paquete quant-shared (schemas Pydantic, features canónicos,
│   └── quant_shared/      model registry) — migración progresiva desde platform/libs/shared
│
└── data/                  Ingestores independientes del venv de research
    ├── ingestion.py       CCXT (Binance, Bybit, Kraken) → Parquet
    └── real_data.py       Yahoo Finance → Parquet (FX + futuros diario)
```

---

## Arranque rápido

### Research (backtesting + ML)

```bash
# Instalar entorno
cd research
pip install -e ../shared
pip install -e .
# Instalar extras de ingesta y dashboard (opcionales)
pip install ccxt yfinance streamlit plotly

# Tests de regresión (deben pasar antes de cualquier experimento)
pytest tests/ -v

# Demo end-to-end con datos reales (EURUSD diario, sin deps externas en runtime)
python examples/pipeline_ml_real_data.py

# Dashboard Streamlit
streamlit run dashboard/app.py
```

### Platform (microservicios + frontend)

```bash
cd platform
make up           # levanta Kafka, Redis, Postgres, todos los servicios y frontend
make infra        # solo infraestructura (Kafka, Redis, Postgres, MongoDB)
make services     # solo microservicios (requiere infra corriendo)
make monitoring   # Prometheus + Grafana
```

URLs locales: dashboard `http://localhost:3000` · Kafka UI `http://localhost:8080` ·
Grafana `http://localhost:3001`

---

## Flujo de investigación (research)

```
Yahoo Finance / CCXT
        │
        ▼
  OHLCV validado (parquet cache)
        │
        ▼
  features/engineering.py  →  features técnicos (RSI, MACD, ATR, z-scores, vol)
  features/regime_gmm.py   →  régimen de mercado (GMM 3–5 componentes)
  features/pca_denoiser.py →  denoising anti-leakage
  features/labeling.py     →  Triple-Barrier Labels {-1, 0, +1}
        │
        ▼
  models/walk_forward_runner.py
  ┌─────────────────────────────────────────────────────────┐
  │ Por fold:                                               │
  │  TRAIN_fit → XGBoost.fit()                             │
  │  TRAIN_calib → IsotonicCalibrator.fit()  (ECE < 0.05) │
  │  TRAIN_calib → EntryFilter.fit()         (threshold)   │
  │  TEST → predict_proba → signal → Kelly → sizing        │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  Métricas OOS: PSR, DSR, Sharpe, MDD, ECE
  + Feature importance cross-fold (Gain + SHAP)
  + Calibration report por fold
```

---

## Componentes estadísticos clave

### Triple-Barrier Labeling (López de Prado)

El label de cada barra no es `sign(retorno)` sino el resultado de la primera
barrera alcanzada. Módulo canónico: [`research/features/labeling.py`](research/features/labeling.py).

- **+1** si `price[t+k] > entry + mult × ATR[t]`
- **−1** si `price[t+k] < entry - mult × ATR[t]`
- **0** si ninguna barrera se toca en `horizon` barras (timeout)

### Calibración de Probabilidades (OvR)

`XGBoost.predict_proba()` devuelve softmax (scores relativos, no probabilidades
bayesianas). Pipeline de calibración con `IsotonicRegression` one-vs-rest en
`TRAIN_calib`, verificado con ECE < 0.05 y Brier < baseline.

### PSR / DSR

```
PSR(SR*) = Φ( (SR - SR*) × √(T-1) / √(1 - γ₃·SR + (γ₄-1)/4·SR²) )
DSR = PSR( E[max SR | n_trials] )
```

Un modelo solo pasa si **PSR(0) > 95% y DSR(0) > 95%** sobre la serie OOS
concatenada. El parámetro `n_trials` debe reflejar el número real de configuraciones
evaluadas (ver [`research/metrics/objective.py`](research/metrics/objective.py)).

### Kelly Fraccional

```
f* = (p·b - q) / b    →   usar f = f* × 0.25 (quarter Kelly)
```

Regla no negociable: `kelly_fraction ≤ 0.25` en producción.

---

## Métricas objetivo (OOS)

| Métrica | Mínimo aceptable | Objetivo |
|---------|-----------------|----------|
| Sharpe anual (OOS) | > 0.8 | > 1.5 |
| PSR(SR>0) | > 90% | > 95% |
| DSR | > 0.4 | > 0.6 |
| ECE (calibración) | < 0.10 | < 0.05 |
| Max Drawdown | < 25% | < 15% |
| Calmar Ratio | > 0.5 | > 1.0 |

---

## Reglas no negociables

1. **Calibra antes de filtrar** — ECE verificado antes de usar `predict_proba()` en decisiones.
2. **Valida OOS siempre** — PSR > 95% y DSR > 95% son el mínimo para considerar una estrategia.
3. **Purge + embargo obligatorio** — sin esto el backtest miente por data leakage temporal.
4. **Kelly fraccional ≤ 0.25** — Kelly completo es teóricamente óptimo, prácticamente ruinoso.
5. **Paper trading mínimo 1 mes** antes de capital real.
6. **`n_trials` honesto en DSR** — si optimizaste 100 combinaciones, usa `n_trials=100`.
7. **Features con justificación económica** — no agregar indicadores porque "funcionaron in-sample".

---

## Referencias

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio". *JPM*.
- Harvey, C. R. & Liu, Y. (2015). "Backtesting". *JPM*.

---

## Disclaimer

Proyecto de investigación y aprendizaje. Operar con dinero real conlleva riesgo de
pérdida total del capital. Las performances pasadas no garantizan resultados futuros.
