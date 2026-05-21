# research/archive — Scripts de sesión (histórico)

Estos scripts son la bitácora de desarrollo del proyecto: sesiones de exploración,
diagnósticos puntuales y verificaciones one-off. Se conservan como referencia
histórica pero **no pertenecen al código de producción**.

## Por qué están aquí

- Usan `sys.path.insert(0, '.')` y rutas relativas a `./cache/*.parquet`
  que solo funcionan desde un directorio de trabajo específico.
- Dependen de datasets locales no versionados (en `cache/`).
- Su lógica reutilizable fue absorbida por los módulos canónicos:
  - Feature engineering → `research/features/engineering.py`
  - Triple-barrier labeling → `research/features/labeling.py`
  - Risk management → `research/risk/kelly.py`, `dynamic_rr.py`
  - Walk-forward → `research/models/walk_forward_runner.py`
  - Pipeline orquestado → `research/examples/pipeline_ml_real_data.py`

## Regla

Nada de este directorio se importa desde tests ni desde producción.
Si necesitas reproducir una sesión, parte del ejemplo canónico en
`research/examples/` en lugar de adaptar estos scripts.

## Contenido

| Archivo | Descripción |
|---------|-------------|
| `session1.py` | Ingesta BTC/USDT diario + detección de regímenes anómalos |
| `session2.py` | Feature engineering completo + visualización |
| `session2_3.py` | Análisis de correlación entre features |
| `session2_4.py` | Filtrado manual con criterio económico |
| `session2_5.py` | Correlaciones macro (SP500, NASDAQ, DXY, VIX, Oro) |
| `session2_6.py` | Target triple-barrier + distribución de labels |
| `session2_6b.py` | Diagnóstico visual mejorado del triple-barrier |
| `session3_1.py` | Baselines triviales + walk-forward splits |
| `session3_2.py` | Logistic Regression baseline walk-forward |
| `session3_3.py` | XGBoost primeros resultados |
| `session4.py` | Pipeline integrado con risk management |
| `session4_diag.py` | Diagnóstico de clasificación por clase |
| `session5.py` | Calibración de probabilidades |
| `session6.py` | Meta-labeling + Bayesian sizing |
| `session7_1.py` | Integración completa walk-forward runner |
| `session7_2.py` | Hyperparameter optimization |
| `session7_2_diag.py` | Diagnóstico de hyperopt |
| `diag_nasdaq.py` | Diagnóstico específico NASDAQ |
| `pipeline.py` | Orquestador antiguo (pre-monorepo, usa risk.management) |
| `test_fix.py` | Verificación de fix en XGBoostClassifier |
| `verify_run.py` | Verificación de entrenamiento walk-forward |
| `verify_run2.py` | Segunda verificación de entrenamiento |
