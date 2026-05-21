# Bugbot — capa research (ML, features, risk, backtest)

## Invariantes ML (bloquear si se violan)

1. **Walk-forward**: cualquier cambio en evaluación de estrategias debe usar
   `WalkForwardRunner` o `WalkForwardSplitter`; prohibido optimizar hiperparámetros
   mirando el fold de test.
2. **Calibración**: `IsotonicCalibrator` / `model.calibrate()` solo sobre `X_calib`
   posterior al `fit` y anterior al test; nunca calibrar con labels del test.
3. **Meta-labeler y Bayesian**: `BayesianWinUpdater.fit` solo en calib;
   `update()` solo en test/inferencia — ver `risk/bayesian_sizer.py`.
4. **GMM régimen**: `GMMRegimeDetector.fit` solo en train del fold;
   `transform` en calib/test por separado — `features/regime_gmm.py`.
5. **PCA**: columnas `regime_*` excluidas del PCA denoiser (no reducir régimen).
6. **Labels**: triple-barrier y ATR canónicos en `features/labeling.py`; no copiar
   funciones inline en `examples/` o `engineering.py`.

## Kelly y riesgo (research)

- Kelly fraccional por defecto `kelly_fraction=0.25`, nunca full Kelly en código
  marcado como producción.
- `max_risk_pct` cap (p. ej. 2%) debe permanecer en `KellyAtrSizer`.
- `DynamicRRManager`: SL en ATR fijo; TP escala con `p_win` — no invertir sin razón.

## Métricas en PRs

- Reportar **PSR/DSR OOS** concatenado, ECE, max drawdown OOS si el PR afecta modelos.
- Rechazar narrativas del tipo "Sharpe subió en train" sin degradación IS→OOS
  (`is_oos_degradation` en `metrics/advanced.py`).

## Tests

- Cambios en `models/`, `risk/`, `features/` requieren test en `research/tests/` o
  junto al módulo.
- No commitear scripts de sesión en `research/scripts/` (usar `archive/`).

## Dependencias

- Nuevas libs ML solo con entrada en `research/pyproject.toml` y justificación en PR.
- No añadir `quant-shared` como dependencia core si research no lo importa.
