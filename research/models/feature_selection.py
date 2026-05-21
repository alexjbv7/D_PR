"""
Feature Selection para modelos de walk-forward
===============================================
Detecta features que no aportan información real en datos OOS.

REGLA FUNDAMENTAL:
  Toda la importancia se calcula sobre datos OOS (test), NUNCA sobre train.
  Calcularla en train mide qué usó el modelo para memorizar, no qué predice.

TRES MÉTODOS IMPLEMENTADOS:

1. Gain importance (XGBoost built-in):
   Rápido, pero sesgado hacia features de alta cardinalidad. Útil para
   ranking relativo, no para threshold absoluto.

2. Permutation importance (OOS):
   Mezcla aleatoriamente cada feature en el test set y mide la caída en
   accuracy. Sin sesgos de cardinalidad. Más lento que gain, más honesto.
   Fórmula: PI(feature_j) = score_original - E[score | X_j permutado]

3. SHAP values (XGBoost built-in, sin librería externa):
   Valores de Shapley via pred_contribs. Descomposición exacta del output
   del modelo. Los más informativos: muestran no solo qué features importan
   sino cómo (dirección y magnitud por muestra).

PROTOCOLO DE PRUNING (cross-fold):
  No descartes una feature porque sea poco importante en UN fold.
  Un feature puede ser irrelevante en tendencia alcista pero crítico en crash.
  Criterio correcto:
    - Mediana de importancia across folds < threshold (importancia promedio baja)
    - Y fracción de folds con importancia > 0 < 30% (inestable/raro)
  Solo ambos a la vez justifican eliminar una feature del modelo.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance as sklearn_perm_importance

logger = logging.getLogger(__name__)


# =====================================================================
# 1. GAIN IMPORTANCE (XGBoost built-in)
# =====================================================================

def compute_gain_importance(xgb_classifier) -> pd.Series:
    """
    Gain importance de XGBoost para el modelo entrenado.

    Gain = reducción total del criterio de impureza atribuible a cada feature.
    Más informativo que 'weight' (frecuencia de uso en splits).

    Returns pd.Series índice=feature_name, valores=gain normalizado [0,1].
    """
    imp = xgb_classifier.feature_importance()    # ya implementado en zoo.py
    total = imp.sum()
    if total > 0:
        imp = imp / total
    return imp


# =====================================================================
# 2. PERMUTATION IMPORTANCE (OOS)
# =====================================================================

def compute_permutation_importance(
    xgb_classifier,
    X_oos: pd.DataFrame,
    y_oos: pd.Series,
    n_repeats: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Permutation importance sobre datos OOS.

    Por cada feature j:
      1. Mezcla aleatoriamente la columna j en X_oos.
      2. Mide la caída en accuracy respecto al score original.
      3. Repite n_repeats veces y promedia.

    Valores NEGATIVOS: la feature mejora el modelo (positivo al barajar lo empeora).
    Valores POSITIVOS (raros): la feature confunde al modelo.
    Valores ≈ 0: feature no aporta nada en OOS → candidata a eliminar.

    Returns
    -------
    pd.DataFrame con columnas: ['mean', 'std', 'ci_lower', 'ci_upper']
    Índice: nombre de cada feature.
    """
    from sklearn.base import BaseEstimator, ClassifierMixin

    # Adaptador sklearn-compatible para nuestro XGBoostClassifier
    class _Adapter(BaseEstimator, ClassifierMixin):
        def __init__(self, model):
            self._m = model

        def fit(self, X, y):
            return self

        def predict(self, X):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self._m.feature_names_)
            return self._m.predict(X)

        def score(self, X, y):
            preds = self.predict(X)
            return float(np.mean(preds == np.asarray(y)))

    adapter = _Adapter(xgb_classifier)
    X_arr = X_oos[xgb_classifier.feature_names_].values
    y_arr = np.asarray(y_oos)

    result = sklearn_perm_importance(
        adapter, X_arr, y_arr,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )

    df = pd.DataFrame(
        {
            "mean": result.importances_mean,
            "std": result.importances_std,
            "ci_lower": result.importances_mean - 2 * result.importances_std,
            "ci_upper": result.importances_mean + 2 * result.importances_std,
        },
        index=xgb_classifier.feature_names_,
    ).sort_values("mean", ascending=False)

    return df


# =====================================================================
# 3. SHAP VALUES (XGBoost built-in, sin librería shap)
# =====================================================================

def compute_shap_importance(
    xgb_classifier,
    X_oos: pd.DataFrame,
    sample_size: int = 500,
    seed: int = 42,
) -> pd.Series:
    """
    SHAP values usando el soporte nativo de XGBoost (pred_contribs=True).
    No requiere la librería `shap` externa.

    Para multiclass (softprob), XGBoost devuelve contribuciones por clase.
    Aquí promediamos el valor absoluto sobre todas las clases para obtener
    la importancia total de cada feature.

    Returns pd.Series con mean(|SHAP|) por feature, normalizado a [0,1].

    Notes
    -----
    - Muestra aleatoria de sample_size filas para reducir coste computacional.
    - El resultado es una aproximación si sample_size < len(X_oos).
    """
    import xgboost as xgb

    n = min(sample_size, len(X_oos))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_oos), size=n, replace=False)
    X_sample = X_oos.iloc[idx][xgb_classifier.feature_names_]

    booster = xgb_classifier.model.get_booster()
    dmatrix = xgb.DMatrix(X_sample.values)

    try:
        # pred_contribs: shape (n_samples, n_features + 1) o
        # (n_samples, n_classes, n_features + 1) para multiclass
        contribs = booster.predict(dmatrix, pred_contribs=True)
    except Exception as e:
        logger.warning(f"SHAP via pred_contribs falló: {e}. Usando gain importance.")
        return compute_gain_importance(xgb_classifier)

    # Eliminar el término de bias (último elemento)
    if contribs.ndim == 3:
        # Multiclass: (n_samples, n_classes, n_features + 1)
        contribs = contribs[:, :, :-1]        # quitar bias
        shap_abs = np.abs(contribs).mean(axis=(0, 1))  # promedio sobre muestras y clases
    else:
        # Binario o regresión: (n_samples, n_features + 1)
        contribs = contribs[:, :-1]
        shap_abs = np.abs(contribs).mean(axis=0)

    if len(shap_abs) != len(xgb_classifier.feature_names_):
        logger.warning(
            f"Dimensión SHAP ({len(shap_abs)}) != n_features "
            f"({len(xgb_classifier.feature_names_)}). Usando gain importance."
        )
        return compute_gain_importance(xgb_classifier)

    imp = pd.Series(shap_abs, index=xgb_classifier.feature_names_)
    total = imp.sum()
    if total > 0:
        imp = imp / total
    return imp.sort_values(ascending=False)


# =====================================================================
# 4. AGREGACIÓN CROSS-FOLD
# =====================================================================

def aggregate_fold_importances(
    importances_per_fold: list[pd.Series],
    method: str = "median",
) -> pd.DataFrame:
    """
    Agrega importancias de múltiples folds en una tabla resumen.

    Regla: no descartes una feature basándote en un solo fold.
    Agrega primero, decide después.

    Parameters
    ----------
    importances_per_fold : lista de pd.Series (una por fold),
        cada una con índice = nombre de feature, valores = importancia [0,1].
    method : 'median' | 'mean'

    Returns
    -------
    pd.DataFrame con columnas:
        - 'median_importance'  : mediana across folds
        - 'mean_importance'    : media across folds
        - 'std_importance'     : desviación estándar
        - 'frac_folds_nonzero' : fracción de folds con importancia > 0
        - 'min_importance'     : mínimo (peor fold)
        - 'max_importance'     : máximo (mejor fold)
    Ordenado por median_importance descendente.
    """
    if not importances_per_fold:
        return pd.DataFrame()

    # Unir todas las series en un DataFrame (NaN si la feature no apareció en un fold)
    df = pd.concat(importances_per_fold, axis=1)
    df.columns = [f"fold_{i}" for i in range(len(importances_per_fold))]
    df = df.fillna(0.0)

    result = pd.DataFrame(index=df.index)
    result["median_importance"] = df.median(axis=1)
    result["mean_importance"] = df.mean(axis=1)
    result["std_importance"] = df.std(axis=1)
    result["frac_folds_nonzero"] = (df > 0).mean(axis=1)
    result["min_importance"] = df.min(axis=1)
    result["max_importance"] = df.max(axis=1)

    sort_col = "median_importance" if method == "median" else "mean_importance"
    return result.sort_values(sort_col, ascending=False)


# =====================================================================
# 5. DECISIÓN DE PRUNING
# =====================================================================

def select_features_to_drop(
    agg_importance: pd.DataFrame,
    median_threshold: float = 0.005,
    frac_nonzero_threshold: float = 0.30,
    method: str = "conservative",
) -> list[str]:
    """
    Identifica features candidatas a eliminación.

    CRITERIO CONSERVADOR (recomendado para producción):
      Eliminar solo si se cumplen AMBAS condiciones:
        1. median_importance < median_threshold  (importancia sistemáticamente baja)
        2. frac_folds_nonzero < frac_nonzero_threshold (aparece en pocos folds)

    CRITERIO AGRESIVO:
      Eliminar si median_importance < median_threshold (solo condición 1).
      Útil para reducción drástica de dimensionalidad pero con más riesgo.

    Parameters
    ----------
    median_threshold : importancia mediana mínima para conservar (normalizada [0,1]).
        Con gain importance, features con < 0.005 (0.5% del total) rara vez aportan.
    frac_nonzero_threshold : mínima fracción de folds en que la feature es no-cero.
    method : 'conservative' (ambas condiciones) | 'aggressive' (solo importancia).

    Returns
    -------
    list[str] : nombres de features a eliminar.
    """
    if agg_importance.empty:
        return []

    low_importance = agg_importance["median_importance"] < median_threshold
    low_consistency = agg_importance["frac_folds_nonzero"] < frac_nonzero_threshold

    if method == "conservative":
        to_drop_mask = low_importance & low_consistency
    elif method == "aggressive":
        to_drop_mask = low_importance
    else:
        raise ValueError(f"method debe ser 'conservative' o 'aggressive', no '{method}'")

    to_drop = agg_importance.index[to_drop_mask].tolist()

    if to_drop:
        logger.info(
            f"Features candidatas a eliminar ({len(to_drop)}): {to_drop}"
        )
    else:
        logger.info("No se identificaron features a eliminar con los thresholds actuales.")

    return to_drop


# =====================================================================
# 6. REPORTE COMPLETO
# =====================================================================

def feature_importance_report(
    agg_importance: pd.DataFrame,
    top_n: int = 20,
    median_threshold: float = 0.005,
) -> None:
    """
    Imprime un reporte formateado de importancia de features cross-fold.
    """
    print("=" * 75)
    print(" FEATURE IMPORTANCE REPORT (cross-fold)")
    print("=" * 75)
    print(f" {'Feature':35s} {'Median':>8s} {'Mean':>8s} {'Std':>7s} {'NonZero%':>9s}")
    print("-" * 75)

    shown = 0
    for feat, row in agg_importance.iterrows():
        if shown >= top_n:
            break
        flag = " <-- BAJO" if row["median_importance"] < median_threshold else ""
        print(
            f" {str(feat):35s} "
            f"{row['median_importance']:8.4f} "
            f"{row['mean_importance']:8.4f} "
            f"{row['std_importance']:7.4f} "
            f"{row['frac_folds_nonzero']:9.0%}"
            f"{flag}"
        )
        shown += 1

    n_total = len(agg_importance)
    n_low = (agg_importance["median_importance"] < median_threshold).sum()
    print("-" * 75)
    print(
        f" Total features: {n_total} | "
        f"Bajo umbral ({median_threshold:.3f}): {n_low} | "
        f"Conservar: {n_total - n_low}"
    )
    print("=" * 75)
