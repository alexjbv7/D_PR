"""
Meta-Labeling (López de Prado, AFML Cap. 3)
============================================
Un segundo clasificador *binario* que predice si la señal del modelo
primario es correcta o no en cada barra activa (long/short).

Flujo por fold:
  1. Primario     : XGBoost {-1, 0, +1}  fit sobre X_fit
  2. Señales calib: primario.predict(X_calib) → signals_calib  (sin ver test)
  3. Meta-labels  : meta_y[t] = 1  si  signals_calib[t] == y_calib[t]
                                      AND  signals_calib[t] ≠ 0
  4. Meta-modelo  : fit sobre X_calib[señal≠0] con meta_y como target
     Features     : X_original + probabilidades del primario + dirección
  5. En test      : P(correcto|x) = meta_modelo.predict_proba(x)
                    → reemplaza p_win en Kelly / sizing

Beneficio:
  - Win rate ↑  porque sólo operamos cuando AMBOS modelos coinciden.
  - Coverage ↓  pero la curva riesgo/retorno mejora.

Anti-leakage:
  - MetaLabeler.fit() SÓLO sobre X_calib (pasado respecto al test).
  - Las probabilidades del primario usadas como features son predicciones
    sobre X_calib, no los labels reales.

Referencia: López de Prado (2018), AFML, capítulo 3, sección 3.5.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURACIÓN
# =====================================================================

@dataclass
class MetaLabelConfig:
    """
    Configuración del MetaLabeler.

    xgb_params          : params de XGBoost para el meta-modelo.
                          Más simples que el primario para evitar overfitting
                          con pocos ejemplos.
    min_samples         : mínimo de señales activas en calib para entrenar.
                          Si hay menos, el meta-labeler no se ajusta y p_win
                          cae al fallback (probabilidad del primario).
    use_original_features : incluir las features originales de X como input.
    use_primary_proba   : incluir las probabilidades del primario como input.
                          Recomendado: True — captura la confianza del primario.
    """
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "reg_lambda": 2.0,
        "reg_alpha": 0.1,
    })
    min_samples: int = 20
    use_original_features: bool = True
    use_primary_proba: bool = True


# =====================================================================
# HELPERS
# =====================================================================

def create_meta_labels(
    primary_signals: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera meta-labels binarios a partir de señales del primario y labels reales.

    Sólo se consideran los instantes con señal activa (≠ 0).

    Parameters
    ----------
    primary_signals : array {-1, 0, +1}
    y_true          : array {-1, 0, +1}

    Returns
    -------
    active_mask : bool array, True donde señal ≠ 0
    meta_y      : int array con {0, 1} en las posiciones activas
                  1 = primario correcto, 0 = primario incorrecto
    """
    primary_signals = np.asarray(primary_signals)
    y_true = np.asarray(y_true)

    active_mask = primary_signals != 0
    meta_y = (primary_signals[active_mask] == y_true[active_mask]).astype(int)
    return active_mask, meta_y


# =====================================================================
# META-LABELER
# =====================================================================

class MetaLabeler:
    """
    Clasificador binario secundario que filtra las señales del modelo primario.

    Uso en WalkForwardRunner (step 4b, tras entry filter):
        meta = MetaLabeler(MetaLabelConfig())
        meta.fit(
            X_calib, proba_calib, signals_calib, y_calib.values, class_labels
        )
        p_win_arr = meta.predict_p_correct(X_test, proba_test, signals_test)

    Si meta.is_fitted es False (muy pocos ejemplos), predict_p_correct
    devuelve None → el runner usa el p_win original del primario.
    """

    def __init__(self, config: Optional[MetaLabelConfig] = None):
        self.cfg = config or MetaLabelConfig()
        self._model = None
        self._feature_names: List[str] = []
        self.is_fitted: bool = False
        self.n_active_train: int = 0
        self.class_labels_: list = []

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        primary_proba: np.ndarray,
        primary_signals: np.ndarray,
        y_true: np.ndarray,
        class_labels: Optional[list] = None,
    ) -> "MetaLabeler":
        """
        Entrena el meta-modelo sobre los instantes con señal primaria activa.

        Parameters
        ----------
        X               : features originales (DataFrame con índice)
        primary_proba   : probabilidades del primario (n_samples x n_classes)
        primary_signals : señales del primario {-1, 0, +1} (n_samples)
        y_true          : labels reales {-1, 0, +1} (n_samples)
        class_labels    : lista de clases del primario (ej. [-1, 0, 1])
        """
        cfg = self.cfg
        self.class_labels_ = class_labels or sorted(set(y_true.tolist()))

        primary_signals = np.asarray(primary_signals)
        y_true = np.asarray(y_true)
        primary_proba = np.asarray(primary_proba)

        active_mask, meta_y = create_meta_labels(primary_signals, y_true)
        n_active = int(active_mask.sum())
        self.n_active_train = n_active

        if n_active < cfg.min_samples:
            logger.debug(
                "MetaLabeler.fit: solo %d muestras activas (mínimo=%d). "
                "Meta-labeler NO entrenado.",
                n_active, cfg.min_samples,
            )
            self.is_fitted = False
            return self

        # Construir features del meta-modelo
        X_meta = self._build_meta_features(
            X.values[active_mask],
            primary_proba[active_mask],
            primary_signals[active_mask],
            X.columns.tolist(),
        )

        # Verificar que hay al menos dos clases en meta_y
        unique_meta = np.unique(meta_y)
        if len(unique_meta) < 2:
            logger.debug(
                "MetaLabeler.fit: meta_y tiene una sola clase (%s). "
                "Meta-labeler NO entrenado.",
                unique_meta,
            )
            self.is_fitted = False
            return self

        # Fit XGBoost binario
        from xgboost import XGBClassifier
        self._model = XGBClassifier(
            **cfg.xgb_params,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )
        self._model.fit(X_meta, meta_y)
        self.is_fitted = True

        win_rate_calib = float(meta_y.mean())
        logger.debug(
            "MetaLabeler fit: %d muestras activas | win_rate_calib=%.1f%%",
            n_active, win_rate_calib * 100,
        )
        return self

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------

    def predict_p_correct(
        self,
        X: pd.DataFrame,
        primary_proba: np.ndarray,
        primary_signals: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Devuelve P(señal primaria correcta) para cada barra.

        Sólo se evalúa sobre señales activas; las neutras reciben 0.0.

        Returns
        -------
        np.ndarray de shape (n_samples,) con valores en [0, 1],
        o None si el meta-labeler no está entrenado (el runner usa fallback).
        """
        if not self.is_fitted:
            return None

        primary_signals = np.asarray(primary_signals)
        primary_proba = np.asarray(primary_proba)
        n = len(primary_signals)

        result = np.zeros(n, dtype=float)
        active_mask = primary_signals != 0
        n_active = int(active_mask.sum())

        if n_active == 0:
            return result

        X_meta = self._build_meta_features(
            X.values[active_mask],
            primary_proba[active_mask],
            primary_signals[active_mask],
            X.columns.tolist(),
        )

        proba_active = self._model.predict_proba(X_meta)
        # Clase 1 = "primario correcto"
        idx_correct = list(self._model.classes_).index(1)
        result[active_mask] = proba_active[:, idx_correct]
        return result

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _build_meta_features(
        self,
        X_arr: np.ndarray,
        proba_arr: np.ndarray,
        signals_arr: np.ndarray,
        feature_names: List[str],
    ) -> np.ndarray:
        """
        Concatena features según la configuración:
          - X original (opcional)
          - probabilidades del primario (opcional): p_{clase_0}..p_{clase_k}
          - dirección de la señal primaria (siempre)
        """
        cfg = self.cfg
        parts = []

        if cfg.use_original_features:
            parts.append(X_arr)

        if cfg.use_primary_proba:
            parts.append(proba_arr)

        # Dirección de la señal: +1 o -1 (siempre incluida)
        parts.append(signals_arr.reshape(-1, 1).astype(float))

        return np.concatenate(parts, axis=1)

    # ------------------------------------------------------------------
    # DIAGNÓSTICOS
    # ------------------------------------------------------------------

    def feature_importance(self) -> Optional[pd.Series]:
        """Importancia de features del meta-modelo (gain)."""
        if not self.is_fitted:
            return None
        booster = self._model.get_booster()
        scores = booster.get_score(importance_type="gain")
        return pd.Series(scores).sort_values(ascending=False)

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"MetaLabeler(not fitted, min_samples={self.cfg.min_samples})"
        return (
            f"MetaLabeler(fitted, "
            f"n_active_train={self.n_active_train}, "
            f"use_proba={self.cfg.use_primary_proba})"
        )
