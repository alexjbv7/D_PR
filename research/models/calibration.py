"""
Model Calibration
=================
Calibra las probabilidades de XGBoost para que sean P(y=c|x) reales.

PROBLEMA SIN CALIBRAR:
  XGBClassifier.predict_proba() devuelve softmax(scores_de_árboles).
  Eso NO es una probabilidad bayesiana. Un valor de 0.72 no significa
  "72% de probabilidad de que sea +1" — significa "score relativo más alto
  que el resto de clases". Esto invalida cualquier umbral de entrada,
  Kelly, o filtro probabilístico.

SOLUCIÓN — Calibración isotónica:
  Aprende una función monótona no-paramétrica f: score → probabilidad real,
  entrenada sobre un conjunto de calibración separado (posterior al train,
  anterior al test). Después de calibrar, el modelo pasa tests de reliability.

PROTOCOLO CORRECTO en walk-forward:
  [TRAIN_fit 80%] | [TRAIN_calib 20%] | [embargo] | [TEST]

  1. Fit XGBoost en TRAIN_fit.
  2. Calibrar sobre TRAIN_calib (no embargo: calibración sigue siendo "pasado").
  3. Evaluar modelo calibrado en TEST.

  Ejemplo con 252 barras train: ~201 fit, ~51 calibración.
  Isotonic necesita ≥ 50 muestras; Platt (sigmoid) funciona con menos.

MÉTRICAS:
  - ECE  (Expected Calibration Error): 0 = perfecto, <0.05 = aceptable.
  - Brier Score: MSE de probabilidades, 0 = perfecto, 0.25 = random.
  - Reliability diagram: curva confianza predicha vs fracción real de positivos.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import logging

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


# =====================================================================
# MÉTRICAS DE CALIBRACIÓN
# =====================================================================

def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE para clasificación binaria (one-vs-rest).

    Divide las predicciones en n_bins intervalos de confianza iguales.
    En cada bin mide |confianza_media - fracción_real_de_positivos|
    ponderado por el tamaño del bin.

    ECE < 0.05 = bien calibrado.
    ECE > 0.10 = calibración pobre, no usar probabilidades ciegamente.

    Parameters
    ----------
    y_true : array de {0,1} (one-vs-rest para la clase de interés)
    y_proba : array de probabilidades predichas en [0,1]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi)
        if i == n_bins - 1:          # último bin: incluye 1.0
            mask = (y_proba >= lo) & (y_proba <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        mean_conf = float(y_proba[mask].mean())
        frac_pos = float(y_true[mask].mean())
        ece += (n_bin / n) * abs(mean_conf - frac_pos)

    return float(ece)


def brier_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    """
    Brier Score = mean squared error de probabilidades.
    Rango [0, 1]. 0 = perfecto, 0.25 = modelo sin información (p=0.5 siempre).

    Para multiclass: promedia el BS one-vs-rest sobre todas las clases.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.ndim == 1:
        return float(np.mean((y_proba - y_true) ** 2))
    # Multiclass: one-hot encode y_true y promediar
    n_classes = y_proba.shape[1]
    bs_per_class = []
    for c in range(n_classes):
        y_c = (y_true == c).astype(float)
        bs_per_class.append(np.mean((y_proba[:, c] - y_c) ** 2))
    return float(np.mean(bs_per_class))


def reliability_diagram_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Devuelve los datos para dibujar un reliability diagram.

    Returns
    -------
    dict con:
        - 'bin_centers': centros de cada bin de confianza
        - 'fraction_positive': fracción real de positivos en cada bin
        - 'mean_confidence': confianza media en cada bin
        - 'counts': número de muestras en cada bin
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, fracs, confs, counts = [], [], [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_proba >= lo) & (y_proba <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        centers.append(float((lo + hi) / 2))
        fracs.append(float(y_true[mask].mean()))
        confs.append(float(y_proba[mask].mean()))
        counts.append(n_bin)

    return {
        "bin_centers": centers,
        "fraction_positive": fracs,
        "mean_confidence": confs,
        "counts": counts,
    }


def calibration_report(
    y_true: np.ndarray,
    y_proba_uncal: np.ndarray,
    y_proba_cal: np.ndarray,
    label: str = "",
    n_bins: int = 10,
) -> dict:
    """
    Compara métricas antes y después de calibrar.

    Parameters
    ----------
    y_true       : etiquetas verdaderas (entero 0..K-1 para la clase de interés)
    y_proba_uncal: probabilidades sin calibrar (array 1D, one-vs-rest)
    y_proba_cal  : probabilidades calibradas (array 1D, one-vs-rest)
    """
    ece_before = expected_calibration_error(y_true, y_proba_uncal, n_bins)
    ece_after = expected_calibration_error(y_true, y_proba_cal, n_bins)
    bs_before = brier_score(y_true, y_proba_uncal)
    bs_after = brier_score(y_true, y_proba_cal)

    report = {
        "label": label,
        "ece_uncalibrated": round(ece_before, 4),
        "ece_calibrated": round(ece_after, 4),
        "ece_improvement": round(ece_before - ece_after, 4),
        "brier_uncalibrated": round(bs_before, 4),
        "brier_calibrated": round(bs_after, 4),
        "brier_improvement": round(bs_before - bs_after, 4),
        "verdict": "OK" if ece_after < 0.05 else ("MARGINAL" if ece_after < 0.10 else "POOR"),
    }

    logger.info(
        f"Calibración {label}: "
        f"ECE {ece_before:.3f} → {ece_after:.3f}  "
        f"Brier {bs_before:.4f} → {bs_after:.4f}  "
        f"[{report['verdict']}]"
    )
    return report


# =====================================================================
# CALIBRADOR — wrapper alrededor de sklearn
# =====================================================================

class IsotonicCalibrator:
    """
    Calibrador one-vs-rest para un clasificador ya entrenado.

    Implementación manual que no depende de CalibratedClassifierCV (cuya API
    interna cambia entre versiones de sklearn). Compatible con sklearn >= 1.3.

    Algoritmo:
      Para cada clase c ∈ {0..K-1}:
        1. Obtiene proba_raw[:, c] del modelo base en el set de calibración.
        2. Ajusta un calibrador sobre (proba_raw[:, c], y_binary_c).
        3. En predict_proba, aplica cada calibrador y renormaliza a suma=1.

    Métodos disponibles:
      - 'isotonic': IsotonicRegression — no-paramétrico, flexible.
                    Necesita ≥ 50 muestras por clase. Recomendado ≥ 200 total.
      - 'sigmoid' : Platt scaling (LogisticRegression 1D).
                    Funciona con pocas muestras (≥ 20). Asume relación log-lineal.

    Parameters
    ----------
    method : 'isotonic' | 'sigmoid'
    min_samples_isotonic : int
        Si el set de calibración tiene menos muestras que este valor,
        cambia a 'sigmoid' automáticamente.
    """

    def __init__(
        self,
        method: str = "isotonic",
        min_samples_isotonic: int = 80,
    ):
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"method debe ser 'isotonic' o 'sigmoid', no '{method}'")
        self.method = method
        self.min_samples_isotonic = min_samples_isotonic
        self._calibrators: list = []   # uno por clase (OvR)
        self._n_classes: int = 0
        self._fitted = False

    def fit(
        self,
        base_model,            # modelo con .predict_proba(X) ya entrenado
        X_calib: np.ndarray,
        y_calib: np.ndarray,   # labels mapeados a 0..K-1
    ) -> "IsotonicCalibrator":
        """
        Ajusta el calibrador OvR sobre el set de calibración.

        Parameters
        ----------
        base_model : modelo con predict_proba (xgb.XGBClassifier, etc.)
        X_calib    : features del set de calibración (numpy array, ya escalado).
        y_calib    : labels mapeados a 0..K-1.
        """
        method = self.method
        n_samples = len(X_calib)

        if method == "isotonic" and n_samples < self.min_samples_isotonic:
            logger.warning(
                f"Set de calibración pequeño ({n_samples} < {self.min_samples_isotonic}). "
                f"Usando 'sigmoid' en vez de 'isotonic'."
            )
            method = "sigmoid"

        # Probabilidades crudas del modelo base en el set de calibración
        proba_raw = base_model.predict_proba(X_calib)  # (n, K)
        n_classes = proba_raw.shape[1]
        self._n_classes = n_classes

        self._calibrators = []
        for c in range(n_classes):
            y_binary = (y_calib == c).astype(float)
            p_c = proba_raw[:, c]

            if method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(p_c, y_binary)
            else:
                # Platt: regresión logística 1D sobre la probabilidad cruda
                cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
                cal.fit(p_c.reshape(-1, 1), y_binary)

            self._calibrators.append((method, cal))

        self._fitted = True
        logger.info(
            f"Calibrador {method} OvR ajustado: {n_classes} clases, "
            f"{n_samples} muestras de calibración."
        )
        return self

    def fit_from_proba(
        self,
        proba_raw: np.ndarray,
        y_calib: np.ndarray,
    ) -> "IsotonicCalibrator":
        """
        Ajusta el calibrador OvR a partir de probabilidades ya computadas.

        Idéntico a fit() pero acepta proba_raw directamente en vez de
        requerir el modelo base + X. Útil cuando ya se tiene predict_proba_raw().

        Parameters
        ----------
        proba_raw : np.ndarray shape (n_samples, n_classes)
            Salida de base_model.predict_proba_raw() en el set de calibración.
        y_calib   : np.ndarray de labels mapeados a 0..K-1.
        """
        method = self.method
        n_samples = len(proba_raw)

        if method == "isotonic" and n_samples < self.min_samples_isotonic:
            logger.warning(
                f"Set de calibración pequeño ({n_samples} < {self.min_samples_isotonic}). "
                f"Usando 'sigmoid' en vez de 'isotonic'."
            )
            method = "sigmoid"

        n_classes = proba_raw.shape[1]
        self._n_classes = n_classes
        self._calibrators = []

        for c in range(n_classes):
            y_binary = (y_calib == c).astype(float)
            p_c = proba_raw[:, c]

            if method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(p_c, y_binary)
            else:
                cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
                cal.fit(p_c.reshape(-1, 1), y_binary)

            self._calibrators.append((method, cal))

        self._fitted = True
        logger.info(
            f"Calibrador {method} OvR (from_proba) ajustado: {n_classes} clases, "
            f"{n_samples} muestras de calibración."
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Probabilidades calibradas. Shape: (n_samples, n_classes).
        Las filas suman a 1 (renormalización después de calibración OvR).
        """
        if not self._fitted:
            raise RuntimeError("Llama a .fit() antes de .predict_proba()")

        # Necesitamos las probabilidades crudas: el caller debe pasar el
        # modelo base o las probabilidades directamente.
        # Diseño: predict_proba recibe probabilidades crudas (no el modelo).
        # Ver _predict_calibrated() para el flujo completo.
        raise NotImplementedError(
            "Usa predict_proba_from_raw(proba_raw) directamente, "
            "o llama a XGBoostClassifier.predict_proba() que gestiona el flujo."
        )

    def predict_proba_from_raw(self, proba_raw: np.ndarray) -> np.ndarray:
        """
        Calibra probabilidades ya obtenidas del modelo base.

        Parameters
        ----------
        proba_raw : np.ndarray shape (n_samples, n_classes)
            Salida de base_model.predict_proba() en nuevos datos.

        Returns
        -------
        np.ndarray shape (n_samples, n_classes), renormalizado a suma=1.
        """
        if not self._fitted:
            raise RuntimeError("Llama a .fit() antes de .predict_proba_from_raw()")

        n = proba_raw.shape[0]
        calibrated = np.zeros((n, self._n_classes))

        for c, (method, cal) in enumerate(self._calibrators):
            p_c = proba_raw[:, c]
            if method == "isotonic":
                calibrated[:, c] = cal.predict(p_c)
            else:  # sigmoid
                calibrated[:, c] = cal.predict_proba(p_c.reshape(-1, 1))[:, 1]

        # Clip a [0,1] por errores numéricos y renormalizar
        calibrated = np.clip(calibrated, 0.0, 1.0)
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)   # evitar div/0
        return calibrated / row_sums

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# =====================================================================
# UTILIDAD: split train → fit + calibración
# =====================================================================

def split_train_for_calibration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    calib_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Divide el conjunto de train en dos partes temporalmente ordenadas:
      [TRAIN_fit | TRAIN_calib]

    La separación respeta el orden temporal (sin shuffle).
    TRAIN_calib es siempre el bloque más reciente del train.

    Parameters
    ----------
    calib_frac : fracción del train destinada a calibración (default 20%).

    Returns
    -------
    X_fit, y_fit, X_calib, y_calib
    """
    n = len(X_train)
    n_calib = max(1, int(n * calib_frac))
    n_fit = n - n_calib

    X_fit = X_train.iloc[:n_fit]
    y_fit = y_train.iloc[:n_fit]
    X_calib = X_train.iloc[n_fit:]
    y_calib = y_train.iloc[n_fit:]

    logger.info(
        f"Split train: {n_fit} barras fit + {n_calib} barras calibración "
        f"(frac={calib_frac:.0%})"
    )
    return X_fit, y_fit, X_calib, y_calib


# =====================================================================
# CALIBRADOR ESCALAR 1D — para señales con un único score de confianza
# =====================================================================

class ScalarProbabilityCalibrator:
    """
    Calibrador 1D ``p_raw -> P(win)`` para señales con un ÚNICO score de
    confianza por muestra (E3 paso 2, arbitraje D).

    A diferencia de ``IsotonicCalibrator`` (OvR multiclase sobre
    ``predict_proba``), aquí el input es un escalar: la confianza de la acción
    elegida (p.ej. ``softmax(Q)[a]`` del DQN — un proxy ordinal, NO una
    probabilidad). Aprende una función monótona ``p_raw -> frecuencia empírica de
    acierto`` sobre un conjunto de calibración OOS por fold.

    Anti-leakage (NO negociable): ajustar SOLO sobre el slice de calibración
    (``TRAIN_calib``, posterior a ``TRAIN_fit``), NUNCA con outcomes de ``TEST``.
    Ver el protocolo ``[TRAIN_fit | TRAIN_calib | embargo | TEST]`` arriba.

    Expone ``__call__(p: float) -> float`` para encajar directamente como
    ``DqnAlphaAgent(calibrator=...)``; al cablearlo, el agente marca
    ``TradeSignal.p_win_calibrated=True`` (habilita el guard de sizing, R-02).

    Parameters
    ----------
    method : 'isotonic' | 'sigmoid'
        'isotonic' = IsotonicRegression no-paramétrica (>= ``min_samples_isotonic``;
        si hay menos, cae a 'sigmoid' automáticamente). 'sigmoid' = Platt 1D
        (LogisticRegression), válido con pocas muestras.
    min_samples_isotonic : int
        Umbral de muestras para usar isotónica; por debajo usa sigmoid.

    Examples
    --------
    >>> cal = ScalarProbabilityCalibrator(method="isotonic").fit(p_raw, outcomes)
    >>> p_cal = cal(0.61)            # escalar -> escalar, en [0, 1]
    """

    def __init__(self, method: str = "isotonic", min_samples_isotonic: int = 80):
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"method debe ser 'isotonic' o 'sigmoid', no '{method}'")
        self.method = method
        self.min_samples_isotonic = min_samples_isotonic
        self._cal = None
        self._kind: Optional[str] = None   # 'isotonic' | 'sigmoid' | 'constant'
        self._constant: float = 0.5
        self._fitted = False

    def fit(self, p_raw, outcomes) -> "ScalarProbabilityCalibrator":
        """
        Ajusta sobre pares ``(p_raw, outcome)`` del slice de calibración.

        Parameters
        ----------
        p_raw : array-like de float en [0, 1]
            Confianza cruda de la acción elegida por muestra (softmax de Q).
        outcomes : array-like de {0, 1}
            1 = la apuesta direccional resultó ganadora, 0 = perdedora.
        """
        p = np.asarray(p_raw, dtype=float).ravel()
        y = np.asarray(outcomes, dtype=float).ravel()
        if p.shape != y.shape:
            raise ValueError(f"p_raw {p.shape} y outcomes {y.shape} deben alinear")
        if len(p) == 0:
            raise ValueError("conjunto de calibración vacío")

        # Degenerado: una sola clase de outcome -> mapeo constante (sin crash).
        if np.unique(y).size < 2:
            self._kind = "constant"
            self._constant = float(np.clip(y.mean(), 0.0, 1.0))
            self._fitted = True
            logger.warning(
                "ScalarProbabilityCalibrator: outcomes de una sola clase "
                "(%d muestras) -> mapeo constante = %.3f", len(y), self._constant
            )
            return self

        method = self.method
        if method == "isotonic" and len(p) < self.min_samples_isotonic:
            logger.warning(
                "Calibración pequeña (%d < %d) -> usando 'sigmoid' en vez de "
                "'isotonic'.", len(p), self.min_samples_isotonic
            )
            method = "sigmoid"

        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(p, y)
        else:
            cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            cal.fit(p.reshape(-1, 1), y)

        self._cal = cal
        self._kind = method
        self._fitted = True
        logger.info(
            "ScalarProbabilityCalibrator %s ajustado sobre %d muestras "
            "(tasa de acierto base = %.3f).", method, len(p), float(y.mean())
        )
        return self

    def transform(self, p_raw) -> np.ndarray:
        """Calibra un array de scores crudos -> probabilidades en [0, 1]."""
        if not self._fitted:
            raise RuntimeError("Llama a .fit() antes de transform/__call__")
        p = np.asarray(p_raw, dtype=float).ravel()
        if self._kind == "constant":
            out = np.full(p.shape, self._constant, dtype=float)
        elif self._kind == "isotonic":
            out = self._cal.predict(p)
        else:  # sigmoid
            out = self._cal.predict_proba(p.reshape(-1, 1))[:, 1]
        return np.clip(out, 0.0, 1.0)

    def __call__(self, p: float) -> float:
        """Calibra un score escalar (firma de ``DqnAlphaAgent(calibrator=...)``)."""
        return float(self.transform([float(p)])[0])

    @property
    def is_fitted(self) -> bool:
        return self._fitted
