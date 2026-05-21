"""
Classification Metrics & Diagnostics
======================================
Implementa las métricas del cheatsheet CS229 (Stanford) para el problema
de trading 3-clases {-1, 0, +1}:

  Métricas de clasificación
  -------------------------
  - Confusion matrix
  - Precision / Recall / F1  (macro + weighted + por clase)
  - AUC-ROC  one-vs-rest, macro average
  - Accuracy

  Diagnóstico Bias/Varianza
  --------------------------
  - Train accuracy  (en X_fit, post-entrenamiento)
  - Calib accuracy  (en X_calib, antes de calibrar)
  - Test  accuracy  (en X_test, OOS)
  - Verdict: "underfitting" | "just_right" | "overfitting"

  El gap train_acc - test_acc > umbral → overfitting (high variance).
  Train y test ambos bajos → underfitting (high bias).

Referencia: CS 229 VIP Cheatsheet: Machine Learning Tips (Amidi & Amidi, 2018).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Umbral de gap (train - test) para declarar overfitting
OVERFIT_GAP = 0.12
# Umbral de accuracy "bajo" (para 3 clases, random = 33%)
UNDERFIT_THRESHOLD = 0.42


# =====================================================================
# DATACLASS DE MÉTRICAS POR FOLD
# =====================================================================

@dataclass
class FoldClassificationMetrics:
    """
    Métricas de clasificación para un fold del walk-forward.

    Atributos "activos": sólo barras con señal ≠ 0 (el modelo tomó posición).
    Atributos "all": todas las barras del test.
    """
    # Accuracy global
    accuracy_all: Optional[float] = None
    accuracy_active: Optional[float] = None    # sólo señales ≠ 0

    # Macro-averages (no penaliza clases mayoritarias)
    precision_macro: Optional[float] = None
    recall_macro: Optional[float] = None
    f1_macro: Optional[float] = None

    # Weighted-averages (pondera por soporte — útil con 68% neutros)
    precision_weighted: Optional[float] = None
    recall_weighted: Optional[float] = None
    f1_weighted: Optional[float] = None

    # Por clase
    precision_long: Optional[float] = None
    recall_long: Optional[float] = None
    f1_long: Optional[float] = None

    precision_short: Optional[float] = None
    recall_short: Optional[float] = None
    f1_short: Optional[float] = None

    precision_neutral: Optional[float] = None
    recall_neutral: Optional[float] = None
    f1_neutral: Optional[float] = None

    # AUC one-vs-rest macro
    auc_macro: Optional[float] = None

    # Confusion matrix (como dict serializable, índices=class_labels)
    confusion_matrix: Optional[dict] = None   # {true_cls: {pred_cls: count}}

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# =====================================================================
# COMPUTE CLASSIFICATION METRICS
# =====================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    class_labels: list,
    signals: Optional[np.ndarray] = None,
) -> FoldClassificationMetrics:
    """
    Calcula métricas de clasificación completas para un fold.

    Parameters
    ----------
    y_true       : labels reales {-1, 0, +1}
    y_pred       : predicciones del modelo {-1, 0, +1}
    y_proba      : probabilidades calibradas (n x n_classes); puede ser None
    class_labels : lista ordenada de clases (ej. [-1, 0, 1])
    signals      : señales del entry filter {-1, 0, +1};
                   si se pasa, se computan métricas "active" (señal ≠ 0)

    Returns
    -------
    FoldClassificationMetrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    m = FoldClassificationMetrics()

    if len(y_true) == 0:
        return m

    # ── Accuracy ──────────────────────────────────────────────────────
    m.accuracy_all = round(float(accuracy_score(y_true, y_pred)), 4)

    if signals is not None:
        active = np.asarray(signals) != 0
        if active.sum() >= 2:
            m.accuracy_active = round(
                float(accuracy_score(y_true[active], y_pred[active])), 4
            )

    # ── Macro / Weighted averages ─────────────────────────────────────
    kw = dict(labels=class_labels, zero_division=0)
    m.precision_macro    = round(float(precision_score(y_true, y_pred, average="macro",    **kw)), 4)
    m.recall_macro       = round(float(recall_score   (y_true, y_pred, average="macro",    **kw)), 4)
    m.f1_macro           = round(float(f1_score       (y_true, y_pred, average="macro",    **kw)), 4)
    m.precision_weighted = round(float(precision_score(y_true, y_pred, average="weighted", **kw)), 4)
    m.recall_weighted    = round(float(recall_score   (y_true, y_pred, average="weighted", **kw)), 4)
    m.f1_weighted        = round(float(f1_score       (y_true, y_pred, average="weighted", **kw)), 4)

    # ── Por clase ─────────────────────────────────────────────────────
    cls_map = {-1: "short", 0: "neutral", 1: "long"}
    per_p = precision_score(y_true, y_pred, average=None, **kw)
    per_r = recall_score   (y_true, y_pred, average=None, **kw)
    per_f = f1_score       (y_true, y_pred, average=None, **kw)

    for i, cls in enumerate(class_labels):
        name = cls_map.get(cls, str(cls))
        setattr(m, f"precision_{name}", round(float(per_p[i]), 4))
        setattr(m, f"recall_{name}",    round(float(per_r[i]), 4))
        setattr(m, f"f1_{name}",        round(float(per_f[i]), 4))

    # ── AUC one-vs-rest ───────────────────────────────────────────────
    if y_proba is not None and len(np.unique(y_true)) >= 2:
        try:
            m.auc_macro = round(float(
                roc_auc_score(
                    y_true, y_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=class_labels,
                )
            ), 4)
        except Exception as e:
            logger.debug("AUC computation failed: %s", e)

    # ── Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_dict: dict = {}
    for i, true_cls in enumerate(class_labels):
        cm_dict[true_cls] = {pred_cls: int(cm[i, j]) for j, pred_cls in enumerate(class_labels)}
    m.confusion_matrix = cm_dict

    return m


def format_confusion_matrix(cm_dict: dict) -> pd.DataFrame:
    """
    Formatea el confusion_matrix dict en un DataFrame legible.

        Predicted →   -1    0   +1
    Actual -1         ...  ...  ...
           0          ...  ...  ...
          +1          ...  ...  ...
    """
    if not cm_dict:
        return pd.DataFrame()
    classes = sorted(cm_dict.keys())
    data = [[cm_dict[t].get(p, 0) for p in classes] for t in classes]
    col_names = {-1: "Pred -1 (Short)", 0: "Pred 0 (Neutral)", 1: "Pred +1 (Long)"}
    row_names = {-1: "Actual -1 (Short)", 0: "Actual 0 (Neutral)", 1: "Actual +1 (Long)"}
    return pd.DataFrame(
        data,
        index=[row_names.get(c, str(c)) for c in classes],
        columns=[col_names.get(c, str(c)) for c in classes],
    )


# =====================================================================
# BIAS / VARIANCE DIAGNOSTIC
# =====================================================================

def compute_bias_variance_verdict(
    train_acc: Optional[float],
    calib_acc: Optional[float],
    test_acc: Optional[float],
    overfit_gap: float = OVERFIT_GAP,
    underfit_threshold: float = UNDERFIT_THRESHOLD,
) -> str:
    """
    Diagnóstico bias/varianza a partir de accuracies en train/calib/test.

    Lógica (CS 229 cheatsheet):
      - Overfitting (high variance) : train_acc >> test_acc (gap > overfit_gap)
      - Underfitting (high bias)    : train_acc ≈ test_acc pero ambos bajos
      - Just right                  : train_acc ≈ test_acc y aceptables

    Para 3 clases equiprobables, azar = 33%.
    Umbral "bajo" por defecto = 42% (poca mejora sobre azar).

    Returns
    -------
    "underfitting" | "just_right" | "overfitting" | "unknown"
    """
    if train_acc is None or test_acc is None:
        return "unknown"

    gap = train_acc - test_acc

    if gap > overfit_gap:
        return "overfitting"
    elif train_acc < underfit_threshold and test_acc < underfit_threshold:
        return "underfitting"
    else:
        return "just_right"


# =====================================================================
# AGGREGATE CROSS-FOLD
# =====================================================================

def aggregate_classification_metrics(
    metrics_list: List[FoldClassificationMetrics],
) -> pd.DataFrame:
    """
    Agrega métricas de clasificación cross-fold.

    Returns DataFrame con media y std de cada métrica numérica.
    """
    if not metrics_list:
        return pd.DataFrame()

    # Extraer métricas numéricas (excluir confusion_matrix)
    numeric_keys = [
        "accuracy_all", "accuracy_active",
        "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "precision_long", "recall_long", "f1_long",
        "precision_short", "recall_short", "f1_short",
        "auc_macro",
    ]

    records = []
    for m in metrics_list:
        row = {}
        for k in numeric_keys:
            v = getattr(m, k, None)
            if v is not None:
                row[k] = v
        records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    return pd.DataFrame({
        "mean": df.mean().round(4),
        "std":  df.std().round(4),
        "min":  df.min().round(4),
        "max":  df.max().round(4),
    })


def bias_variance_summary(fold_results) -> pd.DataFrame:
    """
    Tabla resumen de bias/varianza por fold.

    Columns: fold | train_acc | calib_acc | test_acc | gap | verdict
    """
    rows = []
    for fr in fold_results:
        bv = getattr(fr, "bias_variance", {})
        rows.append({
            "fold":      fr.fold_idx + 1,
            "train_acc": bv.get("train_acc"),
            "calib_acc": bv.get("calib_acc"),
            "test_acc":  bv.get("test_acc"),
            "gap":       round((bv.get("train_acc") or 0) - (bv.get("test_acc") or 0), 4),
            "verdict":   bv.get("verdict", "unknown"),
        })
    return pd.DataFrame(rows)
