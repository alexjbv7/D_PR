"""
Error Analysis
==============
Analiza la raíz de los errores del modelo, comparando el modelo actual
con un "modelo perfecto" (CS229 cheatsheet: error analysis).

Dimensiones de análisis:

  1. Por dirección
     ─ Precision de señales long vs short
     ─ ¿El modelo falla más en un lado?

  2. Por confianza (bins de p_win)
     ─ Cuántos errores ocurren en señales con baja confianza (p_win < 0.5)?
     ─ El "umbral de corte" ideal se visualiza aquí.

  3. Por régimen (si disponible)
     ─ Precision/F1 por régimen GMM → ¿en qué mercado falla más?

  4. Por fold
     ─ Métricas cross-fold para detectar folds problemáticos.

  5. Confusion matrix agregada (suma de todos los folds)

Uso:
    from models.error_analysis import ErrorAnalyzer
    ea = ErrorAnalyzer()
    report = ea.analyze(result.fold_results, result.oos_sizing)
    print(report.summary())

Referencia: CS 229 Tips cheatsheet — "Error analysis is analyzing the root
cause of the difference in performance between the current and the perfect
models."
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_P_WIN_BINS = [0.0, 0.45, 0.55, 0.65, 1.01]
_P_WIN_LABELS = ["very_low (<0.45)", "low (0.45-0.55)", "medium (0.55-0.65)", "high (>0.65)"]


# =====================================================================
# RESULT DATACLASS
# =====================================================================

@dataclass
class ErrorAnalysisReport:
    """Resultado del análisis de errores cross-fold."""

    # 1. Por dirección
    direction_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 2. Por confianza (p_win bins)
    confidence_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 3. Por régimen
    regime_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 4. Por fold
    fold_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 5. Confusion matrix agregada
    confusion_agg: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Diagnóstico textual
    top_issues: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 65,
            " ERROR ANALYSIS REPORT",
            "=" * 65,
        ]

        if not self.direction_table.empty:
            lines += ["", "── Por dirección ──"]
            lines.append(self.direction_table.to_string(index=False))

        if not self.confidence_table.empty:
            lines += ["", "── Por confianza (p_win bins) ──"]
            lines.append(self.confidence_table.to_string(index=False))

        if not self.regime_table.empty:
            lines += ["", "── Por régimen ──"]
            lines.append(self.regime_table.to_string(index=False))

        if not self.fold_table.empty:
            lines += ["", "── Por fold ──"]
            lines.append(self.fold_table.to_string(index=False))

        if not self.confusion_agg.empty:
            lines += ["", "── Confusion matrix agregada ──"]
            lines.append(self.confusion_agg.to_string())

        if self.top_issues:
            lines += ["", "── Top issues ──"]
            for issue in self.top_issues:
                lines.append(f"  • {issue}")

        lines.append("=" * 65)
        return "\n".join(lines)


# =====================================================================
# ANALYZER
# =====================================================================

class ErrorAnalyzer:
    """
    Analiza los errores del walk-forward runner.

    Inputs requeridos por fold:
      - FoldResult.oos_signals (señales del entry filter)
      - FoldResult.oos_sizing  (incluye p_win por barra)
      - FoldResult.classification_metrics (precision/recall por clase)
      - FoldResult.regime_labels (opcional, si use_regime_features=True)

    Para análisis de errores necesitamos también y_true por barra → hay
    que pasar X y y completos con los índices de test de cada fold.
    """

    def analyze(
        self,
        fold_results: list,
        oos_sizing: pd.DataFrame,
        y_true_full: Optional[pd.Series] = None,
    ) -> ErrorAnalysisReport:
        """
        Análisis completo de errores.

        Parameters
        ----------
        fold_results  : lista de FoldResult del WalkForwardResult
        oos_sizing    : oos_sizing del WalkForwardResult (incluye signal, p_win)
        y_true_full   : Serie de labels reales con índice temporal completo.
                        Si None, los análisis que requieren y_true se omiten.
        """
        report = ErrorAnalysisReport()

        if not fold_results:
            return report

        # ── 1. Por dirección ──────────────────────────────────────────
        report.direction_table = self._direction_analysis(fold_results)

        # ── 2. Por confianza ──────────────────────────────────────────
        if "p_win" in oos_sizing.columns and y_true_full is not None:
            report.confidence_table = self._confidence_analysis(
                oos_sizing, y_true_full
            )

        # ── 3. Por régimen ────────────────────────────────────────────
        if y_true_full is not None:
            report.regime_table = self._regime_analysis(fold_results, y_true_full)

        # ── 4. Por fold ───────────────────────────────────────────────
        report.fold_table = self._fold_analysis(fold_results)

        # ── 5. Confusion matrix agregada ──────────────────────────────
        report.confusion_agg = self._aggregate_confusion(fold_results)

        # ── Diagnóstico textual ───────────────────────────────────────
        report.top_issues = self._diagnose(report, fold_results)

        return report

    # ------------------------------------------------------------------
    # 1. Análisis por dirección
    # ------------------------------------------------------------------

    def _direction_analysis(self, fold_results: list) -> pd.DataFrame:
        """Precision/Recall/F1 de señales long y short, cross-fold media."""
        rows = []
        for fr in fold_results:
            cm = fr.classification_metrics
            if cm is None:
                continue
            for direction, cls in [("long", 1), ("short", -1)]:
                p = getattr(cm, f"precision_{direction}", None)
                r = getattr(cm, f"recall_{direction}", None)
                f = getattr(cm, f"f1_{direction}", None)
                if p is not None:
                    rows.append({"fold": fr.fold_idx + 1,
                                 "direction": direction,
                                 "precision": p, "recall": r, "f1": f})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        agg = (df.groupby("direction")[["precision", "recall", "f1"]]
               .agg(["mean", "std"])
               .round(4))
        agg.columns = ["prec_mean", "prec_std", "rec_mean", "rec_std", "f1_mean", "f1_std"]
        return agg.reset_index()

    # ------------------------------------------------------------------
    # 2. Análisis por confianza
    # ------------------------------------------------------------------

    def _confidence_analysis(
        self,
        oos_sizing: pd.DataFrame,
        y_true: pd.Series,
    ) -> pd.DataFrame:
        """Error rate por bucket de p_win."""
        df = oos_sizing[["signal", "p_win"]].copy()
        df["y_true"] = y_true.reindex(df.index)
        active = df[df["signal"] != 0].dropna()
        if active.empty:
            return pd.DataFrame()

        active = active.copy()
        active["correct"] = (active["signal"] == active["y_true"]).astype(int)
        active["p_win_bin"] = pd.cut(
            active["p_win"],
            bins=_P_WIN_BINS,
            labels=_P_WIN_LABELS,
            right=False,
        )

        result = (
            active.groupby("p_win_bin", observed=True)
            .agg(
                n_trades=("correct", "count"),
                precision=("correct", "mean"),
                avg_p_win=("p_win", "mean"),
            )
            .round(4)
            .reset_index()
        )
        result.rename(columns={"p_win_bin": "confidence_bucket"}, inplace=True)
        return result

    # ------------------------------------------------------------------
    # 3. Análisis por régimen
    # ------------------------------------------------------------------

    def _regime_analysis(
        self,
        fold_results: list,
        y_true: pd.Series,
    ) -> pd.DataFrame:
        """Precision por régimen GMM (si hay regime_labels en los fold results)."""
        records = []
        for fr in fold_results:
            if fr.regime_labels is None or fr.oos_signals is None:
                continue
            regimes = fr.regime_labels.reindex(fr.oos_signals.index)
            signals = fr.oos_signals
            y_fold  = y_true.reindex(fr.oos_signals.index)

            for regime in regimes.dropna().unique():
                mask = (regimes == regime) & (signals != 0)
                if mask.sum() == 0:
                    continue
                sig_r  = signals[mask]
                y_r    = y_fold[mask]
                valid  = sig_r.notna() & y_r.notna()
                if valid.sum() == 0:
                    continue
                correct = (sig_r[valid] == y_r[valid]).mean()
                records.append({
                    "fold":      fr.fold_idx + 1,
                    "regime":    int(regime),
                    "n_trades":  int(mask.sum()),
                    "precision": round(float(correct), 4),
                })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        agg = (df.groupby("regime")
               .agg(
                   n_trades=("n_trades", "sum"),
                   precision_mean=("precision", "mean"),
                   precision_std=("precision", "std"),
               )
               .round(4)
               .reset_index())
        return agg

    # ------------------------------------------------------------------
    # 4. Análisis por fold
    # ------------------------------------------------------------------

    def _fold_analysis(self, fold_results: list) -> pd.DataFrame:
        """Resumen de métricas clave por fold."""
        rows = []
        for fr in fold_results:
            bv = fr.bias_variance or {}
            cm = fr.classification_metrics
            rows.append({
                "fold":       fr.fold_idx + 1,
                "n_trades":   fr.metrics.get("n_trades", 0),
                "coverage":   round(fr.metrics.get("coverage", 0), 3),
                "sharpe":     fr.metrics.get("sharpe"),
                "f1_macro":   getattr(cm, "f1_macro", None),
                "auc":        getattr(cm, "auc_macro", None),
                "train_acc":  bv.get("train_acc"),
                "test_acc":   bv.get("test_acc"),
                "gap":        bv.get("gap"),
                "bv_verdict": bv.get("verdict", "unknown"),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 5. Confusion matrix agregada
    # ------------------------------------------------------------------

    def _aggregate_confusion(self, fold_results: list) -> pd.DataFrame:
        """Suma las confusion matrices de todos los folds."""
        agg: dict = {}
        for fr in fold_results:
            cm_obj = fr.classification_metrics
            if cm_obj is None or cm_obj.confusion_matrix is None:
                continue
            for true_cls, preds in cm_obj.confusion_matrix.items():
                if true_cls not in agg:
                    agg[true_cls] = {}
                for pred_cls, cnt in preds.items():
                    agg[true_cls][pred_cls] = agg[true_cls].get(pred_cls, 0) + cnt

        if not agg:
            return pd.DataFrame()

        from models.metrics import format_confusion_matrix
        return format_confusion_matrix(agg)

    # ------------------------------------------------------------------
    # Diagnóstico textual
    # ------------------------------------------------------------------

    def _diagnose(self, report: ErrorAnalysisReport, fold_results: list) -> List[str]:
        """Genera diagnósticos textuales accionables."""
        issues = []

        # Bias/Variance
        verdicts = [
            fr.bias_variance.get("verdict", "unknown")
            for fr in fold_results
            if fr.bias_variance
        ]
        n_overfit  = verdicts.count("overfitting")
        n_underfit = verdicts.count("underfitting")
        if n_overfit > len(verdicts) / 2:
            issues.append(
                f"OVERFITTING en {n_overfit}/{len(verdicts)} folds → "
                "aumentar regularización (reg_alpha/reg_lambda) o reducir max_depth"
            )
        if n_underfit > len(verdicts) / 2:
            issues.append(
                f"UNDERFITTING en {n_underfit}/{len(verdicts)} folds → "
                "añadir features, reducir min_child_weight o aumentar n_estimators"
            )

        # Dirección asimétrica
        if not report.direction_table.empty:
            dt = report.direction_table.set_index("direction")
            if "long" in dt.index and "short" in dt.index:
                diff = abs(float(dt.loc["long", "prec_mean"]) -
                           float(dt.loc["short", "prec_mean"]))
                if diff > 0.10:
                    worse = ("long" if dt.loc["long", "prec_mean"] <
                             dt.loc["short", "prec_mean"] else "short")
                    issues.append(
                        f"ASIMETRÍA {diff:.0%} entre long y short precision → "
                        f"revisar features y class weights para señales {worse}"
                    )

        # Confianza
        if not report.confidence_table.empty:
            ct = report.confidence_table
            low_conf = ct[ct["confidence_bucket"].str.contains("low", na=False)]
            if not low_conf.empty:
                avg_low = low_conf["precision"].mean()
                if avg_low < 0.45:
                    issues.append(
                        f"Precision muy baja ({avg_low:.0%}) en señales de baja confianza → "
                        "considerar subir threshold del entry filter"
                    )

        # AUC general
        aucs = [
            fr.classification_metrics.auc_macro
            for fr in fold_results
            if fr.classification_metrics and fr.classification_metrics.auc_macro
        ]
        if aucs:
            avg_auc = np.mean(aucs)
            if avg_auc < 0.55:
                issues.append(
                    f"AUC macro bajo ({avg_auc:.3f} ≈ aleatorio=0.5) → "
                    "la señal del modelo es débil; revisar features e horizonte"
                )

        if not issues:
            issues.append("No se detectaron problemas críticos — modelo dentro de parámetros normales.")

        return issues
