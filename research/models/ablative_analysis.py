"""
Ablative Analysis
=================
Mide la contribución de cada módulo del pipeline comparando el rendimiento
OOS con y sin ese módulo — análisis ablativo (CS229 cheatsheet).

"Ablative analysis is analyzing the root cause of the difference in performance
between the current and the baseline models."  — Amidi & Amidi (2018)

Configuraciones evaluadas (acumulativas):
  0. baseline   : XGBoost puro, sin GMM / PCA / Meta / Bayes
  1. +gmm       : + Regime features (GMM)
  2. +pca       : + PCA Denoising
  3. +meta      : + Meta-labeling
  4. +bayes     : + Bayesian P(win) (stack completo)

Por cada configuración se corre el WalkForwardRunner completo y se reportan:
  Sharpe | PSR | Win Rate | Coverage | F1(macro) | AUC | n_trades

Delta respecto al baseline muestra el valor incremental de cada módulo.

Uso:
    from models.ablative_analysis import AblativeAnalyzer
    analyzer = AblativeAnalyzer(base_config)
    result = analyzer.run(X, y, prices, atr, all_classes=[-1, 0, 1])
    print(result.summary())
    print(result.delta_table())

Nota: Cada configuración corre el walk-forward completo → puede tardar.
      Para evaluaciones rápidas, usar n_folds_max para limitar los folds.
"""
from __future__ import annotations

import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURACIÓN
# =====================================================================

@dataclass
class AblativeConfig:
    """
    Controla qué ablaciones se ejecutan.

    ablations       : lista de nombres de ablaciones a ejecutar.
                      Default: todas las 5 configuraciones acumulativas.
    n_folds_max     : si > 0, limita los datos a los últimos N folds
                      (recorta X/y para acelerar el análisis).
                      0 = usar todos los datos.
    verbose         : mostrar progreso durante el análisis.
    """
    ablations: List[str] = field(default_factory=lambda: [
        "baseline", "+gmm", "+pca", "+meta", "+bayes"
    ])
    n_folds_max: int = 0
    verbose: bool = True


# =====================================================================
# RESULTADO
# =====================================================================

@dataclass
class AblativeResult:
    """Resultado del análisis ablativo."""

    # Tabla principal: una fila por ablación
    metrics_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Resultados completos por ablación (para drill-down)
    wf_results: dict = field(default_factory=dict)  # nombre → WalkForwardResult

    def summary(self) -> str:
        if self.metrics_table.empty:
            return "AblativeResult vacío."
        lines = [
            "=" * 75,
            " ABLATIVE ANALYSIS — Contribución de cada módulo",
            "=" * 75,
            self.metrics_table.to_string(index=False),
            "=" * 75,
        ]
        return "\n".join(lines)

    def delta_table(self) -> pd.DataFrame:
        """
        Tabla de deltas respecto a baseline.

        Útil para ver cuánto aporta cada módulo.
        """
        df = self.metrics_table.copy()
        if df.empty or "config" not in df.columns:
            return df

        baseline_row = df[df["config"] == "baseline"]
        if baseline_row.empty:
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        baseline_vals = baseline_row[numeric_cols].iloc[0]
        delta = df[numeric_cols].subtract(baseline_vals)
        delta.columns = [f"Δ_{c}" for c in delta.columns]
        result = pd.concat([df[["config"]], delta], axis=1)
        return result.reset_index(drop=True)


# =====================================================================
# ANALYZER
# =====================================================================

class AblativeAnalyzer:
    """
    Corre el walk-forward runner con 5 configuraciones acumulativas
    y compara los resultados.

    Cada ablación desactiva/activa módulos del pipeline:

      baseline : use_regime_features=False, use_pca=False,
                 use_meta_labeling=False, use_bayesian_sizing=False
      +gmm     : use_regime_features=True
      +pca     : use_regime_features=True, use_pca=True
      +meta    : ... + use_meta_labeling=True
      +bayes   : ... + use_bayesian_sizing=True  (stack completo)
    """

    def __init__(
        self,
        base_config,
        ablative_config: Optional[AblativeConfig] = None,
    ):
        """
        Parameters
        ----------
        base_config      : WalkForwardConfig a usar como base.
                           Los flags de módulos se sobreescriben por cada ablación.
        ablative_config  : AblativeConfig con lista de ablaciones a ejecutar.
        """
        self.base_cfg = base_config
        self.abl_cfg  = ablative_config or AblativeConfig()

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prices: Optional[pd.Series] = None,
        atr: Optional[pd.Series] = None,
        all_classes: Optional[list] = None,
    ) -> AblativeResult:
        """
        Ejecuta las ablaciones configuradas y devuelve el resultado comparativo.
        """
        from models.walk_forward_runner import WalkForwardRunner

        abl_cfg = self.abl_cfg
        ablation_configs = self._build_ablation_configs()
        filtered = {k: v for k, v in ablation_configs.items()
                    if k in abl_cfg.ablations}

        # Recortar datos si n_folds_max > 0
        X_run, y_run, prices_run, atr_run = X, y, prices, atr
        if abl_cfg.n_folds_max > 0:
            X_run, y_run, prices_run, atr_run = self._trim_data(
                X, y, prices, atr, abl_cfg.n_folds_max
            )

        rows = []
        wf_results = {}

        for name, cfg in filtered.items():
            if abl_cfg.verbose:
                print(f"  [Ablative] Running '{name}' ...", end="", flush=True)
            t0 = time.time()
            try:
                runner = WalkForwardRunner(cfg)
                result = runner.run(
                    X=X_run, y=y_run,
                    prices=prices_run, atr=atr_run,
                    all_classes=all_classes,
                )
                elapsed = time.time() - t0
                row = self._extract_metrics(name, result, elapsed)
                rows.append(row)
                wf_results[name] = result
                if abl_cfg.verbose:
                    print(f" done ({elapsed:.1f}s)  Sharpe={row.get('sharpe')}  "
                          f"F1={row.get('f1_macro')}")
            except Exception as e:
                elapsed = time.time() - t0
                logger.warning("Ablation '%s' falló: %s", name, e)
                rows.append({"config": name, "error": str(e)})
                if abl_cfg.verbose:
                    print(f" ERROR: {e}")

        metrics_table = pd.DataFrame(rows) if rows else pd.DataFrame()
        return AblativeResult(metrics_table=metrics_table, wf_results=wf_results)

    # ------------------------------------------------------------------
    # BUILD CONFIGS
    # ------------------------------------------------------------------

    def _build_ablation_configs(self) -> dict:
        """Genera las 5 configuraciones acumulativas."""
        def make(
            regime=False, pca=False, meta=False, bayes=False
        ):
            cfg = deepcopy(self.base_cfg)
            cfg.use_regime_features  = regime
            cfg.use_pca              = pca
            cfg.use_meta_labeling    = meta
            cfg.use_bayesian_sizing  = bayes
            # Siempre rastrear métricas extendidas en ablación
            cfg.track_extended_metrics = True
            # Silenciar importancia para acelerar (podría ser lento)
            cfg.track_importance = False
            return cfg

        return {
            "baseline": make(regime=False, pca=False, meta=False, bayes=False),
            "+gmm":     make(regime=True,  pca=False, meta=False, bayes=False),
            "+pca":     make(regime=True,  pca=True,  meta=False, bayes=False),
            "+meta":    make(regime=True,  pca=True,  meta=True,  bayes=False),
            "+bayes":   make(regime=True,  pca=True,  meta=True,  bayes=True),
        }

    # ------------------------------------------------------------------
    # TRIM DATA
    # ------------------------------------------------------------------

    def _trim_data(self, X, y, prices, atr, n_folds_max):
        """
        Recorta los datos para ejecutar sólo los últimos `n_folds_max` folds.
        Calcula cuántas barras se necesitan mínimo.
        """
        cfg = self.base_cfg
        needed = (cfg.train_size
                  + n_folds_max * (cfg.test_size + cfg.embargo)
                  + cfg.embargo)
        if len(X) <= needed:
            return X, y, prices, atr
        start_idx = len(X) - needed
        X_trim = X.iloc[start_idx:]
        y_trim = y.iloc[start_idx:]
        p_trim = prices.iloc[start_idx:] if prices is not None else None
        a_trim = atr.iloc[start_idx:]    if atr is not None     else None
        return X_trim, y_trim, p_trim, a_trim

    # ------------------------------------------------------------------
    # EXTRACT METRICS
    # ------------------------------------------------------------------

    def _extract_metrics(self, name: str, result, elapsed: float) -> dict:
        """Extrae métricas clave de un WalkForwardResult."""
        gm = result.global_metrics or {}
        row = {
            "config":    name,
            "sharpe":    gm.get("sharpe"),
            "psr":       gm.get("psr"),
            "win_rate":  gm.get("win_rate"),
            "coverage":  round(gm.get("coverage", 0), 3),
            "n_trades":  gm.get("n_trades", 0),
            "max_dd":    gm.get("max_drawdown"),
            "elapsed_s": round(elapsed, 1),
        }

        # F1 macro y AUC: media cross-fold
        f1s  = [fr.classification_metrics.f1_macro
                for fr in result.fold_results
                if fr.classification_metrics and fr.classification_metrics.f1_macro]
        aucs = [fr.classification_metrics.auc_macro
                for fr in result.fold_results
                if fr.classification_metrics and fr.classification_metrics.auc_macro]
        bvs  = [fr.bias_variance.get("gap")
                for fr in result.fold_results
                if fr.bias_variance and fr.bias_variance.get("gap") is not None]

        row["f1_macro"]  = round(float(np.mean(f1s)), 4)  if f1s  else None
        row["auc_macro"] = round(float(np.mean(aucs)), 4) if aucs else None
        row["bv_gap"]    = round(float(np.mean(bvs)), 4)  if bvs  else None

        return row
