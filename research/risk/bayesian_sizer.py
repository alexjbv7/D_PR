"""
Bayesian P(win) Updating
========================
Combina dos estimaciones independientes de la probabilidad de ganar un trade:

  1. Prior de régimen  : P(win | régimen, dirección)
     ↳ Win rate histórico por régimen GMM y dirección de la señal.
        Ejemplo: en régimen "volátil" ir long ganó el 38%; en "tranquilo" el 52%.

  2. Likelihood del modelo : p_model
     ↳ Probabilidad del modelo primario (XGBoost) o del meta-labeler.

Combinación (Product of Experts):
  odds_prior = p_prior / (1 - p_prior)
  odds_model = p_model / (1 - p_model)
  odds_post  = odds_prior × odds_model
  p_post     = odds_post / (1 + odds_post)

Cuando ambos están de acuerdo (ambos > 0.5), el posterior es más extremo.
Cuando discrepan, se modula la confianza a la baja → menos trades, mejor precision.

Alternativa lineal:
  p_post = α × p_prior + (1 - α) × p_model  (weighted average)

Anti-leakage:
  BayesianWinUpdater.fit() se llama SÓLO sobre X_calib.
  update() se aplica por separado a X_test.

Requisito:
  use_regime_features=True  en WalkForwardConfig, para tener "regime_label"
  disponible en X_calib y X_test. Si no hay columna de régimen, la función
  devuelve p_model sin modificar (fallback silencioso).

Referencia: Bayes' theorem en sizing como en López de Prado (AFML, cap. 10),
  extendido con "product of experts" (Hinton, 2002).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Etiqueta de columna que busca el updater en X
REGIME_LABEL_COL = "regime_label"


# =====================================================================
# CONFIGURACIÓN
# =====================================================================

@dataclass
class BayesianSizerConfig:
    """
    Configuración del BayesianWinUpdater.

    combination    : "product" → product of experts (multiplicar odds).
                     "weighted" → promedio ponderado lineal.
    smoothing      : conteo de Laplace para el prior (evita 0/1 exactos).
                     1.0 = suavizado Laplace estándar.
    min_samples    : mínimo de observaciones por celda (régimen × dirección)
                     para usar el prior. Celdas con menos muestras usan el
                     prior global como fallback.
    prior_weight   : peso del prior en combinación "weighted" (∈ [0, 1]).
                     0.3 → 30% prior + 70% modelo.
    clip_eps       : cota mínima/máxima de probabilidad para evitar log(0).
    """
    combination: str = "product"    # "product" | "weighted"
    smoothing: float = 1.0
    min_samples: int = 20
    prior_weight: float = 0.3
    clip_eps: float = 1e-4


# =====================================================================
# BAYESIAN WIN UPDATER
# =====================================================================

class BayesianWinUpdater:
    """
    Estima el prior P(win | régimen, dirección) y actualiza p_model.

    Uso típico en WalkForwardRunner (paso 4c, tras meta-labeler):
        updater = BayesianWinUpdater(cfg)
        updater.fit(X_calib, y_calib, signals_calib)
        p_win_final = updater.update(p_win_arr, X_test, signals_test)
    """

    def __init__(self, config: Optional[BayesianSizerConfig] = None):
        self.cfg = config or BayesianSizerConfig()
        # prior_table[(regime, direction)] → float P(win)
        self._prior_table: dict = {}
        # prior global por dirección (fallback si régimen no conocido)
        self._global_prior: dict = {}
        self.is_fitted: bool = False
        self.n_regimes_: int = 0

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        signals: np.ndarray,
    ) -> "BayesianWinUpdater":
        """
        Estima P(win | régimen, dirección) en X_calib.

        Sólo considera barras con señal activa (≠ 0), que son las que luego
        se evaluarán en test. Las barras neutras no aportan información
        sobre la precisión del primario.

        Parameters
        ----------
        X        : DataFrame con al menos la columna REGIME_LABEL_COL.
                   Si falta, is_fitted se queda False (fallback silencioso).
        y_true   : labels reales {-1, 0, +1}
        signals  : señales del primario {-1, 0, +1}
        """
        cfg = self.cfg
        y_true = np.asarray(y_true)
        signals = np.asarray(signals)

        if REGIME_LABEL_COL not in X.columns:
            logger.debug(
                "BayesianWinUpdater.fit: columna '%s' no encontrada. "
                "Updater desactivado.", REGIME_LABEL_COL
            )
            return self

        regime_labels = X[REGIME_LABEL_COL].values.astype(int)
        active_mask = signals != 0

        if active_mask.sum() == 0:
            logger.debug("BayesianWinUpdater.fit: sin señales activas en calib.")
            return self

        # ── Calcular win/loss por (régimen, dirección) ─────────────────
        regimes = np.unique(regime_labels)
        self.n_regimes_ = len(regimes)
        directions = [-1, 1]

        # win = señal activa que coincide con el label real
        wins = (signals[active_mask] == y_true[active_mask]).astype(int)
        r_active = regime_labels[active_mask]
        s_active = signals[active_mask]

        # Prior table con suavizado de Laplace
        prior_table: dict = {}
        for r in regimes:
            for d in directions:
                cell_mask = (r_active == r) & (s_active == d)
                n_cell = int(cell_mask.sum())
                n_win = int(wins[cell_mask].sum())
                if n_cell < cfg.min_samples:
                    # Celda con pocas muestras: registrar pero marcar como fallback
                    prior_table[(int(r), d)] = None
                else:
                    # Laplace smoothing: (n_win + k) / (n_cell + 2k)
                    k = cfg.smoothing
                    prior_table[(int(r), d)] = (n_win + k) / (n_cell + 2 * k)

        self._prior_table = prior_table

        # ── Prior global por dirección (fallback) ─────────────────────
        for d in directions:
            d_mask = s_active == d
            n_d = int(d_mask.sum())
            n_win_d = int(wins[d_mask].sum())
            k = cfg.smoothing
            self._global_prior[d] = (n_win_d + k) / (n_d + 2 * k) if n_d > 0 else 0.5

        self.is_fitted = True
        logger.debug(
            "BayesianWinUpdater fit: %d régimens | prior long=%.2f short=%.2f",
            self.n_regimes_,
            self._global_prior.get(1, 0.5),
            self._global_prior.get(-1, 0.5),
        )
        return self

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

    def update(
        self,
        p_model: np.ndarray,
        X: pd.DataFrame,
        signals: np.ndarray,
    ) -> np.ndarray:
        """
        Actualiza p_win con el prior de régimen.

        Señales neutras (signals=0) → p_win queda 0.0 sin cambios.
        Si not is_fitted o sin columna de régimen → devuelve p_model original.

        Parameters
        ----------
        p_model  : array float [0,1], p_win por barra (del modelo o meta-labeler)
        X        : DataFrame de test con columna REGIME_LABEL_COL
        signals  : señales {-1, 0, +1} correspondientes a p_model

        Returns
        -------
        np.ndarray float [0,1], p_win actualizado
        """
        p_model = np.asarray(p_model, dtype=float)
        signals = np.asarray(signals)

        if not self.is_fitted:
            return p_model

        if REGIME_LABEL_COL not in X.columns:
            return p_model

        cfg = self.cfg
        regime_labels = X[REGIME_LABEL_COL].values.astype(int)
        result = p_model.copy()

        for i in range(len(signals)):
            if signals[i] == 0:
                result[i] = 0.0
                continue

            d = int(signals[i])
            r = int(regime_labels[i])

            p_prior = self._prior_table.get((r, d), None)
            if p_prior is None:
                p_prior = self._global_prior.get(d, 0.5)

            p_m = float(p_model[i])

            if cfg.combination == "product":
                result[i] = self._product_of_experts(p_m, p_prior, cfg.clip_eps)
            else:
                w = cfg.prior_weight
                result[i] = w * p_prior + (1.0 - w) * p_m

        return result

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _product_of_experts(
        p1: float,
        p2: float,
        eps: float = 1e-4,
    ) -> float:
        """
        Product of Experts: combina dos probabilidades multiplicando sus odds.

        odds_post = odds_1 × odds_2
        P_post    = odds_post / (1 + odds_post)
        """
        p1 = float(np.clip(p1, eps, 1.0 - eps))
        p2 = float(np.clip(p2, eps, 1.0 - eps))
        odds1 = p1 / (1.0 - p1)
        odds2 = p2 / (1.0 - p2)
        odds_post = odds1 * odds2
        return odds_post / (1.0 + odds_post)

    # ------------------------------------------------------------------
    # DIAGNÓSTICOS
    # ------------------------------------------------------------------

    def prior_table(self) -> pd.DataFrame:
        """
        Tabla legible del prior estimado.

        Columns: regime, direction, p_win, source ('estimated' | 'global_fallback')
        """
        if not self.is_fitted:
            return pd.DataFrame()

        rows = []
        for (r, d), p in self._prior_table.items():
            src = "estimated" if p is not None else "global_fallback"
            p_val = p if p is not None else self._global_prior.get(d, 0.5)
            rows.append({
                "regime": r,
                "direction": "long" if d == 1 else "short",
                "p_win": round(p_val, 4),
                "source": src,
            })
        return pd.DataFrame(rows).sort_values(["regime", "direction"]).reset_index(drop=True)

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"BayesianWinUpdater(not fitted, combination={self.cfg.combination!r})"
        return (
            f"BayesianWinUpdater("
            f"fitted, "
            f"n_regimes={self.n_regimes_}, "
            f"combination={self.cfg.combination!r}, "
            f"global_prior=long:{self._global_prior.get(1, 0.5):.2f}/"
            f"short:{self._global_prior.get(-1, 0.5):.2f})"
        )
