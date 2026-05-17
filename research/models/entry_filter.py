"""
Filtro de Entrada Probabilístico
=================================
Filtra señales del modelo usando umbrales sobre probabilidades calibradas.

PROBLEMA SIN FILTRO:
  El modelo predice {-1, 0, +1} para cada barra. Sin filtro, entra en toda
  señal no-neutral. Pero muchas señales tienen P(y=c|x) ≈ 0.38 — apenas por
  encima del azar para 3 clases. Entrar en esas señales añade ruido, no alfa.

SOLUCIÓN — Filtro probabilístico:
  Solo entra si la confianza del modelo supera un umbral:
    - Señal larga:  P(y=+1|x) > threshold_long
    - Señal corta:  P(y=-1|x) > threshold_short
    - Si ninguna supera umbral: clase efectiva = 0 (no entrar)

  El threshold óptimo se busca en TRAIN_calib (no en TEST):
    - Grid search sobre [1/K + ε, 0.95] donde K = número de clases
    - Métrica: Sharpe del P&L o proxy label-based
    - Penalización si coverage < min_coverage (Sharpe inflado con pocos trades)

PROTOCOLO CORRECTO en walk-forward:
  [TRAIN_fit] | [TRAIN_calib] | embargo | [TEST]
       ↓              ↓                      ↓
  fit XGBoost   calibrar + optimizar     aplicar threshold
                threshold aquí           NUNCA optimizar aquí

TRADEOFF cobertura vs precisión:
  threshold bajo  → muchas señales, precisión media, Sharpe potencialmente bajo
  threshold alto  → pocas señales, alta precisión, Sharpe potencialmente alto
                    pero riesgo de sobreajuste al set de calibración

  El grid search encuentra el punto óptimo en el set de calibración.
  Coverage mínimo recomendado: 5-10% de las barras (al menos ~10 trades OOS).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# MÉTRICAS AUXILIARES
# =====================================================================

def _sharpe_from_returns(returns: np.ndarray, min_trades: int = 5) -> float:
    """
    Sharpe anualizado (asume barras diarias) de una serie de retornos.
    Devuelve -np.inf si hay menos de min_trades trades.
    """
    if len(returns) < min_trades:
        return -np.inf
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma < 1e-10:
        return -np.inf
    return float(mu / sigma * np.sqrt(252))


def _label_based_returns(
    signals: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """
    Proxy de retornos cuando no se tienen retornos reales de barra.

    Para cada barra con señal != 0:
      - Si signal == y_true: retorno = +1  (trade correcto)
      - Si signal != y_true: retorno = -1  (trade incorrecto)

    Las barras sin señal (signal == 0) se excluyen del cálculo.

    Este proxy asume R:R simétrico. No es perfecto pero es correcto
    para optimizar threshold: si el modelo acerta más con threshold alto,
    el Sharpe proxy aumenta.
    """
    mask = signals != 0
    if mask.sum() == 0:
        return np.array([])
    correct = (signals[mask] == y_true[mask]).astype(float)
    # +1 si correcto, -1 si incorrecto
    return 2.0 * correct - 1.0


# =====================================================================
# FILTRO PRINCIPAL
# =====================================================================

class ProbabilityEntryFilter:
    """
    Filtra señales de un clasificador calibrado por umbral de confianza.

    Flujo:
      1. fit(proba_calib, y_calib) → busca threshold óptimo en set de calibración
      2. predict(proba_new)        → aplica threshold a nuevas barras

    Soporta:
      - Modelos 3-class {-1, 0, +1}  (multiclass, índices = 0,1,2 para -1,0,+1)
      - Modelos 2-class {-1, +1}     (binario)
      - Threshold simétrico (mismo para long y short) o asimétrico

    Parameters
    ----------
    class_labels : list
        Etiquetas originales del modelo, en el mismo orden que las columnas
        de predict_proba(). Ejemplo: [-1, 0, 1] o [-1, 1].
    symmetric : bool
        Si True, usa el mismo threshold para long y short.
        Si False, optimiza threshold_long y threshold_short por separado.
    min_coverage : float
        Fracción mínima de barras que deben generar señal.
        Thresholds con coverage < min_coverage son descartados.
    n_thresholds : int
        Número de puntos en el grid search.
    min_trades : int
        Número mínimo de trades en el set de calibración para que el
        Sharpe sea considerado válido.
    fallback_threshold : float
        Threshold a usar si el set de calibración es demasiado pequeño
        para optimizar (< min_samples_to_optimize).
    min_samples_to_optimize : int
        Mínimo de muestras en el set de calibración para hacer grid search.
        Con menos muestras, usa fallback_threshold.
    """

    def __init__(
        self,
        class_labels: list = [-1, 0, 1],
        symmetric: bool = True,
        min_coverage: float = 0.05,
        n_thresholds: int = 50,
        min_trades: int = 10,
        fallback_threshold: float = 0.45,
        min_samples_to_optimize: int = 60,
    ):
        self.class_labels = list(class_labels)
        self.symmetric = symmetric
        self.min_coverage = min_coverage
        self.n_thresholds = n_thresholds
        self.min_trades = min_trades
        self.fallback_threshold = fallback_threshold
        self.min_samples_to_optimize = min_samples_to_optimize

        self.threshold_long_: float = fallback_threshold
        self.threshold_short_: float = fallback_threshold
        self._fitted: bool = False
        self._search_results_: Optional[pd.DataFrame] = None

        # Índices de las clases de interés en el array de probabilidades
        self._idx_long: Optional[int] = None
        self._idx_short: Optional[int] = None
        self._n_classes: int = len(class_labels)

        # Validar que existan clases +1 y -1
        if 1 in self.class_labels:
            self._idx_long = self.class_labels.index(1)
        if -1 in self.class_labels:
            self._idx_short = self.class_labels.index(-1)

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(
        self,
        proba_calib: np.ndarray,
        y_calib: np.ndarray,
        bar_returns_calib: Optional[np.ndarray] = None,
    ) -> "ProbabilityEntryFilter":
        """
        Busca el threshold óptimo sobre el set de calibración.

        Parameters
        ----------
        proba_calib : np.ndarray, shape (n, n_classes)
            Probabilidades CALIBRADAS del modelo en el set de calibración.
            Columnas en el mismo orden que class_labels.
        y_calib : np.ndarray, shape (n,)
            Etiquetas verdaderas en el set de calibración (valores originales,
            ej. -1, 0, 1).
        bar_returns_calib : np.ndarray, shape (n,), opcional
            Retornos reales de cada barra (ej. log-return o % return).
            Si se proporciona, el Sharpe se calcula sobre retornos reales
            ponderados por la señal (return_i × signal_i).
            Si es None, usa proxy label-based (+1/-1 según acierto).
        """
        n = len(proba_calib)
        y_arr = np.asarray(y_calib)

        if n < self.min_samples_to_optimize:
            logger.warning(
                f"Set de calibración pequeño ({n} < {self.min_samples_to_optimize}). "
                f"Usando threshold fallback = {self.fallback_threshold:.2f}."
            )
            self.threshold_long_ = self.fallback_threshold
            self.threshold_short_ = self.fallback_threshold
            self._fitted = True
            return self

        # Rango de búsqueda: desde ligeramente por encima de azar (1/K)
        # hasta 0.90 (thresholds más altos dan cobertura < 1%)
        t_min = round(1.0 / self._n_classes + 0.05, 3)
        t_max = 0.90
        thresholds = np.linspace(t_min, t_max, self.n_thresholds)

        if self.symmetric:
            results = self._grid_search_symmetric(
                thresholds, proba_calib, y_arr, bar_returns_calib, n
            )
            best = results.loc[results["sharpe"].idxmax()]
            self.threshold_long_ = float(best["threshold"])
            self.threshold_short_ = float(best["threshold"])
            self._search_results_ = results
            logger.info(
                f"Threshold óptimo (simétrico): {self.threshold_long_:.3f}  "
                f"Sharpe={best['sharpe']:.3f}  "
                f"Coverage={best['coverage']:.1%}  "
                f"Precision={best['precision']:.3f}"
            )
        else:
            results_long, results_short = self._grid_search_asymmetric(
                thresholds, proba_calib, y_arr, bar_returns_calib, n
            )
            best_long = results_long.loc[results_long["sharpe"].idxmax()]
            best_short = results_short.loc[results_short["sharpe"].idxmax()]
            self.threshold_long_ = float(best_long["threshold"])
            self.threshold_short_ = float(best_short["threshold"])
            self._search_results_ = (results_long, results_short)
            logger.info(
                f"Threshold long:  {self.threshold_long_:.3f}  "
                f"Sharpe={best_long['sharpe']:.3f}  "
                f"Coverage={best_long['coverage']:.1%}"
            )
            logger.info(
                f"Threshold short: {self.threshold_short_:.3f}  "
                f"Sharpe={best_short['sharpe']:.3f}  "
                f"Coverage={best_short['coverage']:.1%}"
            )

        self._fitted = True
        return self

    def _grid_search_symmetric(
        self,
        thresholds: np.ndarray,
        proba: np.ndarray,
        y_true: np.ndarray,
        bar_returns: Optional[np.ndarray],
        n: int,
    ) -> pd.DataFrame:
        """Grid search con el mismo threshold para long y short."""
        rows = []
        for t in thresholds:
            signals = self._apply_threshold(proba, t, t)
            coverage = float((signals != 0).mean())

            if coverage < self.min_coverage:
                rows.append({
                    "threshold": t, "sharpe": -np.inf,
                    "coverage": coverage, "precision": np.nan,
                    "n_trades": int((signals != 0).sum()),
                })
                continue

            sharpe, precision = self._evaluate_signals(
                signals, y_true, bar_returns
            )
            rows.append({
                "threshold": t, "sharpe": sharpe,
                "coverage": coverage, "precision": precision,
                "n_trades": int((signals != 0).sum()),
            })

        return pd.DataFrame(rows)

    def _grid_search_asymmetric(
        self,
        thresholds: np.ndarray,
        proba: np.ndarray,
        y_true: np.ndarray,
        bar_returns: Optional[np.ndarray],
        n: int,
    ):
        """Grid search independiente para long y short."""
        rows_long, rows_short = [], []

        for t in thresholds:
            # Long: solo señales largas
            signals_long_only = self._apply_threshold(proba, t_long=t, t_short=0.0)
            mask_long = signals_long_only == 1
            cov_long = float(mask_long.mean())

            if cov_long < self.min_coverage / 2:
                rows_long.append({
                    "threshold": t, "sharpe": -np.inf,
                    "coverage": cov_long, "precision": np.nan,
                    "n_trades": int(mask_long.sum()),
                })
            else:
                sh, prec = self._evaluate_signals(
                    signals_long_only, y_true, bar_returns
                )
                rows_long.append({
                    "threshold": t, "sharpe": sh,
                    "coverage": cov_long, "precision": prec,
                    "n_trades": int(mask_long.sum()),
                })

            # Short: solo señales cortas
            signals_short_only = self._apply_threshold(proba, t_long=0.0, t_short=t)
            mask_short = signals_short_only == -1
            cov_short = float(mask_short.mean())

            if cov_short < self.min_coverage / 2:
                rows_short.append({
                    "threshold": t, "sharpe": -np.inf,
                    "coverage": cov_short, "precision": np.nan,
                    "n_trades": int(mask_short.sum()),
                })
            else:
                sh, prec = self._evaluate_signals(
                    signals_short_only, y_true, bar_returns
                )
                rows_short.append({
                    "threshold": t, "sharpe": sh,
                    "coverage": cov_short, "precision": prec,
                    "n_trades": int(mask_short.sum()),
                })

        return pd.DataFrame(rows_long), pd.DataFrame(rows_short)

    def _apply_threshold(
        self,
        proba: np.ndarray,
        t_long: float,
        t_short: float,
    ) -> np.ndarray:
        """
        Aplica threshold a las probabilidades y devuelve señales {-1, 0, +1}.
        """
        n = proba.shape[0]
        signals = np.zeros(n, dtype=int)

        if self._idx_long is not None and t_long > 0:
            signals = np.where(proba[:, self._idx_long] > t_long, 1, signals)
        if self._idx_short is not None and t_short > 0:
            signals = np.where(proba[:, self._idx_short] > t_short, -1, signals)

        # Si ambos superan su umbral (raro pero posible), tomar el mayor
        if self._idx_long is not None and self._idx_short is not None:
            both_mask = (
                (proba[:, self._idx_long] > t_long) &
                (proba[:, self._idx_short] > t_short)
            )
            if both_mask.any():
                # Tomar la clase con mayor probabilidad
                long_wins = (
                    proba[both_mask, self._idx_long] >=
                    proba[both_mask, self._idx_short]
                )
                signals[both_mask] = np.where(long_wins, 1, -1)

        return signals

    def _evaluate_signals(
        self,
        signals: np.ndarray,
        y_true: np.ndarray,
        bar_returns: Optional[np.ndarray],
    ):
        """
        Calcula Sharpe y Precision para un conjunto de señales.
        Devuelve (sharpe, precision).
        """
        mask = signals != 0
        n_trades = int(mask.sum())

        if n_trades < self.min_trades:
            return -np.inf, np.nan

        # Precision: fracción de señales no-neutrales que aciertan
        correct = (signals[mask] == y_true[mask]).sum()
        precision = float(correct / n_trades) if n_trades > 0 else np.nan

        # Sharpe
        if bar_returns is not None:
            # Retornos reales ponderados por la señal: signal_i × return_i
            trade_returns = signals[mask].astype(float) * bar_returns[mask]
        else:
            # Proxy label-based
            trade_returns = _label_based_returns(signals, y_true)

        sharpe = _sharpe_from_returns(trade_returns, self.min_trades)
        return sharpe, precision

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------

    def predict(self, proba: np.ndarray) -> np.ndarray:
        """
        Aplica los thresholds optimizados a nuevas probabilidades.

        Parameters
        ----------
        proba : np.ndarray, shape (n, n_classes)
            Probabilidades calibradas del modelo.

        Returns
        -------
        np.ndarray, shape (n,)
            Señales filtradas {-1, 0, +1}. Los 0 significan "no entrar".
        """
        if not self._fitted:
            raise RuntimeError("Llama a .fit() antes de .predict()")
        return self._apply_threshold(proba, self.threshold_long_, self.threshold_short_)

    # ------------------------------------------------------------------
    # REPORTE
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def filter_stats(
        self,
        proba: np.ndarray,
        y_true: np.ndarray,
        bar_returns: Optional[np.ndarray] = None,
        label: str = "",
    ) -> dict:
        """
        Estadísticas del filtro aplicado a un conjunto de datos.

        Útil para reportar métricas en el set de calibración y de test.
        """
        if not self._fitted:
            raise RuntimeError("Llama a .fit() primero.")

        signals = self.predict(proba)
        n = len(signals)
        mask = signals != 0
        n_trades = int(mask.sum())
        coverage = float(n_trades / n) if n > 0 else 0.0

        precision = np.nan
        sharpe = np.nan
        if n_trades >= self.min_trades:
            correct = int((signals[mask] == y_true[mask]).sum())
            precision = correct / n_trades
            if bar_returns is not None:
                trade_returns = signals[mask].astype(float) * bar_returns[mask]
            else:
                trade_returns = _label_based_returns(signals, y_true)
            sharpe = _sharpe_from_returns(trade_returns, self.min_trades)

        n_long = int((signals == 1).sum())
        n_short = int((signals == -1).sum())

        stats = {
            "label": label,
            "threshold_long": round(self.threshold_long_, 4),
            "threshold_short": round(self.threshold_short_, 4),
            "n_bars": n,
            "n_trades": n_trades,
            "n_long": n_long,
            "n_short": n_short,
            "coverage": round(coverage, 4),
            "precision": round(precision, 4) if not np.isnan(precision) else None,
            "sharpe_proxy": round(sharpe, 4) if not np.isnan(sharpe) else None,
        }

        logger.info(
            f"EntryFilter {label}: "
            f"threshold=({self.threshold_long_:.3f}/{self.threshold_short_:.3f})  "
            f"coverage={coverage:.1%}  "
            f"trades={n_trades} (L:{n_long} S:{n_short})  "
            f"precision={precision:.3f}" if not np.isnan(precision) else
            f"EntryFilter {label}: threshold=({self.threshold_long_:.3f}/{self.threshold_short_:.3f})  "
            f"coverage={coverage:.1%}  trades={n_trades}"
        )
        return stats

    def search_results_dataframe(self) -> pd.DataFrame:
        """
        Devuelve el DataFrame de resultados del grid search.
        Solo disponible si symmetric=True.
        """
        if not self._fitted:
            raise RuntimeError("Llama a .fit() primero.")
        if self._search_results_ is None:
            return pd.DataFrame()
        if isinstance(self._search_results_, tuple):
            long_df, short_df = self._search_results_
            long_df = long_df.copy()
            short_df = short_df.copy()
            long_df["side"] = "long"
            short_df["side"] = "short"
            return pd.concat([long_df, short_df], ignore_index=True)
        return self._search_results_.copy()

    def __repr__(self) -> str:
        if not self._fitted:
            return f"ProbabilityEntryFilter(not fitted)"
        return (
            f"ProbabilityEntryFilter("
            f"threshold_long={self.threshold_long_:.3f}, "
            f"threshold_short={self.threshold_short_:.3f}, "
            f"symmetric={self.symmetric})"
        )


# =====================================================================
# INTEGRACIÓN CON WALK-FORWARD — función de conveniencia
# =====================================================================

def fit_entry_filter(
    model,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    bar_returns_calib: Optional[np.ndarray] = None,
    symmetric: bool = True,
    min_coverage: float = 0.05,
    n_thresholds: int = 50,
) -> ProbabilityEntryFilter:
    """
    Función de conveniencia: dado un modelo ya calibrado y el set de
    calibración, ajusta y devuelve el filtro de entrada.

    Uso típico en walk-forward:
        X_fit, y_fit, X_calib, y_calib = split_train_for_calibration(X, y)
        model.fit(X_fit, y_fit)
        model.calibrate(X_calib, y_calib)
        entry_filter = fit_entry_filter(model, X_calib, y_calib)
        # En test:
        proba = model.predict_proba(X_test)
        signals = entry_filter.predict(proba)

    Parameters
    ----------
    model : XGBoostClassifier con .is_calibrated == True
    X_calib : features del set de calibración
    y_calib : etiquetas originales ({-1, 0, 1}) del set de calibración
    bar_returns_calib : retornos reales de barra (opcional)
    """
    if not model.is_calibrated:
        raise RuntimeError(
            "El modelo no está calibrado. Llama a model.calibrate() primero."
        )

    class_labels = sorted(model.label_map_.keys())
    proba_calib = model.predict_proba(X_calib)
    y_arr = np.asarray(y_calib)

    ef = ProbabilityEntryFilter(
        class_labels=class_labels,
        symmetric=symmetric,
        min_coverage=min_coverage,
        n_thresholds=n_thresholds,
    )
    ef.fit(proba_calib, y_arr, bar_returns_calib)
    return ef
