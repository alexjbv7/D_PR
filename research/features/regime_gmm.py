"""
Regime Detection via Gaussian Mixture Model (GMM)
==================================================
Detecta el régimen de mercado (tendencia / rango / volátil) entrenando un GMM
sobre features estadísticos derivados de {close, ATR}.

Salidas por barra:
  regime_prob_0 .. regime_prob_{n-1}  : P(régimen=k | features_t)
  regime_label                         : régimen más probable (argmax)
  regime_entropy                       : H = -sum(p·log p) — incertidumbre

PROTOCOLO ANTI-LEAKAGE (para uso dentro de walk-forward):
  - El GMM se ajusta SOLO sobre el período de entrenamiento (prices_train, atr_train)
  - El StandardScaler también se ajusta SOLO en train
  - transform() se llama por separado sobre calib y test
  - NUNCA hay información futura en las features de régimen (solo lookback)

Consistencia cross-fold:
  Los componentes se re-ordenan por volatilidad media ascendente tras cada fit:
    regime_0 = régimen de menor volatilidad (ranging/quiet)
    regime_1 = régimen intermedio
    regime_2 = régimen de mayor volatilidad (trending/crisis)
  Esto garantiza que regime_prob_0 tenga el mismo significado semántico en
  todos los folds, aunque el GMM sea re-entrenado.

Features de régimen (todos vectorizados, sin rolling.apply Python):
  vol_20      : vol realizada 20 días (std de log-retornos)
  vol_ratio   : vol_20 / vol_60  — estado de compresión/expansión de vol
  sharpe_roll : media_ret_20 / vol_20  — proxy de trend strength
  autocorr_1  : correlación móvil(20) entre ret[t] y ret[t-1]
                > 0: momentum  |  < 0: mean-reverting
  atr_z       : z-score de ATR sobre ventana 40  — nivel de volatilidad relativa

Uso típico (dentro de WalkForwardRunner):
    det = GMMRegimeDetector(GMMRegimeConfig(n_components=3))
    det.fit(prices.reindex(X_train.index), atr.reindex(X_train.index))

    X_fit   = pd.concat([X_fit,   det.transform(prices_fit,   atr_fit)],   axis=1)
    X_calib = pd.concat([X_calib, det.transform(prices_calib, atr_calib)], axis=1)
    X_test  = pd.concat([X_test,  det.transform(prices_test,  atr_test)],  axis=1)

Uso offline (análisis):
    best_n = select_n_components(close, atr, max_n=6)
    det = GMMRegimeDetector(GMMRegimeConfig(n_components=best_n))
    feats = det.fit_transform(close, atr)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURACIÓN
# =====================================================================

@dataclass
class GMMRegimeConfig:
    """
    Configuración del detector de regímenes.

    n_components    : número fijo de regímenes (usar select_n_components para elegir)
    covariance_type : 'diag' recomendado para datasets chicos (~252 barras);
                      'full' más expresivo pero requiere más datos
    n_init          : restarts del EM (más = más estable, más lento)
    random_state    : semilla para reproducibilidad
    sort_by         : criterio de ordenamiento cross-fold
                      'vol'  — por volatilidad media ascendente (recomendado)
                      'none' — sin ordenar
    min_fit_bars    : mínimo de barras con features completas para ajustar el GMM
    """
    n_components: int = 3
    covariance_type: str = "diag"
    n_init: int = 5
    random_state: int = 42
    sort_by: str = "vol"
    min_fit_bars: int = 80


# =====================================================================
# DETECTOR
# =====================================================================

class GMMRegimeDetector:
    """
    Detector de regímenes de mercado usando Gaussian Mixture Model.

    Ver módulo docstring para protocolo anti-leakage y detalles de diseño.
    """

    def __init__(self, config: Optional[GMMRegimeConfig] = None):
        self.cfg = config or GMMRegimeConfig()
        self._gmm: Optional[GaussianMixture] = None
        self._scaler: Optional[StandardScaler] = None
        self._sort_order: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # FEATURE EXTRACTION (100% vectorizado)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_regime_features(close: pd.Series, atr: pd.Series) -> pd.DataFrame:
        """
        Construye las 5 features de régimen desde {close, atr}.

        Todo rolling vectorizado (sin .apply Python) para máxima velocidad.
        Devuelve DataFrame sin NaN (filas iniciales eliminadas).
        """
        eps = 1e-10
        ret = np.log(close / close.shift(1))

        # 1. Volatilidad realizada multi-escala
        vol_20 = ret.rolling(20).std()
        vol_60 = ret.rolling(60).std()

        # 2. Ratio de volatilidad (compresión/expansión)
        vol_ratio = vol_20 / (vol_60 + eps)

        # 3. Rolling Sharpe proxy = media(ret) / std(ret)  (trend strength)
        #    Positivo = tendencia alcista, negativo = bajista, ~0 = lateral
        mean_ret_20 = ret.rolling(20).mean()
        sharpe_roll = mean_ret_20 / (vol_20 + eps)

        # 4. Autocorrelación lag-1 de retornos (rolling 20)
        #    Vectorizado: cov(ret, ret.shift(1)) / var(ret)
        ret_lag = ret.shift(1)
        roll_cov = (
            ret.rolling(20).cov(ret_lag)
        )
        roll_var = ret.rolling(20).var()
        autocorr_1 = roll_cov / (roll_var + eps)

        # 5. Z-score del ATR (nivel de volatilidad relativa)
        atr_mean = atr.rolling(40).mean()
        atr_std  = atr.rolling(40).std()
        atr_z    = (atr - atr_mean) / (atr_std + eps)

        df = pd.DataFrame({
            "vol_20":     vol_20,
            "vol_ratio":  vol_ratio,
            "sharpe_roll": sharpe_roll,
            "autocorr_1": autocorr_1,
            "atr_z":      atr_z,
        }, index=close.index)

        return df.dropna()

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(self, close: pd.Series, atr: pd.Series) -> "GMMRegimeDetector":
        """
        Ajusta el GMM sobre el período de entrenamiento.

        Parameters
        ----------
        close : pd.Series   Precios de cierre del período de TRAIN solamente.
        atr   : pd.Series   ATR del período de TRAIN solamente.
        """
        cfg = self.cfg
        self.is_fitted = False

        regime_feats = self._build_regime_features(close, atr)

        if len(regime_feats) < cfg.min_fit_bars:
            logger.warning(
                "GMMRegimeDetector.fit: %d barras con features completas "
                "(mínimo=%d). Se usarán features neutras en transform().",
                len(regime_feats), cfg.min_fit_bars
            )
            return self

        X = regime_feats.values

        # Escalar (solo en train)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Ajustar GMM
        self._gmm = GaussianMixture(
            n_components=cfg.n_components,
            covariance_type=cfg.covariance_type,
            n_init=cfg.n_init,
            random_state=cfg.random_state,
        )
        self._gmm.fit(X_scaled)

        # Ordenar componentes por vol media para consistencia cross-fold
        if cfg.sort_by == "vol":
            self._sort_order = self._compute_vol_sort_order(X_scaled, regime_feats)
        else:
            self._sort_order = np.arange(cfg.n_components)

        self.is_fitted = True
        logger.debug(
            "GMMRegimeDetector fit: %d componentes | %d barras | orden=%s",
            cfg.n_components, len(X_scaled), self._sort_order,
        )
        return self

    def _compute_vol_sort_order(
        self, X_scaled: np.ndarray, regime_feats: pd.DataFrame
    ) -> np.ndarray:
        """
        Asigna labels a las barras y ordena componentes por vol_20 media ascendente.
        Garantiza: regime_0 = menor vol (quiet/ranging), regime_{n-1} = mayor vol.
        """
        try:
            raw_labels = self._gmm.predict(X_scaled)
            vol_20 = regime_feats["vol_20"].values
            mean_vol_per_comp = np.array([
                vol_20[raw_labels == k].mean() if (raw_labels == k).sum() > 0 else 0.0
                for k in range(self.cfg.n_components)
            ])
            return np.argsort(mean_vol_per_comp)  # ascendente
        except Exception:
            return np.arange(self.cfg.n_components)

    # ------------------------------------------------------------------
    # TRANSFORM
    # ------------------------------------------------------------------

    def transform(self, close: pd.Series, atr: pd.Series) -> pd.DataFrame:
        """
        Genera features de régimen para un conjunto de precios dado.

        Puede llamarse con el mismo período de train (fit_transform) o
        con un período futuro (calib / test) — ambos son válidos.

        Returns
        -------
        pd.DataFrame con columnas:
          regime_prob_0 .. regime_prob_{n-1}
          regime_label   (int, 0-indexed, ordenado por vol ascendente)
          regime_entropy (float >= 0)
        Mismo índice que `close`. Las barras sin features completas
        reciben probabilidades uniformes (neutras).
        """
        n = self.cfg.n_components
        full_index = close.index

        # Fallback: features neutras cuando el GMM no está ajustado
        if not self.is_fitted or self._gmm is None:
            return self._neutral_features(full_index, n)

        regime_feats = self._build_regime_features(close, atr)

        if regime_feats.empty:
            return self._neutral_features(full_index, n)

        X = regime_feats.values
        X_scaled = self._scaler.transform(X)
        proba_raw = self._gmm.predict_proba(X_scaled)          # (m, n)
        proba_sorted = proba_raw[:, self._sort_order]           # reordenar columnas

        # Construir DataFrame sobre el índice de las barras con features completas
        feat_df = pd.DataFrame(index=regime_feats.index)
        for k in range(n):
            feat_df[f"regime_prob_{k}"] = proba_sorted[:, k]
        feat_df["regime_label"] = proba_sorted.argmax(axis=1).astype(int)

        eps = 1e-10
        entropy = -(proba_sorted * np.log(proba_sorted + eps)).sum(axis=1)
        feat_df["regime_entropy"] = entropy

        # Reindexar al índice completo; rellenar NaN del warm-up con neutras
        result = feat_df.reindex(full_index)
        for k in range(n):
            result[f"regime_prob_{k}"] = result[f"regime_prob_{k}"].fillna(1.0 / n)
        result["regime_label"]   = result["regime_label"].fillna(0).astype(int)
        result["regime_entropy"] = result["regime_entropy"].fillna(float(np.log(n)))

        return result

    def fit_transform(self, close: pd.Series, atr: pd.Series) -> pd.DataFrame:
        """Ajusta y transforma en un paso. Útil para análisis offline."""
        return self.fit(close, atr).transform(close, atr)

    # ------------------------------------------------------------------
    # ANÁLISIS / INTERPRETACIÓN
    # ------------------------------------------------------------------

    @property
    def n_components_(self) -> int:
        return self.cfg.n_components

    def regime_summary(self, close: pd.Series, atr: pd.Series) -> pd.DataFrame:
        """
        Resumen estadístico por régimen: tiempo %, vol media, sharpe, autocorr.
        Útil para etiquetar manualmente los regímenes (trending/ranging/crisis).
        """
        if not self.is_fitted:
            return pd.DataFrame()

        regime_feats = self._build_regime_features(close, atr)
        if regime_feats.empty:
            return pd.DataFrame()

        X_scaled = self._scaler.transform(regime_feats.values)
        raw_labels = self._gmm.predict(X_scaled)

        # Reordenar labels según sort_order
        inv_order = np.argsort(self._sort_order)  # mapeo inverso
        sorted_labels = inv_order[raw_labels]

        rows = []
        n = self.cfg.n_components
        for k in range(n):
            mask = sorted_labels == k
            if mask.sum() == 0:
                continue
            sub = regime_feats.iloc[mask]
            rows.append({
                "Regime":       k,
                "N_bars":       int(mask.sum()),
                "Pct_time":     f"{mask.mean():.1%}",
                "Vol_mean":     round(float(sub["vol_20"].mean()), 5),
                "Sharpe_roll":  round(float(sub["sharpe_roll"].mean()), 3),
                "Autocorr":     round(float(sub["autocorr_1"].mean()), 3),
                "ATR_z":        round(float(sub["atr_z"].mean()), 3),
                "Label":        _auto_label(
                    sub["vol_20"].mean(),
                    sub["sharpe_roll"].mean(),
                    sub["autocorr_1"].mean(),
                ),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # HELPERS PRIVADOS
    # ------------------------------------------------------------------

    @staticmethod
    def _neutral_features(index: pd.Index, n: int) -> pd.DataFrame:
        """DataFrame de features neutras (probabilidades uniformes)."""
        data = {f"regime_prob_{k}": np.full(len(index), 1.0 / n) for k in range(n)}
        data["regime_label"]   = np.zeros(len(index), dtype=int)
        data["regime_entropy"] = np.full(len(index), float(np.log(n)))
        return pd.DataFrame(data, index=index)


# =====================================================================
# UTILIDAD OFFLINE: selección automática de n_components vía BIC
# =====================================================================

def select_n_components(
    close: pd.Series,
    atr: pd.Series,
    max_n: int = 6,
    covariance_type: str = "diag",
    n_init: int = 5,
    random_state: int = 42,
) -> int:
    """
    Elige el número óptimo de componentes GMM por BIC (Bayesian Information Criterion).

    Uso offline (UNA SOLA VEZ sobre el dataset completo) para determinar
    el valor de `n_components` a usar en WalkForwardConfig.

    El BIC penaliza la complejidad del modelo: mínimo BIC = mejor trade-off
    bias-varianza para el GMM.

    Parameters
    ----------
    close, atr : Series   Dataset completo (o una muestra representativa).
    max_n      : int      Número máximo de componentes a evaluar (2..max_n).

    Returns
    -------
    int: mejor n_components según BIC.
    """
    feats = GMMRegimeDetector._build_regime_features(close, atr)
    if feats.empty or len(feats) < 50:
        logger.warning("select_n_components: pocos datos, devolviendo n=3.")
        return 3

    scaler = StandardScaler()
    X = scaler.fit_transform(feats.values)

    best_n, best_bic = 3, np.inf
    bic_scores = {}
    for n in range(2, max_n + 1):
        try:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=covariance_type,
                n_init=n_init,
                random_state=random_state,
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            bic_scores[n] = round(bic, 1)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        except Exception as e:
            logger.debug("select_n_components n=%d falló: %s", n, e)

    logger.info("BIC por n_components: %s -> mejor=%d", bic_scores, best_n)
    return best_n


# =====================================================================
# HELPER: etiqueta automática de régimen
# =====================================================================

def _auto_label(vol: float, sharpe: float, autocorr: float) -> str:
    if vol > 0.009:
        return "Volatil/Crisis"
    elif abs(sharpe) > 0.3 and autocorr > 0.05:
        return "Tendencia" + (" alcista" if sharpe > 0 else " bajista")
    elif autocorr < -0.05:
        return "Media-reversion"
    else:
        return "Lateral/Rango"
