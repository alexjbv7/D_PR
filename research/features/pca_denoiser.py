"""
PCA Denoising de Features
==========================
Reduce la dimensionalidad de las features de trading via PCA (Análisis de
Componentes Principales) para:

  1. Eliminar colinealidad: ret_1/ret_5/ret_10/ret_20 y vol_10/vol_20
     están altamente correlacionados → los componentes son ortogonales.
  2. Reducir overfitting: menos dimensiones = menos parámetros en XGBoost.
  3. Filtrar ruido: los últimos componentes capturan principalmente ruido;
     al descartarlos queda la señal estructural.

Referencia: López de Prado (AFML, cap. 2) recomienda explícitamente PCA
como paso de preprocesamiento para features financieras tabulares.

PROTOCOLO ANTI-LEAKAGE:
  - PCADenoiser.fit() se llama SOLO sobre X_fit (porción de entrenamiento)
  - StandardScaler también se ajusta solo en X_fit
  - transform() se aplica por separado a X_calib y X_test

Columnas excluidas del PCA:
  Las columnas con el prefijo `exclude_prefix` (default "regime_") pasan
  directamente sin transformar, ya que son probabilidades interpretables
  de escala distinta.

Modos de selección de n_components:
  float en (0, 1) : umbral de varianza explicada  (ej. 0.95 = 95%)
  int             : número fijo de componentes
  "mle"           : estimación por MLE de Minka (automático)

Uso típico:
    denoiser = PCADenoiser(PCAConfig(n_components=0.95))
    denoiser.fit(X_fit)
    X_fit_pca   = denoiser.transform(X_fit)
    X_calib_pca = denoiser.transform(X_calib)
    X_test_pca  = denoiser.transform(X_test)

Uso offline:
    denoiser = PCADenoiser()
    X_pca = denoiser.fit_transform(X)
    print(denoiser.scree_summary())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURACIÓN
# =====================================================================

@dataclass
class PCAConfig:
    """
    Configuración del PCA Denoiser.

    n_components     : número de componentes a retener.
                       float (0,1) → umbral de varianza explicada acumulada.
                       int         → número fijo.
                       "mle"       → Minka's MLE (automático).
    scale            : aplicar StandardScaler antes de PCA (recomendado: True).
    whiten           : normalizar varianza de los componentes (False por defecto).
    exclude_prefix   : columnas cuyo nombre empiece con este prefijo se excluyen
                       del PCA y se concatenan tal cual al resultado.
    min_components   : número mínimo de componentes (evita colapsar a 1 o 2).
    """
    n_components: Union[int, float, str] = 0.95
    scale: bool = True
    whiten: bool = False
    exclude_prefix: str = "regime_"
    min_components: int = 3


# =====================================================================
# DENOISER
# =====================================================================

class PCADenoiser:
    """
    Wrapper de sklearn.PCA con soporte para:
      - Exclusión automática de columnas de régimen
      - StandardScaler integrado (ajustado solo en train)
      - API consistente con GMMRegimeDetector (fit / transform / fit_transform)
      - Diagnósticos: scree_summary(), loadings(), n_components_

    Ejemplo dentro de WalkForwardRunner (step 1c, tras añadir régimen):
        denoiser = PCADenoiser(PCAConfig(n_components=0.95))
        denoiser.fit(X_fit)
        X_fit   = denoiser.transform(X_fit)
        X_calib = denoiser.transform(X_calib)
        X_test  = denoiser.transform(X_test)
    """

    def __init__(self, config: Optional[PCAConfig] = None):
        self.cfg = config or PCAConfig()
        self._pca: Optional[PCA] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_cols: list = []    # columnas que entran al PCA
        self._excluded_cols: list = []   # columnas que se pasan directo
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "PCADenoiser":
        """
        Ajusta scaler + PCA sobre las columnas no excluidas de X.

        Parameters
        ----------
        X : pd.DataFrame
            Features de TRAIN (X_fit solamente, sin calib ni test).
        """
        cfg = self.cfg

        # Separar columnas excluidas
        self._excluded_cols = [
            c for c in X.columns if c.startswith(cfg.exclude_prefix)
        ]
        self._feature_cols = [
            c for c in X.columns if c not in self._excluded_cols
        ]

        if not self._feature_cols:
            logger.warning("PCADenoiser.fit: no hay columnas para transformar.")
            return self

        X_feat = X[self._feature_cols].values.astype(float)

        # Escalar (ajustado solo sobre train)
        if cfg.scale:
            self._scaler = StandardScaler()
            X_feat = self._scaler.fit_transform(X_feat)

        # Determinar n_components efectivo
        n_samples, n_features = X_feat.shape
        n_comp = cfg.n_components
        if isinstance(n_comp, int):
            n_comp = min(n_comp, n_features, n_samples)
            n_comp = max(n_comp, cfg.min_components)
        # Para float y "mle" sklearn lo maneja internamente

        self._pca = PCA(n_components=n_comp, whiten=cfg.whiten, random_state=42)
        self._pca.fit(X_feat)

        # Aplicar min_components si el resultado tiene menos del mínimo.
        # Usamos self._pca.n_components_ directamente (no la property, que
        # requiere is_fitted=True y aún no está marcado en este punto).
        actual_n = int(self._pca.n_components_)
        if actual_n < cfg.min_components:
            n_comp_new = min(cfg.min_components, n_features, n_samples)
            self._pca = PCA(
                n_components=n_comp_new, whiten=cfg.whiten, random_state=42
            )
            self._pca.fit(X_feat)

        self.is_fitted = True
        logger.debug(
            "PCADenoiser fit: %d -> %d componentes | varianza explicada acum.=%.3f",
            len(self._feature_cols),
            self.n_components_,
            self.explained_variance_ratio_.sum(),
        )
        return self

    # ------------------------------------------------------------------
    # TRANSFORM
    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Proyecta X al espacio PCA.

        Las columnas excluidas se mantienen intactas y se concatenan al final.
        El índice original se preserva.

        Returns
        -------
        pd.DataFrame con columnas pca_0..pca_{n-1} + columnas excluidas.
        """
        if not self.is_fitted:
            logger.warning("PCADenoiser.transform llamado sin fit(). Devolviendo X original.")
            return X

        X_feat = X[self._feature_cols].values.astype(float)

        if self._scaler is not None:
            X_feat = self._scaler.transform(X_feat)

        X_pca = self._pca.transform(X_feat)

        pca_df = pd.DataFrame(
            X_pca,
            index=X.index,
            columns=[f"pca_{k}" for k in range(X_pca.shape[1])],
        )

        if self._excluded_cols:
            excl_df = X[self._excluded_cols].reindex(X.index)
            return pd.concat([pca_df, excl_df], axis=1)

        return pca_df

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ajusta y transforma en un paso. Útil para análisis offline."""
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # DIAGNÓSTICOS
    # ------------------------------------------------------------------

    @property
    def n_components_(self) -> int:
        """Número de componentes efectivamente usados."""
        if not self.is_fitted or self._pca is None:
            return 0
        return int(self._pca.n_components_)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Varianza explicada por componente."""
        if not self.is_fitted:
            return np.array([])
        return self._pca.explained_variance_ratio_

    @property
    def cumulative_variance_(self) -> float:
        """Varianza total explicada por los n componentes retenidos."""
        if not self.is_fitted:
            return 0.0
        return float(self._pca.explained_variance_ratio_.sum())

    def loadings(self) -> pd.DataFrame:
        """
        Matriz de loadings: contribución de cada feature original a cada PC.

        Rows = features originales, Cols = PC1..PCn.
        Valores altos (abs) → feature tiene fuerte contribución a ese componente.
        """
        if not self.is_fitted:
            return pd.DataFrame()
        return pd.DataFrame(
            self._pca.components_.T,
            index=self._feature_cols,
            columns=[f"PC{k + 1}" for k in range(self.n_components_)],
        )

    def top_loadings(self, pc: int = 1, top_n: int = 5) -> pd.Series:
        """
        Top N features con mayor contribución (abs) al componente `pc` (1-indexed).

        Útil para entender QUÉ captura cada componente principal.
        """
        ld = self.loadings()
        col = f"PC{pc}"
        if col not in ld.columns:
            return pd.Series(dtype=float)
        return ld[col].abs().sort_values(ascending=False).head(top_n)

    def scree_summary(self) -> pd.DataFrame:
        """
        Tabla scree: componente | varianza explicada | varianza acumulada.

        Usar para decidir el n_components óptimo (buscar el "codo").
        """
        if not self.is_fitted:
            return pd.DataFrame()
        evr = self.explained_variance_ratio_
        return pd.DataFrame({
            "Componente": range(1, len(evr) + 1),
            "Var. explicada": np.round(evr, 4),
            "Var. acumulada": np.round(np.cumsum(evr), 4),
        })

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"PCADenoiser(n_components={self.cfg.n_components}, not fitted)"
        return (
            f"PCADenoiser("
            f"{len(self._feature_cols)} features -> {self.n_components_} PCs, "
            f"var_explicada={self.cumulative_variance_:.1%})"
        )
