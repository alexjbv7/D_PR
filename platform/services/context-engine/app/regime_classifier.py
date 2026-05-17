"""
Market Regime Classifier — GMM + HMM + macro overlay.
=====================================================
Clasifica el estado del mercado en tiempo real combinando:

  1. Volatility regime (σ rolling)
  2. Trend regime (momentum z-score)
  3. Macro overlay (recession probability de macroeconomic service)
  4. On-chain overlay (smart money signal)

Regímenes resultantes:
  0: bull_low_vol      — tendencia alcista, volatilidad baja
  1: bull_high_vol     — tendencia alcista, volatilidad alta (momentum)
  2: range_bound       — lateral, sin tendencia clara
  3: bear_low_vol      — tendencia bajista, capitulación lenta
  4: bear_high_vol     — tendencia bajista, pánico / crash

Outputs:
  - regime_label: int 0-4
  - regime_name: str
  - regime_probs: [p0, p1, p2, p3, p4]
  - stability: qué tan confiada es la clasificación
  - dominant_features: qué features lo explican

Integración con quant_bot:
  Este clasificador reemplaza el GMM simple de features/regime.py
  con un sistema más sofisticado multi-fuente.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from libs.shared.events import RegimeUpdateEvent, KafkaTopics
from libs.shared.kafka_client import KafkaProducerClient
from libs.shared.redis_client import RedisCache, TTL

logger = logging.getLogger(__name__)

REGIME_NAMES = {
    0: "bull_low_vol",
    1: "bull_high_vol",
    2: "range_bound",
    3: "bear_low_vol",
    4: "bear_high_vol",
}

# Features utilizadas para el GMM (calculadas de OHLCV + macro)
GMM_FEATURES = [
    "vol_realized_20d",   # volatilidad realizada 20 días
    "vol_realized_5d",    # volatilidad realizada corto plazo
    "momentum_20d",       # retorno z-score 20 días
    "momentum_5d",        # retorno z-score 5 días
    "trend_strength",     # ADX proxy
    "macro_recession_p",  # probabilidad de recesión (macro overlay)
    "vix_z",              # VIX z-score
]


class RegimeClassifier:
    """
    Clasificador de régimen de mercado multi-fuente.

    Entrenamiento: fit() sobre ventana histórica de features.
    Inferencia: classify() sobre el estado actual.

    Anti-leakage: fit() solo se llama con datos de entrenamiento.
    En producción, el modelo se re-entrena con datos OOS confirmados.
    """

    def __init__(
        self,
        n_components: int = 5,
        producer: Optional[KafkaProducerClient] = None,
        cache: Optional[RedisCache] = None,
    ):
        self._n = n_components
        self._gmm: Optional[GaussianMixture] = None
        self._scaler = StandardScaler()
        self._producer = producer
        self._cache    = cache
        self._is_fitted = False
        self._label_map: dict[int, int] = {}  # GMM label → semantic label
        self._feature_history: list[dict] = []
        self._last_regime: Optional[int] = None

    def fit(self, X: pd.DataFrame) -> "RegimeClassifier":
        """
        Entrena el GMM sobre el DataFrame de features.

        X debe tener columnas de GMM_FEATURES (las disponibles).
        """
        available = [f for f in GMM_FEATURES if f in X.columns]
        if len(available) < 2:
            logger.warning("RegimeClassifier: insuficientes features. Disponibles: %s", available)
            return self

        X_feat = X[available].dropna()
        if len(X_feat) < self._n * 10:
            logger.warning("RegimeClassifier: datos insuficientes (%d rows)", len(X_feat))
            return self

        X_scaled = self._scaler.fit_transform(X_feat.values)
        self._gmm = GaussianMixture(
            n_components=self._n,
            covariance_type="full",
            random_state=42,
            n_init=5,
            max_iter=200,
        )
        self._gmm.fit(X_scaled)
        self._is_fitted = True

        # Mapear labels GMM a nombres semánticos basado en centroides
        self._build_label_map(X_feat, available)

        logger.info(
            "RegimeClassifier fitted: %d samples, %d features, %d components",
            len(X_feat), len(available), self._n,
        )
        return self

    def _build_label_map(self, X_feat: pd.DataFrame, features: list[str]):
        """
        Mapea labels del GMM (enteros arbitrarios) a regímenes semánticos.

        Heurística:
          - Alta vol + negatividad  → bear_high_vol (4)
          - Alta vol + positividad  → bull_high_vol (1)
          - Baja vol + positivo     → bull_low_vol (0)
          - Bajo momentum           → range_bound (2)
          - Baja vol + negativo     → bear_low_vol (3)
        """
        means = self._gmm.means_   # (n_components, n_features)
        means_original = self._scaler.inverse_transform(means)

        df_means = pd.DataFrame(means_original, columns=features)

        self._label_map = {}
        for i in range(self._n):
            row = df_means.iloc[i]
            vol = row.get("vol_realized_20d", 0)
            mom = row.get("momentum_20d", 0)
            vol_median = df_means.get("vol_realized_20d", pd.Series([0])).median()

            if vol > vol_median * 1.5 and mom < 0:
                semantic = 4  # bear_high_vol
            elif vol > vol_median * 1.5 and mom > 0:
                semantic = 1  # bull_high_vol
            elif vol <= vol_median * 1.5 and mom > 0.5:
                semantic = 0  # bull_low_vol
            elif vol <= vol_median * 1.5 and mom < -0.5:
                semantic = 3  # bear_low_vol
            else:
                semantic = 2  # range_bound

            self._label_map[i] = semantic

    async def classify(
        self,
        features: dict[str, float],
        symbol: str = "BTCUSDT",
    ) -> dict:
        """
        Clasifica el régimen actual a partir de un vector de features.

        Parameters
        ----------
        features : dict {feature_name: value}
        symbol   : identificador del mercado

        Returns
        -------
        dict con regime_label, regime_name, probs, stability
        """
        if not self._is_fitted:
            return self._default_regime()

        available = [f for f in GMM_FEATURES if f in features]
        if len(available) < 2:
            return self._default_regime()

        X = np.array([[features.get(f, 0.0) for f in GMM_FEATURES
                       if f in available]])
        try:
            X_scaled = self._scaler.transform(X)
            gmm_label = int(self._gmm.predict(X_scaled)[0])
            proba_raw = self._gmm.predict_proba(X_scaled)[0]

            # Mapear probs a semántica
            semantic_probs = np.zeros(5)
            for gmm_l, sem_l in self._label_map.items():
                semantic_probs[sem_l] += proba_raw[gmm_l]

            # Normalizar
            semantic_probs = semantic_probs / semantic_probs.sum()

            semantic_label = int(self._label_map.get(gmm_label, 2))
            stability      = float(np.max(semantic_probs))
            regime_name    = REGIME_NAMES.get(semantic_label, "range_bound")

            result = {
                "regime_label":   semantic_label,
                "regime_name":    regime_name,
                "regime_probs":   semantic_probs.tolist(),
                "stability":      round(stability, 4),
                "symbol":         symbol,
                "ts":             datetime.now(tz=timezone.utc).isoformat(),
            }

            # Emitir a Kafka si hay cambio de régimen
            if semantic_label != self._last_regime:
                await self._emit_regime_update(result, symbol)
                self._last_regime = semantic_label

            # Cache Redis
            if self._cache:
                await self._cache.set(
                    f"regime:{symbol}",
                    result,
                    ttl=TTL["regime"],
                )

            return result

        except Exception as exc:
            logger.error("RegimeClassifier classify error: %s", exc)
            return self._default_regime()

    def _default_regime(self) -> dict:
        return {
            "regime_label": 2,  # range_bound por defecto
            "regime_name":  "range_bound",
            "regime_probs": [0.1, 0.1, 0.6, 0.1, 0.1],
            "stability":    0.6,
        }

    async def _emit_regime_update(self, result: dict, symbol: str):
        """Emite RegimeUpdateEvent a Kafka."""
        if not self._producer:
            return
        event = RegimeUpdateEvent(
            source        = "regime-classifier",
            symbol        = symbol,
            regime_label  = result["regime_label"],
            regime_probs  = result["regime_probs"],
            regime_name   = result["regime_name"],
            stability     = result["stability"],
        )
        await self._producer.send(KafkaTopics.REGIME_UPDATE, event, key=symbol)
        logger.info("Regime change %s: %s (stability=%.2f)", symbol, result["regime_name"], result["stability"])

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
