"""
XgbAlphaAgent — AlphaAgent adapter for the existing XGBoostClassifier (ADR-042).

Wraps a FITTED ``models.zoo.XGBoostClassifier`` behind the ``AlphaAgent``
Protocol. No retraining, no model changes: the adapter only maps
``MarketContext`` → model input and model output → ``TradeSignal``.

Self-contained module (ADR-042 §3.1): declares its own ``AlphaHypothesis``
(falsifiable, with explicit invalidation) and its own ``StrategyConfig`` with
the fee model of ITS market (US equities — taker + slippage + short borrow; a
future crypto-perp module would declare ``FeeModel(funding=True, ...)``) and
its intrinsic stop/target. The agent PROPOSES that intrinsic risk in the
signal; capital sizing and firm-level limits are external
(``PositionSizer`` / ``RiskGate``, ADR-009) — ``kelly_fraction`` and
``size_usd`` are ALWAYS 0.0 here.

Input dependencies (declared, ADR-042 §3.1)
-------------------------------------------
Tabular family: consumes ``MarketContext.features`` only. The exact feature
names are whatever the wrapped model was fitted with (``model.feature_names_``
— for the ADR-040 baseline these are the 21 env-frame features of
``data.drl_dataset``). Missing features raise ``ValueError`` (the adapter
never silently zero-fills its model's input).

Calibration is the model's own concern: ``XGBoostClassifier.predict_proba``
already returns calibrated probabilities when ``.calibrate()`` was called.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from models.zoo import XGBoostClassifier
from quant_shared.contracts import (
    AlphaHypothesis,
    AssetClass,
    Benchmark,
    FeeModel,
    MarketContext,
    StrategyConfig,
    TradingStyle,
)
from quant_shared.schemas.signals import SignalDirection, TradeSignal

#: Falsifiable identity of this module (ADR-042; CATALOGO_ALPHA_HIPOTESIS.md).
XGB_HYPOTHESIS = AlphaHypothesis(
    id="stock.position.xgb_directional",
    asset_class=AssetClass.STOCK,
    style=TradingStyle.POSITION,
    thesis=(
        "Un clasificador XGBoost sobre features técnicos diarios + régimen GMM "
        "predice la dirección de la siguiente barra en equities US con "
        "probabilidad suficiente para batir buy-and-hold neto de costos."
    ),
    horizon_bars=1,  # label = signo de la siguiente barra; la posición persiste vía re-señal diaria
    benchmark=Benchmark.BUY_AND_HOLD,
    invalidation=(
        "DSR deflactado <= 0.4 sobre OOS concatenado del gate ADR-040, o "
        "Sharpe OOS <= Sharpe buy-and-hold en el mismo walk-forward → "
        "hipótesis muerta; el agente no se promueve."
    ),
)

#: Intrinsic strategy parameters + THIS market's fee model (US equities).
XGB_CONFIG = StrategyConfig(
    fees=FeeModel(
        taker_bps=5.0,      # coste efectivo por lado usado en el gate (EnvironmentConfig.fee_bps)
        slippage_bps=2.0,
        borrow_bps=50.0,    # short borrow anualizado, general collateral típico
    ),
    intrinsic_stop_pct=0.02,    # modelo next-bar: stop ceñido
    intrinsic_target_pct=0.04,  # R:R 2:1
    max_holding_bars=20,
    bar_size="1d",
    params={"n_features": 21.0},
)


class XgbAlphaAgent:
    """
    AlphaAgent adapter over a fitted ``XGBoostClassifier``.

    Parameters
    ----------
    model : XGBoostClassifier
        FITTED classifier with labels in {-1, 0, +1} (direction classes).
        Calibrate it before wrapping if calibrated probabilities are required
        (``model.fit(...).calibrate(...)``).
    model_version : str
        Version stamp for traceability (e.g. registry version or artifact
        hash). Travels in ``TradeSignal.model_version``.
    feature_set_hash : str
        Hash of the feature set spec. Default: sha256 over the model's own
        ordered feature names.

    Examples
    --------
    >>> agent = XgbAlphaAgent(fitted_model, model_version="0.1")
    >>> signal = agent.predict(context)   # context.features: name -> value
    """

    hypothesis: AlphaHypothesis = XGB_HYPOTHESIS
    config: StrategyConfig = XGB_CONFIG

    def __init__(
        self,
        model: XGBoostClassifier,
        model_version: str = "",
        feature_set_hash: str = "",
    ) -> None:
        if getattr(model, "model", None) is None or model.feature_names_ is None:
            raise ValueError(
                "XgbAlphaAgent requires a FITTED XGBoostClassifier "
                "(call .fit() before wrapping)"
            )
        self._model = model
        self._model_version = model_version
        self._feature_set_hash = feature_set_hash or _hash_names(model.feature_names_)

    def predict(self, context: MarketContext) -> TradeSignal:
        """
        Map ``context.features`` → model input; winning class → signal.

        ``p_win`` = probability of the winning (argmax) class, calibrated if
        the wrapped model was calibrated. The agent does NOT size capital:
        ``kelly_fraction`` / ``size_usd`` stay 0.0 (ADR-042 §3.1, ADR-009).

        Raises
        ------
        ValueError
            If ``context.features`` is missing any feature the model needs.
        """
        names = self._model.feature_names_
        missing = [f for f in names if f not in context.features]
        if missing:
            raise ValueError(
                f"MarketContext.features missing {len(missing)} feature(s) "
                f"required by {self.hypothesis.id}: {missing[:10]}"
            )

        X = pd.DataFrame([{f: float(context.features[f]) for f in names}])[names]
        proba = self._model.predict_proba(X)[0]
        win_idx = int(proba.argmax())
        label = int(self._model.inv_label_map_[win_idx])
        p_win = float(proba[win_idx])

        return TradeSignal(
            symbol=context.symbol,
            direction=_LABEL_TO_DIRECTION[label],
            p_win=p_win,
            p_win_raw=p_win,  # sin meta-labeler en esta capa
            # Honestidad de calibración (R-02.c): el XGBoostClassifier devuelve
            # proba calibrada solo si se llamó .calibrate(); refleja ese estado real.
            p_win_calibrated=bool(getattr(self._model, "is_calibrated", False)),
            # Regla dura ADR-042 §3.1 / ADR-009: el agente NUNCA dimensiona capital.
            kelly_fraction=0.0,
            size_usd=0.0,
            strategy=self.hypothesis.id,
            model_version=self._model_version,
            feature_set_hash=self._feature_set_hash,
            # Riesgo INTRÍNSECO de la estrategia — el agente PROPONE, la firma dispone.
            stop_loss_pct=self.config.intrinsic_stop_pct,
            take_profit_pct=self.config.intrinsic_target_pct,
        )


_LABEL_TO_DIRECTION: dict[int, SignalDirection] = {
    -1: SignalDirection.SHORT,
    0: SignalDirection.FLAT,
    1: SignalDirection.LONG,
}


def _hash_names(names: list[str]) -> str:
    """Deterministic hash of an ordered feature-name list."""
    return hashlib.sha256(",".join(names).encode("utf-8")).hexdigest()[:16]
