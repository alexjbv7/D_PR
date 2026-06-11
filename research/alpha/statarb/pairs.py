"""
PairStatArb — stat-arb de pares cointegrados, market-neutral (ADR-043).

Primer agente market-neutral del sistema: opera el SPREAD entre dos activos
cointegrados (largo el barato, corto el caro), beta-neutral, medido vs ZERO
(retorno absoluto) — no contra buy-and-hold. Rule-based (cointegración +
z-score), NO DRL (ADR-043 §2; diagnóstico F.4).

Reusa (§20.2): ``statsmodels`` (Engle-Granger ``coint``, OLS) para
cointegración/hedge ratio; ``models.validation.WalkForwardSplitter`` para los
folds; ``deflated_sharpe_ratio``/``probabilistic_sharpe_ratio`` vía la
extensión ``evaluate_zero_gate`` de ``models.drl.dsr_gate``;
``data.drl_dataset.fetch_ohlcv_frame`` para traer las patas reales.

ANTI-LEAKAGE (ADR-043 §4 — no negociable)
-----------------------------------------
Por fold, ``fit`` ajusta SOLO sobre el train: test de cointegración, hedge
ratio β (OLS), media/desv del spread (z-score) y half-life. En test esos
parámetros se APLICAN congelados (``signals``/``returns`` no re-estiman
nada). Embargo ≥ 60 barras, validado en ``walk_forward_pair_returns``.
Ajustar β/mean/std sobre toda la serie hace que el spread "revierta
mágicamente" → edge fantasma. Los helpers ``_coint_pvalue`` / ``_fit_beta``
/ ``_half_life`` son module-level para que el test espía (ADR-040 §5.2
style) verifique qué índices ven.

Retorno market-neutral (ADR-043 §5 — definición única)
------------------------------------------------------
::

    ret_spread_t = (ret_y_t − β·ret_x_t) / (1 + |β|)     # beta-hedged
    r_t          = pos_{t-1} · ret_spread_t − costos_t
    costos_t     = (fee_bps/1e4) · (|Δpos_y_t| + |Δpos_x_t|)
                 = (fee_bps/1e4) · |Δpos_t| · (1 + |β|)   # DOS patas → doble fee

con ``pos ∈ {−1, 0, +1}`` (short/flat/long spread), patas ``pos_y = pos`` y
``pos_x = −β·pos``. Convención temporal ``pos_{t-1}·ret_t`` — la misma del
gate (ADR-040 §3.3): la posición decidida en t cobra el retorno de t+1,
nunca el de la barra que la generó (lookahead).

Alcance (§7): valida el EDGE vía el gate vs ZERO. La integración como
``AlphaAgent``/``TradeSignal`` de 2 patas es follow-up Nivel 2 (el contrato
es por-símbolo; un par necesita ``PairSignal`` o lista de señales).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.drl.dsr_gate import MIN_EMBARGO_BARS
from models.validation import WalkForwardSplitter

logger = logging.getLogger(__name__)


# =====================================================================
# Helpers ajustables SOLO en train (module-level → espiables en tests)
# =====================================================================


def _coint_pvalue(log_y: pd.Series, log_x: pd.Series) -> float:
    """p-value Engle-Granger (``statsmodels.tsa.stattools.coint``) en train."""
    from statsmodels.tsa.stattools import coint

    _, pvalue, _ = coint(log_y.values, log_x.values)
    return float(pvalue)


def _fit_beta(log_y: pd.Series, log_x: pd.Series) -> float:
    """Hedge ratio β: pendiente de OLS ``log(y) ~ const + log(x)`` en train."""
    import statsmodels.api as sm

    X = sm.add_constant(log_x.values)
    model = sm.OLS(log_y.values, X).fit()
    return float(model.params[1])


def _half_life(spread: pd.Series) -> float:
    """
    Half-life de reversión (Ornstein-Uhlenbeck discreto) en train.

    Regresión ``Δs_t = a + b·s_{t−1}``; ``half_life = −ln(2)/b`` si ``b < 0``,
    ``inf`` si no hay reversión (b ≥ 0).
    """
    import statsmodels.api as sm

    s_lag = spread.shift(1).iloc[1:]
    delta = spread.diff().iloc[1:]
    X = sm.add_constant(s_lag.values)
    b = float(sm.OLS(delta.values, X).fit().params[1])
    if b >= 0.0:
        return float("inf")
    return float(-np.log(2.0) / b)


# =====================================================================
# Config y parámetros congelados
# =====================================================================


@dataclass(frozen=True)
class PairStatArbConfig:
    """Parámetros de la estrategia — defaults de ADR-043 §3, sin magic numbers."""

    entry_z: float = 2.0          # |z| de entrada (desviaciones estándar)
    exit_z: float = 0.5           # |z| de cierre (reversión a la media)
    coint_alpha: float = 0.05     # p-value máximo de Engle-Granger
    max_half_life: float = 30.0   # barras; más lento no revierte útilmente
    fee_bps: float = 5.0          # por pata y lado (coste efectivo, como el gate)


@dataclass(frozen=True)
class PairParams:
    """Parámetros ajustados en el TRAIN de un fold; congelados para su test."""

    beta: float
    spread_mean: float
    spread_std: float
    half_life: float
    coint_pvalue: float
    tradeable: bool
    reject_reason: str = ""       # vacío si tradeable=True


# =====================================================================
# Estrategia
# =====================================================================


class PairStatArb:
    """
    Stat-arb de un par cointegrado. Par inicial: SPY/QQQ (ADR-043 §2).

    Parameters
    ----------
    y_col, x_col : str
        Columnas de cierre de las dos patas en el frame de precios
        (``y`` es la pata regresada; ``x`` la de hedge).
    config : PairStatArbConfig, optional

    Examples
    --------
    >>> strategy = PairStatArb("SPY", "QQQ")
    >>> params = strategy.fit(prices.iloc[train_idx])
    >>> positions = strategy.signals(prices.iloc[test_idx], params)
    >>> r = strategy.returns(prices.iloc[test_idx], positions, params)
    """

    def __init__(
        self,
        y_col: str = "SPY",
        x_col: str = "QQQ",
        config: PairStatArbConfig | None = None,
    ) -> None:
        self.y_col = y_col
        self.x_col = x_col
        self.config = config or PairStatArbConfig()

    # ------------------------------------------------------------------
    # FIT — SOLO train (anti-leakage §4)
    # ------------------------------------------------------------------

    def fit(self, train: pd.DataFrame) -> PairParams:
        """
        Ajusta cointegración, β, media/desv del spread y half-life sobre el
        TRAIN de un fold. Cualquier rechazo deja el fold sin operar.
        """
        cfg = self.config
        log_y = np.log(train[self.y_col].astype(float))
        log_x = np.log(train[self.x_col].astype(float))

        pvalue = _coint_pvalue(log_y, log_x)
        beta = _fit_beta(log_y, log_x)
        spread = log_y - beta * log_x
        mean = float(spread.mean())
        std = float(spread.std(ddof=1))
        hl = _half_life(spread)

        if pvalue > cfg.coint_alpha:
            return PairParams(
                beta=beta, spread_mean=mean, spread_std=std,
                half_life=hl, coint_pvalue=pvalue,
                tradeable=False,
                reject_reason=f"not_cointegrated: p={pvalue:.4f} > {cfg.coint_alpha}",
            )
        if not np.isfinite(hl) or hl <= 0.0 or hl > cfg.max_half_life:
            return PairParams(
                beta=beta, spread_mean=mean, spread_std=std,
                half_life=hl, coint_pvalue=pvalue,
                tradeable=False,
                reject_reason=(
                    f"half_life: {hl:.1f} fuera de (0, {cfg.max_half_life}] barras"
                ),
            )
        if std <= 0.0 or not np.isfinite(std):
            return PairParams(
                beta=beta, spread_mean=mean, spread_std=std,
                half_life=hl, coint_pvalue=pvalue,
                tradeable=False,
                reject_reason="degenerate_spread: std no positiva",
            )
        return PairParams(
            beta=beta, spread_mean=mean, spread_std=std,
            half_life=hl, coint_pvalue=pvalue, tradeable=True,
        )

    # ------------------------------------------------------------------
    # SIGNALS — aplica parámetros CONGELADOS sobre test
    # ------------------------------------------------------------------

    def signals(self, test: pd.DataFrame, params: PairParams) -> np.ndarray:
        """
        Posiciones de spread ∈ {−1, 0, +1} sobre el test del fold.

        ``z > +entry`` → spread caro → SHORT spread (short y, long β·x);
        ``z < −entry`` → LONG spread; ``|z| < exit`` → cerrar. No re-estima
        nada: β/mean/std vienen congelados del train (§4).
        """
        n = len(test)
        if not params.tradeable:
            return np.zeros(n, dtype=float)

        cfg = self.config
        log_y = np.log(test[self.y_col].astype(float)).values
        log_x = np.log(test[self.x_col].astype(float)).values
        z = (log_y - params.beta * log_x - params.spread_mean) / params.spread_std

        positions = np.zeros(n, dtype=float)
        pos = 0.0
        for t in range(n):
            if pos == 0.0:
                if z[t] >= cfg.entry_z:
                    pos = -1.0
                elif z[t] <= -cfg.entry_z:
                    pos = 1.0
            elif abs(z[t]) <= cfg.exit_z:
                pos = 0.0
            positions[t] = pos
        return positions

    # ------------------------------------------------------------------
    # RETURNS — definición única §5
    # ------------------------------------------------------------------

    def returns(
        self,
        test: pd.DataFrame,
        positions: np.ndarray,
        params: PairParams,
    ) -> np.ndarray:
        """
        Retornos por barra del spread con doble fee (2 patas), §5.

        ``r_t = pos_{t−1}·ret_spread_t − (fee/1e4)·|Δpos_t|·(1+|β|)`` con
        ``pos_{−1} = 0`` (la primera entrada paga fee). Misma convención
        temporal que ``positions_to_returns`` (ADR-040 §3.3).
        """
        pos = np.asarray(positions, dtype=float)
        n = len(pos)
        if n != len(test):
            raise ValueError(f"positions ({n}) y test ({len(test)}) deben alinear")
        if n == 0:
            return np.empty(0, dtype=float)

        beta = params.beta
        py = test[self.y_col].astype(float).values
        px = test[self.x_col].astype(float).values
        ret_y = np.zeros(n)
        ret_x = np.zeros(n)
        if n > 1:
            ret_y[1:] = py[1:] / py[:-1] - 1.0
            ret_x[1:] = px[1:] / px[:-1] - 1.0
        ret_spread = (ret_y - beta * ret_x) / (1.0 + abs(beta))

        prev = np.concatenate([[0.0], pos[:-1]])
        # Patas: pos_y = pos, pos_x = −β·pos → |Δpos_y|+|Δpos_x| = |Δpos|·(1+|β|)
        leg_turnover = np.abs(pos - prev) * (1.0 + abs(beta))
        fee = self.config.fee_bps / 1e4
        return prev * ret_spread - fee * leg_turnover


# =====================================================================
# Orquestación walk-forward (embargo ≥ 60, ADR-043 §4)
# =====================================================================


def walk_forward_pair_returns(
    prices: pd.DataFrame,
    splitter: WalkForwardSplitter,
    strategy: PairStatArb,
) -> np.ndarray:
    """
    Por fold: ``fit`` en train_k, ``signals``+``returns`` en test_k con los
    parámetros congelados; concatena los retornos OOS en orden de fold.
    Folds rechazados (no cointegra / half-life mala) quedan flat (retorno 0
    — capital ocioso, honesto).

    Parameters
    ----------
    prices : pd.DataFrame
        Cierres de las dos patas (columnas ``strategy.y_col`` /
        ``strategy.x_col``), DatetimeIndex.
    splitter : WalkForwardSplitter
        ``embargo >= MIN_EMBARGO_BARS`` (60) obligatorio — consistente con
        ADR-040; violarlo lanza ``ValueError``.
    strategy : PairStatArb

    Returns
    -------
    np.ndarray
        Retornos OOS concatenados (entrada de ``evaluate_zero_gate``).
    """
    if splitter.embargo < MIN_EMBARGO_BARS:
        raise ValueError(
            f"splitter.embargo={splitter.embargo} < {MIN_EMBARGO_BARS} barras "
            f"(ADR-043 §4, consistente con ADR-040 §4.3)"
        )
    missing = [c for c in (strategy.y_col, strategy.x_col) if c not in prices.columns]
    if missing:
        raise ValueError(f"prices sin columnas de las patas: {missing}")

    out: list[np.ndarray] = []
    n_folds = 0
    for k, (train_idx, test_idx) in enumerate(splitter.split(prices)):
        n_folds += 1
        train = prices.iloc[train_idx]
        test = prices.iloc[test_idx]
        params = strategy.fit(train)
        if not params.tradeable:
            logger.info(
                "pair fold %d: rechazado (%s) — flat", k, params.reject_reason
            )
            out.append(np.zeros(len(test), dtype=float))
            continue
        positions = strategy.signals(test, params)
        r = strategy.returns(test, positions, params)
        logger.info(
            "pair fold %d: beta=%.3f hl=%.1f p=%.4f oos_bars=%d mean_r=%.6f",
            k, params.beta, params.half_life, params.coint_pvalue,
            len(r), float(np.mean(r)),
        )
        out.append(r)

    if n_folds == 0:
        raise ValueError(
            f"splitter no produjo folds sobre {len(prices)} barras "
            f"(train={splitter.train_size}, test={splitter.test_size}, "
            f"embargo={splitter.embargo})"
        )
    return np.concatenate(out)
