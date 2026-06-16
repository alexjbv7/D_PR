"""
StackedPositionSizer — sizing por posición (DIAGNOSTICO §4, contrato ADR-042).

Implementa ``quant_shared.contracts.PositionSizer`` componiendo las piezas
existentes de ``research/risk`` (§20.2 — composición, no reimplementación):

- ``risk.kelly.kelly_fraction_binary`` — Kelly fraccional con cap (ADR-003).
- ``risk.dynamic_rr.compute_dynamic_rr`` — R:R dinámico desde p_win, que se
  alimenta a Kelly (coherencia documentada en ``dynamic_rr.py``).
- El shrink bayesiano del edge consume la incertidumbre del posterior (p.ej.
  un Beta de ``bayesian_sizer``): a más varianza, menos tamaño.

Stack (en este orden):

    vol_scale     = vol_target / max(vol_forecast, eps)        # vol-targeting
    kelly_frac    = frac_kelly(p_eff, rr_dyn, cap)             # quarter-Kelly
    regime_factor = 1.0 normal | 0.5 alta vol | 0.0 defensivo  # atenuación
    raw           = vol_scale * kelly_frac * regime_factor
    target_weight = sign(weight) * min(|raw|, max_position_cap)
    + CVaR-lite   : si ES_posición > presupuesto → reducir hasta cumplir

donde ``p_eff = clip(E[p] − shrink_z·std[p], 0, 1)`` es el edge encogido por
incertidumbre (Kelly bayesiano conservador).

Límites de responsabilidad (ADR-042 §3.1, ADR-009)
--------------------------------------------------
El sizer DIMENSIONA una posición ya direccionada por el allocator/agente.
NUNCA decide límites de firma, caps por sector ni kill switch — eso es
``RiskGate``. El CVaR de cartera con correlaciones es del ``CapitalAllocator``;
aquí solo la restricción CVaR-lite POR POSICIÓN (gaussiana sobre
``vol_forecast``).

Convención de unidades: ``vol_target``, ``vol_forecast`` y ``cvar_budget``
comparten base temporal (p.ej. todo anualizado). El ratio del vol-targeting es
invariante; el presupuesto CVaR se interpreta en esa misma base.
"""
from __future__ import annotations

import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.stats import norm

from quant_shared.contracts import SizeDecision
from quant_shared.schemas.signals import TradeSignal, require_calibrated_signal
from risk.dynamic_rr import compute_dynamic_rr
from risk.kelly import kelly_fraction_binary

_METHOD = "vol_target·frac_kelly·regime·cvar"


def edge_posterior_from_signal(signal: TradeSignal) -> float:
    """
    Convierte una ``TradeSignal`` en el ``edge_posterior`` para ``size()``,
    RECHAZANDO señales con ``p_win`` sin calibrar (arbitraje D / R-02.b).

    Es el ÚNICO punto autorizado para alimentar Kelly desde una señal: pone el
    "seguro" al cable ``p_win``→Kelly. Si la señal no fue calibrada OOS lanza
    ``UncalibratedSignalError`` en vez de propagar un softmax crudo al sizing de
    capital. Para sizing sin Kelly (vol-target puro) no se usa esta función.

    Parameters
    ----------
    signal : TradeSignal
        Señal con ``p_win`` y ``p_win_calibrated``.

    Returns
    -------
    float
        ``signal.p_win`` (ya calibrado) listo para ``size(edge_posterior=...)``.
    """
    require_calibrated_signal(signal)
    return float(signal.p_win)


@dataclass(frozen=True)
class SizerConfig:
    """
    Parámetros del stack de sizing — config con defaults, sin magic numbers
    (§20.6). Todos los términos del stack son ajustables por instancia.
    """

    # Vol-targeting (base robusta)
    vol_target: float = 0.15            # vol objetivo de la posición (anualizada)
    vol_eps: float = 1e-6               # piso numérico del forecast

    # Kelly fraccional bayesiano
    kelly_cap: float = 0.25             # quarter-Kelly; nunca full (ADR-003)
    shrink_z: float = 1.0               # nº de std restadas al edge (shrink bayesiano)

    # R:R dinámico (alimenta a Kelly — risk/dynamic_rr.py)
    rr_min: float = 1.2
    rr_max: float = 2.5
    p_low: float = 0.45
    p_high: float = 0.75
    rr_shape: Literal["linear", "sigmoid", "stepped"] = "sigmoid"

    # Atenuación por régimen (factor 0 = no operar)
    regime_factors: Mapping[str, float] = field(
        default_factory=lambda: {"normal": 1.0, "high_vol": 0.5, "defensive": 0.0}
    )
    unknown_regime_factor: float = 1.0  # régimen no mapeado → tratar como normal

    # Caps por posición
    max_position_cap: float = 0.20      # |peso| máximo por posición

    # CVaR-lite por posición (gaussiano)
    cvar_confidence: float = 0.95
    cvar_budget: float = 0.10           # ES máximo por posición (misma base que vol)


class StackedPositionSizer:
    """
    ``PositionSizer`` del stack del diagnóstico §4.

    Examples
    --------
    >>> sizer = StackedPositionSizer()
    >>> decision = sizer.size(
    ...     weight=0.5, vol_forecast=0.20, regime="normal", edge_posterior=0.62
    ... )
    >>> abs(decision.target_weight) <= 0.20
    True
    """

    def __init__(self, config: SizerConfig | None = None) -> None:
        self.config = config or SizerConfig()

    def size(
        self,
        weight: float,
        vol_forecast: float,
        regime: str,
        edge_posterior: Any,
    ) -> SizeDecision:
        """
        Dimensiona una posición. El signo de ``weight`` fija la dirección
        (nunca se invierte); su magnitud es del allocator y aquí solo aporta
        el signo (stack §4).

        Parameters
        ----------
        weight : float
            Peso direccionado por el allocator; signo = dirección.
        vol_forecast : float
            Volatilidad esperada de la posición (>= 0, misma base que
            ``vol_target``).
        regime : str
            Etiqueta de régimen ("normal" | "high_vol" | "defensive" | otra).
        edge_posterior : float | (mean, std) | objeto con .mean()/.std()
            Posterior de P(win): punto, tupla, o distribución (p.ej.
            ``scipy.stats.beta`` ajustada desde ``bayesian_sizer``). Mayor
            incertidumbre → edge efectivo menor → menos tamaño.
            **DEBE provenir de una fuente CALIBRADA** (arbitraje D / R-02): cuando
            el edge venga de una ``TradeSignal``, constrúyelo con
            ``edge_posterior_from_signal(signal)``, que rechaza ``p_win`` sin
            calibrar. Nunca pasar aquí un softmax de Q-values crudo.

        Returns
        -------
        SizeDecision
            ``target_weight`` con el signo de ``weight``; ``reason`` explica
            qué término del stack mandó.
        """
        cfg = self.config
        if not np.isfinite(vol_forecast) or vol_forecast < 0.0:
            raise ValueError(f"vol_forecast inválido: {vol_forecast}")

        if weight == 0.0:
            return SizeDecision(
                target_weight=0.0, method=_METHOD,
                reason="flat: weight=0 — sin dirección del allocator",
            )

        # Edge bayesiano: encoger por incertidumbre antes de Kelly
        p_mean, p_std = _edge_stats(edge_posterior)
        p_eff = float(np.clip(p_mean - cfg.shrink_z * p_std, 0.0, 1.0))

        rr_dyn = compute_dynamic_rr(
            p_eff, rr_min=cfg.rr_min, rr_max=cfg.rr_max,
            p_low=cfg.p_low, p_high=cfg.p_high, shape=cfg.rr_shape,
        )
        kelly = kelly_fraction_binary(p_eff, rr_dyn, kelly_fraction=cfg.kelly_cap)
        if kelly <= 0.0:
            return SizeDecision(
                target_weight=0.0, method=_METHOD,
                reason=(
                    f"kelly: sin edge tras shrink bayesiano "
                    f"(p_mean={p_mean:.3f}, p_std={p_std:.3f}, p_eff={p_eff:.3f}, "
                    f"rr={rr_dyn:.2f})"
                ),
            )

        regime_factor = float(cfg.regime_factors.get(regime, cfg.unknown_regime_factor))
        if regime_factor <= 0.0:
            return SizeDecision(
                target_weight=0.0, method=_METHOD,
                reason=f"regime: '{regime}' factor=0 — modo defensivo, no operar",
            )

        vol_scale = cfg.vol_target / max(vol_forecast, cfg.vol_eps)
        raw = vol_scale * kelly * regime_factor

        magnitude = abs(raw)
        binding = ""
        if magnitude > cfg.max_position_cap:
            magnitude = cfg.max_position_cap
            binding = "max_position_cap"

        # CVaR-lite por posición: ES(|w|) = |w| · vol · φ(z_α)/(1−α)
        es_unit = _expected_shortfall_unit(vol_forecast, cfg.cvar_confidence)
        if magnitude * es_unit > cfg.cvar_budget:
            magnitude = cfg.cvar_budget / max(es_unit, cfg.vol_eps)
            binding = "cvar"

        if not binding:
            # Sin clip: manda el multiplicador que más atenúa
            terms = {"vol_target": vol_scale, "kelly": kelly, "regime": regime_factor}
            binding = min(terms, key=terms.__getitem__)

        target = float(np.sign(weight) * magnitude)
        return SizeDecision(
            target_weight=target,
            method=_METHOD,
            reason=(
                f"{binding} manda | vol_scale={vol_scale:.3f} "
                f"kelly={kelly:.4f} (p_eff={p_eff:.3f}, rr={rr_dyn:.2f}) "
                f"regime={regime}:{regime_factor:.2f} raw={raw:.4f} → "
                f"|w|={magnitude:.4f}, ES={magnitude * es_unit:.4f} ≤ "
                f"budget={cfg.cvar_budget}"
            ),
        )


def _edge_stats(edge_posterior: Any) -> tuple[float, float]:
    """
    Extrae (media, std) del posterior de P(win).

    Acepta: un float (estimación puntual, std=0), una tupla ``(mean, std)``,
    o un objeto distribución con ``.mean()``/``.std()`` invocables (p.ej.
    ``scipy.stats.beta(a, b)`` construida desde los conteos de
    ``bayesian_sizer``).
    """
    if isinstance(edge_posterior, numbers.Real):
        return float(edge_posterior), 0.0
    if isinstance(edge_posterior, (tuple, list)) and len(edge_posterior) == 2:
        return float(edge_posterior[0]), float(edge_posterior[1])
    mean = getattr(edge_posterior, "mean", None)
    std = getattr(edge_posterior, "std", None)
    if callable(mean) and callable(std):
        return float(mean()), float(std())
    raise TypeError(
        f"edge_posterior no soportado: {type(edge_posterior).__name__}. "
        f"Usa float, (mean, std) o un objeto con .mean()/.std()."
    )


def _expected_shortfall_unit(vol: float, confidence: float) -> float:
    """
    Expected shortfall gaussiano por unidad de peso: ``vol·φ(z_α)/(1−α)``.

    Misma base temporal que ``vol``. Aproximación CVaR-lite por posición; el
    CVaR de cartera con correlaciones es del ``CapitalAllocator``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"cvar_confidence debe estar en (0,1): {confidence}")
    z = norm.ppf(confidence)
    return float(vol * norm.pdf(z) / (1.0 - confidence))
