"""
E4 — Diagnóstico de potencia estadística de la validación (arbitraje D · Fase 6).

El arbitraje señaló una debilidad fatal compartida por A, B y C: **nadie reportó la
potencia del gate** — ni nº de trials de Optuna, ni longitud efectiva de fold tras
purga/embargo, ni el intervalo de confianza del Sharpe. Sin eso, "el gate hizo su
trabajo" y "el gate no es suficiente" son ambas afirmaciones sin numerador.

Este módulo cuantifica esa potencia:

1. ``sharpe_standard_error`` — SE del Sharpe ANUALIZADO bajo IID (Lo 2002 /
   Mertens 2002): ``SE = sqrt((ppy + 0.5·SR²) / T)``. Con folds OOS cortos el SE es
   enorme → el Sharpe no es interpretable (RC-3).
2. ``sharpe_power_diagnostic`` — ¿el SE permite **distinguir** dos Sharpes de
   referencia (p.ej. 0.3 vs 0.8)? Si no, ningún veredicto de edge (positivo o
   negativo) es estadísticamente defendible.
3. ``effective_fold_lengths`` — N efectivo por fold (barras OOS reales).
4. ``deflation_report`` — recalcula la deflación del DSR con el nº REAL de trials
   (reusa ``walk_forward_runner.deflated_sharpe_ratio`` — §20.2).

El núcleo (1–3) es numpy-only y testeable; (4) importa de forma perezosa.

Referencias: Lo (2002); Mertens (2002); Bailey & López de Prado, *The Deflated
Sharpe Ratio* (2014). (CLAUDE.md Apéndice B.)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_PPY_DAILY = 252
_Z95 = 1.959963984540054   # cuantil normal bilateral 95%


# =====================================================================
# Error estándar del Sharpe (IID, Lo/Mertens)
# =====================================================================

def sharpe_standard_error(
    sharpe_ann: float, n_obs: int, ppy: int = _PPY_DAILY
) -> float:
    """
    SE asintótico del Sharpe ANUALIZADO bajo retornos IID.

    Deriva del SE por período de Lo (2002) ``sqrt((1+0.5·SR_pp²)/T)`` anualizado
    por ``sqrt(ppy)``:

        SE_ann = sqrt((ppy + 0.5 · SR_ann²) / T)

    Parameters
    ----------
    sharpe_ann : float
        Sharpe anualizado estimado.
    n_obs : int
        Nº de observaciones OOS (T) usadas para estimar el Sharpe.
    ppy : int
        Períodos por año (252 para barras diarias).

    Returns
    -------
    float
        Error estándar del Sharpe anualizado. ``inf`` si ``n_obs < 2``.
    """
    if n_obs is None or n_obs < 2:
        return float("inf")
    return float(math.sqrt((ppy + 0.5 * sharpe_ann ** 2) / n_obs))


@dataclass(frozen=True)
class PowerResult:
    sharpe_ann: float
    n_obs: int
    ppy: int
    se_ann: float
    ci95_low: float
    ci95_high: float
    ref_gap: float
    margin_of_error: float          # z·SE
    can_resolve_gap: bool
    verdict: str                    # "POWERED" | "UNDERPOWERED"
    reason: str


def sharpe_power_diagnostic(
    sharpe_ann: float,
    n_obs: int,
    *,
    ppy: int = _PPY_DAILY,
    ref_gap: float = 0.5,           # distinguir 0.3 de 0.8 ≈ gap 0.5
    z: float = _Z95,
) -> PowerResult:
    """
    ¿Tiene la muestra potencia para distinguir Sharpes separados por ``ref_gap``?

    Criterio: dos medias separadas ``ref_gap`` tienen ICs 95% no solapados sólo si
    ``z·SE < ref_gap/2``. Si no se cumple, la validación está **subpotenciada** y el
    veredicto de edge (sea cual sea) no es estadísticamente defendible.
    """
    se = sharpe_standard_error(sharpe_ann, n_obs, ppy)
    moe = z * se
    can = moe < ref_gap / 2.0
    verdict = "POWERED" if can else "UNDERPOWERED"
    reason = (
        f"{verdict}: SE_ann={se:.3f}, margen 95%=±{moe:.3f} sobre Sharpe={sharpe_ann:.3f} "
        f"(T={n_obs}). "
        + (
            f"Resuelve un gap de {ref_gap:.2f} (p.ej. 0.3 vs 0.8)."
            if can
            else f"NO resuelve un gap de {ref_gap:.2f}: el IC ({z:.2f}σ) es más ancho "
                 f"que media separación → no se puede afirmar ni negar edge con rigor."
        )
    )
    return PowerResult(
        sharpe_ann=float(sharpe_ann), n_obs=int(n_obs), ppy=ppy, se_ann=se,
        ci95_low=float(sharpe_ann - moe), ci95_high=float(sharpe_ann + moe),
        ref_gap=ref_gap, margin_of_error=float(moe), can_resolve_gap=bool(can),
        verdict=verdict, reason=reason,
    )


# =====================================================================
# N efectivo por fold
# =====================================================================

@dataclass(frozen=True)
class FoldLengthReport:
    n_folds: int
    per_fold: tuple[int, ...]
    total_oos: int
    min_fold: int
    mean_fold: float


def effective_fold_lengths(fold_test_sizes: Sequence[int]) -> FoldLengthReport:
    """
    Resume el N efectivo OOS dado el tamaño de TEST (tras purga/embargo) por fold.

    ``fold_test_sizes`` son las barras OOS *evaluables* por fold (típicamente
    ``len(test_idx) - 1`` en el gate). Reporta total, mínimo y media — insumos para
    leer ``sharpe_power_diagnostic`` por fold y sobre el OOS concatenado.
    """
    sizes = [int(s) for s in fold_test_sizes]
    if not sizes:
        raise ValueError("fold_test_sizes vacío")
    return FoldLengthReport(
        n_folds=len(sizes),
        per_fold=tuple(sizes),
        total_oos=int(sum(sizes)),
        min_fold=int(min(sizes)),
        mean_fold=float(np.mean(sizes)),
    )


def fold_lengths_from_splitter(raw_ohlcv, splitter) -> list[int]:
    """N efectivo (``len(test_idx)-1``) por fold de un splitter del gate (lazy)."""
    from models.drl.dsr_gate import _validated_folds   # lazy: deps del gate

    return [len(test_idx) - 1 for _, test_idx in _validated_folds(raw_ohlcv, splitter)]


# =====================================================================
# Deflación honesta con el nº REAL de trials (reusa walk_forward_runner)
# =====================================================================

def deflation_report(
    returns: np.ndarray, n_trials: int, ppy: int = _PPY_DAILY
) -> dict:
    """
    Recalcula PSR/DSR con el nº REAL de trials buscados y reporta el "haircut".

    Reusa ``deflated_sharpe_ratio`` / ``probabilistic_sharpe_ratio`` (§20.2). El
    ``n_trials`` honesto = nº de configs/seeds de Optuna realmente evaluados; no
    inflar ni sub-reportar (ADR-040 §6).
    """
    from models.walk_forward_runner import (  # lazy
        deflated_sharpe_ratio,
        probabilistic_sharpe_ratio,
    )

    if n_trials < 1:
        raise ValueError(f"n_trials debe ser >= 1, recibido {n_trials}")
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    psr = probabilistic_sharpe_ratio(r, 0.0, ppy)
    dsr_1 = deflated_sharpe_ratio(r, 1, ppy)
    dsr_n = deflated_sharpe_ratio(r, n_trials, ppy)
    return {
        "n_obs": int(len(r)),
        "n_trials_reported": int(n_trials),
        "psr": float(psr),
        "dsr_n_trials_1": float(dsr_1),
        "dsr_n_trials_real": float(dsr_n),
        "deflation_haircut": float(dsr_1 - dsr_n),   # cuánto baja el DSR por buscar N
    }


def e4_report(
    sharpe_ann: float,
    fold_test_sizes: Sequence[int],
    n_trials: int,
    *,
    returns: Optional[np.ndarray] = None,
    ppy: int = _PPY_DAILY,
    ref_gap: float = 0.5,
) -> dict:
    """
    Reporte E4 consolidado: potencia (SE + resolución), N efectivo y deflación.

    ``returns`` es opcional: si se pasa, añade el reporte de deflación (requiere
    ``walk_forward_runner``). Los componentes de potencia y N efectivo son
    numpy-only.
    """
    folds = effective_fold_lengths(fold_test_sizes)
    power = sharpe_power_diagnostic(
        sharpe_ann, folds.total_oos, ppy=ppy, ref_gap=ref_gap
    )
    out = {
        "fold_lengths": folds,
        "power": power,
        "n_trials_reported": int(n_trials),
    }
    if returns is not None:
        out["deflation"] = deflation_report(returns, n_trials, ppy)
    logger.info("E4: %s | folds=%s", power.reason, folds.per_fold)
    return out
