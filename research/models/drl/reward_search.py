"""
Optuna search over MTM reward weights (ADR-041 §5) — cheap→expensive funnel.

SUBORDINATED TO THE STRUCTURAL FIX (§5, non-negotiable sequence)
----------------------------------------------------------------
Do NOT run this search until the MTM reward (``envs.trading_env``,
``reward_mode="mtm"`` with default weights) has been re-gated on SPY and the
gap against buy-and-hold measured. Tuning the weights of a misaligned reward
wastes compute; tuning an aligned one is step 2, only if step 1 is not enough.

Funnel design (§5.2)
--------------------
1. **Cheap proxy**: objective = Sharpe on a validation slice of ONE fold's
   train bars, reduced episodes, ``MedianPruner`` (CLAUDE.md §6.7). Budget
   capped at ``MAX_PROXY_TRIALS`` (50).
2. **Full gate only for finalists**: top-K proxy configs (K <=
   ``MAX_FINALISTS`` = 5) go through the complete walk-forward DSR gate.

Honest DSR deflation (§5.3 — CLAUDE.md §6.10)
---------------------------------------------
``n_trials`` passed to ``evaluate_drl_gate`` must equal the number of configs
evaluated **out-of-sample** (the finalists), NOT the proxy trial count. The
proxy never touches OOS data, so it does not enter the deflation. Use
``honest_gate_n_trials`` to compute it.

Anti-leakage (ADR-041 §6)
-------------------------
The proxy validation slice is carved from the fold's TRAIN bars only, with an
embargo gap, via ``proxy_validation_split``. The test fold used for the final
DSR is never visible to the search objective. Torch-free module: the expensive
evaluation is injected by the caller.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from models.drl.dsr_gate import MIN_EMBARGO_BARS

logger = logging.getLogger(__name__)

#: §5.2 budget caps — exceeding them is a spec violation, not a tunable.
MAX_PROXY_TRIALS: int = 50
MAX_FINALISTS: int = 5

#: Search space over EnvironmentConfig weights (ADR-041 §3/§5; ranges extend
#: ADR-037 §6 — lambda_dd ↔ w_dd, lambda_vol ↔ w_vol, idle_penalty ↔ w_idle).
_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "w_ret": (0.5, 2.0),
    "w_cost": (0.5, 3.0),
    "w_dd": (0.5, 5.0),
    "w_vol": (0.0, 2.0),
    "w_idle": (0.0, 0.01),
}


def proxy_validation_split(
    train_idx: np.ndarray,
    val_frac: float = 0.25,
    embargo: int = MIN_EMBARGO_BARS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split one fold's TRAIN indices into (fit, validation) for the proxy.

    The validation slice is the chronological TAIL of the train bars, with
    ``embargo`` bars dropped between fit and validation (same rationale as the
    walk-forward embargo: rolling features computed near the boundary leak).
    The test fold is, by construction, never part of either output — callers
    must pass train indices only (ADR-041 §6).

    Parameters
    ----------
    train_idx : np.ndarray
        The fold's train indices (chronologically ordered positions).
    val_frac : float
        Fraction of train bars reserved for validation, in (0, 0.5].
    embargo : int
        Bars dropped between fit and validation (>= ``MIN_EMBARGO_BARS``).

    Returns
    -------
    (fit_idx, val_idx) : tuple[np.ndarray, np.ndarray]
        Disjoint subsets of ``train_idx``; ``max(fit) + embargo < min(val)``.

    Raises
    ------
    ValueError
        If the split leaves fewer than 2 bars on either side, or the
        arguments are out of range.
    """
    if not 0.0 < val_frac <= 0.5:
        raise ValueError(f"val_frac must be in (0, 0.5], got {val_frac}")
    if embargo < MIN_EMBARGO_BARS:
        raise ValueError(
            f"embargo {embargo} < MIN_EMBARGO_BARS {MIN_EMBARGO_BARS} (ADR-041 §6)"
        )
    idx = np.sort(np.asarray(train_idx))
    n = len(idx)
    n_val = int(n * val_frac)
    n_fit = n - n_val - embargo
    if n_fit < 2 or n_val < 2:
        raise ValueError(
            f"train fold too small for proxy split: n={n}, "
            f"fit={n_fit}, val={n_val}, embargo={embargo}"
        )
    return idx[:n_fit], idx[n - n_val:]


def suggest_reward_weights(trial: Any) -> dict[str, float]:
    """Sample one weight config from the ADR-041 §5 search space."""
    return {
        name: trial.suggest_float(name, lo, hi)
        for name, (lo, hi) in _SEARCH_SPACE.items()
    }


def search_reward_weights(
    evaluate: Callable[[dict[str, float], Any], float],
    n_trials: int = MAX_PROXY_TRIALS,
    seed: int = 42,
) -> Any:
    """
    Run the cheap-proxy Optuna study over the MTM reward weights.

    Parameters
    ----------
    evaluate : callable
        ``evaluate(weights, trial) -> float`` returning the PROXY objective
        (Sharpe on a validation slice from ``proxy_validation_split`` — never
        on the test fold). May call ``trial.report``/raise ``TrialPruned``.
    n_trials : int
        Proxy budget; capped at ``MAX_PROXY_TRIALS`` (§5.2).
    seed : int
        TPESampler seed (reproducibility, CLAUDE.md §6.7).

    Returns
    -------
    optuna.Study
        Completed study, direction maximize. Take the top-K (K <=
        ``MAX_FINALISTS``) by value to the full walk-forward gate.
    """
    import optuna

    if n_trials > MAX_PROXY_TRIALS:
        raise ValueError(
            f"n_trials={n_trials} exceeds MAX_PROXY_TRIALS={MAX_PROXY_TRIALS} "
            f"(ADR-041 §5.2 budget)"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    def objective(trial: Any) -> float:
        weights = suggest_reward_weights(trial)
        return evaluate(weights, trial)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(
        "reward search done: %d trials, best=%.4f params=%s",
        len(study.trials), study.best_value, study.best_params,
    )
    return study


def honest_gate_n_trials(n_finalists: int) -> int:
    """
    Deflation count for ``evaluate_drl_gate`` after a reward search (§5.3).

    Equals the number of configs evaluated OUT-OF-SAMPLE — the finalists that
    went through the full walk-forward gate. The proxy trials never saw OOS
    data and must not enter the deflation (inflating would over-deflate;
    under-reporting would be selection-bias fraud — CLAUDE.md §6.10).

    Raises
    ------
    ValueError
        If ``n_finalists`` is outside [1, MAX_FINALISTS].
    """
    if not 1 <= n_finalists <= MAX_FINALISTS:
        raise ValueError(
            f"n_finalists={n_finalists} outside [1, {MAX_FINALISTS}] (ADR-041 §5.2)"
        )
    return n_finalists
