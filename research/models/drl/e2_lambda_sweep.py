"""
E2 — Barrido de λ del reward MTM (arbitraje D · Fase 6).

Pregunta que resuelve: ¿el "no edge" del direccional es real o un **artefacto** de
penalizaciones (λ) demasiado agresivas que empujan al agente a NO operar (flat)? Un
agente que aprende a quedarse plano se ve idéntico a "sin edge" en el backtest. E2 lo
descarta barriendo los pesos del reward MTM vigente (ADR-041,
``envs.trading_env.compute_reward_mtm``):

- ``w_dd``  — penalización de drawdown
- ``w_vol`` — penalización de volatilidad
- ``w_idle``— penalización por inactividad (NO debe dominar)

Por combinación se mide: **% flat** (fracción de barras sin posición), **turnover** y
**Sharpe con IC** (por seeds, reusa E1). El λ óptimo es el que **maximiza LB95(S_Δ)
sin colapsar a flat** — alimenta la definición de "λ óptimo" del acta (Fase 0).

Núcleo numpy-only (grid, %flat, turnover, selección) testeable sin torch; el
orquestador ``run_lambda_sweep`` reentrena el DQN por combinación (torch, perezoso).
Reutiliza el gate (``dsr_gate``) y el IC de E1 (§20.2 — no reimplementar).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from models.drl.e1_baseline_comparison import _ann_sharpe, sharpe_ci_from_seeds

logger = logging.getLogger(__name__)


# =====================================================================
# Grid de λ (numpy-only)
# =====================================================================

@dataclass(frozen=True)
class LambdaPoint:
    w_dd: float
    w_vol: float
    w_idle: float

    def as_kwargs(self) -> dict:
        return {"w_dd": self.w_dd, "w_vol": self.w_vol, "w_idle": self.w_idle}


def lambda_grid(
    w_dd_values: Sequence[float],
    w_vol_values: Sequence[float],
    w_idle_values: Sequence[float],
) -> list[LambdaPoint]:
    """Producto cartesiano de los tres ejes de penalización del reward MTM."""
    if not (w_dd_values and w_vol_values and w_idle_values):
        raise ValueError("cada eje del grid necesita >= 1 valor")
    return [
        LambdaPoint(float(a), float(b), float(c))
        for a, b, c in product(w_dd_values, w_vol_values, w_idle_values)
    ]


# =====================================================================
# Métricas de actividad (numpy-only)
# =====================================================================

def flat_fraction(positions: np.ndarray) -> float:
    """Fracción de barras en flat (posición 0). Alto → el agente casi no opera."""
    p = np.asarray(positions, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean(p == 0.0))


def turnover(positions: np.ndarray) -> float:
    """Turnover medio = media de |Δposición| (entrada inicial desde 0)."""
    p = np.asarray(positions, dtype=float)
    if p.size == 0:
        return float("nan")
    prev = np.concatenate([[0.0], p[:-1]])
    return float(np.mean(np.abs(p - prev)))


# =====================================================================
# Resultado + selección del λ óptimo (numpy-only)
# =====================================================================

@dataclass(frozen=True)
class LambdaResult:
    point: LambdaPoint
    sharpe_mean: float
    lb95: float
    flat_fraction: float
    turnover: float
    n_seeds: int
    n_oos_bars: int


@dataclass(frozen=True)
class SweepVerdict:
    best: Optional[LambdaResult]
    all_flat_collapsed: bool
    max_flat: float
    reason: str
    results: tuple[LambdaResult, ...]


def select_optimal_lambda(
    results: Sequence[LambdaResult], *, max_flat: float = 0.90
) -> SweepVerdict:
    """
    Elige el λ que **maximiza LB95** entre los que NO colapsan a flat
    (``flat_fraction <= max_flat``). Mejor por cota inferior, no por media
    (coherente con E1 / el acta).

    Si TODAS las combinaciones colapsan a flat → bandera ``all_flat_collapsed``:
    el "no edge" es, al menos en parte, **artefacto de λ agresivos** (confirma RC-5),
    no necesariamente ausencia de alfa.
    """
    if not results:
        raise ValueError("results vacío")
    res = tuple(results)
    viable = [r for r in res if not np.isnan(r.flat_fraction) and r.flat_fraction <= max_flat]
    if not viable:
        return SweepVerdict(
            best=None, all_flat_collapsed=True, max_flat=max_flat,
            reason=(
                f"TODAS las {len(res)} combinaciones de λ colapsan a flat "
                f"(>{max_flat:.0%} de barras sin posición): el 'no edge' es, al menos "
                f"en parte, artefacto de penalizaciones agresivas (RC-5), no prueba de "
                f"ausencia de alfa. Reducir w_dd/w_vol/w_idle y re-medir."
            ),
            results=res,
        )
    best = max(viable, key=lambda r: r.lb95)
    return SweepVerdict(
        best=best, all_flat_collapsed=False, max_flat=max_flat,
        reason=(
            f"λ óptimo: w_dd={best.point.w_dd}, w_vol={best.point.w_vol}, "
            f"w_idle={best.point.w_idle} → LB95={best.lb95:.3f}, "
            f"flat={best.flat_fraction:.0%}, turnover={best.turnover:.3f}. "
            f"Este λ alimenta la rama 'λ óptimo' del gate (acta Fase 0)."
        ),
        results=res,
    )


# =====================================================================
# Orquestador (torch, perezoso) — reentrena el DQN por combinación
# =====================================================================

def _oos_positions_and_returns(
    raw_ohlcv: pd.DataFrame, splitter, env_cfg, *, seed: int, episodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Posiciones y retornos OOS concatenados (greedy) con un ``env_cfg`` dado.

    Espeja ``dsr_gate._train_eval_one_fold`` pero devuelve TAMBIÉN las posiciones
    (el gate solo expone retornos; E2 necesita %flat/turnover). Anti-leakage idéntico:
    GMM por fold, embargo del splitter, eval greedy (eps=0). Torch perezoso.
    """
    import dataclasses
    import random

    import torch

    from data.drl_dataset import build_env_frame
    from envs import TradingEnvironment
    from models.drl.dqn import TradingDQN
    from models.drl.dqn_trainer import DQNConfig, DQNTrainer
    from models.drl.dsr_gate import (
        _greedy_positions,
        _validated_folds,
        positions_to_returns,
    )

    pos_all, ret_all = [], []
    for k, (train_idx, test_idx) in enumerate(_validated_folds(raw_ohlcv, splitter)):
        random.seed(seed + k)
        np.random.seed((seed + k) % (2**32))
        torch.manual_seed(seed + k)

        frame = build_env_frame(raw_ohlcv, gmm_train_idx=train_idx)
        train_df = frame.iloc[train_idx]
        test_df = frame.iloc[test_idx]
        if len(train_df) < 2 or len(test_df) < 2:
            continue

        train_cfg = dataclasses.replace(
            env_cfg, episode_length=min(env_cfg.episode_length, len(train_df) - 1)
        )
        train_env = TradingEnvironment(train_df, config=train_cfg, seed=seed + k)
        net = TradingDQN(obs_dim=env_cfg.obs_dim)
        trainer = DQNTrainer(net, DQNConfig())
        trainer.train(train_env, n_episodes=episodes, checkpoint_dir=None, log_every=0)

        positions = _greedy_positions(trainer, test_df, env_cfg, seed=seed + k)
        closes = test_df["close"].to_numpy()[: len(positions)]
        pos_all.append(positions)
        ret_all.append(positions_to_returns(positions, closes, env_cfg.fee_bps))
    return np.concatenate(pos_all), np.concatenate(ret_all)


def run_lambda_sweep(
    raw_ohlcv: pd.DataFrame,
    grid: Sequence[LambdaPoint],
    *,
    n_folds: int = 5,
    seeds: Sequence[int] = (0,),
    episodes: int = 100,
    base_env_cfg=None,
    max_flat: float = 0.90,
) -> SweepVerdict:
    """
    Corre el barrido completo y devuelve el ``SweepVerdict``.

    Por cada ``LambdaPoint`` reconstruye ``env_cfg`` con esos pesos, entrena/evalúa el
    DQN en los mismos folds por cada seed, y agrega Sharpe (IC por seeds), %flat y
    turnover. Requiere torch + datos; el núcleo de selección se testea aparte.
    """
    from models.drl.dsr_gate import EnvironmentConfig, make_wf_splitter
    import dataclasses

    cfg0 = base_env_cfg or EnvironmentConfig()
    splitter = make_wf_splitter(raw_ohlcv, n_folds, env_cfg=cfg0)

    results: list[LambdaResult] = []
    for pt in grid:
        env_cfg = dataclasses.replace(cfg0, **pt.as_kwargs())
        sharpes, flats, turns, last_n = [], [], [], 0
        for s in seeds:
            pos, ret = _oos_positions_and_returns(
                raw_ohlcv, splitter, env_cfg, seed=s, episodes=episodes
            )
            sharpes.append(_ann_sharpe(ret))
            flats.append(flat_fraction(pos))
            turns.append(turnover(pos))
            last_n = len(ret)
        mean, lb, _ = sharpe_ci_from_seeds(sharpes)
        results.append(
            LambdaResult(
                point=pt, sharpe_mean=mean, lb95=lb,
                flat_fraction=float(np.nanmean(flats)),
                turnover=float(np.nanmean(turns)),
                n_seeds=len(seeds), n_oos_bars=int(last_n),
            )
        )
        logger.info(
            "λ(w_dd=%.3f,w_vol=%.3f,w_idle=%.4f): LB95=%.3f flat=%.0f%% turn=%.3f",
            pt.w_dd, pt.w_vol, pt.w_idle, lb, 100 * results[-1].flat_fraction,
            results[-1].turnover,
        )
    verdict = select_optimal_lambda(results, max_flat=max_flat)
    logger.info("E2 veredicto: %s", verdict.reason)
    return verdict
