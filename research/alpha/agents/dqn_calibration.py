"""
Calibración OOS por fold del ``p_win`` del DQN (E3 paso 2 · arbitraje D).

El ``DqnAlphaAgent`` emite ``p_win = softmax(Q)[a]``, un proxy ORDINAL que NO es
una probabilidad calibrada. Mientras ``p_win_calibrated=False`` el guard de sizing
(``quant_shared.schemas.signals.require_calibrated_signal``, R-02) rechaza usarlo en
Kelly. Este módulo produce el calibrador que levanta ese bloqueo de forma honesta:

1. ``collect_dqn_calibration_pairs`` — rollout greedy (eps=0) sobre el slice de
   CALIBRACIÓN; registra, por barra direccional, la confianza cruda ``p_raw`` de la
   acción elegida y el ``outcome`` (¿esa apuesta ganó en la barra siguiente?).
2. ``fit_dqn_fold_calibrator`` — ajusta un ``ScalarProbabilityCalibrator`` 1D sobre
   esos pares y devuelve un ``Callable[[float], float]`` listo para
   ``DqnAlphaAgent(calibrator=...)``.
3. ``calibration_diagnostics`` — ECE antes/después + datos de reliability diagram.

Protocolo anti-leakage (NO negociable, igual que ``models.calibration``):

    [ TRAIN_fit | TRAIN_calib | embargo | TEST ]

- Entrena el DQN en ``TRAIN_fit``.
- Ajusta el calibrador en ``TRAIN_calib`` (held-out dentro de train).
- Sirve en ``TEST`` con el calibrador YA fijo. **Jamás** se ajusta con outcomes de
  ``TEST`` (eso sería look-ahead y contaminaría el gate).

El rollout reusa la MISMA definición de posición→retorno del gate
(``models.drl.dsr_gate.positions_to_returns``, ADR-040 §3.3): la posición fijada en
la barra ``j`` gana el movimiento ``close[j+1]/close[j]-1``. Así el ``outcome`` que
calibra el ``p_win`` es coherente con cómo el gate mide retornos.

``torch`` se importa de forma perezosa: el módulo (y su parte 1D) queda importable en
entornos sin torch para tests del calibrador escalar.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd

from envs import EnvironmentConfig, TradingEnvironment
from models.calibration import (
    ScalarProbabilityCalibrator,
    expected_calibration_error,
    reliability_diagram_data,
)

if TYPE_CHECKING:  # pragma: no cover
    from models.drl.dqn_trainer import DQNTrainer

logger = logging.getLogger(__name__)

#: Acciones direccionales del DQN ({0:SHORT, 1:FLAT, 2:LONG} — ver dqn_agent).
_FLAT_ACTION = 1


def collect_dqn_calibration_pairs(
    trainer: "DQNTrainer",
    calib_df: pd.DataFrame,
    env_cfg: EnvironmentConfig,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rollout greedy sobre ``calib_df``; devuelve ``(p_raw, outcomes)`` direccionales.

    Para cada barra ``j`` se calcula la confianza ``p_raw_j = softmax(Q(s_j))[a_j]``
    de la acción greedy ``a_j`` (idéntico a ``DqnAlphaAgent.predict``), se aplica
    ``a_j`` al entorno y se registra la posición resultante ``pos_j``. El
    ``outcome_j`` es 1 si ``pos_j``·(retorno de la barra siguiente) > 0. Solo se
    devuelven barras con ``pos_j != 0`` (las flat no tienen acierto que calibrar).

    Parameters
    ----------
    trainer : DQNTrainer
        Trainer YA entrenado en ``TRAIN_fit`` (se usa ``trainer.online_net``).
    calib_df : pd.DataFrame
        Slice de calibración (env frame con columna ``close``). DEBE ser disjunto
        del TEST y posterior a TRAIN_fit (anti-leakage).
    env_cfg : EnvironmentConfig
        Config del entorno (misma que el gate; ``fee_bps`` no afecta el signo del
        outcome direccional pero se mantiene para coherencia del rollout).
    seed : int
        Semilla del entorno de evaluación (determinista, eps=0).

    Returns
    -------
    (p_raw, outcomes) : tuple[np.ndarray, np.ndarray]
        Arrays alineados con la confianza cruda y el acierto {0,1} de cada barra
        direccional del slice de calibración.
    """
    import torch  # heavy import — perezoso (mantiene la parte 1D testeable sin torch)

    if len(calib_df) < 3:
        raise ValueError(f"calib_df demasiado corto para calibrar: {len(calib_df)}")

    eval_cfg = dataclasses.replace(env_cfg, episode_length=len(calib_df) - 1)
    env = TradingEnvironment(calib_df, config=eval_cfg, seed=seed)
    net = trainer.online_net
    device = getattr(trainer, "device", "cpu")

    obs, _ = env.reset()
    state = torch.tensor(obs, dtype=torch.float32)
    p_raw_steps: list[float] = []
    positions: list[int] = []
    with torch.no_grad():
        while True:
            q = net(state.to(device).unsqueeze(0))[0]
            action = int(q.argmax(dim=-1).item())
            p_raw_steps.append(float(torch.softmax(q, dim=-1)[action].item()))
            obs, _, terminated, truncated, info = env.step(action)
            positions.append(int(info["position"]))
            state = torch.tensor(obs, dtype=torch.float32)
            if terminated or truncated:
                break

    p_raw_arr = np.asarray(p_raw_steps, dtype=float)
    pos_arr = np.asarray(positions, dtype=float)
    closes = calib_df["close"].to_numpy(dtype=float)[: len(pos_arr)]

    # La posición de la barra j gana el movimiento j -> j+1 (misma convención que
    # positions_to_returns, ADR-040 §3.3). Última barra sin barra siguiente: se cae.
    n = min(len(pos_arr), len(closes) - 1)
    if n <= 0:
        return np.empty(0), np.empty(0)
    fwd_ret = closes[1 : n + 1] / closes[:n] - 1.0
    pos_j = pos_arr[:n]
    p_raw_j = p_raw_arr[:n]

    directional = pos_j != 0.0
    gross = pos_j[directional] * fwd_ret[directional]
    outcomes = (gross > 0.0).astype(float)
    p_raw_dir = p_raw_j[directional]

    logger.info(
        "Calibración DQN: %d barras direccionales de %d (tasa acierto cruda=%.3f)",
        int(directional.sum()), n, float(outcomes.mean()) if outcomes.size else float("nan"),
    )
    return p_raw_dir, outcomes


def fit_dqn_fold_calibrator(
    trainer: "DQNTrainer",
    calib_df: pd.DataFrame,
    env_cfg: EnvironmentConfig,
    *,
    seed: int,
    method: str = "isotonic",
    min_samples_isotonic: int = 80,
) -> ScalarProbabilityCalibrator:
    """
    Ajusta y devuelve el calibrador 1D del fold, listo para
    ``DqnAlphaAgent(calibrator=...)``.

    Convención de uso (cierra el guard R-02)::

        cal = fit_dqn_fold_calibrator(trainer, calib_df, env_cfg, seed=seed+k)
        agent = DqnAlphaAgent(net, calibrator=cal)   # -> p_win_calibrated=True
    """
    p_raw, outcomes = collect_dqn_calibration_pairs(
        trainer, calib_df, env_cfg, seed=seed
    )
    if p_raw.size == 0:
        raise ValueError(
            "sin barras direccionales en el slice de calibración: el agente quedó "
            "flat todo el tramo (revisa λ del reward — cruza con E2)."
        )
    return ScalarProbabilityCalibrator(
        method=method, min_samples_isotonic=min_samples_isotonic
    ).fit(p_raw, outcomes)


def calibration_diagnostics(
    p_raw: np.ndarray,
    outcomes: np.ndarray,
    calibrator: ScalarProbabilityCalibrator,
    n_bins: int = 10,
) -> dict:
    """
    Métricas de calibración antes/después + datos del reliability diagram.

    Returns
    -------
    dict con ``ece_uncalibrated``, ``ece_calibrated``, ``ece_improvement``,
    ``verdict`` (OK <0.05 / MARGINAL <0.10 / POOR) y ``reliability`` (curvas para
    graficar). Reusa ``expected_calibration_error`` y ``reliability_diagram_data``
    de ``models.calibration`` (§20.2 — no reimplementar).
    """
    p_cal = calibrator.transform(p_raw)
    ece_before = expected_calibration_error(outcomes, p_raw, n_bins)
    ece_after = expected_calibration_error(outcomes, p_cal, n_bins)
    verdict = "OK" if ece_after < 0.05 else ("MARGINAL" if ece_after < 0.10 else "POOR")
    return {
        "n_samples": int(len(p_raw)),
        "base_win_rate": float(np.mean(outcomes)) if len(outcomes) else float("nan"),
        "ece_uncalibrated": round(ece_before, 4),
        "ece_calibrated": round(ece_after, 4),
        "ece_improvement": round(ece_before - ece_after, 4),
        "verdict": verdict,
        "reliability_uncalibrated": reliability_diagram_data(outcomes, p_raw, n_bins),
        "reliability_calibrated": reliability_diagram_data(outcomes, p_cal, n_bins),
    }
