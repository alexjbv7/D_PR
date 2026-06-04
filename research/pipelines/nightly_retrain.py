"""
NightlyRetrainDAG — S11 nightly retraining pipeline.

Orchestrates MultiHorizonTrainer across the configured horizons, applies
DSR/ECE/class-collapse/regression gates against the current production model,
promotes passing horizons to "staging" (not production — that is S12), and
writes a structured run log to artifacts/runs/{run_id}.json.

Design decisions
----------------
- Trainer is invoked with registry=None so the DAG controls registration;
  the trainer still trains and returns the fitted model_object.
- Gate order is fixed (first failure determines skip_reason).
- UUID v7 for run_id: tries uuid_utils → uuid6 → manual RFC 9562 fallback.
- UTC timestamps throughout; no naive datetimes.
- Dataset loading is stubbed (mirrors train_multi_horizon.py); replace with
  real TimescaleDB / Parquet loader before S12.

References
----------
CLAUDE.md §6.6 (retraining gates)
alpaca_integration.md §5 (S11 roadmap row)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from models.multi_horizon.trainer import MultiHorizonTrainer, TrainResult
from models.multi_horizon.registry_adapter import register_horizon_model
from models.multi_horizon.horizon_config import ALL_HORIZONS
from models.multi_horizon.feature_sets import get_feature_set
from quant_shared.models.registry import ModelCard, ModelRegistry

logger = logging.getLogger(__name__)

_HORIZON_ORDER = ["intraday", "swing", "daily"]


# ============================================================================
# UUID v7 (RFC 9562)
# ============================================================================


_uuid7_last_ms: int = 0
_uuid7_seq: int = 0   # monotonic counter within the same millisecond (12 bits)


def _uuid7() -> str:
    """Generate UUID v7 (time-ordered). Falls back to a manual impl if no library.

    Within the same millisecond, rand_a is a monotonic 12-bit counter so that
    UUIDs remain time-ordered even when generated in tight loops (RFC 9562 §6.2).
    """
    try:
        import uuid_utils  # type: ignore[import]
        return str(uuid_utils.uuid7())
    except ImportError:
        pass
    try:
        import uuid6  # type: ignore[import]
        return str(uuid6.uuid7())
    except ImportError:
        pass

    global _uuid7_last_ms, _uuid7_seq

    # Manual RFC 9562 UUID v7:
    # 48 bits unix-ms | 4 bits ver=7 | 12 bits rand_a (monotonic counter)
    # | 2 bits var=10 | 62 bits rand_b (random)
    ms = int(time.time() * 1000)
    if ms == _uuid7_last_ms:
        _uuid7_seq = (_uuid7_seq + 1) & 0x0FFF   # monotonic within same ms
    else:
        _uuid7_last_ms = ms
        _uuid7_seq = 0                             # always start at 0 on new ms

    rand_b = os.urandom(8)
    b = (
        ms.to_bytes(6, "big")
        + bytes([0x70 | (_uuid7_seq >> 8), _uuid7_seq & 0xFF])
        + bytes([0x80 | (rand_b[0] & 0x3F)])
        + rand_b[1:]
    )
    return str(uuid.UUID(bytes=b))


# ============================================================================
# CONFIG
# ============================================================================


@dataclass
class NightlyRetrainConfig:
    """Configuration for one nightly retrain run."""

    as_of: date = field(default_factory=date.today)
    seed: int = 42
    horizons: list[str] = field(default_factory=lambda: ["intraday", "swing", "daily"])
    n_trials: int = 50
    dsr_min: float = 0.4       # absolute DSR floor (CLAUDE.md §6.6)
    dsr_delta: float = 0.95    # new DSR >= prod_DSR * dsr_delta
    ece_max: float = 0.05      # calibration ceiling (CLAUDE.md §6.6)
    dry_run: bool = False
    run_log_dir: Path = field(default_factory=lambda: Path("artifacts/runs"))


# ============================================================================
# RESULT TYPES
# ============================================================================


@dataclass
class HorizonGateResult:
    """Gate evaluation outcome for one horizon."""

    horizon: str
    dsr_new: float
    dsr_prod: Optional[float]   # None if no production model exists
    ece_new: float
    promoted: bool
    skip_reason: str = ""       # empty string when promoted=True


@dataclass
class RetrainRunLog:
    """Structured record of one nightly retrain execution."""

    run_id: str
    as_of: str                  # ISO date
    started_at: str             # ISO UTC datetime
    finished_at: str
    horizons: list[HorizonGateResult]
    n_promoted: int
    exit_code: int              # 0 = ≥1 promoted, 2 = none


# ============================================================================
# DAG
# ============================================================================


class NightlyRetrainDAG:
    """
    Nightly retraining DAG — S11.

    Parameters
    ----------
    config : NightlyRetrainConfig
    registry : ModelRegistry
        Shared registry used to fetch production baselines and register new
        staging artifacts.
    """

    def __init__(self, config: NightlyRetrainConfig, registry: ModelRegistry) -> None:
        self.config = config
        self.registry = registry

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    async def run(self) -> RetrainRunLog:
        """Execute the full retrain DAG and return a structured run log."""
        cfg = self.config
        started_at = datetime.now(tz=timezone.utc).isoformat()
        run_id = _uuid7()

        logger.info(
            "NightlyRetrainDAG starting run_id=%s as_of=%s dry_run=%s horizons=%s",
            run_id, cfg.as_of, cfg.dry_run, cfg.horizons,
        )

        if cfg.dry_run:
            gate_results = [
                HorizonGateResult(
                    horizon=h,
                    dsr_new=0.0, dsr_prod=None, ece_new=1.0,
                    promoted=False, skip_reason="dry_run",
                )
                for h in cfg.horizons
            ]
            log = RetrainRunLog(
                run_id=run_id,
                as_of=cfg.as_of.isoformat(),
                started_at=started_at,
                finished_at=datetime.now(tz=timezone.utc).isoformat(),
                horizons=gate_results,
                n_promoted=0,
                exit_code=0,
            )
            self._write_run_log(log)
            return log

        # Load datasets (stub — replace with real loader in S12)
        datasets = self._load_datasets()

        # Run trainer (registry=None: DAG controls registration)
        trainer = MultiHorizonTrainer(
            seed=cfg.seed,
            ablate=True,
            n_wf_splits=8,
        )
        results: dict[str, TrainResult] = await trainer.run_all_horizons(
            horizon_datasets={h: datasets[h] for h in cfg.horizons if h in datasets},
            as_of=cfg.as_of,
            symbols=["STUB_SYMBOL"],
            registry=None,
        )

        # Evaluate gates and promote passing horizons
        gate_results: list[HorizonGateResult] = []
        for horizon_name in cfg.horizons:
            result = results.get(horizon_name)
            if result is None or result.model_object is None:
                gate_results.append(HorizonGateResult(
                    horizon=horizon_name,
                    dsr_new=0.0, dsr_prod=None, ece_new=1.0,
                    promoted=False, skip_reason="no_result",
                ))
                continue

            prod_card = self.registry.get_production(f"multi_horizon_{horizon_name}")
            gate = self._check_gates(result, prod_card)
            gate_results.append(gate)

            if gate.promoted:
                self._promote_to_staging(horizon_name, result)
            else:
                logger.info("Horizon %s skipped: %s", horizon_name, gate.skip_reason)

        n_promoted = sum(1 for g in gate_results if g.promoted)
        exit_code = 0 if n_promoted >= 1 else 2

        log = RetrainRunLog(
            run_id=run_id,
            as_of=cfg.as_of.isoformat(),
            started_at=started_at,
            finished_at=datetime.now(tz=timezone.utc).isoformat(),
            horizons=gate_results,
            n_promoted=n_promoted,
            exit_code=exit_code,
        )
        self._write_run_log(log)

        logger.info(
            "NightlyRetrainDAG done run_id=%s n_promoted=%d exit_code=%d",
            run_id, n_promoted, exit_code,
        )
        return log

    # ------------------------------------------------------------------
    # GATES
    # ------------------------------------------------------------------

    def _check_gates(
        self,
        result: TrainResult,
        prod_card: Optional[ModelCard],
    ) -> HorizonGateResult:
        """
        Apply gates in order; first failure determines skip_reason.

        Gate order (CLAUDE.md §6.6):
          1. DSR absolute floor
          2. ECE ceiling
          3. Class collapse
          4. DSR regression vs production model
        """
        cfg = self.config
        dsr_prod = prod_card.dsr if prod_card is not None else None

        if result.dsr < cfg.dsr_min:
            return HorizonGateResult(
                horizon=result.horizon_name,
                dsr_new=result.dsr, dsr_prod=dsr_prod, ece_new=result.ece,
                promoted=False, skip_reason="dsr_below_min",
            )

        if result.ece > cfg.ece_max:
            return HorizonGateResult(
                horizon=result.horizon_name,
                dsr_new=result.dsr, dsr_prod=dsr_prod, ece_new=result.ece,
                promoted=False, skip_reason="ece_above_max",
            )

        if result.class_collapse:
            return HorizonGateResult(
                horizon=result.horizon_name,
                dsr_new=result.dsr, dsr_prod=dsr_prod, ece_new=result.ece,
                promoted=False, skip_reason="class_collapse",
            )

        if prod_card is not None and result.dsr < prod_card.dsr * cfg.dsr_delta:
            return HorizonGateResult(
                horizon=result.horizon_name,
                dsr_new=result.dsr, dsr_prod=dsr_prod, ece_new=result.ece,
                promoted=False, skip_reason="dsr_regression",
            )

        return HorizonGateResult(
            horizon=result.horizon_name,
            dsr_new=result.dsr, dsr_prod=dsr_prod, ece_new=result.ece,
            promoted=True, skip_reason="",
        )

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _promote_to_staging(self, horizon_name: str, result: TrainResult) -> None:
        """Register the artifact and set status='staging'."""
        cfg = self.config
        artifact = register_horizon_model(
            registry=self.registry,
            horizon_name=horizon_name,
            model_object=result.model_object,
            version="0.1",
            psr=result.psr,
            dsr=result.dsr,
            ece=result.ece,
            sharpe_oos=result.sharpe_oos,
            win_rate_oos=result.win_rate,
            train_start=cfg.as_of,
            train_end=cfg.as_of,
            n_folds=8,
            symbols=["STUB_SYMBOL"],
        )
        logger.info(
            "Horizon %s promoted to staging — DSR=%.4f ECE=%.4f artifact=%s",
            horizon_name, result.dsr, result.ece, artifact.artifact_path,
        )

    def _write_run_log(self, log: RetrainRunLog) -> Path:
        """Serialize run log to {run_log_dir}/{run_id}.json and return path."""
        log_dir = Path(self.config.run_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        out_path = log_dir / f"{log.run_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(log), f, indent=2, default=str)
        logger.info("Run log written: %s", out_path)
        return out_path

    def _load_datasets(
        self,
    ) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]]:
        """
        Stub dataset loader.

        Replace with a real TimescaleDB / Parquet loader before S12.
        Mirrors the stub in research/cli/train_multi_horizon.py.
        """
        cfg = self.config
        bar_sizes = {"intraday": "5min", "swing": "4H", "daily": "B"}
        n_bars = {"intraday": 1000, "swing": 800, "daily": 500}

        datasets: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]] = {}
        for hcfg in ALL_HORIZONS:
            if hcfg.name not in cfg.horizons:
                continue
            rng = np.random.default_rng(cfg.seed + _HORIZON_ORDER.index(hcfg.name))
            freq = bar_sizes[hcfg.name]
            n = n_bars[hcfg.name]
            feature_names = get_feature_set(hcfg.feature_set)
            idx = pd.date_range(end=pd.Timestamp(cfg.as_of), periods=n, freq=freq)
            X = pd.DataFrame(
                rng.standard_normal((n, len(feature_names))),
                index=idx,
                columns=feature_names,
            )
            regime_cols = [c for c in feature_names if c.startswith("regime_prob_")]
            if regime_cols:
                raw = X[regime_cols].abs()
                X[regime_cols] = raw.div(raw.sum(axis=1), axis=0)
            for sc in ("session_pre", "session_rth", "session_post"):
                if sc in X.columns:
                    X[sc] = 0
            if "session_rth" in X.columns:
                X["session_rth"] = 1
            y = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx)
            prices = pd.Series(
                100.0 * (1 + rng.normal(0, 0.01, n)).cumprod(), index=idx
            )
            datasets[hcfg.name] = (X, y, prices)
        return datasets
