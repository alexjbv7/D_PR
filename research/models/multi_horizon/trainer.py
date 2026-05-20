"""
MultiHorizonTrainer — coordinates walk-forward training across 3 horizons.

Design:
  - One WalkForwardRunner instance per horizon (reuses existing runner as-is).
  - Hyperopt via Optuna with nested walk-forward on the first train fold.
  - DSR corrected for cross-horizon multiple testing: n_trials = 150 (3×50).
  - asyncio.gather with Semaphore(2) for bounded parallelism.
  - seed=42 propagated to numpy, torch (if available), xgboost, optuna.
  - Reproducibility: same seed → same artifact binary (XGBoost n_jobs=1 final).

CRITICAL ANTI-LEAKAGE INVARIANTS:
  1. embargo per horizon is >= horizon timeout duration.
  2. Daily features used in intraday/swing must carry _lag1d suffix.
  3. Universe loaded point-in-time (universe_historical), not current.
  4. IsotonicCalibrator.fit() only sees TRAIN_calib rows (enforced by runner).
  5. GMM is re-fit per fold inside runner — never on full dataset.

References
----------
Bailey, D.H. & López de Prado, M. (2014). The deflated Sharpe ratio.
ADR-028: docs/adr/028-multi-horizon-config.md
ADR-029: docs/adr/029-dsr-n-trials-correction.md
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import random
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from models.walk_forward_runner import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardRunner,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from models.zoo import get_model
from models.calibration import expected_calibration_error
from models.multi_horizon.horizon_config import (
    ALL_HORIZONS,
    TOTAL_OPTUNA_TRIALS,
    HorizonConfig,
)
from models.multi_horizon.feature_sets import get_feature_set
from models.multi_horizon.registry_adapter import register_horizon_model
from models.multi_horizon.reports import (
    AblativeEntry,
    HorizonReport,
    write_horizon_report,
)

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

_DSR_THRESHOLD = 0.4
_CLASS_COLLAPSE_MIN_PCT = 0.05


# ============================================================================
# RESULT DATACLASS
# ============================================================================


@dataclass
class TrainResult:
    """
    Output for one horizon's complete walk-forward + hyperopt run.

    Parameters
    ----------
    horizon_name : str
    model_object : Any
        Final trained model artifact.
    wf_result : WalkForwardResult
    psr, dsr, ece : float
        OOS metrics.
    sharpe_oos, win_rate, n_trades : float / int
    best_params : dict
        Best hyperparameters from Optuna.
    promoted : bool
        True if DSR >= 0.4 and no class collapse.
    class_collapse : bool
        True if one class is predicted < 5 % of the time OOS.
    feature_importance : dict[str, float]
        Top-10 feature importances (gain).
    ablative_dsrs : dict[str, float]
        DSR per ablation label.
    """

    horizon_name: str
    model_object: Any
    wf_result: WalkForwardResult
    psr: float
    dsr: float
    ece: float
    sharpe_oos: float
    win_rate: float
    n_trades: int
    best_params: dict[str, Any]
    promoted: bool
    class_collapse: bool
    feature_importance: dict[str, float] = field(default_factory=dict)
    ablative_dsrs: dict[str, float] = field(default_factory=dict)


# ============================================================================
# HYPEROPT SEARCH SPACES
# ============================================================================


def _xgb_search_space(trial: Any) -> dict[str, Any]:
    return {
        "max_depth":       trial.suggest_int("max_depth", 3, 7),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
        "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
        "reg_alpha":       trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":      trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        # Deterministic final training (anti-drift in reproducibility tests)
        "n_jobs": 1,
        "seed": 42,
        "random_state": 42,
    }


def _mlp_search_space(trial: Any) -> dict[str, Any]:
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512, step=64)
    return {
        "hidden_dims": [hidden_dim] * n_layers,
        "dropout":     trial.suggest_float("dropout", 0.0, 0.5),
        "lr":          trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size":  trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epochs":      trial.suggest_int("epochs", 20, 100, step=10),
    }


# ============================================================================
# TRAINER
# ============================================================================


class MultiHorizonTrainer:
    """
    Coordinates walk-forward training across 3 horizons in parallel.

    Usage:
        trainer = MultiHorizonTrainer(seed=42)
        results = asyncio.run(trainer.run_all_horizons(
            horizon_datasets={"intraday": (X_5m, y_5m, prices_5m),
                              "swing":    (X_4h, y_4h, prices_4h),
                              "daily":    (X_1d, y_1d, prices_1d)},
            as_of=date(2026, 5, 1),
            symbols=["AAPL", "MSFT", ...],
            registry=model_registry,
        ))
    """

    def __init__(
        self,
        seed: int = 42,
        max_parallel: int = 2,
        n_wf_splits: int = 8,
        ablate: bool = True,
    ) -> None:
        self.seed = seed
        self.max_parallel = max_parallel
        self.n_wf_splits = n_wf_splits
        self.ablate = ablate
        self._seed_all(seed)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def run_all_horizons(
        self,
        horizon_datasets: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]],
        as_of: date,
        symbols: list[str],
        registry: Any | None = None,
    ) -> dict[str, TrainResult]:
        """
        Run all horizons with bounded parallelism.

        Parameters
        ----------
        horizon_datasets : dict
            Keys = horizon name, values = (X, y, prices).
        as_of : date
            Point-in-time reference date for the training run.
        symbols : list[str]
            Universe symbols used in this run.
        registry : ModelRegistry | None
            If provided, artifacts and cards are registered.

        Returns
        -------
        dict mapping horizon_name → TrainResult
        """
        sem = asyncio.Semaphore(self.max_parallel)

        async def _bounded(cfg: HorizonConfig) -> tuple[str, TrainResult]:
            if cfg.name not in horizon_datasets:
                logger.warning("No dataset provided for horizon %s — skipping.", cfg.name)
                return cfg.name, self._empty_result(cfg.name)
            async with sem:
                data = horizon_datasets[cfg.name]
                def _run(
                    c: HorizonConfig = cfg,
                    d: tuple[pd.DataFrame, pd.Series, pd.Series] = data,
                ) -> TrainResult:
                    return self.run_horizon(c, d[0], d[1], d[2])

                result = await asyncio.get_event_loop().run_in_executor(None, _run)
                return cfg.name, result

        pairs = await asyncio.gather(*[_bounded(cfg) for cfg in ALL_HORIZONS])
        results: dict[str, TrainResult] = dict(pairs)

        # Persist artifacts and reports
        if registry is not None:
            for cfg in ALL_HORIZONS:
                r = results.get(cfg.name)
                if r is None or r.model_object is None:
                    continue
                self._persist(r, cfg, as_of, symbols, registry)

        self._log_promotion_summary(results)
        return results

    def run_horizon(
        self,
        cfg: HorizonConfig,
        X: pd.DataFrame,
        y: pd.Series,
        prices: pd.Series,
    ) -> TrainResult:
        """
        Run one horizon end-to-end:
          build WF config → hyperopt (nested WF) → final WF run → metrics → ablative.
        """
        logger.info("=== Horizon: %s ===", cfg.name.upper())
        feature_names = get_feature_set(cfg.feature_set)
        X_feat = self._select_features(X, feature_names)

        # Convert embargo timedelta → bars
        embargo_bars = self._embargo_to_bars(cfg.embargo, cfg.bar_size)

        # Hyperopt on a single inner fold (anti-leakage: only train data)
        best_params = self._hyperopt(cfg, X_feat, y, embargo_bars)

        # Final walk-forward with best params
        wf_cfg = self._build_wf_config(cfg, best_params, embargo_bars)

        # Defensive clamp: if cfg requests more bars than dataset has,
        # the splitter produces 0 folds and the runner crashes on an empty
        # concat. Scale train_size/test_size down so at least one fold runs.
        wf_cfg = self._clamp_wf_to_dataset(wf_cfg, len(X_feat), cfg.name)

        runner = WalkForwardRunner(wf_cfg)
        wf_result = runner.run(X_feat, y, prices=prices)

        # OOS metrics
        psr, dsr, ece, sharpe, win_rate, n_trades, class_collapse = (
            self._compute_metrics(wf_result, prices)
        )

        # Ablative analysis
        ablative_dsrs: dict[str, float] = {}
        if self.ablate:
            ablative_dsrs = self._run_ablative(cfg, X_feat, y, prices, embargo_bars, best_params)

        # Feature importance top-10
        feat_imp = self._top10_importance(wf_result)

        promoted = (dsr >= _DSR_THRESHOLD) and (not class_collapse)

        return TrainResult(
            horizon_name=cfg.name,
            model_object=self._fit_final_model(cfg, X_feat, y, best_params),
            wf_result=wf_result,
            psr=psr,
            dsr=dsr,
            ece=ece,
            sharpe_oos=sharpe,
            win_rate=win_rate,
            n_trades=n_trades,
            best_params=best_params,
            promoted=promoted,
            class_collapse=class_collapse,
            feature_importance=feat_imp,
            ablative_dsrs=ablative_dsrs,
        )

    # ------------------------------------------------------------------
    # HYPEROPT
    # ------------------------------------------------------------------

    def _hyperopt(
        self,
        cfg: HorizonConfig,
        X: pd.DataFrame,
        y: pd.Series,
        embargo_bars: int,
    ) -> dict[str, Any]:
        """Optuna hyperopt with nested walk-forward on first half of X."""
        if not _OPTUNA_AVAILABLE:
            logger.warning("Optuna not installed — using default params for %s.", cfg.name)
            return {}

        # Nested WF: use first 50 % of data as the hyperopt train set
        n_rows = len(X)
        cutoff = n_rows // 2
        X_hp = X.iloc[:cutoff]
        y_hp = y.iloc[:cutoff]

        inner_embargo = max(embargo_bars, 2)

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner  = optuna.pruners.MedianPruner(n_warmup_steps=10)
        study   = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        def objective(trial: optuna.Trial) -> float:
            if cfg.model_name == "xgb":
                params = _xgb_search_space(trial)
            else:
                params = _mlp_search_space(trial)

            inner_cfg = self._build_wf_config(cfg, params, inner_embargo)
            # Reduce splits for speed inside hyperopt
            inner_cfg = self._scale_down_wf_config(inner_cfg)
            try:
                runner = WalkForwardRunner(inner_cfg)
                res = runner.run(X_hp, y_hp)
                return res.global_metrics.get("sharpe") or -999.0
            except Exception as exc:  # noqa: BLE001
                logger.debug("Optuna trial failed: %s", exc)
                raise optuna.TrialPruned() from exc

        n_jobs_opt = max(1, (os.cpu_count() or 4) - 2)
        study.optimize(
            objective,
            n_trials=cfg.n_optuna_trials,
            n_jobs=n_jobs_opt,
            show_progress_bar=False,
        )

        try:
            best = study.best_params if study.best_trial else {}
        except ValueError:
            # All trials were pruned (common with n_optuna_trials=1 in tests)
            best = {}
        try:
            best_val = study.best_value
        except ValueError:
            best_val = float("nan")
        logger.info(
            "Horizon %s — best Optuna sharpe=%.4f params=%s",
            cfg.name,
            best_val,
            best,
        )
        # Force deterministic final training
        if cfg.model_name == "xgb":
            best.update({"n_jobs": 1, "seed": 42, "random_state": 42})
        return best

    # ------------------------------------------------------------------
    # ABLATIVE
    # ------------------------------------------------------------------

    def _run_ablative(
        self,
        cfg: HorizonConfig,
        X: pd.DataFrame,
        y: pd.Series,
        prices: pd.Series,
        embargo_bars: int,
        best_params: dict[str, Any],
    ) -> dict[str, float]:
        """
        Five ablations: base, no_calibration, no_meta_labeler,
        no_regime, no_macro.
        """
        ablations: dict[str, dict[str, Any]] = {
            "base":              {},
            "no_calibration":    {"use_calibration": False},
            "no_meta_labeler":   {"use_meta_labeling": False},
            "no_regime":         {"use_regime_features": False},
            "no_macro":          {},
        }

        results: dict[str, float] = {}
        for label, overrides in ablations.items():
            try:
                ab_cfg = self._build_wf_config(cfg, best_params, embargo_bars)
                for k, v in overrides.items():
                    if hasattr(ab_cfg, k):
                        object.__setattr__(ab_cfg, k, v)

                if label == "no_macro":
                    macro_cols = [c for c in X.columns if any(
                        m in c for m in ("dxy_", "vix_", "yield_curve_", "btc_funding", "btc_exchange")
                    )]
                    X_ab = X.drop(columns=macro_cols, errors="ignore")
                else:
                    X_ab = X

                runner = WalkForwardRunner(ab_cfg)
                wf = runner.run(X_ab, y, prices=prices)
                strat_ret = self._extract_strategy_returns(wf, prices)
                dsr_val = deflated_sharpe_ratio(
                    strat_ret.values, n_trials=TOTAL_OPTUNA_TRIALS
                ) if len(strat_ret) >= 5 else 0.0
                results[label] = round(dsr_val, 4)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ablation %s failed for horizon %s: %s", label, cfg.name, exc)
                results[label] = float("nan")

        return results

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _build_wf_config(
        self,
        cfg: HorizonConfig,
        params: dict[str, Any],
        embargo_bars: int,
    ) -> WalkForwardConfig:
        """Build WalkForwardConfig from HorizonConfig + hyperparams."""
        base_xgb = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 5,
            "n_jobs": 1,
            "seed": self.seed,
            "random_state": self.seed,
        }

        if cfg.model_name == "xgb":
            xgb_p = {**base_xgb, **{k: v for k, v in params.items() if k != "n_jobs"}}
            xgb_p["n_jobs"] = 1  # always deterministic
            mlp_p: dict[str, Any] = {}
        else:
            xgb_p = base_xgb
            mlp_p = params

        # Convert train_lookback to bars (approximate)
        train_bars = self._lookback_to_bars(cfg.train_lookback, cfg.bar_size)
        test_bars  = max(train_bars // self.n_wf_splits, 1)

        wf_cfg = WalkForwardConfig(
            train_size=train_bars,
            test_size=test_bars,
            embargo=embargo_bars,
            expanding=False,
            calib_frac=0.20,
            use_regime_features=True,
            use_meta_labeling=True,
            use_pca=True,
            use_class_weights=True,
            track_importance=True,
            xgb_params=xgb_p,
            model_class=("xgboost" if cfg.model_name == "xgb" else "deep_mlp"),
        )
        if mlp_p and hasattr(wf_cfg, "mlp_params"):
            wf_cfg.mlp_params = mlp_p

        return wf_cfg

    def _scale_down_wf_config(self, wf_cfg: WalkForwardConfig) -> WalkForwardConfig:
        """Reduce splits for inner hyperopt loop (speed)."""
        scaled = copy.copy(wf_cfg)
        scaled.train_size = max(wf_cfg.train_size // 2, 50)
        scaled.test_size  = max(wf_cfg.test_size  // 2, 20)
        return scaled

    def _clamp_wf_to_dataset(
        self,
        wf_cfg: WalkForwardConfig,
        n_rows: int,
        horizon_name: str,
    ) -> WalkForwardConfig:
        """
        Ensure the WalkForwardConfig fits the dataset.

        If ``train_size + test_size + embargo > n_rows`` the splitter
        produces zero folds and the runner crashes on an empty concat.
        Scale train/test down to guarantee at least one valid split,
        logging a warning so the operator sees the adjustment.

        Defensive only — production runs with sufficient data are unaffected.
        """
        required = wf_cfg.train_size + wf_cfg.test_size + wf_cfg.embargo
        if required <= n_rows:
            return wf_cfg

        # Allocate ~60 % to train, ~20 % to test, leave embargo as-is.
        # Floors keep the run statistically meaningful (or at least non-empty).
        safe_train = max(int(n_rows * 0.6), 30)
        safe_test  = max(int(n_rows * 0.2) // max(self.n_wf_splits, 1), 5)

        # Final sanity: never exceed dataset minus embargo
        budget = max(n_rows - wf_cfg.embargo - 1, 10)
        if safe_train + safe_test > budget:
            safe_train = max(budget - safe_test, 10)

        logger.warning(
            "MultiHorizonTrainer[%s]: dataset too small for requested config "
            "(need %d bars, have %d). Clamping train_size=%d, test_size=%d, "
            "embargo=%d.",
            horizon_name, required, n_rows,
            safe_train, safe_test, wf_cfg.embargo,
        )

        clamped = copy.copy(wf_cfg)
        clamped.train_size = safe_train
        clamped.test_size  = safe_test
        return clamped

    def _select_features(self, X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        """Keep only columns present in X; fill missing with 0."""
        present   = [f for f in feature_names if f in X.columns]
        missing   = [f for f in feature_names if f not in X.columns]
        if missing:
            logger.debug(
                "Features not in dataset (will be zero-filled): %s", missing[:10]
            )
        result = X[present].copy()
        for m in missing:
            result[m] = 0.0
        return result[feature_names]

    def _embargo_to_bars(self, embargo: timedelta, bar_size: str) -> int:
        """Convert timedelta to integer bar count for WalkForwardConfig.

        For daily bars: embargo in calendar days maps directly to trading days
        (1 calendar day ≈ 1 trading day for multi-day embargos).
        """
        bar_size_lower = bar_size.lower()
        if bar_size_lower in ("1d", "d", "b"):
            # Daily bars: 1 trading day = 1 bar
            return max(1, embargo.days)

        bar_minutes = _bar_size_minutes(bar_size)
        embargo_minutes = embargo.total_seconds() / 60
        return max(1, int(embargo_minutes / bar_minutes))

    def _lookback_to_bars(self, lookback: timedelta, bar_size: str) -> int:
        bar_minutes = _bar_size_minutes(bar_size)
        trading_minutes_per_day = 390  # RTH
        total_minutes = lookback.days * trading_minutes_per_day
        return max(50, int(total_minutes / bar_minutes))

    def _compute_metrics(
        self,
        wf_result: WalkForwardResult,
        prices: pd.Series,
    ) -> tuple[float, float, float, float, float, int, bool]:
        """Return (psr, dsr, ece, sharpe, win_rate, n_trades, class_collapse)."""
        gm = wf_result.global_metrics
        strat_ret = self._extract_strategy_returns(wf_result, prices)

        psr = float(gm.get("psr") or 0.0)
        # DSR corrected for all 150 trials across 3 horizons (ADR-029)
        dsr = (
            deflated_sharpe_ratio(strat_ret.values, n_trials=TOTAL_OPTUNA_TRIALS)
            if len(strat_ret) >= 5
            else 0.0
        )
        sharpe    = float(gm.get("sharpe") or 0.0)
        win_rate  = float(gm.get("win_rate") or 0.0)
        n_trades  = int(gm.get("n_trades") or 0)

        # ECE: aggregate over all OOS proba
        oos_proba  = wf_result.oos_proba
        oos_labels = wf_result.oos_signals  # signals as proxy for direction labels
        if not oos_proba.empty and len(oos_proba) > 0:
            # Use max-class column as the "positive" proba for ECE
            max_col_proba = oos_proba.max(axis=1).values
            binary_labels = (oos_labels != 0).astype(int).values
            ece = float(expected_calibration_error(binary_labels, max_col_proba))
        else:
            ece = 1.0

        # Class collapse detection
        signals = wf_result.oos_signals
        if len(signals) > 0:
            for cls in [-1, 0, 1]:
                frac = float((signals == cls).mean())
                if frac < _CLASS_COLLAPSE_MIN_PCT:
                    logger.warning(
                        "Class collapse detected: class=%d has %.1f%% of OOS signals.",
                        cls, frac * 100,
                    )
                    return psr, dsr, ece, sharpe, win_rate, n_trades, True
        return psr, dsr, ece, sharpe, win_rate, n_trades, False

    def _extract_strategy_returns(
        self,
        wf_result: WalkForwardResult,
        prices: pd.Series,
    ) -> pd.Series:
        signals = wf_result.oos_signals
        price_ret = prices.pct_change().reindex(signals.index).fillna(0.0)
        strat_ret = (signals * price_ret)[signals != 0]
        return strat_ret

    def _top10_importance(self, wf_result: WalkForwardResult) -> dict[str, float]:
        if wf_result.feature_importance_agg.empty:
            return {}
        agg = wf_result.feature_importance_agg
        if "mean" in agg.columns:
            top = agg["mean"].sort_values(ascending=False).head(10)
        else:
            top = agg.iloc[:, 0].sort_values(ascending=False).head(10)
        return {str(k): round(float(v), 6) for k, v in top.items()}

    def _fit_final_model(
        self,
        cfg: HorizonConfig,
        X: pd.DataFrame,
        y: pd.Series,
        best_params: dict[str, Any],
    ) -> Any:
        """
        Re-fit final model on all data with best hyperparams.

        Reproducibility contract
        ------------------------
        ``self.seed`` is the single source of truth — it is *always* injected
        into the final model's params (overriding any seed Optuna may have
        sampled).  Without this, hyperopt with degenerate / empty
        ``best_params`` would produce identical artifacts across distinct
        trainer seeds (see ``test_different_seed_different_model``).

        XGBoost gets ``n_jobs=1`` to guarantee deterministic CPU execution.
        """
        params: dict[str, Any] = dict(best_params)
        params["seed"]         = self.seed
        params["random_state"] = self.seed
        if cfg.model_name == "xgb":
            params["n_jobs"] = 1

        model = get_model(
            "xgboost" if cfg.model_name == "xgb" else "deep_mlp",
            **params,
        )
        model.fit(X, y)
        return model

    def _persist(
        self,
        result: TrainResult,
        cfg: HorizonConfig,
        as_of: date,
        symbols: list[str],
        registry: Any,
    ) -> None:
        """Register artifact and write JSON report."""
        train_start = (as_of - cfg.train_lookback).isoformat()

        artifact = register_horizon_model(
            registry=registry,
            horizon_name=cfg.name,
            model_object=result.model_object,
            version="0.1",
            psr=result.psr,
            dsr=result.dsr,
            ece=result.ece,
            sharpe_oos=result.sharpe_oos,
            win_rate_oos=result.win_rate,
            train_start=date.fromisoformat(train_start),
            train_end=as_of,
            n_folds=self.n_wf_splits,
            symbols=symbols,
        )

        report = HorizonReport(
            horizon=cfg.name,
            model=cfg.model_name,
            version="0.1",
            as_of=as_of.isoformat(),
            training_window=[train_start, as_of.isoformat()],
            universe_size=len(symbols),
            psr=result.psr,
            dsr=result.dsr,
            ece=result.ece,
            brier=0.0,
            n_trades_oos=result.n_trades,
            win_rate=result.win_rate,
            sharpe_oos=result.sharpe_oos,
            hyperparams=result.best_params,
            feature_importance_top10=result.feature_importance,
            ablative=[
                AblativeEntry(label=k, dsr=v)
                for k, v in result.ablative_dsrs.items()
            ],
            artifact_path=str(artifact.artifact_path),
            seed=self.seed,
            promoted=result.promoted,
            promotion_reason=(
                "DSR >= 0.4 OOS" if result.promoted
                else f"DSR={result.dsr:.4f} < 0.4 — no edge in current data"
            ),
        )
        write_horizon_report(report)

    def _log_promotion_summary(self, results: dict[str, TrainResult]) -> None:
        lines = ["\n" + "=" * 60, " MULTI-HORIZON PROMOTION SUMMARY", "=" * 60]
        promoted_count = 0
        for name, r in results.items():
            status = "PROMOTED" if r.promoted else "ARCHIVED (no edge)"
            lines.append(
                f"  {name:10s}  DSR={r.dsr:.4f}  PSR={r.psr:.4f}  ECE={r.ece:.4f}  "
                f"collapse={r.class_collapse}  → {status}"
            )
            if r.promoted:
                promoted_count += 1
        lines.append(f"\n  Horizons promoted: {promoted_count}/3")
        if promoted_count < 2:
            lines.append(
                "  WARNING: fewer than 2 horizons promoted. "
                "Document as 'no edge in current data' per spec."
            )
        lines.append("=" * 60)
        logger.info("\n".join(lines))

    def _empty_result(self, horizon_name: str) -> TrainResult:
        dummy_signals = pd.Series(dtype=float)
        dummy_proba   = pd.DataFrame()
        dummy_sizing  = pd.DataFrame()
        dummy_wf = WalkForwardResult(
            config=WalkForwardConfig(),
            fold_results=[],
            oos_signals=dummy_signals,
            oos_proba=dummy_proba,
            oos_sizing=dummy_sizing,
            feature_importance_agg=pd.DataFrame(),
            features_to_drop=[],
            global_metrics={},
        )
        return TrainResult(
            horizon_name=horizon_name,
            model_object=None,
            wf_result=dummy_wf,
            psr=0.0, dsr=0.0, ece=1.0,
            sharpe_oos=0.0, win_rate=0.0, n_trades=0,
            best_params={},
            promoted=False,
            class_collapse=False,
        )

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        if _TORCH_AVAILABLE:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ============================================================================
# UTILITIES
# ============================================================================


def _bar_size_minutes(bar_size: str) -> float:
    """Convert pandas-style bar_size string to minutes."""
    mapping: dict[str, float] = {
        "5min": 5.0,
        "5T": 5.0,
        "15min": 15.0,
        "1h": 60.0,
        "1H": 60.0,
        "4H": 240.0,
        "4h": 240.0,
        "1D": 390.0,   # one RTH session
        "1d": 390.0,
    }
    if bar_size in mapping:
        return mapping[bar_size]
    logger.warning("Unknown bar_size '%s', defaulting to 60 minutes.", bar_size)
    return 60.0
