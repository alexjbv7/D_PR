"""
Bayesian Hyperparameter Optimization
=====================================
Usa Optuna (TPE sampler) para encontrar los mejores parámetros de:
  - XGBoost: n_estimators, max_depth, learning_rate, subsample,
             colsample_bytree, reg_alpha, reg_lambda, min_child_weight, gamma
  - Barrera triple: upper_mult, lower_mult, horizon
  - (Opcional) R:R dinámico: rr_min, rr_max

PROTOCOLO ANTI-LEAKAGE:
  - val_frac% inicial del dataset -> SOLO para hyperopt (nunca el test final)
  - Dentro del hyperopt: mini walk-forward de n_val_folds folds -> PSR/Sharpe OOS
  - El conjunto de test final (1 - val_frac) NUNCA se toca durante la búsqueda

Uso:
    from models.hyperopt import BayesianHyperopt, HyperoptConfig

    ho = BayesianHyperopt(HyperoptConfig(n_trials=50))
    result = ho.run(X, close, atr, label_fn, prices, base_wf_config=cfg)
    print(result.summary())

    # Aplicar mejores parámetros
    best_cfg = result.to_walk_forward_config(cfg)
    best_barrier = result.best_barrier_params   # {"upper_mult":..., "lower_mult":..., "horizon":...}
"""
from __future__ import annotations

import dataclasses
import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURACIÓN
# =====================================================================

@dataclass
class HyperoptConfig:
    """
    Espacio de búsqueda y presupuesto para la optimización bayesiana.

    Rangos de búsqueda
    ------------------
    Todos los *_range son tuplas (min, max) pasadas al sampler de Optuna.
    Parámetros con rango de un solo punto quedan fijos.

    Protocolo de validación
    -----------------------
    val_frac     : fracción inicial del dataset usada durante hyperopt
    n_val_folds  : folds internos por trial (3-5 recomendado, +folds = +lento)
    train_size   : barras de entrenamiento por fold interno
    val_size     : barras de validación por fold interno
    embargo      : barras de embargo entre train y val

    Objetivo
    --------
    objective_metric : "psr" | "sharpe" | "coverage_psr"
       - psr            : PSR(SR > 0) — recomendado
       - sharpe         : Sharpe anualizado simple
       - coverage_psr   : PSR * clip(coverage / target_coverage, 0, 1)
    min_trades       : mínimo de trades OOS; si < min_trades -> trial inválido (nan)
    target_coverage  : cobertura objetivo para "coverage_psr"
    """

    # --- XGBoost ---
    n_estimators_range: tuple = (100, 600)
    max_depth_range: tuple = (3, 7)
    learning_rate_range: tuple = (0.01, 0.20)
    subsample_range: tuple = (0.60, 1.00)
    colsample_bytree_range: tuple = (0.50, 1.00)
    reg_alpha_range: tuple = (0.0, 2.0)
    reg_lambda_range: tuple = (0.5, 4.0)
    min_child_weight_range: tuple = (3, 40)
    gamma_range: tuple = (0.0, 0.5)

    # --- Barrera triple ---
    upper_mult_range: tuple = (0.5, 2.5)
    lower_mult_range: tuple = (0.5, 2.5)
    horizon_range: tuple = (3, 15)          # enteros
    symmetric_barriers: bool = False        # True => upper_mult == lower_mult

    # --- R:R (opcional) ---
    search_rr: bool = False
    rr_min_range: tuple = (1.0, 2.0)
    rr_max_range: tuple = (1.5, 4.0)

    # --- Presupuesto ---
    n_trials: int = 50
    timeout: Optional[float] = None         # segundos; None = sin límite
    n_jobs: int = 1                         # trials en paralelo

    # --- Protocolo de validación ---
    val_frac: float = 0.80                  # 80% para hyperopt, 20% test final
    n_val_folds: int = 3
    train_size: int = 252
    val_size: int = 63
    embargo: int = 5
    calib_frac: float = 0.20
    calib_method: str = "sigmoid"
    use_class_weights: bool = True

    # --- Objetivo ---
    objective_metric: str = "psr"
    min_trades: int = 20
    target_coverage: float = 0.15          # para "coverage_psr"

    # --- Sampler / Pruner ---
    sampler: str = "tpe"                   # "tpe" | "random" | "cmaes"
    seed: int = 42
    use_pruner: bool = True
    n_warmup_steps: int = 5

    # --- Verbosity ---
    verbose: bool = True                   # True = progress log por trial


# =====================================================================
# RESULTADO
# =====================================================================

@dataclass
class HyperoptResult:
    """
    Resultado de una corrida de optimización bayesiana.

    Atributos
    ---------
    best_params          : dict completo (keys = nombres de Optuna)
    best_value           : valor del objetivo en el mejor trial
    n_trials_completed   : trials que llegaron al final (no podados ni fallidos)
    n_trials_pruned      : trials podados por el pruner de Optuna
    all_trials           : lista de dicts {params, value, state} para análisis
    study                : objeto optuna.Study (para plots y análisis)
    best_xgb_params      : subdict solo con los params de XGBoost
    best_barrier_params  : subdict {"upper_mult", "lower_mult", "horizon"}
    best_rr_params       : subdict {"rr_min", "rr_max"} (vacío si no se buscó)
    """
    best_params: dict
    best_value: float
    n_trials_completed: int
    n_trials_pruned: int
    all_trials: List[dict]
    study: object                          # optuna.Study

    best_xgb_params: dict = field(default_factory=dict)
    best_barrier_params: dict = field(default_factory=dict)
    best_rr_params: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 65,
            " HYPEROPT RESULT SUMMARY",
            "=" * 65,
            f"  Trials completados : {self.n_trials_completed}",
            f"  Trials podados     : {self.n_trials_pruned}",
            f"  Trials fallidos    : {len(self.all_trials) - self.n_trials_completed - self.n_trials_pruned}",
            f"  Mejor valor        : {self.best_value:.4f}",
            "",
            "  MEJORES PARAMS XGBoost:",
        ]
        for k, v in sorted(self.best_xgb_params.items()):
            fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"    {k:25s}: {fmt}")
        lines.append("")
        lines.append("  MEJORES PARAMS BARRERA:")
        for k, v in self.best_barrier_params.items():
            fmt = f"{v:.3f}" if isinstance(v, float) else str(v)
            lines.append(f"    {k:25s}: {fmt}")
        if self.best_rr_params:
            lines.append("")
            lines.append("  MEJORES PARAMS R:R:")
            for k, v in self.best_rr_params.items():
                lines.append(f"    {k:25s}: {v:.3f}")
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_walk_forward_config(self, base_config=None):
        """
        Devuelve un WalkForwardConfig con los mejores XGBoost + R:R params.

        Los params de barrera (upper_mult, lower_mult, horizon) NO se guardan
        en WalkForwardConfig porque son propiedades del labeling, no del runner.
        El caller debe usarlos por separado para recomputar los labels.

        Parameters
        ----------
        base_config : WalkForwardConfig, optional
            Config base; todos los campos no buscados se heredan de aquí.

        Returns
        -------
        WalkForwardConfig con los mejores params aplicados.
        """
        from models.walk_forward_runner import WalkForwardConfig

        if base_config is None:
            base_config = WalkForwardConfig()

        # Merge: best XGBoost sobre el base
        xgb = dict(base_config.xgb_params)
        xgb.update(self.best_xgb_params)

        kwargs: dict = {"xgb_params": xgb}

        # R:R si se buscó
        if self.best_rr_params:
            if "rr_min" in self.best_rr_params:
                kwargs["rr_min"] = self.best_rr_params["rr_min"]
            if "rr_max" in self.best_rr_params:
                kwargs["rr_max"] = self.best_rr_params["rr_max"]

        return dataclasses.replace(base_config, **kwargs)


# =====================================================================
# OPTIMIZADOR BAYESIANO
# =====================================================================

class BayesianHyperopt:
    """
    Optimizador bayesiano para el pipeline de ML trading usando Optuna.

    Diseño:
      - Cada trial samplea params de XGBoost + barrera
      - Recomputa labels con los params de barrera del trial
      - Corre un mini walk-forward de n_val_folds folds en la porción de hyperopt
      - Devuelve PSR/Sharpe como objetivo

    ANTI-LEAKAGE:
      - val_frac% inicial del dataset -> solo para hyperopt
      - El resto (test final) NUNCA se accede durante la búsqueda

    Ejemplo:
        ho = BayesianHyperopt(HyperoptConfig(n_trials=50, verbose=True))
        result = ho.run(
            X=features,
            close=prices_df["close"],
            atr=atr_series,
            label_fn=lambda um, lm, h: triple_barrier_labels(close, atr, h, um, lm),
            prices=prices_df["close"],
            base_wf_config=cfg,
            all_classes=[-1, 0, 1],
        )
        print(result.summary())
    """

    def __init__(self, config: HyperoptConfig):
        self.cfg = config
        self._best_so_far: float = float("-inf")

    # ------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # ------------------------------------------------------------------

    def run(
        self,
        X: pd.DataFrame,
        close: pd.Series,
        atr: pd.Series,
        label_fn: Callable,
        prices: Optional[pd.Series] = None,
        base_wf_config=None,
        all_classes: Optional[list] = None,
    ) -> HyperoptResult:
        """
        Ejecuta la optimización bayesiana.

        Parameters
        ----------
        X : pd.DataFrame
            Features pre-computados (sin NaN, índice temporal).
        close : pd.Series
            Precios de cierre — mismo índice que X o superset de él.
            Usado para recomputar labels por trial.
        atr : pd.Series
            ATR pre-calculado — mismo índice o superset.
        label_fn : Callable
            Firma: (upper_mult: float, lower_mult: float, horizon: int) -> pd.Series
            Debe devolver labels {-1, 0, +1} alineados con X.index.
            Las últimas `horizon` barras pueden ser NaN (triple-barrier normal).
        prices : pd.Series, optional
            Precios para métricas OOS. Si None, usa `close`.
        base_wf_config : WalkForwardConfig, optional
            Parámetros base del runner (los no buscados se heredan de aquí).
        all_classes : list, optional
            Clases esperadas. Default: [-1, 0, 1].

        Returns
        -------
        HyperoptResult
        """
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "Optuna no instalado. Ejecuta:\n"
                "    pip install optuna>=3.0.0"
            )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if all_classes is None:
            all_classes = [-1, 0, 1]
        if prices is None:
            prices = close

        cfg = self.cfg

        # ── Definir split hyperopt / test final ───────────────────────
        n_total = len(X)
        val_end_idx = int(n_total * cfg.val_frac)
        if val_end_idx < cfg.train_size + cfg.n_val_folds * (cfg.val_size + cfg.embargo):
            raise ValueError(
                f"Dataset demasiado pequeño para hyperopt. "
                f"Necesitas al menos "
                f"{cfg.train_size + cfg.n_val_folds * (cfg.val_size + cfg.embargo)} barras "
                f"en la porción de hyperopt (tienes {val_end_idx})."
            )

        X_ho = X.iloc[:val_end_idx].copy()
        prices_ho = prices.reindex(X_ho.index)

        if cfg.verbose:
            logger.warning(
                f"[Hyperopt] Porcion hyperopt: {len(X_ho)} barras  "
                f"({X_ho.index[0].date()} -> {X_ho.index[-1].date()})"
            )
            logger.warning(
                f"[Hyperopt] Test final reservado: {n_total - val_end_idx} barras"
            )
            logger.warning(
                f"[Hyperopt] Buscando {cfg.n_trials} trials  "
                f"({cfg.n_val_folds} folds internos por trial)"
            )

        # ── Sampler y pruner ──────────────────────────────────────────
        sampler = self._make_sampler(optuna)
        pruner = (
            optuna.pruners.MedianPruner(n_warmup_steps=cfg.n_warmup_steps)
            if cfg.use_pruner
            else optuna.pruners.NopPruner()
        )

        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        # ── Closure objetivo ──────────────────────────────────────────
        def objective(trial):
            return self._objective(
                trial=trial,
                X_ho=X_ho,
                prices_ho=prices_ho,
                label_fn=label_fn,
                base_wf_config=base_wf_config,
                all_classes=all_classes,
            )

        # ── Optimizar ─────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(
                objective,
                n_trials=cfg.n_trials,
                timeout=cfg.timeout,
                n_jobs=cfg.n_jobs,
                show_progress_bar=False,
                callbacks=[self._make_callback()] if cfg.verbose else None,
            )

        # ── Construir resultado ───────────────────────────────────────
        try:
            best = study.best_trial
            best_params = best.params
            best_value = best.value
        except ValueError:
            # Todos los trials fallaron
            logger.warning("[Hyperopt] Ningun trial valido. Devolviendo params por defecto.")
            best_params = {}
            best_value = float("nan")

        _XGB_KEYS = [
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "reg_alpha", "reg_lambda", "min_child_weight", "gamma",
        ]
        _BARRIER_KEYS = ["upper_mult", "lower_mult", "horizon"]
        _RR_KEYS = ["rr_min", "rr_max"]

        best_xgb = {k: best_params[k] for k in _XGB_KEYS if k in best_params}
        best_barrier = {k: best_params[k] for k in _BARRIER_KEYS if k in best_params}
        # lower_mult puede haberse almacenado como user_attr si symmetric_barriers=True
        if cfg.symmetric_barriers and "upper_mult" in best_barrier:
            best_barrier.setdefault("lower_mult", best_barrier["upper_mult"])
        best_rr = {k: best_params[k] for k in _RR_KEYS if k in best_params}

        _COMPLETE = optuna.trial.TrialState.COMPLETE
        _PRUNED = optuna.trial.TrialState.PRUNED
        n_completed = sum(1 for t in study.trials if t.state == _COMPLETE)
        n_pruned = sum(1 for t in study.trials if t.state == _PRUNED)

        all_trial_data = [
            {"number": t.number, "params": t.params,
             "value": t.value, "state": str(t.state.name)}
            for t in study.trials
        ]

        result = HyperoptResult(
            best_params=best_params,
            best_value=best_value,
            n_trials_completed=n_completed,
            n_trials_pruned=n_pruned,
            all_trials=all_trial_data,
            study=study,
            best_xgb_params=best_xgb,
            best_barrier_params=best_barrier,
            best_rr_params=best_rr,
        )

        if cfg.verbose:
            logger.warning("\n" + result.summary())

        return result

    # ------------------------------------------------------------------
    # FUNCIÓN OBJETIVO
    # ------------------------------------------------------------------

    def _objective(
        self,
        trial,
        X_ho: pd.DataFrame,
        prices_ho: pd.Series,
        label_fn: Callable,
        base_wf_config,
        all_classes: list,
    ) -> float:
        cfg = self.cfg

        # ── 1. Samplear parámetros ────────────────────────────────────
        xgb_params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", *cfg.n_estimators_range),
            "max_depth": trial.suggest_int(
                "max_depth", *cfg.max_depth_range),
            "learning_rate": trial.suggest_float(
                "learning_rate", *cfg.learning_rate_range, log=True),
            "subsample": trial.suggest_float(
                "subsample", *cfg.subsample_range),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *cfg.colsample_bytree_range),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", *cfg.reg_alpha_range),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", *cfg.reg_lambda_range),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *cfg.min_child_weight_range),
            "gamma": trial.suggest_float(
                "gamma", *cfg.gamma_range),
        }

        upper_mult = trial.suggest_float("upper_mult", *cfg.upper_mult_range)
        if cfg.symmetric_barriers:
            lower_mult = upper_mult
        else:
            lower_mult = trial.suggest_float("lower_mult", *cfg.lower_mult_range)
        horizon = trial.suggest_int("horizon", *cfg.horizon_range)

        if cfg.search_rr:
            rr_min = trial.suggest_float("rr_min", *cfg.rr_min_range)
            rr_max_lo = max(cfg.rr_max_range[0], rr_min + 0.2)
            rr_max = trial.suggest_float("rr_max", rr_max_lo, cfg.rr_max_range[1])
        else:
            rr_min = base_wf_config.rr_min if base_wf_config else 1.2
            rr_max = base_wf_config.rr_max if base_wf_config else 2.5

        # ── 2. Recomputar labels ──────────────────────────────────────
        try:
            raw_labels = label_fn(upper_mult, lower_mult, horizon)
        except Exception as exc:
            logger.debug(f"Trial {trial.number}: label_fn error: {exc}")
            return float("nan")

        valid_idx = raw_labels.dropna().index.intersection(X_ho.index)

        # Necesitamos suficientes barras para n_val_folds folds
        min_needed = (
            cfg.train_size
            + cfg.n_val_folds * (cfg.val_size + cfg.embargo)
        )
        if len(valid_idx) < min_needed:
            return float("nan")

        # Tomar la ventana mínima necesaria (las últimas min_needed barras)
        # -> preserva temporalidad, da exactamente n_val_folds folds
        valid_idx_sub = valid_idx[-min_needed:]
        X_sub = X_ho.loc[valid_idx_sub]
        y_sub = raw_labels.loc[valid_idx_sub].astype(int)
        prices_sub = prices_ho.reindex(valid_idx_sub)

        # ── 3. Config del mini walk-forward ───────────────────────────
        from models.walk_forward_runner import WalkForwardConfig, WalkForwardRunner

        base_rr_p_low = base_wf_config.rr_p_low if base_wf_config else 0.40
        base_rr_p_high = base_wf_config.rr_p_high if base_wf_config else 0.70

        wf_cfg = WalkForwardConfig(
            train_size=cfg.train_size,
            test_size=cfg.val_size,
            embargo=cfg.embargo,
            expanding=False,
            calib_frac=cfg.calib_frac,
            calib_method=cfg.calib_method,
            # Entry filter — liviano para velocidad
            filter_symmetric=True,
            filter_min_coverage=0.03,
            filter_n_thresholds=25,
            # Kelly / R:R
            kelly_fraction=0.25,
            max_risk_pct=0.02,
            rr_min=rr_min,
            rr_max=rr_max,
            rr_p_low=base_rr_p_low,
            rr_p_high=base_rr_p_high,
            rr_shape="sigmoid",
            atr_sl_mult=2.0,
            use_class_weights=cfg.use_class_weights,
            # Importance: OFF para velocidad
            track_importance=False,
            shap_sample_size=0,
            # Sin sizing (atr=None) -> más rápido
            instrument=None,
            xgb_params=xgb_params,
        )

        # ── 4. Mini walk-forward ──────────────────────────────────────
        try:
            runner = WalkForwardRunner(wf_cfg)
            result = runner.run(
                X=X_sub,
                y=y_sub,
                prices=prices_sub,
                atr=None,
                all_classes=all_classes,
            )
        except Exception as exc:
            logger.debug(f"Trial {trial.number}: runner error: {exc}")
            return float("nan")

        # ── 5. Calcular objetivo ──────────────────────────────────────
        n_trades = result.global_metrics.get("n_trades", 0)
        if n_trades < cfg.min_trades:
            return float("nan")

        val = self._extract_metric(result.global_metrics, cfg.objective_metric,
                                   cfg.target_coverage)

        if val is None or np.isnan(float(val)):
            return float("nan")

        return float(val)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metric(
        global_metrics: dict,
        metric_name: str,
        target_coverage: float = 0.15,
    ) -> Optional[float]:
        """Extrae y transforma la métrica objetivo de los metrics globales."""
        if metric_name == "psr":
            return global_metrics.get("psr")

        elif metric_name == "sharpe":
            s = global_metrics.get("sharpe")
            return s

        elif metric_name == "coverage_psr":
            psr = global_metrics.get("psr") or 0.0
            cov = global_metrics.get("coverage", 0.0)
            scale = min(cov / max(target_coverage, 1e-6), 1.0)
            return psr * scale

        else:
            return global_metrics.get("psr")

    def _make_sampler(self, optuna):
        """Crea el sampler de Optuna según la config."""
        s = self.cfg.sampler.lower()
        seed = self.cfg.seed
        if s == "tpe":
            return optuna.samplers.TPESampler(seed=seed)
        elif s == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        elif s == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=seed)
        else:
            logger.warning(f"Sampler '{s}' desconocido, usando TPE.")
            return optuna.samplers.TPESampler(seed=seed)

    def _make_callback(self):
        """Callback de Optuna para logging de progreso."""
        cfg = self.cfg

        def callback(study, trial):
            try:
                import optuna
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    val = trial.value
                    bp = trial.params
                    best_val = study.best_value
                    logger.warning(
                        f"[Hyperopt] Trial {trial.number+1}/{cfg.n_trials}  "
                        f"val={val:.4f}  "
                        f"best={best_val:.4f}  "
                        f"h={bp.get('horizon','?')}  "
                        f"um={bp.get('upper_mult', 0):.2f}  "
                        f"lr={bp.get('learning_rate', 0):.3f}"
                    )
            except Exception:
                pass

        return callback


# =====================================================================
# FUNCIÓN CONVENIENTE
# =====================================================================

def run_hyperopt(
    X: pd.DataFrame,
    close: pd.Series,
    atr: pd.Series,
    label_fn: Callable,
    prices: Optional[pd.Series] = None,
    base_wf_config=None,
    all_classes: Optional[list] = None,
    n_trials: int = 50,
    n_val_folds: int = 3,
    objective_metric: str = "psr",
    verbose: bool = True,
    **kwargs,
) -> HyperoptResult:
    """
    Función de conveniencia para lanzar la búsqueda bayesiana.

    Equivale a:
        cfg = HyperoptConfig(n_trials=n_trials, n_val_folds=n_val_folds, ...)
        ho = BayesianHyperopt(cfg)
        return ho.run(X, close, atr, label_fn, ...)

    Parameters
    ----------
    label_fn : Callable
        (upper_mult: float, lower_mult: float, horizon: int) -> pd.Series
        Debe devolver labels {-1, 0, +1} alineados con X.index.
        Las últimas `horizon` barras pueden ser NaN.
    **kwargs
        Cualquier campo de HyperoptConfig (e.g., symmetric_barriers=True).

    Returns
    -------
    HyperoptResult
    """
    cfg = HyperoptConfig(
        n_trials=n_trials,
        n_val_folds=n_val_folds,
        objective_metric=objective_metric,
        verbose=verbose,
        **kwargs,
    )
    ho = BayesianHyperopt(cfg)
    return ho.run(
        X=X,
        close=close,
        atr=atr,
        label_fn=label_fn,
        prices=prices,
        base_wf_config=base_wf_config,
        all_classes=all_classes,
    )
