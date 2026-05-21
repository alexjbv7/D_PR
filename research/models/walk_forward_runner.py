"""
Walk-Forward Runner — Pipeline Completo por Fold
=================================================
Ensambla todos los componentes en el orden correcto para cada fold:

  TRAIN_fit (60-80%) → TRAIN_calib (20-40%) → embargo → TEST
       │                      │                            │
  fit XGBoost          calibrar probas             evaluar OOS
                       optimizar threshold
                       (EntryFilter)

Por fold:
  1. split_train_for_calibration(X_train, y_train, calib_frac)
  2. model.fit(X_fit, y_fit)
  3. model.calibrate(X_calib, y_calib)          → P(y=c|x) reales
  4. entry_filter.fit(proba_calib, y_calib)     → threshold óptimo
  5. Per barra en TEST:
       proba    = model.predict_proba(X_test)
       signal   = entry_filter.predict(proba)
       p_win    = extract_p_win(proba, signal, class_labels)
       sizing   = compute_full_sizing(signal, p_win, ...)
  6. Registrar importancia de features (gain/SHAP)
  7. Registrar ECE/Brier del fold

Post walk-forward:
  - Agregar importancias cross-fold
  - PSR / DSR sobre la serie OOS completa
  - Resumen de calibración por fold

PROTOCOLO ANTI-LEAKAGE:
  - Calibración SOLO sobre TRAIN_calib (pasado respecto al test)
  - Entry filter optimizado SOLO sobre TRAIN_calib
  - R:R empírico actualizado SOLO con trades de folds anteriores
  - NUNCA se usa información del test para tomar decisiones
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from models.zoo import XGBoostClassifier
from models.calibration import (
    split_train_for_calibration,
    calibration_report,
    IsotonicCalibrator,
)
from models.entry_filter import ProbabilityEntryFilter, fit_entry_filter
from models.feature_selection import (
    compute_gain_importance,
    compute_shap_importance,
    aggregate_fold_importances,
    select_features_to_drop,
)
from models.validation import WalkForwardSplitter
from risk.kelly import extract_p_win
from risk.dynamic_rr import DynamicRRManager, compute_full_sizing

logger = logging.getLogger(__name__)


# =====================================================================
# PSR / DSR (Probabilistic & Deflated Sharpe Ratio)
# =====================================================================

def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    PSR(SR*) = P(SR_real > SR* | datos)

    Corrección de Mertens para distribución no-normal de retornos:
      SE(SR) = sqrt( (1 - γ₃·SR + (γ₄-1)/4·SR²) / (T-1) )

    donde γ₃ = skewness, γ₄ = kurtosis de los retornos diarios.

    Returns
    -------
    float en [0, 1]: probabilidad de que el Sharpe real > sr_benchmark.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 4:
        return 0.0

    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma < 1e-10:
        return 1.0 if mu > sr_benchmark / np.sqrt(periods_per_year) else 0.0

    sr_hat = mu / sigma * np.sqrt(periods_per_year)

    skew = float(scipy_stats.skew(r))
    kurt = float(scipy_stats.kurtosis(r, fisher=False))  # kurtosis excesiva no, normal

    # Mertens SE
    sr_normalized = sr_hat / np.sqrt(periods_per_year)  # SR en escala de 1 período
    se_sr = np.sqrt(
        max(0.0, (1 - skew * sr_normalized + (kurt - 1) / 4 * sr_normalized ** 2) / (T - 1))
    )

    if se_sr < 1e-10:
        return 1.0 if sr_hat > sr_benchmark else 0.0

    z = (sr_hat - sr_benchmark) / (se_sr * np.sqrt(periods_per_year))
    return float(scipy_stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int = 252,
) -> float:
    """
    DSR = PSR(E[max SR | n_trials])

    Corrige el sesgo de selección cuando se han probado n_trials estrategias
    y se elige la mejor. E[max SR] crece con n_trials.

    Aproximación de Bailey & López de Prado (2014):
      E[max SR] ≈ √(2·log(n)) · (1 - γ_EM/(2·log(n)) - log(log(n))/(2·log(n)))

    donde γ_EM ≈ 0.5772 (constante de Euler-Mascheroni).

    Returns float en [0, 1].
    """
    if n_trials <= 1:
        return probabilistic_sharpe_ratio(returns, 0.0, periods_per_year)

    EULER_MASCHERONI = 0.5772156649
    log_n = np.log(n_trials)

    # E[max SR] de una Normal estándar con n_trials muestras
    e_max_sr = np.sqrt(2.0 * log_n) * (
        1.0
        - EULER_MASCHERONI / (2.0 * log_n)
        - np.log(np.log(n_trials)) / (2.0 * log_n)
    )

    return probabilistic_sharpe_ratio(returns, e_max_sr, periods_per_year)


def compute_oos_metrics(
    signals: pd.Series,
    prices: pd.Series,
    periods_per_year: int = 252,
) -> dict:
    """
    Métricas OOS sobre la serie de señales y precios.

    Usa retornos diarios ponderados por señal como proxy de P&L
    (sin fees ni sizing explícito — para diagnóstico rápido).
    """
    price_returns = prices.pct_change().reindex(signals.index).fillna(0.0)
    strategy_returns = signals * price_returns

    active = strategy_returns[signals != 0]

    if len(active) < 5:
        return {"sharpe": np.nan, "psr": np.nan, "dsr": np.nan,
                "n_trades": 0, "coverage": 0.0}

    mu = float(active.mean())
    sigma = float(active.std(ddof=1)) if len(active) > 1 else 0.0
    sharpe = mu / sigma * np.sqrt(periods_per_year) if sigma > 1e-10 else np.nan

    psr = probabilistic_sharpe_ratio(active.values, 0.0, periods_per_year)
    dsr = deflated_sharpe_ratio(active.values, n_trials=1, periods_per_year=periods_per_year)

    coverage = float((signals != 0).mean())
    win_rate = float((active > 0).mean()) if len(active) > 0 else np.nan

    cumret = (1 + strategy_returns.fillna(0)).cumprod()
    rolling_max = cumret.cummax()
    drawdown = (cumret - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    return {
        "sharpe": round(sharpe, 4) if not np.isnan(sharpe) else None,
        "psr": round(psr, 4),
        "dsr": round(dsr, 4),
        "win_rate": round(win_rate, 4) if not np.isnan(win_rate) else None,
        "n_trades": int((signals != 0).sum()),
        "coverage": round(coverage, 4),
        "max_drawdown": round(max_dd, 4),
    }


# =====================================================================
# CONFIGURACIÓN DEL RUNNER
# =====================================================================

@dataclass
class WalkForwardConfig:
    """
    Configuración completa del walk-forward runner.

    Walk-forward splits
    -------------------
    train_size : int          barras de entrenamiento por fold
    test_size  : int          barras de test por fold
    embargo    : int          barras de embargo entre train y test
    expanding  : bool         True = ventana anchored; False = rolling

    Calibración
    -----------
    calib_frac : float        fracción del train para calibración (0.20)
    calib_method : str        'isotonic' | 'sigmoid'

    Entry filter
    ------------
    filter_symmetric : bool   mismo threshold para long y short
    filter_min_coverage : float
    filter_n_thresholds : int

    Kelly + Dynamic R:R
    -------------------
    kelly_fraction  : float   fracción de Kelly (0.25 = quarter Kelly)
    max_risk_pct    : float   cap duro de riesgo por trade
    rr_min / rr_max : float   rango de R:R dinámico
    rr_p_low / rr_p_high : float   rango de probabilidad para escalar R:R
    rr_shape : str            'sigmoid' | 'linear' | 'stepped'
    atr_sl_mult : float       stop en múltiplos de ATR

    Feature importance
    ------------------
    track_importance : bool   True = calcular gain + SHAP por fold
    shap_sample_size : int    muestras para SHAP (reducir si es lento)

    Instrument (para sizing; None = solo señales, sin sizing)
    ----------------------------------------------------------
    instrument : InstrumentSpec | None
    """
    # Walk-forward
    train_size: int = 252
    test_size: int = 63
    embargo: int = 5
    expanding: bool = False

    # Calibración
    calib_frac: float = 0.20
    calib_method: str = "sigmoid"

    # Entry filter
    filter_symmetric: bool = True
    filter_min_coverage: float = 0.05
    filter_n_thresholds: int = 50

    # Kelly + Dynamic R:R
    kelly_fraction: float = 0.25
    max_risk_pct: float = 0.02
    rr_min: float = 1.2
    rr_max: float = 2.5
    rr_p_low: float = 0.45
    rr_p_high: float = 0.75
    rr_shape: str = "sigmoid"
    atr_sl_mult: float = 2.0

    # Class weights (compensar desbalance de labels)
    use_class_weights: bool = True

    # Feature importance
    track_importance: bool = True
    shap_sample_size: int = 200

    # Instrumento (opcional)
    instrument: object = None   # InstrumentSpec

    # XGBoost params
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
    })

    # Regime detection (GMM)
    use_regime_features: bool = False   # True = añadir regime_prob_k como features
    regime_n_components: int = 3        # número de regímenes (usar select_n_components para elegir)

    # PCA Denoising (aplicado DESPUÉS de añadir régimen, ANTES del XGBoost)
    use_pca: bool = False
    pca_n_components: Union[int, float, str] = 0.95   # float=varianza, int=fijo, "mle"=auto

    # Meta-labeling (segundo clasificador binario que filtra señales)
    use_meta_labeling: bool = False
    meta_min_samples: int = 20   # mínimo señales activas en calib para entrenar
    meta_xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "reg_lambda": 2.0,
        "reg_alpha": 0.1,
    })

    # Métricas extendidas: Precision, Recall, F1, AUC, Confusion Matrix, Bias/Variance
    # (CS229 Tips & Tricks cheatsheet)
    track_extended_metrics: bool = True

    # Bayesian P(win) updating (combina prior de régimen + likelihood del modelo)
    # Requiere use_regime_features=True para tener regime_label en X.
    use_bayesian_sizing: bool = False
    bayesian_combination: str = "product"   # "product" | "weighted"
    bayesian_smoothing: float = 1.0         # Laplace smoothing
    bayesian_min_samples: int = 20          # min obs/celda para prior confiable
    bayesian_prior_weight: float = 0.3      # peso del prior (solo mode "weighted")

    # Selector de modelo primario (CS229 DL cheatsheet)
    # 'xgboost'  : XGBoostClassifier (default, más robusto para finanzas)
    # 'deep_mlp' : DeepMLPClassifier (BatchNorm+LeakyReLU+Dropout+Adam)
    # 'logistic' : LogisticBaseline  (baseline)
    # 'lstm'     : LSTMClassifier    (solo con >100K barras y GPU)
    model_class: str = "xgboost"

    # Parámetros para DeepMLP (solo si model_class='deep_mlp')
    mlp_params: dict = field(default_factory=lambda: {
        "hidden_dims":   [128, 64, 32],
        "dropout":       0.3,
        "learning_rate": 1e-3,
        "weight_decay":  1e-4,
        "batch_size":    256,
        "epochs":        100,
        "patience":      10,
        "device":        "cpu",
    })


# =====================================================================
# RESULTADO POR FOLD
# =====================================================================

@dataclass
class FoldResult:
    fold_idx: int
    train_start: object   # timestamp
    train_end: object
    test_start: object
    test_end: object
    n_train_fit: int
    n_train_calib: int
    n_test: int

    # Señales y probabilidades OOS
    oos_signals: pd.Series        # {-1, 0, +1}
    oos_proba: pd.DataFrame       # probabilidades calibradas (n_test × n_classes)
    oos_sizing: pd.DataFrame      # n_units, sl, tp, risk_pct por barra

    # Metadatos del fold
    threshold_long: float
    threshold_short: float
    calibration: dict             # ECE, Brier antes/después
    metrics: dict                 # sharpe, psr, coverage, n_trades
    gain_importance: pd.Series    # importancia de features en este fold

    # R:R empírico actualizado (tras este fold)
    rr_empirical: Optional[float] = None

    # Resumen de regímenes detectados en el test (None si use_regime_features=False)
    regime_labels: Optional[pd.Series] = None

    # True si se entrenó y usó el meta-labeler en este fold
    meta_labeler_trained: bool = False

    # True si el BayesianWinUpdater se ajustó con éxito en este fold
    bayesian_sizer_fitted: bool = False

    # Métricas extendidas (CS229): precision/recall/F1/AUC + bias-variance
    classification_metrics: Optional[object] = None   # FoldClassificationMetrics
    bias_variance: Optional[dict] = None              # train_acc/calib_acc/test_acc/verdict


# =====================================================================
# RESULTADO GLOBAL
# =====================================================================

@dataclass
class WalkForwardResult:
    config: WalkForwardConfig
    fold_results: List[FoldResult]

    # OOS completo (concatenación de todos los folds)
    oos_signals: pd.Series
    oos_proba: pd.DataFrame
    oos_sizing: pd.DataFrame

    # Feature importance agregada cross-fold
    feature_importance_agg: pd.DataFrame
    features_to_drop: List[str]

    # Métricas globales
    global_metrics: dict

    # Serie de regímenes OOS (None si use_regime_features=False)
    oos_regimes: Optional[pd.Series] = None

    def summary(self) -> str:
        lines = [
            "=" * 65,
            " WALK-FORWARD RESULT SUMMARY",
            "=" * 65,
            f" Folds:        {len(self.fold_results)}",
            f" OOS barras:   {len(self.oos_signals)}",
            f" OOS trades:   {self.global_metrics.get('n_trades', 'N/A')}",
            f" Coverage:     {self.global_metrics.get('coverage', 0):.1%}",
            "",
            " MÉTRICAS OOS GLOBALES:",
            f"   Sharpe:      {self.global_metrics.get('sharpe', 'N/A')}",
            f"   PSR(SR>0):   {self.global_metrics.get('psr', 'N/A'):.2%}" if self.global_metrics.get('psr') else "   PSR:         N/A",
            f"   DSR(SR>0):   {self.global_metrics.get('dsr', 'N/A'):.2%}" if self.global_metrics.get('dsr') else "   DSR:         N/A",
            f"   Win Rate:    {self.global_metrics.get('win_rate', 'N/A')}",
            f"   Max DD:      {self.global_metrics.get('max_drawdown', 'N/A')}",
            "",
            " POR FOLD:",
        ]
        for fr in self.fold_results:
            verdict = "OK" if fr.calibration.get("verdict") == "OK" else fr.calibration.get("verdict", "?")
            lines.append(
                f"   Fold {fr.fold_idx+1:>2}: "
                f"thresh={fr.threshold_long:.3f}  "
                f"trades={fr.metrics.get('n_trades', 0):>4}  "
                f"sharpe={fr.metrics.get('sharpe') or 'N/A':>6}  "
                f"ECE={fr.calibration.get('ece_calibrated', 'N/A')}  "
                f"calib={verdict}"
            )
        lines += [
            "",
            f" FEATURES A ELIMINAR ({len(self.features_to_drop)}): "
            + (", ".join(self.features_to_drop[:10]) or "ninguna"),
            "=" * 65,
        ]
        return "\n".join(lines)


# =====================================================================
# WALK-FORWARD RUNNER
# =====================================================================

class WalkForwardRunner:
    """
    Runner de walk-forward con pipeline completo.

    Uso:
        cfg = WalkForwardConfig(train_size=252, test_size=63, embargo=5)
        runner = WalkForwardRunner(cfg)
        result = runner.run(X, y, prices=close_series)
        print(result.summary())
    """

    def __init__(self, config: WalkForwardConfig):
        self.cfg = config
        self._rr_manager = DynamicRRManager(
            atr_sl_mult=config.atr_sl_mult,
            rr_min=config.rr_min,
            rr_max=config.rr_max,
            p_low=config.rr_p_low,
            p_high=config.rr_p_high,
            shape=config.rr_shape,
        )
        self._accumulated_trade_returns: list = []  # para R:R empírico rolling

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prices: Optional[pd.Series] = None,
        atr: Optional[pd.Series] = None,
        all_classes: Optional[list] = None,
    ) -> WalkForwardResult:
        """
        Ejecuta el walk-forward completo.

        Parameters
        ----------
        X : pd.DataFrame
            Features (índice temporal).
        y : pd.Series
            Labels {-1, 0, +1} (índice temporal).
        prices : pd.Series, opcional
            Precios de cierre para métricas OOS y sizing.
        atr : pd.Series, opcional
            ATR precalculado. Si None, no se hace sizing.
        all_classes : list, opcional
            Clases esperadas. Default: sorted(unique(y)).
        """
        cfg = self.cfg
        if all_classes is None:
            all_classes = sorted(y.unique().tolist())

        splitter = WalkForwardSplitter(
            train_size=cfg.train_size,
            test_size=cfg.test_size,
            embargo=cfg.embargo,
            expanding=cfg.expanding,
        )
        n_splits = splitter.get_n_splits(X)
        logger.info(f"Walk-forward: {n_splits} folds  "
                    f"(train={cfg.train_size}, test={cfg.test_size}, embargo={cfg.embargo})")

        fold_results: List[FoldResult] = []
        gain_importances_per_fold: list = []

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
            logger.info(f"\n--- Fold {fold_idx+1}/{n_splits} ---")
            fold_result = self._run_fold(
                fold_idx=fold_idx,
                X=X, y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                prices=prices,
                atr=atr,
                all_classes=all_classes,
            )
            fold_results.append(fold_result)
            if fold_result.gain_importance is not None:
                gain_importances_per_fold.append(fold_result.gain_importance)

        # ── Concatenar resultados OOS ──────────────────────────────────
        oos_signals = pd.concat([fr.oos_signals for fr in fold_results]).sort_index()
        oos_proba = pd.concat([fr.oos_proba for fr in fold_results]).sort_index()
        oos_sizing = pd.concat([fr.oos_sizing for fr in fold_results]).sort_index()

        # ── Feature importance cross-fold ──────────────────────────────
        if gain_importances_per_fold:
            feat_agg = aggregate_fold_importances(gain_importances_per_fold)
            features_to_drop = select_features_to_drop(feat_agg)
        else:
            feat_agg = pd.DataFrame()
            features_to_drop = []

        # ── Métricas globales OOS ──────────────────────────────────────
        if prices is not None:
            global_metrics = compute_oos_metrics(
                oos_signals, prices.reindex(oos_signals.index), 252
            )
        else:
            global_metrics = {"n_trades": int((oos_signals != 0).sum()),
                              "coverage": float((oos_signals != 0).mean())}

        # DSR con n_trials = n_splits (probamos n estrategias independientes)
        if prices is not None and "psr" in global_metrics:
            price_ret = prices.pct_change().reindex(oos_signals.index).fillna(0)
            strat_ret = (oos_signals * price_ret)[oos_signals != 0]
            if len(strat_ret) >= 5:
                global_metrics["dsr"] = round(
                    deflated_sharpe_ratio(strat_ret.values, n_trials=n_splits), 4
                )

        # Concatenar regímenes OOS si están disponibles
        regime_series_list = [
            fr.regime_labels for fr in fold_results if fr.regime_labels is not None
        ]
        oos_regimes = (
            pd.concat(regime_series_list).sort_index()
            if regime_series_list else None
        )

        result = WalkForwardResult(
            config=cfg,
            fold_results=fold_results,
            oos_signals=oos_signals,
            oos_proba=oos_proba,
            oos_sizing=oos_sizing,
            feature_importance_agg=feat_agg,
            features_to_drop=features_to_drop,
            global_metrics=global_metrics,
            oos_regimes=oos_regimes,
        )

        logger.info("\n" + result.summary())
        return result

    # ------------------------------------------------------------------
    # POR FOLD
    # ------------------------------------------------------------------

    def _run_fold(
        self,
        fold_idx: int,
        X: pd.DataFrame,
        y: pd.Series,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        prices: Optional[pd.Series],
        atr: Optional[pd.Series],
        all_classes: list,
    ) -> FoldResult:
        cfg = self.cfg

        X_train_full = X.iloc[train_idx]
        y_train_full = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        logger.info(
            f"  Train: {X_train_full.index[0].date()} → {X_train_full.index[-1].date()} "
            f"({len(train_idx)} barras)"
        )
        logger.info(
            f"  Test:  {X_test.index[0].date()} → {X_test.index[-1].date()} "
            f"({len(test_idx)} barras)"
        )

        # ── 1. Split train → fit + calib ──────────────────────────────
        X_fit, y_fit, X_calib, y_calib = split_train_for_calibration(
            X_train_full, y_train_full, calib_frac=cfg.calib_frac
        )
        logger.info(f"  Train split: {len(X_fit)} fit + {len(X_calib)} calib")

        # ── 1b. Regime features (GMM) ─────────────────────────────────
        # Anti-leakage: GMM se ajusta SOLO sobre X_train_full (train completo),
        # scaler incluido. Transform se aplica a cada subset por separado.
        regime_labels_test: Optional[pd.Series] = None
        if cfg.use_regime_features and prices is not None and atr is not None:
            X_fit, X_calib, X_test, regime_labels_test = self._add_regime_features(
                X_fit=X_fit,
                X_calib=X_calib,
                X_test=X_test,
                X_train_full=X_train_full,
                prices=prices,
                atr=atr,
                n_components=cfg.regime_n_components,
            )
            logger.info(f"  Regime features añadidas ({cfg.regime_n_components} componentes)")

        # ── 1c. PCA Denoising ──────────────────────────────────────────
        # Anti-leakage: PCA + scaler se ajustan SOLO sobre X_fit.
        # Las columnas de régimen (exclude_prefix="regime_") se excluyen del PCA
        # y se pasan directamente — preservan su interpretabilidad probabilística.
        if cfg.use_pca:
            X_fit, X_calib, X_test = self._add_pca_features(
                X_fit=X_fit,
                X_calib=X_calib,
                X_test=X_test,
                n_components=cfg.pca_n_components,
            )
            logger.info(
                f"  PCA aplicado: {X_fit.shape[1]} columnas resultantes "
                f"(n_components={cfg.pca_n_components})"
            )

        # ── 2. Fit modelo primario ────────────────────────────────────
        from models.zoo import get_model
        model_cls = cfg.model_class.lower()

        if model_cls == "xgboost":
            model = get_model("xgboost", **cfg.xgb_params)
        elif model_cls == "deep_mlp":
            model = get_model("deep_mlp", **cfg.mlp_params)
        elif model_cls in ("logistic", "deep_mlp", "lstm"):
            model = get_model(model_cls)
        else:
            logger.warning(
                f"model_class='{cfg.model_class}' desconocido, usando xgboost"
            )
            model = get_model("xgboost", **cfg.xgb_params)

        logger.info(f"  Modelo: {model.name}  (fold {fold_idx})")

        # Class weights: compensan desbalance en labels (ej. 68% neutros)
        sample_weight = None
        if cfg.use_class_weights:
            try:
                from sklearn.utils.class_weight import compute_sample_weight
                sample_weight = compute_sample_weight("balanced", y_fit.values)
            except Exception as e:
                logger.warning(f"  No se pudieron calcular class weights: {e}")

        # fit() — distintos modelos tienen distintas firmas; normalizamos aquí
        if model_cls == "xgboost":
            model.fit(X_fit, y_fit, all_classes=all_classes,
                      sample_weight=sample_weight)
        elif model_cls == "deep_mlp":
            # DeepMLP acepta eval_set para early stopping (usa calib como val)
            model.fit(X_fit, y_fit, sample_weight=sample_weight,
                      eval_set=(X_calib, y_calib), all_classes=all_classes)
        else:
            model.fit(X_fit, y_fit, sample_weight=sample_weight)

        # ── 3. Calibrar ───────────────────────────────────────────────
        model.calibrate(X_calib, y_calib, method=cfg.calib_method)

        # Calibration report: usar predict_proba_raw() para uncalibrated
        proba_calib_uncal = model.predict_proba_raw(X_calib)
        proba_calib_cal = model.predict_proba(X_calib)

        # Usar clase +1 para ECE one-vs-rest (índice de la clase +1)
        _lm = getattr(model, 'label_map_', None)
        if _lm is not None:
            class_labels = sorted(_lm.keys())
        else:
            class_labels = sorted(np.unique(y_fit.values))
        idx_pos = class_labels.index(1) if 1 in class_labels else 0
        cal_report = calibration_report(
            y_true=(y_calib.values == 1).astype(float),
            y_proba_uncal=proba_calib_uncal[:, idx_pos],
            y_proba_cal=proba_calib_cal[:, idx_pos],
            label=f"fold_{fold_idx+1}",
        )

        # ── 4. Entry filter ───────────────────────────────────────────
        # min_samples_to_optimize adaptativo: siempre intenta optimizar
        # si hay al menos 20 muestras (nunca usa fallback por datasets pequeños)
        min_to_opt = max(20, int(len(X_calib) * 0.7))
        entry_filter = ProbabilityEntryFilter(
            class_labels=class_labels,
            symmetric=cfg.filter_symmetric,
            min_coverage=cfg.filter_min_coverage,
            n_thresholds=cfg.filter_n_thresholds,
            min_samples_to_optimize=min_to_opt,
            fallback_threshold=1.0 / len(class_labels) + 0.05,  # ligeramente > azar
        )
        entry_filter.fit(proba_calib_cal, y_calib.values)

        logger.info(
            f"  Threshold: long={entry_filter.threshold_long_:.3f}  "
            f"short={entry_filter.threshold_short_:.3f}"
        )

        # ── 4b. Meta-labeler ──────────────────────────────────────────
        # Entrenado sobre X_calib donde primario tuvo señal activa.
        # Anti-leakage: usa sólo predicciones del primario sobre calib,
        # nunca etiquetas del test.
        meta_labeler = None
        if cfg.use_meta_labeling:
            meta_labeler = self._fit_meta_labeler(
                X_calib=X_calib,
                proba_calib=proba_calib_cal,
                y_calib=y_calib,
                entry_filter=entry_filter,
                class_labels=class_labels,
            )
            if meta_labeler is not None and meta_labeler.is_fitted:
                logger.info(
                    f"  MetaLabeler entrenado "
                    f"({meta_labeler.n_active_train} señales activas en calib)"
                )
            else:
                logger.info("  MetaLabeler NO entrenado (pocas señales activas en calib)")

        # ── 4c. Bayesian Win Updater ──────────────────────────────────
        # Estima P(win|régimen, dirección) en calib → combinación con
        # p_model en test vía product-of-experts.
        # Requiere use_regime_features=True (necesita columna regime_label).
        bayesian_updater = None
        if cfg.use_bayesian_sizing:
            bayesian_updater = self._fit_bayesian_updater(
                X_calib=X_calib,
                y_calib=y_calib,
                entry_filter=entry_filter,
                proba_calib=proba_calib_cal,
            )
            if bayesian_updater is not None and bayesian_updater.is_fitted:
                logger.info(
                    f"  BayesianWinUpdater ajustado "
                    f"({bayesian_updater.n_regimes_} régimens)"
                )
            else:
                logger.info(
                    "  BayesianWinUpdater NO ajustado "
                    "(régimen no disponible o pocas muestras)"
                )

        # ── 5. Predicciones en TEST ───────────────────────────────────
        proba_test = model.predict_proba(X_test)
        signals_arr = entry_filter.predict(proba_test)
        oos_signals = pd.Series(signals_arr, index=X_test.index, name="signal")
        oos_proba = pd.DataFrame(
            proba_test, index=X_test.index, columns=[str(c) for c in class_labels]
        )

        # ── 6. Sizing por barra ───────────────────────────────────────
        # p_win pipeline:
        #   base    : extract_p_win(primario)
        #   step 1  : meta-labeler.predict_p_correct()  (si disponible)
        #   step 2  : bayesian_updater.update()          (si disponible)
        meta_p_win_arr = None
        if meta_labeler is not None and meta_labeler.is_fitted:
            meta_p_win_arr = meta_labeler.predict_p_correct(
                X=X_test,
                primary_proba=proba_test,
                primary_signals=signals_arr,
            )

        # Bayesian update sobre lo que tengamos (meta o base)
        if (bayesian_updater is not None
                and bayesian_updater.is_fitted
                and meta_p_win_arr is not None):
            meta_p_win_arr = bayesian_updater.update(
                p_model=meta_p_win_arr,
                X=X_test,
                signals=signals_arr,
            )
        elif bayesian_updater is not None and bayesian_updater.is_fitted:
            # No hay meta-labeler → aplicar Bayes sobre p_win del primario
            base_p_win = np.array([
                extract_p_win(proba_test[i], int(signals_arr[i]), class_labels)
                if signals_arr[i] != 0 else 0.0
                for i in range(len(signals_arr))
            ])
            meta_p_win_arr = bayesian_updater.update(
                p_model=base_p_win,
                X=X_test,
                signals=signals_arr,
            )

        oos_sizing = self._compute_sizing_series(
            signals_arr=signals_arr,
            proba_arr=proba_test,
            index=X_test.index,
            class_labels=class_labels,
            prices=prices,
            atr=atr,
            meta_p_win_arr=meta_p_win_arr,
        )

        # ── 7. Feature importance ─────────────────────────────────────
        gain_imp = None
        if cfg.track_importance:
            try:
                gain_imp = compute_gain_importance(model)
            except Exception as e:
                logger.warning(f"  Gain importance falló en fold {fold_idx+1}: {e}")

        # ── 8. Métricas del fold ──────────────────────────────────────
        if prices is not None:
            fold_metrics = compute_oos_metrics(
                oos_signals, prices.reindex(X_test.index), 252
            )
        else:
            fold_metrics = {
                "n_trades": int((signals_arr != 0).sum()),
                "coverage": float((signals_arr != 0).mean()),
            }

        logger.info(
            f"  Trades: {fold_metrics.get('n_trades', 0)}  "
            f"Coverage: {fold_metrics.get('coverage', 0):.1%}  "
            f"ECE: {cal_report.get('ece_calibrated', 'N/A')}  "
            f"[{cal_report.get('verdict', '?')}]"
        )

        # ── 9. Extended metrics: Precision/Recall/F1/AUC + Bias-Variance ─
        # (CS229 Tips cheatsheet: classification metrics + diagnostics)
        classification_metrics = None
        bias_variance_info = None
        if cfg.track_extended_metrics:
            try:
                from models.metrics import (
                    compute_classification_metrics,
                    compute_bias_variance_verdict,
                )
                # Predictions on test for classification metrics
                y_pred_test = model.predict(X_test)
                classification_metrics = compute_classification_metrics(
                    y_true=y_test.values,
                    y_pred=y_pred_test,
                    y_proba=proba_test,
                    class_labels=class_labels,
                    signals=signals_arr,
                )

                # Bias/Variance: accuracy on train_fit, calib, test
                y_pred_fit   = model.predict(X_fit)
                y_pred_calib = model.predict(X_calib)
                from sklearn.metrics import accuracy_score
                train_acc = float(accuracy_score(y_fit.values, y_pred_fit))
                calib_acc = float(accuracy_score(y_calib.values, y_pred_calib))
                test_acc  = float(accuracy_score(y_test.values, y_pred_test))
                verdict = compute_bias_variance_verdict(train_acc, calib_acc, test_acc)
                bias_variance_info = {
                    "train_acc": round(train_acc, 4),
                    "calib_acc": round(calib_acc, 4),
                    "test_acc":  round(test_acc, 4),
                    "gap":       round(train_acc - test_acc, 4),
                    "verdict":   verdict,
                }
                logger.info(
                    f"  F1(macro)={classification_metrics.f1_macro}  "
                    f"AUC={classification_metrics.auc_macro}  "
                    f"B/V={verdict} "
                    f"(train={train_acc:.2%} calib={calib_acc:.2%} test={test_acc:.2%})"
                )
            except Exception as e:
                logger.warning(f"  Extended metrics falló en fold {fold_idx+1}: {e}")

        meta_trained = (
            meta_labeler is not None and meta_labeler.is_fitted
        ) if cfg.use_meta_labeling else False

        bayes_fitted = (
            bayesian_updater is not None and bayesian_updater.is_fitted
        ) if cfg.use_bayesian_sizing else False

        return FoldResult(
            fold_idx=fold_idx,
            train_start=X_train_full.index[0],
            train_end=X_train_full.index[-1],
            test_start=X_test.index[0],
            test_end=X_test.index[-1],
            n_train_fit=len(X_fit),
            n_train_calib=len(X_calib),
            n_test=len(X_test),
            oos_signals=oos_signals,
            oos_proba=oos_proba,
            oos_sizing=oos_sizing,
            threshold_long=entry_filter.threshold_long_,
            threshold_short=entry_filter.threshold_short_,
            calibration=cal_report,
            metrics=fold_metrics,
            gain_importance=gain_imp,
            regime_labels=regime_labels_test,
            meta_labeler_trained=meta_trained,
            bayesian_sizer_fitted=bayes_fitted,
            classification_metrics=classification_metrics,
            bias_variance=bias_variance_info,
        )

    def _add_regime_features(
        self,
        X_fit: pd.DataFrame,
        X_calib: pd.DataFrame,
        X_test: pd.DataFrame,
        X_train_full: pd.DataFrame,
        prices: pd.Series,
        atr: pd.Series,
        n_components: int,
    ):
        """
        Ajusta un GMMRegimeDetector sobre el train completo y augmenta
        X_fit, X_calib y X_test con las features de régimen.

        Returns (X_fit_aug, X_calib_aug, X_test_aug, regime_labels_test).
        """
        from features.regime_gmm import GMMRegimeDetector, GMMRegimeConfig

        det = GMMRegimeDetector(GMMRegimeConfig(n_components=n_components))
        det.fit(
            prices.reindex(X_train_full.index),
            atr.reindex(X_train_full.index),
        )

        def _aug(X_sub: pd.DataFrame) -> pd.DataFrame:
            regime_feats = det.transform(
                prices.reindex(X_sub.index),
                atr.reindex(X_sub.index),
            ).reindex(X_sub.index)
            return pd.concat([X_sub, regime_feats], axis=1)

        X_fit_aug   = _aug(X_fit)
        X_calib_aug = _aug(X_calib)
        X_test_aug  = _aug(X_test)

        # Serie de labels de régimen del test (para dashboard / análisis)
        regime_labels_test = pd.Series(
            det.transform(
                prices.reindex(X_test.index),
                atr.reindex(X_test.index),
            ).reindex(X_test.index)["regime_label"].values,
            index=X_test.index,
            name="regime_label",
        )

        return X_fit_aug, X_calib_aug, X_test_aug, regime_labels_test

    def _add_pca_features(
        self,
        X_fit: pd.DataFrame,
        X_calib: pd.DataFrame,
        X_test: pd.DataFrame,
        n_components: Union[int, float, str] = 0.95,
    ):
        """
        Aplica PCA Denoising a X_fit, X_calib y X_test.

        Protocolo anti-leakage:
          - PCADenoiser.fit() SOLO sobre X_fit (sin calib ni test)
          - StandardScaler ajustado solo sobre X_fit
          - transform() aplicado por separado a cada subset

        Las columnas con prefijo "regime_" se excluyen del PCA y se
        concatenan intactas al resultado (probabilidades interpretables).

        Returns (X_fit_pca, X_calib_pca, X_test_pca).
        """
        from features.pca_denoiser import PCADenoiser, PCAConfig

        denoiser = PCADenoiser(PCAConfig(n_components=n_components))
        denoiser.fit(X_fit)

        X_fit_pca   = denoiser.transform(X_fit)
        X_calib_pca = denoiser.transform(X_calib)
        X_test_pca  = denoiser.transform(X_test)

        logger.debug(
            "  PCA: %d features → %d PCs (var_explicada=%.1%%)",
            len(denoiser._feature_cols),
            denoiser.n_components_,
            denoiser.cumulative_variance_ * 100,
        )

        return X_fit_pca, X_calib_pca, X_test_pca

    def _fit_meta_labeler(
        self,
        X_calib: pd.DataFrame,
        proba_calib: np.ndarray,
        y_calib: pd.Series,
        entry_filter,
        class_labels: list,
    ):
        """
        Entrena el MetaLabeler sobre X_calib.

        Pasos:
          1. Obtener señales del primario en calib (via entry_filter)
          2. Crear meta-labels: 1 si señal == y_real, 0 si no
          3. Fit MetaLabeler sobre las señales activas

        Returns MetaLabeler (puede no estar fitted si hay pocos ejemplos).
        """
        from models.meta_labeler import MetaLabeler, MetaLabelConfig

        cfg = self.cfg
        signals_calib = entry_filter.predict(proba_calib)

        meta = MetaLabeler(MetaLabelConfig(
            xgb_params=cfg.meta_xgb_params,
            min_samples=cfg.meta_min_samples,
        ))
        meta.fit(
            X=X_calib,
            primary_proba=proba_calib,
            primary_signals=signals_calib,
            y_true=y_calib.values,
            class_labels=class_labels,
        )
        return meta

    def _fit_bayesian_updater(
        self,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
        entry_filter,
        proba_calib: np.ndarray,
    ):
        """
        Ajusta BayesianWinUpdater sobre X_calib.

        Requiere que X_calib contenga la columna "regime_label"
        (generada por _add_regime_features). Si no está, devuelve
        un updater no ajustado (fallback silencioso).
        """
        from risk.bayesian_sizer import BayesianWinUpdater, BayesianSizerConfig

        cfg = self.cfg
        signals_calib = entry_filter.predict(proba_calib)

        updater = BayesianWinUpdater(BayesianSizerConfig(
            combination=cfg.bayesian_combination,
            smoothing=cfg.bayesian_smoothing,
            min_samples=cfg.bayesian_min_samples,
            prior_weight=cfg.bayesian_prior_weight,
        ))
        updater.fit(X_calib, y_calib.values, signals_calib)
        return updater

    def _compute_sizing_series(
        self,
        signals_arr: np.ndarray,
        proba_arr: np.ndarray,
        index: pd.Index,
        class_labels: list,
        prices: Optional[pd.Series],
        atr: Optional[pd.Series],
        meta_p_win_arr: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Calcula sizing bar-a-bar para el test del fold.

        p_win por barra:
          - Si meta_p_win_arr disponible: usa P(meta=correcto) del meta-labeler.
          - Si no: usa extract_p_win(probabilidades del primario).
        """
        records = []
        for i, idx in enumerate(index):
            sig = int(signals_arr[i])
            p_win = 0.0
            if sig != 0:
                if meta_p_win_arr is not None:
                    p_win = float(meta_p_win_arr[i])
                else:
                    try:
                        p_win = extract_p_win(proba_arr[i], sig, class_labels)
                    except Exception:
                        p_win = 0.0

            if (sig != 0
                    and self.cfg.instrument is not None
                    and prices is not None
                    and atr is not None
                    and idx in prices.index
                    and idx in atr.index):
                price = float(prices.loc[idx])
                atr_val = float(atr.loc[idx])
                sizing = compute_full_sizing(
                    signal=sig,
                    p_win=p_win,
                    current_equity=100_000.0,  # placeholder; reemplazar con equity real
                    current_price=price,
                    current_atr=atr_val,
                    instrument=self.cfg.instrument,
                    rr_manager=self._rr_manager,
                    kelly_fraction=self.cfg.kelly_fraction,
                    max_risk_pct=self.cfg.max_risk_pct,
                )
            else:
                sizing = {
                    "n_units": 0.0, "sl_price": None, "tp_price": None,
                    "risk_pct": 0.0, "risk_usd": 0.0,
                    "rr_dynamic": None, "kelly_raw": 0.0,
                }

            records.append({
                "signal": sig,
                "p_win": round(p_win, 4),
                **sizing,
            })

        return pd.DataFrame(records, index=index)
