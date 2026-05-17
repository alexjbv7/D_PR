"""
Charts — Constructores de gráficos Plotly reutilizables
=======================================================
Todas las funciones reciben datos puros (DataFrames, dicts, listas)
y devuelven plotly.graph_objects.Figure.

Paleta: tema oscuro coherente con el estilo quant.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────
# Constantes de estilo
# ─────────────────────────────────────────────────────────────────────
_BG = "#0e1117"
_PAPER = "#0e1117"
_GRID = "#1e2130"
_TEXT = "#c8ccd4"
_BLUE = "#4c8ef7"
_GREEN = "#26c281"
_RED = "#e05252"
_YELLOW = "#f5c842"
_GREY = "#4a4f5c"

_LAYOUT_BASE = dict(
    paper_bgcolor=_PAPER,
    plot_bgcolor=_BG,
    font=dict(color=_TEXT, size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
    yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
)


# =====================================================================
# EQUITY CURVE
# =====================================================================

def equity_curve(
    oos_signals: pd.Series,
    prices: pd.Series,
    title: str = "Equity Curve OOS",
    fold_boundaries: Optional[List] = None,
) -> go.Figure:
    """
    Curva de equity acumulada de la estrategia vs buy & hold.
    Panel inferior: drawdown.
    """
    price_ret = prices.pct_change().reindex(oos_signals.index).fillna(0.0)
    strat_ret = (oos_signals * price_ret).fillna(0.0)

    strat_cum = (1 + strat_ret).cumprod()
    bh_cum = (1 + price_ret).cumprod()

    # Drawdown
    roll_max = strat_cum.cummax()
    dd = (strat_cum - roll_max) / roll_max * 100  # %

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )

    # Estrategia
    fig.add_trace(go.Scatter(
        x=strat_cum.index, y=strat_cum.values,
        name="Estrategia", line=dict(color=_BLUE, width=1.8),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # Buy & hold
    fig.add_trace(go.Scatter(
        x=bh_cum.index, y=bh_cum.values,
        name="Buy & Hold", line=dict(color=_GREY, width=1.2, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # Línea base = 1
    fig.add_hline(y=1.0, line_color=_GREY, line_dash="dash", line_width=0.8, row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name="Drawdown %", fill="tozeroy",
        line=dict(color=_RED, width=1),
        fillcolor="rgba(224,82,82,0.15)",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    # Líneas de fold
    if fold_boundaries:
        for dt in fold_boundaries:
            fig.add_vline(
                x=dt, line_color=_YELLOW, line_dash="dot",
                line_width=0.7, opacity=0.4,
            )

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(size=15, color=_TEXT)),
        legend=dict(bgcolor="rgba(0,0,0,0)", x=0.01, y=0.99),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Retorno acumulado", row=1, col=1,
                     gridcolor=_GRID, zerolinecolor=_GRID)
    fig.update_yaxes(title_text="DD %", row=2, col=1,
                     gridcolor=_GRID, zerolinecolor=_GRID)
    return fig


# =====================================================================
# FOLD METRICS
# =====================================================================

def fold_metrics_chart(fold_results: list, title: str = "Métricas por Fold") -> go.Figure:
    """
    Gráfico de barras agrupadas: Sharpe y PSR*2 por fold.
    Barras coloreadas: verde si Sharpe > 0, rojo si < 0.
    """
    fold_nums = [f"F{fr.fold_idx+1}" for fr in fold_results]
    sharpes = [fr.metrics.get("sharpe") or 0.0 for fr in fold_results]
    psrs = [(fr.metrics.get("psr") or 0.0) * 2 for fr in fold_results]   # escalar para comparar
    coverages = [fr.metrics.get("coverage", 0.0) * 100 for fr in fold_results]
    n_trades = [fr.metrics.get("n_trades", 0) for fr in fold_results]

    colors_sharpe = [_GREEN if s > 0 else _RED for s in sharpes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Sharpe",
        x=fold_nums, y=sharpes,
        marker_color=colors_sharpe,
        text=[f"{s:.2f}" for s in sharpes],
        textposition="outside",
        hovertemplate="Fold %{x}<br>Sharpe: %{y:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="PSR × 2",
        x=fold_nums, y=psrs,
        marker_color=_BLUE,
        opacity=0.65,
        hovertemplate="Fold %{x}<br>PSR×2: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_color=_GREY, line_width=1.2)

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(size=15)),
        barmode="group",
        bargap=0.20,
        bargroupgap=0.05,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def fold_coverage_chart(fold_results: list) -> go.Figure:
    """Coverage y n_trades por fold."""
    fold_nums = [f"F{fr.fold_idx+1}" for fr in fold_results]
    coverages = [fr.metrics.get("coverage", 0.0) * 100 for fr in fold_results]
    n_trades = [fr.metrics.get("n_trades", 0) for fr in fold_results]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        name="Coverage %",
        x=fold_nums, y=coverages,
        marker_color=_BLUE, opacity=0.7,
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        name="N Trades",
        x=fold_nums, y=n_trades,
        mode="lines+markers",
        line=dict(color=_YELLOW, width=1.8),
        hovertemplate="%{x}: %{y} trades<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Coverage y Trades por Fold", font=dict(size=15)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_yaxes(title_text="Coverage %", secondary_y=False, gridcolor=_GRID)
    fig.update_yaxes(title_text="N Trades", secondary_y=True, gridcolor=_GRID)
    return fig


# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================

def feature_importance_chart(feat_agg: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Barras horizontales: top N features por importancia mediana."""
    if feat_agg.empty:
        return go.Figure()

    df = feat_agg.head(top_n).copy()
    df = df.sort_values("median_importance")  # ascendente para que top quede arriba

    colors = [
        _RED if row.get("frac_folds_nonzero", 1) < 0.5 else _BLUE
        for _, row in df.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=df["median_importance"],
        y=df.index.tolist(),
        orientation="h",
        marker_color=colors,
        error_x=dict(
            type="data",
            array=df.get("std_importance", pd.Series([0] * len(df))).tolist(),
            color=_GREY,
        ),
        hovertemplate="<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=f"Feature Importance (top {top_n}, mediana cross-fold)",
                   font=dict(size=15)),
        height=max(300, top_n * 32),
        xaxis_title="Importancia mediana",
        yaxis=dict(gridcolor=_GRID, tickfont=dict(size=11)),
    )
    return fig


# =====================================================================
# CALIBRACIÓN
# =====================================================================

def calibration_ece_chart(fold_results: list) -> go.Figure:
    """ECE antes y después de calibración por fold."""
    fold_nums = [f"F{fr.fold_idx+1}" for fr in fold_results]
    ece_before = []
    ece_after = []
    for fr in fold_results:
        ece_before.append(fr.calibration.get("ece_uncalibrated") or 0.0)
        ece_after.append(fr.calibration.get("ece_calibrated") or 0.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fold_nums, y=ece_before,
        name="ECE sin calibrar",
        mode="lines+markers",
        line=dict(color=_RED, width=1.8),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=fold_nums, y=ece_after,
        name="ECE calibrado",
        mode="lines+markers",
        line=dict(color=_GREEN, width=1.8),
        marker=dict(size=7),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="ECE antes/después de calibración por fold",
                   font=dict(size=15)),
        yaxis_title="ECE (menor = mejor)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# =====================================================================
# DISTRIBUCIÓN DE LABELS
# =====================================================================

def label_distribution_chart(y: pd.Series) -> go.Figure:
    """Pie chart de distribución de labels {-1, 0, +1}."""
    counts = y.value_counts().sort_index()
    labels_map = {-1: "Short (-1)", 0: "Neutral (0)", 1: "Long (+1)"}
    labels = [labels_map.get(k, str(k)) for k in counts.index]
    colors = [_RED, _GREY, _GREEN]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts.values,
        hole=0.42,
        marker_colors=colors,
        textfont=dict(size=13),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Distribución de Labels", font=dict(size=15)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# =====================================================================
# P(WIN) HISTOGRAM
# =====================================================================

def p_win_histogram(oos_sizing: pd.DataFrame) -> go.Figure:
    """Histograma de P(win) para señales activas."""
    active = oos_sizing[oos_sizing["signal"] != 0]["p_win"].dropna()
    if active.empty:
        return go.Figure()

    long_mask = oos_sizing.loc[active.index, "signal"] > 0

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=active[long_mask].values,
        name="Long",
        nbinsx=25,
        marker_color=_GREEN,
        opacity=0.75,
    ))
    fig.add_trace(go.Histogram(
        x=active[~long_mask].values,
        name="Short",
        nbinsx=25,
        marker_color=_RED,
        opacity=0.75,
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Distribución P(win) — señales activas", font=dict(size=15)),
        barmode="overlay",
        xaxis_title="P(win)",
        yaxis_title="Frecuencia",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# =====================================================================
# PRICE CHART (OHLCV)
# =====================================================================

def price_chart(
    prices: pd.DataFrame,
    signals: Optional[pd.Series] = None,
    title: str = "Precios OHLCV",
) -> go.Figure:
    """Candlestick con señales de entrada superpuestas."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=prices.index,
        open=prices["open"],
        high=prices["high"],
        low=prices["low"],
        close=prices["close"],
        name="OHLCV",
        increasing_line_color=_GREEN,
        decreasing_line_color=_RED,
        increasing_fillcolor=_GREEN,
        decreasing_fillcolor=_RED,
    ))

    if signals is not None:
        # Long entries
        long_idx = signals.index[signals == 1]
        if len(long_idx) > 0 and "close" in prices.columns:
            long_prices = prices["close"].reindex(long_idx)
            fig.add_trace(go.Scatter(
                x=long_idx, y=long_prices.values * 0.9995,
                mode="markers",
                name="Long",
                marker=dict(symbol="triangle-up", size=8, color=_GREEN),
            ))
        # Short entries
        short_idx = signals.index[signals == -1]
        if len(short_idx) > 0:
            short_prices = prices["close"].reindex(short_idx)
            fig.add_trace(go.Scatter(
                x=short_idx, y=short_prices.values * 1.0005,
                mode="markers",
                name="Short",
                marker=dict(symbol="triangle-down", size=8, color=_RED),
            ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(size=15)),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=500,
    )
    return fig


# =====================================================================
# HYPEROPT — HISTORIAL DE OPTIMIZACIÓN
# =====================================================================

def regime_timeline_chart(
    prices: pd.Series,
    regime_labels: pd.Series,
    n_components: int = 3,
    title: str = "Regímenes Detectados (GMM)",
) -> go.Figure:
    """
    Precio de cierre con fondo coloreado por régimen detectado.
    regime_0 = bajo vol (verde/oscuro)
    regime_1 = intermedio (amarillo)
    regime_2 = alto vol / crisis (rojo)
    """
    _REGIME_COLORS = {
        0: "rgba(38,194,129,0.18)",   # verde — quiet/ranging
        1: "rgba(76,142,247,0.18)",   # azul — trending
        2: "rgba(224,82,82,0.22)",    # rojo — volatile/crisis
        3: "rgba(245,200,66,0.18)",   # amarillo — extra
    }
    _REGIME_NAMES = {
        0: "Quiet/Rango", 1: "Tendencia", 2: "Volatil/Crisis", 3: "Régimen 3"
    }

    common_idx = prices.index.intersection(regime_labels.index)
    prices_al = prices.reindex(common_idx)
    regimes_al = regime_labels.reindex(common_idx)

    fig = go.Figure()

    # Precio
    fig.add_trace(go.Scatter(
        x=prices_al.index, y=prices_al.values,
        name="Precio", line=dict(color=_BLUE, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.5f}<extra></extra>",
    ))

    # Fondo por régimen (agrupa franjas consecutivas del mismo régimen)
    if not regimes_al.empty:
        prev_r = None
        seg_start = None
        for dt, r in regimes_al.items():
            r = int(r)
            if prev_r is None:
                prev_r, seg_start = r, dt
            elif r != prev_r:
                _add_regime_band(fig, seg_start, dt, prev_r, _REGIME_COLORS, _REGIME_NAMES)
                prev_r, seg_start = r, dt
        if prev_r is not None:
            _add_regime_band(fig, seg_start, regimes_al.index[-1],
                             prev_r, _REGIME_COLORS, _REGIME_NAMES)

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title, font=dict(size=15)),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=420,
    )
    return fig


def _add_regime_band(fig, x0, x1, regime_id, colors, names):
    color = colors.get(regime_id, "rgba(100,100,100,0.1)")
    fig.add_vrect(
        x0=x0, x1=x1,
        fillcolor=color, opacity=1.0,
        layer="below", line_width=0,
        annotation_text="",
    )


def hyperopt_history_chart(study) -> go.Figure:
    """
    Scatter de valores por trial + línea del mejor acumulado.
    Requiere un objeto optuna.Study.
    """
    try:
        import optuna
    except ImportError:
        return go.Figure()

    trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not trials:
        return go.Figure()

    nums = [t.number + 1 for t in trials]
    vals = [t.value for t in trials]

    best_so_far = []
    current_best = float("-inf")
    for v in vals:
        current_best = max(current_best, v)
        best_so_far.append(current_best)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=nums, y=vals,
        mode="markers",
        name="Trial",
        marker=dict(color=vals, colorscale="Viridis", size=7,
                    colorbar=dict(title="Valor")),
        hovertemplate="Trial %{x}: %{y:.4f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=nums, y=best_so_far,
        mode="lines",
        name="Mejor acumulado",
        line=dict(color=_YELLOW, width=2),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Historial de Optimización (Optuna)", font=dict(size=15)),
        xaxis_title="Trial",
        yaxis_title="Valor objetivo (PSR / Sharpe)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def hyperopt_param_scatter(study, param_x: str, param_y: str) -> go.Figure:
    """Scatter de dos parámetros coloreado por valor objetivo."""
    try:
        import optuna
    except ImportError:
        return go.Figure()

    trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and param_x in t.params and param_y in t.params
    ]
    if not trials:
        return go.Figure()

    xs = [t.params[param_x] for t in trials]
    ys = [t.params[param_y] for t in trials]
    vals = [t.value for t in trials]

    fig = go.Figure(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            color=vals,
            colorscale="RdYlGn",
            size=9,
            colorbar=dict(title="Objetivo"),
            showscale=True,
        ),
        text=[f"Trial {t.number+1}: {v:.4f}" for t, v in zip(trials, vals)],
        hovertemplate="<b>%{text}</b><br>"
                      f"{param_x}: %{{x:.3f}}<br>"
                      f"{param_y}: %{{y:.3f}}<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=f"{param_x} vs {param_y}", font=dict(size=15)),
        xaxis_title=param_x,
        yaxis_title=param_y,
    )
    return fig
