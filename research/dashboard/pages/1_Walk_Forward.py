"""
Página 1 — Walk-Forward Detallado
===================================
- Selector de fold
- Equity curve del fold seleccionado
- Feature importance cross-fold
- Calibración ECE antes/después
- Sizing sample (p_win, R:R, kelly)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.state import load_wf_result, load_prices, list_available_symbols
from dashboard.components.charts import (
    equity_curve,
    feature_importance_chart,
    calibration_ece_chart,
    fold_coverage_chart,
    p_win_histogram,
    label_distribution_chart,
)

st.set_page_config(page_title="Walk-Forward | Quant Bot", layout="wide", page_icon="📊")
st.markdown("## 📊 Walk-Forward — Análisis Detallado")
st.caption("Métricas OOS por fold, feature importance y calibración")
st.divider()

# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Símbolo")
    available = list_available_symbols()
    if not available:
        available = ["EURUSD"]
    symbol = st.selectbox("Símbolo", available, index=0)

result = load_wf_result(symbol)
if result is None:
    st.info(f"No hay resultados para **{symbol}**. Ve al home y corre el pipeline.")
    st.stop()

prices_cached = load_prices(symbol)
fold_results = result.fold_results
n_folds = len(fold_results)

# ─────────────────────────────────────────────────────────────────────
# Overview rápido
# ─────────────────────────────────────────────────────────────────────
gm = result.global_metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sharpe global", f"{gm.get('sharpe') or 0:.3f}")
c2.metric("PSR",           f"{gm.get('psr') or 0:.3f}")
c3.metric("Trades totales", gm.get("n_trades", 0))
c4.metric("Coverage",      f"{gm.get('coverage', 0):.1%}")

st.divider()

# ─────────────────────────────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Equity & Folds",
    "🌟 Feature Importance",
    "🎯 Calibración",
    "💰 Sizing",
])


# ────── Tab 1: Equity & Folds ─────────────────────────────────────────
with tab1:
    st.markdown("#### Equity Curve completa (OOS)")
    if prices_cached is not None:
        fold_dates = [fr.test_start for fr in fold_results]
        fig_eq = equity_curve(
            result.oos_signals,
            prices_cached["close"].reindex(result.oos_signals.index),
            title=f"OOS completo — {symbol}",
            fold_boundaries=fold_dates,
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.warning("Precios no disponibles.")

    st.markdown("#### Coverage y Trades por Fold")
    st.plotly_chart(fold_coverage_chart(fold_results), use_container_width=True)

    st.markdown("#### Detalle por Fold")
    fold_sel = st.slider(
        "Seleccionar fold", 1, n_folds, 1,
        format="Fold %d",
        help="Navega fold a fold para ver la equity curve individual",
    )
    fr = fold_results[fold_sel - 1]

    info_cols = st.columns(5)
    info_cols[0].metric("Test inicio", str(fr.test_start.date() if hasattr(fr.test_start, "date") else fr.test_start))
    info_cols[1].metric("Trades",    fr.metrics.get("n_trades", 0))
    info_cols[2].metric("Sharpe",    f"{fr.metrics.get('sharpe') or 0:.3f}")
    info_cols[3].metric("Threshold", f"{fr.threshold_long:.3f}")
    info_cols[4].metric("ECE cal",   fr.calibration.get("ece_calibrated", "—"))

    if prices_cached is not None:
        fig_fold = equity_curve(
            fr.oos_signals,
            prices_cached["close"].reindex(fr.oos_signals.index),
            title=f"Fold {fold_sel} — {fr.test_start} → {fr.test_end}",
        )
        st.plotly_chart(fig_fold, use_container_width=True)


# ────── Tab 2: Feature Importance ─────────────────────────────────────
with tab2:
    st.markdown("#### Feature Importance — Mediana cross-fold (Gain)")
    top_n = st.slider("Top N features", 5, 20, 12)

    if not result.feature_importance_agg.empty:
        fig_fi = feature_importance_chart(result.feature_importance_agg, top_n=top_n)
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("##### Tabla completa")
        st.dataframe(
            result.feature_importance_agg.reset_index().rename(
                columns={"index": "Feature"}
            ),
            use_container_width=True,
        )

        if result.features_to_drop:
            st.warning(
                f"**Candidatas a eliminar** (importancia baja cross-fold): "
                f"{', '.join(result.features_to_drop)}"
            )
    else:
        st.info("No se calculó feature importance en esta corrida.")


# ────── Tab 3: Calibración ────────────────────────────────────────────
with tab3:
    st.markdown("#### ECE antes / después de calibración por fold")
    st.caption("ECE = Expected Calibration Error · Menor = mejor calibrado")

    fig_cal = calibration_ece_chart(fold_results)
    st.plotly_chart(fig_cal, use_container_width=True)

    st.markdown("#### Tabla de calibración por fold")
    cal_rows = []
    for fr in fold_results:
        cal = fr.calibration
        cal_rows.append({
            "Fold": fr.fold_idx + 1,
            "ECE (sin cal)": cal.get("ece_uncalibrated", "—"),
            "ECE (cal)":     cal.get("ece_calibrated", "—"),
            "Brier (sin cal)": cal.get("brier_uncalibrated", "—"),
            "Brier (cal)":   cal.get("brier_calibrated", "—"),
            "Veredicto":     cal.get("verdict", "—"),
        })
    st.dataframe(pd.DataFrame(cal_rows), use_container_width=True, hide_index=True)


# ────── Tab 4: Sizing ─────────────────────────────────────────────────
with tab4:
    st.markdown("#### Distribución de P(win) — señales activas")
    fig_pw = p_win_histogram(result.oos_sizing)
    st.plotly_chart(fig_pw, use_container_width=True)

    active_sizing = result.oos_sizing[result.oos_sizing["signal"] != 0].copy()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Estadísticas de sizing")
        if not active_sizing.empty:
            cols_show = ["p_win", "rr_dynamic", "kelly_raw", "risk_pct"]
            cols_avail = [c for c in cols_show if c in active_sizing.columns]
            st.dataframe(
                active_sizing[cols_avail].describe().round(4),
                use_container_width=True,
            )

    with col_r:
        st.markdown("#### Primeras 20 entradas activas")
        if not active_sizing.empty:
            cols_show = ["signal", "p_win", "rr_dynamic", "kelly_raw", "risk_pct"]
            cols_avail = [c for c in cols_show if c in active_sizing.columns]
            st.dataframe(
                active_sizing[cols_avail].head(20).round(4),
                use_container_width=True,
            )
