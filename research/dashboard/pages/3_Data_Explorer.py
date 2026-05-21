"""
Página 3 — Data Explorer
==========================
- Gráfico de precios OHLCV con señales superpuestas
- Distribución de labels
- Distribución de features
- Estadísticas descriptivas
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.state import load_prices, load_wf_result, list_available_symbols
from dashboard.components.charts import price_chart, label_distribution_chart

st.set_page_config(page_title="Data Explorer | Quant Bot", layout="wide", page_icon="🗂️")
st.markdown("## 🗂️ Data Explorer")
st.caption("Precios, features, labels y señales OOS")
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

    st.divider()
    st.markdown("### Parámetros de Labels")
    upper_mult = st.slider("Upper mult", 0.5, 2.5, 1.0, step=0.1)
    lower_mult = st.slider("Lower mult", 0.5, 2.5, 1.0, step=0.1)
    horizon = st.slider("Horizon", 3, 15, 5)

prices = load_prices(symbol)
result = load_wf_result(symbol)

# ─────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📉 Precios y Señales",
    "🏷️ Distribución de Labels",
    "📐 Features",
])


# ────── Tab 1: Precios ────────────────────────────────────────────────
with tab1:
    if prices is None:
        st.info("No hay precios en cache. Corre el pipeline desde el Home.")
        st.stop()

    # Selector de rango de fechas
    min_date = prices.index.min().date()
    max_date = prices.index.max().date()
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        date_from = st.date_input("Desde", value=max(min_date,
                                   pd.Timestamp(max_date) - pd.DateOffset(months=12)), min_value=min_date, max_value=max_date)
    with col_d2:
        date_to = st.date_input("Hasta", value=max_date, min_value=min_date, max_value=max_date)

    prices_slice = prices.loc[str(date_from):str(date_to)]

    # Señales del resultado si están disponibles
    signals_slice = None
    if result is not None:
        signals_slice = result.oos_signals.reindex(prices_slice.index)

    fig_price = price_chart(
        prices_slice,
        signals=signals_slice,
        title=f"{symbol} — OHLCV ({date_from} → {date_to})",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("#### Estadísticas del período")
    c_price = prices_slice["close"]
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Barras", len(prices_slice))
    col_s2.metric("Cierre inicio", f"{c_price.iloc[0]:.5f}" if len(c_price) else "—")
    col_s3.metric("Cierre fin", f"{c_price.iloc[-1]:.5f}" if len(c_price) else "—")
    if len(c_price) > 1:
        ret_total = (c_price.iloc[-1] / c_price.iloc[0] - 1) * 100
        col_s4.metric("Retorno período", f"{ret_total:.2f}%")

    if signals_slice is not None:
        active = signals_slice[signals_slice != 0]
        n_long = int((signals_slice == 1).sum())
        n_short = int((signals_slice == -1).sum())
        st.caption(
            f"Señales en período: **{len(active)}** activas "
            f"(Long: {n_long} · Short: {n_short})"
        )

    st.markdown("#### OHLCV — últimas 20 barras")
    st.dataframe(
        prices_slice.tail(20).round(5),
        use_container_width=True,
    )


# ────── Tab 2: Labels ─────────────────────────────────────────────────
with tab2:
    if prices is None:
        st.info("No hay precios en cache.")
        st.stop()

    # Calcular labels con los params del sidebar
    with st.spinner("Calculando labels..."):
        try:
            from examples.pipeline_ml_real_data import (
                build_features, compute_atr_series, triple_barrier_labels,
            )
            features_raw = build_features(prices)
            atr_raw = compute_atr_series(prices)
            atr_al = atr_raw.reindex(features_raw.index)
            close_al = prices["close"].reindex(features_raw.index)

            raw_labels = triple_barrier_labels(
                close_al, atr_al,
                horizon=horizon,
                upper_mult=upper_mult,
                lower_mult=lower_mult,
            )
            y = raw_labels.dropna().astype(int)

        except Exception as e:
            st.error(f"Error calculando labels: {e}")
            st.stop()

    col_l, col_r = st.columns(2)
    with col_l:
        fig_ld = label_distribution_chart(y)
        st.plotly_chart(fig_ld, use_container_width=True)

    with col_r:
        st.markdown("#### Estadísticas de labels")
        dist = y.value_counts().sort_index()
        total = len(y)
        for lbl, cnt in dist.items():
            name = {-1: "Short (-1)", 0: "Neutral (0)", 1: "Long (+1)"}.get(lbl, str(lbl))
            st.metric(name, f"{cnt}  ({cnt/total:.1%})")

        st.divider()
        st.markdown("##### Con estos params:")
        st.markdown(f"- `upper_mult = {upper_mult}` · `lower_mult = {lower_mult}`")
        st.markdown(f"- `horizon = {horizon}` días")
        neutrals_pct = dist.get(0, 0) / total * 100
        if neutrals_pct > 60:
            st.warning(f"Neutrales: {neutrals_pct:.1f}% — considera reducir los mults.")
        elif neutrals_pct < 25:
            st.warning(f"Neutrales: {neutrals_pct:.1f}% — considera aumentar los mults.")
        else:
            st.success(f"Neutrales: {neutrals_pct:.1f}% — distribución razonable.")

    st.markdown("#### Evolución temporal de labels")
    import plotly.graph_objects as go
    rolling_pct = pd.DataFrame({
        "Long": (y == 1).rolling(63).mean() * 100,
        "Short": (y == -1).rolling(63).mean() * 100,
        "Neutral": (y == 0).rolling(63).mean() * 100,
    }).dropna()

    fig_roll = go.Figure()
    colors = {"Long": "#26c281", "Short": "#e05252", "Neutral": "#4a4f5c"}
    for col, color in colors.items():
        fig_roll.add_trace(go.Scatter(
            x=rolling_pct.index, y=rolling_pct[col],
            name=col, mode="lines",
            line=dict(color=color, width=1.5),
            stackgroup="one",
            hovertemplate=f"{col}: %{{y:.1f}}%<extra></extra>",
        ))
    fig_roll.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#c8ccd4"),
        title=dict(text="% Labels rolling 63 barras", font=dict(size=14)),
        yaxis_title="%", legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor="#1e2130"),
        yaxis=dict(gridcolor="#1e2130"),
    )
    st.plotly_chart(fig_roll, use_container_width=True)


# ────── Tab 3: Features ───────────────────────────────────────────────
with tab3:
    if prices is None:
        st.info("No hay precios en cache.")
        st.stop()

    with st.spinner("Calculando features..."):
        try:
            from examples.pipeline_ml_real_data import build_features
            features_df = build_features(prices)
        except Exception as e:
            st.error(f"Error calculando features: {e}")
            st.stop()

    st.markdown(f"**{features_df.shape[1]} features · {len(features_df)} barras**")

    feat_sel = st.selectbox("Seleccionar feature", features_df.columns.tolist())

    import plotly.graph_objects as go

    serie = features_df[feat_sel].dropna()

    col_hist, col_ts = st.columns([1, 2])

    with col_hist:
        st.markdown(f"##### Distribución: `{feat_sel}`")
        fig_h = go.Figure(go.Histogram(
            x=serie.values, nbinsx=50,
            marker_color="#4c8ef7", opacity=0.8,
        ))
        fig_h.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#c8ccd4"),
            xaxis=dict(gridcolor="#1e2130"),
            yaxis=dict(gridcolor="#1e2130", title="Frecuencia"),
            margin=dict(l=30, r=10, t=30, b=30),
        )
        st.plotly_chart(fig_h, use_container_width=True)

        # Stats
        st.markdown("##### Estadísticas")
        stats = serie.describe()
        st.dataframe(
            stats.to_frame().rename(columns={feat_sel: "Valor"}).round(6),
            use_container_width=True,
        )

    with col_ts:
        st.markdown(f"##### Serie temporal: `{feat_sel}`")
        # Rango — últimos 2 años por defecto
        ts_slice = serie.iloc[-504:]
        fig_ts = go.Figure(go.Scatter(
            x=ts_slice.index, y=ts_slice.values,
            mode="lines", line=dict(color="#4c8ef7", width=1.2),
            name=feat_sel,
        ))
        fig_ts.add_hline(y=0, line_color="#4a4f5c", line_dash="dash", line_width=0.8)
        fig_ts.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#c8ccd4"),
            xaxis=dict(gridcolor="#1e2130"),
            yaxis=dict(gridcolor="#1e2130"),
            margin=dict(l=40, r=10, t=30, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("#### Correlación entre features (top 10)")
    if "feature_importance_agg" in dir(load_wf_result(symbol) or object()):
        top_feats = (
            (load_wf_result(symbol) or object()).feature_importance_agg.head(10).index.tolist()
            if load_wf_result(symbol) and not load_wf_result(symbol).feature_importance_agg.empty
            else features_df.columns[:10].tolist()
        )
    else:
        top_feats = features_df.columns[:10].tolist()

    corr = features_df[top_feats].corr().round(2)
    import plotly.express as px
    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=True,
    )
    fig_corr.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#c8ccd4", size=10),
        margin=dict(l=10, r=10, t=30, b=10),
        coloraxis_colorbar=dict(title="r"),
        title=dict(text="Correlación Pearson", font=dict(size=14)),
    )
    st.plotly_chart(fig_corr, use_container_width=True)
