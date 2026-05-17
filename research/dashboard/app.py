"""
Quant Bot Dashboard — Home / Overview
======================================
Página principal: métricas globales + equity curve + tabla de folds.

Levantar:
    cd C:\\Users\\alexj\\OneDrive\\Desktop\\quant_bot
    .venv-1\\Scripts\\streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────
# Config de página (DEBE ser la primera llamada a st)
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Imports internos (después del set_page_config)
# ─────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from dashboard.state import (
    load_wf_result, load_wf_meta, save_wf_result, save_prices,
    load_prices, list_available_symbols,
)
from dashboard.components.charts import equity_curve, fold_metrics_chart


# ─────────────────────────────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { color: #8a8f9e; font-size: 13px; margin-bottom: 4px; }
    .metric-value { color: #e8eaf0; font-size: 28px; font-weight: 700; }
    .metric-sub   { color: #6a6f7e; font-size: 11px; margin-top: 4px; }
    .verdict-pass     { background:#0d2b1a; border:1px solid #26c281; color:#26c281;
                        border-radius:8px; padding:10px 20px; font-weight:700; font-size:16px; }
    .verdict-marginal { background:#2b2200; border:1px solid #f5c842; color:#f5c842;
                        border-radius:8px; padding:10px 20px; font-weight:700; font-size:16px; }
    .verdict-fail     { background:#2b0d0d; border:1px solid #e05252; color:#e05252;
                        border-radius:8px; padding:10px 20px; font-weight:700; font-size:16px; }
    .verdict-none     { background:#1a1d2e; border:1px solid #4a4f5c; color:#8a8f9e;
                        border-radius:8px; padding:10px 20px; font-weight:700; font-size:16px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────
st.markdown("## 📈 Quant Bot Dashboard")
st.caption("Pipeline ML · López de Prado · Walk-Forward OOS")

st.divider()


# ─────────────────────────────────────────────────────────────────────
# Sidebar: símbolo + acción
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    available = list_available_symbols()
    symbol_options = sorted(set(["EURUSD", "GBPUSD", "ES"] + available))
    symbol = st.selectbox("Símbolo", symbol_options, index=0)

    st.divider()
    st.markdown("### ▶ Correr Pipeline")

    start_date = st.date_input("Fecha inicio", value=pd.Timestamp("2018-01-01"))
    train_size = st.slider("Train size (barras)", 126, 504, 252, step=21)
    test_size = st.slider("Test size (barras)", 21, 126, 63, step=21)
    upper_mult = st.slider("Upper mult (ATR)", 0.5, 2.5, 1.0, step=0.1)
    lower_mult = st.slider("Lower mult (ATR)", 0.5, 2.5, 1.0, step=0.1)
    horizon = st.slider("Horizon (días)", 3, 15, 5)

    run_btn = st.button("🚀 Correr Pipeline", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# Run pipeline si se presionó el botón
# ─────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Corriendo pipeline {symbol} ({train_size}/{test_size})..."):
        try:
            from data.real_data import fetch_real_data
            from models.walk_forward_runner import WalkForwardConfig, WalkForwardRunner
            from examples.pipeline_ml_real_data import (
                build_features, compute_atr_series, triple_barrier_labels,
            )

            CACHE_DIR = ROOT / "cache"
            prices_df = fetch_real_data(
                symbol, interval="1d",
                start=str(start_date), cache_dir=CACHE_DIR,
            )
            save_prices(prices_df, symbol)

            features = build_features(prices_df)
            atr = compute_atr_series(prices_df)
            atr_aligned = atr.reindex(features.index)
            close_aligned = prices_df["close"].reindex(features.index)

            raw_labels = triple_barrier_labels(
                close_aligned, atr_aligned,
                horizon=horizon, upper_mult=upper_mult, lower_mult=lower_mult,
            )
            valid_idx = raw_labels.dropna().index
            X = features.loc[valid_idx]
            y = raw_labels.loc[valid_idx].astype(int)
            prices_aligned = close_aligned.reindex(valid_idx)
            atr_final = atr_aligned.reindex(valid_idx)

            cfg = WalkForwardConfig(
                train_size=train_size,
                test_size=test_size,
                embargo=5,
                use_class_weights=True,
                track_importance=True,
            )
            runner = WalkForwardRunner(cfg)
            result = runner.run(
                X=X, y=y,
                prices=prices_aligned,
                atr=atr_final,
                all_classes=[-1, 0, 1],
            )
            save_wf_result(result, symbol)
            st.success("Pipeline completado. Resultados guardados.")
            st.rerun()

        except Exception as e:
            st.error(f"Error en el pipeline: {e}")


# ─────────────────────────────────────────────────────────────────────
# Cargar resultados
# ─────────────────────────────────────────────────────────────────────
result = load_wf_result(symbol)
meta = load_wf_meta(symbol)

if result is None:
    st.info(
        f"No hay resultados guardados para **{symbol}**. "
        "Configura los parámetros en el sidebar y presiona **Correr Pipeline**."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────
# Métricas globales (cards)
# ─────────────────────────────────────────────────────────────────────
gm = result.global_metrics

sharpe_val = gm.get("sharpe")
psr_val = gm.get("psr")
dsr_val = gm.get("dsr")
n_trades = gm.get("n_trades", 0)
coverage = gm.get("coverage", 0.0)
win_rate = gm.get("win_rate")
max_dd = gm.get("max_drawdown")
n_folds = len(result.fold_results)

saved_at = meta.get("saved_at", "—") if meta else "—"

# Veredicto
if psr_val is not None and dsr_val is not None and sharpe_val is not None:
    if psr_val >= 0.95 and dsr_val >= 0.90 and sharpe_val > 0:
        verdict_cls, verdict_txt = "pass", "✅ PASS"
    elif psr_val >= 0.80 or (sharpe_val or 0) > 0:
        verdict_cls, verdict_txt = "marginal", "⚠️ MARGINAL"
    else:
        verdict_cls, verdict_txt = "fail", "❌ FAIL"
else:
    verdict_cls, verdict_txt = "none", "— INSUFICIENTE"

col_v, col_ts = st.columns([3, 1])
with col_v:
    st.markdown(
        f'<div class="verdict-{verdict_cls}">{verdict_txt} — {symbol}</div>',
        unsafe_allow_html=True,
    )
with col_ts:
    st.caption(f"Última ejecución: {saved_at}")

st.markdown("")

# Cards de métricas
c1, c2, c3, c4, c5, c6 = st.columns(6)

def _card(col, label, value, sub=""):
    fmt = f"{value:.4f}" if isinstance(value, float) else str(value) if value is not None else "N/A"
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{fmt}</div>'
            f'<div class="metric-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

_card(c1, "Sharpe OOS", sharpe_val, "anualizado")
_card(c2, "PSR", psr_val, "P(SR > 0)")
_card(c3, "DSR", dsr_val, "ajustado trials")
_card(c4, "Trades", n_trades, f"{n_folds} folds")
_card(c5, "Coverage", f"{coverage:.1%}", "barras activas")
_card(c6, "Win Rate", win_rate, "trades ganadores")

st.markdown("")


# ─────────────────────────────────────────────────────────────────────
# Equity Curve
# ─────────────────────────────────────────────────────────────────────
st.markdown("### Equity Curve")

prices_cached = load_prices(symbol)
if prices_cached is not None:
    fold_dates = [fr.test_start for fr in result.fold_results]
    fig_eq = equity_curve(
        result.oos_signals,
        prices_cached["close"].reindex(result.oos_signals.index),
        title=f"Estrategia vs Buy & Hold — {symbol}",
        fold_boundaries=fold_dates,
    )
    st.plotly_chart(fig_eq, use_container_width=True)
else:
    st.warning("Precios no disponibles en cache. Re-ejecuta el pipeline.")


# ─────────────────────────────────────────────────────────────────────
# Fold metrics
# ─────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)
with col_l:
    st.markdown("### Métricas por Fold")
    fig_folds = fold_metrics_chart(result.fold_results)
    st.plotly_chart(fig_folds, use_container_width=True)

with col_r:
    st.markdown("### Resumen de Folds")
    rows = []
    for fr in result.fold_results:
        rows.append({
            "Fold": fr.fold_idx + 1,
            "Test inicio": fr.test_start.date() if hasattr(fr.test_start, "date") else fr.test_start,
            "Trades": fr.metrics.get("n_trades", 0),
            "Coverage": f"{fr.metrics.get('coverage', 0):.1%}",
            "Sharpe": f"{fr.metrics.get('sharpe') or 0:.3f}",
            "PSR": f"{fr.metrics.get('psr') or 0:.3f}",
            "ECE": fr.calibration.get("ece_calibrated", "—"),
            "Thresh": f"{fr.threshold_long:.3f}",
        })
    df_folds = pd.DataFrame(rows)
    st.dataframe(df_folds, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────
# Navegación
# ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### Explorar en detalle")
col_n1, col_n2, col_n3 = st.columns(3)
with col_n1:
    st.page_link("pages/1_Walk_Forward.py",
                 label="📊 Walk-Forward Detallado", icon="📊")
with col_n2:
    st.page_link("pages/2_Hyperopt.py",
                 label="🔬 Hyperopt Bayesiano", icon="🔬")
with col_n3:
    st.page_link("pages/3_Data_Explorer.py",
                 label="🗂️ Data Explorer", icon="🗂️")
