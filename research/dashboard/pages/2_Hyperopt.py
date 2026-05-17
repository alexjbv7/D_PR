"""
Página 2 — Hyperopt Bayesiano
================================
- Mejores parámetros (XGBoost + barrera)
- Historial de optimización
- Scatter de parámetros coloreado por objetivo
- Tabla completa de trials
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.state import (
    load_ho_result, load_ho_meta, save_ho_result, load_wf_result,
    list_available_symbols, save_wf_result, save_prices,
)
from dashboard.components.charts import hyperopt_history_chart, hyperopt_param_scatter

st.set_page_config(page_title="Hyperopt | Quant Bot", layout="wide", page_icon="🔬")
st.markdown("## 🔬 Hyperopt Bayesiano — Búsqueda de Hiperparámetros")
st.caption("Optuna TPE · XGBoost + barrera triple · Anti-leakage")
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
    st.markdown("### ▶ Correr Hyperopt")
    n_trials = st.slider("N trials", 10, 200, 50, step=10)
    n_val_folds = st.slider("Folds internos", 2, 5, 3)
    objective = st.selectbox("Objetivo", ["psr", "sharpe", "coverage_psr"])
    sym_barriers = st.checkbox("Barreras simétricas", value=False)
    search_rr = st.checkbox("Buscar R:R", value=False)

    run_ho_btn = st.button("🚀 Correr Hyperopt", type="primary", use_container_width=True)

    if run_ho_btn:
        st.markdown("---")
        st.caption("**Nota:** después del hyperopt se recomienda correr el "
                   "pipeline completo desde el Home con los mejores params.")


# ─────────────────────────────────────────────────────────────────────
# Run Hyperopt
# ─────────────────────────────────────────────────────────────────────
if run_ho_btn:
    with st.spinner(f"Corriendo hyperopt {symbol} ({n_trials} trials)..."):
        try:
            from data.real_data import fetch_real_data
            from models.hyperopt import BayesianHyperopt, HyperoptConfig
            from examples.pipeline_ml_real_data import (
                build_features, compute_atr_series, triple_barrier_labels,
            )

            CACHE_DIR = ROOT / "cache"
            prices_df = fetch_real_data(
                symbol, interval="1d", start="2018-01-01", cache_dir=CACHE_DIR
            )
            save_prices(prices_df, symbol)
            features = build_features(prices_df)
            atr = compute_atr_series(prices_df)
            atr_al = atr.reindex(features.index)
            close_al = prices_df["close"].reindex(features.index)

            def label_fn(um, lm, h):
                return triple_barrier_labels(close_al, atr_al, h, um, lm)

            cfg = HyperoptConfig(
                n_trials=n_trials,
                n_val_folds=n_val_folds,
                objective_metric=objective,
                symmetric_barriers=sym_barriers,
                search_rr=search_rr,
                val_frac=0.80,
                train_size=252,
                val_size=63,
                embargo=5,
                use_class_weights=True,
                use_pruner=True,
                verbose=False,
            )
            ho = BayesianHyperopt(cfg)
            ho_result = ho.run(
                X=features,
                close=close_al,
                atr=atr_al,
                label_fn=label_fn,
                prices=close_al,
                all_classes=[-1, 0, 1],
            )
            save_ho_result(ho_result, symbol)
            st.success("Hyperopt completado. Resultados guardados.")
            st.rerun()

        except Exception as e:
            st.error(f"Error en hyperopt: {e}")
            st.exception(e)


# ─────────────────────────────────────────────────────────────────────
# Cargar resultado
# ─────────────────────────────────────────────────────────────────────
ho_result = load_ho_result(symbol)
meta = load_ho_meta(symbol)

if ho_result is None:
    st.info(
        f"No hay resultados de hyperopt para **{symbol}**. "
        "Configura los parámetros en el sidebar y presiona **Correr Hyperopt**."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────
# Resumen superior
# ─────────────────────────────────────────────────────────────────────
saved_at = meta.get("saved_at", "—") if meta else "—"
st.caption(f"Última ejecución: {saved_at}")

c1, c2, c3 = st.columns(3)
c1.metric("Mejor valor (objetivo)", f"{ho_result.best_value:.4f}")
c2.metric("Trials completados", ho_result.n_trials_completed)
c3.metric("Trials podados", ho_result.n_trials_pruned)

st.divider()


# ─────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🏆 Mejores Parámetros",
    "📈 Historial & Scatter",
    "📋 Todos los Trials",
])


# ────── Tab 1: Mejores Parámetros ─────────────────────────────────────
with tab1:
    col_xgb, col_barrier = st.columns(2)

    with col_xgb:
        st.markdown("#### XGBoost")
        xgb = ho_result.best_xgb_params
        for k, v in sorted(xgb.items()):
            fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
            st.markdown(f"- **{k}**: `{fmt}`")

    with col_barrier:
        st.markdown("#### Barrera triple")
        bp = ho_result.best_barrier_params
        for k, v in bp.items():
            fmt = f"{v:.3f}" if isinstance(v, float) else str(v)
            st.markdown(f"- **{k}**: `{fmt}`")

        if ho_result.best_rr_params:
            st.markdown("#### R:R")
            for k, v in ho_result.best_rr_params.items():
                st.markdown(f"- **{k}**: `{v:.3f}`")

    st.divider()

    # Aplicar mejores params al walk-forward
    st.markdown("#### Aplicar mejores params")
    st.info(
        "Presiona el botón para correr el walk-forward final "
        "con los mejores parámetros encontrados."
    )

    apply_btn = st.button("⚡ Correr WF Final con mejores params", type="secondary")
    if apply_btn:
        with st.spinner("Corriendo walk-forward con mejores params..."):
            try:
                from data.real_data import fetch_real_data
                from models.walk_forward_runner import WalkForwardConfig, WalkForwardRunner
                from examples.pipeline_ml_real_data import (
                    build_features, compute_atr_series, triple_barrier_labels,
                )

                CACHE_DIR = ROOT / "cache"
                prices_df = fetch_real_data(
                    symbol, interval="1d", start="2018-01-01", cache_dir=CACHE_DIR
                )
                features = build_features(prices_df)
                atr = compute_atr_series(prices_df)
                atr_al = atr.reindex(features.index)
                close_al = prices_df["close"].reindex(features.index)

                best_um = bp.get("upper_mult", 1.0)
                best_lm = bp.get("lower_mult", 1.0)
                best_h  = bp.get("horizon", 5)

                raw_labels = triple_barrier_labels(
                    close_al, atr_al, best_h, best_um, best_lm
                )
                valid_idx = raw_labels.dropna().index
                X = features.loc[valid_idx]
                y = raw_labels.loc[valid_idx].astype(int)
                prices_alig = close_al.reindex(valid_idx)
                atr_fin = atr_al.reindex(valid_idx)

                best_wf_cfg = ho_result.to_walk_forward_config()
                final_cfg = WalkForwardConfig(
                    train_size=252, test_size=63, embargo=5,
                    use_class_weights=True, track_importance=True,
                    xgb_params=best_wf_cfg.xgb_params,
                    rr_min=best_wf_cfg.rr_min,
                    rr_max=best_wf_cfg.rr_max,
                )
                runner = WalkForwardRunner(final_cfg)
                wf_result = runner.run(
                    X=X, y=y, prices=prices_alig,
                    atr=atr_fin, all_classes=[-1, 0, 1],
                )
                save_wf_result(wf_result, symbol)
                save_prices(prices_df, symbol)
                st.success("Walk-forward completado. Ve al Home para ver resultados.")

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)


# ────── Tab 2: Historial & Scatter ────────────────────────────────────
with tab2:
    st.markdown("#### Historial de Optimización")
    try:
        fig_hist = hyperopt_history_chart(ho_result.study)
        st.plotly_chart(fig_hist, use_container_width=True)
    except Exception as e:
        st.warning(f"No se puede mostrar el historial: {e}")

    st.markdown("#### Scatter de Parámetros")
    all_param_keys = list(ho_result.best_params.keys())
    if len(all_param_keys) >= 2:
        col_px, col_py = st.columns(2)
        with col_px:
            px_key = st.selectbox("Eje X", all_param_keys,
                                  index=all_param_keys.index("upper_mult")
                                  if "upper_mult" in all_param_keys else 0)
        with col_py:
            remaining = [k for k in all_param_keys if k != px_key]
            py_key = st.selectbox("Eje Y", remaining,
                                  index=remaining.index("horizon")
                                  if "horizon" in remaining else 0)
        try:
            fig_sc = hyperopt_param_scatter(ho_result.study, px_key, py_key)
            st.plotly_chart(fig_sc, use_container_width=True)
        except Exception as e:
            st.warning(f"No se puede mostrar el scatter: {e}")
    else:
        st.info("No hay suficientes parámetros para el scatter.")


# ────── Tab 3: Todos los Trials ───────────────────────────────────────
with tab3:
    st.markdown("#### Tabla completa de trials")
    if ho_result.all_trials:
        rows = []
        for t in ho_result.all_trials:
            row = {"Trial": t["number"] + 1, "Estado": t["state"],
                   "Valor": t["value"]}
            row.update({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in t["params"].items()})
            rows.append(row)

        df_trials = pd.DataFrame(rows)

        # Filtro por estado
        estados = df_trials["Estado"].unique().tolist()
        estado_sel = st.multiselect("Filtrar por estado", estados,
                                    default=["COMPLETE"])
        df_show = df_trials[df_trials["Estado"].isin(estado_sel)]

        # Ordenar por valor descendente
        if "Valor" in df_show.columns:
            df_show = df_show.sort_values("Valor", ascending=False)

        st.dataframe(df_show, use_container_width=True, hide_index=True)
        st.caption(f"{len(df_show)} trials mostrados de {len(df_trials)} totales")
    else:
        st.info("No hay datos de trials disponibles.")
