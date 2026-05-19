"""
Página 4 — Multi-Horizon Trainer Results
==========================================
Displays PSR / DSR / ECE metrics for the 3 trained horizons,
ablative analysis table, and feature importance per horizon.

Reads JSON reports from research/reports/multi_horizon_v1/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research"))

st.set_page_config(
    page_title="Multi-Horizon | Quant Bot",
    layout="wide",
    page_icon="🌐",
)

_REPORTS_DIR = ROOT / "research" / "reports" / "multi_horizon_v1"

HORIZON_ORDER = ["intraday", "swing", "daily"]
MODEL_MAP = {"intraday": "xgb", "swing": "xgb", "daily": "mlp"}


def _load_report(horizon: str) -> dict | None:
    model = MODEL_MAP[horizon]
    candidates = list(_REPORTS_DIR.glob(f"{horizon}_{model}_v*.json"))
    if not candidates:
        return None
    latest = sorted(candidates)[-1]
    with open(latest, encoding="utf-8") as f:
        return json.load(f)


def _load_ablative() -> dict | None:
    candidates = list(_REPORTS_DIR.glob("ablative_v*.json"))
    if not candidates:
        return None
    latest = sorted(candidates)[-1]
    with open(latest, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────

st.markdown("## 🌐 Multi-Horizon Trainer — v1 Results")
st.caption(
    "Semana 7 — 3 calibrated models across intraday (5m), swing (4h), daily (1d)"
)
st.divider()

if not _REPORTS_DIR.exists() or not any(_REPORTS_DIR.iterdir()):
    st.warning(
        "No reports found in `research/reports/multi_horizon_v1/`. "
        "Run the trainer first:\n"
        "```bash\n"
        "python -m research.cli.train_multi_horizon --as-of 2026-05-01 --seed 42\n"
        "```"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────
# Sidebar — horizon selector
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔍 Horizon")
    selected_horizon = st.selectbox(
        "Select horizon",
        HORIZON_ORDER,
        format_func=lambda h: {"intraday": "Intraday (5m)", "swing": "Swing (4h)", "daily": "Daily (1d)"}[h],
    )

# ─────────────────────────────────────────────────────────────────────
# Cross-horizon overview metrics
# ─────────────────────────────────────────────────────────────────────

st.markdown("### 📊 Resumen de los 3 horizontes")

overview_cols = st.columns(3)
for i, horizon in enumerate(HORIZON_ORDER):
    report = _load_report(horizon)
    with overview_cols[i]:
        st.markdown(
            f"**{horizon.capitalize()} ({MODEL_MAP[horizon].upper()})**"
        )
        if report is None:
            st.warning("No report")
            continue
        m = report.get("metrics", report)
        psr = m.get("psr", 0.0)
        dsr = m.get("dsr", 0.0)
        ece = m.get("ece", 1.0)
        promoted = report.get("promoted", dsr >= 0.4)

        st.metric("PSR", f"{psr:.4f}")
        st.metric("DSR", f"{dsr:.4f}", delta="✅ promoted" if promoted else "❌ archived")
        st.metric("ECE", f"{ece:.4f}")
        if promoted:
            st.success("Promoted to staging")
        else:
            st.error(f"No edge — DSR={dsr:.4f} < 0.4")

st.divider()

# ─────────────────────────────────────────────────────────────────────
# Horizon detail panel
# ─────────────────────────────────────────────────────────────────────

report = _load_report(selected_horizon)

if report is None:
    st.info(f"No report found for **{selected_horizon}**.")
else:
    m = report.get("metrics", report)
    st.markdown(f"### 🔬 Detail — {selected_horizon.capitalize()}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("PSR",       f"{m.get('psr', 0.0):.4f}")
    col2.metric("DSR",       f"{m.get('dsr', 0.0):.4f}")
    col3.metric("ECE",       f"{m.get('ece', 1.0):.4f}")
    col4.metric("Sharpe OOS",f"{m.get('sharpe_oos', 0.0):.3f}")
    col5.metric("Win Rate",  f"{m.get('win_rate', 0.0):.2%}")

    st.markdown(f"**Trades OOS:** {m.get('n_trades_oos', m.get('n_trades', 'N/A'))}")
    st.markdown(f"**Training window:** {report.get('training_window', ['N/A', 'N/A'])}")
    st.markdown(f"**Universe size:** {report.get('universe_size', 'N/A')} symbols")
    st.markdown(f"**Seed:** {report.get('seed', 'N/A')}")

    # Feature importance
    feat_imp = report.get("feature_importance_top10", {})
    if feat_imp:
        st.divider()
        st.markdown("#### 🌟 Feature Importance — Top 10")
        df_imp = pd.DataFrame(
            list(feat_imp.items()), columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=False)
        st.bar_chart(df_imp.set_index("Feature"))

    # Hyperparams
    hp = report.get("hyperparams", {})
    if hp:
        st.divider()
        st.markdown("#### ⚙️ Best Hyperparameters")
        st.json(hp)

    # Ablative from report
    abl = report.get("ablative", {})
    if abl:
        st.divider()
        st.markdown("#### 🔬 Ablative Analysis")
        if isinstance(abl, dict):
            rows = [{"Ablation": k, "DSR": v.get("dsr", v) if isinstance(v, dict) else v}
                    for k, v in abl.items()]
        elif isinstance(abl, list):
            rows = [{"Ablation": a.get("label", "?"), "DSR": a.get("dsr", 0.0)}
                    for a in abl]
        else:
            rows = []
        if rows:
            df_abl = pd.DataFrame(rows)
            st.dataframe(df_abl, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────
# Cross-horizon ablative summary
# ─────────────────────────────────────────────────────────────────────

ablative = _load_ablative()
if ablative:
    st.markdown("### 🔬 Cross-Horizon Ablative Summary")
    rows_all = []
    for horizon, ablations in ablative.items():
        for label, dsr_val in ablations.items():
            rows_all.append({"Horizon": horizon, "Ablation": label, "DSR": dsr_val})
    if rows_all:
        df_all = pd.DataFrame(rows_all).pivot(
            index="Ablation", columns="Horizon", values="DSR"
        )
        st.dataframe(df_all.style.highlight_min(axis=1, color="#ff9999")
                                  .highlight_max(axis=1, color="#99ff99"),
                     use_container_width=True)

st.divider()
st.caption(
    "DSR corrected with n_trials=150 (3 horizons × 50 Optuna trials). "
    "ADR-029. Reports: research/reports/multi_horizon_v1/"
)
