"""
Dashboard State — Persistencia de Resultados
=============================================
Guarda y carga WalkForwardResult / HyperoptResult como pickle
en cache/dashboard/ para que el dashboard no necesite reejecutar
el pipeline en cada refresh.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "dashboard"


def _ensure_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Walk-Forward Result
# ─────────────────────────────────────────────────────────────────────

def save_wf_result(result, symbol: str = "EURUSD") -> None:
    """Serializa WalkForwardResult a disco."""
    _ensure_dir()
    path = CACHE_DIR / f"wf_result_{symbol}.pkl"
    with open(path, "wb") as f:
        pickle.dump(result, f)

    # Metadata legible (JSON)
    meta = {
        "symbol": symbol,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "n_folds": len(result.fold_results),
        "n_trades": result.global_metrics.get("n_trades", 0),
        "sharpe": result.global_metrics.get("sharpe"),
        "psr": result.global_metrics.get("psr"),
        "dsr": result.global_metrics.get("dsr"),
        "coverage": result.global_metrics.get("coverage"),
    }
    with open(CACHE_DIR / f"wf_meta_{symbol}.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_wf_result(symbol: str = "EURUSD"):
    """Carga WalkForwardResult desde disco. Devuelve None si no existe."""
    path = CACHE_DIR / f"wf_result_{symbol}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_wf_meta(symbol: str = "EURUSD") -> Optional[dict]:
    path = CACHE_DIR / f"wf_meta_{symbol}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# Hyperopt Result
# ─────────────────────────────────────────────────────────────────────

def save_ho_result(result, symbol: str = "EURUSD") -> None:
    """Serializa HyperoptResult a disco."""
    _ensure_dir()
    path = CACHE_DIR / f"ho_result_{symbol}.pkl"
    with open(path, "wb") as f:
        pickle.dump(result, f)

    meta = {
        "symbol": symbol,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "n_trials_completed": result.n_trials_completed,
        "n_trials_pruned": result.n_trials_pruned,
        "best_value": result.best_value,
        "best_barrier": result.best_barrier_params,
    }
    with open(CACHE_DIR / f"ho_meta_{symbol}.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_ho_result(symbol: str = "EURUSD"):
    path = CACHE_DIR / f"ho_result_{symbol}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_ho_meta(symbol: str = "EURUSD") -> Optional[dict]:
    path = CACHE_DIR / f"ho_meta_{symbol}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# Prices cache
# ─────────────────────────────────────────────────────────────────────

def save_prices(prices_df, symbol: str) -> None:
    _ensure_dir()
    prices_df.to_parquet(CACHE_DIR / f"prices_{symbol}.parquet")


def load_prices(symbol: str):
    import pandas as pd
    path = CACHE_DIR / f"prices_{symbol}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def list_available_symbols() -> list[str]:
    """Lista símbolos con resultados guardados."""
    _ensure_dir()
    return sorted({
        p.stem.replace("wf_result_", "")
        for p in CACHE_DIR.glob("wf_result_*.pkl")
    })
