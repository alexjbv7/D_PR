"""
Tests del screening de pares E5 (ADR-043 · Rama B).

Cubren la lógica pura (generación de pares, alineación, sizing del splitter,
ranking) y el veredicto vs ZERO con deflación. ``screen_pair`` sobre datos reales
usa statsmodels y se ejercita en el venv con los parquet de 4h.
"""
import numpy as np
import pandas as pd
import pytest

from alpha.statarb.screen import (
    PairScreenResult,
    align_pair_prices,
    candidate_pairs,
    evaluate_screen,
    make_pair_splitter,
    rank_pairs,
)


# ---- generación de pares ----

def test_candidate_pairs_anchor():
    pairs = candidate_pairs(["XRP/USD", "BTC/USD", "ETH/USD"], anchor="XRP/USD")
    assert pairs == [("XRP/USD", "BTC/USD"), ("XRP/USD", "ETH/USD")]


def test_candidate_pairs_all_combinations():
    pairs = candidate_pairs(["A", "B", "C"])
    assert set(pairs) == {("A", "B"), ("A", "C"), ("B", "C")}


def test_candidate_pairs_anchor_not_in_universe_raises():
    with pytest.raises(ValueError):
        candidate_pairs(["BTC/USD", "ETH/USD"], anchor="XRP/USD")


# ---- alineación ----

def test_align_pair_prices_intersection():
    idx_a = pd.date_range("2021-01-01", periods=10, freq="4h", tz="UTC")
    idx_b = idx_a[2:]                          # solapa parcial
    a = pd.DataFrame({"close": np.arange(10.0)}, index=idx_a)
    b = pd.DataFrame({"close": np.arange(8.0) + 100}, index=idx_b)
    frame = align_pair_prices({"A": a, "B": b}, "A", "B")
    assert list(frame.columns) == ["y", "x"]
    assert len(frame) == 8                     # intersección
    assert frame["x"].iloc[0] == 100.0


def test_align_pair_missing_symbol_raises():
    a = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError):
        align_pair_prices({"A": a}, "A", "B")


# ---- splitter ----

def test_make_pair_splitter_produces_folds():
    sp = make_pair_splitter(1500, n_folds=5)
    folds = list(sp.split(pd.DataFrame(index=np.arange(1500))))
    assert len(folds) == 5
    assert sp.embargo >= 60


def test_make_pair_splitter_too_few_bars_raises():
    with pytest.raises(ValueError):
        make_pair_splitter(120, n_folds=5)


# ---- ranking ----

def test_rank_pairs_by_lb95():
    r1 = PairScreenResult("A", "B", sharpe=0.5, lb95=0.10, p95=1.0, n_oos=500, frac_traded=0.5)
    r2 = PairScreenResult("A", "C", sharpe=0.3, lb95=0.40, p95=0.9, n_oos=500, frac_traded=0.6)
    ranked = rank_pairs([r1, r2])
    assert ranked[0].x_sym == "C"              # mayor LB95 primero


# ---- veredicto vs ZERO ----

def _res(y, x, lb95, returns):
    return PairScreenResult(y, x, sharpe=float(np.mean(returns)) * 252, lb95=lb95,
                            p95=lb95 + 0.3, n_oos=len(returns),
                            frac_traded=float(np.mean(returns != 0)), oos_returns=returns)


def test_evaluate_screen_viable():
    rng = np.random.default_rng(0)
    strong = rng.normal(0.003, 0.005, 900)     # Sharpe alto → DSR pasa
    v = evaluate_screen([_res("XRP/USD", "BTC/USD", 0.8, strong)])
    assert v.branch == "VIABLE" and v.gate_passed
    assert v.best.label == "XRP/USD/BTC/USD"


def test_evaluate_screen_none_when_lb95_negative():
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 0.01, 900)
    v = evaluate_screen([_res("XRP/USD", "ETH/USD", -0.2, noise)])
    assert v.branch == "NONE" and not v.gate_passed


def test_evaluate_screen_marginal_when_gate_fails():
    short = np.array([0.01, 0.02, 0.015])      # <4 barras → gate ZERO falla
    v = evaluate_screen([_res("XRP/USD", "SOL/USD", 0.10, short)])
    assert v.branch == "MARGINAL"


def test_evaluate_screen_empty_raises():
    with pytest.raises(ValueError):
        evaluate_screen([])
