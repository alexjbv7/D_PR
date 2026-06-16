"""
Tests del diagnóstico de potencia E4 (arbitraje D · núcleo torch-free).

Verifican el SE del Sharpe (Lo/Mertens), el criterio de resolución de potencia y
el N efectivo por fold. La deflación (reusa walk_forward_runner) se ejercita aparte.
"""
import math

import numpy as np
import pytest

from models.drl.e4_power_diagnostics import (
    effective_fold_lengths,
    sharpe_power_diagnostic,
    sharpe_standard_error,
)


def test_se_formula_matches_closed_form():
    se = sharpe_standard_error(0.5, 300, ppy=252)
    expected = math.sqrt((252 + 0.5 * 0.5 ** 2) / 300)
    assert se == pytest.approx(expected)


def test_se_decreases_with_more_data():
    assert sharpe_standard_error(0.5, 5000) < sharpe_standard_error(0.5, 300)


def test_se_infinite_for_degenerate_n():
    assert math.isinf(sharpe_standard_error(0.5, 1))
    assert math.isinf(sharpe_standard_error(0.5, 0))


def test_short_oos_is_underpowered():
    # ~1.2 años de barras diarias: el SE anualizado domina al Sharpe.
    res = sharpe_power_diagnostic(0.5, n_obs=300, ppy=252, ref_gap=0.5)
    assert res.verdict == "UNDERPOWERED"
    assert not res.can_resolve_gap
    assert res.margin_of_error > 0.25          # IC más ancho que media separación


def test_large_sample_becomes_powered():
    # Con muchísimas barras el margen cae por debajo de gap/2.
    res = sharpe_power_diagnostic(0.5, n_obs=200_000, ppy=252, ref_gap=0.5)
    assert res.verdict == "POWERED"
    assert res.can_resolve_gap
    assert res.ci95_low < res.sharpe_ann < res.ci95_high


def test_power_ci_brackets_point_estimate():
    res = sharpe_power_diagnostic(0.8, n_obs=1000)
    assert res.ci95_low < 0.8 < res.ci95_high
    assert res.margin_of_error == pytest.approx(res.sharpe_ann - res.ci95_low)


def test_effective_fold_lengths():
    rep = effective_fold_lengths([120, 130, 110, 140])
    assert rep.n_folds == 4
    assert rep.total_oos == 500
    assert rep.min_fold == 110
    assert rep.mean_fold == pytest.approx(125.0)


def test_effective_fold_lengths_empty_raises():
    with pytest.raises(ValueError):
        effective_fold_lengths([])
