"""Wiring tests for the PPO/SAC gate extension ``models.drl.multi_algo_gate``.

Training functions are monkeypatched, so these validate dispatch + the gate
verdict plumbing (correct periods_per_year, n_trials = n_seeds, median-seed
selection, dqn delegation, unsupported-algo error) WITHOUT torch. Importing the
module needs gymnasium/sklearn (present in the repo venv / on the Brev box);
they are not exercised at runtime here.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

mag = pytest.importorskip("models.drl.multi_algo_gate")
from models.drl.dsr_gate import AgentSpec  # noqa: E402
from envs import EnvironmentConfig  # noqa: E402


def _fake_gate_result(agent_r):
    return types.SimpleNamespace(
        dsr_agent=0.5, psr_agent=0.6, sharpe_agent=1.0, sharpe_buyhold=0.2,
        dsr_xgb=0.1, n_oos_bars=len(agent_r), passed=True, reason="PASS",
    )


def test_run_gate_passes_correct_ppy_and_deflation(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(mag, "make_wf_splitter", lambda o, nf, env_cfg=None: "SPL")
    monkeypatch.setattr(mag, "xgb_oos_returns",
                        lambda o, s, fee_bps=None, seed=42: np.random.default_rng(1).normal(0, 0.01, 100))
    monkeypatch.setattr(mag, "buyhold_oos_returns",
                        lambda o, s, fee_bps=None: np.random.default_rng(2).normal(0, 0.01, 100))

    def fake_eval(a, b, x, n_trials, dsr_threshold=0.4, periods_per_year=252):
        seen["ppy"] = periods_per_year
        seen["n_trials"] = n_trials
        return _fake_gate_result(a)

    monkeypatch.setattr(mag, "evaluate_drl_gate", fake_eval)

    # Three seeds with increasing mean → median is the middle series.
    rng = np.random.default_rng(0)
    seqs = iter([rng.normal(m, 0.01, 100) for m in (0.0, 0.001, 0.002)])
    monkeypatch.setattr(mag, "walk_forward_oos_returns",
                        lambda spec, o, s, e, seed=42, n_jobs=1: next(seqs))

    res = mag.run_gate(None, "dqn", n_folds=4, episodes=2, seeds=[1, 2, 3],
                       env_cfg=EnvironmentConfig(), periods_per_year=2190)

    assert seen["ppy"] == 2190           # 4H crypto annualization, not daily 252
    assert seen["n_trials"] == 3         # deflate by number of seeds searched
    assert res["algo"] == "dqn" and res["n_seeds"] == 3 and res["passed"] is True


def test_walk_forward_routing(monkeypatch) -> None:
    monkeypatch.setattr(mag, "_dqn_walk_forward_oos_returns", lambda *a, **k: np.array([0.1]))
    monkeypatch.setattr(mag, "_validated_folds",
                        lambda o, s: iter([(np.array([0, 1]), np.array([2, 3])),
                                           (np.array([0, 1, 2]), np.array([3, 4]))]))
    monkeypatch.setattr(mag, "_train_eval_one_fold",
                        lambda k, tr, te, o, e, spec, seed, tpw=0: (k, np.array([float(k)])))
    monkeypatch.setattr(mag, "_concat_fold_returns",
                        lambda res: np.concatenate([r for _, r in sorted(res)]))

    env = EnvironmentConfig()
    # dqn delegates to the audited gate path
    assert mag.walk_forward_oos_returns(AgentSpec(algo="dqn"), None, "SPL", env)[0] == 0.1
    # ppo uses the per-fold trainer over both folds
    out = mag.walk_forward_oos_returns(AgentSpec(algo="ppo"), None, "SPL", env, n_jobs=1)
    assert len(out) == 2
    # unsupported algo rejected
    with pytest.raises(NotImplementedError):
        mag.walk_forward_oos_returns(AgentSpec(algo="foo"), None, "SPL", env)


def test_supported_algos_constant() -> None:
    assert mag.SUPPORTED_ALGOS == ("dqn", "ppo", "sac")
