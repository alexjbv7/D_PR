"""Wiring tests for the master driver ``pipelines.train_all`` (no heavy deps).

Stage functions are monkeypatched, so this validates orchestration logic
(stage selection, report shape, config parsing) without torch/xgboost/gymnasium.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pipelines import train_all as TA


def test_config_from_args_defaults() -> None:
    cfg = TA._config_from_args(TA._parse_args([]))
    assert cfg.assets == ("BTC/USD", "ETH/USD", "EUR/USD")
    assert cfg.algos == ("dqn", "ppo", "sac")
    assert set(cfg.stages) == {"drl", "supervised", "statarb"}


def test_config_from_args_subsets() -> None:
    cfg = TA._config_from_args(
        TA._parse_args(["--stages", "drl", "--algos", "ppo,sac", "--assets", "BTC/USD"])
    )
    assert cfg.stages == ("drl",)
    assert cfg.algos == ("ppo", "sac")
    assert cfg.assets == ("BTC/USD",)


def test_smoke_config_is_tiny() -> None:
    cfg = TA.TrainAllConfig.smoke()
    assert cfg.n_seeds == 1 and cfg.episodes == 2 and cfg.n_folds == 2


def _stub_drl(symbol, ohlcv, cfg):
    return {
        "dqn": {
            "algo": "dqn", "dsr_agent": 0.5, "sharpe_agent": 1.0, "dsr_xgb": 0.1,
            "sharpe_buyhold": 0.2, "passed": True, "reason": "PASS",
        }
    }


def test_run_all_only_selected_stages(monkeypatch) -> None:
    calls = {"load": 0, "drl": 0, "sup": 0, "stat": 0}
    monkeypatch.setattr(TA, "_load_asset",
                        lambda s, c: calls.__setitem__("load", calls["load"] + 1) or pd.DataFrame({"close": [1.0, 2.0]}))
    monkeypatch.setattr(TA, "run_drl_for_asset",
                        lambda s, o, c: (calls.__setitem__("drl", calls["drl"] + 1), _stub_drl(s, o, c))[1])
    monkeypatch.setattr(TA, "run_supervised_for_asset",
                        lambda s, o, c: calls.__setitem__("sup", calls["sup"] + 1) or {})
    monkeypatch.setattr(TA, "run_statarb",
                        lambda c: calls.__setitem__("stat", calls["stat"] + 1) or {})

    cfg = TA.TrainAllConfig(assets=("BTC/USD", "ETH/USD"), stages=("drl",))
    report = TA.run_all(cfg)

    assert calls["drl"] == 2 and calls["sup"] == 0 and calls["stat"] == 0
    assert set(report["results"]["drl"].keys()) == {"BTC/USD", "ETH/USD"}
    assert report["experiment"] == "train_all"


def test_run_all_statarb_only_skips_asset_loop(monkeypatch) -> None:
    calls = {"load": 0, "stat": 0}
    monkeypatch.setattr(TA, "_load_asset",
                        lambda s, c: calls.__setitem__("load", calls["load"] + 1) or pd.DataFrame({"close": [1.0]}))
    monkeypatch.setattr(TA, "run_statarb",
                        lambda c: calls.__setitem__("stat", calls["stat"] + 1) or {"BTC/USD~ETH/USD": {"dsr": 0.3, "sharpe": 0.5, "passed": False}})

    report = TA.run_all(TA.TrainAllConfig(stages=("statarb",)))
    assert calls["stat"] == 1 and calls["load"] == 0
    assert "BTC/USD~ETH/USD" in report["results"]["statarb"]


def test_format_report_handles_errors() -> None:
    res = {
        "drl": {"BTC/USD": {"dqn": {"error": "boom"}}},
        "supervised": {},
        "statarb": {"BTC/USD~ETH/USD": {"error": "nope"}},
    }
    out = TA._format_report(res)
    assert "ERROR" in out


def test_config_jsonable_pairs_are_lists() -> None:
    cfg = TA.TrainAllConfig()
    d = TA._config_to_jsonable(cfg)
    assert d["pairs"] == [["BTC/USD", "ETH/USD"]]
