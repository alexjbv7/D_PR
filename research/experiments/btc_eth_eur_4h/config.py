"""Experiment configuration and market-type helpers (no heavy deps).

Kept import-light on purpose: ``data_sources`` and the tests import this module
without pulling torch / xgboost / gymnasium / sklearn, so routing and
annualization logic stay unit-testable in isolation.

Annualization (the one correction vs the gate's daily default)
--------------------------------------------------------------
``models.drl.dsr_gate.evaluate_drl_gate`` defaults ``periods_per_year=252``
(daily bars). For 4H bars the Sharpe/DSR annualization factor is the number of
4H bars actually present in a year:

* Crypto (24/7): ``6 bars/day * 365 ≈ 2190``.
* FX (24/5, weekends closed): ``6 bars/day * ~260 trading days ≈ 1560``.

Passing the wrong factor mis-scales every annualized metric by
``sqrt(true_ppy / 252)`` (~2.9x for crypto), so ``periods_per_year`` is routed
per instrument from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

#: 4H bars per year, continuous 24/7 markets (crypto).
CRYPTO_PPY_4H: int = 2190
#: 4H bars per year, 24/5 markets (spot FX) — weekends excluded.
FX_PPY_4H: int = 1560

#: Quote/base currencies that mark an instrument as spot FX (yfinance route).
FIAT_BASES: frozenset[str] = frozenset(
    {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SEK", "NOK"}
)


def is_fx(symbol: str) -> bool:
    """Return True if ``symbol`` (e.g. ``"EUR/USD"``) is spot FX.

    The base leg (left of the slash) being a fiat currency distinguishes FX
    (``EUR/USD`` -> base ``EUR``) from crypto quoted in fiat (``BTC/USD`` ->
    base ``BTC``). Crypto routes to Alpaca; FX routes to yfinance.

    Parameters
    ----------
    symbol : str
        Instrument in ``BASE/QUOTE`` form.

    Returns
    -------
    bool
        True for spot FX, False otherwise.
    """
    base = symbol.split("/")[0].strip().upper()
    return base in FIAT_BASES


def periods_per_year(symbol: str, timeframe: str = "4h") -> int:
    """Annualization factor for ``symbol`` at ``timeframe``.

    Parameters
    ----------
    symbol : str
        Instrument in ``BASE/QUOTE`` form.
    timeframe : str
        Bar timeframe. Only ``"4h"`` is supported by this experiment.

    Returns
    -------
    int
        Bars per year for Sharpe/DSR annualization.

    Raises
    ------
    NotImplementedError
        If ``timeframe`` is not ``"4h"``.
    """
    if timeframe != "4h":
        raise NotImplementedError(
            f"this experiment is 4H-only; got timeframe={timeframe!r}"
        )
    return FX_PPY_4H if is_fx(symbol) else CRYPTO_PPY_4H


@dataclass
class ExperimentConfig:
    """Parameters for the 4H BTC/ETH/EUR XGBoost-vs-DQN run.

    Parameters
    ----------
    symbols : tuple[str, ...]
        Instruments to evaluate (BASE/QUOTE form).
    timeframe : str
        Bar timeframe (4H only).
    crypto_start : str
        ISO date for the crypto history start (Alpaca crypto begins ~2021).
    fx_lookback_days : int
        FX history window in days. yfinance caps 1H bars at ~730 days, so 4H
        FX history is bounded; this keeps the request inside that window.
    end : str | None
        ISO end date; ``None`` means "today" (UTC).
    n_folds : int
        Walk-forward OOS folds (ADR-040).
    n_seeds : int
        DQN seeds; the confidence interval comes from their dispersion and the
        agent DSR is deflated by ``n_seeds`` (honest selection-bias deflation).
    episodes : int
        DQN training episodes per fold.
    dsr_threshold : float
        Minimum deflated Sharpe for promotion condition 1 (ADR-040 §3.2).
    fee_bps : float
        Proportional fee in basis points (shared by env, baselines and gate).
    episode_length : int
        4H bars per training episode (~180 bars ≈ 30 days).
    device : str
        Torch device ("cpu" or "cuda").
    n_jobs : int
        Folds trained concurrently (ADR-040 §6).
    seed : int
        Base seed (XGBoost random_state and first DQN seed).
    feed : str
        Alpaca data feed ("iex" free, "sip" paid).
    out_dir : str
        Directory (relative to research/) for the JSON run log.
    """

    symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD", "EUR/USD")
    timeframe: str = "4h"
    crypto_start: str = "2021-01-01"
    fx_lookback_days: int = 700
    end: str | None = None
    n_folds: int = 4
    n_seeds: int = 5
    episodes: int = 150
    dsr_threshold: float = 0.4
    fee_bps: float = 5.0
    episode_length: int = 180
    device: str = "cpu"
    n_jobs: int = 1
    seed: int = 42
    feed: str = "iex"
    out_dir: str = "artifacts/runs"

    def end_date(self) -> str:
        """ISO end date, defaulting to today (UTC)."""
        if self.end:
            return self.end
        return datetime.now(timezone.utc).date().isoformat()

    def start_for(self, symbol: str) -> str:
        """ISO start date for ``symbol`` (FX is windowed to the yfinance cap)."""
        if is_fx(symbol):
            end = datetime.fromisoformat(self.end_date()).replace(tzinfo=timezone.utc)
            return (end - timedelta(days=self.fx_lookback_days)).date().isoformat()
        return self.crypto_start

    @classmethod
    def smoke(cls) -> "ExperimentConfig":
        """Tiny config for a wiring smoke test (fast, synthetic-data sized)."""
        return cls(
            n_folds=2, n_seeds=1, episodes=2, episode_length=40, fx_lookback_days=120
        )
