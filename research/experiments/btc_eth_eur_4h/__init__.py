"""BTC/USD + ETH/USD + EUR/USD on 4H bars — XGBoost vs DQN (ADR-040 gate).

Study scope
-----------
Compare the DRL agent (DQN) against the supervised baseline (XGBoost) and
buy-and-hold on three 4H instruments, using the EXISTING walk-forward DSR
promotion gate (``models.drl.dsr_gate``, ADR-040). The only new surface here is

* ``data_sources`` — a unified 4H OHLCV loader: crypto (BTC/ETH) via the
  canonical ``data.drl_dataset.fetch_ohlcv_frame`` (Alpaca), EUR/USD via
  yfinance (Alpaca has no FX), both returning the same
  ``[open, high, low, close, volume]`` UTC schema the gate consumes.
* ``config`` — symbols, date windows and the 4H annualization factors
  (crypto 24/7 vs FX 24/5), which differ from the gate's daily default of 252.
* ``run_experiment`` — the orchestrator that calls the gate per asset.

Everything statistical (feature build, anti-leakage GMM per fold, DQN training,
XGBoost folds, DSR/PSR, the three §3.2 promotion conditions) is delegated to
the audited gate — this package adds no new modeling assumptions.
"""
