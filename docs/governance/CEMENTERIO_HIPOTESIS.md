# Cementerio de hipótesis de alfa

Hipótesis evaluadas y **formalmente invalidadas** (no reabrir sin datos nuevos y re-gate).

---

## stock.position.dqn_directional (A-001) — 2026-07-17

| Campo | Valor |
|-------|--------|
| **ID de hipótesis** | `stock.position.dqn_directional` (`research/alpha/agents/dqn_agent.py` · `DQN_HYPOTHESIS`) |
| **Tesis** | Política DQN con reward MTM sobre features diarios + GMM captura tendencias multi-día mejor que buy-and-hold. |
| **Invalidación declarada** | DSR deflactado ≤ 0.4 OOS, o Sharpe OOS ≤ Sharpe B&H en el mismo walk-forward. |
| **Resultado empírico** | Gate FAIL en ≥3 runs: Sharpe agente 0.30–0.55 vs B&H 0.8–1.3 (SPY/BTC logs en repo). Condición B&H fallida de forma consistente. |
| **Estado** | **MUERTA** — no promocionar. Código y gate se mantienen para harness y comparación. |
| **Siguiente paso** | Hipótesis market-neutral / stat-arb (ADR-043) u otras tesis falsables con SNR creíble. |
