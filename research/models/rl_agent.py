"""
RL Agent — Tabular Q-learning para Trading
===========================================
Implementación del Q-learning tabular del CS229 Deep Learning cheatsheet.

MDP de trading:
  Estado  s = (regime_bin, p_win_bin, trend_bin)
  Acciones a ∈ {-1, 0, +1}  (short, flat, long)
  Reward  R(s,a) = retorno ajustado por ATR del bar siguiente × signo(a)
  Bellman: Q(s,a) ← Q(s,a) + α [R + γ max_a' Q(s',a') − Q(s,a)]

Protocolo anti-leakage:
  - El agente se entrena SOLO sobre X_fit (o X_calib).
  - update_step() consume barras en orden temporal (no shuffle).
  - En producción, act() solo consulta Q sin modificarlo.

Por qué tabular y no DQN aquí:
  - Espacios de estado pequeños (3×4×3 = 36 estados) → tabla suficiente.
  - DQN necesita miles de episodios y añade inestabilidad difícil de
    interpretar en datos financieros escasos.
  - Si tu espacio de estado crece, considera DQN; aquí el tabular es
    más robusto y debuggeable.

Uso:
    from models.rl_agent import QLearningAgent, QLearningConfig

    cfg = QLearningConfig(alpha=0.1, gamma=0.95, epsilon=0.1)
    agent = QLearningAgent(cfg)
    agent.train(X_fit, rewards, regime_labels, p_win_series, price_returns)
    signal = agent.act(state)
    report = agent.summary()
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Discretización del espacio de estado
# ---------------------------------------------------------------------------

# Régimen: bins 0..N_REGIME-1 (índice del estado GMM)
# p_win:   cuatro bins de confianza
# trend:   -1=bajista, 0=lateral, +1=alcista → bins 0,1,2

_PWIN_BINS   = [0.0, 0.45, 0.55, 0.65, 1.01]   # 4 bins
_PWIN_LABELS = [0, 1, 2, 3]

_ACTIONS     = [-1, 0, 1]   # short, flat, long
_ACTION_IDX  = {a: i for i, a in enumerate(_ACTIONS)}

N_REGIME_DEFAULT = 3
N_PWIN           = 4
N_TREND          = 3   # 0=down, 1=flat, 2=up


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class QLearningConfig:
    """
    Hiperparámetros del Q-learning tabular.

    alpha   : learning rate (0.05–0.2 típico para finanzas)
    gamma   : discount factor (0.9–0.99)
    epsilon : exploración ε-greedy durante training (0.05–0.2)
    epsilon_decay : multiplicador por episodio (0.999 → decay gradual)
    epsilon_min   : piso de exploración
    n_regimes     : número de regímenes GMM (dimension del estado)
    reward_scale  : normaliza rewards dividiendo por este valor (e.g. ATR medio)
    transaction_cost : coste por cambio de posición (fracción del retorno)
    """
    alpha:            float = 0.10
    gamma:            float = 0.95
    epsilon:          float = 0.15
    epsilon_decay:    float = 0.999
    epsilon_min:      float = 0.02
    n_regimes:        int   = N_REGIME_DEFAULT
    reward_scale:     float = 1.0
    transaction_cost: float = 0.001


# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Agente Q-learning tabular para decisiones de trading.

    Estado: tupla (regime_bin, p_win_bin, trend_bin)
    Tabla Q: dict[(estado, acción)] → valor
    Inicialización optimista: 0.0 (favorece exploración inicial)

    Entrenamiento:
      Iterar barras en orden temporal. Para cada barra:
        1. Construir estado s_t
        2. Con prob ε elegir acción aleatoria, si no argmax Q(s_t, ·)
        3. Observar reward r_t
        4. Construir estado s_{t+1}
        5. Bellman update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

    Predicción (act):
      Solo ejecuta argmax Q(s, ·) — sin exploración.
    """

    def __init__(self, config: Optional[QLearningConfig] = None):
        self.cfg = config or QLearningConfig()
        self._Q: dict = {}           # (state_tuple, action_idx) → float
        self._epsilon = self.cfg.epsilon
        self._is_trained = False
        self._train_stats: dict = {}

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        price_returns: pd.Series,
        regime_labels: Optional[pd.Series] = None,
        p_win_series: Optional[pd.Series] = None,
        primary_signals: Optional[pd.Series] = None,
    ) -> "QLearningAgent":
        """
        Entrena el agente sobre datos históricos en orden temporal.

        Parameters
        ----------
        X               : features (mismo índice que price_returns)
        price_returns   : retornos (barra siguiente) alineados con X.
                          R_t = price_{t+1} / price_t - 1
        regime_labels   : etiquetas GMM (int 0..n_regimes-1). Si None → 0.
        p_win_series    : probabilidad de señal correcta [0,1]. Si None → 0.5.
        primary_signals : señales del modelo primario {-1,0,+1}. Si None → 0.
        """
        cfg = self.cfg
        n = len(X)
        if n < 2:
            logger.warning("QLearningAgent: datos insuficientes para entrenar")
            return self

        rewards = self._compute_rewards(
            price_returns=price_returns,
            primary_signals=primary_signals,
            cost=cfg.transaction_cost,
            scale=cfg.reward_scale,
        )

        # Construir estados para cada barra
        states = self._build_states(X, regime_labels, p_win_series)

        # Estadísticas de entrenamiento
        total_reward = 0.0
        n_updates = 0
        prev_action_idx = 1  # empieza flat

        for t in range(n - 1):
            s = states[t]
            s_next = states[t + 1]
            r = rewards[t]

            # ε-greedy
            if np.random.rand() < self._epsilon:
                a_idx = np.random.randint(len(_ACTIONS))
            else:
                a_idx = self._best_action(s)

            # Bellman update
            q_sa = self._Q.get((s, a_idx), 0.0)
            q_next_max = max(
                self._Q.get((s_next, ai), 0.0) for ai in range(len(_ACTIONS))
            )
            td_error = r + cfg.gamma * q_next_max - q_sa
            self._Q[(s, a_idx)] = q_sa + cfg.alpha * td_error

            total_reward += r
            n_updates += 1
            prev_action_idx = a_idx

            # Epsilon decay
            self._epsilon = max(cfg.epsilon_min, self._epsilon * cfg.epsilon_decay)

        self._is_trained = True
        self._train_stats = {
            "n_updates":    n_updates,
            "total_reward": round(total_reward, 4),
            "avg_reward":   round(total_reward / max(n_updates, 1), 6),
            "n_states":     len(set(s for s, _ in self._Q)),
            "epsilon_final": round(self._epsilon, 4),
        }
        logger.info(
            f"QLearningAgent entrenado: {n_updates} updates, "
            f"avg_reward={self._train_stats['avg_reward']:.5f}, "
            f"estados únicos={self._train_stats['n_states']}"
        )
        return self

    def act(
        self,
        X_row: pd.Series | pd.DataFrame,
        regime: int = 0,
        p_win: float = 0.5,
    ) -> int:
        """
        Elige la acción greedy (sin exploración) dado el estado actual.

        Parameters
        ----------
        X_row  : una fila de features (pd.Series o DataFrame de 1 fila)
        regime : índice del régimen actual (0..n_regimes-1)
        p_win  : probabilidad de señal correcta [0,1]

        Returns
        -------
        int en {-1, 0, +1}
        """
        if isinstance(X_row, pd.DataFrame):
            X_row = X_row.iloc[0]
        trend_bin = self._trend_bin(X_row)
        p_win_bin = self._pwin_bin(float(p_win))
        regime_bin = int(np.clip(regime, 0, self.cfg.n_regimes - 1))
        s = (regime_bin, p_win_bin, trend_bin)
        a_idx = self._best_action(s)
        return int(_ACTIONS[a_idx])

    def act_series(
        self,
        X: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
        p_win_series: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Aplica act() sobre todo un DataFrame. Devuelve señales {-1,0,+1}.
        """
        states = self._build_states(X, regime_labels, p_win_series)
        actions = [_ACTIONS[self._best_action(s)] for s in states]
        return pd.Series(actions, index=X.index, name="rl_signal")

    def q_table(self) -> pd.DataFrame:
        """
        Devuelve la tabla Q como DataFrame legible.
        Columnas: regime_bin, p_win_bin, trend_bin, action, q_value.
        """
        rows = []
        for (state, a_idx), q_val in self._Q.items():
            rows.append({
                "regime_bin": state[0],
                "p_win_bin":  state[1],
                "trend_bin":  state[2],
                "action":     _ACTIONS[a_idx],
                "q_value":    round(q_val, 6),
            })
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .sort_values(["regime_bin", "p_win_bin", "trend_bin", "action"])
            .reset_index(drop=True)
        )

    def summary(self) -> str:
        lines = [
            "=" * 55,
            " Q-LEARNING AGENT — Resumen",
            "=" * 55,
            f"  Trained     : {self._is_trained}",
            f"  n_states_Q  : {len(set(s for s, _ in self._Q))}",
            f"  n_entries_Q : {len(self._Q)}",
            f"  epsilon_cur : {self._epsilon:.4f}",
        ]
        for k, v in self._train_stats.items():
            lines.append(f"  {k:<18}: {v}")
        lines.append("=" * 55)
        return "\n".join(lines)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _best_action(self, state: tuple) -> int:
        """Índice de la acción con mayor Q(s,·). Empate → flat (idx=1)."""
        q_vals = [self._Q.get((state, ai), 0.0) for ai in range(len(_ACTIONS))]
        return int(np.argmax(q_vals))

    def _build_states(
        self,
        X: pd.DataFrame,
        regime_labels: Optional[pd.Series],
        p_win_series: Optional[pd.Series],
    ) -> list:
        """
        Construye lista de tuplas de estado (regime_bin, p_win_bin, trend_bin)
        para cada barra de X.
        """
        n = len(X)

        # Régimen
        if regime_labels is not None:
            reg_arr = regime_labels.reindex(X.index).fillna(0).values.astype(int)
            reg_arr = np.clip(reg_arr, 0, self.cfg.n_regimes - 1)
        else:
            reg_arr = np.zeros(n, dtype=int)

        # P(win)
        if p_win_series is not None:
            pw_arr = p_win_series.reindex(X.index).fillna(0.5).values
        else:
            pw_arr = np.full(n, 0.5)

        states = []
        for i in range(n):
            trend_bin = self._trend_bin(X.iloc[i])
            p_win_bin = self._pwin_bin(float(pw_arr[i]))
            regime_bin = int(reg_arr[i])
            states.append((regime_bin, p_win_bin, trend_bin))
        return states

    def _pwin_bin(self, p: float) -> int:
        """Discretiza p_win en 4 bins: 0=muy_bajo, 1=bajo, 2=medio, 3=alto."""
        p = float(np.clip(p, 0.0, 1.0))
        for i, (lo, hi) in enumerate(zip(_PWIN_BINS[:-1], _PWIN_BINS[1:])):
            if lo <= p < hi:
                return i
        return len(_PWIN_LABELS) - 1

    def _trend_bin(self, row: pd.Series) -> int:
        """
        Discretiza la tendencia a partir de columnas disponibles en la fila.
        Busca (en orden): 'trend', 'sma_ratio', 'return_1' como proxy.
        Devuelve 0=bajista, 1=lateral, 2=alcista.
        """
        val = None
        for col in ('trend', 'sma_ratio', 'return_1', 'close_ret_1'):
            if col in row.index:
                val = float(row[col])
                break

        if val is None:
            return 1  # neutral por defecto

        if col == 'sma_ratio':
            # sma_ratio = price/SMA: >1 = alcista, <1 = bajista
            if val > 1.005:
                return 2
            elif val < 0.995:
                return 0
            return 1
        else:
            # Retorno u otro indicador de tendencia
            if val > 0.001:
                return 2
            elif val < -0.001:
                return 0
            return 1

    @staticmethod
    def _compute_rewards(
        price_returns: pd.Series,
        primary_signals: Optional[pd.Series],
        cost: float,
        scale: float,
    ) -> np.ndarray:
        """
        R_t = signal_t × return_{t+1} - cost × |signal_t - signal_{t-1}|

        El término de coste penaliza cambios frecuentes de posición.
        Si no hay señal primaria, el reward es simplemente el retorno.
        """
        ret = price_returns.values.astype(float)
        n = len(ret)

        if primary_signals is not None:
            sig = primary_signals.reindex(price_returns.index).fillna(0).values
        else:
            sig = np.ones(n)

        # R_t = sig_t × ret_t - coste × |Δsig_t|
        rewards = np.zeros(n)
        prev_sig = 0.0
        for t in range(n):
            trade_ret = float(sig[t]) * float(ret[t])
            turnover_cost = cost * abs(float(sig[t]) - prev_sig)
            rewards[t] = (trade_ret - turnover_cost) / (scale if scale != 0 else 1.0)
            prev_sig = float(sig[t])

        return rewards


# ---------------------------------------------------------------------------
# Integración con WalkForwardRunner: función de conveniencia
# ---------------------------------------------------------------------------

def train_rl_agent(
    X_fit: pd.DataFrame,
    price_returns: pd.Series,
    regime_labels: Optional[pd.Series] = None,
    p_win_series: Optional[pd.Series] = None,
    primary_signals: Optional[pd.Series] = None,
    config: Optional[QLearningConfig] = None,
) -> QLearningAgent:
    """
    Entrena un Q-learning agent sobre el fold de entrenamiento.

    Convenience function para llamar desde WalkForwardRunner o notebooks.

    Parameters
    ----------
    X_fit          : features del train (anti-leakage: solo X_fit, no X_calib)
    price_returns  : retornos forward-shifted del train
    regime_labels  : etiquetas GMM del train
    p_win_series   : p_win del modelo primario en el train
    primary_signals: señales del modelo primario en el train
    config         : QLearningConfig; si None usa defaults

    Returns
    -------
    QLearningAgent entrenado
    """
    agent = QLearningAgent(config)
    agent.train(
        X=X_fit,
        price_returns=price_returns,
        regime_labels=regime_labels,
        p_win_series=p_win_series,
        primary_signals=primary_signals,
    )
    return agent
