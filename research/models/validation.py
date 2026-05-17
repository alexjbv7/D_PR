"""
Validation Module
=================
Implementaciones de validación robusta para series temporales financieras.

Incluye:
- WalkForwardSplitter: rolling/expanding window en cronológico
- PurgedKFold: K-Fold con purge + embargo (López de Prado)
- TimeSeriesCV con ventanas

REGLA: Nunca, jamás, uses sklearn.model_selection.KFold con datos financieros.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class WalkForwardSplitter:
    """
    Walk-forward validation splitter.

    Parameters
    ----------
    train_size : int
        Número de muestras para entrenamiento en cada fold.
    test_size : int
        Número de muestras para test en cada fold.
    step : int, optional
        Avance entre folds. Si None, igual a test_size (no hay solapamiento de tests).
    expanding : bool
        Si True, train crece (anchored). Si False, ventana rolling de tamaño fijo.
    embargo : int
        Número de muestras a excluir entre train y test (para evitar leakage por
        targets que miran al futuro).

    Yields
    ------
    (train_idx, test_idx) : tuplas de arrays con índices posicionales
    """
    train_size: int
    test_size: int
    step: Optional[int] = None
    expanding: bool = False
    embargo: int = 0

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        step = self.step if self.step is not None else self.test_size

        train_start = 0
        train_end = self.train_size
        test_start = train_end + self.embargo
        test_end = test_start + self.test_size

        while test_end <= n:
            if self.expanding:
                train_idx = np.arange(0, train_end)
            else:
                train_idx = np.arange(train_start, train_end)

            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx

            # Avanzar
            train_start += step
            train_end += step
            test_start = train_end + self.embargo
            test_end = test_start + self.test_size

    def get_n_splits(self, X) -> int:
        n = len(X)
        step = self.step if self.step is not None else self.test_size
        usable = n - self.train_size - self.embargo - self.test_size
        if usable < 0:
            return 0
        return usable // step + 1


@dataclass
class PurgedKFold:
    """
    K-Fold purged con embargo (López de Prado, AFML cap. 7).

    Diseñado para targets construidos con triple-barrier o lookforward,
    donde hay solapamiento temporal entre muestras.

    Parameters
    ----------
    n_splits : int
        Número de folds.
    pct_embargo : float
        Fracción de muestras a embarcar entre train y test. Típicamente 0.01-0.02.
    """
    n_splits: int = 5
    pct_embargo: float = 0.01

    def split(
        self,
        X: pd.DataFrame,
        target_times: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Parameters
        ----------
        target_times : pd.Series
            Para cada índice de X (series.index), el TIMESTAMP en el que el target
            está completamente determinado. Por ejemplo, si entrenas con horizonte 20,
            target_times[i] = X.index[i + 20].
            Esto permite hacer purge correctamente.
        """
        n = len(X)
        if target_times is None:
            # Sin info de target_times, asumimos que cada fila es independiente.
            target_times = pd.Series(X.index, index=X.index)

        embargo_size = int(n * self.pct_embargo)
        indices = np.arange(n)
        fold_size = n // self.n_splits

        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < self.n_splits - 1 else n
            test_idx = indices[test_start:test_end]

            test_time_start = X.index[test_idx[0]]
            test_time_end = X.index[test_idx[-1]]

            # Purge: eliminar muestras de train cuyo target se solape con el período de test
            # Una muestra i está "contaminada" si su target_times[i] cae dentro del período de test
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False

            for i in indices:
                if i in test_idx:
                    continue
                t_start = X.index[i]
                t_end = target_times.iloc[i]
                # Solapamiento con período de test
                if t_end >= test_time_start and t_start <= test_time_end:
                    train_mask[i] = False

            # Embargo: bloquear muestras justo después del test
            embargo_end = min(test_end + embargo_size, n)
            train_mask[test_end:embargo_end] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx
