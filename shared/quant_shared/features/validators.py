"""
Validación de feature vectors — detecta NaN, inf, y valores fuera de rango.

Usable tanto en research/ (antes de entrenar) como en platform/
(antes de hacer inference en tiempo real).
"""
from __future__ import annotations

import numpy as np
from .definitions import FEATURES, FEATURE_NAMES


class FeatureValidationError(ValueError):
    """Feature vector inválido."""
    pass


def validate_feature_vector(
    arr: np.ndarray,
    strict: bool = False,
    symbol: str = "",
) -> list[str]:
    """
    Valida un vector numpy de 19 features.

    Args:
        arr:    Array de shape (19,) o (N, 19)
        strict: Si True, lanza FeatureValidationError en vez de devolver warnings
        symbol: Para mensajes de error más descriptivos

    Returns:
        Lista de strings con warnings encontrados (vacía = OK)

    Raises:
        FeatureValidationError: si strict=True y hay problemas
    """
    prefix = f"[{symbol}] " if symbol else ""
    warnings: list[str] = []

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    n_features = arr.shape[-1]
    if n_features != 19:
        msg = f"{prefix}Se esperan 19 features, se recibieron {n_features}"
        if strict:
            raise FeatureValidationError(msg)
        return [msg]

    # NaN / Inf
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    if nan_mask.any():
        cols = np.where(nan_mask.any(axis=0))[0]
        names = [FEATURE_NAMES[c] for c in cols]
        warnings.append(f"{prefix}NaN en features: {names}")

    if inf_mask.any():
        cols = np.where(inf_mask.any(axis=0))[0]
        names = [FEATURE_NAMES[c] for c in cols]
        warnings.append(f"{prefix}Inf en features: {names}")

    # Rangos esperados
    for i, fdef in enumerate(FEATURES):
        col = arr[:, i]
        valid = col[~np.isnan(col) & ~np.isinf(col)]
        if len(valid) == 0:
            continue
        if valid.min() < fdef.min_val - 1e-6 or valid.max() > fdef.max_val + 1e-6:
            warnings.append(
                f"{prefix}{fdef.name} fuera de rango [{fdef.min_val}, {fdef.max_val}]: "
                f"min={valid.min():.4f}, max={valid.max():.4f}"
            )

    if strict and warnings:
        raise FeatureValidationError("\n".join(warnings))

    return warnings
