"""
Registry adapter for multi-horizon models.

Wraps shared.quant_shared.models.registry.ModelRegistry with the
multi_horizon_v1 tag convention and horizon-specific defaults.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from quant_shared.models.registry import ModelCard, ModelRegistry

_ARTIFACTS_DIR = Path(__file__).parents[4] / "artifacts" / "multi_horizon"


@dataclass
class HorizonArtifact:
    """Serialized model + full metadata for one horizon."""

    horizon_name: str
    model_object: Any
    card: ModelCard
    artifact_path: Path


def register_horizon_model(
    registry: ModelRegistry,
    horizon_name: str,
    model_object: Any,
    version: str,
    psr: float,
    dsr: float,
    ece: float,
    sharpe_oos: float,
    win_rate_oos: float,
    train_start: date,
    train_end: date,
    n_folds: int,
    symbols: list[str],
    notes: str = "",
) -> HorizonArtifact:
    """
    Persist model artifact and register ModelCard.

    Parameters
    ----------
    registry : ModelRegistry
        Shared registry instance.
    horizon_name : "intraday" | "swing" | "daily"
    model_object : Any
        Trained model (XGBoostClassifier or DeepMLPClassifier).
    version : str
        Semantic version string, e.g. "0.1".
    psr, dsr, ece, sharpe_oos, win_rate_oos : float
        OOS metrics.
    train_start, train_end : date
    n_folds : int
    symbols : list[str]
    notes : str

    Returns
    -------
    HorizonArtifact with artifact_path populated.
    """
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_id = f"multi_horizon_{horizon_name}_v{version}"
    artifact_path = _ARTIFACTS_DIR / f"{model_id}.pkl"

    # Serialize
    with open(artifact_path, "wb") as f:
        pickle.dump(model_object, f, protocol=5)

    # Compute content hash for reproducibility checks
    content_hash = _file_sha256(artifact_path)

    card = ModelCard(
        model_id=model_id,
        name=f"multi_horizon_{horizon_name}",
        version=version,
        model_class=_infer_model_class(model_object),
        artifact_path=str(artifact_path),
        feature_set_version="1.0.0",
        psr=round(psr, 6),
        dsr=round(dsr, 6),
        ece=round(ece, 6),
        sharpe_oos=round(sharpe_oos, 6),
        win_rate_oos=round(win_rate_oos, 6),
        train_start=train_start.isoformat(),
        train_end=train_end.isoformat(),
        n_folds=n_folds,
        symbols=symbols[:50],
        strategy=f"multi_horizon_{horizon_name}",
        notes=f"tag=multi_horizon_v1 hash={content_hash[:12]} {notes}",
    )

    if dsr >= 0.4:
        card.status = "staging"
    else:
        card.status = "deprecated"
        card.notes = f"no_edge: DSR={dsr:.4f} < 0.4 " + card.notes

    registry.register(card)
    return HorizonArtifact(
        horizon_name=horizon_name,
        model_object=model_object,
        card=card,
        artifact_path=artifact_path,
    )


def _infer_model_class(obj: Any) -> str:
    cls_name = type(obj).__name__.lower()
    if "xgboost" in cls_name or "xgb" in cls_name:
        return "xgboost"
    if "mlp" in cls_name or "deep" in cls_name:
        return "deep_mlp"
    return cls_name


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
