"""
Serialización / deserialización de modelos ML entrenados.

Soporta:
  - joblib (sklearn, XGBoost)
  - torch.save (DeepMLP, LSTM)
  - onnx (futuro — inferencia multi-runtime)

Convención de paths:
  artifacts/<name>/<version>/model.<ext>
  artifacts/<name>/<version>/metadata.json
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_ARTIFACTS_ROOT = Path(os.getenv(
    "MODEL_ARTIFACTS_PATH",
    str(Path(__file__).parent.parent.parent.parent / "artifacts" / "models")
))


@dataclass
class ModelArtifact:
    """Contenedor de modelo cargado + metadata."""
    name:     str
    version:  str
    model:    Any          # el objeto modelo (sklearn, XGBoost, nn.Module)
    metadata: dict


def _artifact_dir(name: str, version: str) -> Path:
    return _ARTIFACTS_ROOT / name / version


def save_model(
    model: Any,
    name: str,
    version: str,
    metadata: Optional[dict] = None,
    backend: str = "auto",
) -> Path:
    """
    Serializa un modelo al artifact store.

    Args:
        model:    Objeto modelo (XGBoost, sklearn, torch.nn.Module)
        name:     Nombre del modelo (e.g. "xgboost_swing")
        version:  Version semántica (e.g. "1.2.3")
        metadata: Dict con metadatos adicionales (features, métricas, etc.)
        backend:  "joblib" | "torch" | "auto"

    Returns:
        Path al archivo serializado.
    """
    artifact_dir = _artifact_dir(name, version)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Detectar backend
    if backend == "auto":
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                backend = "torch"
            else:
                backend = "joblib"
        except ImportError:
            backend = "joblib"

    if backend == "torch":
        import torch
        model_path = artifact_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
    else:
        import joblib
        model_path = artifact_dir / "model.joblib"
        joblib.dump(model, model_path)

    # Guardar metadata
    meta = metadata or {}
    meta.update({"name": name, "version": version, "backend": backend})
    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    return model_path


def load_model(
    name: str,
    version: str,
    model_class: Optional[Any] = None,
) -> ModelArtifact:
    """
    Carga un modelo desde el artifact store.

    Args:
        name:        Nombre del modelo.
        version:     Versión a cargar.
        model_class: Clase del modelo (requerido para backend=torch).

    Returns:
        ModelArtifact con el modelo y su metadata.
    """
    artifact_dir = _artifact_dir(name, version)
    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"No se encontró el artifact {name}/{version} en {artifact_dir}"
        )

    # Leer metadata
    meta_path = artifact_dir / "metadata.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    backend = metadata.get("backend", "joblib")

    if backend == "torch":
        import torch
        if model_class is None:
            raise ValueError("model_class es requerido para cargar modelos PyTorch")
        model = model_class()
        model.load_state_dict(torch.load(artifact_dir / "model.pt", weights_only=True))
        model.eval()
    else:
        import joblib
        model = joblib.load(artifact_dir / "model.joblib")

    return ModelArtifact(
        name=name,
        version=version,
        model=model,
        metadata=metadata,
    )


def list_versions(name: str) -> list[str]:
    """Lista todas las versiones disponibles de un modelo."""
    model_dir = _ARTIFACTS_ROOT / name
    if not model_dir.exists():
        return []
    return sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
