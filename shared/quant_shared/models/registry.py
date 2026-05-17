"""
Model Registry — catálogo de modelos entrenados.

research/ registra modelos tras el walk-forward.
platform/ml-feature-store los carga para inference en tiempo real.

Implementación local (archivo JSON). En producción se reemplaza
por MLflow o un registry en Postgres.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_REGISTRY_PATH = Path(os.getenv(
    "MODEL_REGISTRY_PATH",
    str(Path(__file__).parent.parent.parent.parent / "artifacts" / "registry.json")
))


@dataclass
class ModelCard:
    """Metadata de un modelo registrado."""
    model_id:           str
    name:               str             # "xgboost_swing_v1"
    version:            str             # "1.2.3"
    model_class:        str             # "xgboost" | "deep_mlp" | "lstm"
    artifact_path:      str             # path al archivo serializado
    feature_set_version: str = "1.0.0"

    # Métricas OOS (walk-forward)
    psr:                float = 0.0     # Probabilistic Sharpe Ratio
    dsr:                float = 0.0     # Deflated Sharpe Ratio
    ece:                float = 1.0     # Expected Calibration Error (lower = better)
    sharpe_oos:         float = 0.0
    win_rate_oos:       float = 0.0

    # Contexto de entrenamiento
    train_start:        str = ""
    train_end:          str = ""
    n_folds:            int = 0
    symbols:            list[str] = field(default_factory=list)
    strategy:           str = ""

    # Estado
    status:             str = "staging"  # "staging" | "canary" | "production" | "deprecated"
    promoted_at:        Optional[str] = None
    deprecated_at:      Optional[str] = None
    notes:              str = ""

    registered_at:      str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def is_production_ready(self) -> bool:
        """Verifica que las métricas superan umbrales mínimos de producción."""
        return (
            self.psr > 0.5
            and self.dsr > 0.4
            and self.ece < 0.05
            and self.sharpe_oos > 0.8
        )

    def promote_to(self, status: str) -> None:
        self.status = status
        self.promoted_at = datetime.now(tz=timezone.utc).isoformat()


class ModelRegistry:
    """
    Registry local de modelos.

    Uso en research/:
        registry = ModelRegistry()
        card = ModelCard(model_id="...", name="xgboost_swing", ...)
        registry.register(card)
        registry.promote("model-id", "canary")

    Uso en platform/:
        registry = ModelRegistry()
        card = registry.get_production("xgboost_swing")
        model = load_model(card.artifact_path)
    """

    def __init__(self, path: Path = _REGISTRY_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cards: dict[str, ModelCard] = {}
        self._load()

    def register(self, card: ModelCard) -> None:
        """Registra o actualiza un modelo."""
        self._cards[card.model_id] = card
        self._save()

    def get(self, model_id: str) -> Optional[ModelCard]:
        return self._cards.get(model_id)

    def get_production(self, name: str) -> Optional[ModelCard]:
        """Devuelve el modelo en producción para un nombre dado."""
        candidates = [
            c for c in self._cards.values()
            if c.name == name and c.status == "production"
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.registered_at)

    def get_by_status(self, status: str) -> list[ModelCard]:
        return [c for c in self._cards.values() if c.status == status]

    def promote(self, model_id: str, status: str) -> None:
        """Promueve un modelo a un nuevo estado."""
        if model_id not in self._cards:
            raise KeyError(f"Model {model_id} not found in registry")
        self._cards[model_id].promote_to(status)
        self._save()

    def list_all(self) -> list[ModelCard]:
        return list(self._cards.values())

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._cards = {k: ModelCard(**v) for k, v in data.items()}

    def _save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(
                {k: asdict(v) for k, v in self._cards.items()},
                f, indent=2, default=str,
            )


_default_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Singleton del registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry()
    return _default_registry
