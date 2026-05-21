"""
List registered models filtered by tag.

Usage:
    python -m research.tools.list_models --tag multi_horizon_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).parents[3]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quant_shared.models.registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="List registered models")
    parser.add_argument("--tag", type=str, default="multi_horizon_v1")
    args = parser.parse_args()

    registry = ModelRegistry()
    all_cards = registry.list_all()

    matching = [
        c for c in all_cards
        if args.tag in c.notes or args.tag in c.name or args.tag in c.strategy
    ]

    if not matching:
        print(f"No models found with tag '{args.tag}'.")
        return

    print(f"\nModels tagged '{args.tag}':\n")
    for card in matching:
        status_icon = "✅" if card.status in ("staging", "production") else "🗄"
        print(
            f"  {status_icon} {card.model_id:<40} "
            f"DSR={card.dsr:.4f}  PSR={card.psr:.4f}  ECE={card.ece:.4f}  "
            f"status={card.status}"
        )
    print(f"\nTotal: {len(matching)} model(s)")


if __name__ == "__main__":
    main()
