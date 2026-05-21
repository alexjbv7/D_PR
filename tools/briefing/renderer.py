"""Jinja2 rendering helpers for briefing templates."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).parent / "templates"


def render_template(template_name: str, **context: Any) -> str:
    """Render a template from ``tools/briefing/templates/``."""
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(enabled_extensions=()),
    )
    template = env.get_template(template_name)
    return template.render(**context)
