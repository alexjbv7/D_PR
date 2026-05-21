"""
Root conftest.py — Adds each service's parent directory to sys.path
so that `from app.module import X` works when running pytest from repo root.

Usage:
    cd los_ojos
    PYTHONPATH=. pytest services/strategy-orchestrator/tests -v

Or via Makefile:
    make test
"""
import sys
import os

# Add the repo root so `libs/shared` is importable
_ROOT = os.path.dirname(__file__)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# When pytest discovers tests inside services/<svc>/tests/, make `app`
# importable by adding services/<svc>/ to sys.path dynamically.
# This is done by pytest-pythonpath or manually here via conftest hooks.

def pytest_configure(config):
    """Add each service directory to sys.path for test discovery."""
    services_dir = os.path.join(_ROOT, "services")
    if not os.path.isdir(services_dir):
        return
    for svc in os.listdir(services_dir):
        svc_path = os.path.join(services_dir, svc)
        if os.path.isdir(svc_path) and svc_path not in sys.path:
            sys.path.insert(0, svc_path)
