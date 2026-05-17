"""
conftest.py — pytest configuration for execution-engine service.

Adds both the service root (so `app` is importable) and the monorepo's
shared library (so `quant_shared` is importable without a pip install).
"""
import os
import sys

_SVC_ROOT    = os.path.dirname(__file__)
_SHARED_ROOT = os.path.normpath(
    os.path.join(_SVC_ROOT, "..", "..", "..", "shared")
)

for _p in (_SVC_ROOT, _SHARED_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
