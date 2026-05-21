import sys, os
# Ensure 'app' package is importable from this service's directory
_SVC = os.path.dirname(__file__)
if _SVC not in sys.path:
    sys.path.insert(0, _SVC)
