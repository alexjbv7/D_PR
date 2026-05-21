"""
Drift detection — Population Stability Index, Expected Calibration Error,
KS test, and bucket-edge persistence.

Sub-modules
-----------
psi         — compute_psi, severity_label, compute_psi_categorical
ece         — compute_ece, compute_brier
ks_test     — compute_ks, ks_severity
bucketizer  — BucketEdgesRepository (Redis, TTL 30 days)

Anti-leakage invariant (ADR-030):
  Bucket edges for PSI MUST be derived from training data only and
  persisted in Redis before computing PSI on recent data.
  Recomputing edges from recent data is a tautology that suppresses
  all PSI signal (PSI ≈ 0 always).
"""
from .psi import compute_psi, severity_label, compute_psi_categorical
from .ece import compute_ece, compute_brier
from .ks_test import compute_ks, ks_severity

__all__ = [
    "compute_psi",
    "severity_label",
    "compute_psi_categorical",
    "compute_ece",
    "compute_brier",
    "compute_ks",
    "ks_severity",
]
