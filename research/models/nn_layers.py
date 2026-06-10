"""
Neural network building blocks shared across zoo.py and DRL agents (ADR-034, ADR-038).

Modules
-------
ResBlock          : Residual block — LayerNorm → SwiGLU gate → Linear → Dropout → skip.
TemperatureScaling: Post-hoc calibration via a single scalar T (Guo et al. 2017).
                    Divides logits by T before softmax to reduce ECE without touching accuracy.

Design decisions
----------------
- LayerNorm (not BatchNorm): stable with tabular small batches; no batch-size dependency.
- SwiGLU gate: SwiGLU(x) = sigmoid(xW1) ⊙ (xW2). Empirically better than ReLU/GELU on
  tabular data with low signal-to-noise (Shazeer 2020).
- Orthogonal init (via caller): stabilises RL gradient flow (Schulman et al.).
- Temperature found by minimising NLL on a held-out val set with LBFGS.

References
----------
Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML.
Shazeer, N. (2020). GLU Variants Improve Transformer.
He, K. et al. (2015). Deep Residual Learning for Image Recognition.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ============================================================================
# RESBLOCK — shared by zoo.ResMLPClassifier and drl.TradingResMLP
# ============================================================================


class ResBlock(nn.Module):
    """
    Residual block with pre-activation LayerNorm and SwiGLU gate.

    Architecture: y = x + Dropout(Linear(SwiGLU(gate(LayerNorm(x)))))

    Parameters
    ----------
    dim : int
        Input/output dimension (skip connection requires equal dims).
    dropout : float
        Dropout probability applied after the projection.

    Examples
    --------
    >>> block = ResBlock(dim=128, dropout=0.1)
    >>> x = torch.randn(16, 128)
    >>> block(x).shape
    torch.Size([16, 128])
    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Projects dim → 2*dim; chunked into gate + linear halves (SwiGLU)
        self.gate = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gate_half, linear_half = self.gate(h).chunk(2, dim=-1)
        h = self.proj(torch.sigmoid(gate_half) * linear_half)
        return x + self.drop(h)


# ============================================================================
# TEMPERATURE SCALING — post-hoc calibration for NNs
# ============================================================================


class TemperatureScaling(nn.Module):
    """
    Post-hoc probability calibration via a single learned temperature T.

    Replaces raw logits z with z / T before softmax.
    T > 1 → softer distribution (less overconfident).
    T < 1 → harder distribution (rarely needed in practice).

    Usage:
        ts = TemperatureScaling()
        ts.fit(logits_val, labels_val)   # fit on calibration set
        calibrated_proba = ts.calibrate_proba(raw_proba)  # at inference

    Parameters
    ----------
    init_temp : float
        Initial temperature value (1.5 is a conservative warm start).

    References
    ----------
    Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.

    Examples
    --------
    >>> ts = TemperatureScaling()
    >>> logits = torch.randn(100, 3)
    >>> labels = torch.randint(0, 3, (100,))
    >>> ts.fit(logits, labels)
    >>> proba = torch.softmax(logits / ts.temperature, dim=1).detach().numpy()
    """

    def __init__(self, init_temp: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temp]))
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by T (call before softmax)."""
        return logits / self.temperature.clamp(min=0.05)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> "TemperatureScaling":
        """
        Find T that minimises NLL on the calibration set.

        Parameters
        ----------
        logits : (n, n_classes) raw logits from the model (pre-softmax).
        labels : (n,) integer class indices.
        max_iter : LBFGS max iterations.
        lr : learning rate for LBFGS.
        """
        self.temperature.requires_grad_(True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        # LBFGS can overshoot into the clamped (zero-gradient) region and leave
        # the raw parameter negative, while forward() still clamps to min=0.05.
        # Persist the effective (clamped) temperature so the stored value is
        # consistent with inference and stays strictly positive.
        with torch.no_grad():
            self.temperature.clamp_(min=0.05)
        self.temperature.requires_grad_(False)
        self._fitted = True
        logger.info(
            "TemperatureScaling fitted: T=%.4f", self.temperature.item()
        )
        return self

    def calibrate_proba(self, raw_proba: "torch.Tensor | None" = None) -> None:
        """
        Not used directly — calibration is applied inside ResMLPClassifier
        by scaling logits before softmax. Kept for interface clarity.
        """
        raise NotImplementedError(
            "Call ResMLPClassifier.predict_proba() which applies T internally."
        )
