"""Held-out temperature fitting for classifiers with named species logits."""

import math

import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Minimize validation cross entropy with one positive temperature.

    Use a separate calibration split, never the final test set. Fit on logits
    produced by the same model, crop policy, and preprocessing used at inference.
    This preserves class rankings and does not train an unknown-species detector.
    The search includes T=1 so fitting cannot worsen calibration-set log loss.
    """
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must be a nonempty [samples, classes] tensor with at least two classes")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must contain one class index per sample")
    if labels.is_floating_point() or labels.dtype == torch.bool:
        raise ValueError("labels must contain integer class indices")
    if not torch.isfinite(logits).all() or labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError("Logits must be finite and labels must be in the class range")

    values = logits.detach().to(device="cpu", dtype=torch.float64)
    targets = labels.detach().to(device="cpu", dtype=torch.int64)

    def objective(log_temperature: float) -> float:
        return float(F.cross_entropy(values / math.exp(log_temperature), targets).item())

    result = minimize_scalar(objective, bounds=(math.log(0.01), math.log(100.0)), method="bounded")
    if not result.success or not math.isfinite(result.fun):
        raise RuntimeError("Temperature optimization did not converge")
    if result.fun >= objective(0.0):
        return 1.0

    return math.exp(result.x)
