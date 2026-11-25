"""
Split Conformal Risk Control for Evidence Admission (Proposition 5 & Proposition 6).
Guarantees marginal over-assertion risk E[ell(Z_{n+1}, theta_hat)] <= alpha.
"""

import numpy as np
from typing import Tuple, Dict, Any


def calibrate_conformal_threshold(
    confidences: np.ndarray,
    ground_truth_labels: np.ndarray,
    alpha: float = 0.05
) -> float:
    """
    Computes calibrated threshold theta_hat under Proposition 6:
      inf { theta in [0, 1] : (n / (n+1)) * R_cal(theta) + 1 / (n+1) <= alpha }
    """
    n = len(confidences)
    if n == 0:
        return 1.0

    candidates = np.array(sorted(set(confidences) | {0.0, 1.0}))
    err_counts = np.sum((confidences[:, None] > candidates[None, :]) & (ground_truth_labels[:, None] == 0), axis=0)
    R_cal = err_counts / float(n)
    adjusted_bound = (n / (n + 1.0)) * R_cal + (1.0 / (n + 1.0))

    valid_indices = np.where(adjusted_bound <= alpha)[0]
    if len(valid_indices) > 0:
        return float(candidates[valid_indices[0]])
    return 1.0


def evaluate_conformal_risk(
    confidences: np.ndarray,
    ground_truth_labels: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """Evaluates empirical risk and assertion coverage at given threshold."""
    asserted = confidences > threshold
    n = len(confidences)
    if n == 0:
        return {"empirical_risk": 0.0, "coverage": 0.0}

    over_assertions = np.sum(asserted & (ground_truth_labels == 0))
    empirical_risk = float(over_assertions / n)
    coverage = float(np.sum(asserted) / n)

    return {
        "empirical_risk": empirical_risk,
        "coverage": coverage
    }
