"""
Landmark Gating & Calibration Module (Section 4.1 & Section 6.4, Table 6).
Evaluates landmark probability mu_L(L_k | V) with temperature scaling TL=0.5.
"""

import numpy as np
from typing import Dict, List, Tuple


def softmax_with_temperature(logits: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """Scales logits by temperature TL and returns calibrated softmax distribution."""
    scaled = logits / max(1e-4, temperature)
    scaled -= np.max(scaled, axis=-1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)


def compute_expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE):
      ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)
