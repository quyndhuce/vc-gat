"""
Decision-Certified Bounds and Stopping Criteria (Proposition 2, 2a, Corollary 1).
"""

import math
from typing import List, Tuple, Optional


def compute_monotone_bounds(evaluated_degrees: List[float], max_residual_degree: float = 1.0) -> Tuple[float, float]:
    """
    Computes monotone bounds [L_k, U_k] after evaluating k derivations.
    Under noisy-OR t-conorm:
      L_k = 1 - prod_{i=1}^k (1 - d_i)
      U_k = 1 - (1 - L_k) * (1 - max_residual_degree)
    """
    rem = 1.0
    for d in evaluated_degrees:
        rem *= max(0.0, 1.0 - float(d))
    L_k = 1.0 - rem
    U_k = min(1.0, 1.0 - rem * max(0.0, 1.0 - float(max_residual_degree)))
    return L_k, U_k


def check_certified_stopping(L_k: float, U_k: float, tau_d: float = 0.50) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether the interval [L_k, U_k] certifies the downstream decision.
    Returns:
      (True, 'OMIT')   if U_k < tau_d  (certified omit)
      (True, 'ASSERT') if L_k >= tau_d (certified assert)
      (False, 'UNDECIDED') otherwise   (must continue derivation)
    """
    if U_k < tau_d:
        return True, "OMIT"
    elif L_k >= tau_d:
        return True, "ASSERT"
    else:
        return False, "UNDECIDED"
