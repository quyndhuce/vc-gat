"""
Decision-Certified Fuzzy Horn Reasoning Solver (3-Stage Architecture).
Proposition 2, Proposition 2a, Corollary 1.
"""

from typing import List, Dict, Any, Tuple, Optional
from .bounds import compute_monotone_bounds, check_certified_stopping


class DecisionCertifiedHornSolver:
    def __init__(self, tau_d: float = 0.50):
        self.tau_d = tau_d

    def solve_fact(
        self,
        base_degree: float,
        derivations: List[float],
        is_c2_compliant: bool = True
    ) -> Dict[str, Any]:
        """
        Solves query fact under 3-stage Decision-Certified Horn architecture:
          Stage 1: Analytical exact (if 0 or 1 derivation)
          Stage 2: Decision-Certified BnB (if C2 compliant)
          Stage 3: Exact Fallback (if non-compliant diamond DAG)
        """
        if len(derivations) == 0:
            mu_f = float(base_degree)
            decision = "ASSERT" if mu_f >= self.tau_d else "OMIT"
            return {
                "stage": 1,
                "mu_f": mu_f,
                "decision": decision,
                "derivations_evaluated": 0,
                "certified_stopped": True
            }

        # Stage 3 Fallback for non-compliant DAGs
        if not is_c2_compliant:
            rem = 1.0 - base_degree
            for d in derivations:
                rem *= (1.0 - d)
            mu_f = 1.0 - rem
            decision = "ASSERT" if mu_f >= self.tau_d else "OMIT"
            return {
                "stage": 3,
                "mu_f": mu_f,
                "decision": decision,
                "derivations_evaluated": len(derivations),
                "certified_stopped": False
            }

        # Stage 2: Decision-Certified BnB
        sorted_derivs = sorted(derivations, reverse=True)
        eval_list = [base_degree]
        stopped = False
        decision = "UNDECIDED"
        eval_count = 0

        for idx, d in enumerate(sorted_derivs):
            eval_list.append(d)
            eval_count += 1
            max_res = sorted_derivs[idx + 1] if idx + 1 < len(sorted_derivs) else 0.0
            L_k, U_k = compute_monotone_bounds(eval_list, max_residual_degree=max_res)
            is_cert, dec = check_certified_stopping(L_k, U_k, self.tau_d)
            if is_cert:
                stopped = True
                decision = dec
                break

        if not stopped:
            L_k, _ = compute_monotone_bounds(eval_list, max_residual_degree=0.0)
            decision = "ASSERT" if L_k >= self.tau_d else "OMIT"

        rem = 1.0
        for d in eval_list:
            rem *= (1.0 - d)
        mu_f = 1.0 - rem

        return {
            "stage": 2,
            "mu_f": mu_f,
            "decision": decision,
            "derivations_evaluated": eval_count,
            "certified_stopped": stopped
        }
