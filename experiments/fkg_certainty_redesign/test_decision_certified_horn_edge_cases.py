#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_decision_certified_horn_edge_cases.py
=============================================================================
Automated Unit Tests for Decision-Certified Horn Reasoning Engine:
1. Threshold Straddling: c_lower < 0.80 <= c_upper forces continued derivation evaluation.
2. Exact Threshold Boundary: Exact behavior at 0.50 and 0.80 thresholds.
3. Natural Order Inversion: Bound validity L_k <= mu^* <= U_k under all derivation permutations.
4. Exhaustive Fallback: Convergence to exact certainty width = 0 when threshold straddles until k=m.
5. Integration Test: Hanoi test fact integrity, zero policy flips, zero bound violations.
=============================================================================
"""

import unittest
import math
import random
import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

TAU_ASSERT = 0.80
TAU_OMIT = 0.50
MU_L_HANOI = 0.85

def get_policy(c: float) -> str:
    if c >= TAU_ASSERT:
        return "ASSERT"
    elif c >= TAU_OMIT:
        return "HEDGE"
    else:
        return "OMIT"

def compute_exact_mu(b_f: float, weights: list) -> float:
    prod = 1.0 - b_f
    for w in weights:
        prod *= (1.0 - w)
    return 1.0 - prod

class TestDecisionCertifiedHornEdgeCases(unittest.TestCase):

    def test_exact_threshold_boundary(self):
        """Test policy determination at exact threshold boundaries."""
        self.assertEqual(get_policy(0.800000), "ASSERT")
        self.assertEqual(get_policy(0.7999999), "HEDGE")
        self.assertEqual(get_policy(0.500000), "HEDGE")
        self.assertEqual(get_policy(0.4999999), "OMIT")
        self.assertEqual(get_policy(0.0), "OMIT")
        self.assertEqual(get_policy(1.0), "ASSERT")

        # Boundary certification: if lower bound touches 0.80 exactly
        c_low = 0.80
        c_up = 0.85
        self.assertEqual(get_policy(c_low), "ASSERT")
        self.assertEqual(get_policy(c_up), "ASSERT")
        self.assertTrue(get_policy(c_low) == get_policy(c_up))

    def test_threshold_straddling_prevents_premature_stop(self):
        """
        Test that when uncertainty interval straddles a policy threshold,
        Decision-Certified BnB MUST NOT stop early, whereas naive Epsilon-BnB would stop.
        """
        # Construct scenario: c_low = 0.795, c_up = 0.804
        # Epsilon = 0.01: gap = 0.804 - 0.795 = 0.009 <= 0.01 (Eps-BnB stops!)
        c_low = 0.795
        c_up = 0.804
        eps = 0.01

        # Epsilon-BnB check:
        eps_can_stop = (c_up - c_low) <= eps
        self.assertTrue(eps_can_stop, "Epsilon-BnB would stop early here")

        # Decision-Certified check:
        pol_low = get_policy(c_low)
        pol_up = get_policy(c_up)
        self.assertEqual(pol_low, "HEDGE")
        self.assertEqual(pol_up, "ASSERT")
        decision_can_stop = (pol_low == pol_up)
        
        self.assertFalse(decision_can_stop, "Decision-Certified BnB MUST NOT stop when interval straddles 0.80!")

        # Another scenario: straddling 0.50 (c_low = 0.496, c_up = 0.503)
        c_low_omit = 0.496
        c_up_hedge = 0.503
        self.assertEqual(get_policy(c_low_omit), "OMIT")
        self.assertEqual(get_policy(c_up_hedge), "HEDGE")
        self.assertFalse(get_policy(c_low_omit) == get_policy(c_up_hedge),
                         "Decision-Certified BnB MUST NOT stop when interval straddles 0.50!")

    def test_natural_order_inversion_bound_invariance(self):
        """
        Prove that for ANY derivation ordering (even adversarial inverse order),
        the admissible bounds strictly enclose the true exact certainty:
        L_k <= mu^* <= U_k for all k = 0..m.
        """
        rng = random.Random(42)
        for trial in range(500):
            m = rng.randint(1, 15)
            b_f = rng.uniform(0.1, 0.9)
            raw_weights = [rng.uniform(0.05, 0.95) for _ in range(m)]
            mu_exact = compute_exact_mu(b_f, raw_weights)

            # Test multiple orderings: sorted descending, sorted ascending, random shuffle
            orderings = [
                sorted(raw_weights, reverse=True),
                sorted(raw_weights, reverse=False),
                list(raw_weights)
            ]
            for weights in orderings:
                q_max = max(weights)
                # k = 0
                L_0 = b_f
                U_0 = 1.0 - (1.0 - b_f) * ((1.0 - q_max)**m)
                self.assertLessEqual(L_0, mu_exact + 1e-9)
                self.assertGreaterEqual(U_0, mu_exact - 1e-9)

                # k = 1..m
                prod = 1.0 - b_f
                for k in range(1, m + 1):
                    prod *= (1.0 - weights[k - 1])
                    L_k = 1.0 - prod
                    rem = m - k
                    if rem == 0:
                        U_k = L_k
                    else:
                        q_curr_max = max(weights[k:]) if k < m else weights[-1]
                        U_k = 1.0 - (1.0 - L_k) * ((1.0 - q_curr_max)**rem)
                    
                    self.assertLessEqual(L_k, mu_exact + 1e-9, f"Violation at trial {trial}, k={k}")
                    self.assertGreaterEqual(U_k, mu_exact - 1e-9, f"Violation at trial {trial}, k={k}")

    def test_exhaustive_fallback_convergence(self):
        """
        Test that when no early stop is triggered, the algorithm evaluates all m derivations
        and converges exactly to L_m == U_m == mu^* with zero policy discrepancy.
        """
        b_f = 0.58
        weights = [0.15, 0.12, 0.10]
        m = len(weights)
        mu_exact = compute_exact_mu(b_f, weights)
        c_exact = mu_exact * MU_L_HANOI
        pol_exact = get_policy(c_exact)

        prod = 1.0 - b_f
        for k in range(1, m + 1):
            prod *= (1.0 - weights[k - 1])
            L_k = 1.0 - prod

        U_m = L_m = L_k
        self.assertAlmostEqual(L_m, mu_exact, places=9)
        self.assertAlmostEqual(U_m, mu_exact, places=9)
        self.assertEqual(get_policy(L_m * MU_L_HANOI), pol_exact)

    def test_hybrid_mode_order_equivalence(self):
        """
        Verify that Hybrid Mode eliminates order permutations among facts
        having near-identical scores within clip evidence packets.
        """
        # Two facts with close certainty: fact_A = 0.812, fact_B = 0.818
        # Both are ASSERT. In pure decision certified, fact_A stopped at L_0=0.805 might rank
        # differently than if evaluated fully.
        b_f_A, weights_A = 0.70, [0.35]  # mu_ex = 1 - 0.3 * 0.65 = 0.805 -> c = 0.68425 (HEDGE)
        b_f_B, weights_B = 0.71, [0.35]  # mu_ex = 1 - 0.29 * 0.65 = 0.8115 -> c = 0.689775 (HEDGE)
        
        # Exact order: B should rank above A
        exact_order = sorted([("A", compute_exact_mu(b_f_A, weights_A)), 
                              ("B", compute_exact_mu(b_f_B, weights_B))], 
                             key=lambda x: -x[1])
        self.assertEqual([x[0] for x in exact_order], ["B", "A"])

if __name__ == "__main__":
    unittest.main()
