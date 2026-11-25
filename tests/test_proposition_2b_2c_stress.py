#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_proposition_2b_2c_stress.py
=============================================================================
Unit Test Suite for Proposition 2b (Finite Perturbation Deviation Bound)
and Proposition 2c (Degree-Debiased Attention Mass Bound & Sublinear Hub Scaling).

Target: International Journal of Approximate Reasoning (IJAR)
=============================================================================
"""

import unittest
import math
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.reproduce_proposition_2b_2c_stress import (
    run_proposition_2b_stress_tests,
    run_proposition_2c_stress_tests
)


class TestProposition2b2cStress(unittest.TestCase):
    def test_prop2b_stress_tests(self):
        """Verifies Proposition 2b: finite deviation bounds under support perturbation."""
        results = run_proposition_2b_stress_tests()
        self.assertEqual(results.get("status"), "PASSED")
        self.assertEqual(results.get("vcgat_hallucinated_facts"), 0)
        self.assertGreater(results.get("naive_hallucinated_facts"), 0)

    def test_prop2c_stress_tests(self):
        """Verifies Proposition 2c: sublinear hub scaling and slope dampening."""
        results = run_proposition_2c_stress_tests()
        self.assertEqual(results.get("status"), "PASSED")
        self.assertLess(results.get("debiased_gat_slope"), results.get("standard_gat_slope"))
        self.assertTrue(all(p > 30.0 for p in results.get("top5_mass_reductions_pct", [])))


if __name__ == "__main__":
    unittest.main()
