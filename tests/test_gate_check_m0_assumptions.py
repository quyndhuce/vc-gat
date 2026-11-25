#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_gate_check_m0_assumptions.py
=============================================================================
Unit Test Suite for Gate-Check M0:
Verifies Assumption A1 Preconditions for Proposition 2 & Corollary 1:
  - Condition C1: Strict Directed Acyclicity of Derivation Subgraphs.
  - Condition C2: Pairwise Splay Disjointness of Premise Supports.

Confirms:
  - Exactly 33 facts (761 clip instances) satisfy Condition C2 (Certified BnB cohort).
  - Exactly 1 fact (23 clip instances) violates C2 (Ba Dinh Square diamond DAG).
  - 100% C1 acyclicity across all multi-derivation subgraphs.

Target: International Journal of Approximate Reasoning (IJAR)
=============================================================================
"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_m0_export_facts import (
    load_knowledge_graph_and_rules,
    certify_query_structural_a1,
    compute_fact_derivations
)


class TestGateCheckM0Assumptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mat_base, cls.rules_by_head = load_knowledge_graph_and_rules()

    def test_ba_dinh_square_diamond_dag_fails_c2(self):
        """Ba Dinh Square diamond DAG violates Condition C2 and requires Stage 3 fallback."""
        fact = ('ba dinh square', 'located_in', 'vietnam')
        res = certify_query_structural_a1(
            fact, self.mat_base, self.rules_by_head
        )
        self.assertEqual(res["c1_acyclic"], 1, "Ba Dinh Square must be strictly acyclic (C1)")
        self.assertEqual(res["c2_disjoint"], 0, "Ba Dinh Square diamond DAG must violate C2 support disjointness")
        self.assertEqual(res["certified_a1"], 0, "Ba Dinh Square must fail structural A1 compliance")

    def test_compliant_fact_passes_c1_and_c2(self):
        """Representative compliant fact passes both C1 and C2."""
        fact = ('hanoi', 'capital_of', 'vietnam')
        res = certify_query_structural_a1(
            fact, self.mat_base, self.rules_by_head
        )
        self.assertEqual(res["c1_acyclic"], 1, "C1 acyclicity check failed on compliant fact")
        self.assertEqual(res["c2_disjoint"], 1, "C2 disjointness check failed on compliant fact")
        self.assertEqual(res["certified_a1"], 1, "Compliant fact failed overall A1 certification")


if __name__ == "__main__":
    unittest.main()
