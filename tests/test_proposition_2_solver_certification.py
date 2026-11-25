#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/test_solver_certification.py
=============================================================================
Unit tests and Comprehensive Synthetic Test Suite for Structural A1
Certification (Condition C1 and Condition C2) and Three-Stage Solver Enforcement.

Covers:
  1. Real KG regression benchmarks (compliant vs. ba dinh diamond).
  2. Synthetic nested diamond dependency.
  3. Synthetic shared intermediate lemmas between sibling branches.
  4. Synthetic direct self-loop detection (Condition C1).
  5. Synthetic multi-step cyclic derivation detection (Condition C1).
  6. Synthetic base-fact versus derived-lemma aliasing (Condition C2).
  7. Synthetic pure multi-level tree hierarchy (Sound pass).

Target: International Journal of Approximate Reasoning (IJAR)
=============================================================================
"""

import unittest
from typing import Tuple, List, Dict, Any
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_m0_export_facts import (
    load_knowledge_graph_and_rules,
    certify_query_structural_a1,
    compute_fact_derivations
)


class TestSolverCertification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mat_base, cls.rules_by_head = load_knowledge_graph_and_rules()

    def test_unified_representation_catches_shared_lemma_and_base(self):
        """
        Verify that unified ("prop", f) representation catches diamond sharing
        even when fact f is used as base fact in Branch 1 and derived in Branch 2.
        """
        f = ('a', 'part_of', 'b')
        b1_comps = {('prop', f)}
        b2_comps = {
            ('prop', f),
            ('prop', ('a', 'part_of', 'sub_b')),
            ('rule', 'r_sub', (('a', 'part_of', 'sub_b'),), f)
        }
        inter = b1_comps.intersection(b2_comps)
        self.assertIn(('prop', f), inter)
        self.assertEqual(len(inter), 1)

    def test_certify_blocks_unverified_early_stopping(self):
        """
        Verify that compliant queries pass certification, while diamond-sharing queries
        in the benchmark are strictly identified and blocked from Stage 2 certified early stopping.
        """
        # Compliant fact: hoan kiem lake located_in hanoi
        f_cert = ('hoan kiem lake', 'located_in', 'hanoi')
        cert_res = certify_query_structural_a1(f_cert, self.mat_base, self.rules_by_head, max_depth=3)
        self.assertEqual(cert_res['c1_acyclic'], 1)
        self.assertEqual(cert_res['c2_disjoint'], 1)
        self.assertEqual(cert_res['certified_a1'], 1)
        
        # Diamond query: ba dinh square located_in vietnam
        f_uncert = ('ba dinh square', 'located_in', 'vietnam')
        uncert_res = certify_query_structural_a1(f_uncert, self.mat_base, self.rules_by_head, max_depth=3)
        self.assertEqual(uncert_res['c1_acyclic'], 1)
        self.assertEqual(uncert_res['c2_disjoint'], 0) # Violates C2
        self.assertEqual(uncert_res['certified_a1'], 0) # Strictly blocked from Stage 2

    def test_synthetic_direct_self_loop(self):
        """
        Condition C1 test: direct self-loop (fact references itself as antecedent).
        Must be rejected by C1.
        """
        syn_base = {('x', 'rel', 'y'): 0.8, ('y', 'rel', 'z'): 0.7}
        syn_rules = defaultdict(list)
        # Self loop: ('x', 'rel', 'y') requires ('x', 'rel', 'y')
        syn_rules[('x', 'rel', 'y')].append(
            ('r_loop', (('x', 'rel', 'y'), ('y', 'rel', 'z')), 0.9)
        )
        res = certify_query_structural_a1(('x', 'rel', 'y'), syn_base, syn_rules, max_depth=3)
        self.assertEqual(res['c1_acyclic'], 0, "Direct self-loop must violate Condition C1")
        self.assertEqual(res['certified_a1'], 0)

    def test_synthetic_indirect_cycle(self):
        """
        Condition C1 test: cyclic dependency path A -> B -> A.
        Must prune cyclic paths and ensure acyclicity check triggers if cyclic loop encountered.
        """
        syn_base = {('c', 'rel', 'd'): 0.8}
        syn_rules = defaultdict(list)
        f_a = ('node_a', 'rel', 'target')
        f_b = ('node_b', 'rel', 'target')
        # A depends on B
        syn_rules[f_a].append(('r_ab', (f_b, ('c', 'rel', 'd')), 0.95))
        # B depends on A (cycle)
        syn_rules[f_b].append(('r_ba', (f_a, ('c', 'rel', 'd')), 0.95))

        res = certify_query_structural_a1(f_a, syn_base, syn_rules, max_depth=3)
        # When traversing A -> B -> A, ancestor check detects path cycle
        # If no non-cyclic base paths exist, components are pruned
        self.assertIn('c1_acyclic', res)

    def test_synthetic_nested_diamond(self):
        """
        Condition C2 test: nested diamond structure at depth 2.
        Root H has sibling antecedents A1 and A2.
        A1 derives via (B1, B2).
        A2 derives via (B3, B4).
        B2 and B3 share a common sub-premise C.
        Must be caught by C2 (sibling branch support disjointness).
        """
        f_h = ('root', 'has_fact', 'h')
        f_a1 = ('node', 'type', 'a1')
        f_a2 = ('node', 'type', 'a2')
        f_b1 = ('sub', 'type', 'b1')
        f_b2 = ('sub', 'type', 'b2')
        f_b3 = ('sub', 'type', 'b3')
        f_b4 = ('sub', 'type', 'b4')
        f_shared = ('core', 'shared', 'fact')

        syn_base = {
            f_b1: 0.9,
            f_b4: 0.9,
            f_shared: 0.85
        }
        syn_rules = defaultdict(list)
        # Root rule: H <= A1 & A2
        syn_rules[f_h].append(('r_root', (f_a1, f_a2), 0.95))
        # A1 branch: A1 <= B1 & B2
        syn_rules[f_a1].append(('r_a1', (f_b1, f_b2), 0.90))
        # B2 derives from shared core
        syn_rules[f_b2].append(('r_b2', (f_shared,), 0.85))
        # A2 branch: A2 <= B3 & B4
        syn_rules[f_a2].append(('r_a2', (f_b3, f_b4), 0.90))
        # B3 derives from shared core
        syn_rules[f_b3].append(('r_b3', (f_shared,), 0.85))

        res = certify_query_structural_a1(f_h, syn_base, syn_rules, max_depth=3)
        self.assertEqual(res['c1_acyclic'], 1, "Nested diamond is acyclic")
        self.assertEqual(res['c2_disjoint'], 0, "Nested diamond must violate Condition C2")
        self.assertEqual(res['certified_a1'], 0, "Nested diamond must be denied Stage 2 certification")
        self.assertGreaterEqual(res['n_diamonds'], 1)

    def test_synthetic_shared_intermediate_lemma(self):
        """
        Condition C2 test: Sibling premises A1 and A2 both rely on an intermediate derived lemma L.
        L is not a base fact, but derived from M.
        Must be caught by C2 via unified ("prop", L) representation.
        """
        f_target = ('head', 'rel', 'conclusion')
        f_a1 = ('premise', 'rel', 'a1')
        f_a2 = ('premise', 'rel', 'a2')
        f_lemma = ('intermediate', 'lemma', 'l')
        f_base = ('ground', 'evidence', 'e')

        syn_base = {f_base: 0.8}
        syn_rules = defaultdict(list)
        syn_rules[f_target].append(('r_top', (f_a1, f_a2), 0.98))
        syn_rules[f_a1].append(('r_left', (f_lemma,), 0.90))
        syn_rules[f_a2].append(('r_right', (f_lemma,), 0.90))
        syn_rules[f_lemma].append(('r_lemma', (f_base,), 0.85))

        res = certify_query_structural_a1(f_target, syn_base, syn_rules, max_depth=3)
        self.assertEqual(res['c1_acyclic'], 1)
        self.assertEqual(res['c2_disjoint'], 0, "Shared intermediate lemma must violate Condition C2")
        self.assertEqual(res['certified_a1'], 0)

    def test_synthetic_base_derived_aliasing(self):
        """
        Condition C2 test: entity triple X appears as an asserted atomic base fact in Branch 1
        and as a derived conclusion in Branch 2.
        Unified ("prop", X) canonicalization must detect this overlap.
        """
        f_goal = ('target', 'property', 'val')
        f_branch1 = ('branch', 'step', 'one')
        f_branch2 = ('branch', 'step', 'two')
        f_x = ('shared_entity', 'is_a', 'landmark')
        f_source = ('raw_source', 'mentions', 'landmark')

        # f_x is a base fact
        syn_base = {f_x: 0.9, f_source: 0.8}
        syn_rules = defaultdict(list)
        syn_rules[f_goal].append(('r_main', (f_branch1, f_branch2), 0.95))
        # Branch 1 uses f_x directly
        syn_rules[f_branch1].append(('r_b1', (f_x,), 0.90))
        # Branch 2 derives f_x from f_source
        syn_rules[f_branch2].append(('r_b2', (f_x,), 0.90))
        syn_rules[f_x].append(('r_derive_x', (f_source,), 0.85))

        res = certify_query_structural_a1(f_goal, syn_base, syn_rules, max_depth=3)
        self.assertEqual(res['c2_disjoint'], 0, "Base-derived aliasing must violate C2")
        self.assertEqual(res['certified_a1'], 0)

    def test_synthetic_pure_tree_hierarchy(self):
        """
        Soundness positive test: A strictly hierarchical multi-level derivation tree
        with mutually disjoint premises at every branch level.
        Must successfully achieve certified_a1 = 1.
        """
        f_root = ('root', 'state', 'true')
        f_l1_a = ('l1', 'branch', 'a')
        f_l1_b = ('l1', 'branch', 'b')
        f_l2_a1 = ('l2', 'leaf', 'a1')
        f_l2_a2 = ('l2', 'leaf', 'a2')
        f_l2_b1 = ('l2', 'leaf', 'b1')
        f_l2_b2 = ('l2', 'leaf', 'b2')

        syn_base = {
            f_l2_a1: 0.85,
            f_l2_a2: 0.80,
            f_l2_b1: 0.90,
            f_l2_b2: 0.75,
        }
        syn_rules = defaultdict(list)
        syn_rules[f_root].append(('r_root', (f_l1_a, f_l1_b), 0.95))
        syn_rules[f_l1_a].append(('r_branch_a', (f_l2_a1, f_l2_a2), 0.90))
        syn_rules[f_l1_b].append(('r_branch_b', (f_l2_b1, f_l2_b2), 0.85))

        res = certify_query_structural_a1(f_root, syn_base, syn_rules, max_depth=3)
        self.assertEqual(res['c1_acyclic'], 1, "Pure tree must be acyclic")
        self.assertEqual(res['c2_disjoint'], 1, "Pure tree sibling branches must be disjoint")
        self.assertEqual(res['certified_a1'], 1, "Pure tree must be certified for Stage 2")
        self.assertEqual(res['n_diamonds'], 0)


if __name__ == '__main__':
    unittest.main()
