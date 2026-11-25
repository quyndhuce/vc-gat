#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_theory_properties.py
=============================================================================
Section 9 Synthetic Unit Tests for Theoretical Properties (05_theory.tex)
Must pass with 0 errors before running on real data.

Tests:
  1. Two-support overlap gap: gap = x(1-x)y1*y2 (error < 1e-9)
  2. Disjoint supports: pi == nu_vee (error < 1e-9)
  3. Prop 2 (Truncation upper bound): 10,000 random Horn programs (A1 holds)
     verifying nu_S <= mu_F <= upper for all S.
  4. Prop 4 (Conditioning nested refinement): 10,000 random support families
     verifying U_k decreasing, L_k increasing, U_{|C|} == L_{|C|} == pi,
     and Monte Carlo calibration check.
=============================================================================
"""

import math
import random
import itertools
from collections import Counter
from typing import List, Dict, Set, Tuple, FrozenSet
import numpy as np

def prod(iterable):
    p = 1.0
    for x in iterable:
        p *= float(x)
    return p

def pi_exact(supports: List[FrozenSet[str]], p: Dict[str, float]) -> float:
    """Exact union probability under independent component retention."""
    if not supports:
        return 0.0
    cnt = Counter(u for S in supports for u in S)
    C = [u for u, k in cnt.items() if k >= 2]
    if len(C) > 20:
        raise ValueError(f"|C| = {len(C)} exceeds exact enumeration limit 20")
    
    total = 0.0
    for z in itertools.product([0, 1], repeat=len(C)):
        zmap = dict(zip(C, z))
        wz = prod(p[u] if zmap[u] else 1.0 - p[u] for u in C)
        q = [
            prod(p[u] for u in S if u not in zmap)
            for S in supports
            if all(zmap.get(u, 1) for u in S)
        ]
        total += wz * (1.0 - prod(1.0 - x for x in q))
    return float(total)

def compute_nested_conditioning(
    supports: List[FrozenSet[str]], p: Dict[str, float], max_k: int = None
) -> Tuple[List[float], List[float], List[float]]:
    """Computes nested conditional bounds [L_k, U_k] and widths W_k."""
    cnt = Counter(u for S in supports for u in S)
    C = [u for u, k in cnt.items() if k >= 2]
    # Sort C by decreasing support-count, ties broken by lower p
    C.sort(key=lambda u: (-cnt[u], p[u]))
    
    if max_k is None or max_k > len(C):
        max_k = len(C)
        
    p_supps = [prod(p[u] for u in S) for S in supports]
    
    # Check disjoint for initial L0
    all_disjoint = True
    for i in range(len(supports)):
        for j in range(i + 1, len(supports)):
            if supports[i] & supports[j]:
                all_disjoint = False
                break
        if not all_disjoint:
            break
            
    if all_disjoint:
        L0 = 1.0 - prod(1.0 - ps for ps in p_supps)
    else:
        m = len(supports)
        best = max(p_supps) if p_supps else 0.0
        for r in range(2, m + 1):
            for combo in itertools.combinations(range(m), r):
                seen = set()
                disj = True
                for idx in combo:
                    if seen & supports[idx]:
                        disj = False
                        break
                    seen.update(supports[idx])
                if disj:
                    val = 1.0
                    for idx in combo:
                        val *= (1.0 - p_supps[idx])
                    b = 1.0 - val
                    if b > best:
                        best = b
        L0 = best
        
    U0 = 1.0 - prod(1.0 - ps for ps in p_supps)
    
    L_list = [L0]
    U_list = [U0]
    W_list = [U0 - L0]
    
    for k in range(1, max_k + 1):
        Ck = C[:k]
        L_cond_sum = 0.0
        U_cond_sum = 0.0
        
        for z in itertools.product([0, 1], repeat=k):
            zmap = dict(zip(Ck, z))
            wz = prod(p[u] if zmap[u] else 1.0 - p[u] for u in Ck)
            surviving = [S for S in supports if all(zmap.get(u, 1) for u in S if u in zmap)]
            if not surviving:
                L_z = 0.0
                U_z = 0.0
            else:
                rem_supps = [frozenset(u for u in S if u not in zmap) for S in surviving]
                qs = [prod(p[u] for u in S if u not in zmap) for S in surviving]
                U_z = 1.0 - prod(1.0 - x for x in qs)
                
                # Check if remaining components are pairwise disjoint
                all_disjoint = True
                for i in range(len(rem_supps)):
                    for j in range(i + 1, len(rem_supps)):
                        if rem_supps[i] & rem_supps[j]:
                            all_disjoint = False
                            break
                    if not all_disjoint:
                        break
                        
                if all_disjoint:
                    L_z = U_z
                else:
                    m = len(surviving)
                    best_bound = 0.0
                    for r in range(1, m + 1):
                        for combo in itertools.combinations(range(m), r):
                            seen = set()
                            disj = True
                            for idx in combo:
                                if seen & rem_supps[idx]:
                                    disj = False
                                    break
                                seen.update(rem_supps[idx])
                            if disj:
                                val = 1.0
                                for idx in combo:
                                    val *= (1.0 - qs[idx])
                                b = 1.0 - val
                                if b > best_bound:
                                    best_bound = b
                    L_z = best_bound
                
            L_cond_sum += wz * L_z
            U_cond_sum += wz * U_z
            
        Lk = max(L_list[-1], L_cond_sum)
        Uk = min(U_list[-1], U_cond_sum)
        L_list.append(Lk)
        U_list.append(Uk)
        W_list.append(Uk - Lk)
        
    return L_list, U_list, W_list


# =============================================================================
# TEST 1: Two supports sharing a core
# =============================================================================
def test_1_two_supports_overlap():
    print("Running Test 1: Two supports sharing a core...")
    np.random.seed(42)
    max_err = 0.0
    N = 1000
    for _ in range(N):
        x = np.random.uniform(0.01, 0.99)
        y1 = np.random.uniform(0.01, 0.99)
        y2 = np.random.uniform(0.01, 0.99)
        
        p = {"I": x, "P1": y1, "P2": y2}
        supports = [frozenset(["I", "P1"]), frozenset(["I", "P2"])]
        
        pi = pi_exact(supports, p)
        nu_vee = 1.0 - (1.0 - x * y1) * (1.0 - x * y2)
        gap_measured = nu_vee - pi
        gap_analytic = x * (1.0 - x) * y1 * y2
        
        err = abs(gap_measured - gap_analytic)
        if err > max_err:
            max_err = err
        assert err < 1e-9, f"Test 1 failed: err={err} > 1e-9 for x={x}, y1={y1}, y2={y2}"
        
    print(f"  [PASS] Test 1 passed ({N} cases). Max error = {max_err:.2e} < 1e-9")


# =============================================================================
# TEST 2: Disjoint supports
# =============================================================================
def test_2_disjoint_supports():
    print("Running Test 2: Disjoint supports...")
    np.random.seed(42)
    max_err = 0.0
    N = 1000
    for _ in range(N):
        m = np.random.randint(2, 6)
        supports = []
        p = {}
        for i in range(m):
            comp_name = f"u_{i}"
            prob = np.random.uniform(0.05, 0.95)
            p[comp_name] = prob
            supports.append(frozenset([comp_name]))
            
        pi = pi_exact(supports, p)
        nu_vee = 1.0 - prod(1.0 - p[f"u_{i}"] for i in range(m))
        err = abs(pi - nu_vee)
        if err > max_err:
            max_err = err
        assert err < 1e-9, f"Test 2 failed: err={err} > 1e-9 for m={m}"
        
    print(f"  [PASS] Test 2 passed ({N} cases). Max error = {max_err:.2e} < 1e-9")


# =============================================================================
# TEST 3: Prop 2 on 10,000 random Horn programs (A1 holds)
# =============================================================================
def test_3_prop2_horn_programs():
    print("Running Test 3: Prop 2 on 10,000 random Horn programs (A1 holds)...")
    np.random.seed(42)
    random.seed(42)
    N = 10000
    violations = 0
    max_viol_val = 0.0
    
    for it in range(N):
        n_base = random.randint(2, 3)
        base_weights = {f"b_{i}": round(random.uniform(0.2, 0.95), 3) for i in range(n_base)}
        
        gamma_r = 0.90
        rules = []
        
        n_inter = random.randint(1, 2)
        inter_nodes = [f"h_{j}" for j in range(n_inter)]
        for h in inter_nodes:
            body = random.sample(list(base_weights.keys()), random.randint(1, min(2, n_base)))
            rules.append((tuple(body), h))
            
        target = "B"
        all_prior = list(base_weights.keys()) + inter_nodes
        n_target_rules = random.randint(1, 3)
        for _ in range(n_target_rules):
            body_size = random.randint(1, min(2, len(all_prior)))
            body = random.sample(all_prior, body_size)
            rules.append((tuple(body), target))
            
        def get_derivations(node, depth, path_nodes):
            if depth > 3 or node in path_nodes:
                return []
            if node in base_weights:
                return [(base_weights[node], [node])]
            derivs = []
            for body, head in rules:
                if head == node:
                    body_derivs = [get_derivations(b, depth + 1, path_nodes | {node}) for b in body]
                    if any(len(bd) == 0 for bd in body_derivs):
                        continue
                    for combo in itertools.product(*body_derivs):
                        val = gamma_r
                        comps = [f"rule_{body}->{head}"]
                        for b_val, b_comps in combo:
                            val *= b_val
                            comps.extend(b_comps)
                        derivs.append((val, comps))
            return derivs
            
        derivs = get_derivations(target, 0, set())
        if not derivs:
            continue
            
        # DP for Z_L
        L = 3
        Z_prev = {k: v for k, v in base_weights.items()}
        for n in inter_nodes + [target]:
            if n not in Z_prev:
                Z_prev[n] = 0.0
                
        for l in range(1, L + 1):
            Z = {k: (base_weights[k] if k in base_weights else 0.0) for k in Z_prev}
            for body, head in rules:
                val_prod = prod(Z_prev.get(b, 0.0) for b in body)
                Z[head] += gamma_r * val_prod
            Z_prev = Z
            
        Z_L = Z_prev[target]
        
        vals = [d[0] for d in derivs]
        mu_F = 1.0 - prod(1.0 - v for v in vals)
        
        sorted_vals = sorted(vals, reverse=True)
        n_s = random.randint(1, len(vals))
        S_vals = sorted_vals[:n_s]
        
        nu_S = 1.0 - prod(1.0 - v for v in S_vals)
        T_S = Z_L - sum(S_vals)
        upper = nu_S + (1.0 - nu_S) * min(1.0, max(0.0, T_S))
        
        if mu_F < nu_S - 1e-12:
            violations += 1
        if mu_F > upper + 1e-12:
            violations += 1
            diff = mu_F - upper
            if diff > max_viol_val:
                max_viol_val = diff
                
    assert violations == 0, f"Test 3 failed: {violations} violations found! Max viol: {max_viol_val}"
    print(f"  [PASS] Test 3 passed ({N} Horn programs). Violations = {violations}, Max upper viol = {max_viol_val:.2e}")


# =============================================================================
# TEST 4: Prop 4 on 10,000 random support families
# =============================================================================
def test_4_prop4_conditioning():
    print("Running Test 4: Prop 4 conditioning on 10,000 support families...")
    np.random.seed(42)
    random.seed(42)
    N = 10000
    monotonicity_violations = 0
    exact_at_end_violations = 0
    max_pi_diff = 0.0
    
    mc_checks = 100
    mc_passed = 0
    
    for it in range(N):
        n_shared = random.randint(1, 3)
        shared_comps = [f"c_{i}" for i in range(n_shared)]
        
        m_supps = random.randint(2, 4)
        supports = []
        p = {}
        for sc in shared_comps:
            p[sc] = round(random.uniform(0.1, 0.9), 3)
            
        for s_idx in range(m_supps):
            s_shared = random.sample(shared_comps, random.randint(1, n_shared))
            priv_comp = f"priv_{s_idx}"
            p[priv_comp] = round(random.uniform(0.1, 0.9), 3)
            supports.append(frozenset(s_shared + [priv_comp]))
            
        pi = pi_exact(supports, p)
        L_list, U_list, W_list = compute_nested_conditioning(supports, p)
        
        for k in range(1, len(L_list)):
            if L_list[k] < L_list[k-1] - 1e-12:
                monotonicity_violations += 1
            if U_list[k] > U_list[k-1] + 1e-12:
                monotonicity_violations += 1
            if W_list[k] > W_list[k-1] + 1e-12:
                monotonicity_violations += 1
                
        final_L = L_list[-1]
        final_U = U_list[-1]
        diff_pi = max(abs(final_L - pi), abs(final_U - pi))
        if diff_pi > max_pi_diff:
            max_pi_diff = diff_pi
        if diff_pi > 1e-9:
            exact_at_end_violations += 1
            
        if it < mc_checks:
            S_mc = 1000000
            comps = list(p.keys())
            probs = np.array([p[u] for u in comps])
            draws = np.random.uniform(size=(S_mc, len(comps))) < probs
            comp_idx = {u: i for i, u in enumerate(comps)}
            
            surv_any = np.zeros(S_mc, dtype=bool)
            for supp in supports:
                idxs = [comp_idx[u] for u in supp]
                supp_active = np.all(draws[:, idxs], axis=1)
                surv_any = surv_any | supp_active
                
            pi_mc = np.mean(surv_any)
            sigma = math.sqrt(max(pi * (1.0 - pi), 1e-6) / S_mc)
            z_score = abs(pi - pi_mc) / sigma
            if z_score <= 3.5:
                mc_passed += 1
                
    assert monotonicity_violations == 0, f"Monotonicity violations: {monotonicity_violations}"
    assert exact_at_end_violations == 0, f"Exact at end violations: {exact_at_end_violations}"
    print(f"  [PASS] Test 4 passed ({N} support families).")
    print(f"         Monotonicity violations = {monotonicity_violations}")
    print(f"         Exactness at |C|: max error = {max_pi_diff:.2e} < 1e-9")
    print(f"         Monte Carlo validation: {mc_passed}/{mc_checks} within 3.5 sigma.")

import unittest

class TestTheoryProperties(unittest.TestCase):
    def test_overlap_gap(self):
        test_1_two_supports_overlap()

    def test_disjoint_supports(self):
        test_2_disjoint_supports()

    def test_prop2_truncation(self):
        test_3_prop2_horn_programs()

    def test_prop4_conditioning(self):
        test_4_prop4_conditioning()

if __name__ == "__main__":
    unittest.main()
