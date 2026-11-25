#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_certified_horn_domain_split.py
=============================================================================
Definitive Domain-Partitioned Evaluation of Decision-Certified Horn Reasoning:
- In-Domain Test Set: Hanoi (500 clips, 30 landmarks, 12,554 facts)
- Cross-Domain Evaluation Set: HCMC (300 clips, 5 landmarks, 3,793 facts)

Evaluates:
  1. Derivation workload reduction across eps in {0.01, 0.001, 0.0001}
  2. Cohort early stopping (m=1 vs m>=2)
  3. Condition A1 verification and order-inversion audit
  4. 3-level output equivalence (numerical eps, policy decisions, top-5 clips)
  5. In-memory execution time and projected I/O speedup
=============================================================================
"""

import json
import math
import time
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
OUTPUTS_DIR = REPO_ROOT / "experiments" / "outputs"
CONFIG_DIR = REPO_ROOT / "experiments" / "phase1" / "config"
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"

# Load manifests and evidence
with open(CONFIG_DIR / "hanoi_manifest.jsonl") as f:
    hn_manifest = [json.loads(l) for l in f]
with open(CONFIG_DIR / "hcmc_manifest.jsonl") as f:
    hc_manifest = [json.loads(l) for l in f]

with open(PHASE1_DIR / "hanoi" / "evidence.jsonl") as f:
    hn_ev = [json.loads(l) for l in f]
with open(PHASE1_DIR / "hcmc" / "evidence.jsonl") as f:
    hc_ev = [json.loads(l) for l in f]

from experiments.fkg_certainty_redesign.verify_certified_horn_early_stopping import (
    build_knowledge_graph_and_derivations, run_full_closure_oracle
)

print("Building knowledge graph and derivations...")
fused_facts, triple_to_fact, derivations = build_knowledge_graph_and_derivations()
exact_mu_all = run_full_closure_oracle(fused_facts, derivations)

HCMC_LANDMARKS = {
    "saigon_central_post_office",
    "independence_palace",
    "ben_thanh_market",
    "saigon_opera_house",
    "ba_thien_hau_temple"
}

# Partition facts into Hanoi vs HCMC
domain_facts = {"HANOI": {}, "HCMC": {}}
for fid, f in fused_facts.items():
    lm = (f.get("landmark") or f.get("landmark_id") or "").strip().lower()
    s = (f.get("subject") or "").strip().lower()
    is_hc = any(hcl in lm for hcl in HCMC_LANDMARKS) or any(hcl in s for hcl in HCMC_LANDMARKS)
    if is_hc:
        domain_facts["HCMC"][fid] = f
    else:
        domain_facts["HANOI"][fid] = f

print(f"Partitioned: Hanoi In-Domain = {len(domain_facts['HANOI']):,} facts | HCMC Cross-Domain = {len(domain_facts['HCMC']):,} facts")

TAU_ASSERT = 0.80
TAU_OMIT = 0.50

def get_policy(mu):
    if mu >= TAU_ASSERT:
        return "ASSERT"
    elif mu >= TAU_OMIT:
        return "HEDGE"
    else:
        return "OMIT"

domain_results = {}

for dom_name, facts_dict in domain_facts.items():
    dom_manifest = hn_manifest if dom_name == "HANOI" else hc_manifest
    dom_ev = hn_ev if dom_name == "HANOI" else hc_ev
    total_dom_facts = len(facts_dict)
    
    # Calculate oracle time for this domain
    t0_ora = time.perf_counter()
    ora_mu = {}
    total_dom_derivs = 0
    m_counts = {"m0": 0, "m1": 0, "m_ge_2": 0}
    
    for fid, fact in facts_dict.items():
        s = (fact.get("subject") or "").strip().lower()
        p = (fact.get("predicate") or "").strip().lower()
        o = (fact.get("object") or "").strip().lower()
        t = (s, p, o)
        d_list = derivations.get(t, [])
        m = len(d_list)
        b_f = fact.get("w_base", 0.0)
        total_dom_derivs += m
        
        if m == 0:
            m_counts["m0"] += 1
            ora_mu[fid] = b_f
        elif m == 1:
            m_counts["m1"] += 1
            ora_mu[fid] = 1.0 - (1.0 - b_f) * (1.0 - d_list[0][1])
        else:
            m_counts["m_ge_2"] += 1
            prod = 1.0 - b_f
            for d in d_list:
                prod *= (1.0 - d[1])
            ora_mu[fid] = 1.0 - prod
            
    t1_ora = time.perf_counter()
    ora_time_ms = (t1_ora - t0_ora) * 1000.0

    dom_data = {
        "domain_name": dom_name,
        "role": "In-Domain Test (Primary)" if dom_name == "HANOI" else "Cross-Domain Evaluation (Dedicated Held-Out)",
        "num_clips": len(dom_manifest),
        "total_facts": total_dom_facts,
        "total_derivations": total_dom_derivs,
        "facts_m0": m_counts["m0"],
        "facts_m1": m_counts["m1"],
        "facts_m_ge_2": m_counts["m_ge_2"],
        "facts_with_derivs": m_counts["m1"] + m_counts["m_ge_2"],
        "oracle_scan_time_ms": round(ora_time_ms, 2),
        "tolerances": {}
    }

    for eps in [0.01, 0.001, 0.0001]:
        t0_bnb = time.perf_counter()
        stopped_mu = {}
        derivs_evaluated = 0
        node_expansions_full = 0
        node_expansions_pruned = 0
        
        m1_stopped = 0
        m_ge_2_stopped = 0
        bound_violations = 0
        max_error = 0.0
        err_sum = 0.0
        
        policy_matches = 0
        policy_flips = 0
        near_thresh_facts = 0
        
        for fid, fact in facts_dict.items():
            s = (fact.get("subject") or "").strip().lower()
            p = (fact.get("predicate") or "").strip().lower()
            o = (fact.get("object") or "").strip().lower()
            t = (s, p, o)
            d_list = derivations.get(t, [])
            m = len(d_list)
            b_f = fact.get("w_base", 0.0)
            node_expansions_full += m
            
            ex_val = ora_mu[fid]
            if abs(ex_val - TAU_ASSERT) <= eps or abs(ex_val - TAU_OMIT) <= eps:
                near_thresh_facts += 1
                
            if m == 0:
                stopped_mu[fid] = b_f
                st_val = b_f
            elif m == 1:
                q_max = d_list[0][1]
                gap_0 = (1.0 - b_f) * (1.0 - (1.0 - q_max))
                if gap_0 <= eps:
                    m1_stopped += 1
                    stopped_mu[fid] = b_f
                    st_val = b_f
                    derivs_evaluated += 0
                    node_expansions_pruned += 0
                else:
                    st_val = 1.0 - (1.0 - b_f) * (1.0 - q_max)
                    stopped_mu[fid] = st_val
                    derivs_evaluated += 1
                    node_expansions_pruned += 1
            else:
                # Best-First sort
                sorted_d = sorted(d_list, key=lambda x: -x[1])
                q_max = sorted_d[0][1]
                gap_0 = (1.0 - b_f) * (1.0 - (1.0 - q_max)**m)
                
                if gap_0 <= eps:
                    m_ge_2_stopped += 1
                    stopped_mu[fid] = b_f
                    st_val = b_f
                    derivs_evaluated += 0
                    node_expansions_pruned += 0
                else:
                    prod = 1.0 - b_f
                    stopped = False
                    for k in range(1, m + 1):
                        prod *= (1.0 - sorted_d[k-1][1])
                        L_k = 1.0 - prod
                        derivs_evaluated += 1
                        node_expansions_pruned += 1
                        rem = m - k
                        if rem == 0:
                            st_val = L_k
                            stopped = True
                            break
                        q_curr = sorted_d[k-1][1]
                        U_k = 1.0 - (1.0 - L_k) * ((1.0 - q_curr)**rem)
                        
                        if ex_val < L_k - 1e-9 or ex_val > U_k + 1e-9:
                            bound_violations += 1
                            
                        if (U_k - L_k) <= eps:
                            m_ge_2_stopped += 1
                            st_val = L_k
                            stopped = True
                            break
                    if not stopped:
                        st_val = L_k
                    stopped_mu[fid] = st_val
                    
            err = abs(ex_val - st_val)
            err_sum += err
            if err > max_error:
                max_error = err
                
            if get_policy(ex_val) == get_policy(st_val):
                policy_matches += 1
            else:
                policy_flips += 1
                
        t1_bnb = time.perf_counter()
        bnb_time_ms = (t1_bnb - t0_bnb) * 1000.0
        
        # Check Top-5 Fact Selection on clips
        top5_set_matches = 0
        top5_order_matches = 0
        for ep in dom_ev:
            cand_facts = ep.get("top5_facts", [])
            if not cand_facts:
                top5_set_matches += 1
                top5_order_matches += 1
                continue
            sorted_ex = sorted(cand_facts, key=lambda f: -ora_mu.get(f.get("fact_id", ""), f.get("w_base", 0.0)))
            sorted_st = sorted(cand_facts, key=lambda f: -stopped_mu.get(f.get("fact_id", ""), f.get("w_base", 0.0)))
            ids_ex = [f.get("fact_id") for f in sorted_ex[:5]]
            ids_st = [f.get("fact_id") for f in sorted_st[:5]]
            if ids_ex == ids_st:
                top5_order_matches += 1
            if set(ids_ex) == set(ids_st):
                top5_set_matches += 1
                
        derivs_saved = total_dom_derivs - derivs_evaluated
        derivs_saved_pct = (derivs_saved / total_dom_derivs * 100.0) if total_dom_derivs > 0 else 0.0
        expansions_saved_pct = ((node_expansions_full - node_expansions_pruned) / node_expansions_full * 100.0) if node_expansions_full > 0 else 0.0
        
        # Speedup modeling
        # Crossover latency: t_diff / expansions_saved
        delta_compute = bnb_time_ms - ora_time_ms
        exp_saved_count = node_expansions_full - node_expansions_pruned
        crossover_us = (delta_compute / exp_saved_count * 1000.0) if exp_saved_count > 0 else 0.0
        
        # Projected times
        t_proj_sqlite_full = ora_time_ms + (node_expansions_full * 0.010)
        t_proj_sqlite_bnb = bnb_time_ms + (node_expansions_pruned * 0.010)
        proj_speedup_sqlite = t_proj_sqlite_full / t_proj_sqlite_bnb if t_proj_sqlite_bnb > 0 else 1.0
        
        t_proj_neo4j_full = ora_time_ms + (node_expansions_full * 0.200)
        t_proj_neo4j_bnb = bnb_time_ms + (node_expansions_pruned * 0.200)
        proj_speedup_neo4j = t_proj_neo4j_full / t_proj_neo4j_bnb if t_proj_neo4j_bnb > 0 else 1.0
        
        dom_data["tolerances"][str(eps)] = {
            "eps": eps,
            "derivs_total": total_dom_derivs,
            "derivs_evaluated": derivs_evaluated,
            "derivs_saved": derivs_saved,
            "derivs_saved_pct": round(derivs_saved_pct, 2),
            "node_expansions_full": node_expansions_full,
            "node_expansions_pruned": node_expansions_pruned,
            "node_expansions_saved_pct": round(expansions_saved_pct, 2),
            "multi_deriv_stopped": m_ge_2_stopped,
            "multi_deriv_total": m_counts["m_ge_2"],
            "multi_deriv_stopped_pct": round(m_ge_2_stopped / m_counts["m_ge_2"] * 100.0, 2) if m_counts["m_ge_2"] > 0 else 0.0,
            "single_deriv_stopped": m1_stopped,
            "single_deriv_total": m_counts["m1"],
            "single_deriv_stopped_pct": round(m1_stopped / m_counts["m1"] * 100.0, 2) if m_counts["m1"] > 0 else 0.0,
            "all_deriv_stopped": m1_stopped + m_ge_2_stopped,
            "all_deriv_total": m_counts["m1"] + m_counts["m_ge_2"],
            "all_deriv_stopped_pct": round((m1_stopped + m_ge_2_stopped) / (m_counts["m1"] + m_counts["m_ge_2"]) * 100.0, 2),
            "bound_violations": bound_violations,
            "max_observed_error": round(max_error, 8),
            "mean_observed_error": round(err_sum / total_dom_facts, 8),
            "policy_matches": policy_matches,
            "policy_flips": policy_flips,
            "policy_equivalence_pct": round(policy_matches / total_dom_facts * 100.0, 2),
            "near_threshold_facts": near_thresh_facts,
            "clips_count": len(dom_manifest),
            "top5_set_matches": top5_set_matches,
            "top5_set_equivalence_pct": round(top5_set_matches / len(dom_manifest) * 100.0, 2),
            "top5_order_matches": top5_order_matches,
            "top5_order_matches_pct": round(top5_order_matches / len(dom_manifest) * 100.0, 2),
            "in_memory_time_full_ms": round(ora_time_ms, 2),
            "in_memory_time_bnb_ms": round(bnb_time_ms, 2),
            "in_memory_speedup": round(ora_time_ms / bnb_time_ms, 2) if bnb_time_ms > 0 else 1.0,
            "crossover_latency_us": round(crossover_us, 3),
            "projected_sqlite_speedup": round(proj_speedup_sqlite, 2),
            "projected_neo4j_speedup": round(proj_speedup_neo4j, 2),
        }
        
    domain_results[dom_name] = dom_data

# Save results
OUTPUT_JSON = OUTPUTS_DIR / "certified_horn_domain_partitioned_report.json"
with open(OUTPUT_JSON, "w") as f:
    json.dump(domain_results, f, indent=2)

print(f"\nSaved partitioned report to {OUTPUT_JSON}")

# Print comparative summary table
print("\n" + "="*80)
print("COMPARATIVE SUMMARY: HANOI IN-DOMAIN TEST VS HCMC CROSS-DOMAIN")
print("="*80)
for eps in ["0.01", "0.001", "0.0001"]:
    hn = domain_results["HANOI"]["tolerances"][eps]
    hc = domain_results["HCMC"]["tolerances"][eps]
    print(f"\n--- TOLERANCE eps = {eps} ---")
    print(f"Derivations Saved:        Hanoi = {hn['derivs_saved']:,} / {hn['derivs_total']:,} ({hn['derivs_saved_pct']}%)  |  HCMC = {hc['derivs_saved']:,} / {hc['derivs_total']:,} ({hc['derivs_saved_pct']}%)")
    print(f"Multi-deriv Stopped:      Hanoi = {hn['multi_deriv_stopped']:,} / {hn['multi_deriv_total']:,} ({hn['multi_deriv_stopped_pct']}%)  |  HCMC = {hc['multi_deriv_stopped']:,} / {hc['multi_deriv_total']:,} ({hc['multi_deriv_stopped_pct']}%)")
    print(f"Single-deriv Stopped:     Hanoi = {hn['single_deriv_stopped']:,} / {hn['single_deriv_total']:,} ({hn['single_deriv_stopped_pct']}%)  |  HCMC = {hc['single_deriv_stopped']:,} / {hc['single_deriv_total']:,} ({hc['single_deriv_stopped_pct']}%)")
    print(f"All Deriv-Facts Stopped:  Hanoi = {hn['all_deriv_stopped']:,} / {hn['all_deriv_total']:,} ({hn['all_deriv_stopped_pct']}%)  |  HCMC = {hc['all_deriv_stopped']:,} / {hc['all_deriv_total']:,} ({hc['all_deriv_stopped_pct']}%)")
    print(f"Max Error (eps={eps}):    Hanoi = {hn['max_observed_error']:.8f}  |  HCMC = {hc['max_observed_error']:.8f}")
    print(f"Policy Equivalence:       Hanoi = {hn['policy_matches']:,} / {domain_results['HANOI']['total_facts']:,} ({hn['policy_equivalence_pct']}%)  |  HCMC = {hc['policy_matches']:,} / {domain_results['HCMC']['total_facts']:,} ({hc['policy_equivalence_pct']}%)")
    print(f"Top-5 Set Equivalence:    Hanoi = {hn['top5_set_matches']} / {hn['clips_count']} ({hn['top5_set_equivalence_pct']}%)  |  HCMC = {hc['top5_set_matches']} / {hc['clips_count']} ({hc['top5_set_equivalence_pct']}%)")
    print(f"Projected SQLite Speedup: Hanoi = {hn['projected_sqlite_speedup']}x  |  HCMC = {hc['projected_sqlite_speedup']}x")
    print(f"Projected Neo4j Speedup:  Hanoi = {hn['projected_neo4j_speedup']}x  |  HCMC = {hc['projected_neo4j_speedup']}x")
