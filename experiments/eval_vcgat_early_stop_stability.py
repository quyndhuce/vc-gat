#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/eval_vcgat_early_stop_stability.py
=============================================================================
Evaluates the sensitivity and stability of VC-GAT (NewDeepVCGAT) relevance
scoring, candidate ranking, and stratified admitted fact set A_KG when
supplied with Corollary-1 early-stopping interval proxies:
  - Lower bound nu_S
  - Midpoint 0.5 * (nu_S + upper_S)
  - Upper bound upper_S
  - Fixed truncation K in {1, 2, 3, 5} (lower bound & midpoint)
compared against exact depth-bounded Horn closure mu_F_exact (and static base FKG).

Runs cleanly on CPU (CUDA_VISIBLE_DEVICES="" recommended when GPU is occupied).
Outputs:
  - results/vcgat_early_stop_stability.json
  - results/vcgat_early_stop_stability.csv
=============================================================================
"""

import itertools
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from transformers import AutoModel, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_2x2_factorial_vcgat_experiment import (
    DEFAULT_SIGLIP2_MODEL,
    NewDeepVCGAT,
    build_candidate_subgraph_new_sparse,
    format_fact_claim,
)
from experiments.run_m0_export_facts import (
    GAMMA_R,
    load_knowledge_graph_and_rules,
    run_dynamic_programming_zl,
)
from experiments.vcgat_semantic_anchoring_engine import (
    DIRECT_RELS,
    INFERRED_RELS,
    VISUAL_RELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("VCGATEarlyStopStability")

DATA_DIR = REPO_ROOT / "data" / "caption_dataset"
FEAT_DIR = REPO_ROOT / "experiments" / "features"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_JSONL = DATA_DIR / "generated_captions_test_balanced_500_vcgat.jsonl"
FKG_PATH = REPO_ROOT / "data" / "fkg_per_kb" / "fkg_global_merged.json"
SIGLIP2_EMB_PATH = FEAT_DIR / "siglip2_test_video_embeddings.npy"
CHECKPOINT_FULL = FEAT_DIR / "best_scientific_deep_vc_gat.pt"

THETA_M = 0.50
THETA_H = 0.80


def get_regime(c_val: float) -> str:
    if c_val < THETA_M:
        return "omit"
    elif c_val < THETA_H:
        return "hedge"
    return "assert"


def compute_sorted_derivations(
    fact_triple: Tuple[str, str, str],
    mat_base: Dict[Tuple[str, str, str], float],
    rules_by_head: Dict[Tuple[str, str, str], List[Any]],
    Z_L_map: Dict[Tuple[str, str, str], float],
    fallback_mu: float,
    max_depth: int = 3,
) -> Dict[str, Any]:
    """Computes canonical derivations sorted descending + A1 flag + Z_L."""
    memo = {}

    def dfs(f, cur_depth, ancestor_facts):
        state = (f, cur_depth, tuple(sorted(ancestor_facts)))
        if state in memo:
            return memo[state]
        trees = []
        if f in mat_base:
            trees.append((mat_base[f], frozenset([f]), False))
        if cur_depth < max_depth:
            for rid, body, r_conf in rules_by_head.get(f, []):
                if any(b in ancestor_facts for b in body):
                    continue
                new_anc = ancestor_facts | {f}
                body_trees = [dfs(b, cur_depth + 1, new_anc) for b in body]
                if any(len(bt) == 0 for bt in body_trees):
                    continue
                for combo in itertools.product(*body_trees):
                    val = GAMMA_R * r_conf
                    comps = {f"{rid}_{body}->{f}"}
                    viol = any(bt[2] for bt in combo)
                    for b_val, b_comps, _ in combo:
                        val *= b_val
                        if not comps.isdisjoint(b_comps):
                            viol = True
                        comps.update(b_comps)
                    trees.append((val, frozenset(comps), viol))
        memo[state] = trees
        return trees

    trees = dfs(fact_triple, 0, set())
    can_d = {}
    a1_viol = False
    for val, comps, viol in trees:
        if viol:
            a1_viol = True
        if comps not in can_d or val > can_d[comps]:
            can_d[comps] = val

    if not can_d:
        mu_exact = float(fallback_mu)
        return {
            "vals": [mu_exact],
            "Z_L": mu_exact,
            "a1_viol": False,
            "mu_exact": mu_exact,
            "n_deriv": 1,
        }

    sorted_vals = sorted(can_d.values(), reverse=True)
    mu_exact = float(1.0 - np.prod([1.0 - min(0.999999, v) for v in sorted_vals]))
    Z_L = float(Z_L_map.get(fact_triple, mat_base.get(fact_triple, mu_exact)))
    return {
        "vals": sorted_vals,
        "Z_L": Z_L,
        "a1_viol": a1_viol,
        "mu_exact": mu_exact,
        "n_deriv": len(sorted_vals),
    }


def eval_stopping_interval(
    deriv_info: Dict[str, Any],
    mu_L: float,
    mode: str,
    fixed_k: int = 0,
) -> Tuple[float, float, float, int, bool]:
    """
    Returns (mu_lower, mu_upper, mu_exact, n_stop, truncated_flag)
    for a triple given mu_L under:
      - mode == 'exact': (mu_exact, mu_exact, mu_exact, n_deriv, False)
      - mode == 'adaptive': Corollary-1 regime stopping with running-min upper bound
      - mode == 'fixed': top-K truncation with running-min upper bound
    """
    mu_exact = deriv_info["mu_exact"]
    vals = deriv_info["vals"]
    Z_L = deriv_info["Z_L"]
    M = len(vals)

    if mode == "exact" or M <= 1 or deriv_info["a1_viol"]:
        return mu_exact, mu_exact, mu_exact, M, False

    if mode == "fixed":
        n_eval = min(fixed_k, M)
        up_min = 1.0
        nu_n = 0.0
        for n in range(1, n_eval + 1):
            S_n = vals[:n]
            nu_n = float(1.0 - np.prod([1.0 - min(0.999999, v) for v in S_n]))
            T_n = float(max(0.0, Z_L - sum(S_n)))
            raw_up = float(min(1.0, nu_n + (1.0 - nu_n) * min(1.0, T_n)))
            up_min = min(up_min, raw_up)
        return nu_n, max(nu_n, up_min), mu_exact, n_eval, (n_eval < M and up_min > nu_n + 1e-12)

    # Adaptive Corollary-1 stopping
    up_min = 1.0
    for n in range(1, M + 1):
        S_n = vals[:n]
        nu_n = float(1.0 - np.prod([1.0 - min(0.999999, v) for v in S_n]))
        T_n = float(max(0.0, Z_L - sum(S_n)))
        raw_up = float(min(1.0, nu_n + (1.0 - nu_n) * min(1.0, T_n)))
        up_min = min(up_min, raw_up)
        if get_regime(mu_L * nu_n) == get_regime(mu_L * up_min):
            return nu_n, max(nu_n, up_min), mu_exact, n, (n < M and up_min > nu_n + 1e-12)

    # Exhausted without early stop -> exact
    return mu_exact, mu_exact, mu_exact, M, False


def rebuild_mu_edge(
    cands: List[Dict[str, Any]],
    mu_vec: np.ndarray,
    landmark: str,
) -> np.ndarray:
    """Matches build_candidate_subgraph_new_sparse edge-reliability matrix for an arbitrary mu_vec."""
    N = len(cands)
    lm_low = landmark.lower().strip()
    mu_edge = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        mu_edge[i, i] = float(mu_vec[i])
    for i in range(N):
        h_i = cands[i]["head"].lower().strip()
        t_i = cands[i]["tail"].lower().strip()
        mu_i = float(mu_vec[i])
        for j in range(i + 1, N):
            h_j = cands[j]["head"].lower().strip()
            t_j = cands[j]["tail"].lower().strip()
            mu_j = float(mu_vec[j])
            if t_i == h_j or h_i == t_j:
                w = float(min(mu_i, mu_j))
                mu_edge[i, j] = w
                mu_edge[j, i] = w
            elif t_i == t_j and t_i != lm_low and len(t_i) > 2:
                w = 0.8 * float(min(mu_i, mu_j))
                mu_edge[i, j] = w
                mu_edge[j, i] = w
    return mu_edge


def select_admitted_and_pools(
    cands: List[Dict[str, Any]],
    scores: np.ndarray,
    mu_vec: np.ndarray,
    mu_L: float,
    certainty: str,
    quota: Tuple[int, int, int],
) -> Tuple[List[Tuple[str, str, str]], Dict[str, List[Tuple[int, float]]], Dict[Tuple[str, str, str], str]]:
    """
    Reproduces exact VCGATSemanticAnchoringEngine.select_stratified_quota_facts
    selection + returns deduplicated sorted pools per stratum for margin analysis.
    """
    k_vis, k_dir, k_inf = quota
    N = len(cands)
    if N == 0:
        return [], {"vis": [], "dir": [], "inf": []}, {}

    def triple_key(f):
        return (f["head"].lower().strip(), f["relation"].lower().strip(), f["tail"].lower().strip())

    if certainty == "LOW":
        vis_idxs = [i for i, f in enumerate(cands) if f["relation"].lower() in VISUAL_RELS]
        if not vis_idxs:
            vis_idxs = list(range(N))
        vis_idxs.sort(key=lambda i: float(mu_vec[i]), reverse=True)
        sel = []
        reg = {}
        for i in vis_idxs[:k_vis]:
            tk = triple_key(cands[i])
            sel.append(tk)
            reg[tk] = get_regime(mu_L * float(mu_vec[i]))
        return sel, {"vis": [(i, float(mu_vec[i])) for i in vis_idxs], "dir": [], "inf": []}, reg

    raw_vis, raw_dir, raw_inf = [], [], []
    seen_vis, seen_dir, seen_inf = set(), set(), set()

    for i, f in enumerate(cands):
        r = f["relation"].lower()
        t = f["tail"].lower()
        key = (r, t)
        sc = float(scores[i])
        if r in VISUAL_RELS and key not in seen_vis:
            seen_vis.add(key)
            raw_vis.append((i, sc))
        elif r in DIRECT_RELS and key not in seen_dir:
            seen_dir.add(key)
            raw_dir.append((i, sc))
        elif (r in INFERRED_RELS or f.get("is_inferred", False)) and key not in seen_inf:
            seen_inf.add(key)
            raw_inf.append((i, sc))

    raw_vis.sort(key=lambda x: x[1], reverse=True)
    raw_dir.sort(key=lambda x: x[1], reverse=True)
    raw_inf.sort(key=lambda x: x[1], reverse=True)

    selected = []
    regimes = {}
    seen_all = set()

    for pool, limit in ((raw_vis, k_vis), (raw_dir, k_dir), (raw_inf, k_inf)):
        cnt = 0
        for idx, _ in pool:
            f = cands[idx]
            key = (f["relation"].lower(), f["tail"].lower())
            if key not in seen_all:
                seen_all.add(key)
                tk = triple_key(f)
                selected.append(tk)
                regimes[tk] = get_regime(mu_L * float(mu_vec[idx]))
                cnt += 1
                if cnt >= limit:
                    break

    total_target = k_vis + k_dir + k_inf
    if len(selected) < total_target:
        for idx, _ in (raw_vis + raw_dir + raw_inf):
            f = cands[idx]
            key = (f["relation"].lower(), f["tail"].lower())
            if key not in seen_all:
                seen_all.add(key)
                tk = triple_key(f)
                selected.append(tk)
                regimes[tk] = get_regime(mu_L * float(mu_vec[idx]))
                if len(selected) >= total_target:
                    break

    return selected, {"vis": raw_vis, "dir": raw_dir, "inf": raw_inf}, regimes


def load_siglip2_cpu(model_id: str):
    try:
        proc = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        mod = AutoModel.from_pretrained(model_id, local_files_only=True)
        LOGGER.info("Loaded %s from local HF cache on CPU.", model_id)
        return proc, mod
    except Exception:
        LOGGER.info("Local cache miss; downloading %s...", model_id)
        proc = AutoProcessor.from_pretrained(model_id)
        mod = AutoModel.from_pretrained(model_id)
        return proc, mod


def main():
    t_start = time.time()
    device = torch.device("cpu")
    LOGGER.info("Running VC-GAT early-stopping stability audit on %s...", device)

    # 1. Load Horn closure machinery + base FKG
    mat_base, rules_by_head = load_knowledge_graph_and_rules()
    Z_L_map = run_dynamic_programming_zl(mat_base, rules_by_head)

    with open(FKG_PATH, "r", encoding="utf-8") as f:
        fkg_raw = json.load(f)

    head_map = defaultdict(list)
    for fact in fkg_raw.get("multi_valued_facts", []):
        h = fact.get("head_surface", "").strip()
        r = fact.get("relation", "").strip()
        t = fact.get("tail_surface", "").strip()
        w = float(fact.get("mu_F", 0.5))
        if h and r and t:
            head_map[h.lower()].append({"head": h, "relation": r, "tail": t, "mu_F": w, "is_inferred": False})

    inferred_map = defaultdict(list)
    for h_low, facts in head_map.items():
        for fc in facts:
            if fc["relation"] in {"architectural_style", "has_style"}:
                inferred_map[h_low].append({
                    "head": fc["head"],
                    "relation": "features_style",
                    "tail": fc["tail"],
                    "mu_F": round(fc["mu_F"] * 0.9, 4),
                    "is_inferred": True,
                    "parent_triple": (fc["head"].lower().strip(), fc["relation"].lower().strip(), fc["tail"].lower().strip()),
                })

    # 2. Load 500 clips + video embeddings + VC-GAT checkpoint
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip()]
    siglip2_embeddings = np.load(SIGLIP2_EMB_PATH)

    vcgat = NewDeepVCGAT(d_model=768, num_heads=4, num_layers=2, dropout=0.0, tau_v=0.15).to(device)
    vcgat.load_state_dict(torch.load(CHECKPOINT_FULL, map_location=device))
    vcgat.eval()
    LOGGER.info("Loaded NewDeepVCGAT checkpoint from %s", CHECKPOINT_FULL)

    # 3. Collect all subgraphs and unique triples, encode with SigLIP-2 text encoder once
    clip_subgraphs = []
    unique_triples: Dict[Tuple[str, str, str], float] = {}
    unique_texts: Dict[Tuple[str, str, str], str] = {}

    for s in test_samples:
        cert = s.get("certainty_level", "HIGH")
        active_lm = s.get("landmark_gt", "") if cert in ("HIGH", "MEDIUM") else s.get("landmark_pred", "")
        cands, adj_mask_np, _ = build_candidate_subgraph_new_sparse(active_lm, head_map, inferred_map)
        clip_subgraphs.append((s, active_lm, cands, adj_mask_np))
        for fc in cands:
            tk = (fc["head"].lower().strip(), fc["relation"].lower().strip(), fc["tail"].lower().strip())
            if tk not in unique_triples:
                unique_triples[tk] = float(fc.get("mu_F", 0.5))
                unique_texts[tk] = format_fact_claim(fc["head"], fc["relation"], fc["tail"])

    LOGGER.info("Collected %d unique candidate triples across %d clips.", len(unique_triples), len(clip_subgraphs))

    # Precompute derivations for base/rule triples (and link inferred features_style parent)
    deriv_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for tk, fb_mu in unique_triples.items():
        if tk[1] == "features_style":
            # Inherit from architectural_style / has_style parent if present
            parent_tk = None
            for cand_rel in ("architectural_style", "has_style"):
                cand_k = (tk[0], cand_rel, tk[2])
                if cand_k in unique_triples or cand_k in mat_base or cand_k in rules_by_head:
                    parent_tk = cand_k
                    break
            if parent_tk is not None:
                if parent_tk not in deriv_cache:
                    deriv_cache[parent_tk] = compute_sorted_derivations(
                        parent_tk, mat_base, rules_by_head, Z_L_map, unique_triples.get(parent_tk, fb_mu / 0.9)
                    )
                pinfo = deriv_cache[parent_tk]
                deriv_cache[tk] = {
                    "vals": [0.9 * v for v in pinfo["vals"]],
                    "Z_L": 0.9 * pinfo["Z_L"],
                    "a1_viol": pinfo["a1_viol"],
                    "mu_exact": 0.9 * pinfo["mu_exact"],
                    "n_deriv": pinfo["n_deriv"],
                    "scale_from_parent": True,
                    "parent_tk": parent_tk,
                }
            else:
                deriv_cache[tk] = compute_sorted_derivations(tk, mat_base, rules_by_head, Z_L_map, fb_mu)
        else:
            if tk not in deriv_cache:
                deriv_cache[tk] = compute_sorted_derivations(tk, mat_base, rules_by_head, Z_L_map, fb_mu)

    # Encode all unique candidate texts cleanly on CPU
    processor, siglip_model = load_siglip2_cpu(DEFAULT_SIGLIP2_MODEL)
    siglip_model.to(device).eval()

    keys_list = list(unique_texts.keys())
    text_emb_map: Dict[Tuple[str, str, str], np.ndarray] = {}
    bs = 128
    for i in range(0, len(keys_list), bs):
        chunk_keys = keys_list[i : i + bs]
        texts = [unique_texts[k] for k in chunk_keys]
        inputs = processor(text=texts, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            txt_out = siglip_model.get_text_features(inputs["input_ids"])
            txt_vec = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out
            txt_vec = F.normalize(txt_vec, p=2, dim=-1)
            arr = txt_vec.cpu().numpy().astype(np.float32)
        for j, k in enumerate(chunk_keys):
            text_emb_map[k] = arr[j]
    LOGGER.info("Encoded %d candidate texts in %.1fs.", len(text_emb_map), time.time() - t_start)

    # Free SigLIP text model
    del siglip_model, processor

    # Helper to get mu vector over candidates for a clip under a stopping variant
    def get_clip_mu_variants(cands, mu_L, mode, fixed_k=0):
        lo_arr = np.zeros(len(cands), dtype=np.float32)
        up_arr = np.zeros(len(cands), dtype=np.float32)
        ex_arr = np.zeros(len(cands), dtype=np.float32)
        base_arr = np.zeros(len(cands), dtype=np.float32)
        any_trunc = False
        widths = []
        stops = []
        for idx, fc in enumerate(cands):
            tk = (fc["head"].lower().strip(), fc["relation"].lower().strip(), fc["tail"].lower().strip())
            base_arr[idx] = float(fc.get("mu_F", 0.5))
            dinfo = deriv_cache[tk]
            if dinfo.get("scale_from_parent", False):
                pinfo = deriv_cache[dinfo["parent_tk"]]
                plo, pup, pex, ns, tr = eval_stopping_interval(pinfo, mu_L, mode=mode, fixed_k=fixed_k)
                lo, up, ex = 0.9 * plo, 0.9 * pup, 0.9 * pex
            else:
                lo, up, ex, ns, tr = eval_stopping_interval(dinfo, mu_L, mode=mode, fixed_k=fixed_k)
            lo_arr[idx] = lo
            up_arr[idx] = up
            ex_arr[idx] = ex
            if tr:
                any_trunc = True
                widths.append(up - lo)
            stops.append(ns)
        mid_arr = 0.5 * (lo_arr + up_arr)
        return lo_arr, mid_arr, up_arr, ex_arr, base_arr, any_trunc, widths, stops

    def run_vcgat(cands, adj_mask_np, active_lm, v_vec, mu_vec):
        N = len(cands)
        if N == 0:
            return np.zeros((0,), dtype=np.float32)
        embs = np.stack([
            text_emb_map[(fc["head"].lower().strip(), fc["relation"].lower().strip(), fc["tail"].lower().strip())]
            for fc in cands
        ])
        h_t = torch.from_numpy(embs)
        v_t = torch.from_numpy(v_vec).unsqueeze(0)
        adj_t = torch.from_numpy(adj_mask_np)
        mu_edge_np = rebuild_mu_edge(cands, mu_vec, active_lm)
        mu_edge_t = torch.from_numpy(mu_edge_np)
        mu_f_t = torch.from_numpy(mu_vec)
        with torch.no_grad():
            logits, _, _ = vcgat(h_t, v_t, adj_t, mu_edge_t, mu_f_t)
        return logits.numpy().astype(np.float64)

    # Evaluate across all 500 clips
    variants = [
        ("Adaptive-LB", "adaptive", 0, "lo"),
        ("Adaptive-Mid", "adaptive", 0, "mid"),
        ("Adaptive-UB", "adaptive", 0, "up"),
        ("Fixed-K1-LB", "fixed", 1, "lo"),
        ("Fixed-K1-Mid", "fixed", 1, "mid"),
        ("Fixed-K2-LB", "fixed", 2, "lo"),
        ("Fixed-K2-Mid", "fixed", 2, "mid"),
        ("Fixed-K3-LB", "fixed", 3, "lo"),
        ("Fixed-K3-Mid", "fixed", 3, "mid"),
        ("Fixed-K5-LB", "fixed", 5, "lo"),
        ("Fixed-K5-Mid", "fixed", 5, "mid"),
        ("Static-Base-FKG", "base", 0, "base"),
    ]

    records = {v[0]: [] for v in variants}
    # Cache per-clip adaptive exact & interval stats
    adaptive_cache = {}
    clip_quota_margins = []

    for clip_idx, (s, active_lm, cands, adj_mask_np) in enumerate(clip_subgraphs):
        N = len(cands)
        if N == 0:
            continue
        mu_L = float(s.get("mu_L", 0.5))
        cert = s.get("certainty_level", "HIGH")
        quota = tuple(s.get("fact_quota", [2, 2, 1]))
        orig_idx = s["original_index"]
        v_vec = siglip2_embeddings[orig_idx]

        # Exact reference
        lo_ad, mid_ad, up_ad, ex_arr, base_arr, any_trunc_ad, widths_ad, _ = get_clip_mu_variants(
            cands, mu_L, mode="adaptive"
        )
        z_exact = run_vcgat(cands, adj_mask_np, active_lm, v_vec, ex_arr)
        sig_exact = 1.0 / (1.0 + np.exp(-z_exact))
        sel_exact, pools_exact, reg_exact = select_admitted_and_pools(
            cands, z_exact, ex_arr, mu_L, cert, quota
        )
        top1_exact = int(np.argmax(z_exact))

        # Check worst-case logit shift among {lo, mid, up} under Adaptive vs exact for HIGH/MEDIUM
        if cert in ("HIGH", "MEDIUM"):
            z_lo_ad = run_vcgat(cands, adj_mask_np, active_lm, v_vec, lo_ad)
            z_mid_ad = run_vcgat(cands, adj_mask_np, active_lm, v_vec, mid_ad)
            z_up_ad = run_vcgat(cands, adj_mask_np, active_lm, v_vec, up_ad)

            max_dz_endpoints = float(
                max(
                    np.max(np.abs(z_lo_ad - z_exact)),
                    np.max(np.abs(z_mid_ad - z_exact)),
                    np.max(np.abs(z_up_ad - z_exact)),
                    np.max(np.abs(z_up_ad - z_lo_ad)) / 2.0,
                )
            )
            # Compute intra-stratum quota margins at exact / mid
            min_stratum_margin = float("inf")
            for p_name, q_k in zip(("vis", "dir", "inf"), quota):
                pool = pools_exact[p_name]
                if len(pool) > q_k and q_k > 0:
                    gap = pool[q_k - 1][1] - pool[q_k][1]
                    if gap < min_stratum_margin:
                        min_stratum_margin = gap
            clip_quota_margins.append({
                "clip_idx": clip_idx,
                "cert": cert,
                "any_trunc_ad": any_trunc_ad,
                "max_width_ad": float(max(widths_ad)) if widths_ad else 0.0,
                "max_dz_endpoints": max_dz_endpoints,
                "min_stratum_margin": min_stratum_margin if math.isfinite(min_stratum_margin) else None,
            })

            adaptive_cache[clip_idx] = (z_lo_ad, z_mid_ad, z_up_ad, lo_ad, mid_ad, up_ad, any_trunc_ad)

        # Evaluate each variant
        for vname, mode, fk, pick in variants:
            if mode == "adaptive" and cert in ("HIGH", "MEDIUM"):
                z_lo_ad, z_mid_ad, z_up_ad, lo_v, mid_v, up_v, any_trunc = adaptive_cache[clip_idx]
                if pick == "lo":
                    mu_use, z_use = lo_v, z_lo_ad
                elif pick == "mid":
                    mu_use, z_use = mid_v, z_mid_ad
                else:
                    mu_use, z_use = up_v, z_up_ad
                widths_v = (up_v - lo_v)
            elif mode == "base":
                mu_use = base_arr
                z_use = run_vcgat(cands, adj_mask_np, active_lm, v_vec, mu_use)
                any_trunc = bool(np.max(np.abs(mu_use - ex_arr)) > 1e-9)
                widths_v = np.abs(mu_use - ex_arr)
            else:
                lo_v, mid_v, up_v, _, _, any_trunc, wlist, _ = get_clip_mu_variants(
                    cands, mu_L, mode=("adaptive" if mode == "adaptive" else "fixed"), fixed_k=fk
                )
                mu_use = lo_v if pick == "lo" else (mid_v if pick == "mid" else up_v)
                z_use = run_vcgat(cands, adj_mask_np, active_lm, v_vec, mu_use)
                widths_v = up_v - lo_v

            sig_use = 1.0 / (1.0 + np.exp(-z_use))
            sel_use, pools_use, reg_use = select_admitted_and_pools(
                cands, z_use, mu_use, mu_L, cert, quota
            )

            set_ex = set(sel_exact)
            set_us = set(sel_use)
            inter = len(set_ex & set_us)
            union = len(set_ex | set_us)
            jacc = (inter / union) if union > 0 else 1.0
            exact_set_match = 1 if set_ex == set_us else 0
            exact_ordered_match = 1 if sel_exact == sel_use else 0
            exact_set_and_reg = 1 if (set_ex == set_us and all(reg_exact.get(k) == reg_use.get(k) for k in set_ex)) else 0

            # Ranking stats (meaningful for HIGH/MEDIUM where VC-GAT ranks)
            if N >= 2 and np.std(z_exact) > 0 and np.std(z_use) > 0:
                sp_rho, _ = stats.spearmanr(z_exact, z_use)
                kd_tau, _ = stats.kendalltau(z_exact, z_use)
                if math.isnan(sp_rho):
                    sp_rho = 1.0
                if math.isnan(kd_tau):
                    kd_tau = 1.0
            else:
                sp_rho, kd_tau = 1.0, 1.0

            top1_flip = 1 if int(np.argmax(z_use)) != top1_exact else 0
            # Per-stratum top1 flips
            stratum_top1_flip = 0
            for p_name in ("vis", "dir", "inf"):
                pe = pools_exact[p_name]
                pu = pools_use[p_name]
                if len(pe) >= 1 and len(pu) >= 1 and pe[0][0] != pu[0][0]:
                    stratum_top1_flip = 1

            records[vname].append({
                "clip_idx": clip_idx,
                "cert": cert,
                "any_trunc": int(any_trunc),
                "n_cands": N,
                "max_dmu": float(np.max(np.abs(mu_use - ex_arr))),
                "mean_dmu": float(np.mean(np.abs(mu_use - ex_arr))),
                "max_dz": float(np.max(np.abs(z_use - z_exact))),
                "mean_dz": float(np.mean(np.abs(z_use - z_exact))),
                "max_dsig": float(np.max(np.abs(sig_use - sig_exact))),
                "spearman": float(sp_rho),
                "kendall": float(kd_tau),
                "top1_flip": top1_flip,
                "stratum_top1_flip": stratum_top1_flip,
                "exact_set_match": exact_set_match,
                "exact_ordered_match": exact_ordered_match,
                "exact_set_and_reg": exact_set_and_reg,
                "jaccard": float(jacc),
            })

    # Summarize results
    summary_rows = []
    for vname, _, _, _ in variants:
        df_v = pd.DataFrame(records[vname])
        df_hm = df_v[df_v["cert"].isin(["HIGH", "MEDIUM"])]
        df_tr = df_v[df_v["any_trunc"] == 1]

        row = {
            "variant": vname,
            "n_clips": len(df_v),
            "n_truncated_clips": int(df_v["any_trunc"].sum()),
            "max_dmu": round(float(df_v["max_dmu"].max()), 6),
            "max_dz": round(float(df_hm["max_dz"].max()), 6),
            "mean_dz_trunc": round(float(df_tr["mean_dz"].mean()), 6) if len(df_tr) else 0.0,
            "max_dsig": round(float(df_hm["max_dsig"].max()), 6),
            "spearman_mean": round(float(df_hm["spearman"].mean()), 6),
            "spearman_min": round(float(df_hm["spearman"].min()), 6),
            "kendall_mean": round(float(df_hm["kendall"].mean()), 6),
            "kendall_min": round(float(df_hm["kendall"].min()), 6),
            "top1_flip_pct": round(float(df_hm["top1_flip"].mean() * 100.0), 2),
            "stratum_top1_flip_pct": round(float(df_hm["stratum_top1_flip"].mean() * 100.0), 2),
            "set_match_all_pct": round(float(df_v["exact_set_match"].mean() * 100.0), 2),
            "set_match_hm_pct": round(float(df_hm["exact_set_match"].mean() * 100.0), 2),
            "set_match_trunc_pct": round(float(df_tr["exact_set_match"].mean() * 100.0), 2) if len(df_tr) else 100.0,
            "set_and_reg_match_all_pct": round(float(df_v["exact_set_and_reg"].mean() * 100.0), 2),
            "jaccard_all_mean": round(float(df_v["jaccard"].mean()), 6),
            "jaccard_trunc_mean": round(float(df_tr["jaccard"].mean()), 6) if len(df_tr) else 1.0,
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    csv_out = RESULTS_DIR / "vcgat_early_stop_stability.csv"
    summary_df.to_csv(csv_out, index=False)

    # Compute joint Selection-Stability Margin statistics on Adaptive Stopping
    df_m = pd.DataFrame(clip_quota_margins)
    valid_margins = df_m[df_m["min_stratum_margin"].notna()].copy()
    certified_margin = (valid_margins["min_stratum_margin"] > 2.0 * valid_margins["max_dz_endpoints"]).mean() * 100.0
    max_dz_observed_ad = float(df_m["max_dz_endpoints"].max())
    min_gap_observed = float(valid_margins["min_stratum_margin"].min())
    median_gap_observed = float(valid_margins["min_stratum_margin"].median())

    out_json = {
        "n_clips_total": len(clip_subgraphs),
        "n_unique_triples": len(unique_triples),
        "n_multi_deriv_triples": sum(1 for v in deriv_cache.values() if v["n_deriv"] >= 2),
        "n_a1_viol_triples": sum(1 for v in deriv_cache.values() if v["a1_viol"]),
        "margin_diagnostics": {
            "n_hm_with_quota_boundary": len(valid_margins),
            "max_dz_adaptive_endpoints": max_dz_observed_ad,
            "min_quota_gap": min_gap_observed,
            "median_quota_gap": median_gap_observed,
            "pct_strict_endpoint_margin_certified": round(float(certified_margin), 2),
            "n_below_endpoint_margin": int((valid_margins["min_stratum_margin"] <= 2.0 * valid_margins["max_dz_endpoints"]).sum()),
        },
        "summary": summary_rows,
    }
    json_out = RESULTS_DIR / "vcgat_early_stop_stability.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    LOGGER.info("Completed in %.1fs. Summary:\n%s", time.time() - t_start, summary_df.to_string(index=False))
    LOGGER.info("Margin diagnostics: %s", json.dumps(out_json["margin_diagnostics"], indent=2))


if __name__ == "__main__":
    main()
