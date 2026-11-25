#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/hcmc_14_experiment_x5_300.py
=============================================================================
Experiment X5: Cross-Region and Unseen-Landmark Generalization (Hanoi -> HCMC)
Canonical 300-Clip Balanced Benchmark with Restored Original 285-Concept Bank
=============================================================================
Scientific Guarantees:
1. Canonical Unseen Dataset (N = 300 = 60 x 5 landmarks, G = 26 YouTube videos):
   Rebuilt strictly using the OLD 285-concept bank (`concepts_285.json` /
   `prototypes_285x768.pt`) and duration-stratified quotas from Section 6.8.
2. Zero Cross-Region Reasoning/Generation Leakage:
   - Deep VC-GAT (`best_scientific_deep_vc_gat.pt`), Qwen2.5-3B LoRA adapters
     (`experiments/checkpoints/`), and Hanoi-149 thresholds (`THETA_M=0.50`,
     `THETA_H=0.80`, `T=0.5`) remain 100% FROZEN from Hanoi.
   - Zero HCMC clips touch reasoning, VC-GAT, thresholds, or LoRA training.
3. Strict Protocol Separation:
   - Protocol A-OracleHigh (Oracle gt_parent, mu_L=0.95 HIGH): pure transfer
     of HCMC URI-safe KG + Horn Rules + Deep VC-GAT + LoRA without visual
     grounding attenuation. NEVER reports Top-1=100% as a model score.
   - Protocol A-ZeroShotVis (Oracle gt_parent, mu_L=P_Z4_5way[gt]): tests
     whether certainty control responsibly abstains under frozen zero-shot
     visual uncertainty.
   - Protocol B-ZeroShot (35-way `Z4_MultiView_OOD_Center`, 0 HCMC train clips):
     full end-to-end zero-shot pipeline including open-set grounding errors.
   - Protocol B-Adaptation (35-way `SuperEnsemble_T05` 3-fold OOF on Raw_300):
     explicitly flagged *Cross-Region Grounding Adaptation (NOT zero-shot)*,
     with VC-GAT, HCMC KG, thresholds, and LoRA remaining 100% frozen.
4. Full 6 Metric Groups + Statistical Reliability:
   - Group 1: Landmark Grounding (Top-1, Top-5, MRR, ECE, ConfHanoi)
   - Group 2: Knowledge Grounding (K-Recall + dir/inf/vis, VSP [Sec.6.8, Uniform, SG])
   - Group 3: Factual Reliability (EAP, Factual Precision, OAR_fact, strict audit)
   - Group 4: Linguistic Policy (Policy Adherence, OAR_gen, BLEU-4, ROUGE-L)
   - Group 5: Reasoning Efficiency on Pure HCMC KB (Certification Rate, Agreement, Latency)
   - Group 6: Generalization Gap vs Hanoi (500), Per-Landmark (60x5), Macro-Avg,
     Clean (154) vs Noisy (146), Clip & Hierarchical Video-Clustered B=10,000 CIs.
=============================================================================
"""

import gc
import itertools
import json
import logging
import math
import os
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from peft import PeftModel
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("ExpX5_HCMC300")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.config import LANDMARK_DESCRIPTIVE_PROMPTS, PARENT_CLASSES_30
from experiments.evaluate_table4_on_500_test import compute_ece, evaluate_predictions_from_probs
from experiments.hcmc_03_frozen_calib_and_zeroshot import HCMC_CLASSES_5, THETA_H, THETA_M
from experiments.hcmc_08_audit_dual_judge_and_pack import build_fkg_years_map, find_mentioned_landmarks
from experiments.hcmc_11_unified_concept_grounding import (
    compute_fkg_grounding_matrix,
    compute_multiprompt_max,
    load_stop_filtered_uri_fkg,
    softmax_probs,
)
from experiments.hcmc_13_c285_noise_mix_grounding import (
    HANOI_TEST_500_JSONL,
    HANOI_TEST_DINO,
    HANOI_TEST_MANIFEST,
    HANOI_TEST_SIG,
    HANOI_TRAIN_COCO,
    HANOI_TRAIN_DINO,
    HANOI_TRAIN_SIG6,
    OPEN35,
    _calibrate_probs,
    _inner_cv_temp_and_weights,
    _l2_norm,
    _make_class_weights,
    train_mlp_seeds,
)
from experiments.hcmc_uri_fkg_utils import (
    DIRECT_RELS,
    INFERRED_RELS,
    VISUAL_RELS,
    build_uri_safe_subgraph,
    is_junk_surface,
    precompute_uri_safe_subgraphs,
    select_uri_safe_quota_facts,
)
from experiments.run_comprehensive_1119_benchmark import (
    ASSERTIVE_MARKERS,
    HEDGING_MARKERS,
    SYSTEM_PROMPT,
    UNCERTAIN_MARKERS,
    compute_bleu4,
    compute_rouge_l,
    match_fact,
)
from experiments.run_m0_export_facts import compute_fact_derivations
from experiments.vcgat_semantic_anchoring_engine import VCGATSemanticAnchoringEngine
from experiments.visual_anchoring_engine import (
    VISUAL_TESTABLE_RELATIONS,
    format_fact_claim_sentence,
    is_entity_in_scene,
    parse_scene_graph,
)

# =============================================================================
# PATHS & FROZEN CONSTANTS
# =============================================================================

HCMC_DIR = REPO_ROOT / "data" / "hcmc_benchmark"
HCMC_FKG_DIR = REPO_ROOT / "data" / "fkg_per_kb_with_hcmc"
MANIFEST_JSON = REPO_ROOT / "data" / "VIDEO_TEST_HCMC" / "hcmc_segments_manifest.json"
BAL300_IDX_JSON = HCMC_DIR / "hcmc_balanced_300_indices.json"
VLM_CURATION_JSON = HCMC_DIR / "hcmc_vlm_curation_scores_1161.json"
GOLD_READY_JSONL = HCMC_DIR / "hcmc_balanced_300_gold_ready.jsonl"
FEATS_1161_NPZ = HCMC_DIR / "hcmc_1161_c285_grounding_feats.npz"
CONCEPTS_285_JSON = HCMC_DIR / "concepts_285.json"
FROZEN_CFG_JSON = HCMC_DIR / "frozen_unified_grounding_cfg.json"
PROMPTS_HCMC_JSON = HCMC_DIR / "hcmc_landmarks_knowledge_and_prompts.json"

HANOI_VAL_JSONL = REPO_ROOT / "data" / "caption_dataset" / "generated_captions_val_vcgat.jsonl"
HANOI_VAL_EMB_PKL = REPO_ROOT / "experiments" / "features" / "siglip2_train_val_embeddings.pkl"
HANOI_TRAIN_CONCEPTS = REPO_ROOT / "data" / "unified_coco_dataset" / "features" / "train_concepts_285d.npy"
HANOI_TEST_CONCEPTS = REPO_ROOT / "data" / "unified_coco_dataset" / "features" / "test_concepts_285d.npy"

CHECKPOINTS_BASE = REPO_ROOT / "experiments" / "checkpoints"
CKPT_CONTINUOUS = CHECKPOINTS_BASE / "qwen25_3b_lora_continuous"
CKPT_WO_FUSION = CHECKPOINTS_BASE / "component_ablation" / "wo_uncertainty_fusion"

OUT_DATASET_JSONL = HCMC_DIR / "hcmc_300_x5_dataset.jsonl"
OUT_CAPTIONS_JSON = HCMC_DIR / "hcmc_300_x5_captions.json"
OUT_REPORT_JSON = HCMC_DIR / "hcmc_experiment_x5_300_report.json"

HCMC_KB_DIRS = [
    "ba_thien_hau_temple",
    "ben_thanh_market",
    "independence_palace",
    "saigon_central_post_office",
    "saigon_opera_house",
    "ho_chi_minh_city",
]

# Hanoi (500) Reference Baseline (Paper Table 4 & comprehensive_500_balanced_scientific_vcgat_benchmark_report.json)
HANOI_500_REF = {
    "k_recall": 51.84,
    "vsp": 96.89,
    "fact_p_audited": 97.10,
    "fact_p_rule": 100.00,
    "oar_fact_audited": 2.90,
    "oar_fact_rule": 0.00,
    "policy_adherence": 89.40,
    "oar_gen": 0.00,
    "eap": 100.00,
    "cert_rate": 97.10,
    "decision_agreement": 100.00,
    "latency_adapt_ms_clip": 1.42,
    "latency_exhaust_ms_clip": 21.65,
}

B_BOOT = 10000
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_qwen3b_path() -> str:
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct"))
    if cache_dir.is_dir():
        snaps = sorted(cache_dir.glob("snapshots/*/"))
        if snaps:
            return str(snaps[0])
    return "Qwen/Qwen2.5-3B-Instruct"


def get_duration_tier_and_quota(dur: float) -> Tuple[str, Tuple[int, int, int], str, str, int, int, int]:
    if dur < 7.5:
        return "5s", (1, 1, 1), "2–3 sentence", "30–50 words", 3, 50, 90
    elif dur < 15.0:
        return "10s", (2, 2, 1), "3–4 sentence", "40–70 words", 4, 70, 130
    elif dur < 25.0:
        return "20s", (3, 3, 2), "4–6 sentence", "60–100 words", 6, 100, 180
    else:
        return "30s", (4, 4, 3), "6–8 sentence", "90–140 words", 8, 140, 250


def format_observed_scene_graph_c285(
    c285_vec: np.ndarray,
    concept_names: List[str],
    top_k: int = 10,
    min_score: float = 0.05,
) -> Tuple[str, List[Dict[str, Any]]]:
    order = np.argsort(-c285_vec)[:top_k]
    lines: List[str] = []
    top_items: List[Dict[str, Any]] = []
    for ci in order:
        sc = float(c285_vec[ci])
        if sc < min_score:
            continue
        raw_name = concept_names[int(ci)]
        clean_name = raw_name.replace("bg_", "").replace("_", " ").strip()
        lines.append(f"- {clean_name} ({sc:.2f})")
        top_items.append({"concept": clean_name, "concept_id": raw_name, "score": round(sc, 4)})
    if not lines:
        lines = ["- historical heritage site in landscaped setting (0.85)"]
        top_items = [{"concept": "historical heritage site in landscaped setting", "concept_id": "fallback", "score": 0.85}]
    return "\n".join(lines), top_items


def get_regime(val: float) -> str:
    if val >= THETA_H:
        return "HIGH"
    elif val >= THETA_M:
        return "MEDIUM"
    return "LOW"


# =============================================================================
# MODULE 1: GROUP 5 REASONING EFFICIENCY ON PURE HCMC KB
# =============================================================================

def _build_pure_hcmc_horn_closure() -> Tuple[Dict[Tuple[str, str, str], float], Dict[Tuple[str, str, str], List[Any]], Dict[Tuple[str, str, str], float]]:
    base_facts: Dict[Tuple[str, str, str], float] = {}
    for kb in HCMC_KB_DIRS:
        p = HCMC_FKG_DIR / kb / "fkg_facts.json"
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        for m in d.get("multi_valued_facts", []):
            h = (m.get("head_surface") or "").strip()
            r = (m.get("relation") or "").strip()
            t = (m.get("tail_surface") or "").strip()
            w = float(m.get("mu_F", 0.5))
            if h and r and t and not is_junk_surface(h) and not is_junk_surface(t) and h.lower() != t.lower():
                k = (h.lower(), r.lower(), t.lower())
                base_facts[k] = max(base_facts.get(k, 0.0), w)
        for g in d.get("single_valued_groups", []):
            h = (g.get("head_surface") or "").strip()
            r = (g.get("relation") or "").strip()
            t = (g.get("winner_surface") or "").strip()
            w = float(g.get("winner_bel", 0.8))
            if h and r and t and not is_junk_surface(h) and not is_junk_surface(t) and h.lower() != t.lower():
                k = (h.lower(), r.lower(), t.lower())
                base_facts[k] = max(base_facts.get(k, 0.0), w)

    sym_rules = {
        "adjacent_to": 1.00,
        "near_to": 0.95,
        "near": 0.95,
        "bordering": 1.00,
        "associated_with": 0.90,
    }
    mat_base = dict(base_facts)
    for (h, r, t), w in list(base_facts.items()):
        if r in sym_rules:
            inv_k = (t, r, h)
            mat_base[inv_k] = max(mat_base.get(inv_k, 0.0), w * sym_rules[r])
    for (h, r, t), w in list(mat_base.items()):
        if r == "part_of" and (t, "has_part", h) not in mat_base:
            mat_base[(t, "has_part", h)] = w
        elif r == "has_part" and (t, "part_of", h) not in mat_base:
            mat_base[(t, "part_of", h)] = w

    head_rel_tails: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (h, r, t), w in mat_base.items():
        head_rel_tails[h][r][t] = w

    rules_by_head: Dict[Tuple[str, str, str], List[Any]] = defaultdict(list)
    for x in head_rel_tails:
        for y in head_rel_tails[x].get("located_in", {}):
            for z in head_rel_tails.get(y, {}).get("located_in", {}):
                if x != z:
                    rules_by_head[(x, "located_in", z)].append(("r5", ((x, "located_in", y), (y, "located_in", z)), 0.98))
        for y in head_rel_tails[x].get("has_part", {}):
            if x == y:
                continue
            for z in head_rel_tails[x].get("located_in", {}):
                if y != z:
                    rules_by_head[(y, "located_in", z)].append(("r6", ((x, "has_part", y), (x, "located_in", z)), 0.95))
            for z in head_rel_tails[x].get("belongs_to_dynasty", {}):
                rules_by_head[(y, "belongs_to_dynasty", z)].append(("r7", ((x, "has_part", y), (x, "belongs_to_dynasty", z)), 0.85))
            for z in head_rel_tails[x].get("architectural_style", {}):
                rules_by_head[(y, "architectural_style", z)].append(("r8", ((x, "has_part", y), (x, "architectural_style", z)), 0.80))
            for z in head_rel_tails[x].get("dedicated_to", {}):
                rules_by_head[(y, "worships", z)].append(("r9", ((x, "has_part", y), (x, "dedicated_to", z)), 0.88))
            for z in head_rel_tails[x].get("worships", {}):
                rules_by_head[(y, "worships", z)].append(("r9", ((x, "has_part", y), (x, "worships", z)), 0.88))
            for z in head_rel_tails[x].get("material", {}):
                rules_by_head[(y, "material", z)].append(("r10", ((x, "has_part", y), (x, "material", z)), 0.70))
        for y in head_rel_tails[x].get("commemorates", {}):
            for z in head_rel_tails.get(y, {}).get("ruled_by", {}):
                rules_by_head[(x, "belongs_to_dynasty", z)].append(("r11", ((x, "commemorates", y), (y, "ruled_by", z)), 0.90))
        for y in head_rel_tails[x].get("built_by", {}):
            for z in head_rel_tails.get(y, {}).get("ruled_by", {}):
                rules_by_head[(x, "belongs_to_dynasty", z)].append(("r12", ((x, "built_by", y), (y, "ruled_by", z)), 0.92))

    GAMMA_R = 0.90
    all_facts = set(mat_base.keys()) | set(rules_by_head.keys())
    Z_prev = {f: mat_base.get(f, 0.0) for f in all_facts}
    for _ in range(3):
        Z = {f: mat_base.get(f, 0.0) for f in all_facts}
        for head, r_list in rules_by_head.items():
            for _, body, r_conf in r_list:
                term = GAMMA_R * r_conf
                for b in body:
                    term *= Z_prev.get(b, 0.0)
                Z[head] += term
        Z_prev = Z
    return len(base_facts), mat_base, rules_by_head, Z_prev


def _extract_sorted_deriv_vals(
    fact: Tuple[str, str, str],
    mat_base: Dict[Tuple[str, str, str], float],
    rules_by_head: Dict[Tuple[str, str, str], List[Any]],
    max_depth: int = 3,
) -> List[float]:
    memo = {}

    def dfs(f, d, anc):
        st = (f, d, tuple(sorted(anc)))
        if st in memo:
            return memo[st]
        trees = []
        if f in mat_base:
            trees.append((mat_base[f], frozenset([f]), False))
        if d < max_depth:
            for rid, body, r_conf in rules_by_head.get(f, []):
                if any(b in anc for b in body):
                    continue
                n_anc = anc | {f}
                bts = [dfs(b, d + 1, n_anc) for b in body]
                if any(len(bt) == 0 for bt in bts):
                    continue
                for combo in itertools.product(*bts):
                    val = 0.90 * r_conf
                    comps = {f"{rid}_{body}->{f}"}
                    viol = any(bt[2] for bt in combo)
                    for bv, bc, _ in combo:
                        val *= bv
                        if not comps.isdisjoint(bc):
                            viol = True
                        comps.update(bc)
                    trees.append((val, frozenset(comps), viol))
        memo[st] = trees
        return trees

    trees = dfs(fact, 0, set())
    can_d: Dict[frozenset, float] = {}
    for val, comps, viol in trees:
        if not viol and (comps not in can_d or val > can_d[comps]):
            can_d[comps] = val
    return sorted(can_d.values(), reverse=True)


def run_group5_efficiency_benchmark(
    mu_L_adapt_300: np.ndarray,
    vcgat_engine: VCGATSemanticAnchoringEngine,
    sg35: Dict[str, Dict[str, Any]],
    gts_300: List[str],
    sig300: np.ndarray,
) -> Dict[str, Any]:
    LOGGER.info("=== [Module 1 / Group 5] Pure HCMC Horn Reasoning & Truncation Efficiency ===")
    n_base_facts, mat_base, rules_by_head, Z_L_map = _build_pure_hcmc_horn_closure()

    # Collect clean A1=0 multi-derivation facts in pure HCMC KG
    fact_cache: List[Tuple[Tuple[str, str, str], List[float], float]] = []
    for f in sorted(rules_by_head.keys()):
        vals = _extract_sorted_deriv_vals(f, mat_base, rules_by_head, max_depth=3)
        if len(vals) >= 2:
            fact_cache.append((f, vals, float(Z_L_map.get(f, mat_base.get(f, 0.0)))))

    N_facts = len(fact_cache)
    LOGGER.info("Pure HCMC KB: mat_base=%d, rules_by_head=%d, multi-deriv(A1=0)=%d",
                len(mat_base), len(rules_by_head), N_facts)

    # Sample clip visual confidences cyclically across the 300 HCMC clips (deterministic, same distribution as E2E)
    rng = np.random.RandomState(SEED)
    mu_Ls = mu_L_adapt_300[rng.randint(0, len(mu_L_adapt_300), size=N_facts)]

    # Precompute true exhaustive mu_F and regimes
    true_muFs = []
    true_regimes = []
    for (_, vals, _), mu_l in zip(fact_cache, mu_Ls):
        muf = 1.0 - math.prod(1.0 - min(0.999999, v) for v in vals)
        true_muFs.append(muf)
        true_regimes.append(get_regime(float(mu_l) * muf))

    def _dfs_horn_budgeted(
        fact: Tuple[str, str, str],
        max_derivs: Optional[int] = None,
        max_depth: int = 3,
    ) -> List[float]:
        can_d: Dict[frozenset, float] = {}

        def dfs(f, d, anc):
            if max_derivs is not None and d == 0 and len(can_d) >= max_derivs:
                return []
            trees = []
            if f in mat_base:
                fc = frozenset([f])
                trees.append((mat_base[f], fc, False))
                if d == 0:
                    if fc not in can_d or mat_base[f] > can_d[fc]:
                        can_d[fc] = mat_base[f]
                    if max_derivs is not None and len(can_d) >= max_derivs:
                        return trees
            if d < max_depth:
                for rid, body, r_conf in rules_by_head.get(f, []):
                    if any(b in anc for b in body):
                        continue
                    n_anc = anc | {f}
                    bts = [dfs(b, d + 1, n_anc) for b in body]
                    if any(len(bt) == 0 for bt in bts):
                        continue
                    for combo in itertools.product(*bts):
                        val = 0.90 * r_conf
                        comps = {f"{rid}_{body}->{f}"}
                        viol = any(bt[2] for bt in combo)
                        for bv, bc, _ in combo:
                            val *= bv
                            if not comps.isdisjoint(bc):
                                viol = True
                            comps.update(bc)
                        fc = frozenset(comps)
                        trees.append((val, fc, viol))
                        if d == 0 and not viol:
                            if fc not in can_d or val > can_d[fc]:
                                can_d[fc] = val
                            if max_derivs is not None and len(can_d) >= max_derivs:
                                return trees
            return trees

        dfs(fact, 0, set())
        return sorted(can_d.values(), reverse=True)

    def _dfs_horn_adaptive(
        fact: Tuple[str, str, str],
        Z_L: float,
        mu_l: float,
        max_depth: int = 3,
    ) -> Tuple[List[float], bool]:
        can_d: Dict[frozenset, float] = {}
        stopped = [False]

        def try_emit(val: float, fc: frozenset, viol: bool) -> bool:
            if viol or (fc in can_d and val <= can_d[fc]):
                return False
            can_d[fc] = val
            vals = sorted(can_d.values(), reverse=True)
            nu_n = 1.0 - math.prod(1.0 - min(0.999999, v) for v in vals)
            T_n = max(0.0, Z_L - sum(vals))
            up_n = min(1.0, nu_n + (1.0 - nu_n) * min(1.0, T_n))
            if get_regime(mu_l * nu_n) == get_regime(mu_l * up_n):
                stopped[0] = True
                return True
            return False

        def dfs(f, d, anc):
            if d == 0 and stopped[0]:
                return []
            trees = []
            if f in mat_base:
                fc = frozenset([f])
                trees.append((mat_base[f], fc, False))
                if d == 0 and try_emit(mat_base[f], fc, False):
                    return trees
            if d < max_depth:
                for rid, body, r_conf in rules_by_head.get(f, []):
                    if d == 0 and stopped[0]:
                        return trees
                    if any(b in anc for b in body):
                        continue
                    n_anc = anc | {f}
                    bts = [dfs(b, d + 1, n_anc) for b in body]
                    if any(len(bt) == 0 for bt in bts):
                        continue
                    for combo in itertools.product(*bts):
                        val = 0.90 * r_conf
                        comps = {f"{rid}_{body}->{f}"}
                        viol = any(bt[2] for bt in combo)
                        for bv, bc, _ in combo:
                            val *= bv
                            if not comps.isdisjoint(bc):
                                viol = True
                            comps.update(bc)
                        fc = frozenset(comps)
                        trees.append((val, fc, viol))
                        if d == 0 and try_emit(val, fc, viol):
                            return trees
            return trees

        dfs(fact, 0, set())
        return sorted(can_d.values(), reverse=True), stopped[0]

    N_REP_DFS = 3
    t0 = time.perf_counter()
    for _ in range(N_REP_DFS):
        for fact_t, _, _ in fact_cache:
            vals_ex = _dfs_horn_budgeted(fact_t, max_derivs=None)
            _ = 1.0 - math.prod(1.0 - min(0.999999, v) for v in vals_ex)
    t_exhaust_ms = (time.perf_counter() - t0) / (N_REP_DFS * max(1, N_facts)) * 1000.0

    k_table = []
    for K in [1, 2, 3, 5, 10, 20]:
        certs, agrees, errs, widths = 0, 0, [], []
        t0 = time.perf_counter()
        for rep_k in range(N_REP_DFS):
            c_r, a_r = 0, 0
            for (fact_t, vals, Z_L), mu_l, tr_reg, tr_muf in zip(fact_cache, mu_Ls, true_regimes, true_muFs):
                _ = _dfs_horn_budgeted(fact_t, max_derivs=K)
                S_k = vals[: min(K, len(vals))]
                nu_s = 1.0 - math.prod(1.0 - min(0.999999, v) for v in S_k)
                T_s = max(0.0, Z_L - sum(S_k))
                up_s = min(1.0, nu_s + (1.0 - nu_s) * min(1.0, T_s))
                reg_lo = get_regime(float(mu_l) * nu_s)
                reg_up = get_regime(float(mu_l) * up_s)
                if reg_lo == reg_up:
                    c_r += 1
                if reg_lo == tr_reg:
                    a_r += 1
                if rep_k == 0:
                    errs.append(max(0.0, tr_muf - nu_s))
                    widths.append(max(0.0, up_s - nu_s))
            certs, agrees = c_r, a_r
        t_k_ms = (time.perf_counter() - t0) / (N_REP_DFS * max(1, N_facts)) * 1000.0
        k_table.append({
            "mechanism": f"Fixed K={K}",
            "K": K,
            "certification_rate_pct": round(certs / N_facts * 100.0, 2),
            "decision_agreement_pct": round(agrees / N_facts * 100.0, 2),
            "mean_error": round(float(np.mean(errs)), 5),
            "mean_width": round(float(np.mean(widths)), 5),
            "latency_ms_fact": round(t_k_ms, 5),
            "speedup_vs_exhaust": round(t_exhaust_ms / t_k_ms if t_k_ms > 0 else 1.0, 2),
        })

    # Adaptive Stopping (Corollary 1)
    certs_ad, agrees_ad, stops_ad, errs_ad, widths_ad = 0, 0, [], [], []
    t0 = time.perf_counter()
    for rep_ad in range(N_REP_DFS):
        c_r, a_r = 0, 0
        for (fact_t, vals, Z_L), mu_l, tr_reg, tr_muf in zip(fact_cache, mu_Ls, true_regimes, true_muFs):
            _ = _dfs_horn_adaptive(fact_t, Z_L, float(mu_l))
            M = len(vals)
            stop_n = M
            nu_final, up_final = 0.0, 1.0
            for n in range(1, M + 1):
                S_n = vals[:n]
                nu_n = 1.0 - math.prod(1.0 - min(0.999999, v) for v in S_n)
                T_n = max(0.0, Z_L - sum(S_n))
                up_n = min(1.0, nu_n + (1.0 - nu_n) * min(1.0, T_n))
                reg_lo = get_regime(float(mu_l) * nu_n)
                reg_up = get_regime(float(mu_l) * up_n)
                nu_final, up_final = nu_n, up_n
                if reg_lo == reg_up:
                    stop_n = n
                    c_r += 1
                    break
            if get_regime(float(mu_l) * nu_final) == tr_reg:
                a_r += 1
            if rep_ad == 0:
                stops_ad.append(stop_n)
                errs_ad.append(max(0.0, tr_muf - nu_final))
                widths_ad.append(max(0.0, up_final - nu_final))
        certs_ad, agrees_ad = c_r, a_r
    t_adapt_ms = (time.perf_counter() - t0) / (N_REP_DFS * max(1, N_facts)) * 1000.0

    k_table.append({
        "mechanism": "Adaptive Stopping (Cor. 1)",
        "K": "Adaptive",
        "certification_rate_pct": round(certs_ad / N_facts * 100.0, 2),
        "decision_agreement_pct": round(agrees_ad / N_facts * 100.0, 2),
        "median_stop_k": float(np.median(stops_ad)),
        "mean_stop_k": round(float(np.mean(stops_ad)), 2),
        "mean_error": round(float(np.mean(errs_ad)), 5),
        "mean_width": round(float(np.mean(widths_ad)), 5),
        "latency_ms_fact": round(t_adapt_ms, 5),
        "speedup_vs_exhaust": round(t_exhaust_ms / t_adapt_ms if t_adapt_ms > 0 else 1.0, 2),
    })
    k_table.append({
        "mechanism": "Exhaustive (K=Inf)",
        "K": "Inf",
        "certification_rate_pct": 100.00,
        "decision_agreement_pct": 100.00,
        "mean_error": 0.0,
        "mean_width": 0.0,
        "latency_ms_fact": round(t_exhaust_ms, 5),
        "speedup_vs_exhaust": 1.00,
    })

    # Per-clip end-to-end reasoning + VC-GAT profiling over the 300 clips
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(300):
        sub = sg35[gts_300[i]]
        v_t = torch.from_numpy(sig300[i].astype(np.float32)).unsqueeze(0).to(vcgat_engine.device)
        with torch.no_grad():
            _ = vcgat_engine.vc_gat(sub["K_embs_t"], v_t, sub["adj_t"], sub["mu_edge_t"], sub["mu_f_t"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    vcgat_ms_clip = (time.perf_counter() - t0) / 300.0 * 1000.0

    # Clip-level symbolic DFS reasoning time (all active inferred + fallback multi-deriv facts per clip)
    clip_fact_keys = []
    for i in range(300):
        lm = gts_300[i]
        sub_cands = sg35[lm]["cands"]
        f_items = []
        seen_t = set()
        for c in sub_cands:
            f_key = (c["head"].lower().strip(), c["relation"].lower().strip(), c["tail"].lower().strip())
            if f_key in rules_by_head and f_key not in seen_t:
                seen_t.add(f_key)
                f_items.append((f_key, float(Z_L_map.get(f_key, mat_base.get(f_key, 0.0)))))
        if len(f_items) < 5:
            for ft, _, zl in fact_cache[: max(5, 10 - len(f_items))]:
                if ft not in seen_t:
                    seen_t.add(ft)
                    f_items.append((ft, zl))
        clip_fact_keys.append(f_items)

    mean_facts_per_clip = float(np.mean([len(x) for x in clip_fact_keys]))

    t0 = time.perf_counter()
    for _ in range(3):
        for items in clip_fact_keys:
            for fk, _ in items:
                vals_ex = _dfs_horn_budgeted(fk, max_derivs=None)
                _ = 1.0 - math.prod(1.0 - min(0.999999, v) for v in vals_ex)
    sym_exhaust_ms_clip = (time.perf_counter() - t0) / (3 * 300.0) * 1000.0

    t0 = time.perf_counter()
    for _ in range(3):
        for i, items in enumerate(clip_fact_keys):
            mu_l = float(mu_L_adapt_300[i])
            for fk, Z_L in items:
                _ = _dfs_horn_adaptive(fk, Z_L, mu_l)
    sym_adapt_ms_clip = (time.perf_counter() - t0) / (3 * 300.0) * 1000.0

    t0 = time.perf_counter()
    for _ in range(3):
        for i, items in enumerate(clip_fact_keys):
            mu_l = float(mu_L_adapt_300[i])
            for fk, _ in items:
                vals_k = _dfs_horn_budgeted(fk, max_derivs=5)
                _ = get_regime(mu_l * (1.0 - math.prod(1.0 - min(0.999999, v) for v in vals_k)))
    sym_k5_ms_clip = (time.perf_counter() - t0) / (3 * 300.0) * 1000.0

    return {
        "hcmc_kb_stats": {
            "clean_base_facts": n_base_facts,
            "materialized_base_facts": len(mat_base),
            "rules_by_head": len(rules_by_head),
            "clean_multi_deriv_facts": N_facts,
        },
        "truncation_table": k_table,
        "clip_level_latency": {
            "mean_facts_evaluated_per_clip": round(mean_facts_per_clip, 2),
            "vcgat_forward_ms_clip": round(vcgat_ms_clip, 4),
            "symbolic_adapt_ms_clip": round(sym_adapt_ms_clip, 4),
            "symbolic_k5_ms_clip": round(sym_k5_ms_clip, 4),
            "symbolic_exhaust_ms_clip": round(sym_exhaust_ms_clip, 4),
            "total_reasoning_adapt_ms_clip": round(vcgat_ms_clip + sym_adapt_ms_clip, 4),
            "total_reasoning_k5_ms_clip": round(vcgat_ms_clip + sym_k5_ms_clip, 4),
            "total_reasoning_exhaust_ms_clip": round(vcgat_ms_clip + sym_exhaust_ms_clip, 4),
            "symbolic_speedup_adapt_vs_exhaust": round(sym_exhaust_ms_clip / sym_adapt_ms_clip if sym_adapt_ms_clip > 0 else 1.0, 2),
        },
    }


# =============================================================================
# MODULE 2: GROUNDING PROBS (PROTOCOL B-ZEROSHOT & PROTOCOL B-ADAPTATION)
# =============================================================================

def compute_protocol_b_grounding(
    idx300: np.ndarray,
    segs: List[Dict[str, Any]],
    feats_1161: Dict[str, np.ndarray],
    vlm_scores: Dict[str, Any],
    vcgat_engine: VCGATSemanticAnchoringEngine,
    sg35: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    prompts_map: Dict[str, List[str]],
    device: str,
) -> Dict[str, Any]:
    LOGGER.info("=== [Module 2 / Group 1] Protocol B Grounding (Z4 Zero-Shot & SuperEnsemble Adaptation) ===")
    sig1161 = feats_1161["siglip2"]
    cls35_to_i = {c: i for i, c in enumerate(OPEN35)}
    cls5_to_i = {c: i for i, c in enumerate(HCMC_CLASSES_5)}

    cache_npz = HCMC_DIR / "hcmc_300_x5_grounding_cache.npz"
    if cache_npz.exists():
        c = np.load(cache_npz)
        z4_probs_35 = c["z4_probs_35"]
        z4_probs_5 = c["z4_probs_5"]
        se300_probs_35 = c["se300_probs_35"]
        LOGGER.info("Loaded cached Module 2 grounding matrices from %s", cache_npz)
    else:
        # 1. Zero-Shot Z4_MultiView_OOD_Center (OOD bias strictly from Hanoi Val 117)
        with open(HANOI_VAL_EMB_PKL, "rb") as f:
            emb_pkl = pickle.load(f)
        val_rows = [json.loads(l) for l in open(HANOI_VAL_JSONL, "r", encoding="utf-8") if l.strip()]
        tr_anns = json.loads(HANOI_TRAIN_COCO.read_text(encoding="utf-8"))["annotations"]

        val_vids = sorted({int(r["video_id"]) for r in val_rows if int(r["video_id"]) in emb_pkl})
        v_val = np.stack([emb_pkl[v] for v in val_vids]).astype(np.float32)
        v_val /= (np.linalg.norm(v_val, axis=1, keepdims=True) + 1e-8)

        mp_val = compute_multiprompt_max(v_val, prompts_map, OPEN35, vcgat_engine.processor, vcgat_engine.siglip_model, vcgat_engine.device)
        fk_val = compute_fkg_grounding_matrix(v_val, OPEN35, sg35, vcgat_engine, top_k=5)["vis_cos_top5"]
        raw_val_35 = cfg["w_maxp_star"] * mp_val + cfg["alpha_fkg_star"] * fk_val
        ood_bias = raw_val_35.mean(axis=0, keepdims=True)

        mp_300 = compute_multiprompt_max(sig1161[idx300], prompts_map, OPEN35, vcgat_engine.processor, vcgat_engine.siglip_model, vcgat_engine.device)
        fk_300 = compute_fkg_grounding_matrix(sig1161[idx300], OPEN35, sg35, vcgat_engine, top_k=5)["vis_cos_top5"]
        z4_300_35 = (cfg["w_maxp_star"] * mp_300 + cfg["alpha_fkg_star"] * fk_300) - ood_bias

        z4_probs_35 = softmax_probs(z4_300_35, cfg["tau_35_mv"])
        z4_probs_5 = softmax_probs(z4_300_35[:, 30:], cfg["tau_5_mv"])

        # 2. Protocol B-Adaptation: 3-fold youtube_id-disjoint SuperEnsemble_T05 OOF on Mix_100_Raw (285 concepts)
        all_names = [s["landmark_en"] for s in segs]
        y5_all = np.array([cls5_to_i[n] for n in all_names], dtype=np.int64)
        y35_all = np.array([cls35_to_i[n] for n in all_names], dtype=np.int64)
        g_all = np.array([s["youtube_id"] for s in segs])
        fkg_p35_1161 = compute_fkg_grounding_matrix(sig1161, OPEN35, sg35, vcgat_engine, top_k=5)["vis_cos_top5"]

        # Hanoi Unified P_285 train bundle
        hn_tr_dino = _l2_norm(np.load(HANOI_TRAIN_DINO).astype(np.float32))
        hn_tr_sig = _l2_norm(np.load(HANOI_TRAIN_SIG6).astype(np.float32))
        hn_tr_cw = np.load(HANOI_TRAIN_CONCEPTS).astype(np.float32)
        hn_tr_y = np.array([cls35_to_i[a["parent_landmark"]] for a in tr_anns], dtype=np.int64)
        hn_tr_g = np.array([f"hn_tr_{a['video_id']}" for a in tr_anns])
        hn_tr_fkg = compute_fkg_grounding_matrix(hn_tr_sig, OPEN35, sg35, vcgat_engine, top_k=5)["vis_cos_top5"]

        N = len(segs)
        dino_all = feats_1161["dinov2_mid"]
        cw_all = feats_1161["c285_whole"]
        oof_se_t05 = np.zeros((N, 35), dtype=np.float32)

        sgkf = StratifiedGroupKFold(n_splits=3)
        for fi, (tr, te) in enumerate(sgkf.split(np.zeros(N), y5_all, g_all)):
            LOGGER.info("  [SuperEnsemble 35-way OOF] Training fold %d/3 (tr=%d HCMC + %d Hanoi -> te=%d HCMC)...",
                        fi + 1, len(tr), len(hn_tr_y), len(te))
            tr_dino = np.concatenate([hn_tr_dino, dino_all[tr]], axis=0)
            tr_sig = np.concatenate([hn_tr_sig, sig1161[tr]], axis=0)
            tr_cw_raw = np.concatenate([hn_tr_cw, cw_all[tr]], axis=0)
            tr_y = np.concatenate([hn_tr_y, y35_all[tr]], axis=0)
            tr_g = np.concatenate([hn_tr_g, g_all[tr]], axis=0)
            tr_fkg = np.concatenate([hn_tr_fkg, fkg_p35_1161[tr]], axis=0)

            mu_w, sd_w = tr_cw_raw.mean(0, keepdims=True), np.clip(tr_cw_raw.std(0, keepdims=True), 1e-4, None)
            cw_tr = (tr_cw_raw - mu_w) / sd_w
            cw_te = (cw_all[te] - mu_w) / sd_w

            pw, T_star = _inner_cv_temp_and_weights(
                {"sig": tr_sig, "conc_w": tr_cw_raw}, tr_y, tr_g, tr_fkg, 35, device, is_5way=False,
            )
            w = _make_class_weights(tr_y, 35, power=pw)

            pd_hc = train_mlp_seeds(tr_dino, tr_y, [dino_all[te]], 35, w, device)[0]
            ps_hc = train_mlp_seeds(tr_sig, tr_y, [sig1161[te]], 35, w, device)[0]
            pc_hc = train_mlp_seeds(cw_tr, tr_y, [cw_te], 35, w, device)[0]

            jf_tr = np.concatenate([tr_sig, tr_dino * 0.5, cw_tr * 0.1], axis=1)
            jf_hc = np.concatenate([sig1161[te], dino_all[te] * 0.5, cw_te * 0.1], axis=1)
            pj_hc = train_mlp_seeds(jf_tr, tr_y, [jf_hc], 35, w, device)[0]

            p_ens = 0.45 * ps_hc + 0.25 * pc_hc + 0.15 * pj_hc + 0.15 * pd_hc
            p_sup = 0.88 * p_ens + 0.12 * fkg_p35_1161[te]
            oof_se_t05[te] = _calibrate_probs(p_sup, 0.5)

        se300_probs_35 = oof_se_t05[idx300]
        np.savez(cache_npz, z4_probs_35=z4_probs_35, z4_probs_5=z4_probs_5, se300_probs_35=se300_probs_35)
        LOGGER.info("Cached Module 2 grounding matrices -> %s", cache_npz)
    gt300_idx = np.array([cls35_to_i[segs[i]["landmark_en"]] for i in idx300], dtype=np.int64)
    gt300_names = [segs[i]["landmark_en"] for i in idx300]
    clean_mask = np.array([vlm_scores[str(int(i))]["p_onsite"] >= 0.70 for i in idx300], dtype=bool)

    def summarize_35(probs: np.ndarray, label: str) -> Dict[str, Any]:
        ev = evaluate_predictions_from_probs(probs, gt300_idx)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        order = np.argsort(-probs, axis=1)
        conf_hn = float(np.mean(preds < 30) * 100.0)

        per_lm = {}
        for lm in HCMC_CLASSES_5:
            m = np.array([g == lm for g in gt300_names])
            per_lm[lm] = {
                "n": int(m.sum()),
                "top1": round(float(np.mean(preds[m] == gt300_idx[m]) * 100.0), 2),
                "top5": round(float(np.mean([gt300_idx[j] in order[j, :5] for j in np.where(m)[0]]) * 100.0), 2),
                "mean_muL": round(float(confs[m].mean()), 4),
            }
        return {
            "label": label,
            "n": 300,
            "top1": round(float(ev["R@1"]), 2),
            "macro_top1": round(float(np.mean([per_lm[lm]["top1"] for lm in HCMC_CLASSES_5])), 2),
            "top5": round(float(ev["R@5"]), 2),
            "mrr": round(float(ev["MRR"]), 4),
            "ece_pct": round(float(ev["ECE"]), 2),
            "confused_hanoi_pct": round(conf_hn, 2),
            "mean_muL": round(float(confs.mean()), 4),
            "regimes": {
                "HIGH": int((confs >= THETA_H).sum()),
                "MEDIUM": int(((confs >= THETA_M) & (confs < THETA_H)).sum()),
                "LOW": int((confs < THETA_M).sum()),
            },
            "clean_154": {
                "n": int(clean_mask.sum()),
                "top1": round(float(np.mean(preds[clean_mask] == gt300_idx[clean_mask]) * 100.0), 2),
                "mean_muL": round(float(confs[clean_mask].mean()), 4),
            },
            "noisy_146": {
                "n": int((~clean_mask).sum()),
                "top1": round(float(np.mean(preds[~clean_mask] == gt300_idx[~clean_mask]) * 100.0), 2),
                "mean_muL": round(float(confs[~clean_mask].mean()), 4),
            },
            "per_landmark": per_lm,
        }

    return {
        "z4_35way_probs": z4_probs_35,
        "z4_5way_probs": z4_probs_5,
        "se_35way_probs": se300_probs_35,
        "summary": {
            "Z4_MultiView_OOD_Center_35way_ZeroShot": summarize_35(z4_probs_35, "Protocol B-ZeroShot (35-way Z4_MV)"),
            "SuperEnsemble_T05_35way_Adaptation": summarize_35(se300_probs_35, "Protocol B-Adaptation (35-way SuperEnsemble OOF)"),
            "Z4_MultiView_OOD_Center_5way_ZeroShot_Ref": {
                "top1": round(float(np.mean(z4_probs_5.argmax(axis=1) == (gt300_idx - 30)) * 100.0), 2),
                "mean_muL_gt": round(float(z4_probs_5[np.arange(300), gt300_idx - 30].mean()), 4),
            },
        },
    }


# =============================================================================
# MODULE 3: FACT SELECTION, COSINE ABLATION, PROMPT BUILDERS & GPU GENERATION
# =============================================================================

def select_cosine_without_vcgat(
    vcgat_engine: VCGATSemanticAnchoringEngine,
    video_vec: np.ndarray,
    landmark: str,
    mu_L: float,
    certainty: str,
    precomputed_subgraph: Dict[str, Any],
    quota: Tuple[int, int, int],
) -> List[Dict[str, Any]]:
    """Without VC-GAT: pointwise SigLIP-2 cosine ranking (no GAT message passing)."""
    k_vis, k_dir, k_inf = quota
    cands = [dict(f) for f in precomputed_subgraph.get("cands", [])]
    if not cands:
        return []
    k_embs = precomputed_subgraph["K_embs_np"]
    sims = k_embs @ video_vec.astype(np.float32)

    for idx, f in enumerate(cands):
        f["vcgat_score"] = round(float(sims[idx]), 4)
        f["siglip_similarity"] = round(float(sims[idx]), 4)

    if certainty == "LOW":
        vis_c = [f for f in cands if f["relation"].lower() in VISUAL_RELS] or cands
        vis_c.sort(key=lambda x: x["vcgat_score"], reverse=True)
        return [{**f, "category": "visual", "c_f": round(mu_L * f["mu_F"], 4)} for f in vis_c[:k_vis]]

    raw_vis, raw_dir, raw_inf = [], [], []
    sv, sd, si = set(), set(), set()
    for f in cands:
        r, t = f["relation"].lower(), f["tail"].lower()
        key = (r, t)
        if r in VISUAL_RELS and key not in sv:
            sv.add(key); raw_vis.append(f)
        elif r in DIRECT_RELS and key not in sd:
            sd.add(key); raw_dir.append(f)
        elif (r in INFERRED_RELS or f.get("is_inferred", False)) and key not in si:
            si.add(key); raw_inf.append(f)

    raw_vis.sort(key=lambda x: x["vcgat_score"], reverse=True)
    raw_dir.sort(key=lambda x: x["vcgat_score"], reverse=True)
    raw_inf.sort(key=lambda x: x["vcgat_score"], reverse=True)

    sel, seen_all = [], set()
    for bucket, cat, k_lim in [(raw_vis, "visual", k_vis), (raw_dir, "direct", k_dir), (raw_inf, "inferred", k_inf)]:
        cnt = 0
        for f in bucket:
            if cnt >= k_lim:
                break
            key = (f["relation"].lower(), f["tail"].lower())
            if key not in seen_all:
                seen_all.add(key)
                sel.append({**f, "category": cat, "c_f": round(mu_L * f["mu_F"], 4)})
                cnt += 1
    return sel


def attach_uniform_siglip_similarity(
    facts: List[Dict[str, Any]],
    video_vec: np.ndarray,
    vcgat_engine: VCGATSemanticAnchoringEngine,
) -> List[Dict[str, Any]]:
    """Populates SigLIP-2 similarity uniformly on every fact for fair VSP checks."""
    out = []
    for f in facts:
        fc = dict(f)
        if fc.get("siglip_similarity") is None:
            k_emb = vcgat_engine.encode_fact_claim(fc["head"], fc["relation"], fc["tail"])
            fc["siglip_similarity"] = round(float(np.dot(video_vec.astype(np.float32), k_emb)), 4)
        out.append(fc)
    return out


def _strip_membership_degrees(sg_text: str) -> str:
    return "\n".join(re.sub(r"\s*\([0-9\.]+\)", "", l).strip() for l in sg_text.splitlines() if l.strip())


def build_hcmc_prompt_anchored(sample: Dict[str, Any], facts: List[Dict[str, Any]], landmark: str, mu_L: float, cert: str) -> str:
    sg_text = sample["observed_scene_graph"]
    tgt_s = sample["target_sentences"]
    tgt_w = sample["target_words"]
    facts_lines = [
        f"- {format_fact_claim_sentence(f.get('head', landmark), f.get('relation', 'features'), f.get('tail', ''))} "
        f"(c = {f.get('c_f', f.get('mu_F', 0.5) * mu_L):.2f})"
        for f in facts
    ]
    facts_str = "\n".join(facts_lines) if facts_lines else "- None"

    if cert == "HIGH":
        return f"""CERTAINTY LEVEL: HIGH (confidence = {mu_L:.2f})
LANDMARK: {landmark}
GIVEN FACTS:
{facts_str}

OBSERVED (visual scene graph from video, with membership degree):
{sg_text}

INSTRUCTION:
Write a cohesive {tgt_s} narration paragraph (approx. {tgt_w}).
- State the facts directly and confidently without hedging (do NOT use 'likely', 'appears', 'might').
- STRICT INVARIANT: NO NEW ATOMIC FACTUAL CLAIMS.
- Target length: exactly {tgt_s}s, between {tgt_w}.

CAPTION:"""
    elif cert == "MEDIUM":
        return f"""CERTAINTY LEVEL: MEDIUM (confidence = {mu_L:.2f})
LANDMARK CANDIDATE: {landmark}
GIVEN FACTS:
{facts_str}

OBSERVED (visual scene graph from video, with membership degree):
{sg_text}

INSTRUCTION:
Write a cohesive {tgt_s} narration paragraph (approx. {tgt_w}) using hedged language.
- Use hedging phrases for the landmark: 'likely shows', 'appears to be', 'possibly', 'if this is indeed {landmark}'.
- State only facts from the given list; do NOT assert them as absolute certainty.
- STRICT INVARIANT: NO NEW ATOMIC FACTUAL CLAIMS.
- Target length: exactly {tgt_s}s, between {tgt_w}.

CAPTION:"""
    else:
        return f"""CERTAINTY LEVEL: LOW (confidence = {mu_L:.2f})
LANDMARK: Unknown / Inconclusive
GIVEN FACTS:
- None

OBSERVED (visual scene graph from video, with membership degree):
{sg_text}

INSTRUCTION:
Write a cohesive {tgt_s} narration paragraph (approx. {tgt_w}).
- Explicitly state that the exact landmark identity is uncertain or cannot be definitively confirmed.
- Describe ONLY the visually observed elements from the scene graph.
- STRICT INVARIANT: NO NEW ATOMIC FACTUAL CLAIMS.
- Target length: exactly {tgt_s}s, between {tgt_w}.

CAPTION:"""


def build_hcmc_prompt_wo_fusion(sample: Dict[str, Any], facts: List[Dict[str, Any]], landmark: str) -> str:
    sg_text = _strip_membership_degrees(sample["observed_scene_graph"])
    tgt_s = sample["target_sentences"]
    tgt_w = sample["target_words"]
    city = "Ho Chi Minh City" if landmark in HCMC_CLASSES_5 else "Hanoi"
    facts_lines = [
        f"- {format_fact_claim_sentence(f.get('head', landmark), f.get('relation', 'features'), f.get('tail', ''))}"
        for f in facts
    ]
    facts_str = "\n".join(facts_lines) if facts_lines else f"- {landmark} is located in {city}."
    return f"""LANDMARK: {landmark}
GIVEN FACTS:
{facts_str}

OBSERVED:
{sg_text}

INSTRUCTION:
Write a cohesive {tgt_s} narration paragraph (approx. {tgt_w}) describing the scene.
- STRICT INVARIANT: NO NEW ATOMIC FACTUAL CLAIMS.
- Target length: exactly {tgt_s}s, between {tgt_w}.

CAPTION:"""


def build_hcmc_prompt_crisp_kg(sample: Dict[str, Any], facts: List[Dict[str, Any]], landmark: str) -> str:
    tgt_s = sample["target_sentences"]
    tgt_w = sample["target_words"]
    city = "Ho Chi Minh City" if landmark in HCMC_CLASSES_5 else "Hanoi"
    facts_lines = [
        f"- {landmark} {f['relation'].replace('_', ' ')} {f['tail']}."
        for f in facts
    ]
    facts_str = "\n".join(facts_lines) if facts_lines else f"- {landmark} is located in {city}."
    return f"""LANDMARK: {landmark}
KNOWLEDGE FACTS:
{facts_str}

OBSERVED (visual scene graph from video):
{sample['observed_scene_graph']}

INSTRUCTION:
Write a cohesive {tgt_s} narration paragraph (approx. {tgt_w}) describing the landmark and scene. Confidently state the knowledge facts.
- STRICT INVARIANT: NO NEW ATOMIC FACTUAL CLAIMS.
- Target length: exactly {tgt_s}s, between {tgt_w}.

CAPTION:"""


def build_hcmc_prompt_visual_only(sample: Dict[str, Any]) -> str:
    tgt_s = sample["target_sentences"]
    tgt_w = sample["target_words"]
    return f"""OBSERVED (visual scene graph from video):
{sample['observed_scene_graph']}

INSTRUCTION:
Identify the location shown if possible, and write a cohesive {tgt_s} narration paragraph (approx. {tgt_w}) describing the scene.
- STRICT INVARIANT: NO NEW ATOMIC FACTUAL CLAIMS.
- Target length: exactly {tgt_s}s, between {tgt_w}.

CAPTION:"""


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    samples: List[Dict[str, Any]],
    batch_size: int = 32,
) -> List[str]:
    model.eval()
    captions: List[str] = []
    N = len(prompts)
    for i in range(0, N, batch_size):
        b_prompts = prompts[i: i + batch_size]
        b_samples = samples[i: i + batch_size]
        b_max_toks = max(s["max_new_tokens"] for s in b_samples)
        chat_texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in b_prompts
        ]
        enc = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True, max_length=1800)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=b_max_toks,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for bi in range(len(b_prompts)):
            gen = tokenizer.decode(out[bi][prompt_len:], skip_special_tokens=True).strip()
            for pref in ["CAPTION:", "Caption:", "caption:", "Sure!", "Here is a caption:"]:
                if gen.startswith(pref):
                    gen = gen[len(pref):].strip()
            gen = re.split(r"\n\s*(?:Note:|\(Note:|###|Based on)", gen)[0].strip()
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", gen) if len(s.strip()) > 5]
            max_s = b_samples[bi]["max_sentences"]
            max_w = b_samples[bi]["max_words"]
            if sents:
                chosen, wc = [], 0
                for s in sents[:max_s]:
                    chosen.append(s)
                    wc += len(s.split())
                    if wc >= max_w:
                        break
                gen = " ".join(chosen)
                if not gen.endswith((".", "!", "?")):
                    gen += "."
            captions.append(gen)
    return captions


# =============================================================================
# MODULE 4: SECTION 6.8 METRICS + STRICT AUDIT + HIERARCHICAL BOOTSTRAP
# =============================================================================

def evaluate_caption_x5(
    caption: str,
    sample: Dict[str, Any],
    given_facts: List[Dict[str, Any]],
    pred_lm: str,
    cert: str,
    fkg_years_map: Dict[str, Set[str]],
    all_visual_cand_facts: List[Dict[str, Any]],
) -> Dict[str, float]:
    cap_lower = caption.lower()
    sg_entities = parse_scene_graph(sample["observed_scene_graph"])
    gold_facts = sample["gold_facts"]
    gt_lm = sample["landmark_gt"]

    # 1. Knowledge Recalls (against Oracle gt_parent duration-quota F_gold)
    n_gold = len(gold_facts)
    n_matched = 0
    inf_m, n_inf, dir_m, n_dir, vis_m, n_vis = 0, 0, 0, 0, 0, 0
    for gf in gold_facts:
        tail = gf.get("tail", "")
        cat = gf.get("category", "")
        m = match_fact(tail, cap_lower)
        if m:
            n_matched += 1
        if cat == "inferred" or gf.get("is_inferred"):
            n_inf += 1
            if m:
                inf_m += 1
        elif cat == "direct":
            n_dir += 1
            if m:
                dir_m += 1
        elif cat == "visual":
            n_vis += 1
            if m:
                vis_m += 1

    k_recall = (n_matched / n_gold * 100.0) if n_gold > 0 else 100.0
    inf_recall = (inf_m / n_inf * 100.0) if n_inf > 0 else 0.0
    dir_recall = (dir_m / n_dir * 100.0) if n_dir > 0 else 0.0
    vis_recall = (vis_m / n_vis * 100.0) if n_vis > 0 else 0.0

    # 2. VSP: check Section 6.8 (siglip >= 0.02 OR is_entity_in_scene), Uniform SigLIP-2 (>= 0.02), and Strict SG
    n_vc_68, n_sup_68 = 0, 0
    n_vc_uni, n_sup_uni = 0, 0
    n_vc_sg, n_sup_sg = 0, 0

    checked: Set[str] = set()
    # Pool visual facts from given_facts, gold_facts, and active visual candidate pool
    for vf in given_facts + gold_facts + all_visual_cand_facts:
        if not (vf.get("relation", "").lower() in VISUAL_TESTABLE_RELATIONS or vf.get("category") == "visual"):
            continue
        tail = vf.get("tail", "").lower().strip()
        if not tail or len(tail) < 3 or tail in checked:
            continue
        words = [w for w in tail.split() if len(w) > 3]
        if tail in cap_lower or (len(words) >= 2 and sum(1 for w in words if w in cap_lower) >= 2):
            checked.add(tail)
            sim = vf.get("siglip_similarity", None)
            in_sg = is_entity_in_scene(tail, sg_entities)
            in_clip = (sim >= 0.02) if sim is not None else in_sg
            n_vc_68 += 1
            if in_clip:
                n_sup_68 += 1
            n_vc_uni += 1
            if sim is not None and sim >= 0.02:
                n_sup_uni += 1
            elif sim is None and in_sg:
                n_sup_uni += 1
            n_vc_sg += 1
            if in_sg:
                n_sup_sg += 1

    for ent in sg_entities:
        if ent in checked:
            continue
        words = [w for w in ent.split() if len(w) > 3]
        if ent in cap_lower or (len(words) >= 2 and sum(1 for w in words if w in cap_lower) >= 2):
            checked.add(ent)
            n_vc_68 += 1; n_sup_68 += 1
            n_vc_uni += 1; n_sup_uni += 1
            n_vc_sg += 1; n_sup_sg += 1

    vsp = (n_sup_68 / n_vc_68 * 100.0) if n_vc_68 > 0 else 100.0
    vsp_uniform = (n_sup_uni / n_vc_uni * 100.0) if n_vc_uni > 0 else 100.0
    vsp_sg = (n_sup_sg / n_vc_sg * 100.0) if n_vc_sg > 0 else 100.0

    # 3. Epistemic markers, Policy Adherence, OAR_gen, EAP
    n_assert = sum(1 for m in ASSERTIVE_MARKERS if m in cap_lower)
    n_hedge = sum(1 for m in HEDGING_MARKERS if m in cap_lower)
    n_uncert = sum(1 for m in UNCERTAIN_MARKERS if m in cap_lower)

    if cert == "HIGH":
        pol = (n_assert >= 1 and n_hedge == 0 and n_uncert == 0)
        oar_gen = 0.0
    elif cert == "MEDIUM":
        pol = (n_hedge >= 1)
        oar_gen = 100.0 if (n_hedge == 0 and n_uncert == 0 and n_assert >= 1) else 0.0
    else:  # LOW
        pol = (n_uncert >= 1 or (n_hedge >= 1 and n_assert <= 1))
        oar_gen = 100.0 if (n_assert >= 2 and n_uncert == 0 and n_hedge == 0) else 0.0

    # 4. Strict Factual Audit: wrong landmark assertion + year hallucinations
    # Mask generic visual-concept tag 'old post office' from concepts_285 so it does not false-trigger 'Saigon Central Post Office'
    cap_for_lm = re.sub(r"\bold post office\b", "old_pc_tag", cap_lower)
    mentioned_lms = find_mentioned_landmarks(cap_for_lm)
    wrong_lms = [m for m in mentioned_lms if m.lower() != gt_lm.lower()]
    hedged_flag = (n_hedge >= 1 or n_uncert >= 1)
    wrong_lm_assert = 100.0 if (wrong_lms and not hedged_flag) else 0.0

    cap_years = re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", caption)
    allowed_years_gt = set(fkg_years_map.get(gt_lm, set()))
    for gf in gold_facts:
        allowed_years_gt.update(re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", str(gf.get("tail", ""))))
    allowed_years_given = set()
    for f in given_facts:
        allowed_years_given.update(re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", str(f.get("tail", ""))))
    year_halluc_given = any(y not in allowed_years_given for y in cap_years) if given_facts else bool(cap_years)
    real_year_err = any(y not in allowed_years_gt for y in cap_years)

    # Rule-based Fact-P / OAR_fact (run_comprehensive_1119_benchmark semantics)
    oar_fact_rule = 100.0 if (year_halluc_given or wrong_lm_assert > 0.0 or oar_gen > 0.0) else 0.0
    # Strict Factual Precision / OAR_fact (strict wrong-landmark or hallucinated year or unhedged LOW wrong claims)
    true_factual_err = bool(real_year_err or wrong_lm_assert > 0.0 or (cert == "LOW" and wrong_lms))
    strict_oar_fact = 100.0 if true_factual_err else 0.0
    strict_fact_p = 0.0 if true_factual_err else 100.0

    # EAP: on LOW (abstain/uncertain), did it avoid wrong unhedged landmark/year assertions?
    # Or overall: fraction of asserted factual claims (non-abstained) / abstention correctness
    eap = 100.0 if not true_factual_err else 0.0

    ref_cap = sample.get("reference_caption", "")
    bleu4 = compute_bleu4(ref_cap, caption) if ref_cap else 0.0
    rougel = compute_rouge_l(ref_cap, caption) if ref_cap else 0.0

    return {
        "k_recall": k_recall,
        "inf_recall": inf_recall,
        "dir_recall": dir_recall,
        "vis_recall": vis_recall,
        "vsp": vsp,
        "vsp_uniform": vsp_uniform,
        "vsp_sg": vsp_sg,
        "fact_p_rule": 100.0 if not (real_year_err or wrong_lm_assert > 0.0) else 0.0,
        "fact_p_strict": strict_fact_p,
        "eap": eap,
        "policy": 100.0 if pol else 0.0,
        "oar_gen": oar_gen,
        "oar_fact_rule": oar_fact_rule,
        "oar_fact_strict": strict_oar_fact,
        "wrong_lm_assert": wrong_lm_assert,
        "year_err": 100.0 if real_year_err else 0.0,
        "bleu4": bleu4,
        "rougel": rougel,
    }


def compute_dual_bootstrap(
    evals: List[Dict[str, float]],
    gts: List[str],
    yt_ids: List[str],
    metric_keys: List[str],
    B: int = B_BOOT,
    seed: int = SEED,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    N = len(evals)
    M = len(metric_keys)
    mat = np.array([[e[k] for k in metric_keys] for e in evals], dtype=np.float64)

    # 1. Clip-level i.i.d. bootstrap (matching Section 6.8 Hanoi 500)
    idx_clip = rng.randint(0, N, size=(B, N))
    clip_means = np.mean(mat[idx_clip], axis=1)

    # 2. Hierarchical Landmark-Stratified + YouTube-Cluster Bootstrap
    lm_clusters: Dict[str, List[np.ndarray]] = {}
    for lm in HCMC_CLASSES_5:
        by_yt: Dict[str, List[int]] = defaultdict(list)
        for i, (g, yt) in enumerate(zip(gts, yt_ids)):
            if g == lm:
                by_yt[yt].append(i)
        lm_clusters[lm] = [np.array(idxs, dtype=np.int64) for idxs in by_yt.values()]

    hier_means = np.zeros((B, M), dtype=np.float64)
    for b in range(B):
        chosen_idx = []
        for lm in HCMC_CLASSES_5:
            clist = lm_clusters[lm]
            picks = rng.randint(0, len(clist), size=len(clist))
            for p in picks:
                chosen_idx.append(clist[p])
        flat_idx = np.concatenate(chosen_idx)
        hier_means[b] = np.mean(mat[flat_idx], axis=0)

    # Macro-average across the 5 landmarks
    lm_masks = [np.array([g == lm for g in gts], dtype=bool) for lm in HCMC_CLASSES_5]
    macro_vals = np.mean(np.stack([mat[m].mean(axis=0) for m in lm_masks], axis=0), axis=0)

    summary = {}
    for j, k in enumerate(metric_keys):
        pm = float(mat[:, j].mean())
        summary[k] = {
            "mean": round(pm, 2),
            "macro_5lm": round(float(macro_vals[j]), 2),
            "clip_std": round(float(np.std(clip_means[:, j], ddof=1)), 2),
            "clip_ci95": [round(float(np.percentile(clip_means[:, j], 2.5)), 2), round(float(np.percentile(clip_means[:, j], 97.5)), 2)],
            "hier_std": round(float(np.std(hier_means[:, j], ddof=1)), 2),
            "hier_ci95": [round(float(np.percentile(hier_means[:, j], 2.5)), 2), round(float(np.percentile(hier_means[:, j], 97.5)), 2)],
        }
    return summary


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main() -> None:
    t_global = time.time()
    set_seed(SEED)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Starting Experiment X5 on %s | THETA_M=%.2f, THETA_H=%.2f", device, THETA_M, THETA_H)

    # 1. Load datasets & RESTORED 285-concept bank
    with open(BAL300_IDX_JSON, "r", encoding="utf-8") as f:
        idx300 = np.array(json.load(f)["indices"], dtype=np.int64)
    with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
        segs = json.load(f)["segments"]
    with open(VLM_CURATION_JSON, "r", encoding="utf-8") as f:
        vlm_scores = json.load(f)
    with open(CONCEPTS_285_JSON, "r", encoding="utf-8") as f:
        concepts_285 = json.load(f)
    with open(FROZEN_CFG_JSON, "r", encoding="utf-8") as f:
        frozen_cfg = json.load(f)
    with open(PROMPTS_HCMC_JSON, "r", encoding="utf-8") as f:
        prompts_map = {**LANDMARK_DESCRIPTIVE_PROMPTS, **json.load(f)["descriptive_prompts"]}

    ref_by_clip_idx = {}
    if GOLD_READY_JSONL.exists():
        with open(GOLD_READY_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    ref_by_clip_idx[int(rec["clip_idx"])] = rec.get("reference_caption", "")

    npz = np.load(FEATS_1161_NPZ)
    feats_1161 = {k: npz[k] for k in npz.files}

    # 2. Load URI-safe HCMC FKG + Deep VC-GAT engine (frozen from Hanoi)
    fkg_index = load_stop_filtered_uri_fkg(HCMC_FKG_DIR)
    vcgat_engine = VCGATSemanticAnchoringEngine(device=device)
    sg35 = precompute_uri_safe_subgraphs(OPEN35, fkg_index, vcgat_engine)
    fkg_years_map = build_fkg_years_map(fkg_index)

    # 3. Module 2: Grounding (Protocol B-ZeroShot & Protocol B-Adaptation)
    grounding_out = compute_protocol_b_grounding(
        idx300, segs, feats_1161, vlm_scores, vcgat_engine, sg35, frozen_cfg, prompts_map, device
    )
    z4_probs_35 = grounding_out["z4_35way_probs"]
    z4_probs_5 = grounding_out["z4_5way_probs"]
    se_probs_35 = grounding_out["se_35way_probs"]

    # 4. Module 1: Group 5 Reasoning Efficiency on Pure HCMC KB
    gts_300 = [segs[int(i)]["landmark_en"] for i in idx300]
    yt_300 = [segs[int(i)]["youtube_id"] for i in idx300]
    sig300 = feats_1161["siglip2"][idx300]
    muL_adapt_300 = se_probs_35.max(axis=1)

    group5_report = run_group5_efficiency_benchmark(muL_adapt_300, vcgat_engine, sg35, gts_300, sig300)
    LOGGER.info("Group 5 Pure HCMC: adapt_ms_clip=%.4f vs exhaust_ms_clip=%.4f (sym speedup=%.2fx)",
                group5_report["clip_level_latency"]["total_reasoning_adapt_ms_clip"],
                group5_report["clip_level_latency"]["total_reasoning_exhaust_ms_clip"],
                group5_report["clip_level_latency"]["symbolic_speedup_adapt_vs_exhaust"])

    # 5. Build Canonical 300-Clip Dataset with OLD 285-concept SG & Precomputed Facts
    LOGGER.info("Building canonical 300-clip dataset with restored 285-concept bank...")
    cls35_to_i = {c: i for i, c in enumerate(OPEN35)}
    cls5_to_i = {c: i for i, c in enumerate(HCMC_CLASSES_5)}

    samples: List[Dict[str, Any]] = []
    # Precompute fact bundles keyed by (sample_pos, track_name, mode)
    prompt_bundles: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for pos, raw_idx in enumerate(idx300):
        idx_i = int(raw_idx)
        seg = segs[idx_i]
        gt_lm = seg["landmark_en"]
        dur = float(seg["duration"])
        tier, quota, tgt_s, tgt_w, max_s, max_w, max_toks = get_duration_tier_and_quota(dur)

        c285_vec = feats_1161["c285_whole"][idx_i]
        obs_sg, top_c = format_observed_scene_graph_c285(c285_vec, concepts_285)
        v_vec = sig300[pos]

        # Canonical Oracle Duration-Quota F_gold
        gold_facts = select_uri_safe_quota_facts(
            vcgat_engine, v_vec, gt_lm, mu_L=1.0, certainty="HIGH",
            fkg_index=fkg_index, quota=quota, anchored=True, precomputed_subgraph=sg35[gt_lm],
        )

        rec = {
            "sample_id": f"hcmc_x5_{idx_i:04d}",
            "clip_idx": idx_i,
            "youtube_id": seg["youtube_id"],
            "landmark_gt": gt_lm,
            "duration": dur,
            "duration_tier": tier,
            "fact_quota": list(quota),
            "target_sentences": tgt_s,
            "target_words": tgt_w,
            "max_sentences": max_s,
            "max_words": max_w,
            "max_new_tokens": max_toks,
            "p_onsite": float(vlm_scores[str(idx_i)]["p_onsite"]),
            "is_clean_onsite": bool(vlm_scores[str(idx_i)]["p_onsite"] >= 0.70),
            "observed_scene_graph": obs_sg,
            "top_concepts_285": top_c,
            "gold_facts": gold_facts,
            "reference_caption": ref_by_clip_idx.get(idx_i, ""),
        }
        samples.append(rec)

        # Grounding signals for the 4 protocol tracks
        gt_5i = cls5_to_i[gt_lm]
        z4_pred_lm = OPEN35[int(np.argmax(z4_probs_35[pos]))]
        z4_mu_L = float(np.max(z4_probs_35[pos]))
        se_pred_lm = OPEN35[int(np.argmax(se_probs_35[pos]))]
        se_mu_L = float(np.max(se_probs_35[pos]))
        z4_5_gt_muL = float(z4_probs_5[pos, gt_5i])

        track_specs = [
            ("ProtocolA_OracleHigh", gt_lm, 0.95, "HIGH"),
            ("ProtocolA_ZeroShotVis", gt_lm, z4_5_gt_muL, get_regime(z4_5_gt_muL)),
            ("ProtocolB_ZeroShot", z4_pred_lm, z4_mu_L, get_regime(z4_mu_L)),
            ("ProtocolB_Adaptation", se_pred_lm, se_mu_L, get_regime(se_mu_L)),
        ]
        for tname, alm, amu, acert in track_specs:
            # Anchored VC-GAT facts
            f_anch = select_uri_safe_quota_facts(
                vcgat_engine, v_vec, alm, mu_L=amu, certainty=acert,
                fkg_index=fkg_index, quota=quota, anchored=True, precomputed_subgraph=sg35[alm],
            )
            # Unanchored static mu_F (Without VC-GAT - matching Hanoi 500 ours_unanchored)
            f_unan_raw = select_uri_safe_quota_facts(
                vcgat_engine, v_vec, alm, mu_L=amu, certainty=acert,
                fkg_index=fkg_index, quota=quota, anchored=False, precomputed_subgraph=sg35[alm],
            )
            f_unan = attach_uniform_siglip_similarity(f_unan_raw, v_vec, vcgat_engine)
            # Cosine-only ranking (Without VC-GAT graph attention)
            f_cos = select_cosine_without_vcgat(
                vcgat_engine, v_vec, alm, mu_L=amu, certainty=acert,
                precomputed_subgraph=sg35[alm], quota=quota,
            )
            # Crisp-KG (unanchored HIGH quota, ignores mu_L thresholding)
            f_crisp_raw = select_uri_safe_quota_facts(
                vcgat_engine, v_vec, alm, mu_L=1.0, certainty="HIGH",
                fkg_index=fkg_index, quota=quota, anchored=False, precomputed_subgraph=sg35[alm],
            )
            f_crisp = attach_uniform_siglip_similarity(f_crisp_raw, v_vec, vcgat_engine)

            prompt_bundles[(tname, "Full_Framework")].append({
                "alm": alm, "mu_L": amu, "cert": acert, "facts": f_anch,
                "prompt": build_hcmc_prompt_anchored(rec, f_anch, alm, amu, acert),
            })
            prompt_bundles[(tname, "Without_VCGAT_Unanchored")].append({
                "alm": alm, "mu_L": amu, "cert": acert, "facts": f_unan,
                "prompt": build_hcmc_prompt_anchored(rec, f_unan, alm, amu, acert),
            })
            if tname == "ProtocolA_OracleHigh":
                prompt_bundles[(tname, "Without_VCGAT_Cosine")].append({
                    "alm": alm, "mu_L": amu, "cert": acert, "facts": f_cos,
                    "prompt": build_hcmc_prompt_anchored(rec, f_cos, alm, amu, acert),
                })
            prompt_bundles[(tname, "Without_Certainty_Control")].append({
                "alm": alm, "mu_L": amu, "cert": acert, "facts": f_anch,
                "prompt": build_hcmc_prompt_wo_fusion(rec, f_anch, alm),
            })
            prompt_bundles[(tname, "Prompt_Only_LLM")].append({
                "alm": alm, "mu_L": amu, "cert": acert, "facts": f_anch,
                "prompt": build_hcmc_prompt_anchored(rec, f_anch, alm, amu, acert),
            })
            if tname != "ProtocolA_ZeroShotVis":  # Identical to ProtocolA_OracleHigh Crisp-KG
                prompt_bundles[(tname, "Crisp_KG_RAG")].append({
                    "alm": alm, "mu_L": 1.0, "cert": "HIGH", "facts": f_crisp,
                    "prompt": build_hcmc_prompt_crisp_kg(rec, f_crisp, alm),
                })

    prompt_bundles[("Shared", "Visual_Only")].extend([
        {
            "alm": s["landmark_gt"], "mu_L": 0.0, "cert": "LOW", "facts": [],
            "prompt": build_hcmc_prompt_visual_only(s),
        }
        for s in samples
    ])

    with open(OUT_DATASET_JSONL, "w", encoding="utf-8") as f:
        for rec in samples:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    LOGGER.info("Saved rebuilt 300-clip 285-concept dataset -> %s", OUT_DATASET_JSONL)

    # Free SigLIP2/VC-GAT GPU memory before Qwen2.5-3B generation
    all_visual_cand_map: Dict[int, List[Dict[str, Any]]] = {}
    for pos in range(300):
        # Store SigLIP-2 similarity for candidate pool of ground-truth & predicted landmarks if needed later
        all_visual_cand_map[pos] = prompt_bundles[("ProtocolA_OracleHigh", "Full_Framework")][pos]["facts"]

    del vcgat_engine, sg35
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. GPU Batch Generation (single base model load, adapter swapping)
    all_captions: Dict[str, List[str]] = {}
    if OUT_CAPTIONS_JSON.exists():
        all_captions = json.loads(OUT_CAPTIONS_JSON.read_text(encoding="utf-8"))

    base_tasks = [(k, v) for k, v in prompt_bundles.items() if k[1] in ("Prompt_Only_LLM", "Crisp_KG_RAG", "Visual_Only")]
    cont_tasks = [(k, v) for k, v in prompt_bundles.items() if k[1] in ("Full_Framework", "Without_VCGAT_Unanchored", "Without_VCGAT_Cosine")]
    wof_tasks = [(k, v) for k, v in prompt_bundles.items() if k[1] == "Without_Certainty_Control"]

    needed = [f"{t}::{c}" for (t, c), _ in base_tasks + cont_tasks + wof_tasks if f"{t}::{c}" not in all_captions or len(all_captions[f"{t}::{c}"]) != 300]
    if needed:
        qwen_path = resolve_qwen3b_path()
        LOGGER.info("Loading Qwen2.5-3B-Instruct from %s (%d missing config passes)...", qwen_path, len(needed))
        tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=torch.bfloat16, device_map=device)

        for (tname, cfg_name), plist in base_tasks:
            key_str = f"{tname}::{cfg_name}"
            if key_str in all_captions and len(all_captions[key_str]) == 300:
                continue
            t0 = time.time()
            all_captions[key_str] = generate_batch(base_model, tokenizer, [x["prompt"] for x in plist], samples, batch_size=32)
            LOGGER.info("  Generated [BASE | %s] 300 captions in %.1fs", key_str, time.time() - t0)
            OUT_CAPTIONS_JSON.write_text(json.dumps(all_captions, indent=2, ensure_ascii=False), encoding="utf-8")

        peft_model = PeftModel.from_pretrained(base_model, str(CKPT_CONTINUOUS), adapter_name="continuous")
        peft_model.set_adapter("continuous")
        for (tname, cfg_name), plist in cont_tasks:
            key_str = f"{tname}::{cfg_name}"
            if key_str in all_captions and len(all_captions[key_str]) == 300:
                continue
            t0 = time.time()
            all_captions[key_str] = generate_batch(peft_model, tokenizer, [x["prompt"] for x in plist], samples, batch_size=32)
            LOGGER.info("  Generated [LORA-CONT | %s] 300 captions in %.1fs", key_str, time.time() - t0)
            OUT_CAPTIONS_JSON.write_text(json.dumps(all_captions, indent=2, ensure_ascii=False), encoding="utf-8")

        peft_model.load_adapter(str(CKPT_WO_FUSION), adapter_name="wo_fusion")
        peft_model.set_adapter("wo_fusion")
        for (tname, cfg_name), plist in wof_tasks:
            key_str = f"{tname}::{cfg_name}"
            if key_str in all_captions and len(all_captions[key_str]) == 300:
                continue
            t0 = time.time()
            all_captions[key_str] = generate_batch(peft_model, tokenizer, [x["prompt"] for x in plist], samples, batch_size=32)
            LOGGER.info("  Generated [LORA-WOF | %s] 300 captions in %.1fs", key_str, time.time() - t0)
            OUT_CAPTIONS_JSON.write_text(json.dumps(all_captions, indent=2, ensure_ascii=False), encoding="utf-8")

        del peft_model, base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Alias shared baselines across ProtocolA_ZeroShotVis & Visual_Only
    all_captions["ProtocolA_ZeroShotVis::Crisp_KG_RAG"] = all_captions["ProtocolA_OracleHigh::Crisp_KG_RAG"]
    prompt_bundles[("ProtocolA_ZeroShotVis", "Crisp_KG_RAG")] = prompt_bundles[("ProtocolA_OracleHigh", "Crisp_KG_RAG")]
    for tn in ["ProtocolA_OracleHigh", "ProtocolA_ZeroShotVis", "ProtocolB_ZeroShot", "ProtocolB_Adaptation"]:
        all_captions[f"{tn}::Visual_Only"] = all_captions["Shared::Visual_Only"]
        prompt_bundles[(tn, "Visual_Only")] = prompt_bundles[("Shared", "Visual_Only")]

    with open(OUT_CAPTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved all generated captions -> %s", OUT_CAPTIONS_JSON)

    # 7. Module 4: Evaluate Section 6.8 + Strict Audit + Clip/Hierarchical Bootstraps + Gap + Per-Landmark
    metric_keys = [
        "k_recall", "inf_recall", "dir_recall", "vis_recall",
        "vsp", "vsp_uniform", "vsp_sg",
        "fact_p_rule", "fact_p_strict", "eap",
        "policy", "oar_gen", "oar_fact_rule", "oar_fact_strict",
        "wrong_lm_assert", "year_err", "bleu4", "rougel",
    ]
    clean_mask = np.array([s["is_clean_onsite"] for s in samples], dtype=bool)
    eval_report: Dict[str, Any] = {}

    for (tname, cfg_name), caps in sorted(
        [(tuple(k.split("::")), v) for k, v in all_captions.items() if not k.startswith("Shared::")]
    ):
        plist = prompt_bundles[(tname, cfg_name)]
        row_evals = []
        for pos in range(300):
            ev = evaluate_caption_x5(
                caps[pos],
                samples[pos],
                plist[pos]["facts"],
                plist[pos]["alm"],
                plist[pos]["cert"],
                fkg_years_map,
                all_visual_cand_map[pos],
            )
            row_evals.append(ev)

        boot_all = compute_dual_bootstrap(row_evals, gts_300, yt_300, metric_keys, B=B_BOOT, seed=SEED)

        # Clean vs Noisy slices + Per-Landmark
        clean_means = {k: round(float(np.mean([row_evals[i][k] for i in range(300) if clean_mask[i]])), 2) for k in metric_keys}
        noisy_means = {k: round(float(np.mean([row_evals[i][k] for i in range(300) if not clean_mask[i]])), 2) for k in metric_keys}
        per_lm = {}
        for lm in HCMC_CLASSES_5:
            sub = [row_evals[i] for i in range(300) if gts_300[i] == lm]
            per_lm[lm] = {k: round(float(np.mean([x[k] for x in sub])), 2) for k in ["k_recall", "vsp", "vsp_sg", "fact_p_strict", "policy", "oar_gen", "oar_fact_strict", "wrong_lm_assert"]}

        reg_cnt = dict(Counter(p["cert"] for p in plist))
        gap_vs_hanoi = {
            "k_recall_gap": round(boot_all["k_recall"]["mean"] - HANOI_500_REF["k_recall"], 2),
            "vsp_gap": round(boot_all["vsp"]["mean"] - HANOI_500_REF["vsp"], 2),
            "fact_p_strict_vs_audited_gap": round(boot_all["fact_p_strict"]["mean"] - HANOI_500_REF["fact_p_audited"], 2),
            "policy_gap": round(boot_all["policy"]["mean"] - HANOI_500_REF["policy_adherence"], 2),
            "oar_gen_gap": round(boot_all["oar_gen"]["mean"] - HANOI_500_REF["oar_gen"], 2),
            "oar_fact_strict_vs_audited_gap": round(boot_all["oar_fact_strict"]["mean"] - HANOI_500_REF["oar_fact_audited"], 2),
            "eap_gap": round(boot_all["eap"]["mean"] - HANOI_500_REF["eap"], 2),
        }

        eval_report.setdefault(tname, {})[cfg_name] = {
            "n_samples": 300,
            "regimes": reg_cnt,
            "metrics_with_dual_ci": boot_all,
            "gap_vs_hanoi_500": gap_vs_hanoi,
            "clean_154": clean_means,
            "noisy_146": noisy_means,
            "per_landmark": per_lm,
        }

    final_report = {
        "experiment": "Experiment X5: Cross-Region & Unseen-Landmark Generalization (300 HCMC clips, Restored 285 Concepts)",
        "dataset_summary": {
            "n_clips": 300,
            "n_landmarks": 5,
            "clips_per_landmark": 60,
            "n_youtube_clusters": len(set(yt_300)),
            "clean_onsite_154": int(clean_mask.sum()),
            "noisy_146": int((~clean_mask).sum()),
            "concept_bank": "Restored Original 285 Concepts (concepts_285.json / prototypes_285x768.pt)",
            "thresholds_frozen_hanoi_149": {"THETA_M": THETA_M, "THETA_H": THETA_H, "T_se": 0.5},
        },
        "hanoi_500_reference": HANOI_500_REF,
        "group1_grounding": grounding_out["summary"],
        "group5_reasoning_efficiency_pure_hcmc": group5_report,
        "protocols_e2e_evaluation": eval_report,
        "statistical_caveat": (
            "Conformal risk guarantees established on Hanoi i.i.d./within-region data cannot be asserted "
            "directly on HCMC without distribution-shift recalibration because cross-region exchangeability does not hold."
        ),
        "runtime_seconds": round(time.time() - t_global, 1),
    }

    with open(OUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    LOGGER.info("Experiment X5 COMPLETED in %.1fs -> %s", time.time() - t_global, OUT_REPORT_JSON)


if __name__ == "__main__":
    main()
