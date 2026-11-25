#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/hcmc_07_uri_safe_rebuild_and_e2e.py
=============================================================================
Clean URI-Safe Rebuild of the Unseen HCMC Generalization Pipeline (Zero Leakage):
  Phase A:
    - Hanoi Validation (117 unique clips) calibration with URI-safe FKG subgraphs
      -> writes & freezes `frozen_zeroshot_config_urisafe.json` BEFORE HCMC.
    - Evaluates Z1–Z4 zero-shot recognition on Raw (1,161), VLM-Clean (481),
      and VLM-Clean Balanced (180) HCMC splits with 0% cross-city FKG leaks.
  Phase B:
    - Incrementally extends Multi-VLM visual target probing (`Qwen3-VL-8B` +
      `CLIP ViT-B/32`) for any newly surfaced clean URI-safe visual tails on the
      180 balanced clean clips (reusing cached scene descriptions & existing probs).
    - Builds BOTH evaluation-only ground-truth files:
        * `hcmc_clean_180_uri_std_gold.jsonl`   (URI-safe VC-GAT gold + 14B ref)
        * `hcmc_clean_180_uri_indep_gold.jsonl` (100% VC-GAT/SigLIP-2 independent
          Multi-VLM + static mu_F gold + 14B ref without scene graph)
  Phase C-Gen:
    - Strictly label-free end-to-end caption generation under `z4` & `z3` across
      all 6 systems (`B1`, `B2`, `B3`, `B4`, `Proposed_5way`, `Proposed_35way`)
      using URI-safe FKG subgraphs and frozen checkpoints only, recording exact
      `facts_given` and `prompt_regime` per sample for full retention/error audit.
=============================================================================
"""

import argparse
import gc
import json
import logging
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from scipy.optimize import minimize_scalar
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPModel,
    CLIPProcessor,
)
from qwen_vl_utils import process_vision_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("HCMCStage7URISafe")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.config import LANDMARK_DESCRIPTIVE_PROMPTS, PARENT_CLASSES_30
from experiments.hcmc_03_frozen_calib_and_zeroshot import (
    HCMC_CLASSES_5,
    THETA_H,
    THETA_M,
    compute_fkg_grounding_matrix,
    encode_text_prompts,
    evaluate_score_matrix,
    fit_temperature,
    softmax_probs,
)
from experiments.hcmc_04_llm_gt_labeling import (
    CLIP_ID,
    NEGATIVE_CONTROLS,
    QWEN3_VL_ID,
    QWEN_14B_ID,
    extract_frames,
    qwen3vl_yesno_prob,
)
from experiments.hcmc_04b_vlm_clean_curation import prf_metrics
from experiments.hcmc_05_e2e_captioning_and_eval import (
    BASE_QWEN3B_SNAPSHOT,
    CHECKPOINT_DIR,
    SYSTEM_PROMPT,
    build_operational_prompt,
    build_prompt_crisp_kg,
    build_prompt_visual_only,
    generate_batch_captions,
)
from experiments.hcmc_uri_fkg_utils import (
    VISUAL_RELS,
    build_uri_safe_subgraph,
    load_uri_safe_fkg,
    precompute_uri_safe_subgraphs,
    select_uri_safe_quota_facts,
)
from experiments.vcgat_semantic_anchoring_engine import VCGATSemanticAnchoringEngine
from experiments.visual_anchoring_engine import format_fact_claim_sentence

HCMC_DIR = REPO_ROOT / "data" / "hcmc_benchmark"
HANOI_FKG_DIR = REPO_ROOT / "data" / "fkg_per_kb"
HCMC_FKG_DIR = REPO_ROOT / "data" / "fkg_per_kb_with_hcmc"

VCGAT_CKPT = REPO_ROOT / "experiments" / "features" / "best_scientific_deep_vc_gat.pt"
HANOI_VAL_JSONL = REPO_ROOT / "data" / "caption_dataset" / "generated_captions_val_vcgat.jsonl"
HANOI_TRAIN_COCO = REPO_ROOT / "data" / "unified_coco_dataset" / "annotations" / "train_coco.json"
HANOI_VAL_EMB_PKL = REPO_ROOT / "experiments" / "features" / "siglip2_train_val_embeddings.pkl"

RAW_MANIFEST_JSON = REPO_ROOT / "data" / "VIDEO_TEST_HCMC" / "hcmc_segments_manifest.json"
CLEAN_MANIFEST_JSON = REPO_ROOT / "data" / "VIDEO_TEST_HCMC_CLEAN" / "hcmc_clean_manifest.json"
CLEAN_BAL_ZS_JSONL = HCMC_DIR / "hcmc_clean_balanced_zeroshot.jsonl"
HCMC_KNOWLEDGE_JSON = HCMC_DIR / "hcmc_landmarks_knowledge_and_prompts.json"
EMB_1161_PATH = HCMC_DIR / "hcmc_siglip2_embeddings_1161.npy"
OLD_VLM_LABELS_JSON = HCMC_DIR / "hcmc_vlm_gt_labels_clean_180.json"

FROZEN_CFG_URI_PATH = HCMC_DIR / "frozen_zeroshot_config_urisafe.json"
URI_ZS_RESULTS_PATH = HCMC_DIR / "hcmc_uri_safe_zeroshot_results.json"
CLEAN_180_URI_ZS_JSONL = HCMC_DIR / "hcmc_clean_180_uri_safe_zeroshot.jsonl"
URI_VLM_LABELS_JSON = HCMC_DIR / "hcmc_vlm_gt_labels_clean_180_urisafe.json"
URI_VLM_PARTIAL_JSON = HCMC_DIR / "hcmc_vlm_gt_partial_clean_180_urisafe.json"
STD_GOLD_URI_JSONL = HCMC_DIR / "hcmc_clean_180_uri_std_gold.jsonl"
INDEP_GOLD_URI_JSONL = HCMC_DIR / "hcmc_clean_180_uri_indep_gold.jsonl"

OPEN35 = list(PARENT_CLASSES_30) + list(HCMC_CLASSES_5)
SYSTEM_ORDER = [
    "B1_Visual_Only",
    "B2_Crisp_KG_RAG_5way",
    "B3_Prompt_Only_Certainty_5way",
    "B4_Ours_No_VCGAT_5way",
    "Proposed_VCGAT_E2E_5way",
    "Proposed_VCGAT_E2E_35way",
]


def _cert_tier(mu_l: float) -> str:
    if mu_l >= THETA_H:
        return "HIGH"
    if mu_l >= THETA_M:
        return "MEDIUM"
    return "LOW"


# =============================================================================
# Phase A: URI-Safe Hanoi Val Calibration + HCMC Zero-Shot Evaluation
# =============================================================================
def run_phase_a_calibration_and_zeroshot(device: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    LOGGER.info("=" * 80)
    LOGGER.info("PHASE A: URI-SAFE HANOI VAL CALIBRATION + HCMC ZERO-SHOT (Z1-Z4)")
    LOGGER.info("=" * 80)

    vcgat_engine = VCGATSemanticAnchoringEngine(checkpoint_path=str(VCGAT_CKPT), device=device)
    sig_proc = vcgat_engine.processor
    sig_model = vcgat_engine.siglip_model
    dev_t = torch.device(device)

    # 1. Strictly Hanoi-only Validation Calibration
    rows = [json.loads(l) for l in HANOI_VAL_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen_vids = set()
    val_items = []
    for r in rows:
        if r["video_id"] not in seen_vids:
            seen_vids.add(r["video_id"])
            val_items.append(r)

    with open(HANOI_TRAIN_COCO, "r", encoding="utf-8") as f:
        vid2parent = {a["video_id"]: a["parent_landmark"] for a in json.load(f)["annotations"]}
    with open(HANOI_VAL_EMB_PKL, "rb") as f:
        val_emb_dict = pickle.load(f)

    val_embs = np.stack([val_emb_dict[it["video_id"]].astype(np.float32) for it in val_items], axis=0)
    val_embs = val_embs / np.linalg.norm(val_embs, axis=1, keepdims=True)
    val_gt = [vid2parent[it["video_id"]] for it in val_items]
    cls2idx30 = {c: i for i, c in enumerate(PARENT_CLASSES_30)}
    y_val = np.array([cls2idx30[g] for g in val_gt], dtype=np.int64)
    N_val = len(val_items)

    naive_30 = {c: [f"a photo of {c}"] for c in PARENT_CLASSES_30}
    proto_naive_30 = encode_text_prompts(naive_30, PARENT_CLASSES_30, sig_proc, sig_model, dev_t)
    proto_desc_30 = encode_text_prompts(LANDMARK_DESCRIPTIVE_PROMPTS, PARENT_CLASSES_30, sig_proc, sig_model, dev_t)
    sims_val_z1 = val_embs @ proto_naive_30.T
    sims_val_z2 = val_embs @ proto_desc_30.T

    hanoi_fkg_idx = load_uri_safe_fkg(HANOI_FKG_DIR)
    hanoi_subgraphs = precompute_uri_safe_subgraphs(list(PARENT_CLASSES_30), hanoi_fkg_idx, vcgat_engine)
    val_fkg_mats = compute_fkg_grounding_matrix(val_embs, list(PARENT_CLASSES_30), hanoi_subgraphs, vcgat_engine, top_k=5)

    mode_scale_grids = {
        "vcgat_prob_top5": [0.02, 0.05, 0.10, 0.15, 0.25, 0.40],
        "vcgat_logit_top5": [0.005, 0.01, 0.02, 0.03, 0.05],
        "vis_cos_top5": [0.05, 0.10, 0.25, 0.50, 0.75, 1.00],
    }

    z2_ev = evaluate_score_matrix(sims_val_z2, val_gt, list(PARENT_CLASSES_30))
    tau_z2, nll_z2, _ = fit_temperature(sims_val_z2, y_val)
    best_cfg = {
        "mode": "none", "alpha": 0.0,
        "micro_top1": z2_ev["micro_top1"], "mrr": z2_ev["mrr"],
        "tau_open": float(tau_z2), "nll_open": float(nll_z2),
        "scores": sims_val_z2,
    }
    sweep_results = []
    for mode_name, a_grid in mode_scale_grids.items():
        mat = val_fkg_mats[mode_name]
        for alpha in a_grid:
            comb = sims_val_z2 + alpha * mat
            ev = evaluate_score_matrix(comb, val_gt, list(PARENT_CLASSES_30))
            tau_c, nll_c, _ = fit_temperature(comb, y_val)
            sweep_results.append({
                "mode": mode_name, "alpha": alpha,
                "micro_top1": ev["micro_top1"], "macro_top1": ev["macro_top1"],
                "top5": ev["top5"], "mrr": ev["mrr"],
                "tau_open": round(tau_c, 6), "nll_open": round(nll_c, 4),
            })
            if (ev["micro_top1"], ev["mrr"], -nll_c) > (best_cfg["micro_top1"], best_cfg["mrr"], -best_cfg["nll_open"]):
                best_cfg = {
                    "mode": mode_name, "alpha": float(alpha),
                    "micro_top1": ev["micro_top1"], "mrr": ev["mrr"],
                    "tau_open": float(tau_c), "nll_open": float(nll_c),
                    "scores": comb,
                }

    rng = np.random.default_rng(42)
    pool_logits_5, pool_labels_5 = [], []
    best_scores_30 = best_cfg["scores"]
    for _ in range(500):
        i = int(rng.integers(0, N_val))
        gt_i = int(y_val[i])
        neg_pool = [c for c in range(len(PARENT_CLASSES_30)) if c != gt_i]
        cand_5 = [gt_i] + rng.choice(neg_pool, size=4, replace=False).tolist()
        pool_logits_5.append(best_scores_30[i, cand_5])
        pool_labels_5.append(0)
    tau_5_star, nll_5_star, _ = fit_temperature(np.stack(pool_logits_5), np.array(pool_labels_5, dtype=np.int64))

    frozen_cfg = {
        "calibrated_on": f"Hanoi Validation Set ONLY (N={N_val} unique clips, URI-safe FKG)",
        "zero_hcmc_test_leakage": True,
        "fkg_grounding_mode": best_cfg["mode"],
        "alpha_fkg_star": best_cfg["alpha"],
        "tau_5_star": round(float(tau_5_star), 6),
        "nll_5_star": round(float(nll_5_star), 4),
        "tau_35_star": round(float(best_cfg["tau_open"]), 6),
        "nll_30_star": round(float(best_cfg["nll_open"]), 4),
        "theta_M": THETA_M,
        "theta_H": THETA_H,
        "hanoi_val_z1_top1": evaluate_score_matrix(sims_val_z1, val_gt, list(PARENT_CLASSES_30))["micro_top1"],
        "hanoi_val_z2_top1": z2_ev["micro_top1"],
        "hanoi_val_z3_top1": best_cfg["micro_top1"],
        "hanoi_val_z3_mrr": best_cfg["mrr"],
        "sweep_results": sweep_results,
    }
    with open(FROZEN_CFG_URI_PATH, "w", encoding="utf-8") as f:
        json.dump(frozen_cfg, f, indent=2, ensure_ascii=False)
    LOGGER.info(
        "FROZEN URI-SAFE CONFIG (Hanoi Val): mode=%s, alpha=%.3f, tau_5=%.6f, tau_35=%.6f, Z3_Val_Top1=%.2f%%",
        frozen_cfg["fkg_grounding_mode"], frozen_cfg["alpha_fkg_star"],
        frozen_cfg["tau_5_star"], frozen_cfg["tau_35_star"], frozen_cfg["hanoi_val_z3_top1"],
    )

    # 2. Evaluate Zero-Shot on Extended URI-Safe Global FKG (35 Open-World Classes)
    hcmc_fkg_idx = load_uri_safe_fkg(HCMC_FKG_DIR)
    leak_terms = ["lý dynasty", "three holy mothers", "hanoi", "ga hà nội", "district 12"]
    for lm in HCMC_CLASSES_5:
        cands, _, _ = build_uri_safe_subgraph(lm, hcmc_fkg_idx)
        assert cands, f"Empty URI-safe subgraph for {lm}"
        assert not any(
            lk in (c["head"] + " " + c["tail"]).lower() for c in cands for lk in leak_terms
        ), f"Contamination detected in {lm}"

    with open(HCMC_KNOWLEDGE_JSON, "r", encoding="utf-8") as f:
        hcmc_know = json.load(f)
    desc_35 = {**LANDMARK_DESCRIPTIVE_PROMPTS, **hcmc_know["descriptive_prompts"]}
    naive_35 = {c: [f"a photo of {c}"] for c in OPEN35}
    p_naive_35 = encode_text_prompts(naive_35, OPEN35, sig_proc, sig_model, dev_t)
    p_desc_35 = encode_text_prompts(desc_35, OPEN35, sig_proc, sig_model, dev_t)

    embs_1161 = np.load(EMB_1161_PATH)
    sims_z1_35 = embs_1161 @ p_naive_35.T
    sims_z2_35 = embs_1161 @ p_desc_35.T

    subgraphs_35 = precompute_uri_safe_subgraphs(OPEN35, hcmc_fkg_idx, vcgat_engine)
    alpha_star = float(frozen_cfg["alpha_fkg_star"])
    mode_star = frozen_cfg["fkg_grounding_mode"]
    tau_5 = float(frozen_cfg["tau_5_star"])
    tau_35 = float(frozen_cfg["tau_35_star"])

    if mode_star != "none":
        fkg_mats_35 = compute_fkg_grounding_matrix(embs_1161, OPEN35, subgraphs_35, vcgat_engine, top_k=5)
        scores_z3_35 = sims_z2_35 + alpha_star * fkg_mats_35[mode_star]
        hanoi_fkg_mats_35 = compute_fkg_grounding_matrix(val_embs, OPEN35, subgraphs_35, vcgat_engine, top_k=5)
        hanoi_z3_35 = (val_embs @ p_desc_35.T) + alpha_star * hanoi_fkg_mats_35[mode_star]
    else:
        scores_z3_35 = sims_z2_35
        hanoi_z3_35 = val_embs @ p_desc_35.T

    ood_bias_35 = hanoi_z3_35.mean(axis=0, keepdims=True)
    scores_z4_35 = scores_z3_35 - ood_bias_35

    del vcgat_engine
    gc.collect()
    torch.cuda.empty_cache()

    raw_segs = json.loads(RAW_MANIFEST_JSON.read_text(encoding="utf-8"))["segments"]
    clean_segs = json.loads(CLEAN_MANIFEST_JSON.read_text(encoding="utf-8"))["segments"]
    clean_indices = [s["clip_idx"] for s in clean_segs]
    clean_bal_rows = [json.loads(l) for l in CLEAN_BAL_ZS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    clean_bal_indices = [r["clip_idx"] for r in clean_bal_rows]
    gt_all = [s["landmark_en"] for s in raw_segs]

    def _eval_split(indices: List[int], name: str) -> Dict[str, Any]:
        gt = [gt_all[i] for i in indices]
        s5 = {
            "Z1_Naive_SigLIP2": sims_z1_35[indices, 30:35],
            "Z2_Descriptive_SigLIP2": sims_z2_35[indices, 30:35],
            "Z3_Proposed_Desc_FKG": scores_z3_35[indices, 30:35],
            "Z4_Proposed_FKG_OOD_Centered": scores_z4_35[indices, 30:35],
        }
        s35 = {
            "Z1_Naive_SigLIP2": sims_z1_35[indices],
            "Z2_Descriptive_SigLIP2": sims_z2_35[indices],
            "Z3_Proposed_Desc_FKG": scores_z3_35[indices],
            "Z4_Proposed_FKG_OOD_Centered": scores_z4_35[indices],
        }
        res5 = {}
        for k, m in s5.items():
            d = evaluate_score_matrix(m, gt, HCMC_CLASSES_5, tau=tau_5)
            d.update(prf_metrics(m, gt, HCMC_CLASSES_5))
            res5[k] = d
        res35 = {k: evaluate_score_matrix(m, gt, OPEN35, tau=tau_35) for k, m in s35.items()}
        LOGGER.info("[%s | n=%d] 5-way Z4 R@1=%.2f%% MacroF1=%.2f%% MRR=%.4f | 35-way Z4 R@1=%.2f%% Top5=%.2f%%",
                    name, len(indices),
                    res5["Z4_Proposed_FKG_OOD_Centered"]["micro_top1"],
                    res5["Z4_Proposed_FKG_OOD_Centered"]["macro_f1"],
                    res5["Z4_Proposed_FKG_OOD_Centered"]["mrr"],
                    res35["Z4_Proposed_FKG_OOD_Centered"]["micro_top1"],
                    res35["Z4_Proposed_FKG_OOD_Centered"]["top5"])
        return {"hcmc_5way": res5, "openworld_35way": res35}

    zs_report = {
        "frozen_config": frozen_cfg,
        "uri_safe_subgraph_sizes": {lm: len(subgraphs_35[lm]["cands"]) for lm in HCMC_CLASSES_5},
        "raw_all_1161": _eval_split(list(range(len(raw_segs))), "RAW_ALL_1161"),
        "clean_all": _eval_split(clean_indices, "VLM_CLEAN_ALL"),
        "clean_balanced": _eval_split(clean_bal_indices, "VLM_CLEAN_BALANCED"),
    }
    with open(URI_ZS_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(zs_report, f, indent=2, ensure_ascii=False)

    probs_z3_5 = softmax_probs(scores_z3_35[:, 30:35], tau_5)
    probs_z3_35 = softmax_probs(scores_z3_35, tau_35)
    probs_z4_5 = softmax_probs(scores_z4_35[:, 30:35], tau_5)
    probs_z4_35 = softmax_probs(scores_z4_35, tau_35)

    updated_bal_rows = []
    for row in clean_bal_rows:
        idx = int(row["clip_idx"])
        r = dict(row)
        p3_5, p3_35 = int(np.argmax(probs_z3_5[idx])), int(np.argmax(probs_z3_35[idx]))
        p4_5, p4_35 = int(np.argmax(probs_z4_5[idx])), int(np.argmax(probs_z4_35[idx]))
        mu3_5, mu3_35 = float(probs_z3_5[idx, p3_5]), float(probs_z3_35[idx, p3_35])
        mu4_5, mu4_35 = float(probs_z4_5[idx, p4_5]), float(probs_z4_35[idx, p4_35])
        r.update({
            "z3_pred_5way": HCMC_CLASSES_5[p3_5],
            "z3_mu_L_5way": round(mu3_5, 4),
            "z3_cert_5way": _cert_tier(mu3_5),
            "z3_pred_35way": OPEN35[p3_35],
            "z3_mu_L_35way": round(mu3_35, 4),
            "z3_cert_35way": _cert_tier(mu3_35),
            "z4_pred_5way": HCMC_CLASSES_5[p4_5],
            "z4_mu_L_5way": round(mu4_5, 4),
            "z4_cert_5way": _cert_tier(mu4_5),
            "z4_pred_35way": OPEN35[p4_35],
            "z4_mu_L_35way": round(mu4_35, 4),
            "z4_cert_35way": _cert_tier(mu4_35),
        })
        updated_bal_rows.append(r)

    with open(CLEAN_180_URI_ZS_JSONL, "w", encoding="utf-8") as f:
        for r in updated_bal_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    LOGGER.info("Saved URI-safe 180 balanced zeroshot -> %s", CLEAN_180_URI_ZS_JSONL)
    return updated_bal_rows, hcmc_fkg_idx


# =============================================================================
# Phase B: Incremental Multi-VLM Probing + Dual URI-Safe Ground Truth
# =============================================================================
def collect_uri_safe_visual_targets(
    hcmc_fkg_idx: Dict[str, Any],
    landmarks: List[str],
    max_targets: int = 16,
) -> Dict[str, List[str]]:
    out = {}
    for lm in landmarks:
        cands, _, _ = build_uri_safe_subgraph(lm, hcmc_fkg_idx)
        vis_cands = [f for f in cands if f["relation"].lower() in VISUAL_RELS]
        vis_cands.sort(key=lambda x: float(x.get("mu_F", 0.5)), reverse=True)
        seen = set()
        targets = []
        for f in vis_cands:
            t_raw = f["tail"].strip()
            t_low = t_raw.lower()
            if len(t_low) < 3 or t_low in seen:
                continue
            seen.add(t_low)
            targets.append(t_raw)
            if len(targets) >= max_targets:
                break
        out[lm] = targets
    return out


def incremental_vlm_probe(
    samples: List[Dict[str, Any]],
    lm_targets: Dict[str, List[str]],
    device: str,
) -> Dict[str, Dict[str, Any]]:
    LOGGER.info("=" * 80)
    LOGGER.info("PHASE B.1: INCREMENTAL MULTI-VLM VISUAL PROBING ON CLEAN URI-SAFE TARGETS")
    LOGGER.info("=" * 80)
    if URI_VLM_LABELS_JSON.exists():
        data = json.loads(URI_VLM_LABELS_JSON.read_text(encoding="utf-8"))
        all_ok = all(
            all(t in data.get(s["sample_id"], {}).get("visual_targets_evaluated", {}) for t in lm_targets[s["landmark_gt"]])
            for s in samples
        )
        if all_ok and len(data) == len(samples):
            LOGGER.info("Loaded complete URI-safe Multi-VLM labels from %s", URI_VLM_LABELS_JSON)
            return data

    seed_vlm = json.loads(OLD_VLM_LABELS_JSON.read_text(encoding="utf-8")) if OLD_VLM_LABELS_JSON.exists() else {}
    partial = json.loads(URI_VLM_PARTIAL_JSON.read_text(encoding="utf-8")) if URI_VLM_PARTIAL_JSON.exists() else {}

    # Merge cached probabilities from old clean_180 labels
    work_records: Dict[str, Dict[str, Any]] = {}
    missing_qwen: List[Tuple[Dict[str, Any], List[str]]] = []
    for s in samples:
        sid = s["sample_id"]
        gt_lm = s["landmark_gt"]
        rec = dict(partial.get(sid) or seed_vlm.get(sid) or {})
        rec.setdefault("sample_id", sid)
        rec.setdefault("clip_idx", s["clip_idx"])
        rec.setdefault("landmark", gt_lm)
        rec.setdefault("segment_path", s["segment_path"])
        t_probs = dict(rec.get("target_p_qwen", {}))
        vt_old = rec.get("visual_targets_evaluated", {})
        for k, v in vt_old.items():
            if k not in t_probs and "p_qwen" in v:
                t_probs[k] = float(v["p_qwen"])
        rec["target_p_qwen"] = t_probs
        todo = [t for t in lm_targets[gt_lm] if t not in t_probs]
        if todo:
            missing_qwen.append((s, todo))
        work_records[sid] = rec

    total_new_queries = sum(len(td) for _, td in missing_qwen)
    LOGGER.info("Clips needing incremental Qwen3-VL target probes: %d/%d (%d total target queries)",
                len(missing_qwen), len(samples), total_new_queries)

    if missing_qwen:
        processor = AutoProcessor.from_pretrained(QWEN3_VL_ID, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            QWEN3_VL_ID, torch_dtype=torch.bfloat16, device_map=device, local_files_only=True
        ).eval()
        tok = processor.tokenizer
        yes_ids = sorted({tok.encode(w, add_special_tokens=False)[-1] for w in ["Yes", "yes", "YES", " Yes", " yes"]})
        no_ids = sorted({tok.encode(w, add_special_tokens=False)[-1] for w in ["No", "no", "NO", " No", " no"]})

        t0 = time.time()
        for idx, (s, todo) in enumerate(missing_qwen):
            sid = s["sample_id"]
            frames = extract_frames(s["segment_path"], num_frames=8)
            dummy = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": "hi"}]}]
            img_in, vid_in = process_vision_info(dummy)
            t_probs = work_records[sid]["target_p_qwen"]
            for tgt in todo:
                q = f"Is '{tgt}' visually visible in this video?"
                t_probs[tgt] = round(qwen3vl_yesno_prob(model, processor, img_in, vid_in, frames, q, yes_ids, no_ids, device), 4)
            if (idx + 1) % 25 == 0 or (idx + 1) == len(missing_qwen):
                LOGGER.info("  Incremental Qwen3-VL [%d/%d clips] (%.1fs)", idx + 1, len(missing_qwen), time.time() - t0)
                with open(URI_VLM_PARTIAL_JSON, "w", encoding="utf-8") as f:
                    json.dump(work_records, f, indent=2, ensure_ascii=False)

        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    # Pass B: CLIP ViT-B/32 for clean targets & consensus
    clip_proc = CLIPProcessor.from_pretrained(CLIP_ID, local_files_only=True)
    clip_model = CLIPModel.from_pretrained(CLIP_ID, local_files_only=True).to(device).eval()

    all_terms = sorted({t for tl in lm_targets.values() for t in tl} | set(NEGATIVE_CONTROLS))
    term_embeds = {}
    with torch.no_grad():
        for term in all_terms:
            texts = [f"a photo showing {term}", f"a generic street or room scene without {term}"]
            inp = clip_proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
            t_out = clip_model.get_text_features(**inp)
            t_emb = t_out.pooler_output if hasattr(t_out, "pooler_output") else t_out
            term_embeds[term] = t_emb / t_emb.norm(dim=-1, keepdim=True)
        logit_scale = clip_model.logit_scale.exp().item()

    for s in samples:
        sid = s["sample_id"]
        gt_lm = s["landmark_gt"]
        rec = work_records[sid]
        vt_old = rec.get("visual_targets_evaluated", {})
        targets = lm_targets[gt_lm]
        need_clip = [t for t in targets if t not in vt_old]
        vis_eval = {}
        if need_clip:
            frames = extract_frames(s["segment_path"], num_frames=8)
            inp_img = clip_proc(images=frames, return_tensors="pt").to(device)
            with torch.no_grad():
                i_out = clip_model.get_image_features(**inp_img)
                img_feats = i_out.pooler_output if hasattr(i_out, "pooler_output") else i_out
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            for tgt in targets:
                if tgt in vt_old:
                    vis_eval[tgt] = vt_old[tgt]
                else:
                    p_q = float(rec["target_p_qwen"][tgt])
                    logits = logit_scale * (img_feats @ term_embeds[tgt].t())
                    p_c = float(torch.softmax(logits, dim=-1)[:, 0].max().item())
                    vis_r = 1.0 if (p_q >= 0.60 and p_c >= 0.45) else (0.0 if (p_q <= 0.25 and p_c <= 0.35) else None)
                    vis_eval[tgt] = {"p_qwen": p_q, "p_clip": round(p_c, 4), "visibility_R": vis_r}
        else:
            for tgt in targets:
                vis_eval[tgt] = vt_old[tgt]
        rec["visual_targets_evaluated"] = vis_eval
        rec["is_vlm_verified_onsite"] = bool(rec["p_qwen_onsite"] >= 0.50)

    del clip_model, clip_proc
    gc.collect()
    torch.cuda.empty_cache()

    with open(URI_VLM_LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(work_records, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved URI-safe Multi-VLM labels -> %s", URI_VLM_LABELS_JSON)
    return work_records


def _select_independent_uri_gold_facts(
    landmark: str,
    quota: Tuple[int, int, int],
    vt_map: Dict[str, Dict[str, Any]],
    hcmc_fkg_idx: Dict[str, Any],
) -> List[Dict[str, Any]]:
    k_vis, k_dir, k_inf = quota
    cands, _, _ = build_uri_safe_subgraph(landmark, hcmc_fkg_idx)
    raw_vis, raw_dir, raw_inf = [], [], []
    seen_k = set()
    for f in cands:
        r = f["relation"].lower()
        t = f["tail"].strip()
        key = (r, t.lower())
        if key in seen_k:
            continue
        seen_k.add(key)
        if r in VISUAL_RELS:
            vt = vt_map.get(t, {})
            vis_r = vt.get("visibility_R", None)
            p_q = float(vt.get("p_qwen", 0.0))
            p_c = float(vt.get("p_clip", 0.0))
            tier = 2 if vis_r == 1.0 else (1 if vis_r is None and t in vt_map else (0 if t not in vt_map else -1))
            score = tier * 10.0 + (0.5 * p_q + 0.5 * p_c) + 0.1 * float(f.get("mu_F", 0.5))
            raw_vis.append({**f, "category": "visual", "_rank": score, "visibility_R": vis_r, "p_qwen": p_q, "p_clip": p_c})
        elif r in ("located_in", "founded_by", "founded_in", "built_by", "constructed_in", "symbol_of",
                   "renamed_to", "named_after", "built_in", "dated_to", "used_as", "associated_dynasty",
                   "dynasty", "history", "emperor", "established_by", "historical_period", "origin",
                   "reign", "country", "province", "postal_code", "ruled_by", "commissioned_by", "designed_by",
                   "renovated_in"):
            raw_dir.append({**f, "category": "direct", "_rank": float(f.get("mu_F", 0.5))})
        elif f.get("is_inferred", False) or r in ("belongs_to_dynasty", "ruled_by", "architectural_style",
                                                  "has_style", "worships", "dedicated_to", "associated_with",
                                                  "commemorates", "features_style", "incorporates_material"):
            raw_inf.append({**f, "category": "inferred", "_rank": float(f.get("mu_F", 0.5))})

    raw_vis.sort(key=lambda x: x["_rank"], reverse=True)
    raw_dir.sort(key=lambda x: x["_rank"], reverse=True)
    raw_inf.sort(key=lambda x: x["_rank"], reverse=True)
    chosen = []
    for pool, k in ((raw_vis, k_vis), (raw_dir, k_dir), (raw_inf, k_inf)):
        for it in pool[:k]:
            chosen.append({kk: vv for kk, vv in it.items() if kk != "_rank"})
    return chosen


def _synthesize_refs_qwen14b_multi(tasks: List[Tuple[List[Dict[str, Any]], bool, Path]]) -> None:
    """Loads Qwen2.5-14B-Instruct (8-bit) once and synthesizes reference captions for pending gold files."""
    if not tasks:
        return
    LOGGER.info("Loading %s (8-bit) once for %d gold dataset(s)...", QWEN_14B_ID, len(tasks))
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained(QWEN_14B_ID, local_files_only=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        QWEN_14B_ID, quantization_config=bnb_cfg, device_map="auto", torch_dtype=torch.float16, local_files_only=True
    ).eval()

    sys_prompt = (
        "You are a strictly faithful cultural heritage video narrator. "
        "Write a cohesive, accurate narration matching the exact target sentence and word counts.\n\n"
        "RULES:\n"
        "1. Mention ONLY facts given in VERIFIED GOLD FACTS or visually visible in SCENE DESCRIPTION.\n"
        "2. Do NOT add dates, architects, or claims that are not provided.\n"
        "3. No promotional filler or viewer invitations."
    )
    batch_size = 8
    for records, include_sg, out_path in tasks:
        t0 = time.time()
        LOGGER.info("Synthesizing %d reference captions -> %s (include_sg=%s)", len(records), out_path.name, include_sg)
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            prompts = []
            max_new = max(r["max_new_tokens"] for r in batch) + 20
            for r in batch:
                gt_lm = r["landmark_gt"]
                facts_str = "\n".join(
                    f"- {format_fact_claim_sentence(f.get('head', gt_lm), f['relation'], f['tail'])}"
                    for f in r["gold_facts"]
                ) or "- None"
                sg_block = f"OBSERVED CONCEPTS:\n{r['observed_scene_graph']}\n" if include_sg else ""
                user_msg = (
                    f"LANDMARK: {gt_lm}\n"
                    f"SCENE DESCRIPTION: {r['vlm_visual_description']}\n"
                    f"{sg_block}"
                    f"VERIFIED GOLD FACTS:\n{facts_str}\n\n"
                    f"INSTRUCTION: Write a cohesive {r['target_sentences']} narration paragraph "
                    f"(approx. {r['target_words']}). Output ONLY the narration paragraph."
                )
                msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}]
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1300).to(llm.device)
            p_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = llm.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, rec in enumerate(batch):
                rec["reference_caption"] = tok.decode(out[j, p_len:], skip_special_tokens=True).strip()
            if (start // batch_size + 1) % 6 == 0 or (start + batch_size) >= len(records):
                LOGGER.info("  Refs [%d/%d] (%.1fs)", min(start + batch_size, len(records)), len(records), time.time() - t0)

        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        LOGGER.info("Saved -> %s", out_path)

    del llm, tok
    gc.collect()
    torch.cuda.empty_cache()


def run_phase_b_dual_gt(
    samples: List[Dict[str, Any]],
    vlm_labels: Dict[str, Dict[str, Any]],
    hcmc_fkg_idx: Dict[str, Any],
    device: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    LOGGER.info("=" * 80)
    LOGGER.info("PHASE B.2: DUAL URI-SAFE GOLD DATASETS (STD-GT + 100% INDEP-GT)")
    LOGGER.info("=" * 80)

    std_ready = STD_GOLD_URI_JSONL.exists() and len(STD_GOLD_URI_JSONL.read_text(encoding="utf-8").splitlines()) == len(samples)
    indep_ready = INDEP_GOLD_URI_JSONL.exists() and len(INDEP_GOLD_URI_JSONL.read_text(encoding="utf-8").splitlines()) == len(samples)

    if std_ready and indep_ready:
        std_recs = [json.loads(l) for l in STD_GOLD_URI_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
        indep_recs = [json.loads(l) for l in INDEP_GOLD_URI_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
        LOGGER.info("Loaded cached URI-safe Std-GT & Indep-GT datasets.")
        return std_recs, indep_recs

    embs_1161 = np.load(EMB_1161_PATH)
    vcgat_engine = VCGATSemanticAnchoringEngine(checkpoint_path=str(VCGAT_CKPT), device=device)
    subgraphs_5 = precompute_uri_safe_subgraphs(list(HCMC_CLASSES_5), hcmc_fkg_idx, vcgat_engine)

    std_recs, indep_recs = [], []
    for s in samples:
        sid = s["sample_id"]
        gt_lm = s["landmark_gt"]
        v_vec = embs_1161[s["clip_idx"]]
        quota = tuple(s["fact_quota"])
        vl = vlm_labels[sid]
        vt_map = vl["visual_targets_evaluated"]

        gf_std = select_uri_safe_quota_facts(
            vcgat_engine, v_vec, gt_lm, 1.0, "HIGH", hcmc_fkg_idx,
            quota=quota, anchored=True, precomputed_subgraph=subgraphs_5.get(gt_lm),
        )
        for gf in gf_std:
            tail = gf.get("tail", "")
            if tail in vt_map:
                gf["visibility_R"] = vt_map[tail]["visibility_R"]
                gf["p_qwen"] = vt_map[tail]["p_qwen"]
                gf["p_clip"] = vt_map[tail]["p_clip"]

        gf_indep = _select_independent_uri_gold_facts(gt_lm, quota, vt_map, hcmc_fkg_idx)

        base = dict(s)
        base["vlm_visual_description"] = vl["vlm_visual_description"]
        base["p_qwen_onsite"] = vl["p_qwen_onsite"]
        base["is_vlm_verified_onsite"] = vl["is_vlm_verified_onsite"]
        base["visual_targets_evaluated"] = vt_map

        r_std = dict(base); r_std["gold_facts"] = gf_std
        r_ind = dict(base); r_ind["gold_facts"] = gf_indep
        std_recs.append(r_std)
        indep_recs.append(r_ind)

    del vcgat_engine
    gc.collect()
    torch.cuda.empty_cache()

    tasks = []
    if not std_ready:
        tasks.append((std_recs, True, STD_GOLD_URI_JSONL))
    else:
        std_recs = [json.loads(l) for l in STD_GOLD_URI_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not indep_ready:
        tasks.append((indep_recs, False, INDEP_GOLD_URI_JSONL))
    else:
        indep_recs = [json.loads(l) for l in INDEP_GOLD_URI_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

    _synthesize_refs_qwen14b_multi(tasks)
    return std_recs, indep_recs


# =============================================================================
# Phase C-Gen: Strictly Label-Free E2E Generation (z4 & z3) with Exact Metadata
# =============================================================================
def run_phase_c_generation(
    samples: List[Dict[str, Any]],
    hcmc_fkg_idx: Dict[str, Any],
    device: str,
    prefixes: List[str] = ["z4", "z3"],
) -> None:
    LOGGER.info("=" * 80)
    LOGGER.info("PHASE C-GEN: STRICTLY LABEL-FREE E2E CAPTIONING (%s)", prefixes)
    LOGGER.info("=" * 80)

    need_prefixes = []
    for pfx in prefixes:
        out_p = HCMC_DIR / f"hcmc_clean_180_uri_e2e_captions_{pfx}.jsonl"
        if out_p.exists() and len(out_p.read_text(encoding="utf-8").splitlines()) == len(samples):
            first = json.loads(out_p.read_text(encoding="utf-8").splitlines()[0])
            if "facts_given" in first and "prompt_regime" in first:
                LOGGER.info("Skipping %s (already cached with facts_given at %s)", pfx, out_p)
                continue
        need_prefixes.append(pfx)
    if not need_prefixes:
        return

    embs_1161 = np.load(EMB_1161_PATH)
    vcgat_engine = VCGATSemanticAnchoringEngine(checkpoint_path=str(VCGAT_CKPT), device=device)
    subgraphs_35 = precompute_uri_safe_subgraphs(OPEN35, hcmc_fkg_idx, vcgat_engine)

    # Prepare strict label-free prompts & facts_given for all requested prefixes
    prep: Dict[str, Dict[str, Any]] = {}
    for pfx in need_prefixes:
        prompts_map: Dict[str, List[str]] = defaultdict(list)
        facts_map: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(list)
        regime_map: Dict[str, List[str]] = defaultdict(list)
        max_toks = []
        for s in samples:
            v_vec = embs_1161[s["clip_idx"]]
            quota = tuple(s["fact_quota"])
            sg_text = s["observed_scene_graph"]
            tgt_s, tgt_w = s["target_sentences"], s["target_words"]
            max_toks.append(int(s["max_new_tokens"]))

            pred5 = s[f"{pfx}_pred_5way"]
            mu5 = float(s[f"{pfx}_mu_L_5way"])
            pred35 = s[f"{pfx}_pred_35way"]
            mu35 = float(s[f"{pfx}_mu_L_35way"])

            # B1: Visual Only
            prompts_map["B1_Visual_Only"].append(build_prompt_visual_only(sg_text, tgt_s, tgt_w))
            facts_map["B1_Visual_Only"].append([])
            regime_map["B1_Visual_Only"].append("LOW")

            # B2: Crisp KG-RAG (always injects static facts under predicted 5-way landmark)
            f_crisp = select_uri_safe_quota_facts(
                vcgat_engine, v_vec, pred5, 1.0, "HIGH", hcmc_fkg_idx,
                quota=quota, anchored=False, precomputed_subgraph=subgraphs_35.get(pred5),
            )
            prompts_map["B2_Crisp_KG_RAG_5way"].append(build_prompt_crisp_kg(pred5, sg_text, f_crisp, tgt_s, tgt_w))
            facts_map["B2_Crisp_KG_RAG_5way"].append(f_crisp)
            regime_map["B2_Crisp_KG_RAG_5way"].append("CRISP")

            cert5 = _cert_tier(mu5)
            f_vcgat5 = (
                select_uri_safe_quota_facts(
                    vcgat_engine, v_vec, pred5, mu5, cert5, hcmc_fkg_idx,
                    quota=quota, anchored=True, precomputed_subgraph=subgraphs_35.get(pred5),
                )
                if mu5 >= THETA_M else []
            )
            f_unanc5 = (
                select_uri_safe_quota_facts(
                    vcgat_engine, v_vec, pred5, mu5, cert5, hcmc_fkg_idx,
                    quota=quota, anchored=False, precomputed_subgraph=subgraphs_35.get(pred5),
                )
                if mu5 >= THETA_M else []
            )

            # B3: Prompt-Only Certainty (same VC-GAT facts + certainty prompt, untuned base 3B)
            p_b3, rg_b3 = build_operational_prompt(pred5, mu5, sg_text, f_vcgat5, tgt_s, tgt_w)
            prompts_map["B3_Prompt_Only_Certainty_5way"].append(p_b3)
            facts_map["B3_Prompt_Only_Certainty_5way"].append(f_vcgat5)
            regime_map["B3_Prompt_Only_Certainty_5way"].append(rg_b3)

            # B4: Ours No VC-GAT (LoRA 3B + unanchored static facts)
            p_b4, rg_b4 = build_operational_prompt(pred5, mu5, sg_text, f_unanc5, tgt_s, tgt_w)
            prompts_map["B4_Ours_No_VCGAT_5way"].append(p_b4)
            facts_map["B4_Ours_No_VCGAT_5way"].append(f_unanc5)
            regime_map["B4_Ours_No_VCGAT_5way"].append(rg_b4)

            # Proposed 5-way
            p_p5, rg_p5 = build_operational_prompt(pred5, mu5, sg_text, f_vcgat5, tgt_s, tgt_w)
            prompts_map["Proposed_VCGAT_E2E_5way"].append(p_p5)
            facts_map["Proposed_VCGAT_E2E_5way"].append(f_vcgat5)
            regime_map["Proposed_VCGAT_E2E_5way"].append(rg_p5)

            # Proposed 35-way Open-World
            cert35 = _cert_tier(mu35)
            f_vcgat35 = (
                select_uri_safe_quota_facts(
                    vcgat_engine, v_vec, pred35, mu35, cert35, hcmc_fkg_idx,
                    quota=quota, anchored=True, precomputed_subgraph=subgraphs_35.get(pred35),
                )
                if mu35 >= THETA_M else []
            )
            p_p35, rg_p35 = build_operational_prompt(pred35, mu35, sg_text, f_vcgat35, tgt_s, tgt_w)
            prompts_map["Proposed_VCGAT_E2E_35way"].append(p_p35)
            facts_map["Proposed_VCGAT_E2E_35way"].append(f_vcgat35)
            regime_map["Proposed_VCGAT_E2E_35way"].append(rg_p35)

        prep[pfx] = {
            "prompts": prompts_map, "facts": facts_map,
            "regimes": regime_map, "max_toks": max_toks,
            "captions": {},
        }

    del vcgat_engine
    gc.collect()
    torch.cuda.empty_cache()

    # Load Base Qwen2.5-3B-Instruct once
    tok = AutoTokenizer.from_pretrained(str(BASE_QWEN3B_SNAPSHOT), local_files_only=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_QWEN3B_SNAPSHOT), torch_dtype=torch.bfloat16, device_map=device, local_files_only=True
    ).eval()

    # Generate B1 once + B2/B3 for each prefix on base model
    b1_caps = generate_batch_captions(
        base_model, tok, prep[need_prefixes[0]]["prompts"]["B1_Visual_Only"],
        prep[need_prefixes[0]]["max_toks"], batch_size=16, device=device,
    )
    for pfx in need_prefixes:
        prep[pfx]["captions"]["B1_Visual_Only"] = b1_caps
        for sys_name in ["B2_Crisp_KG_RAG_5way", "B3_Prompt_Only_Certainty_5way"]:
            LOGGER.info("Generating %s [%s] (Base Qwen2.5-3B)...", pfx, sys_name)
            prep[pfx]["captions"][sys_name] = generate_batch_captions(
                base_model, tok, prep[pfx]["prompts"][sys_name], prep[pfx]["max_toks"], batch_size=16, device=device,
            )

    # Wrap with Frozen LoRA for B4 & Proposed
    lora_model = PeftModel.from_pretrained(base_model, str(CHECKPOINT_DIR), local_files_only=True).eval()
    for pfx in need_prefixes:
        for sys_name in ["B4_Ours_No_VCGAT_5way", "Proposed_VCGAT_E2E_5way", "Proposed_VCGAT_E2E_35way"]:
            LOGGER.info("Generating %s [%s] (Frozen LoRA)...", pfx, sys_name)
            prep[pfx]["captions"][sys_name] = generate_batch_captions(
                lora_model, tok, prep[pfx]["prompts"][sys_name], prep[pfx]["max_toks"], batch_size=16, device=device,
            )

    del lora_model, base_model, tok
    gc.collect()
    torch.cuda.empty_cache()

    for pfx in need_prefixes:
        out_path = HCMC_DIR / f"hcmc_clean_180_uri_e2e_captions_{pfx}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for i, s in enumerate(samples):
                row = {
                    "sample_id": s["sample_id"],
                    "clip_idx": s["clip_idx"],
                    "youtube_id": s["youtube_id"],
                    "segment_path": s["segment_path"],
                    "landmark_gt": s["landmark_gt"],
                    "is_vlm_verified_onsite": s["is_vlm_verified_onsite"],
                    "duration_tier": s["duration_tier"],
                    "fact_quota": s["fact_quota"],
                    "observed_scene_graph": s["observed_scene_graph"],
                    "vlm_visual_description": s["vlm_visual_description"],
                    f"{pfx}_pred_5way": s[f"{pfx}_pred_5way"],
                    f"{pfx}_mu_L_5way": s[f"{pfx}_mu_L_5way"],
                    f"{pfx}_cert_5way": s[f"{pfx}_cert_5way"],
                    f"{pfx}_pred_35way": s[f"{pfx}_pred_35way"],
                    f"{pfx}_mu_L_35way": s[f"{pfx}_mu_L_35way"],
                    f"{pfx}_cert_35way": s[f"{pfx}_cert_35way"],
                    "captions": {sn: prep[pfx]["captions"][sn][i] for sn in SYSTEM_ORDER},
                    "facts_given": {
                        sn: [
                            {
                                "head": ft.get("head", ""),
                                "relation": ft.get("relation", ""),
                                "tail": ft.get("tail", ""),
                                "mu_F": float(ft.get("mu_F", 0.0)),
                                "c_f": float(ft.get("c_f", ft.get("mu_F", 0.0))),
                                "category": ft.get("category", ""),
                                "vcgat_score": ft.get("vcgat_score"),
                                "siglip_similarity": ft.get("siglip_similarity"),
                                "is_inferred": bool(ft.get("is_inferred", False)),
                            }
                            for ft in prep[pfx]["facts"][sn][i]
                        ]
                        for sn in SYSTEM_ORDER
                    },
                    "prompt_regime": {sn: prep[pfx]["regimes"][sn][i] for sn in SYSTEM_ORDER},
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        LOGGER.info("Saved URI-safe E2E captions -> %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", nargs="+", default=["z4", "z3"], choices=["z4", "z3"])
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    t_start = time.time()

    bal_rows, hcmc_fkg_idx = run_phase_a_calibration_and_zeroshot(device=device)
    lm_targets = collect_uri_safe_visual_targets(hcmc_fkg_idx, list(HCMC_CLASSES_5), max_targets=16)
    vlm_labels = incremental_vlm_probe(bal_rows, lm_targets, device=device)
    std_recs, _indep_recs = run_phase_b_dual_gt(bal_rows, vlm_labels, hcmc_fkg_idx, device=device)
    run_phase_c_generation(std_recs, hcmc_fkg_idx, device=device, prefixes=args.prefixes)

    LOGGER.info("HCMC Stage 7 URI-Safe Rebuild & E2E Generation COMPLETE in %.1fs.", time.time() - t_start)


if __name__ == "__main__":
    main()
