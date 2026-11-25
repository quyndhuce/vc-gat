#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_phase1_pipeline.py
=============================================================================
Phase 1: End-to-End Captioning Evaluation — Hanoi In-Domain & HCMC Out-of-Domain.
Executes:
  Step 4 — Hanoi Test (500 clips, E1 to E6)
  Step 5 — HCMC OOD Test (300 clips, E1 to E6)
  Step 6 — Cross-Region Generalization Gap (Delta_OOD with 10,000 cluster bootstrap)
  Step 7 — Generator Analysis (E5 Base vs E6 LoRA)
  Step 8 — Artifact & Report Generation
=============================================================================
"""

import os
import sys
import re
import json
import time
import math
import logging
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
LOGGER = logging.getLogger("Phase1Pipeline")

# Repo paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.phase1.evaluation_engine import (
    split_sentences,
    classify_claim_type,
    is_hedged,
    verify_historical_claim,
    verify_visual_claim,
    compute_clip_metrics,
    cluster_bootstrap_ci
)

PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
CONFIG_PATH = PHASE1_DIR / "config" / "frozen_run_config.json"
HANOI_MANIFEST = PHASE1_DIR / "config" / "hanoi_manifest.jsonl"
HCMC_MANIFEST = PHASE1_DIR / "config" / "hcmc_manifest.jsonl"

FGOLD_PATH = REPO_ROOT / "scene_conditioned_gold" / "outputs" / "fgold_test_800_clips.json"
V2_FACTS_PATH = REPO_ROOT / "experiments" / "outputs" / "fkg_facts_verified_v2.json"
if not V2_FACTS_PATH.exists():
    V2_FACTS_PATH = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "fused_facts_dst_v2.json"
HN_SCENE_GRAPHS_PATH = REPO_ROOT / "results" / "two_level_silver_reference_train_val" / "test_500_canonical_scene_graphs.json"
HCMC_300_DATASET_PATH = REPO_ROOT / "data" / "hcmc_benchmark" / "hcmc_300_x5_dataset.jsonl"

# Cached paper baseline paths
HN_CACHED_BASELINES = REPO_ROOT / "data" / "caption_dataset" / "test_500_balanced_all_models_captions_scientific_vcgat.json"
HCMC_CACHED_BASELINES = REPO_ROOT / "data" / "hcmc_benchmark" / "hcmc_300_decoupled_captions.json"

CHECKPOINT_DIR = REPO_ROOT / "experiments" / "checkpoints_caption" / "qwen25_3b_certainty_lora"
BASE_MODEL_PATH = Path("/home/nguyendinhquy/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
if not BASE_MODEL_PATH.exists():
    BASE_MODEL_PATH = Path("Qwen/Qwen2.5-3B-Instruct")


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_evidence_packet(
    clip_info: Dict[str, Any],
    candidate_facts: List[Dict[str, Any]],
    v2_facts: Dict[str, Any],
    scene_graph_str: str,
    mu_L: float = 0.85
) -> Dict[str, Any]:
    """
    Builds the exact evidence packet for E5 and E6:
    1. Re-rank candidate facts by mu_F.
    2. Select top k = 5 facts.
    3. Certainty c = mu_F * mu_L.
    4. Regime: ASSERT (c >= 0.65), HEDGE (0.25 <= c < 0.65), OMIT (c < 0.25).
    5. Scene graph: direct visual observations unhedged (c = 1.0, ASSERT).
    """
    scored_facts = []
    for f in candidate_facts:
        fid = f.get("fact_id")
        v2_item = v2_facts.get(fid, {})
        mu_F = v2_item.get("mu_F_v2", v2_item.get("w_base", f.get("confidence_score", 0.5)))
        c = float(mu_F) * float(mu_L)

        if c >= 0.65:
            regime = "ASSERT"
        elif c >= 0.25:
            regime = "HEDGE"
        else:
            regime = "OMIT"

        tail = f.get("object") or f.get("tail") or ""
        subj = f.get("subject") or f.get("head") or ""
        pred = f.get("predicate") or f.get("relation") or ""
        fact_text = f.get("fact_text") or f"{subj} {pred.replace('_', ' ')} {tail}."

        scored_facts.append({
            "fact_id": fid,
            "subject": subj,
            "predicate": pred,
            "object": tail,
            "fact_text": fact_text,
            "mu_F": float(mu_F),
            "mu_L": float(mu_L),
            "c_certainty": float(c),
            "policy": regime,
            "gold_review": f.get("review", {}).get("result", "N/A")
        })

    # Sort by mu_F descending, take top 5
    scored_facts.sort(key=lambda x: x["mu_F"], reverse=True)
    top5_facts = scored_facts[:5]

    # Parse scene graph concepts
    sg_lines = [s.strip().lstrip("- ") for s in scene_graph_str.split("\n") if s.strip()]
    sg_concepts = [re.sub(r"\(\d+\.\d+\)", "", s).strip() for s in sg_lines if len(s.strip()) > 3][:6]

    return {
        "clip_id": clip_info["clip_id"],
        "landmark_name": clip_info["landmark_name"],
        "source_video_id": clip_info.get("source_video_id", "unknown"),
        "top5_facts": top5_facts,
        "scene_graph_concepts": sg_concepts,
        "raw_scene_graph": scene_graph_str
    }


def format_prompt_for_system(sys_id: str, ep: Dict[str, Any]) -> str:
    lm = ep["landmark_name"]
    sg = ep["raw_scene_graph"]
    top5 = ep["top5_facts"]

    if sys_id == "E1":  # Video-only
        return (
            f"OBSERVED (visual scene graph from video):\n{sg}\n\n"
            f"INSTRUCTION:\n"
            f"Write a concise narration describing what is visually observed in the scene. "
            f"Do not state unobserved historical facts or uncertain assertions.\n\n"
            f"CAPTION:"
        )
    elif sys_id == "E2":  # Text-RAG
        # Retrieved text summary
        facts_text = " ".join([f["fact_text"] for f in top5[:3]])
        return (
            f"RETRIEVED TOURISM DOCUMENT:\n"
            f"{lm} is a notable historic heritage site in Vietnam. {facts_text}\n\n"
            f"OBSERVED:\n{sg}\n\n"
            f"INSTRUCTION:\n"
            f"Write a narration describing the landmark and scene using the retrieved text.\n\n"
            f"CAPTION:"
        )
    elif sys_id == "E3":  # Crisp-KG RAG
        facts_lines = [f"- {f['fact_text']}" for f in top5]
        facts_str = "\n".join(facts_lines) if facts_lines else f"- {lm} is located in Vietnam."
        return (
            f"LANDMARK: {lm}\n"
            f"KNOWLEDGE FACTS (top-5 from KG):\n{facts_str}\n\n"
            f"OBSERVED:\n{sg}\n\n"
            f"INSTRUCTION:\n"
            f"Write a cohesive narration describing the landmark and scene. Confidently state the knowledge facts.\n\n"
            f"CAPTION:"
        )
    elif sys_id == "E4":  # Prompt-only LLM
        facts_lines = [f"- {f['fact_text']}" for f in top5]
        facts_str = "\n".join(facts_lines) if facts_lines else f"- {lm} is located in Vietnam."
        return (
            f"LANDMARK: {lm}\n"
            f"GIVEN FACTS:\n{facts_str}\n\n"
            f"OBSERVED:\n{sg}\n\n"
            f"INSTRUCTION:\n"
            f"Write a narration describing the scene. Carefully consider the uncertainty of given facts.\n\n"
            f"CAPTION:"
        )
    elif sys_id in ("E5", "E6"):  # Proposed (Base vs LoRA)
        lines = []
        # Visual facts (ASSERT)
        for c in ep["scene_graph_concepts"]:
            lines.append(f"- [VISUAL OBSERVATION] The video visibly shows {c}.")
        # Top-5 KG facts with policy
        for f in top5:
            lines.append(f"- [{f['policy']}] {f['fact_text']} (Certainty c = {f['c_certainty']:.2f})")
        facts_str = "\n".join(lines)
        return (
            f"[LANDMARK SCENE]\nSite: {lm}\n\n"
            f"[GROUNDED EVIDENCE & CERTAINTY POLICIES]\n{facts_str}\n\n"
            f"INSTRUCTION:\n"
            f"Narrate the video scene accurately. Assert confirmed visual and high-certainty facts (ASSERT). "
            f"Use cautious, hedged language for uncertain facts (HEDGE). Completely omit ungrounded facts (OMIT).\n\n"
            f"CAPTION:"
        )
    return ""


def main():
    LOGGER.info("=" * 80)
    LOGGER.info("STARTING PHASE 1: END-TO-END CAPTIONING EVALUATION PIPELINE")
    LOGGER.info("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Target compute device: {device}")

    # 1. Load manifests
    hanoi_clips = load_manifest(HANOI_MANIFEST)
    hcmc_clips = load_manifest(HCMC_MANIFEST)
    LOGGER.info(f"Loaded Hanoi Test Clips: {len(hanoi_clips)}")
    LOGGER.info(f"Loaded HCMC OOD Test Clips: {len(hcmc_clips)}")

    # 2. Load references
    with open(FGOLD_PATH, "r", encoding="utf-8") as f:
        fgold_all = json.load(f).get("clips", {})
    LOGGER.info(f"Loaded Gold Reference Benchmark ({len(fgold_all)} clips total).")

    with open(V2_FACTS_PATH, "r", encoding="utf-8") as f:
        v2_facts = json.load(f)
    LOGGER.info(f"Loaded FKG V2 Facts ({len(v2_facts)} facts).")

    # Load scene graphs
    with open(HN_SCENE_GRAPHS_PATH, "r", encoding="utf-8") as f:
        hn_sg_map = json.load(f)
    LOGGER.info(f"Loaded Hanoi Canonical Scene Graphs ({len(hn_sg_map)} clips).")

    hcmc_sg_map = {}
    if HCMC_300_DATASET_PATH.exists():
        with open(HCMC_300_DATASET_PATH, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    item = json.loads(l)
                    cid = f"hcmc_x5_{item['clip_idx']:04d}" if 'clip_idx' in item else item.get('sample_id')
                    hcmc_sg_map[cid] = item.get("observed_scene_graph", "")
    LOGGER.info(f"Loaded HCMC Scene Graphs ({len(hcmc_sg_map)} clips).")

    # Load cached baseline captions if available
    hn_cached_caps = {}
    if HN_CACHED_BASELINES.exists():
        with open(HN_CACHED_BASELINES, "r", encoding="utf-8") as f:
            hn_cached_caps = json.load(f)
        LOGGER.info(f"Loaded Hanoi Paper Baselines: {list(hn_cached_caps.keys())}")

    hcmc_cached_caps = {}
    if HCMC_CACHED_BASELINES.exists():
        with open(HCMC_CACHED_BASELINES, "r", encoding="utf-8") as f:
            hcmc_cached_caps = json.load(f)
        LOGGER.info(f"Loaded HCMC Paper Baselines: {list(hcmc_cached_caps.keys())}")

    # 3. Model loading
    LOGGER.info(f"Loading Base Qwen2.5-3B-Instruct from {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load in bfloat16 or 4-bit depending on GPU
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            str(BASE_MODEL_PATH),
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True
        )
        LOGGER.info("Base Qwen2.5-3B loaded in bfloat16 successfully.")
    except Exception as e:
        LOGGER.warning(f"Failed bfloat16 load ({e}), falling back to 8-bit...")
        base_model = AutoModelForCausalLM.from_pretrained(
            str(BASE_MODEL_PATH),
            load_in_8bit=True,
            device_map={"": device},
            trust_remote_code=True
        )

    # Load LoRA adapter
    LOGGER.info(f"Loading LoRA Adapter from {CHECKPOINT_DIR}...")
    lora_model = PeftModel.from_pretrained(base_model, str(CHECKPOINT_DIR))
    lora_model.eval()
    LOGGER.info("LoRA Adapter attached successfully.")

    # Prepare evidence packets
    domains = [
        ("hanoi", hanoi_clips, hn_sg_map, hn_cached_caps),
        ("hcmc", hcmc_clips, hcmc_sg_map, hcmc_cached_caps)
    ]

    all_domain_results = {}

    for domain_name, clips, sg_map, cached_caps in domains:
        LOGGER.info("=" * 80)
        LOGGER.info(f"EVALUATING DOMAIN: {domain_name.upper()} ({len(clips)} CLIPS)")
        LOGGER.info("=" * 80)

        domain_dir = PHASE1_DIR / domain_name
        domain_dir.mkdir(parents=True, exist_ok=True)

        captions_out = []
        evidence_out = []
        claims_out = []
        verification_out = []
        metrics_by_clip = []

        system_metrics_collector = defaultdict(lambda: defaultdict(list))
        sources_collector = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        systems = ["E1", "E2", "E3", "E4", "E5", "E6"]

        for idx, c in enumerate(clips):
            cid = c["clip_id"]
            src_video = c.get("source_video_id", "src_unknown")
            fgold_data = fgold_all.get(cid, {})
            cand_facts = fgold_data.get("gold_facts", [])
            sg_str = sg_map.get(cid, "")

            # Build exact evidence packet (shared between E5 and E6)
            mu_L_val = 0.85 if domain_name == "hanoi" else 0.70
            ep = build_evidence_packet(c, cand_facts, v2_facts, sg_str, mu_L=mu_L_val)
            evidence_out.append(ep)

            gold_facts_true = [f for f in cand_facts if f.get("review", {}).get("result") == "đúng"]
            negative_facts = [f for f in cand_facts if f.get("review", {}).get("result") == "sai"]
            vis_concepts = ep["scene_graph_concepts"]

            clip_metrics_entry = {"clip_id": cid, "source_video_id": src_video}

            for sys_id in systems:
                # 1. Obtain Caption
                caption = ""
                # Check cached baseline if available
                if sys_id == "E1" and "visual_only" in cached_caps and idx < len(cached_caps["visual_only"]):
                    caption = cached_caps["visual_only"][idx]
                elif sys_id == "E2" and "text_rag" in cached_caps and idx < len(cached_caps["text_rag"]):
                    caption = cached_caps["text_rag"][idx]
                elif sys_id == "E3" and "crisp_kg" in cached_caps and idx < len(cached_caps["crisp_kg"]):
                    caption = cached_caps["crisp_kg"][idx]
                elif sys_id == "E4" and "prompt_only" in cached_caps and idx < len(cached_caps["prompt_only"]):
                    caption = cached_caps["prompt_only"][idx]
                elif sys_id == "E1" and "Visual_Only" in cached_caps and idx < len(cached_caps["Visual_Only"]):
                    caption = cached_caps["Visual_Only"][idx]
                elif sys_id == "E3" and "Crisp_KG_RAG" in cached_caps and idx < len(cached_caps["Crisp_KG_RAG"]):
                    caption = cached_caps["Crisp_KG_RAG"][idx]
                elif sys_id == "E4" and "Prompt_Only_LLM" in cached_caps and idx < len(cached_caps["Prompt_Only_LLM"]):
                    caption = cached_caps["Prompt_Only_LLM"][idx]

                # If not cached or E5/E6, generate via LLM
                if not caption or sys_id in ("E5", "E6"):
                    prompt = format_prompt_for_system(sys_id, ep)
                    messages = [
                        {"role": "system", "content": "You are an expert cultural heritage narrator for Vietnamese landmarks."},
                        {"role": "user", "content": prompt}
                    ]
                    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

                    with torch.no_grad():
                        if sys_id == "E6":
                            # Use LoRA
                            out = lora_model.generate(
                                **inputs,
                                max_new_tokens=140,
                                do_sample=False,
                                repetition_penalty=1.1,
                                pad_token_id=tokenizer.pad_token_id
                            )
                        else:
                            # Use Base (temporarily disable adapter)
                            with lora_model.disable_adapter():
                                out = base_model.generate(
                                    **inputs,
                                    max_new_tokens=140,
                                    do_sample=False,
                                    repetition_penalty=1.1,
                                    pad_token_id=tokenizer.pad_token_id
                                )
                    caption = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

                captions_out.append({
                    "clip_id": cid,
                    "domain": domain_name,
                    "system_id": sys_id,
                    "caption": caption
                })

                # 2. Extract Atomic Claims
                sentences = split_sentences(caption)
                clip_claims = []
                for s in sentences:
                    ctype = classify_claim_type(s)
                    style = "HEDGED" if is_hedged(s) else "ASSERTED"

                    if ctype in ("HISTORICAL", "MIXED"):
                        v_label, matched_f = verify_historical_claim(s, gold_facts_true, negative_facts)
                        fid = matched_f.get("fact_id") if matched_f else None
                    else:
                        v_label = verify_visual_claim(s, vis_concepts)
                        fid = None

                    claim_obj = {
                        "clip_id": cid,
                        "domain": domain_name,
                        "system_id": sys_id,
                        "sentence": s,
                        "claim": s,
                        "type": ctype,
                        "claim_type": ctype,
                        "style": style,
                        "verification": v_label,
                        "matched_fact_id": fid
                    }
                    claims_out.append(claim_obj)
                    verification_out.append(claim_obj)
                    clip_claims.append(claim_obj)

                # 3. Compute Metrics
                regimes = ep["top5_facts"] if sys_id in ("E5", "E6") else None
                m = compute_clip_metrics(clip_claims, gold_facts_true, regimes)

                for k, val in m.items():
                    clip_metrics_entry[f"{sys_id}_{k}"] = val
                    if val is not None:
                        system_metrics_collector[sys_id][k].append(val)
                        sources_collector[sys_id][k][src_video].append(val)

            metrics_by_clip.append(clip_metrics_entry)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(clips):
                LOGGER.info(f"Processed {idx + 1}/{len(clips)} clips in {domain_name}...")

        # Save domain outputs
        with open(domain_dir / "captions.jsonl", "w", encoding="utf-8") as f:
            for item in captions_out:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(domain_dir / "evidence.jsonl", "w", encoding="utf-8") as f:
            for item in evidence_out:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(domain_dir / "claims.jsonl", "w", encoding="utf-8") as f:
            for item in claims_out:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(domain_dir / "verification.jsonl", "w", encoding="utf-8") as f:
            for item in verification_out:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        df_metrics = pd.DataFrame(metrics_by_clip)
        df_metrics.to_csv(domain_dir / "metrics_per_clip.csv", index=False)

        # Aggregate summary and bootstrap
        summary_stats = {}
        for sys_id in systems:
            sys_summary = {}
            for metric in ["fp", "coverage", "fp_conservative", "kr", "bkr5", "vsp", "vfar", "aur", "eap", "ovr", "oargen"]:
                vals = system_metrics_collector[sys_id][metric]
                n_valid = len(vals)
                if n_valid > 0:
                    macro_mean = float(np.mean(vals))
                    _, low, high = cluster_bootstrap_ci(sources_collector[sys_id][metric], b_resamples=10000)
                    sys_summary[metric] = {
                        "macro_mean": round(macro_mean, 4),
                        "ci_95": [round(low, 4), round(high, 4)],
                        "valid_clips": n_valid
                    }
                else:
                    sys_summary[metric] = {
                        "macro_mean": None,
                        "ci_95": None,
                        "valid_clips": 0
                    }
            summary_stats[sys_id] = sys_summary

        with open(domain_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=2)

        all_domain_results[domain_name] = summary_stats

    # =========================================================================
    # GENERATE REQUIRED TABLES
    # =========================================================================
    tables_dir = PHASE1_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Table 7A: Hanoi
    t7a_rows = []
    hn_res = all_domain_results["hanoi"]
    sys_names = {
        "E1": "E1 Qwen2.5-VL-3B",
        "E2": "E2 Text-RAG",
        "E3": "E3 Crisp-KG RAG",
        "E4": "E4 Prompt-only LLM",
        "E5": "E5 DEC + Base Qwen3B",
        "E6": "E6 Proposed (Full LoRA)"
    }
    for sid in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        s_data = hn_res[sid]
        fp_str = f"{s_data['fp']['macro_mean']:.4f}" if s_data['fp']['macro_mean'] is not None else "N/A"
        bkr_str = f"{s_data['bkr5']['macro_mean']:.4f}" if s_data['bkr5']['macro_mean'] is not None else "N/A"
        kr_str = f"{s_data['kr']['macro_mean']:.4f}" if s_data['kr']['macro_mean'] is not None else "N/A"
        vsp_str = f"{s_data['vsp']['macro_mean']:.4f}" if s_data['vsp']['macro_mean'] is not None else "N/A"
        vfar_str = f"{s_data['vfar']['macro_mean']:.4f}" if s_data['vfar']['macro_mean'] is not None else "N/A"
        cov_str = f"{s_data['coverage']['macro_mean']:.4f}" if s_data['coverage']['macro_mean'] is not None else "N/A"
        t7a_rows.append({
            "System": sys_names[sid],
            "FP": fp_str,
            "BKR@5": bkr_str,
            "KR": kr_str,
            "VSP": vsp_str,
            "VFAR": vfar_str,
            "Verify Cov.": cov_str
        })
    pd.DataFrame(t7a_rows).to_csv(tables_dir / "table7a_hanoi.csv", index=False)

    # Table 7B: HCMC
    t7b_rows = []
    hcmc_res = all_domain_results["hcmc"]
    for sid in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        s_data = hcmc_res[sid]
        fp_str = f"{s_data['fp']['macro_mean']:.4f}" if s_data['fp']['macro_mean'] is not None else "N/A"
        bkr_str = f"{s_data['bkr5']['macro_mean']:.4f}" if s_data['bkr5']['macro_mean'] is not None else "N/A"
        kr_str = f"{s_data['kr']['macro_mean']:.4f}" if s_data['kr']['macro_mean'] is not None else "N/A"
        vsp_str = f"{s_data['vsp']['macro_mean']:.4f}" if s_data['vsp']['macro_mean'] is not None else "N/A"
        vfar_str = f"{s_data['vfar']['macro_mean']:.4f}" if s_data['vfar']['macro_mean'] is not None else "N/A"
        cov_str = f"{s_data['coverage']['macro_mean']:.4f}" if s_data['coverage']['macro_mean'] is not None else "N/A"
        t7b_rows.append({
            "System": sys_names[sid],
            "FP": fp_str,
            "BKR@5": bkr_str,
            "KR": kr_str,
            "VSP": vsp_str,
            "VFAR": vfar_str,
            "Verify Cov.": cov_str
        })
    pd.DataFrame(t7b_rows).to_csv(tables_dir / "table7b_hcmc.csv", index=False)

    # Table 7C: Cross-Region Generalization Gap (HCMC - Hanoi)
    t7c_rows = []
    for sid in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        hn_s = hn_res[sid]
        hc_s = hcmc_res[sid]
        
        def diff_str(m):
            v_hn = hn_s[m]["macro_mean"]
            v_hc = hc_s[m]["macro_mean"]
            if v_hn is not None and v_hc is not None:
                d = v_hc - v_hn
                return f"{d:+.4f}"
            return "N/A"

        t7c_rows.append({
            "System": sys_names[sid],
            "ΔFP": diff_str("fp"),
            "ΔBKR@5": diff_str("bkr5"),
            "ΔKR": diff_str("kr"),
            "ΔVSP": diff_str("vsp"),
            "ΔVFAR": diff_str("vfar")
        })
    pd.DataFrame(t7c_rows).to_csv(tables_dir / "table7c_generalization.csv", index=False)

    # Table 7D: Base vs LoRA Generation Quality (E5 vs E6)
    t7d_rows = []
    for dom in ["hanoi", "hcmc"]:
        d_res = all_domain_results[dom]
        for sid in ["E5", "E6"]:
            s_data = d_res[sid]
            t7d_rows.append({
                "Domain": dom.upper(),
                "System": sys_names[sid],
                "EAP": f"{s_data['eap']['macro_mean']:.4f}" if s_data['eap']['macro_mean'] is not None else "N/A",
                "OARgen": f"{s_data['oargen']['macro_mean']:.4f}" if s_data['oargen']['macro_mean'] is not None else "N/A",
                "OVR": f"{s_data['ovr']['macro_mean']:.4f}" if s_data['ovr']['macro_mean'] is not None else "N/A",
                "FP": f"{s_data['fp']['macro_mean']:.4f}" if s_data['fp']['macro_mean'] is not None else "N/A",
                "VSP": f"{s_data['vsp']['macro_mean']:.4f}" if s_data['vsp']['macro_mean'] is not None else "N/A"
            })
    pd.DataFrame(t7d_rows).to_csv(tables_dir / "table7d_generation.csv", index=False)

    LOGGER.info("ALL PIPELINE STAGES, METRICS AND TABLES COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
