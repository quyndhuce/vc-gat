#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fact_drop_funnel_waterfall.py
=============================================================================
End-to-End Fact-Drop Funnel (Waterfall) Analysis:
Traces every gold fact through:
  Stage 0: Ground Truth Reference (F_gold, review=='đúng')
  Stage 1: DEC Candidate Retrieval
  Stage 2: Top-5 Selection (Capacity / Ranking Bottleneck)
  Stage 3: FKG Calibration & Regime Assignment (OMIT vs HEDGE vs ASSERT)
  Stage 4: LoRA Generation (Text Expression Bottleneck)
  Stage 5: Evaluator Matcher (Recognition / Crediting Bottleneck)
=============================================================================
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
AUDIT_DIR = PHASE1_DIR / "audit"

# 1. Load Ground Truth
with open(REPO_ROOT / "scene_conditioned_gold" / "outputs" / "fgold_test_800_clips.json") as f:
    fgold_all = json.load(f)["clips"]

# 2. Load Evidence Packets
with open(PHASE1_DIR / "hanoi" / "evidence.jsonl") as f:
    hn_ev = {x["clip_id"]: x for x in [json.loads(l) for l in f]}
with open(PHASE1_DIR / "hcmc" / "evidence.jsonl") as f:
    hc_ev = {x["clip_id"]: x for x in [json.loads(l) for l in f]}

# 3. Load LoRA Captions
with open(PHASE1_DIR / "hanoi" / "captions.jsonl") as f:
    hn_caps = {x["clip_id"]: x["caption"] for x in [json.loads(l) for l in f] if x["system_id"] == "E6"}
with open(PHASE1_DIR / "hcmc" / "captions.jsonl") as f:
    hc_caps = {x["clip_id"]: x["caption"] for x in [json.loads(l) for l in f] if x["system_id"] == "E6"}

# 4. Load Claims & Verified Facts
with open(PHASE1_DIR / "hanoi" / "claims.jsonl") as f:
    hn_claims = [json.loads(l) for l in f if l.strip()]
with open(PHASE1_DIR / "hcmc" / "claims.jsonl") as f:
    hc_claims = [json.loads(l) for l in f if l.strip()]

matched_e6_facts = defaultdict(set)
for c in hn_claims + hc_claims:
    if c["system_id"] == "E6" and c.get("verification") == "SUPPORTED" and c.get("matched_fact_id"):
        matched_e6_facts[c["clip_id"]].add(c["matched_fact_id"])

def analyze_domain(dom_name, manifest_ev, caps_dict):
    total_gold_facts = 0
    in_candidate_pool = 0
    selected_top5 = 0
    dropped_rank_gt_5 = 0
    regime_assert = 0
    regime_hedge = 0
    regime_omit = 0
    passed_to_prompt = 0
    expressed_by_lora = 0
    dropped_by_lora = 0
    credited_by_evaluator = 0
    missed_by_evaluator = 0

    fact_records = []

    for cid, ep in manifest_ev.items():
        gdata = fgold_all.get(cid, {})
        cand_facts = gdata.get("gold_facts", [])
        gold_true = [f for f in cand_facts if f.get("review", {}).get("result") == "đúng"]
        total_gold_facts += len(gold_true)

        top5_fids = {f["fact_id"]: f for f in ep["top5_facts"]}
        cap = caps_dict.get(cid, "")
        cap_lower = cap.lower()
        credited_fids = matched_e6_facts[cid]

        for gf in gold_true:
            fid = gf.get("fact_id")
            f_text = gf.get("fact_text", "")
            tail = (gf.get("object") or gf.get("tail") or "").lower().strip()
            subj = (gf.get("subject") or gf.get("head") or "").lower().strip()
            pred = (gf.get("predicate") or gf.get("relation") or "").lower().strip().replace("_", " ")

            in_candidate_pool += 1

            # Stage 2: Top-5
            is_in_top5 = fid in top5_fids
            policy = None
            mu_f_val = None
            c_val = None

            if is_in_top5:
                selected_top5 += 1
                f_top = top5_fids[fid]
                policy = f_top.get("policy")
                mu_f_val = f_top.get("mu_F")
                c_val = f_top.get("c_certainty")

                if policy == "ASSERT":
                    regime_assert += 1
                    passed_to_prompt += 1
                elif policy == "HEDGE":
                    regime_hedge += 1
                    passed_to_prompt += 1
                elif policy == "OMIT":
                    regime_omit += 1
            else:
                dropped_rank_gt_5 += 1

            # Stage 4: LoRA Expression
            is_expressed = False
            if is_in_top5 and policy in ("ASSERT", "HEDGE"):
                words_tail = [w for w in tail.split() if len(w) > 3 and w not in ["and", "the", "walls", "street", "building"]]
                if len(words_tail) >= 2:
                    is_expressed = sum(1 for w in words_tail if w in cap_lower) >= 2
                elif len(words_tail) == 1:
                    is_expressed = words_tail[0] in cap_lower
                elif len(tail) > 3 and tail in cap_lower:
                    is_expressed = True

                if not is_expressed and len(pred) > 4 and pred in cap_lower:
                    is_expressed = True

                if is_expressed:
                    expressed_by_lora += 1
                else:
                    dropped_by_lora += 1

            # Stage 5: Evaluator Credited
            is_credited = False
            if is_expressed:
                if fid in credited_fids:
                    is_credited = True
                    credited_by_evaluator += 1
                else:
                    missed_by_evaluator += 1

            # Determine final fate / drop stage
            final_status = ""
            drop_stage = ""
            if not is_in_top5:
                final_status = "DROPPED_AT_STAGE_2_CAPACITY"
                drop_stage = "Stage 2: DEC / Capacity (Rank > 5)"
            elif policy == "OMIT":
                final_status = "SUPPRESSED_AT_STAGE_3_REGIME"
                drop_stage = "Stage 3: FKG / Regime (c < 0.25 OMIT)"
            elif not is_expressed:
                final_status = "OMITTED_AT_STAGE_4_LORA"
                drop_stage = "Stage 4: LoRA Generation (Not Written)"
            elif not is_credited:
                final_status = "MISSED_AT_STAGE_5_EVALUATOR"
                drop_stage = "Stage 5: Evaluator Matcher (Uncredited)"
            else:
                final_status = "VERIFIED_AT_STAGE_5_SUPPORTED"
                drop_stage = "Successfully Verified (N_S)"

            fact_records.append({
                "Domain": dom_name,
                "Clip_ID": cid,
                "Fact_ID": fid,
                "Fact_Text": f_text,
                "Tail": tail,
                "Predicate": pred,
                "Is_In_Top5": is_in_top5,
                "Policy": policy,
                "mu_F": mu_f_val,
                "Certainty_c": c_val,
                "Is_Expressed_in_LoRA": is_expressed,
                "Is_Credited_Supported": is_credited,
                "Drop_Stage": drop_stage,
                "Final_Status": final_status
            })

    funnel_summary = {
        "Domain": dom_name,
        "Total_Gold_True": total_gold_facts,
        "Stage1_In_Candidate_Pool": in_candidate_pool,
        "Stage2_Selected_Top5": selected_top5,
        "Stage2_Loss_Rank_gt_5": dropped_rank_gt_5,
        "Stage3_Regime_ASSERT": regime_assert,
        "Stage3_Regime_HEDGE": regime_hedge,
        "Stage3_Regime_OMIT_Suppressed": regime_omit,
        "Stage3_Passed_To_Prompt": passed_to_prompt,
        "Stage4_Expressed_by_LoRA": expressed_by_lora,
        "Stage4_Loss_LoRA_Omission": dropped_by_lora,
        "Stage5_Credited_Supported": credited_by_evaluator,
        "Stage5_Loss_Evaluator_Missed": missed_by_evaluator
    }

    return funnel_summary, pd.DataFrame(fact_records)

hn_summary, hn_df = analyze_domain("HANOI", hn_ev, hn_caps)
hc_summary, hc_df = analyze_domain("HCMC", hc_ev, hc_caps)

# Save combined facts
combined_df = pd.concat([hn_df, hc_df], ignore_index=True)
combined_df.to_csv(AUDIT_DIR / "phase1_fact_drop_funnel_all_facts.csv", index=False)

summary_rows = [hn_summary, hc_summary]
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(AUDIT_DIR / "phase1_fact_drop_funnel_summary.csv", index=False)

print("Saved phase1_fact_drop_funnel_all_facts.csv and summary.")
