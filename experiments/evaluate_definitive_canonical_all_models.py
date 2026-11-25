#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/evaluate_definitive_canonical_all_models.py
=============================================================================
Definitive, exact Protocol B Canonical Evaluator across ALL 13 models
using the canonical functions from export_per_clip_vsp_breakdown.py.
=============================================================================
"""

import collections
import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("DefinitiveCanonicalEval")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "data" / "caption_dataset" / "generated_captions_test_balanced_500_vcgat.jsonl"
CAPS_FILE = REPO_ROOT / "results" / "two_level_silver_reference_train_val" / "captions_fair_budget_full_500.json"
OUT_JSON = REPO_ROOT / "results" / "two_level_silver_reference_train_val" / "table3_canonical_protocol_b_final.json"
OUT_CSV = REPO_ROOT / "results" / "two_level_silver_reference_train_val" / "table3_canonical_protocol_b_final.csv"
OUT_PER_CLIP_CSV = REPO_ROOT / "results" / "two_level_silver_reference_train_val" / "table3_canonical_per_clip_audit.csv"

VISUAL_TESTABLE_RELATIONS = {
    "has_part", "part_of", "features", "includes", "architecture",
    "architectural_style", "material", "contains", "structure",
    "appearance", "adjacent_to", "next_to", "spans_across", "inside",
    "situated_on", "surrounded_by", "decorated_with", "made_of", "stands_on",
    "over", "faces", "in_front_of", "connects", "connected_to"
}

ASSERTIVE_MARKERS = [
    "is known for", "was built", "was constructed", "was founded",
    "dates back", "was established", "is located", "has been", "features",
    "is a", "are a", "stands as", "serves as", "showcases", "houses",
    "honoring", "commemorates", "contains", "stands proudly", "is situated",
]
HEDGING_MARKERS = [
    "likely", "appears to", "possibly", "might be", "could be",
    "seems to", "suggests", "may have", "if this is", "believed to",
    "probably", "potentially", "perhaps", "may include", "typical of",
    "depending on", "indicates", "resembles", "evokes",
]
UNCERTAIN_MARKERS = [
    "uncertain", "cannot be confirmed", "cannot confirm", "unclear",
    "remains unknown", "not confirmed", "inconclusive",
    "insufficient evidence", "difficult to determine", "difficult to verify",
    "exact landmark remains", "exact landmark is", "exact identity",
    "exact location remains", "remains unclear", "unidentified",
]

def parse_scene_graph(s_text: str) -> set:
    entities = set()
    if not s_text:
        return entities
    for line in s_text.split("\n"):
        line = line.strip().lstrip("- ")
        m = re.match(r"\(([^,]+),\s*([^,]+),\s*([^\)]+)\)", line)
        if m:
            entities.add(m.group(1).strip().lower())
            entities.add(m.group(3).strip().lower())
        else:
            c = re.sub(r"\s*\([0-9\.]+\)", "", line).strip().lower()
            if c:
                entities.add(c)
    return entities

def match_fact(tail_phrase: str, text_lower: str) -> bool:
    t = str(tail_phrase).lower().strip()
    if len(t) < 3:
        return False
    if t in text_lower:
        return True
    words = [w for w in re.split(r"[\s_]+", t) if len(w) > 3]
    if len(words) >= 2:
        return sum(1 for w in words if w in text_lower) >= max(2, int(len(words) * 0.7))
    return False

def is_entity_in_scene(tail_str: str, sg_entities: set) -> bool:
    t = tail_str.strip().lower()
    for ent in sg_entities:
        if t in ent or ent in t:
            return True
        words_t = [w for w in t.split() if len(w) > 3]
        words_e = [w for w in ent.split() if len(w) > 3]
        if words_t and words_e:
            shared = set(words_t) & set(words_e)
            if len(shared) >= max(1, len(words_t) // 2):
                return True
    return False

def compute_rouge_l(ref: str, hyp: str) -> float:
    ref_toks = re.findall(r"\w+", ref.lower())
    hyp_toks = re.findall(r"\w+", hyp.lower())
    m, n = len(ref_toks), len(hyp_toks)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_toks[i] == hyp_toks[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    p = lcs / n
    r = lcs / m
    return (2 * p * r / (p + r)) * 100.0 if (p + r) > 0 else 0.0

def compute_bleu4(ref: str, hyp: str) -> float:
    ref_toks = re.findall(r"\w+", ref.lower())
    hyp_toks = re.findall(r"\w+", hyp.lower())
    c, r_len = len(hyp_toks), len(ref_toks)
    if c == 0 or r_len == 0:
        return 0.0
    bp = 1.0 if c > r_len else (math.exp(1 - r_len / c) if c > 0 else 0.0)
    precisions = []
    for n in range(1, 5):
        if c < n:
            precisions.append(1e-10)
            continue
        hyp_ngrams = collections.Counter(tuple(hyp_toks[i:i+n]) for i in range(c - n + 1))
        ref_ngrams = collections.Counter(tuple(ref_toks[i:i+n]) for i in range(r_len - n + 1))
        overlap = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
        total = max(1, sum(hyp_ngrams.values()))
        precisions.append((overlap + 1e-10) / total)
    log_prec = sum(math.log(p) for p in precisions) / 4.0
    return bp * math.exp(log_prec) * 100.0

def bootstrap_ci(arr, B=10000, seed=42):
    np.random.seed(seed)
    boot = np.mean(np.random.choice(arr, size=(B, len(arr)), replace=True), axis=1)
    mean_val = float(np.mean(arr))
    sd_val = float(np.std(boot))
    ci_lower = float(np.percentile(boot, 2.5))
    ci_upper = float(np.percentile(boot, 97.5))
    return mean_val, sd_val, ci_lower, ci_upper

def main():
    LOGGER.info("Loading test dataset from %s...", DATA_FILE)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    N = len(samples)
    LOGGER.info("Loaded %d test samples.", N)

    LOGGER.info("Loading captions from %s...", CAPS_FILE)
    with open(CAPS_FILE, "r", encoding="utf-8") as f:
        all_caps_dict = json.load(f)

    DISPLAY_NAMES = {
        "visual_only":              "Qwen2.5-3B Visual-only (No KG)",
        "crisp_kg_fair":            "Crisp-KG RAG (Fair: top-Mq)",
        "hard_threshold_fair":      "Hard-threshold KG (Fair: top-Mq, θ=0.50)",
        "prompt_only_fair":         "Prompt-only LLM (Fair: top-Mq)",
        "crisp_kg_unconstrained":   "Crisp-KG RAG (Unconstrained Literature)",
        "binary_adapter":           "Binary Adapter (LoRA, Mq)",
        "ternary_adapter":          "Ternary Adapter (LoRA, Mq)",
        "ours_unanchored":          "Ours w/o VC-GAT (Unanchored, Mq)",
        "flat_siglip2":             "Flat SigLIP-2 Cosine (No Graph Attention, Mq)",
        "merged_raw":               "Merged (s = r · c, Raw Thresholds)",
        "merged_calibrated":        "Merged (s = r · c, Calibrated Thresholds)",
        "proposed_full":            "Decoupled Ranking VC-GAT (Ours, Mq)",
        "proposed_full_guardrail":  "Proposed + Epistemic Modal Guardrail",
    }

    results = {}
    per_clip_records = []

    for key, model_label in DISPLAY_NAMES.items():
        if key not in all_caps_dict:
            LOGGER.warning("Key %s not found in captions!", key)
            continue

        raw_caps = all_caps_dict[key]
        if isinstance(raw_caps, list) and isinstance(raw_caps[0], dict):
            captions = [item.get("caption", "") for item in raw_caps]
        else:
            captions = raw_caps

        k_rec_all = []
        canon_vsp_all = []
        canon_vsp_active = []
        policy_all = []
        oar_gen_all = []
        oar_fact_all = []
        fact_p_all = []
        bleu4_all = []
        rougel_all = []

        tot_gold_facts = 0
        tot_matched_gold = 0
        tot_canon_claims = 0
        tot_canon_supported = 0

        for idx, s in enumerate(samples):
            cap = str(captions[idx]).lower()
            ref = s.get("reference_narration") or s.get("caption") or s.get("prompt", "")
            gold_facts = s.get("gold_facts") or s.get("eligible_facts", [])
            cert = s.get("certainty_level", "HIGH")
            sg_entities = parse_scene_graph(s.get("observed_scene_graph", ""))

            # 1. K-Recall
            n_gold = len(gold_facts)
            n_matched = sum(1 for f in gold_facts if match_fact(f.get("tail", ""), cap))
            k_rec = (n_matched / max(n_gold, 1)) * 100.0

            tot_gold_facts += n_gold
            tot_matched_gold += n_matched
            k_rec_all.append(k_rec)

            # 2. Protocol B: Canonical VSP
            vis_facts = [
                f for f in gold_facts
                if f.get("relation", "").lower() in VISUAL_TESTABLE_RELATIONS or f.get("category") == "visual"
            ]
            canon_claims = 0
            canon_supported = 0
            checked_tails = set()

            for vf in vis_facts:
                t = vf.get("tail", "").lower()
                if match_fact(t, cap):
                    checked_tails.add(t)
                    canon_claims += 1
                    if is_entity_in_scene(t, sg_entities):
                        canon_supported += 1

            for ent in sg_entities:
                if ent in checked_tails:
                    continue
                if len(ent) >= 3 and (ent in cap or match_fact(ent, cap)):
                    canon_claims += 1
                    canon_supported += 1

            vsp_canonical = (canon_supported / canon_claims * 100.0) if canon_claims > 0 else 100.0
            tot_canon_claims += canon_claims
            tot_canon_supported += canon_supported
            canon_vsp_all.append(vsp_canonical)
            if canon_claims > 0:
                canon_vsp_active.append(vsp_canonical)

            # 3. Policy & OAR_gen
            n_assertive = sum(1 for m in ASSERTIVE_MARKERS if m in cap)
            n_hedging = sum(1 for m in HEDGING_MARKERS if m in cap)
            n_uncertain = sum(1 for m in UNCERTAIN_MARKERS if m in cap)

            if cert == "HIGH":
                policy_compliant = (n_assertive >= 1 and n_hedging == 0 and n_uncertain == 0)
                oar_gen = 0.0
            elif cert == "MEDIUM":
                policy_compliant = (n_hedging >= 1 or n_uncertain >= 1)
                oar_gen = 100.0 if (n_assertive >= 1 and n_hedging == 0 and n_uncertain == 0) else 0.0
            else:  # LOW
                policy_compliant = (n_assertive == 0 and n_hedging == 0)
                oar_gen = 100.0 if (n_assertive >= 1 or n_hedging >= 1) else 0.0

            policy_val = 100.0 if policy_compliant else 0.0
            policy_all.append(policy_val)
            oar_gen_all.append(oar_gen)

            # 4. OAR_fact & Fact-P
            alt = (s.get("alt_landmark") or "").lower()
            years = re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", cap)
            gt_tails = [str(f.get("tail", "")) for f in gold_facts]
            has_year_halluc = any(not any(y in t for t in gt_tails) for y in years)
            has_wrong_lm = (
                s.get("scenario") == "D_wrong" and alt and alt in cap
                and not any(m in cap for m in HEDGING_MARKERS)
            )
            oar_fact = 50.0 if (has_year_halluc or has_wrong_lm) else 0.0
            fact_p = 100.0 - oar_fact

            oar_fact_all.append(oar_fact)
            fact_p_all.append(fact_p)

            # 5. Lexical
            b4 = compute_bleu4(ref, cap)
            rl = compute_rouge_l(ref, cap)
            bleu4_all.append(b4)
            rougel_all.append(rl)

            per_clip_records.append({
                "model_key": key,
                "model_name": model_label,
                "sample_id": s.get("sample_id", f"test_{idx+1:04d}"),
                "landmark_gt": s.get("landmark_gt", ""),
                "certainty_level": cert,
                "n_gold": n_gold,
                "n_matched": n_matched,
                "k_rec_pct": round(k_rec, 2),
                "canon_claims": canon_claims,
                "canon_supported": canon_supported,
                "canon_vsp_pct": round(vsp_canonical, 2),
                "policy_pct": round(policy_val, 2),
                "oar_gen_pct": round(oar_gen, 2),
                "oar_fact_pct": round(oar_fact, 2),
                "bleu4": round(b4, 4),
                "rouge_l": round(rl, 4),
            })

        # Bootstrap stats
        B = 10000
        k_all_m, k_all_sd, _, _ = bootstrap_ci(np.array(k_rec_all), B=B)
        vsp_all_m, vsp_all_sd, _, _ = bootstrap_ci(np.array(canon_vsp_all), B=B)
        vsp_act_m, vsp_act_sd, _, _ = bootstrap_ci(np.array(canon_vsp_active), B=B) if canon_vsp_active else (0.0, 0.0, 0.0, 0.0)
        pol_m, pol_sd, _, _ = bootstrap_ci(np.array(policy_all), B=B)
        ogen_m, ogen_sd, _, _ = bootstrap_ci(np.array(oar_gen_all), B=B)
        ofact_m, ofact_sd, _, _ = bootstrap_ci(np.array(oar_fact_all), B=B)
        factp_m, factp_sd, _, _ = bootstrap_ci(np.array(fact_p_all), B=B)
        b4_m, b4_sd, _, _ = bootstrap_ci(np.array(bleu4_all), B=B)
        rl_m, rl_sd, _, _ = bootstrap_ci(np.array(rougel_all), B=B)

        k_rec_micro = (tot_matched_gold / max(tot_gold_facts, 1)) * 100.0
        vsp_micro = (tot_canon_supported / max(tot_canon_claims, 1)) * 100.0

        results[key] = {
            "model_name": model_label,
            "k_recall_macro": f"{k_all_m:.1f} ± {k_all_sd:.1f}",
            "k_recall_micro": f"{k_rec_micro:.2f}% ({tot_matched_gold}/{tot_gold_facts})",
            "k_recall_micro_val": round(k_rec_micro, 2),
            "canonical_vsp_macro": f"{vsp_all_m:.1f} ± {vsp_all_sd:.1f}",
            "canonical_vsp_micro": f"{vsp_micro:.2f}% ({tot_canon_supported}/{tot_canon_claims})",
            "canonical_vsp_micro_val": round(vsp_micro, 2),
            "canonical_vsp_active": f"{vsp_act_m:.1f} ± {vsp_act_sd:.1f}",
            "policy": f"{pol_m:.1f} ± {pol_sd:.1f}",
            "oar_gen": f"{ogen_m:.1f} ± {ogen_sd:.1f}",
            "oar_fact": f"{ofact_m:.1f} ± {ofact_sd:.1f}",
            "fact_p": f"{factp_m:.1f} ± {factp_sd:.1f}",
            "bleu4": f"{b4_m:.2f} ± {b4_sd:.2f}",
            "rouge_l": f"{rl_m:.2f} ± {rl_sd:.2f}",
            "tot_canon_claims": tot_canon_claims,
            "tot_canon_supported": tot_canon_supported,
        }

    # Print Table
    print("\n" + "=" * 120)
    print("TABLE 3: DEFINITIVE RE-EVALUATED BENCHMARK UNDER PROTOCOL B CANONICAL EVALUATOR")
    print("=" * 120)
    header = f"{'Model / Architecture':<45} {'K-Rec Macro':>11} {'K-Rec Micro':>12} {'Canonical VSP':>14} {'VSP Micro':>12} {'OAR_gen':>9} {'OAR_fact':>9} {'Fact-P':>9} {'BLEU-4':>8} {'ROUGE-L':>9}"
    print(header)
    print("-" * 120)

    for k in DISPLAY_NAMES.keys():
        if k not in results:
            continue
        r = results[k]
        print(
            f"{r['model_name']:<45} "
            f"{r['k_recall_macro']:>11} "
            f"{r['k_recall_micro'].split()[0]:>12} "
            f"{r['canonical_vsp_macro']:>14} "
            f"{r['canonical_vsp_micro'].split()[0]:>12} "
            f"{r['oar_gen']:>9} "
            f"{r['oar_fact']:>9} "
            f"{r['fact_p']:>9} "
            f"{r['bleu4']:>8} "
            f"{r['rouge_l']:>9}"
        )
    print("=" * 120)

    # Save JSON & CSV
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved final JSON to %s", OUT_JSON)

    # Save CSV
    cols = ["model_name", "k_recall_macro", "k_recall_micro", "canonical_vsp_macro", "canonical_vsp_micro", "canonical_vsp_active", "policy", "oar_gen", "oar_fact", "fact_p", "bleu4", "rouge_l"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key"] + cols)
        for k in DISPLAY_NAMES.keys():
            if k not in results:
                continue
            r = results[k]
            w.writerow([k] + [r.get(c, "") for c in cols])
    LOGGER.info("Saved final CSV to %s", OUT_CSV)

    # Save per clip
    df_per_clip = pd.DataFrame(per_clip_records)
    df_per_clip.to_csv(OUT_PER_CLIP_CSV, index=False)
    LOGGER.info("Saved per-clip audit CSV to %s", OUT_PER_CLIP_CSV)

if __name__ == "__main__":
    main()
