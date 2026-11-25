#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_100_kb_base_misranked.py
=============================================================================
Audit of 100 Misranked KB_Base Facts:
  - 50 Worst False Positives (y=0, highest w_base)
  - 50 Worst False Negatives (y=1, lowest w_base == 0.0)

Extracts:
  - subject, predicate, object, landmark_id
  - y_true, review_result, review_reason
  - bel_T, bel_F, uncertainty, conflict, w_base, old_mu_F
  - num_independent_sources
  - Full list of NLI evidence (source_type, reliability, p_E, p_C, p_N, premise)
  - Diagnosed Failure Mode:
      * ONTOLOGY_FRAGMENT: Dummy headers, punctuation, stopwords, abstract phrases
      * NLI_FALSE_CONTRADICTION: DeBERTa predicts p_C instead of p_N on hierarchy/omission
      * RETRIEVAL_TOPIC_DRIFT: BM25 retrieves irrelevant sub-entity passage
      * DST_OVERPENALIZATION: Single conflicting evidence zeroes out high entailment
      * INVERTED_RELATION: Directional meronym inverted in triple but marked true by human

Exports:
  - experiments/outputs/audit_100_kb_base_misranked.json
  - experiments/outputs/audit_100_kb_base_misranked.tsv
=============================================================================
"""

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("AuditKBBase")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
VERIFIED_PATH = REPO_ROOT / "experiments" / "outputs" / "fkg_facts_verified.json"
OUTPUT_JSON_PATH = REPO_ROOT / "experiments" / "outputs" / "audit_100_kb_base_misranked.json"
OUTPUT_TSV_PATH = REPO_ROOT / "experiments" / "outputs" / "audit_100_kb_base_misranked.tsv"

STOPWORDS = {"the", "a", "an", "and", "or", "in", "on", "at", "it", "this", "that", "there", ".", ",", "-", ";"}
DUMMY_HEADERS = {
    "characteristic physical features", "visual components", "physical features",
    "named places", "sections", "parts", "exterior", "inside", "performance", "design"
}


def diagnose_failure_mode(f: Dict[str, Any], error_type: str) -> str:
    s = (f.get("subject") or "").strip().lower()
    o = (f.get("object") or "").strip().lower()
    reason = (f.get("review_reason") or "").lower()

    if error_type == "False_Positive":
        # Check ontology/schema fragment
        if s in STOPWORDS or o in STOPWORDS or len(s) <= 1 or len(o) <= 1:
            return "ONTOLOGY_STOPWORD_PUNCTUATION"
        if s in DUMMY_HEADERS or o in DUMMY_HEADERS or "mảnh vụn" in reason:
            return "ONTOLOGY_SYNTACTIC_FRAGMENT"
        if "không hợp lệ" in reason or "không hợp lý" in reason:
            return "ONTOLOGY_SEMANTIC_INVALIDITY"
        return "NLI_HALLUCINATION_FALSE_ENTAILMENT"

    else:  # False_Negative
        ev_list = f.get("evidence_details", [])
        has_high_pe = any(ev["p_E"] > 0.7 for ev in ev_list)
        has_high_pc = any(ev["p_C"] > 0.7 for ev in ev_list)
        dst = f.get("dst", {})

        if "đảo chiều" in reason or ("bao hàm" in reason and has_high_pc):
            return "INVERTED_RELATION_TRIPLE"
        if has_high_pe and has_high_pc:
            return "DST_CONFLICT_ZERO_KILL"
        if has_high_pc and not has_high_pe:
            # Check if premise doesn't even mention the object
            top_c_ev = max(ev_list, key=lambda x: x["p_C"])
            if o not in top_c_ev["premise"].lower() and s not in top_c_ev["premise"].lower():
                return "RETRIEVAL_TOPIC_DRIFT"
            return "NLI_SPURIOUS_CONTRADICTION"
        if all(ev["p_N"] > 0.6 for ev in ev_list):
            return "RETRIEVAL_MISSING_EVIDENCE"

        return "DST_LOW_BELIEF"


def generate_audit_report():
    LOGGER.info(f"Loading verified facts from {VERIFIED_PATH}...")
    with open(VERIFIED_PATH, "r", encoding="utf-8") as f:
        facts = json.load(f)

    kb_base = [f for f in facts.values() if f.get("provenance_type") == "KB_Base"]
    LOGGER.info(f"Loaded {len(kb_base):,} KB_Base facts.")

    false_facts = [f for f in kb_base if f["y_true"] == 0]
    true_facts = [f for f in kb_base if f["y_true"] == 1]

    # Select top 50 False Positives (sorted by w_base descending)
    top_50_fp = sorted(false_facts, key=lambda f: (-f["dst"]["w_base"], -f["dst"]["bel_T"]))[:50]

    # Select top 50 False Negatives (sorted by w_base ascending, conflict descending, old_mu_F descending)
    top_50_fn = sorted(
        true_facts,
        key=lambda f: (f["dst"]["w_base"], -f["dst"]["conflict"], -f["old_mu_F"])
    )[:50]

    audit_records = []

    # Process False Positives
    for rank, f in enumerate(top_50_fp, 1):
        mode = diagnose_failure_mode(f, "False_Positive")
        record = {
            "audit_id": f"FP_{rank:02d}",
            "error_type": "False_Positive",
            "rank_in_group": rank,
            "fact_id": f["fact_id"],
            "subject": f["subject"],
            "predicate": f["predicate"],
            "object": f["object"],
            "landmark_id": f["landmark_id"],
            "y_true": f["y_true"],
            "review_result": f["review_result"],
            "review_reason": f.get("review_reason", ""),
            "w_base": f["dst"]["w_base"],
            "old_mu_F": f["old_mu_F"],
            "bel_T": f["dst"]["bel_T"],
            "bel_F": f["dst"]["bel_F"],
            "uncertainty": f["dst"]["uncertainty"],
            "conflict": f["dst"]["conflict"],
            "num_evidence": f["num_evidence"],
            "diagnosed_failure_mode": mode,
            "evidence_details": f.get("evidence_details", [])
        }
        audit_records.append(record)

    # Process False Negatives
    for rank, f in enumerate(top_50_fn, 1):
        mode = diagnose_failure_mode(f, "False_Negative")
        record = {
            "audit_id": f"FN_{rank:02d}",
            "error_type": "False_Negative",
            "rank_in_group": rank,
            "fact_id": f["fact_id"],
            "subject": f["subject"],
            "predicate": f["predicate"],
            "object": f["object"],
            "landmark_id": f["landmark_id"],
            "y_true": f["y_true"],
            "review_result": f["review_result"],
            "review_reason": f.get("review_reason", ""),
            "w_base": f["dst"]["w_base"],
            "old_mu_F": f["old_mu_F"],
            "bel_T": f["dst"]["bel_T"],
            "bel_F": f["dst"]["bel_F"],
            "uncertainty": f["dst"]["uncertainty"],
            "conflict": f["dst"]["conflict"],
            "num_evidence": f["num_evidence"],
            "diagnosed_failure_mode": mode,
            "evidence_details": f.get("evidence_details", [])
        }
        audit_records.append(record)

    LOGGER.info(f"Generated {len(audit_records)} audit records (50 FP, 50 FN).")

    # Save to JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_records, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved audit JSON to {OUTPUT_JSON_PATH}")

    # Save to TSV
    tsv_fields = [
        "audit_id", "error_type", "fact_id", "subject", "predicate", "object",
        "landmark_id", "y_true", "review_reason", "w_base", "bel_T", "bel_F",
        "conflict", "uncertainty", "num_evidence", "diagnosed_failure_mode",
        "top_evidence_pe", "top_evidence_pc", "top_evidence_pn", "top_evidence_premise"
    ]
    with open(OUTPUT_TSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(tsv_fields)
        for r in audit_records:
            top_ev = r["evidence_details"][0] if r["evidence_details"] else {}
            premise_short = top_ev.get("premise", "").replace("\t", " ").replace("\n", " ")[:150]
            writer.writerow([
                r["audit_id"],
                r["error_type"],
                r["fact_id"],
                r["subject"],
                r["predicate"],
                r["object"],
                r["landmark_id"],
                r["y_true"],
                r["review_reason"],
                r["w_base"],
                r["bel_T"],
                r["bel_F"],
                r["conflict"],
                r["uncertainty"],
                r["num_evidence"],
                r["diagnosed_failure_mode"],
                top_ev.get("p_E", 0.0),
                top_ev.get("p_C", 0.0),
                top_ev.get("p_N", 0.0),
                premise_short
            ])
    LOGGER.info(f"Saved audit TSV to {OUTPUT_TSV_PATH}")

    # Print summary statistics
    fp_modes = Counter(r["diagnosed_failure_mode"] for r in audit_records if r["error_type"] == "False_Positive")
    fn_modes = Counter(r["diagnosed_failure_mode"] for r in audit_records if r["error_type"] == "False_Negative")

    print("\n" + "=" * 70)
    print("AUDIT FAILURE MODE BREAKDOWN (100 MISRANKED KB_BASE FACTS)")
    print("=" * 70)
    print("--- 1. False Positives (Worst 50 FALSE facts with high w_base) ---")
    for m, c in fp_modes.most_common():
        print(f"  {m:<35}: {c:>2} facts ({c/50*100:>5.1f}%)")

    print("\n--- 2. False Negatives (Worst 50 TRUE facts with w_base == 0.0) ---")
    for m, c in fn_modes.most_common():
        print(f"  {m:<35}: {c:>2} facts ({c/50*100:>5.1f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    generate_audit_report()
