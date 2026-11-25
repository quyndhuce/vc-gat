#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_tau_1_test_instances.py
=============================================================================
Audit of Test Clip Instances Retained at tau = 1.0 (Precision 90.30%):

Answers User's Exact Scientific Inquiry:
  - Total retained clip-fact instances: 835 (precision 90.30%)
  - How many unique facts do they correspond to?
  - How many unique landmarks are represented?
  - Is the high precision dominated by duplicate scenes / single landmark repetition,
    or does it reflect genuine cross-landmark certainty?
=============================================================================
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("AuditTau1")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
TEST_PAIRS_PATH = REPO_ROOT / "experiments" / "outputs" / "mu_f_vs_fref_test_pairs.json"
VERIFIED_PATH = REPO_ROOT / "experiments" / "outputs" / "fkg_facts_verified_v2.json"
OUTPUT_REPORT = REPO_ROOT / "experiments" / "outputs" / "audit_tau_1_instances_breakdown.json"


def audit_tau_1():
    LOGGER.info(f"Loading test pairs from {TEST_PAIRS_PATH}...")
    with open(TEST_PAIRS_PATH, "r", encoding="utf-8") as f:
        test_pairs = json.load(f)

    LOGGER.info(f"Loading verified facts V2 from {VERIFIED_PATH}...")
    with open(VERIFIED_PATH, "r", encoding="utf-8") as f:
        verified_facts = json.load(f)

    # Filter to instances with mu_F_v2 >= 1.0 (or rounded to 1.0, >= 0.999)
    tau_1_instances = []
    for p in test_pairs:
        fid = p["fact_id"]
        vf = verified_facts.get(fid)
        if vf:
            score = vf.get("mu_F_v2", 0.0)
            if score >= 0.999:
                tau_1_instances.append({
                    **p,
                    "score": score,
                    "subject": vf.get("subject"),
                    "predicate": vf.get("predicate"),
                    "object": vf.get("object"),
                    "landmark_id": vf.get("landmark_id"),
                    "provenance_type": vf.get("provenance_type")
                })

    LOGGER.info(f"Found {len(tau_1_instances):,} instances at tau = 1.0.")

    # Unique facts
    unique_fids = set(inst["fact_id"] for inst in tau_1_instances)
    unique_triples = set((inst["subject"], inst["predicate"], inst["object"]) for inst in tau_1_instances)
    unique_landmarks = set(inst["landmark_id"] for inst in tau_1_instances)
    unique_clips = set(inst.get("clip_id") for inst in tau_1_instances if "clip_id" in inst)

    # Ground truth precision
    y_true = [inst["y_true"] for inst in tau_1_instances]
    tp = sum(y_true)
    fp = len(y_true) - tp
    precision = tp / len(y_true) * 100.0 if y_true else 0.0

    # Landmark distribution
    lm_counts = Counter(inst["landmark_id"] for inst in tau_1_instances)
    provenance_counts = Counter(inst["provenance_type"] for inst in tau_1_instances)
    fact_repetition_counts = Counter(inst["fact_id"] for inst in tau_1_instances)

    # Top repeated facts
    top_repeated_facts = []
    for fid, count in fact_repetition_counts.most_common(10):
        sample = next(inst for inst in tau_1_instances if inst["fact_id"] == fid)
        top_repeated_facts.append({
            "fact_id": fid,
            "triple": f"({sample['subject']}, {sample['predicate']}, {sample['object']})",
            "landmark_id": sample["landmark_id"],
            "count": count,
            "y_true": sample["y_true"]
        })

    print("\n" + "=" * 85)
    print("AUDIT OF TEST CLIP INSTANCES RETAINED AT TAU = 1.0")
    print("=" * 85)
    print(f"Total Clip-Fact Instances Retained:  {len(tau_1_instances):,}")
    print(f"Ground Truth Precision:             {precision:.2f}% (TP = {tp}, FP = {fp})")
    print(f"Unique Fact IDs Represented:        {len(unique_fids):,}")
    print(f"Unique Semantic Triples:            {len(unique_triples):,}")
    print(f"Unique Landmarks Represented:       {len(unique_landmarks):,} / 35")
    print(f"Unique Video Clips Represented:     {len(unique_clips):,}")
    print(f"Average Instances per Unique Fact:  {len(tau_1_instances) / len(unique_fids):.2f}")
    print(f"Max Instances for Any Single Fact:  {max(fact_repetition_counts.values())}")
    print("-" * 85)
    print("Provenance Breakdown:")
    for prov, count in provenance_counts.items():
        print(f"  {prov:<25}: {count:4d} instances ({count / len(tau_1_instances) * 100:.1f}%)")
    print("-" * 85)
    print(f"Top 5 Landmarks Represented:")
    for lm, count in lm_counts.most_common(5):
        print(f"  {lm:<32}: {count:4d} instances ({count / len(tau_1_instances) * 100:.1f}%)")
    print("=" * 85 + "\n")

    report = {
        "total_instances_retained": len(tau_1_instances),
        "precision": round(precision, 2),
        "tp": tp,
        "fp": fp,
        "unique_fact_ids": len(unique_fids),
        "unique_triples": len(unique_triples),
        "unique_landmarks_count": len(unique_landmarks),
        "unique_clips_count": len(unique_clips),
        "avg_instances_per_fact": round(len(tau_1_instances) / len(unique_fids), 2),
        "max_instances_per_fact": max(fact_repetition_counts.values()),
        "provenance_distribution": dict(provenance_counts),
        "landmark_distribution": dict(lm_counts),
        "top_repeated_facts": top_repeated_facts
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved audit report to {OUTPUT_REPORT}")


if __name__ == "__main__":
    audit_tau_1()
