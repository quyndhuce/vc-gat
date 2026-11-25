#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_review_into_scene_conditioned.py
=============================================================================
Merges the binary review results from:
  /Users/quynd/Downloads/fact_review_results_binary.json
into the Scene-Conditioned F_gold datasets:
  - scene_conditioned_gold/outputs/scene_conditioned_fgold_horn_complete.json (Master 1,163 clips)
  - scene_conditioned_gold/outputs/fgold_test_800_clips.json (Test 800 clips)
  - scene_conditioned_gold/outputs/fgold_facts_dictionary.json (16,347 ontology facts)
  - scene_conditioned_gold/outputs/fgold_test_evidence_audit_table.tsv (TSV table with review columns)
  - scene_conditioned_gold/outputs/scene_conditioned_gold_horn_complete.jsonl
  - scene_conditioned_gold/outputs/scene_conditioned_gold_train_val_horn_complete.jsonl

Generates detailed binary statistics across all videos, by split, and by landmark.
=============================================================================
"""

import csv
import json
import logging
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("MergeFactReviewBinary")

REPO_ROOT = Path("/Users/quynd/Downloads/TransMTL_OntoKG-main")
REVIEW_FILE = Path("/Users/quynd/Downloads/fact_review_results_binary.json")
OUTPUT_DIR = REPO_ROOT / "scene_conditioned_gold" / "outputs"

MASTER_FILE = OUTPUT_DIR / "scene_conditioned_fgold_horn_complete.json"
TEST_FILE = OUTPUT_DIR / "fgold_test_800_clips.json"
DICT_FILE = OUTPUT_DIR / "fgold_facts_dictionary.json"
TSV_FILE = OUTPUT_DIR / "fgold_test_evidence_audit_table.tsv"
REPORT_JSON = OUTPUT_DIR / "fact_review_aggregation_report.json"
TEST_JSONL = OUTPUT_DIR / "scene_conditioned_gold_horn_complete.jsonl"
TRAIN_VAL_JSONL = OUTPUT_DIR / "scene_conditioned_gold_train_val_horn_complete.jsonl"


def main():
    LOGGER.info("=" * 70)
    LOGGER.info("MERGING BINARY FACT REVIEW RESULTS INTO SCENE-CONDITIONED DATASET")
    LOGGER.info("=" * 70)

    # 1. Load Binary Review Results
    LOGGER.info(f"Loading binary review results from {REVIEW_FILE}...")
    with open(REVIEW_FILE, "r", encoding="utf-8") as f:
        review_data = json.load(f)

    reviews_list = review_data.get("reviews", [])
    review_map = {}
    for r in reviews_list:
        review_map[r["fact_id"]] = {
            "result": r.get("result", "sai"),
            "reason": r.get("reason", "")
        }
    LOGGER.info(f"Loaded {len(review_map):,} binary fact reviews ({review_data.get('metadata', {}).get('counts', {})}).")

    # 2. Update fgold_facts_dictionary.json
    LOGGER.info(f"Updating ontology facts dictionary in {DICT_FILE}...")
    with open(DICT_FILE, "r", encoding="utf-8") as f:
        dict_data = json.load(f)

    dict_review_counts = Counter()
    for fid, fact in dict_data["facts"].items():
        rev = review_map.get(fid, {"result": "sai", "reason": "chưa có review"})
        fact["review"] = rev
        dict_review_counts[rev["result"]] += 1

    dict_data["metadata"]["review_integration"] = {
        "review_file": "fact_review_results_binary.json",
        "review_mode": "binary",
        "timestamp": datetime.now().isoformat(),
        "counts": dict(dict_review_counts)
    }

    with open(DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(dict_data, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Updated dictionary: {dict_review_counts}")

    # 3. Update scene_conditioned_fgold_horn_complete.json (Master)
    LOGGER.info(f"Updating Master dataset in {MASTER_FILE}...")
    with open(MASTER_FILE, "r", encoding="utf-8") as f:
        master_data = json.load(f)

    overall_stats = Counter()
    split_stats = defaultdict(Counter)
    lm_stats = defaultdict(lambda: {"clips": 0, "total": 0, "đúng": 0, "sai": 0})
    clip_stats = {}

    for split in ["test", "train", "val"]:
        clips = master_data[split]
        for cid, cdata in clips.items():
            lm = cdata["landmark_name"]
            lm_id = cdata["landmark_id"]
            lm_stats[lm]["clips"] += 1

            c_counts = Counter()
            for gf in cdata["gold_facts"]:
                fid = gf["fact_id"]
                rev = review_map.get(fid, {"result": "sai", "reason": "chưa có review"})
                gf["review"] = rev

                res = rev["result"]
                overall_stats[res] += 1
                split_stats[split][res] += 1
                lm_stats[lm]["total"] += 1
                lm_stats[lm][res] += 1
                c_counts[res] += 1

            cdata["review_summary"] = dict(c_counts)
            clip_stats[cid] = {
                "clip_id": cid,
                "split": split,
                "landmark_name": lm,
                "total_gold_facts": len(cdata["gold_facts"]),
                "review_summary": dict(c_counts)
            }

    master_data["metadata"]["review_integration"] = {
        "review_file": "fact_review_results_binary.json",
        "review_mode": "binary",
        "timestamp": datetime.now().isoformat(),
        "overall_gold_counts": dict(overall_stats),
        "split_gold_counts": {sp: dict(split_stats[sp]) for sp in split_stats}
    }

    with open(MASTER_FILE, "w", encoding="utf-8") as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Master file updated: Overall Gold Facts Review = {dict(overall_stats)}")

    # 4. Update fgold_test_800_clips.json
    LOGGER.info(f"Updating Test dataset in {TEST_FILE}...")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_stats = Counter()
    audit_rows = []

    for cid, cdata in test_data["clips"].items():
        c_counts = Counter()
        for gf in cdata["gold_facts"]:
            fid = gf["fact_id"]
            rev = review_map.get(fid, {"result": "sai", "reason": "chưa có review"})
            gf["review"] = rev
            res = rev["result"]
            test_stats[res] += 1
            c_counts[res] += 1

            ev = gf.get("evidence", {})
            audit_rows.append({
                "clip_id": cid,
                "landmark_name": cdata.get("landmark_name", ""),
                "fact_id": fid,
                "fact_text": gf.get("fact_text", ""),
                "provenance_type": gf.get("provenance_type", "KB_Base"),
                "visual_grounding": gf.get("visual_grounding", "UNKNOWN"),
                "review_result": res,
                "review_reason": rev.get("reason", ""),
                "original_sentence": ev.get("original_sentence", ""),
                "source_document": ev.get("source_document", ""),
                "extraction_method": ev.get("extraction_method", ""),
                "derivation_rule": ev.get("derivation_rule", ""),
                "confidence_score": ev.get("confidence_score", 1.0),
            })
        cdata["review_summary"] = dict(c_counts)

    test_data["metadata"]["review_integration"] = {
        "review_file": "fact_review_results_binary.json",
        "review_mode": "binary",
        "timestamp": datetime.now().isoformat(),
        "test_gold_counts": dict(test_stats)
    }

    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Test file updated: Test Gold Facts Review = {dict(test_stats)}")

    # 5. Update TSV Audit Table with Binary Review Result and Reason
    LOGGER.info(f"Updating TSV audit table in {TSV_FILE}...")
    fieldnames = [
        "clip_id",
        "landmark_name",
        "fact_id",
        "fact_text",
        "provenance_type",
        "visual_grounding",
        "review_result",
        "review_reason",
        "original_sentence",
        "source_document",
        "extraction_method",
        "derivation_rule",
        "confidence_score",
    ]
    with open(TSV_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(audit_rows)
    LOGGER.info(f"TSV audit table updated with {len(audit_rows):,} rows.")

    # 6. Update JSONL files
    for p in [TEST_JSONL, TRAIN_VAL_JSONL]:
        if p.exists():
            LOGGER.info(f"Updating JSONL file: {p.name}...")
            tmp_p = p.with_suffix(".tmp")
            with open(p, "r", encoding="utf-8") as fin, open(tmp_p, "w", encoding="utf-8") as fout:
                for line in fin:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    fid = item.get("fact_id")
                    rev = review_map.get(fid, {"result": "sai", "reason": "chưa có review"})
                    item["review_result"] = rev["result"]
                    item["review_reason"] = rev["reason"]
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            tmp_p.replace(p)
            LOGGER.info(f"Finished updating {p.name}.")

    # 7. Save Aggregated Report JSON
    report_dict = {
        "metadata": {
            "title": "Scene-Conditioned F_gold Binary Review Statistics Report",
            "source_review_file": str(REVIEW_FILE),
            "generated_at": datetime.now().isoformat(),
            "review_mode": "binary (đúng / sai)",
            "total_clips": len(clip_stats),
            "total_positive_gold_fact_instances": sum(overall_stats.values()),
        },
        "overall_summary": {
            "total_instances": sum(overall_stats.values()),
            "đúng": overall_stats["đúng"],
            "đúng_percent": round(overall_stats["đúng"] / sum(overall_stats.values()) * 100, 2),
            "sai": overall_stats["sai"],
            "sai_percent": round(overall_stats["sai"] / sum(overall_stats.values()) * 100, 2),
        },
        "split_summary": {
            sp: {
                "total": sum(split_stats[sp].values()),
                "đúng": split_stats[sp]["đúng"],
                "đúng_percent": round(split_stats[sp]["đúng"] / sum(split_stats[sp].values()) * 100, 2),
                "sai": split_stats[sp]["sai"],
                "sai_percent": round(split_stats[sp]["sai"] / sum(split_stats[sp].values()) * 100, 2),
            }
            for sp in split_stats
        },
        "per_landmark_summary": {
            lm: {
                "clips": data["clips"],
                "total_facts": data["total"],
                "đúng": data["đúng"],
                "đúng_percent": round(data["đúng"] / data["total"] * 100, 2) if data["total"] else 0,
                "sai": data["sai"],
                "sai_percent": round(data["sai"] / data["total"] * 100, 2) if data["total"] else 0,
            }
            for lm, data in sorted(lm_stats.items(), key=lambda x: x[0])
        }
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved aggregated binary report JSON to {REPORT_JSON}")

    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("BINARY REVIEW MERGE & STATISTICS COMPLETED SUCCESSFULLY!")
    LOGGER.info(f"  Overall: Đúng = {overall_stats['đúng']:,} ({overall_stats['đúng']/sum(overall_stats.values())*100:.2f}%), Sai = {overall_stats['sai']:,} ({overall_stats['sai']/sum(overall_stats.values())*100:.2f}%)")
    LOGGER.info(f"  Test Split: Đúng = {test_stats['đúng']:,} ({test_stats['đúng']/sum(test_stats.values())*100:.2f}%), Sai = {test_stats['sai']:,} ({test_stats['sai']/sum(test_stats.values())*100:.2f}%)")
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()
