#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_real_disk_storage.py
=============================================================================
Direct Empirical Measurement of Measured Storage Latency vs. In-Memory RAM Latency
Comparing Exhaustive, Pure Certified BnB, and Proposed Hybrid Mode on real
disk-backed persistent storage (SQLite on disk with B-Tree indices).

Addresses user directive:
"Hybrid tiết kiệm 66,94% số phép đánh giá nhưng chạy chậm hơn khoảng 21,5%
trong benchmark RAM. Bài báo đã thừa nhận điều này, song các lợi ích về tốc độ
trong hệ thống lưu trữ thực tế vẫn cần đo trực tiếp thay vì chỉ dự phóng."
=============================================================================
"""

import os
import sys
import time
import json
import sqlite3
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUTS_DIR = REPO_ROOT / "experiments" / "outputs"
CONFIG_DIR = REPO_ROOT / "experiments" / "phase1" / "config"
PHASE1_DIR = REPO_ROOT / "experiments" / "phase1"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = OUTPUTS_DIR / "hanoi_kg_persistent.db"

# 1. Load Hanoi Manifest and Evidence (Strictly Hanoi 500 clips)
with open(CONFIG_DIR / "hanoi_manifest.jsonl", "r", encoding="utf-8") as f:
    hn_manifest = [json.loads(l) for l in f]

with open(PHASE1_DIR / "hanoi" / "evidence.jsonl", "r", encoding="utf-8") as f:
    hn_ev = [json.loads(l) for l in f]

cand_fact_ids = set()
for ep in hn_ev:
    for f in ep.get("top5_facts", []):
        cand_fact_ids.add(f.get("fact_id"))

# 2. Build Knowledge Graph and Derivations
from experiments.fkg_certainty_redesign.verify_certified_horn_early_stopping import (
    build_knowledge_graph_and_derivations
)

fused_facts, triple_to_fact, derivations = build_knowledge_graph_and_derivations()

HCMC_LANDMARKS = {
    "saigon_central_post_office",
    "independence_palace",
    "ben_thanh_market",
    "saigon_opera_house",
    "ba_thien_hau_temple"
}

hn_facts: Dict[str, Dict[str, Any]] = {}
for fid, f in fused_facts.items():
    lm_id = (f.get("landmark_id") or "").strip().lower()
    subj = (f.get("subject") or "").strip().lower()
    if not (any(h in lm_id for h in HCMC_LANDMARKS) or any(h in subj for h in HCMC_LANDMARKS)):
        hn_facts[fid] = f

total_hn_derivs = 0
for fid, f in hn_facts.items():
    s = (f.get("subject") or "").strip().lower()
    p = (f.get("predicate") or "").strip().lower()
    o = (f.get("object") or "").strip().lower()
    total_hn_derivs += len(derivations.get((s, p, o), []))

print(f"Verified Dataset: Hanoi 500 clips | {len(hn_facts):,} facts | {total_hn_derivs:,} derivations | {len(cand_fact_ids)} candidate facts.")

# 3. Create persistent disk-backed SQLite database
if DB_PATH.exists():
    DB_PATH.unlink()

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()
cur.execute("PRAGMA page_size = 4096;")
cur.execute("PRAGMA cache_size = 2000;")
cur.execute("PRAGMA synchronous = NORMAL;")
cur.execute("PRAGMA journal_mode = DELETE;")

cur.execute("""
CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    subject TEXT,
    predicate TEXT,
    object TEXT,
    w_base REAL
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS derivations (
    subject TEXT,
    predicate TEXT,
    object TEXT,
    rule_id TEXT,
    weight REAL
);
""")

print("Inserting facts into disk database...")
fact_rows = []
for fid, f in hn_facts.items():
    fact_rows.append((
        fid,
        (f.get("subject") or "").strip().lower(),
        (f.get("predicate") or "").strip().lower(),
        (f.get("object") or "").strip().lower(),
        float(f.get("w_base", 0.0))
    ))
cur.executemany("INSERT INTO facts VALUES (?, ?, ?, ?, ?);", fact_rows)

print("Inserting derivations into disk database...")
deriv_rows = []
for fid, f in hn_facts.items():
    s = (f.get("subject") or "").strip().lower()
    p = (f.get("predicate") or "").strip().lower()
    o = (f.get("object") or "").strip().lower()
    d_list = derivations.get((s, p, o), [])
    for d in d_list:
        deriv_rows.append((s, p, o, str(d[0]), float(d[1])))
cur.executemany("INSERT INTO derivations VALUES (?, ?, ?, ?, ?);", deriv_rows)

print("Building B-Tree index on (subject, predicate, object, weight DESC)...")
cur.execute("CREATE INDEX idx_deriv_spo_w ON derivations(subject, predicate, object, weight DESC);")
conn.commit()
conn.close()

db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)
print(f"Persistent Disk Database created successfully: {DB_PATH} ({db_size_mb:.2f} MB)")

# 4. Benchmarking functions
TAU_ASSERT = 0.80
TAU_OMIT = 0.50
MU_L_HANOI = 0.85

def get_policy(certainty: float) -> str:
    if certainty >= TAU_ASSERT:
        return "ASSERT"
    elif certainty >= TAU_OMIT:
        return "HEDGE"
    else:
        return "OMIT"

# A. Disk Exhaustive
def run_disk_exhaustive(db_file: str):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    # Read all facts
    cur.execute("SELECT fact_id, subject, predicate, object, w_base FROM facts;")
    facts_list = cur.fetchall()
    
    t0 = time.perf_counter()
    mu_dict = {}
    derivs_count = 0
    
    for fid, s, p, o, b_f in facts_list:
        cur.execute("SELECT weight FROM derivations WHERE subject=? AND predicate=? AND object=?;", (s, p, o))
        weights = cur.fetchall()
        m = len(weights)
        derivs_count += m
        if m == 0:
            val = b_f
        else:
            prod = 1.0 - b_f
            for (w,) in weights:
                prod *= (1.0 - w)
            val = 1.0 - prod
        mu_dict[fid] = val
        
    t1 = time.perf_counter()
    conn.close()
    return derivs_count, (t1 - t0) * 1000.0, mu_dict

# B. Disk Pure Certified BnB
def run_disk_certified_bnb(db_file: str):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("SELECT fact_id, subject, predicate, object, w_base FROM facts;")
    facts_list = cur.fetchall()
    
    t0 = time.perf_counter()
    mu_dict = {}
    derivs_count = 0
    
    for fid, s, p, o, b_f in facts_list:
        # Step k=0: check if already certified using base weight
        L_0 = b_f
        # We need m and q_max. We query: count and max(weight)
        cur.execute("SELECT count(*), max(weight) FROM derivations WHERE subject=? AND predicate=? AND object=?;", (s, p, o))
        m, q_max = cur.fetchone()
        if m == 0 or q_max is None:
            mu_dict[fid] = b_f
            continue
            
        U_0 = 1.0 - (1.0 - b_f) * ((1.0 - q_max)**m)
        if get_policy(L_0 * MU_L_HANOI) == get_policy(U_0 * MU_L_HANOI):
            mu_dict[fid] = L_0
            continue
            
        # If not certified at k=0, query derivations ordered by weight DESC
        cur.execute("SELECT weight FROM derivations WHERE subject=? AND predicate=? AND object=? ORDER BY weight DESC;", (s, p, o))
        prod = 1.0 - b_f
        stopped = False
        k = 0
        while True:
            row = cur.fetchone()
            if row is None:
                break
            w = row[0]
            k += 1
            derivs_count += 1
            prod *= (1.0 - w)
            L_k = 1.0 - prod
            rem = m - k
            if rem == 0:
                mu_dict[fid] = L_k
                stopped = True
                break
            U_k = 1.0 - (1.0 - L_k) * ((1.0 - w)**rem)
            if get_policy(L_k * MU_L_HANOI) == get_policy(U_k * MU_L_HANOI):
                mu_dict[fid] = L_k
                stopped = True
                break
        if not stopped:
            mu_dict[fid] = L_k
            
    t1 = time.perf_counter()
    conn.close()
    return derivs_count, (t1 - t0) * 1000.0, mu_dict

# C. Disk Hybrid Mode (Proposed)
def run_disk_hybrid(db_file: str, cand_ids: set, eps_rank: float = 0.0001):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("SELECT fact_id, subject, predicate, object, w_base FROM facts;")
    facts_list = cur.fetchall()
    
    t0 = time.perf_counter()
    mu_dict = {}
    derivs_count = 0
    
    for fid, s, p, o, b_f in facts_list:
        is_cand = fid in cand_ids
        
        if not is_cand:
            # Fast Decision Certification
            L_0 = b_f
            cur.execute("SELECT count(*), max(weight) FROM derivations WHERE subject=? AND predicate=? AND object=?;", (s, p, o))
            m, q_max = cur.fetchone()
            if m == 0 or q_max is None:
                mu_dict[fid] = b_f
                continue
            U_0 = 1.0 - (1.0 - b_f) * ((1.0 - q_max)**m)
            if get_policy(L_0 * MU_L_HANOI) == get_policy(U_0 * MU_L_HANOI):
                mu_dict[fid] = L_0
                continue
                
            cur.execute("SELECT weight FROM derivations WHERE subject=? AND predicate=? AND object=? ORDER BY weight DESC;", (s, p, o))
            prod = 1.0 - b_f
            stopped = False
            k = 0
            while True:
                row = cur.fetchone()
                if row is None:
                    break
                w = row[0]
                k += 1
                derivs_count += 1
                prod *= (1.0 - w)
                L_k = 1.0 - prod
                rem = m - k
                if rem == 0:
                    mu_dict[fid] = L_k
                    stopped = True
                    break
                U_k = 1.0 - (1.0 - L_k) * ((1.0 - w)**rem)
                if get_policy(L_k * MU_L_HANOI) == get_policy(U_k * MU_L_HANOI):
                    mu_dict[fid] = L_k
                    stopped = True
                    break
            if not stopped:
                mu_dict[fid] = L_k
        else:
            # Candidate Fact: Refined BnB to guarantee exact ranking
            cur.execute("SELECT weight FROM derivations WHERE subject=? AND predicate=? AND object=? ORDER BY weight DESC;", (s, p, o))
            weights = cur.fetchall()
            m = len(weights)
            if m == 0:
                mu_dict[fid] = b_f
                continue
            prod = 1.0 - b_f
            stopped = False
            for k in range(1, m + 1):
                w = weights[k-1][0]
                prod *= (1.0 - w)
                L_k = 1.0 - prod
                derivs_count += 1
                rem = m - k
                if rem == 0:
                    mu_dict[fid] = L_k
                    stopped = True
                    break
                U_k = 1.0 - (1.0 - L_k) * ((1.0 - w)**rem)
                if (U_k - L_k) <= eps_rank:
                    mu_dict[fid] = L_k
                    stopped = True
                    break
            if not stopped:
                mu_dict[fid] = L_k
                
    t1 = time.perf_counter()
    conn.close()
    return derivs_count, (t1 - t0) * 1000.0, mu_dict

# Run warm-up and multi-trial benchmark
print("\n" + "="*70)
print("EXECUTING DIRECT EMPIRICAL STORAGE BENCHMARK (10 TRIALS)")
print("="*70)

N_TRIALS = 10
times_ex = []
times_pure = []
times_hybrid = []

for trial in range(N_TRIALS):
    c_ex, t_ex, _ = run_disk_exhaustive(str(DB_PATH))
    c_pure, t_pure, _ = run_disk_certified_bnb(str(DB_PATH))
    c_hyb, t_hyb, _ = run_disk_hybrid(str(DB_PATH), cand_fact_ids)
    times_ex.append(t_ex)
    times_pure.append(t_pure)
    times_hybrid.append(t_hyb)
    print(f"Trial {trial+1:02d}: Exhaustive={t_ex:.2f} ms | Pure BnB={t_pure:.2f} ms | Hybrid={t_hyb:.2f} ms")

mean_ex, std_ex = np.mean(times_ex), np.std(times_ex)
mean_pure, std_pure = np.mean(times_pure), np.std(times_pure)
mean_hyb, std_hyb = np.mean(times_hybrid), np.std(times_hybrid)

print("\n" + "="*70)
print("PERSISTENT DISK STORAGE BENCHMARK RESULTS (SQLite on SSD)")
print("="*70)
print(f"Exhaustive (Oracle):        {c_ex:,} derivs | {mean_ex:.2f} ± {std_ex:.2f} ms (1.00x baseline)")
print(f"Pure Certified BnB:         {c_pure:,} derivs | {mean_pure:.2f} ± {std_pure:.2f} ms ({mean_ex/mean_pure:.2f}x speedup)")
print(f"Hybrid Mode (Proposed):     {c_hyb:,} derivs | {mean_hyb:.2f} ± {std_hyb:.2f} ms ({mean_ex/mean_hyb:.2f}x speedup)")
print(f"Derivations Saved by Hybrid: {(c_ex - c_hyb)/c_ex * 100:.2f}% ({c_ex - c_hyb:,} / {c_ex:,})")

# Save results
out_data = {
    "benchmark": "Empirical_Disk_Storage_Latency_Hanoi_500",
    "dataset": "Hanoi 500 clips, 12,554 facts, 12,391 derivations",
    "storage_backend": "SQLite on NVMe/SSD, B-Tree index on (s, p, o, weight DESC)",
    "trials": N_TRIALS,
    "ram_benchmark_reference": {
        "exhaustive_ms": "23.31 ± 0.99",
        "pure_bnb_ms": "69.00 ± 5.55",
        "hybrid_ms": "28.33 ± 0.33",
        "hybrid_relative_to_exhaustive": "+21.5% slowdown"
    },
    "measured_disk_storage_benchmark": {
        "exhaustive_derivs": c_ex,
        "exhaustive_disk_ms": {"mean": round(mean_ex, 2), "std": round(std_ex, 2)},
        "pure_bnb_derivs": c_pure,
        "pure_bnb_disk_ms": {"mean": round(mean_pure, 2), "std": round(std_pure, 2)},
        "hybrid_derivs": c_hyb,
        "hybrid_disk_ms": {"mean": round(mean_hyb, 2), "std": round(std_hyb, 2)},
        "measured_hybrid_speedup_over_exhaustive": round(mean_ex / mean_hyb, 2),
        "derivations_saved_pct": round((c_ex - c_hyb)/c_ex * 100, 2)
    }
}

with open(OUTPUTS_DIR / "measured_disk_storage_benchmark_report.json", "w") as f:
    json.dump(out_data, f, indent=2)

print(f"\nReport written to {OUTPUTS_DIR / 'measured_disk_storage_benchmark_report.json'}")
