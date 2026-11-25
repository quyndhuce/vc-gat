import os
import sys
import pickle
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'fkg'))

from relations import RELATION_NAMES

with open("/home/nguyendinhquy/OvSGTR/fkg/output/grounded_segments_cache.pkl", "rb") as f:
    cache = pickle.load(f)

test_cache = cache["test_videos"]
sup_cache = cache["support_segments"]
concepts = cache["engine_concepts"]

# Let's inspect a segment from One Pillar Pagoda, Hoan Kiem Lake, and Bat Trang
for lm in ["One Pillar Pagoda", "Bat Trang Pottery Village", "The Old Quarter"]:
    seg = test_cache[lm][0]["segments"][0]
    triplets = seg["sparse_triplets"]
    print(f"\n=== Landmark: {lm} (Segment 0 has {len(triplets)} relations) ===")
    count = 0
    for k, sc in triplets.items():
        cA, r_idx, cB = [int(p) for p in k.split('_')]
        # Only print if score is meaningful
        if sc >= 0.15:
            count += 1
            print(f"  {concepts[cA]['label_en']} --[{RELATION_NAMES[r_idx]}]--> {concepts[cB]['label_en']} (membership={sc:.3f})")
            if count >= 5:
                break

# Now let's check the balance of term_nodes vs beta * term_rels during scoring!
from test_scoring import evaluate_configuration

print("\n" + "=" * 70)
print("COMPARING EVIDENCE CONTRIBUTIONS: NODES vs RELATIONS")
print("=" * 70)

# Evaluate with beta = 0.0 (Only Nodes, No Relations)
acc_no_rels, per_lm_0, _, _ = evaluate_configuration({}, beta=0.0, norm_mode="sum_rels")
print(f"Beta = 0.0 (CHỈ DÙNG ĐỈNH KHÁI NIỆM, KHÔNG DÙNG QUAN HỆ): Accuracy = {acc_no_rels*100:.1f}%")

# Evaluate with beta = 1.0 (Nodes + Relations)
acc_rels_1, per_lm_1, _, _ = evaluate_configuration({}, beta=1.0, norm_mode="sum_rels")
print(f"Beta = 1.0 (KẾT HỢP ĐỈNH + 7 QUAN HỆ HÌNH HỌC):          Accuracy = {acc_rels_1*100:.1f}%")

# Evaluate with beta = 2.0 (Emphasize Relations)
acc_rels_2, per_lm_2, _, _ = evaluate_configuration({}, beta=2.0, norm_mode="sum_rels")
print(f"Beta = 2.0 (TĂNG TRỌNG SỐ QUAN HỆ HÌNH HỌC):             Accuracy = {acc_rels_2*100:.1f}%")
