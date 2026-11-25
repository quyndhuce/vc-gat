import pickle
import numpy as np
from sklearn.cluster import KMeans

with open("/home/nguyendinhquy/OvSGTR/fkg/output/grounded_segments_cache.pkl", "rb") as f:
    data = pickle.load(f)

support_segs = data["support_segments"]
concepts = data["engine_concepts"]
target_lms = ["Van Mieu Quoc Tu Giam", "HoangThanhThangLong", "Van Phuc Silk Village"]

for lm in target_lms:
    segs = support_segs[lm]
    X = np.stack([s["concept_membership"] for s in segs])
    for K in [3, 4]:
        km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
        print(f"\n=== {lm} (K={K}) ===")
        for c_idx in range(K):
            c_segs = [segs[i] for i, lbl in enumerate(km.labels_) if lbl == c_idx]
            mean_c = np.mean([s["concept_membership"] for s in c_segs], axis=0)
            top5 = np.argsort(-mean_c)[:5]
            top_str = ", ".join([f"{concepts[idx]['label_en']} ({mean_c[idx]:.2f})" for idx in top5])
            print(f"  View {c_idx} ({len(c_segs)} segs): {top_str}")
