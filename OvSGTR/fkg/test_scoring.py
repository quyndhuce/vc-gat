import pickle
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

with open("/home/nguyendinhquy/OvSGTR/fkg/output/grounded_segments_cache.pkl", "rb") as f:
    data = pickle.load(f)

support_segs = data["support_segments"]
test_vids = data["test_videos"]
concepts = data["engine_concepts"]
landmarks = list(support_segs.keys())
lm_to_idx = {lm: i for i, lm in enumerate(landmarks)}
num_landmarks = len(landmarks)
num_concepts = len(concepts)

def evaluate_configuration(cluster_dict, beta=1.0, norm_mode="support_max"):
    """
    norm_mode:
      - 'sum_rels': original sum of rels (baseline formula)
      - 'support_max': max value achieved by support segments of that landmark/view
      - 'avg_rels': term_rels normalized by average support relations
    """
    landmark_views = {}
    for lm in landmarks:
        segs = support_segs[lm]
        k = cluster_dict.get(lm, 1)
        if k > 1:
            X = np.stack([s["concept_membership"] for s in segs])
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            clusters = [[] for _ in range(k)]
            for s, lbl in zip(segs, km.labels_):
                clusters[lbl].append(s)
            clusters = [c for c in clusters if len(c) > 0]
        else:
            clusters = [segs]

        views = []
        for v_idx, c_segs in enumerate(clusters):
            Nv = len(c_segs)
            Sc = np.sum([s["concept_membership"] for s in c_segs], axis=0)
            node_m = Sc / max(1, Nv)

            PR_k = defaultdict(float)
            P_k = defaultdict(float)
            for s in c_segs:
                c_mem = s["concept_membership"]
                for trip_key, trip_score in s["sparse_triplets"].items():
                    PR_k[trip_key] += trip_score
                    parts = trip_key.split('_')
                    cA, cB = int(parts[0]), int(parts[2])
                    P_k[trip_key] += float(c_mem[cA] * c_mem[cB])

            rel_m = {}
            for trip_key, pr_val in PR_k.items():
                p_val = P_k.get(trip_key, 0.0)
                rm = min(1.0, pr_val / (p_val + 1e-4)) if p_val > 1e-6 else min(1.0, pr_val / Nv)
                if rm >= 0.05:
                    rel_m[trip_key] = round(float(rm), 4)

            views.append({
                "view_idx": v_idx,
                "num_segments": Nv,
                "node_mem": node_m,
                "rel_mem": rel_m,
                "segments": c_segs
            })
        landmark_views[lm] = views

    # Landmark-level peak for discriminativeness
    peak_nodes = np.zeros((num_landmarks, num_concepts), dtype=np.float32)
    for l_idx, lm in enumerate(landmarks):
        all_node_mems = [v["node_mem"] for v in landmark_views[lm]]
        peak_nodes[l_idx] = np.max(all_node_mems, axis=0)

    sum_across = np.sum(peak_nodes, axis=0, keepdims=True)

    # Compute disc & norm_const
    for l_idx, lm in enumerate(landmarks):
        for v in landmark_views[lm]:
            node_m = v["node_mem"]
            disc_v = node_m / (sum_across.squeeze(0) + 1e-6)
            v["disc"] = disc_v

            if norm_mode == "sum_rels":
                term_nodes = float(np.sum(disc_v * node_m))
                term_rels = float(sum(v["rel_mem"].values()))
                v["norm_const"] = max(1e-4, term_nodes + beta * term_rels)
            elif norm_mode == "support_max":
                # Compute raw score on support segments of this view
                raw_scores = []
                for s in v["segments"]:
                    tn = float(np.sum(disc_v * s["concept_membership"] * node_m))
                    tr = sum(v["rel_mem"].get(k, 0.0) * sc for k, sc in s["sparse_triplets"].items())
                    raw_scores.append(tn + beta * tr)
                v["norm_const"] = max(1e-4, float(np.percentile(raw_scores, 95)))
            elif norm_mode == "view_scale":
                # Scale relative to view's segment count Nv vs total N_l
                term_nodes = float(np.sum(disc_v * node_m))
                term_rels = float(sum(v["rel_mem"].values())) * (len(v["segments"]) / len(support_segs[lm]))
                v["norm_const"] = max(1e-4, term_nodes + beta * term_rels)

    # Evaluate test videos
    correct = 0
    total = 0
    per_lm = {lm: [0, 0] for lm in landmarks}
    predictions = []

    for gt_lm, vids in test_vids.items():
        gt_idx = lm_to_idx[gt_lm]
        for v in vids:
            total += 1
            per_lm[gt_lm][1] += 1
            segs = v["segments"]
            seg_scores = []

            for s in segs:
                s_scores = []
                for l_idx, lm in enumerate(landmarks):
                    views = landmark_views[lm]
                    v_scores = []
                    for vi in views:
                        tn = float(np.sum(vi["disc"] * s["concept_membership"] * vi["node_mem"]))
                        tr = sum(vi["rel_mem"].get(k, 0.0) * sc for k, sc in s["sparse_triplets"].items())
                        v_score = (tn + beta * tr) / vi["norm_const"]
                        v_scores.append(v_score)
                    s_scores.append(max(v_scores))
                seg_scores.append(s_scores)

            v_scores = np.mean(seg_scores, axis=0)
            pred_idx = int(np.argmax(v_scores))
            is_corr = (pred_idx == gt_idx)
            if is_corr:
                correct += 1
                per_lm[gt_lm][0] += 1
            predictions.append((v["video_id"], gt_lm, landmarks[pred_idx], is_corr, v_scores[pred_idx], v_scores[gt_idx]))

    acc = correct / total
    return acc, per_lm, predictions, landmark_views


# Test different setups
print("1. Baseline (K=1 all, sum_rels):")
acc, per_lm, preds, _ = evaluate_configuration({}, beta=1.0, norm_mode="sum_rels")
print(f"Overall Accuracy: {acc*100:.1f}%")
for lm, (c, t) in per_lm.items():
    print(f"  {lm:35s}: {c}/{t} ({c/t*100:.0f}%)")

print("\n" + "="*60)
for K in [3, 4]:
    print(f"\n2. K={K} views for 3 weak landmarks (norm_mode='support_max'):")
    c_cfg = {"Van Mieu Quoc Tu Giam": K, "HoangThanhThangLong": K, "Van Phuc Silk Village": K}
    acc, per_lm, preds, _ = evaluate_configuration(c_cfg, beta=1.0, norm_mode="support_max")
    print(f"Overall Accuracy: {acc*100:.1f}%")
    for lm, (c, t) in per_lm.items():
        print(f"  {lm:35s}: {c}/{t} ({c/t*100:.0f}%)")

print("\n" + "="*60)
for K in [3, 4]:
    print(f"\n3. K={K} views for ALL 10 landmarks (norm_mode='sum_rels'):")
    c_cfg = {lm: K for lm in landmarks}
    acc, per_lm, preds, _ = evaluate_configuration(c_cfg, beta=1.0, norm_mode="sum_rels")
    print(f"Overall Accuracy: {acc*100:.1f}%")
    for lm, (c, t) in per_lm.items():
        print(f"  {lm:35s}: {c}/{t} ({c/t*100:.0f}%)")
