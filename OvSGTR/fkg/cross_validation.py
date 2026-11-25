import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'fkg'))


def load_all_grounded_videos(
    cache_path="/home/nguyendinhquy/OvSGTR/fkg/output/grounded_segments_cache.pkl",
    stage1_path="/home/nguyendinhquy/OvSGTR/fkg/output/stage1_detections.json"
):
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    with open(stage1_path, "r", encoding="utf-8") as f:
        stage1 = json.load(f)

    s1_vids = stage1["videos"]
    sup_cache = cache["support_segments"]
    test_cache = cache["test_videos"]
    concepts = cache["engine_concepts"]

    landmarks = list(sup_cache.keys())
    all_videos_by_landmark = defaultdict(list)

    for lm in landmarks:
        lm_s1_sup = [v for v in s1_vids if v["landmark"] == lm and v["split"] == "support"]
        sup_counts = [len(v["segments"]) for v in lm_s1_sup]

        # Reconstruct support videos 0, 1, 2
        offset = 0
        for v_rec, count in zip(lm_s1_sup, sup_counts):
            v_segs = sup_cache[lm][offset : offset + count]
            offset += count
            all_videos_by_landmark[lm].append({
                "video_id": v_rec["video_id"],
                "landmark": lm,
                "segments": v_segs
            })

        # Reconstruct test videos 3, 4
        for v_rec in test_cache[lm]:
            all_videos_by_landmark[lm].append({
                "video_id": v_rec["video_id"],
                "landmark": lm,
                "segments": v_rec["segments"]
            })

    return all_videos_by_landmark, landmarks, concepts


def run_cross_validation_experiment(
    all_videos_by_landmark,
    landmarks,
    concepts,
    k_views=3,
    beta=1.0,
    scheme="3sup_2test"
):
    """
    scheme:
      - '3sup_2test': 5 folds, each fold has 3 support videos and 2 test videos per landmark.
      - '4sup_1test': 5 folds, each fold has 4 support videos and 1 test video per landmark.
    """
    num_landmarks = len(landmarks)
    lm_to_idx = {lm: i for i, lm in enumerate(landmarks)}
    num_concepts = len(concepts)

    if scheme == "3sup_2test":
        # 5 folds of (3 support, 2 test)
        folds_indices = [
            ([0, 1, 2], [3, 4]),  # Fold 0 (original split)
            ([1, 2, 3], [4, 0]),  # Fold 1
            ([2, 3, 4], [0, 1]),  # Fold 2
            ([3, 4, 0], [1, 2]),  # Fold 3
            ([4, 0, 1], [2, 3]),  # Fold 4
        ]
    else:
        # 5-fold Leave-One-Video-Out (4 support, 1 test)
        folds_indices = [
            ([1, 2, 3, 4], [0]),
            ([0, 2, 3, 4], [1]),
            ([0, 1, 3, 4], [2]),
            ([0, 1, 2, 4], [3]),
            ([0, 1, 2, 3], [4]),
        ]

    fold_accuracies = []
    overall_confusion_matrix = np.zeros((num_landmarks, num_landmarks), dtype=np.int32)
    fold_details = []

    per_lm_correct = {lm: 0 for lm in landmarks}
    per_lm_total = {lm: 0 for lm in landmarks}

    for fold_idx, (sup_idxs, test_idxs) in enumerate(folds_indices):
        # 1. Build support segments
        landmark_views = {}
        for lm in landmarks:
            segs = []
            for i in sup_idxs:
                segs.extend(all_videos_by_landmark[lm][i]["segments"])

            # Multi-view clustering
            if k_views > 1 and len(segs) > k_views:
                X = np.stack([s["concept_membership"] for s in segs])
                km = KMeans(n_clusters=k_views, random_state=42, n_init=10).fit(X)
                clusters = [[] for _ in range(k_views)]
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
                    "rel_mem": rel_m
                })
            landmark_views[lm] = views

        # Landmark-level peak for discriminativeness
        peak_nodes = np.zeros((num_landmarks, num_concepts), dtype=np.float32)
        for l_idx, lm in enumerate(landmarks):
            all_node_mems = [v["node_mem"] for v in landmark_views[lm]]
            peak_nodes[l_idx] = np.max(all_node_mems, axis=0)

        sum_across = np.sum(peak_nodes, axis=0, keepdims=True)

        for l_idx, lm in enumerate(landmarks):
            for v in landmark_views[lm]:
                node_m = v["node_mem"]
                disc_v = node_m / (sum_across.squeeze(0) + 1e-6)
                v["disc"] = disc_v

                term_nodes = float(np.sum(disc_v * node_m))
                term_rels = float(sum(v["rel_mem"].values()))
                v["norm_const"] = max(1e-4, term_nodes + beta * term_rels)

        # 2. Evaluate on test set of this fold
        fold_correct = 0
        fold_total = 0
        fold_preds = []

        for gt_lm in landmarks:
            gt_idx = lm_to_idx[gt_lm]
            for t_idx in test_idxs:
                v_record = all_videos_by_landmark[gt_lm][t_idx]
                segs = v_record["segments"]
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
                sorted_scores = np.sort(v_scores)[::-1]
                margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else float(sorted_scores[0])

                is_corr = (pred_idx == gt_idx)
                if is_corr:
                    fold_correct += 1
                    per_lm_correct[gt_lm] += 1
                per_lm_total[gt_lm] += 1
                fold_total += 1

                overall_confusion_matrix[gt_idx, pred_idx] += 1

                fold_preds.append({
                    "video_id": v_record["video_id"],
                    "ground_truth": gt_lm,
                    "predicted": landmarks[pred_idx],
                    "is_correct": is_corr,
                    "margin": round(margin, 4)
                })

        fold_acc = fold_correct / fold_total
        fold_accuracies.append(fold_acc)
        fold_details.append({
            "fold": fold_idx,
            "support_indices": sup_idxs,
            "test_indices": test_idxs,
            "accuracy": round(fold_acc, 4),
            "correct": fold_correct,
            "total": fold_total,
            "predictions": fold_preds
        })

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))

    per_lm_acc = {lm: round(per_lm_correct[lm] / max(1, per_lm_total[lm]), 4) for lm in landmarks}

    results = {
        "scheme": scheme,
        "k_views": k_views,
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "fold_accuracies": [round(a, 4) for a in fold_accuracies],
        "total_evaluations": sum(f["total"] for f in fold_details),
        "total_correct": sum(f["correct"] for f in fold_details),
        "per_landmark_accuracy": per_lm_acc,
        "confusion_matrix": overall_confusion_matrix.tolist(),
        "folds": fold_details
    }

    return results


def main():
    print("=" * 70)
    print("RUNNING CROSS-VALIDATION EXPERIMENTS ON 50 TOURISM VIDEOS")
    print("=" * 70)

    all_vids, landmarks, concepts = load_all_grounded_videos()
    print(f"[INFO] Loaded {len(landmarks)} landmarks, 5 videos each (50 videos total).")

    output_dir = "/home/nguyendinhquy/OvSGTR/fkg/output"

    # 1. Scheme A: 5-Fold Cross-Validation (3 Support, 2 Test - Matches exact original setup)
    print("\n" + "#" * 70)
    print("SCHEME A: 5-Fold Cross-Validation (3 Support / 2 Test per landmark, 20 test vids/fold)")
    print("#" * 70)

    for k_views in [1, 3]:
        label = "Baseline (K=1)" if k_views == 1 else f"Multi-View (K={k_views})"
        print(f"\n--- Running Scheme A with {label} ---")
        res = run_cross_validation_experiment(all_vids, landmarks, concepts, k_views=k_views, beta=1.0, scheme="3sup_2test")
        print(f"Mean Accuracy: {res['mean_accuracy']*100:.2f}% ± {res['std_accuracy']*100:.2f}%")
        print(f"Fold Accuracies: {[f'{a*100:.1f}%' for a in res['fold_accuracies']]}")
        print("Per-Landmark Accuracy:")
        for lm, acc in res["per_landmark_accuracy"].items():
            print(f"  {lm:35s}: {acc*100:5.1f}%")

        out_path = os.path.join(output_dir, f"cv_results_3sup2test_k{k_views}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[INFO] Saved to {out_path}")

    # 2. Scheme B: 5-Fold Leave-One-Video-Out (4 Support, 1 Test - Every video tested once)
    print("\n" + "#" * 70)
    print("SCHEME B: 5-Fold Cross-Validation (4 Support / 1 Test per landmark, 50 videos evaluated)")
    print("#" * 70)

    for k_views in [1, 3]:
        label = "Baseline (K=1)" if k_views == 1 else f"Multi-View (K={k_views})"
        print(f"\n--- Running Scheme B with {label} ---")
        res = run_cross_validation_experiment(all_vids, landmarks, concepts, k_views=k_views, beta=1.0, scheme="4sup_1test")
        print(f"Mean Accuracy: {res['mean_accuracy']*100:.2f}% ± {res['std_accuracy']*100:.2f}%")
        print(f"Fold Accuracies: {[f'{a*100:.1f}%' for a in res['fold_accuracies']]}")
        print("Per-Landmark Accuracy:")
        for lm, acc in res["per_landmark_accuracy"].items():
            print(f"  {lm:35s}: {acc*100:5.1f}%")

        out_path = os.path.join(output_dir, f"cv_results_4sup1test_k{k_views}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()
