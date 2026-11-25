import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict


def cluster_landmark_segments(segments, n_clusters=3, random_state=42):
    """
    Phân cụm các đoạn của một địa danh thành n_clusters views dựa trên vector khái niệm.
    """
    if len(segments) <= n_clusters:
        return [[s] for s in segments]

    # Feature matrix: (N_segments, 207)
    X = np.stack([s["concept_membership"] for s in segments])

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    clusters = [[] for _ in range(n_clusters)]
    for seg, lbl in zip(segments, labels):
        clusters[lbl].append(seg)

    # Filter out empty clusters if any
    clusters = [c for c in clusters if len(c) > 0]
    return clusters


def build_multiview_support_profiles(
    support_segments_by_landmark,
    clustered_landmarks=None,
    n_clusters=3,
    num_concepts=207,
    beta=1.0
):
    """
    Xây dựng bảng đặc trưng tri thức mờ hỗ trợ Multi-View.
    - Với các landmark trong clustered_landmarks: phân cụm thành n_clusters view profiles.
    - Với các landmark khác: giữ 1 view profile chuẩn.
    """
    if clustered_landmarks is None:
        clustered_landmarks = {
            "Van Mieu Quoc Tu Giam": n_clusters,
            "HoangThanhThangLong": n_clusters,
            "Van Phuc Silk Village": n_clusters
        }
    elif isinstance(clustered_landmarks, list):
        clustered_landmarks = {lm: n_clusters for lm in clustered_landmarks}

    landmarks = list(support_segments_by_landmark.keys())
    num_landmarks = len(landmarks)

    # Structure: landmark -> list of views
    # Each view: { "view_idx": int, "N": int, "node_mem": (207,), "rel_mem": dict, "norm_const": float }
    landmark_views = {}

    for lm in landmarks:
        segments = support_segments_by_landmark[lm]
        k = clustered_landmarks.get(lm, 1)

        if k > 1:
            clusters = cluster_landmark_segments(segments, n_clusters=k)
        else:
            clusters = [segments]

        views = []
        for v_idx, cluster_segs in enumerate(clusters):
            N_v = max(1, len(cluster_segs))
            S_c = np.zeros(num_concepts, dtype=np.float32)
            P_k = defaultdict(float)
            PR_k = defaultdict(float)

            for seg in cluster_segs:
                c_mem = seg["concept_membership"]
                S_c += c_mem

                triplets = seg["sparse_triplets"]
                for trip_key, trip_score in triplets.items():
                    PR_k[trip_key] += trip_score
                    parts = trip_key.split('_')
                    cA, cB = int(parts[0]), int(parts[2])
                    P_k[trip_key] += float(c_mem[cA] * c_mem[cB])

            node_mem = S_c / N_v

            rel_mem = {}
            for trip_key, pr_val in PR_k.items():
                p_val = P_k.get(trip_key, 0.0)
                if p_val > 1e-6:
                    rm = min(1.0, pr_val / (p_val + 1e-4))
                else:
                    rm = min(1.0, pr_val / N_v)
                if rm >= 0.05:
                    rel_mem[trip_key] = round(float(rm), 4)

            views.append({
                "view_idx": v_idx,
                "num_segments": N_v,
                "node_mem": node_mem,
                "rel_mem": rel_mem,
                "segments": cluster_segs
            })

        landmark_views[lm] = views

    # Calculate Landmark-Level Peak Node Memberships for Discriminativeness
    landmark_peak_nodes = np.zeros((num_landmarks, num_concepts), dtype=np.float32)
    for l_idx, lm in enumerate(landmarks):
        all_node_mems = [v["node_mem"] for v in landmark_views[lm]]
        landmark_peak_nodes[l_idx] = np.max(all_node_mems, axis=0)

    sum_across_landmarks = np.sum(landmark_peak_nodes, axis=0, keepdims=True)  # (1, 207)

    # Compute Discriminativeness & Norm Constants for each view of each landmark
    for l_idx, lm in enumerate(landmarks):
        for v in landmark_views[lm]:
            node_m = v["node_mem"]
            # Discriminativeness: node_mem of this view over all landmarks
            disc_v = node_m / (sum_across_landmarks.squeeze(0) + 1e-6)
            v["disc"] = disc_v

            term_nodes = float(np.sum(disc_v * node_m))
            term_rels = float(sum(v["rel_mem"].values()))
            v["norm_const"] = max(1e-4, term_nodes + beta * term_rels)

    profiles = {
        "landmarks": landmarks,
        "landmark_views": landmark_views,
        "clustered_landmarks": clustered_landmarks,
        "beta": beta
    }

    return profiles


def compute_multiview_segment_evidence(segment, multiview_profiles, beta=1.0):
    """
    Tính điểm bằng chứng của 1 đoạn test với từng landmark qua các views:
    score(s, l) = max_{v} score(s, l, v)
    """
    landmarks = multiview_profiles["landmarks"]
    landmark_views = multiview_profiles["landmark_views"]
    num_landmarks = len(landmarks)

    mu_s_c = segment["concept_membership"]
    mu_s_k = segment["sparse_triplets"]

    scores = np.zeros(num_landmarks, dtype=np.float32)
    best_concept_contribs = np.zeros((num_landmarks, len(mu_s_c)), dtype=np.float32)

    for l_idx, lm in enumerate(landmarks):
        views = landmark_views[lm]
        v_scores = []
        v_contribs = []

        for v in views:
            disc_v = v["disc"]
            node_m = v["node_mem"]
            c_contrib = disc_v * mu_s_c * node_m
            term_nodes = float(np.sum(c_contrib))

            term_rels = 0.0
            v_rels = v["rel_mem"]
            for trip_key, trip_score in mu_s_k.items():
                if trip_key in v_rels:
                    term_rels += v_rels[trip_key] * trip_score

            total_raw = term_nodes + beta * term_rels
            norm_c = v["norm_const"]
            v_score = min(1.0, max(0.0, total_raw / norm_c))

            v_scores.append(v_score)
            v_contribs.append(c_contrib)

        # Max-pooling across views (Gödel s-norm: segment matches best matching view)
        best_v_idx = int(np.argmax(v_scores))
        scores[l_idx] = v_scores[best_v_idx]
        best_concept_contribs[l_idx] = v_contribs[best_v_idx]

    return scores, best_concept_contribs


def evaluate_multiview_test_videos(test_video_segments, multiview_profiles, concepts_meta, beta=1.0):
    """
    Đánh giá 20 video test trên hệ thống hỗ trợ Multi-View Profiles.
    """
    landmarks = multiview_profiles["landmarks"]
    lm_to_idx = {lm: i for i, lm in enumerate(landmarks)}
    num_landmarks = len(landmarks)

    confusion_matrix = np.zeros((num_landmarks, num_landmarks), dtype=np.int32)
    video_predictions = []

    correct_margins = []
    incorrect_margins = []
    landmark_contributions = [np.zeros(len(concepts_meta), dtype=np.float32) for _ in range(num_landmarks)]

    prefix_cutoffs = [1, 3, 5, 999]
    prefix_correct = {k: 0 for k in prefix_cutoffs}
    total_eval_videos = 0

    for gt_lm, videos in test_video_segments.items():
        gt_idx = lm_to_idx[gt_lm]

        for v_record in videos:
            vid = v_record["video_id"]
            segments = v_record["segments"]
            if not segments:
                continue

            total_eval_videos += 1

            seg_scores_list = []
            seg_contribs_list = []

            for seg in segments:
                s_scores, s_contrib = compute_multiview_segment_evidence(seg, multiview_profiles, beta=beta)
                seg_scores_list.append(s_scores)
                seg_contribs_list.append(s_contrib)

            seg_scores_arr = np.array(seg_scores_list)

            # Step 11: Video score is mean of segment scores
            video_scores = np.mean(seg_scores_arr, axis=0)
            pred_idx = int(np.argmax(video_scores))
            sorted_scores = np.sort(video_scores)[::-1]
            margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else float(sorted_scores[0])

            confusion_matrix[gt_idx, pred_idx] += 1
            is_correct = (pred_idx == gt_idx)

            if is_correct:
                correct_margins.append(margin)
            else:
                incorrect_margins.append(margin)

            avg_contrib = np.mean(seg_contribs_list, axis=0)
            landmark_contributions[gt_idx] += avg_contrib[gt_idx]

            # Step 13: Early recognition curve
            for cut in prefix_cutoffs:
                p_scores = np.mean(seg_scores_arr[:min(len(seg_scores_arr), cut)], axis=0)
                if int(np.argmax(p_scores)) == gt_idx:
                    prefix_correct[cut] += 1

            video_predictions.append({
                "video_id": vid,
                "ground_truth": gt_lm,
                "predicted": landmarks[pred_idx],
                "is_correct": is_correct,
                "top1_score": round(float(sorted_scores[0]), 4),
                "top2_score": round(float(sorted_scores[1]), 4) if len(sorted_scores) > 1 else 0.0,
                "margin": round(margin, 4),
                "num_segments": len(segments),
                "video_scores": {lm: round(float(video_scores[i]), 4) for i, lm in enumerate(landmarks)}
            })

    total_correct = int(np.trace(confusion_matrix))
    overall_acc = total_correct / max(1, total_eval_videos)

    per_landmark_acc = {}
    for i, lm in enumerate(landmarks):
        row_sum = np.sum(confusion_matrix[i])
        per_landmark_acc[lm] = float(confusion_matrix[i, i] / max(1, row_sum))

    mean_margin_correct = float(np.mean(correct_margins)) if correct_margins else 0.0
    mean_margin_incorrect = float(np.mean(incorrect_margins)) if incorrect_margins else 0.0

    top5_per_landmark = {}
    for i, lm in enumerate(landmarks):
        c_scores = landmark_contributions[i]
        top_indices = np.argsort(-c_scores)[:5]
        top5_per_landmark[lm] = [
            {
                "id": concepts_meta[c]["id"],
                "label_en": concepts_meta[c]["label_en"],
                "score": round(float(c_scores[c]), 4)
            }
            for c in top_indices
        ]

    early_curves = {}
    for cut in [1, 3, 5, 999]:
        label = f"first_{cut}_segs" if cut < 999 else "all_segs"
        early_curves[label] = round(float(prefix_correct[cut] / max(1, total_eval_videos)), 4)

    results = {
        "overall_accuracy": round(overall_acc, 4),
        "total_test_videos": total_eval_videos,
        "total_correct": total_correct,
        "per_landmark_accuracy": per_landmark_acc,
        "confusion_matrix": confusion_matrix.tolist(),
        "mean_margin_correct": round(mean_margin_correct, 4),
        "mean_margin_incorrect": round(mean_margin_incorrect, 4),
        "top5_contributing_concepts": top5_per_landmark,
        "early_recognition_curve": early_curves,
        "predictions": video_predictions
    }

    return results
