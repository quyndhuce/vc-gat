import numpy as np


def compute_segment_evidence_scores(segment, profiles, beta=1.0):
    """
    Bước 10. Tính điểm bằng chứng cho từng đoạn kiểm tra:
    Với đoạn s và landmark l:
    điểm = [ tổng theo c của (độ phân biệt(l,c) × độ thuộc đoạn(c) × độ thuộc đỉnh(l,c))
             + beta × tổng theo k của (độ thuộc quan hệ(l,k) × độ thuộc đoạn(k)) ]
           chia cho hằng số chuẩn hóa của l
    """
    landmarks = profiles["landmarks"]
    num_landmarks = len(landmarks)
    disc_matrix = profiles["discriminativeness_matrix"]      # (10, 207)
    node_mem_matrix = profiles["node_membership_matrix"]      # (10, 207)
    rel_mem_dicts = profiles["relation_membership_dicts"]      # list of 10 dicts
    norm_consts = profiles["norm_constants"]                   # (10,)

    mu_s_c = segment["concept_membership"]                     # (207,)
    mu_s_k = segment["sparse_triplets"]                        # dict of (cA, r, cB) -> score

    scores = np.zeros(num_landmarks, dtype=np.float32)
    concept_contributions = np.zeros((num_landmarks, len(mu_s_c)), dtype=np.float32)

    for l_idx in range(num_landmarks):
        # 1. Điểm thành phần đỉnh (Nodes): disc(l,c) * mu_s(c) * node_mem(l,c)
        c_contrib = disc_matrix[l_idx] * mu_s_c * node_mem_matrix[l_idx]
        term_nodes = float(np.sum(c_contrib))
        concept_contributions[l_idx] = c_contrib

        # 2. Điểm thành phần quan hệ (Relations): rel_mem(l,k) * mu_s(k)
        term_rels = 0.0
        l_rels = rel_mem_dicts[l_idx]
        for trip_key, trip_score in mu_s_k.items():
            if trip_key in l_rels:
                term_rels += l_rels[trip_key] * trip_score

        # Chuẩn hóa
        total_raw = term_nodes + beta * term_rels
        norm_c = norm_consts[l_idx]
        scores[l_idx] = min(1.0, max(0.0, total_raw / norm_c))

    return scores, concept_contributions


def evaluate_test_videos(test_video_segments, profiles, concepts_meta, beta=1.0):
    """
    Bước 11, 12, 13:
    - Cộng dồn điểm các đoạn của mỗi video (Bước 11)
    - Ma trận nhầm lẫn 10x10, độ chính xác, margin, top-5 khái niệm (Bước 12)
    - Đường cong theo số đoạn (Bước 13)
    """
    landmarks = profiles["landmarks"]
    lm_to_idx = {lm: i for i, lm in enumerate(landmarks)}
    num_landmarks = len(landmarks)

    confusion_matrix = np.zeros((num_landmarks, num_landmarks), dtype=np.int32)
    video_predictions = []

    correct_margins = []
    incorrect_margins = []
    landmark_contributions = [np.zeros(len(concepts_meta), dtype=np.float32) for _ in range(num_landmarks)]

    # Bước 13: Evaluation with different prefix segments [1, 3, 5, all]
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

            # Compute segment scores
            seg_scores_list = []
            seg_contribs_list = []

            for seg in segments:
                s_scores, s_contrib = compute_segment_evidence_scores(seg, profiles, beta=beta)
                seg_scores_list.append(s_scores)
                seg_contribs_list.append(s_contrib)

            seg_scores_arr = np.array(seg_scores_list)  # (N_segments, 10)

            # Bước 11: Điểm của video bằng trung bình điểm các đoạn
            video_scores = np.mean(seg_scores_arr, axis=0)  # (10,)
            pred_idx = int(np.argmax(video_scores))
            sorted_scores = np.sort(video_scores)[::-1]
            margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else float(sorted_scores[0])

            # Record confusion matrix
            confusion_matrix[gt_idx, pred_idx] += 1
            is_correct = (pred_idx == gt_idx)

            if is_correct:
                correct_margins.append(margin)
            else:
                incorrect_margins.append(margin)

            # Accumulate concept contributions for gt_landmark
            avg_contrib = np.mean(seg_contribs_list, axis=0)  # (10, 207)
            landmark_contributions[gt_idx] += avg_contrib[gt_idx]

            # Bước 13: Test accuracy with prefixes of segments
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
                "num_segments": len(segments)
            })

    # Overall & Per-landmark accuracy
    total_correct = int(np.trace(confusion_matrix))
    overall_acc = total_correct / max(1, total_eval_videos)

    per_landmark_acc = {}
    for i, lm in enumerate(landmarks):
        row_sum = np.sum(confusion_matrix[i])
        per_landmark_acc[lm] = float(confusion_matrix[i, i] / max(1, row_sum))

    mean_margin_correct = float(np.mean(correct_margins)) if correct_margins else 0.0
    mean_margin_incorrect = float(np.mean(incorrect_margins)) if incorrect_margins else 0.0

    # Top-5 contributing concepts per landmark
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

    # Early recognition curve
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
