import numpy as np
from collections import defaultdict


def aggregate_support_statistics(landmark_support_segments, num_concepts=207):
    """
    Bước 9. Cộng dồn thống kê:
    Với mỗi landmark, gộp mọi đoạn của 3 video support:
      - Số đoạn N_l
      - Với mỗi khái niệm: tổng độ thuộc S(c)
      - Với mỗi bộ ba: tổng mức cùng xuất hiện P(k) và tổng mức cùng xuất hiện kèm quan hệ PR(k)
    Rồi tính:
      - Độ thuộc đỉnh: S(c) chia N_l
      - Độ thuộc quan hệ: PR(k) chia P(k)
      - Độ phân biệt: độ thuộc đỉnh chia cho tổng độ thuộc đỉnh của khái niệm đó trên cả 10 landmark
    Kết quả: ma trận 10 × 207 cho khái niệm, ma trận 10 × 207 cho độ phân biệt, và một ma trận thưa cho quan hệ.
    """
    landmarks = list(landmark_support_segments.keys())
    num_landmarks = len(landmarks)

    node_mem_matrix = np.zeros((num_landmarks, num_concepts), dtype=np.float32)
    rel_mem_dicts = [{} for _ in range(num_landmarks)]
    norm_const_vector = np.zeros(num_landmarks, dtype=np.float32)

    support_N = {}

    for l_idx, lm in enumerate(landmarks):
        segments = landmark_support_segments[lm]
        N_l = max(1, len(segments))
        support_N[lm] = N_l

        S_c = np.zeros(num_concepts, dtype=np.float32)
        P_k = defaultdict(float)
        PR_k = defaultdict(float)

        for seg in segments:
            c_mem = seg["concept_membership"]  # (207,)
            S_c += c_mem

            triplets = seg["sparse_triplets"]  # dict of "cA_r_cB" -> score
            # Co-occurrence with relation PR(k) and pair co-occurrence P(k)
            for trip_key, trip_score in triplets.items():
                PR_k[trip_key] += trip_score
                parts = trip_key.split('_')
                cA, cB = int(parts[0]), int(parts[2])
                P_k[trip_key] += float(c_mem[cA] * c_mem[cB])

        # Độ thuộc đỉnh: S(c) / N
        node_mem_matrix[l_idx] = S_c / N_l

        # Độ thuộc quan hệ: PR(k) / P(k)
        for trip_key, pr_val in PR_k.items():
            p_val = P_k.get(trip_key, 0.0)
            if p_val > 1e-6:
                rel_mem = min(1.0, pr_val / (p_val + 1e-4))
            else:
                rel_mem = min(1.0, pr_val / N_l)
            if rel_mem >= 0.05:
                rel_mem_dicts[l_idx][trip_key] = round(float(rel_mem), 4)

    # Độ phân biệt: node_mem(l, c) chia cho tổng độ thuộc đỉnh của khái niệm đó trên cả 10 landmark
    sum_across_landmarks = np.sum(node_mem_matrix, axis=0, keepdims=True)  # (1, 207)
    disc_matrix = node_mem_matrix / (sum_across_landmarks + 1e-6)

    # Tính Hằng số chuẩn hóa NormConst(l) cho mỗi landmark (để điểm segment evidence nằm trong [0, 1])
    # NormConst(l) = sum_c (disc(l,c) * node_mem(l,c)) + beta * sum_k rel_mem(l,k)
    for l_idx in range(num_landmarks):
        term_nodes = float(np.sum(disc_matrix[l_idx] * node_mem_matrix[l_idx]))
        term_rels = float(sum(rel_mem_dicts[l_idx].values()))
        norm_const_vector[l_idx] = max(1e-4, term_nodes + 1.0 * term_rels)

    profiles = {
        "landmarks": landmarks,
        "support_counts": support_N,
        "node_membership_matrix": node_mem_matrix,
        "discriminativeness_matrix": disc_matrix,
        "relation_membership_dicts": rel_mem_dicts,
        "norm_constants": norm_const_vector
    }

    return profiles
