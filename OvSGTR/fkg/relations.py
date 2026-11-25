import numpy as np

# ---------------------------------------------------------------------------
# CANONICAL VISUAL RELATIONS FROM OVSGTR (VISUAL GENOME 150 - VG150)
# Standardized visual predicate vocabulary:
#   - 'on'            (standard VG150 predicate, replaces heuristic 'on_top_of')
#   - 'in'            (standard VG150 predicate, replaces heuristic 'inside')
#   - 'near'          (standard VG150 predicate, replaces heuristic 'next_to')
#   - 'surrounded_by' (OvSGTR open-vocabulary spatial relation)
#   - 'across'        (standard VG150 predicate, replaces heuristic 'spanning_across')
#   - 'hanging_from'  (standard VG150 predicate 'hanging from')
#   - 'attached_to'   (standard VG150 predicate 'attached to', replaces heuristic 'emitting')
# ---------------------------------------------------------------------------
OVSGTR_RELATION_NAMES = [
    'on',
    'in',
    'near',
    'surrounded_by',
    'across',
    'hanging_from',
    'attached_to'
]

# Primary relation names list (Canonical OvSGTR)
RELATION_NAMES = OVSGTR_RELATION_NAMES

# Complete alias mapping for 100% backward and cross-format compatibility:
RELATION_ALIASES = {
    'on_top_of': 'on',
    'inside': 'in',
    'next_to': 'near',
    'surrounded_by': 'surrounded_by',
    'spanning_across': 'across',
    'hanging_from': 'hanging_from',
    'hanging from': 'hanging_from',
    'emitting': 'attached_to',
    'attached to': 'attached_to',
    'attached_to': 'attached_to',
    'on': 'on',
    'in': 'in',
    'near': 'near',
    'across': 'across',
}

def normalize_relation_name(rel_name: str) -> str:
    """Standardizes any predicate name to canonical OvSGTR relation name."""
    clean = rel_name.strip().lower()
    return RELATION_ALIASES.get(clean, clean)

def get_relation_index(rel_name: str) -> int:
    """Returns the canonical 0..6 integer index for the relation in RELATION_NAMES."""
    canonical = normalize_relation_name(rel_name)
    if canonical in RELATION_NAMES:
        return RELATION_NAMES.index(canonical)
    raise ValueError(f"Unknown visual relation '{rel_name}' (canonical: '{canonical}'). Supported: {RELATION_NAMES}")


def compute_box_geometry(boxA, boxB, orig_w, orig_h, frames_A=None, frames_B=None):
    """
    Tính 7 quan hệ thị giác chuẩn tắc OvSGTR giữa Track A và Track B từ bbox đại diện [x0, y0, x1, y1]:
    ['on', 'in', 'near', 'surrounded_by', 'across', 'hanging_from', 'attached_to']
    """
    xA0, yA0, xA1, yA1 = boxA
    xB0, yB0, xB1, yB1 = boxB

    wA = max(1.0, xA1 - xA0)
    hA = max(1.0, yA1 - yA0)
    areaA = wA * hA

    wB = max(1.0, xB1 - xB0)
    hB = max(1.0, yB1 - yB0)
    areaB = wB * hB

    # Intersection
    inter_x0 = max(xA0, xB0)
    inter_y0 = max(yA0, yB0)
    inter_x1 = min(xA1, xB1)
    inter_y1 = min(yA1, yB1)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    # 1. on (A ở trên B - OvSGTR canonical predicate)
    overlap_x = inter_w / min(wA, wB)
    dy = (yB0 - yA1) / max(hA, hB)
    if dy >= -0.2 and dy <= 0.3:
        mu_dy = max(0.0, 1.0 - abs(dy) / 0.3)
    else:
        mu_dy = 0.0
    r_on = float(overlap_x * mu_dy)

    # 2. in (A nằm trong B - OvSGTR canonical predicate)
    r_in = float(inter_area / areaA)

    # 3. near (A đứng gần/cạnh B - OvSGTR canonical predicate)
    cxA, cyA = (xA0 + xA1) / 2.0, (yA0 + yA1) / 2.0
    cxB, cyB = (xB0 + xB1) / 2.0, (yB0 + yB1) / 2.0
    dist_centers = np.sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2)
    norm_factor = np.sqrt(wB ** 2 + hB ** 2)
    norm_dist = dist_centers / max(1.0, norm_factor)
    r_near = float(np.exp(- (norm_dist ** 2) / 1.5))

    # 4. surrounded_by (A bị B bao quanh - OvSGTR open-vocabulary predicate)
    surround_count = 0
    if xB0 <= xA0: surround_count += 1
    if xB1 >= xA1: surround_count += 1
    if yB0 <= yA0: surround_count += 1
    if yB1 >= yA1: surround_count += 1
    r_surrounded_by = float(surround_count / 4.0 if inter_area > 0 else 0.0)

    # 5. across (A vắt ngang qua B - OvSGTR canonical predicate)
    width_ratio = min(2.0, wA / wB)
    crossing_level = inter_w / wB
    r_across = float(min(1.0, width_ratio * crossing_level) if crossing_level >= 0.7 else 0.0)

    # 6. hanging_from (A treo lơ lửng bên dưới B - OvSGTR canonical predicate)
    dy_hang = (yA0 - yB1) / max(hA, hB)
    if dy_hang >= -0.2 and dy_hang <= 0.3:
        mu_hang = max(0.0, 1.0 - abs(dy_hang) / 0.3)
    else:
        mu_hang = 0.0
    r_hanging_from = float(overlap_x * mu_hang)

    # 7. attached_to (gắn liền / đồng xuất hiện liên tục - OvSGTR canonical predicate)
    if frames_A and frames_B:
        common_frames = len(set(frames_A).intersection(set(frames_B)))
        total_frames = max(1, len(set(frames_A).union(set(frames_B))))
        co_persist = common_frames / total_frames
    else:
        co_persist = 0.5
    r_attached_to = float(overlap_x * co_persist)

    rels = {
        'on': round(min(1.0, max(0.0, r_on)), 4),
        'in': round(min(1.0, max(0.0, r_in)), 4),
        'near': round(min(1.0, max(0.0, r_near)), 4),
        'surrounded_by': round(min(1.0, max(0.0, r_surrounded_by)), 4),
        'across': round(min(1.0, max(0.0, r_across)), 4),
        'hanging_from': round(min(1.0, max(0.0, r_hanging_from)), 4),
        'attached_to': round(min(1.0, max(0.0, r_attached_to)), 4),
        # Alias keys for transparent legacy compatibility:
        'on_top_of': round(min(1.0, max(0.0, r_on)), 4),
        'inside': round(min(1.0, max(0.0, r_in)), 4),
        'next_to': round(min(1.0, max(0.0, r_near)), 4),
        'spanning_across': round(min(1.0, max(0.0, r_across)), 4),
        'emitting': round(min(1.0, max(0.0, r_attached_to)), 4),
        'attached to': round(min(1.0, max(0.0, r_attached_to)), 4),
        'hanging from': round(min(1.0, max(0.0, r_hanging_from)), 4),
    }

    return rels


def compute_segment_relations_and_triplets(tracks, track_memberships, rel_threshold=0.20, top_concepts_per_track=5):
    """
    Bước 8. Tính quan hệ bằng hàm hình học:
    - Với mỗi cặp track trong đoạn có participate_in_relations == True: tính 7 quan hệ.
    - Giữ các cặp có ít nhất 1 quan hệ >= rel_threshold.
    - Với mỗi bộ ba (khái niệm A, quan hệ r, khái niệm B):
      Độ thuộc bằng tích (t-norm tích) mu_A(cA) * mu_B(cB) * mu_r(A,B),
      rồi Noisy-OR trên mọi cặp track cho cùng bộ ba.
    - Trả về dictionary thưa: sparse_triplets[(cA_idx, r_idx, cB_idx)] = membership_score.
    """
    N_tracks = len(tracks)
    if N_tracks < 2:
        return {}

    # Find pairs of tracks eligible for relation calculation
    eligible_indices = [
        i for i, t in enumerate(tracks) if t.get("participate_in_relations", True)
    ]

    sparse_triplets = {}

    for idx_A in eligible_indices:
        tA = tracks[idx_A]
        memA = track_memberships[idx_A]
        fA = [d[0] for d in tA["detections"]]

        # Consider top-K most probable concepts for track A to keep compute sharp & efficient
        top_cA = np.argsort(-memA)[:top_concepts_per_track]

        for idx_B in eligible_indices:
            if idx_A == idx_B:
                continue

            tB = tracks[idx_B]
            memB = track_memberships[idx_B]
            fB = [d[0] for d in tB["detections"]]
            top_cB = np.argsort(-memB)[:top_concepts_per_track]

            # Compute the 7 geometric relations
            geom_rels = compute_box_geometry(tA["rep_box"], tB["rep_box"], orig_w=1920, orig_h=1080, frames_A=fA, frames_B=fB)

            # Filter active canonical OvSGTR relations with threshold
            active_rels = {r: geom_rels[r] for r in RELATION_NAMES if geom_rels.get(r, 0.0) >= rel_threshold}
            if not active_rels:
                continue

            for r_name, r_score in active_rels.items():
                r_idx = RELATION_NAMES.index(r_name)

                for cA in top_cA:
                    pA = float(memA[cA])
                    if pA < 0.05:
                        continue

                    for cB in top_cB:
                        pB = float(memB[cB])
                        if pB < 0.05:
                            continue

                        # T-norm tích: pA * r_score * pB
                        pair_triplet_score = pA * r_score * pB
                        triplet_key = f"{cA}_{r_idx}_{cB}"

                        # Noisy-OR accumulation over track pairs for this triplet
                        if triplet_key not in sparse_triplets:
                            sparse_triplets[triplet_key] = pair_triplet_score
                        else:
                            current = sparse_triplets[triplet_key]
                            sparse_triplets[triplet_key] = float(1.0 - (1.0 - current) * (1.0 - pair_triplet_score))

    return sparse_triplets
