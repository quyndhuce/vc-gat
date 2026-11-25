"""
Knowledge Reliability & Captioning Evaluation Metrics (Section 6.1).
BKR@5, CSR, FP, KR, VFAR, nDCG@5, P@5, R@5.
"""

import numpy as np
from typing import List, Dict, Set, Any, Tuple


def compute_ndcg_at_k(ranked_labels: List[int], k: int = 5) -> float:
    """Computes Normalized Discounted Cumulative Gain at K (nDCG@K)."""
    k = min(k, len(ranked_labels))
    if k == 0:
        return 0.0

    dcg = 0.0
    for i in range(k):
        rel = ranked_labels[i]
        dcg += (2.0**rel - 1.0) / np.log2(i + 2.0)

    ideal_labels = sorted(ranked_labels, reverse=True)[:k]
    idcg = 0.0
    for i in range(k):
        rel = ideal_labels[i]
        idcg += (2.0**rel - 1.0) / np.log2(i + 2.0)

    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_precision_recall_at_k(ranked_labels: List[int], k: int = 5) -> Tuple[float, float]:
    """Computes Precision@K and Recall@K."""
    k = min(k, len(ranked_labels))
    if k == 0:
        return 0.0, 0.0

    hits = sum(ranked_labels[:k])
    total_pos = sum(ranked_labels)

    prec = float(hits / k)
    rec = float(hits / total_pos) if total_pos > 0 else 0.0
    return prec, rec


def compute_clip_bkr_at_k(attributed_facts: Set[str], gold_facts: Set[str], k: int = 5) -> float:
    """
    Bounded Knowledge Recall at K (BKR@K):
      BKR@K = |Attributed intersect Gold| / min(K, |Gold|)
    """
    if not gold_facts:
        return 0.0
    denom = min(k, len(gold_facts))
    correct = len(attributed_facts & gold_facts)
    return float(correct / denom)
