"""
Hierarchical Paired Cluster Bootstrap across Parent Video Clusters (Section 6.1).
"""

import numpy as np
from typing import Dict, List, Tuple


def cluster_bootstrap_ci(
    cluster_scores: Dict[str, List[float]],
    b_resamples: int = 10000,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Computes cluster bootstrap mean and (1 - alpha) confidence interval.
    Resamples clusters with replacement to preserve intra-cluster dependencies.
    """
    cluster_keys = list(cluster_scores.keys())
    n_clusters = len(cluster_keys)
    if n_clusters == 0:
        return 0.0, 0.0, 0.0

    cluster_means = np.array([np.mean(cluster_scores[k]) for k in cluster_keys])
    cluster_weights = np.array([len(cluster_scores[k]) for k in cluster_keys])
    point_estimate = float(np.sum(cluster_means * cluster_weights) / np.sum(cluster_weights))

    boot_means = []
    rng = np.random.default_rng(42)

    for _ in range(b_resamples):
        sample_indices = rng.choice(n_clusters, size=n_clusters, replace=True)
        s_means = cluster_means[sample_indices]
        s_weights = cluster_weights[sample_indices]
        w_sum = np.sum(s_weights)
        if w_sum > 0:
            boot_means.append(np.sum(s_means * s_weights) / w_sum)

    boot_means = np.sort(boot_means)
    low_idx = int((alpha / 2.0) * len(boot_means))
    high_idx = int((1.0 - alpha / 2.0) * len(boot_means))

    ci_low = float(boot_means[low_idx])
    ci_high = float(boot_means[high_idx])

    return point_estimate, ci_low, ci_high
