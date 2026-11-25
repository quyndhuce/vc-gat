#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiment_prop2c_prime.py
=============================================================================
Rigorous empirical verification of Proposition 2c' (Exact Stationary
Influence of Degree-Damped Attention):
- Detailed balance verification: pi_i * A_ij == pi_j * A_ji
- Closed form pi_i = a_i * S_i / sum_k a_k * S_k
- Power-law scaling pi_i propto (1 + d_i)^{1 - gamma} * a_bar_i

Experiments:
  X1: Exact stationary distribution pi_gamma on:
      (a) Hanoi KG (merged landmark subgraphs, 2,118 nodes)
      (b) HCMC KG (merged landmark subgraphs, 1,116 nodes)
      (c) Synthetic Barabási-Albert graph (N=500, m=2) with self-loops
      For gamma in {0.0, 0.20, 0.40, 0.60, 1.00}:
      - Check detailed balance (max error < 1e-14)
      - OLS log-log slope: log(pi_i) = beta * log(1 + d_i) + c (beta, SE, R^2)
      - Top-1% hub mass fraction
      - Gini coefficient of pi
      - Spectral gap: 1 - |lambda_2(A)|
  X2: Real attention matrix vs theoretical pi_gamma across Hanoi-500 test clips:
      - Extract real A (averaged across 4 heads) from best_dec_vcgat_v2.pt
      - Compute A^L column sums u_L for L in {1, 2, 3, 4}
      - Compare against theoretical stationary distribution pi_0.4:
        Spearman rho, Relative L1 error, Total Variation distance
      - Quantify graph asymmetry ||A - A^T||_F / ||A||_F
      - Quantify content logit spread M^* = max |score_ij - (-gamma * log(1+d_j))|
  X3: Damping sweep on Hanoi-500 test clips:
      - Post-hoc sweep gamma in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}:
        Overall nDCG@5, Hard-Negative AUC, Hub-Exposure@5
      - Compare against retrained ablation checkpoint without degree debiasing
        (best_ablation_wo_degreedebiasing.pt)
=============================================================================
"""

import sys
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import scipy.sparse as sp
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.fact_selection_benchmark.train_dec_vcgat_v2 import DecoupledEvidenceCalibratedVCGATv2, build_clip_graph
from experiments.fact_selection_benchmark.dataset_loader import (
    load_gold_data,
    load_normalized_facts,
    compute_ranking_metrics,
    compute_hard_negative_auc,
)
from experiments.fact_selection_benchmark.feature_cache import BenchmarkFeatureCache

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# HELPER FUNCTIONS FOR PROPOSITION 2c'
# =============================================================================

def compute_theoretical_pi(adj, degrees, gamma):
    """
    Computes exact stationary distribution pi for transition matrix:
    A_ij = a_j / sum_k a_k for (i, j) in E, with a_j = (1 + d_j)^{-gamma}.
    Here adj is N x N float tensor/array with self-loops (adj_ii = 1).
    Returns:
      pi: (N,) stationary distribution
      A:  (N, N) transition matrix
      detailed_balance_err: max_{i, j} |pi_i A_ij - pi_j A_ji|
      spectral_gap: 1 - |lambda_2(A)|
    """
    N = len(degrees)
    d = np.array(degrees, dtype=np.float64)
    # Damping vector
    a = (1.0 + d) ** (-gamma)  # shape (N,)
    
    # Adjacency with weights W = adj (assuming unweighted undirected graph with self-loops)
    # S_i = sum_{k in N(i)} W_ik * a_k
    W = np.array(adj, dtype=np.float64)
    S = W @ a  # shape (N,)
    
    # Avoid zero division if isolated node
    S_safe = np.where(S > 1e-12, S, 1.0)
    
    # Transition matrix A_ij = W_ij * a_j / S_i
    A = (W * a[None, :]) / S_safe[:, None]
    # For isolated nodes, A_ii = 1
    isolated = (S <= 1e-12)
    A[isolated, :] = 0.0
    A[isolated, isolated] = 1.0
    
    # Theoretical pi_i = a_i * S_i / sum_k (a_k * S_k)
    numerator = a * S
    Z = np.sum(numerator)
    if Z > 1e-12:
        pi = numerator / Z
    else:
        pi = np.ones(N) / N
        
    # Check detailed balance: pi_i * A_ij == pi_j * A_ji
    P_sym = pi[:, None] * A
    db_err = float(np.max(np.abs(P_sym - P_sym.T)))
    
    # Spectral gap
    if N > 2:
        try:
            # For reversible chain, D_pi^{1/2} A D_pi^{-1/2} is symmetric
            sqrt_pi = np.sqrt(np.maximum(pi, 1e-16))
            A_sym = (sqrt_pi[:, None] * A) / sqrt_pi[None, :]
            # Make exactly symmetric
            A_sym = 0.5 * (A_sym + A_sym.T)
            evals = np.linalg.eigvalsh(A_sym)
            evals = np.sort(evals)[::-1]  # descending
            # evals[0] should be 1.0
            lambda_2 = evals[1]
            spectral_gap = float(1.0 - np.abs(lambda_2))
        except Exception as e:
            spectral_gap = np.nan
    else:
        spectral_gap = 1.0
        
    return pi, A, db_err, spectral_gap


def compute_gini(x):
    """Computes Gini coefficient of array x."""
    x = np.asarray(x, dtype=np.float64)
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


def fit_log_log_slope(d, pi):
    """Fits OLS slope log(pi) = beta * log(1 + d) + c."""
    valid = (pi > 1e-16) & (d >= 0)
    x = np.log(1.0 + d[valid])
    y = np.log(pi[valid])
    # OLS
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return float(slope), float(std_err), float(r_value ** 2)


# =============================================================================
# TASK X1: EXACT STATIONARY DISTRIBUTION ACROSS GRAPHS
# =============================================================================

def run_task_x1(subgraphs, cache_dir):
    print("\n" + "=" * 70)
    print("TASK X1: EXACT STATIONARY DISTRIBUTION pi_gamma ON REAL & SYNTHETIC KGs")
    print("=" * 70)
    
    gammas = [0.0, 0.20, 0.40, 0.60, 1.00]
    
    # 1. Hanoi KG: Merged 13 landmark subgraphs
    hcmc_lids = {"ba_thien_hau_temple", "ben_thanh_market", "independence_palace", "saigon_central_post_office", "saigon_opera_house"}
    hanoi_lids = [lid for lid in subgraphs if lid not in hcmc_lids]
    
    def merge_subgraphs(lids):
        all_nodes = set()
        for lid in lids:
            all_nodes.update(subgraphs[lid]["fact_ids"])
        node_list = sorted(list(all_nodes))
        node2idx = {n: i for i, n in enumerate(node_list)}
        N = len(node_list)
        
        # Build adjacency
        adj_dense = np.zeros((N, N), dtype=np.float32)
        # Self-loops
        np.fill_diagonal(adj_dense, 1.0)
        
        for lid in lids:
            g = subgraphs[lid]
            fids = g["fact_ids"]
            fid2local = g["fid2idx"]
            edge_idx = g["edge_index"].cpu().numpy()
            for src, dst in zip(edge_idx[0], edge_idx[1]):
                u = node2idx[fids[src]]
                v = node2idx[fids[dst]]
                adj_dense[u, v] = 1.0
                adj_dense[v, u] = 1.0
                
        # Degrees (excluding self-loop for d_i, or including: degree = sum - 1)
        degrees = np.maximum(0, adj_dense.sum(axis=1) - 1.0)
        return node_list, adj_dense, degrees

    hanoi_nodes, hanoi_adj, hanoi_degrees = merge_subgraphs(hanoi_lids)
    print(f"Hanoi Merged KG: N = {len(hanoi_nodes)} nodes, Total edges = {int((hanoi_adj.sum() - len(hanoi_nodes)) / 2)}")
    
    # 2. HCMC KG: Merged 5 landmark subgraphs
    hcmc_nodes, hcmc_adj, hcmc_degrees = merge_subgraphs(list(hcmc_lids))
    print(f"HCMC Merged KG: N = {len(hcmc_nodes)} nodes, Total edges = {int((hcmc_adj.sum() - len(hcmc_nodes)) / 2)}")
    
    # 3. Synthetic Barabási-Albert graph (N=500, m=2)
    def generate_ba_graph(N=500, m=2, seed=SEED):
        import networkx as nx
        G = nx.barabasi_albert_graph(N, m, seed=seed)
        adj = nx.to_numpy_array(G, dtype=np.float32)
        np.fill_diagonal(adj, 1.0)
        degrees = adj.sum(axis=1) - 1.0
        return list(range(N)), adj, degrees
        
    ba_nodes, ba_adj, ba_degrees = generate_ba_graph(N=500, m=2)
    print(f"Synthetic BA Graph: N = {len(ba_nodes)} nodes, m = 2")
    
    graphs = {
        "Hanoi KG (N=2,118)": (hanoi_nodes, hanoi_adj, hanoi_degrees),
        "HCMC KG (N=1,116)": (hcmc_nodes, hcmc_adj, hcmc_degrees),
        "Synthetic BA (N=500, m=2)": (ba_nodes, ba_adj, ba_degrees)
    }
    
    x1_results = {}
    
    for gname, (nodes, adj, degrees) in graphs.items():
        print(f"\n--- Graph: {gname} ---")
        N = len(nodes)
        top1_pct_k = max(1, int(math.ceil(0.01 * N)))
        # Find top 1% hubs by degree
        hub_indices = np.argsort(degrees)[::-1][:top1_pct_k]
        
        x1_results[gname] = []
        
        for gamma in gammas:
            pi, A, db_err, gap = compute_theoretical_pi(adj, degrees, gamma)
            slope, se, r2 = fit_log_log_slope(degrees, pi)
            hub_mass = float(np.sum(pi[hub_indices]))
            gini = compute_gini(pi)
            
            rec = {
                "gamma": gamma,
                "theory_slope": float(1.0 - gamma),
                "empirical_slope": slope,
                "slope_se": se,
                "r2": r2,
                "top1_hub_mass_pct": hub_mass * 100.0,
                "gini": gini,
                "spectral_gap": gap,
                "detailed_balance_max_err": db_err
            }
            x1_results[gname].append(rec)
            print(f"  gamma={gamma:.2f} | Theory slope: {1.0-gamma:.2f} | Emp slope: {slope:.3f} +/- {se:.3f} (R2={r2:.3f}) | "
                  f"Top-1% Hub: {hub_mass*100:.2f}% | Gini: {gini:.3f} | Gap: {gap:.4f} | DB err: {db_err:.1e}")
            
    return x1_results, graphs


# =============================================================================
# TASK X2: REAL VC-GAT ATTENTION VS THEORETICAL pi_gamma ACROSS TEST CLIPS
# =============================================================================

def run_task_x2(test_clips, hanoi_cids, subgraphs, feature_cache, dec_model):
    print("\n" + "=" * 70)
    print("TASK X2: REAL ATTENTION MATRIX VS THEORETICAL pi_gamma ACROSS HANOI-500")
    print("=" * 70)
    
    dec_model.eval()
    
    # Store per-clip metrics for L in {1, 2, 3, 4}
    layers = [1, 2, 3, 4]
    per_clip_spearman = {L: [] for L in layers}
    per_clip_rel_l1 = {L: [] for L in layers}
    per_clip_tv = {L: [] for L in layers}
    per_clip_asym = []
    per_clip_logit_spread = []
    
    # Hook into forward pass to extract attention weights and raw scores
    attn_dict = {}
    def hook_fn(module, input, output):
        pass

    with torch.no_grad():
        for cid in hanoi_cids:
            recs = test_clips[cid]
            lid = recs[0].get("landmark_id", "unknown")
            g = subgraphs.get(lid)
            v_frames = feature_cache.video_frames.get(cid)
            v_pooled = feature_cache.video_pooled.get(cid)
            if g is None or v_frames is None:
                continue
                
            cand_fids = [r["fact_id"] for r in recs]
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device) if v_pooled is not None else None
            
            h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                g, cand_fids, feature_cache.fact_embeddings, device
            )
            N = h_tensor.size(0)
            
            # Replicate VC-GAT attention computation manually to extract exact scores & attn
            # 1. Visual Saliency Gate
            q = h_tensor.unsqueeze(1)
            kv = v_f_tensor.unsqueeze(0).expand(N, -1, -1)
            attn_out, _ = dec_model.cross_attn(query=q, key=kv, value=kv)
            ctx = dec_model.norm1(attn_out.squeeze(1) + h_tensor)
            vis_feat = torch.cat([h_tensor, ctx, h_tensor * ctx], dim=-1)
            g_nodes = dec_model.saliency_gate(vis_feat)
            
            # 2. Graph Attention
            Q = dec_model.q_proj(h_tensor).view(N, dec_model.num_heads, dec_model.head_dim).transpose(0, 1)
            h_trans = h_tensor * g_nodes
            K = dec_model.k_proj(h_trans).view(N, dec_model.num_heads, dec_model.head_dim).transpose(0, 1)
            scores = torch.bmm(Q, K.transpose(1, 2)) * dec_model.scale
            
            rel_emb = dec_model.rel_embed(rel_adj)
            rel_k = dec_model.rel_k_proj(rel_emb).view(N, N, dec_model.num_heads, dec_model.head_dim).permute(2, 0, 1, 3)
            scores_rel = torch.sum(Q.unsqueeze(2) * rel_k, dim=-1) * dec_model.scale
            scores = scores + scores_rel + dec_model.rel_bias(rel_adj).permute(2, 0, 1)
            
            content_scores = scores.clone() # before damping
            
            # Degree debiasing: gamma = 0.40
            deg_penalty = dec_model.degree_damping * torch.log1p(degrees.float()).unsqueeze(0).unsqueeze(1)
            scores_damped = scores - deg_penalty
            
            mask = (adj == 0).unsqueeze(0).expand(dec_model.num_heads, -1, -1)
            scores_damped = scores_damped.masked_fill(mask, -1e9)
            attn = F.softmax(scores_damped, dim=-1)
            attn_mean = attn.mean(dim=0).cpu().numpy() # [N, N]
            
            # Content logit spread M^* on graph edges: |content_scores_ij - (-gamma * log(1+d_j))|
            adj_np = adj.cpu().numpy()
            edge_mask = (adj_np > 0)
            # content logits mean across heads
            content_mean = content_scores.mean(dim=0).cpu().numpy()
            deg_pen_np = (dec_model.degree_damping * torch.log1p(degrees.float())).cpu().numpy()
            # M^* is maximum absolute difference on existing edges
            diff = np.abs(content_mean[edge_mask])
            m_star = float(np.max(diff)) if len(diff) > 0 else 0.0
            per_clip_logit_spread.append(m_star)
            
            # Graph asymmetry ||A - A^T||_F / ||A||_F
            norm_A = np.linalg.norm(attn_mean, 'fro')
            asym = np.linalg.norm(attn_mean - attn_mean.T, 'fro') / (norm_A + 1e-12)
            per_clip_asym.append(float(asym))
            
            # Theoretical stationary distribution on this clip graph
            degrees_np = degrees.cpu().numpy()
            pi_theo, A_theo, _, _ = compute_theoretical_pi(adj_np, degrees_np, gamma=0.40)
            
            # Matrix powers A^L
            A_cur = np.eye(N)
            for L in range(1, max(layers) + 1):
                A_cur = A_cur @ attn_mean
                if L in layers:
                    # Column sum normalized by N: average influence of node j on all nodes
                    u_L = np.mean(A_cur, axis=0) # shape (N,)
                    
                    # Spearman rank correlation
                    spearman_rho, _ = stats.spearmanr(u_L, pi_theo)
                    if np.isnan(spearman_rho):
                        spearman_rho = 0.0
                    per_clip_spearman[L].append(float(spearman_rho))
                    
                    # Relative L1 error: mean_j |u_L(j) - pi(j)| / pi(j)
                    rel_err = np.mean(np.abs(u_L - pi_theo) / np.maximum(pi_theo, 1e-12))
                    per_clip_rel_l1[L].append(float(rel_err))
                    
                    # Total Variation distance: 0.5 * sum |u_L(j) - pi(j)|
                    tv = 0.5 * np.sum(np.abs(u_L - pi_theo))
                    per_clip_tv[L].append(float(tv))

    x2_results = {
        "n_clips": len(per_clip_asym),
        "mean_asymmetry": float(np.mean(per_clip_asym)),
        "std_asymmetry": float(np.std(per_clip_asym)),
        "mean_logit_spread_M_star": float(np.mean(per_clip_logit_spread)),
        "max_logit_spread_M_star": float(np.max(per_clip_logit_spread)),
        "layers": {}
    }
    
    print(f"Evaluated across {len(per_clip_asym)} Hanoi test clips:")
    print(f"Graph Asymmetry ||A - A^T||_F / ||A||_F: {x2_results['mean_asymmetry']:.3f} +/- {x2_results['std_asymmetry']:.3f}")
    print(f"Content Logit Spread M^*: mean {x2_results['mean_logit_spread_M_star']:.2f}, max {x2_results['max_logit_spread_M_star']:.2f}")
    
    for L in layers:
        mean_sp = float(np.mean(per_clip_spearman[L]))
        std_sp = float(np.std(per_clip_spearman[L]))
        mean_rel = float(np.mean(per_clip_rel_l1[L]))
        std_rel = float(np.std(per_clip_rel_l1[L]))
        mean_tv = float(np.mean(per_clip_tv[L]))
        std_tv = float(np.std(per_clip_tv[L]))
        
        x2_results["layers"][f"L={L}"] = {
            "spearman": {"mean": mean_sp, "std": std_sp},
            "relative_l1": {"mean": mean_rel, "std": std_rel},
            "total_variation": {"mean": mean_tv, "std": std_tv}
        }
        print(f"  Layer L={L}: Spearman rho = {mean_sp:.3f} +/- {std_sp:.3f} | Rel L1 Err = {mean_rel:.3f} +/- {std_rel:.3f} | TV = {mean_tv:.4f} +/- {std_tv:.4f}")
        
    return x2_results


# =============================================================================
# TASK X3: DAMPING SWEEP & HUB-EXPOSURE ON HANOI-500
# =============================================================================

def run_task_x3(test_clips, hanoi_cids, subgraphs, feature_cache, dec_model, ablation_ckpt_path):
    print("\n" + "=" * 70)
    print("TASK X3: DAMPING SWEEP (gamma in [0.0, 1.0]) & HUB-EXPOSURE ON HANOI-500")
    print("=" * 70)
    
    # Pre-identify top 1% hubs for each landmark subgraph
    landmark_top1_hubs = {}
    for lid, g in subgraphs.items():
        degrees = g["degrees"].cpu().numpy()
        fids = g["fact_ids"]
        k = max(1, int(math.ceil(0.01 * len(fids))))
        top_idx = np.argsort(degrees)[::-1][:k]
        landmark_top1_hubs[lid] = set(fids[idx] for idx in top_idx)
        
    gammas = [0.0, 0.20, 0.40, 0.60, 0.80, 1.00]
    sweep_results = []
    
    for gamma in gammas:
        dec_model.eval()
        dec_model.degree_damping = gamma # set damping factor
        
        ndcg_list = []
        auc_list = []
        hub_exposure_list = []
        
        with torch.no_grad():
            for cid in hanoi_cids:
                recs = test_clips[cid]
                lid = recs[0].get("landmark_id", "unknown")
                g = subgraphs.get(lid)
                v_frames = feature_cache.video_frames.get(cid)
                v_pooled = feature_cache.video_pooled.get(cid)
                if g is None or v_frames is None:
                    continue
                    
                cand_fids = [r["fact_id"] for r in recs]
                gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
                hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
                
                v_f_tensor = torch.from_numpy(v_frames).float().to(device)
                v_p_tensor = torch.from_numpy(v_pooled).float().to(device) if v_pooled is not None else None
                
                h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                    g, cand_fids, feature_cache.fact_embeddings, device
                )
                
                logits_d, _, _, _ = dec_model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
                scores_d = logits_d.cpu().numpy()
                f_scores_d = {fid: float(scores_d[fid2idx[fid]]) for fid in cand_fids}
                ranked_d = sorted(cand_fids, key=lambda fid: f_scores_d[fid], reverse=True)
                
                m = compute_ranking_metrics(ranked_d, gold_fids, cand_fids, top_k_list=[5])
                ndcg_list.append(m["nDCG@5"])
                
                auc = compute_hard_negative_auc(f_scores_d, gold_fids, hard_neg_fids)
                if auc is not None:
                    auc_list.append(auc)
                    
                # Hub exposure @ 5: fraction of top-5 predicted facts that are top 1% hubs of landmark
                hubs = landmark_top1_hubs.get(lid, set())
                top5_preds = ranked_d[:5]
                hub_count = sum(1 for fid in top5_preds if fid in hubs)
                hub_exposure_list.append(hub_count / 5.0)
                
        rec = {
            "mode": f"Post-hoc damping gamma={gamma:.2f}",
            "gamma": gamma,
            "overall_ndcg5": float(np.mean(ndcg_list)),
            "hard_neg_auc": float(np.mean(auc_list)),
            "hub_exposure_at_5": float(np.mean(hub_exposure_list)) * 100.0 # percentage
        }
        sweep_results.append(rec)
        print(f"gamma={gamma:.2f} | Overall nDCG@5: {rec['overall_ndcg5']:.4f} | Hard-Neg AUC: {rec['hard_neg_auc']:.4f} | "
              f"Hub-Exposure@5: {rec['hub_exposure_at_5']:.2f}%")

    # Reset dec_model damping to default 0.40
    dec_model.degree_damping = 0.40

    # Also evaluate the retrained ablation checkpoint without degree debiasing
    print("\n--- Evaluating Retrained Ablation Checkpoint (best_ablation_wo_degreedebiasing.pt) ---")
    ablation_model = DecoupledEvidenceCalibratedVCGATv2(d_model=768, degree_damping=0.0).to(device)
    ablation_model.load_state_dict(torch.load(ablation_ckpt_path, map_location=device))
    ablation_model.eval()
    
    abl_ndcg = []
    abl_auc = []
    abl_hub_exposure = []
    
    with torch.no_grad():
        for cid in hanoi_cids:
            recs = test_clips[cid]
            lid = recs[0].get("landmark_id", "unknown")
            g = subgraphs.get(lid)
            v_frames = feature_cache.video_frames.get(cid)
            v_pooled = feature_cache.video_pooled.get(cid)
            if g is None or v_frames is None:
                continue
                
            cand_fids = [r["fact_id"] for r in recs]
            gold_fids = set(r["fact_id"] for r in recs if r.get("is_scene_gold"))
            hard_neg_fids = set(r["fact_id"] for r in recs if r.get("gold_label") == "HARD_NEGATIVE")
            
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device) if v_pooled is not None else None
            
            h_tensor, adj, rel_adj, degrees, all_fids, fid2idx = build_clip_graph(
                g, cand_fids, feature_cache.fact_embeddings, device
            )
            
            logits_d, _, _, _ = ablation_model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            scores_d = logits_d.cpu().numpy()
            f_scores_d = {fid: float(scores_d[fid2idx[fid]]) for fid in cand_fids}
            ranked_d = sorted(cand_fids, key=lambda fid: f_scores_d[fid], reverse=True)
            
            m = compute_ranking_metrics(ranked_d, gold_fids, cand_fids, top_k_list=[5])
            abl_ndcg.append(m["nDCG@5"])
            
            auc = compute_hard_negative_auc(f_scores_d, gold_fids, hard_neg_fids)
            if auc is not None:
                abl_auc.append(auc)
                
            hubs = landmark_top1_hubs.get(lid, set())
            top5_preds = ranked_d[:5]
            hub_count = sum(1 for fid in top5_preds if fid in hubs)
            abl_hub_exposure.append(hub_count / 5.0)

    abl_rec = {
        "mode": "Retrained without debiasing (gamma=0)",
        "gamma": 0.0,
        "overall_ndcg5": float(np.mean(abl_ndcg)),
        "hard_neg_auc": float(np.mean(abl_auc)),
        "hub_exposure_at_5": float(np.mean(abl_hub_exposure)) * 100.0
    }
    sweep_results.append(abl_rec)
    print(f"Retrained (gamma=0.0) | Overall nDCG@5: {abl_rec['overall_ndcg5']:.4f} | Hard-Neg AUC: {abl_rec['hard_neg_auc']:.4f} | "
          f"Hub-Exposure@5: {abl_rec['hub_exposure_at_5']:.2f}%")
          
    return sweep_results


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main():
    print("=" * 70)
    print("STARTING EXPERIMENT SUITE: PROPOSITION 2c'")
    print("=" * 70)
    
    # 1. Load data and models
    facts = load_normalized_facts()
    test_clips, _ = load_gold_data(split_type="test")
    hanoi_cids = sorted([cid for cid in test_clips if not cid.startswith("hcmc_")])
    print(f"Loaded {len(facts)} facts, {len(hanoi_cids)} Hanoi test clips.")
    
    feature_cache = BenchmarkFeatureCache(device=device)
    feature_cache.extract_fact_embeddings(facts)
    feature_cache.extract_video_features([])
    subgraphs = torch.load(feature_cache.cache_dir / "landmark_subgraphs_rel.pt", weights_only=False)
    
    dec_ckpt = REPO_ROOT / "experiments" / "checkpoints_fact_selection" / "best_dec_vcgat_v2.pt"
    dec_model = DecoupledEvidenceCalibratedVCGATv2(d_model=768).to(device)
    dec_model.load_state_dict(torch.load(dec_ckpt, map_location=device))
    dec_model.eval()
    
    abl_ckpt = REPO_ROOT / "experiments" / "checkpoints_fact_selection" / "best_ablation_wo_degreedebiasing.pt"
    
    # 2. Run Task X1
    x1_results, _ = run_task_x1(subgraphs, feature_cache.cache_dir)
    
    # 3. Run Task X2
    x2_results = run_task_x2(test_clips, hanoi_cids, subgraphs, feature_cache, dec_model)
    
    # 4. Run Task X3
    x3_results = run_task_x3(test_clips, hanoi_cids, subgraphs, feature_cache, dec_model, abl_ckpt)
    
    # 5. Export comprehensive results
    output_data = {
        "task_x1_exact_pi": x1_results,
        "task_x2_real_attention_vs_pi": x2_results,
        "task_x3_damping_sweep": x3_results
    }
    
    out_file = REPO_ROOT / "results_prop2c_prime.json"
    out_file.write_text(json.dumps(output_data, indent=2))
    print(f"\nAll Prop 2c' results exported to: {out_file}")


if __name__ == "__main__":
    main()
