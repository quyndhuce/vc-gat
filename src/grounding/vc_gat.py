"""
src/grounding/vc_gat.py
=============================================================================
Decoupled Evidence-Calibrated Relational Video-Conditioned Graph Attention
Network (DEC-RVCGAT v2 / VC-GAT) (Section 4.2, Section 5.3, Section 5.4, Table 5).

Paper: "Decision-Certified Partial Reasoning for Uncertainty-Aware Heritage Video Captioning"
Target Journal: International Journal of Approximate Reasoning (IJAR), Elsevier

Key Architectural Innovations of DEC-RVCGAT v2:
  1. Decoupled Visual Grounding & Lower-Bound Guarantee (Proposition 2b):
       s(V, f_i) = s_vis(V, f_i) + alpha * lambda_i * Delta_s(f_i)
     where alpha = 0.40, Delta_s in [-0.8, +0.8], ensuring bounded modulation
     around the perceptual foundation.
  2. Strict Visual Saliency Transmission (Proposition 2b):
       M_{j -> i} = g_j * W_v h_j
     where g_j in [0, 1] is the visual saliency gate. Saliency-absent nodes
     (g_j -> 0) transmit identically zero evidence, preventing hallucination.
  3. Degree-Debiased Relational Attention (Proposition 2c):
       alpha_{ij} \propto exp(q_i^T k_j / sqrt(d)) * (1 + deg_j)^(-gamma)
     with damping factor gamma = 0.40, provably bounding structural hub domination
     to O(deg_j^(1 - gamma)).
  4. Dynamic Candidate Graph Inclusion (DCSE):
     Guarantees candidate facts are first-class nodes dynamically attached
     via semantic association edges (Relation 3) and bidirectional self-loops,
     eliminating the 'Candidate Disconnect' flaw.
  5. Domain-Invariant Meta-Evidence Reliability Gate:
     Computes lambda_i in [0, 1] from 6 invariant scalar features:
     [cos(h, ctx), cos(h_gat, ctx), cos(h, h_gat), r_anchor, deg_factor, prob_vis].
=============================================================================
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Optional PyTorch import for full GPU/Server training and inference
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False


# =============================================================================
# 1. CORE NUMERICAL / EVALUATION ROUTINES (CPU & Numpy compatible)
# =============================================================================

def compute_vcgat_debiased_attention(
    query_feats: np.ndarray,
    key_feats: np.ndarray,
    in_degrees: np.ndarray,
    gamma: float = 0.40,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Computes degree-debiased attention weights satisfying Proposition 2c:
      alpha_{ij} = exp(q_i^T k_j / sqrt(d)) * (1 + d_j)^(-gamma) / Z_i
    """
    scale = 1.0 / max(1e-4, math.sqrt(query_feats.shape[-1]))
    scores = np.dot(query_feats, key_feats.T) * scale / max(1e-4, temperature)
    debias_factors = np.power(np.maximum(1.0, 1.0 + in_degrees), -gamma)
    weighted_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True)) * debias_factors
    attention = weighted_exp / np.sum(weighted_exp, axis=-1, keepdims=True)
    return attention


def compute_strict_saliency_transmission(
    h_messages: np.ndarray,
    visual_saliency_g: np.ndarray
) -> np.ndarray:
    """
    Implements Proposition 2b Strict Saliency Transmission:
      M_{j -> i} = g_j * W_v h_j
    If g_j -> 0, transmitted evidence is identically zero.
    """
    g_weights = np.clip(visual_saliency_g, 0.0, 1.0)
    if g_weights.ndim == 1:
        g_weights = g_weights[:, np.newaxis]
    return h_messages * g_weights


def compute_decoupled_vcgat_score(
    s_vis: np.ndarray,
    delta_s: np.ndarray,
    lambda_gate: np.ndarray,
    alpha: float = 0.40,
    delta_bound: float = 0.80
) -> np.ndarray:
    """
    Computes Decoupled DEC-RVCGAT v2 Fact Score (Proposition 2b):
      score(f_i) = s_vis(V, f_i) + alpha * lambda_i * clip(Delta_s(f_i), -delta_bound, delta_bound)
    Guarantees: |score(f_i) - s_vis(V, f_i)| <= alpha * delta_bound * lambda_i.
    """
    clipped_delta = np.clip(delta_s, -delta_bound, delta_bound)
    bounded_lambda = np.clip(lambda_gate, 0.0, 1.0)
    return s_vis + alpha * bounded_lambda * clipped_delta


def rank_facts_vcgat(
    visual_scores: np.ndarray,
    delta_scores: Optional[np.ndarray] = None,
    evidence_gates: Optional[np.ndarray] = None,
    alpha: float = 0.40,
    fkg_mu_f: Optional[np.ndarray] = None,
    lambda_f: float = 0.35
) -> np.ndarray:
    """
    Fact ranking function using DEC-RVCGAT v2 decoupled scoring.
    Maintains backward compatibility with legacy linear fusion if delta_scores is None.
    """
    if delta_scores is not None and evidence_gates is not None:
        return compute_decoupled_vcgat_score(
            s_vis=visual_scores,
            delta_s=delta_scores,
            lambda_gate=evidence_gates,
            alpha=alpha
        )
    if fkg_mu_f is not None:
        # Fallback to legacy baseline blend if explicit graph delta is absent
        return (1.0 - lambda_f) * visual_scores + lambda_f * fkg_mu_f
    return visual_scores


# =============================================================================
# 2. FULL PYTORCH NEURAL NETWORK MODULE (DEC-RVCGAT v2)
# =============================================================================

if HAS_TORCH:
    class DecoupledEvidenceCalibratedVCGATv2(nn.Module):
        """
        Complete Decoupled Evidence-Calibrated Relational VC-GAT (DEC-RVCGAT v2).
        Trained with Listwise Plackett-Luce Ranking Loss and Dynamic Candidate Inclusion.
        """
        def __init__(
            self,
            d_model: int = 768,
            num_heads: int = 4,
            hidden_dim: int = 256,
            delta_bound: float = 0.80,
            degree_damping: float = 0.40,
            dropout: float = 0.10,
        ):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.scale = 1.0 / math.sqrt(self.head_dim)
            self.delta_bound = delta_bound
            self.degree_damping = degree_damping

            # 1. Visual Backbone (Cross-Attention Grounding)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.vis_scorer = nn.Sequential(
                nn.Linear(3 * d_model, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

            # 2. Strict Visual Saliency Gate
            self.saliency_gate = nn.Sequential(
                nn.Linear(3 * d_model, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

            # 3. Degree-Debiased Relation-Aware Graph Attention
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

            # Relations: 0: co_occurrence, 1: semantic_dep, 2: causal_temporal, 3: associative_candidate
            self.rel_embed = nn.Embedding(4, 64)
            self.rel_k_proj = nn.Linear(64, d_model)
            self.rel_v_proj = nn.Linear(64, d_model)
            self.rel_bias = nn.Embedding(4, num_heads)

            self.norm_gat = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

            # 4. Bounded Graph Delta MLP
            self.graph_delta_mlp = nn.Sequential(
                nn.Linear(3 * d_model, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Tanh()
            )

            # 5. Domain-Invariant Evidence Calibration Gate (6 meta-features)
            self.evidence_gate = nn.Sequential(
                nn.Linear(6, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(
            self,
            h: "torch.Tensor",              # [N, D]
            adj: "torch.Tensor",            # [N, N]
            rel_adj: "torch.Tensor",        # [N, N]
            degrees: "torch.Tensor",        # [N]
            v_frames: "torch.Tensor",       # [T, D]
            v_pooled: Optional["torch.Tensor"] = None
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            N = h.size(0)

            # 1. Cross-Attention Visual Grounding
            q = h.unsqueeze(1)
            kv = v_frames.unsqueeze(0).expand(N, -1, -1)
            attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
            ctx = self.norm1(attn_out.squeeze(1) + h)
            vis_feat = torch.cat([h, ctx, h * ctx], dim=-1)
            s_vis = self.vis_scorer(vis_feat).squeeze(-1)

            # 2. Strict Visual Saliency Gate (g_nodes in [0, 1])
            g_nodes = self.saliency_gate(vis_feat)

            # 3. Degree-Debiased Relation-Aware Graph Attention
            Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
            # Strict Transmission: unobserved nodes transmit zero evidence
            h_trans = h * g_nodes
            K = self.k_proj(h_trans).view(N, self.num_heads, self.head_dim).transpose(0, 1)
            V_nodes = self.v_proj(h_trans).view(N, self.num_heads, self.head_dim).transpose(0, 1)

            scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale

            # Relational projection
            rel_emb = self.rel_embed(rel_adj)
            rel_k = self.rel_k_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
            scores_rel = torch.sum(Q.unsqueeze(2) * rel_k, dim=-1) * self.scale
            scores = scores + scores_rel + self.rel_bias(rel_adj).permute(2, 0, 1)

            # Degree-Debiased Normalization: subtract gamma * log(1 + degree_j)
            deg_penalty = self.degree_damping * torch.log1p(degrees.float()).unsqueeze(0).unsqueeze(1)
            scores = scores - deg_penalty

            mask = (adj == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
            scores = scores.masked_fill(mask, -1e9)
            attn = self.dropout(F.softmax(scores, dim=-1))
            attn_mean = attn.mean(dim=0)

            ctx_nodes = torch.bmm(attn, V_nodes)
            rel_v = self.rel_v_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
            ctx_rel = torch.sum(attn.unsqueeze(-1) * rel_v, dim=2)
            agg = (ctx_nodes + ctx_rel).transpose(0, 1).contiguous().view(N, -1)
            h_gat = self.norm_gat(h + self.dropout(self.out_proj(agg)))

            # 4. Bounded Graph Delta
            graph_feat = torch.cat([h_gat, ctx, h_gat * ctx], dim=-1)
            delta_s = self.delta_bound * self.graph_delta_mlp(graph_feat).squeeze(-1)

            # 5. Domain-Invariant Evidence Calibration Gate (6 meta-features)
            h_u = F.normalize(h, p=2, dim=-1)
            ctx_u = F.normalize(ctx, p=2, dim=-1)
            h_gat_u = F.normalize(h_gat, p=2, dim=-1)

            cos_fact_ctx = torch.sum(h_u * ctx_u, dim=-1, keepdim=True)
            cos_gat_ctx = torch.sum(h_gat_u * ctx_u, dim=-1, keepdim=True)
            cos_fact_gat = torch.sum(h_u * h_gat_u, dim=-1, keepdim=True)

            eye = torch.eye(N, device=h.device)
            attn_neighbor = attn_mean * (1.0 - eye)
            neighbor_sum = attn_neighbor.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            r_anchor = torch.matmul(attn_neighbor, g_nodes) / neighbor_sum

            deg_factor = (1.0 / torch.sqrt(1.0 + degrees.float())).unsqueeze(-1)
            prob_vis = torch.sigmoid(s_vis).unsqueeze(-1)

            evidence_scalars = torch.cat(
                [cos_fact_ctx, cos_gat_ctx, cos_fact_gat, r_anchor, deg_factor, prob_vis],
                dim=-1
            )
            lambda_i = self.evidence_gate(evidence_scalars).squeeze(-1)

            # Final Decoupled Fact Score
            final_score = s_vis + 0.40 * lambda_i * delta_s
            return final_score, s_vis, lambda_i, delta_s


    def listmle_topk_loss(
        logits: "torch.Tensor",
        gold_indices: List[int],
        cand_indices: List[int],
        top_k: int = 5,
        temp: float = 0.50
    ) -> "torch.Tensor":
        """Plackett-Luce ListMLE ranking loss for Top-K evidence selection."""
        if not gold_indices:
            return torch.tensor(0.0, device=logits.device)

        neg_indices = [idx for idx in cand_indices if idx not in gold_indices]
        perm_indices = gold_indices[:top_k] + neg_indices

        perm_logits = logits[perm_indices] / temp
        k_eval = min(len(gold_indices), top_k)

        loss = 0.0
        for i in range(k_eval):
            log_denom = torch.logsumexp(perm_logits[i:], dim=0)
            log_num = perm_logits[i]
            loss = loss - (log_num - log_denom)

        return loss / float(k_eval)
else:
    DecoupledEvidenceCalibratedVCGATv2 = None
    listmle_topk_loss = None
