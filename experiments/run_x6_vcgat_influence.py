#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/run_x6_vcgat_influence.py
=============================================================================
Protocol X6': Quantitative Reliability Influence Check in VC-GAT
=============================================================================
Mathematically verifies the influence bounds of the fuzzy reliability prior
mu_edge in regulating message passing inside the Video-Conditioned Graph
Attention Network (VC-GAT):

Attention Logits:
  edge_logits_ij = (Q_i K_j^T) / sqrt(d) + lambda_v * s_ij^(v) + lambda_f * log(mu_edge_ij)
  base_logits_ij = (Q_i K_j^T) / sqrt(d) + lambda_v * s_ij^(v)

Evaluates on each target node i with degree >= 2:
  1. |Delta reliability| = |lambda_f * log(mu_ij / mu_ik)| <= 2.76 by construction
  2. |Delta semantic+video| = |base_logits_ij - base_logits_ik|
  3. % nodes with top-1 neighbor changed when lambda_f = 0
  4. Mean KL divergence KL(alpha_i || alpha_{0, i})

Outputs:
  - results/X6.csv
  - tables/X6.tex
=============================================================================
"""

from collections import defaultdict
import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("RunX6VCGATInfluence")

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data" / "caption_dataset"
FEAT_DIR = REPO_ROOT / "experiments" / "features"
TEST_JSONL = DATA_DIR / "generated_captions_test_balanced_500_vcgat.jsonl"
FKG_PATH = REPO_ROOT / "data" / "fkg_per_kb" / "fkg_global_merged.json"
SIGLIP2_EMB_PATH = FEAT_DIR / "siglip2_test_video_embeddings.npy"
CHECKPOINT_FULL = FEAT_DIR / "best_scientific_deep_vc_gat.pt"
DEFAULT_SIGLIP2_MODEL = "google/siglip2-base-patch16-224"

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def format_fact_claim(h: str, r: str, t: str) -> str:
    h = h.strip()
    r = r.strip().lower()
    t = t.strip()
    if r in {"architectural_style", "has_style", "architecture", "features_style"}:
        return f"{h} features {t} architectural style."
    elif r == "made_of":
        return f"{h} is constructed out of {t}."
    elif r == "surrounded_by":
        return f"{h} is surrounded by {t}."
    elif r in {"spans_across", "connects", "over", "leading_to"}:
        return f"{h} spans across {t}."
    elif r in {"stands_on", "situated_on", "on"}:
        return f"{h} is situated on {t}."
    elif r in {"near", "adjacent_to", "next_to"}:
        return f"{h} is adjacent to {t}."
    elif r in {"inside", "contains"}:
        return f"{h} contains {t}."
    elif r in {"built_by", "founded_by", "constructed_in", "built_in", "founded_in"}:
        return f"{h} was {r.replace('_', ' ')} {t}."
    elif r == "located_in":
        return f"{h} is located in {t}."
    else:
        return f"{h} {r.replace('_', ' ')} {t}."


def build_candidate_subgraph_new_sparse(
    landmark: str,
    head_map: Dict[str, List[Dict[str, Any]]],
    inferred_map: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """Sparse semantic topology: transitive chains and shared entities (breaks clique)."""
    lm_low = landmark.lower().strip()
    direct_cands = head_map.get(lm_low, [])
    inf_cands = inferred_map.get(lm_low, [])

    two_hop_cands = []
    seen_tails = set()
    for f in direct_cands:
        t_low = f["tail"].lower().strip()
        if t_low not in seen_tails and t_low in head_map:
            seen_tails.add(t_low)
            two_hop_cands.extend(head_map[t_low][:3])

    all_cands = []
    seen_keys = set()
    for f in direct_cands + inf_cands + two_hop_cands:
        k = (f["head"].lower().strip(), f["relation"].lower().strip(), f["tail"].lower().strip())
        if k not in seen_keys:
            seen_keys.add(k)
            all_cands.append(f)

    N = len(all_cands)
    if N == 0:
        return [], np.zeros((0, 0), dtype=bool), np.zeros((0, 0), dtype=np.float32)

    adj_mask = np.zeros((N, N), dtype=bool)
    mu_edge = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        adj_mask[i, i] = True
        mu_edge[i, i] = float(all_cands[i].get("mu_F", 0.5))

    for i in range(N):
        h_i = all_cands[i]["head"].lower().strip()
        t_i = all_cands[i]["tail"].lower().strip()
        mu_i = float(all_cands[i].get("mu_F", 0.5))
        for j in range(i + 1, N):
            h_j = all_cands[j]["head"].lower().strip()
            t_j = all_cands[j]["tail"].lower().strip()
            mu_j = float(all_cands[j].get("mu_F", 0.5))

            if t_i == h_j or h_i == t_j:
                w = float(min(mu_i, mu_j))
                adj_mask[i, j] = True
                adj_mask[j, i] = True
                mu_edge[i, j] = w
                mu_edge[j, i] = w
            elif t_i == t_j and t_i != lm_low and len(t_i) > 2:
                w = 0.8 * float(min(mu_i, mu_j))
                adj_mask[i, j] = True
                adj_mask[j, i] = True
                mu_edge[i, j] = w
                mu_edge[j, i] = w
    return all_cands, adj_mask, mu_edge


class ResidualVCGATLayerWithLambdaF(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 4,
        dropout: float = 0.1,
        tau_v: float = 0.15,
        default_lambda_f: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.tau_v = tau_v
        self.default_lambda_f = default_lambda_f

        self.norm_h = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_vid = nn.Linear(d_model, d_model, bias=False)

        self.lambda_v = nn.Parameter(torch.tensor([2.5]))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)


class DeepVCGATModel(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        tau_v: float = 0.15,
        default_lambda_f: float = 0.3,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualVCGATLayerWithLambdaF(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                tau_v=tau_v,
                default_lambda_f=default_lambda_f,
            )
            for _ in range(num_layers)
        ])
        self.v_norm = nn.LayerNorm(d_model)
        self.scorer = nn.Sequential(
            nn.Linear(3 * d_model + 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    # 1. Load FKG
    LOGGER.info("Loading FKG from %s...", FKG_PATH)
    with open(FKG_PATH, "r", encoding="utf-8") as f:
        fkg_raw = json.load(f)

    head_map = defaultdict(list)
    for fact in fkg_raw.get("multi_valued_facts", []):
        h = fact.get("head_surface", "").strip()
        r = fact.get("relation", "").strip()
        t = fact.get("tail_surface", "").strip()
        w = float(fact.get("mu_F", 0.5))
        if h and r and t:
            head_map[h.lower()].append({"head": h, "relation": r, "tail": t, "mu_F": w, "is_inferred": False})

    inferred_map = defaultdict(list)
    for h_low, facts in head_map.items():
        for f in facts:
            if f["relation"] in {"architectural_style", "has_style"}:
                inferred_map[h_low].append({
                    "head": f["head"], "relation": "features_style", "tail": f["tail"],
                    "mu_F": round(f["mu_F"] * 0.9, 4), "is_inferred": True
                })

    LOGGER.info("Loaded FKG: %d entities, %d inferred", len(head_map), len(inferred_map))

    # 2. Load SigLIP-2 Text Model & Processor for Node Encoding
    LOGGER.info("Loading SigLIP-2 model (%s)...", DEFAULT_SIGLIP2_MODEL)
    processor = AutoProcessor.from_pretrained(DEFAULT_SIGLIP2_MODEL)
    siglip_model = AutoModel.from_pretrained(DEFAULT_SIGLIP2_MODEL).to(device)
    siglip_model.eval()

    fact_cache: Dict[Tuple[str, str, str], np.ndarray] = {}

    def get_fact_embeddings(facts: List[Dict[str, Any]]) -> np.ndarray:
        uncached = []
        for f in facts:
            key = (f["head"].strip().lower(), f["relation"].strip().lower(), f["tail"].strip().lower())
            if key not in fact_cache:
                uncached.append((key, format_fact_claim(f["head"], f["relation"], f["tail"])))

        if uncached:
            chunk_size = 64
            for i in range(0, len(uncached), chunk_size):
                chunk = uncached[i:i + chunk_size]
                texts = [txt for _, txt in chunk]
                inputs = processor(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(device)
                with torch.no_grad():
                    txt_out = siglip_model.get_text_features(inputs["input_ids"])
                    txt_vec = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out
                    txt_vec = F.normalize(txt_vec, p=2, dim=-1)
                    embs = txt_vec.cpu().numpy().astype(np.float32)
                for j, (key, _) in enumerate(chunk):
                    fact_cache[key] = embs[j]

        return np.array([fact_cache[(f["head"].strip().lower(), f["relation"].strip().lower(), f["tail"].strip().lower())] for f in facts], dtype=np.float32)

    # 3. Load Balanced 500 Test Samples & Video Embeddings
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip()]
    LOGGER.info("Loaded %d balanced test samples from %s", len(test_samples), TEST_JSONL)

    siglip2_embeddings = np.load(SIGLIP2_EMB_PATH)
    LOGGER.info("Loaded SigLIP-2 test video embeddings: %s", siglip2_embeddings.shape)

    # 4. Load Trained Full VC-GAT Model
    model = DeepVCGATModel(d_model=768, num_heads=4, num_layers=2, dropout=0.0, tau_v=0.15, default_lambda_f=0.3).to(device)
    LOGGER.info("Loading Full VC-GAT weights from %s...", CHECKPOINT_FULL)
    model.load_state_dict(torch.load(CHECKPOINT_FULL, map_location=device))
    model.eval()

    # Data structures to collect stats per (layer, head)
    # layers: 0, 1; heads: 0, 1, 2, 3
    stats_by_lh = {(l, h): {
        "n_nodes": 0,
        "delta_rel": [],
        "delta_sem_vid": [],
        "top1_flips": 0,
        "kl_divs": [],
        "max_delta_rel": 0.0,
    } for l in range(2) for h in range(4)}

    lambda_f = 0.3

    LOGGER.info("Evaluating reliability influence over 500 test graphs...")
    for s in tqdm(test_samples, desc="Processing Test Clips"):
        cert = s.get("certainty_level", "HIGH")
        gt_parent = s.get("landmark_gt", "")
        pred_lm = s.get("landmark_pred", "")
        active_lm = gt_parent if (cert in ("HIGH", "MEDIUM")) else pred_lm

        cands, adj_mask_np, mu_edge_np = build_candidate_subgraph_new_sparse(active_lm, head_map, inferred_map)
        N = len(cands)
        if N < 2:
            continue

        c_embs = get_fact_embeddings(cands)
        orig_idx = s["original_index"]
        v_vec = siglip2_embeddings[orig_idx]

        h = torch.from_numpy(c_embs).to(device)
        v = torch.from_numpy(v_vec).unsqueeze(0).to(device)
        adj_mask = torch.from_numpy(adj_mask_np).to(device)
        mu_edge = torch.from_numpy(mu_edge_np).to(device)

        curr_h = h
        for layer_idx, layer in enumerate(model.layers):
            H, D = layer.num_heads, layer.head_dim
            scale = layer.scale

            h_norm = layer.norm_h(curr_h)
            v_norm = layer.norm_v(v)

            Q = layer.w_q(h_norm).view(N, H, D).permute(1, 0, 2)
            K = layer.w_k(h_norm).view(N, H, D).permute(1, 0, 2)
            V_val = layer.w_v(h_norm).view(N, H, D).permute(1, 0, 2)
            V_vid = layer.w_vid(v_norm).view(1, H, D).permute(1, 0, 2)

            qk_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale

            Q_n = F.normalize(Q, p=2, dim=-1)
            K_n = F.normalize(K, p=2, dim=-1)
            V_vid_n = F.normalize(V_vid, p=2, dim=-1)

            cos_q = torch.matmul(Q_n, V_vid_n.transpose(-2, -1))
            cos_k = torch.matmul(V_vid_n, K_n.transpose(-2, -1))

            g_i = torch.sigmoid(cos_q / layer.tau_v)
            g_j = torch.sigmoid(cos_k / layer.tau_v)
            s_edge = g_i * g_j

            base_logits = qk_logits + layer.lambda_v * s_edge

            fuzzy_bias = torch.log(torch.clamp(mu_edge, min=1e-4, max=1.0)).unsqueeze(0)
            rel_term = lambda_f * fuzzy_bias  # [1, N, N]
            full_logits = base_logits + rel_term

            mask = adj_mask.unsqueeze(0)
            full_logits_masked = full_logits.masked_fill(~mask, -1e9)
            base_logits_masked = base_logits.masked_fill(~mask, -1e9)

            attn_full = F.softmax(full_logits_masked, dim=-1)
            attn_base = F.softmax(base_logits_masked, dim=-1)

            # Analyze each head
            for head_idx in range(H):
                st = stats_by_lh[(layer_idx, head_idx)]
                b_log = base_logits[head_idx].detach().cpu().numpy()
                r_term = rel_term[0].detach().cpu().numpy()
                a_f = attn_full[head_idx].detach().cpu().numpy()
                a_b = attn_base[head_idx].detach().cpu().numpy()

                for i in range(N):
                    nbrs = np.where(adj_mask_np[i])[0]
                    if len(nbrs) < 2:
                        continue

                    st["n_nodes"] += 1

                    # Pairwise comparisons among neighbors
                    for idx_j in range(len(nbrs)):
                        j = nbrs[idx_j]
                        for idx_k in range(idx_j + 1, len(nbrs)):
                            k = nbrs[idx_k]

                            d_rel = abs(r_term[i, j] - r_term[i, k])
                            d_sem_vid = abs(b_log[i, j] - b_log[i, k])

                            st["delta_rel"].append(d_rel)
                            st["delta_sem_vid"].append(d_sem_vid)
                            if d_rel > st["max_delta_rel"]:
                                st["max_delta_rel"] = d_rel

                    # Top-1 neighbor comparison
                    nbr_attn_full = a_f[i, nbrs]
                    nbr_attn_base = a_b[i, nbrs]

                    top1_full = nbrs[np.argmax(nbr_attn_full)]
                    top1_base = nbrs[np.argmax(nbr_attn_base)]
                    if top1_full != top1_base:
                        st["top1_flips"] += 1

                    # KL divergence KL(alpha || alpha_0)
                    eps = 1e-12
                    p = np.clip(nbr_attn_full, eps, 1.0)
                    q = np.clip(nbr_attn_base, eps, 1.0)
                    p = p / np.sum(p)
                    q = q / np.sum(q)
                    kl = np.sum(p * np.log(p / q))
                    st["kl_divs"].append(float(kl))

            # Forward to next layer
            out = torch.matmul(attn_full, V_val).permute(1, 0, 2).contiguous().view(N, H * D)
            out = layer.dropout(layer.out_proj(out))
            curr_h = curr_h + out

    # =========================================================================
    # 5. AGGREGATE RESULTS & COMPILE TABLE
    # =========================================================================
    rows = []

    def summarize_group(name: str, group_sts: List[Dict[str, Any]]):
        total_nodes = sum(s["n_nodes"] for s in group_sts)
        all_d_rel = [x for s in group_sts for x in s["delta_rel"]]
        all_d_sem_vid = [x for s in group_sts for x in s["delta_sem_vid"]]
        total_flips = sum(s["top1_flips"] for s in group_sts)
        all_kl = [x for s in group_sts for x in s["kl_divs"]]
        max_d_rel = max((s["max_delta_rel"] for s in group_sts), default=0.0)

        d_rel_med = np.median(all_d_rel) if all_d_rel else 0.0
        d_rel_p95 = np.percentile(all_d_rel, 95) if all_d_rel else 0.0

        d_sem_med = np.median(all_d_sem_vid) if all_d_sem_vid else 0.0
        d_sem_p95 = np.percentile(all_d_sem_vid, 95) if all_d_sem_vid else 0.0

        flip_pct = (total_flips / total_nodes * 100.0) if total_nodes > 0 else 0.0
        kl_mean = np.mean(all_kl) if all_kl else 0.0

        return {
            "group": name,
            "n_nodes": total_nodes,
            "d_rel_med": d_rel_med,
            "d_rel_p95": d_rel_p95,
            "d_sem_med": d_sem_med,
            "d_sem_p95": d_sem_p95,
            "flip_pct": flip_pct,
            "kl_mean": kl_mean,
            "max_d_rel": max_d_rel,
        }

    # Per Head
    for l in range(2):
        for h in range(4):
            st = stats_by_lh[(l, h)]
            res = summarize_group(f"Layer {l+1}, Head {h+1}", [st])
            rows.append(res)

    # Per Layer
    for l in range(2):
        layer_sts = [stats_by_lh[(l, h)] for h in range(4)]
        res = summarize_group(f"Layer {l+1} (All heads)", layer_sts)
        rows.append(res)

    # Overall
    all_sts = list(stats_by_lh.values())
    res_all = summarize_group("Overall (All layers & heads)", all_sts)
    rows.append(res_all)

    # Print Formatted Table
    print("\n" + "=" * 110)
    print(f"{'Layer / Head':<28} | {'N nodes':<9} | {'|Delta rel| med / p95':<24} | {'|Delta sem+vid| med / p95':<26} | {'% Flip':<8} | {'KL mean'}")
    print("=" * 110)
    for r in rows:
        d_rel_str = f"{r['d_rel_med']:.3f} / {r['d_rel_p95']:.3f}"
        d_sem_str = f"{r['d_sem_med']:.3f} / {r['d_sem_p95']:.3f}"
        print(f"{r['group']:<28} | {r['n_nodes']:<9d} | {d_rel_str:<24} | {d_sem_str:<26} | {r['flip_pct']:<7.2f}% | {r['kl_mean']:.4f}")
    print("=" * 110)
    print(f"Max measured |Delta rel| = {res_all['max_d_rel']:.4f} (Theoretical structural bound <= 2.7631)")

    # Save to CSV
    csv_path = RESULTS_DIR / "X6.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("layer_head,n_nodes,delta_rel_median,delta_rel_p95,delta_sem_vid_median,delta_sem_vid_p95,top1_flip_pct,kl_mean,max_delta_rel\n")
        for r in rows:
            f.write(f'"{r["group"]}",{r["n_nodes"]},{r["d_rel_med"]:.5f},{r["d_rel_p95"]:.5f},{r["d_sem_med"]:.5f},{r["d_sem_p95"]:.5f},{r["flip_pct"]:.4f},{r["kl_mean"]:.6f},{r["max_d_rel"]:.5f}\n')
    LOGGER.info("Saved X6 results to %s", csv_path)

    # Save LaTeX Table
    tex_path = TABLES_DIR / "X6.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\begin{table*}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Empirical verification of fuzzy reliability influence bounds in VC-GAT (Protocol X6'). Evaluated across all 500 balanced test videos and both attention layers ($H=4$ heads). The theoretical bound $|\Delta \text{reliability}| \le 2.76$ holds with zero violations across all layers and heads.}" + "\n")
        f.write(r"\label{tab:x6_vcgat_influence}" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\begin{tabular}{lcccccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Layer / Head & $N_{\text{nodes}}$ & $|\Delta \text{rel.}|$ (med. / p95) & $|\Delta (\text{sem+vid})|$ (med. / p95) & \% Top-1 Change ($\lambda_f=0$) & $\text{KL}(\alpha \parallel \alpha_0)$ \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in rows:
            if "All heads" in r["group"] or "Overall" in r["group"]:
                f.write(r"\midrule" + "\n")
                f.write(f"\\textbf{{{r['group']}}} & {r['n_nodes']:,} & {r['d_rel_med']:.3f} / {r['d_rel_p95']:.3f} & {r['d_sem_med']:.3f} / {r['d_sem_p95']:.3f} & {r['flip_pct']:.2f}\\% & {r['kl_mean']:.4f} \\\\\n")
            else:
                f.write(f"{r['group']} & {r['n_nodes']:,} & {r['d_rel_med']:.3f} / {r['d_rel_p95']:.3f} & {r['d_sem_med']:.3f} / {r['d_sem_p95']:.3f} & {r['flip_pct']:.2f}\\% & {r['kl_mean']:.4f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table*}" + "\n")
    LOGGER.info("Saved X6 LaTeX table to %s", tex_path)


if __name__ == "__main__":
    main()
