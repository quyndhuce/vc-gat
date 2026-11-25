import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.optim import AdamW
import numpy as np

REPO_ROOT = Path(\"TransMTL_OntoKG-main\")
sys.path.insert(0, str(REPO_ROOT))

from experiments.fact_selection_benchmark.dataset_loader import (
    load_normalized_facts, load_gold_data, load_contrastive_pairs,
    compute_ranking_metrics, compute_hard_negative_auc, compute_contrastive_accuracy
)
from experiments.fact_selection_benchmark.feature_cache import BenchmarkFeatureCache

device = torch.device(\"cuda:0\")
facts = load_normalized_facts()
train_clips, _ = load_gold_data(split_type=\"train\")
val_clips, _ = load_gold_data(split_type=\"val\")
test_clips, _ = load_gold_data(split_type=\"test\")
train_contrastive = load_contrastive_pairs(split_type=\"train_val\")
test_contrastive = load_contrastive_pairs(split_type=\"test\")

feature_cache = BenchmarkFeatureCache(device=device)
feature_cache.extract_fact_embeddings(facts)
feature_cache.extract_video_features([])
rel_subgraphs = torch.load(feature_cache.cache_dir / \"landmark_subgraphs_rel.pt\", weights_only=False)

# Architecture: Cross-Attention Visual Backbone + Relation-Aware Residual Graph + Scene Hard-Neg Margin
class FullUnifiedRVCGAT(nn.Module):
    def __init__(self, d_model=768, num_heads=4, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 1. Cross-Attention Visual Backbone (Exact same as B5)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)
        self.vis_scorer = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # 2. Node Saliency Gate
        self.saliency_gate = nn.Sequential(
            nn.Linear(3 * d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 3. Relation-Aware Graph Attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rel_embed = nn.Embedding(4, 64)
        self.rel_k_proj = nn.Linear(64, d_model)
        self.rel_v_proj = nn.Linear(64, d_model)
        self.rel_bias = nn.Embedding(4, num_heads)
        self.norm_gat = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 4. Graph Relational Delta Scorer
        self.graph_delta_scorer = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # 5. Evidence-Based Reliability Gate
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model + 3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        nn.init.constant_(self.gate_mlp[-1].bias, -0.6)

    def forward(self, h, adj_mask, rel_adj, degrees, v_frames, v_pooled):
        # h: [N, D]
        # v_frames: [K=8, D]
        if v_frames.dim() == 2:
            v_frames = v_frames.unsqueeze(0) # [1, K, D]
        N = h.size(0)
        q = h.unsqueeze(1)            # [N, 1, D]
        kv = v_frames.expand(N, -1, -1) # [N, K, D]

        # A. Visual Cross-Attention Backbone
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        ctx = self.norm_cross(attn_out.squeeze(1) + h) # [N, D]
        vis_feat = torch.cat([h, ctx, h * ctx], dim=-1) # [N, 3D]
        s_vis = self.vis_scorer(vis_feat).squeeze(-1) # [N]

        # B. Node Saliency Gate
        g_nodes = self.saliency_gate(vis_feat) # [N, 1]

        # C. Relation-Aware VC-GAT
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        h_gated = h * (0.5 + 0.5 * g_nodes)
        K = self.k_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V_nodes = self.v_proj(h_gated).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale

        rel_emb = self.rel_embed(rel_adj) # [N, N, 64]
        rel_k = self.rel_k_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        q_exp = Q.unsqueeze(2)
        scores_rel = torch.sum(q_exp * rel_k, dim=-1) * self.scale
        scores = scores + scores_rel + self.rel_bias(rel_adj).permute(2, 0, 1)

        mask = (adj_mask == 0).unsqueeze(0).expand(self.num_heads, -1, -1)
        scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1)) # [H, N, N]
        attn_mean = attn.mean(dim=0) # [N, N]

        ctx_nodes = torch.bmm(attn, V_nodes)
        rel_v = self.rel_v_proj(rel_emb).view(N, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        ctx_rel = torch.sum(attn.unsqueeze(-1) * rel_v, dim=2)
        h_gat = self.norm_gat(h + self.dropout(self.out_proj((ctx_nodes + ctx_rel).transpose(0, 1).contiguous().view(N, -1))))

        # D. Graph Delta Score
        graph_feat = torch.cat([h_gat, ctx, h_gat * ctx], dim=-1)
        s_graph_delta = self.graph_delta_scorer(graph_feat).squeeze(-1)

        # E. Evidence-Based Reliability Gate
        eye = torch.eye(N, device=h.device)
        attn_neighbor = attn_mean * (1.0 - eye)
        neighbor_sum = attn_neighbor.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        r_anchor = torch.matmul(attn_neighbor, g_nodes) / neighbor_sum

        h_gat_u = F.normalize(h_gat, p=2, dim=-1)
        ctx_u = F.normalize(ctx, p=2, dim=-1)
        cos_gat = torch.sum(h_gat_u * ctx_u, dim=-1, keepdim=True)
        deg_factor = (1.0 / torch.sqrt(1.0 + degrees.float())).unsqueeze(-1)

        gate_feat = torch.cat([h * ctx, h_gat * ctx, cos_gat, r_anchor, deg_factor], dim=-1)
        lambda_gate = torch.sigmoid(self.gate_mlp(gate_feat)).squeeze(-1)

        # Final Score
        final_score = s_vis + lambda_gate * s_graph_delta
        return final_score, s_vis, lambda_gate

def evaluate_on_test_full(model, test_clip_records, test_contrastive_pairs):
    model.eval()
    flat_scores = {}
    ndcg5, p5, aucs = [], [], []
    seen_ndcg, unseen_ndcg = [], []
    SEEN_LANDMARKS = {\"hoan_kiem_lake\", \"temple_of_literature\", \"ba_dinh_square\", \"imperial_citadel_of_thang_long\", \"one_pillar_pagoda\", \"hanoi_old_quarter\", \"west_lake\"}

    with torch.no_grad():
        for c_id, recs in test_clip_records.items():
            l_id = recs[0].get(\"landmark_id\", \"unknown\")
            g = rel_subgraphs.get(l_id)
            v_frames = feature_cache.video_frames.get(c_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if g is None or v_frames is None or v_pooled is None: continue

            sub_fids = g[\"fact_ids\"]
            h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device)
            adj = g[\"adj\"].to(device)
            rel_adj = g[\"rel_adj\"].to(device)
            degrees = g[\"degrees\"].to(device)

            logits, _, _ = model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            scores_np = logits.cpu().numpy()

            idx_map = g[\"fid2idx\"]
            f_scores = {}
            gold_fids = set(r[\"fact_id\"] for r in recs if r.get(\"is_scene_gold\"))
            hard_neg_fids = set(r[\"fact_id\"] for r in recs if r.get(\"gold_label\") == \"HARD_NEGATIVE\")
            cand_fids = [r[\"fact_id\"] for r in recs]

            for fid in cand_fids:
                s_val = float(scores_np[idx_map[fid]]) if fid in idx_map else 0.0
                f_scores[fid] = s_val
                flat_scores[(c_id, fid)] = s_val

            ranked_fids = sorted(cand_fids, key=lambda fid: f_scores[fid], reverse=True)
            m = compute_ranking_metrics(ranked_fids, gold_fids, cand_fids, top_k_list=[5])
            c_ndcg = m.get(\"nDCG@5\", 0.0)
            ndcg5.append(c_ndcg)
            p5.append(m.get(\"P@5\", 0.0))

            auc = compute_hard_negative_auc(f_scores, gold_fids, hard_neg_fids)
            if auc is not None: aucs.append(auc)

            if l_id in SEEN_LANDMARKS: seen_ndcg.append(c_ndcg)
            else: unseen_ndcg.append(c_ndcg)

    cont_acc = compute_contrastive_accuracy(flat_scores, test_contrastive_pairs)
    return {
        \"ndcg@5\": float(np.mean(ndcg5)),
        \"p@5\": float(np.mean(p5)),
        \"hard_neg_auc\": float(np.mean(aucs)),
        \"contrastive_acc\": float(cont_acc),
        \"seen_ndcg@5\": float(np.mean(seen_ndcg)),
        \"unseen_ndcg@5\": float(np.mean(unseen_ndcg))
    }

# Train model
model = FullUnifiedRVCGAT(d_model=768).to(device)
optimizer = AdamW(model.parameters(), lr=1.2e-4, weight_decay=1e-4)
train_cids = [c for c in train_clips.keys() if any(r.get(\"is_scene_gold\") for r in train_clips[c])]

best_val_ndcg = -1.0
best_state = None

print(\"Training FullUnifiedRVCGAT (Cross-Attention Visual Backbone + Relation-Aware Residual Graph)...\")
for epoch in range(1, 16):
    model.train()
    np.random.shuffle(train_cids)
    for c_id in train_cids:
        recs = train_clips[c_id]
        l_id = recs[0].get(\"landmark_id\", \"unknown\")
        g = rel_subgraphs.get(l_id)
        v_frames = feature_cache.video_frames.get(c_id)
        v_pooled = feature_cache.video_pooled.get(c_id)
        if g is None or v_frames is None or v_pooled is None: continue

        pos_fids = set(r[\"fact_id\"] for r in recs if r.get(\"is_scene_gold\"))
        hard_neg_fids = set(r[\"fact_id\"] for r in recs if r.get(\"gold_label\") == \"HARD_NEGATIVE\")
        all_neg_fids = set(r[\"fact_id\"] for r in recs if not r.get(\"is_scene_gold\"))
        if not pos_fids or not all_neg_fids: continue

        optimizer.zero_grad()
        sub_fids = g[\"fact_ids\"]
        h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
        h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
        v_f_tensor = torch.from_numpy(v_frames).float().to(device)
        v_p_tensor = torch.from_numpy(v_pooled).float().to(device)
        adj = g[\"adj\"].to(device)
        rel_adj = g[\"rel_adj\"].to(device)
        degrees = g[\"degrees\"].to(device)

        logits, _, _ = model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
        idx_map = g[\"fid2idx\"]
        pos_idx = [idx_map[fid] for fid in pos_fids if fid in idx_map]
        hard_neg_idx = [idx_map[fid] for fid in hard_neg_fids if fid in idx_map]
        neg_idx = [idx_map[fid] for fid in list(all_neg_fids)[:len(pos_fids)*6] if fid in idx_map]

        if not pos_idx or not neg_idx: continue

        # BCE
        y_true = [1.0]*len(pos_idx) + [0.0]*len(neg_idx)
        preds = torch.cat([logits[pos_idx], logits[neg_idx]])
        targets = torch.tensor(y_true, dtype=torch.float32, device=device)
        loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([4.0], device=device))

        # Top Ranking loss (margin 0.6)
        p_scores = logits[pos_idx].unsqueeze(1)
        n_scores = logits[neg_idx].unsqueeze(0)
        rank_loss = torch.relu(0.6 - (p_scores - n_scores)).mean()
        loss = loss + 0.8 * rank_loss

        # Scene Hard Negative margin (margin 0.8)
        if hard_neg_idx:
            hn_scores = logits[hard_neg_idx].unsqueeze(0)
            hn_loss = torch.relu(0.8 - (p_scores - hn_scores)).mean()
            loss = loss + 0.7 * hn_loss

        loss.backward()
        optimizer.step()

    # Contrastive
    if train_contrastive:
        np.random.shuffle(train_contrastive)
        for p in train_contrastive[:150]:
            c_a = p.get(\"clip_a\") or p.get(\"clip_A\", {}).get(\"clip_id\")
            c_b = p.get(\"clip_b\") or p.get(\"clip_B\", {}).get(\"clip_id\")
            f_id = p.get(\"fact_id\") or p.get(\"fact_a\")
            l_id = p.get(\"landmark_id\")

            v_fa = feature_cache.video_frames.get(c_a)
            v_fb = feature_cache.video_frames.get(c_b)
            v_pa = feature_cache.video_pooled.get(c_a)
            v_pb = feature_cache.video_pooled.get(c_b)
            g = rel_subgraphs.get(l_id)
            if v_fa is None or v_fb is None or g is None or f_id not in g[\"fid2idx\"]:
                continue

            f_idx = g[\"fid2idx\"][f_id]
            sub_fids = g[\"fact_ids\"]
            h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            adj = g[\"adj\"].to(device)
            rel_adj = g[\"rel_adj\"].to(device)
            degrees = g[\"degrees\"].to(device)

            optimizer.zero_grad()
            s_a, _, _ = model(h_tensor, adj, rel_adj, degrees, torch.from_numpy(v_fa).float().to(device), torch.from_numpy(v_pa).float().to(device))
            s_b, _, _ = model(h_tensor, adj, rel_adj, degrees, torch.from_numpy(v_fb).float().to(device), torch.from_numpy(v_pb).float().to(device))

            cont_loss = torch.relu(0.8 - (s_a[f_idx] - s_b[f_idx]))
            if cont_loss > 0:
                cont_loss.backward()
                optimizer.step()

    # Eval on Val
    model.eval()
    val_ndcg_list = []
    val_auc_list = []
    with torch.no_grad():
        for c_id, recs in val_clips.items():
            l_id = recs[0].get(\"landmark_id\", \"unknown\")
            g = rel_subgraphs.get(l_id)
            v_frames = feature_cache.video_frames.get(c_id)
            v_pooled = feature_cache.video_pooled.get(c_id)
            if g is None or v_frames is None or v_pooled is None: continue

            sub_fids = g[\"fact_ids\"]
            h_list = [feature_cache.fact_embeddings.get(fid, np.zeros(768, dtype=np.float32)) for fid in sub_fids]
            h_tensor = torch.from_numpy(np.array(h_list, dtype=np.float32)).to(device)
            v_f_tensor = torch.from_numpy(v_frames).float().to(device)
            v_p_tensor = torch.from_numpy(v_pooled).float().to(device)
            adj = g[\"adj\"].to(device)
            rel_adj = g[\"rel_adj\"].to(device)
            degrees = g[\"degrees\"].to(device)

            logits, _, _ = model(h_tensor, adj, rel_adj, degrees, v_f_tensor, v_p_tensor)
            scores_np = logits.cpu().numpy()
            idx_map = g[\"fid2idx\"]
            gold_fids = set(r[\"fact_id\"] for r in recs if r.get(\"is_scene_gold\"))
            hard_neg_fids = set(r[\"fact_id\"] for r in recs if r.get(\"gold_label\") == \"HARD_NEGATIVE\")
            cand_fids = [r[\"fact_id\"] for r in recs]
            f_scores = {fid: float(scores_np[idx_map[fid]]) if fid in idx_map else 0.0 for fid in cand_fids}

            ranked_fids = sorted(cand_fids, key=lambda fid: f_scores[fid], reverse=True)
            m = compute_ranking_metrics(ranked_fids, gold_fids, cand_fids, top_k_list=[5])
            val_ndcg_list.append(m.get(\"nDCG@5\", 0.0))
            auc = compute_hard_negative_auc(f_scores, gold_fids, hard_neg_fids)
            if auc is not None: val_auc_list.append(auc)

    val_ndcg = float(np.mean(val_ndcg_list))
    val_auc = float(np.mean(val_auc_list))
    print(f\"Epoch {epoch:02d} | Val nDCG@5: {val_ndcg:.4f} | Val AUC: {val_auc:.4f}\")

    if val_ndcg > best_val_ndcg:
        best_val_ndcg = val_ndcg
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
test_res = evaluate_on_test_full(model, test_clips, test_contrastive)
print(\"\n\" + \"=\"*75)
print(\"DEFINITIVE TEST RESULTS ON 800 TEST CLIPS:\")
print(f\"  Overall nDCG@5      = {test_res['ndcg@5']:.4f}\")
print(f\"  Overall P@5         = {test_res['p@5']:.4f}\")
print(f\"  Hard Negative AUC   = {test_res['hard_neg_auc']:.4f}\")
print(f\"  Contrastive Acc     = {test_res['contrastive_acc']:.1f}%\")
print(f\"  Seen nDCG@5         = {test_res['seen_ndcg@5']:.4f}\")
print(f\"  Unseen nDCG@5       = {test_res['unseen_ndcg@5']:.4f}\")
print(\"=\"*75)