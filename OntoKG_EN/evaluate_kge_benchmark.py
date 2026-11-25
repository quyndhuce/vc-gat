"""
Systematic Link Prediction Benchmark for Fuzzy Knowledge Graph Embeddings.
Evaluates Standard TransE vs. FTransE (Fuzzy-Weighted TransE) on 90/10 Transductive Masking.

Protocol:
  1. 10% edges randomly masked as Test Set; 90% retained as Training Set.
  2. Filtered Ranking (Bordes et al., 2013) on both Head & Tail prediction.
  3. Metrics: Hits@1, Hits@3, Hits@10, Mean Reciprocal Rank (MRR), Mean Rank (MR).
"""

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("KGEBenchmark")


# ══════════════════════════════════════════════════════════
# 1. DATASET & TRANSDUCTIVE MASKING (90% TRAIN / 10% TEST)
# ══════════════════════════════════════════════════════════

def load_and_split_fkg(
    json_path: Path = Path("data/fkg_per_kb/fkg_global_merged.json"),
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[
    List[Tuple[int, int, int, float]],  # train_triples
    List[Tuple[int, int, int, float]],  # test_triples
    Dict[str, int],                     # entity_to_id
    Dict[int, str],                     # id_to_entity
    Dict[str, int],                     # relation_to_id
    Dict[int, str],                     # id_to_relation
    Set[Tuple[int, int, int]],          # all_true_triples (for filtered ranking)
    Dict[str, str],                     # surface_map
]:
    random.seed(seed)
    torch.manual_seed(seed)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    surface_map: Dict[str, str] = {}
    raw_triples: List[Tuple[str, str, str, float]] = []

    # 1. Single-valued winners
    for g in data.get("single_valued_groups", []):
        h = g.get("head_uri")
        r = g.get("relation")
        w = g.get("winner_uri")
        bel = float(g.get("winner_bel", 0.8))
        if h and r and w:
            raw_triples.append((h, r, w, max(0.1, min(1.0, bel))))
            if g.get("head_surface"):
                surface_map[h] = g["head_surface"]
            if g.get("winner_surface"):
                surface_map[w] = g["winner_surface"]

    # 2. Multi-valued facts
    for m in data.get("multi_valued_facts", []):
        h = m.get("head_uri")
        r = m.get("relation")
        t = m.get("tail_uri")
        mu = float(m.get("mu_F", 0.5))
        if h and r and t:
            raw_triples.append((h, r, t, max(0.1, min(1.0, mu))))
            if m.get("head_surface"):
                surface_map[h] = m["head_surface"]
            if m.get("tail_surface"):
                surface_map[t] = m["tail_surface"]

    # Deduplicate (keep max weight if multiple)
    dedup_dict: Dict[Tuple[str, str, str], float] = {}
    for h, r, t, w in raw_triples:
        k = (h, r, t)
        if k not in dedup_dict or w > dedup_dict[k]:
            dedup_dict[k] = w

    unique_triples = [(h, r, t, w) for (h, r, t), w in dedup_dict.items()]
    random.shuffle(unique_triples)

    # Build vocabularies
    entities = sorted(list(set(h for h, _, _, _ in unique_triples) | set(t for _, _, t, _ in unique_triples)))
    relations = sorted(list(set(r for _, r, _, _ in unique_triples)))

    ent2id = {e: i for i, e in enumerate(entities)}
    id2ent = {i: e for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}
    id2rel = {i: r for i, r in enumerate(relations)}

    # Transductive split: ensure all entities appear in training set at least once
    ent_freq = defaultdict(int)
    for h, r, t, _ in unique_triples:
        ent_freq[h] += 1
        ent_freq[t] += 1

    train_raw = []
    candidate_test = []

    for h, r, t, w in unique_triples:
        # If entity only appears once, it MUST go to train set
        if ent_freq[h] <= 1 or ent_freq[t] <= 1:
            train_raw.append((h, r, t, w))
        else:
            candidate_test.append((h, r, t, w))

    # Sample test set from candidates
    target_test_size = int(len(unique_triples) * test_ratio)
    test_raw = candidate_test[:target_test_size]
    train_raw.extend(candidate_test[target_test_size:])

    # Convert to IDs
    train_triples = [(ent2id[h], rel2id[r], ent2id[t], w) for h, r, t, w in train_raw]
    test_triples = [(ent2id[h], rel2id[r], ent2id[t], w) for h, r, t, w in test_raw]

    all_true_triples = {(h, r, t) for h, r, t, _ in train_triples} | {(h, r, t) for h, r, t, _ in test_triples}

    LOGGER.info(
        "Dataset Split: Total=%d | Train=%d (%.1f%%) | Test (Masked)=%d (%.1f%%)",
        len(unique_triples), len(train_triples), len(train_triples)/len(unique_triples)*100,
        len(test_triples), len(test_triples)/len(unique_triples)*100
    )
    LOGGER.info("Vocabulary: %d Entities, %d Relations", len(ent2id), len(rel2id))

    return (
        train_triples, test_triples,
        ent2id, id2ent, rel2id, id2rel,
        all_true_triples, surface_map
    )


class TrainDataset(Dataset):
    def __init__(self, triples: List[Tuple[int, int, int, float]], num_entities: int, num_neg: int = 5):
        self.triples = triples
        self.num_entities = num_entities
        self.num_neg = num_neg
        self.existing_edges = {(h, r, t) for h, r, t, _ in triples}

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        h, r, t, w = self.triples[idx]
        neg_tails = []
        for _ in range(self.num_neg):
            while True:
                corrupt_t = random.randint(0, self.num_entities - 1)
                if (h, r, corrupt_t) not in self.existing_edges or corrupt_t == t:
                    neg_tails.append(corrupt_t)
                    break
        return (
            torch.tensor(h, dtype=torch.long),
            torch.tensor(r, dtype=torch.long),
            torch.tensor(t, dtype=torch.long),
            torch.tensor(w, dtype=torch.float),
            torch.tensor(neg_tails, dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════
# 2. MODEL DEFINITION: FTransE / TransE
# ══════════════════════════════════════════════════════════

class TransEModule(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int = 128, margin: float = 6.0):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.margin = margin

        self.ent_embeddings = nn.Embedding(num_entities, dim)
        self.rel_embeddings = nn.Embedding(num_relations, dim)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, p=2, dim=-1)

    def normalize_entities(self):
        with torch.no_grad():
            self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, p=2, dim=-1)

    def forward(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        weights: torch.Tensor,
        neg_t: torch.Tensor,
        use_fuzzy_weight: bool = True,
    ) -> torch.Tensor:
        h_vec = self.ent_embeddings(h)
        r_vec = self.rel_embeddings(r)
        t_vec = self.ent_embeddings(t)

        pos_dist = torch.norm(h_vec + r_vec - t_vec, p=2, dim=-1)

        B, K = neg_t.shape
        h_exp = h_vec.unsqueeze(1).expand(-1, K, -1)
        r_exp = r_vec.unsqueeze(1).expand(-1, K, -1)
        neg_t_vec = self.ent_embeddings(neg_t)
        neg_dist = torch.norm(h_exp + r_exp - neg_t_vec, p=2, dim=-1)

        pos_loss = F.softplus(pos_dist - self.margin)
        neg_loss = F.softplus(self.margin - neg_dist).mean(dim=-1)

        if use_fuzzy_weight:
            return (weights * (pos_loss + neg_loss)).mean()
        else:
            return (pos_loss + neg_loss).mean()


# ══════════════════════════════════════════════════════════
# 3. TRAINING ROUTINE
# ══════════════════════════════════════════════════════════

def train_model(
    model: TransEModule,
    train_triples: List[Tuple[int, int, int, float]],
    num_entities: int,
    epochs: int = 300,
    batch_size: int = 512,
    lr: float = 0.005,
    use_fuzzy_weight: bool = True,
    device: str = "cuda",
) -> TransEModule:
    dev = torch.device(device)
    model = model.to(dev)

    dataset = TrainDataset(train_triples, num_entities, num_neg=5)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    name = "FTransE (Fuzzy-Weighted)" if use_fuzzy_weight else "Standard TransE (Unweighted)"
    LOGGER.info("Starting training for %s (%d epochs)...", name, epochs)

    t0 = time.time()
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for h, r, t, w, neg_t in dataloader:
            h, r, t, w, neg_t = h.to(dev), r.to(dev), t.to(dev), w.to(dev), neg_t.to(dev)

            optimizer.zero_grad()
            loss = model(h, r, t, w, neg_t, use_fuzzy_weight=use_fuzzy_weight)
            loss.backward()
            optimizer.step()

            model.normalize_entities()
            total_loss += loss.item() * len(h)

        scheduler.step()
        if ep % 100 == 0 or ep == epochs:
            LOGGER.info("  [%s] Epoch %3d/%3d — Loss: %.5f", name, ep, epochs, total_loss / len(train_triples))

    LOGGER.info("✓ %s training complete in %.2fs", name, time.time() - t0)
    return model


# ══════════════════════════════════════════════════════════
# 4. FILTERED LINK PREDICTION BENCHMARK
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_link_prediction(
    model: TransEModule,
    test_triples: List[Tuple[int, int, int, float]],
    all_true_triples: Set[Tuple[int, int, int]],
    device: str = "cuda",
) -> Dict[str, float]:
    """Standard Filtered Link Prediction protocol (Bordes et al. 2013).

    For each test triple (h, r, t):
      1. Tail prediction: Rank all candidates (h, r, t'). Filter out other known true triples.
      2. Head prediction: Rank all candidates (h', r, t). Filter out other known true triples.
    """
    dev = torch.device(device)
    model.eval()

    all_ent_vecs = model.ent_embeddings.weight.data  # [N, D]
    num_entities = all_ent_vecs.size(0)

    # Pre-build true triple lookup for fast filtering
    # (h, r) -> set of true tails
    hr_to_true_tails = defaultdict(set)
    # (r, t) -> set of true heads
    rt_to_true_heads = defaultdict(set)
    for (h, r, t) in all_true_triples:
        hr_to_true_tails[(h, r)].add(t)
        rt_to_true_heads[(r, t)].add(h)

    ranks_raw = []
    ranks_filtered = []

    # Batch test triples for GPU matrix multiplication
    batch_size = 128
    for i in range(0, len(test_triples), batch_size):
        chunk = test_triples[i:i + batch_size]

        h_idx = torch.tensor([h for h, _, _, _ in chunk], dtype=torch.long, device=dev)
        r_idx = torch.tensor([r for _, r, _, _ in chunk], dtype=torch.long, device=dev)
        t_idx = torch.tensor([t for _, _, t, _ in chunk], dtype=torch.long, device=dev)

        h_vecs = model.ent_embeddings(h_idx)  # [B, D]
        r_vecs = model.rel_embeddings(r_idx)  # [B, D]
        t_vecs = model.ent_embeddings(t_idx)  # [B, D]

        # ── 1. Tail Prediction: score(h, r, ?) -> target = h + r
        targets_t = h_vecs + r_vecs  # [B, D]
        # Distances to all entities: ||target - all_e||_2: [B, N]
        # Using ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
        dists_t = torch.cdist(targets_t, all_ent_vecs, p=2)  # [B, N]

        # ── 2. Head Prediction: score(?, r, t) -> target = t - r
        targets_h = t_vecs - r_vecs  # [B, D]
        dists_h = torch.cdist(targets_h, all_ent_vecs, p=2)  # [B, N]

        dists_t_cpu = dists_t.cpu().numpy()
        dists_h_cpu = dists_h.cpu().numpy()

        for b_idx, (h, r, t, _) in enumerate(chunk):
            # ── Process Tail Prediction ─────────────────────
            scores_t = dists_t_cpu[b_idx]
            target_score_t = scores_t[t]

            # Raw rank
            raw_rank_t = 1 + (scores_t < target_score_t).sum()
            ranks_raw.append(raw_rank_t)

            # Filtered rank: ignore other known true tails
            known_tails = hr_to_true_tails[(h, r)]
            filtered_better = 0
            for cand_id in range(num_entities):
                if cand_id != t and scores_t[cand_id] < target_score_t:
                    if cand_id not in known_tails:
                        filtered_better += 1
            ranks_filtered.append(1 + filtered_better)

            # ── Process Head Prediction ─────────────────────
            scores_h = dists_h_cpu[b_idx]
            target_score_h = scores_h[h]

            # Raw rank
            raw_rank_h = 1 + (scores_h < target_score_h).sum()
            ranks_raw.append(raw_rank_h)

            # Filtered rank: ignore other known true heads
            known_heads = rt_to_true_heads[(r, t)]
            filtered_better_h = 0
            for cand_id in range(num_entities):
                if cand_id != h and scores_h[cand_id] < target_score_h:
                    if cand_id not in known_heads:
                        filtered_better_h += 1
            ranks_filtered.append(1 + filtered_better_h)

    # Compute Standard Metrics
    mrr_filt = sum(1.0 / r for r in ranks_filtered) / len(ranks_filtered)
    hits1_filt = sum(1 for r in ranks_filtered if r <= 1) / len(ranks_filtered) * 100.0
    hits3_filt = sum(1 for r in ranks_filtered if r <= 3) / len(ranks_filtered) * 100.0
    hits10_filt = sum(1 for r in ranks_filtered if r <= 10) / len(ranks_filtered) * 100.0
    mr_filt = sum(ranks_filtered) / len(ranks_filtered)

    mrr_raw = sum(1.0 / r for r in ranks_raw) / len(ranks_raw)
    hits10_raw = sum(1 for r in ranks_raw if r <= 10) / len(ranks_raw) * 100.0
    mr_raw = sum(ranks_raw) / len(ranks_raw)

    return {
        "num_test_queries": len(ranks_filtered),
        "filtered_mrr": round(mrr_filt, 4),
        "filtered_hits@1": round(hits1_filt, 2),
        "filtered_hits@3": round(hits3_filt, 2),
        "filtered_hits@10": round(hits10_filt, 2),
        "filtered_mr": round(mr_filt, 1),
        "raw_mrr": round(mrr_raw, 4),
        "raw_hits@10": round(hits10_raw, 2),
        "raw_mr": round(mr_raw, 1),
    }


# ══════════════════════════════════════════════════════════
# 5. MAIN BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("════════════════════════════════════════════════════════════")
    LOGGER.info("  SYSTEMATIC KGE LINK PREDICTION BENCHMARK (10%% MASKED)   ")
    LOGGER.info("  Device: %s                                                ", device)
    LOGGER.info("════════════════════════════════════════════════════════════")

    # 1. Load & Split Data (90/10 Transductive Masking)
    (
        train_triples, test_triples,
        ent2id, id2ent, rel2id, id2rel,
        all_true_triples, surface_map
    ) = load_and_split_fkg(test_ratio=0.10, seed=42)

    num_entities = len(ent2id)
    num_relations = len(rel2id)

    # 2. Train Model A: Standard TransE (Unweighted baseline)
    model_unweighted = TransEModule(num_entities, num_relations, dim=128, margin=6.0)
    train_model(
        model_unweighted, train_triples, num_entities,
        epochs=300, batch_size=512, lr=0.005,
        use_fuzzy_weight=False, device=device
    )
    metrics_unweighted = evaluate_link_prediction(
        model_unweighted, test_triples, all_true_triples, device=device
    )

    # 3. Train Model B: FTransE (Fuzzy-Weighted TransE with μ_F & Bel)
    model_fuzzy = TransEModule(num_entities, num_relations, dim=128, margin=6.0)
    train_model(
        model_fuzzy, train_triples, num_entities,
        epochs=300, batch_size=512, lr=0.005,
        use_fuzzy_weight=True, device=device
    )
    metrics_fuzzy = evaluate_link_prediction(
        model_fuzzy, test_triples, all_true_triples, device=device
    )

    # 4. Print Comparative Results Table
    print("\n" + "═" * 80)
    print("  🏆 KGE BENCHMARK RESULTS — 10% RANDOM MASKING (FILTERED LINK PREDICTION)")
    print("═" * 80)
    print(f"Total Triples in Graph : {len(all_true_triples)}")
    print(f"Training Set (90%)     : {len(train_triples)} edges")
    print(f"Masked Test Set (10%)  : {len(test_triples)} edges ({metrics_fuzzy['num_test_queries']} total evaluation queries)")
    print(f"Vocabulary Size        : {num_entities} Entities | {num_relations} Relations")
    print("─" * 80)
    header = f"{'Metric (Filtered)':<25} | {'Standard TransE':<18} | {'FTransE (Fuzzy μ_F)':<20} | {'Gain / Impact':<15}"
    print(header)
    print("─" * 80)

    rows = [
        ("Hits@10 (%)", metrics_unweighted["filtered_hits@10"], metrics_fuzzy["filtered_hits@10"], "%"),
        ("Hits@3 (%)", metrics_unweighted["filtered_hits@3"], metrics_fuzzy["filtered_hits@3"], "%"),
        ("Hits@1 (%)", metrics_unweighted["filtered_hits@1"], metrics_fuzzy["filtered_hits@1"], "%"),
        ("MRR (Mean Recip. Rank)", metrics_unweighted["filtered_mrr"], metrics_fuzzy["filtered_mrr"], ""),
        ("Mean Rank (MR)", metrics_unweighted["filtered_mr"], metrics_fuzzy["filtered_mr"], " (lower is better)"),
        ("Raw Hits@10 (%)", metrics_unweighted["raw_hits@10"], metrics_fuzzy["raw_hits@10"], "%"),
        ("Raw MRR", metrics_unweighted["raw_mrr"], metrics_fuzzy["raw_mrr"], ""),
    ]

    for label, val_std, val_fuz, unit in rows:
        if "lower is better" in unit:
            diff = val_std - val_fuz
            diff_str = f"-{diff:.1f} ranks" if diff >= 0 else f"+{-diff:.1f} ranks"
        else:
            diff = val_fuz - val_std
            diff_str = f"+{diff:.2f}{unit}" if diff >= 0 else f"{diff:.2f}{unit}"
        print(f"{label:<25} | {str(val_std) + unit:<18} | {str(val_fuz) + unit:<20} | {diff_str:<15}")

    print("═" * 80)

    # Save benchmark report to JSON
    report = {
        "dataset": {
            "total_triples": len(all_true_triples),
            "train_triples": len(train_triples),
            "test_triples": len(test_triples),
            "num_entities": num_entities,
            "num_relations": num_relations,
        },
        "standard_transe": metrics_unweighted,
        "ftranse_fuzzy": metrics_fuzzy,
    }
    out_file = Path("data/fkg_per_kb/kge_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved benchmark report to %s", out_file)


if __name__ == "__main__":
    run_benchmark()
