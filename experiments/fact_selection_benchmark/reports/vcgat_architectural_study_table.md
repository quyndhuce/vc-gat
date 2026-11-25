# VC-GAT Architectural Progression Study (Simpler to Complex)
Evaluated on 800 independent test clips (301 Seen Hanoi, 499 Unseen Hanoi+HCMC) over 143,124 candidate pairs and 9,040 contrastive pairs.
## Table 1: Progressive Performance Comparison
| Level | Variant / Architectural Paradigm | nDCG@5 ↑ | P@5 ↑ | Hard Neg AUC ↑ | Contrastive Acc (%) ↑ | Seen nDCG@5 | Unseen nDCG@5 | Generalization Gap |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| L1 | **Base: Direct Visual Matching (No Graph)** | **0.1131** | 0.1173 | **0.6250** | **47.5%** | 0.0748 | 0.1280 | -0.0533 |
| L2 | **Naive VC-GAT (Raw OntoKG Subgraph)** | **0.2657** | 0.2308 | **0.6314** | **55.2%** | 0.3974 | 0.2145 | +0.1829 |
| L3 | **Pruned VC-GAT + Saliency Gating** | **0.2450** | 0.2172 | **0.6508** | **54.5%** | 0.3313 | 0.2115 | +0.1198 |
| L4 | **Residual-Gated VC-GAT (Visual Lower-Bound)** | **0.2382** | 0.2148 | **0.6401** | **51.3%** | 0.3218 | 0.2056 | +0.1162 |
| L5 | **Multi-Scale Temporal Residual VC-GAT** | **0.2217** | 0.1960 | **0.6515** | **59.1%** | 0.3180 | 0.1842 | +0.1338 |

## Table 2: Fact-Type Ranking Breakdown (nDCG@5)
| Level | Variant | Visual Facts | Historical Facts | Inferred Facts |
| :---: | :--- | :---: | :---: | :---: |
| L1 | Base: Direct Visual Matching (No Graph) | 0.1888 | 0.0910 | 0.7093 |
| L2 | Naive VC-GAT (Raw OntoKG Subgraph) | 0.3469 | 0.1553 | 0.9192 |
| L3 | Pruned VC-GAT + Saliency Gating | 0.3455 | 0.1091 | 0.8125 |
| L4 | Residual-Gated VC-GAT (Visual Lower-Bound) | 0.3241 | 0.1144 | 0.7916 |
| L5 | Multi-Scale Temporal Residual VC-GAT | 0.2864 | 0.1162 | 0.8033 |
