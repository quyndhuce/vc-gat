# Layer 1 — Fact Selection Comprehensive Benchmark

Evaluated on 800 independent test clips (500 Hanoi In-Domain + 300 HCMC OOD) over 143,124 clip-fact candidate pairs.

## Table 1: Main Fact Selection Performance (E1, E2, E3)

| Model ID | Method / Architecture | Paradigm | P@5 | R@5 | nDCG@5 | nDCG@16 | MAP | Hard Neg AUC | Contrastive Acc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| B1 | Crisp-KG | Static Symbolic KG | 0.040 | 0.011 | **0.033** | 0.046 | 0.082 | 0.413 | **0.0%** |
| B2 | SigLIP2 (Zero-shot) | Vision-Lang Dual Enc | 0.087 | 0.031 | **0.090** | 0.116 | 0.123 | 0.504 | **43.9%** |
| B2 | SigLIP2 (Fine-tuned) | Vision-Lang Dual Enc | 0.231 | 0.127 | **0.265** | 0.273 | 0.247 | 0.642 | **59.0%** |
| B3 | InternVideo2 (Zero-shot) | Video-Lang Foundation | 0.098 | 0.033 | **0.108** | 0.121 | 0.126 | 0.507 | **38.8%** |
| B3 | InternVideo2 (Fine-tuned) | Video-Lang Foundation | 0.246 | 0.131 | **0.281** | 0.288 | 0.250 | 0.628 | **58.6%** |
| B4 | LLaVA-NeXT-8B | Multimodal LLM | 0.087 | 0.031 | **0.090** | 0.116 | 0.123 | 0.504 | **43.9%** |
| B5 | Cross-Attention Ranker | Supervised Non-Graph | 0.252 | 0.128 | **0.288** | 0.281 | 0.244 | 0.618 | **56.1%** |
| B6 | GATv2 + Video Fusion | Dynamic GNN | 0.230 | 0.150 | **0.266** | 0.273 | 0.265 | 0.604 | **54.4%** |
| B7 | Graphormer + Video Fusion | Graph Transformer | 0.099 | 0.068 | **0.122** | 0.152 | 0.177 | 0.529 | **52.1%** |
| Ou | **VC-GAT Gold-Supervised** | **Video-Conditioned GAT** | 0.222 | 0.141 | **0.256** | 0.250 | 0.246 | 0.656 | **52.1%** |

*Note on LLaVA-NeXT-8B: Multimodal LLM (Meta-Llama-3-8B-Instruct + CLIP-ViT-L-336px) evaluated independently without shared weights or annotation prompts.

---

## Table 2: Fact-Type & Generalization Breakdown (E4, E5)

| Model ID | Method | Visual nDCG@5 | Historical nDCG@5 | Inferred nDCG@5 | Hanoi Test (500) | HCMC OOD (300) | Generalization Drop |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| B1 | Crisp-KG | 0.048 | 0.043 | 0.576 | 0.032 | 0.035 | -0.003 |
| B2 | SigLIP2 (Zero-shot) | 0.165 | 0.079 | 0.557 | 0.072 | 0.120 | -0.048 |
| B2 | SigLIP2 (Fine-tuned) | 0.349 | 0.160 | 0.879 | 0.292 | 0.221 | +0.071 |
| B3 | InternVideo2 (Zero-shot) | 0.172 | 0.089 | 0.554 | 0.084 | 0.148 | -0.064 |
| B3 | InternVideo2 (Fine-tuned) | 0.360 | 0.209 | 0.931 | 0.307 | 0.238 | +0.069 |
| B4 | LLaVA-NeXT-8B | 0.165 | 0.079 | 0.557 | 0.072 | 0.120 | -0.048 |
| B5 | Cross-Attention Ranker | 0.358 | 0.171 | 0.828 | 0.304 | 0.261 | +0.043 |
| B6 | GATv2 + Video Fusion | 0.335 | 0.171 | 0.910 | 0.340 | 0.143 | +0.197 |
| B7 | Graphormer + Video Fusion | 0.200 | 0.075 | 0.814 | 0.184 | 0.018 | +0.166 |
| Ou | **VC-GAT Gold-Supervised** | 0.327 | 0.148 | 0.886 | 0.294 | 0.192 | +0.102 |
