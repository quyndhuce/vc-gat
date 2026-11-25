# Definitive IJAR Fact Selection Benchmark & Controlled Ablation Table
Evaluated on 800 independent test clips (301 Seen Hanoi, 499 Unseen Hanoi+HCMC) over 143,124 candidate pairs and 9,040 contrastive pairs.

## Table 1: Main Comparison vs. Competitive External Baselines
| Model / Architecture | Overall nDCG@5 ↑ | P@5 ↑ | Hard Neg AUC ↑ | Contrastive Acc (%) ↑ | Seen nDCG@5 | Unseen nDCG@5 | Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Cross-Attention Ranker (Non-Graph) | 0.2882 | 0.2520 | 0.6180 | 56.1% | 0.4196 | 0.2083 | +0.2113 |
| InternVideo2 (Fine-tuned) | 0.2811 | 0.2463 | 0.6284 | 58.6% | 0.4182 | 0.1983 | +0.2199 |
| SigLIP2 (Fine-tuned) | 0.2654 | 0.2312 | 0.6421 | 59.0% | 0.4039 | 0.1820 | +0.2219 |
| GATv2 + Video Fusion (Raw KG) | 0.2661 | 0.2304 | 0.6042 | 54.4% | 0.3402 | 0.1431 | +0.1971 |
| Graphormer + Video Fusion | 0.1221 | 0.0988 | 0.5287 | 52.1% | 0.1504 | 0.1050 | +0.0454 |
| Crisp-KG (Static Symbolic) | 0.0331 | 0.0402 | 0.4128 | 0.0% | 0.0410 | 0.0283 | +0.0127 |
| **R-VCGAT (Proposed Full)** | **0.2771** | **0.2395** | **0.6300** | **45.1%** | **0.4567** | **0.2072** | **+0.2495** |

## Table 2: Controlled Component Ablation Study
| Configuration | Overall nDCG@5 | Delta nDCG | Hard Neg AUC | Delta AUC | Contrastive Acc | Delta Cont | Unseen nDCG@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| R_VCGAT_Proposed_Full | 0.2771 | +0.0000 | 0.6300 | +0.0000 | 45.1% | +0.0% | 0.2072 |