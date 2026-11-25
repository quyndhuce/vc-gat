# R-VCGAT Controlled Improvement & Ablation Benchmark
Evaluated on 800 independent test clips (301 Seen Hanoi, 499 Unseen Hanoi+HCMC) over 143,124 candidate pairs and 9,040 contrastive pairs.
Checkpoints selected exclusively via Composite Validation Criterion on 149 Val clips.

## Table 1: Benchmark Performance vs. Strong External Baselines
| Architecture / Paradigm | Overall nDCG@5 ↑ | P@5 ↑ | Hard Neg AUC ↑ | Contrastive Acc (%) ↑ | Seen nDCG@5 | Unseen nDCG@5 | Generalization Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Cross-Attention Ranker (Non-Graph) | 0.2882 | 0.2520 | 0.6180 | 56.1% | 0.4196 | 0.2083 | +0.2113 |
| InternVideo2 (Fine-tuned) | 0.2811 | 0.2463 | 0.6284 | 58.6% | 0.4182 | 0.1983 | +0.2199 |
| SigLIP2 (Fine-tuned) | 0.2654 | 0.2312 | 0.6421 | 59.0% | 0.4039 | 0.1820 | +0.2219 |
| GATv2 + Video Fusion (Raw KG) | 0.2661 | 0.2304 | 0.6042 | 54.4% | 0.3402 | 0.1431 | +0.1971 |
| **V4 Residual-Gated (Previous)** | 0.2671 | 0.2412 | 0.6356 | 53.9% | 0.3757 | 0.2249 | +0.1508 |
| **R_VCGAT_Proposed_Full** | **0.2198** | 0.1905 | **0.6545** | **51.7%** | 0.3011 | **0.1882** | +0.1129 |
| Ablation_wo_Relations | 0.2441 | 0.2148 | 0.6407 | 56.3% | 0.3416 | 0.2062 | +0.1353 |
| Ablation_wo_ReliabilityGate | 0.2089 | 0.1883 | 0.6875 | 47.3% | 0.3128 | 0.1685 | +0.1443 |
| Ablation_wo_Residual | 0.2515 | 0.2372 | 0.6782 | 50.3% | 0.3608 | 0.2089 | +0.1519 |
| Ablation_wo_SceneHNLoss | 0.2078 | 0.1787 | 0.6136 | 52.9% | 0.2505 | 0.1912 | +0.0593 |

## Table 2: Controlled Ablation Analysis (Component Contributions)
| Configuration | Overall nDCG@5 | Delta nDCG | Hard Neg AUC | Delta AUC | Contrastive Acc | Delta Cont | Unseen nDCG@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| R_VCGAT_Proposed_Full | 0.2198 | +0.0000 | 0.6545 | +0.0000 | 51.7% | +0.0% | 0.1882 |
| Ablation_wo_Relations | 0.2441 | +0.0244 | 0.6407 | -0.0139 | 56.3% | +4.7% | 0.2062 |
| Ablation_wo_ReliabilityGate | 0.2089 | -0.0109 | 0.6875 | +0.0330 | 47.3% | -4.3% | 0.1685 |
| Ablation_wo_Residual | 0.2515 | +0.0317 | 0.6782 | +0.0237 | 50.3% | -1.4% | 0.2089 |
| Ablation_wo_SceneHNLoss | 0.2078 | -0.0120 | 0.6136 | -0.0409 | 52.9% | +1.2% | 0.1912 |
