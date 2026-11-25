# Independent Reproducibility & Verification Guide — IJAR Manuscript
**Manuscript Title:** *Decision-Certified Partial Reasoning for Uncertainty-Aware Heritage Video Captioning*  
**Target Journal:** *International Journal of Approximate Reasoning (IJAR)*  
**Verification Level:** Level 1 (Raw Recomputation), Level 2 (Inference on Frozen Checkpoints), Level 3 (Training Provenance)  
**Git Commit Hash:** `efc8364`  
**Overall Verdict:** **PASS (GO for Submission)**

---

## 1. Quick Start (Master Reproduction in One Command)

To reproduce all experimental tables, theoretical property audits, and statistical confidence intervals:

```bash
python3 reproduce_all.py
```

This master script verifies:
1. **Cryptographic Checkpoint Integrity:** Computes and verifies SHA-256 hashes of all frozen checkpoints against `checkpoint_hashes.json`.
2. **Theoretical Certificates:** Runs 23 automated unit tests verifying Assumption A1 acyclicity, C1/C2 bounds, Harris--FKG correlation bounds, and Conformal Risk Control.
3. **Table 2 (Certified Horn Reasoning Benchmark):** Recomputes derivation counts, 66.94% computational savings, zero policy flips, and 1.72x physical NVMe disk speedup.
4. **Table 3 (Anytime Conditioning Protocol X3b):** Evaluates envelope contraction across depths $k \in \{0, 1, 2, 3\}$.
5. **Table 5 (Exact Ranking Improvement Decomposition):** Recomputes Panel A (pairwise risk change across $\alpha$) and Panel B (exact nDCG@5 partition: entering, leaving, within, outside).
6. **Table 6 (Risk-Constrained Evidence Admission):** Recomputes Cluster Conformal Risk Control admission across nominal $\alpha \in \{0.05, 0.10, 0.20\}$.
7. **Table 20 (Appendix Table 22 / Paired Grounding Comparison):** Recomputes 6 evaluation slices, paired point estimates, 95% clip-level CIs, and 95% cluster-aware bootstrap CIs ($B=10{,}000$).
8. **Table 7 & Table 23 (End-to-End Captioning & Cross-Domain HCMC):** Recomputes generative captioning metrics (CSR, BKR@5, KR, VFAR, EAP).

---

## 2. Hardware and Software Environment Specifications

Detailed specifications are exported in `environment_dependencies.json`.

- **Training Cluster (`aircsrv4.airc.lab`):**
  - **OS:** Linux 6.8.0-49-generic (Ubuntu 22.04 LTS)
  - **GPU:** NVIDIA GeForce RTX 5090 (32 GB VRAM, CUDA 12.8)
  - **CPU / RAM:** 64 GB DDR5 RAM
  - **Storage:** Physical NVMe PCIe 4.0 SSD (used for persistent disk latency benchmarks)
  - **Python Environment:** Python 3.10.16, PyTorch 2.11.0+cu128, Transformers 4.46.3, PEFT 0.13.2, Accelerate 1.1.1, NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.5.2.
- **Audit Workstation:**
  - **OS:** Apple macOS Darwin (arm64)
  - **Python Environment:** Python 3.14 / 3.10+, NumPy 2.5.3, SciPy 1.18.1, Pandas 3.0.6, scikit-learn 1.9.1.
- **Random Seeds:**
  - Global Default Seed: `42`
  - Paired Bootstrap Resamples: $B = 10{,}000$ (seed `42`)
  - Retraining Seed Ensemble: $\{42, 123, 456, 789, 999\}$

---

## 3. Cryptographically Frozen Checkpoints

All model checkpoints used in this benchmark are frozen and verified via SHA-256 in `checkpoint_hashes.json`:

| Checkpoint Name | File Path | File Size | SHA-256 Hash | Role in Benchmark |
| :--- | :--- | :---: | :--- | :--- |
| `best_dec_vcgat_v2.pt` | `experiments/checkpoints_fact_selection/best_dec_vcgat_v2.pt` | 24,774,433 bytes | `01f3b3742d5e55a296c3d99c784244c66ecb32a0870ad1d3f7b0773745f36633` | Operational DEC-v2 model for Hanoi-500 test ranking |
| `best_cross_attention.pt` | `experiments/checkpoints_fact_selection/best_cross_attention.pt` | 11,896,017 bytes | `e243799dad2701d0821bd18fd63254013df255127a4a0a51ce1c7d5b31476fa8` | Supervised multimodal non-graph baseline ranker |
| `best_dec_vcgat_v2_full_horn.pt` | `experiments/checkpoints_fact_selection/best_dec_vcgat_v2_full_horn.pt` | 24,775,017 bytes | `b8ddb441a1d1ec3be6e0b83090781058493159ae4d3ab44c8f0c8828dd31f54a` | Horn saturated fixpoint closure ablation |
| `best_decoupled_ranking_vcgat.pt` | `experiments/features/best_decoupled_ranking_vcgat.pt` | 44,571,181 bytes | `cf19893f3b8f5a57960e22ea49d6e32b8a604d81c3c69433db3d969a1535930e` | Decoupled ranking dual-scorer network |
| `adapter_model.safetensors` | `experiments/checkpoints_caption/qwen25_3b_certainty_lora/adapter_model.safetensors` | 119,801,528 bytes | `70d0c2083d287e82df493ea74dfd2013e63a43a580713fb19c953f50dfa81dfc` | LoRA certainty adapter for Qwen2.5-3B generator |

---

## 4. Dataset Provenance and Leakage-Free Verification

Audit results exported in `data_leakage_report.json`:
- **Hanoi-500 Test Cohort:** Exactly 500 clips partitioned into 60 disjoint parent videos (`scene_conditioned_gold/outputs/evaluation_clips_manifest.json`).
- **HCMC-300 OOD Cohort:** Exactly 300 clips across 26 parent videos in Ho Chi Minh City. Overlap with Hanoi: 0 clips, 0 parent videos.
- **Caption Fine-Tuning Cohort:** 702 training samples, 149 validation instances, 500 test clips. Overlap: exactly 0.
- **Parent Video Separation (Scheme A):** 965 train videos, 207 val videos, 208 test videos. Pairwise video-level overlap: exactly 0.

---

## 5. Table Provenance and Source Mapping

All 23 tables of the final manuscript are mapped to their underlying raw experimental data in `table_provenance.csv`:

- **Table 1 (`tab:dataset`):** Cohort sizes from `evaluation_clips_manifest.json` and `dataset_splits_scheme_A_parent.json`.
- **Table 2 (`tab:x2`):** Reasoning timings from `experiments/outputs/measured_persistent_storage_benchmark_report.json`.
- **Table 3 (`tab:x3`):** Dependency bounds from `experiments/outputs/certified_horn_early_stopping_report.json`.
- **Table 4 (`tab:grounding`):** Benchmark scores from `table22_per_clip_metrics.csv` and `evaluation_manifest.json`.
- **Table 5 (`tab:ranking_decomposition`):** Panel A and Panel B calculated from `audit_artifacts_reconciliation/raw_candidate_predictions_hanoi500.csv`.
- **Table 6 (`tab:evidence_admission`):** Cluster CRC thresholds and empirical test risk from `audit_artifacts_reconciliation/master_reconciliation_audit_report.json`.
- **Table 7 (`tab:end2end`):** Hanoi captioning metrics from `data/caption_dataset/test_500_balanced_all_models_captions_scientific_vcgat.json`.
- **Table 8 (`tab:ablation`):** Retrained component ablation from `data/caption_dataset/retrained_component_ablation_report.json`.
- **Table 17 (`tab:app_statistical_uncertainty`):** Bootstrap intervals from `audit_artifacts_reconciliation/paired_cluster_bootstrap_replicates.csv` and `table22_bootstrap_replicates.npz`.
- **Table 20 (`tab:vcgat_bootstrap`):** Paired grounding comparison on Hanoi-500 from `table22_verified_results.json`.
- **Table 21 (`tab:calibration`):** Multilevel conformal risk control empirical results from `data/caption_dataset/audited_claims_644_x4_x5.json`.
- **Table 23 (`tab:cross`):** HCMC-300 zero-shot transfer from `experiments/outputs/hanoi_vs_hcmc_held_out_evaluation_report.json`.

---

## 6. Audit Artifacts and Manifests

The complete audit generated the following machine-readable records:
1. `checkpoint_hashes.json`: Cryptographic hashes of all model weights.
2. `data_leakage_report.json`: Verification of split isolation.
3. `table_provenance.csv`: Traceability matrix from LaTeX tables to raw data.
4. `manuscript_comparison.csv`: Claim-by-claim numerical verification against `main.pdf`.
5. `recomputed_metrics.json`: Independently calculated metric values.
6. `experiment_manifest.json`: Master freeze manifest and audit metadata.
7. `environment_dependencies.json`: Full environment and package configuration.
8. `EXPERIMENTAL_AUDIT_REPORT.md`: Comprehensive audit report and final submission recommendation.
