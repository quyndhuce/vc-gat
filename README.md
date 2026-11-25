# Decision-Certified Partial Reasoning for Uncertainty-Aware Heritage Video Captioning

Official code and evaluation repository for the paper:  
**"Decision-Certified Partial Reasoning for Uncertainty-Aware Heritage Video Captioning"**  
Submitted to the *International Journal of Approximate Reasoning* (IJAR), Elsevier.

- **Anonymous Code Repository:** [https://anonymous.4open.science/r/vc-gat-E1C7/](https://anonymous.4open.science/r/vc-gat-E1C7/)
- **Dataset Archive (Figshare DOI):** [10.6084/m9.figshare.35113231](https://doi.org/10.6084/m9.figshare.35113231)
- **Dataset & Video Mirror (Google Drive):** [https://drive.google.com/drive/folders/1iTEHWfA5XfYW5Y1KvoFMofbhSpki2Dwv?usp=sharing](https://drive.google.com/drive/folders/1iTEHWfA5XfYW5Y1KvoFMofbhSpki2Dwv?usp=sharing)

---

## 📌 Executive Summary & Core Contributions

Video captioning in high-stakes cultural heritage domains requires not only identifying relevant historical facts, but also determining **how assertively each fact should be communicated** under epistemic uncertainty.

In fuzzy knowledge graphs, when a candidate fact is supported by multiple Horn derivations, computing its aggregate support $\mu_F$ via noisy-OR over all paths is computationally expensive. We show that **exhaustive derivation evaluation is unnecessary** when feeding thresholded epistemic policies ($D(f) \in \{\text{omit}, \text{hedge}, \text{assert}\}$).

```
   Candidate Facts & Rules
             │
             ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Structural Certification Audit     │
│  (Verify Theorem 1: C1 Acyclic & C2 Disjoint)│
└──────────────────────┬───────────────────────┘
                       │ Undecided Facts (A1 Holds: 97.07%)
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 2: Certified Early-Stopping Inference │
│  (Compute [L_k, U_k] bounds; Corollary 1)    │
│  ──> Terminates once [L_k, U_k] ⊆ Cell       │
└──────────────────────┬───────────────────────┘
                       │ Flagged Non-A1 Topology (2.93%)
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 3: Exact Fallback Solver              │
│  (Complete fuzzy algebraic-sum aggregation   │
│   over all distinct Horn rule derivations)   │
└──────────────────────────────────────────────┘
```

> **Note on Aggregation Semantics:** The operational score $\mu_F(B) = 1 - (1 - w_0)\prod_{D \in \mathcal{D}_L(B)}(1 - \operatorname{val}(D))$ is a **fuzzy t-conorm algebraic sum**, not a probabilistic possible-worlds calculation. When the structural audit flags potential diamond-sharing or cyclic topologies ($2.93\%$ of complex facts), Stage 3 evaluates all distinct derivations directly under this algebraic sum with zero decision error ($T_{\mathrm{exact}}$). This is distinct from Section 5.3's auxiliary possible-world model, where true event union probability $\pi(B)$ is \#P-hard and bounded via anytime conditioning.

### Key Theoretical Guarantees (Section 5)

1. **Proposition 1 (Best Derivation; Inherited):** Over the max--product semiring, the valuation of the optimal derivation is computable in polynomial time via Knuth's generalized Dijkstra algorithm.
2. **Proposition 2 & 2a (Admissible Support Envelopes; New):** For depth-bounded fuzzy Horn programs satisfying component-simple derivation trees (**Assumption A1**), any evaluated prefix of derivations yields computable, monotone lower and upper bounds:
   $$\nu_S \le \mu_F(B) \le \nu_S + (1 - \nu_S)\min\{1, T_S\}$$
   When derivations are sorted offline by valuation, residual bounds satisfy $L_k = \mu_k$ and $U_k = \mu_k + (1 - \mu_k)(1 - (1 - q_k)^{m-k})$.
3. **Corollary 1 & 2 (Decision-Certified Early Stopping; New):** Halts immediately once the scaled interval $[\mu_L L_k, \mu_L U_k]$ is entirely contained within a single convex decision cell $\mathcal{C} \in \{[0, \theta_M), [\theta_M, \theta_H], (\theta_H, 1]\}$, provably guaranteeing:
   $$\operatorname{Regime}(\mu_L L_k) = \operatorname{Regime}(\mu_L U_k) \implies \operatorname{Regime}(c^*) = \operatorname{Regime}(\mu_L L_k)$$
4. **Theorem 1 (Structural A1 Certificate; New):** Sufficient conditions for component-simple derivations (A1) are verifiable prior to search in linear time via hypergraph acyclicity (C1) and pairwise disjoint reachable component sets (C2).
5. **Proposition 2b (Risk-Controlled Evidence Admission; Adapted):** Conformal risk control over exchangeable parent-video clusters guarantees expected video-average false-admission loss $\mathbb{E}[L_{n+1}(\hat\theta)] \le \varepsilon$.
6. **Propositions 3 & 4 (Dependency Envelope in Auxiliary Possible-World Model):** Under shared premise components, noisy-OR is provably optimistic ($\pi(B) \le \nu_\vee$; Prop 3). Anytime conditioning (Prop 4) contracts monotone bounds $\bar L_k \le \pi(B) \le \bar U_k$, converging to the exact union upon full component instantiation.
7. **Propositions 5 & 6 (Bayes Policy & Conformal Risk Control; Inherited):** Proposition 5 establishes the three-action Bayes policy; Proposition 6 guarantees distribution-free finite-sample marginal over-assertion risk control $\mathbb{E}[\ell_{n+1}(\hat\theta_H)] \le \alpha$. Remark 1 proves this guarantee is preserved identically under decision-certified early stopping.

### Three-Tier Assurance Hierarchy

- **Tier 1 (Per-Fact Decision Certification):** Mathematical certainty that the epistemic action ($D(f) \in \{\text{omit}, \text{hedge}, \text{assert}\}$) of an individual candidate fact is invariant to unexpanded derivations under Corollary 1.
- **Tier 2 (Empirical Top-K Agreement):** Fact-level decision invariance does *not* imply pairwise score or ranking invariance. The observed complete Top-5 set and order agreement ($500/500$ clips, $100\%$) on Hanoi-500 is an **empirical benchmark finding** achieved via candidate ranking refinement ($\varepsilon_{\mathrm{rank}} \le 10^{-4}$), not an unconditional mathematical theorem across arbitrary candidate pools.
- **Tier 3 (Caption-Level Factuality):** Neither fact certification nor Top-K ranking certificates guarantee correctness of generated text. Caption-level over-assertion risk is bounded in expectation ($\mathbb{E}[R] \le \alpha$) via conformal risk control (Proposition 6) and calibrated prompting, but verbalization may still introduce hallucination outside certified evidence (coverage-reliability trade-off).

---

## 📊 Key Benchmark Results (Matching 39-Page Manuscript)

All results reported below are verified by running `python3 reproduce_all.py` directly on the codebase:

### 1. Decision-Certified Fuzzy Horn Reasoning (Table 2)
| Metric / Benchmark | Exhaustive (Oracle) | Pure Certified BnB | Hybrid Proposed (Ours) | Manuscript Reference |
|:---|:---:|:---:|:---:|:---:|
| **Evaluated Derivations** | $12{,}391$ | $4{,}020$ | **$4{,}096$** | $-66.94\%$ evaluations avoided |
| **Policy Flips ($761$ instances)** | Baseline | **$0$** | **$0$** | Provably $0$ flips (Corollary 1) |
| **Top-5 Set / Order Match** | $500/500$ (100\%) | $500/500$ (100\%) / $313/500$ | **$500/500$ ($100\%$) / $500/500$ ($100\%$)** | Exact empirical agreement |
| **In-Memory RAM Latency** | $23.31 \pm 0.99$\,ms | $69.00 \pm 5.55$\,ms | **$28.33 \pm 0.33$\,ms** | Bound tracking overhead |
| **Persistent NVMe SSD Latency** | $114.42 \pm 14.68$\,ms | $61.68 \pm 8.39$\,ms | **$66.57 \pm 9.64$\,ms** | **$1.72\times$ disk speedup ($p < 0.001$)** |
| **Storage Latency Crossover** | — | — | **$t_{\text{crossover}} = 0.605\,\mu$s** | Break-even persistent read threshold |

### 2. VC-GAT Grounding Fact Selection (Table 4 & Table 20)
| Model / Architecture | Overall nDCG@5 | Precision@5 | Hard Neg AUC | Seen nDCG@5 | Unseen nDCG@5 | Paired Delta vs Cross-Attn |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Crisp-KG Baseline | $0.0320$ | $0.0402$ | $0.4130$ | $0.0410$ | $0.0250$ | — |
| SigLIP-2 Zero-Shot | $0.0720$ | $0.0871$ | $0.5040$ | $0.1120$ | $0.0390$ | — |
| GATv2 + Video Fusion | $0.3400$ | $0.2304$ | $0.6042$ | $0.3402$ | $0.3400$ | $+0.0362$ |
| Cross-Attention Ranker | $0.3038$ | $0.2460$ | $0.7064$ | $0.3958$ | $0.2291$ | Baseline |
| **VC-GAT (Ours, Proposed)** | **$0.3899$** | **$0.3196$** | **$0.7501$** | **$0.5049$** | **$0.2966$** | **$+0.0861$ [$+0.0432, +0.1335$]** ($p < 0.001$) |

### 3. End-to-End Multimodal Captioning & Cross-Region Transfer (Table 7 & Table 23)
| Evaluation Benchmark | System / Configuration | FP Macro $\uparrow$ | CSR $\uparrow$ | BKR@5 $\uparrow$ | VFAR $\downarrow$ | Notes |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Hanoi-500 In-Domain (Table 7)** | E1 Qwen2.5-VL-3B | $0.7355 \pm 0.1335$ | $0.1463 \pm 0.0615$ | $0.2794 \pm 0.1200$ | $0.2743 \pm 0.1425$ | Visual-only foundation |
| | E5 DEC + Base Qwen3B | $\mathbf{0.8093} \pm 0.0710$ | $0.2454 \pm 0.0705$ | $0.5683 \pm 0.1035$ | $\mathbf{0.1940} \pm 0.0700$ | High precision, lower recall |
| | **E6 Proposed (Full LoRA)** | $0.7679 \pm 0.0895$ | $\mathbf{0.4642} \pm 0.0745$ | $\mathbf{0.6813} \pm 0.0740$ | $0.2613 \pm 0.0965$ | **$+89.2\%$ CSR coverage gain** |
| **HCMC-300 Transfer (Table 23)** | E5 DEC + Base Qwen3B | $0.8679 \pm 0.0525$ | $0.5717 \pm 0.0935$ | $0.8782 \pm 0.0615$ | $0.1279 \pm 0.0530$ | Cross-region baseline |
| | **E6 Proposed (Full LoRA)** | $\mathbf{0.8920} \pm 0.0525$ | $\mathbf{0.7196} \pm 0.1155$ | $\mathbf{0.8744} \pm 0.0800$ | $\mathbf{0.0596} \pm 0.0330$ | **Landmark-adapted transfer** |

> **Cross-Region Evaluation Scope:** HCMC evaluation (Table 23) tests cross-region transfer with *landmark-recognizer adaptation* (source-video grouped split, Recall@1 $91.33\%$), while VC-GAT weights, Horn reasoning rules, decision thresholds ($\theta_M=0.50, \theta_H=0.80$), and the generator remain frozen from Hanoi. Recognizer adaptation prevents interpreting this as fully zero-shot transfer. Conformal bounds from Hanoi are not asserted on HCMC because exchangeability across regions is not established.

---

## 📂 Repository Directory Layout

```text
TransMTL_OntoKG-main/
├── README.md                             # Publication guide & reproduction documentation
├── requirements.txt                      # Dependencies tiered by reproduction vs. full training
├── config.yaml                           # Master system configuration and default hyperparameters
│
├── src/                                  # Reusable core library for reasoning, grounding & evaluation
│   ├── reasoning/                        # Fuzzy Horn reasoning, monotone bounds & solver
│   │   ├── bounds.py                     # Proposition 2, 2a, Corollary 1 bounding routines
│   │   ├── horn_rules.py                 # Ontology Horn rules and attenuation factors (lambda=0.90)
│   │   ├── engine.py                     # DecisionCertifiedHornSolver (Analytical + BnB + Fallback)
│   │   └── persistent_store.py           # Disk-backed B-Tree storage store on NVMe SSD
│   ├── grounding/                        # Visual grounding and graph attention modules
│   │   ├── vc_gat.py                     # Degree-debiased attention (gamma=0.40) & hub dampening
│   │   └── landmark_gate.py              # Visual landmark gate, temperature scaling & calibration
│   ├── conformal/                        # Conformal risk control (Propositions 5 & 6)
│   │   └── risk_control.py               # Distribution-free finite-sample calibration
│   ├── captioning/                       # Prompt formatting & epistemic regimes
│   │   └── prompt_builder.py             # Assert / Hedge / Omit prompt compilation
│   └── evaluation/                       # Evaluation metrics and statistical testing
│       ├── metrics.py                    # nDCG@K, Precision@K, Recall@K, BKR@K, VFAR, CSR
│       └── bootstrap.py                  # Hierarchical parent-video cluster bootstrap (B=10,000)
│
├── experiments/                          # Verified reproduction runners matching paper tables
│   ├── reproduce_all.sh                  # Master script running all 10 reproduction steps
│   ├── reproduce_table3_certified_horn.py # Table 3: Decision-Certified Horn Reasoning Benchmark
│   ├── reproduce_table4_derivation_depth.py # Table 4: Anytime conditioning & derivation depth
│   ├── reproduce_table5_vcgat_ranking.py # Table 5: VC-GAT paired cluster bootstrap significance
│   ├── reproduce_table6_landmark_calibration.py # Table 6: Landmark score calibration & Bayes risk
│   ├── reproduce_table7_end2end_captioning.py # Tables 7 & 8: End-to-end captioning on Hanoi-500
│   ├── reproduce_table11_cross_domain_hcmc.py # Table 11: Cross-domain zero-shot transfer on HCMC-300
│   ├── reproduce_proposition_2b_2c_stress.py # Stress tests for Propositions 2b & 2c
│   ├── reproduce_persistent_storage_benchmark.py # Physical NVMe SSD persistent storage benchmark
│   ├── reproduce_figure2_certification_dependency.py # Figure 2 data & vector plot generator
│   ├── reproduce_figure3_conformal_risk.py # Figure 3 conformal risk-coverage curve generator
│   └── outputs/                          # JSON reports and benchmark logs
│
├── tests/                                # Automated unit test suite (100% passing)
│   ├── test_proposition_2_solver_certification.py # Solver correctness & zero-flip guarantee
│   ├── test_proposition_2b_2c_stress.py  # Proposition 2b/2c empirical stress tests
│   ├── test_proposition_5_6_conformal_risk.py # Conformal risk bounds & finite-sample coverage
│   ├── test_theory_properties.py         # Numerical validation on 10,000 Horn programs
│   ├── test_gate_check_m0_assumptions.py # Verification of Assumption A1 structural conditions
│   └── test_persistent_storage.py        # NVMe SSD disk store speedup verification
│
├── data/                                 # Datasets, Ground Truth FKGs, and Manifests
│   ├── fkg_per_kb/                       # Fused heritage knowledge graphs (Hanoi & HCMC)
│   ├── caption_dataset/                  # Benchmark captions (Hanoi-500, HCMC-300, Val-149)
│   ├── unified_coco_dataset/             # Grounding annotations and bounding boxes
│   └── scene_conditioned_gold/           # Gold-standard scene-conditioned Horn evidence
│
├── figures/                              # Publication vector PDF and high-res PNG figures
│   ├── Figure1_overall_architecture.png  # System architecture diagram
│   ├── Figure2_theory_certification_dependency.pdf # Early stopping dependency envelope
│   └── Figure3_conformal_risk_coverage.pdf # Conformal risk-coverage calibration curve
│
├── tables/                               # Generated publication LaTeX table snippets
├── results/                              # Benchmark CSV summaries and raw metric logs
└── legacy_archive/                       # Safely archived exploratory and historical scripts
```

---

## 💾 Dataset & Video Clips Download

The complete cultural heritage knowledge graph, verified scene-conditioned Horn evidence, benchmark splits, and 4,346 cut video clips are accessible via:

1. **Google Drive Mirror (Clean Benchmark Package & Cut Video Clips):**  
   🔗 **[Google Drive Dedicated Folder](https://drive.google.com/drive/folders/1iTEHWfA5XfYW5Y1KvoFMofbhSpki2Dwv?usp=sharing)**
   - `vc_gat_dataset_clean_462M.tar.gz` (2.24 GB): Master benchmark archive containing:
     - `data/caption_dataset/`: Benchmark captions across epistemic regimes (Hanoi-500, HCMC-300, Val-149).
     - `data/fkg_per_kb/`: Complete fused fuzzy knowledge graphs (13,553 facts across 4,093 cultural landmarks).
     - `data/unified_coco_dataset/annotations/`: Object and concept grounding annotations with bounding boxes.
     - `data/dataset_splits_scheme_A_parent.json`: Canonical parent-video disjoint train/val/test splits.
     - `scene_conditioned_gold/`: Ground-truth Horn-complete evidence and contrastive challenge pairs.
     - `experiments/features/`: Pre-extracted visual scene graph features (`.pkl`, `.npy`), enabling instant reproduction without raw video decoding.
   - `videos/` (50.6 GB): Complete 4,346 cut video clips (Train: 3,227 clips, Test: 1,119 clips) directly fed into the multimodal models.

2. **Figshare Open Science Repository (DOI):**  
   🔗 **[Figshare Dataset Record (DOI: 10.6084/m9.figshare.35113231)](https://doi.org/10.6084/m9.figshare.35113231)**  
   Permanent academic repository for long-term open access preservation.

---

## ⚡ Quickstart

### 1. Installation

To reproduce the paper's theoretical validations, benchmarks, and statistical evaluations on CPU, install Tier 1 dependencies:

```bash
# Clone the repository (or unpack the supplementary archive)
git clone <ANONYMIZED_REPOSITORY_URL>
cd vc-gat

# Create environment (Python 3.10+)
python3 -m venv venv
source venv/bin/activate

# Install reproduction dependencies
pip install -r requirements.txt
```

### 2. Four Tiers of Reproducibility

To support diverse review workflows, the repository provides four distinct verification tiers:

| Tier | Scope | Hardware | Latency | Command |
|:---|:---|:---:|:---:|:---|
| **Level 1: Cryptographic Integrity** | SHA-256 verification of all checkpoints, splits, and manifests | Any CPU | < 2 sec | `python3 tests/test_checkpoint_hashes.py` |
| **Level 2: Metric Recomputation** | Recomputes all manuscript tables & cluster bootstraps from saved outputs | Any CPU | ~12 sec | `python3 reproduce_all.py` |
| **Level 3: Fresh Inference** | End-to-end forward passes using pre-extracted visual features | CPU / 1 GPU | 2–5 min | `python3 experiments/run_strictly_controlled_clean_end2end_ablation.py` |
| **Level 4: Full Training from Scratch** | Full retraining of VC-GAT ranker, LoRA adapter, & recognizer | GPU (RTX 5090 / A100) | 4–6 hrs | `./experiments/train_all_from_scratch.sh` |

### 3. Run Automated Unit Tests

Run the full automated test suite verifying theoretical properties across 10,000 synthetic Horn programs and dataset queries:

```bash
python3 -m unittest discover tests/
```
*Expected Output:*
```text
Ran 23 tests in ~11s
OK
```

### 4. One-Click Master Reproduction of All Paper Results (Level 2)

Execute the master reproduction script to run cryptographic checkpoint audits, theoretical property unit tests, and reproduce all key paper tables (Tables 2, 3, 4, 5, 6, 7, 8, 20, 23):

```bash
# Master reproduction suite (Level 1 & Level 2 verification in ~12 seconds)
python3 reproduce_all.py

# Recompute paired cluster bootstrap statistics (B = 10,000 resamples for Table 20)
python3 recompute_table22.py
```

---

## 🔬 Table-by-Table Reproduction Guide

Every empirical claim in the 39-page manuscript can be verified individually:

### Table 2: Decision-Certified Partial Horn Reasoning Benchmark
Demonstrates that certified early stopping avoids $66.94\%$ of derivation evaluations ($12{,}391 \to 4{,}096$) while preserving $100\%$ of Top-5 decisions with zero policy flips and a measured $1.72\times$ wall-clock speedup on physical NVMe SSD:
```bash
python3 experiments/reproduce_table3_certified_horn.py
```
*(Script retained as `reproduce_table3_certified_horn.py` for backward compatibility with submission checkpoints; corresponds to **Table 2** in the final manuscript).*

### Table 3: Anytime Conditioning and Derivation Depth
Evaluates residual interval contraction $W_k = U_k - L_k$ (contracting from $0.1521$ to $0.0290$), resolution percentages ($81.6\% \to 96.9\%$), and zero monotonicity violations on the fixed X3b cohort:
```bash
python3 experiments/reproduce_table4_derivation_depth.py
```
*(Corresponds to **Table 3** in the final manuscript).*

### Table 4 & Table 20: VC-GAT Grounding Fact Selection Benchmark
Calculates ranking metrics (VC-GAT Overall $\text{nDCG}@5 = 0.3899$, $\text{P}@5 = 0.3196$, $\text{R}@5 = 0.1746$, Hard Neg $\text{AUC} = 0.7501$) and paired cluster bootstrap significance ($B=10{,}000$ resamples over 60 parent videos; $\Delta = +0.0861$ [$+0.0432, +0.1335$], $p < 0.001$):
```bash
python3 experiments/reproduce_table5_vcgat_ranking.py
```
*(Corresponds to **Table 4** and **Table 20** in the final manuscript).*

### Table 6 & Table 21: Risk-Constrained Evidence Admission & Conformal Calibration
Assesses conformal risk control under Proposition 2b (at $\alpha=0.05$: test cluster risk $0.89\%$, grounded recall $10.27\%$, precision $46.29\%$) and isotonic calibration (ECE reduced from $10.87\%$ to $4.10\%$):
```bash
python3 experiments/reproduce_table6_landmark_calibration.py
```
*(Corresponds to **Table 6** and **Table 21** in the final manuscript).*

### Table 7 & Table 8: End-to-End Multimodal Video Captioning (Hanoi-500)
Reproduces end-to-end captioning metrics for configurations E1 through E6 (E6 Full LoRA: CSR $0.4642 \pm 0.0745$, BKR@5 $0.6813 \pm 0.0740$, VFAR $0.2613 \pm 0.0965$, FP Macro $0.7679 \pm 0.0895$) with 95% cluster-bootstrap confidence intervals:
```bash
python3 experiments/reproduce_table7_end2end_captioning.py
```
*(Corresponds to **Table 7** and **Table 8** in the final manuscript).*

### Table 23: Cross-Region Transfer on HCMC-300 with Landmark-Recognizer Adaptation
Validates cross-region transfer performance on Ho Chi Minh City cultural landmarks with source-video grouped landmark-recognizer adaptation (CSR $0.7196 \pm 0.1155$, VFAR $0.0596 \pm 0.0330$, FP Macro $0.8920 \pm 0.0525$):
```bash
python3 experiments/reproduce_table11_cross_domain_hcmc.py
```
*(Script retained as `reproduce_table11_cross_domain_hcmc.py` for backward compatibility; corresponds to **Table 23** in the final manuscript).*

### Measured Persistent Storage Benchmark on NVMe SSD
Directly measures the $1.72\times$ disk I/O speedup ($66.57 \pm 9.64$\,ms vs.\ $114.42 \pm 14.68$\,ms on disk) on physical NVMe SSD storage and computes the $0.605\,\mu$s crossover point:
```bash
python3 experiments/reproduce_persistent_storage_benchmark.py
```

### Figures 2 & 3: Theoretical Envelopes & Risk-Coverage Curves
Verifies data integrity and regenerates publication-quality vector PDF figures for Section 5 and Section 6:
```bash
python3 experiments/reproduce_figure2_certification_dependency.py
python3 experiments/reproduce_figure3_conformal_risk.py
```

---

## 🛠 Citation & Contact

```bibtex
@article{anonymous2026decisioncertified,
  title={Decision-Certified Partial Reasoning for Uncertainty-Aware Heritage Video Captioning},
  author={Anonymous Authors},
  journal={Submitted to International Journal of Approximate Reasoning},
  year={2026}
}
```

*Note: Author identities, contact emails, and institutional affiliations have been omitted for double-blind peer review.*
