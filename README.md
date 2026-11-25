# Decision-Certified Partial Reasoning for Uncertainty-Aware Heritage Video Captioning

Official code and evaluation repository for the paper:  
**"Decision-Certified Partial Reasoning for Uncertainty-Aware Heritage Video Captioning"**  
Submitted to the *International Journal of Approximate Reasoning* (IJAR), Elsevier.

---

## 📌 Executive Summary & Core Contributions

Video captioning in high-stakes cultural heritage domains requires not only identifying relevant historical facts, but also determining **how assertively each fact should be communicated** under epistemic uncertainty.

In fuzzy knowledge graphs, when a candidate fact is supported by multiple Horn derivations, computing its aggregate support $\mu_F$ via noisy-OR over all paths is computationally expensive. We show that **exhaustive derivation evaluation is unnecessary** when feeding thresholded epistemic policies ($D(f) \in \{\text{omit}, \text{hedge}, \text{assert}\}$).

```
   Candidate Facts & Rules
             │
             ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Analytical Decision Certification  │
│  (Check Assumption A1: Acyclic & Disjoint)   │
└──────────────────────┬───────────────────────┘
                       │ Undecided Facts
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 2: Certified Branch-and-Bound (BnB)   │
│  (Compute [L_k, U_k] bounds; Corollary 1)    │
│  ──> Terminates once bounds fall in cell     │
└──────────────────────┬───────────────────────┘
                       │ Complex Overlapping Cases
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 3: Exact Fallback Solver              │
│  (Full inclusion-exclusion aggregation)      │
└──────────────────────────────────────────────┘
```

### Key Theoretical Guarantees

1. **Proposition 2 & 2a (Admissible Support Envelopes):** For depth-bounded fuzzy Horn programs satisfying a checkable structural condition (**Assumption A1**), any evaluated prefix of derivations yields computable, monotone lower and upper bounds $[L_k, U_k]$ on residual truth support.
2. **Corollary 1 (Certified Early Stopping):** Once the scaled interval $[s_{\text{vis}} \cdot L_k, s_{\text{vis}} \cdot U_k]$ is entirely contained within a single epistemic decision cell $(\tau_D, \tau_A]$, derivation search halts immediately with **provable decision preservation**.
3. **Propositions 2b & 2c (Perceptual Lower-Bound & Degree-Debiasing):** VC-GAT degree-debiased attention ($\gamma=0.40$) strictly dampens structural graph hub domination ($\mathcal{O}(d_j^{1-\gamma})$) while the visual saliency gate ensures zero ungrounded hallucinated facts into Top-5 evidence.
4. **Propositions 5 & 6 (Conformal Risk Control):** Distribution-free finite-sample risk control guaranteeing $\mathbb{E}[R(\tau)] \le \alpha$ under parent-video clustered exchangeability.
5. **Three-Tier Assurance Hierarchy:**
   - **Tier 1 (Per-Fact Decision Certification):** Mathematical certainty that the omit/hedge/assert label of each evaluated fact is invariant to unexpanded derivations.
   - **Tier 2 (Top-K Set Agreement):** Empirical agreement on the evaluated cohort (e.g., 500/500 clips on Hanoi-500).
   - **Tier 3 (Caption-Level Factuality):** Controlled hallucination risk bounded via conformal thresholds and calibrated LLM prompting.

---

## 📊 Key Benchmark Results

All results reported below are verified by running `./experiments/reproduce_all.sh` directly on the codebase:

| Metric / Benchmark | Exhaustive (Oracle) | Pure Certified BnB | Hybrid Proposed (Ours) | Paper Target |
|:---|:---:|:---:|:---:|:---:|
| **Evaluated Derivations** | $12{,}391$ | $4{,}020$ | **$4{,}096$** | $-66.94\%$ evaluations avoided |
| **Policy Flips ($761$ instances)** | Baseline | **$0$** | **$0$** | Provably $0$ flips |
| **Top-5 Clip Agreement** | $100\%$ | $100\%$ | **$500/500$ ($100\%$)** | Exact empirical agreement |
| **In-Memory RAM Latency** | $23.31 \pm 0.99$\,ms | $69.00 \pm 5.55$\,ms | **$28.33 \pm 0.33$\,ms** | Low overhead |
| **Persistent NVMe SSD Latency** | $140.76 \pm 31.79$\,ms | $72.82 \pm 11.37$\,ms | **$77.28 \pm 18.86$\,ms** | **$1.72\times$ – $1.82\times$ disk speedup** |
| **Analytical Crossover Latency** | — | — | **$t_{\text{crossover}} = 605$\,ns** | Measured crossover |
| **VC-GAT nDCG@5 (Table 5)** | $0.6276$ (GAT) | $0.6865$ (TempMax) | **$0.7231 \pm 0.0460$** | $+0.0955$, $p < 0.001$ (paired cluster bootstrap) |
| **Hanoi-500 E6 Captioning (Table 7)** | — | — | **CSR: $0.4642 \pm 0.075$** | **VFAR: $0.2613 \pm 0.096$**, **BKR@5: $0.6813 \pm 0.074$** |
| **HCMC-300 Zero-Shot Transfer (Table 11)**| — | — | **CSR: $0.7196 \pm 0.041$** | **VFAR: $0.0596 \pm 0.016$**, **OVR: $0.0\%$** |

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

### 2. Run Automated Unit Tests

Run the full automated test suite verifying theoretical properties across 10,000 synthetic Horn programs and dataset queries:

```bash
python3 -m unittest discover tests/
```
*Expected Output:*
```text
Ran 23 tests in ~11s
OK
```

### 3. One-Click Master Reproduction of All Paper Results

Execute the master reproduction script to run cryptographic checkpoint audits, theoretical property unit tests, and reproduce all key paper tables (Tables 2, 5, 6, 7/11, 20/22):

```bash
# Master reproduction suite (Level 1 & Level 2 verification in ~12 seconds)
python3 reproduce_all.py

# Recompute paired bootstrap statistics (B = 10,000 resamples for Table 20/22)
python3 recompute_table22.py
```

---

## 🔬 Table-by-Table Reproduction Guide

Every empirical claim in the manuscript can be verified individually:

### Table 3: Decision-Certified Partial Horn Reasoning Benchmark
Demonstrates that certified early stopping avoids $66.94\%$ of derivation evaluations ($12{,}391 \to 4{,}096$) while preserving $100\%$ of Top-5 decisions with zero policy flips:
```bash
python3 experiments/reproduce_table3_certified_horn.py
```

### Table 4: Anytime Conditioning and Derivation Depth
Evaluates residual interval contraction $W_k = U_k - L_k$, resolution percentages, and decision latencies across evaluation prefixes:
```bash
python3 experiments/reproduce_table4_derivation_depth.py
```

### Table 5: Video-Conditioned Graph Attention (VC-GAT) Ranking
Calculates ranking metrics ($\text{nDCG}@5$, $\text{P}@5$, $\text{R}@5$) and paired cluster bootstrap significance ($B=10{,}000$ resamples over 60 parent videos):
```bash
python3 experiments/reproduce_table5_vcgat_ranking.py
```

### Table 6: Visual Landmark Grounding & Conformal Risk Control
Assesses expected calibration error (ECE), negative log-likelihood (NLL), and conformal risk calibration across varying risk tolerances $\alpha$:
```bash
python3 experiments/reproduce_table6_landmark_calibration.py
```

### Table 7 & Table 8: End-to-End Multimodal Video Captioning (Hanoi-500)
Reproduces end-to-end captioning metrics for configurations E1 through E6 (CSR, BKR@5, VFAR, KR, FP) with 95% cluster-bootstrap confidence intervals:
```bash
python3 experiments/reproduce_table7_end2end_captioning.py
```

### Table 11: Cross-Domain Zero-Shot Transfer (HCMC-300)
Validates zero-shot transfer performance on Ho Chi Minh City cultural landmarks without retraining (CSR $0.7196 \pm 0.041$, VFAR $0.0596 \pm 0.016$, OVR $0.0\%$):
```bash
python3 experiments/reproduce_table11_cross_domain_hcmc.py
```

### Measured Persistent Storage Benchmark on NVMe SSD
Directly measures the $1.72\times$ – $1.82\times$ disk I/O speedup on physical NVMe SSD storage and computes the $605$\,ns crossover point:
```bash
python3 experiments/reproduce_persistent_storage_benchmark.py
```

### Figures 2 & 3: Theoretical Envelopes & Risk-Coverage Curves
Verifies data integrity and regenerates vector PDF figures for Section 5 and Section 6:
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
