# Project Structure and Module Mapping Specification

> **Manuscript:** *Decision-Certified Partial Reasoning for Uncertainty-Aware Heritage Video Captioning*  
> **Target Journal:** *International Journal of Approximate Reasoning* (Elsevier CAS)  
> **Repository:** `TransMTL_OntoKG`  
> **Last Updated:** November 23, 2026

---

## 1. Executive Architecture & Pipeline Flow

The codebase implements an end-to-end knowledge-grounded multimodal video captioning framework designed for cultural heritage preservation under epistemic uncertainty:

```
                  ┌───────────────────────────────┐
                  │ 1. Visual Universe Extraction │
                  │  (285 Concepts & SGG BBoxes)  │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │ 2. Fuzzy Knowledge Graph      │
                  │   (Hanoi & HCMC Ontologies)   │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │ 3. Certified Horn Reasoning   │
                  │    - Stage 1: Analytical A1   │
                  │    - Stage 2: Certified BnB   │
                  │    - Stage 3: Exact Fallback  │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │ 4. VC-GAT Grounding           │
                  │  (Degree-Debiasing gamma=0.40)│
                  │  (Landmark Saliency Gating)   │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │ 5. Conformal Risk Control     │
                  │  (Propositions 5 & 6; E[R]<=a)│
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │ 6. LLM Fact-to-Caption Engine │
                  │  (Assert / Hedge / Omit)      │
                  └───────────────────────────────┘
```

---

## 2. Directory Tree Layout

```text
TransMTL_OntoKG-main/
│
├── README.md                             # Publication guide & reproduction documentation
├── PROJECT_STRUCTURE.md                  # This architectural specification document
├── requirements.txt                      # Dependencies tiered by reproduction vs. full training
├── config.yaml                           # Master system configuration and default hyperparameters
│
├── src/                                  # Core Python package (Modular Architecture)
│   ├── __init__.py                       # Package initialization (v2.0.0)
│   ├── reasoning/                        # Fuzzy Horn reasoning, bounds, and solvers
│   │   ├── __init__.py
│   │   ├── bounds.py                     # Proposition 2, 2a, Corollary 1 bounding routines
│   │   ├── horn_rules.py                 # Ontology Horn rules and attenuation factors (lambda=0.90)
│   │   ├── persistent_store.py           # Disk-backed B-Tree storage on NVMe SSD
│   │   └── engine.py                     # DecisionCertifiedHornSolver (Analytical, BnB, Fallback)
│   ├── grounding/                        # Visual grounding and graph attention modules
│   │   ├── __init__.py
│   │   ├── landmark_gate.py              # Visual landmark gate, temperature scaling & calibration
│   │   └── vc_gat.py                     # Degree-debiased attention (gamma=0.40) & hub dampening
│   ├── conformal/                        # Conformal risk control (Propositions 5 & 6)
│   │   ├── __init__.py
│   │   └── risk_control.py               # Distribution-free finite-sample risk calibration
│   ├── captioning/                       # Epistemic prompt generation
│   │   ├── __init__.py
│   │   └── prompt_builder.py             # Assert / Hedge / Omit prompt compilation
│   └── evaluation/                       # Statistical testing and metric computation
│       ├── __init__.py
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
├── data/                                 # Curated benchmark datasets and knowledge graphs
│   ├── fkg_per_kb/                       # Fused heritage knowledge graphs (Hanoi & HCMC)
│   │   ├── fkg_global_merged.json        # Merged Hanoi base knowledge graph (13,553 facts)
│   │   ├── fkg_van_mieu.json             # Temple of Literature sub-graph
│   │   ├── fkg_hoan_kiem.json            # Hoan Kiem Lake sub-graph
│   │   └── fkg_tran_quoc.json            # Tran Quoc Pagoda sub-graph
│   ├── caption_dataset/                  # Benchmark captions and evaluation splits
│   │   ├── generated_captions_test_full_1119.jsonl
│   │   ├── generated_captions_val.jsonl
│   │   └── generated_captions_train.jsonl
│   ├── unified_coco_dataset/             # Grounding annotations and visual features
│   └── scene_conditioned_gold/           # Gold-standard scene-conditioned Horn evidence
│
├── figures/                              # Publication vector PDF and high-res PNG figures
│   ├── Figure1_overall_architecture.png  # System architecture diagram
│   ├── Figure2_theory_certification_dependency.pdf # Early stopping dependency envelope
│   └── Figure3_conformal_risk_coverage.pdf # Conformal risk-coverage calibration curve
│
├── tables/                               # Generated publication LaTeX table snippets
│   ├── table3_certified_horn.tex
│   ├── table4_derivation_depth.tex
│   ├── table5_vcgat_ranking.tex
│   ├── table6_landmark_calibration.tex
│   ├── table7_captioning_e1_e6.tex
│   ├── table11_cross_domain_hcmc.tex
│   └── prop2c_hub_scaling.tex
│
├── results/                              # Benchmark CSV summaries and raw metric logs
└── legacy_archive/                       # Safely archived exploratory and historical scripts
```

---

## 3. Paper-to-Code Mapping Matrix

| Manuscript Target | Theoretical / Empirical Claim | Code Module / Reproduction Script | Primary Output Artifact |
|:---|:---|:---|:---|
| **Section 5.1 / Prop 2** | Monotone Lower & Upper Bounds under Assumption A1 | `src/reasoning/bounds.py`<br>`tests/test_theory_properties.py` | Validated over 10,000 Horn programs (0 violations) |
| **Section 5.2 / Cor 1** | Certified Early Stopping with Provable Decision Preservation | `src/reasoning/engine.py`<br>`experiments/reproduce_table3_certified_horn.py` | `tables/table3_certified_horn.tex`<br>`results/table3_metrics.json` |
| **Section 5.3 / Prop 2b** | Perceptual Lower Bound & Ungrounded Hallucination Immunity | `src/grounding/vc_gat.py`<br>`experiments/reproduce_proposition_2b_2c_stress.py` | `results/prop2b_stress_results.json` (0/5 hallucinations) |
| **Section 5.4 / Prop 2c** | Degree-Debiased Attention Mass Bound ($\mathcal{O}(d_j^{1-\gamma})$) | `src/grounding/vc_gat.py`<br>`tests/test_proposition_2b_2c_stress.py` | `tables/prop2c_hub_scaling.tex` (Sublinear hub scaling) |
| **Section 5.5 / Prop 5-6**| Finite-Sample Conformal Risk Control ($\mathbb{E}[R(\tau)] \le \alpha$) | `src/conformal/risk_control.py`<br>`tests/test_proposition_5_6_conformal_risk.py` | `results/conformal_risk_audit.json` |
| **Section 6.2 / Table 3** | Reasoning Complexity, RAM time vs. Disk Speedup | `experiments/reproduce_table3_certified_horn.py`<br>`experiments/reproduce_persistent_storage_benchmark.py` | 66.94% evaluations avoided, $1.72\times$ disk speedup |
| **Section 6.3 / Table 4** | Derivation Depth & Residual Interval Contraction | `experiments/reproduce_table4_derivation_depth.py` | `tables/table4_derivation_depth.tex` |
| **Section 6.4 / Table 5** | VC-GAT Evidence Ranking (nDCG@5, P@5, R@5) | `experiments/reproduce_table5_vcgat_ranking.py` | `tables/table5_vcgat_ranking.tex` ($+0.0955$, $p < 0.001$) |
| **Section 6.5 / Table 6** | Landmark Calibration & Conformal Thresholds | `experiments/reproduce_table6_landmark_calibration.py` | `tables/table6_landmark_calibration.tex` |
| **Section 6.6 / Table 7** | End-to-End Captioning Evaluation (E1–E6) | `experiments/reproduce_table7_end2end_captioning.py` | `tables/table7_captioning_e1_e6.tex` |
| **Section 6.7 / Table 11**| Cross-Domain Zero-Shot Transfer on HCMC-300 | `experiments/reproduce_table11_cross_domain_hcmc.py` | `tables/table11_cross_domain_hcmc.tex` |
| **Figures 2 & 3** | Early Stopping Envelope & Conformal Coverage Curves | `experiments/reproduce_figure2_certification_dependency.py`<br>`experiments/reproduce_figure3_conformal_risk.py` | `figures/Figure2_theory_certification_dependency.pdf`<br>`figures/Figure3_conformal_risk_coverage.pdf` |

---

## 4. Verification and Reproducibility Standards

1. **Deterministic Reproducibility:** All seeds are fixed (`seed = 42`) in `config.yaml`.
2. **Paired Cluster Bootstrap:** Clustered resampling ($B = 10{,}000$) operates at the parent-video level (60 clusters in Hanoi-500, 36 clusters in HCMC-300) to account for intra-video correlation.
3. **Hardware Independence:** All theoretical checks, bounding algorithms, and evaluation scripts execute deterministically on CPU in $< 30$ seconds.
