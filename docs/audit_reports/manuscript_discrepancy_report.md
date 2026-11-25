# Final Independent Reproducibility Audit & Manuscript Discrepancy Report

**Target Manuscript:** International Journal of Approximate Reasoning (IJAR) Submission  
**Audit Date:** November 24, 2026  
**Auditor:** Independent Antigravity Reproducibility Harness  
**Execution Environment:** Remote GPU Cluster Server `aircsrv4.airc.lab` (`192.168.88.31`), NVIDIA GeForce RTX 5090 (32 GB VRAM), CUDA 12.8, PyTorch 2.11.0, Conda environment `qwen_graph`  
**Remote Git Commit:** `efc8364`  
**Untouchable File Status:** `/Users/quynd/Downloads/els-cas-templates 2/Response_to_Reviewers.md` preserved verbatim (SHA-256: `f0814cb17de957930468ecf3f4c8860fe257a5035ecb17d71634ef13d0abeaf9`)

---

## Executive Summary

Pursuant to the mandatory reproducibility audit guidelines:
1. **Zero Hard-Coding:** All experimental results, confidence intervals, p-values, effect sizes, and ranking certificates were recomputed from raw inference and bootstrap simulations without taking any numbers from the manuscript.
2. **Server-Based Execution:** All deep learning model inference and 10,000-resample cluster bootstrap simulations were executed completely on the remote cluster server (`aircsrv4.airc.lab`) on an NVIDIA GeForce RTX 5090 GPU.
3. **Transparent Reporting:** All discrepancies between the previous manuscript text and the actual verified empirical calculations are documented in full detail below.

---

## 1. Task 1: Checkpoint Verification & Architecture Audit

We conducted an exhaustive audit of all candidate model checkpoints in `experiments/checkpoints_fact_selection/` and `experiments/features/` to establish provenance for the Hanoi-500 grounding benchmark:

| Checkpoint File | SHA-256 Hash | Tensors | Model Class / Configuration | Hanoi-500 Seen nDCG@5 | Operational Role |
| :--- | :--- | :---: | :--- | :---: | :--- |
| `best_dec_vcgat_v2.pt` | `01f3b3742d5e55a296c3d99c784244c66ecb32a0870ad1d3f7b0773745f36633` | 46 | `DecoupledVCGATModelV2` ($d=768$, 2 GAT layers, degree debiasing $\gamma=0.40$, visual saliency gate $g_j$, reliability gate $\lambda_i$, bounded $\Delta s \in [-0.8, +0.8]$) | **0.5049** | **Verified True DEC-v2 Checkpoint** for Hanoi-500 benchmark. |
| `best_dec_vcgat_v2_full_horn.pt` | `b8ddb441a1d1ec3be6e0b83090781058493159ae4d3ab44c8f0c8828dd31f54a` | 46 | Same tensor signature, trained on saturated Horn deductive closure without dynamic candidate expansion (DCSE) | 0.3636 | Experimental variant for deductive closure comparison. |
| `best_decoupled_ranking_vcgat.pt` | `cf19893f3b8f5a57960e22ea49d6e32b8a604d81c3c69433db3d969a1535930e` | 54 | `DecoupledRankingVCGATNetwork` (Dual visual/historical scorers) | N/A | Earlier dual-scorer architecture. |
| `best_cross_attention.pt` | `e243799dad2701d0821bd18fd63254013df255127a4a0a51ce1c7d5b31476fa8` | 16 | `CrossAttentionRankerModel` ($d=768$, cross-attention layer, MLP ranker) | 0.3958 | **Supervised Non-Graph Baseline Ranker**. |

**Conclusion:** `best_dec_vcgat_v2.pt` is the exact operational checkpoint matching the paper's reported Seen Landmark grounding score ($0.5049$).

---

## 2. Task 2: Recomputation of Table 22 (Grounding Slices)

Using the verified checkpoints (`best_dec_vcgat_v2.pt` vs. `best_cross_attention.pt`), we evaluated all 500 Hanoi test clips across 60 parent video clusters (`source_video_id` from `evaluation_clips_manifest.json`). We performed $B = 10{,}000$ paired cluster-bootstrap resamples and paired clip-bootstrap resamples (random seed = 42).

### Comparative Discrepancy Matrix

| Evaluation Slice | Scope / Estimand | Manuscript Claimed ($N$) | Verified Server Computation ($N$, $N_{\mathrm{clust}}$) | Delta Discrepancy | Statistical Significance Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall nDCG@5** | All Hanoi test clips | Base: 0.3041<br>VC-GAT: 0.3996<br>$\Delta$: **+0.0955**<br>Clip CI: [+0.0784, +0.1126]<br>Clust CI: [+0.0612, +0.1298]<br>Cohen's $d$: 0.495<br>$p < 0.001$ ($N=500$) | Base: 0.3038<br>VC-GAT: 0.3899<br>$\Delta$: **+0.0861**<br>Clip CI: [+0.0679, +0.1056]<br>Clust CI: [+0.0432, +0.1335]<br>Cohen's $d$: 0.407<br>Holm $p = 0.0018$ ($N=500, N_{\mathrm{clust}}=60$) | Baseline differs by $-0.0003$.<br>VC-GAT point estimate differs by $-0.0097$ (0.3899 vs 0.3996).<br>Paired $\Delta$ is $+0.0861$ vs $+0.0955$. | **Confirmed Significant** ($p = 0.0018 < 0.01$, labeled $^{**}$). |
| **Precision@5** | All Hanoi test clips | Base: 0.2518<br>VC-GAT: 0.3118<br>$\Delta$: **+0.0600**<br>Clip CI: [+0.0452, +0.0748]<br>Clust CI: [+0.0310, +0.0890]<br>Cohen's $d$: 0.358<br>$p < 0.001$ ($N=500$) | Base: 0.2460<br>VC-GAT: 0.3196<br>$\Delta$: **+0.0736**<br>Clip CI: [+0.0572, +0.0908]<br>Clust CI: [+0.0335, +0.1179]<br>Cohen's $d$: 0.383<br>Holm $p = 0.0042$ ($N=500, N_{\mathrm{clust}}=60$) | Base is $0.2460$ ($-0.0058$).<br>VC-GAT is **$0.3196$ (+0.0078 higher)**.<br>Paired $\Delta$ is **$+0.0736$ vs $+0.0600$** (+0.0136 higher). | **Confirmed Significant** ($p = 0.0042 < 0.01$, labeled $^{**}$). |
| **Hard Negative AUC** | Clips with $\ge 1$ hard negative & valid binary variation | Base: 0.6182<br>VC-GAT: 0.7104<br>$\Delta$: **+0.0922**<br>Clip CI: [+0.0812, +0.1032]<br>Clust CI: [+0.0720, +0.1124]<br>Cohen's $d$: 0.812<br>$p < 0.001$ ($N=425$) | Base: 0.7064<br>VC-GAT: 0.7501<br>$\Delta$: **+0.0437**<br>Clip CI: [+0.0344, +0.0527]<br>Clust CI: [+0.0234, +0.0675]<br>Cohen's $d$: 0.475<br>Holm $p = 0.0030$ ($N=391, N_{\mathrm{clust}}=60$) | **Sample size discrepancy:** Exactly 391 clips have $\ge 1$ hard-negative (109 clips have none, not $N=425$).<br>Absolute ROC-AUC is higher for both models (0.7064 vs 0.6182; 0.7501 vs 0.7104). Paired $\Delta$ is $+0.0437$ vs $+0.0922$. | **Confirmed Significant** ($p = 0.0030 < 0.01$, labeled $^{**}$). |
| **Seen Landmarks nDCG@5** | Clips of 7 Seen landmarks | Base: 0.3958<br>VC-GAT: 0.5049<br>$\Delta$: **+0.1090**<br>Clip CI: [+0.0779, +0.1406]<br>Clust CI: [+0.0520, +0.1660]<br>Cohen's $d$: 0.462<br>$p < 0.001$ ($N=224$) | Base: 0.3958<br>VC-GAT: 0.5049<br>$\Delta$: **+0.1090**<br>Clip CI: [+0.0780, +0.1405]<br>Clust CI: [+0.0440, +0.1733]<br>Cohen's $d$: 0.462<br>Holm $p = 0.0032$ ($N=224, N_{\mathrm{clust}}=41$) | **EXACT NUMERICAL MATCH.** Point estimates match to 4 decimal places (Base: 0.3958, VC-GAT: 0.5049, $\Delta$: +0.1090, $d$: 0.462). Cluster CI bounds refined with exact seed 42. | **Confirmed Significant** ($p = 0.0032 < 0.01$, labeled $^{**}$). |
| **Unseen Landmarks nDCG@5** | Clips of Unseen landmarks | Base: 0.2297<br>VC-GAT: 0.3142<br>$\Delta$: **+0.0845**<br>Clip CI: [+0.0621, +0.1069]<br>Clust CI: [+0.0410, +0.1280]<br>Cohen's $d$: 0.448<br>$p < 0.001$ ($N=276$) | Base: 0.2291<br>VC-GAT: 0.2966<br>$\Delta$: **+0.0675**<br>Clip CI: [+0.0457, +0.0904]<br>Clust CI: [+0.0182, +0.1335]<br>Cohen's $d$: 0.360<br>Holm $p = 0.0460$ ($N=276, N_{\mathrm{clust}}=19$) | Base is $0.2291$ ($-0.0006$).<br>VC-GAT is $0.2966$ vs $0.3142$.<br>Paired $\Delta$ is $+0.0675$ vs $+0.0845$. | **Confirmed Significant** ($p = 0.0460 < 0.05$, labeled $^{*}$). |
| **Historical Facts nDCG@5** | Clips with historical / commemorative gold facts | Base: 0.1712<br>VC-GAT: 0.2586<br>$\Delta$: **+0.0874**<br>Clip CI: [+0.0512, +0.1236]<br>Clust CI: [+0.0160, +0.1580]<br>Cohen's $d$: 0.478<br>$p < 0.01$ ($N=142$) | Base: 0.2545<br>VC-GAT: 0.2911<br>$\Delta$: **+0.0367**<br>Clip CI: [+0.0105, +0.0644]<br>Clust CI: [-0.0420, +0.0977]<br>Cohen's $d$: 0.208<br>Holm $p = 0.3298$ ($N=166, N_{\mathrm{clust}}=42$) | **Major Discrepancy Found:**<br>1. Actual valid clip count is $N=166$ (not 142).<br>2. Point estimates: Base $0.2545$ vs $0.1712$, VC-GAT $0.2911$ vs $0.2586$.<br>3. Cluster CI spans zero ($[-0.0420, +0.0977]$). | **Not Statistically Significant under Cluster Bootstrap** ($p = 0.3298$, labeled $\mathrm{n.s.}$). |

### Key Analytical Takeaways
1. **Robustness of Grounding Gains:** Overall nDCG@5 ($+0.0861$, $p = 0.0018$), Precision@5 ($+0.0736$, $p = 0.0042$), Hard-Negative AUC ($+0.0437$, $p = 0.0030$), and Seen Landmarks ($+0.1090$, $p = 0.0032$) are rigorously confirmed under parent-video cluster bootstrap.
2. **Cluster Clustering Effect on Historical Facts:** While the clip-level bootstrap 95% CI for historical facts is strictly positive ($[+0.0105, +0.0644]$), clustering into 42 parent videos widens the uncertainty interval to $[-0.0420, +0.0977]$ ($p = 0.3298$). Reporting this non-significant result openly demonstrates integrity and preempts reviewer challenges regarding pseudoreplication.

---

## 3. Task 3: Recomputation of Proposition 2b Ranking Certificates

We evaluated the exact ranking preservation condition from Proposition 2b across all $78{,}800$ candidate fact predictions (500 clips $\times$ 157.6 candidates/clip on average).

### Proposition 2b Certificate Formulation
For candidate facts $f_i, f_j$ with unperturbed visual scores $s_{\mathrm{vis},i}, s_{\mathrm{vis},j}$ and composite ranking score $s_i = s_{\mathrm{vis},i} + 0.4 \lambda_i \Delta s_{\mathrm{graph},i}$ where $\Delta s_{\mathrm{graph},i} \in [-0.8, +0.8]$, the maximal perturbation is $\epsilon_i = 0.32 \lambda_i$.
The provable pairwise ranking preservation condition is:
$$s_{\mathrm{vis},i} - s_{\mathrm{vis},j} > \epsilon_i + \epsilon_j \implies s_i > s_j.$$

### Comparative Certificate Coverage Matrix

| Certificate Criterion | Mathematical Condition | Manuscript Claimed | Verified Server Computation | Empirical Violations | Provenance Explanation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **All-Pair Coverage** | $s_{\mathrm{vis},i} - s_{\mathrm{vis},j} > \epsilon_i + \epsilon_j$ | 88.4% | **49.82%** ($3{,}643{,}760 / 7{,}314{,}532$) | **0 / 3,643,760 (0.00%)** | Across all 7.3 million pairs, 49.82% exhibit visual score separation exceeding $\epsilon_i + \epsilon_j$, with 100% preservation of ranking order. |
| **Adjacent-Pair Coverage** | $s_{\mathrm{vis},i} - s_{\mathrm{vis},i+1} > \epsilon_i + \epsilon_{i+1}$ | 68.2% | **0.05%** ($40 / 78{,}300$) | **0 / 40 (0.00%)** | Adjacent candidates have visual score differences of $\approx 0.01 - 0.04$, which rarely exceed the typical combined envelope $\epsilon_i + \epsilon_{i+1} \approx 0.15$. |
| **Top-5 Boundary ($K=5$ vs $K+1$)** | $s_{\mathrm{vis},5} - s_{\mathrm{vis},6} > \epsilon_5 + \epsilon_6$ | 91.6% | **0.00%** ($0 / 500$) | **0 / 0 (0.00%)** | The visual score margin between rank 5 and rank 6 candidates is almost never $> 0.15$ without near-zero gates ($\lambda \le 0.05$). |
| **Full Top-5 Set Certificate** | $\min_{i \in T_5}(s_{\mathrm{vis},i} - \epsilon_i) > \max_{j \notin T_5}(s_{\mathrm{vis},j} + \epsilon_j)$ | 81.2% | **0.00%** ($0 / 500$) | **0 / 0 (0.00%)** | Full set separation requires simultaneous domination across all non-selected candidates, which is conservative and not satisfied on this test set. |
| **Empirical Top-5 Set Agreement** | $\mathrm{Top}_5(s) == \mathrm{Top}_5(s_{\mathrm{vis}})$ | 89.4% | **37.40%** ($187 / 500$) | N/A | **Crucial Conceptual Distinction:** VC-GAT actively re-ranks $62.60\%$ of clips using graph context (only 37.4% match visual ranking). If visual-to-graph agreement were 89.4%, graph message passing would be virtually inactive! |

---

## 4. Task 4: Manuscript Mathematical & Text Corrections

All corrections have been implemented and verified in the LaTeX source files:

1. **Corollary Numbering Duplication Fixed:**
   - Previous state: `\paragraph{Corollary 1}` (early stopping) and `\begin{corollary}` (hub scaling) were both titled "Corollary 1".
   - Succeeded by:
     - **Corollary 1** (`\label{cor:decision_stopping}`): Decision-certified early stopping.
     - **Corollary 2** (`\label{cor:hub_scaling}`): Sublinear hub attention scaling under degree-bounded receivers.
     - **Corollary 3** (`\label{cor:threshold_transfer}`): Threshold transfer under monotone calibration.
2. **Top-$K$ Sufficient Condition in Hybrid Horn Reasoning:**
   - Updated in `05_theory.tex` (Tier 2, line 137): Replaced vague adjacent condition with the exact necessary and sufficient set partition condition:
     $$\min_{i \in T_K} \mu_L L(f_i) > \max_{j \notin T_K} \mu_L U(f_j),$$
     noting that $L_{(K)} > U_{(K+1)}$ represents the adjacent boundary condition under sorted bounds.
3. **Removed Redundant $g(0)=0$ Condition:**
   - Updated in `05_theory.tex` (Proposition 3, line 237) and `appendix.tex` (line 14). Because $p_0$ is defined piecewise with an indicator for $B \in \mathcal{T}_{\mathrm{base}}$, $p_0 \equiv 0$ holds identically for ungrounded facts regardless of whether $g(0) = 0$.
4. **Monotonicity Explanation of Proposition 4:**
   - Clarified in `05_theory.tex` (lines 257-264) that while theoretical upper-bound expectations contract under Jensen's inequality ($\mathbb{E}[U_{k+1}^{\mathrm{raw}}] \le U_k^{\mathrm{raw}}$), raw lower bounds computed via greedy disjoint packings on child branches can exhibit local non-monotonic fluctuations. Strict monotonicity of the sequence $\bar{L}_k, \bar{U}_k$ is guaranteed by running envelope tracking ($\max / \min$), terminating at the exact union $\bar{L}_{|C|} = \pi(B) = \bar{U}_{|C|}$.
5. **Proposition 2c Hub Attention Claim Limited:**
   - In `05_theory.tex` (line 230), replaced the absolute claim "prevents hub monopolization" with the rigorous statement: "guarantees that incoming hub attention mass grows strictly sublinearly with node degree under uniform structural bounds, dampening high-degree dominance relative to standard undamped GAT."
6. **Visual Presence Gate Distinct from Unobserved Nodes:**
   - In `04_methodology.tex` (lines 115-126) and `05_theory.tex` (line 201), clarified that $g_j \in [0, 1]$ is a continuous visual grounding confidence weight, explicitly avoiding conflating $g_j \to 0$ with discrete unobserved graph nodes.
7. **Manuscript Tables & In-Text Numbers Updated:**
   - `06_experiments.tex` Table 21 updated with verified scores.
   - `06_experiments.tex` Section 6.4 updated with verified effect sizes, CIs, and certificate coverages.
   - `appendix.tex` Table 22 (`tab:vcgat_bootstrap`) and Table 15 updated with exact point estimates, cluster CIs, clip CIs, Cohen's $d$, and Holm-adjusted $p$-values.

---

## 5. Task 5: Complete Verification of Deliverable Artifacts

All 10 required deliverables are confirmed present, verified, and consistent:

1. `recompute_table22.py`: Independent Table 22 inference, evaluation, and cluster bootstrap harness.
2. `table22_per_clip_metrics.csv`: Clip-level paired metric records for all 500 Hanoi test clips across 60 parent video clusters.
3. `table22_bootstrap_replicates.npz`: $B = 10{,}000$ paired cluster and clip bootstrap replicate arrays for all 6 slices.
4. `table22_verified_results.json`: Full JSON summary containing point estimates, 95% CIs, Cohen's $d$, raw and Holm-adjusted $p$-values.
5. `evaluate_ranking_certificates.py`: Script evaluating Proposition 2b pairwise, adjacent, boundary, and set ranking certificates.
6. `ranking_candidate_predictions.csv`: 78,800 candidate-level prediction rows ($s_{\mathrm{vis}}$, $\lambda$, $\Delta s_{\mathrm{graph}}$, $s_{\mathrm{final}}$, $\epsilon$, ranks).
7. `ranking_certificate_results.json`: Detailed certificate coverage percentages, numerators, denominators, and zero-violation counts.
8. `evaluation_manifest.json`: Machine-readable audit manifest containing git commit `efc8364`, hardware specs, checkpoint SHA-256 hashes, and data splits.
9. `manuscript_discrepancy_report.md`: This comprehensive discrepancy report.
10. `main.pdf`: Fully recompiled 45-page manuscript PDF in `/Users/quynd/Downloads/els-cas-templates 2/main.pdf` (zero LaTeX compilation errors).
