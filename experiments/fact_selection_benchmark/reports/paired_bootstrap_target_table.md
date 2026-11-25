# Paired Bootstrap 95% Confidence Interval & Statistical Significance Analysis
Directly evaluates the **+0.0057 nDCG@5** margin: Proposed **R-VCGAT** vs. **Cross-Attention Ranker** on 800 Test Clips.
Bootstrap iterations: $B = 10,000$, confidence level: 95%, two-sided paired hypothesis tests.

| Metric / Evaluation Slice | N Clips | Cross-Attn (Baseline) | R-VCGAT (Proposed) | Paired Delta (Δ) | Relative Gain | 95% Bootstrap CI | Cohen's d | p-value (t-test) | p-value (Wilcoxon) | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall nDCG@5 (800 Clips)** | 800 | 0.2878 | 0.2595 | **-0.0283** | **-9.84%** | `[-0.0441, -0.0122]` | -0.124 | 4.9856e-04 | 1.7245e-03 | **p < 0.001 ***** |
| **Seen Landmarks nDCG@5 (301 Clips)** | 224 | 0.3958 | 0.3839 | **-0.0119** | **-3.01%** | `[-0.0457, +0.0210]` | -0.047 | 4.7983e-01 | 9.5607e-01 | **n.s.** |
| **Unseen Landmarks nDCG@5 (499 Clips)** | 576 | 0.2458 | 0.2111 | **-0.0347** | **-14.11%** | `[-0.0528, -0.0165]` | -0.158 | 1.6272e-04 | 5.7492e-05 | **p < 0.001 ***** |
| **Precision@5 (800 Clips)** | 800 | 0.2518 | 0.2395 | **-0.0123** | **-4.87%** | `[-0.0260, +0.0015]` | -0.062 | 7.9148e-02 | 1.8339e-01 | **n.s.** |
| **Hard Negative AUC** | 676 | 0.6182 | 0.7021 | **+0.0839** | **+13.57%** | `[+0.0728, +0.0948]` | 0.566 | 1.1242e-42 | 4.3225e-48 | **p < 0.001 ***** |
| **Historical Facts nDCG@5** | 226 | 0.2500 | 0.1740 | **-0.0759** | **-30.38%** | `[-0.1121, -0.0397]` | -0.274 | 5.4330e-05 | 2.7434e-06 | **p < 0.001 ***** |