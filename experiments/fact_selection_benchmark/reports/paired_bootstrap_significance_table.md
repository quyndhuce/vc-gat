# Paired Bootstrap 95% Confidence Interval & Statistical Significance Analysis
Comparing Proposed **R-VCGAT** against Strongest Baseline (**Cross-Attention Ranker**) on 800 Test Clips.
Bootstrap iterations: $B = 10,000$, confidence level: 95%, two-sided paired hypothesis tests.

| Metric / Evaluation Slice | N Clips | Cross-Attn (Baseline) | R-VCGAT (Proposed) | Paired Delta (Δ) | Relative Gain | 95% Bootstrap CI | Cohen's d | p-value (t-test) | p-value (Wilcoxon) | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall nDCG@5 (800 Clips)** | 800 | 0.2878 | 0.2673 | **-0.0206** | **-7.15%** | `[-0.0378, -0.0023]` | -0.081 | 2.2739e-02 | 6.0023e-02 | **p < 0.05 *** |
| **Seen Landmarks nDCG@5 (301 Clips)** | 224 | 0.3958 | 0.4346 | **+0.0388** | **+9.80%** | `[+0.0080, +0.0696]` | 0.167 | 1.3036e-02 | 1.5589e-03 | **p < 0.01 **** |
| **Unseen Landmarks nDCG@5 (499 Clips)** | 576 | 0.2458 | 0.2022 | **-0.0436** | **-17.76%** | `[-0.0649, -0.0221]` | -0.168 | 6.2832e-05 | 6.9606e-06 | **p < 0.001 ***** |
| **Precision@5 (800 Clips)** | 800 | 0.2518 | 0.2303 | **-0.0215** | **-8.54%** | `[-0.0358, -0.0068]` | -0.102 | 3.9636e-03 | 2.6310e-03 | **p < 0.01 **** |
| **Hard Negative AUC** | 676 | 0.6182 | 0.6929 | **+0.0747** | **+12.08%** | `[+0.0644, +0.0849]` | 0.548 | 1.7427e-40 | 1.1437e-47 | **p < 0.001 ***** |
| **Historical Facts nDCG@5** | 226 | 0.2500 | 0.2485 | **-0.0015** | **-0.59%** | `[-0.0376, +0.0338]` | -0.005 | 9.3535e-01 | 9.0137e-01 | **n.s.** |