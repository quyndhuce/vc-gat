# Decoupled Visual Grounding + Relational Evidence Branch with ListMLE Loss
Evaluated on 800 independent Test clips against Cross-Attention Baseline.

| Metric | Cross-Attention (Baseline) | Decoupled R-VCGAT + ListMLE | Delta (Δ) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Overall nDCG@5** | 0.2882 | **0.2519** | **-0.0363** | **-12.59%** |
| **Precision@5 (P@5)** | 0.2520 | **0.2225** | **-0.0295** | **-11.71%** |
| **Hard Negative AUC** | 0.6180 | **0.6574** | **+0.0394** | **+6.38%** |
| **Contrastive Accuracy (%)** | 56.1000% | **58.7942%** | **+2.6942** | **+4.80%** |
| **Seen Landmarks nDCG@5** | 0.4196 | **0.3816** | **-0.0380** | **-9.06%** |
| **Unseen Landmarks nDCG@5** | 0.2083 | **0.2015** | **-0.0068** | **-3.27%** |
| **Visual Facts nDCG@5** | 0.3512 | **0.3482** | **-0.0030** | **-0.86%** |
| **Historical Facts nDCG@5 (Target)** | 0.2245 | **0.1744** | **-0.0501** | **-22.32%** |