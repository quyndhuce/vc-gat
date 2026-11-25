#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/rerun_protocol_x4b_server_verification.py
=============================================================================
Protocol X4b Complete Re-execution and Loss Verification (Server Version)
=============================================================================
Mục đích:
  - Chạy lại 100% Protocol X4b trên server với 200 random 50/50 clip splits.
  - Kiểm tra tính nhất quán đại số với Mệnh đề 5:
      lambda_H = kappa * rho * lambda_A
      q_M = lambda_H / (rho + lambda_H)
      q_H = (lambda_A - lambda_H) / ((1 - rho) + (lambda_A - lambda_H))
      Điều kiện 3 miền quyết định: lambda_H < rho * lambda_A (luôn thoả vì kappa < 1).
  - So sánh trực tiếp Theoretical Loss vs Empirical Loss vs Heuristic Loss
    trên toàn bộ 27 cấu hình (3x3x3 grid) và 9 cấu hình đại diện của Table 16.
  - Xuất file CSV và JSON raw outputs để làm bằng chứng khoa học cho Reviewer.
=============================================================================
"""

import csv
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("ProtocolX4bVerification")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_x4_x5_evaluation import extract_audited_claims

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_DIR / "X4b_rerun_server_verified.csv"
OUT_JSON = RESULTS_DIR / "X4b_rerun_server_verified.json"

SEED = 42
N_SPLITS = 200

def fast_loss(
    c: np.ndarray, y: np.ndarray, th_M: float, th_H: float,
    rho: float, lam_A: float, lam_H: float
) -> float:
    assert_mask = (c >= th_H)
    hedge_mask = (c >= th_M) & (c < th_H)
    omit_mask = (c < th_M)
    loss = (
        np.sum((y == 1) & hedge_mask) * (1.0 - rho) +
        np.sum((y == 1) & omit_mask) * 1.0 +
        np.sum((y == 0) & assert_mask) * (1.0 + lam_A) +
        np.sum((y == 0) & hedge_mask) * (1.0 + lam_H) +
        np.sum((y == 0) & omit_mask) * 1.0
    )
    return float(loss / len(c))


def run_verification():
    LOGGER.info("=" * 80)
    LOGGER.info("PROTOCOL X4b SERVER VERIFICATION & RE-EXECUTION")
    LOGGER.info("=" * 80)
    t0 = time.time()

    LOGGER.info("Step 1: Extracting audited claims (N=644 cohort)...")
    claims = extract_audited_claims()
    N = len(claims)
    LOGGER.info("Loaded %d audited claims.", N)

    c_arr = np.array([c["c_prod"] for c in claims], dtype=np.float64)
    y_arr = np.array([c["y_correct"] for c in claims], dtype=int)

    rhos = [0.5, 0.7, 0.9]
    lambda_As = [2.0, 4.0, 8.0]
    kappas = [0.25, 0.50, 0.75]

    cand_th = np.linspace(0.05, 0.95, 19)
    cand_pairs = [(t_m, t_h) for t_m in cand_th for t_h in cand_th if t_h >= t_m]
    c_grid = np.linspace(0.0, 1.0, 1001)

    grid_results = []
    cfg_idx = 0

    LOGGER.info("Step 2: Sweeping 27 parameter configurations across %d splits...", N_SPLITS)
    for rho in rhos:
        for lam_A in lambda_As:
            for kappa in kappas:
                lam_H = kappa * rho * lam_A

                # Algebraic check of Proposition 5
                valid_condition = (lam_H < rho * lam_A)
                q_M = lam_H / (rho + lam_H)
                q_H = (lam_A - lam_H) / ((1.0 - rho) + (lam_A - lam_H))
                order_check = (0.0 < q_M < q_H < 1.0)

                theo_losses = []
                emp_losses = []
                manual_losses = []
                th_M_theos = []
                th_H_theos = []

                rng = np.random.RandomState(SEED)
                for _ in range(N_SPLITS):
                    perm = rng.permutation(N)
                    cal_idx = perm[:N // 2]
                    test_idx = perm[N // 2:]

                    c_cal, y_cal = c_arr[cal_idx], y_arr[cal_idx]
                    c_test, y_test = c_arr[test_idx], y_arr[test_idx]

                    # 1. Theoretical thresholds via isotonic inversion on calibration split
                    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                    iso.fit(c_cal, y_cal)
                    q_pred = iso.predict(c_grid)

                    m_cands = c_grid[q_pred >= q_M]
                    tm_theo = float(m_cands[0]) if len(m_cands) > 0 else 1.0
                    h_cands = c_grid[q_pred >= q_H]
                    th_theo = float(h_cands[0]) if len(h_cands) > 0 else 1.0
                    if th_theo < tm_theo:
                        th_theo = tm_theo

                    th_M_theos.append(tm_theo)
                    th_H_theos.append(th_theo)

                    # Theoretical loss on test split
                    l_theo = fast_loss(c_test, y_test, tm_theo, th_theo, rho, lam_A, lam_H)
                    theo_losses.append(l_theo)

                    # 2. Empirical 2D grid search on calibration split
                    best_l_cal = float("inf")
                    best_th_M, best_th_H = tm_theo, th_theo
                    for t_m, t_h in cand_pairs:
                        l_cal = fast_loss(c_cal, y_cal, t_m, t_h, rho, lam_A, lam_H)
                        if l_cal < best_l_cal:
                            best_l_cal = l_cal
                            best_th_M, best_th_H = t_m, t_h

                    # Evaluate optimal calibration thresholds on test split
                    l_emp = fast_loss(c_test, y_test, best_th_M, best_th_H, rho, lam_A, lam_H)
                    emp_losses.append(l_emp)

                    # 3. Fixed heuristic baseline (0.50, 0.80) on test split
                    l_man = fast_loss(c_test, y_test, 0.50, 0.80, rho, lam_A, lam_H)
                    manual_losses.append(l_man)

                mean_theo = float(np.mean(theo_losses))
                std_theo = float(np.std(theo_losses))
                mean_emp = float(np.mean(emp_losses))
                std_emp = float(np.std(emp_losses))
                mean_man = float(np.mean(manual_losses))
                std_man = float(np.std(manual_losses))
                gap = float(mean_theo - mean_emp)

                rec = {
                    "config_idx": cfg_idx,
                    "rho": rho,
                    "lambda_A": lam_A,
                    "kappa": kappa,
                    "lambda_H": round(lam_H, 4),
                    "valid_condition": valid_condition,
                    "order_check": order_check,
                    "q_M": round(q_M, 4),
                    "q_H": round(q_H, 4),
                    "th_M_theo_mean": round(float(np.mean(th_M_theos)), 4),
                    "th_H_theo_mean": round(float(np.mean(th_H_theos)), 4),
                    "loss_theo_mean": round(mean_theo, 4),
                    "loss_theo_std": round(std_theo, 4),
                    "loss_emp_mean": round(mean_emp, 4),
                    "loss_emp_std": round(std_emp, 4),
                    "loss_manual_mean": round(mean_man, 4),
                    "loss_manual_std": round(std_man, 4),
                    "gap_theo_emp": round(gap, 4),
                    "abs_gap": round(abs(gap), 4),
                }
                grid_results.append(rec)
                cfg_idx += 1

    # Save to CSV
    fieldnames = list(grid_results[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(grid_results)
    LOGGER.info("Saved verified results to CSV: %s", OUT_CSV)

    # Save to JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(grid_results, f, indent=2)
    LOGGER.info("Saved verified results to JSON: %s", OUT_JSON)

    # Analyze 9 Representative Configurations (Table 16)
    rep_indices = [0, 4, 8, 9, 13, 17, 18, 22, 26]
    rep_results = [grid_results[i] for i in rep_indices]

    max_gap_9 = max(r["abs_gap"] for r in rep_results)
    max_gap_27 = max(r["abs_gap"] for r in grid_results)

    print("\n" + "=" * 100)
    print("TABLE 16 RE-VERIFICATION REPORT (9 REPRESENTATIVE CONFIGURATIONS)")
    print("=" * 100)
    print(f"{'rho':4s} | {'lam_A':5s} | {'kappa':5s} | {'lam_H':5s} | {'q_M':5s} | {'q_H':5s} | {'th_M*':5s} | {'th_H*':5s} | {'Theo Loss':17s} | {'Emp Loss':17s} | {'Heur Loss':9s} | {'|Delta|':7s}")
    print("-" * 100)
    for r in rep_results:
        th_loss_str = f"{r['loss_theo_mean']:.3f} ± {r['loss_theo_std']:.3f}"
        emp_loss_str = f"{r['loss_emp_mean']:.3f} ± {r['loss_emp_std']:.3f}"
        print(f"{r['rho']:4.1f} | {r['lambda_A']:5.1f} | {r['kappa']:5.2f} | {r['lambda_H']:5.2f} | {r['q_M']:5.3f} | {r['q_H']:5.3f} | {r['th_M_theo_mean']:5.3f} | {r['th_H_theo_mean']:5.3f} | {th_loss_str:17s} | {emp_loss_str:17s} | {r['loss_manual_mean']:9.3f} | {r['abs_gap']:7.4f}")

    print("=" * 100)
    print(f"Max |Delta| across 9 Representative Configurations (Table 16): {max_gap_9:.4f}")
    print(f"Max |Delta| across ALL 27 Configurations in Full Grid         : {max_gap_27:.4f}")
    print("=" * 100)
    LOGGER.info("Protocol X4b Verification completed in %.2f seconds.", time.time() - t0)

if __name__ == "__main__":
    run_verification()
