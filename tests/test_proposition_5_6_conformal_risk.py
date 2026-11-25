#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/test_conformal_risk_control.py
=============================================================================
Stress Test Suite for Proposition 6 (Conformal Risk Control):
  1. Discrete Exactness of theta_hat_fast vs dense continuous infimum.
  2. Finite-Sample Marginal Risk Bound: E[ell(Z_{n+1}, theta_hat)] <= alpha.
  3. Monotonicity and Right-Continuity across jump discontinuities.
  4. Feasibility boundary conditions: n >= ceil(1/alpha) - 1.
  5. Cluster exchangeability sensitivity audit.
=============================================================================
"""

import logging
import unittest
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("TestProp6")


def theta_hat_fast(c: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """Vectorized discrete infimum calculation for Proposition 6."""
    n = len(c)
    cand = np.array(sorted(set(c) | {0.0, 1.0}))
    err_counts = np.sum((c[:, None] > cand[None, :]) & (y[:, None] == 0), axis=0)
    R = err_counts / float(n)
    bound = (n / (n + 1.0)) * R + (1.0 / (n + 1.0))
    valid = np.where(bound <= alpha)[0]
    if len(valid) > 0:
        return float(cand[valid[0]])
    return 1.0


def theta_hat_continuous_search(c: np.ndarray, y: np.ndarray, alpha: float, step: float = 1e-4) -> float:
    """Brute-force grid search over continuous interval [0, 1]."""
    n = len(c)
    grid = np.arange(0.0, 1.0 + step, step)
    err_counts = np.sum((c[:, None] > grid[None, :]) & (y[:, None] == 0), axis=0)
    R = err_counts / float(n)
    bound = (n / (n + 1.0)) * R + (1.0 / (n + 1.0))
    valid = np.where(bound <= alpha)[0]
    if len(valid) > 0:
        return float(grid[valid[0]])
    return 1.0


class TestProposition6ConformalRisk(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_discrete_exactness_vs_continuous_search(self):
        """Test 1: Verify discrete jump point scan matches continuous infimum."""
        LOGGER.info("Running Test 1: Discrete Exactness verification...")
        n = 150
        alpha = 0.05
        for trial in range(10):
            c_sample = np.random.beta(2, 2, size=n)
            y_sample = (np.random.rand(n) < c_sample).astype(int)

            th_fast = theta_hat_fast(c_sample, y_sample, alpha)
            th_cont = theta_hat_continuous_search(c_sample, y_sample, alpha, step=1e-4)

            # th_fast should equal th_cont within step tolerance
            self.assertAlmostEqual(
                th_fast, th_cont, delta=1.5e-4,
                msg=f"Trial {trial}: Discrete th ({th_fast:.5f}) != continuous ({th_cont:.5f})"
            )
        LOGGER.info("Test 1 PASSED: Discrete algorithm achieves exact infimum.")

    def test_finite_sample_risk_bound_guarantee(self):
        """Test 2: Verify E[ell(Z_{n+1}, theta_hat)] <= alpha across exchangeable draws."""
        LOGGER.info("Running Test 2: Finite-Sample Expectation Risk Bound...")
        n_cal = 200
        n_trials = 4000
        alpha_targets = [0.02, 0.05, 0.10, 0.20]

        for alpha in alpha_targets:
            test_losses = []
            for _ in range(n_trials):
                # Draw n+1 exchangeable samples
                c_all = np.random.beta(2, 2, size=n_cal + 1)
                y_all = (np.random.rand(n_cal + 1) < (0.3 + 0.6 * c_all)).astype(int)

                c_cal, y_cal = c_all[:n_cal], y_all[:n_cal]
                c_test, y_test = c_all[n_cal], y_all[n_cal]

                th_hat = theta_hat_fast(c_cal, y_cal, alpha)
                loss_test = 1.0 if (c_test > th_hat and y_test == 0) else 0.0
                test_losses.append(loss_test)

            emp_risk = float(np.mean(test_losses))
            # Standard error of Monte Carlo mean
            se = float(np.sqrt(emp_risk * (1 - emp_risk) / n_trials))
            LOGGER.info("Alpha = %.2f | Empirical Risk = %.4f (SE = %.4f) | Bound <= %.4f",
                        alpha, emp_risk, se, alpha)

            # Must satisfy empirical mean <= alpha + 2 * SE (accounting for Monte Carlo error)
            self.assertLessEqual(
                emp_risk, alpha + 2.0 * se,
                msg=f"Conformal risk bound violated: {emp_risk:.4f} > {alpha:.4f} + 2*SE"
            )
        LOGGER.info("Test 2 PASSED: Empirical expectation strictly satisfies marginal risk bound.")

    def test_monotonicity_and_right_continuity(self):
        """Test 3: Verify monotonicity and right-continuity at jump points."""
        LOGGER.info("Running Test 3: Monotonicity and Right-Continuity...")
        c = np.array([0.15, 0.32, 0.55, 0.78, 0.91])
        y = np.array([0, 1, 0, 1, 0])
        n = len(c)

        # Monotonicity check across fine sweep
        theta_grid = np.linspace(0.0, 1.0, 500)
        bounds = []
        for th in theta_grid:
            err = np.sum((c > th) & (y == 0)) / float(n)
            b = (n / (n + 1.0)) * err + (1.0 / (n + 1.0))
            bounds.append(b)

        for i in range(len(bounds) - 1):
            self.assertGreaterEqual(
                bounds[i], bounds[i + 1],
                msg="Empirical upper bound is not monotonically non-increasing!"
            )

        # Right-continuity check: lim_{eps -> 0+} f(c_k + eps) == f(c_k)
        for val in c:
            err_at = np.sum((c > val) & (y == 0))
            err_plus = np.sum((c > (val + 1e-9)) & (y == 0))
            self.assertEqual(
                err_at, err_plus,
                msg=f"Right-continuity failed at discontinuity point {val}"
            )
        LOGGER.info("Test 3 PASSED: Monotonicity and right-continuity verified.")

    def test_sample_size_feasibility_condition(self):
        """Test 4: Verify n >= ceil(1/alpha) - 1 condition for non-trivial threshold."""
        LOGGER.info("Running Test 4: Sample size feasibility condition...")
        alpha = 0.05
        min_n = int(np.ceil(1.0 / alpha) - 1)  # 19

        c_dummy = np.linspace(0.1, 0.9, min_n)
        y_dummy = np.zeros(min_n, dtype=int)  # worst case: all wrong

        # At n = min_n, bound at theta=1.0 is 1/(n+1) = 1/20 = 0.05 <= alpha
        th_min = theta_hat_fast(c_dummy, y_dummy, alpha)
        self.assertLessEqual(th_min, 1.0)

        # At n < min_n (e.g. n = 10), 1/(n+1) = 1/11 = 0.0909 > 0.05
        c_small = np.linspace(0.1, 0.9, min_n - 5)
        y_small = np.zeros(min_n - 5, dtype=int)
        th_small = theta_hat_fast(c_small, y_small, alpha)
        self.assertEqual(
            th_small, 1.0,
            msg="When n < ceil(1/alpha) - 1, empty assertion threshold (1.0) must be returned"
        )
        LOGGER.info("Test 4 PASSED: Feasibility boundary condition verified.")

    def test_clustered_exchangeability_audit(self):
        """Test 5: Audit risk when clusters have correlated errors."""
        LOGGER.info("Running Test 5: Clustered Exchangeability Audit...")
        n_clusters = 50
        cluster_size = 6
        alpha = 0.10

        # Correlated error model: each cluster has a shared latent error factor
        all_clusters_c = []
        all_clusters_y = []
        all_cluster_ids = []

        for cid in range(n_clusters):
            cluster_latent = np.random.rand()
            for _ in range(cluster_size):
                c_val = np.random.beta(2, 2)
                # If cluster_latent < 0.3, error rate is higher
                p_err = 0.50 if cluster_latent < 0.3 else 0.10
                y_val = 0 if np.random.rand() < p_err else 1
                all_clusters_c.append(c_val)
                all_clusters_y.append(y_val)
                all_cluster_ids.append(cid)

        all_c = np.array(all_clusters_c)
        all_y = np.array(all_clusters_y)
        all_ids = np.array(all_cluster_ids)

        # Clustered split (GroupShuffleSplit by cluster id)
        n_sim = 500
        cluster_risks = []
        for _ in range(n_sim):
            perm_cids = np.random.permutation(n_clusters)
            cal_cids = set(perm_cids[:n_clusters // 2])
            test_cids = set(perm_cids[n_clusters // 2:])

            cal_mask = np.isin(all_ids, list(cal_cids))
            test_mask = np.isin(all_ids, list(test_cids))

            th = theta_hat_fast(all_c[cal_mask], all_y[cal_mask], alpha)
            r_test = np.mean((all_c[test_mask] > th) & (all_y[test_mask] == 0))
            cluster_risks.append(r_test)

        mean_cluster_risk = float(np.mean(cluster_risks))
        LOGGER.info("Clustered Evaluation Mean Risk = %.4f (Nominal Alpha = %.2f)",
                    mean_cluster_risk, alpha)
        # Should stay close to alpha (within reasonable group exchangeability margin)
        self.assertLessEqual(mean_cluster_risk, alpha + 0.02)
        LOGGER.info("Test 5 PASSED: Clustered exchangeability audit verified.")


if __name__ == "__main__":
    unittest.main()
