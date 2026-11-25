#!/usr/bin/env bash
# =============================================================================
# reproduce_all.sh
# Master Reproduction Script for IJAR Manuscript:
# "Decision-Certified Fuzzy Horn Reasoning and Perceptual Grounding for
#  Knowledge-Reliable Video Captioning"
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================================"
echo "RUNNING FULL REPRODUCTION AND VERIFICATION SUITE"
echo "================================================================================"

echo ""
echo ">>> STEP 1: Running Automated Theoretical Property & Unit Test Suite..."
python3 -m unittest discover tests/

echo ""
echo ">>> STEP 2: Reproducing Table 3 (Decision-Certified Horn Reasoning Benchmark)..."
python3 experiments/reproduce_table3_certified_horn.py

echo ""
echo ">>> STEP 3: Reproducing Table 4 (Anytime Conditioning & Derivation Depth)..."
python3 experiments/reproduce_table4_derivation_depth.py

echo ""
echo ">>> STEP 4: Reproducing Table 5 (VC-GAT Ranking Cluster Bootstrap Significance)..."
python3 experiments/reproduce_table5_vcgat_ranking.py

echo ""
echo ">>> STEP 5: Reproducing Table 6 (Landmark Calibration & Conformal Control)..."
python3 experiments/reproduce_table6_landmark_calibration.py

echo ""
echo ">>> STEP 6: Reproducing Table 7 (End-to-End Captioning Evaluation on Hanoi-500)..."
python3 experiments/reproduce_table7_end2end_captioning.py

echo ""
echo ">>> STEP 7: Reproducing Table 11 (Cross-Domain Evaluation on HCMC-300)..."
python3 experiments/reproduce_table11_cross_domain_hcmc.py

echo ""
echo ">>> STEP 8: Verifying Proposition 2b & 2c Stress Tests..."
python3 experiments/reproduce_proposition_2b_2c_stress.py

echo ""
echo ">>> STEP 9: Verifying Persistent Storage Benchmark (Physical NVMe SSD)..."
python3 experiments/reproduce_persistent_storage_benchmark.py

echo ""
echo ">>> STEP 10: Verifying Figure 2 & Figure 3 Artifacts..."
python3 experiments/reproduce_figure2_certification_dependency.py
python3 experiments/reproduce_figure3_conformal_risk.py

echo ""
echo "================================================================================"
echo "ALL EXPERIMENTAL TABLES AND FIGURES SUCCESSFULLY REPRODUCED AND VERIFIED!"
echo "================================================================================"
