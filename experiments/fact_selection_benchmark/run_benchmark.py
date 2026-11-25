#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/fact_selection_benchmark/run_benchmark.py
=============================================================================
Master Orchestrator for Fact Selection Benchmark.

1. Precomputes & verifies multimodal feature caches.
2. Trains all 6 supervised models on Gold Train (702 clips) with early stopping on Val (149 clips).
3. Evaluates all 8 models (including Zero-Shot & Fine-Tuned variants) on 800 test clips across E1..E5.
4. Generates publication-ready Markdown and JSON reports.
=============================================================================
"""

import sys
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

BENCHMARK_DIR = REPO_ROOT / "experiments" / "fact_selection_benchmark"


def run_cmd(cmd: list, desc: str):
    print("\n" + "=" * 80)
    print(f"RUNNING: {desc}")
    print("Command:", " ".join(cmd))
    print("=" * 80)
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"FAILED: {desc} (code {res.returncode}) after {elapsed:.1f}s")
        sys.exit(res.returncode)
    print(f"SUCCESS: {desc} in {elapsed:.1f}s")


def main():
    python_bin = sys.executable

    # 1. Train supervised models
    train_script = BENCHMARK_DIR / "train_baselines.py"
    run_cmd([python_bin, str(train_script), "--model", "all", "--epochs", "15"], "Stage 1: Training Supervised Baseline Models")

    # 2. Evaluate all models on 800 test clips
    eval_script = BENCHMARK_DIR / "evaluate_benchmark.py"
    run_cmd([python_bin, str(eval_script)], "Stage 2: Evaluating Fact Selection Benchmark on Test Set")

    print("\n" + "=" * 80)
    print("ALL FACT SELECTION BENCHMARK STAGES COMPLETED SUCCESSFULLY!")
    print(f"Results Report: {BENCHMARK_DIR / 'reports' / 'fact_selection_benchmark_table.md'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
