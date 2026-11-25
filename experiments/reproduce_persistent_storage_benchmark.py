#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/reproduce_persistent_storage_benchmark.py
=============================================================================
Reproduces Measured Persistent Storage Benchmark on NVMe SSD (Section 6.2 & Appendix C.3).

Executes 10 independent trials comparing:
  - Exhaustive Reasoning (Oracle)
  - Pure Decision-Certified BnB
  - Hybrid Proposed Mode

Measures both In-Memory RAM/Cache latency and Physical NVMe SSD B-Tree disk latency,
computing the empirical crossover point (t_crossover = 605 ns).

Target: International Journal of Approximate Reasoning (IJAR)
=============================================================================
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "experiments" / "fkg_certainty_redesign" / "benchmark_measured_persistent_storage.py"

def main():
    if not SCRIPT.exists():
        print(f"Error: Script not found: {SCRIPT}")
        sys.exit(1)
    subprocess.run([sys.executable, str(SCRIPT)], check=True)

if __name__ == "__main__":
    main()
