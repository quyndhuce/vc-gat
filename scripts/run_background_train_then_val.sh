#!/bin/bash
set -e
set -o pipefail

cd /home/nguyendinhquy/TransMTL_OntoKG-main
mkdir -p logs

echo "================================================================================"
echo "[$(date)] STARTING UNIFIED VLM GROUNDING: TRAIN SPLIT FIRST, THEN VAL SPLIT"
echo "================================================================================"

# 1. Run Grounding on TRAIN (702 clips) then VAL (149 clips) in a single run
PYTHONUNBUFFERED=1 /home/nguyendinhquy/miniconda3/envs/qwen_graph/bin/python experiments/build_scientific_visual_and_context_labels.py --split train_val 2>&1 | tee logs/vlm_grounding_train_val.log

echo "================================================================================"
echo "[$(date)] VLM GROUNDING (TRAIN + VAL) COMPLETED!"
echo "================================================================================"

# 2. Merge train and val into combined file for 2x2 Factorial Ablation
/home/nguyendinhquy/miniconda3/envs/qwen_graph/bin/python -c '
import json
from pathlib import Path

out_dir = Path("data/caption_dataset")
train_p = out_dir / "scientific_vlm_grounding_labels_train.json"
val_p = out_dir / "scientific_vlm_grounding_labels_val.json"
merged_p = out_dir / "scientific_vlm_grounding_labels_train_val.json"

merged = {}
if train_p.exists():
    with open(train_p) as f:
        merged.update(json.load(f))
if val_p.exists():
    with open(val_p) as f:
        merged.update(json.load(f))

with open(merged_p, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"Merged {len(merged)} clips into {merged_p}")
'

echo "================================================================================"
echo "[$(date)] RUNNING 2X2 FACTORIAL ABLATION EXPERIMENT..."
echo "================================================================================"

# 3. Run the 2x2 Factorial Ablation Benchmark
PYTHONUNBUFFERED=1 /home/nguyendinhquy/miniconda3/envs/qwen_graph/bin/python experiments/run_2x2_factorial_vcgat_experiment.py 2>&1 | tee logs/factorial_2x2_experiment.log

echo "================================================================================"
echo "[$(date)] ALL EXPERIMENTS AND PIPELINES COMPLETED SUCCESSFULLY!"
echo "================================================================================"
