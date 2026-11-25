#!/usr/bin/env bash
set -e

REPO_ROOT="/home/nguyendinhquy/OvSGTR"
cd "$REPO_ROOT"

echo "========================================================================"
echo "STARTING FUZZY KNOWLEDGE GRAPH (FKG) 13-STEP PIPELINE"
echo "Host: $(hostname) | Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "========================================================================"

# Stage 1: Coarse Detection, Filtering & Tracking (Steps 1-4)
echo ""
echo ">>> [STAGE 1] Running Detection & Tracking with qwen_graph environment..."
/home/nguyendinhquy/miniconda3/envs/qwen_graph/bin/python3 "$REPO_ROOT/fkg/run_stage1_detector.py"

# Stage 2: Semantic Grounding, Relations, Profiles & Evaluation (Steps 5-13)
echo ""
echo ">>> [STAGE 2] Running Semantic Grounding, Fuzzy Relations & Reasoning with base environment..."
/home/nguyendinhquy/miniconda3/bin/python3 "$REPO_ROOT/fkg/run_stage2_reasoning.py"

echo ""
echo "========================================================================"
echo "FKG PIPELINE COMPLETED SUCCESSFULLY!"
echo "========================================================================"
