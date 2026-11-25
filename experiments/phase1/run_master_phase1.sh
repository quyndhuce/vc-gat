#!/bin/bash
REPO_DIR="/home/nguyendinhquy/TransMTL_OntoKG-main"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "======================================================================"
echo "STARTING MASTER PHASE 1 PIPELINE: HANOI & HCMC EVALUATION"
echo "Date: $(date)"
echo "======================================================================"

# Step 1: Temporarily pause VRAM holder to free GPU memory
echo "Pausing GPU VRAM holder..."
pkill -f "python app.py" || true
sleep 3
nvidia-smi

# Step 2: Run Phase 1 End-to-End Pipeline
echo "Running Phase 1 Pipeline..."
/home/nguyendinhquy/miniconda3/envs/qwen_graph/bin/python experiments/phase1/run_phase1_pipeline.py
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Phase 1 pipeline completed successfully!"
    touch experiments/phase1/PHASE1_SUCCESS.flag
else
    echo "Phase 1 pipeline FAILED with exit code $EXIT_CODE!"
fi

# Step 3: Always resume 90% VRAM holder
echo "Resuming 90% VRAM holder..."
nohup /home/nguyendinhquy/miniconda3/envs/qwen_graph/bin/python "$REPO_DIR/app.py" > /dev/null 2>&1 &
sleep 5
nvidia-smi
echo "All done at $(date)"
exit $EXIT_CODE
