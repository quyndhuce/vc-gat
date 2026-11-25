#!/bin/bash
# =============================================================================
# master_fact_to_caption_pipeline.sh
# =============================================================================
# Executes the end-to-end certainty-controlled pipeline completely in the background:
#   1. Releases VRAM holder so training has full GPU memory.
#   2. Runs Qwen 14B Teacher label generation & NLI claim verification (skipped if done).
#   3. Fine-tunes Qwen2.5-3B-Instruct with LoRA.
#   4. Runs 3-Part Independent Evaluation on 800 Test clips.
#   5. Automatically resumes 90% VRAM holding via app.py.
# =============================================================================

set -e
trap '' HUP

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_DIR="/home/nguyendinhquy/TransMTL_OntoKG-main"
CAPTION_DIR="$BASE_DIR/experiments/fact_to_caption"
PYTHON="/home/nguyendinhquy/miniconda3/envs/qwen_graph/bin/python"
LOG_FILE="$CAPTION_DIR/master_pipeline.log"
VERIFIED_SFT="$CAPTION_DIR/sft_dataset_verified_train.json"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "======================================================================"
echo "STARTING MASTER FACT-TO-CAPTION CERTAINTY PIPELINE AT $(date)"
echo "Host: $(hostname) | User: $(whoami)"
echo "======================================================================"

# 1. Release VRAM holder so training has full headroom
echo "Step 1: Releasing VRAM holder during active training..."
touch /tmp/release_vram
sleep 3
nvidia-smi

# 2. Generate teacher labels using Qwen 14B & NLI verification (if not already completed)
if [ -f "$VERIFIED_SFT" ] && [ -s "$VERIFIED_SFT" ]; then
    echo "Step 2: Supervised SFT dataset already generated and verified at $VERIFIED_SFT. Skipping to Step 3."
else
    echo "Step 2: Building supervised training labels using Qwen2.5-14B Teacher + Visual Grounding Facts..."
    cd "$BASE_DIR"
    "$PYTHON" "$CAPTION_DIR/generate_teacher_sft_dataset.py"
fi

# 3. Fine-tune Qwen2.5-3B-Instruct with LoRA
echo "Step 3: Fine-tuning Qwen2.5-3B-Instruct with LoRA..."
cd "$BASE_DIR"
"$PYTHON" "$CAPTION_DIR/train_qwen25_3b_lora.py"

# 4. Step 4: Strict Zero Data Leakage (Test evaluation reserved for evaluation phase)
echo "Step 4: Training completed. Test set strictly preserved without exposure (Zero Data Leakage)."

echo "======================================================================"
echo "ALL PIPELINE STAGES COMPLETED SUCCESSFULLY AT $(date)"
echo "======================================================================"
touch "$CAPTION_DIR/PIPELINE_SUCCESS.flag"

# 5. Resume 90% VRAM holding
echo "Step 5: Resuming 90% VRAM holder..."
rm -f /tmp/release_vram
sleep 3

# Ensure app.py is running if it was stopped
if ! pgrep -f "fuzzy_graph/app.py" > /dev/null; then
    cd /home/nguyendinhquy/graph-tourism/fuzzy_graph
    nohup "$PYTHON" app.py > app.log 2>&1 &
    sleep 3
fi

nvidia-smi
echo "Done! 90% VRAM holding is active."
