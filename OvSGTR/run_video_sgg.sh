#!/usr/bin/env bash
# ==============================================================================
# Full Video Scene Graph Generation Pipeline:
# 1. Scene detection & Keyframe Extraction (PySceneDetect)
# 2. Open-Vocabulary SGG (OvSGTR Swin-B)
# 3. Domain Refinement with SigLIP 2 + Material/Colour Attributes
# 4. Cross-Frame Object Tracking for Entity Consistency
# 5. Export Per-Scene Scene Graph JSONs
# ==============================================================================

set -e

VIDEO="${1:-videos/cau_the_huc_full.mp4}"
NUM_KEYFRAMES="${2:-4}"
MAX_SCENES="${3:-3}"
OUT_DIR="${4:-video_output}"

echo "=================================================================="
echo " Starting Video Scene Graph Pipeline"
echo " Video:         $VIDEO"
echo " Keyframes/Scene: $NUM_KEYFRAMES"
echo " Max Scenes:    $MAX_SCENES"
echo " Output Dir:    $OUT_DIR"
echo "=================================================================="

# Detect dynamic conda paths
CONDA_BASE="${CONDA_PREFIX:-$HOME/miniconda3}"
if [ -d "$HOME/miniconda3/envs/qwen_graph" ]; then
    PYTHON_SGG="$HOME/miniconda3/envs/qwen_graph/bin/python3"
else
    PYTHON_SGG="python3"
fi

if [ -f "$HOME/miniconda3/bin/python3" ]; then
    PYTHON_SIGLIP="$HOME/miniconda3/bin/python3"
else
    PYTHON_SIGLIP="python3"
fi

# PyTorch library path for GroundingDINO CUDA ops
TORCH_LIB=$($PYTHON_SGG -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false

echo ""
echo ">>> [STAGE 1] PySceneDetect Scene Cutting & OvSGTR SGG on Keyframes..."
$PYTHON_SGG video_extract_and_detect.py \
    --video "$VIDEO" \
    --output_dir "$OUT_DIR" \
    --num_keyframes "$NUM_KEYFRAMES" \
    --max_scenes "$MAX_SCENES" \
    --score_threshold 0.20

echo ""
echo ">>> [STAGE 2] SigLIP 2 Refinement, Object Tracking & Per-Scene JSON Export..."
$PYTHON_SIGLIP video_refine_track_export.py \
    --input_json "$OUT_DIR/detector_detections.json" \
    --universe "vn_tourism_visual_universe.json" \
    --output_dir "$OUT_DIR" \
    --edge_conf_threshold 0.050

echo ""
echo "=================================================================="
echo " [SUCCESS] Video Pipeline Completed!"
echo " Scene JSONs are ready in: $OUT_DIR"
echo "=================================================================="
ls -lh "$OUT_DIR"/*.json
