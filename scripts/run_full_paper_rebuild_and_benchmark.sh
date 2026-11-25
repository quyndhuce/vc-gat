#!/bin/bash
set -e
set -o pipefail

cd /home/nguyendinhquy/TransMTL_OntoKG-main
mkdir -p logs

echo "================================================================================"
echo "[$(date)] STARTING COMPLETE PAPER RE-EVALUATION & BENCHMARK SUITE"
echo "================================================================================"

# Step 1: Retrain Production NewDeepVCGAT on all 813 VLM-grounded clips
echo "[$(date)] [STEP 1/4] Retraining Production NewDeepVCGAT..."
PYTHONUNBUFFERED=1 /home/nguyendinhquy/miniconda3/envs/qwen_graph/bin/python experiments/run_2x2_factorial_vcgat_experiment.py 2>&1 | tee logs/step1_retrain_vcgat.log

# Step 2: Rebuilding 1,119 Test Dataset with New Mathematical Deep VC-GAT
echo "[$(date)] [STEP 2/4] Rebuilding 1,119 Test Dataset with New Deep VC-GAT..."
PYTHONUNBUFFERED=1 /home/nguyendinhquy/miniconda3/envs/qwen_graph/bin/python experiments/build_full_1119_test_dataset_vcgat.py 2>&1 | tee logs/step2_build_1119_dataset.log

# Step 3: Run Full VLM Grounding on remaining test clips for 100% 1,119 coverage
echo "[$(date)] [STEP 3/4] Expanding Multi-VLM Grounding to Full 1,119 Test Set..."
if [ ! -f "data/caption_dataset/checkpoint_vlm_labels_test_1119.json" ] && [ -f "data/caption_dataset/checkpoint_vlm_labels_test_500.json" ]; then
    cp data/caption_dataset/checkpoint_vlm_labels_test_500.json data/caption_dataset/checkpoint_vlm_labels_test_1119.json
fi
PYTHONUNBUFFERED=1 /home/nguyendinhquy/miniconda3/envs/qwen_graph/bin/python experiments/build_scientific_visual_and_context_labels.py --split test_1119 2>&1 | tee logs/step3_vlm_grounding_1119.log

# Step 4: Run Comprehensive 1,119 Benchmark Suite (B=10,000 bootstrap, Tables 4, 5, 6, 7, 8, 10, 11)
echo "[$(date)] [STEP 4/4] Running Comprehensive 1,119 Benchmark Suite..."
PYTHONUNBUFFERED=1 /home/nguyendinhquy/miniconda3/envs/qwen_graph/bin/python experiments/run_comprehensive_1119_benchmark.py --device cuda:0 --batch_size 32 --mode full 2>&1 | tee logs/step4_comprehensive_1119_benchmark.log

echo "================================================================================"
echo "[$(date)] ALL STEPS COMPLETED! FULL PAPER BENCHMARK READY."
echo "================================================================================"

