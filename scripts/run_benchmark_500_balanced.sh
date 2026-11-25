#!/bin/bash
set -e
set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/nguyendinhquy/TransMTL_OntoKG-main
mkdir -p logs

# Remove old cached captions so all models regenerate under the correct Hanoi 30 class prompts
if [ -f "data/caption_dataset/test_500_balanced_all_models_captions_scientific_vcgat.json" ]; then
    echo "Archiving old caption cache..."
    mv data/caption_dataset/test_500_balanced_all_models_captions_scientific_vcgat.json \
       data/caption_dataset/test_500_balanced_all_models_captions_scientific_vcgat.old.json
fi

echo "================================================================================"
echo "[$(date)] STARTING BENCHMARK ON FIXED BALANCED 500 TEST CLIPS (BATCH SIZE 24)"
echo "================================================================================"

# 1. Run Comprehensive Benchmark Suite on the Fixed 500 Balanced Test Set
# Evaluates all 18 models & configurations with B=10,000 bootstrap resamples
PYTHONUNBUFFERED=1 /home/nguyendinhquy/miniconda3/envs/qwen_graph/bin/python experiments/run_comprehensive_1119_benchmark.py \
    --device cuda:0 \
    --batch_size 24 \
    --mode full \
    --test_data data/caption_dataset/generated_captions_test_balanced_500_vcgat.jsonl \
    2>&1 | tee logs/benchmark_500_balanced.log

echo "================================================================================"
echo "[$(date)] ALL BENCHMARK TABLES (TABLES 4, 5, 6, 7, 8, 10, 11) COMPLETED!"
echo "================================================================================"
