#!/bin/bash
#SBATCH --job-name=kvc_bench_aime24_eager
#SBATCH --partition=general
#SBATCH --output=/home/%u/workspace/kvcache/slurm_logs/phase1/bench_aime2024_eager_%j.out
#SBATCH --error=/home/%u/workspace/kvcache/slurm_logs/phase1/bench_aime2024_eager_%j.err
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# Phase 1 benchmark — AIME 2024, all eager-mode methods.
# 30 problems, 16384 token budget.
# Covers:
#   Baselines:   none h2o thinKV raas
#   Attn+HS:     attn_hs_product hybrid_seg_hs
# Pair with run_benchmark_aime2024_sdpa.sh for FA2-compatible HS/KV methods.
# Results: results/phase1/benchmark_aime2024_eager.json

source $(conda info --base)/etc/profile.d/conda.sh
conda activate kvcache

export HF_HOME=/data/hf_cache/skolawol
export HF_HUB_CACHE=/data/hf_cache/skolawol/hub
export HF_DATASETS_CACHE=/data/hf_cache/skolawol/datasets
export TRANSFORMERS_CACHE=/data/hf_cache/skolawol

export CUDA_LAUNCH_BLOCKING=1

WORKDIR=/home/skolawol/workspace/kvcache
cd "$WORKDIR"
mkdir -p slurm_logs/phase1 /data/user_data/skolawol/kvcache/results/phase1

echo "=========================================="
echo "Phase 1 benchmark — AIME 2024 (eager, all attn-dependent methods)"
echo "Node: $(hostname)   Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Started: $(date)"
echo "=========================================="

python scripts/benchmark.py \
    --dataset aime2024 \
    --n_samples 30 \
    --max_new_tokens 16384 \
    --cache_sizes 512 1024 2048 4096 8192 \
    --keep_recent_k 128 \
    --methods none h2o thinKV raas attn_hs_product hybrid_seg_hs \
    --attn_impl eager \
    --output /data/user_data/skolawol/kvcache/results/phase1/benchmark_aime2024_eager.json \
    --resume

EXIT=$?
echo ""
echo "=========================================="
echo "Complete at $(date) (exit $EXIT)"
echo "Output: /data/user_data/skolawol/kvcache/results/phase1/benchmark_aime2024_eager.json"
echo "=========================================="
exit $EXIT
