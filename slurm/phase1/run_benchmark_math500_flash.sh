#!/bin/bash
#SBATCH --job-name=kvc_bench_math500_flash
#SBATCH --partition=general
#SBATCH --output=/home/%u/workspace/kvcache/slurm_logs/phase1/bench_math500_flash_%j.out
#SBATCH --error=/home/%u/workspace/kvcache/slurm_logs/phase1/bench_math500_flash_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=48G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# Phase 1 benchmark — MATH-500, FA2-compatible methods (no attn matrix needed).
# Single GPU (48G) — previous 2-GPU flash_attn runs hit CUDA launch failures;
# single GPU avoids multi-GPU flash_attn kernel coordination issues.
# Covers:
#   Baseline:    none
#   HS methods:  hs_variance  hs_variance_detrend  band_adaptive_hs
#   KV methods:  kv_val  kv_key  lag_kv_key  lag_kv
# Pair with run_benchmark_math500_eager.sh for attn-dependent methods.
# Results: results/phase1/benchmark_math500_flash.json

source $(conda info --base)/etc/profile.d/conda.sh
conda activate kvcache

export HF_HOME=/data/hf_cache/skolawol
export HF_HUB_CACHE=/data/hf_cache/skolawol/hub
export HF_DATASETS_CACHE=/data/hf_cache/skolawol/datasets
export TRANSFORMERS_CACHE=/data/hf_cache/skolawol

WORKDIR=/home/skolawol/workspace/kvcache
cd "$WORKDIR"
mkdir -p slurm_logs/phase1 /data/user_data/skolawol/kvcache/results/phase1

echo "=========================================="
echo "Phase 1 benchmark — MATH-500 (flash_attention_2, FA2-compatible methods)"
echo "Node: $(hostname)   Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Started: $(date)"
echo "=========================================="

python scripts/benchmark.py \
    --dataset math500 \
    --n_samples 100 \
    --max_new_tokens 8192 \
    --cache_sizes 512 1024 2048 4096 \
    --keep_recent_k 128 \
    --methods none hs_variance hs_variance_detrend band_adaptive_hs \
              kv_val kv_key lag_kv_key lag_kv \
    --attn_impl flash_attention_2 \
    --output /data/user_data/skolawol/kvcache/results/phase1/benchmark_math500_flash.json \
    --resume

EXIT=$?
echo ""
echo "=========================================="
echo "Complete at $(date) (exit $EXIT)"
echo "Output: /data/user_data/skolawol/kvcache/results/phase1/benchmark_math500_flash.json"
echo "=========================================="
exit $EXIT
