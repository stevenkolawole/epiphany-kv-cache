#!/bin/bash
#SBATCH --job-name=kvc_aime2024_eager
#SBATCH --partition=preempt
#SBATCH --output=/home/%u/workspace/kvcache/slurm_logs/aime2024_eager_%j.out
#SBATCH --error=/home/%u/workspace/kvcache/slurm_logs/aime2024_eager_%j.err
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# AIME 2024 — eager attention (preempt partition for long labelling times).
# --requeue: SLURM automatically restarts the job if preempted.
# Both collect and label skip already-completed entries on restart — safe to requeue.
# Writes to *_eager_* files so existing aime2024 data is untouched.
#
# max_new_tokens capped at 16384 for the eager run: at 32768 tokens the full
# attention matrix is ~64 GB (32768² × 32 layers × fp16) — OOM on most GPUs.
# For a non-eager AIME collection (KV signals only), use run_aime2024.sh at 32768.

# ── Environment ────────────────────────────────────────────────────────────────
source $(conda info --base)/etc/profile.d/conda.sh
conda activate kvcache

export HF_HOME=/data/hf_cache/skolawol
export HF_HUB_CACHE=/data/hf_cache/skolawol/hub
export HF_DATASETS_CACHE=/data/hf_cache/skolawol/datasets
export TRANSFORMERS_CACHE=/data/hf_cache/skolawol

WORKDIR=/home/skolawol/workspace/kvcache
cd "$WORKDIR"
mkdir -p slurm_logs results data

TRACES=data/aime2024_eager_traces.jsonl
LABELS=data/aime2024_eager_traces_labelled.jsonl
RESULTS=results/aime2024_eager_signal_ablation.csv

echo "=========================================="
echo "aime2024 — eager attention (preempt, requeue-safe)"
echo "Node: $(hostname)   Job: $SLURM_JOB_ID   Restart: ${SLURM_RESTART_COUNT:-0}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Started: $(date)"
echo "Outputs: $TRACES / $LABELS / $RESULTS"
echo "=========================================="

# ── Step 1: Collect traces ────────────────────────────────────────────────────
echo ""
echo "[Step 1] Collecting aime2024 traces with eager attention ..."
python scripts/collect_traces.py \
    --dataset aime2024 \
    --n_samples 30 \
    --max_new_tokens 16384 \
    --force_eager_attn \
    --output "$TRACES"

STEP1=$?
echo "[Step 1] Done (exit $STEP1) at $(date)"
[ $STEP1 -ne 0 ] && { echo "[ERROR] collect_traces failed. Aborting."; exit $STEP1; }

# ── Step 2a: Sanity check (1 trace) ──────────────────────────────────────────
echo ""
echo "[Step 2a] Labelling 1 trace (sanity check) ..."
python scripts/label_importance.py \
    --input "$TRACES" \
    --output "$LABELS" \
    --max_traces 1

STEP2A=$?
echo "[Step 2a] Done (exit $STEP2A) at $(date)"
[ $STEP2A -ne 0 ] && { echo "[ERROR] label_importance (1 trace) failed. Aborting."; exit $STEP2A; }

# ── Step 2b: Full labelling ───────────────────────────────────────────────────
echo ""
echo "[Step 2b] Labelling all correctly-answered traces ..."
python scripts/label_importance.py \
    --input "$TRACES" \
    --output "$LABELS"

STEP2B=$?
echo "[Step 2b] Done (exit $STEP2B) at $(date)"
[ $STEP2B -ne 0 ] && { echo "[ERROR] label_importance (full) failed. Aborting."; exit $STEP2B; }

# ── Step 3: Signal ablation ───────────────────────────────────────────────────
echo ""
echo "[Step 3] Signal ablation ..."
python scripts/signal_ablation.py \
    --traces "$TRACES" \
    --labels "$LABELS" \
    --output "$RESULTS"

STEP3=$?
echo "[Step 3] Done (exit $STEP3) at $(date)"

echo ""
echo "=========================================="
echo "Complete at $(date)   exit=$STEP3"
echo "  Traces:  $TRACES"
echo "  Labels:  $LABELS"
echo "  Results: $RESULTS"
echo "  Note: h2o_attn + attn_entropy collected (eager). Non-eager at 32768 tokens: run_aime2024.sh"
echo "=========================================="
exit $STEP3
