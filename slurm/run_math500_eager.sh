#!/bin/bash
#SBATCH --job-name=kvc_math500_eager
#SBATCH --partition=general
#SBATCH --output=/home/%u/workspace/kvcache/slurm_logs/math500_eager_%j.out
#SBATCH --error=/home/%u/workspace/kvcache/slurm_logs/math500_eager_%j.err
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# Math500 — eager attention only (adds h2o_attn + attn_entropy signals)
# Writes to separate *_eager_* files so existing math500 data is untouched.
# Preempt-safe: both collect and label skip already-completed entries on restart.

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

TRACES=data/math500_eager_traces.jsonl
LABELS=data/math500_eager_traces_labelled.jsonl
RESULTS=results/math500_eager_signal_ablation.csv

echo "=========================================="
echo "math500 — eager attention"
echo "Node: $(hostname)   Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Started: $(date)"
echo "Outputs: $TRACES / $LABELS / $RESULTS"
echo "=========================================="

# ── Step 1: Collect traces ────────────────────────────────────────────────────
echo ""
echo "[Step 1] Collecting math500 traces with eager attention ..."
python scripts/collect_traces.py \
    --dataset math500 \
    --n_samples 100 \
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
echo "=========================================="
exit $STEP3
