#!/bin/bash
#SBATCH --job-name=kvc_math500e_lbl
#SBATCH --partition=general
#SBATCH --output=/home/%u/workspace/kvcache/slurm_logs/math500_eager_label_%j.out
#SBATCH --error=/home/%u/workspace/kvcache/slurm_logs/math500_eager_label_%j.err
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# Math500 eager — labelling + signal ablation only.
# Reads from data/math500_eager_traces.jsonl (already collected).
# Clears stale label/result files before re-running (fix: occlusion masking).

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
echo "math500 — eager label + ablate (general, 48h)"
echo "Node: $(hostname)   Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Started: $(date)"
echo "Inputs:  $TRACES"
echo "Outputs: $LABELS / $RESULTS"
echo "=========================================="

if [ ! -f "$TRACES" ]; then
    echo "[ERROR] Traces file not found: $TRACES"
    echo "[ERROR] Run run_math500_eager.sh (Step 1) first."
    exit 1
fi

# Clear stale labels and results so resume logic doesn't skip re-labelling.
echo ""
echo "[Cleanup] Removing stale label and result files ..."
rm -f "$LABELS" "$RESULTS"

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
echo "  Labels:  $LABELS"
echo "  Results: $RESULTS"
echo "=========================================="
exit $STEP3
