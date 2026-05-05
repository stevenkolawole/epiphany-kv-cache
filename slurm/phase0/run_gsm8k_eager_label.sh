#!/bin/bash
#SBATCH --job-name=kvc_gsm8k_e_lbl
#SBATCH --partition=general
#SBATCH --output=/home/%u/workspace/kvcache/slurm_logs/gsm8k_eager_label_%j.out
#SBATCH --error=/home/%u/workspace/kvcache/slurm_logs/gsm8k_eager_label_%j.err
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# GSM8K — eager labelling + signal ablation only.
# Reads from data/gsm8k_eager_traces.jsonl (produced by run_gsm8k_eager_collect.sh).
# Resume-safe: label_importance.py appends and skips traces already labelled,
# so rerunning this job on general continues from the last written trace.
#
# Prerequisite: run_gsm8k_eager_collect.sh must be complete before submitting this.

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

TRACES=data/gsm8k_eager_traces.jsonl
LABELS=data/gsm8k_eager_traces_labelled.jsonl
RESULTS=results/gsm8k_eager_signal_ablation.csv

echo "=========================================="
echo "gsm8k — eager label + ablate (general, 48h)"
echo "Node: $(hostname)   Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Started: $(date)"
echo "Inputs:  $TRACES"
echo "Outputs: $LABELS / $RESULTS"
echo "=========================================="

if [ ! -f "$TRACES" ]; then
    echo "[ERROR] Traces file not found: $TRACES"
    echo "[ERROR] Run run_gsm8k_eager_collect.sh first."
    exit 1
fi

# No rm -f here: resume logic in label_importance.py skips already-labelled traces
# and appends to the existing file, so reruns continue where they left off.

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
echo "  Note: h2o_attn + attn_entropy collected (eager). Non-eager at 32768: run_gsm8k_*.sh"
echo "=========================================="
exit $STEP3
