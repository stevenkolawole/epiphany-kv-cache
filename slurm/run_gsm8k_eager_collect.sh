#!/bin/bash
#SBATCH --job-name=kvc_gsm8k_e_col
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/%u/workspace/kvcache/slurm_logs/gsm8k_eager_collect_%j.out
#SBATCH --error=/home/%u/workspace/kvcache/slurm_logs/gsm8k_eager_collect_%j.err
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# GSM8K eager — collect → posthoc cross-validate → auto-chain label job.
# Phase 0B edition: all Phase 0 signals (incl. h2o_attn + attn_entropy via
# --force_eager_attn) + pre-RoPE key variance + per-layer HS at all 32 layers (0–31).
# GSM8K traces are short (~1-2k tokens) so 16384 token budget is never the bottleneck;
# eager mode is safe without OOM risk even at this limit.
#
# Steps run in this script:
#   1. Clean start (rm stale traces)
#   2. collect_traces.py --force_eager_attn --phase0b
#   3. extract_phase0b_signals.py (posthoc extraction for cross-validation)
#   4. Cross-validate posthoc vs collected signals (PASS required to proceed)
#   5. Auto-submit run_gsm8k_eager_label.sh with afterok dependency
#
# User only needs to sbatch this script; label job is chained automatically.

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

echo "=========================================="
echo "gsm8k — collect traces (general, 48h, eager, 16384 tokens, Phase 0B)"
echo "Node: $(hostname)   Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Started: $(date)"
echo "Output:  $TRACES"
echo "=========================================="

# No rm -f here: resume logic in collect_traces.py skips already-collected problems
# and appends to the existing file, so preempt+requeue continues where it left off.

python scripts/collect_traces.py \
    --dataset gsm8k \
    --n_samples 500 \
    --max_new_tokens 16384 \
    --force_eager_attn \
    --phase0b \
    --output "$TRACES"

EXIT=$?
echo "[Step 1] Done (exit $EXIT) at $(date)"
[ $EXIT -ne 0 ] && { echo "[ERROR] collect_traces failed. Aborting."; exit $EXIT; }

# ── Step 2: Posthoc Phase 0B extraction ───────────────────────────────────────
TRACES_POSTHOC="${TRACES%.jsonl}_posthoc.jsonl"
echo ""
echo "[Step 2] Posthoc Phase 0B extraction ..."
python scripts/extract_phase0b_signals.py \
    --input  "$TRACES" \
    --output "$TRACES_POSTHOC"

EXIT2=$?
echo "[Step 2] Done (exit $EXIT2) at $(date)"
[ $EXIT2 -ne 0 ] && { echo "[ERROR] Posthoc extraction failed. Aborting."; exit $EXIT2; }

# ── Step 3: Cross-validate ────────────────────────────────────────────────────
echo ""
echo "[Step 3] Cross-validating posthoc vs collected Phase 0B signals ..."
python scripts/extract_phase0b_signals.py \
    --input   "$TRACES_POSTHOC" \
    --compare "$TRACES"

XVAL=$?
echo "[Step 3] Cross-val done (exit $XVAL) at $(date)"
if [ $XVAL -ne 0 ]; then
    echo "[ERROR] Cross-validation FAILED — signal mismatch between collection and posthoc."
    echo "[ERROR] Inspect $TRACES_POSTHOC vs $TRACES before proceeding."
    echo "[ERROR] Label job NOT submitted. Fix the issue and rerun manually."
    exit $XVAL
fi

echo ""
echo "Cross-validation PASSED. Auto-submitting label job ..."
LABEL_JOB=$(sbatch --dependency=afterok:$SLURM_JOB_ID \
    "$WORKDIR/slurm/run_gsm8k_eager_label.sh" | awk '{print $4}')
echo "  Label job submitted: $LABEL_JOB (runs after job $SLURM_JOB_ID completes)"

echo ""
echo "=========================================="
echo "Complete at $(date)"
echo "  Traces:    $TRACES"
echo "  Posthoc:   $TRACES_POSTHOC"
echo "  Label job: $LABEL_JOB (queued, afterok:$SLURM_JOB_ID)"
echo "=========================================="
exit 0
