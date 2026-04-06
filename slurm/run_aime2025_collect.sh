#!/bin/bash
#SBATCH --job-name=kvc_aime25_col
#SBATCH --partition=general
#SBATCH --output=/home/%u/workspace/kvcache/slurm_logs/aime2025_collect_%j.out
#SBATCH --error=/home/%u/workspace/kvcache/slurm_logs/aime2025_collect_%j.err
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# AIME 2025 (MathArena/aime_2025) — collect → posthoc cross-validate → auto-chain label.
# Non-eager (FlashAttention), 32768 tokens. Phase 0B: all 32 HS layers + preRoPE keys.
# h2o_attn + attn_entropy NOT collected (OOM at 32k without eager attn).

source $(conda info --base)/etc/profile.d/conda.sh
conda activate kvcache

export HF_HOME=/data/hf_cache/skolawol
export HF_HUB_CACHE=/data/hf_cache/skolawol/hub
export HF_DATASETS_CACHE=/data/hf_cache/skolawol/datasets
export TRANSFORMERS_CACHE=/data/hf_cache/skolawol

WORKDIR=/home/skolawol/workspace/kvcache
cd "$WORKDIR"
mkdir -p slurm_logs results data

TRACES=data/aime2025_traces.jsonl

echo "=========================================="
echo "aime2025 — collect traces (general, 48h, non-eager, 32768 tokens)"
echo "Node: $(hostname)   Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Started: $(date)"
echo "Output:  $TRACES"
echo "=========================================="

echo "[Cleanup] Removing stale traces ..."
rm -f "$TRACES"

python scripts/collect_traces.py \
    --dataset aime2025 \
    --n_samples 30 \
    --max_new_tokens 32768 \
    --phase0b \
    --output "$TRACES"

EXIT=$?
echo "[Step 1] Done (exit $EXIT) at $(date)"
[ $EXIT -ne 0 ] && { echo "[ERROR] collect_traces failed. Aborting."; exit $EXIT; }

TRACES_POSTHOC="${TRACES%.jsonl}_posthoc.jsonl"
echo ""
echo "[Step 2] Posthoc Phase 0B extraction ..."
python scripts/extract_phase0b_signals.py \
    --input  "$TRACES" \
    --output "$TRACES_POSTHOC"

EXIT2=$?
echo "[Step 2] Done (exit $EXIT2) at $(date)"
[ $EXIT2 -ne 0 ] && { echo "[ERROR] Posthoc extraction failed. Aborting."; exit $EXIT2; }

echo ""
echo "[Step 3] Cross-validating posthoc vs collected Phase 0B signals ..."
python scripts/extract_phase0b_signals.py \
    --input   "$TRACES_POSTHOC" \
    --compare "$TRACES"

XVAL=$?
echo "[Step 3] Cross-val done (exit $XVAL) at $(date)"
if [ $XVAL -ne 0 ]; then
    echo "[ERROR] Cross-validation FAILED. Label job NOT submitted."
    exit $XVAL
fi

echo ""
echo "Cross-validation PASSED. Auto-submitting label job ..."
LABEL_JOB=$(sbatch --dependency=afterok:$SLURM_JOB_ID \
    "$WORKDIR/slurm/run_aime2025_label.sh" | awk '{print $4}')
echo "  Label job submitted: $LABEL_JOB (runs after job $SLURM_JOB_ID completes)"

echo ""
echo "=========================================="
echo "Complete at $(date)"
echo "  Traces:    $TRACES"
echo "  Posthoc:   $TRACES_POSTHOC"
echo "  Label job: $LABEL_JOB (queued, afterok:$SLURM_JOB_ID)"
echo "=========================================="
exit 0
