#!/bin/bash
#SBATCH --job-name=randmulti
#SBATCH --partition=general
#SBATCH --output=/home/%u/workspace/kvcache/slurm_logs/randmulti_%j.out
#SBATCH --error=/home/%u/workspace/kvcache/slurm_logs/randmulti_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=48G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# High-entropy occlusion arm — the control that discriminates between the two
# readings of the labeling-bias result.
#
# The three-arm test found EOS flips 0.176 of windows against 0.115 (pad) and
# 0.114 (random), with pad and random indistinguishable (p=0.55) and each
# differing from EOS at p=0.0009. That was read as "EOS's stop semantics inflate
# the labels" -- but the reading does not follow, because the `random` arm filled
# each window with 32 copies of ONE token. That is another degenerate repeated
# pattern, structurally the same manipulation as `pad`, so the two agreeing tells
# us nothing about whether EOS is special.
#
# `randmulti` writes a DISTINCT random in-vocabulary token at every masked
# position: a genuine high-entropy perturbation with no stop semantics.
#   flips at ~EOS rate  -> EOS is merely a high-gain probe; published bands stand
#   flips at ~pad rate  -> EOS's stop semantics are doing the work
#
# Runs on the same 60-trace slice the other arms used, so the comparison is
# paired trace-for-trace rather than against a historical aggregate.

# ── Environment ────────────────────────────────────────────────────────────────
source $(conda info --base)/etc/profile.d/conda.sh
conda activate kvcache

export HF_HOME=/data/hf_cache/skolawol
export HF_HUB_CACHE=/data/hf_cache/skolawol/hub
export HF_DATASETS_CACHE=/data/hf_cache/skolawol/datasets
export TRANSFORMERS_CACHE=/data/hf_cache/skolawol
# tqdm writing to the NFS-backed log killed a shard earlier with OSError 512.
export TQDM_DISABLE=1

WORKDIR=/home/skolawol/workspace/kvcache
cd "$WORKDIR"
mkdir -p slurm_logs results data

TRACES=data/slim60.jsonl
LABELS=data/slim60_labelled_randmulti.jsonl
N_TRACES=30

echo "=========================================="
echo "High-entropy occlusion arm (randmulti)"
echo "  node    : $(hostname)"
echo "  gpu     : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  traces  : $TRACES (first $N_TRACES)"
echo "  output  : $LABELS"
echo "  start   : $(date)"
echo "=========================================="

[ -f "$TRACES" ] || { echo "[ERROR] $TRACES missing"; exit 1; }

python scripts/label_importance.py \
    --input "$TRACES" \
    --output "$LABELS" \
    --mask_token randmulti \
    --mask_seed 0 \
    --max_traces "$N_TRACES"

RC=$?
echo "=========================================="
echo "Done (exit $RC) at $(date)"
if [ -f "$LABELS" ]; then
    echo "  labelled traces: $(wc -l < "$LABELS")"
fi
echo "=========================================="
exit $RC
