#!/bin/bash
#SBATCH --job-name=kvc_prefill_mem
#SBATCH --partition=general
#SBATCH --output=/home/%u/workspace/kvcache/slurm_logs/phase2/prefill_mem_%j.out
#SBATCH --error=/home/%u/workspace/kvcache/slurm_logs/phase2/prefill_mem_%j.err
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# Engineering figure: peak GPU memory of a single (prefill) forward pass vs.
# context length, comparing the two scoring regimes on an 80 GB card.
#   eager_attn : output_attentions=True  -> materialises (L x H x n x n) maps
#                                           (what H2O/ThinKV/RaaS must request)
#   flash      : FlashAttention-2, hidden states only (our FA2-compatible path)
# Forward-only, no generation -> a few minutes total.
#
# GPU TYPE: this requests an A100 80GB (Babel: babel-v5-[20,28,32]). This cluster
# has NO H100; the other big-memory option is H200 (141GB, babel-y5-16):
#   #SBATCH --gres=gpu:H200:1
# Check availability with: sinfo -p general -o "%N %G"
# Single GPU on purpose (multi-GPU flash_attn hit CUDA launch failures in Phase 1).
#
# Output:
#   reports/prefill_memory_80gb.json   (microbenchmark — Figure 2)
#   reports/analytical_batch_80gb.{json,pdf}  (analytical batch/throughput)

source $(conda info --base)/etc/profile.d/conda.sh
conda activate kvcache

export HF_HOME=/data/hf_cache/skolawol
export HF_HUB_CACHE=/data/hf_cache/skolawol/hub
export HF_DATASETS_CACHE=/data/hf_cache/skolawol/datasets
export TRANSFORMERS_CACHE=/data/hf_cache/skolawol

WORKDIR=/home/skolawol/workspace/kvcache
cd "$WORKDIR"
mkdir -p slurm_logs/phase2 reports

echo "=========================================="
echo "Prefill memory microbenchmark (eager output_attentions vs FlashAttention-2)"
echo "Node: $(hostname)   Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Started: $(date)"
echo "=========================================="

# Low end included so eager produces a curve before its OOM cliff; high end
# pushed to 131072 so the FA2 path's linear scaling is visible on 80 GB.
python scripts/bench_prefill_memory.py \
    --seq_lens 512 1024 2048 4096 8192 16384 32768 65536 131072 \
    --batch_sizes 1 \
    --modes eager_attn flash \
    --output reports/prefill_memory_80gb.json

echo ""
echo "--- analytical batch/throughput (80 GB) ---"
python scripts/analytical_batch_throughput.py \
    --gpu_gb 80 --weight_gb 16 \
    --budgets 2048 4096 \
    --context_lens 4096 8192 16384 32768 65536 131072 \
    --output reports/analytical_batch_80gb.json \
    --plot reports/analytical_batch_80gb.pdf

EXIT=$?
echo ""
echo "=========================================="
echo "Complete at $(date) (exit $EXIT)"
echo "Outputs: reports/prefill_memory_80gb.json , reports/analytical_batch_80gb.{json,pdf}"
echo "=========================================="
exit $EXIT
