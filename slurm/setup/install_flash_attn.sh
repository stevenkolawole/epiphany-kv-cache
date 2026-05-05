#!/bin/bash
#SBATCH --job-name=install_flash_attn
#SBATCH --partition=general
#SBATCH --output=/home/%u/workspace/kvcache/slurm_logs/install_flash_attn_%j.out
#SBATCH --error=/home/%u/workspace/kvcache/slurm_logs/install_flash_attn_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# Try a prebuilt flash-attn wheel first; fall back to source compilation if
# no matching wheel exists for this CUDA/Python/PyTorch combination.
# flash-attn source builds can take a long time. MAX_JOBS controls how many
# parallel compiler processes ninja/setuptools spawns; set it to match
# the allocated CPU count so we use everything we asked for.
# 128 GB RAM guards against the linker OOMing during heavy parallel builds.

source $(conda info --base)/etc/profile.d/conda.sh
conda activate kvcache

mkdir -p /home/skolawol/workspace/kvcache/slurm_logs

echo "=========================================="
echo "Installing flash-attn"
echo "Node: $(hostname)   Job: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU:  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.version.cuda)')"
echo "Started: $(date)"
echo "=========================================="

# flash-attn source builds are dramatically slower without ninja.
if ! command -v ninja >/dev/null 2>&1; then
    echo "ninja not found; installing into current environment..."
    python -m pip install -U ninja
fi
echo "ninja: $(command -v ninja || echo 'not found')"
ninja --version || true

export MAX_JOBS=32
export NINJA_NUM_JOBS=$MAX_JOBS
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
export NVCC_THREADS=2

echo "Build parallelism: MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS"

FLASH_ATTN_VERSION=2.8.3

if python -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --only-binary=:all: -v; then
    echo "Installed flash-attn wheel ${FLASH_ATTN_VERSION}"
else
    echo "No matching flash-attn wheel found; building from source"
    python -m pip install flash-attn --no-build-isolation -v
fi

EXIT=$?
echo ""
echo "=========================================="
echo "pip install exited with: $EXIT"
echo "Finished: $(date)"
echo "=========================================="

if [ $EXIT -eq 0 ]; then
    echo ""
    echo "Verifying import..."
    python -c "
import flash_attn
import torch
print('flash_attn version:', flash_attn.__version__)
print('CUDA available:', torch.cuda.is_available())
print('OK')
"
fi

exit $EXIT
