#!/bin/bash
#SBATCH --job-name=aime_setup
#SBATCH --partition=general
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20,babel-n9-32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/skolawol/rebuttal_aime/logs/setup_%j.out
# Builds the Table 2 library stack (p4d venv: py3.12, torch 2.7.0+cu126, transformers 4.57.1, flash-attn 2.8.3) under
# /data, fetches Qwen3-8B, the DMS checkpoint and AIME 2024-2026, then runs a short smoke of every method.
set -x
source ~/miniconda3/etc/profile.d/conda.sh
E=/data/user_data/skolawol/envs/tf457
export HF_HOME=/data/user_data/skolawol/hf_home USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
[ -x $E/bin/python ] || conda create -y -q -p $E python=3.12 || exit 1
$E/bin/pip install -q torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126 || exit 1
$E/bin/pip install -q transformers==4.57.1 accelerate==1.15.0 datasets==5.0.1 huggingface_hub==0.36.2 tokenizers==0.22.2 \
  safetensors==0.8.0 einops==0.8.2 math-verify==0.9.0 sympy==1.14.0 numpy==2.5.4 tqdm || exit 1
$E/bin/pip install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" || exit 1
$E/bin/python -c "import torch, transformers, flash_attn; assert torch.cuda.is_available(); print('STACK', torch.__version__, transformers.__version__, flash_attn.__version__, torch.cuda.get_device_name(0))" || exit 1
$E/bin/python -c "
from huggingface_hub import snapshot_download as s; import os
for r in ('Qwen/Qwen3-8B', 'nvidia/Qwen3-8B-DMS-8x'): print('MODEL', s(r, cache_dir=os.environ['HF_HOME']))" || exit 1
cd ~/workspace/kvcache_aime || exit 1
$E/bin/python -c "
import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src'); import benchmark as b
for d in ('aime2024', 'aime2025', 'aime2026'): print('DATA', d, len(b.load_problems(d, 30, 0)))" || exit 1
C="--model Qwen/Qwen3-8B --dataset aime2024 --max_new_tokens 1024 --logical_positions --keep_recent_k 128 --n_samples 1 --start_idx 0 --attn_impl flash_attention_2"
EP="--refresh_tau 128 --tau_mode steps --query_sim_mode replace --query_sim_weight 1 --query_mode qk --query_sim_window 8 --band_a_layer 11 --band_b_layer 24"
O=~/rebuttal_aime/smoke; mkdir -p $O
$E/bin/python scripts/benchmark.py $C --cache_sizes 512 --methods none --output $O/none.json > $O/none.log 2>&1; echo "SMOKE none rc=$?"
$E/bin/python scripts/benchmark.py $C --cache_sizes 512 --methods kv_seg_hs_fill $EP --output $O/epikv.json > $O/epikv.log 2>&1; echo "SMOKE epikv rc=$?"
$E/bin/python scripts/benchmark.py $C --cache_sizes 512 --methods raas --output $O/raas.json > $O/raas.log 2>&1; echo "SMOKE raas rc=$?"
$E/bin/python scripts/benchmark.py --model nvidia/Qwen3-8B-DMS-8x --tokenizer Qwen/Qwen3-8B --dataset aime2024 --max_new_tokens 1024 \
  --logical_positions --keep_recent_k 128 --n_samples 1 --start_idx 0 --cache_sizes 8 --methods dms --output $O/dms.json > $O/dms.log 2>&1; echo "SMOKE dms rc=$?"
for f in $O/*.json; do $E/bin/python -c "
import json, sys; r = json.load(open(sys.argv[1]))['results']; b = list(list(r.values())[0].values())[0]; p = b['per_problem'][0]
print('REC', sys.argv[1].split('/')[-1], p.get('n_tokens_generated'), round(p.get('n_tokens_generated', 0) / max(p.get('wall_time_s') or 1e-9, 1e-9), 1), 'tok/s', p.get('error'))" $f; done
echo SETUP_DONE
