#!/bin/bash
# Slurm array task: one 50-problem piece of one (model, method, K) MATH-500 cell, LongFlow native or EpiKV, in the
# LongFlow venv (torch 2.4 / transformers 4.51). Task N runs line N of $MANIFEST: "<model> <method> <K> <start>".
# Submit (from the login node):
#   MANIFEST=$HOME/lf_results/grid/manifest.txt sbatch --array=1-160%24 scripts/longflow_grid.sh
#SBATCH --job-name=lf_grid
#SBATCH --partition=general
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/skolawol/lf_results/grid/logs/%x_%A_%a.out
V=/data/user_data/skolawol/lf_venv
export HF_HOME=/data/user_data/skolawol/hf_home HF_HUB_OFFLINE=1 USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1
$V/bin/python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print(torch.cuda.get_device_name(0), torch.__version__)" || { echo ABORT_NOCUDA; exit 1; }
read -r MODEL METHOD K START < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
N=50
TAG=$(basename $MODEL)_${METHOD}_K${K}_s${START}
OUT=$HOME/lf_results/grid/results/$TAG.json
mkdir -p $(dirname $OUT)
hostname; echo "$TAG $(date -u)"
cd $HOME/workspace/lf_smoke/epikv
if [ "$METHOD" = longflow_native ]; then
  $V/bin/python scripts/longflow_native.py --longflow_dir $HOME/workspace/lf_smoke/LongFlow --model $MODEL \
    --n_samples $N --start_idx $START --cache_size $K --max_new_tokens 16384 --resume --output $OUT
else
  case $MODEL in *Qwen3*) BA=11; BB=24;; *) BA=20; BB=24;; esac
  $V/bin/python scripts/benchmark.py --model $MODEL --dataset math500 --n_samples $N --start_idx $START \
    --max_new_tokens 16384 --logical_positions --keep_recent_k 128 --cache_sizes $K --methods kv_seg_hs_fill \
    --attn_impl sdpa --refresh_tau 128 --tau_mode steps --query_sim_mode replace --query_sim_weight 1 \
    --query_mode qk --query_sim_window 8 --band_a_layer $BA --band_b_layer $BB --resume --output $OUT
fi
rc=$?
# a piece is done only if every problem ran without error and generated tokens
$V/bin/python - "$OUT" <<'PY' || rc=1
import json,sys
d=json.load(open(sys.argv[1]))["results"]
e=next(iter(next(iter(d.values())).values()))
pp=e["per_problem"]; bad=[r for r in pp if r.get("error") or not r.get("n_tokens_generated")]
ok=(not e.get("partial")) and len(pp)==50 and not bad
print("PIECE_OK" if ok else f"PIECE_BAD partial={e.get('partial')} n={len(pp)} bad={len(bad)}"); sys.exit(0 if ok else 1)
PY
echo "$TAG rc=$rc $(date -u)"; exit $rc
