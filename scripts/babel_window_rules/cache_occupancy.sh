#!/bin/bash
# Cache occupancy (kv_live: mean and peak cached positions per problem) of each eviction method at K=512..4096 on
# Llama-8B, MATH-500 problems 0-14. One array task per method; results in $HOME/rebuttal_occupancy/cells.
#SBATCH --job-name=occupancy
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=smithv
#SBATCH --requeue
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20,babel-n9-32,babel-x9-32,babel-n9-28,babel-q5-28,babel-m5-20,babel-u5-28
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-8
#SBATCH --output=/home/skolawol/rebuttal_occupancy/logs/slurm_%a.out
source $HOME/workspace/kvcache_window/scripts/babel_window_rules/common.sh
O=$HOME/rebuttal_occupancy; mkdir -p $O/cells $O/logs
M=(kv_seg_hs_fill kv_seg_hs raas h2o r_kv longflow lag_kv knorm kvpress_snapkv_decoding)
m=${M[$SLURM_ARRAY_TASK_ID]}; f=""
[ $m = kv_seg_hs_fill ] && f="$EPIKV_FLAGS"
[ $m = kv_seg_hs ] && f="--refresh_tau 128 --tau_mode steps"
cd $CODE && preflight || exit 1
$PY scripts/benchmark.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B $COMMON --resume --n_samples 15 --start_idx 0 \
  --cache_sizes 512 1024 2048 4096 --methods $m $f --output $O/cells/llama_$m.json >> $O/logs/$m.log 2>&1
