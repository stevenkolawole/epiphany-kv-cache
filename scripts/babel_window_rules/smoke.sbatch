#!/bin/bash
#SBATCH --job-name=wr_smoke
#SBATCH --partition=general
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20,babel-n9-32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/skolawol/rebuttal_window_rules/logs/smoke_%j.out
# Smoke: two methods side by side on one card (the grid's density), MATH-500 problems 0-4 at K=512, with one
# COMPACT line per compaction in each log. Usage: sbatch smoke.sbatch <llama|qwen> <method> <method> [pytest]
source $HOME/workspace/kvcache_window/scripts/babel_window_rules/common.sh
preflight || exit 1
cd $CODE || exit 1
[ "$4" = pytest ] && { $PY -m pytest -q tests/test_window_rules.py 2>&1 | tail -3; }
S=$OUT/smoke; mkdir -p $S
run(){ $PY scripts/trace_compactions.py $(model_flags $1) $COMMON --n_samples 5 --start_idx 0 --cache_sizes 512 \
         --methods $2 $(method_flags $2) --output $S/$1_$2.json > $S/$1_$2.log 2>&1; echo "SMOKE $1 $2 rc=$?"; }
run $1 $2 & run $1 $3 & wait
echo SMOKE_DONE
