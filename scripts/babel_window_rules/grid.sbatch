#!/bin/bash
#SBATCH --job-name=wr_grid
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20,babel-n9-32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/home/skolawol/rebuttal_window_rules/logs/grid_%A_%a.out
# Array task i runs lines 2i and 2i+1 of lines.txt side by side on one card. A preempted task is requeued and each
# line continues from its last finished problem (--resume; greedy, so the finished prefix is reused verbatim).
# done/<tag> marks rc=0; whether a cell is complete and error-free is decided by window_rule_tables.py.
source $HOME/workspace/kvcache_window/scripts/babel_window_rules/common.sh
preflight || exit 1
cd $CODE || exit 1
mkdir -p $OUT/cells $OUT/done $OUT/logs
run(){ L=$(sed -n "$(( $1 + 1 ))p" $OUT/lines.txt); tag=${L%% *}; args=${L#* }
  [ -n "$tag" ] || return 0
  [ -e $OUT/done/$tag ] && { echo "SKIP $tag"; return 0; }
  echo "$(date -u +%FT%TZ) $(hostname) START $tag"
  $PY scripts/benchmark.py $args --output $OUT/cells/$tag.json >> $OUT/logs/$tag.log 2>&1; rc=$?
  [ $rc -eq 0 ] && touch $OUT/done/$tag
  echo "$(date -u +%FT%TZ) END $tag rc=$rc"; return $rc; }
i=$SLURM_ARRAY_TASK_ID
run $((2 * i)) & a=$!
run $((2 * i + 1)) & b=$!
wait $a; ra=$?; wait $b; rb=$?
[ $ra -eq 0 ] && [ $rb -eq 0 ]
