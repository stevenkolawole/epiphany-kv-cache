#!/bin/bash
# Pull-based worker for Babel: two slots per card, each pops the next line of $Q and runs it until the queue is empty.
# Submit on both partitions (any number; spare workers exit when the queue is empty):
#   sbatch -p general scripts/babel_queue_worker.sh
#   sbatch -p preempt --qos=preempt_qos scripts/babel_queue_worker.sh
# Line kinds:
#   wr <n>                     line n of the window-rule grid ($HOME/rebuttal_window_rules/lines.txt)
#   lf <model> <K> <start> <n> one LongFlow-native MATH-500 piece in the LongFlow venv
# A popped line sits in held/<job>.<slot> until it ends, so a requeued worker resumes it, and any worker returns lines
# held by jobs that no longer exist to the head of the queue. A failed line goes to failed.txt and is not retried.
#SBATCH --job-name=q_worker
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20,babel-n9-32,babel-x9-32,babel-n9-28
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/home/skolawol/gridq/logs/worker_%j.out
G=$HOME/gridq; Q=$G/queue.txt; LK=$G/queue.lock; H=$G/held; mkdir -p $H $G/logs
WR=$HOME/rebuttal_window_rules; LF=$HOME/lf_results/grid; mkdir -p $LF/results $LF/done $LF/logs
LFV=/data/user_data/skolawol/lf_venv; LFCODE=$HOME/workspace/lf_smoke/epikv
J=$SLURM_JOB_ID
log(){ echo "$(date -u +%FT%TZ) $(hostname) $J $*" | tee -a $G/events.log; }

pop(){ ( flock 9; l=$(head -n 1 $Q); [ -n "$l" ] && sed -i 1d $Q; printf '%s' "$l" ) 9>$LK; }
push_head(){ ( flock 9; { printf '%s\n' "$1"; cat $Q; } > $Q.tmp && mv $Q.tmp $Q ) 9>$LK; }
reclaim(){ for f in $H/*; do [ -e "$f" ] || continue; j=${f##*/}; j=${j%%.*}
    [ "$j" = "$J" ] && continue
    squeue -h -j "$j" 2>/dev/null | grep -q . && continue
    ( flock 8; [ -e "$f" ] || exit 0; l=$(cat $f); push_head "$l"; rm -f $f
      echo "$(date -u +%FT%TZ) $J RECLAIM $l from $j" >> $G/events.log ) 8>$H.lock
  done; }

run_wr(){ local n=$1 l tag args; l=$(sed -n "${n}p" $WR/lines.txt); tag=${l%% *}; args=${l#* }
  [ -n "$tag" ] || return 1; [ -e $WR/done/$tag ] && return 0
  ( source $HOME/workspace/kvcache_window/scripts/babel_window_rules/common.sh; cd $CODE || exit 1; preflight || exit 1
    $PY scripts/benchmark.py $args --output $OUT/cells/$tag.json >> $OUT/logs/$tag.log 2>&1 ) && touch $WR/done/$tag; }

run_lf(){ local model=$1 K=$2 s=$3 n=$4 tag out; tag=$(basename $model)_longflow_native_K${K}_s${s}; out=$LF/results/$tag.json
  [ -e $LF/done/$tag ] && return 0
  ( export HF_HOME=/data/user_data/skolawol/hf_home HF_HUB_OFFLINE=1 USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1
    $LFV/bin/python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'" || exit 1
    cd $LFCODE || exit 1
    $LFV/bin/python scripts/longflow_native.py --longflow_dir $HOME/workspace/lf_smoke/LongFlow --model $model \
      --n_samples $n --start_idx $s --cache_size $K --max_new_tokens 16384 --resume --output $out >> $LF/logs/$tag.log 2>&1 || exit 1
    $LFV/bin/python - "$out" "$n" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))["results"]; n=int(sys.argv[2])
e=next(iter(next(iter(d.values())).values())); pp=e["per_problem"]
bad=[r for r in pp if r.get("error") or not r.get("n_tokens_generated")]
sys.exit(0 if (not e.get("partial")) and len(pp)==n and not bad else 1)
PY
  ) && touch $LF/done/$tag; }

slot(){ local s=$1 hf=$H/$J.$1 line rc t0 fast=0
  while :; do
    if [ -s $hf ]; then line=$(cat $hf); log "RESUME s$s $line"
    else reclaim; line=$(pop); [ -n "$line" ] || { log "EMPTY s$s"; return; }; printf '%s\n' "$line" > $hf; fi
    t0=$(date +%s); log "START s$s $line"
    set -- $line; case $1 in wr) run_wr $2;; lf) run_lf $2 $3 $4 $5;; *) false;; esac; rc=$?
    log "END s$s rc=$rc $(( $(date +%s) - t0 ))s $line"
    if [ $rc -ne 0 ]; then printf '%s rc=%s job=%s\n' "$line" $rc $J >> $G/failed.txt
      [ $(( $(date +%s) - t0 )) -lt 300 ] && fast=$((fast+1)) || fast=0
      rm -f $hf; [ $fast -ge 3 ] && { log "BREAKER s$s: 3 fast failures"; return; }
    else fast=0; rm -f $hf; fi
  done; }

log "WORKER_UP partition=$SLURM_JOB_PARTITION"
slot 0 & slot 1 & wait
log "WORKER_DONE"
