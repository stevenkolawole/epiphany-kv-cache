#!/bin/bash
#SBATCH --job-name=ovl_w
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20,babel-n9-32,babel-v9-28
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --output=/home/skolawol/rebuttal_dms_overlap/logs/w_%j.out
# DMS keep-agreement grid: lines are "<tag> <need> <dms_keep_agreement.py args>"; a line is claimable once its need (a done
# marker of the DMS record it replays, or "-") exists. Claims by mkdir (owner = job id); claims whose owner is not RUNNING,
# or idle 3 h, go back to the pool. A worker waits while blocked lines remain and exits when every line is done or parked.
E=/data/user_data/skolawol/envs/tf457
export USE_TF=0 TRANSFORMERS_NO_TF=1 HF_HOME=/data/user_data/skolawol/hf_home TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1
$E/bin/python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print(torch.cuda.get_device_name(0))" || exit 1
D=$HOME/rebuttal_dms_overlap; mkdir -p $D/claims $D/done $D/failcount $D/out
fails=0
field(){ sed -n "$(( $1 + 1 ))p" $D/lines.txt | cut -d' ' -f$2; }
run(){ L=$(sed -n "$(( $1 + 1 ))p" $D/lines.txt); tag=${L%% *}; rest=${L#* }; args=${rest#* }
  [ -e $D/done/$tag ] && return 0
  echo "$(date -u +%FT%TZ) $SLURM_JOB_ID $(hostname) START $1 $tag"
  cd ~/workspace/kvcache_overlap || { fails=$((fails+1)); rm -rf $D/claims/$1; return 1; }
  $E/bin/python scripts/dms_keep_agreement.py $args --output $D/out/$tag.json >> $D/logs/$tag.log 2>&1; rc=$?
  if [ $rc -eq 0 ]; then touch $D/done/$tag; fails=0
  else fails=$((fails+1)); n=$(( $(cat $D/failcount/$1 2>/dev/null || echo 0) + 1 )); echo $n > $D/failcount/$1
    [ $n -lt 3 ] && rm -rf $D/claims/$1 || echo "PARKED $1 $tag after $n failures"; fi
  echo "$(date -u +%FT%TZ) $SLURM_JOB_ID END $1 $tag rc=$rc"
  [ $fails -ge 2 ] && { echo "$(hostname): two failures in a row, exiting"; exit 1; } }
release(){ R=$(squeue -u $USER -h -t RUNNING -o %i) || return; echo "$R" | grep -qx "$SLURM_JOB_ID" || return
  for c in $D/claims/*; do [ -d $c ] || continue; i=$(basename $c); o=$(cat $c/owner 2>/dev/null)
    [ -z "$o" ] && [ $(( $(date +%s) - $(stat -c %Y $c) )) -lt 120 ] && continue
    tag=$(field $i 1); [ -e $D/done/$tag ] && continue; [ "$(cat $D/failcount/$i 2>/dev/null || echo 0)" -ge 3 ] && continue
    echo "$R" | grep -qx "$o" || { rm -rf $c && echo "$(date -u +%FT%TZ) RELEASED $i $tag from ${o:-none}"; continue; }
    a=$(stat -c %Y $D/done/$tag 2>/dev/null || stat -c %Y $c); [ $(( $(date +%s) - a )) -gt 10800 ] && rm -rf $c && echo "$(date -u +%FT%TZ) RELEASED-STALE $i $tag from $o"; done; }
# prints a claimed index, or nothing while unclaimed lines wait on a record; returns 1 when no unclaimed line is left
# (lines other workers hold are theirs; at the tail idle workers exit instead of parking a GPU)
next(){ wait=0; i=0
  while read -r tag need _; do
    if [ ! -e $D/done/$tag ] && [ ! -e $D/claims/$i ] && [ "$(cat $D/failcount/$i 2>/dev/null || echo 0)" -lt 3 ]; then
      if [ "$need" = - ] || [ -e $D/done/$need ]; then
        mkdir $D/claims/$i 2>/dev/null && { echo $SLURM_JOB_ID > $D/claims/$i/owner; echo $i; return 0; }
      else wait=1; fi; fi
    i=$((i+1)); done < $D/lines.txt
  [ $wait = 1 ]; }
for c in $D/claims/*; do [ "$(cat $c/owner 2>/dev/null)" = "$SLURM_JOB_ID" ] && run $(basename $c); done
while :; do release; i=$(next) || break; if [ -n "$i" ]; then run $i; else sleep 60; fi; done
echo "$(date -u +%FT%TZ) $SLURM_JOB_ID NOTHING LEFT"
