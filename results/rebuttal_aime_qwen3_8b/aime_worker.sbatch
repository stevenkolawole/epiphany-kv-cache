#!/bin/bash
#SBATCH --job-name=aime_w
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20,babel-n9-32,babel-v9-28
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --output=/home/skolawol/rebuttal_aime/logs/w_%j.out
# AIME 2024-2026 (n=90) on Qwen3-8B: no eviction, DMS, EpiKV and RaaS at K=4096/8192, cap 32768, greedy.
# Claims lines of lines.txt by atomic mkdir (claims/<i>, owner = job id); a requeued (preempted) job first finishes its
# own claims, and --resume continues each piece from its last finished problem. Stack: /data env tf457 (Table 2 versions).
E=/data/user_data/skolawol/envs/tf457
export USE_TF=0 TRANSFORMERS_NO_TF=1 HF_HOME=/data/user_data/skolawol/hf_home TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1
$E/bin/python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print(torch.cuda.get_device_name(0))" || exit 1
D=$HOME/rebuttal_aime; mkdir -p $D/claims $D/done $D/failcount
fails=0
run(){ L=$(sed -n "$(( $1 + 1 ))p" $D/lines.txt); tag=${L%% *}; args=${L#* }
  [ -e $D/done/$tag ] && return 0
  echo "$(date -u +%FT%TZ) $SLURM_JOB_ID $(hostname) START $1 $tag"
  cd ~/workspace/kvcache_aime || { fails=$((fails+1)); rm -rf $D/claims/$1; return 1; }   # re-enter: stale NFS handle
  $E/bin/python scripts/benchmark.py $args --output $D/$tag.json >> $D/logs/$tag.log 2>&1; rc=$?
  if [ $rc -eq 0 ]; then touch $D/done/$tag; fails=0
  else fails=$((fails+1)); n=$(( $(cat $D/failcount/$1 2>/dev/null || echo 0) + 1 )); echo $n > $D/failcount/$1
    [ $n -lt 3 ] && rm -rf $D/claims/$1 || echo "PARKED $1 $tag after $n failures"; fi
  echo "$(date -u +%FT%TZ) $SLURM_JOB_ID END $1 $tag rc=$rc"
  [ $fails -ge 2 ] && { echo "$(hostname): two failures in a row, exiting"; exit 1; } }
for c in $D/claims/*; do [ "$(cat $c/owner 2>/dev/null)" = "$SLURM_JOB_ID" ] && run $(basename $c); done
n=$(wc -l < $D/lines.txt)
# Free claims whose owner job is not running (preempted, timed out, crashed) so any worker can take them; --resume keeps
# their finished problems. Skipped when squeue fails or does not list this job, so a controller hiccup frees nothing.
release(){ R=$(squeue -u $USER -h -t RUNNING -o %i) || return; echo "$R" | grep -qx "$SLURM_JOB_ID" || return
  for c in $D/claims/*; do [ -d $c ] || continue; i=$(basename $c); o=$(cat $c/owner 2>/dev/null)
    [ -z "$o" ] && [ $(( $(date +%s) - $(stat -c %Y $c) )) -lt 120 ] && continue   # just claimed, owner not written yet
    L=$(sed -n "$((i+1))p" $D/lines.txt); tag=${L%% *}
    [ -e $D/done/$tag ] && continue; [ "$(cat $D/failcount/$i 2>/dev/null || echo 0)" -ge 3 ] && continue
    echo "$R" | grep -qx "$o" || { rm -rf $c && echo "$(date -u +%FT%TZ) RELEASED $i $tag from ${o:-none}"; continue; }
    # A hung node (Nov 14: babel-v9-28 lost its controller and its worker sat 12 h on a finished piece) keeps its job
    # RUNNING; a piece with no write for 3 h (one problem takes under 2 h even for DMS at the cap) goes back to the pool.
    a=$(stat -c %Y $D/$tag.json 2>/dev/null || stat -c %Y $c); [ $(( $(date +%s) - a )) -gt 10800 ] && rm -rf $c && echo "$(date -u +%FT%TZ) RELEASED-STALE $i $tag from $o"; done; }
next(){ for i in $(seq 0 $((n-1))); do mkdir $D/claims/$i 2>/dev/null && { echo $SLURM_JOB_ID > $D/claims/$i/owner; echo $i; return 0; }; done; return 1; }
while :; do release; i=$(next) || break; run $i; done
echo "$(date -u +%FT%TZ) $SLURM_JOB_ID NOTHING LEFT"
