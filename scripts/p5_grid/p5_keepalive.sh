#!/bin/bash
# p5_keepalive.sh -- stateless; cron every 10 minutes. Three jobs:
#   1. respawn runners while the queue has lines (a queue with no runners behind
#      it looks exactly like a finished grid);
#   2. reap hung jobs. On Oct 6, 10 of 37 processes were zombies holding 20-40 GB
#      each -- one had printed a CUDA OOM eleven hours earlier and never exited --
#      and admission control was waiting on them forever;
#   3. trip a breaker on fast failure. The CPU-fallback loop ran 15 hours at
#      250 failures/hour because nothing distinguished it from throughput.
# Hang signals, calibrated against 36 live jobs: a working job touches its result
# file every 0-40 min and shows SM% well above zero. TQDM is off, so the LOG is not
# a heartbeat -- the result file is.
set -u
K=/opt/dlami/nvme/kvcache; Q=$K/queue3.txt; M=$K/epikv_master.txt
LOG=$HOME/p5_keepalive.log; RL=$K/runner_log.txt; WANT=32
# Oct 7 22:15: lock moved to $HOME/.p5_keepalive.lock2. The old lock was inherited as fd 8 by every runner
# this script spawned (and by their jobs), so it stayed held for as long as any of them
# lived and every later run exited at `flock -n` -- no respawn, reaper, guard or mirror
# for ~14 h (p5: cards 4-7 drained and never refilled). Children now get `8>&-`.
exec 8>$HOME/.p5_keepalive.lock2; flock -n 8 || exit 0
now(){ date -u +%FT%TZ; }

# ---- 3. breaker first: if the box is failing fast, park the queue -------------
c10=$(date -u -d '-10 min' +%FT%TZ)
# Oct 28: count genuine failures only. A pause or an OOM requeue also ends a job with rc!=0, and the
# old END-rc count read a 44-job hard pause as a failure storm and parked the queue.
bad=$(awk -v c=$c10 '$1>c && (/REQUEUE try=/ || /GIVEUP/ || /CPUFALLBACK/) && !/OOM-/' $RL 2>/dev/null | wc -l)
if [ "$bad" -ge 30 ] && [ ! -e $K/HALT ]; then
  flock $Q.lock bash -c "cat $Q >> $Q.halted; : > $Q"
  cat > $K/HALT <<H
$(now) $bad failures in 10 min. Queue parked in $Q.halted; runners will drain and exit.
Inspect $K/logs/<tag>.log for the last few END rc!=0 tags in $RL, fix the cause, then:
  cat $Q.halted >> $Q && : > $Q.halted && rm $K/HALT
H
  echo "$(now) HALT $bad failures/10min; queue parked" >> $LOG
fi


# ---- offload guard (Oct 6): accelerate can place PART of a model on the CPU
# ("offloaded to the cpu" / "on the meta device"); the harness only checks where
# the model as a whole landed, so such a job passes and crawls at ~1 tok/s.
# Kill it and requeue without charging a retry; it reloads cleanly on the GPU.
for pid in $(pgrep -f benchmark.py); do
  tag=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null | grep -o -- "--output [^ ]*" | sed "s|.*/||;s|\.json$||"); [ -n "$tag" ] || continue
  if grep -qiE "offloaded to the cpu|on the meta device" $K/logs/$tag.log 2>/dev/null; then
    kill $pid 2>/dev/null; sleep 5; kill -9 $pid 2>/dev/null
    echo "$(now) OFFLOAD-KILLED pid=$pid $tag" >> $LOG
  fi
done

# ---- 2. reaper ----------------------------------------------------------------
pm=$(for i in 1 2 3; do nvidia-smi pmon -c 1 -s u 2>/dev/null | awk 'NR>2 && $2 ~ /^[0-9]+$/ {print $2, $4}'; sleep 6; done \
     | awk '{s[$1]+=$2;n[$1]++} END {for(p in s) printf "%s %.0f\n", p, s[p]/n[p]}')
ps -eo etimes,pid,args | grep benchmark.py | grep -v 'grep\|bash -c' | while read -r et pid rest; do
  out=$(grep -o '\-\-output [^ ]*' <<<"$rest" | cut -d' ' -f2); [ -n "$out" ] || continue
  tag=$(basename "$out" .json); age=$((et/60))
  if [ -e "$out" ]; then fi=$(( ( $(date +%s) - $(stat -c %Y "$out") ) / 60 )); else fi=-1; fi
  sm=$(awk -v p=$pid '$1==p{print $2}' <<<"$pm"); sm=${sm:-99}
  # Oct 6 11:20 recalibration. Per-process SM% is unreliable with four jobs time-slicing a
  # card (it reads ~0 for working jobs), and one problem legitimately takes up to ~80 min
  # (16k-token loops at ~9 tok/s; p90 49 min, max 78 min measured on p5). The 30-45 min
  # rules reaped working jobs. Only the result file decides now, at 2.5x the worst
  # observed problem; this morning's zombies were idle 96-694 min and are still caught.
  why=""
  # Oct 7 22:15: idle is measured from the LATER of process start and last write. A job
  # resuming from an old partial file (--resume) has an old mtime until its first problem
  # finishes; on file age alone the reaper killed two 10-minute-old p4d jobs (files idle
  # 2060 min) the first time it ran after the lock fix.
  [ $fi -ge 200 ] && [ $age -ge 200 ] && why="result file idle ${fi}m, process ${age}m"
  [ $fi -eq -1 ] && [ $age -ge 200 ] && why="no result file after ${age}m"
  [ -z "$why" ] && continue
  kill $pid 2>/dev/null; sleep 5; kill -9 $pid 2>/dev/null; sleep 8
  echo "$(now) REAPED pid=$pid $tag: $why" >> $LOG
  # Requeue unless its runner already did (a runner-held job is requeued by the
  # runner's own wait; an orphan has nobody). --resume continues from the last
  # finished problem, so the partial file is kept.
  if ! grep -q " $tag " $Q 2>/dev/null; then
    line=$(awk -v t="$tag" 'NF>=5 && $4==t' $M 2>/dev/null | head -1)
    [ -n "$line" ] && ( flock 9; printf '%s\n' "$line" >> "$Q" ) 9>"$Q.lock" && echo "$(now) REQUEUED $tag" >> $LOG
  fi
done

# ---- MPS daemon (Oct 6 17:00) -------------------------------------------------
# Four batch-1 jobs time-slice a card; through MPS their kernels overlap. Card 7
# A/B, Qwen3-14B EpiKV x4: 59.4 -> 101.6 tok/s per card, identical outputs.
# The daemon must see ALL devices: with CUDA_VISIBLE_DEVICES=7 on the daemon, a
# client asking for device 7 gets no CUDA at all (silent CPU fallback).
if [ -s $K/mps_cards ] && ! pgrep -f "[n]vidia-cuda-mps-control" >/dev/null; then
  mkdir -p $K/mps/pipe $K/mps/log
  env -u CUDA_VISIBLE_DEVICES CUDA_MPS_PIPE_DIRECTORY=$K/mps/pipe CUDA_MPS_LOG_DIRECTORY=$K/mps/log nvidia-cuda-mps-control -d 8>&- \
    && echo "$(now) MPS daemon started" >> $LOG
fi

# ---- 1. respawn ----------------------------------------------------------------
# Per card/slot (Oct 6 16:40). The old block spawned a full set of 32 whenever ANY
# runner was missing, doubling the runners. A card with $K/PAUSE_CARD_<n> is left
# alone (held for a test). A card listed in $K/mps_cards gets runners whose jobs
# attach to the MPS daemon at $K/mps/pipe.
d=$(wc -l < $Q 2>/dev/null || echo 0)
if [ "$d" -gt 0 ] && [ ! -e $K/HALT ]; then
  n=0
  for i in 0 1 2 3 4 5 6 7; do
    [ -e $K/PAUSE_CARD_$i ] && continue
    mps=""; grep -qx "$i" $K/mps_cards 2>/dev/null && mps="CUDA_MPS_PIPE_DIRECTORY=$K/mps/pipe CUDA_MPS_LOG_DIRECTORY=$K/mps/log"
    for s in 1 2 3 4 5 6; do   # Oct 28: 6 slots, admission by free memory decides density
      pgrep -f "[r]unner4.sh $i R$s " >/dev/null && continue
      setsid nohup env $mps bash $K/runner4.sh $i R$s $Q >/dev/null 2>&1 8>&- & sleep 0.2; n=$((n+1))
    done
  done
  [ $n -gt 0 ] && echo "$(now) RESPAWN queue=$d spawned=$n -> $(pgrep -fc '[r]unner4.sh')" >> $LOG
fi

# ---- mirror finished cells off instance store ----------------------------------
for dd in results_newmodels results_gpqa_p4dmoved results_gpqa results_gpqa_new; do
  [ -d $K/$dd ] && { mkdir -p $HOME/epikv_results/$dd; cp -u $K/$dd/*.json $HOME/epikv_results/$dd/ 2>/dev/null; }
done
