#!/bin/bash
# runner4.sh <card> <slot> <queuefile>  (runner3 + quoting-safe requeue, 2026-10-06; see runner_fix.sh on p4d for the bug)
# 16:45: a popped line is recorded in held/<card>.<slot> until its job starts.
#
# Replaces runner2.sh, which launched a job whenever a slot was free. On an H200
# four Qwen3-14B jobs do not fit, accelerate silently places the model on the CPU,
# the job dies in seconds, and the runner requeues it -- which reads as throughput
# in the log. Oct 6: 667 of ~695 ENDs in three hours were that loop.
#
# Two changes:
#   1. Admission control. A job starts only when its card really has room for it,
#      and only one job per card is admitted at a time, with a settle window so the
#      weights are resident before the next admission. The memory estimate is the
#      queue line's own second field, in GB.
#   2. A CPU fallback is an environment failure, not a job failure, so it requeues
#      without burning a retry. Only genuine non-zero exits count toward MAXTRY.
R=/opt/dlami/nvme/kvcache; CARD=$1; SLOT=$2; Q=$3; L=$Q.lock; MAXTRY=3
ADMIT=$R/admit_$CARD.lock; HEADROOM=${HEADROOM:-12}; SETTLE=${SETTLE:-150}
mkdir -p $R/retries $R/logs $R/held $R/oom_tries $R/backup_oom; HELD=$R/held/$CARD.$SLOT; : > $ADMIT 2>/dev/null
requeue() { ( flock 9; printf '%s\n' "$1" >> "$Q" ) 9>"$L"; }
# Pause (Oct 28): PAUSE_ALL or PAUSE_CARD_<card> in $R stops new work on this card. A runner waits
# before taking a line, gives back a line it holds in admission, and a job killed by pause.sh hard
# (marker in $R/paused_kill/<tag>) goes back to the head of the queue without charging a retry.
paused(){ [ -e $R/PAUSE_ALL ] || [ -e $R/PAUSE_CARD_$CARD ]; }
requeue_head() { ( flock 9; { printf '%s\n' "$1"; cat "$Q"; } > "$Q.h.$CARD.$SLOT" && mv "$Q.h.$CARD.$SLOT" "$Q" ) 9>"$L"; }
freemb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $CARD 2>/dev/null | tr -d ' '; }
while true; do
  while paused; do sleep 30; done
  line=$( ( flock 8; head -n1 "$Q"; sed -i 1d "$Q" ) 8>"$L" )
  [ -z "$line" ] && { rm -f $HELD; break; }
  # A popped line waits here, invisible to the queue, until admission (up to 3 h).
  # Record it so status/reconcile can see it and a killed runner's line can be recovered.
  printf '%s\n' "$line" > $HELD
  read -r prio mem owner tag cmd <<<"$line"
  if [ -z "$cmd" ] || ! printf '%s' "$cmd" | grep -q -- "--output "; then echo "$(date -u +%FT%TZ) card=$CARD slot=$SLOT MALFORMED-DROPPED $(printf '%s' "$line" | cut -c1-80)" >> $R/runner_log.txt; continue; fi
  case "$mem" in ''|*[!0-9]*) mem=24;; esac
  need=$(( (mem + HEADROOM) * 1024 ))

  exec 9>$ADMIT; flock 9
  f=$(freemb); waited=0
  while [ -n "$f" ] && [ "$f" -lt "$need" ] && [ $waited -lt 10800 ] && ! paused; do
    sleep 15; waited=$((waited+15)); f=$(freemb)
  done
  if paused; then requeue_head "$line"; rm -f $HELD; flock -u 9; exec 9>&-; continue; fi
  echo "$(date -u +%FT%TZ) card=$CARD slot=$SLOT START $tag need=${need}MiB free=${f}MiB waited=${waited}s" >> $R/runner_log.txt
  rm -f $HELD
  CUDA_VISIBLE_DEVICES=$CARD bash -c "$cmd" > $R/logs/$tag.log 2>&1 &
  pid=$!
  s=0; while [ $s -lt $SETTLE ] && kill -0 $pid 2>/dev/null; do sleep 5; s=$((s+5)); done
  flock -u 9; exec 9>&-

  # Out of memory (Oct 28). benchmark.py records an OOM as a wrong answer with 0 tokens and goes on,
  # and the rest of the process usually fails too. Stop the job at the first OOM in its log (its
  # python child, not only the bash wrapper), then cut the result file back to the problem before the
  # first error (oom_fix.py, backup in backup_oom/) and requeue at the head with 24 GB more, so fewer
  # jobs share its card; --resume reruns only the cut problems. Up to 4 attempts (max 120 GB).
  OOMRE="CUDA out of memory|OutOfMemoryError"
  while kill -0 $pid 2>/dev/null; do
    if grep -qE "$OOMRE" $R/logs/$tag.log 2>/dev/null; then
      kill $(pgrep -P $pid) $pid 2>/dev/null; sleep 5; kill -9 $(pgrep -P $pid) $pid 2>/dev/null; break; fi
    sleep 60
  done
  wait $pid; rc=$?
  echo "$(date -u +%FT%TZ) card=$CARD slot=$SLOT END rc=$rc $tag" >> $R/runner_log.txt
  if [ -e $R/paused_kill/$tag ]; then rm -f $R/paused_kill/$tag; requeue_head "$line"
    echo "$(date -u +%FT%TZ) card=$CARD slot=$SLOT PAUSED-REQUEUE $tag (no retry charged)" >> $R/runner_log.txt; continue; fi
  out=$(printf '%s' "$cmd" | grep -o -- "--output [^ ]*" | cut -d" " -f2)
  oomlog=0; grep -qE "$OOMRE" $R/logs/$tag.log 2>/dev/null && oomlog=1
  nerr=$(python3 $R/oom_fix.py "$out" 2>>$R/logs/$tag.oomfix.err); case "$nerr" in ''|*[!0-9]*) nerr=0;; esac
  if [ $oomlog = 1 ] || [ $nerr -gt 0 ]; then
    k=$(( $(cat $R/oom_tries/$tag 2>/dev/null || echo 0) + 1 )); echo $k > $R/oom_tries/$tag
    nm=$(( mem + 24 )); [ $nm -gt 120 ] && nm=120
    cp $R/logs/$tag.log $R/logs/$tag.oom$k.log 2>/dev/null
    if [ $k -le 4 ]; then requeue_head "$prio $nm $owner $tag $cmd"
      echo "$(date -u +%FT%TZ) card=$CARD slot=$SLOT OOM-REQUEUE try=$k $tag mem $mem->$nm cut $nerr problems" >> $R/runner_log.txt
    else ( flock 8; printf '%s\n' "$line" >> "$Q.failed" ) 8>"$L"
      echo "$(date -u +%FT%TZ) card=$CARD slot=$SLOT OOM-GIVEUP $tag after $k attempts" >> $R/runner_log.txt; fi
    sleep 20; continue
  fi
  [ $rc -eq 0 ] && continue

  if grep -qE "loaded on CPU despite CUDA|Model loaded on cpu" $R/logs/$tag.log 2>/dev/null; then
    requeue "$line"
    echo "$(date -u +%FT%TZ) card=$CARD slot=$SLOT CPUFALLBACK-REQUEUE $tag (no retry charged)" >> $R/runner_log.txt
    sleep 60; continue
  fi
  n=$(( $(cat $R/retries/$tag 2>/dev/null || echo 0) + 1 )); echo $n > $R/retries/$tag
  if [ $n -lt $MAXTRY ]; then
    requeue "$line"
    echo "$(date -u +%FT%TZ) card=$CARD slot=$SLOT REQUEUE try=$n $tag" >> $R/runner_log.txt; sleep 20
  else
    ( flock 8; printf '%s\n' "$line" >> "$Q.failed" ) 8>"$L"
    echo "$(date -u +%FT%TZ) card=$CARD slot=$SLOT GIVEUP after=$n $tag" >> $R/runner_log.txt
  fi
done
