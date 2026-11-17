#!/bin/bash
# Runs queue lines (runner4 format: prio mem owner tag cmd) on a box without the keepalive: SLOTS jobs per card, each
# slot pops the next line under a lock. A line that fails goes to <queue>.failed. Usage: slot_runner.sh <queue> <slots>
Q=$1; SLOTS=${2:-2}; D=$(dirname $Q); mkdir -p $D/slot_logs
pop(){ ( flock 9; l=$(head -1 $Q); [ -n "$l" ] && sed -i 1d $Q; printf '%s\n' "$l" ) 9>$Q.lock; }
for c in $(nvidia-smi --query-gpu=index --format=csv,noheader); do for s in $(seq $SLOTS); do
  ( sleep $(( (s - 1) * 120 ))
    while l=$(pop); [ -n "$l" ]; do read -r prio mem owner tag cmd <<<"$l"; t0=$(date +%s)
      echo "$(date -u +%FT%TZ) START card=$c $tag" >> $D/slot_events.log
      CUDA_VISIBLE_DEVICES=$c bash -c "$cmd" > $D/slot_logs/$tag.log 2>&1; rc=$?
      echo "$(date -u +%FT%TZ) END rc=$rc $(( $(date +%s) - t0 ))s card=$c $tag" >> $D/slot_events.log
      [ $rc -ne 0 ] && ( flock 9; printf '%s\n' "$l" >> $Q.failed ) 9>$Q.lock
    done ) 8>&- &
done; done; wait
