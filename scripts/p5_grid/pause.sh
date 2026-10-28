#!/bin/bash
# pause.sh soft|hard [card ...]   (no cards = all cards)
#   soft: no new jobs start; running jobs finish.  hard: also stop running jobs; their lines go back to the
#   head of the queue without a retry charge, and --resume restarts each from its last finished problem.
# resume.sh [card ...] removes the pause.
R=/opt/dlami/nvme/kvcache; mode=$1; shift; mkdir -p $R/paused_kill
[ "$mode" = soft ] || [ "$mode" = hard ] || { echo "usage: pause.sh soft|hard [card ...]"; exit 1; }
if [ $# -eq 0 ]; then touch $R/PAUSE_ALL; cards="0 1 2 3 4 5 6 7"; else cards="$*"; for c in $cards; do touch $R/PAUSE_CARD_$c; done; fi
[ "$mode" = soft ] && { echo "soft pause on cards: $cards"; exit 0; }
n=0
for p in $(pgrep -f "^/opt/dlami/nvme/kvcache/venv/bin/python scripts/benchmark.py"); do
  c=$(tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep "^CUDA_VISIBLE_DEVICES=" | cut -d= -f2)
  case " $cards " in *" $c "*) ;; *) continue;; esac
  t=$(tr "\0" " " < /proc/$p/cmdline | grep -o -- "--output [^ ]*" | sed "s|.*/||;s|\.json$||"); [ -n "$t" ] || continue
  touch $R/paused_kill/$t; kill $p; n=$((n+1))
done
echo "hard pause on cards: $cards; stopped $n jobs (requeued at the head by their runners)"
