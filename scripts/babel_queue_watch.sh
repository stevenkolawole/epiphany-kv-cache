# Watches the Babel queue (~/gridq) from the laptop and exits, printing why, on: a failed line, a new breaker trip,
# a running piece idle 90 min (stalled.sh), no workers while lines remain, or everything done. Checks every 30 min.
B0=$(ssh babel "grep -c BREAKER ~/gridq/events.log")
while :; do
  s=$(ssh babel 'f=$(cat ~/gridq/failed.txt 2>/dev/null | wc -l); b=$(( $(grep -c BREAKER ~/gridq/events.log) - '$B0' ))
    q=$(wc -l < ~/gridq/queue.txt); h=$(ls ~/gridq/held | wc -l); w=$(squeue -u skolawol -n q_worker -h | wc -l)
    st=$(bash ~/gridq/stalled.sh | wc -l); echo "$f $b $q $h $w $st"') || { sleep 600; continue; }
  set -- $s
  [ "$1" -gt 0 ] && { echo "QUEUE_FAIL $s"; break; }
  [ "$2" -gt 0 ] && { echo "BREAKER $s"; break; }
  [ "$6" -gt 0 ] && { echo "STALL $s"; break; }
  [ "$5" -eq 0 ] && [ $(( $3 + $4 )) -gt 0 ] && { echo "NO_WORKERS $s"; break; }
  [ "$3" -eq 0 ] && [ "$4" -eq 0 ] && { echo "ALL_DONE $s"; break; }
  sleep 1800
done
echo "fields: failed breaker queue held workers stalled"
