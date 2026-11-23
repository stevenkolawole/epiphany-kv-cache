#!/bin/bash
# Appends p5_camera_fill.txt to queue3.txt once no GPU process has run for two checks 5 minutes apart.
K=/opt/dlami/nvme/kvcache; F=$K/p5_camera_fill.txt; quiet=0
while [ $quiet -lt 2 ]; do
  if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; then quiet=$((quiet+1)); else quiet=0; fi
  sleep 300
done
( flock 9; cat "$F" >> $K/queue3.txt ) 9>$K/queue3.txt.lock
echo "$(date -u +%FT%TZ) appended $(wc -l < $F) lines" >> $K/p5_camera_gate.log
