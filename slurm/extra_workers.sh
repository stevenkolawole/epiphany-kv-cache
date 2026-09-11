#!/bin/bash
# Add N extra grid workers per card on a running box, draining the SAME
# queue file (flock) with the same tags as the grid script. Exists because
# nvidia-smi's "utilisation" hid that each batch-size-one decode process
# only uses ~1/3 of the memory bus: three per card left power at half the
# limit, so more processes per card raise aggregate throughput.
#   MODE=aime MODEL=... BANDS="" OUT=~/results_aime_logical LOGS=~/logs_aime_logical PER_CARD=1 NGPU=8 bash extra_workers.sh
#   MODE=math MODEL=... OUT=~/results_llama_logical LOGS=~/logs_llama_logical PER_CARD=1 bash extra_workers.sh
set -u
MODE=${MODE:-aime}
MODEL=${MODEL:?}
BANDS=${BANDS:-}
OUT=${OUT:?}
LOGS=${LOGS:?}
PER_CARD=${PER_CARD:-1}
NGPU=${NGPU:-8}
GPU0=${GPU0:-0}          # first card to use; skip cards already near their memory limit
NSAMP=${NSAMP:-30}
QUEUE=$OUT/queue.txt
LOCK=$OUT/queue.lock
cd "$HOME/kvcache"
export HF_HOME=$HOME/hf_cache TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export USE_TF=0 TRANSFORMERS_NO_TF=1 HF_HUB_OFFLINE=1

next_cell () { flock "$LOCK" bash -c "head -n1 '$QUEUE'; sed -i '1d' '$QUEUE'"; }

worker () {
  local gpu=$1 slot=$2
  while :; do
    local cell; cell=$(next_cell)
    [ -z "$cell" ] && break
    local method impl K ds tag extra=""
    if [ "$MODE" = aime ]; then
      IFS=: read -r method impl K ds <<< "$cell"
      tag="${method}_${ds}_k${K}_s0"
      n=$NSAMP; cap=16384
    else
      method=${cell%%:*}; local rest=${cell#*:}; impl=${rest%%:*}; K=${rest##*:}; ds=math500
      tag="${method}_K${K}"
      n=100; cap=8192
    fi
    if [ -f "$OUT/$tag.json" ]; then echo "[extra] skip $tag"; continue; fi
    if [ "$impl" = "flash_attention_2" ] && [ "$method" != "none" ]; then extra="$BANDS"; fi
    echo "[extra] gpu=$gpu slot=$slot start $tag $(date -u)"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/benchmark.py \
        --model "$MODEL" --dataset "$ds" --methods "$method" \
        --cache_sizes "$K" --max_new_tokens "$cap" --n_samples "$n" --start_idx 0 \
        --attn_impl "$impl" --logical_positions $extra \
        --output "$OUT/$tag.json" > "$LOGS/$tag.log" 2>&1
    local rc=$?
    local errs; errs=$(grep -c ERROR "$LOGS/$tag.log" 2>/dev/null || echo 0)
    echo "[extra] gpu=$gpu slot=$slot done  $tag rc=$rc errors=$errs $(date -u)"
    if [ "$errs" -gt 0 ]; then
      echo "[extra] QUARANTINE $tag: $errs per-problem errors"
      mv "$OUT/$tag.json" "$OUT/$tag.json.ERRORED" 2>/dev/null
    fi
  done
  echo "[extra] gpu=$gpu slot=$slot idle $(date -u)"
}

for gpu in $(seq "$GPU0" $((GPU0 + NGPU - 1))); do
  for s in $(seq 1 "$PER_CARD"); do
    worker "$gpu" "x$s" &
    sleep 10
  done
done
wait
echo "[extra] all extra workers finished $(date -u)"
