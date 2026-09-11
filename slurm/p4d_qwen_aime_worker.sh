#!/bin/bash
# Qwen-7B pooled-AIME logical cells handed off from the 4xH100 box to p4d
# (8x A100-40GB, one cell per card, daily reboot ~00:10 UTC). Same cell
# format and tags as qwen_aime_logical_grid.sh. Cells are moved from the
# queue to inflight.txt while running and removed on completion; on
# (re)launch, inflight cells without a result go back to the queue, so a
# reboot loses nothing but the partial run.
#   NGPU=5 GPU0=3 bash p4d_qwen_aime_worker.sh
set -u
MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}
BANDS=${BANDS:-"--band_a_layer 20 --band_b_layer 24"}   # Qwen bands; set BANDS="" for Llama
NSAMP=30
NGPU=${NGPU:-5}
GPU0=${GPU0:-3}
OUT=${OUT:-$HOME/results_qwen_aime_logical}
LOGS=${LOGS:-$HOME/logs_qwen_aime_logical}
QUEUE=$OUT/queue.txt
INFLIGHT=$OUT/inflight.txt
LOCK=$OUT/queue.lock
cd "$HOME/kvcache"
export HF_HOME=$HOME/hf_cache TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export USE_TF=0 TRANSFORMERS_NO_TF=1 HF_HUB_OFFLINE=1
mkdir -p "$OUT" "$LOGS"
touch "$QUEUE" "$INFLIGHT"
PY=$HOME/venvs/kvcache/bin/python

$PY - <<PREFLIGHT || { echo "[p4d] ABORT: CUDA"; exit 1; }
import sys, torch
ok = torch.cuda.is_available()
print(f"[preflight] torch={torch.__version__} cuda={ok} n={torch.cuda.device_count()}", flush=True)
sys.exit(0 if ok else 1)
PREFLIGHT

# Re-queue inflight cells that have no result (killed by a reboot).
flock "$LOCK" bash -c "
  while read -r cell; do
    [ -z \"\$cell\" ] && continue
    IFS=: read -r method impl K ds <<< \"\$cell\"
    tag=\"\${method}_\${ds}_k\${K}_s0\"
    if [ ! -f '$OUT'/\$tag.json ]; then echo \"\$cell\"; fi
  done < '$INFLIGHT' > '$OUT'/requeue.tmp
  cat '$OUT'/requeue.tmp '$QUEUE' > '$OUT'/q.tmp && mv '$OUT'/q.tmp '$QUEUE'
  : > '$INFLIGHT'
  echo \"[p4d] requeued \$(wc -l < '$OUT'/requeue.tmp) inflight cells; queue now \$(wc -l < '$QUEUE')\"
"

next_cell () { flock "$LOCK" bash -c "c=\$(head -n1 '$QUEUE'); sed -i '1d' '$QUEUE'; [ -n \"\$c\" ] && echo \"\$c\" >> '$INFLIGHT'; echo \"\$c\""; }
finish_cell () { flock "$LOCK" bash -c "grep -vxF -- '$1' '$INFLIGHT' > '$INFLIGHT'.tmp; mv '$INFLIGHT'.tmp '$INFLIGHT'"; }

worker () {
  local gpu=$1
  while :; do
    local cell; cell=$(next_cell)
    [ -z "$cell" ] && break
    IFS=: read -r method impl K ds <<< "$cell"
    local tag="${method}_${ds}_k${K}_s0"
    if [ -f "$OUT/$tag.json" ]; then finish_cell "$cell"; echo "[p4d] skip $tag"; continue; fi
    local extra=""
    if [ "$impl" = "flash_attention_2" ] && [ "$method" != "none" ]; then
      extra="$BANDS"
    fi
    echo "[p4d] gpu=$gpu start $tag $(date -u)"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/benchmark.py \
        --model "$MODEL" --dataset "$ds" --methods "$method" \
        --cache_sizes "$K" --max_new_tokens 16384 --n_samples "$NSAMP" --start_idx 0 \
        --attn_impl "$impl" --logical_positions $extra \
        --output "$OUT/$tag.json" > "$LOGS/$tag.log" 2>&1
    local rc=$?
    local errs; errs=$(grep -c ERROR "$LOGS/$tag.log" 2>/dev/null || echo 0)
    echo "[p4d] gpu=$gpu done  $tag rc=$rc errors=$errs $(date -u)"
    if [ "$errs" -gt 0 ] || [ ! -f "$OUT/$tag.json" ]; then
      echo "[p4d] QUARANTINE $tag: rc=$rc errors=$errs"
      mv "$OUT/$tag.json" "$OUT/$tag.json.ERRORED" 2>/dev/null
    fi
    finish_cell "$cell"
  done
  echo "[p4d] gpu=$gpu idle $(date -u)"
}

for gpu in $(seq "$GPU0" $((GPU0 + NGPU - 1))); do
  worker "$gpu" &
  sleep 15
done
wait
echo "[p4d] all workers finished $(date -u)"
