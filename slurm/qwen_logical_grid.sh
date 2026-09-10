#!/bin/bash
# Qwen-7B MATH-500 grid under LOGICAL post-eviction positions on one 8xA100
# box. Same queue-of-cells design as llama_logical_grid.sh; bands travel
# explicitly (Qwen's are 20/24, from the June signal ablation).
set -u
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
BAND_A=20
BAND_B=24
NSAMP=${NSAMP:-100}
OUT=$HOME/results_qwen_logical
LOGS=$HOME/logs_qwen_logical
QUEUE=$OUT/queue.txt
LOCK=$OUT/queue.lock
cd "$HOME/kvcache"
export HF_HOME=$HOME/hf_cache TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export USE_TF=0 TRANSFORMERS_NO_TF=1 HF_HUB_OFFLINE=1
mkdir -p "$OUT" "$LOGS"

python3 - <<'PREFLIGHT' || { echo "[grid] ABORT: need 8 CUDA devices"; exit 1; }
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 8
print(f"[preflight] torch={torch.__version__} cuda={ok} n={torch.cuda.device_count()}", flush=True)
sys.exit(0 if ok else 1)
PREFLIGHT

METHODS=(
  "h2o:eager" "raas:eager" "thinkv_faithful:eager" "r_kv:eager" "longflow:eager"
  "kv_seg_hs:flash_attention_2" "hs_variance_detrend:flash_attention_2"
  "hs_variance:flash_attention_2" "band_adaptive_hs:flash_attention_2"
  "kv_val:flash_attention_2" "kv_key:flash_attention_2"
  "lag_kv_key:flash_attention_2" "lag_kv:flash_attention_2"
)
if [ ! -f "$QUEUE" ]; then
  : > "$QUEUE"
  for K in 512 1024 2048 4096; do
    for spec in "${METHODS[@]}"; do echo "$spec:$K" >> "$QUEUE"; done
  done
  echo "none:flash_attention_2:1024" >> "$QUEUE"
  echo "[grid] queue built: $(wc -l < "$QUEUE") cells"
fi

next_cell () { flock "$LOCK" bash -c "head -n1 '$QUEUE'; sed -i '1d' '$QUEUE'"; }

worker () {
  local gpu=$1 slot=$2
  while :; do
    local cell; cell=$(next_cell)
    [ -z "$cell" ] && break
    local method=${cell%%:*}; local rest=${cell#*:}; local impl=${rest%%:*}; local K=${rest##*:}
    local tag="${method}_K${K}"
    if [ -f "$OUT/$tag.json" ]; then echo "[grid] skip $tag"; continue; fi
    local extra=""
    if [ "$impl" = "flash_attention_2" ] && [ "$method" != "none" ]; then
      extra="--band_a_layer $BAND_A --band_b_layer $BAND_B"
    fi
    echo "[grid] gpu=$gpu slot=$slot start $tag $(date -u)"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/benchmark.py \
        --model "$MODEL" --dataset math500 --methods "$method" \
        --cache_sizes "$K" --max_new_tokens 8192 --n_samples "$NSAMP" --start_idx 0 \
        --attn_impl "$impl" --logical_positions $extra \
        --output "$OUT/$tag.json" > "$LOGS/$tag.log" 2>&1
    local rc=$?
    local errs; errs=$(grep -c ERROR "$LOGS/$tag.log" 2>/dev/null || echo 0)
    echo "[grid] gpu=$gpu slot=$slot done  $tag rc=$rc errors=$errs $(date -u)"
    if [ "$errs" -gt 0 ]; then
      echo "[grid] QUARANTINE $tag: $errs per-problem errors"
      mv "$OUT/$tag.json" "$OUT/$tag.json.ERRORED" 2>/dev/null
    fi
  done
  echo "[grid] gpu=$gpu slot=$slot idle $(date -u)"
}

for gpu in 0 1 2 3 4 5 6 7; do
  worker "$gpu" a &
  sleep 20
  worker "$gpu" b &
done
wait
echo "[grid] all workers finished $(date -u)"
