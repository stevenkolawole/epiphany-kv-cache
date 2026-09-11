#!/bin/bash
# Equal-load, long-context timing run for the trade figure and the speed
# table: ONE process per card, AIME-2024 (30 problems, 16,384-token cap),
# K=8192, the main methods. Throughput = generated tokens / wall time.
# Run on an otherwise idle box (all grid workers finished).
#   MODEL=... [BANDS=...] OUT=~/results_timing_llama bash timing_cells.sh
set -u
MODEL=${MODEL:?}
BANDS=${BANDS:-}
OUT=${OUT:?}
LOGS=${LOGS:-$OUT/../logs_$(basename "$OUT")}
DS=${DS:-aime2024}
K=${K:-8192}
NGPU=${NGPU:-8}
cd "$HOME/kvcache"
export HF_HOME=$HOME/hf_cache TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export USE_TF=0 TRANSFORMERS_NO_TF=1 HF_HUB_OFFLINE=1
mkdir -p "$OUT" "$LOGS"
# method:impl:extra ; none first (the reference), then the paper's main rows
CELLS=(
  "none:flash_attention_2:"
  "kv_seg_hs:flash_attention_2:--refresh_tau 128"
  "kv_seg_hs:flash_attention_2:--refresh_tau 1"
  "hs_variance:flash_attention_2:"
  "kv_key:flash_attention_2:"
  "raas:eager:"
  "h2o:eager:"
  "r_kv:eager:"
  "longflow:eager:"
  "thinkv_faithful:eager:"
)
i=0
for cell in "${CELLS[@]}"; do
  method=${cell%%:*}; rest=${cell#*:}; impl=${rest%%:*}; extra=${rest#*:}
  tag="${method}$( [ -n "$extra" ] && echo "_$(echo $extra | tr -d ' -' )" )_${DS}_k${K}"
  gpu=$((i % NGPU)); i=$((i + 1))
  band=""; [ "$impl" = "flash_attention_2" ] && [ "$method" != "none" ] && band="$BANDS"
  (
    while [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader -i $gpu | wc -l)" -gt 0 ]; do sleep 60; done
    echo "[timing] gpu=$gpu start $tag $(date -u)"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/benchmark.py \
        --model "$MODEL" --dataset "$DS" --methods "$method" \
        --cache_sizes "$K" --max_new_tokens 16384 --n_samples 30 --start_idx 0 \
        --attn_impl "$impl" --logical_positions $band $extra \
        --output "$OUT/$tag.json" > "$LOGS/$tag.log" 2>&1
    echo "[timing] gpu=$gpu done  $tag rc=$? $(date -u)"
  ) &
  sleep 5
done
wait
echo "[timing] all done $(date -u)"
