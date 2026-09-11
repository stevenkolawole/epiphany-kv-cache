#!/bin/bash
# EpiKV-Seg at refresh_tau=128 (evict at a segment boundary every 128 decode
# steps, the vLLM port's operating point) under LOGICAL positions, alongside a
# running grid: one extra process per card, results in the grid's own
# results dir with a tau128 tag. The tau sweep on Babel (Llama MATH, n=100)
# gave 62 -> 74% at K=1024 for tau 1 -> 128, so this is the candidate
# headline configuration and must exist for every model x dataset.
#   MODE=math  MODEL=... [BANDS="--band_a_layer 20 --band_b_layer 24"] OUT=~/results_llama_logical bash tau128_cells.sh
#   MODE=aime  MODEL=... [BANDS=...] OUT=~/results_aime_logical bash tau128_cells.sh
set -u
MODE=${MODE:-math}
MODEL=${MODEL:?}
BANDS=${BANDS:-}
OUT=${OUT:?}
NGPU=${NGPU:-8}
EXTRA=${EXTRA:-}          # e.g. "--band_mode a_only"; TAGSUF names the variant
TAGSUF=${TAGSUF:-tau128}
LOGS=${LOGS:-$OUT/../logs_tau128_$(basename "$OUT")}
cd "$HOME/kvcache"
export HF_HOME=$HOME/hf_cache TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export USE_TF=0 TRANSFORMERS_NO_TF=1 HF_HUB_OFFLINE=1
mkdir -p "$OUT" "$LOGS"

run_cell () {  # gpu tag dataset K nsamp cap
  local gpu=$1 tag=$2 ds=$3 K=$4 n=$5 cap=$6
  if [ -f "$OUT/$tag.json" ]; then echo "[tau128] skip $tag"; return; fi
  echo "[tau128] gpu=$gpu start $tag $(date -u)"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/benchmark.py \
      --model "$MODEL" --dataset "$ds" --methods kv_seg_hs \
      --cache_sizes "$K" --max_new_tokens "$cap" --n_samples "$n" --start_idx 0 \
      --attn_impl flash_attention_2 --logical_positions --refresh_tau 128 $BANDS $EXTRA \
      --output "$OUT/$tag.json" > "$LOGS/$tag.log" 2>&1
  local rc=$?
  local errs; errs=$(grep -c ERROR "$LOGS/$tag.log" 2>/dev/null || echo 0)
  echo "[tau128] gpu=$gpu done  $tag rc=$rc errors=$errs $(date -u)"
  if [ "$errs" -gt 0 ]; then mv "$OUT/$tag.json" "$OUT/$tag.json.ERRORED" 2>/dev/null; fi
}

gpu=0
if [ "$MODE" = math ]; then
  for K in 512 1024 2048 4096; do
    run_cell $((gpu % NGPU)) "kv_seg_hs_${TAGSUF}_K$K" math500 "$K" 100 8192 &
    gpu=$((gpu + 1)); sleep 20
  done
else
  for K in 8192 4096; do
    for ds in aime2024 aime2025 aime2026; do
      run_cell $((gpu % NGPU)) "kv_seg_hs_${TAGSUF}_${ds}_k${K}_s0" "$ds" "$K" 30 16384 &
      gpu=$((gpu + 1)); sleep 20
    done
  done
fi
wait
echo "[tau128] all done $(date -u)"
