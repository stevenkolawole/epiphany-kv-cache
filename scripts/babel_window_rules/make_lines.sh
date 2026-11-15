#!/bin/bash
# Write the grid's lines (tag, then benchmark.py arguments) to stdout: 2 models x 4 methods x K in {512,1024,2048,4096}
# x MATH-500 problems in five 100-problem pieces = 160 lines. Lines 2i and 2i+1 are two pieces of one cell, run side
# by side by array task i (grid.sbatch).
source "$(dirname "$0")/common.sh"
for m in llama qwen; do for meth in kv_seg_hs_fill raas_epikv_window r_kv_epikv_window kvpress_snapkv_decoding; do
  for K in 512 1024 2048 4096; do for s in 0 100 200 300 400; do
    echo "${m}_${meth}_K${K}_s${s} $(model_flags $m) $COMMON --resume --n_samples 100 --start_idx $s --cache_sizes $K --methods $meth $(method_flags $meth)"
  done; done; done; done
