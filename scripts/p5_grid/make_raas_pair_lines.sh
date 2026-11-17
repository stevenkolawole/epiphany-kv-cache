# Queue lines (runner4 format: prio mem owner tag cmd) for original RaaS (eager attention) against RaaS's rule on
# EpiKV's recomputed window, both on one host so every pair shares hardware and libraries. MATH-500 in 25-problem
# pieces, greedy, cap 16384. Usage: make_raas_pair_lines.sh <root with kvcache_window/, venv/, hf_cache/> <K ...>
R=$1; shift
for K in "$@"; do for m in llama qwen; do for meth in raas raas_epikv_window; do for s in $(seq 0 25 475); do
  n=25; [ $s -eq 475 ] && n=20
  [ $m = llama ] && mf="--model deepseek-ai/DeepSeek-R1-Distill-Llama-8B" || mf="--model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --band_a_layer 20 --band_b_layer 24"
  tag=rp_${m}_${meth}_K${K}_s${s}
  echo "1 20 EPIKV $tag cd $R/kvcache_window && export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1 && mkdir -p $R/results_raas_pair && $R/venv/bin/python scripts/benchmark.py $mf --dataset math500 --max_new_tokens 16384 --logical_positions --keep_recent_k 128 --resume --n_samples $n --start_idx $s --cache_sizes $K --methods $meth --attn_impl flash_attention_2 --output $R/results_raas_pair/$tag.json"
done; done; done; done
