# Sourced by the window-rule scripts: one interpreter, offline caches, a CUDA preflight, model and method flags.
# The interpreter is a venv over the shared kvcache conda env (torch 2.10.0+cu128, transformers 5.2.0, flash-attn
# 2.8.3) with only kvpress 0.5.5, cachetools, fire 0.6.0, termcolor and pytest added (setup_overlay.sh), so every
# method, kvpress included, runs on one library stack and the shared env is not modified.
# HF_HOME: the cache the rebuttal sampling runs loaded both distill models and MATH-500 from with this harness;
# under /data/hf_cache/skolawol the models are not where benchmark.py (cache_dir=HF_HOME) looks.
PY=$HOME/envs/kvpress_overlay/bin/python
CODE=$HOME/workspace/kvcache_window
OUT=$HOME/rebuttal_window_rules
export USE_TF=0 TRANSFORMERS_NO_TF=1 HF_HOME=/data/user_data/skolawol/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1
preflight(){ $PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('GPU', torch.cuda.get_device_name(0))"; }
model_flags(){ case $1 in
  llama) echo "--model deepseek-ai/DeepSeek-R1-Distill-Llama-8B";;
  qwen)  echo "--model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --band_a_layer 20 --band_b_layer 24";;
  *) echo "unknown model $1" >&2; exit 1;; esac; }
# EpiKV as shipped; the window rules hard-wire the same window, interval and step rule (WindowRuleEviction).
EPIKV_FLAGS="--refresh_tau 128 --tau_mode steps --query_sim_mode replace --query_sim_weight 1 --query_mode qk --query_sim_window 8"
method_flags(){ [ "$1" = kv_seg_hs_fill ] && echo "$EPIKV_FLAGS"; }
COMMON="--dataset math500 --max_new_tokens 16384 --logical_positions --keep_recent_k 128 --attn_impl flash_attention_2"
