# Queue lines (runner4 format: prio mem owner tag cmd) for kvpress (SnapKV scoring in its decoding press) on
# GPQA-Diamond with the two paper models at the settings of the GPQA table: greedy, cap 16384, the main library stack
# plus kvpress 0.5.5 (kvpress_overlay). 18 pieces of 11 problems. Usage: make_kvpress_gpqa_paper_models_lines.py <root> <K ...>
import sys
R, Ks = sys.argv[1], sys.argv[2:]
MODELS = [("llama", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"), ("qwen", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")]
OUT = f"{R}/results_kvpress_gpqa"
ENV = (f"cd {R}/kvcache_window && export HF_HOME={R}/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 USE_TF=0 "
       f"TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1 && mkdir -p {OUT} && ")
for K in Ks:
    for tag, model in MODELS:
        for s in range(0, 198, 11):
            t = f"kg_{tag}_kvpress_K{K}_s{s}"
            cmd = (f"{R}/kvpress_overlay/bin/python scripts/benchmark.py --model {model} --dataset gpqa "
                   f"--max_new_tokens 16384 --logical_positions --keep_recent_k 128 --resume --n_samples 11 "
                   f"--start_idx {s} --cache_sizes {K} --methods kvpress_snapkv_decoding --attn_impl flash_attention_2")
            print(f"1 20 EPIKV {t} {ENV}{cmd} --output {OUT}/{t}.json")
