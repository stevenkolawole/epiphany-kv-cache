# Queue lines (runner4 format: prio mem owner tag cmd) for kvpress (SnapKV scoring in its decoding press) on MATH-500
# with the two paper models at the main accuracy table's settings: greedy, cap 8192, the main library stack plus
# kvpress 0.5.5 (kvpress_overlay). 25-problem pieces. Usage: make_kvpress_main_table_lines.py <root> <K ...>
# With SMOKE=1 in the environment it prints one 2-problem line per model instead (output under smoke/).
import os, sys
R, Ks = sys.argv[1], sys.argv[2:]
SMOKE = os.environ.get("SMOKE") == "1"
MODELS = [("llama", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"), ("qwen", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")]
OUT = f"{R}/results_kvpress_main" + ("/smoke" if SMOKE else "")
ENV = (f"cd {R}/kvcache_window && export HF_HOME={R}/hf_cache HF_HUB_OFFLINE=1 USE_TF=0 TRANSFORMERS_NO_TF=1 "
       f"TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1 && mkdir -p {OUT} && ")
for K in Ks:
    for tag, model in MODELS:
        for s in ([0] if SMOKE else range(0, 500, 25)):
            n = 2 if SMOKE else (20 if s == 475 else 25)
            t = f"kp_{'smoke_' if SMOKE else ''}{tag}_kvpress_snapkv_decoding_K{K}_s{s}"
            cmd = (f"{R}/kvpress_overlay/bin/python scripts/benchmark.py --model {model} --dataset math500 "
                   f"--max_new_tokens 8192 --logical_positions --keep_recent_k 128 --resume --n_samples {n} "
                   f"--start_idx {s} --cache_sizes {K} --methods kvpress_snapkv_decoding --attn_impl flash_attention_2")
            print(f"1 20 EPIKV {t} {ENV}{cmd} --output {OUT}/{t}.json")
