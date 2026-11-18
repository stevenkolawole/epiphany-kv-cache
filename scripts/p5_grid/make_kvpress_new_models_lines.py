# Queue lines (runner4 format: prio mem owner tag cmd) for kvpress (SnapKV scoring in its decoding press) on MATH-500
# and GPQA-Diamond with Qwen3-8B, Qwen3-14B and Nemotron-Nano-8B at the settings of their EpiKV rows: greedy, cap 16384,
# the main library stack plus kvpress 0.5.5 (kvpress_overlay). MATH-500 in 25-problem pieces, GPQA in 11-problem pieces.
# Usage: make_kvpress_new_models_lines.py <root> math|gpqa <K ...>; SMOKE=1 prints one 2-problem line per model instead.
import os, sys
R, bench, Ks = sys.argv[1], sys.argv[2], sys.argv[3:]
SMOKE = os.environ.get("SMOKE") == "1"
NEMO = r'--system_prompt "detailed thinking on. Please reason step by step, and put your final answer within \\boxed{}."'
MODELS = [("q3_8b", "Qwen/Qwen3-8B", "", 20), ("nemo", "nvidia/Llama-3.1-Nemotron-Nano-8B-v1", NEMO, 20),
          ("q3_14b", "Qwen/Qwen3-14B", "", 34)]
OUT = f"{R}/results_kvpress_newmodels" + ("_gpqa" if bench == "gpqa" else "") + ("/smoke" if SMOKE else "")
ENV = (f"cd {R}/kvcache_window && export HF_HOME={R}/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 USE_TF=0 "
       f"TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1 && mkdir -p {OUT} && ")
N, PIECE, DS = (500, 25, "math500") if bench == "math" else (198, 11, "gpqa")
for K in Ks:
    for tag, model, extra, mem in MODELS:
        for s in ([0] if SMOKE else range(0, N, PIECE)):
            n = 2 if SMOKE else min(PIECE, N - s)
            t = f"kn_{bench}_{'smoke_' if SMOKE else ''}{tag}_kvpress_K{K}_s{s}"
            cmd = (f"{R}/kvpress_overlay/bin/python scripts/benchmark.py --model {model} --dataset {DS} "
                   f"--max_new_tokens 16384 --logical_positions --keep_recent_k 128 --resume --n_samples {n} "
                   f"--start_idx {s} --cache_sizes {K} --methods kvpress_snapkv_decoding --attn_impl flash_attention_2 {extra}")
            print(f"1 {mem} EPIKV {t} {ENV}{' '.join(cmd.split())} --output {OUT}/{t}.json")
