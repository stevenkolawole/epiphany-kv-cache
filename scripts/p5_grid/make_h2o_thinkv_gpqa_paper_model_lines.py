# Queue lines (runner4 format: prio mem owner tag cmd) for H2O and ThinKV on GPQA-Diamond with the two paper models,
# under the protocol of the paper's GPQA cells: 18 shards of 11 problems, greedy, cap 16384, K = 1024, 2048, 4096.
# Usage: make_h2o_thinkv_gpqa_paper_model_lines.py <root>
import sys
R = sys.argv[1]
MODELS = [("llama", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"), ("qwen", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")]
ENV = (f"cd {R}/kvcache && export HF_HOME={R}/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 USE_TF=0 TRANSFORMERS_NO_TF=1 "
       f"TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1 && mkdir -p {R}/results_baselines && ")
for tag, model in MODELS:
    for meth in ("h2o", "thinkv_faithful"):
        for K in (1024, 2048, 4096):
            for s in range(0, 198, 11):
                t = f"gpqa_{tag}_{meth}_K{K}_s{s}"
                print(f"3 30 EPIKV {t} {ENV}{R}/venv/bin/python scripts/benchmark.py --model {model} --dataset gpqa "
                      f"--max_new_tokens 16384 --logical_positions --keep_recent_k 128 --resume --n_samples 11 --start_idx {s} "
                      f"--cache_sizes {K} --methods {meth} --attn_impl flash_attention_2 --output {R}/results_baselines/{t}.json")
