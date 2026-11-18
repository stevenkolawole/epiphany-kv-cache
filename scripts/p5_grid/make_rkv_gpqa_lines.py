# Queue lines (runner4 format: prio mem owner tag cmd) for the released R-KV code on GPQA-Diamond with the two paper
# models, in the R-KV environment (transformers 4.55), plus EpiKV in that environment (K=1024 on both models, K=4096 on Qwen-7B) to pair
# with R-KV under one library version. 18 shards of 11 problems, greedy, cap 16384. Usage: make_rkv_gpqa_lines.py <root>
import sys
R = sys.argv[1]
EPI = "--refresh_tau 128 --tau_mode steps --query_sim_mode replace --query_sim_weight 1 --query_mode qk --query_sim_window 8"
MODELS = [("llama", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", ""),
          ("qwen", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "--band_a_layer 20 --band_b_layer 24")]
ENV = (f"cd {R}/kvcache_window && export HF_HOME={R}/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 USE_TF=0 TRANSFORMERS_NO_TF=1 "
       f"TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1 && mkdir -p {R}/results_rkv_gpqa && ")
for tag, model, bands in MODELS:
    for meth, flags, Ks in (("rkv_official", "", (1024, 2048, 4096)), ("kv_seg_hs_fill", f"{EPI} {bands}".strip(), (1024, 4096) if tag == "qwen" else (1024,))):
        for K in Ks:
            for s in range(0, 198, 11):
                t = f"rg_{tag}_{meth}_K{K}_s{s}"
                cmd = (f"{R}/rkv_venv/bin/python scripts/benchmark.py --model {model} --dataset gpqa --max_new_tokens 16384 "
                       f"--logical_positions --keep_recent_k 128 --resume --n_samples 11 --start_idx {s} --cache_sizes {K} "
                       f"--methods {meth} --attn_impl flash_attention_2 {flags}").rstrip()
                print(f"1 20 EPIKV {t} {ENV}{cmd} --output {R}/results_rkv_gpqa/{t}.json")
