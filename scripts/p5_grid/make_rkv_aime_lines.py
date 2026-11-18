# Queue lines (runner4 format: prio mem owner tag cmd) for the released R-KV code against EpiKV on pooled AIME
# (2024, 2025, 2026; 30 problems each) with the two paper models, both in the R-KV environment (transformers 4.55) so
# each pair shares one stack. 10-problem pieces, greedy, cap 16384. Usage: make_rkv_aime_lines.py <root> <K ...>
import sys
R, Ks = sys.argv[1], sys.argv[2:]
EPI = "--refresh_tau 128 --tau_mode steps --query_sim_mode replace --query_sim_weight 1 --query_mode qk --query_sim_window 8"
MODELS = [("llama", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", ""),
          ("qwen", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "--band_a_layer 20 --band_b_layer 24")]
OUT = f"{R}/results_rkv_aime"
ENV = (f"cd {R}/kvcache_window && export HF_HOME={R}/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 USE_TF=0 "
       f"TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1 && mkdir -p {OUT} && ")
for K in Ks:
    for tag, model, bands in MODELS:
        for year in (2024, 2025, 2026):
            for meth, flags in (("rkv_official", ""), ("kv_seg_hs_fill", f"{EPI} {bands}")):
                for s in (0, 10, 20):
                    t = f"ra_{tag}_{meth}_aime{year}_K{K}_s{s}"
                    cmd = (f"{R}/rkv_venv/bin/python scripts/benchmark.py --model {model} --dataset aime{year} "
                           f"--max_new_tokens 16384 --logical_positions --keep_recent_k 128 --resume --n_samples 10 "
                           f"--start_idx {s} --cache_sizes {K} --methods {meth} --attn_impl flash_attention_2 {flags}")
                    print(f"1 20 EPIKV {t} {ENV}{' '.join(cmd.split())} --output {OUT}/{t}.json")
