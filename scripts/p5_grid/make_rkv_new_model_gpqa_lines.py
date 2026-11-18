# Queue lines (runner4 format: prio mem owner tag cmd) for the released R-KV code against EpiKV on GPQA-Diamond with
# Qwen3-8B, Qwen3-14B and Nemotron-Nano-8B, both in the R-KV environment (transformers 4.55) so each pair shares one
# stack. 18 shards of 11 problems, greedy, cap 16384. Usage: make_rkv_new_model_gpqa_lines.py <root> <K ...>
import sys
R, Ks = sys.argv[1], sys.argv[2:]
EPI = "--refresh_tau 128 --tau_mode steps --query_sim_mode replace --query_sim_weight 1 --query_mode qk --query_sim_window 8"
NEMO = r'--system_prompt "detailed thinking on. Please reason step by step, and put your final answer within \\boxed{}."'
MODELS = [("q3_8b", "Qwen/Qwen3-8B", "--band_a_layer 11 --band_b_layer 24", "", 20),
          ("nemo", "nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "", NEMO, 20),
          ("q3_14b", "Qwen/Qwen3-14B", "--band_a_layer 12 --band_b_layer 26", "", 34)]
OUT = f"{R}/results_rkv_newmodels_gpqa"
ENV = (f"cd {R}/kvcache_window && export HF_HOME={R}/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 USE_TF=0 "
       f"TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1 && mkdir -p {OUT} && ")
for K in Ks:
    for tag, model, bands, extra, mem in MODELS:
        for meth, flags in (("rkv_official", ""), ("kv_seg_hs_fill", f"{EPI} {bands}")):
            for s in range(0, 198, 11):
                t = f"rn_gpqa_{tag}_{meth}_K{K}_s{s}"
                cmd = (f"{R}/rkv_venv/bin/python scripts/benchmark.py --model {model} --dataset gpqa --max_new_tokens 16384 "
                       f"--logical_positions --keep_recent_k 128 --resume --n_samples 11 --start_idx {s} --cache_sizes {K} "
                       f"--methods {meth} --attn_impl flash_attention_2 {flags} {extra}")
                print(f"1 {mem} EPIKV {t} {ENV}{' '.join(cmd.split())} --output {OUT}/{t}.json")
