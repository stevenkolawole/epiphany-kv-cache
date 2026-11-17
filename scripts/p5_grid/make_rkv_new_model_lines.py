# Queue lines (runner4 format: prio mem owner tag cmd) for the released R-KV code against EpiKV on the three new models,
# both in the R-KV environment (transformers 4.55) so each pair shares one stack. MATH-500 in 25-problem pieces,
# greedy, cap 16384, K given on the command line. Usage: make_rkv_new_model_lines.py <root> <K ...>
import sys
R, Ks = sys.argv[1], sys.argv[2:]
EPI = "--refresh_tau 128 --tau_mode steps --query_sim_mode replace --query_sim_weight 1 --query_mode qk --query_sim_window 8"
NEMO = r'--system_prompt "detailed thinking on. Please reason step by step, and put your final answer within \\boxed{}."'
MODELS = [("q3_8b", "Qwen/Qwen3-8B", "--band_a_layer 11 --band_b_layer 24", "", 20),
          ("nemo", "nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "", NEMO, 20),
          ("q3_14b", "Qwen/Qwen3-14B", "--band_a_layer 12 --band_b_layer 26", "", 34)]
ENV = (f"cd {R}/kvcache_window && export HF_HOME={R}/hf_cache HF_HUB_OFFLINE=1 USE_TF=0 TRANSFORMERS_NO_TF=1 "
       f"TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1 && mkdir -p {R}/results_rkv_newmodels && ")
for K in Ks:
    for tag, model, bands, extra, mem in MODELS:
        for meth, flags in (("rkv_official", ""), ("kv_seg_hs_fill", f"{EPI} {bands}")):
            for s in range(0, 500, 25):
                n = 20 if s == 475 else 25
                t = f"rn_{tag}_{meth}_K{K}_s{s}"
                cmd = (f"{R}/rkv_venv/bin/python scripts/benchmark.py --model {model} --dataset math500 --max_new_tokens 16384 "
                       f"--logical_positions --keep_recent_k 128 --resume --n_samples {n} --start_idx {s} --cache_sizes {K} "
                       f"--methods {meth} --attn_impl flash_attention_2 {flags} {extra}").rstrip()
                print(f"1 {mem} EPIKV {t} {ENV}{' '.join(cmd.split(' '))} --output {R}/results_rkv_newmodels/{t}.json")
