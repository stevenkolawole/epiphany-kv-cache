# Lines that fill the open cells of the AIME and GPQA tables (Qwen3-8B, Qwen3-14B, Nemotron-Nano-8B), greedy, AIME cap 32,768.
#   p5     baselines + ablations on Qwen3-14B and Nemotron, one problem per piece     -> <out>/p5_aime_fill.txt
#   p5     released R-KV + EpiKV on R-KV's stack, all three models, 5-problem pieces   -> <out>/p5_aime_rkv.txt
#   babel  baselines + ablations on Qwen3-8B (same stack as that column), one problem  -> <out>/babel_aime_fill_lines.txt
#   babel  LongFlow (native) on Qwen3-8B, GPQA-Diamond and AIME                        -> <out>/babel_longflow_lines.txt
#   p5     EpiKV^-Q (no query flags) on Qwen3-14B and Nemotron MATH-500, and on all three models GPQA-Diamond
#          -> <out>/p5_epikv_noq.txt (the earlier "seg" cells of those models carried the query flags)
# Usage: make_aime_fill_lines.py <p5 root> <out dir>
import sys
R, O = sys.argv[1], sys.argv[2]
EPI = "--refresh_tau 128 --tau_mode steps --query_sim_mode replace --query_sim_weight 1 --query_mode qk --query_sim_window 8"
NEMO = r'--system_prompt "detailed thinking on. Please reason step by step, and put your final answer within \\boxed{}."'
M = {"q3_8b": ("Qwen/Qwen3-8B", "--band_a_layer 11 --band_b_layer 24", "", 20),
     "q14b": ("Qwen/Qwen3-14B", "--band_a_layer 12 --band_b_layer 26", "", 34),
     "nemo": ("nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "", NEMO, 20)}
DS = ("aime2024", "aime2025", "aime2026")
SEG = "--refresh_tau 128 --tau_mode steps"   # EpiKV^-Q: no query flags (as the paper models' cells)
FILL = [("kv_seg_hs", "seg"), ("lag_kv", ""), ("h2o", ""), ("thinkv_faithful", ""),
        ("hs_variance_detrend", "bands"), ("hs_variance", ""), ("kv_key", "")]
base = "--logical_positions --keep_recent_k 128 --resume --attn_impl flash_attention_2"

def fill_args(model, bands, extra, ds, s, meth, fl, K):
    fl = {"seg": f"{SEG} {bands}", "bands": bands, "": ""}[fl]
    return " ".join(f"--dataset {ds} --max_new_tokens 32768 {base} --n_samples 1 --start_idx {s} --model {model} "
                    f"--cache_sizes {K} --methods {meth} {fl} {extra}".split())

with open(f"{O}/p5_aime_fill.txt", "w") as p5, open(f"{O}/babel_aime_fill_lines.txt", "w") as bb:
    for key, (model, bands, extra, mem) in M.items():
        out = f"{R}/results_aime_newmodels"
        env = (f"cd {R}/kvcache && export HF_HOME={R}/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 USE_TF=0 TRANSFORMERS_NO_TF=1 "
               f"TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1 && mkdir -p {out} && ")
        for meth, fl in FILL:
            for K in (4096, 8192):
                for ds in DS:
                    for s in range(30):
                        t = f"{key}_{meth}_{ds}_k{K}_s{s}"
                        a = fill_args(model, bands, extra, ds, s, meth, fl, K)
                        if key == "q3_8b":
                            bb.write(f"{t.replace('q3_8b_', '', 1)} {a}\n")
                        else:
                            p5.write(f"5 {mem} EPIKV aime_{t} {env}{R}/venv/bin/python scripts/benchmark.py {a} --output {out}/{t}.json\n")

with open(f"{O}/p5_aime_rkv.txt", "w") as f:
    out = f"{R}/results_rkv_aime_newmodels"
    env = (f"cd {R}/kvcache_window && export HF_HOME={R}/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 USE_TF=0 TRANSFORMERS_NO_TF=1 "
           f"TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1 && mkdir -p {out} && ")
    for K in (4096, 8192):
        for key, (model, bands, extra, mem) in M.items():
            for ds in DS:
                for meth, fl in (("rkv_official", ""), ("kv_seg_hs_fill", f"{EPI} {bands}")):
                    for s in range(0, 30, 5):
                        t = f"ra_{key}_{meth}_{ds}_K{K}_s{s}"
                        cmd = (f"{R}/rkv_venv/bin/python scripts/benchmark.py --model {model} --dataset {ds} --max_new_tokens 32768 "
                               f"--logical_positions --keep_recent_k 128 --resume --n_samples 5 --start_idx {s} --cache_sizes {K} "
                               f"--methods {meth} --attn_impl flash_attention_2 {fl} {extra}")
                        f.write(f"4 {mem} EPIKV {t} {env}{' '.join(cmd.split())} --output {out}/{t}.json\n")

with open(f"{O}/babel_longflow_lines.txt", "w") as f:
    for K in (512, 1024, 2048, 4096):
        for s in range(0, 198, 11):
            f.write(f"lf Qwen/Qwen3-8B {K} {s} 11 16384 $HOME/lf_results/gpqa_q3_8b gpqa\n")
    for K in (4096, 8192):
        for ds in DS:
            for s in range(0, 30, 5):
                f.write(f"lf Qwen/Qwen3-8B {K} {s} 5 32768 $HOME/lf_results/aime_q3_8b {ds}\n")

with open(f"{O}/p5_epikv_noq.txt", "w") as f:
    env = (f"cd {R}/kvcache && export HF_HOME={R}/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 USE_TF=0 TRANSFORMERS_NO_TF=1 "
           f"TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1 PYTHONUNBUFFERED=1 && mkdir -p {R}/results_epikv_noq && ")
    grid = [("math500", key, K, s, 25) for key in ("q14b", "nemo") for K in (512, 1024, 2048, 4096) for s in range(0, 500, 25)]
    grid += [("gpqa", key, K, s, 11) for key in M for K in (512, 1024, 2048, 4096) for s in range(0, 198, 11)]
    for ds, key, K, s, n in grid:
        model, bands, extra, mem = M[key]
        t = f"{'math' if ds == 'math500' else ds}_{key}_seg_noquery_K{K}_s{s}"
        cmd = (f"{R}/venv/bin/python scripts/benchmark.py --model {model} --dataset {ds} --max_new_tokens 16384 --logical_positions "
               f"--keep_recent_k 128 --resume --n_samples {n} --start_idx {s} --cache_sizes {K} --methods kv_seg_hs "
               f"--attn_impl flash_attention_2 {SEG} {bands} {extra}")
        f.write(f"3 {max(mem, 30)} EPIKV {t} {env}{' '.join(cmd.split())} --output {R}/results_epikv_noq/{t}.json\n")
