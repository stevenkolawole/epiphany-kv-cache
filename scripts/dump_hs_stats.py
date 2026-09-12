#!/usr/bin/env python3
"""Dump per-index statistics of the hidden_states tuple for a fixed prompt, to
compare how two transformers versions index/return hidden states (the
hidden-state scorers gain 4-6 points on 5.2 vs 4.57.1 while attention-based
ones do not). Run on each stack:

    python scripts/dump_hs_stats.py --out hs_stats_<stack>.json

Compare: index k on one stack should match some index on the other by the
per-token L2 norms and the cosine between the vectors at the last token.
"""
import argparse
import json
import os

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
ap.add_argument("--out", required=True)
a = ap.parse_args()
cache = os.environ.get("HF_HOME")
tok = AutoTokenizer.from_pretrained(a.model, cache_dir=cache)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, cache_dir=cache).cuda().eval()
prompt = "Let $n$ be the smallest positive integer such that $n^2 + 1$ is divisible by 13. Then n equals"
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    out = model(ids, output_hidden_states=True, use_cache=True)
hs = out.hidden_states
rec = {"transformers": transformers.__version__, "torch": torch.__version__,
       "num_hidden_layers": model.config.num_hidden_layers, "len_hidden_states": len(hs),
       "n_tokens": int(ids.shape[1]),
       "norm_last_token": [float(h[0, -1].float().norm()) for h in hs],
       "mean_abs_last_token": [float(h[0, -1].float().abs().mean()) for h in hs],
       "last_token_first8": [[float(x) for x in h[0, -1, :8].float()] for h in hs],
       "final_norm_applied_to_last": float(model.model.norm(hs[-2][0, -1:]).float().norm()) if len(hs) > 1 else None}
json.dump(rec, open(a.out, "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("transformers", "torch", "num_hidden_layers", "len_hidden_states")}))
print("norms:", [round(x, 1) for x in rec["norm_last_token"]])
