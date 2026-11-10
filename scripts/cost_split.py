"""Where EpiKV's time goes, per decode step and per compaction (rebuttal, reviewer dspk W2).

Runs the shipped policy (kv_seg_hs_fill, query read in replace mode, tau=128 steps) through the harness's own
run_one on real MATH-500 problems and times, with a CUDA synchronize around each call:
  forward      the model's decode forward pass (FlashAttention-2), every step
  score        the policy call on steps without a compaction (hidden-state difference, rolling z-score, banking)
  query        _qk_relevance: recompute 8 queries from the stored Band-A states and read them against the cached keys
  segments     _segment_keyvar + _classify_segments: key-variance tiers
  gather       _gather_kept: build the compacted cache tensors
  write_back   benchmark._write_back: put the compacted tensors into the live cache
  select       the rest of a compaction call (z-scores, per-segment top-k, fill)
Also runs no eviction and EpiKV^-Q (kv_seg_hs, tau=128 steps) for the forward-time baseline.
Synchronous timing adds a little to every timed call, so shares are upper bounds on the policy's cost.
One process, one exclusive card. Output: one JSON with per-run totals and per-compaction records.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B  # noqa: E402
import eviction as E  # noqa: E402  benchmark put src/ on the path; this is its module

T = defaultdict(float)
N = defaultdict(int)
COMP = []
_in_comp = {"on": False, "parts": None}


def _sync():
    torch.cuda.synchronize()


def timed(name, fn):
    def wrap(*a, **k):
        _sync(); t0 = time.perf_counter()
        r = fn(*a, **k)
        _sync(); dt = time.perf_counter() - t0
        T[name] += dt; N[name] += 1
        if _in_comp["parts"] is not None:
            _in_comp["parts"][name] = _in_comp["parts"].get(name, 0.0) + dt
        return r
    return wrap


def patch():
    cls = E.KVSegHSEviction
    cls._qk_relevance = timed("query", cls._qk_relevance)
    cls._segment_keyvar = timed("segments", cls._segment_keyvar)
    cls._classify_segments = timed("segments", cls._classify_segments)
    E._gather_kept = timed("gather", E._gather_kept)
    B._write_back = timed("write_back", B._write_back)
    orig_evict = cls.evict_past_key_values

    def evict(self, pkv, hs):
        seq = pkv[0][0].shape[2]
        _in_comp["parts"] = {}
        _sync(); t0 = time.perf_counter()
        out = orig_evict(self, pkv, hs)
        _sync(); dt = time.perf_counter() - t0
        parts = _in_comp["parts"]; _in_comp["parts"] = None
        if out[0][0].shape[2] != seq:
            sub = sum(parts.get(k, 0.0) for k in ("query", "segments", "gather"))
            T["select"] += dt - sub; N["compactions"] += 1
            COMP.append({"cache_before": seq, "cache_after": int(out[0][0].shape[2]),
                         "total_ms": dt * 1e3, **{k + "_ms": v * 1e3 for k, v in parts.items()},
                         "select_ms": (dt - sub) * 1e3})
        else:
            T["score"] += dt; N["score"] += 1
        return out
    cls.evict_past_key_values = evict

    orig_run_one = B.run_one

    def run_one(model, *a, **k):
        if not getattr(model, "_cost_wrapped", False):
            f = model.forward
            model.forward = timed("forward", f)
            model._cost_wrapped = True
        return orig_run_one(model, *a, **k)
    B.run_one = run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--band_a_layer", type=int, default=None)
    ap.add_argument("--band_b_layer", type=int, default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--Ks", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    patch()
    B.LOGICAL_POSITIONS = True
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, cache_dir=B._hf_cache())
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0",
                                                 attn_implementation="flash_attention_2", cache_dir=B._hf_cache())
    model.eval()
    dev = next(model.parameters()).device
    probs = B.load_problems("math500", a.n, a.start_idx)
    ship = dict(refresh_tau=128, tau_mode="steps", query_sim_weight=1.0, query_sim_mode="replace",
                query_mode="qk", query_sim_window=8, band_a_layer=a.band_a_layer, band_b_layer=a.band_b_layer)
    runs = [("none", 1024, {})]
    for K in a.Ks:
        runs += [("kv_seg_hs_fill", K, ship),
                 ("kv_seg_hs", K, dict(refresh_tau=128, tau_mode="steps",
                                       band_a_layer=a.band_a_layer, band_b_layer=a.band_b_layer))]
    out = {"meta": {"model": a.model, "n": len(probs), "start_idx": a.start_idx, "max_new_tokens": a.max_new_tokens,
                    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                    "note": "synchronous timing around every timed call"}, "runs": []}
    for method, K, kw in runs:
        T.clear(); N.clear(); COMP.clear()
        tokens = 0; wall = 0.0; correct = 0
        for p in probs:
            ev = B.make_eviction(method, K, 128, **{k: v for k, v in kw.items() if v is not None})
            if ev is not None and hasattr(ev, "set_model"):
                ev.set_model(model)
            r = B.run_one(model, tok, p, method, ev, a.max_new_tokens, dev)
            tokens += r["n_tokens_generated"]; wall += r["wall_time_s"]; correct += int(bool(r["correct"]))
        rec = {"method": method, "K": K, "tokens": tokens, "wall_s": wall, "correct": correct,
               "seconds": dict(T), "calls": dict(N),
               "ms_per_token": {k: v * 1e3 / max(tokens, 1) for k, v in T.items()},
               "compactions": list(COMP)}
        out["runs"].append(rec)
        print(method, K, tokens, {k: round(v, 3) for k, v in rec["ms_per_token"].items()}, flush=True)
        Path(a.output).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
