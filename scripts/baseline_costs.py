"""Measure the time and GPU memory each KV-eviction method costs on one card (rebuttal, submission 50503).

Modes (one method per process: R-KV and kvpress patch transformers globally, H2O/RaaS need eager attention):
  steady      decode windows of 128 steps from a cache at the budget K, for K x batch: ms per decode token, the
              compaction step, the policy's own time, resident and peak memory, cache length after each compaction.
  compaction  one compaction from a cache of length L down to K, for L x K x batch: its time and the memory it
              allocates above what is resident before the step (the transient).
  capacity    the largest batch that decodes through one compaction without CUDA OOM at budget K (no eviction:
              at context C, given as --Ks). Prefill of new requests is not part of the probe.
  prefill     one full-prompt prefill at length L, batch 1, with what the method asks of it (all hidden states,
              attention weights, or a prompt compression to K), logits for the last position only.
  dms_decode  the DMS checkpoint's own decode from a short prompt, recording step time against its live cache.

Caches are built from real MATH-500 text (problems and solutions concatenated) by chunked prefill of one sequence
and replicated across the batch. Decode steps feed the next tokens of the same text (teacher forcing) at logical
positions, as the harness does with --logical_positions. The first P=128 tokens are the prompt; the rest count as
decoded tokens, and EpiKV's per-token state is replayed from the prefill's hidden states through the policy's own
update, so the policy holds the state it would have after decoding them.
Batch > 1 for EpiKV: the repo's policy is batch-1 code, so each sequence gets its own policy object; their keep
masks are captured and the batch is gathered once (PerSeq). H2O and RaaS run as written, which shares one keep mask
across the batch (same work as per-sequence masks). R-KV official and kvpress are batched natively.
Every timed call is bracketed by torch.cuda.synchronize(), so times include the synchronisation.
"""
import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B  # noqa: E402
import eviction as E  # noqa: E402  benchmark put src/ on the path
from epikv_batched import BatchedEpiKV, trim_bank  # noqa: E402

P = 128          # prompt tokens at the start of every cache
INTERVAL = 128   # compaction interval shared by EpiKV, R-KV and the kvpress decoding press
LLAMA = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def sync():
    torch.cuda.synchronize()


def gb(x):
    return round(x / 2 ** 30, 3)


def free():
    gc.collect()
    torch.cuda.empty_cache()



def load_model(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, cache_dir=B._hf_cache())
    attn = "eager" if a.method in ("h2o", "raas") else "flash_attention_2"
    kw = dict(device_map="cuda:0", cache_dir=B._hf_cache(), **{B._DTYPE_KW: torch.bfloat16})
    if a.method == "rkv":
        import transformers
        assert transformers.__version__.split(".")[:2] < ["4", "56"], "R-KV official needs transformers < 4.56"
        sys.path.insert(0, str(B._REPO_ROOT))
        from third_party.rkv.monkeypatch import replace_llama, replace_qwen3
        cc = {"method": "rkv", "method_config": {"budget": 10 ** 9, "window_size": 8, "mix_lambda": 0.1,
                                                 "retain_ratio": 0.2, "retain_direction": "last",
                                                 "first_tokens": 4},
              "compression": None, "update_kv": True}
        (replace_qwen3 if "qwen3" in a.model.lower() else replace_llama)(cc)
        model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation=attn, use_cache=True, **kw)
        model.config.update({"divide_method": "step_length", "divide_length": INTERVAL,
                             "compression_content": "all"})
    elif a.method == "dms":
        sys.path.insert(0, str(B._REPO_ROOT))
        from third_party.dms.modeling_qwen3_dms import Qwen3ForCausalLM
        model = Qwen3ForCausalLM.from_pretrained(a.model, **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation=attn, **kw)
    model.eval()
    assert next(model.parameters()).device.type == "cuda", "model is not on the GPU"
    return model, tok, attn


def text_ids(tok, n, device):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=B._hf_cache())
    ids = []
    while len(ids) < n:
        for it in ds:
            ids += tok(it["problem"] + "\n\n" + it["solution"] + "\n\n", add_special_tokens=False).input_ids
            if len(ids) >= n:
                break
    return torch.tensor([ids[:n]], device=device)



def kv_pairs(cache):
    return [(l.keys, l.values) for l in cache.layers]


def set_kv(cache, pairs):
    for layer, (k, v) in zip(cache.layers, pairs):
        layer.keys, layer.values = k, v


def seqlen(cache):
    return int(cache.layers[0].keys.shape[2])


def build(model, ids, S, hs_layers, chunk):
    """Prefill ids[:, :S] in chunks into a fresh DynamicCache; return it and the hidden states at hs_layers."""
    from transformers import DynamicCache
    cache = DynamicCache()
    hs = {l: [] for l in hs_layers}
    dev = ids.device
    for s in range(0, S, chunk):
        e = min(S, s + chunk)
        with torch.no_grad():
            out = model.model(input_ids=ids[:, s:e], past_key_values=cache, use_cache=True,
                              position_ids=torch.arange(s, e, device=dev)[None],
                              cache_position=torch.arange(s, e, device=dev),
                              output_hidden_states=bool(hs_layers))
        for l in hs_layers:
            hs[l].append(out.hidden_states[l][0])
        del out
    return cache, {l: torch.cat(v) for l, v in hs.items()}


def snapshot(cache):
    """CPU copy of one sequence's cache (and R-KV's query window), to restore before each repeat."""
    snap = {"kv": [(k.cpu(), v.cpu()) for k, v in kv_pairs(cache)]}
    if hasattr(cache, "query_cache"):
        snap["q"] = {i: q.cpu() for i, q in cache.query_cache.items()}
    return snap


def restore(snap, Bsz, device):
    """A fresh GPU cache holding Bsz copies of the snapshot's sequence."""
    from transformers import DynamicCache
    cache = DynamicCache()
    for i, (k, v) in enumerate(snap["kv"]):
        cache.update(k.to(device).repeat(Bsz, 1, 1, 1), v.to(device).repeat(Bsz, 1, 1, 1), i)
    if "q" in snap:
        cache.query_cache = {i: q.to(device).repeat(Bsz, 1, 1, 1) for i, q in snap["q"].items()}
    return cache



def make_policy(a, model, K):
    bands = dict(band_a_layer=a.band_a_layer, band_b_layer=a.band_b_layer)
    bands = {k: v for k, v in bands.items() if v is not None}
    if a.method in ("epikv", "epikv_batched"):
        pol = B.make_eviction("kv_seg_hs_fill", K, 128, refresh_tau=INTERVAL, tau_mode="steps",
                              query_sim_weight=1.0, query_sim_mode="replace", query_mode="qk",
                              query_sim_window=8, **bands)
        pol.set_model(model)
        return pol
    if a.method == "epikv_noq":
        return B.make_eviction("kv_seg_hs", K, 128, refresh_tau=INTERVAL, tau_mode="steps", **bands)
    if a.method in ("h2o", "raas"):
        return B.make_eviction(a.method, K, 128)
    return None


def hs_layers(a, pol):
    if isinstance(pol, E.KVSegHSEviction):
        return (pol.band_a_layer + 1, pol.band_b_layer + 1)
    return ()


class _HS:
    """Stands in for a forward pass's hidden_states tuple at one position (only band layers are read)."""

    def __init__(self, hs, t):
        self.hs, self.t = hs, t

    def __getitem__(self, i):
        return self.hs[i][self.t:self.t + 1][None]


def prime(a, pol, hs, S0, since):
    """Put a fresh policy into the state it would hold after decoding tokens P..S0-1 of the text."""
    if isinstance(pol, E.KVSegHSEviction):
        pol.reset(prefill_len=P)
        pol.set_prefill_end(_HS(hs, P - 1))
        nl = len(pol._model.model.layers) if pol._model is not None else 64
        dummy = torch.zeros(1, 1, 1, 1, device=next(iter(hs.values())).device)
        dkv = tuple((dummy, dummy) for _ in range(nl))   # seq_len 1 <= budget: state update only
        for t in range(P, S0):
            pol.evict_past_key_values(dkv, _HS(hs, t))
        pol._since_compaction = since
    elif isinstance(pol, E.H2OEviction):
        pol.reset()
    elif isinstance(pol, E.RaaSEviction):
        pol.reset(prefill_len=P)
    return pol


def clone_policy(pol):
    """An independent copy of a primed policy (lists and tensors copied; the model reference shared)."""
    new = copy.copy(pol)
    for k, v in vars(pol).items():
        if isinstance(v, list):
            setattr(new, k, [x.clone() if torch.is_tensor(x) else x for x in v])
        elif torch.is_tensor(v):
            setattr(new, k, v.clone())
    return new


class PerSeq:
    """Run a batch-1 policy once per sequence; capture each keep mask and gather the batch once."""

    def __init__(self, pols):
        self.pols = pols

    def evict_past_key_values(self, kv, hs):
        masks = []
        orig = E._gather_kept
        E._gather_kept = lambda pkv, m: (masks.append(m), pkv)[1]
        try:
            for b, p in enumerate(self.pols):
                p.evict_past_key_values(tuple((k[b:b + 1], v[b:b + 1]) for k, v in kv),
                                        tuple(h[b:b + 1] for h in hs))
        finally:
            E._gather_kept = orig
        if not masks:
            return kv
        assert len(masks) == len(self.pols), "sequences disagreed on whether to compact"
        idx = torch.stack([m.nonzero().squeeze(1) for m in masks])     # (B, kept); equal counts asserted by stack
        g = idx[:, None, :, None]
        return tuple((k.gather(2, g.expand(-1, k.shape[1], -1, k.shape[3])),
                      v.gather(2, g.expand(-1, v.shape[1], -1, v.shape[3]))) for k, v in kv)



def decode(a, model, cache, ids, start, n, Bsz, pol, force_at=None):
    """Run n decode steps feeding ids[start + i]; return one record per step.

    force_at: steps (0-based) at which R-KV is told to compress (its flag is otherwise set by its own
    length counter). Other methods compact on their own schedule.
    """
    need_hs = isinstance(pol, (E.KVSegHSEviction, PerSeq, BatchedEpiKV))
    need_attn = a.method in ("h2o", "raas")
    dev = ids.device
    recs = []
    for i in range(n):
        pos = start + i
        tok = ids[:, pos:pos + 1].expand(Bsz, 1)
        if a.method == "rkv":
            model.config.compression = bool(force_at and i in force_at)
            for layer in model.model.layers:
                layer.self_attn.config.compression = model.config.compression
        L0 = seqlen(cache)
        sync()
        a0 = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=tok, past_key_values=cache, use_cache=True,
                        position_ids=torch.full((Bsz, 1), pos, device=dev),
                        cache_position=torch.tensor([pos], device=dev),
                        output_hidden_states=need_hs, output_attentions=need_attn, logits_to_keep=1)
        sync()
        t1 = time.perf_counter()
        L1 = seqlen(cache)
        if isinstance(pol, BatchedEpiKV):
            with torch.no_grad():
                pol.step(cache, out.hidden_states)      # compacts the cache in place, one layer at a time
        elif pol is not None:
            kv = tuple(kv_pairs(cache))
            with torch.no_grad():
                new = pol.evict_past_key_values(kv, out.attentions if need_attn else out.hidden_states)
            if new[0][0].shape[2] != kv[0][0].shape[2]:
                set_kv(cache, new)
            del kv, new
        del out
        sync()
        t2 = time.perf_counter()
        pk = torch.cuda.max_memory_allocated()
        recs.append({"fwd_ms": (t1 - t0) * 1e3, "pol_ms": (t2 - t1) * 1e3, "len_before": L0,
                     "len_fwd": L1, "len_after": seqlen(cache), "resident": a0, "peak": pk,
                     "reserved": torch.cuda.max_memory_reserved()})
    return recs


def is_compaction(r):
    return r["len_after"] < r["len_before"] + 1


def summarize(recs):
    import statistics as st
    tot = [r["fwd_ms"] + r["pol_ms"] for r in recs]
    comp = [r for r in recs if is_compaction(r)]
    norm = [r for r in recs if not is_compaction(r)]
    s = {"steps": len(recs), "ms_per_token": sum(tot) / len(tot),
         "fwd_ms_median": st.median(r["fwd_ms"] for r in recs),
         "pol_ms_mean": sum(r["pol_ms"] for r in recs) / len(recs),
         "n_compactions": len(comp),
         "cache_len_min": min(r["len_after"] for r in recs), "cache_len_max": max(r["len_fwd"] for r in recs),
         "resident_gb": gb(recs[0]["resident"]), "peak_gb": gb(max(r["peak"] for r in recs)),
         "transient_gb": gb(max(r["peak"] - r["resident"] for r in recs)),
         "reserved_gb": gb(max(r["reserved"] for r in recs))}
    if comp:
        s["compaction"] = [{"ms": r["fwd_ms"] + r["pol_ms"], "fwd_ms": r["fwd_ms"], "pol_ms": r["pol_ms"],
                            "from": r["len_before"] + 1, "to": r["len_after"],
                            "transient_gb": gb(r["peak"] - r["resident"])} for r in comp[:4]]
    if norm:
        s["normal_step_ms_median"] = st.median(r["fwd_ms"] + r["pol_ms"] for r in norm)
        s["normal_pol_ms_mean"] = sum(r["pol_ms"] for r in norm) / len(norm)
    return s



def setup_state(a, model, ids, S0, Bsz, since, hs=None, snap=None, K=None):
    """Cache of length S0 replicated Bsz times and a primed policy (PerSeq for EpiKV at Bsz > 1)."""
    cache = restore(snap, Bsz, ids.device)
    pol = make_policy(a, model, K)
    if pol is not None:
        prime(a, pol, hs, S0, since)
        if a.method == "epikv_batched":
            trim_bank(pol)          # before cloning, so the clones never hold the full bank
            pol = BatchedEpiKV([pol] + [clone_policy(pol) for _ in range(Bsz - 1)])
        elif isinstance(pol, E.KVSegHSEviction) and Bsz > 1:
            pol = PerSeq([pol] + [clone_policy(pol) for _ in range(Bsz - 1)])
    if a.method == "rkv":
        for layer in model.model.layers:
            layer.self_attn.kv_cluster.budget = K if K is not None else 10 ** 9
    return cache, pol


def make_press(a, K):
    from kvpress import DecodingPress, SnapKVPress
    return DecodingPress(base_press=SnapKVPress(), compression_interval=INTERVAL, target_size=K,
                         hidden_states_buffer_size=256)


def run_with_press(a, K, fn):
    if a.method != "snapkv_decoding":
        return fn()
    press = make_press(a, K)
    with press(_MODEL[0]):
        return fn()


_MODEL = [None]


def base_snapshot(a, model, ids, S0):
    pol = make_policy(a, model, 1024)
    chunk = 512 if a.method in ("h2o", "raas") else 4096
    if a.method == "rkv":
        # Prefill goes through R-KV's own prompt compression; an unbounded budget keeps the whole text.
        model.config.compression = None
        for layer in model.model.layers:
            layer.self_attn.kv_cluster.budget = 10 ** 9
    cache, hs = build(model, ids, S0, hs_layers(a, pol), chunk)
    snap = snapshot(cache)
    del cache
    free()
    return snap, hs


def mode_steady(a, model, ids, out):
    for K in a.Ks:
        snap, hs = base_snapshot(a, model, ids, K)
        for Bsz in a.batches:
            key = f"K{K}_B{Bsz}"
            if key in out["cells"]:
                continue
            phase = "setup"
            try:
                if a.restart_windows:
                    # Every window starts from the cache at K with a freshly primed policy, so each one holds
                    # 127 steps at K..K+126 and one compaction from K+128, whatever the method keeps afterwards.
                    recs = []
                    for w in range(a.windows):
                        phase = "setup"
                        cache = pol = None
                        free()
                        cache, pol = setup_state(a, model, ids, K, Bsz, 0, hs, snap, K)
                        phase = "decode"
                        recs += run_with_press(a, K, lambda: decode(a, model, cache, ids, K, INTERVAL, Bsz, pol,
                                                                    force_at={INTERVAL - 1}))
                else:
                    cache, pol = setup_state(a, model, ids, K, Bsz, 0, hs, snap, K)
                    phase = "decode"
                    force = {w * INTERVAL + INTERVAL - 1 for w in range(a.windows)}
                    recs = run_with_press(a, K, lambda: decode(a, model, cache, ids, K, a.windows * INTERVAL, Bsz,
                                                               pol, force_at=force))
                wins = [summarize(recs[w * INTERVAL:(w + 1) * INTERVAL]) for w in range(1, a.windows)]
                out["cells"][key] = {"K": K, "batch": Bsz, "windows": wins}
                print(key, [round(w["ms_per_token"], 2) for w in wins],
                      [w.get("compaction", [{}])[0].get("ms") for w in wins], wins[0]["peak_gb"], flush=True)
            except torch.OutOfMemoryError as e:
                out["cells"][key] = {"K": K, "batch": Bsz, "oom": str(e)[:200], "oom_phase": phase}
                print(key, "OOM", phase, flush=True)
            cache = pol = None
            free()
            dump(a, out)
            if a.first_fit and "oom" not in out["cells"][key]:
                break
        del snap, hs
        free()


def mode_compaction(a, model, ids, out):
    for L in a.Ls:
        S0 = L - (INTERVAL - 1) if a.method == "snapkv_decoding" else L
        try:
            snap, hs = base_snapshot(a, model, ids, S0)
        except torch.OutOfMemoryError as e:
            out["cells"][f"L{L}"] = {"L": L, "oom_build": str(e)[:200]}
            free()
            continue
        for K in a.Ks:
            if K >= L:
                continue
            for Bsz in a.batches:
                key = f"L{L}_K{K}_B{Bsz}"
                if key in out["cells"]:
                    continue
                reps = []
                phase = "setup"
                try:
                    for r in range(a.repeats + 1):
                        phase = "setup"
                        cache, pol = setup_state(a, model, ids, S0, Bsz, INTERVAL - 1, hs, snap, K)
                        phase = "decode"
                        n = INTERVAL if a.method == "snapkv_decoding" else 1
                        recs = run_with_press(a, K, lambda: decode(a, model, cache, ids, S0, n, Bsz, pol,
                                                                   force_at={0}))
                        rec = recs[-1]
                        if a.method != "none":
                            assert is_compaction(rec), f"no compaction at {key}: {rec}"
                        if r:
                            reps.append({"ms": rec["fwd_ms"] + rec["pol_ms"], "fwd_ms": rec["fwd_ms"],
                                         "pol_ms": rec["pol_ms"], "from": rec["len_before"] + 1, "to": rec["len_after"],
                                         "resident_gb": gb(rec["resident"]), "peak_gb": gb(rec["peak"]),
                                         "transient_gb": gb(rec["peak"] - rec["resident"])})
                        cache = pol = None
                        free()
                    out["cells"][key] = {"L": L, "K": K, "batch": Bsz, "repeats": reps}
                    print(key, [round(x["ms"], 1) for x in reps], reps[0]["transient_gb"], flush=True)
                except torch.OutOfMemoryError as e:
                    out["cells"][key] = {"L": L, "K": K, "batch": Bsz, "oom": str(e)[:200], "oom_phase": phase,
                                         "repeats": reps}
                    print(key, "OOM", phase, flush=True)
                cache = pol = None
                free()
                dump(a, out)
        del snap, hs
        free()


def probe(a, model, ids, K, Bsz, snap, hs, S0, n, force):
    cache = pol = None
    try:
        since = INTERVAL - 8 if a.method.startswith("epikv") else 0   # EpiKV compacts on the 8th step
        cache, pol = setup_state(a, model, ids, S0, Bsz, since, hs, snap, None if a.method == "none" else K)
        recs = run_with_press(a, K, lambda: decode(a, model, cache, ids, S0, n, Bsz, pol, force_at=force))
        if a.method != "none":
            assert any(is_compaction(r) for r in recs), "probe saw no compaction"
        return True, gb(max(r["peak"] for r in recs))
    except torch.OutOfMemoryError:
        return False, None
    finally:
        cache = pol = None
        free()


def mode_capacity(a, model, ids, out):
    for K in a.Ks:
        key = f"K{K}"
        if key in out["cells"]:
            continue
        if a.method == "none":
            S0, n, force = K - 8, 8, None
        elif a.method in ("h2o", "raas"):
            S0, n, force = K, 8, None
        elif a.method == "snapkv_decoding":
            S0, n, force = K, INTERVAL + 4, None
        else:   # EpiKV and R-KV: compaction on the 8th step, then 4 more
            S0, n, force = K + INTERVAL - 8, 12, {7}
        snap, hs = base_snapshot(a, model, ids, S0)
        if a.probe_only:
            # One probe per batch size, no search (capacity_search.py drives one process per probe).
            for b in a.batches:
                ok, pk = probe(a, model, ids, K, b, snap, hs, S0, n, force)
                out["cells"][f"K{K}_B{b}"] = {"K_or_context": K, "batch": b, "ok": ok, "peak_gb": pk}
                print(f"K{K}_B{b}", "OK" if ok else "OOM", pk, flush=True)
            dump(a, out)
            continue
        tried = {}
        lo, hi, b = 0, None, 1
        while hi is None and b <= a.max_batch:
            ok, pk = probe(a, model, ids, K, b, snap, hs, S0, n, force)
            tried[b] = pk if ok else "OOM"
            print(key, b, tried[b], flush=True)
            if ok:
                lo, b = b, b * 2
            else:
                hi = b
        if hi is not None:
            while hi - lo > max(1, lo // 32):
                mid = (lo + hi) // 2
                ok, pk = probe(a, model, ids, K, mid, snap, hs, S0, n, force)
                tried[mid] = pk if ok else "OOM"
                print(key, mid, tried[mid], flush=True)
                lo, hi = (mid, hi) if ok else (lo, mid)
        out["cells"][key] = {"K_or_context": K, "max_batch": lo, "first_oom": hi, "probes": tried,
                             "start_len": S0, "steps": n}
        print(key, "MAX", lo, flush=True)
        dump(a, out)
        del snap, hs
        free()


def mode_prefill(a, model, ids, out):
    K = a.Ks[0]
    need_hs = a.method in ("epikv", "epikv_noq")
    need_attn = a.method in ("h2o", "raas")
    for j, L in enumerate([a.Ls[0]] + a.Ls):     # j == 0 is a warmup and is not recorded
        key = f"L{L}"
        reps = []
        try:
            for r in range(a.repeats):
                from transformers import DynamicCache
                cache = DynamicCache()
                if a.method == "rkv":
                    model.config.compression = None
                    for layer in model.model.layers:
                        layer.self_attn.kv_cluster.budget = K
                    if hasattr(model, "length"):
                        del model.length
                if a.method == "dms":
                    cache = None
                press = None
                if a.method == "snapkv_prefill" and L > K:
                    from kvpress import SnapKVPress
                    press = SnapKVPress(compression_ratio=1 - K / L)
                free()
                sync()
                a0 = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                with torch.no_grad():
                    kw = dict(input_ids=ids[:, :L], use_cache=True, logits_to_keep=1)
                    if cache is not None:
                        kw["past_key_values"] = cache
                    if need_hs:
                        kw["output_hidden_states"] = True
                    if need_attn:
                        kw["output_attentions"] = True
                    if press is not None:
                        with press(model):
                            o = model(**kw)
                    else:
                        o = model(**kw)
                sync()
                dt = time.perf_counter() - t0
                pk = torch.cuda.max_memory_allocated()
                kept = None
                pkv = o.past_key_values
                if hasattr(pkv, "layers") and hasattr(pkv.layers[0], "keys") and pkv.layers[0].keys is not None:
                    kept = int(pkv.layers[0].keys.shape[2])
                del o, pkv, cache
                reps.append({"ms": dt * 1e3, "peak_gb": gb(pk), "transient_gb": gb(pk - a0), "cache_len": kept})
            if j:
                out["cells"][key] = {"L": L, "K": K, "repeats": reps}
            print(key, [round(x["ms"], 1) for x in reps], reps[0]["peak_gb"], reps[0]["cache_len"], flush=True)
        except torch.OutOfMemoryError as e:
            out["cells"][key] = {"L": L, "K": K, "oom": str(e)[:200]}
            print(key, "OOM", flush=True)
        free()
        dump(a, out)


def mode_dms_decode(a, model, ids, out):
    """DMS from a P-token prompt: per 128-step window, ms per token and the live KV entries (mean over layers, heads)."""
    dev = ids.device
    for Bsz in a.batches:
        key = f"B{Bsz}"
        if key in out["cells"]:
            continue
        try:
            with torch.no_grad():
                o = model(input_ids=ids[:, :P].expand(Bsz, P), use_cache=True, logits_to_keep=1)
            cache = o.past_key_values
            del o
            wins = []
            for w in range(a.steps // INTERVAL):
                live, ms = [], []
                sync()
                torch.cuda.reset_peak_memory_stats()
                a0 = torch.cuda.memory_allocated()
                for i in range(INTERVAL):
                    pos = P + w * INTERVAL + i
                    sync()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        o = model(input_ids=ids[:, pos:pos + 1].expand(Bsz, 1), past_key_values=cache,
                                  use_cache=True, logits_to_keep=1)
                    sync()
                    ms.append((time.perf_counter() - t0) * 1e3)
                    cache = o.past_key_values
                    del o
                    live.append(torch.stack([l.cache_seq_lengths.float().mean() for l in cache.layers]).mean())
                live = torch.stack(live)
                wins.append({"context_end": P + (w + 1) * INTERVAL, "ms_per_token": sum(ms) / len(ms),
                             "live_mean": float(live.mean()), "live_end": float(live[-1]),
                             "resident_gb": gb(a0), "peak_gb": gb(torch.cuda.max_memory_allocated())})
                if w % 8 == 0:
                    print(key, wins[-1], flush=True)
            out["cells"][key] = {"batch": Bsz, "windows": wins}
        except torch.OutOfMemoryError as e:
            out["cells"][key] = {"batch": Bsz, "oom": str(e)[:200]}
        cache = None
        free()
        dump(a, out)


def mode_dms_capacity(a, model, ids, out):
    """DMS's largest batch at context C (--Ks): prefill C tokens per sequence through its own code, decode 8 steps.

    Doubling, then bisection; the search stops early (bracket recorded) once a probe takes longer than
    --probe_timeout seconds, since DMS's prefill time grows with the batch.
    """
    dev = ids.device

    def probe(C, Bsz):
        o = cache = None
        try:
            free()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.no_grad():
                o = model(input_ids=ids[:, :C].expand(Bsz, C), use_cache=True, logits_to_keep=1)
                cache = o.past_key_values
                o = None
                for i in range(8):
                    o = model(input_ids=ids[:, C + i:C + i + 1].expand(Bsz, 1), past_key_values=cache,
                              use_cache=True, logits_to_keep=1)
                    cache = o.past_key_values
                    o = None
            sync()
            return True, gb(torch.cuda.max_memory_allocated()), time.perf_counter() - t0
        except torch.OutOfMemoryError:
            return False, None, time.perf_counter() - t0
        finally:
            o = cache = None
            free()

    for C in a.Ks:
        key = f"C{C}"
        tried, lo, hi, b, slow = {}, 0, None, 1, False
        while hi is None and b <= a.max_batch and not slow:
            ok, pk, dt = probe(C, b)
            tried[b] = {"ok": ok, "peak_gb": pk, "seconds": round(dt, 1)}
            print(key, b, tried[b], flush=True)
            slow = dt > a.probe_timeout
            lo, hi, b = (b, None, b * 2) if ok else (lo, b, b)
        while hi is not None and hi - lo > max(1, lo // 32) and not slow:
            mid = (lo + hi) // 2
            ok, pk, dt = probe(C, mid)
            tried[mid] = {"ok": ok, "peak_gb": pk, "seconds": round(dt, 1)}
            print(key, mid, tried[mid], flush=True)
            slow = dt > a.probe_timeout
            lo, hi = (mid, hi) if ok else (lo, mid)
        out["cells"][key] = {"context": C, "max_batch": lo, "first_oom": hi, "stopped_on_time": slow,
                             "probes": tried}
        print(key, "MAX", lo, "first_oom", hi, flush=True)
        dump(a, out)


def mode_check_perseq(a, model, ids, out):
    """PerSeq at batch 2 must choose exactly the slots the unmodified batch-1 policy chooses, for every K.

    Compares keep masks and the kept K/V except the newest slot: that token's K/V come from a batch-2 forward,
    whose bf16 kernels may round differently from a batch-1 forward.
    """
    for K in a.Ks:
        snap, hs = base_snapshot(a, model, ids, K + INTERVAL - 1)
        res, masks = [], []
        for Bsz in (1, 2):
            cache, pol = setup_state(a, model, ids, K + INTERVAL - 1, Bsz, INTERVAL - 1, hs, snap, K)
            recs = decode(a, model, cache, ids, K + INTERVAL - 1, 1, Bsz, pol)
            assert is_compaction(recs[0]), recs[0]
            res.append(kv_pairs(cache))
            masks += [p.last_keep_mask for p in (pol.pols if isinstance(pol, PerSeq) else [pol])]
        same_mask = all(torch.equal(masks[0], m) for m in masks[1:])
        same_kv = all(torch.equal(k1[0, :, :-1], k2[b, :, :-1]) and torch.equal(v1[0, :, :-1], v2[b, :, :-1])
                      for (k1, v1), (k2, v2) in zip(res[0], res[1]) for b in range(2))
        out["cells"][f"K{K}"] = {"same_keep_mask": same_mask, "same_kept_kv_but_newest": same_kv,
                                 "kept": int(res[0][0][0].shape[2])}
        print(f"K{K}", "mask", same_mask, "kv", same_kv, res[0][0][0].shape[2], flush=True)
        dump(a, out)


def dump(a, out):
    p = Path(a.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["steady", "compaction", "capacity", "prefill", "dms_decode", "check_perseq",
                             "dms_capacity"])
    ap.add_argument("--method", required=True,
                    choices=["none", "epikv", "epikv_batched", "epikv_noq", "h2o", "raas", "rkv", "snapkv_decoding",
                             "snapkv_prefill", "dms"])
    ap.add_argument("--model", default=LLAMA)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--band_a_layer", type=int, default=None)
    ap.add_argument("--band_b_layer", type=int, default=None)
    ap.add_argument("--Ks", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--Ls", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--windows", type=int, default=4, help="steady: 128-step windows; the first is warmup")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max_batch", type=int, default=1024)
    ap.add_argument("--restart_windows", action="store_true",
                    help="steady: start every window from the cache at K (one compaction per window)")
    ap.add_argument("--first_fit", action="store_true",
                    help="steady: stop at the first batch size (in --batches order) that runs without OOM")
    ap.add_argument("--probe_timeout", type=float, default=600, help="dms_capacity: stop searching after a slower probe")
    ap.add_argument("--probe_only", action="store_true", help="capacity: probe each --batches size, no search")
    ap.add_argument("--steps", type=int, default=8192, help="dms_decode: decode steps")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    import transformers
    try:
        import flash_attn
        fa = flash_attn.__version__
    except ImportError:
        fa = None
    model, tok, attn = load_model(a)
    _MODEL[0] = model
    dev = next(model.parameters()).device
    need = max(a.Ls + a.Ks) + a.windows * INTERVAL + 2 * INTERVAL + a.steps
    ids = text_ids(tok, need, dev)
    out = {"meta": {"mode": a.mode, "method": a.method, "model": a.model, "attn_implementation": attn,
                    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                    "transformers": transformers.__version__, "flash_attn": fa,
                    "prompt_tokens": P, "interval": INTERVAL, "args": vars(a),
                    "weights_gb": gb(torch.cuda.memory_allocated()),
                    "note": "synchronised timing around every step; times in ms; memory from "
                            "torch.cuda.max_memory_allocated (resident = allocated before the step)"},
           "cells": {}}
    if Path(a.output).exists():
        prev = json.loads(Path(a.output).read_text())
        if prev.get("meta", {}).get("args") == vars(a):
            out["cells"] = prev["cells"]
    dump(a, out)
    {"steady": mode_steady, "compaction": mode_compaction, "capacity": mode_capacity,
     "prefill": mode_prefill, "dms_decode": mode_dms_decode, "check_perseq": mode_check_perseq,
     "dms_capacity": mode_dms_capacity}[a.mode](a, model, ids, out)
    out["meta"]["finished"] = True
    dump(a, out)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
