#!/usr/bin/env python3
"""Per generated token: what the DMS checkpoint decides to evict, and what a training-free policy keeps on the same tokens.

    dms_keep_agreement.py record <trace_dir> <benchmark.py args with --methods dms>
    dms_keep_agreement.py forced <trace_dir> <benchmark.py args with the policy and --cache_sizes>

record runs benchmark.py unchanged on the DMS checkpoint and saves, per problem, <dataset>_<crc32 of problem>.npz:
  ids      generated token ids
  dec      (layers, kv heads, decode steps) bool, True = evict. Column j is produced by the query of generated token j and
           decides the token before it (modeling_qwen3_dms.py: "token t+1 predicts whether to evict token t"), so the
           decision on generated token i is dec[:, :, i + 1].
  pre      the same for the prompt pass; fin = live entries per (layer, head) at the end; prompt_len.
forced replays those ids through the base model (benchmark._pick returns the recorded token instead of the argmax), with
the policy evicting as usual, and saves <dataset>_<crc>__<method>_k<K>.npy: bool per generated token, True = still in the
cache at the end. Kept positions are followed through eviction._gather_kept, the one place every policy drops tokens.
Any inconsistency exits non-zero (benchmark.py would otherwise score the problem wrong and carry on).
"""
import os, sys, traceback, zlib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
mode, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
sys.argv = ["benchmark.py"] + sys.argv[3:]
dataset = sys.argv[sys.argv.index("--dataset") + 1]
import benchmark as B
import eviction as E


def stem(prob):
    return f"{dataset}_{zlib.crc32(prob['problem'].encode()):08x}"


def loud(fn):
    def g(*a, **k):
        try:
            return fn(*a, **k)
        except BaseException:
            traceback.print_exc(); sys.stdout.flush(); os._exit(1)
    return g


if mode == "record":
    from third_party.dms import dms_cache
    layers, step, pre, picks = [], {}, {}, []
    upd = dms_cache.DMSPagedCacheLayer.update

    def update(self, *a, **k):
        ei = (k.get("cache_kwargs") or a[2])["eviction_info"]
        if id(self) not in step:
            layers.append(self); step[id(self)] = []; pre[id(self)] = []
        (step if ei.shape[2] == 1 else pre)[id(self)].append(ei.reshape(-1, ei.shape[2]).bool().clone())
        return upd(self, *a, **k)
    dms_cache.DMSPagedCacheLayer.update = update
    pick = B._pick

    def _pick(logits, gen):
        t = pick(logits, gen); picks.append(t); return t
    B._pick = _pick
    run = B.run_one_dms

    @loud
    def run_one_dms(model, tok, prob, mnt, dev):
        layers.clear(); step.clear(); pre.clear(); picks.clear()
        res = run(model, tok, prob, mnt, dev)
        n = res["n_tokens_generated"]
        ids = torch.cat(picks[:n]).flatten().cpu().numpy()
        cat = lambda d: torch.stack([torch.cat(d[id(l)], 1) for l in layers]).cpu().numpy()
        dec = cat(step) if step[id(layers[0])] else np.zeros((len(layers), layers[0].page_batch, 0), bool)
        assert len(ids) == n and dec.shape[2] in (n, n - 1), (len(ids), n, dec.shape)
        np.savez_compressed(out / (stem(prob) + ".npz"), ids=ids, dec=dec, pre=cat(pre),
                            fin=torch.stack([l.cache_seq_lengths for l in layers]).cpu().numpy(),
                            prompt_len=res["kv_live"]["total_tokens"] - n, correct=bool(res["correct"]))
        return res
    B.run_one_dms = run_one_dms
else:
    st, size = {}, {}
    mk = B.make_eviction

    def make_eviction(method, cache_size, *a, **k):
        ev = mk(method, cache_size, *a, **k); size[id(ev)] = cache_size; return ev
    B.make_eviction = make_eviction
    gather = E._gather_kept

    @loud
    def _gather_kept(pkv, keep_mask):
        add = keep_mask.numel() - st["alive"].numel()
        assert add >= 0, add
        a = torch.cat([st["alive"], torch.arange(st["next"], st["next"] + add)]); st["next"] += add
        st["alive"] = a[keep_mask.cpu()]
        return gather(pkv, keep_mask)
    E._gather_kept = _gather_kept

    def _pick(logits, gen):
        return torch.tensor([[st["q"].pop() if st["q"] else st["eos"]]], device=logits.device)
    B._pick = _pick
    run = B.run_one

    @loud
    def run_one(model, tok, prob, method, eviction, mnt, *a, **k):
        z = np.load(out / (stem(prob) + ".npz")); ids = z["ids"].tolist(); K = size[id(eviction)]
        st.update(q=ids[::-1], alive=torch.empty(0, dtype=torch.long), next=0, eos=tok.eos_token_id)
        res = run(model, tok, prob, method, eviction, len(ids), *a, **k)
        total = int(z["prompt_len"]) + len(ids) - (ids[-1] == tok.eos_token_id)   # an ending eos is never fed
        assert res["n_tokens_generated"] == len(ids) and not st["q"], (res["n_tokens_generated"], len(ids))
        assert st["next"] <= total, (st["next"], total)
        alive = torch.cat([st["alive"], torch.arange(st["next"], total)]) - int(z["prompt_len"])
        assert total <= K + 129 or alive.numel() <= K + 129, ("policy bypassed _gather_kept?", alive.numel(), K)
        kept = np.zeros(len(ids), bool); kept[alive[alive >= 0].numpy()] = True
        np.save(out / f"{stem(prob)}__{method}_k{K}.npy", kept)
        print(f"  kept {int(kept.sum())}/{len(ids)} generated tokens, cache {alive.numel()} at K={K}", flush=True)
        return res
    B.run_one = run_one

B.main()
