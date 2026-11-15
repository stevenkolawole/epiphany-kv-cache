"""Check that BatchedEpiKV keeps exactly the slots the per-sequence KVSegHSEviction keeps, on real reasoning traces.

Each batch row is a different stream of real DeepSeek-R1-Distill-Llama-8B reasoning traces (data_slim60.jsonl:
prompt + generated token ids; row b concatenates traces b, b+1, ... until it is long enough). One batched cache is
decoded teacher-forced; at every step the reference (one unmodified KVSegHSEviction per row, keep masks captured,
no gather) and BatchedEpiKV (its own primed copies) see the same cache and the same hidden states; the reference
runs first, then BatchedEpiKV compacts the cache in place. Asserted per step: every row's newest score is the same
float. Per compaction: per row, torch.equal of the kept index tensors, equal score lists, bank length equal to the
number of cached decode tokens, and the gathered K/V of the first and last layer equal to an index_select with the
reference indices. Writes one JSON record with pass/fail per (B, K, compaction).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_costs as C  # noqa: E402
from epikv_batched import BatchedEpiKV, trim_bank  # noqa: E402

E = C.E


def streams(path, B, T, device):
    rows = [json.loads(line)["token_ids"] for line in open(path)]
    out = []
    for b in range(B):
        ids, j = [], b
        while len(ids) < T:
            ids += rows[j % len(rows)]
            j += 1
        out.append(ids[:T])
    return torch.tensor(out, device=device)


def run(a, model, ids, K, n_comp):
    from transformers import DynamicCache
    Bsz = ids.shape[0]
    S0 = K
    ns = argparse.Namespace(method="epikv", band_a_layer=None, band_b_layer=None)
    caches, refs = [], []
    for b in range(Bsz):
        pol = C.make_policy(ns, model, K)
        cache_b, hs = C.build(model, ids[b:b + 1], S0, C.hs_layers(ns, pol), 4096)
        caches.append(C.kv_pairs(cache_b))
        refs.append(C.prime(ns, pol, hs, S0, 0))
        del cache_b, hs
    cache = DynamicCache()
    for i in range(len(caches[0])):
        cache.update(torch.cat([c[i][0] for c in caches]), torch.cat([c[i][1] for c in caches]), i)
    del caches
    C.free()
    bats = [C.clone_policy(p) for p in refs]
    for p in bats:
        trim_bank(p)
    bat = BatchedEpiKV(bats)
    dev = ids.device
    recs, score_mismatch = [], 0
    for i in range(n_comp * C.INTERVAL):
        pos = S0 + i
        with torch.no_grad():
            out = model(input_ids=ids[:, pos:pos + 1], past_key_values=cache, use_cache=True,
                        position_ids=torch.full((Bsz, 1), pos, device=dev),
                        cache_position=torch.tensor([pos], device=dev), output_hidden_states=True,
                        logits_to_keep=1)
        hs = out.hidden_states
        masks = []
        orig = E._gather_kept
        E._gather_kept = lambda pkv, m: (masks.append(m), pkv)[1]
        try:
            kv = C.kv_pairs(cache)
            for b, p in enumerate(refs):
                p.evict_past_key_values(tuple((k[b:b + 1], v[b:b + 1]) for k, v in kv), tuple(h[b:b + 1] for h in hs))
            del kv
        finally:
            E._gather_kept = orig
        expect = None
        if masks:
            ridx = [m.nonzero().squeeze(1) for m in masks]
            expect = {li: [torch.cat([cache.layers[li].keys[b:b + 1].index_select(2, ridx[b]) for b in range(Bsz)]),
                           torch.cat([cache.layers[li].values[b:b + 1].index_select(2, ridx[b]) for b in range(Bsz)])]
                      for li in (0, len(cache.layers) - 1)}
        with torch.no_grad():
            did = bat.step(cache, hs)
        del out, hs
        score_mismatch += sum(r._scores[-1] != q._scores[-1] for r, q in zip(refs, bat.pols))
        if masks or did:
            bidx = [m.nonzero().squeeze(1) for m in bat.last_masks] if did else []
            same_idx = bool(masks) and did and all(torch.equal(x, y) for x, y in zip(ridx, bidx))
            same_scores = all(r._scores == q._scores for r, q in zip(refs, bat.pols))
            num_decode = C.seqlen(cache) - C.P
            bank_ok = all(len(q._hs_bank) == num_decode == len(r._hs_bank) for r, q in zip(refs, bat.pols))
            kv_ok = all(torch.equal(cache.layers[li].keys, ek) and torch.equal(cache.layers[li].values, ev)
                        for li, (ek, ev) in expect.items()) if expect else False
            rec = {"step": i, "kept": int(ridx[0].numel()) if masks else None, "same_indices": same_idx,
                   "same_scores": same_scores, "bank_len_ok": bank_ok, "same_gathered_kv": kv_ok,
                   "distinct_rows": Bsz == 1 or not torch.equal(ridx[0], ridx[-1])}
            recs.append(rec)
            print(K, Bsz, rec, flush=True)
        del expect, masks
    return {"K": K, "batch": Bsz, "compactions": recs, "steps": n_comp * C.INTERVAL,
            "steps_with_score_mismatch": int(score_mismatch),
            "pass": len(recs) == n_comp and score_mismatch == 0 and all(
                r["same_indices"] and r["same_scores"] and r["bank_len_ok"] and r["same_gathered_kv"] for r in recs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--compactions", type=int, default=5)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    ns = argparse.Namespace(method="epikv", model=C.LLAMA, tokenizer=None, band_a_layer=None, band_b_layer=None)
    model, tok, _ = C.load_model(ns)
    dev = next(model.parameters()).device
    out = {"meta": {"traces": a.traces, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
           "cells": []}
    for K in a.Ks:
        for Bsz in a.batches:
            ids = streams(a.traces, Bsz, K + a.compactions * C.INTERVAL + 1, dev)
            out["cells"].append(run(a, model, ids, K, a.compactions))
            print("CELL", K, Bsz, "PASS" if out["cells"][-1]["pass"] else "FAIL", flush=True)
            Path(a.output).write_text(json.dumps(out, indent=1))
            C.free()
    out["all_pass"] = all(c["pass"] for c in out["cells"])
    Path(a.output).write_text(json.dumps(out, indent=1))
    print("ALL", "PASS" if out["all_pass"] else "FAIL")


if __name__ == "__main__":
    main()
