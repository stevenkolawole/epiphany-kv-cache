"""Analytical FLOPs, bytes moved and transient memory per compaction (or per evicting step) for each KV-eviction method.

Counts follow the code that the measurements in baseline_costs.py run (src/eviction.py, third_party/rkv,
kvpress 0.5.5 SnapKVPress inside DecodingPress), for one model's shapes. Element sizes: bf16 2 B, fp32 4 B,
int64 8 B, bool 1 B. Bytes are what each step writes plus reads from GPU memory, ignoring caches; FLOPs count
multiply-adds as 2. These are analytical estimates, not measurements, and they explain rather than replace them.
Output: one JSON with a record per (method, L, K, B) and a printed summary.
"""
import argparse
import json
from pathlib import Path

LLAMA = dict(layers=32, q_heads=32, kv_heads=8, head_dim=128, hidden=4096, weights_gb=16.06)


def kv_bytes_per_token(m):
    return m["layers"] * m["kv_heads"] * m["head_dim"] * 2 * 2


def epikv(m, L, K, B, qk=True, w=8):
    nl, kvh, d, h, qh = m["layers"], m["kv_heads"], m["head_dim"], m["hidden"], m["q_heads"]
    flops = 3 * nl * kvh * L * d * B                              # key variance over head_dim, every layer
    byts = nl * kvh * L * d * (2 + 4 + 4) * B                     # read bf16 keys, write and read the fp32 copy
    trans = kvh * L * d * 4 * B                                   # one layer's fp32 key copy at a time
    if qk:
        flops += (2 * w * h * h + 2 * qh * w * L * d) * B         # q_proj of 8 banked states; 8 queries x L keys
        byts += (qh * L * d * 2 * 2 + qh * w * L * 4 * 3) * B     # repeat_interleave keys to 32 heads; fp32 logits
        trans = max(trans, (qh * L * d * 2 + qh * w * L * 4 * 2) * B)
    keep = K * kv_bytes_per_token(m) * B
    byts += 2 * keep                                              # gather the kept cache (read + write)
    trans += keep                                                 # the kept copy coexists with the old cache
    return dict(flops=flops, bytes=byts, transient=trans, per="compaction (every 128 steps)")


def rkv(m, L, K, B, w=8):
    nl, kvh, d, qh = m["layers"], m["kv_heads"], m["head_dim"], m["q_heads"]
    g = qh // kvh
    per_layer_flops = 2 * qh * w * L * d + 2 * kvh * L * L * d + 6 * kvh * L * L
    # similarity: bf16 matmul out (w2), eye mask (L^2 bool), threshold mask (w1), zeros_like int64 (w8), where
    # int64 (r1 + r8 + w8), max over it (r8), scatter, mean (r2), plus softmax of the window attention (fp32)
    per_layer_bytes = (kvh * L * L * (2 + 1 + 1 + 8 + 1 + 8 + 8 + 8 + 2) + L * L
                       + qh * w * L * 4 * 2 + kvh * g * w * L * 4)
    per_layer_trans = kvh * L * L * (2 + 1 + 8 + 8) + L * L      # sim, mask, zeros_like and where alive together
    keep = K * kv_bytes_per_token(m) * B
    return dict(flops=nl * per_layer_flops * B, bytes=nl * per_layer_bytes * B + 2 * keep,
                transient=per_layer_trans * B + keep // nl, per="compaction (every 128 steps)")


def h2o_like(m, L, K, B):
    nl, qh = m["layers"], m["q_heads"]
    attn_out = nl * qh * L * 2 * B                                # weights returned for every layer (bf16)
    softmax_tmp = qh * L * 4 * B                                  # fp32 softmax inside eager attention, per layer
    keep = K * kv_bytes_per_token(m) * B
    return dict(flops=nl * qh * L * 5 * B, bytes=attn_out * 2 + nl * softmax_tmp * 2 + 2 * keep,
                transient=attn_out + keep, per="every decode step")


def snapkv_decoding(m, L, K, B, w=64, buf=128):
    nl, kvh, d, h, qh = m["layers"], m["kv_heads"], m["head_dim"], m["hidden"], m["q_heads"]
    per_layer_flops = 2 * w * h * h + 2 * qh * w * L * d
    per_layer_bytes = qh * L * d * 2 * 2 + qh * w * L * (2 + 2 + 4 + 2) + buf * h * 2
    per_layer_trans = qh * L * d * 2 + qh * w * L * (2 + 2 + 4)
    keep = K * kv_bytes_per_token(m) * B
    return dict(flops=nl * per_layer_flops * B, bytes=nl * per_layer_bytes * B + 2 * keep,
                transient=per_layer_trans * B + keep // nl, per="compaction (every 128 steps)",
                resident_extra=nl * 256 * h * 2 * B)


def decode_step(m, L, B):
    """Bytes one decode step must read: the weights once and the live cache."""
    return dict(bytes=m["weights_gb"] * 2 ** 30 + L * kv_bytes_per_token(m) * B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    m = LLAMA
    recs = []
    for B in (1, 4, 16):
        for K in (512, 1024, 2048, 4096):
            for L in sorted({K + 128, 4096, 8192, 16384, 32768}):
                if L < K + 1:
                    continue
                for name, fn in (("epikv", lambda: epikv(m, L, K, B)), ("epikv_noq", lambda: epikv(m, L, K, B, False)),
                                 ("rkv", lambda: rkv(m, L, K, B)), ("h2o_raas", lambda: h2o_like(m, L, K, B)),
                                 ("snapkv_decoding", lambda: snapkv_decoding(m, L, K, B))):
                    r = fn()
                    recs.append(dict(method=name, L=L, K=K, B=B, gflop=round(r["flops"] / 1e9, 2),
                                     gb_moved=round(r["bytes"] / 2 ** 30, 3),
                                     transient_gb=round(r["transient"] / 2 ** 30, 3), per=r["per"],
                                     resident_extra_gb=round(r.get("resident_extra", 0) / 2 ** 30, 3)))
                ds = decode_step(m, L, B)
                recs.append(dict(method="decode_step_floor", L=L, K=K, B=B, gb_moved=round(ds["bytes"] / 2 ** 30, 3)))
    Path(a.output).write_text(json.dumps({"model_shape": m, "note": "analytical", "records": recs}, indent=1))
    for r in recs:
        if r["B"] == 1 and r["L"] == r["K"] + 128 and r["method"] != "decode_step_floor":
            print(r["method"], r["K"], r["gflop"], "GFLOP", r["gb_moved"], "GB moved", r["transient_gb"], "GB transient")


if __name__ == "__main__":
    main()
