"""Derive decode energy per correct answer: measured J/token (energy_* records) x measured tokens per correct answer.

Tokens per correct answer come from rebuttals/scripts/baseline_decode_work_per_correct.py (run here; its markdown is
parsed): MATH-500 n=495, greedy, cap 8192 for Llama-8B (Table 2 records, and official R-KV in the transformers 4.55.4
group), cap 16384 for Qwen3-8B (none, EpiKV, DMS-8x). J/token is the median over the three measured 128-step windows
of the cell, from the NVML energy counter; net subtracts the card's idle power. Pairing:
  batch 1  -> energy_b1_<method> at the same K (decode at batch 1 is bound by reading the weights, so J/token moves
              about 2% between K=1024 and 4096; no eviction is measured at a cache of K tokens);
  max batch -> energy_max_<method>_K<K> at K in {2048, 4096}; no eviction pairs with its 8k-context max batch
              (the Llama traces are capped at 8192 tokens).
DMS: its J/token is the token-weighted mean of its decode from a 128-token prompt to 8320 tokens of context.
kvpress has no accuracy record in that script, so it has no row. Output: records/energy_per_correct.json.
Usage: energy_per_correct.py <records folder>
"""
import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

SCRIPT = Path.home() / "workspace/emnlp-26/rebuttals/scripts/baseline_decode_work_per_correct.py"
SECTIONS = {"Llama-8B MATH-500 n=495, A100, Table 2 records (cap 8192)": "llama",
            "Llama-8B MATH-500, official R-KV vs EpiKV (transformers 4.55.4, A100) (cap 8192)": "llama_rkv",
            "Qwen3-8B MATH with DMS-8x at K=1024 (cap 16384)": "qwen3",
            "Qwen3-8B MATH with DMS-8x at K=2048 (cap 16384)": "qwen3",
            "Qwen3-8B MATH with DMS-8x at K=4096 (cap 16384)": "qwen3"}
NAMES = {("llama", "none"): "none", ("llama", "EpiKV"): "epikv_batched", ("llama", "EpiKV^-Q"): "epikv_noq",
         ("llama", "RaaS"): "raas", ("llama", "H2O"): "h2o", ("llama_rkv", "R-KV official"): "rkv",
         ("qwen3", "none"): "none_qwen3", ("qwen3", "EpiKV"): "epikv_qwen3", ("qwen3", "DMS-8x"): "dms"}


def tokens_per_correct():
    md = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout
    out, group = {}, None
    for line in md.splitlines():
        if line.startswith("### "):
            group = SECTIONS.get(line[4:].strip())
            continue
        m = re.match(r"\| (\d+) \| ([^|]+) \| 495 \| [\d.]+ \| \d+ \| (\d+) \|", line)
        if group and m:
            name = NAMES.get((group, m.group(2).strip()))
            if name:
                out[(name, int(m.group(1)))] = int(m.group(3))
    return out


def jpt(folder, name, key):
    p = Path(folder) / f"{name}.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())["cells"].get(key)
    if not c or "oom" in c:
        return None
    e = [w["energy"] for w in c["windows"]]
    return st.median(x["j_per_token"] for x in e), st.median(x["net_j_per_token"] for x in e)


def dms_jpt(folder, batch):
    p = Path(folder) / "energy_dms_decode.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())["cells"].get(f"B{batch}")
    if not c or "oom" in c:
        return None
    e = [w["energy"] for w in c["windows"][1:]]
    n = sum(x["tokens"] for x in e)
    return sum(x["energy_j"] for x in e) / n, sum(x["net_j_per_token"] * x["tokens"] for x in e) / n


def main():
    folder = sys.argv[1]
    tpc = tokens_per_correct()
    rows = []
    for (name, K), t in sorted(tpc.items()):
        cells = []
        if name == "dms":       # one checkpoint, no budget: the same tokens per correct answer at every K
            if K != 1024:
                continue
            cells = [("batch 1", "energy_dms_decode B1", dms_jpt(folder, 1)),
                     ("batch 8", "energy_dms_decode B8", dms_jpt(folder, 8))]
        elif K in (1024, 4096):
            cells.append(("batch 1", f"energy_b1_{name} K{K}_B1", jpt(folder, f"energy_b1_{name}", f"K{K}_B1")))
        if K in (2048, 4096) and name in ("epikv_batched", "epikv_noq", "rkv", "h2o", "raas", "none"):
            if name == "none":
                cells.append(("max batch", "energy_max_none_K8192", jpt(folder, "energy_max_none_K8192", "K8192_B23")))
            else:
                f = Path(folder) / f"energy_max_{name}_K{K}.json"
                key = next(iter(json.loads(f.read_text())["cells"])) if f.exists() else None
                cells.append(("max batch", f"energy_max_{name}_K{K} {key}", jpt(folder, f"energy_max_{name}_K{K}", key)))
        for regime, cell, v in cells:
            if v is None:
                continue
            group = "Qwen3-8B MATH cap 16384" if name in ("none_qwen3", "epikv_qwen3", "dms") else "Llama-8B MATH cap 8192"
            rows.append({"group": group, "regime": regime, "method": name, "K": "-" if name == "dms" else K, "energy_cell": cell,
                         "j_per_token": v[0], "net_j_per_token": v[1], "tokens_per_correct": t,
                         "j_per_correct": v[0] * t, "net_j_per_correct": v[1] * t})
    Path(folder, "energy_per_correct.json").write_text(json.dumps(
        {"meta": {"note": "derived: measured J/token x measured tokens per correct answer", "source": str(SCRIPT)},
         "rows": rows}, indent=1))
    for r in rows:
        print(r["group"], r["method"], r["K"], r["regime"], round(r["j_per_token"], 3), r["tokens_per_correct"],
              round(r["j_per_correct"]))


if __name__ == "__main__":
    main()
