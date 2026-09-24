#!/usr/bin/env python3
"""Print the attribute path to vLLM's GPU model runner on this install.

find_model_runner in modal_infra/vllm_poc.py was written against an earlier
0.11 layout and its BFS gives up before reaching the runner here. Rather than
loosen the search blindly, this walks the engine graph to depth 14 with no
type filter and prints every object that carries `input_batch`, `kv_caches`,
or `_prepare_inputs`, with its path and type. The winning path gets hardcoded
in vllm_seg_validate.py with a comment saying which vLLM it was found on.
"""
import os
import sys
from collections import deque

os.environ.setdefault("USE_TF", "0")
import torch  # noqa: E402
from vllm import LLM  # noqa: E402

llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", enforce_eager=True,
          gpu_memory_utilization=0.45, max_model_len=2048, dtype="bfloat16",
          max_num_seqs=1)

root = llm.llm_engine
print("engine type:", type(root).__name__, flush=True)
ec = getattr(root, "engine_core", None)
print("engine_core type:", type(ec).__name__, flush=True)
if ec is not None:
    print("engine_core attrs:", [a for a in vars(ec) if not a.startswith("__")][:40], flush=True)

seen, q, hits = set(), deque([(root, "llm_engine", 0)]), []
while q:
    obj, path, d = q.popleft()
    if id(obj) in seen or d > 14:
        continue
    seen.add(id(obj))
    marks = [a for a in ("input_batch", "kv_caches", "_prepare_inputs", "execute_model",
                         "scheduler", "kv_cache_manager", "block_pool", "requests")
             if hasattr(obj, a)]
    if marks:
        hits.append((path, type(obj).__name__, marks))
    if not hasattr(obj, "__dict__"):
        continue
    for name, child in list(vars(obj).items()):
        if name.startswith("__") or child is None:
            continue
        if isinstance(child, (int, float, str, bool, bytes, torch.Tensor, torch.nn.Module)):
            continue
        if isinstance(child, (list, tuple)):
            for i, c in enumerate(child[:8]):
                if hasattr(c, "__dict__"):
                    q.append((c, f"{path}.{name}[{i}]", d + 1))
            continue
        if isinstance(child, dict):
            for k, c in list(child.items())[:8]:
                if hasattr(c, "__dict__"):
                    q.append((c, f"{path}.{name}[{k!r}]", d + 1))
            continue
        q.append((child, f"{path}.{name}", d + 1))

for path, tname, marks in hits:
    print(f"HIT {path}  ({tname})  {marks}", flush=True)
print("DONE", flush=True)
