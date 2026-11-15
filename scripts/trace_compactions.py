#!/usr/bin/env python3
"""Run scripts/benchmark.py with its arguments and print one COMPACT line per KV compaction.

For EpiKV (KVSegHSEviction) and the window rules (WindowRuleEviction) a line gives the decode step, the cache
length before and after, and whether every prompt slot was kept; for kvpress DecodingPress it gives the layer-0
length before and after and the buffered window. Used by the smoke runs to show that compaction fires, keeps K and
peaks at K + 128; the policies and the harness are not changed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import eviction as E  # noqa: E402
import benchmark  # noqa: E402


def _trace(cls):
    orig = cls.evict_past_key_values

    def evict(self, past_key_values, *rest):
        n = past_key_values[0][0].shape[2]
        out = orig(self, past_key_values, *rest)
        if out[0][0].shape[2] != n:
            p = min(self._prefill_len, n)
            print(f"COMPACT {cls.__name__} step {self._step} {n} -> {out[0][0].shape[2]} "
                  f"prompt {p} kept {bool(self.last_keep_mask[:p].all())}", flush=True)
        return out

    cls.evict_past_key_values = evict


_trace(E.KVSegHSEviction)
_trace(E.WindowRuleEviction)

if "kvpress_snapkv_decoding" in sys.argv:
    from kvpress.presses.decoding_press import DecodingPress
    _orig = DecodingPress.compress

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        out = _orig(self, module, hidden_states, keys, values, attentions, kwargs)
        if module.layer_idx == 0 and out[0].shape[2] != keys.shape[2]:
            print(f"COMPACT kvpress {keys.shape[2]} -> {out[0].shape[2]} window {hidden_states.shape[1]}", flush=True)
        return out

    DecodingPress.compress = compress

benchmark.main()
