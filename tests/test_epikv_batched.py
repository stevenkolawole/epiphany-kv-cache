"""BatchedEpiKV keeps exactly what one KVSegHSEviction per sequence keeps (tiny random Llama, CPU)."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import eviction as E  # noqa: E402
from epikv_batched import BatchedEpiKV  # noqa: E402

transformers = pytest.importorskip("transformers")


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=24, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, vocab_size=512, max_position_embeddings=4096,
                      attn_implementation="eager")
    return LlamaForCausalLM(cfg).eval()


def _policy(model, K, query):
    cfg = E.EvictionConfig(cache_size=K, keep_recent_k=128)
    kw = dict(fill_budget=True, refresh_tau=128, tau_mode="steps")
    if query:
        kw.update(query_sim_weight=1.0, query_sim_mode="replace", query_mode="qk", query_sim_window=8)
    p = E.KVSegHSEviction(cfg, **kw)
    p.set_model(model)
    return p


def _run(B, K, query, steps=700, prompt=32):
    from transformers import DynamicCache
    model = _tiny_llama()
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 512, (B, prompt + steps), generator=g)
    cache = DynamicCache()
    with torch.no_grad():
        out = model(input_ids=ids[:, :prompt], past_key_values=cache, use_cache=True, output_hidden_states=True)
    refs = [_policy(model, K, query) for _ in range(B)]
    bats = [_policy(model, K, query) for _ in range(B)]
    for b in range(B):
        for p in (refs[b], bats[b]):
            p.reset(prefill_len=prompt)
            p.set_prefill_end(tuple(h[b:b + 1] for h in out.hidden_states))
    bat = BatchedEpiKV(bats)
    n_comp = 0
    for i in range(steps):
        pos = prompt + i
        with torch.no_grad():
            out = model(input_ids=ids[:, pos:pos + 1], past_key_values=cache, use_cache=True,
                        position_ids=torch.full((B, 1), pos), cache_position=torch.tensor([pos]),
                        output_hidden_states=True)
        masks = []
        orig = E._gather_kept
        E._gather_kept = lambda pkv, m: (masks.append(m), pkv)[1]
        try:
            kv = [(layer.keys, layer.values) for layer in cache.layers]
            for b, p in enumerate(refs):
                p.evict_past_key_values(tuple((k[b:b + 1], v[b:b + 1]) for k, v in kv),
                                        tuple(h[b:b + 1] for h in out.hidden_states))
            del kv
        finally:
            E._gather_kept = orig
        with torch.no_grad():
            did = bat.step(cache, out.hidden_states)
        assert did == bool(masks)
        for r, q in zip(refs, bat.pols):
            assert r._scores == q._scores
            assert len(r._hs_bank) == len(q._hs_bank)
        if did:
            n_comp += 1
            for m, bm in zip(masks, bat.last_masks):
                assert torch.equal(m.nonzero(), bm.nonzero())
            assert cache.layers[0].keys.shape[2] == K
    return n_comp


@pytest.mark.parametrize("B", [1, 3])
def test_batched_keeps_the_same_slots_with_the_query_read(B):
    assert _run(B, K=256, query=True) >= 3


def test_batched_keeps_the_same_slots_without_the_query_read():
    assert _run(2, K=256, query=False) >= 3


def test_batched_rejects_the_hs_query_mode():
    model = _tiny_llama()
    p = _policy(model, 256, True)
    p.query_mode = "hs"
    with pytest.raises(ValueError, match="query_mode"):
        BatchedEpiKV([p])
