"""WindowRuleEviction reads EpiKV's window and keeps EpiKV's frame (tiny random Llama, CPU)."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import eviction as E  # noqa: E402

transformers = pytest.importorskip("transformers")


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=24, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, vocab_size=512, max_position_embeddings=4096,
                      attn_implementation="eager")
    return LlamaForCausalLM(cfg).eval()


def _decode(policy, model, K, steps=700, prompt=32):
    """Decode random tokens under policy; return (length before, length after) of every step."""
    from transformers import DynamicCache
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 512, (1, prompt + steps), generator=g)
    cache = DynamicCache()
    with torch.no_grad():
        out = model(input_ids=ids[:, :prompt], past_key_values=cache, use_cache=True, output_hidden_states=True)
    policy.reset(prefill_len=prompt)
    policy.set_prefill_end(out.hidden_states)
    lens, masks = [], []
    for i in range(steps):
        pos = prompt + i
        with torch.no_grad():
            out = model(input_ids=ids[:, pos:pos + 1], past_key_values=cache, use_cache=True,
                        position_ids=torch.tensor([[pos]]), cache_position=torch.tensor([pos]),
                        output_hidden_states=True)
        kv = tuple((layer.keys, layer.values) for layer in cache.layers)
        policy.last_keep_mask = None
        new = policy.evict_past_key_values(kv, out.hidden_states)
        lens.append((kv[0][0].shape[2], new[0][0].shape[2]))
        masks.append(policy.last_keep_mask)
        for layer, (k, v) in zip(cache.layers, new):
            layer.keys, layer.values = k, v
    return lens, masks


def test_window_rows_average_to_epikv_relevance():
    model = _tiny_llama()
    p = E.WindowRuleEviction(E.EvictionConfig(cache_size=256), "raas")
    p.set_model(model)
    _decode(p, model, K=256, steps=200)
    from transformers import DynamicCache
    ids = torch.randint(0, 512, (1, 300))
    cache = DynamicCache()
    with torch.no_grad():
        model(input_ids=ids, past_key_values=cache, use_cache=True)
    kv = tuple((layer.keys, layer.values) for layer in cache.layers)
    rel = E.KVSegHSEviction._qk_relevance(p, kv, 300, torch.device("cpu"))
    probs = p._window_probs(kv, torch.device("cpu"))
    assert probs.shape[2] == 8
    assert torch.equal(probs.mean(dim=(0, 1, 2)), rel)


@pytest.mark.parametrize("rule", ["raas", "r_kv"])
def test_compaction_keeps_epikvs_frame(rule):
    K, prompt = 256, 32
    model = _tiny_llama()
    p = E.WindowRuleEviction(E.EvictionConfig(cache_size=K, keep_recent_k=128), rule)
    p.set_model(model)
    lens, masks = _decode(p, model, K=K, prompt=prompt)
    fired = [i for i, (a, b) in enumerate(lens) if a != b]
    assert len(fired) >= 3
    assert max(a for a, _ in lens) <= K + 128
    last = -1
    for i in fired:
        a, b = lens[i]
        assert b == K and a > K
        assert i - last >= 128
        last = i
        m = masks[i]
        assert m[:prompt].all() and m[-min(128, K // 4):].all()
    assert len(p._stamps) == lens[-1][1] - prompt


def test_compaction_steps_match_epikv():
    """The window rules compact on exactly the steps EpiKV does."""
    K = 256
    model = _tiny_llama()
    ep = E.KVSegHSEviction(E.EvictionConfig(cache_size=K), fill_budget=True, refresh_tau=128, tau_mode="steps",
                           query_sim_weight=1.0, query_sim_mode="replace", query_mode="qk", query_sim_window=8)
    ep.set_model(model)
    wr = E.WindowRuleEviction(E.EvictionConfig(cache_size=K), "r_kv")
    wr.set_model(model)
    a, _ = _decode(ep, model, K=K)
    b, _ = _decode(wr, model, K=K)
    assert a == b


def test_raas_rule_keeps_the_newest_stamps():
    p = E.WindowRuleEviction(E.EvictionConfig(cache_size=8), "raas")
    p._step = 20
    rows = torch.rand(8, 30, generator=torch.Generator().manual_seed(0)) * 0.1
    rows[:, 10] = 1.0
    stamps = torch.arange(1, 21)
    order = p._raas_rank(rows, stamps, prefill_len=10, lo=10, hi=22)
    assert stamps[0].item() == 20       # slot 10 is in the top half of every row
    assert order[-1].item() == 0
