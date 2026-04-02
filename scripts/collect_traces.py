#!/usr/bin/env python3
"""
collect_traces.py

Generates reasoning traces from a DeepSeek-R1-Distill model on MATH-500 or AIME 2024,
and computes per-token importance signals for the variance signal ablation study
(see experiments/research_overview.md §3.1).

Runs a manual step-by-step decode loop so that KV signals and H2O attention can
be accumulated incrementally without materializing the full attention matrix.

Usage:
    python scripts/collect_traces.py --dataset math500 --n_samples 100
    python scripts/collect_traces.py --dataset aime2024 --n_samples 30
    python scripts/collect_traces.py --dataset math500 --n_samples 5 --dry_run

Output JSONL — one object per trace (written incrementally to avoid data loss):
    problem         str             the input problem
    ground_truth    str             correct answer
    generated_text  str             full model output (CoT + answer)
    token_ids       List[int]       all token IDs (prompt + generated)
    prompt_len      int             number of prompt tokens
    correct         bool            whether the extracted answer matched ground truth
    signals         Dict[str, List[float]]
        ── Phase 0 signals (always collected) ──────────────────────────────────
        kv_key_var      KV key variance across head_dim, mean over heads+layers
                        (post-RoPE; compared against pre-RoPE via --phase0b)
        kv_key_norm     KV key L2 norm, mean over heads+layers (post-RoPE)
        kv_val_var      KV value variance across head_dim, mean over heads+layers
        cross_head_var  Variance of per-head key means across heads, mean over layers
        h2o_attn        H2O cumulative attention column sums (None if flash attn inactive)
        attn_entropy    Per-token attention entropy H = -sum(a*log(a)), mean over heads+layers
                        (None if flash attn active; requires --force_eager_attn)
        hs_l2_diff      L2 norm of consecutive last-layer hidden-state diffs
                        (0.0 for first token; -1.0 sentinel if seq > hs_max_len)
        hs_cos_dist     Cosine distance between consecutive last-layer hidden states
                        (0.0 for first token; -1.0 sentinel if seq > hs_max_len)
        hs_norm         L2 norm of the last-layer hidden state at each position
                        (-1.0 sentinel if seq > hs_max_len)
        ── Phase 0B signals (collected when --phase0b is set) ──────────────────
        kv_key_var_preRoPE   Key variance before rotary position embedding is applied.
                             Extracted via k_proj forward hook; all-layer mean.
                             (-1.0 sentinel if seq > hs_max_len)
        kv_key_norm_preRoPE  Key L2 norm before RoPE, all-layer mean.
                             (-1.0 sentinel if seq > hs_max_len)
        hs_l2_diff_lN        L2 diff of hidden states at layer N (e.g. hs_l2_diff_l16).
                             Layers are specified via --hs_layers (default: all 32 layers, 0-31).
                             (-1.0 sentinel if seq > hs_max_len)
        hs_cos_dist_lN       Cosine distance of hidden states at layer N.
                             (-1.0 sentinel if seq > hs_max_len)

Design notes:
  - KV signals are read from past_key_values at each decode step — zero extra compute.
    Post-RoPE keys are stored in the cache. Pre-RoPE keys require a forward hook on
    k_proj and are computed in the post-generation forward pass (--phase0b).
  - H2O and attn_entropy both require the softmax attention weight matrix. FlashAttention
    never materialises this, so both signals are gated behind --force_eager_attn.
    attn_entropy at position t: entropy of the attention distribution over t's context.
    Low entropy = model is sharply attending to a few tokens (ThinKV "Thinking" type).
    High entropy = diffuse attention (ThinKV "Rambling" type).
  - Hidden-state signals (Phase 0 and 0B) all use a single post-generation forward pass
    with output_hidden_states=True plus optional k_proj hooks for pre-RoPE keys.
    Limited to sequences <= --hs_max_len tokens. Longer traces get -1.0 sentinel.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# ── Imports with helpful error messages ──────────────────────────────────────

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Missing dependency: pip install datasets")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("Missing dependency: pip install transformers")


# ── KV cache compatibility helpers ────────────────────────────────────────────

def _as_legacy_kv(past_key_values) -> list:
    """
    Normalise past_key_values to a list of (key, value) tensor pairs per layer.

    This codebase targets the transformers version installed on this cluster, where
    DynamicCache stores layers as cache.layers[i] objects with .keys / .values tensors,
    and __iter__ yields (keys, values, sliding_window_tensor_or_None) 3-tuples.

    We always return plain (key, value) 2-tuples so the rest of the code is uniform.
    Tensors have shape (batch, n_heads, seq_len, head_dim).
    """
    if hasattr(past_key_values, 'layers'):
        # New-style DynamicCache: layers list with .keys / .values per layer
        return [(layer.keys, layer.values) for layer in past_key_values.layers]
    if hasattr(past_key_values, 'key_cache'):
        # Older DynamicCache (transformers ~4.38–4.44)
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    # Legacy tuple-of-tuples: each element is (key, value[, ...])
    return [(layer[0], layer[1]) for layer in past_key_values]


# ── Answer extraction ─────────────────────────────────────────────────────────

# Handles nested braces one level deep: \boxed{a+b} and \boxed{\frac{a}{b}}
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(text: str) -> Optional[str]:
    """Return the last \\boxed{} content in text, or None."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def answers_match(pred: Optional[str], gt: str) -> bool:
    if pred is None:
        return False

    def norm(s: str) -> str:
        s = s.strip()
        s = s.replace("\\dfrac", "\\frac")                          # \dfrac == \frac
        s = re.sub(r"\\left\s*([(\[{|])", r"\1", s)                 # \left( → (
        s = re.sub(r"\\right\s*([)\]}|])", r"\1", s)                # \right) → )
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)                  # \text{X} → X
        m = re.match(r"^[a-zA-Z]\s*=\s*(.+)$", s.strip())          # x=5 → 5
        if m:
            s = m.group(1)
        return s.lower().replace(" ", "").replace(",", "")

    if norm(pred) == norm(gt):
        return True

    # Order-insensitive set comparison: "-2,1" vs "1,-2"
    # Split before normalising so commas inside numbers don't break element boundaries
    pred_parts = sorted(norm(p) for p in pred.split(","))
    gt_parts   = sorted(norm(g) for g in gt.split(","))
    return pred_parts == gt_parts


# ── Dataset loaders ───────────────────────────────────────────────────────────

_DEFAULT_HF_CACHE = "/data/hf_cache/skolawol"


def _hf_cache() -> str:
    """
    Return the HuggingFace cache directory.
    Uses HF_HOME env var if set, otherwise /data/hf_cache/skolawol.
    All models and datasets are cached here.
    """
    return os.environ.get("HF_HOME", _DEFAULT_HF_CACHE)


def load_math500(n_samples: int) -> List[Dict]:
    """
    HuggingFaceH4/MATH-500, split=test (500 rows).
    Columns: problem, solution (solution contains \\boxed{answer} at the end).
    Source: OpenAI's PRM800K paper subset.
    """
    print("Loading HuggingFaceH4/MATH-500 (test) ...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=_hf_cache())
    print(f"  {len(ds)} problems available.")

    problems = []
    for item in ds:
        answer = extract_boxed(item["solution"])
        if answer:
            problems.append({
                "problem":      item["problem"].strip(),
                "ground_truth": answer,
                "dataset":      "MATH-500",
                "level":        item.get("level", ""),
                "type":         item.get("type", ""),
            })
        if len(problems) >= n_samples:
            break

    print(f"  Collected {len(problems)} usable problems.")
    return problems


def load_aime2024(n_samples: int) -> List[Dict]:
    """
    Maxwell-Jia/AIME_2024, split=train (30 rows).
    Columns: ID, Problem, Solution, Answer.
    Answers are integers 0–999.
    """
    print("Loading Maxwell-Jia/AIME_2024 (train) ...")
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train", cache_dir=_hf_cache())
    print(f"  {len(ds)} problems available.")

    problems = []
    for item in ds:
        problems.append({
            "problem":      item["Problem"].strip(),
            "ground_truth": str(item["Answer"]).strip(),
            "dataset":      "AIME2024",
            "id":           item.get("ID", ""),
        })
        if len(problems) >= n_samples:
            break

    print(f"  Collected {len(problems)} usable problems.")
    return problems


def load_livecodebench(n_samples: int) -> List[Dict]:
    """
    livecodebench/code_generation, split=test (121 rows).
    Generates the longest reasoning traces (~14k tokens avg per ThinKV).
    Correctness for code requires executing the generated code against test cases —
    that is handled separately; here we set ground_truth=None and correct=None.
    Column used: question_content (full problem description with I/O examples).
    """
    print("Loading livecodebench/code_generation (test) ...")
    ds = load_dataset("livecodebench/code_generation", split="test", cache_dir=_hf_cache())
    print(f"  {len(ds)} problems available.")
    print("  Note: correctness checking requires code execution — set correct=None for now.")

    # Inspect columns on first item so we pick the right field
    first = ds[0]
    problem_col = next(
        (c for c in ["question_content", "problem", "prompt", "description"] if c in first),
        None
    )
    if problem_col is None:
        raise RuntimeError(
            f"Could not find problem column in LiveCodeBench. "
            f"Available columns: {list(first.keys())}"
        )

    problems = []
    for item in ds:
        problems.append({
            "problem":      item[problem_col].strip(),
            "ground_truth": None,   # code execution required
            "dataset":      "LiveCodeBench",
            "difficulty":   item.get("difficulty", ""),
        })
        if len(problems) >= n_samples:
            break

    print(f"  Collected {len(problems)} problems.")
    return problems


def load_gsm8k(n_samples: int) -> List[Dict]:
    """
    openai/gsm8k, split=test (~1319 rows).
    Columns: question, answer.
    Answer format: "... #### 42" — extract the number after ####.
    Used as low-pressure control: traces are short (~200 tokens), cache never fills.
    """
    print("Loading openai/gsm8k (test) ...")
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=_hf_cache())
    print(f"  {len(ds)} problems available.")

    _gsm_re = re.compile(r"####\s*(-?[\d,]+)")

    problems = []
    for item in ds:
        m = _gsm_re.search(item["answer"])
        if m:
            answer = m.group(1).replace(",", "").strip()
            problems.append({
                "problem":      item["question"].strip(),
                "ground_truth": answer,
                "dataset":      "GSM8K",
            })
        if len(problems) >= n_samples:
            break

    print(f"  Collected {len(problems)} usable problems.")
    return problems


# ── Prompt construction ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def build_prompt_ids(tokenizer, problem: str) -> torch.Tensor:
    """
    Build tokenized prompt using the model's chat template.
    Returns (1, prompt_len) int64 tensor.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt")["input_ids"]  # (1, prompt_len)


# ── Signal accumulation ───────────────────────────────────────────────────────

class SignalAccumulator:
    """
    Collects per-token importance signals during and after decode.

    KV signals (kv_key_var, kv_key_norm, kv_val_var, cross_head_var):
        Read from past_key_values at each decode step.  For decode tokens,
        only the newly appended position is read.  For the prompt, all
        prompt positions are extracted after the prefill pass.
        Cost: negligible — tensors already in GPU memory.

    Attention signals (h2o_attn, attn_entropy) — require --force_eager_attn:
        h2o_attn:     cumulative attention column sums (how much each token was attended to).
        attn_entropy: per-token entropy of the attention distribution over context.
                      Low = sharp/focused (ThinKV "Thinking"); high = diffuse ("Rambling").
        Both require output_attentions=True, incompatible with FlashAttention.

    Hidden-state signals (hs_l2_diff, hs_cos_dist, hs_norm):
        Computed via a single post-generation forward pass with output_hidden_states=True.
        Uses only the last hidden layer.  Limited to sequences <= hs_max_len; longer
        traces get -1.0 sentinel.
    """

    def __init__(self, max_len: int):
        self.max_len = max_len
        # Phase 0 — KV signals (post-RoPE, from past_key_values)
        self._kv_key_var     = np.zeros(max_len, dtype=np.float32)
        self._kv_key_norm    = np.zeros(max_len, dtype=np.float32)
        self._kv_val_var     = np.zeros(max_len, dtype=np.float32)
        self._cross_head_var = np.zeros(max_len, dtype=np.float32)
        # Phase 0 — attention signals (require --force_eager_attn)
        self._h2o_attn       = np.zeros(max_len, dtype=np.float32)
        self._attn_entropy   = np.zeros(max_len, dtype=np.float32)
        self._h2o_available  = True  # flipped to False if flash attn active
        # Phase 0 — last-layer hidden-state signals (post-generation forward pass)
        self.hs_l2_diff      = np.zeros(max_len, dtype=np.float32)
        self.hs_cos_dist     = np.zeros(max_len, dtype=np.float32)
        self.hs_norm         = np.zeros(max_len, dtype=np.float32)
        # Phase 0B — pre-RoPE key signals (populated by fill_hidden_states when
        #             collect_preRoPE=True; captured via k_proj forward hooks)
        self._kv_key_var_preRoPE  = np.zeros(max_len, dtype=np.float32)
        self._kv_key_norm_preRoPE = np.zeros(max_len, dtype=np.float32)
        self._preRoPE_collected   = False   # set to True once hooks have fired
        # Phase 0B — per-layer hidden-state signals keyed by layer index
        self._hs_by_layer: Dict[int, Dict[str, np.ndarray]] = {}

    # ── KV signal helpers ─────────────────────────────────────────────────

    def _extract_kv_at_pos(self, past_key_values, pos: int):
        """Extract and accumulate KV signals for a single token position."""
        layers = _as_legacy_kv(past_key_values)
        n_layers = len(layers)
        kv_key_var = kv_key_norm = kv_val_var = cross_head_var = 0.0

        for layer_k, layer_v in layers:
            # layer_k: (1, n_heads, seq_len, head_dim)
            k = layer_k[0, :, pos, :].float().cpu()   # (n_heads, head_dim)
            v = layer_v[0, :, pos, :].float().cpu()

            kv_key_var    += k.var(dim=-1).mean().item()
            kv_key_norm   += k.norm(dim=-1).mean().item()
            kv_val_var    += v.var(dim=-1).mean().item()
            # cross-head variance: variance across heads of per-head key mean
            cross_head_var += k.mean(dim=-1).var().item()

        self._kv_key_var[pos]     = kv_key_var    / n_layers
        self._kv_key_norm[pos]    = kv_key_norm   / n_layers
        self._kv_val_var[pos]     = kv_val_var    / n_layers
        self._cross_head_var[pos] = cross_head_var / n_layers

    def fill_prompt_kv(self, past_key_values, prompt_len: int):
        """Extract KV signals for all prompt positions after the prefill pass."""
        layers = _as_legacy_kv(past_key_values)
        n_layers = len(layers)
        kv_key_var    = np.zeros(prompt_len, dtype=np.float64)
        kv_key_norm   = np.zeros(prompt_len, dtype=np.float64)
        kv_val_var    = np.zeros(prompt_len, dtype=np.float64)
        cross_head_var = np.zeros(prompt_len, dtype=np.float64)

        for layer_k, layer_v in layers:
            k = layer_k[0].float().cpu()   # (n_heads, prompt_len, head_dim)
            v = layer_v[0].float().cpu()

            kv_key_var    += k.var(dim=-1).mean(dim=0).numpy()
            kv_key_norm   += k.norm(dim=-1).mean(dim=0).numpy()
            kv_val_var    += v.var(dim=-1).mean(dim=0).numpy()
            cross_head_var += k.mean(dim=-1).var(dim=0).numpy()

        self._kv_key_var[:prompt_len]    = (kv_key_var    / n_layers).astype(np.float32)
        self._kv_key_norm[:prompt_len]   = (kv_key_norm   / n_layers).astype(np.float32)
        self._kv_val_var[:prompt_len]    = (kv_val_var    / n_layers).astype(np.float32)
        self._cross_head_var[:prompt_len] = (cross_head_var / n_layers).astype(np.float32)

    def fill_prompt_attn_entropy(self, attentions, prompt_len: int):
        """
        Extract attention entropy for all prompt positions from the prefill attention matrix.
        attentions: tuple of (1, n_heads, prompt_len, prompt_len) per layer (causal).
        Requires output_attentions=True (i.e. --force_eager_attn).
        """
        if attentions is None:
            return
        n_layers = len(attentions)
        entropy_sum = np.zeros(prompt_len, dtype=np.float64)
        for layer_attn in attentions:
            # (1, n_heads, prompt_len, prompt_len) — last dim is attended positions
            a = layer_attn[0].float().cpu().clamp(min=1e-10)  # (n_heads, prompt_len, prompt_len)
            # Masked positions have near-zero weight; their contribution is negligible after clamp
            H = -(a * a.log()).sum(dim=-1)   # (n_heads, prompt_len)
            entropy_sum += H.mean(dim=0).numpy()
        self._attn_entropy[:prompt_len] = (entropy_sum / n_layers).astype(np.float32)

    def update_decode_step(self, pos: int, past_key_values, attn_weights):
        """
        Called once per decode step.
        pos: 0-indexed position of the token just generated in the full sequence.
        attn_weights: tuple of (1, n_heads, 1, seq_len_so_far) per layer, or None.
        """
        self._extract_kv_at_pos(past_key_values, pos)

        if attn_weights is not None and self._h2o_available:
            n_layers = len(attn_weights)
            entropy_sum = 0.0
            for layer_attn in attn_weights:
                # layer_attn: (1, n_heads, 1, pos+1) — current token attending to all prior
                a = layer_attn[0, :, 0, :].float().cpu()   # (n_heads, pos+1)
                # H2O: accumulate attention column sums (how much each prior token was attended to)
                self._h2o_attn[:a.shape[-1]] += a.mean(dim=0).numpy() / n_layers
                # Attention entropy at this position: H = -sum(a * log(a)), mean over heads
                a_clamped = a.clamp(min=1e-10)
                entropy_sum += -(a_clamped * a_clamped.log()).sum(dim=-1).mean().item()
            self._attn_entropy[pos] = entropy_sum / n_layers
        elif attn_weights is None:
            self._h2o_available = False

    # ── Hidden-state signals ──────────────────────────────────────────────

    def _compute_hs_signals(self, h: torch.Tensor, seq_len: int):
        """
        Compute l2_diff, cos_dist, and norm arrays from a hidden-state tensor.

        Args:
            h: (seq_len, d_model) float tensor for one layer.
            seq_len: number of valid positions.

        Returns:
            (l2_diff, cos_dist, norm) each as (seq_len,) np.float32 array.
            Token 0 always gets 0.0 for diff/dist (no predecessor).
        """
        norms = h.norm(dim=-1)                                       # (seq_len,)
        diffs = (h[1:] - h[:-1]).norm(dim=-1)                       # (seq_len-1,)
        normed = h / norms.clamp(min=1e-8).unsqueeze(-1)
        cos_sim = (normed[1:] * normed[:-1]).sum(dim=-1).clamp(-1.0, 1.0)

        l2_diff  = np.zeros(seq_len, dtype=np.float32)
        cos_dist = np.zeros(seq_len, dtype=np.float32)
        l2_diff[1:]  = diffs.numpy()
        cos_dist[1:] = (1.0 - cos_sim).numpy()
        return l2_diff, cos_dist, norms.numpy().astype(np.float32)

    def fill_hidden_states(
        self,
        model,
        input_ids: torch.Tensor,
        hs_max_len: int,
        hs_layers: Optional[List[int]] = None,
        collect_preRoPE: bool = False,
    ):
        """
        Single post-generation forward pass computing all hidden-state and
        pre-RoPE key signals.

        Phase 0 (always):
          - Last-layer hs_l2_diff, hs_cos_dist, hs_norm.

        Phase 0B (when hs_layers or collect_preRoPE are set):
          - Per-layer hs_l2_diff_lN / hs_cos_dist_lN for each layer in hs_layers.
          - kv_key_var_preRoPE / kv_key_norm_preRoPE via k_proj forward hooks.

        Sequences longer than hs_max_len receive -1.0 sentinel for all signals.

        Args:
            model:            loaded HuggingFace model (eval mode).
            input_ids:        (1, seq_len) full token ID tensor.
            hs_max_len:       sentinel threshold in tokens.
            hs_layers:        list of layer indices for per-layer HS (e.g. [16, 20, 24]).
            collect_preRoPE:  whether to register k_proj hooks and collect pre-RoPE keys.
        """
        if hs_layers is None:
            hs_layers = []

        seq_len = input_ids.shape[1]

        # ── Sentinel path: sequence too long ─────────────────────────────
        if seq_len > hs_max_len:
            self.hs_l2_diff[:seq_len]  = -1.0
            self.hs_cos_dist[:seq_len] = -1.0
            self.hs_norm[:seq_len]     = -1.0
            for layer_idx in hs_layers:
                self._hs_by_layer[layer_idx] = {
                    "l2_diff":  np.full(seq_len, -1.0, dtype=np.float32),
                    "cos_dist": np.full(seq_len, -1.0, dtype=np.float32),
                }
            if collect_preRoPE:
                self._kv_key_var_preRoPE[:seq_len]  = -1.0
                self._kv_key_norm_preRoPE[:seq_len] = -1.0
            return

        device = next(model.parameters()).device

        # ── Register pre-RoPE hooks on k_proj of every layer ─────────────
        # k_proj output: (1, seq_len, n_kv_heads * head_dim) — before RoPE.
        # We capture and accumulate across layers to compute an all-layer mean.
        preRoPE_key_var_sum  = np.zeros(seq_len, dtype=np.float64)
        preRoPE_key_norm_sum = np.zeros(seq_len, dtype=np.float64)
        preRoPE_n_layers     = 0
        hooks = []

        if collect_preRoPE:
            cfg        = model.config
            n_kv_heads = cfg.num_key_value_heads
            head_dim   = cfg.hidden_size // cfg.num_attention_heads

            def _make_preRoPE_hook():
                # Closure captures the mutable accumulators above.
                def _hook(module, inp, output):
                    nonlocal preRoPE_n_layers
                    # output: (1, seq_len, n_kv_heads * head_dim)
                    k = output.detach().float().cpu()
                    k = k.view(k.shape[0], k.shape[1], n_kv_heads, head_dim)
                    k = k[0]  # (seq_len, n_kv_heads, head_dim)
                    # Variance and norm across head_dim, averaged over n_kv_heads
                    preRoPE_key_var_sum  [:] += k.var(dim=-1).mean(dim=-1).numpy()
                    preRoPE_key_norm_sum [:] += k.norm(dim=-1).mean(dim=-1).numpy()
                    preRoPE_n_layers += 1
                return _hook

            for layer in model.model.layers:
                hooks.append(layer.self_attn.k_proj.register_forward_hook(
                    _make_preRoPE_hook()
                ))

        # ── Single forward pass ───────────────────────────────────────────
        with torch.no_grad():
            out = model(
                input_ids=input_ids.to(device),
                output_hidden_states=True,
                use_cache=False,
            )

        for h in hooks:
            h.remove()

        # out.hidden_states: tuple (n_layers+1) × (1, seq_len, d_model)
        # Index 0 = embedding output; index i+1 = output of transformer layer i.

        # ── Last-layer HS signals (Phase 0) ──────────────────────────────
        last_h = out.hidden_states[-1][0].float().cpu()  # (seq_len, d_model)
        l2_diff, cos_dist, norm = self._compute_hs_signals(last_h, seq_len)
        self.hs_l2_diff[:seq_len]  = l2_diff
        self.hs_cos_dist[:seq_len] = cos_dist
        self.hs_norm[:seq_len]     = norm

        # ── Per-layer HS signals (Phase 0B) ──────────────────────────────
        for layer_idx in hs_layers:
            hs_index = layer_idx + 1   # hidden_states[0]=embed, [i+1]=layer i output
            if hs_index >= len(out.hidden_states):
                continue
            h = out.hidden_states[hs_index][0].float().cpu()
            l2_diff, cos_dist, _ = self._compute_hs_signals(h, seq_len)
            self._hs_by_layer[layer_idx] = {
                "l2_diff":  l2_diff,
                "cos_dist": cos_dist,
            }

        del out
        torch.cuda.empty_cache()

        # ── Pre-RoPE key signals (Phase 0B) ──────────────────────────────
        if collect_preRoPE and preRoPE_n_layers > 0:
            self._kv_key_var_preRoPE[:seq_len]  = (
                preRoPE_key_var_sum  / preRoPE_n_layers
            ).astype(np.float32)
            self._kv_key_norm_preRoPE[:seq_len] = (
                preRoPE_key_norm_sum / preRoPE_n_layers
            ).astype(np.float32)
            self._preRoPE_collected = True

    # ── Export ────────────────────────────────────────────────────────────

    def to_dict(self, seq_len: int) -> Dict[str, List]:
        attn_signals = (
            {
                "h2o_attn":     self._h2o_attn[:seq_len].tolist(),
                "attn_entropy": self._attn_entropy[:seq_len].tolist(),
            }
            if self._h2o_available
            else {"h2o_attn": None, "attn_entropy": None}
        )
        result = {
            # Phase 0 — KV signals (post-RoPE)
            "kv_key_var":     self._kv_key_var[:seq_len].tolist(),
            "kv_key_norm":    self._kv_key_norm[:seq_len].tolist(),
            "kv_val_var":     self._kv_val_var[:seq_len].tolist(),
            "cross_head_var": self._cross_head_var[:seq_len].tolist(),
            # Phase 0 — attention signals
            **attn_signals,
            # Phase 0 — last-layer hidden-state signals
            "hs_l2_diff":     self.hs_l2_diff[:seq_len].tolist(),
            "hs_cos_dist":    self.hs_cos_dist[:seq_len].tolist(),
            "hs_norm":        self.hs_norm[:seq_len].tolist(),
        }
        # Phase 0B — pre-RoPE key signals (only if collected; non-zero = populated)
        if self._preRoPE_collected:
            result["kv_key_var_preRoPE"]  = self._kv_key_var_preRoPE[:seq_len].tolist()
            result["kv_key_norm_preRoPE"] = self._kv_key_norm_preRoPE[:seq_len].tolist()
        # Phase 0B — per-layer hidden-state signals
        for layer_idx, sigs in sorted(self._hs_by_layer.items()):
            result[f"hs_l2_diff_l{layer_idx}"]  = sigs["l2_diff"][:seq_len].tolist()
            result[f"hs_cos_dist_l{layer_idx}"] = sigs["cos_dist"][:seq_len].tolist()
        return result


# ── Core generation loop ──────────────────────────────────────────────────────

def run_trace(
    model,
    tokenizer,
    problem: Dict,
    max_new_tokens: int,
    hs_max_len: int,
    collect_h2o: bool,
    device: torch.device,
    hs_layers: Optional[List[int]] = None,
    collect_preRoPE: bool = False,
) -> Optional[Dict]:
    """
    Run a single problem through the model and collect all signals.
    Returns None on error (logged to stderr).
    """
    prompt_ids = build_prompt_ids(tokenizer, problem["problem"]).to(device)
    prompt_len = prompt_ids.shape[1]
    max_total = prompt_len + max_new_tokens

    accumulator = SignalAccumulator(max_len=max_total)

    # ── Prefill pass ──────────────────────────────────────────────────────
    with torch.no_grad():
        prefill_out = model(
            input_ids=prompt_ids,
            use_cache=True,
            output_attentions=collect_h2o,
        )

    past_kv = prefill_out.past_key_values
    accumulator.fill_prompt_kv(past_kv, prompt_len)

    # H2O + attn_entropy for prompt tokens (requires --force_eager_attn).
    # Prefill attention matrix: (1, n_heads, prompt_len, prompt_len) per layer.
    if collect_h2o and prefill_out.attentions is not None:
        n_layers = len(prefill_out.attentions)
        for layer_attn in prefill_out.attentions:
            # H2O: sum over query dimension to get column sums (how much each token was attended to)
            col_sums = layer_attn[0].float().cpu().mean(dim=0).sum(dim=0).numpy()
            accumulator._h2o_attn[:prompt_len] += col_sums / n_layers
        # Attention entropy: H_j = -sum_k a_{j,k} * log(a_{j,k}), mean over heads+layers
        accumulator.fill_prompt_attn_entropy(prefill_out.attentions, prompt_len)
    elif collect_h2o:
        accumulator._h2o_available = False

    # Seed the decode loop with the first predicted token from the prefill.
    next_token = prefill_out.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)  # (1, 1)
    del prefill_out
    torch.cuda.empty_cache()

    # ── Decode loop ───────────────────────────────────────────────────────

    generated_ids: List[int] = []

    eos_id = tokenizer.eos_token_id
    pos = prompt_len  # position of next generated token in full sequence

    for _ in range(max_new_tokens):
        token_id = int(next_token[0, 0])
        generated_ids.append(token_id)

        if token_id == eos_id:
            break

        with torch.no_grad():
            step_out = model(
                input_ids=next_token.to(device),
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=collect_h2o,
            )

        past_kv = step_out.past_key_values
        attn = step_out.attentions if collect_h2o else None

        accumulator.update_decode_step(pos, past_kv, attn)

        next_token = step_out.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        pos += 1

        del step_out
        if pos % 512 == 0:
            torch.cuda.empty_cache()

    # ── Hidden-state signals (post-hoc) ───────────────────────────────────
    full_ids = torch.cat([
        prompt_ids.cpu(),
        torch.tensor(generated_ids, dtype=torch.long).unsqueeze(0)
    ], dim=1)  # (1, total_len)

    accumulator.fill_hidden_states(
        model, full_ids, hs_max_len,
        hs_layers=hs_layers,
        collect_preRoPE=collect_preRoPE,
    )

    # ── Answer extraction ─────────────────────────────────────────────────
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    pred_answer = extract_boxed(generated_text)
    if problem["ground_truth"] is None:
        correct = None   # LiveCodeBench: requires code execution, checked separately
    else:
        correct = answers_match(pred_answer, problem["ground_truth"])

    total_len = prompt_len + len(generated_ids)

    return {
        "problem":        problem["problem"],
        "ground_truth":   problem["ground_truth"],
        "generated_text": generated_text,
        "token_ids":      prompt_ids[0].tolist() + generated_ids,
        "prompt_len":     prompt_len,
        "correct":        correct,
        "pred_answer":    pred_answer,
        "dataset":        problem.get("dataset", ""),
        "signals":        accumulator.to_dict(total_len),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",           default="deepseek-ai/deepseek-r1-distill-llama-8b",
                   help="HuggingFace model ID (default: deepseek-r1-distill-llama-8b)")
    p.add_argument("--dataset",
                   choices=["math500", "aime2024", "livecodebench", "gsm8k"],
                   default="math500",
                   help="Dataset to use (default: math500)")
    p.add_argument("--n_samples",       type=int, default=20,
                   help="Number of problems to process (default: 20)")
    p.add_argument("--max_new_tokens",  type=int, default=4096,
                   help="Max tokens to generate per problem (default: 4096)")
    p.add_argument("--hs_max_len",      type=int, default=None,
                   help="Max sequence length for hidden-state forward pass (default: matches --max_new_tokens)")
    p.add_argument("--output",          type=Path, default=None,
                   help="Output JSONL path (default: data/<dataset>_traces.jsonl)")
    p.add_argument("--force_eager_attn", action="store_true",
                   help="Force eager attention (disables flash attn) to enable H2O collection")
    p.add_argument("--phase0b",          action="store_true",
                   help="Enable Phase 0B signal collection: pre-RoPE key variance and "
                        "per-layer hidden-state signals. Adds kv_key_var_preRoPE, "
                        "kv_key_norm_preRoPE, hs_l2_diff_lN, hs_cos_dist_lN to output. "
                        "Uses a post-generation forward pass with k_proj hooks.")
    p.add_argument("--hs_layers",        type=str,
                   default=",".join(str(i) for i in range(32)),
                   help="Comma-separated layer indices for per-layer HS signals when "
                        "--phase0b is set (default: all 32 layers, 0-31). Ignored without --phase0b.")
    p.add_argument("--dry_run",         action="store_true",
                   help="Load model and run 1 problem to verify setup, then exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        args.n_samples = 1
        args.max_new_tokens = 128
        print("DRY RUN: 1 sample, 128 max tokens")

    # hs_max_len defaults to match generation budget so hidden-state signals
    # cover the same sequences as all other signals
    if args.hs_max_len is None:
        args.hs_max_len = args.max_new_tokens

    # Output path
    output_path = args.output or Path(f"data/{args.dataset}_traces.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load problems
    loader = {
        "math500":       load_math500,
        "aime2024":      load_aime2024,
        "livecodebench": load_livecodebench,
        "gsm8k":         load_gsm8k,
    }[args.dataset]
    problems = loader(args.n_samples)
    if not problems:
        sys.exit("No problems loaded. Check dataset name and network access.")

    # Load model
    print(f"\nLoading model: {args.model}")
    cache_dir = _hf_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)

    model_kwargs = dict(
        dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    if args.force_eager_attn:
        model_kwargs["attn_implementation"] = "eager"
        print("  Forcing eager attention (flash attention disabled) for H2O collection.")

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    collect_h2o = args.force_eager_attn
    hs_layers = (
        [int(x.strip()) for x in args.hs_layers.split(",") if x.strip()]
        if args.phase0b else []
    )
    collect_preRoPE = args.phase0b

    if args.phase0b:
        print(f"  Phase 0B: pre-RoPE key signals ENABLED; "
              f"per-layer HS at layers {hs_layers}")
    if not collect_h2o:
        print(
            "  Note: H2O signal will be None (flash attention active). "
            "Use --force_eager_attn to enable. H2O can also be computed "
            "separately in signal_ablation.py."
        )

    device = next(model.parameters()).device
    print(f"  Model on device: {device}")
    print(f"  Collecting signals: KV-based always; H2O={'yes' if collect_h2o else 'no'}; "
          f"hidden-states for seqs<={args.hs_max_len} tokens\n")

    # Resume support: skip problems already written to output
    done_problems: set = set()
    if output_path.exists():
        with open(output_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    try:
                        done_problems.add(json.loads(_line)["problem"])
                    except Exception:
                        pass
    if done_problems:
        n_skip = sum(1 for p in problems if p["problem"] in done_problems)
        problems = [p for p in problems if p["problem"] not in done_problems]
        print(f"  Resuming: {n_skip} problems already done, {len(problems)} remaining.")
    write_mode = "a" if done_problems else "w"

    # Run
    n_correct = 0
    n_total = 0
    n_hs_skipped = 0

    with open(output_path, write_mode) as f_out:
        for problem in tqdm(problems, desc="Generating traces"):
            try:
                result = run_trace(
                    model=model,
                    tokenizer=tokenizer,
                    problem=problem,
                    max_new_tokens=args.max_new_tokens,
                    hs_max_len=args.hs_max_len,
                    collect_h2o=collect_h2o,
                    device=device,
                    hs_layers=hs_layers,
                    collect_preRoPE=collect_preRoPE,
                )
            except Exception as e:
                import traceback
                print(f"\n  [SKIP] Error on problem: {e}", file=sys.stderr)
                if args.dry_run:
                    traceback.print_exc(file=sys.stderr)
                continue

            if result is None:
                continue

            n_total += 1
            if result["correct"] is True:
                n_correct += 1

            total_len = len(result["token_ids"])
            if total_len > args.hs_max_len:
                n_hs_skipped += 1

            # Write immediately — don't lose data if interrupted
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

            tqdm.write(
                f"  len={total_len:>6}  correct={result['correct']}  "
                f"pred={result['pred_answer']!r:>12}  gt={result['ground_truth']!r}"
            )

    # Summary
    print(f"\n{'='*60}")
    print(f"Done. Wrote {n_total} traces to {output_path}")
    if args.dataset == "livecodebench":
        print(f"Accuracy:          N/A (code execution required)")
    else:
        print(f"Accuracy:          {n_correct}/{n_total} = {n_correct/max(n_total,1):.1%}")
    print(f"HS signals skipped (seq > {args.hs_max_len}): {n_hs_skipped}/{n_total}")
    if not collect_h2o:
        print("H2O + attn_entropy: not collected (re-run with --force_eager_attn)")
    if args.phase0b:
        print(f"Phase 0B: pre-RoPE keys + HS layers {hs_layers} collected")
    else:
        print("Phase 0B signals: not collected (re-run with --phase0b)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
