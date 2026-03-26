# Epiphany-Aware KV Cache Management — Technical Research Overview

*Last updated: March 2026*

---

## 1. Problem Statement and Motivation

### The Core Problem

Transformer-based reasoning models — DeepSeek-R1, QwQ, NVIDIA AceReason, and their distilled variants — generate extremely long chain-of-thought (CoT) sequences during inference. A single AIME problem may require 20,000–60,000 tokens of internal reasoning before producing a final answer. This creates a KV cache that grows linearly with sequence length:

```
Memory ≈ 2 × n_layers × n_heads × d_head × seq_len × bytes_per_element
```

For LLaMA-3-70B at 60k tokens (FP16): approximately 120GB — well beyond a single H100's 80GB. Even at 8B parameters, a 20k-token reasoning trace consumes ~16GB, dominating the available memory budget and drastically reducing viable batch sizes.

**The standard response to this is KV cache eviction**: keep only the most important K token entries, discard the rest. But this introduces the central question: *what counts as important?*

### The Hypothesis

All mainstream eviction methods use attention scores as their importance proxy. This project's core hypothesis is that **attention weight is a systematically bad proxy for semantic importance in long reasoning traces**, and that **hidden-state variance** — broadly, signals derived from representational change in the model's internal state — is a better proxy.

The reasoning:

1. **Attention sinks**: The first few tokens in any sequence receive disproportionately high attention scores regardless of content (Xiao et al., StreamingLLM 2023). This is structural/positional, not semantic.

2. **Thinking filler tokens**: In reasoning traces, tokens like "Let me think...", "Hmm, wait...", "So if we consider..." receive high attention during generation because the model is actively attending to its current position. But these tokens carry no reusable semantic content — the model will never need to reference "Let me" again once it has moved on.

3. **Hidden-state variance as a semantic signal**: When a model generates a semantically significant token — a key intermediate result, a decision point, a concluded insight — the hidden state undergoes a larger representational shift than when generating filler. Some form of representational change signal (see Section 3 for the full design space) should be higher at semantically significant positions.

4. **The epiphany hypothesis**: Reasoning models have identifiable transition moments where they shift from exploratory/uncertain generation to convergent/committed generation. These transitions should be detectable as large representational shifts. Tokens near these transitions are candidates for high-priority retention.

### What "Hidden-State Variance" Actually Means — The Design Space

"Hidden-state variance" is not a single signal. There are at least five independent dimensions along which different measurement approaches diverge, and the best combination is an empirical question that must be ablated. See Section 3.1 for the full ablation plan.

### What Is Valid

The general claim — that attention weight is an imperfect proxy for semantic importance in reasoning traces — is **well-supported by existing empirical evidence**:

- StreamingLLM demonstrates attention sinks exist and persist regardless of content
- RaaS (2025) empirically proves that H2O-style eviction causes 24.2% of attention maps to fail in reasoning traces, forcing models into infinite repetition loops
- ThinKV (ICLR 2026 oral) demonstrates that reasoning traces have structurally distinct attention sparsity patterns corresponding to different reasoning phases
- SideQuest (2026) demonstrates that non-monotonic token utility in long-horizon tasks causes catastrophic failure for all attention-based heuristics

The use case (long *generation* traces from reasoning models) is also genuinely distinct from prior work, which targets long *input* contexts (document summarization, RAG). SnapKV, PyramidKV, and ChunkKV are all prefill-time methods. FreeKV and Quest are designed for retrieval, not generation-side eviction. ThinKV and RaaS are the only decode-time methods designed specifically for reasoning traces.

**A practical engineering advantage (separate from accuracy)**: Every attention-score-based eviction method (H2O, SnapKV, ThinKV) requires materialising the full $n \times n$ attention weight matrix at each decode step — which is exactly what FlashAttention (FA2) is designed to avoid. FA2 tiles the attention computation in SRAM, never writing the full matrix to HBM, achieving O(n) peak memory and 2–4× throughput gains (Dao et al., FlashAttention, NeurIPS 2022; FlashAttention-2, ICLR 2024). Requesting `output_attentions=True` forces a fallback to standard (eager) attention, eliminating these gains. At the 8k–32k sequence lengths typical of reasoning traces, this is the difference between fitting comfortably on one GPU and OOMing. Our method reads only from `past_key_values` (already in HBM) and the hidden states (post-hoc pass, also FA2-compatible) — **fully compatible with FA2 inference without modification**. This should be stated explicitly in the introduction: we propose the first decode-time eviction method for reasoning traces that works within standard production inference stacks (vLLM, TGI, SGLang) without disabling their primary throughput optimisation.

**A signal-quality degradation at long context (separate from cost)**: Even when eager attention is available, attention-based signals become *semantically unreliable* at long contexts due to the **attention sink** phenomenon. Xiao et al. (StreamingLLM, ICLR 2024) first documented that transformer models concentrate disproportionate attention weight on the first few tokens (positions 0–4) regardless of their semantic content — an artifact of softmax normalisation over very long sequences. At 32k tokens, a large fraction of cumulative attention goes to these sink tokens, drowning out genuinely informative positions. This makes H2O and entropy-based signals progressively noisier as context grows. KV-vector signals and hidden-state signals do not have this specific failure mode: variance and norm are computed per-position independently of the softmax distribution. This is a second, distinct reason to expect our signals to outperform attention-based baselines on long reasoning traces — and it is separately documented in the literature from the O(n²) memory argument above.

### What Is Unproven

The *specific* claim — that hidden-state variance is a better signal than attention scores for token importance in reasoning traces — has not been empirically validated yet. The following remain open questions:

- Does L2 hidden-state difference actually correlate with semantic importance, or does it correlate with syntactic/punctuation boundaries?
- Does KV-vector variance (the available proxy at decode time, since full hidden states are expensive) carry enough signal to be useful?
- Is key-vector variance meaningfully different from what cumulative attention (H2O-style) already captures?

These must be answered empirically before the hypothesis can be claimed as validated.

### Known Methodological Limitations (current implementation)

Documented approximations in the Phase 0 pipeline that may affect signal ablation results. None are blockers; all are noted for paper writeup.

1. **Window overlap — overwrite vs. OR semantics** *(fixed)*: In `label_importance.py`, overlapping windows (stride=16, window=32 means each interior position is covered by 2 windows) originally used overwrite semantics — the last covering window's label won. A position could be labeled 0 even if a *prior* window covering it caused an answer flip. Fixed to OR semantics: a position is labeled important if *any* covering window flipped. This increases measured importance slightly but is more conservative and correct.

2. **Cumsum = position proxy** *(fixed)*: Cumulative sum of non-negative signals (variance, L2-diff) is monotonically increasing — the Spearman rank-order is equivalent to sequence position, not signal content. Empirically confirmed: all three cumsum variants produced identical ρ = -0.6043 on first test trace. Replaced with `_rolling64` (rolling mean over 64 tokens), which tests "sustained elevated signal" without the artifact.

3. **Answer normaliser coverage**: `answers_match` handles LaTeX variants (`\dfrac`/`\frac`, `\left(`/`(`, `\text{}`) and set reordering. Does not handle: symbolic equivalence (`\sin(\pi/6)` ≠ `1/2` in string comparison), approximate decimal equality, or multi-line answers. Some `correct=False` traces may be false negatives.

4. **Post-RoPE key signals**: All KV signals are computed from post-RoPE keys (as stored in the cache). RoPE rotation inflates variance at large positions regardless of content. Pre-RoPE ablation (Dimension 2) is deferred to Phase 0B.

5. **Single-layer hidden states**: `hs_*` signals use only the final transformer layer. Layer-wise ablation (Dimension 3) — whether earlier layers provide cleaner importance signals — is deferred.

6. **Short regeneration budget in labelling**: `max_new_tokens=512` per masked window. For problems requiring >512 tokens to reach `\boxed{}`, a window might appear unimportant (no flip) simply because the model couldn't finish the answer. Estimates suggest this affects ~5–15% of windows on hard AIME traces.

---

## 2. Related Work — Detailed Comparison

### 2.1 StreamingLLM (Xiao et al., 2023)

**Core mechanism**: Keep the first 4 "attention sink" tokens (which attract disproportionate attention for positional/structural reasons regardless of content) plus a fixed sliding window of the most recent W tokens. Everything else is permanently discarded.

**Novelty**: First formal characterization of the attention sink phenomenon and a principled response to it. Enables infinite-length text generation without restarting.

**What it misses**: No content-aware selection at all. Every token outside the recent window is discarded regardless of semantic importance. Completely unsuitable for tasks requiring recall of specific facts from earlier in the sequence (multi-step math, long-range reasoning).

**Relevance to this project**: Baseline to beat trivially. The claim that "attention sinks exist and attention scores are noisy" directly motivates our work.

---

### 2.2 H2O — Heavy Hitter Oracle (Zhang et al., 2023)

**Core mechanism**: Tracks cumulative attention received by each token across all decode steps. Tokens that have been repeatedly attended to ("heavy hitters") are retained; tokens with low cumulative attention are evicted. Eviction is permanent.

**Novelty**: Cumulative attention is a significant improvement over single-step attention as an importance proxy. A token used repeatedly across many steps is more plausible as a "load-bearing" memory item than one with a single high-attention step.

**What it misses**: In reasoning traces, H2O fails catastrophically (RaaS empirically confirms 24.2% failure rate). The problem: milestone tokens have high cumulative attention *while they are being used* for an intermediate step, then their cumulative score stops growing. A token that was heavily used 5k tokens ago for an intermediate calculation now looks stale by cumulative metrics, but H2O continues to retain it because its historical cumulative score is high. Conversely, it may discard a token with low but recent attention that will be needed soon.

**Relevance**: Standard baseline for eviction. Our attention-based baseline should be H2O-style (cumulative), not single-step, to ensure fair comparison.

---

### 2.3 SnapKV (Li et al., 2024)

**Core mechanism**: Prefill-time eviction. Uses the attention pattern from the last α tokens of the prompt (typically the instruction/question) to identify which context tokens are most relevant to the query. Retains top-K tokens by this attention score; discards the rest before decode begins.

**Novelty**: Observation-driven: keep what the model's own question-processing actually attends to. Applied at prefill so there is zero per-step overhead.

**What it misses**: Entirely inapplicable to long *generation* traces — SnapKV cannot know at prefill time which tokens a model will need 15,000 decode steps later. It solves the long-input problem, not the long-output problem. Also the selection signal (last α prompt tokens) may be far removed from what the model needs during synthesis steps much later in generation.

**Relevance**: Upstream baseline for long-input tasks; not directly competing with our work.

---

### 2.4 PyramidKV (Cai et al., 2024)

**Core mechanism**: Non-uniform layer-wise cache budget based on **pyramidal information funneling**: lower transformer layers have dense, broad attention (need more KV budget); upper layers focus sharply on a few tokens (need less). Arithmetic allocation: lower layers get more, upper layers get fewer.

**Novelty**: First principled characterization of cross-layer attention behavior differences as a basis for non-uniform budget allocation. The layer-wise insight is architecture-grounded, not empirically calibrated per-model.

**What it misses**: Designed for prefill-heavy long-context tasks. No investigation on whether the pyramidal pattern holds for long generation (decode-heavy) traces — it may not, because the attention pattern of self-generated reasoning tokens is different from attention over a given long document. No content-aware selection within each layer's budget.

**Relevance**: The layer-wise budget allocation principle is orthogonal to our importance signal and worth incorporating as a free efficiency gain. A future version of our method should allocate different cache budgets per layer using PyramidKV's insight.

---

### 2.5 ChunkKV (NeurIPS 2025)

**Core mechanism**: Uses contiguous 10-token chunks as the atomic compression unit rather than individual tokens. Score each chunk by summing all attention received by its constituent tokens. Retain top-K chunks; discard the rest. Additionally, reuse chunk selection indices across adjacent transformer layers (cross-layer similarity is high), saving ~20% computation.

**Novelty**: Addresses semantic fragmentation — token-level eviction can retain "Eiffel" and discard "Tower" because individual token scores diverged. Chunk-level eviction preserves syntactic coherence (phrases, clauses, entities). Layer-wise index reuse is a systems efficiency innovation orthogonal to the chunk idea.

**What it misses**: Fixed 10-token chunk boundaries have no linguistic grounding — they may straddle phrase or clause boundaries arbitrarily. The paper explicitly acknowledges: "chunk-level methods may underperform token-level methods on global synthesis tasks" requiring recall across the entire context. Still uses attention scores for chunk scoring. No evaluation on decode-heavy reasoning traces.

**Relevance**: The chunk-level eviction unit principle is directly applicable to our method. Instead of scoring individual tokens by hidden-state variance, we should score contiguous segments and evict/retain whole chunks. This is a concrete improvement that sidesteps the fragmentation problem without requiring semantic segmentation infrastructure.

---

### 2.6 RaaS — Reasoning-Aware Attention Sparsity (Hu et al., 2025)

**Core mechanism**: Empirically identifies two token categories in reasoning decode traces:
- **Milestone tokens**: Tokens heavily attended during an intermediate reasoning step, then permanently irrelevant once that step concludes. Premature eviction forces the model into repetition loops.
- **Phoenix tokens**: Tokens that dip to low attention for long stretches then surge back. These nearly always live in the **prefill** (the original user query), not the generated trace.

The algorithm: LRU eviction for decode-stage tokens (a token's "timestamp" is refreshed whenever it falls in the top-50% of attention scores at any step); unconditional preservation of all prefill tokens.

**Novelty**: First paper to characterize milestone vs. phoenix as distinct empirical categories in reasoning traces. The LRU timestamp approach handles the temporal importance pattern that cumulative attention (H2O) cannot. Achieves O(L) memory *and* O(L) time complexity simultaneously — prior retrieval methods (Quest) achieve both but keep all tokens in memory.

**What it misses**:
- No semantic signal beyond attention scores
- No forward-looking capacity: cannot predict which currently-attended milestone token is about to become permanently irrelevant vs. which will continue to be needed
- Eviction is permanent: no recovery if a token is evicted prematurely
- Long-prefill failure mode: if the user prompt is long, prefill preservation may consume the entire budget
- Tested only on four models and three math benchmarks; no analysis of non-math reasoning

**Relevance**: Direct baseline. The milestone/phoenix characterization is important empirical grounding for why attention-based approaches are unstable in reasoning traces. Demonstrates that even a sophisticated attention-based eviction strategy (LRU timestamps) requires careful engineering to avoid loop failures. Our hidden-state variance signal could potentially detect milestone tokens *before* their attention drops — a strictly more predictive approach.

---

### 2.7 ThinKV (ICLR 2026, Oral)

**Core mechanism**: Classifies reasoning trace tokens into three thought types via KDE on attention sparsity from 4 layers, refreshed every 128 decode steps:
- **Reasoning (R)**: systematic deduction, moderate sparsity, highest importance
- **Execution (E)**: calculation/code, low sparsity (dense attention)
- **Transition (T)**: uncertainty markers, backtracking, highest sparsity, lowest importance

Two compression operations:
- **Think Before you Quantize (TBQ)**: R→FP8, E→NVFP4, T→ternary (2-bit). Mean ~3.4 bits.
- **Think Before you Evict (TBE)**: Progressive retention schedule {64, 32, 16, 8, 4}. Within each segment, K-means on post-RoPE keys selects representatives. In-place eviction via PagedAttention block table extensions (no compaction overhead).

**Novelty**: First work to classify reasoning thought types from attention sparsity alone (no keyword matching). Hybrid quantization+eviction at thought-segment granularity. The counterfactual importance ordering (R >> E >> T) is a direct empirical contribution. PagedAttention kernel extension for in-place eviction is a concrete systems contribution. Near-lossless at <5% KV cache retention, 5.8x throughput over SOTA.

**What it misses**:
- Hard floor: minimum 4 tokens per segment must be retained; complete eviction of any segment causes reasoning loops (same failure mode as RaaS)
- Classification signal is attention sparsity (a property *of* attention), not hidden states — it cannot detect semantic transitions, only statistical properties of the attention distribution
- Does not detect the moment of insight — classifies thought *type* retroactively, after the segment has been generated
- Tested only on distilled/RLVR-trained models; unknown behavior on other architectures
- Aggressive quantization alone inflates generation length by 5.1x; hybrid approach is required but adds complexity
- No analysis of how thought type distribution shifts across difficulty levels or problem domains

**Relevance**: This is the primary baseline to beat. SOTA on reasoning-model KV compression. The thought-type classification framework is the closest existing work to our segment-based approach. The key differentiator: ThinKV uses attention sparsity as its classification signal; our approach proposes hidden-state variance, which is a fundamentally different signal with different failure modes and different predictive capacity.

---

### 2.8 FreeKV (ICLR 2026)

**Core mechanism**: A KV *retrieval* method (keeps all tokens in memory, selects which to bring to GPU per step). Key insight: adjacent decode steps have >0.84 query vector similarity, so the set of important tokens barely changes step-to-step. Reuse the retrieved KV set from the previous step instead of recomputing selection. Correction mechanism for low-similarity steps. Hardware-aware dual memory layout (NHD for GPU transposes, HND for CPU-GPU page transfers) + double-buffered streamed recall.

**Novelty**: Speculative retrieval off the critical path; algorithm-system co-design for efficient CPU-GPU transfers. 13x speedup over prior retrieval methods. O(B) GPU memory regardless of context length.

**What it misses**:
- Retrieval, not eviction: all tokens still live in CPU memory — no reduction in total memory, only in GPU memory
- No reasoning awareness for selection priority: token selection is based on query similarity, not semantic importance
- Query similarity assumption may break down during reasoning transitions where the model's focus shifts sharply
- Page-wise granularity "less effective for small budgets" (explicitly stated)
- Not evaluated on reasoning models or long generation traces

**Relevance**: Demonstrates the ceiling of reasoning-agnostic retrieval. A unified system combining FreeKV's retrieval efficiency with reasoning-aware priority scoring (our signal) would outperform both independently — this is Gap 2 (see Section 4).

---

### 2.9 SideQuest (Kariyappa and Suh, 2026)

**Core mechanism**: Agentic multi-turn tasks accumulate tool responses (retrieved documents) in the KV cache. Every K=4 turns, a parallel generation thread is prompted with "Memory management mode", causing the fine-tuned model to output structured JSON identifying which retrieved documents (cursors) are stale vs. still needed. Stale cursors are deleted from the KV cache. The management thread's tokens are then discarded. Fine-tuned on 215 traces with hindsight-annotated last-use indices + logit distillation (λ=500).

**Novelty**: First work where the model reasons about its own context to make cache eviction decisions. Parallel-thread design prevents management tokens from contaminating the main context. Hindsight supervision learns globally optimal eviction policy rather than greedy per-step heuristics. 56–65% token reduction, +83.9% throughput, ~2% accuracy loss.

**What it misses**:
- Scope restricted to tool responses (retrieved documents): the model's own CoT reasoning tokens are completely untouched and grow without bound
- For our use case (single-turn reasoning without tool calls), SideQuest provides no compression at all
- Requires fine-tuning; cursor-indexed structure assumed; small training set (215 traces)
- Eviction is permanent; no retrieval for non-monotonic token utility

**Relevance**: Demonstrates the ceiling of model-driven semantic cache management for tool responses. Gap 3 (Section 4) extends this idea to the model's own CoT tokens — a significantly harder problem since CoT tokens lack explicit cursor identifiers.

---

### 2.10 KVQuant (Hooper et al., NeurIPS 2024)

**Core mechanism**: Quantization-based compression for KV cache. Four innovations: per-channel key quantization (aligns with activation outlier structure), pre-RoPE key quantization (applies quantization before rotary embeddings to avoid RoPE distortion), non-uniform per-layer sensitivity-weighted datatypes, per-vector dense-and-sparse quantization for outlier values.

**Novelty**: First systematic treatment of KV quantization's unique challenges vs. weight quantization, especially the RoPE interaction. Enables 1M context on a single A100 at <0.1 perplexity degradation.

**Relevance**: Quantization is orthogonal and complementary to eviction. Our method determines *which* tokens to retain; KVQuant determines *at what precision* to retain them. ThinKV already combines these (TBQ). A complete system would use both.

---

### 2.11 MiniKV (arXiv 2411)

**Core mechanism**: Layer-discriminative 2-bit KV quantization. Different transformer layers get different quantization precision (not all layers need the same), with custom CUDA kernels compatible with FlashAttention.

**Novelty**: Viable 2-bit KV caching with 86% compression, 98.5% accuracy recovery. Layer-wise precision mirrors PyramidKV's layer-wise budget insight but applied to quantization bits.

**Relevance**: Edge/memory-constrained deployment. Complementary to eviction; may be relevant if our target deployment includes memory-constrained hardware.

---

## 3. Gaps in Existing Literature

The following gaps are identified in the current landscape. They are ordered from most tractable (near-term research contribution) to most speculative (longer-term or separate research threads).

---

### Gap A: Hidden-State Variance as Importance Signal

**Status**: Core project hypothesis, unvalidated.

**What exists**: Every existing method — ThinKV, RaaS, H2O, PyramidKV, ChunkKV, FreeKV — uses attention scores or attention sparsity as its importance signal. Signals derived from hidden-state or KV-vector representational change have not been evaluated as a KV cache importance proxy in any published work.

**The argument**: Representational change signals capture *how much the model's internal state shifts* at each token. High-change positions correspond to tokens where the model is processing something new or consequential (insights, conclusions, transitions). Low-change positions correspond to fluent, predictable filler. This is semantically grounded in a way that attention scores are not.

**The risk**: These signals may correlate more with syntactic/punctuation boundaries than semantic importance. Periods and newlines produce large representational resets. Pre-RoPE vs. post-RoPE computation matters. The proxy available at decode time (KV-vector signals) may lose signal compared to full hidden-state access. The best signal variant is unknown — see Section 3.1 for the ablation plan.

**Validation test**: Run decode at extreme cache pressure (cache_size=32). Compare: which tokens does semantic eviction retain vs. attention eviction? Do semantic-retained tokens correspond to identifiable insights/conclusions? Does accuracy hold better? Repeat across all signal variants in Section 3.1.

---

### Gap B: The Two-Signal Eviction Framework (Hybrid Semantic + Structural)

**Status**: Partially implemented in `src/eviction.py` as `semantic_alpha`.

**What exists**: RaaS preserves all prefill tokens (structural) + LRU for decode tokens (attention-based). ThinKV classifies by thought type (attention-sparsity-based) + quantization + eviction. No method combines hidden-state variance with structural signals (recency, attention sinks, prefill preservation).

**The argument**: Optimal eviction likely requires both signals:
- Structural signals: always keep the first few tokens (attention sinks), always keep prefill tokens (phoenix pattern confirmed by RaaS), always keep very recent tokens (recent context window)
- Semantic signal: within the non-structural portion, keep tokens with high hidden-state variance (insight positions)

**The current implementation**: `SemanticEviction` blends these at a fixed ratio (`semantic_alpha=0.5`). The ratio is untested and the blend function is naive. A proper ablation would sweep `semantic_alpha` from 0 (pure attention) to 1 (pure semantic), with structural tokens hardcoded in.

---

### Gap C: Chunk-Level Semantic Eviction

**Status**: Not yet implemented.

**What ChunkKV showed**: Token-level eviction destroys syntactic coherence. Chunk-level eviction preserves it. This applies equally whether the scoring signal is attention (ChunkKV) or hidden-state variance (ours).

**The extension**: Score contiguous chunks of 8–16 tokens by their *mean* hidden-state variance across member tokens. Evict/retain at chunk granularity. This combines our novel signal with ChunkKV's structural insight.

**Why this matters**: For reasoning traces, the meaningful unit of "an insight" or "a reasoning step" is a multi-token span, not a single token. A single token with high variance may be a punctuation mark; a span with consistently elevated variance is more likely to be a genuine insight.

---

### Gap D: Layer-Wise Budget Allocation for Decode-Heavy Traces

**Status**: Not yet implemented.

**What PyramidKV showed**: Lower transformer layers need more KV budget (dense attention); upper layers need less (sparse, selective attention). This was shown for long-input tasks.

**The open question**: Does the pyramidal pattern hold for long-*generation* reasoning traces? If so, applying PyramidKV's allocation formula on top of our importance scoring is a free orthogonal gain.

**What to do**: Run an attention sparsity analysis across layers on actual DeepSeek-R1-Distill trace generation. Plot attention entropy per layer across decode steps. If the pyramid pattern holds, adopt the arithmetic allocation formula.

---

### Gap E: Dead-End Branch Detection (Predictive, Not Post-Hoc)

**Status**: Open research problem. Most novel of the identified gaps. Likely a separate paper.

**What exists**: ThinKV's Transition category identifies backtracking *after* the backtracking tokens have been generated. RaaS's milestone tokens age out *after* attention to them drops. Both are backward-looking. No method predicts that a currently-active reasoning branch will be abandoned *before* the abandonment marker appears.

**The problem**: In DeepSeek-R1 traces, the pattern "Wait, that approach doesn't work. Let me try..." often follows thousands of tokens of exploration that will never be referenced again. These tokens consume cache until a heuristic eventually evicts them. If we could detect that the model was *heading toward* a pivot before generating "Wait...", we could preemptively compress that branch.

**Candidate signals**:
- **Attention entropy increase**: rising attention entropy (more diffuse attention over many tokens) may precede backtracking, as the model becomes uncertain about which prior context to reference
- **Hidden-state variance spike pattern**: an unusually large hidden-state shift followed by rapid reversion to low variance might signal a failed hypothesis test
- **KV-key clustering divergence**: if the current query vector is becoming dissimilar to recent key vectors (FreeKV's inter-step similarity drops), the model may be pivoting

**Why it's hard**: Requires a predictive model trained on labeled trace data (knowing which spans were ultimately abandoned). This is a supervised learning problem that presupposes a large corpus of reasoning traces with span-level annotation.

**Scope decision**: This is the most scientifically interesting gap but the least tractable. It should be named and motivated as future work in the primary paper rather than attempted in the first experiment phase.

---

### Gap F: Reasoning-Aware Eviction + Recallable Offloading (Unified Framework)

**Status**: Architectural proposal. Strong candidate for primary paper system contribution.

**What exists**: The literature is split into two non-overlapping camps:
- **Eviction methods** (ThinKV, RaaS, H2O): permanently destroy tokens. O(L) memory, fast, but irreversible.
- **Retrieval methods** (FreeKV, Quest, ShadowKV): keep all tokens in CPU memory, retrieve relevant subset per step. Lossless but O(N) total memory.

No method combines both: reasoning-aware eviction *with* a recoverability mechanism for tokens evicted prematurely.

**Why this matters for reasoning**: Reasoning traces have non-monotonic attention patterns. A branch that looks like a dead end at token 8,000 may become relevant again at token 15,000 if the model's alternative approach fails and it loops back. Permanent eviction based on current signals is therefore risky in a way it isn't for document summarization, where you rarely need a sentence back after discarding it.

**Proposed architecture (three-tier memory)**:
```
Hot  (GPU):  actively needed tokens, full precision, ~L tokens
Warm (CPU):  recently evicted but recoverable, quantized (INT4), ~4L tokens
Cold (evicted): permanently dropped, no recovery
```

Token placement is driven by the reasoning-type classification:
- **R tokens** (reasoning/insight): hot by default
- **E tokens** (execution): potentially warm after segment closes
- **T tokens** (transition/exploration): cold after segment closes, except for explicitly marked backtracking points which go warm

Recovery mechanism: if the model's attention shifts back to a warm token (query vector is similar to a warm key), trigger a recall before the step completes. This is FreeKV's speculative retrieval, but applied only over the reasoning-aware warm tier rather than all N tokens — a strictly smaller and more relevant search space.

**Why it's better than either approach alone**:
- Better than pure eviction (ThinKV): warm tier provides a safety net for non-monotonic attention patterns
- Better than pure retrieval (FreeKV): reasoning-aware placement means the warm tier is semantically filtered, reducing recall compute and improving precision

---

### Gap G: Model-Driven Cache Management for CoT Tokens

**Status**: Research direction. Harder than Gap F; likely a separate paper.

**What SideQuest showed**: A fine-tuned model can reason about which retrieved documents in its context are stale, enabling semantically correct eviction with ~2% accuracy cost. This outperforms all attention heuristics on non-monotonic utility tasks.

**The gap**: SideQuest compresses only tool responses (discrete cursor-identified documents). The model's own chain-of-thought tokens — which in reasoning models are the dominant memory consumer — are completely untouched. For single-turn reasoning (no tool calls), SideQuest provides zero compression.

**The extension**: Fine-tune a model to reason about which of its own past CoT spans are now closed/settled vs. still active. The model would output: "The reasoning at positions 2000–3500 explored approach A; that approach was abandoned. Those tokens can be compressed." This generalizes SideQuest to the reasoning CoT use case.

**Why it's hard**:
- Training signal problem: SideQuest's last-use indices are well-defined for discrete cursor-identified documents. For unstructured CoT, "last use" of a reasoning span is harder to define — the model may implicitly rely on a conclusion without explicit attention to the span tokens.
- Bootstrapping problem: need correct, complete reasoning traces to construct hindsight supervision; incorrect traces (wrong final answers) cannot be labeled.
- Contamination risk: if the management thread tokens leak into the main generation context, they degrade the primary reasoning task.

**A lighter version**: Instead of full model-driven reasoning, use hidden-state variance (our signal) as a lightweight proxy for "active computation is happening here." This is cheaper than a fine-tuned auxiliary thread and doesn't require training data. This is exactly what our current project builds — it's the lightweight version of Gap G.

---

### Gap H: Memory-Constrained / Edge Deployment for Reasoning Models

**Status**: Largely unaddressed in current literature.

**What exists**: KVQuant and MiniKV address quantization-based compression. KVSwap addresses disk offloading for mobile/embedded (unified memory devices). Persistent Q4 KV (arXiv 2603) addresses multi-agent context switching on Apple Silicon.

**The specific gap**: All edge/memory-constrained methods address pressure from long *inputs*. None address the case where memory pressure comes from long *generation* — reasoning model traces that grow during decode. A mobile device running a 7B reasoning model for 20k decode steps faces fundamentally different memory dynamics than a server handling a 128k-token RAG query.

**What's needed**: Methods that can cap reasoning trace memory growth while maintaining accuracy, specifically targeting:
- Devices with <16GB unified memory (Apple M4 Pro, NVIDIA Jetson Orin)
- No reliable PCIe bandwidth for CPU offloading
- Single-token decode latency sensitive (no batch opportunity)

This is a deployment target, not a research contribution on its own, but it's a concrete application domain where our method could claim impact.

---

## 3.1 The Hidden-State Variance Signal — Full Ablation Design Space

Before any eviction comparison is meaningful, we must determine *which variant* of the variance signal to use. There are five independent dimensions of choice. Each should be ablated independently, then the best combination selected for the main method.

### Dimension 1: Signal Type

What quantity captures token importance?

Two families: **residual-stream signals** (what the model *learned* at this token) and **attention signals** (what this token *attended to*, or was attended by). Our primary hypothesis is that residual-stream signals are better proxies for *semantic importance* in long CoT traces. Attention signals are the current SOTA baseline and must be included for a fair ablation.

**Residual-stream signals** (collected in `collect_traces.py`; post-hoc forward pass for hs_* signals):

| Variant | Formula | Intuition | Risk |
|---|---|---|---|
| **L2 hidden-state diff** | `‖h_t − h_{t−1}‖₂` | How much the residual stream changed | Post-hoc pass only; expensive at decode time |
| **Cosine distance (hidden)** | `1 − cos(h_t, h_{t−1})` | Directional change, magnitude-invariant | Same cost; may miss magnitude signals |
| **HS L2 norm** | `‖h_t‖₂` | Absolute magnitude of the representation | Captures "energy" not change; possible proxy for token richness |
| **KV-key variance (head_dim)** | `k_t.var(dim=-1)` | How spread the key vector is across channels | Post-RoPE (mixes positional signal; see Dim 2) |
| **KV-key L2 norm** | `‖k_t‖₂` | Magnitude of the key vector | Representational energy, not change; different failure modes |
| **KV-value variance (head_dim)** | `v_t.var(dim=-1)` | How spread the value vector is | Values carry "content to retrieve"; may be cleaner semantic signal than keys |
| **Cross-head key variance** | `var({mean(k_t^h) : h in heads})` | Disagreement across attention heads about how to index this token | High = token is interpreted differently by different heads |

**Attention signals** (require `--force_eager_attn`; FlashAttention cannot materialise these):

| Variant | Formula | Intuition | Used by |
|---|---|---|---|
| **H2O cumulative attention** | `score[j] = sum_{t>j} a[t,j]` | How much future tokens attended back to token j | H2O, SnapKV, our baseline |
| **Attention entropy** | `H_t = -sum_j a[t,j] log a[t,j]` | How focused vs. diffuse the model was at step t. Low H = sharp/confident ("Thinking" in ThinKV); high H = diffuse ("Rambling") | ThinKV (R/E/T classifier), our attention baseline |

Note: ThinKV uses **attention entropy/sparsity**, not activation sparsity. Activation sparsity (Deja Vu, PowerInfer) measures how many FFN neurons fire near-zero — a completely separate concept used for compute reduction, not KV cache management.

**Hypothesis ordering (prior expectation)**: L2 hidden-state diff > cosine distance ≈ value variance > key variance > key L2 norm > attention entropy ≈ H2O. The key question is whether *any* residual-stream signal beats the attention signals. This is unproven.

**Practical constraints**:
- `kv_*` signals: free — K/V tensors are already in HBM; reading them is negligible. Fully online (per decode step) and FA2-compatible.
- `hs_*` signals: one post-hoc forward pass per trace, FA2-compatible, O(n) peak memory. Computed after generation (offline/batch). Better than attention signals on memory and FA2 compatibility; whether to extend to real-time use is a question for after signal ablation confirms usefulness.
- Attention signals (`h2o_attn`, `attn_entropy`): require eager attention (no FlashAttention), O(n²) peak VRAM. Per-step online, but at significant memory cost and with FlashAttention disabled. ThinKV is also not fully online — it accumulates entropy statistics over a window before classifying a thought segment.

### Dimension 2: RoPE Interaction

Rotary position embeddings (RoPE) apply a position-dependent rotation to key (and sometimes query) vectors before attention. This rotation is orthogonal to semantic content — it encodes position, not meaning. Computing KV-key variance *after* RoPE mixes semantic signal with positional signal.

| Variant | When computed | Issue |
|---|---|---|
| **Post-RoPE** (current) | After rotary embedding applied | Variance at token position t is inflated by positional distance from position 0 regardless of content |
| **Pre-RoPE** | Before rotary embedding applied | Cleaner semantic signal; requires hooking an earlier computation point |

**Hypothesis**: Pre-RoPE key variance will show less position-correlated noise than post-RoPE. Tokens at large positions will not appear spuriously important.

**Implementation note**: In HuggingFace transformers, keys are typically returned post-RoPE from `past_key_values`. Pre-RoPE access requires a forward hook on the attention layer before the `apply_rotary_pos_emb` call.

### Dimension 3: Layer Aggregation

Which transformer layers should we read the signal from?

| Variant | Formula | Intuition |
|---|---|---|
| **Last layer only** | `signal(layer L)` | Highest-level representation; most abstract |
| **Mean across all layers** | `mean over l of signal(layer l)` | Robust, but lower layers add noise |
| **Upper-layer weighted mean** | `sum_l (l/L) * signal(layer l)` | PyramidKV insight: upper layers are more semantically selective; weight them higher |
| **Optimal single layer** | argmax layer by correlation with importance | Find empirically on validation set |
| **Multi-layer variance** | `var over l of signal(layer l)` | Tokens whose representation changes across *depth* are more complex/important |

**Hypothesis**: Upper-layer weighted mean will outperform uniform mean; last layer alone may miss information consolidated at intermediate layers. Optimal single layer will likely be in the upper third.

### Dimension 4: Temporal / Sequence Aggregation

At which temporal scale should variance be measured?

| Variant | Formula | Intuition |
|---|---|---|
| **Single-step snapshot** (current) | `signal(t)` | Immediate change; noisy |
| **Exponential decay running mean** | `score[t] = α·score[t−1] + (1−α)·signal[t]` | Smooths noise; tokens with consistently elevated variance score higher |
| **Max over sliding window** | `max(signal[t−w:t])` | Captures any spike in window; less decay than EMA |
| **Cumulative sum** | `sum(signal[0:t])` | Analogous to H2O's cumulative attention; tokens with repeated high-variance steps stay important |

**Hypothesis**: EMA with α≈0.9 will outperform single-step snapshot by reducing noise; cumulative sum may over-retain early tokens (same failure mode as H2O in a different form).

### Dimension 5: Multi-Head Aggregation

How do we aggregate signals across the H attention heads?

| Variant | Formula | Intuition |
|---|---|---|
| **Mean across heads** (current) | `mean_h(signal(head h))` | Average behavior |
| **Max across heads** | `max_h(signal(head h))` | Most active head determines importance |
| **Cross-head variance** | `var_h(signal(head h))` | High = heads disagree strongly; potentially signals a structurally interesting token |

**Hypothesis**: Max across heads will outperform mean for detecting rare high-signal tokens that only activate in specific heads. Cross-head variance is the most novel and highest-risk variant.

### Ablation Plan

The full design space is 8 × 2 × 5 × 4 × 3 = 960 combinations — far too many to sweep exhaustively. The ablation strategy is:

1. **Fix a reasonable default**: post-RoPE, KV-key variance, mean across layers, single-step snapshot, mean across heads
2. **Sweep Dimension 1** (signal type, 8 variants including attention baselines): identify best signal type. If attention entropy or H2O wins, the core hypothesis is wrong — pivot.
3. **Sweep Dimension 2** (RoPE, 2 variants) with best signal type fixed: pre vs. post
4. **Sweep Dimension 3** (layer aggregation, 5 variants) with best of 1+2
5. **Sweep Dimension 4** (temporal, 4 variants) with best of 1+2+3
6. **Sweep Dimension 5** (multi-head, 3 variants) with best of 1+2+3+4

This reduces to ~8+2+5+4+3 = 22 experiments for a sequential ablation. Each experiment: run MATH-500 (100-sample subset) at cache_size=128 with DeepSeek-R1-Distill-LLaMA-8B. Metric: Spearman correlation with counterfactual importance labels.

**Note**: Dim 1 is the most critical experiment. If KV-key variance is not competitive with full hidden-state L2 diff, we need hidden-state hooks in the decode loop (higher latency). If attention signals win outright, the core hypothesis needs revision.

---

## 4. Paper Framing — Three Potential Contributions

The three "new gaps" introduced in discussion (E, F, G above) can each be framed as independent papers. Here is how they relate:

| Paper | Core Claim | Primary Gap | Key Signal | Key Baseline |
|---|---|---|---|---|
| **Paper 1 (this project)** | Hidden-state variance is a better importance signal than attention for reasoning CoT eviction | Gaps A, B, C, D | Hidden-state / KV-vector variance | ThinKV, RaaS, H2O |
| **Paper 2** | Reasoning-aware tiered eviction with recallable offloading outperforms pure eviction and pure retrieval | Gap F | Same signal, new architecture | ThinKV + FreeKV combined |
| **Paper 3** | Models can learn to reason about which of their own CoT spans are stale | Gap G | Model's semantic self-assessment | SideQuest extension |

Dead-end detection (Gap E) is likely a 4th paper, highly novel, high difficulty.

**The first paper must prove Gaps A and B** before Papers 2 and 3 can be built — they all depend on hidden-state variance being a validated signal.

---

## 5. What to Do Forward — Prioritized Execution Plan

### Phase 0: Validate the Core Signal (Before Anything Else)

Everything downstream depends on whether hidden-state variance actually correlates with semantic importance in reasoning traces. This must be answered first.

**Experiment 0.1 — Signal Validation (which variance variant is best?)**:
1. Load DeepSeek-R1-Distill-LLaMA-8B
2. Generate 50–100 traces on MATH-500 problems (verifiable answers)
3. At each token position, record all six signal-type variants (Dimension 1) plus cumulative attention (H2O-style) and ThinKV-style segment label
4. Label tokens as "important" via counterfactual ablation: re-run inference with sliding windows of tokens masked; record which masked windows cause the answer to change. These are ground-truth importance labels.
5. Compute Spearman correlation of each signal variant with importance labels
6. This directly answers: does *any* variance signal outperform H2O cumulative attention? Which variant is best?

This is the single most important experiment. If no variance signal outperforms cumulative attention on importance correlation, the core hypothesis is wrong and the project direction must change.

**Experiment 0.2 — Sequential Ablation Across All Five Dimensions**:
Run the 20-experiment sequential ablation defined in Section 3.1. Fix each best-so-far setting and sweep the next dimension. Use 100-sample MATH-500 subset at cache_size=128 for speed. Identifies the best signal configuration for all downstream experiments.

**Experiment 0.3 — Baseline Calibration**:
1. Implement H2O (cumulative attention) as the proper attention baseline
2. Implement ThinKV's thought classification (KDE on 4 layers, refreshed every 128 steps) as SOTA baseline
3. Establish accuracy vs. cache-size curves for both on MATH-500 + AIME at cache sizes {32, 64, 128, 256, 512, 1024}
4. These curves define the target to beat in Phase 1

### Phase 1: Core Method Development

Once Experiment 0.1 validates the signal:

1. **Chunk-level eviction**: Score 10-token chunks by mean KV-vector variance; evict at chunk granularity (Gap C)
2. **Structural + semantic hybrid**: Hardcode: always keep first 4 tokens, always keep prefill, always keep last 32 tokens; apply variance-based scoring to the remaining budget (Gap B)
3. **`semantic_alpha` ablation**: Sweep 0.0 to 1.0 in 0.2 increments; find optimal blend on validation set (Gap B calibration)
4. **Layer-wise analysis**: Plot attention entropy per layer on DeepSeek-R1 traces; if pyramidal, adopt PyramidKV's allocation formula (Gap D)

### Phase 2: Benchmark Suite

Baselines to beat (in order of importance):
1. **ThinKV** — primary target, SOTA on reasoning models
2. **RaaS** — reasoning-aware attention baseline
3. **H2O** — standard cumulative attention
4. **PyramidKV** — layer-wise allocation
5. **ChunkKV** — chunk-level attention

Benchmarks:
- **MATH-500** (primary — graded math, verifiable answers)
- **AIME** (high difficulty, long traces)
- **HotpotQA** (multi-hop, tests non-monotonic recall)
- **LiveCodeBench** (code generation, long execution traces)

Metrics:
- Accuracy at cache budget K ∈ {32, 64, 128, 256, 512, 1024} tokens
- Peak GPU memory consumption (per-example, not cumulative)
- Tokens generated before correct answer (generation efficiency)
- Eviction recall rate (fraction of "important" tokens retained)

### Phase 3: System Contributions

Once the method is validated:

1. **Implement warm tier (Gap F prototype)**: Add INT4 CPU offloading for recently evicted tokens; implement recall trigger on query similarity
2. **Run ablation**: Hot-only vs. Hot+Warm at same total memory budget
3. **Layer-wise budget**: Combine with Gap D analysis

---

## 6. Current Implementation Status

| Component | Status | Issues |
|---|---|---|
| `src/eviction.py` — AttentionBasedEviction | Implemented, single-step attention | Needs H2O upgrade (cumulative) |
| `src/eviction.py` — SemanticEviction | Implemented, KV-vector variance proxy | Not validated against real traces |
| `src/eviction.py` — evict_past_key_values() | Implemented, both classes | Works, needs per-layer budget option |
| `scripts/poc_harness.py` — step-by-step loop | Fixed, eviction triggers | Needs per-example memory reset |
| `scripts/poc_harness.py` — mock results | Removed | OK |
| `src/data_collection.py` — segment classifier | Consolidated, word-boundary regex | Synthetic traces only; needs real data |
| `scripts/analyze_traces.py` | Working on synthetic data | Needs real DeepSeek-R1 traces |
| `scripts/visualize.py` — memory plot | Replaced with theoretical plot | Correct, clearly labeled |
| H2O baseline | Not implemented | Needed before any comparison |
| ThinKV baseline | Not implemented | Needed to claim SOTA comparison |
| Real data pipeline | Not implemented | Critical blocker |

---

## 7. Further Optimizations

**Value-vector variance over key-vector variance**: Values carry the "content to retrieve" while keys carry the "address to match against queries." If the goal is to retain tokens with semantically rich content, value-vector variance (`v.var(dim=-1)`) may be a more direct signal than key-vector variance. This should be an explicit ablation variant (see Section 3.1, Dimension 1).

**Pre-RoPE variance**: Compute KV-vector variance before applying rotary positional embeddings. RoPE adds a position-dependent rotation that inflates variance for tokens at large positions regardless of content. Pre-RoPE variance is a cleaner semantic signal. Requires hooking into the attention layer before `apply_rotary_pos_emb` rather than reading from `past_key_values` directly.

**Layer-wise variance weighting**: Rather than averaging KV-vector variance across all layers, weight by layer index (PyramidKV insight: upper layers are more semantically selective). Upper-layer variance may be a more reliable importance signal.

**Cross-head key variance**: Compute the variance of key vector means across attention heads. A token where heads disagree sharply on how to index it may carry structural or semantic significance not captured by within-head signal.

**Exponential decay on importance scores**: Maintain a running importance score per token: `score[t] = decay * score[t-1] + (1 - decay) * signal[t]`. Smooths single-step noise; tokens with consistently elevated signal score higher than one-off spikes.

**Variance-gated quantization**: Rather than a hard evict/retain decision, apply graded quantization: high-variance tokens stay FP16, medium-variance go INT8, low-variance go INT4, very-low-variance get evicted. This is ThinKV's TBQ insight applied to our variance signal rather than thought-type classification.

**Batch eviction scheduling**: Evict in batches (when cache reaches 1.5L, evict down to 0.5L) rather than at every step. Reduces topK overhead. Especially useful for the warm-tier design where eviction decisions interact with recall triggers.

**Contrastive validation**: For each reasoning trace, generate two versions — one producing a correct answer, one a wrong answer. Tokens with systematically higher variance in the correct-answer trace are likely load-bearing. Use as supervision for which signal variant to trust.

---

## 8. Summary of Differentiation

| Dimension | This Project | ThinKV (closest) | RaaS | FreeKV |
|---|---|---|---|---|
| Target | Long-generation reasoning traces | Long-generation reasoning | Long-generation reasoning | Long-input retrieval |
| Importance signal | Hidden-state variance (proposed) | Attention sparsity | Attention LRU timestamps | Query vector similarity |
| Predictive capacity | Hidden states may precede transitions | Post-hoc classification | Post-hoc LRU aging | Inter-step extrapolation |
| Eviction unit | Chunk-level (proposed) | Segment-level | Page-level | Page-level |
| Recoverability | Warm tier (proposed) | Permanent eviction | Permanent eviction | Full retrieval (all tokens) |
| Training required | No | No | No | No |
| Layer-wise budget | Yes (proposed, Gap D) | No | No | No |
| Dead-end prediction | Future work (Gap E) | Not addressed | Not addressed | Not addressed |
