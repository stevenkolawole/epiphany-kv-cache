# Epiphany-Aware KV Cache Paper — Agent Writing Prompt

---

## How this session works

Two agents run sequentially. Both read and update the same TODO.md.

1. **GitHub Copilot** runs first — works through all phases, checks off tasks,
   logs confidence and blockers at each checkpoint
2. **Claude Code** runs second — reads TODO.md first, independently verifies,
   critiques, and fills gaps. Adds a verdict beneath each Copilot checkpoint.

Neither agent deletes the other's notes. They accumulate.

**First action for both agents**: read TODO.md and understand the current state
before doing anything else.

---

## Who you are and what you are doing

You are a rigorous research writing agent working on an EMNLP 2026 Industry
Track paper (deadline: June 16, 2026):

**Title**: Epiphany-Aware KV Cache Eviction: Why Hidden States Outperform
Attention

You are not making research decisions. You are writing from the evidence as it
exists. Your job is to produce paper prose that is:
- Grounded entirely in the experimental record
- Calibrated against what the best papers in this subfield actually look like
- Honest about what is solid, what is fragile, and what is pending
- Pitched correctly for EMNLP Industry Track — engineering wins and deployment
  practicality matter here as much as algorithmic novelty

**Golden rule**: Do not invent results. Do not smooth over methodological
discoveries by omitting them. When evidence is fragile or pending, write
[NEEDS ROBUSTNESS: reason] and move on.

---

## The paper in one paragraph (your anchor)

Reasoning models (DeepSeek-R1 class) generate 10k–100k token traces, creating
KV caches that exceed GPU memory. Existing eviction methods use attention weights
as importance proxies — but attention weight is semantically noisy in long
reasoning traces and architecturally incompatible with FlashAttention 2, which
never materialises the attention matrix. We show that hidden-state variance at
specific mid-layers (Band A, layers 7–13) and its complement (Band B, layers
18–25) predicts token importance better than any attention-based signal, and can
be computed entirely within a FA2-compatible forward pass with negligible
overhead. On MATH-500 at a 4096-token cache budget, our detrended HS method
reaches 72% accuracy, beating ThinKV (71%) and H2O (67%), with a ceiling of
75%. On AIME-2024 at 8192 tokens, our FA2-compatible lag_kv method reaches 37%
vs 33% for the best attention-requiring baseline. H2O collapses to empty
generations on 93/100 MATH-500 problems at cache=1024 — the sharpest
illustration of why attention is the wrong signal in this regime. The method
requires no training, no classifiers, no attention matrix, and is fully
compatible with standard production inference stacks (vLLM, TGI, SGLang).

---

## Source of truth — what to read and how

The repo contains extensive documentation. Your job is to mine it for
paper-ready content, not summarise it. Treat these files as draft material
to be refined into LaTeX prose.

**Read all of these before writing anything**:

### Primary source files (in repo)
- `README.md` — canonical project overview; Phase 0B key findings; Phase 1
  headline numbers; project structure
- `experiments/research_overview.md` — the scientific core: problem statement,
  epiphany hypothesis, full related work comparison (§2), gap analysis (§3),
  full ablation design space (§3.1), paper framing (§4), execution plan (§5),
  differentiation table (§8), temporal trend problem and mitigation (§9),
  literature survey table (§10)
- `experiments/progress.md` — full chronological experiment log; all Phase 0,
  0B, and Phase 1 results; all methodology fixes and their rationale; Phase 1
  debugging saga; Phase 2 plan
- `experiments/phase0b_ablation_results.md` — full signal ablation results
  with Spearman ρ values, confidence intervals, per-dataset breakdowns
- `experiments/paper_strategy.md` — figure plan, section notes, presentation
  strategy originally targeting NeurIPS — adapt for EMNLP Industry Track
- `experiments/signals_reference.md` — technical reference for all signals
- `src/eviction.py` — all implemented eviction classes; read to understand
  exactly what each method does and how it differs from the paper description
- `scripts/analyze_phase1.py` — result tables and plots; read for exact numbers
- `reports/phase1_plots/` — accuracy vs cache-size PDFs; describe these in
  figure specs

### What the EMNLP Industry Track framing requires (different from NeurIPS)
The paper_strategy.md was written for NeurIPS. Adjust as follows:
- Industry Track values: does this work in production? what's the engineering
  cost? who can deploy this today?
- Lead with the FA2 compatibility claim earlier and more prominently
- The deployment stack compatibility (vLLM, TGI, SGLang) should appear in the
  intro, not just the conclusion
- Ablation depth matters less than result clarity and practical takeaways
- "This is the first decode-time eviction method for reasoning traces that works
  within standard production inference stacks without disabling FlashAttention"
  is an Industry Track contribution statement. Use it.

---

## Critical methodological facts — handle these honestly

These are not weaknesses to hide. They are part of the story. An agent that
papers over them will produce a paper that a reviewer tears apart.

### 1. The temporal trend discovery (Section 9 of research_overview.md)
The Phase 0B aggregate Spearman ρ was driven partly by cross-problem structure,
not within-trace discrimination. The raw combined score (l10−l21) evicts the
WRONG tokens within simple traces because l10 decreases and l21 increases
monotonically with position. This was discovered mid-project. Rolling z-score
detrending (`DetrendendHSVarianceEviction`) was developed as a fix. The
detrended variant is what beats ThinKV in Phase 1.

In the paper: this should appear in the methods section as a methodological
finding, not be buried or omitted. The narrative is: "We discovered that raw
HS variance signals carry a temporal trend that correlates with position rather
than content; we address this with rolling z-score detrending." This is honest
and adds to the paper — it reveals something about how these signals behave.

### 2. AIME n=30 fragility
The AIME-2024 results (30 problems) make a 3-point absolute win for lag_kv
fragile — it is one problem difference. Phase 2 will combine AIME 2024+2025+2026
to n=90. Until then, tag every AIME win claim with [NEEDS ROBUSTNESS: n=30,
combine AIME 2024+2025+2026 in Phase 2].

### 3. kv_val_var correction
Early documentation claimed kv_val_var as "consistently non-negative" and a
"primary online fallback." This was corrected April 13: kv_val_var is NEGATIVE
in math500 (−0.135) and math500_eager (−0.145). Do not use the earlier claim.

### 4. GSM8K layer anatomy divergence
The Band A/B anatomy (l7–l13 positive, l18–l25 negative) holds for competition
math (MATH-500, AIME). For GSM8K (grade-school math), the anatomy shifts:
l0–l7 positive, l10–l30 negative, l31 strongly positive. KV signal sign flips.
attn_entropy sign flips. This is a genuine finding about difficulty-regime
dependency, not a failure. Frame it as: the "epiphany layers" shift by task
difficulty, which suggests these signals are picking up something real about
the structure of reasoning at different cognitive loads.

### 5. FA2 compatibility scope
"FA2-compatible" means the method reads only from past_key_values (already in
HBM) and hidden states (standard output, FA2-compatible). It does NOT mean the
method was benchmarked with FA2 enabled throughout — the Phase 1 setup ran
eager and flash as separate configurations. Be precise about this in the methods.

### 6. Latency vs memory claims
Phase 1 shows speed wins (lag_kv is 2.8× faster than raas at AIME@8192).
Memory savings in the decode regime measured are modest (~3GB at cache=512,
shrinking at larger budgets). The known O(n²)→O(n) FA2 memory advantage for
prefill is real but not measured in these short-prompt benchmarks. Claim what
was measured; note what wasn't.

---

## TODO.md — initialise before doing anything else

Create TODO.md at the repo root with exactly this content:

```markdown
# EpiphanyKV Paper Writing — Shared Agent TODO

**Confidence scale**: HIGH = verified against experimental record, no gaps
| MEDIUM = mostly done, minor uncertainties flagged
| LOW = placeholder or significant gaps remain
| BLOCKED = cannot proceed without human or missing result

Last updated by: [AGENT NAME + timestamp]

---

## PHASE 1 — Literature

- [ ] 1A: Read all repo docs (research_overview, progress, phase0b_results,
      paper_strategy, signals_reference, src/eviction.py, analyze_phase1.py)
      Copilot: 
      Claude Code verdict: 

- [ ] 1B: Read priority external papers from literature survey (§2 and §10 of
      research_overview.md) — fetch from arXiv where available
      Copilot: Papers read in full: 
            Papers abstract-only: 
            Papers inaccessible: 
      Claude Code verdict: 

- [ ] 1C: Web search for any 2025–2026 papers on reasoning-model KV eviction
      not already in the literature survey
      Copilot: New papers found: 
      Claude Code verdict: 

- [ ] 1D: Write LITERATURE_NOTES.md
      Copilot: CONFIDENCE [ ]
      Claude Code verdict: 

### PHASE 1 CHECKPOINT
Copilot notes:
Claude Code notes:

---

## PHASE 2 — Standards

- [ ] 2A: Abstract patterns — study 5–10 from read papers, derive structure
      Copilot: CONFIDENCE [ ]
      Claude Code verdict: 

- [ ] 2B: Introduction patterns — study 5–10, derive structure
      Copilot: CONFIDENCE [ ]
      Claude Code verdict: 

- [ ] 2C: Results section patterns — finding-led vs method-led; table/prose
      Copilot: CONFIDENCE [ ]
      Claude Code verdict: 

- [ ] 2D: Related work patterns — cluster structure, citation density, positioning
      Copilot: CONFIDENCE [ ]
      Claude Code verdict: 

- [ ] 2E: Methods section patterns — especially Industry Track papers on
      KV cache / inference efficiency; deployment environment description
      Copilot: CONFIDENCE [ ]
      Claude Code verdict: 

- [ ] 2F: Write STANDARDS.md
      Copilot: CONFIDENCE [ ]
      Claude Code verdict: 

### PHASE 2 CHECKPOINT
Copilot notes:
Claude Code notes:

---

## PHASE 3 — Write sections fresh

- [ ] 3.1: Related Work → sections/related_work.tex
      Copilot: CONFIDENCE [ ]
      Self-critique done: [ ]
      Key uncertainties: 
      Claude Code verdict: 

- [ ] 3.2: Methods / Experimental Setup → sections/methods.tex
      Copilot: CONFIDENCE [ ]
      Self-critique done: [ ]
      Temporal trend handling: [ ]
      FA2 scope stated precisely: [ ]
      Claude Code verdict: 

- [ ] 3.3: Results → sections/results.tex
      Copilot: CONFIDENCE [ ]
      Self-critique done: [ ]
      Numbers verified against experimental record: [ ]
      NEEDS ROBUSTNESS tags applied where appropriate: [ ]
      Claude Code verdict: 

- [ ] 3.4: Discussion / Conclusion → sections/discussion.tex
      Copilot: CONFIDENCE [ ]
      Self-critique done: [ ]
      Claude Code verdict: 

- [ ] 3.5: Introduction → sections/intro.tex
      Copilot: CONFIDENCE [ ]
      Self-critique done: [ ]
      FA2 compatibility claim in intro: [ ]
      Claude Code verdict: 

- [ ] 3.6: Abstract → sections/abstract.tex
      Copilot: CONFIDENCE [ ]
      Self-critique done: [ ]
      Word count: 
      Claude Code verdict: 

- [ ] 3.7: Figure specifications → sections/figures.md
      Copilot: CONFIDENCE [ ]
      Claude Code verdict: 

- [ ] 3.8: Title alternatives (3–5 with rationale) → sections/titles.md
      Copilot: CONFIDENCE [ ]
      Claude Code verdict: 

- [ ] 3.9: REVISION_LOG.md
      Copilot: CONFIDENCE [ ]
      Claude Code verdict: 

### PHASE 3 CHECKPOINT
Copilot notes:
Claude Code notes:

---

## PHASE 4 — Repo doc synthesis

- [ ] 4A: Re-read all repo docs as source material for refinement
      Copilot: 
      Claude Code verdict: 

- [ ] 4B: Write REPO_CRITIQUE.md — what each doc contributes and where it
      conflicts with or extends Phase 3 sections
      Copilot: CONFIDENCE [ ]
      Claude Code verdict: 

- [ ] 4C: Write SYNTHESIS_PLAN.md — section-by-section merge decisions
      Copilot: CONFIDENCE [ ]
      Claude Code verdict: 

- [ ] 4D: Produce final merged .tex files
      Copilot: CONFIDENCE [ ]
      Claude Code verdict: 

### PHASE 4 CHECKPOINT
Copilot notes:
Claude Code notes:

---

## FINAL STATUS

- [ ] STATUS.md written
      Copilot: 
      Claude Code additions: 

### Section confidence summary
| Section         | Copilot confidence | Claude Code verdict |
|-----------------|-------------------|---------------------|
| Related Work    |                   |                     |
| Methods         |                   |                     |
| Results         |                   |                     |
| Discussion      |                   |                     |
| Introduction    |                   |                     |
| Abstract        |                   |                     |

### Claims requiring Phase 2 robustness runs
(both agents append here)

### Reviewer risks flagged
(both agents append here)

### First actions for Steven in the morning
(both agents append here — Claude Code may revise Copilot's list)
```

---

## PHASE 1 — Literature

### Step 1A: Read all repo documentation

Read in full — do not skim:
- `README.md`
- `experiments/research_overview.md` (all sections, especially §2, §3, §8, §9, §10)
- `experiments/progress.md` (all entries, especially April 13 and April 24)
- `experiments/phase0b_ablation_results.md`
- `experiments/paper_strategy.md`
- `experiments/signals_reference.md`
- `src/eviction.py` (read all class implementations)
- `scripts/analyze_phase1.py`

For each file, note: what paper-ready prose or arguments does this contain?
What numbers are stated? What claims are made with what confidence?

### Step 1B: Read external papers

The literature survey in §2 and the survey table in §10 of `research_overview.md`
describe 12+ papers. Fetch and read in full from arXiv where available:
- ThinKV (ICLR 2026 oral) — primary baseline to beat
- RaaS (Hu et al., 2025)
- H2O (Zhang et al., 2023)
- StreamingLLM (Xiao et al., 2023/2024)
- SnapKV (Li et al., 2024)
- PyramidKV (Cai et al., 2024)
- ChunkKV (NeurIPS 2025)
- FreeKV (ICLR 2026)
- SideQuest (Kariyappa and Suh, 2026)
- KVQuant (Hooper et al., NeurIPS 2024)
- LagKV, AhaKV, CAOTE, LongFlow, VATP (from §10 survey table)
- EAGLE / EAGLE-2 (for HS-as-information-carrier convergent evidence)
- ROME / MEMIT (for mid-layer importance grounding)
- FlashAttention / FlashAttention-2 (Dao et al., for FA2 compatibility claim)

### Step 1C: Search for new papers (2025–2026)

Search for any reasoning-model KV eviction papers published after the repo's
April 2026 literature survey. The field is moving fast.

### Step 1D: Write LITERATURE_NOTES.md

For every paper read:
```
## [Citation key]
- Full title + venue + year
- Core claim in one sentence
- What they did well methodologically
- What they claimed vs what evidence actually supported (be critical)
- Writing quality: what did they do well at the section level?
- Relevance to this paper: where we cite them, how we differ, what it implies
- Any finding that complicates or contradicts ours
- FA2 compatibility: does their method work with FlashAttention? (yes/no/unclear)
```

**→ After 1D: update TODO.md Phase 1 checkpoint.**

---

## PHASE 2 — Standards

Write STANDARDS.md derived entirely from papers you read — not generic advice.

### 2A: Abstracts in this subfield

Study 5–10 abstracts from the papers you read. For EMNLP Industry Track
specifically, look for: how do papers frame engineering contributions alongside
algorithmic ones? How do they handle numbers vs qualitative claims?

Record:
- Length (EMNLP typical: 150–200 words)
- Problem framing: deployment-first or algorithm-first?
- How many results named with numbers?
- Voice and tense conventions
- Final sentence: takeaway, generalisation, or deployment claim?
- One strong and one weak abstract opening sentence with explanation

Write: the exact structural pattern for this paper's abstract.

### 2B: Introductions in this subfield

Study 5–10. Record:
- Paragraph count and purpose of each
- Opening sentence — what does it accomplish?
- When and how contributions appear (numbered? verb-first?)
- How the gap is established for a deployment/engineering audience
- Citation density and style
- Industry Track specific: do they include a "deployable today" claim?

Write: paragraph-by-paragraph arc for this paper's intro.

### 2C: Results sections

Study 5–10. Record:
- Finding-led or method-led organisation?
- How do they handle negative results and baselines that collapse?
- How do they handle statistically fragile wins (small n)?
- Prose vs table relationship: what does prose add beyond the table?
- Precision of language

Write: the prose style rules for this paper's results section.

### 2D: Related work sections

Study 5–10. Record:
- Cluster-by-theme vs chronological?
- How do papers position against a SOTA baseline they claim to beat?
- Citation density per paragraph
- How do they handle work published concurrently or very recently?

Write: the cluster structure and positioning strategy for this paper.

### 2E: Methods sections in inference efficiency / Industry Track papers

This is the most important standards section for this paper. Study carefully.
Record:
- How precise is the deployment environment description?
- Do they name specific hardware, memory budgets, inference stacks?
- How do they describe the compatibility argument (FA2, vLLM, etc.)?
- How do they handle discovered-and-corrected methodological issues?
- How do they describe a signal with multiple ablated variants?

Write: the subsection structure for this paper's methods.

**→ After 2F: update TODO.md Phase 2 checkpoint.**

---

## PHASE 3 — Write sections fresh

Use only:
- LITERATURE_NOTES.md and STANDARDS.md
- The repo documentation (as source material, not as draft to copy)
- The experimental record (numbers from progress.md and phase0b_ablation_results.md)

Do NOT copy text from repo docs verbatim — refine and rewrite into paper prose.
Work in priority order. Update TODO.md after each section. Stop after Related
Work if time runs short.

---

### Section 3.1: Related Work → sections/related_work.tex

**Goal**: Place the work precisely in the literature. Be generous with citations.

**Cluster structure** (derive subsection names from literature, cover these):

**Cluster 1 — KV cache eviction: attention-based methods**
Cover H2O, SnapKV, PyramidKV, ChunkKV. What they share: attention weight as
importance proxy. What makes cumulative attention (H2O) better than single-step.
What all attention-based methods share as a failure mode in reasoning traces
(RaaS's 24.2% failure rate; H2O's 93/100 collapse in your own experiments).

**Cluster 2 — Reasoning-aware eviction**
Cover ThinKV, RaaS, LongFlow, AhaKV, LagKV. What distinguishes reasoning traces
from document-retrieval tasks (non-monotonic attention, milestone/phoenix tokens,
dead-end branches). What ThinKV does (R/E/T classification via attention sparsity)
and why it's SOTA. Where it fails: still uses attention-derived signal; no FA2
compatibility; CT kernel fork required.

**Cluster 3 — KV retrieval (not eviction)**
Cover FreeKV, Quest, SideQuest. Distinguish from eviction: retrieval keeps all
tokens in memory, selects per-step. FreeKV's inter-step query similarity
assumption. SideQuest's model-driven approach for tool responses. Why retrieval
doesn't solve the memory problem for long-generation traces.

**Cluster 4 — Hidden states as information carriers**
Cover ROME/MEMIT (mid-layer factual retrieval), EAGLE/EAGLE-2 (HS for speculative
decoding — convergent evidence that HS carries richer predictive signal than
token embeddings), logit lens (layer-wise information flow). This is the
theoretical grounding for why Band A/B carries signal. One paragraph; one or
two sentences per reference. Do not over-explain — the point is convergent
evidence, not a full review.

**Cluster 5 — FlashAttention and the O(n²) materialization constraint**
Cover FlashAttention 1 and 2 (Dao et al.). The tiling mechanism. Why
`output_attentions=True` forces eager fallback. Why this is not a minor
inconvenience — at 32k token reasoning traces, this is the difference between
viable and OOM. Position our work: first reasoning-aware eviction that is
natively FA2-compatible.

**Positioning paragraph** (end of section):
State the gap directly: "No prior decode-time eviction method for reasoning
traces combines (a) a non-attention importance signal and (b) full FlashAttention
2 compatibility." ThinKV has (a) partially (attention sparsity is still attention-
derived) and not (b). Our method has both.

**Citation rules**:
- Every claim about prior work needs a citation
- Do not cite papers you haven't read — mark [UNREAD] if needed
- Citation density: aim for 30–45 citations in related work
- Do not use "to the best of our knowledge" — state gaps as facts

**Self-critique pass**:
- Would a ThinKV or RaaS author accept our characterisation of their work?
- Is the FA2 compatibility gap stated precisely enough to defend?
- Is Cluster 4 (HS grounding) proportionate — informative without dominating?

---

### Section 3.2: Methods / Experimental Setup → sections/methods.tex

**Goal**: Full reproducibility. Precise enough that an ML engineer could
evaluate whether this works in their inference stack.

#### 3.2.1 Problem Setup and Notation
- Define the decode-time KV cache eviction problem precisely
- Notation: cache budget K, sequence length n, layers L, heads H
- Define what "FA2-compatible" means precisely in this context: reads only from
  past_key_values (already in HBM) and hidden states (standard FA2 output);
  never calls output_attentions=True; never materialises the n×n attention matrix
- State what FA2-incompatible means: any method requiring output_attentions=True

#### 3.2.2 The Hidden-State Variance Signal

**Why attention is the wrong signal (two arguments, keep separate):**

Argument 1 — Semantic noise: Attention sinks (StreamingLLM) concentrate
disproportionate attention on the first few tokens regardless of content.
Thinking filler tokens ("Let me...", "Hmm...") attract high attention during
generation because the model is attending to its current position, but carry no
reusable semantic content. Cumulative attention (H2O) suffers: milestone tokens
score high while being used, then stop accumulating, while new tokens that will
be needed accumulate less history. Concrete evidence: H2O produces empty
generations on 93/100 MATH-500 problems at cache=1024 in our experiments.

Argument 2 — Architectural cost: FA2 tiles attention computation in SRAM,
never writing the full matrix to HBM. Requesting output_attentions=True forces
eager fallback: O(n²) peak memory, elimination of 2–4× throughput gains. At
32k–60k token reasoning traces, this forces single-example batching or OOM.
Our method avoids this entirely.

**The epiphany hypothesis:**
When a model generates a semantically significant token — a key intermediate
result, a concluded insight, a transition from exploratory to convergent
reasoning — the hidden state undergoes a larger representational shift than when
generating filler. This is grounded in the ROME/MEMIT finding that mid-layers
(7–13 in 32-layer LLaMA architectures) perform factual retrieval and feature
routing — precisely the layers where we observe Band A positive ρ. The name
"epiphany" refers to these transition moments: detectable, load-bearing, worth
retaining.

**The two-band anatomy (state precisely):**
- Band A (layers 7–13): consistently positive Spearman ρ with counterfactual
  importance labels across MATH-500 and AIME-2024. High HS L2 diff at these
  layers = token is important.
- Band B (layers 18–25): consistently negative ρ. High HS L2 diff here =
  token is dispensable.
- Combined score: l10_rolling64 − l21_rolling64 (rolling mean over 64 tokens
  for each; subtracted to get a single score where higher = more important)
- Grounding: Band A corresponds to mid-layer feature routing (ROME/MEMIT);
  Band B to upper-layer prediction preparation layers that are active for all
  tokens but carry less retention value.

**The temporal trend problem and its fix (handle this honestly):**
During analysis, we found that the raw combined score (l10−l21) carries a
monotonic temporal trend within traces: l10 decreases with position, l21
increases. This means early tokens score artificially high and late tokens
artificially low — the signal tracks position as much as content. Fix: rolling
z-score detrending.

z(t) = (signal(t) − rolling_mean[t]) / (rolling_std[t] + ε)

This converts absolute magnitude (position-contaminated) to local deviation
(position-agnostic). The detrended variant (DetrendendHSVarianceEviction) is
what achieves 72% on MATH-500 in Phase 1. Describe this as a methodological
finding — something discovered and corrected, not a design choice made a priori.

#### 3.2.3 Signal Variants Implemented

Brief table or list of all five Phase 1 eviction classes — what each does,
whether it requires eager attention, FA2-compatible or not:
- HSVarianceEviction: raw l10−l21, FA2-compatible
- DetrendendHSVarianceEviction: z-score detrended, FA2-compatible (primary)
- BandAdaptiveHSEviction: all Band A/B layers, ρ-weighted, FA2-compatible
- AttentionHSProductEviction: cumulative attn + detrended HS; eager-only
- HybridSegmentHSEviction: ThinKV R/E/T segments + HS within-segment; eager-only
- lag_kv: lag-relative KV key variance; FA2-compatible; best AIME result

Also list baselines:
- H2OEviction: cumulative attention, eager-only
- ThinKVEviction: R/E/T segment classification, eager-only
- RaaSEviction: LRU + prefill preservation, eager-only

#### 3.2.4 Counterfactual Importance Labels (Phase 0B validation)

How importance labels were generated — this is the validation methodology that
justifies the Phase 0B signal claims:
- Sliding window occlusion: replace window tokens with pad_id; feed full
  modified trace up to </think> boundary; generate answer; record flip
- Important = any covering window caused answer flip (OR semantics, fixed from
  overwrite)
- Label density: ~0.20 for MATH-500 (1 in 5 tokens truly load-bearing),
  ~0.52–0.64 for AIME (harder problems, more load-bearing content)
- Key fix: true occlusion (full context, window content replaced) vs. prior
  truncation bug (which was a position test, not a content test)

Note the truncation bug clearly: early code truncated at mask_start (testing
how much prefix is needed, not whether content matters). This inflated h2o_attn's
apparent advantage. The fix changes the interpretation of Phase 0 results.

#### 3.2.5 Experimental Setup

**Models:**
- Primary: DeepSeek-R1-Distill-LLaMA-8B (matches ThinKV and ChunkKV exactly)
- Rationale for model choice: comparability with SOTA baselines; open-weight;
  representative of the reasoning-model class

**Datasets:**
- MATH-500 (n=100): competition mathematics, verifiable answers, ~4k–16k token
  traces; primary benchmark
- AIME-2024 (n=30): harder competition math, ~16k–32k token traces; tests high
  cache pressure [NEEDS ROBUSTNESS: n=30; combine AIME 2024+2025+2026 in Phase 2]
- GSM8K (n=355 correct traces): grade-school math; difficulty-regime validation;
  shows layer anatomy shift — included as supplementary finding, not as primary
  head-to-head benchmark against ThinKV/RaaS

**Cache budgets tested:**
- MATH-500: K ∈ {512, 1024, 2048, 4096}
- AIME-2024: K ∈ {512, 1024, 2048, 4096, 8192}

**Evaluation:**
- Accuracy: exact match on final boxed answer
- Speed: wall-time per problem (seconds)
- Memory: peak GPU memory (MB per example, not cumulative)
- Run configurations: eager (output_attentions=True) and flash_attention_2
  (FA2-compatible methods only) as separate configurations

**Hardware**: [read from progress.md — state the cluster configuration used;
note single-GPU flash benchmarks after multi-GPU crash issue resolved]

**Infrastructure note**: Multi-GPU runs with flash_attn crashed (CUDA unspecified
launch failure); all flash benchmarks use single-GPU 48G allocation. State this.

**Self-critique pass:**
- Is FA2 compatibility defined precisely enough that a reviewer could evaluate it?
- Is the temporal trend fix described honestly — discovered, not designed?
- Could a researcher reproduce the Phase 1 results from this section?
- Are all limitations of the hardware setup stated?

---

### Section 3.3: Results → sections/results.tex

**Goal**: Let numbers speak. Organise by finding, not by method.

**Prose rules (apply strictly):**
- Number first, interpretation second. Always.
- Do not repeat table contents in prose — add what the table cannot show.
- One interpretive sentence per finding. One.
- Never write "significantly" — write the delta.
- Negative results (H2O collapse, kv_val_var sign flip) get equal space.
- Tag fragile results: [NEEDS ROBUSTNESS: reason].
- No editorialising in results — save for discussion.

#### 3.3.1 H2O Catastrophic Collapse — Motivating the Signal Choice
Start here, not with your method winning. This motivates everything.
- Numbers: 93/100 MATH-500 problems produce empty generations at cache=1024;
  48/100 at 2048; 27/100 at 4096.
- Interpretation: H2O does not degrade gracefully; it collapses. This matches
  RaaS's documented 24.2% attention-map failure rate on reasoning traces.
- This is the empirical motivation for why attention weight is the wrong signal.

#### 3.3.2 Phase 0B Signal Validation
- Band A (l7–l13): consistently positive Spearman ρ across MATH-500 and AIME.
  State representative ρ values from phase0b_ablation_results.md.
- Band B (l18–l25): consistently negative ρ.
- h2o_attn: weakest signal tested — 3–12× weaker than Band A HS signals.
  State the specific ρ values.
- Rolling64 smoothing: outperforms raw and EMA by 30–57% universally.
- Temporal trend: l10 decreases, l21 increases with position within traces.
  Detrending fixes this — report the within-trace validation.
- [FIGURE placeholder: per-layer Spearman ρ heatmap or bar chart]

#### 3.3.3 Accuracy vs Cache Budget — MATH-500
- Table: all methods × cache sizes {512, 1024, 2048, 4096} × accuracy
- Key numbers:
  - Ceiling (no eviction): 75%
  - hs_variance_detrend @ 4096: 72% — beats ThinKV (71%), beats H2O (67%)
  - band_adaptive_hs / kv_val_var @ 2048: ~57% vs raas @ 60%
  - hs_variance @ 1024: 28% vs hybrid_seg_hs @ 36%
- Note: at 2048, attention-requiring methods (raas, hybrid_seg_hs) have an
  advantage — the FA2-compatible methods close the gap at higher budgets.
  Interpret honestly: the FA2 methods are not uniformly better at all budgets.
- [FIGURE placeholder: accuracy vs cache-size curves, grouped by FA2-compatible
  vs eager-only]

#### 3.3.4 Accuracy vs Cache Budget — AIME-2024
- Table: all methods × cache sizes × accuracy
- Key numbers:
  - Ceiling: 43%
  - lag_kv @ 8192: 37% — outperforms every attention-requiring method (best: 33%)
  - lag_kv @ 4096: 20% (tied with thinKV, h2o, hybrid_seg_hs)
- [NEEDS ROBUSTNESS: n=30, Phase 2 targets n=90 with AIME 2024+2025+2026]
- State the n explicitly in the table caption and the text.

#### 3.3.5 Speed Results
- lag_kv (FA2) is 2.8× faster than raas (eager) at AIME@8192 (441s vs 1239s
  per problem mean wall-time).
- Several FA2 methods are faster than no-eviction baseline at large cache budgets
  (smaller cache reduces per-step decode cost; signal extraction overhead is
  negligible).
- State the single-GPU constraint for flash benchmarks.
- [FIGURE placeholder: speed vs cache-size bar chart for FA2 vs eager methods]

#### 3.3.6 Layer Anatomy and Difficulty Dependence (GSM8K finding)
- For competition math (MATH-500, AIME): Band A = l7–l13, Band B = l18–l25.
- For grade-school math (GSM8K, n_eff=352): anatomy shifts — l0–l7 positive,
  l10–l30 negative, l31 strongly positive (+0.231).
- KV signal sign flips confirmed: kv_key_var = +0.380 (math500) vs −0.261
  (gsm8k) — both high-n, confirmed real.
- Interpretation: the "epiphany layers" shift by task difficulty, suggesting
  these signals are picking up genuine differences in where reasoning is
  consolidated at different cognitive loads.
- Note scope: Phase 1 targets competition math; GSM8K findings are supplementary.

**Self-critique pass:**
- Does every number in the text match the experimental record?
- Are all AIME wins tagged [NEEDS ROBUSTNESS: n=30]?
- Is the H2O collapse reported with the same prominence as our wins?
- Is the 2048-budget FA2 disadvantage reported, not buried?

---

### Section 3.4: Discussion / Conclusion → sections/discussion.tex

**Goal**: Tell the reader what to think and why it matters beyond this model
and these benchmarks.

**Para 1 — What the Band A/B anatomy tells us**
The two-band structure maps onto known mid-layer function (ROME/MEMIT factual
retrieval in l7–l13) and upper-layer prediction preparation (l18–l25). The
fact that Band B is negatively correlated — high variance there means the token
is LESS important — is the most counterintuitive finding. Develop what this
suggests: Band B layers may be active precisely when the model is generating
predictable, low-surprise tokens (filler, transitions), which means high variance
there is a signal of uninformative fluency, not of semantic load.

**Para 2 — The difficulty-regime dependence**
The GSM8K layer shift is a finding about where reasoning is consolidated by
difficulty. Harder problems consolidate reasoning in deeper mid-layers (l7–l13);
simpler problems show signal earlier (l0–l7). This suggests the signals are
picking up something real about the model's processing depth. Hedge appropriately:
this is one model family, one architecture — the pattern may shift for other
sizes and architectures.

**Para 3 — The FA2 compatibility argument as a practical contribution**
This is the Industry Track paragraph. No prior decode-time eviction method for
reasoning traces works natively within a standard FA2 inference stack. ThinKV
requires a custom CT kernel fork. RaaS, H2O, and LongFlow all require eager
attention. The practical cost of this incompatibility at 32k–60k sequences is
not academic — it determines whether a method can be deployed in production
(vLLM, TGI, SGLang). State this plainly.

**Para 4 — What the temporal trend discovery implies more broadly**
The finding that raw HS variance signals carry a monotonic positional trend
within traces has implications for anyone using residual-stream signals as
importance proxies in long sequences. Z-score detrending is a cheap fix, but
the root cause — that certain layers have systematically different baseline
activation magnitudes early vs late in generation — is worth investigating.
This is not unique to our method.

**Para 5 — Limitations (state these directly)**
- Results are from one model family (DeepSeek-R1-Distill-LLaMA-8B).
  Generalisability to other reasoning model architectures is unvalidated.
- AIME results (n=30) are statistically fragile. Phase 2 will address this
  with n=90.
- Latency measurements are single-GPU, single-example; multi-GPU and batched
  inference behaviour is not characterised.
- The FA2 memory advantage (O(n²)→O(n) for prefill) is real but not measured
  in these short-prompt benchmarks.
- Phase 2 robustness runs (AIME 2024+2025+2026, FA2-compatible hybrid for
  tight budgets) are pending.

**Para 6 — Future work (be specific)**
- Phase 2: AIME n=90 robustness; FA2-compatible analog of hybrid_seg_hs for
  tight-budget regimes; GSM8K as primary difficulty-transfer test.
- Chunk-level HS eviction (Gap C from research_overview.md §3): score contiguous
  8–16 token spans rather than individual tokens — avoids semantic fragmentation.
- Layer-wise budget allocation (Gap D): test whether PyramidKV's pyramidal
  pattern holds for decode-heavy reasoning traces; if so, apply their allocation
  formula as an orthogonal gain.
- Tiered memory architecture (Gap F): warm CPU tier for recently evicted tokens
  with recall on query similarity — handling non-monotonic attention patterns
  that permanent eviction cannot.

---

### Section 3.5: Introduction → sections/intro.tex

**Goal**: A skeptical Industry Track reviewer wants to keep reading by the end
of the first page.

Follow the arc from STANDARDS.md 2B. For this paper:

**Para 1 — The tension**
Open with something specific and true. Not "large language models have achieved
remarkable results." Consider: open with the scale of the problem — a single
AIME problem may require 60k tokens of internal reasoning; a single H100 has
80GB. The KV cache is the bottleneck. Every existing method that addresses it
uses attention weights. Attention weights are the wrong signal, and they come
at architectural cost that production inference stacks cannot pay.

**Para 2 — Two reasons attention fails**
Keep these separate and precise:
(1) Semantic: attention sinks, thinking filler tokens, milestone token timing —
the attention signal is noisy for reasoning traces (RaaS: 24.2% map failure;
ours: 93/100 collapse at cache=1024).
(2) Architectural: FlashAttention 2 cannot materialise the attention matrix.
output_attentions=True forces eager fallback. At 32k–60k tokens, this eliminates
the primary throughput optimisation of every modern inference stack.

**Para 3 — What we do**
One sentence: "We introduce epiphany-aware KV cache eviction, which uses
hidden-state variance at specific mid-layers — rather than attention weights —
as the importance signal, and is fully compatible with FlashAttention 2."

**Para 4 — Contributions (numbered, verb-first)**
1. We identify a two-band layer anatomy in 32-layer reasoning models: Band A
   (layers 7–13) shows consistently positive correlation with token importance;
   Band B (layers 18–25) shows consistently negative correlation. The combined
   score outperforms all attention-based signals tested.
2. We discover that raw HS variance signals carry a monotonic positional trend
   within reasoning traces, and show that rolling z-score detrending eliminates
   this artifact, recovering eviction quality.
3. Our detrended HS method (DetrendendHSVarianceEviction) reaches 72% on
   MATH-500 at a 4096-token cache budget, beating ThinKV (71%) while being
   the only reasoning-aware eviction method fully compatible with FlashAttention 2.
4. Our FA2-compatible lag_kv method reaches 37% on AIME-2024 at 8192 tokens
   — outperforming every attention-requiring baseline (33%) — at 2.8× the speed
   of the fastest attention-based method.
5. We provide counterfactual occlusion importance labels for 9 datasets as a
   validation resource for future eviction research.

**Para 5 — Practical upshot**
One sentence: "The method requires no training, no fine-tuning, no custom
kernels, and no modification to the inference stack — it can be dropped into
any vLLM, TGI, or SGLang deployment today."

**Rules:**
- No "to the best of our knowledge" — state gaps as facts
- No throat-clearing opening sentence
- The FA2 compatibility argument must appear in the intro, not just discussion
- Contributions must be specific — no vague "comprehensive study"
- ~10–15 citations in the intro; draw from LITERATURE_NOTES.md

**Self-critique pass:**
- Does para 2 keep the two failure modes of attention clearly separate?
- Is every contribution in para 4 backed by the experimental record?
- Does the opening sentence make a reviewer want to read the second?

---

### Section 3.6: Abstract → sections/abstract.tex

**Goal**: 150–200 words. Every sentence earns its place. Industry Track framing.

Follow the pattern from STANDARDS.md 2A. For this paper:

- **Sentences 1–2**: The problem. Reasoning models generate long traces; KV
  cache pressure is the deployment bottleneck; existing eviction methods use
  attention weights, which are semantically noisy in reasoning traces and
  architecturally incompatible with FlashAttention 2.
- **Sentence 3**: What we propose. One sentence, specific.
- **Sentences 4–7**: Findings with numbers:
  - Band A/B anatomy and what it means (with ρ values)
  - Temporal trend discovery and fix
  - MATH-500 result: 72% (ours) vs 71% (ThinKV), FA2-compatible
  - AIME result: 37% vs 33%, 2.8× faster [NEEDS ROBUSTNESS: n=30]
  - H2O collapse: 93/100 at cache=1024
- **Sentence 8**: Deployment claim. No training, no custom kernels, works in
  vLLM/TGI/SGLang today.

**Rules:**
- Lead with the deployment problem, not the algorithm
- Do not use the word "novel"
- Numbers are not optional
- Self-contained: a reader who reads only the abstract understands what was
  done and found

---

### Section 3.7: Figure specifications → sections/figures.md

Write implementation-ready specs. Include matplotlib stub code for each.

**Figure 1: Accuracy vs cache-size curves (primary result figure)**
- Type: line chart, 2 panels side by side (MATH-500, AIME-2024)
- x-axis: cache budget K (log scale)
- y-axis: accuracy (%)
- Line groups: FA2-compatible methods (solid lines) vs eager-only (dashed)
- Highlight: hs_variance_detrend and lag_kv as bold lines
- Show ceiling (no eviction) as horizontal dotted line
- Caption must: state n for each dataset; note that dashed = requires eager attn

**Figure 2: Per-layer Spearman ρ heatmap**
- Type: heatmap or grouped bar chart
- x-axis: layer index 0–31
- y-axis: Spearman ρ with importance labels
- Rows/groups: math500, math500_eager, aime2024, aime2024_eager, gsm8k_eager
- Highlight Band A (l7–l13) and Band B (l18–l25) with shaded regions
- Caption must: identify the band boundaries; note GSM8K divergence

**Figure 3: H2O collapse illustration**
- Type: bar chart or table visualisation
- x-axis: cache budget {512, 1024, 2048, 4096}
- y-axis: fraction of problems producing non-empty generation
- Bars: H2O vs hs_variance_detrend vs no-eviction
- Story: H2O collapses; ours degrades gracefully
- Caption: state the 93/100 number explicitly

**Figure 4: Speed comparison (optional but strong for Industry Track)**
- Type: grouped bar chart
- x-axis: methods (FA2-compatible vs eager-only)
- y-axis: wall-time per problem (seconds)
- Dataset: AIME-2024 @ 8192
- Highlight: lag_kv vs raas (2.8× difference)
- Caption: state single-GPU constraint

---

### Section 3.8: Title alternatives → sections/titles.md

Evaluate the current title:
> "Epiphany-Aware KV Cache Eviction: Why Hidden States Outperform Attention"

Then propose 4 alternatives. For each including the current title, write:
- The title
- What framing it leads with (deployment? signal? architecture?)
- Who it speaks to most
- One strength and one weakness
- EMNLP Industry Track fit (high/medium/low)

Recommend one with rationale. Note: "epiphany-aware" is doing real work in the
current title — any alternative that loses the conceptual hook should say why
the trade-off is worth it.

---

### Section 3.9: REVISION_LOG.md

For every section, record:
- What the self-critique found
- What was changed
- What was left as [NEEDS RESULT] or [NEEDS ROBUSTNESS] and why
- Any claim you were uncertain about and how you handled it

**→ After all Phase 3 sections: update TODO.md Phase 3 checkpoint.**

---

## PHASE 4 — Repo doc synthesis

Unlike EdgeTRM, there are no existing .tex drafts. Instead, the repo contains
dense scientific prose in the experiment docs that may contain paper-ready
arguments, framings, and sentences. This phase extracts the best of those.

### Step 4A: Re-read repo docs as source material

Re-read with fresh eyes after having written your own versions:
- research_overview.md §2 related work characterisations — are these better or
  worse than what you wrote in 3.1?
- research_overview.md §8 differentiation table — can this become a table in
  the paper? (It's very close to paper-ready already)
- research_overview.md §9 temporal trend analysis — does your methods section
  capture this as well as the original?
- progress.md Phase 1 debugging saga — are any of the infrastructure findings
  worth a brief mention in the paper (e.g., multi-GPU DynamicCache fix)?
- paper_strategy.md figure plan — how does it compare to your 3.7 figure specs?

### Step 4B: Write REPO_CRITIQUE.md

For each repo doc section worth mining:
```
## [doc section]
### Content worth incorporating into paper
- Specific sentences or arguments that are better than Phase 3 versions

### Content that contradicts or needs updating
- Any claim in the doc that Phase 1 results have superseded or corrected
  (e.g., kv_val_var "primary fallback" claim — corrected April 13)

### Content out of scope
- Gaps F, G, H from research_overview.md are future work — do not let the
  agent expand scope into these
```

### Step 4C: Write SYNTHESIS_PLAN.md

Section-by-section merge decisions. For each section:
- What Phase 3 version contributes
- What repo docs contribute
- Final structure decision
- Any unresolved conflicts for human judgment

### Step 4D: Produce final merged .tex files

**→ After Phase 4: update TODO.md Phase 4 checkpoint and final status table.**

---

## FINAL: Write STATUS.md

First file read in the morning. Make it useful.

```markdown
## What was completed
[Each section: SOLID / NEEDS REVIEW / PLACEHOLDER]

## Claims safe to submit (solid experimental backing)
[Be specific]

## Claims requiring Phase 2 robustness runs before submission
[Be specific — which claims, which experiments]

## Methodological facts that must appear in the final paper
[The temporal trend, the truncation bug fix, the multi-GPU constraint,
the latency simulation scope — these cannot be omitted]

## Reviewer risks
[What a skeptical EMNLP Industry Track reviewer would push back on hardest]

## Suggested first actions in the morning
[Ordered list, most critical first]
```

---

## Tone and style rules (non-negotiable)

- British spelling: colour, generalise, behaviour, neighbour, artefact, recognise
- No AI filler: "notably", "importantly", "it is worth noting", "crucially",
  "in this paper we", "we present a comprehensive", "leveraging", "showcasing"
- No throat-clearing opening sentences anywhere
- Active voice preferred; passive only where the subject is genuinely unknown
- Numbers always: never "significantly faster", always "2.8× faster"
- Hedge precisely: "suggests", "is consistent with", "indicates" — not "might"
- Industry Track register: a paper for engineers who will deploy this, not just
  researchers who will read it. Deployment practicality is a first-class concern.
- Tables: numbers right-aligned, header bolded, best result per column bolded
- Tag all fragile results: [NEEDS ROBUSTNESS: reason]
- Never use "to the best of our knowledge"

---

## Output file index

| File | Written by | Contents |
|---|---|---|
| TODO.md | Both | Shared task ledger |
| LITERATURE_NOTES.md | Copilot | Per-paper notes |
| STANDARDS.md | Copilot | Derived writing standards |
| sections/related_work.tex | Both | Phase 3 → Phase 4 merged |
| sections/methods.tex | Both | Phase 3 → Phase 4 merged |
| sections/results.tex | Both | Phase 3 → Phase 4 merged |
| sections/discussion.tex | Both | Phase 3 → Phase 4 merged |
| sections/intro.tex | Both | Phase 3 → Phase 4 merged |
| sections/abstract.tex | Both | Phase 3 → Phase 4 merged |
| sections/figures.md | Copilot | Figure specs + matplotlib stubs |
| sections/titles.md | Copilot | Title evaluation + alternatives |
| REPO_CRITIQUE.md | Both | What repo docs contribute to paper |
| SYNTHESIS_PLAN.md | Both | Phase 4 merge decisions |
| REVISION_LOG.md | Both | Self-critique log per section |
| STATUS.md | Both | Final state for morning review |