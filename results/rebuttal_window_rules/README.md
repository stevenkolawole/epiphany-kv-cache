# Baseline scoring rules on EpiKV's recomputed query window, and kvpress SnapKV accuracy (rebuttal, submission 50503)

Two reviewer requests. Vjb9 (Q3b): evaluate RaaS and R-KV with the same recomputed query window as EpiKV, so that the
scoring rule is separated from kernel-level choices. dspk (W1): recomputing a small query window is standard practice
(SnapKV via kvpress); our cost runs (`../rebuttal_baseline_costs`) show kvpress DecodingPress matches EpiKV's speed when
batched, so this measures its accuracy at the same budgets. Tables: `TABLES.md`, generated from `cells/` by
`python3 scripts/window_rule_tables.py results/rebuttal_window_rules`.

These numbers come from Babel (L40S, transformers 5.2.0). The paper's numbers come from other hardware and library
versions; compare these only with the EpiKV rows in this folder, never with the paper's tables.

## Methods

All four run in `scripts/benchmark.py` with the same prompts, decoding loop, answer extraction and scorer.

- **EpiKV** (`kv_seg_hs_fill` with `--refresh_tau 128 --tau_mode steps --query_sim_mode replace --query_sim_weight 1
  --query_mode qk --query_sim_window 8`): rerun on this stack as the reference for the paired tests.
- **RaaS rule on EpiKV's window** (`raas_epikv_window`) and **R-KV rule on EpiKV's window** (`r_kv_epikv_window`):
  `WindowRuleEviction` in `src/eviction.py`. Everything except the score is EpiKV's:
  - the window: at a compaction the queries of the 8 most recent decode tokens are recomputed at layer band_a + 1
    (Llama 11, Qwen 21) from their stored Band-A hidden states, through the model's own input norm, q_proj and RoPE
    at their logical positions, and scored against that layer's cached keys with a causal mask; `_window_probs` is
    `KVSegHSEviction._qk_relevance` up to its softmax, and `tests/test_window_rules.py` checks that the head and
    row mean of its output equals `_qk_relevance` bit for bit;
  - the schedule: a compaction fires when 128 steps have passed since the last one and the cache exceeds K (the
    test checks that the window rules compact on exactly EpiKV's steps); exactly K tokens survive, so the cache
    peaks at K + 128;
  - the frame: every prompt token is kept, and so are the last min(128, K/4) tokens; the rule ranks the decode
    tokens in between and keeps the top K - prompt - recent.
  EpiKV's segment tiers and its fill pass are part of EpiKV's rule and are not used.
  - **RaaS mapping.** RaaS gives each decode token a timestamp, refreshes it whenever the token is in the top half
    of the current step's attention over decode tokens, and evicts the oldest timestamps. Here each token starts
    with its arrival step. Window row j is the query of step t-7+j (t = the compaction step); taking the rows in
    order, every decode token row j can see whose probability (mean over heads) is at least the row's median over
    the decode tokens it can see is stamped t-7+j. The newest timestamps are kept; ties go to the higher mean
    window probability (EpiKV's relevance). Against `raas` (the paper's RaaS): timestamps are refreshed from 8 rows
    every 128 steps instead of one row every step, from one layer instead of the mean over all layers, and the
    cache is compacted to K every 128 steps instead of one token per step.
  - **R-KV mapping** (as adapted in `RKVEviction`, the paper's R-KV row). Importance: each window row (mean over
    heads) max-pooled with kernel 7, then averaged over the 8 rows. Redundancy: per layer and KV head, each key's
    mean cosine similarity to all cached keys, softmax over positions, averaged over heads and layers.
    Score = 0.1 * z(importance) - 0.9 * z(redundancy), each standardised over the whole cache. Against `r_kv`:
    the attention rows come from the recomputed window at one layer instead of the model's own attention averaged
    over all layers (so it runs under FlashAttention), and the prompt is always kept (`r_kv` can evict it); R-KV's
    8 observation tokens are inside the recent window.
- **kvpress SnapKV DecodingPress** (`kvpress_snapkv_decoding`, `run_one_kvpress`): kvpress 0.5.5,
  `DecodingPress(base_press=SnapKVPress(), compression_interval=128, target_size=K, hidden_states_buffer_size=256)`,
  the configuration of the cost runs, built fresh for each problem. kvpress's own rule, unchanged: every 128 decode
  steps each layer recomputes the queries of its last 64 buffered hidden states, averages their attention over the
  window, smooths it with a width-5 average pool, and keeps the top K per KV head (the last 64 positions always
  kept). It can evict prompt tokens, and in decoding it rotates all 64 window queries with the current token's
  RoPE position (kvpress takes `position_embeddings[:, -64:]`, which holds one position during decoding).

## Protocol

MATH-500, all 500 problems, greedy, max_new_tokens 16384, logical positions, R1-Distill-Llama-8B and
R1-Distill-Qwen-7B (bands 10/21 and 20/24), K in {512, 1024, 2048, 4096}, bf16, FlashAttention-2. Babel L40S
(`--constraint=L40S`), two processes per card. A loop is a trace that reached the 16384-token cap. Paired tests:
on each problem, EpiKV right and the other wrong ("EpiKV only") or the reverse ("other only"); exact two-sided
McNemar p (binomial on the discordant pairs).

## Code and stack

`src/eviction.py` md5 b1fe2b9d, `scripts/benchmark.py` md5 3a7e598a (the same on Babel, `~/workspace/kvcache_window`).
Interpreter: a venv over Babel's shared `kvcache` conda env (torch 2.10.0+cu128, transformers 5.2.0, flash-attn
2.8.3) that adds only kvpress 0.5.5, cachetools, fire 0.6.0, termcolor and pytest
(`scripts/babel_window_rules/setup_overlay.sh`); all four methods run on it. HF_HOME=/data/user_data/skolawol/hf_home,
offline.

## Smoke (measured; `smoke/`)

MATH-500 problems 0-4, K=512, cap 16384, two methods per L40S card (`smoke.sbatch`, jobs 11158973-11158976), run
through `scripts/trace_compactions.py`, which prints one COMPACT line per compaction. `tests/test_window_rules.py`:
5 passed on the same stack. All 8 runs exit 0, with no problem errors and no CPU or meta-device offload. Every
compaction goes from at most 640 tokens (K + 128) down to 512 and keeps every prompt slot. The window rules compact
on the same steps as EpiKV (first at step 446, from 513 tokens). Correct answers out of 5 (Llama / Qwen): EpiKV 4 / 4, RaaS rule
5 / 4, R-KV rule 5 / 3, kvpress 3 / 2. Decode speed is 10-33 tokens/s per process with two processes per card. Five
problems are too few to say anything about accuracy.

## Grid

`lines.txt` (from `scripts/babel_window_rules/make_lines.sh`) holds 160 lines: 2 models x 4 methods x 4 K x five
100-problem pieces. `grid.sbatch` is an array on the preempt partition (L40S, `--requeue`). Task i runs lines 2i
and 2i+1 side by side on one card, each with `--resume`. Records are in `cells/<model>_<method>_K<K>_s<start>.json`.
Launched 2026-11-15 as array 11159252 (`--array=0-79%20`, so 20 of the 24 preempt-QOS GPUs; general's 8 are left for
other lanes). Expected time (inferred from the smoke wall times, n=5 per method, which includes one cap-hitting
problem): 3.0-13.0 h per 100-problem piece, mean 6.7 h. That comes to about 27 h of array time, plus the tail of the
Qwen kvpress pieces (~13 h) and any preemption, so roughly 1.5 days.

## Finishing the grid

1. State: `ssh babel 'sacct -j 11159252 -X -n -o JobID%20,State%20 | sort -k2 | uniq -c -f1; ls ~/rebuttal_window_rules/done | wc -l'`
   (160 done markers when every line exited 0). Waiter, which exits when the array leaves the queue or a task fails:
   `J=11159252; until s=$(ssh babel "squeue -h -j $J | wc -l; sacct -j $J -X -n -o State%20 | grep -cE 'FAILED|NODE_FAIL|OUT_OF_ME|TIMEOUT|CANCELLED'; true") && { [ "$(echo "$s" | sed -n 1p)" = 0 ] || [ "$(echo "$s" | sed -n 2p)" -gt 0 ]; }; do sleep 600; done`
2. A FAILED or TIMEOUT task i: read `logs/grid_11159252_<i>.out` and `logs/<tag>.log`, then
   `sbatch --array=<i> grid.sbatch`. Its partner line skips if it is done; `--resume` continues from the last finished problem.
3. `rsync -a babel:rebuttal_window_rules/cells/ results/rebuttal_window_rules/cells/`, then
   `python3 scripts/window_rule_tables.py results/rebuttal_window_rules`. Quote no number until its
   "Incomplete cells" list is empty (that check covers partial pieces, gaps, and errored or 0-token problems). Then
   commit `cells/` and `TABLES.md`.
