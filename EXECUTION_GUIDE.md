# Execution Guide for KV Cache Experiments

Quick reference for running experiments.

## Setup
```bash
# Already done, but for reference:
pip install -r requirements.txt
```

## Phase 1: Data Collection (✓ DONE)
```bash
# Analyze and segment reasoning traces
python scripts/analyze_traces.py

# Output: data/synthetic_math_traces.jsonl
```

## Phase 2: Test Eviction Logic (✓ DONE)
```bash
# Verify eviction implementations work
python src/eviction.py

# Expected output:
# Baseline: 1000 → 512 tokens (cache size: 512)
# Semantic: 1000 → 512 tokens (cache size: 512)
```

## Phase 3: POC Testing (NEXT)
```bash
# Run POC with GPT2 (quick test, ~10 examples)
python scripts/poc_harness.py

# Results saved to: experiments/poc_results.jsonl
```

## Phase 4: Scale to Real Models (TODO)
```bash
# Modify scripts/poc_harness.py to use actual models:
# harness = POCHarness(model_name="meta-llama/Llama-2-7b", cache_size=512)
# or
# harness = POCHarness(model_name="Qwen/Qwen-7B", cache_size=512)

# Then run:
python scripts/poc_harness.py
```

## Monitoring
- Check `experiments/progress.md` for experiment status
- Results logged to `experiments/poc_results.jsonl`
- Traces stored in `data/`

## Debugging
```bash
# Verify imports
python -c "import torch, transformers; print('OK')"

# Check traces
python -c "import json; print(json.load(open('data/synthetic_math_traces.jsonl')))"

# Quick model load test (replace model name)
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('gpt2'); print('OK')"
```
