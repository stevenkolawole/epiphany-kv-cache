# Model Variants & Testing Strategy

## Motivation
Testing both vanilla and reasoning-enabled LLaMA versions reveals:
- **Vanilla LLaMA**: Standard reasoning tasks; shorter traces; baseline behavior
- **Reasoning-enabled**: Longer traces (10k+ tokens); more repetition/exploration; highest eviction potential

## Model Variants

### Vanilla Models
- **LLaMA-3.1-8B-Instruct** (`meta-llama/Llama-3.1-8B-Instruct`)
  - Standard instruction-tuned version
  - Shorter reasoning traces (~100-300 tokens)
  - Good baseline for comparison

- **Qwen-7B-Instruct** (`Qwen/Qwen2-7B-Instruct`)
  - Qwen's instruction-tuned version
  - Good multilingual support
  - Comparable trace lengths to LLaMA-3

### Reasoning-Enabled Models
- **DeepSeek-R1-Distill-LLaMA-8B** (`deepseek-ai/deepseek-r1-distill-llama-8b`)
  - DeepSeek-R1 distilled into LLaMA-3.1-8B
  - Longer traces (1k-5k+ tokens)
  - More exploration, self-correction
  - **High potential for KV cache eviction gains**

- **DeepSeek-R1-Distill-Qwen-7B** (`deepseek-ai/deepseek-r1-distill-qwen-7b`)
  - DeepSeek-R1 distilled into Qwen-7B
  - Similar characteristics to LLaMA distilled version

- **Mistral-7B-Instruct-v0.3** (`mistralai/Mistral-7B-Instruct-v0.3`)
  - Mistral's latest instruction-tuned version
  - Capable of reasoning tasks
  - Moderate trace length increase

## Testing Plan

### Phase 1: Vanilla Models (Quick POC)
1. **LLaMA-3-8B** (baseline)
   - 100 math reasoning examples (GSM8K)
   - Baseline vs Semantic eviction
   - Expected: 15-25% memory reduction

2. **Qwen-7B** (comparison)
   - Same 100 examples
   - Expected: Similar to LLaMA-3

### Phase 2: Reasoning-Enabled Models (Full POC)
1. **DeepSeek-R1-Distill-7B** (high potential)
   - Same 100 examples
   - Expected: 30-50% memory reduction due to longer traces
   - **This is where semantic eviction should shine**

2. **A/B Analysis**: Compare trace patterns
   - Vanilla: Shorter, more direct reasoning
   - Reasoning: Longer, more exploration/backtracking
   - Visualize segment distributions (rambling%, insight%, etc.)

### Phase 3: Large-Scale Comparison
- Scale to 1000+ examples
- Vary cache sizes (256, 512, 1024, 2048)
- Measure accuracy impact at different eviction ratios

## How to Run

### Test vanilla LLaMA-3
```python
python scripts/poc_harness.py
# Internally uses: model_variants=['llama3_vanilla'], eviction_methods=['baseline', 'semantic']
```

### Test reasoning-enabled DeepSeek
```python
from scripts.poc_harness import run_poc
run_poc(model_variants=['deepseek_r1'], eviction_methods=['baseline', 'semantic'])
```

### Test all models
```python
from scripts.poc_harness import run_poc
run_poc(
    model_variants=['llama3_vanilla', 'qwen_vanilla', 'deepseek_r1'],
    eviction_methods=['baseline', 'semantic']
)
```

## Expected Key Findings

### Vanilla vs Reasoning Comparison
| Metric | Vanilla | Reasoning-Enabled |
|--------|---------|-------------------|
| Avg trace length | ~150 tokens | ~1000 tokens |
| Rambling % | ~10% | ~30% |
| Insight % | ~60% | ~40% |
| Memory reduction potential | 15-25% | 30-50% |
| Semantic > Baseline gap | ~5% | ~15% |

### Semantic vs Attention-Based Eviction
- **Vanilla models**: Minimal difference (~5%)
  - Attention patterns already somewhat aligned with importance
  - Less rambling to evict
  
- **Reasoning-enabled models**: Large difference (~15%+)
  - More rambling tokens (low semantic importance, high attention)
  - Semantic methods better distinguish thinking vs conclusions
  - Potential to preserve quality while cutting 40%+ memory

## Notes
- Prefer reasoning-enabled for experiments if compute allows (higher signal)
- Use vanilla for faster iteration / debugging
- Update traces after testing each model (store separate files with model name)
