# Epiphany-Aware KV Cache Management

This project explores advanced KV cache management strategies for reasoning models, focusing on semantic importance over traditional attention-based eviction.

## Problem Statement

Reasoning models generate extremely long traces (10k-100k tokens), leading to KV caches that can exceed 84GB and overflow even high-end GPUs like H100. Current eviction strategies rely on attention scores, which don't align with semantic importance in reasoning contexts.

## Key Insights

- Attention ≠ semantic importance for reasoning
- "Let me think..." (high attention, low value) vs "The key insight is..." (medium attention, critical content)
- Reasoning traces are generally disposable except for interpretability purposes

## Project Structure

- `src/`: Core implementation
- `tests/`: Unit tests
- `notebooks/`: Experimental notebooks
- `data/`: Datasets and cached data
- `experiments/`: Experiment scripts and results

## Installation

```bash
pip install -r requirements.txt
```

## Usage

[To be added]