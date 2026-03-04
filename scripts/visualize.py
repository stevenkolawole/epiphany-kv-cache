"""
Visualization scripts for trace analysis and eviction results.
Generates plots for traces, segments, and eviction performance comparisons.
"""

import json
import re
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Allow running from project root or from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Segment classification (single definition — no copies)
# Uses the same compiled patterns as src/data_collection.py and
# scripts/analyze_traces.py. Word-boundary regex avoids false positives.
# ---------------------------------------------------------------------------
_RAMBLING = re.compile(
    r'\b(i think|let me|hmm|wait|actually|i\'m|i was)\b', re.IGNORECASE
)
_EXPLORATION = re.compile(
    r'\?|\b(what if|why|how|could)\b', re.IGNORECASE
)
_INSIGHT = re.compile(
    r'\b(thus|therefore|conclusion|answer|result)\b', re.IGNORECASE
)

SEGMENT_COLORS = {
    'rambling':    '#FF6B6B',
    'exploration': '#4ECDC4',
    'insight':     '#45B7D1',
    'neutral':     '#95E1D3',
}


def _classify_line(line: str) -> str:
    """Return the segment type for a single line."""
    if _RAMBLING.search(line):
        return 'rambling'
    if _EXPLORATION.search(line):
        return 'exploration'
    if _INSIGHT.search(line):
        return 'insight'
    return 'neutral'


def _segment_trace(reasoning: str) -> List[tuple]:
    """Return list of (line_idx, seg_type) for non-empty lines."""
    return [
        (i, _classify_line(line))
        for i, line in enumerate(reasoning.split('\n'))
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_trace_segments(
    traces: List[Dict[str, Any]],
    output_path: str = "experiments/viz_trace_segments.png",
):
    """Visualise segment types across traces as horizontal colour bars."""
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(len(traces), 1, figsize=(14, 3 * len(traces)))
    if len(traces) == 1:
        axes = [axes]

    for idx, trace in enumerate(traces):
        reasoning = trace.get('reasoning', '')
        segments = _segment_trace(reasoning)

        ax = axes[idx]
        for line_idx, seg_type in segments:
            color = SEGMENT_COLORS.get(seg_type, SEGMENT_COLORS['neutral'])
            ax.add_patch(
                Rectangle((line_idx, 0), 1, 1,
                           facecolor=color, edgecolor='black', linewidth=0.5)
            )

        max_x = max((s[0] for s in segments), default=0) + 2
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"Trace {idx+1}", fontsize=10, fontweight='bold')
        ax.set_yticks([])
        ax.set_title(trace['question'][:60] + "...", fontsize=10)
        if idx < len(traces) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Line Index", fontsize=10)

    legend_elements = [
        mpatches.Patch(facecolor=color, label=seg_type.capitalize())
        for seg_type, color in SEGMENT_COLORS.items()
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved segment visualization to {output_path}")
    plt.close()


def plot_segment_distribution(
    traces: List[Dict[str, Any]],
    output_path: str = "experiments/viz_segment_dist.png",
):
    """Plot segment type distribution across all traces."""
    segment_counts = {k: 0 for k in SEGMENT_COLORS}

    for trace in traces:
        for _, seg_type in _segment_trace(trace.get('reasoning', '')):
            segment_counts[seg_type] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = list(SEGMENT_COLORS.values())
    ax1.pie(
        segment_counts.values(), labels=segment_counts.keys(),
        autopct='%1.1f%%', colors=colors, startangle=90,
    )
    ax1.set_title("Segment Type Distribution", fontsize=12, fontweight='bold')

    ax2.bar(list(segment_counts.keys()), list(segment_counts.values()), color=colors)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Segment Counts by Type", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved segment distribution to {output_path}")
    plt.close()


def plot_trace_length_distribution(
    traces: List[Dict[str, Any]],
    output_path: str = "experiments/viz_trace_lengths.png",
):
    """Plot distribution of reasoning trace lengths."""
    lengths = [len(trace.get('reasoning', '').split()) for trace in traces]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(lengths, bins=10, color='#45B7D1', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(lengths):.0f}')
    ax1.axvline(np.median(lengths), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(lengths):.0f}')
    ax1.set_xlabel("Tokens", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Trace Length Distribution", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2.boxplot(lengths, vert=True)
    ax2.set_ylabel("Tokens", fontsize=11)
    ax2.set_title("Trace Length Statistics", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    stats_text = (
        f"Min: {min(lengths)}\nMax: {max(lengths)}\n"
        f"Mean: {np.mean(lengths):.1f}\nStd: {np.std(lengths):.1f}"
    )
    ax2.text(1.3, np.mean(lengths), stats_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved trace length distribution to {output_path}")
    plt.close()


def plot_eviction_comparison(
    results_path: str = "experiments/poc_results.jsonl",
    output_path: str = "experiments/viz_eviction_comparison.png",
):
    """Plot comparison of eviction methods from measured POC results."""
    results = []
    if Path(results_path).exists():
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

    if not results:
        print(f"No results in {results_path}. Run POC first.")
        return

    methods = sorted({r['eviction_method'] for r in results})
    colors = ['#FF6B6B', '#45B7D1', '#FFC857']

    metrics = ['avg_accuracy', 'peak_memory_mb', 'avg_time_per_example', 'avg_tokens_generated']
    labels = ['Accuracy', 'Peak Memory (MB)', 'Time/Example (s)', 'Avg Tokens']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx // 2, idx % 2]
        method_avgs = []
        for m in methods:
            vals = [r[metric] for r in results if r['eviction_method'] == m]
            method_avgs.append(np.mean(vals) if vals else 0)

        bars = ax.bar(
            range(len(methods)), method_avgs,
            color=colors[:len(methods)], alpha=0.7, edgecolor='black', linewidth=1.5,
        )
        for bar, val in zip(bars, method_avgs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.capitalize() for m in methods])
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{label} Comparison", fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved eviction comparison to {output_path}")
    plt.close()


def plot_theoretical_eviction_savings(
    max_seq_len: int = 2048,
    cache_sizes: List[int] = None,
    output_path: str = "experiments/viz_memory_reduction.png",
):
    """
    Plot the THEORETICAL fraction of KV cache tokens retained at different cache
    budgets, assuming a fixed maximum sequence length.

    This is a linear relationship: retention = cache_size / max_seq_len.
    It does NOT depend on model architecture constants (num_layers, num_heads,
    head_dim) because those cancel out when computing the ratio.

    IMPORTANT: This is a theoretical bound, not a measured result.
    Actual savings depend on (a) how often the sequence exceeds cache_size,
    (b) eviction quality (are the right tokens kept?), and (c) model architecture.
    """
    if cache_sizes is None:
        cache_sizes = [256, 512, 1024, 2048]

    retention_fractions = [min(cs / max_seq_len, 1.0) for cs in cache_sizes]
    eviction_fractions = [1.0 - r for r in retention_fractions]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(cache_sizes))
    width = 0.35
    bars1 = ax.bar(x - width / 2, retention_fractions, width,
                   label='Tokens Retained', color='#45B7D1', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width / 2, eviction_fractions, width,
                   label='Tokens Evicted', color='#FF6B6B', alpha=0.7, edgecolor='black')

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f'{h:.0%}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Cache Budget (tokens)", fontsize=11)
    ax.set_ylabel("Fraction of Sequence", fontsize=11)
    ax.set_title(
        f"Theoretical KV Cache Retention vs Eviction\n"
        f"(assumes max sequence length = {max_seq_len} tokens; "
        f"THEORETICAL — not measured)",
        fontsize=11, fontweight='bold',
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(cs) for cs in cache_sizes])
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.text(
        0.5, 0.01,
        "Note: Actual memory savings additionally depend on num_layers × num_heads × head_dim × dtype.",
        ha='center', fontsize=9, style='italic', color='gray',
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved theoretical memory reduction to {output_path}")
    plt.close()


def generate_all_visualizations():
    """Generate all visualizations."""
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60 + "\n")

    traces_path = Path("data/synthetic_math_traces.jsonl")
    traces = []
    if traces_path.exists():
        with open(traces_path) as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))

    if traces:
        print(f"Loaded {len(traces)} traces\n")
        plot_trace_segments(traces)
        plot_segment_distribution(traces)
        plot_trace_length_distribution(traces)
    else:
        print("No traces found. Run scripts/analyze_traces.py first.\n")

    plot_theoretical_eviction_savings()
    plot_eviction_comparison()

    print("\n" + "=" * 60)
    print("Visualizations complete.")
    print("=" * 60)
    print("\nGenerated files:")
    print("  experiments/viz_trace_segments.png")
    print("  experiments/viz_segment_dist.png")
    print("  experiments/viz_trace_lengths.png")
    print("  experiments/viz_memory_reduction.png  (theoretical)")
    print("  experiments/viz_eviction_comparison.png  (requires POC results)")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    generate_all_visualizations()
