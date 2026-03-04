"""
Visualization scripts for trace analysis and eviction results.
Generates plots for traces, segments, and eviction performance comparisons.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def plot_trace_segments(traces: List[Dict[str, Any]], output_path: str = "experiments/viz_trace_segments.png"):
    """
    Visualize segment types across traces.
    Shows distribution of rambling, exploration, insight, and neutral segments.
    """
    from matplotlib.patches import Rectangle
    
    fig, axes = plt.subplots(len(traces), 1, figsize=(14, 3 * len(traces)))
    if len(traces) == 1:
        axes = [axes]
    
    segment_map = {
        'rambling': '#FF6B6B',      # Red
        'exploration': '#4ECDC4',    # Teal
        'insight': '#45B7D1',        # Blue
        'neutral': '#95E1D3'         # Light green
    }
    
    def identify_segments(reasoning: str):
        """Segment reasoning by type."""
        lines = reasoning.split('\n')
        segments = []
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            line_lower = line.lower()
            if any(phrase in line_lower for phrase in ['hmm', "i think", "let me", "wait", "actually", "i'm"]):
                seg_type = 'rambling'
            elif any(phrase in line_lower for phrase in ['?', 'what if', 'why', 'how']):
                seg_type = 'exploration'
            elif any(phrase in line_lower for phrase in ['thus', 'therefore', 'conclusion', 'answer', 'result', 'so']):
                seg_type = 'insight'
            else:
                seg_type = 'neutral'
            segments.append((i, seg_type))
        
        return segments
    
    for idx, trace in enumerate(traces):
        reasoning = trace.get('reasoning', '')
        segments = identify_segments(reasoning)
        
        ax = axes[idx]
        
        # Plot segments as horizontal bars
        for line_idx, seg_type in segments:
            color = segment_map.get(seg_type, '#95E1D3')
            ax.add_patch(Rectangle((line_idx, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5))
        
        ax.set_ylim(0, 1)
        ax.set_xlim(0, max([s[0] for s in segments]) + 2 if segments else 1)
        ax.set_ylabel(f"Trace {idx+1}", fontsize=10, fontweight='bold')
        ax.set_yticks([])
        ax.set_title(trace['question'][:60] + "...", fontsize=10)
        
        if idx == len(traces) - 1:
            ax.set_xlabel("Line Index", fontsize=10)
        else:
            ax.set_xticklabels([])
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=color, label=seg_type.capitalize())
                      for seg_type, color in segment_map.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.01))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved segment visualization to {output_path}")
    plt.close()


def plot_segment_distribution(traces: List[Dict[str, Any]], output_path: str = "experiments/viz_segment_dist.png"):
    """Plot segment type distribution across all traces."""
    segment_counts = {'rambling': 0, 'exploration': 0, 'insight': 0, 'neutral': 0}
    
    def identify_segments(reasoning: str):
        lines = reasoning.split('\n')
        segments = []
        for line in lines:
            if not line.strip():
                continue
            line_lower = line.lower()
            if any(phrase in line_lower for phrase in ['hmm', "i think", "let me", "wait", "actually"]):
                seg_type = 'rambling'
            elif any(phrase in line_lower for phrase in ['?', 'what if', 'why', 'how']):
                seg_type = 'exploration'
            elif any(phrase in line_lower for phrase in ['thus', 'therefore', 'conclusion', 'answer', 'result']):
                seg_type = 'insight'
            else:
                seg_type = 'neutral'
            segments.append(seg_type)
        return segments
    
    for trace in traces:
        reasoning = trace.get('reasoning', '')
        segments = identify_segments(reasoning)
        for seg_type in segments:
            segment_counts[seg_type] += 1
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']
    ax1.pie(segment_counts.values(), labels=segment_counts.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title("Segment Type Distribution", fontsize=12, fontweight='bold')
    
    # Bar chart
    types = list(segment_counts.keys())
    counts = list(segment_counts.values())
    ax2.bar(types, counts, color=colors)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Segment Counts by Type", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved segment distribution to {output_path}")
    plt.close()


def plot_trace_length_distribution(traces: List[Dict[str, Any]], output_path: str = "experiments/viz_trace_lengths.png"):
    """Plot distribution of reasoning trace lengths."""
    lengths = [len(trace.get('reasoning', '').split()) for trace in traces]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(lengths, bins=10, color='#45B7D1', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lengths):.0f}')
    ax1.axvline(np.median(lengths), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(lengths):.0f}')
    ax1.set_xlabel("Tokens", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Trace Length Distribution", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Box plot
    ax2.boxplot(lengths, vert=True)
    ax2.set_ylabel("Tokens", fontsize=11)
    ax2.set_title("Trace Length Statistics", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    stats_text = f"Min: {min(lengths)}\nMax: {max(lengths)}\nMean: {np.mean(lengths):.1f}\nStd: {np.std(lengths):.1f}"
    ax2.text(1.3, np.mean(lengths), stats_text, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved trace length distribution to {output_path}")
    plt.close()


def plot_eviction_comparison(results_path: str = "experiments/poc_results.jsonl", 
                            output_path: str = "experiments/viz_eviction_comparison.png"):
    """Plot comparison of baseline vs semantic eviction."""
    results = []
    if Path(results_path).exists():
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    
    if not results:
        print(f"No results in {results_path}. Run POC first.")
        return
    
    # Group by eviction method
    baseline_results = [r for r in results if r.get('eviction_method') == 'baseline']
    semantic_results = [r for r in results if r.get('eviction_method') == 'semantic']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['avg_accuracy', 'peak_memory_mb', 'avg_time_per_example', 'avg_tokens_generated']
    labels = ['Accuracy', 'Peak Memory (MB)', 'Time/Example (s)', 'Avg Tokens']
    
    methods = ['Baseline', 'Semantic']
    colors_comp = ['#FF6B6B', '#45B7D1']
    
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx // 2, idx % 2]
        
        baseline_vals = [r.get(metric, 0) for r in baseline_results]
        semantic_vals = [r.get(metric, 0) for r in semantic_results]
        
        if baseline_vals or semantic_vals:
            baseline_avg = np.mean(baseline_vals) if baseline_vals else 0
            semantic_avg = np.mean(semantic_vals) if semantic_vals else 0
            
            x_pos = [0, 1]
            values = [baseline_avg, semantic_avg]
            
            bars = ax.bar(x_pos, values, color=colors_comp, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(f"{label} Comparison", fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved eviction comparison to {output_path}")
    plt.close()


def plot_memory_reduction(cache_sizes: List[int] = [256, 512, 1024, 2048],
                         reduction_pcts: List[float] = [25, 35, 40, 45],
                         output_path: str = "experiments/viz_memory_reduction.png"):
    """Plot potential memory reduction vs cache size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_memory = np.array(cache_sizes) * 2  # Assume 2 bytes per token (rough estimate)
    evicted_memory = baseline_memory * (1 - np.array(reduction_pcts) / 100)
    
    x = np.arange(len(cache_sizes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_memory, width, label='Baseline', color='#FF6B6B', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, evicted_memory, width, label='With Eviction', color='#45B7D1', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel("Max Cache Size (tokens)", fontsize=11)
    ax.set_ylabel("Memory (arbitrary units)", fontsize=11)
    ax.set_title("Potential Memory Reduction with Semantic Eviction", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{size}' for size in cache_sizes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved memory reduction visualization to {output_path}")
    plt.close()


def generate_all_visualizations():
    """Generate all visualizations."""
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60 + "\n")
    
    # Load traces
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
    
    # Memory reduction projection
    plot_memory_reduction()
    
    # POC results (if available)
    plot_eviction_comparison()
    
    print("\n" + "="*60)
    print("Visualizations Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - experiments/viz_trace_segments.png")
    print("  - experiments/viz_segment_dist.png")
    print("  - experiments/viz_trace_lengths.png")
    print("  - experiments/viz_memory_reduction.png")
    print("  - experiments/viz_eviction_comparison.png (after POC)")


if __name__ == "__main__":
    # Ensure matplotlib non-interactive backend
    import matplotlib
    matplotlib.use('Agg')
    
    generate_all_visualizations()
