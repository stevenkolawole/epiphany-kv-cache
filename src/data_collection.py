"""
Data collection and analysis for reasoning traces.

Focuses on DeepSeek and other OSS reasoning models.
Goals: Identify patterns in "rambling" vs "insights" for semantic importance scoring.
"""

import json
import requests
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


class DeepSeekTraceCollector:
    """Collect and analyze DeepSeek reasoning traces."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def fetch_from_huggingface(self, repo_id: str, subset: str = "full", split: str = "train", limit: int = 100):
        """
        Fetch reasoning traces from HuggingFace datasets.
        
        Args:
            repo_id: HuggingFace repo ID (e.g., "deepseek-ai/DeepSeek-Math" or similar)
            subset: Dataset subset/config
            split: Dataset split (train/test/validation)
            limit: Max number of examples to fetch
            
        Returns:
            List of trace dictionaries with 'question', 'reasoning', 'answer' fields
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("Install datasets: pip install datasets")
            return []
        
        try:
            print(f"Loading {repo_id} ({subset}/{split}, limit={limit})...")
            dataset = load_dataset(repo_id, subset, split=split, streaming=True)
            traces = []
            
            for i, example in enumerate(dataset):
                if i >= limit:
                    break
                traces.append(example)
            
            print(f"Loaded {len(traces)} traces")
            return traces
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
    
    def save_traces(self, traces: List[Dict[str, Any]], name: str):
        """Save traces to JSONL."""
        output_path = self.data_dir / f"{name}_traces.jsonl"
        with open(output_path, 'w') as f:
            for trace in traces:
                f.write(json.dumps(trace) + '\n')
        print(f"Saved {len(traces)} traces to {output_path}")
    
    def load_traces(self, name: str) -> List[Dict[str, Any]]:
        """Load traces from JSONL."""
        path = self.data_dir / f"{name}_traces.jsonl"
        traces = []
        if path.exists():
            with open(path, 'r') as f:
                for line in f:
                    traces.append(json.loads(line))
        return traces


class TraceAnalyzer:
    """Analyze reasoning traces for semantic patterns."""
    
    def __init__(self):
        pass
    
    def analyze_trace(self, reasoning: str) -> Dict[str, Any]:
        """
        Analyze a single reasoning trace for characteristics.
        
        Args:
            reasoning: The reasoning/thinking section as text
            
        Returns:
            Dictionary with analysis metrics
        """
        tokens = reasoning.split()
        
        analysis = {
            'total_tokens': len(tokens),
            'lines': len(reasoning.split('\n')),
            'avg_line_length': np.mean([len(line.split()) for line in reasoning.split('\n') if line.strip()]),
            'punctuation_density': reasoning.count('.') + reasoning.count('?'),
        }
        
        return analysis
    
    def identify_segments(self, reasoning: str) -> List[Dict[str, Any]]:
        """
        Segment reasoning into potential "rambling", "exploration", "insight" phases.
        
        Uses heuristics:
        - High repetition or backtracking = rambling
        - Questions/exploration = exploration
        - Assertions/conclusions = insights
        
        Args:
            reasoning: The reasoning text
            
        Returns:
            List of segments with (start, end, type, text)
        """
        lines = reasoning.split('\n')
        segments = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Simple heuristics
            if any(phrase in line_lower for phrase in ['i think', 'let me', 'hmm', 'wait', 'actually']):
                seg_type = 'rambling'
            elif any(phrase in line_lower for phrase in ['?', 'what if', 'why', 'how']):
                seg_type = 'exploration'
            elif any(phrase in line_lower for phrase in ['thus', 'therefore', 'conclusion', 'answer is', 'result']):
                seg_type = 'insight'
            else:
                seg_type = 'neutral'
            
            segments.append({
                'line_idx': i,
                'type': seg_type,
                'text': line,
            })
        
        return segments
    
    def summarize_dataset(self, traces: List[Dict[str, Any]], keys_to_check: List[str] = ['reasoning', 'thinking', 'solution']):
        """
        Summarize patterns across a dataset of traces.
        
        Args:
            traces: List of trace dictionaries
            keys_to_check: Fields to look for reasoning text
        """
        total_traces = len(traces)
        token_lengths = []
        segment_stats = {'rambling': 0, 'exploration': 0, 'insight': 0, 'neutral': 0}
        
        for trace in traces:
            # Find reasoning field
            reasoning_text = None
            for key in keys_to_check:
                if key in trace:
                    reasoning_text = str(trace[key])
                    break
            
            if not reasoning_text:
                continue
            
            token_lengths.append(len(reasoning_text.split()))
            segments = self.identify_segments(reasoning_text)
            for seg in segments:
                segment_stats[seg['type']] += 1
        
        print(f"\n=== Dataset Summary ({total_traces} traces) ===")
        print(f"Avg reasoning length: {np.mean(token_lengths):.0f} tokens (std: {np.std(token_lengths):.0f})")
        print(f"Min/Max reasoning length: {min(token_lengths):.0f} / {max(token_lengths):.0f}")
        print(f"\nSegment type distribution:")
        for seg_type, count in segment_stats.items():
            pct = 100 * count / sum(segment_stats.values()) if sum(segment_stats.values()) > 0 else 0
            print(f"  {seg_type}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    # Example: Fetch and analyze DeepSeek traces
    collector = DeepSeekTraceCollector(data_dir="data")
    analyzer = TraceAnalyzer()
    
    # For now, use synthetic reasoning traces for testing
    # (datasets.load_dataset can take a long time on first run)
    print("\n=== Using synthetic traces for initial analysis ===\n")
    
    synthetic_traces = [
        {
            "question": "If a baker made 24 loaves of bread and sold 18, how many are left?",
            "reasoning": "Let me think about this. So the baker made 24 loaves total. Then sold 18 of them. So I need to subtract. 24 minus 18 is... let me count on my fingers... 6. Wait, let me double-check. 18 plus 6 equals 24, yes. So the answer is 6.",
            "answer": "6"
        },
        {
            "question": "A train travels at 60 mph for 3 hours. How far does it travel?",
            "reasoning": "I need to find the distance. Distance = Speed × Time. Speed is 60 mph. Time is 3 hours. So distance = 60 × 3 = 180 miles. Let me verify: 60 × 3, that's 60 + 60 + 60 = 120 + 60 = 180. Yes, 180 miles.",
            "answer": "180 miles"
        },
        {
            "question": "What is 25% of 80?",
            "reasoning": "25% means 1/4. So I need to find 1/4 of 80. 80 ÷ 4 = 20. Or I could calculate it as 80 × 0.25 = 20. Actually, let me think about this differently. If 100% is 80, then 25% is a quarter. A quarter of 80 is... hmm, let me see. 80/4 is definitely 20. So 25% of 80 is 20.",
            "answer": "20"
        },
    ]
    
    # Save and analyze
    collector.save_traces(synthetic_traces, "synthetic_math")
    print("\nAnalyzing synthetic traces...\n")
    analyzer.summarize_dataset(synthetic_traces, keys_to_check=['reasoning'])
    
    # Load and show detailed segment analysis
    print("\n=== Detailed Segment Analysis ===\n")
    for i, trace in enumerate(synthetic_traces[:2]):
        print(f"Example {i+1}: {trace['question'][:50]}...")
        segments = analyzer.identify_segments(trace['reasoning'])
        for seg in segments:
            print(f"  Line {seg['line_idx']:2d} [{seg['type']:12s}]: {seg['text'][:60]}")
        print()
