"""Simple analysis of reasoning traces - minimal dependencies."""

import json
from pathlib import Path
from typing import List, Dict, Any


def analyze_trace(reasoning: str) -> Dict[str, Any]:
    """Analyze a single reasoning trace for characteristics."""
    tokens = reasoning.split()
    lines = [l for l in reasoning.split('\n') if l.strip()]
    
    return {
        'total_tokens': len(tokens),
        'num_lines': len(lines),
        'avg_line_length': sum(len(l.split()) for l in lines) / len(lines) if lines else 0,
        'punctuation_marks': reasoning.count('.') + reasoning.count('?'),
    }


def identify_segments(reasoning: str) -> List[Dict[str, Any]]:
    """Segment reasoning by type: rambling, exploration, insight, neutral."""
    lines = reasoning.split('\n')
    segments = []
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        line_lower = line.lower()
        
        # Simple heuristics for segment classification
        if any(phrase in line_lower for phrase in ['hmm', "i think", "let me", "wait", "actually", "i'm", "i was"]):
            seg_type = 'rambling'
        elif any(phrase in line_lower for phrase in ['?', 'what if', 'why', 'how', 'could']):
            seg_type = 'exploration'
        elif any(phrase in line_lower for phrase in ['thus', 'therefore', 'conclusion', 'answer', 'result', 'so', '=']):
            seg_type = 'insight'
        else:
            seg_type = 'neutral'
        
        segments.append({
            'line_idx': i,
            'type': seg_type,
            'text': line,
        })
    
    return segments


def analyze_dataset(traces: List[Dict[str, Any]]) -> None:
    """Summarize patterns across dataset."""
    token_lengths = []
    segment_types = {'rambling': 0, 'exploration': 0, 'insight': 0, 'neutral': 0}
    
    for trace in traces:
        reasoning = trace.get('reasoning', trace.get('thinking', ''))
        if not reasoning:
            continue
        
        tokens = reasoning.split()
        token_lengths.append(len(tokens))
        
        segments = identify_segments(reasoning)
        for seg in segments:
            segment_types[seg['type']] += 1
    
    avg_len = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    max_len = max(token_lengths) if token_lengths else 0
    min_len = min(token_lengths) if token_lengths else 0
    
    print(f"\n=== Dataset Summary ({len(traces)} traces) ===")
    print(f"Avg reasoning length: {avg_len:.0f} tokens (min: {min_len}, max: {max_len})")
    print(f"\nSegment type distribution:")
    total_segs = sum(segment_types.values())
    for seg_type, count in segment_types.items():
        pct = 100 * count / total_segs if total_segs > 0 else 0
        print(f"  {seg_type:12s}: {count:4d} ({pct:5.1f}%)")


# Synthetic reasoning traces
synthetic_traces = [
    {
        "question": "If a baker made 24 loaves of bread and sold 18, how many are left?",
        "reasoning": "Let me think about this. So the baker made 24 loaves total. Then sold 18 of them. So I need to subtract. 24 minus 18 is... let me count. 6. Wait, let me double-check. 18 plus 6 equals 24, yes. So the answer is 6.",
        "answer": "6"
    },
    {
        "question": "A train travels at 60 mph for 3 hours. How far does it travel?",
        "reasoning": "I need to find the distance. Distance = Speed × Time. Speed is 60 mph. Time is 3 hours. So distance = 60 × 3 = 180 miles. Let me verify: 60 × 3, that's 60 + 60 + 60 = 120 + 60 = 180. Yes, 180 miles. Therefore, the answer is 180.",
        "answer": "180 miles"
    },
    {
        "question": "What is 25% of 80?",
        "reasoning": "25% means 1/4. So I need to find 1/4 of 80. Hmm, let me think about this. 80 ÷ 4 = 20. Or I could calculate it as 80 × 0.25 = 20. Actually, wait, let me reconsider. If 100% is 80, then 25% is a quarter. A quarter of 80 is 20. Thus, 25% of 80 is 20.",
        "answer": "20"
    },
]

# Analysis
print("=" * 60)
print("Reasoning Trace Analysis")
print("=" * 60)

analyze_dataset(synthetic_traces)

# Detailed segment analysis
print("\n=== Detailed Segment Analysis ===\n")
for i, trace in enumerate(synthetic_traces):
    print(f"Example {i+1}: {trace['question'][:50]}...")
    segments = identify_segments(trace['reasoning'])
    for seg in segments:
        print(f"  Line {seg['line_idx']:2d} [{seg['type']:12s}]: {seg['text'][:65]}")
    print()

# Save traces
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
output_path = data_dir / "synthetic_math_traces.jsonl"
with open(output_path, 'w') as f:
    for trace in synthetic_traces:
        f.write(json.dumps(trace) + '\n')

print(f"Saved {len(synthetic_traces)} traces to {output_path}")
