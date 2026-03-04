"""Simple analysis of reasoning traces - minimal dependencies."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Segment classification helpers
# ---------------------------------------------------------------------------

# Compiled patterns with word boundaries — avoids false positives like
# "how" matching "however" or "so" matching "also".
_RAMBLING = re.compile(
    r'\b(i think|let me|hmm|wait|actually|i\'m|i was)\b', re.IGNORECASE
)
_EXPLORATION = re.compile(
    r'\?|\b(what if|why|how|could)\b', re.IGNORECASE
)
_INSIGHT = re.compile(
    r'\b(thus|therefore|conclusion|answer|result)\b', re.IGNORECASE
)


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
            
        # Priority: rambling > exploration > insight > neutral
        if _RAMBLING.search(line):
            seg_type = 'rambling'
        elif _EXPLORATION.search(line):
            seg_type = 'exploration'
        elif _INSIGHT.search(line):
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


# Synthetic reasoning traces (expanded to 10 examples for better analysis)
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
    {
        "question": "Solve: 2x + 3 = 7",
        "reasoning": "I have the equation 2x + 3 = 7. I need to solve for x. First, subtract 3 from both sides. 2x + 3 - 3 = 7 - 3. So 2x = 4. Now divide both sides by 2. x = 4 ÷ 2 = 2. Let me check: 2×2 + 3 = 4 + 3 = 7. Yes, correct.",
        "answer": "x = 2"
    },
    {
        "question": "If 5 apples cost $2, how much do 12 apples cost?",
        "reasoning": "First, find the cost per apple. 5 apples cost $2, so one apple costs $2 ÷ 5 = $0.40. Now, 12 apples would cost 12 × $0.40 = $4.80. Let me double-check the calculation. 12 × 0.40 = 4.80, yes. So the answer is $4.80.",
        "answer": "$4.80"
    },
    {
        "question": "What is the area of a rectangle with length 8 cm and width 5 cm?",
        "reasoning": "The area of a rectangle is length × width. So, 8 cm × 5 cm = 40 cm². That's straightforward. Let me confirm: 8 × 5 = 40, yes. The area is 40 square centimeters.",
        "answer": "40 cm²"
    },
    {
        "question": "If a box contains 24 balls and you take out 9, how many are left?",
        "reasoning": "There are 24 balls in the box. I take out 9. So I need to subtract. 24 - 9 = 15. Let me count backwards to verify: 24, 23, 22, 21, 20, 19, 18, 17, 16, 15. Yes, 15 balls left.",
        "answer": "15"
    },
    {
        "question": "Convert 3.5 hours to minutes.",
        "reasoning": "1 hour = 60 minutes. So 3.5 hours = 3.5 × 60. First, 3 × 60 = 180. Then 0.5 × 60 = 30. So 180 + 30 = 210 minutes. Let me think: 3 hours is 180 minutes, half an hour is 30 minutes, total 210. Yes.",
        "answer": "210 minutes"
    },
    {
        "question": "What is 15% of 200?",
        "reasoning": "15% means 15/100 = 0.15. So 0.15 × 200. First, 0.1 × 200 = 20. Then 0.05 × 200 = 10. So 20 + 10 = 30. Alternatively, 15% of 200 is (15/100) × 200 = (15 × 200) / 100 = 3000 / 100 = 30. Yes, 30.",
        "answer": "30"
    },
    {
        "question": "If you have $50 and spend $27, how much do you have left?",
        "reasoning": "I start with $50. I spend $27. So I subtract. 50 - 27 = 23. Let me check: 27 + 23 = 50, yes. So I have $23 left. That seems correct.",
        "answer": "$23"
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
