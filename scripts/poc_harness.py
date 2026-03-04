"""
Proof of Concept harness for KV cache eviction.
Compares baseline (attention) vs semantic eviction on reasoning tasks.
"""

import re
import sys
import torch
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import time

# Allow running from the project root as well as from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eviction import AttentionBasedEviction, SemanticEviction, EvictionConfig


@dataclass
class POCResult:
    """Results from a POC run."""
    model_name: str
    eviction_method: str
    num_examples: int
    avg_accuracy: float
    avg_tokens_generated: float
    peak_memory_mb: float
    avg_time_per_example: float
    cache_size_tokens: int


class POCHarness:
    """Harness for running POC experiments."""

    MODEL_VARIANTS = {
        'llama3_vanilla': 'meta-llama/Llama-3.1-8B-Instruct',
        'llama3_reasoning': 'deepseek-ai/deepseek-r1-distill-llama-8b',
        'qwen_vanilla': 'Qwen/Qwen2-7B-Instruct',
        'qwen_reasoning': 'deepseek-ai/deepseek-r1-distill-qwen-7b',
        'mistral_reasoning': 'mistralai/Mistral-7B-Instruct-v0.3',
        'gpt2_test': 'gpt2',
    }

    def __init__(self, model_name: str = "gpt2", variant: str = None, cache_size: int = 512):
        if variant and variant in self.MODEL_VARIANTS:
            self.model_name = self.MODEL_VARIANTS[variant]
            self.variant_name = variant
        else:
            self.model_name = model_name
            self.variant_name = None

        self.cache_size = cache_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.results: List[POCResult] = []

        eviction_config = EvictionConfig(cache_size=cache_size)
        self.attention_eviction = AttentionBasedEviction(eviction_config)
        self.semantic_eviction = SemanticEviction(eviction_config)

    def load_model_and_tokenizer(self) -> bool:
        """Load model and tokenizer from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("Install transformers: pip install transformers")
            return False

        try:
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model.eval()
            print(f"✓ Loaded {self.model_name}")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False

    def generate_reasoning(
        self,
        question: str,
        max_tokens: int = 256,
        eviction_method: str = "none",
    ) -> str:
        """
        Generate reasoning with a manual step-by-step loop so that KV cache
        eviction can be applied between decoding steps.

        Args:
            question: The question to reason about
            max_tokens: Maximum new tokens to generate
            eviction_method: "baseline", "semantic", or "none"

        Returns:
            Full decoded text (prompt + generated tokens)
        """
        if self.model is None:
            return "Model not loaded"

        prompt = f"Question: {question}\nLet me think step by step.\n"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # Flags: only request these when the eviction method actually needs them,
        # because output_attentions=True can be expensive for large models.
        need_attentions = eviction_method == "baseline"
        need_hidden = eviction_method == "semantic"

        past_key_values = None
        generated_ids: List[int] = []

        with torch.no_grad():
            for step in range(max_tokens):
                if step == 0:
                    current_input = input_ids
                    current_mask = attention_mask
                else:
                    # Only feed the most recent token; the rest is in past_key_values
                    current_input = torch.tensor(
                        [[generated_ids[-1]]], dtype=torch.long, device=self.device
                    )
                    current_mask = None

                outputs = self.model(
                    input_ids=current_input,
                    attention_mask=current_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=need_attentions,
                    output_hidden_states=need_hidden,
                )

                # Greedy decoding
                next_token_id = int(outputs.logits[:, -1, :].argmax(dim=-1))
                generated_ids.append(next_token_id)

                past_key_values = outputs.past_key_values
                current_cache_len = past_key_values[0][0].shape[2]

                # Apply eviction once the cache exceeds the budget
                if current_cache_len > self.cache_size:
                    if eviction_method == "baseline":
                        past_key_values = self.attention_eviction.evict_past_key_values(
                            past_key_values, outputs.attentions
                        )
                    elif eviction_method == "semantic":
                        past_key_values = self.semantic_eviction.evict_past_key_values(
                            past_key_values,
                            hidden_states=outputs.hidden_states,
                            attention_weights=outputs.attentions,
                        )

                if next_token_id == self.tokenizer.eos_token_id:
                    break

        all_ids = input_ids[0].tolist() + generated_ids
        return self.tokenizer.decode(all_ids, skip_special_tokens=True)

    @staticmethod
    def _check_answer(expected_answer: str, generated_text: str) -> float:
        """
        Check whether the expected answer appears in generated text.

        Uses a word-boundary pattern so "2" does not falsely match "12" or "20".
        Falls back to a simple containment check only when the answer string
        contains characters that make word-boundary matching ambiguous
        (e.g. "$4.80" — the dollar sign sits at a natural boundary).
        """
        answer = str(expected_answer).strip()
        text = generated_text.lower()
        answer_lower = answer.lower()

        # Try a word-boundary regex first
        try:
            pattern = re.compile(r'(?<!\w)' + re.escape(answer_lower) + r'(?!\w)')
            return 1.0 if pattern.search(text) else 0.0
        except re.error:
            # Fallback for pathological answer strings
            return 1.0 if answer_lower in text else 0.0

    def evaluate_on_dataset(
        self,
        traces: List[Dict[str, Any]],
        eviction_method: str = "none",
        max_examples: int = 50,
    ) -> Optional[POCResult]:
        """
        Evaluate model on a dataset of reasoning traces.

        Args:
            traces: List of {question, answer, reasoning} dicts
            eviction_method: "baseline", "semantic", or "none"
            max_examples: Maximum number of examples to evaluate

        Returns:
            POCResult with metrics
        """
        if self.model is None:
            print("Model not loaded. Call load_model_and_tokenizer() first.")
            return None

        print(f"\n{'='*60}")
        print(f"POC Evaluation: {eviction_method.upper()}")
        print(f"Model: {self.model_name}, Cache size: {self.cache_size}")
        print(f"{'='*60}\n")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        total_accuracy = 0.0
        total_tokens = 0.0
        num_examples = min(max_examples, len(traces))
        start_time = time.time()

        for i, trace in enumerate(traces[:num_examples]):
            question = trace['question']
            expected_answer = trace.get('answer', '')

            reasoning = self.generate_reasoning(
                question, max_tokens=256, eviction_method=eviction_method
            )
            num_tokens = len(reasoning.split())
            accuracy = self._check_answer(expected_answer, reasoning)

            total_accuracy += accuracy
            total_tokens += num_tokens

            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{num_examples} examples...")

        elapsed = time.time() - start_time
        peak_memory = (
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else 0.0
        )

        result = POCResult(
            model_name=self.model_name,
            eviction_method=eviction_method,
            num_examples=num_examples,
            avg_accuracy=total_accuracy / num_examples,
            avg_tokens_generated=total_tokens / num_examples,
            peak_memory_mb=peak_memory,
            avg_time_per_example=elapsed / num_examples,
            cache_size_tokens=self.cache_size,
        )

        print(f"\n✓ Results:")
        print(f"  Accuracy:      {result.avg_accuracy:.1%}")
        print(f"  Avg tokens:    {result.avg_tokens_generated:.0f}")
        print(f"  Peak memory:   {result.peak_memory_mb:.0f} MB")
        print(f"  Time/example:  {result.avg_time_per_example:.2f}s")

        return result

    def save_results(self, output_path: str = "experiments/poc_results.jsonl"):
        """Append results to JSONL file."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'a') as f:
            for result in self.results:
                f.write(json.dumps(asdict(result)) + '\n')
        print(f"\n✓ Results saved to {output_path}")


def run_poc(
    model_variants: List[str] = None,
    eviction_methods: List[str] = None,
    max_examples: int = 10,
):
    """
    Run POC experiments comparing model variants and eviction methods.

    Args:
        model_variants: List of variant keys from POCHarness.MODEL_VARIANTS.
                        Defaults to ['gpt2_test'] for a quick smoke test.
        eviction_methods: Eviction methods to compare. Defaults to ['none', 'baseline', 'semantic'].
        max_examples: Number of evaluation examples per method.
    """
    if model_variants is None:
        model_variants = ['gpt2_test']
    if eviction_methods is None:
        eviction_methods = ['none', 'baseline', 'semantic']

    traces_path = Path("data/synthetic_math_traces.jsonl")
    traces = []
    if traces_path.exists():
        with open(traces_path) as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))

    if not traces:
        print("No traces found. Run scripts/analyze_traces.py first.")
        return

    print(f"Loaded {len(traces)} traces")
    print(f"Testing models: {model_variants}")
    print(f"Testing eviction methods: {eviction_methods}\n")

    for variant in model_variants:
        print(f"\n{'='*60}")
        print(f"Testing: {variant}")
        print(f"{'='*60}\n")

        harness = POCHarness(variant=variant, cache_size=512)

        if not harness.load_model_and_tokenizer():
            print(f"  Skipping {variant}: model failed to load. No results recorded.")
            print("  Tip: ensure you have access to the model and enough GPU memory.")
            continue

        for method in eviction_methods:
            result = harness.evaluate_on_dataset(
                traces, eviction_method=method, max_examples=max_examples
            )
            if result:
                harness.results.append(result)

        harness.save_results()


if __name__ == "__main__":
    run_poc()
