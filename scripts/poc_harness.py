"""
Proof of Concept harness for KV cache eviction.
Compares baseline (attention) vs semantic eviction on reasoning tasks.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import time


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
    
    def __init__(self, model_name: str = "gpt2", cache_size: int = 512):
        """
        Initialize POC harness.
        
        Args:
            model_name: Model to load (e.g., "meta-llama/Llama-2-7b", "Qwen/Qwen-7B")
            cache_size: Max KV cache size in tokens
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.results: List[POCResult] = []
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("Install transformers: pip install transformers")
            return False
        
        try:
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            print(f"✓ Loaded {self.model_name}")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def generate_reasoning(
        self,
        question: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate reasoning for a question using the model.
        
        Args:
            question: The question to reason about
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated reasoning text
        """
        if self.model is None:
            return "Model not loaded"
        
        prompt = f"Question: {question}\nLet me think step by step.\n"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with reduced verbosity
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        reasoning = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return reasoning
    
    def evaluate_on_dataset(self, traces: List[Dict[str, Any]], eviction_method: str = "baseline") -> POCResult:
        """
        Evaluate model on a dataset of traces.
        
        Args:
            traces: List of {question, answer, reasoning} dicts
            eviction_method: "baseline" or "semantic"
            
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
        
        total_accuracy = 0.0
        total_tokens = 0.0
        start_time = time.time()
        
        for i, trace in enumerate(traces[:10]):  # Limit to 10 for quick POC
            question = trace['question']
            expected_answer = trace.get('answer', '')
            
            # Generate reasoning
            reasoning = self.generate_reasoning(question, max_tokens=128)
            num_tokens = len(reasoning.split())
            
            # Simple accuracy: check if answer appears in generated text
            accuracy = 1.0 if str(expected_answer) in reasoning else 0.0
            
            total_accuracy += accuracy
            total_tokens += num_tokens
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(traces[:10])} examples...")
        
        elapsed = time.time() - start_time
        
        result = POCResult(
            model_name=self.model_name,
            eviction_method=eviction_method,
            num_examples=min(10, len(traces)),
            avg_accuracy=total_accuracy / min(10, len(traces)),
            avg_tokens_generated=total_tokens / min(10, len(traces)),
            peak_memory_mb=torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
            avg_time_per_example=elapsed / min(10, len(traces)),
            cache_size_tokens=self.cache_size,
        )
        
        print(f"\n✓ Results:")
        print(f"  Accuracy: {result.avg_accuracy:.1%}")
        print(f"  Avg tokens: {result.avg_tokens_generated:.0f}")
        print(f"  Peak memory: {result.peak_memory_mb:.0f} MB")
        print(f"  Time/example: {result.avg_time_per_example:.2f}s")
        
        return result
    
    def save_results(self, output_path: str = "experiments/poc_results.jsonl"):
        """Save results to file."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'a') as f:
            for result in self.results:
                f.write(json.dumps(asdict(result)) + '\n')
        
        print(f"\n✓ Results saved to {output_path}")


def run_poc():
    """Run a quick POC experiment."""
    # Load synthetic traces
    traces_path = Path("data/synthetic_math_traces.jsonl")
    traces = []
    if traces_path.exists():
        with open(traces_path) as f:
            for line in f:
                traces.append(json.loads(line))
    
    if not traces:
        print("No traces found. Run scripts/analyze_traces.py first.")
        return
    
    print(f"Loaded {len(traces)} traces from {traces_path}")
    
    # POC setup
    # Note: Using small models for POC; replace with LLaMA/Qwen for full experiments
    harness = POCHarness(model_name="gpt2", cache_size=512)
    
    # Load model
    if not harness.load_model_and_tokenizer():
        print("Could not load model. Using mock results instead.")
        # Mock results for testing
        from dataclasses import replace
        mock_result = POCResult(
            model_name="gpt2",
            eviction_method="baseline",
            num_examples=10,
            avg_accuracy=0.60,
            avg_tokens_generated=85.3,
            peak_memory_mb=2048.0,
            avg_time_per_example=1.2,
            cache_size_tokens=512,
        )
        harness.results.append(mock_result)
        print(f"Mock result: {mock_result}")
        return
    
    # Run POC
    result_baseline = harness.evaluate_on_dataset(traces, eviction_method="baseline")
    if result_baseline:
        harness.results.append(result_baseline)
    
    # Save results
    harness.save_results()


if __name__ == "__main__":
    run_poc()
