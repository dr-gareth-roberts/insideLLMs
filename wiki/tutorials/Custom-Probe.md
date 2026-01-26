---
title: Custom Probe
parent: Tutorials
nav_order: 4
---

# Custom Probe Tutorial

Build your own evaluation probe for domain-specific testing.

**Time:** 25 minutes  
**Prerequisites:** Python basics, [First Run](../getting-started/First-Run.md) completed

---

## Goal

By the end of this tutorial, you'll:
- Understand the Probe interface
- Create a custom probe class
- Add scoring logic
- Register and use your probe

---

## Step 1: Understand the Probe Interface

Every probe implements this interface:

```python
class Probe:
    name: str           # Unique identifier
    category: ProbeCategory  # LOGIC, BIAS, SAFETY, etc.
    
    def run(self, model, data, **kwargs) -> Any:
        """Execute the probe on a single input."""
        pass
    
    def score(self, results) -> ProbeScore:
        """Aggregate results into a score (optional)."""
        pass
```

The `run` method is called for each item in your dataset.

---

## Step 2: Create a Simple Probe

Let's create a probe that tests if models follow word count constraints.

Create `word_count_probe.py`:

```python
from insideLLMs.probes.base import Probe
from insideLLMs.types import ProbeCategory, ProbeScore

class WordCountProbe(Probe[dict]):
    """Tests if model responses meet word count constraints."""
    
    name = "word_count"
    default_category = ProbeCategory.CUSTOM
    
    def __init__(self, name: str = "word_count", **kwargs):
        super().__init__(name=name, **kwargs)
    
    def run(self, model, data, **kwargs) -> dict:
        """
        Args:
            model: The model to test
            data: Dict with 'prompt' and optional 'min_words'/'max_words'
        
        Returns:
            Dict with response and word count analysis
        """
        prompt = data.get("prompt", str(data))
        min_words = data.get("min_words", 0)
        max_words = data.get("max_words", float("inf"))
        
        # Add constraint to prompt
        constraint = f"Respond in {min_words}-{max_words} words."
        full_prompt = f"{prompt}\n\n{constraint}"
        
        # Get model response
        response = model.generate(full_prompt, **kwargs)
        
        # Analyze
        word_count = len(response.split())
        meets_constraint = min_words <= word_count <= max_words
        
        return {
            "response": response,
            "word_count": word_count,
            "min_words": min_words,
            "max_words": max_words,
            "meets_constraint": meets_constraint
        }
    
    def score(self, results) -> ProbeScore:
        """Calculate constraint compliance rate."""
        if not results:
            return ProbeScore(value=0.0)
        
        compliant = sum(
            1 for r in results 
            if r.status.value == "success" 
            and r.output.get("meets_constraint", False)
        )
        
        return ProbeScore(
            value=compliant / len(results),
            details={
                "compliant": compliant,
                "total": len(results),
                "compliance_rate": compliant / len(results)
            }
        )
```

---

## Step 3: Test Your Probe

Create a test script `test_word_count.py`:

```python
from insideLLMs.models import DummyModel
from insideLLMs.runtime.runner import ProbeRunner
from word_count_probe import WordCountProbe

# Create instances
model = DummyModel()
probe = WordCountProbe()

# Test data
test_data = [
    {"prompt": "Explain gravity", "min_words": 10, "max_words": 50},
    {"prompt": "What is Python?", "min_words": 20, "max_words": 100},
    {"prompt": "Say hello", "min_words": 1, "max_words": 5},
]

# Run
runner = ProbeRunner(model, probe)
results = runner.run(test_data)

# Print results
for i, result in enumerate(results):
    print(f"\nTest {i+1}:")
    print(f"  Word count: {result['output']['word_count']}")
    print(f"  Constraint: {result['output']['min_words']}-{result['output']['max_words']}")
    print(f"  Meets constraint: {result['output']['meets_constraint']}")

print(f"\nSuccess rate: {runner.success_rate:.1%}")
```

Run it:

```bash
python test_word_count.py
```

---

## Step 4: Register Your Probe

To use your probe in configs and CLI, register it:

```python
from insideLLMs.registry import probe_registry
from word_count_probe import WordCountProbe

# Register the probe
probe_registry.register("word_count", WordCountProbe)

# Now you can use it by name
probe = probe_registry.get("word_count")
```

For automatic registration, create a plugin entry point in `pyproject.toml`:

```toml
[project.entry-points."insidellms.plugins"]
my_probes = "my_package:register_probes"
```

And in `my_package/__init__.py`:

```python
def register_probes(probe_registry):
    from .word_count_probe import WordCountProbe
    probe_registry.register("word_count", WordCountProbe)
```

---

## Step 5: Use in a Harness Config

Create `word_count_harness.yaml`:

```yaml
models:
  - type: openai
    args:
      model_name: gpt-4o-mini

probes:
  - type: word_count  # Your custom probe!
    args: {}

dataset:
  format: inline
  items:
    - prompt: "Explain machine learning"
      min_words: 50
      max_words: 100
    - prompt: "What is a neural network?"
      min_words: 30
      max_words: 75
    - prompt: "Define AI briefly"
      min_words: 5
      max_words: 20

output_dir: ./word_count_results
```

Run with the probe registered:

```python
# register_and_run.py
from insideLLMs.registry import probe_registry
from word_count_probe import WordCountProbe

# Register
probe_registry.register("word_count", WordCountProbe)

# Run harness
from insideLLMs.runtime.runner import run_harness_from_config
run_harness_from_config("word_count_harness.yaml")
```

---

## Step 6: Add More Sophisticated Scoring

Enhance the probe with detailed metrics:

```python
def score(self, results) -> ProbeScore:
    """Calculate detailed metrics."""
    if not results:
        return ProbeScore(value=0.0)
    
    successful = [r for r in results if r.status.value == "success"]
    
    if not successful:
        return ProbeScore(value=0.0, details={"error": "no successful runs"})
    
    # Calculate metrics
    compliant = sum(1 for r in successful if r.output.get("meets_constraint"))
    
    word_counts = [r.output["word_count"] for r in successful]
    avg_word_count = sum(word_counts) / len(word_counts)
    
    over_limit = sum(
        1 for r in successful 
        if r.output["word_count"] > r.output["max_words"]
    )
    under_limit = sum(
        1 for r in successful 
        if r.output["word_count"] < r.output["min_words"]
    )
    
    return ProbeScore(
        value=compliant / len(successful),
        details={
            "compliance_rate": compliant / len(successful),
            "avg_word_count": avg_word_count,
            "over_limit": over_limit,
            "under_limit": under_limit,
            "total_tested": len(successful)
        }
    )
```

---

## Common Patterns

### Comparative Probe

Compare model output against a reference:

```python
class ExactMatchProbe(Probe[dict]):
    def run(self, model, data, **kwargs) -> dict:
        response = model.generate(data["prompt"])
        expected = data["expected"]
        
        return {
            "response": response,
            "expected": expected,
            "exact_match": response.strip().lower() == expected.strip().lower()
        }
```

### Multi-Turn Probe

Test conversation handling:

```python
class ConversationProbe(Probe[dict]):
    def run(self, model, data, **kwargs) -> dict:
        messages = data["messages"]
        responses = []
        
        for msg in messages:
            response = model.chat(messages[:messages.index(msg)+1])
            responses.append(response)
        
        return {"responses": responses, "turn_count": len(responses)}
```

---

## Verification

✅ Created custom probe class  
✅ Implemented run method  
✅ Added scoring logic  
✅ Registered probe  
✅ Used in harness config  

---

## What's Next?

- [Probes Catalog](../reference/Probes-Catalog.md) - See built-in probe implementations
- [API Reference](../reference/API.md) - Full Probe API documentation
- [Plugins Guide](../guides/Plugins.md) - Package and distribute your probe
