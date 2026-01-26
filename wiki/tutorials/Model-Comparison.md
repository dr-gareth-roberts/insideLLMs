---
title: Model Comparison
parent: Tutorials
nav_order: 2
---

# Model Comparison Tutorial

**Compare GPT-4 vs Claude on your prompts.**

**Time:** 20 minutes
**Prerequisites:** API keys

---

## Step 1: Create Dataset

```bash
cat > test.jsonl << 'EOF'
{"question": "Explain quantum entanglement simply.", "category": "explanation"}
{"question": "Write a haiku about programming.", "category": "creative"}
{"question": "What is 15% of 240?", "category": "math"}
{"question": "Debug: for i in range(10) print(i)", "category": "code"}
EOF
```

---

## Step 2: Configure

```yaml
# comparison.yaml
models:
  - type: openai
    args: {model_name: gpt-4o}
  - type: anthropic
    args: {model_name: claude-3-5-sonnet-20241022}
probes:
  - type: logic
  - type: factuality
dataset:
  format: jsonl
  path: test.jsonl
async: true
concurrency: 5
```

## Step 3: Run

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
insidellms harness comparison.yaml
```

---

## Step 6: Analyse Results

### View Summary Statistics

```bash
cat comparison_results/summary.json | python -m json.tool
```

```json
{
  "models": {
    "gpt-4o": {
      "success_rate": 1.0,
      "example_count": 16
    },
    "gpt-4o-mini": {
      "success_rate": 0.9375,
      "example_count": 16
    },
    "claude-3-5-sonnet-20241022": {
      "success_rate": 1.0,
      "example_count": 16
    }
  }
}
```

### View HTML Report

Open `comparison_results/report.html` for:
- Side-by-side response comparison
- Success rate by model and probe
- Individual response inspection
- Filter by category

### Analyse by Category

```python
import json
from collections import defaultdict

records = []
with open("comparison_results/records.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

# Group by category and model
by_category = defaultdict(lambda: defaultdict(list))
for r in records:
    category = r["input"].get("category", "unknown")
    model_id = r["model"]["model_id"]
    by_category[category][model_id].append(r)

# Print success rates by category
for category, models in by_category.items():
    print(f"\n{category.upper()}")
    for model_id, results in models.items():
        success = sum(1 for r in results if r["status"] == "success")
        print(f"  {model_id}: {success}/{len(results)}")
```

---

## Step 7: Compare Response Quality

For qualitative comparison, extract and compare responses:

```python
# Compare responses for a specific question
question = "Explain quantum entanglement in simple terms."

for r in records:
    if r["input"]["question"] == question:
        print(f"\n=== {r['model']['model_id']} ===")
        print(r["output"][:500])
```

---

## Step 8: Generate Comparison Report

Create a markdown summary:

```python
# Generate comparison report
with open("comparison_report.md", "w") as f:
    f.write("# Model Comparison Report\n\n")
    f.write("## Overall Success Rates\n\n")
    f.write("| Model | Success Rate |\n")
    f.write("|-------|-------------|\n")

    for model_id in set(r["model"]["model_id"] for r in records):
        model_records = [r for r in records if r["model"]["model_id"] == model_id]
        success = sum(1 for r in model_records if r["status"] == "success")
        rate = success / len(model_records)
        f.write(f"| {model_id} | {rate:.1%} |\n")

print("Report written to comparison_report.md")
```

---

## Advanced: Diff Between Models

Compare specific model outputs:

```bash
# Extract single-model runs
insidellms harness model_comparison.yaml \
  --model-filter gpt-4o \
  --run-dir ./gpt4o_only

insidellms harness model_comparison.yaml \
  --model-filter claude-3-5-sonnet \
  --run-dir ./claude_only

# Diff the outputs
insidellms diff ./gpt4o_only ./claude_only --output model_diff.json
```

---

## Verification

 Configured multi-model harness
 Ran comparison across models and probes
 Analysed success rates by model
 Compared response quality
 Generated comparison report

---

## What's Next?

- [CI Integration Tutorial](CI-Integration.md) - Automate comparisons in CI
- [Custom Probe Tutorial](Custom-Probe.md) - Create domain-specific evaluations
- [Performance and Caching](../Performance-and-Caching.md) - Speed up large comparisons
