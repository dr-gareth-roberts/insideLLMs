---
title: First Harness
parent: Getting Started
nav_order: 3
---

# First Harness

Compare multiple models side-by-side using the harness. This guide takes about 15 minutes.

## What is a Harness?

The harness runs the same probes across multiple models and datasets in a single command. It produces:

- `records.jsonl` - Every model×probe×example result
- `summary.json` - Aggregated metrics with confidence intervals
- `report.html` - Human-readable comparison report

## Create a Harness Config

Create `my_harness.yaml`:

```yaml
# my_harness.yaml
models:
  - type: dummy
    args:
      name: "baseline"
  - type: dummy
    args:
      name: "candidate"

probes:
  - type: logic
    args: {}

dataset:
  format: inline
  items:
    - question: "What is 2 + 2?"
      expected: "4"
    - question: "What comes next: 1, 4, 9, 16, ?"
      expected: "25"
    - question: "If A > B and B > C, is A > C?"
      expected: "yes"

output_dir: ./harness_results
```

## Run the Harness

```bash
insidellms harness my_harness.yaml
```

**Expected output:**

```
Running harness with 2 models, 1 probe, 3 examples
Processing: baseline × logic [████████████████████] 3/3
Processing: candidate × logic [████████████████████] 3/3

Results written to: ./harness_results/
- records.jsonl (6 records)
- summary.json
- report.html
```

## Inspect the Results

### Records (raw data)

```bash
wc -l harness_results/records.jsonl
# 6 (2 models × 3 examples)
```

### Summary (aggregated metrics)

```bash
cat harness_results/summary.json | python -m json.tool
```

```json
{
  "models": {
    "baseline": {
      "success_rate": 1.0,
      "example_count": 3
    },
    "candidate": {
      "success_rate": 1.0,
      "example_count": 3
    }
  }
}
```

### Report (visual comparison)

Open `harness_results/report.html` in your browser to see:

- Side-by-side model comparison
- Success/failure breakdown by probe
- Individual response inspection

## With Real Models

To compare GPT-4 and Claude:

```yaml
# real_harness.yaml
models:
  - type: openai
    args:
      model_name: gpt-4o
  - type: anthropic
    args:
      model_name: claude-3-5-sonnet-20241022

probes:
  - type: logic
  - type: factuality

dataset:
  format: jsonl
  path: data/questions.jsonl

output_dir: ./real_comparison
```

Set API keys and run:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
insidellms harness real_harness.yaml
```

## Harness Options

| Option | Description | Example |
|--------|-------------|---------|
| `--async` | Run concurrently | `--async --concurrency 10` |
| `--max-examples` | Limit dataset size | `--max-examples 50` |
| `--overwrite` | Replace existing output | `--overwrite` |
| `--skip-report` | Skip HTML generation | `--skip-report` |
| `--run-dir` | Custom output directory | `--run-dir ./my_run` |

## What's Next?

You've successfully:
- ✅ Created a harness configuration
- ✅ Compared multiple models
- ✅ Generated comparison reports

**Next steps:**

- [Understanding Outputs](Understanding-Outputs.md) - Learn what each artifact contains
- [CI Integration Tutorial](../tutorials/CI-Integration.md) - Use harness for regression testing
- [Model Comparison Tutorial](../tutorials/Model-Comparison.md) - Advanced comparison techniques
