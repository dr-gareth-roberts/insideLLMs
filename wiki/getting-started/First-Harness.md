---
title: First Harness
parent: Getting Started
nav_order: 3
---

# First Harness

**10 minutes. Compare two models.**

The harness runs identical tests across multiple models. Output: side-by-side comparison.

## Config

```yaml
# my_harness.yaml
models:
  - type: dummy
    args: {name: baseline}
  - type: dummy
    args: {name: candidate}

probes:
  - type: logic

dataset:
  format: inline
  items:
    - question: "What is 2 + 2?"
    - question: "If A > B and B > C, is A > C?"

output_dir: ./harness_results
```

## Run

```bash
insidellms harness my_harness.yaml
# Creates: records.jsonl (4 records), summary.json, report.html
```

## View Results

```bash
# Raw records
wc -l harness_results/records.jsonl
# 4 (2 models × 2 examples)

# HTML report
open harness_results/report.html
```

## Real Models

```yaml
models:
  - type: openai
    args: {model_name: gpt-4o}
  - type: anthropic
    args: {model_name: claude-3-5-sonnet-20241022}
probes:
  - type: logic
  - type: bias
dataset:
  format: jsonl
  path: data/test.jsonl
```

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
insidellms harness real_harness.yaml
```

## Common Options

```bash
--async --concurrency 10    # Parallel execution
--max-examples 50           # Limit dataset
--overwrite                 # Replace existing
```

## Next

[Understanding Outputs →](Understanding-Outputs.md) Learn what each artefact contains.
