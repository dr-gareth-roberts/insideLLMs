---
title: First Harness
parent: Getting Started
nav_order: 3
---

# First Harness

**10 minutes. Run and diff a complete offline matrix.**

The harness runs every configured model/probe combination against the same
dataset and produces canonical records plus a comparison report.

## Generate the Config and Dataset

Run the initializer from the directory where you want to keep the example:

```bash
insidellms init harness.yaml --template harness
```

It creates `harness.yaml` and `data/harness_dataset.jsonl`. The generated
harness uses one DummyModel, four probes, and three examples, so it performs 12
offline evaluations. The `--model` and `--probe` initializer options do not
customize the `harness` template; edit the generated YAML to change its matrix.

Keep `harness.yaml` in this directory. The sample data is created relative to
your current directory, while the path in YAML is resolved relative to the
config file.

## Check the Plan, Then Run

```bash
insidellms harness harness.yaml --dry-run
insidellms harness harness.yaml --run-dir ./runs/baseline
# Creates records.jsonl (12 records), summary.json, report.html, and metadata
```

## View Results

```bash
# Raw records
wc -l runs/baseline/records.jsonl
# 12 (1 model × 4 probes × 3 examples)

# HTML report
open runs/baseline/report.html
```

To exercise the comparison workflow, produce another snapshot and diff it:

```bash
insidellms harness harness.yaml --run-dir ./runs/candidate
insidellms diff ./runs/baseline ./runs/candidate
```

Plain `diff` reports differences but does not fail on them. Use
`--fail-on-changes` to return exit code 2 for regressions, other changes, or
records present on only one side. Improvements alone remain informational;
trace and trajectory findings have dedicated gate flags.

## Real Models

```yaml
models:
  - type: openai
    args: {model_name: gpt-4o}
  - type: anthropic
    args: {model_name: claude-3-5-sonnet-20241022}
probes:
  - type: logic
  - type: instruction_following
dataset:
  format: jsonl
  path: data/harness_dataset.jsonl

generation:
  temperature: 0.2
  max_tokens: 500
```

```bash
python3 -m pip install "insidellms[openai,anthropic]"
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
insidellms harness real_harness.yaml --dry-run
insidellms harness real_harness.yaml --run-dir ./runs/real-models
```

Provider calls may cost money and may be stochastic even when insideLLMs emits
stable, diffable artifact structures.

`BiasProbe` expects a dataset made of paired prompts for comparison; do not add
it to this generated ordinary-row dataset without replacing the data shape.

## Common Options

```bash
--dry-run                           # Resolve and print the evaluation plan only
--run-dir .tmp/runs/first-harness   # Explicit artifact directory
--overwrite                          # Replace existing output directory
--skip-report                        # Do not create report.html
--report-title "Release candidate"  # Set the HTML report title
--profile healthcare-hipaa           # Apply compliance probe preset
--explain                            # Emit explain.json metadata
```

Unlike `run`, `harness` does not accept `--resume`, `--async`, or
`--concurrency`.

## Next

[Understanding Outputs →](Understanding-Outputs.md) Learn what each artefact contains.
