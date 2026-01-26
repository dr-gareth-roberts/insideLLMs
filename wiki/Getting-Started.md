---
title: Getting Started
nav_order: 2
---

## Requirements

- Python 3.10+

## Install

Not published on PyPI. Install from source:

```bash
git clone https://github.com/dr-gareth-roberts/insideLLMs.git
cd insideLLMs
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[all]"
```

If you use [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -e ".[all]"
```

## Quickstart (Deterministic Run Dir + Validation)

```bash
insidellms run config.yaml --run-dir ./my_run
# Creates: ./my_run/manifest.json, ./my_run/records.jsonl, ./my_run/config.resolved.yaml
insidellms validate ./my_run
```

Tip: relative paths in configs are resolved relative to the config file’s directory. See
[Configuration](Configuration.md).

Optional extras:

```bash
pip install -e ".[nlp]"
pip install -e ".[visualization]"
pip install -e ".[dev]"
```

## Quick Test (No API Keys)

```bash
insidellms quicktest "What is 2 + 2?" --model dummy
```

## Golden Path (Offline)

The repo includes an offline CI harness config that uses `DummyModel` only:

```bash
insidellms harness ci/harness.yaml --run-dir .tmp/runs/baseline --overwrite --skip-report
insidellms harness ci/harness.yaml --run-dir .tmp/runs/candidate --overwrite --skip-report
insidellms diff .tmp/runs/baseline .tmp/runs/candidate --fail-on-changes
```

Or run the scripted workflow:

```bash
python examples/example_cli_golden_path.py
```

## First Harness Run

1) Create a config:

```yaml
# harness.yaml
models:
  - type: openai
    args:
      model_name: gpt-4o
probes:
  - type: logic
    args: {}
dataset:
  format: jsonl
  path: data/questions.jsonl
max_examples: 20
output_dir: results
```

2) Run the harness:

```bash
insidellms harness harness.yaml
```

You will get:
- `records.jsonl` (one row per example per model per probe)
- `summary.json` (aggregates and confidence intervals)
- `report.html` (human-readable comparison)

For backwards compatibility, `results.jsonl` is kept alongside `records.jsonl`.

## API Keys

Set the provider keys for hosted models:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `CO_API_KEY` or `COHERE_API_KEY`
- `HUGGINGFACEHUB_API_TOKEN` (optional, private models)

## Next

- [Examples](Examples.md)
- [Determinism and CI](Determinism-and-CI.md)
