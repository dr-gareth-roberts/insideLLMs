# Ollama Cloud: multi-model prompt benchmark with harness + probes + diff

This example shows how to benchmark a series of prompts across **all** models available in your
Ollama Cloud account, using the harness to run multiple probes and then diffing two runs to
spot regressions or behavioral drift.

## 1) Prepare credentials

```bash
export OLLAMA_API_KEY="..."
```

## 2) Create a harness config that enumerates all Ollama Cloud models

Create `examples/harness_ollama_cloud_all_models.yaml` and list every model you have available in
Ollama Cloud (the dashboard shows your full model list). Each entry below runs the same probe
battery against a shared prompt dataset.

```yaml
# examples/harness_ollama_cloud_all_models.yaml
# Run all Ollama Cloud models against a probe battery + prompt series.
models:
  # Add *every* Ollama Cloud model you want to benchmark.
  - type: ollama
    args:
      model_name: llama3.1:cloud
      base_url: https://ollama.com
      timeout: 600
  - type: ollama
    args:
      model_name: qwen2.5:cloud
      base_url: https://ollama.com
      timeout: 600
  - type: ollama
    args:
      model_name: mistral:cloud
      base_url: https://ollama.com
      timeout: 600
  # ...repeat for the rest of your Ollama Cloud models.

probes:
  # The harness runs each probe for each model on every dataset row.
  - type: logic
    args: {}
  - type: factuality
    args: {}
  - type: prompt_injection
    args: {}
  - type: instruction_following
    args: {}
  - type: code_generation
    args:
      language: python

# This dataset contains a series of prompts that cover all of the probes above.
# (See examples/probe_battery.jsonl for the actual prompt content.)
dataset:
  format: jsonl
  path: examples/probe_battery.jsonl

max_examples: 3

generation:
  temperature: 0.0
  seed: 42
  max_tokens: 512

report_title: Ollama Cloud All-Model Benchmark
output_dir: .tmp/runs/ollama-cloud/all-models
```

**Why this config matters**
- **Harness**: orchestrates every model/probe combination, producing deterministic run artifacts
  (records, summary, report).
- **Probes**: stress different behaviors (logic, factuality, prompt injection resilience, instruction
  following, code generation) over the same prompt series.
- **Diffing**: compares two run directories to spot regressions, improvements, or drift.

## 3) Run the harness across all models

```bash
insidellms harness examples/harness_ollama_cloud_all_models.yaml \
  --run-dir .tmp/runs/ollama-cloud/all-models
```

This produces artifacts like:
- `records.jsonl` (canonical per-example outputs)
- `summary.json` (aggregate metrics)
- `report.html` (human-readable report)

## 4) Re-run and diff to detect regressions or drift

Re-run the same config (e.g., after upgrading a model or changing prompt content) into a second
run directory, then diff the outputs.

```bash
insidellms harness examples/harness_ollama_cloud_all_models.yaml \
  --run-dir .tmp/runs/ollama-cloud/all-models-candidate

insidellms diff .tmp/runs/ollama-cloud/all-models \
  .tmp/runs/ollama-cloud/all-models-candidate \
  --fail-on-changes
```

The diff report highlights:
- **Regressions**: examples where candidate outputs scored worse than baseline
- **Improvements**: examples where candidate outputs improved
- **Trace drift**: changes in trace fingerprints when enabled

## 5) Tips for scaling the benchmark

- Keep `temperature: 0.0` + a fixed `seed` to maximize determinism.
- Start with a small `max_examples` while validating connectivity, then scale up.
- Use the diff output (`diff.json`) to drive CI gating once the baseline is stable.

## 6) Python double-run + diff

If you want a pure-Python workflow that exercises the harness, probes, and diffing APIs, run the
example script below.

```bash
export OLLAMA_API_KEY="..."
export OLLAMA_CLOUD_MODELS="llama3.1:cloud,qwen2.5:cloud,mistral:cloud"

python examples/example_ollama_cloud_harness_diff.py
```

Optional flags:
- `--models` (comma/space-separated list of models; overrides `OLLAMA_CLOUD_MODELS`)
- `--run-root` (default: `.tmp/runs/ollama-cloud`)
- `--dataset` (default: `examples/probe_battery.jsonl`)

## 7) Scripted double-run + diff

If you prefer a scripted workflow, use the helper script that runs the harness twice and diffs the
artifacts.

```bash
export OLLAMA_API_KEY="..."
export OLLAMA_CLOUD_MODELS="llama3.1:cloud,qwen2.5:cloud,mistral:cloud"

./scripts/ollama_cloud_harness_diff.sh
```

Optional environment overrides:
- `RUN_ROOT` (default: `.tmp/runs/ollama-cloud`)
- `CONFIG_PATH` (default: `.tmp/harness_ollama_cloud_all_models.yaml`)
- `DATASET_PATH` (default: `examples/probe_battery.jsonl`)
- `BASELINE_DIR` / `CANDIDATE_DIR` (default: `<RUN_ROOT>/baseline` and `<RUN_ROOT>/candidate`)
