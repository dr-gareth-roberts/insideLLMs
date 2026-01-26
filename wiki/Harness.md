---
title: Harness
nav_order: 20
---

The harness runs a cross-model, cross-probe sweep over a shared dataset.
It produces three outputs:

- `records.jsonl` per example per model per probe
- `summary.json` with aggregates and confidence intervals
- `report.html` for human-readable comparison

For backwards compatibility, `results.jsonl` is kept alongside `records.jsonl`.

## Configuration

```yaml
models:
  - type: openai
    args:
      model_name: gpt-4o
  - type: anthropic
    args:
      model_name: claude-opus-4-5-20251101
probes:
  - type: logic
    args: {}
  - type: bias
    args: {}
dataset:
  format: jsonl
  path: data/questions.jsonl
max_examples: 50
output_dir: results
confidence_level: 0.95
report_title: Behavioural Probe Report
```

Notes:

- Relative dataset paths are resolved relative to the config file location. If your config lives in
  `examples/`, you’ll typically want `../data/...` paths. See [Configuration](Configuration.md).
- If you pass `--run-dir`, it overrides `output_dir` and writes artifacts exactly there.
- Optional `generation` (or `probe_kwargs`) is passed through to `probe.run(...)` and typically ends
  up in `model.generate(...)` as `temperature`, `max_tokens`, `seed`, etc.

## Run

```bash
insidellms harness harness.yaml
```

If `plotly` is installed, the report is interactive. Otherwise a basic HTML report
is generated.

## Dataset Design for Multi-Probe Harnesses

Harness datasets are usually dict records containing multiple fields so different probes can
pull the input they need.

Common built-in probes accept either a string or a dict. For dict inputs they look for these keys:

| Probe | Dict key(s) (first match wins) |
|------:|--------------------------------|
| `logic` | `problem`, `question` |
| `factuality` | `question` + `reference_answer` (or `factual_questions`, `questions`) |
| `bias` | `prompt_pairs`, `pairs` (or `prompt_a` + `prompt_b`) |
| `attack` / `prompt_injection` / `jailbreak` | `prompt`, `attack` |
| `instruction_following` | `task`, `instruction` (+ optional `constraints`) |
| `constraint_compliance` | `task` |
| `multi_step_task` | `steps` (+ optional `preamble`) |
| `code_generation` | `task`, `description` |
| `code_explanation` | `code` |
| `code_debug` | `code` (+ optional `error`) |

## Generation Parameters

You can pass generation parameters via config:

```yaml
generation:
  temperature: 0.0
  max_tokens: 256
```

These are passed to `probe.run(model, item, **generation)` and typically forwarded to
`model.generate(...)`. Not all models accept all parameters.

## Dataset Formats

- `jsonl`: one JSON object per line (recommended)
- `csv`: header row plus records
- `hf`: Hugging Face datasets (`name` and optional `split`)

Example `hf` dataset section:

```yaml
dataset:
  format: hf
  name: squad
  split: validation
```
