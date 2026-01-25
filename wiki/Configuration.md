# Configuration

insideLLMs supports two primary YAML/JSON config shapes:

- `insidellms run`: single model × single probe
- `insidellms harness`: many models × many probes

## Path Resolution

Relative paths in configs (e.g. dataset `path:`) are resolved relative to the config file’s
directory, not the current working directory.

## `insidellms run` (single experiment)

Minimal config:

```yaml
model:
  type: dummy
  args: {}
probe:
  type: logic
  args: {}
dataset:
  format: jsonl
  path: ../ci/harness_dataset.jsonl
  input_field: question
max_examples: 3
```

Key fields:

- `model.type`: registry name (`dummy`, `openai`, `anthropic`, …)
- `model.args`: provider/model-specific parameters
- `probe.type`: registry name (`logic`, `bias`, `attack`, …)
- `probe.args`: probe-specific parameters
- `dataset.format`: `jsonl`, `csv`, or `hf`
- `dataset.path`: for `jsonl`/`csv`
- `dataset.input_field`: which field from each row becomes the prompt (common for `run`)
- `max_examples`: cap dataset size for quick runs

## `insidellms harness` (cross-model sweep)

Minimal config:

```yaml
models:
  - type: dummy
    args: {}
probes:
  - type: logic
    args: {}
  - type: attack
    args:
      attack_type: prompt_injection
dataset:
  format: jsonl
  path: ci/harness_dataset.jsonl
max_examples: 3
report_title: Behavioural Probe Report
confidence_level: 0.95
```

Notes:

- The harness iterates all `models × probes × dataset` combinations.
- Harness datasets are usually “wide” dict records so different probes can pick the fields they
  need (e.g. `question`, `prompt`, `task`).
- If you don’t pass `--run-dir`, the harness writes to `output_dir` (default: `results`).

## Dataset Formats

- `jsonl`: one JSON object per line
- `csv`: header row + records
- `hf`: Hugging Face dataset (`name` + optional `split`)

Example `hf` dataset section:

```yaml
dataset:
  format: hf
  name: squad
  split: validation
```

