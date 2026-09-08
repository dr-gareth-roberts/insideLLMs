---
title: Configuration
parent: Reference
nav_order: 2
---

# Configuration Reference

Complete reference for YAML/JSON configuration files.

## Which Config API to Use

Run and harness files share the version 1 schema. Direct runner calls retain
their separate execution-options dataclass:

| Use case | Surface | Shape |
|----------|---------|-------|
| **CLI YAML/JSON** | `init`, `validate`, `run`, `harness` | Validated mappings with `config_version: "1"` and `type`/`args` |
| **Application config** | `insideLLMs.config_schema.RuntimeConfiguration` | The same Pydantic schema; also accepted by `insideLLMs.config.load_config` |
| **Programmatic runner control** | `insideLLMs.config_types` | `RunConfig`, `RunConfigBuilder`, `ProgressInfo` |

Pydantic is required by the base package; validation cannot silently turn off.
Existing files without `config_version` are treated as version 1. Dataset paths
resolve relative to the configuration file for both validation and execution.

The old `insideLLMs.config.ExperimentConfig` builder uses `provider`/`model_id` and
dataset `source`. Call its `to_runtime_config()` method to convert explicitly.
The CLI can also convert legacy files with a deprecation warning. Unsupported
legacy settings raise errors rather than being ignored.

Unknown top-level execution settings are rejected. Earlier generated `benchmark`,
`tracking`, `async`, and `output` blocks were not executed; remove those blocks and
use the `benchmark` command or `run --track`, `--async`, and output CLI flags.
The current `full` template emits supported runner and determinism settings only.

## Config Types

| Type | Command | Purpose |
|------|---------|---------|
| [Run Config](#run-config) | `insidellms run` | Single model/probe execution |
| [Harness Config](#harness-config) | `insidellms harness` | Multi-model comparison |

---

## Run Config

Generate a working single-run config and its dataset with:

```bash
insidellms init run.yaml --template basic
```

The generated shape for `insidellms run` is:

```yaml
config_version: "1"
# model: The model to use
model:
  type: openai           # Model type (required)
  args:                  # Model constructor arguments
    model_name: gpt-4o

# probe: The probe to run
probe:
  type: logic            # Probe type (required)
  args: {}               # Probe constructor arguments

# dataset: Input data
dataset:
  format: jsonl          # Format: jsonl, csv, hf
  path: data/questions.jsonl  # Relative to this config file

# Optional settings
generation:              # Passed to probe/model generate call
  temperature: 0.7
  max_tokens: 800
```

For execution controls (validation/resume/overwrite/async), use CLI flags:

```bash
insidellms run config.yaml --async --concurrency 10
insidellms run config.yaml --validate-output --validation-mode warn
insidellms run config.yaml --resume
insidellms run config.yaml --overwrite
```

### Minimal Example

```yaml
model:
  type: dummy

probe:
  type: logic

dataset:
  format: jsonl
  path: data/questions.jsonl
```

---

## Harness Config

Generate a portable offline harness and sample dataset with:

```bash
insidellms init harness.yaml --template harness
insidellms harness harness.yaml --dry-run
```

Run the initializer in the directory where the config will live. It creates
`data/harness_dataset.jsonl` relative to the current directory, while execution
resolves the path relative to the config file.

The shape for `insidellms harness` is:

```yaml
# models: List of models to compare
models:
  - type: openai
    args:
      model_name: gpt-4o
  - type: anthropic
    args:
      model_name: claude-3-5-sonnet-20241022

# probes: List of probes to run
probes:
  - type: logic
  - type: instruction_following
  - type: attack
    args:
      attack_type: prompt_injection
  - type: code_generation
    args:
      language: python

# dataset: Shared dataset
dataset:
  format: jsonl
  path: data/harness_dataset.jsonl

# Output settings
output_dir: ./comparison_results

# Optional settings
max_examples: 50
```

`max_examples` is applied by `harness`; it is not applied by `run`.

---

## Dataset Formats

### JSONL

```yaml
dataset:
  format: jsonl
  path: data/questions.jsonl
```

File format:
```jsonl
{"question": "What is 2 + 2?", "expected": "4"}
{"question": "What colour is the sky?", "expected": "blue"}
```

### CSV

```yaml
dataset:
  format: csv
  path: path/to/your/evaluation.csv
```

### HuggingFace

```yaml
dataset:
  format: hf
  name: cais/mmlu
  split: test

# Harness-only limit
max_examples: 100
```

---

## Model Configuration

### Common Options

`model.args` are constructor arguments. Put sampling and token limits in the
top-level `generation` mapping so they are passed to the probe/model call.

```yaml
model:
  type: openai           # Required: model type
  args:
    model_name: gpt-4o   # Model identifier

generation:
  temperature: 0.7
  max_tokens: 1000
```

### Provider-Specific

#### OpenAI

```yaml
model:
  type: openai
  args:
    model_name: gpt-4o

generation:
  temperature: 0.7
  max_tokens: 1000
  top_p: 1.0
  frequency_penalty: 0.0
  presence_penalty: 0.0
```

#### Anthropic

```yaml
model:
  type: anthropic
  args:
    model_name: claude-3-5-sonnet-20241022

generation:
  max_tokens: 1000
  temperature: 0.7
```

#### Ollama

```yaml
model:
  type: ollama
  args:
    model_name: llama3
    base_url: http://localhost:11434
```

#### DummyModel

```yaml
model:
  type: dummy
  args:
    name: test_model
    canned_response: "Fixed test response"
```

---

## Probe Configuration

### Basic

```yaml
probe:
  type: logic
  args: {}
```

### With Options

```yaml
probe:
  type: logic
  args:
    extract_answer: true
```

### Multiple Probes (Harness)

```yaml
probes:
  - type: logic
  - type: attack
    args:
      attack_type: prompt_injection
  - type: code_generation
    args:
      language: python
```

Probe constructor options and row contracts differ. In particular,
`BiasProbe` takes `bias_dimension`/`analyze_sentiment` constructor arguments and
expects each invocation to receive a collection of paired prompts; an ordinary
question row is not valid bias input.

---

## Path Resolution

Relative paths are resolved relative to the **config file's directory**, not the current working directory.

```yaml
# If config is at /project/configs/harness.yaml
dataset:
  path: ../data/harness_dataset.jsonl
# Resolves to /project/data/harness_dataset.jsonl
```

---

## Environment Variables

Model arguments are literal values. `${NAME}` placeholders in `model.args` are
**not** expanded by the CLI runtime. Export the provider's supported variable
and omit `api_key` from the config:

```bash
export OPENAI_API_KEY="sk-..."
insidellms run config.yaml
```

Avoid inline secrets because the resolved mapping is written to
`config.resolved.yaml`. Environment expansion is used for supported file paths,
not arbitrary model arguments.

---

## Execution Options

```yaml
# Limit the dataset in a harness (ignored by `run`)
max_examples: 100

# Optional generation kwargs passed through to probes/models
generation:
  temperature: 0.3
  max_tokens: 500
```

`probe_kwargs` and the legacy `run_kwargs` are also accepted. When more than
one is present, later mappings override earlier ones in this order:
`generation`, `probe_kwargs`, `run_kwargs`.

Execution controls are CLI flags:

```bash
insidellms run config.yaml --async --concurrency 10
insidellms run config.yaml --validate-output --schema-version 1.0.0
insidellms run config.yaml --resume
insidellms run config.yaml --overwrite
```

---

## Complete Examples

### Minimal Run Config

```yaml
model:
  type: dummy
probe:
  type: logic
dataset:
  format: jsonl
  path: data/questions.jsonl
```

### Production Harness

```yaml
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
  - type: instruction_following

dataset:
  format: jsonl
  path: path/to/your/evaluation_set.jsonl

generation:
  temperature: 0.3

output_dir: ./evaluation_results
max_examples: 500
```

### CI Baseline Config

```yaml
models:
  - type: dummy
    args:
      name: baseline

probes:
  - type: logic

dataset:
  format: jsonl
  path: data/harness_dataset.jsonl

output_dir: ci/baseline
```

---

## Validation

`validate` has two different scopes:

- For a single-run config containing singular `model`, `probe`, and `dataset`
  entries, it performs the current legacy config checks.
- For a completed run directory, it validates `manifest.json` and
  `records.jsonl` against their schema contracts.

It does not validate a harness config containing plural `models` and `probes`.
For a harness, resolve and check its evaluation plan first, run it, and then
validate the output directory:

```bash
# Single-run config only
insidellms validate run.yaml

# Harness config and then its artifacts
insidellms harness harness.yaml --dry-run
insidellms harness harness.yaml --run-dir ./runs/candidate
insidellms validate ./runs/candidate
```

The legacy config validator checks relative dataset paths from the current
working directory, whereas execution resolves them from the config file's
directory. Treat a path warning from another working directory accordingly.

---

## See Also

- [CLI Reference](CLI.md) - Command-line options
- [Models Catalog](Models-Catalog.md) - All model configurations
- [Probes Catalog](Probes-Catalog.md) - All probe configurations
