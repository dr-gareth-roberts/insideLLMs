---
title: Configuration
parent: Reference
nav_order: 2
---

# Configuration Reference

Complete reference for YAML/JSON configuration files.

## Config Types

| Type | Command | Purpose |
|------|---------|---------|
| [Run Config](#run-config) | `insidellms run` | Single model/probe execution |
| [Harness Config](#harness-config) | `insidellms harness` | Multi-model comparison |

---

## Run Config

For `insidellms run`:

```yaml
# model: The model to use
model:
  type: openai           # Model type (required)
  args:                  # Model constructor arguments
    model_name: gpt-4o
    temperature: 0.7

# probe: The probe to run
probe:
  type: logic            # Probe type (required)
  args: {}               # Probe constructor arguments

# dataset: Input data
dataset:
  format: jsonl          # Format: jsonl, csv, inline, huggingface
  path: data/test.jsonl  # Path to dataset file

# Optional settings
max_examples: 100        # Limit number of examples
output_dir: ./results    # Output directory
async: false             # Enable async execution
concurrency: 1           # Concurrent requests (if async)
```

### Minimal Example

```yaml
model:
  type: dummy

probe:
  type: logic

dataset:
  format: inline
  items:
    - question: "What is 2 + 2?"
```

---

## Harness Config

For `insidellms harness`:

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
  - type: factuality
  - type: bias

# dataset: Shared dataset
dataset:
  format: jsonl
  path: data/test.jsonl

# Output settings
output_dir: ./comparison_results

# Optional settings
max_examples: 50
async: true
concurrency: 10
```

---

## Dataset Formats

### JSONL

```yaml
dataset:
  format: jsonl
  path: data/test.jsonl
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
  path: data/test.csv
  columns:
    question: prompt_column
    expected: answer_column
```

### Inline

```yaml
dataset:
  format: inline
  items:
    - question: "What is 2 + 2?"
      expected: "4"
    - question: "What colour is the sky?"
      expected: "blue"
```

### HuggingFace

```yaml
dataset:
  format: huggingface
  name: cais/mmlu
  split: test
  subset: abstract_algebra
  max_examples: 100
```

---

## Model Configuration

### Common Options

```yaml
model:
  type: openai           # Required: model type
  args:
    model_name: gpt-4o   # Model identifier
    temperature: 0.7     # Sampling temperature (0.0-2.0)
    max_tokens: 1000     # Max response tokens
    timeout: 60          # Request timeout in seconds
```

### Provider-Specific

#### OpenAI

```yaml
model:
  type: openai
  args:
    model_name: gpt-4o
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
    response: "Fixed test response"
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
    strict: true
    timeout: 30
```

### Multiple Probes (Harness)

```yaml
probes:
  - type: logic
  - type: bias
    args:
      sensitivity: high
  - type: factuality
```

---

## Path Resolution

Relative paths are resolved relative to the **config file's directory**, not the current working directory.

```yaml
# If config is at /project/configs/harness.yaml
dataset:
  path: ../data/test.jsonl  # Resolves to /project/data/test.jsonl
```

---

## Environment Variables

Reference environment variables in configs:

```yaml
model:
  type: openai
  args:
    api_key: ${OPENAI_API_KEY}  # Expanded at runtime
```

---

## Execution Options

```yaml
# Async execution
async: true
concurrency: 10

# Limit dataset
max_examples: 100

# Output control
output_dir: ./results
skip_report: false

# Validation
validate_output: true
schema_version: "1.0.0"

# Resume/overwrite
overwrite: false
resume: false
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
  format: inline
  items:
    - question: "What is 2 + 2?"
```

### Production Harness

```yaml
models:
  - type: openai
    args:
      model_name: gpt-4o
      temperature: 0.3
  - type: anthropic
    args:
      model_name: claude-3-5-sonnet-20241022
      temperature: 0.3

probes:
  - type: logic
  - type: factuality
  - type: bias
  - type: instruction_following

dataset:
  format: jsonl
  path: data/evaluation_set.jsonl

output_dir: ./evaluation_results
async: true
concurrency: 20
max_examples: 500
validate_output: true
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
  path: ci/test_data.jsonl

output_dir: ci/baseline
skip_report: true
```

---

## Validation

Validate your config before running:

```bash
# Check config syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Dry run (coming soon)
insidellms run config.yaml --dry-run
```

---

## See Also

- [CLI Reference](CLI.md) - Command-line options
- [Models Catalog](Models-Catalog.md) - All model configurations
- [Probes Catalog](Probes-Catalog.md) - All probe configurations
