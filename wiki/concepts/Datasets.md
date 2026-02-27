---
title: Datasets
parent: Concepts
nav_order: 4
---

# Datasets

Datasets provide the inputs that probes use to test models.

## Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| JSONL | `.jsonl` | Structured data with fields |
| CSV | `.csv` | Tabular data |
| Inline | (in config) | Small test sets |
| HuggingFace | (remote) | Standard benchmarks |

## JSONL Format

Most flexible format. One JSON object per line:

```jsonl
{"question": "What is 2 + 2?", "expected": "4"}
{"question": "What colour is the sky?", "expected": "blue"}
{"question": "Name a prime number", "expected": "2"}
```

### Config

```yaml
dataset:
  format: jsonl
  path: data/test.jsonl
```

### Loading Programmatically

```python
from insideLLMs.dataset_utils import load_jsonl_dataset

items = load_jsonl_dataset("data/test.jsonl")
```

## CSV Format

For tabular data:

```csv
question,expected
"What is 2 + 2?","4"
"What colour is the sky?","blue"
```

### Config

```yaml
dataset:
  format: csv
  path: data/test.csv
  columns:
    question: question
    expected: expected
```

### Loading Programmatically

```python
from insideLLMs.dataset_utils import load_csv_dataset

items = load_csv_dataset("data/test.csv")
```

## Inline Format

For small datasets directly in config:

```yaml
dataset:
  format: inline
  items:
    - question: "What is 2 + 2?"
      expected: "4"
    - question: "What colour is the sky?"
      expected: "blue"
```

## HuggingFace Datasets

Load standard benchmarks:

```yaml
dataset:
  format: huggingface
  name: cais/mmlu
  split: test
  subset: abstract_algebra
  max_examples: 100
```

### Programmatically

```python
from insideLLMs.dataset_utils import load_huggingface_dataset

items = load_huggingface_dataset(
    name="cais/mmlu",
    split="test",
    subset="abstract_algebra"
)
```

## Content Hashing

Local datasets are content-addressed:

```yaml
dataset:
  format: jsonl
  path: data/test.jsonl
  # Automatically computed:
  dataset_hash: sha256:abc123def456...
```

The hash is included in the run_id, ensuring:
- Different data → Different run_id
- Same data → Same run_id (determinism)

## Path Resolution

Relative paths resolve from the **config file's directory**:

```
project/
├── configs/
│   └── harness.yaml      # dataset.path: ../data/test.jsonl
└── data/
    └── test.jsonl        # ← Resolved path
```

## Limiting Examples

For development/testing:

```yaml
dataset:
  format: jsonl
  path: data/large_dataset.jsonl

max_examples: 50  # Only use first 50
```

Then run normally:

```bash
insidellms run config.yaml
```

## Dataset Registry

Register custom loaders:

```python
from insideLLMs.registry import dataset_registry

def load_my_format(path, **kwargs):
    # Custom loading logic
    return items

dataset_registry.register("my_format", load_my_format)
```

Then in config:

```yaml
dataset:
  format: my_format
  path: data/custom.xyz
```

## Input Structure

Probes expect specific input structures:

### Simple String

```jsonl
"What is the capital of France?"
```

### Dict with Fields

```jsonl
{"question": "...", "expected": "..."}
{"prompt": "...", "constraints": [...]}
```

### Chat Messages

```jsonl
{"messages": [{"role": "user", "content": "Hello!"}]}
```

Check [Probes Catalog](../reference/Probes-Catalog.md) for each probe's expected format.

## Best Practices

### Do

-  Use JSONL for structured data
-  Include `expected` for evaluation
-  Use meaningful field names
-  Keep datasets version-controlled

### Don't

-  Include sensitive data
-  Use very large inline datasets
-  Rely on file modification times
-  Use absolute paths in configs

## See Also

- [Configuration Reference](../reference/Configuration.md) - Dataset config options
- [Probes Catalog](../reference/Probes-Catalog.md) - Expected input formats
- [Determinism](Determinism.md) - How dataset hashing works
