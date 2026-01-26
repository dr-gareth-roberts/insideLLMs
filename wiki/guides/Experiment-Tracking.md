---
title: Experiment Tracking
parent: Guides
nav_order: 3
---

# Experiment Tracking

Log runs to external tracking systems for visualization and comparison.

## Supported Backends

| Backend | Use Case |
|---------|----------|
| [Local File](#local-file) | Simple file-based logging |
| [Weights & Biases](#weights--biases) | Team collaboration, dashboards |
| [MLflow](#mlflow) | Model lifecycle management |
| [TensorBoard](#tensorboard) | TensorFlow ecosystem |

## Enabling Tracking

### In Config

```yaml
tracking:
  enabled: true
  backend: wandb
  project: my-llm-evaluation
  tags:
    - experiment
    - v1
```

### Programmatically

```python
from insideLLMs.tracking import get_tracker

tracker = get_tracker("wandb", project="my-llm-evaluation")

with tracker.start_run(name="bias-test"):
    # Run your experiment
    results = runner.run(prompt_set)
    
    # Log metrics
    tracker.log_metrics({
        "success_rate": runner.success_rate,
        "error_count": runner.error_count
    })
    
    # Log artifacts
    tracker.log_artifact("results", results)
```

---

## Local File

Simple file-based logging for local analysis.

### Config

```yaml
tracking:
  enabled: true
  backend: local
  output_dir: ./tracking_logs
```

### What Gets Logged

```
tracking_logs/
├── run_abc123/
│   ├── metrics.json
│   ├── params.json
│   └── artifacts/
│       └── results.json
```

---

## Weights & Biases

Full-featured experiment tracking with dashboards.

### Setup

```bash
pip install wandb
wandb login
```

### Config

```yaml
tracking:
  enabled: true
  backend: wandb
  project: llm-evaluation
  entity: my-team  # Optional
  tags:
    - production
```

### Features

- Real-time dashboards
- Team collaboration
- Hyperparameter comparison
- Artifact versioning

### Viewing Results

```bash
# Link printed after run
# https://wandb.ai/my-team/llm-evaluation/runs/abc123
```

---

## MLflow

Open-source platform for ML lifecycle.

### Setup

```bash
pip install mlflow

# Start tracking server (optional)
mlflow server --host 127.0.0.1 --port 5000
```

### Config

```yaml
tracking:
  enabled: true
  backend: mlflow
  tracking_uri: http://localhost:5000
  experiment_name: llm-evaluation
```

### Features

- Model registry
- Experiment comparison
- Deployment integration
- Open source

### Viewing Results

```bash
mlflow ui
# Open http://localhost:5000
```

---

## TensorBoard

TensorFlow's visualization toolkit.

### Setup

```bash
pip install tensorboard
```

### Config

```yaml
tracking:
  enabled: true
  backend: tensorboard
  log_dir: ./tb_logs
```

### Viewing Results

```bash
tensorboard --logdir ./tb_logs
# Open http://localhost:6006
```

---

## What Gets Tracked

| Category | Examples |
|----------|----------|
| **Metrics** | success_rate, error_count, latency |
| **Parameters** | model_name, temperature, probe_type |
| **Artifacts** | records.jsonl, summary.json, config |
| **Tags** | experiment name, version, environment |

### Custom Metrics

```python
tracker.log_metrics({
    "custom_score": calculate_score(results),
    "bias_index": calculate_bias(results),
})
```

### Custom Parameters

```python
tracker.log_params({
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1000,
})
```

---

## Comparison with Deterministic Artifacts

| Aspect | Tracking | Artifacts |
|--------|----------|-----------|
| Purpose | Analysis, dashboards | CI, reproducibility |
| Format | Backend-specific | Stable JSON |
| Determinism | No | Yes |
| Team sharing | Yes | Via git |

**Use both:**
- Tracking for exploration and visualization
- Artifacts for CI diff-gating and reproducibility

---

## Best Practices

### Do

- ✅ Tag experiments meaningfully
- ✅ Log hyperparameters consistently
- ✅ Use project/experiment hierarchy
- ✅ Include model and dataset info

### Don't

- ❌ Rely on tracking for determinism
- ❌ Log sensitive data (API keys, PII)
- ❌ Skip local testing before logging

---

## See Also

- [Understanding Outputs](../getting-started/Understanding-Outputs.md) - Deterministic artifacts
- [Determinism](../concepts/Determinism.md) - Why artifacts are separate
