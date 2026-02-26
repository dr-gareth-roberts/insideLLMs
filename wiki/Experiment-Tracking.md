---
title: Experiment Tracking
nav_order: 25
---

Experiment tracking logs metrics, params, and artifacts to a backend (local or hosted).
It complements insideLLMs' deterministic run artifacts (`records.jsonl`, `manifest.json`,
`summary.json`, `report.html`), which are intended for CI diff-gating and reproducibility.

Tracking lives in `insideLLMs.experiment_tracking` and provides a unified API across backends.

## What Gets Tracked

All trackers support the same core interface:

- `start_run(run_name, run_id, nested)`
- `log_metrics(metrics, step)`
- `log_params(params)`
- `log_artifact(path, name, type)`
- `log_experiment_result(result, prefix)`
- `end_run(status)`

Core data types:

- Run metadata: project, run_name/run_id, tags, notes, start/end timestamps, status
- Metrics: numeric key/value pairs, typically with step and timestamp
- Params/config: key/value pairs (MLflow coerces values to strings)
- Artifacts: files copied or uploaded (backend-specific)

`log_experiment_result(...)` extracts and logs:

- Metrics: `success_rate`, `total_count`, `success_count`, `error_count`
- Score metrics (if present): `accuracy`, `precision`, `recall`, `f1_score`,
  `mean_latency_ms`, `total_tokens`, `error_rate`
- Duration: `duration_seconds` (if present)
- Params: `experiment_id`, `model_name`, `model_provider`, `probe_name`, `probe_category`

Artifacts are copied or uploaded; they are not removed from the source path.

## TrackingConfig (Shared Settings)

`TrackingConfig` provides common metadata for all backends:

- `project` (default: `insideLLMs`)
- `experiment_name`
- `tags`
- `notes`
- `log_artifacts`, `log_code`, `auto_log_metrics`

Today, built-in trackers use `project`, `experiment_name`, `tags`, and `notes`.
The other flags are reserved for future use.

## Backends

### LocalFileTracker (local)

- Default output dir: `./experiments`
- Run ID: `run_name` if provided, else `config.experiment_name`, else timestamp `YYYYMMDD_HHMMSS`

Layout:

```
output_dir/
  project_name/
    run_id/
      metadata.json
      metrics.json
      params.json
      artifacts.json
      final_state.json
      artifacts/
        <copied files>
```

Notes:

- Metrics/params are buffered and written at `end_run`.
- JSON serialization uses `default=str` for non-JSON values.

### WandBTracker (wandb)

- Uses `wandb.init(project=..., entity=..., name=..., tags=..., notes=...)`
- Extra features: `log_table(...)`, `watch_model(...)`
- Run ID is assigned by W&B (optional `run_id` can resume)

### MLflowTracker (mlflow)

- `tracking_uri` optional; if not set, MLflow uses `MLFLOW_TRACKING_URI` or local `./mlruns`
- `experiment_name` defaults to `config.experiment_name` or `config.project`
- Extra features: `log_model(...)`, `register_model(...)`
- MLflow params are stored as strings

### TensorBoardTracker (tensorboard / tensorboardX)

- `log_dir` default: `./runs`
- Run directory uses `run_name` or ISO timestamp
- Params and artifacts are logged as text
- Extra feature: `log_histogram(...)`

### MultiTracker

Fan-out to multiple backends (e.g., local + W&B).

## Enabling Tracking

### CLI (`run` and `harness`)

```bash
insidellms run experiment.yaml --track local --track-project my-project
insidellms run experiment.yaml --track wandb --track-project my-project
insidellms run experiment.yaml --track mlflow --track-project my-project
insidellms run experiment.yaml --track tensorboard --track-project my-project

insidellms harness harness.yaml --track local --track-project my-project
```

Notes:

- For `local`, insideLLMs writes tracking logs under `<run_dir_parent>/tracking/<track-project>/<run-id>/`.
- For `tensorboard`, insideLLMs writes TensorBoard logs under `<run_dir_parent>/tracking/tensorboard/<track-project>/<run-id>/`.
- For hosted backends (W&B/MLflow), `--track-project` maps to the backend’s project/experiment name.
- Tracking is best-effort. If the backend dependency is missing, the run continues and tracking is
  disabled with a warning.

### Python API (minimal)

```python
from insideLLMs import create_tracker

with create_tracker("local", output_dir="./experiments") as tracker:
    tracker.log_params({"model": "gpt-4"})
    tracker.log_metrics({"accuracy": 0.95})
    tracker.log_artifact("results.json")
```

Backend-specific construction:

- W&B: `create_tracker("wandb", project="my-project", entity="my-team", mode="offline")`
- MLflow: `create_tracker("mlflow", tracking_uri="http://localhost:5000", experiment_name="my-exp")`
- TensorBoard: `create_tracker("tensorboard", log_dir="./runs")`

Auto-track a function:

```python
from insideLLMs import LocalFileTracker, auto_track

@auto_track(LocalFileTracker(output_dir="./experiments"), experiment_name="baseline")
def run_eval():
    return {"accuracy": 0.93, "f1": 0.91}
```

Log an ExperimentResult:

```python
from insideLLMs import LocalFileTracker

# result = ... (ExperimentResult)
with LocalFileTracker(output_dir="./experiments") as tracker:
    tracker.log_experiment_result(result, prefix="eval_")
```

## Dependencies and Environment

- local: no extra deps
- wandb: `pip install wandb`, authenticate via `wandb login` or `WANDB_API_KEY`
- mlflow: `pip install mlflow`, optionally set `MLFLOW_TRACKING_URI`
- tensorboard: `pip install tensorboard` or `pip install tensorboardX`
  (TensorBoardTracker uses `torch.utils.tensorboard` if available, else `tensorboardX`)

## Data Retention and Privacy

- **Retention**: Backend-specific. LocalFileTracker stores files indefinitely in `output_dir`. Hosted backends (W&B, MLflow) follow their own retention policies.
- **Privacy**: Do not log PII, secrets, or sensitive prompts. Artifacts are copied as-is; ensure source paths contain only appropriate data.
- **Compliance**: For compliance-sensitive adopters, consider: (1) using local-only tracking, (2) disabling artifact logging, (3) scrubbing params/metrics before logging, (4) reviewing backend terms for data residency and retention.

## CI and Determinism Notes

- Canonical run artifacts are deterministic and used for CI diff-gating.
- Tracking backends are not deterministic: they include timestamps, backend-generated run IDs,
  and external side effects.
- Treat tracked data as sensitive. Do not store secrets or PII.
- To correlate tracking with run artifacts, log the run ID or run directory as params explicitly.
