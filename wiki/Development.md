# Development

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Optional extras:

```bash
pip install -e ".[nlp]"
pip install -e ".[visualization]"
```

## Checks

Manual:

```bash
ruff check .
ruff format --check .
mypy insideLLMs
pytest
```

Or helpers:

```bash
make check
# or
bash scripts/checks.sh
```

## Golden Path (Offline)

```bash
make golden-path
```

Or run the scripted workflow:

```bash
python examples/example_cli_golden_path.py
```

## Run Artifacts Location

To keep outputs local to the repo:

```bash
INSIDELLMS_RUN_ROOT=.tmp/insidellms_runs insidellms run examples/experiment.yaml
```

Note: experiment tracking backends (W&B/MLflow/TensorBoard/local tracking logs) write to separate
locations from the deterministic run artifacts. See (Experiment-Tracking).
