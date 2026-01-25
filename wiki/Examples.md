# Examples

Runnable scripts live in `examples/`.

## Quickstart

```bash
python examples/example_quickstart.py
```

## Programmatic Harness

```bash
python examples/example_harness_programmatic.py
```

## End-to-end CLI Workflow (Offline)

```bash
python examples/example_cli_golden_path.py
```

## Experiment Tracking

```bash
python examples/example_experiment_tracking.py
```

## Example Configs

- `examples/experiment.yaml`: minimal offline `insidellms run` config
- `examples/harness.yaml`: hosted-model harness config (requires API keys)
- `examples/harness_ollama_cloud_deepseek_battery.yaml`: Ollama DeepSeek battery (set `OLLAMA_API_KEY`)
- `examples/probe_battery.jsonl`: small multi-probe dataset for battery runs
- `ci/harness.yaml`: offline CI diff-gating harness
