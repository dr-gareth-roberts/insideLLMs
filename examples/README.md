# Examples

This directory contains runnable examples for common insideLLMs workflows.

## Quickstart (pure Python)

```bash
python examples/example_quickstart.py
```

## End-to-end CLI workflow (offline, deterministic)

Runs the built-in CI harness (DummyModel only) twice into `.tmp/runs/`, generates reports, then diffs
the results (should be identical).

```bash
python examples/example_cli_golden_path.py
```

## Programmatic harness (no CLI)

```bash
python examples/example_harness_programmatic.py
```

## More examples

- `examples/example_models.py`: Dummy/OpenAI/HuggingFace model usage
- `examples/example_factuality.py`: probe usage
- `examples/example_registry.py`: registry patterns
- `examples/example_benchmark_suite.py`: benchmarking
- `examples/example_experiment_tracking.py`: tracking backends
- `examples/example_interactive_visualization.py`: report/visualization helpers
- `examples/example_nlp.py`: NLP utilities
- `examples/example_ollama_cloud_harness_diff.py`: Ollama Cloud harness double-run + diff (Python)
- `examples/example_openrouter_advanced.py`: OpenRouter advanced harness + stats + diff
- `examples/killer_features/feature_1_ci_behavior_gate.py`: deterministic CI behaviour gate
- `examples/killer_features/feature_2_trace_drift_gate.py`: trace drift / violation gating demo
- `examples/killer_features/feature_3_model_tournament.py`: multi-model tournament + stats report

## Example configs

- `examples/experiment.yaml`: minimal offline `insidellms run` config
- `examples/harness.yaml`: sample hosted-model harness config (requires API keys)
- `examples/harness_ollama_cloud_deepseek_battery.yaml`: Ollama DeepSeek battery (set `OLLAMA_API_KEY`)
- `examples/ollama_cloud_benchmark.md`: detailed Ollama Cloud all-model benchmark walkthrough
- `examples/probe_battery.jsonl`: small multi-probe dataset for battery runs
- `scripts/ollama_cloud_harness_diff.sh`: helper script to run the Ollama Cloud harness twice + diff
- `ci/harness.yaml`: minimal offline harness used for CI diff-gating
