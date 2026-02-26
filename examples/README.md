# Examples

This directory contains runnable examples for common insideLLMs workflows.

## Compatibility Matrix

| Example | Offline? | API keys required |
|---------|----------|-------------------|
| `example_quickstart.py` | Yes | None |
| `example_cli_golden_path.py` | Yes | None |
| `example_harness_programmatic.py` | Yes | None |
| `example_models.py` | Partial | Dummy/HF work offline; OpenAI needs `OPENAI_API_KEY` |
| `example_factuality.py` | Partial | Skips OpenAI/Anthropic if keys missing |
| `example_registry.py` | Yes | None |
| `example_benchmark_suite.py` | Yes | None |
| `example_experiment_tracking.py` | Yes | None (local backend) |
| `example_interactive_visualization.py` | Yes | None |
| `example_nlp.py` | Yes | None |
| `examples/experiment.yaml` | Yes | None (DummyModel) |
| `examples/harness.yaml` | No | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| `examples/harness_ollama_cloud_deepseek_battery.yaml` | No | `OLLAMA_API_KEY` |
| `ci/harness.yaml` | Yes | None (DummyModel only) |

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
- `examples/killer_feature_1_ci_gate.py`: deterministic behavioral CI gate workflow
- `examples/killer_feature_2_trace_guard.py`: trace-aware drift + violation guardrails
- `examples/killer_feature_3_model_selection.py`: model-selection workflow + statistical report
- `examples/parallel_killer_features.md`: map for splitting killer features across parallel envs

## Example configs

- `examples/experiment.yaml`: minimal offline `insidellms run` config
- `examples/harness.yaml`: sample hosted-model harness config (requires API keys)
- `examples/harness_ollama_cloud_deepseek_battery.yaml`: Ollama DeepSeek battery (set `OLLAMA_API_KEY`)
- `examples/ollama_cloud_benchmark.md`: detailed Ollama Cloud all-model benchmark walkthrough
- `examples/probe_battery.jsonl`: small multi-probe dataset for battery runs
- `scripts/ollama_cloud_harness_diff.sh`: helper script to run the Ollama Cloud harness twice + diff
- `ci/harness.yaml`: minimal offline harness used for CI diff-gating
