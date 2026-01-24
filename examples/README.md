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

## Example configs

- `examples/experiment.yaml`: minimal offline `insidellms run` config
- `examples/harness.yaml`: sample hosted-model harness config (requires API keys)
- `ci/harness.yaml`: minimal offline harness used for CI diff-gating
