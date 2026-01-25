# insideLLMs Wiki

insideLLMs is a Python library and CLI for comparing LLM behaviour across models using
shared probes and datasets. It produces deterministic run artefacts for reporting and CI
diffs.

## Quick Links

- [Getting Started](Getting-Started)
- [Harness](Harness)
- [CLI](CLI)
- [Configuration](Configuration)
- [Providers and Models](Providers-and-Models)
- [Probes and Models](Probes-and-Models)
- [Datasets and Harness](Datasets-and-Harness)
- [Results and Reports](Results-and-Reports)
- [Experiment Tracking](Experiment-Tracking)
- [Performance and Caching](Performance-and-Caching)
- [Tracing and Fingerprinting](Tracing-and-Fingerprinting)
- [Determinism and CI](Determinism-and-CI)
- [Examples](Examples)
- [Development](Development)
- [Troubleshooting](Troubleshooting)
- [FAQ](FAQ)

## Core Concepts

- Models: Providers and local models share a single interface.
- Probes: Small, focused tests for specific behaviours (logic, bias, safety).
- Harness: Run the same probe suite across models and datasets in one pass.
- Outputs: `records.jsonl`, `summary.json`, `report.html`, and `diff.json`.
