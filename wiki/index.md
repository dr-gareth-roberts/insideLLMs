---
title: insideLLMs
nav_order: 1
---

insideLLMs is a Python library and CLI for comparing LLM behaviour across models using
shared probes and datasets. It produces deterministic run artefacts for reporting and CI
diffs.

## Quick Links

- [Getting Started](Getting-Started.md)
- [Harness](Harness.md)
- [CLI](CLI.md)
- [Configuration](Configuration.md)
- [Providers and Models](Providers-and-Models.md)
- [Probes and Models](Probes-and-Models.md)
- [Datasets and Harness](Datasets-and-Harness.md)
- [Results and Reports](Results-and-Reports.md)
- [Experiment Tracking](Experiment-Tracking.md)
- [Performance and Caching](Performance-and-Caching.md)
- [Tracing and Fingerprinting](Tracing-and-Fingerprinting.md)
- [Determinism and CI](Determinism-and-CI.md)
- [LangChain and LangGraph](LangChain-and-LangGraph.md)
- [Examples](Examples.md)
- [Development](Development.md)
- [Troubleshooting](Troubleshooting.md)
- [FAQ](FAQ.md)

## Core Concepts

- Models: Providers and local models share a single interface.
- Probes: Small, focused tests for specific behaviours (logic, bias, safety).
- Harness: Run the same probe suite across models and datasets in one pass.
- Outputs: `records.jsonl`, `summary.json`, `report.html`, and `diff.json`.
