<div align="center">
  <img src="insidellms.jpg" alt="insideLLMs" width="720">
  <h1>insideLLMs</h1>
  <p><strong>Behavioural regression testing for LLM systems.</strong></p>
  <p>Detect model behaviour drift with deterministic artefacts, readable diffs, and CI gates.</p>
</div>

<p align="center">
  <a href="https://github.com/dr-gareth-roberts/insideLLMs/actions/workflows/ci.yml"><img src="https://github.com/dr-gareth-roberts/insideLLMs/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://codecov.io/gh/dr-gareth-roberts/insideLLMs"><img src="https://codecov.io/gh/dr-gareth-roberts/insideLLMs/branch/main/graph/badge.svg" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

---

## What insideLLMs solves

Most eval tooling answers: **"How good is this model on benchmark X?"**

insideLLMs answers: **"Did my application's behaviour change in ways I care about?"**

This project is designed for product teams that need safe, repeatable model changes:

- **Deterministic outputs** so behaviour diffs are stable and reviewable.
- **Response-level artefact tracking** (`records.jsonl`) rather than only aggregate scores.
- **CI enforcement** to fail pull requests when behaviour shifts unexpectedly.
- **Provider-agnostic execution** across hosted and local model backends.

---

## Table of contents

- [Quick start (offline, no API keys)](#quick-start-offline-no-api-keys)
- [Typical workflow](#typical-workflow)
- [Install](#install)
- [Core concepts](#core-concepts)
- [CLI commands you'll use most](#cli-commands-youll-use-most)
- [CI integration](#ci-integration)
- [Configuration and examples](#configuration-and-examples)
- [Documentation map](#documentation-map)
- [Contributing](#contributing)

---

## Quick start (offline, no API keys)

The fastest way to validate your setup is the built-in deterministic golden path.

```bash
# 1) Install development environment
pip install -e ".[dev]"

# 2) Run deterministic harness + diff cycle using DummyModel
make golden-path
```

Expected result: the diff exits cleanly (no unexpected changes) and writes artefacts under `.tmp/runs/`.

---

## Typical workflow

### 1) Create a baseline run

```bash
insidellms harness ci/harness.yaml --run-dir runs/baseline --overwrite
```

### 2) Create a candidate run

```bash
insidellms harness ci/harness.yaml --run-dir runs/candidate --overwrite
```

### 3) Compare and optionally gate

```bash
insidellms diff runs/baseline runs/candidate
insidellms diff runs/baseline runs/candidate --fail-on-changes
```

---

## Install

### End users

```bash
pip install insidellms
```

Optional extras:

```bash
pip install "insidellms[nlp]"            # NLP-related functionality (e.g., nltk)
pip install "insidellms[visualization]"  # visualization/report extras
pip install "insidellms[all]"            # broad optional dependency set
```

### Contributors

```bash
git clone https://github.com/dr-gareth-roberts/insideLLMs.git
cd insideLLMs
pip install -e ".[dev]"
make check-fast
```

> Note: Some optional features require extras not included in `[dev]` (for example `nltk`/`pydantic`-dependent flows).

---

## Core concepts

### Probe
A behavioural test that exercises a model capability or risk area.

### Harness run
A deterministic execution over probe(s) + data + model config.

### Artefacts
A run directory usually includes:

- `records.jsonl` — canonical per-example input/output records.
- `manifest.json` — normalized run metadata.
- `config.resolved.yaml` — effective configuration snapshot.
- `summary.json` — aggregate counters and statistics.
- `report.html` — optional human-readable report.

### Diff
Compares two run directories to identify behavioural deltas.

---

## CLI commands you'll use most

```bash
# Run harness
insidellms harness <config.yaml> --run-dir <output_dir>

# Compare two runs
insidellms diff <baseline_dir> <candidate_dir>

# Fail pipeline if changes exist
insidellms diff <baseline_dir> <candidate_dir> --fail-on-changes

# Show supported schemas
insidellms schema list

# Validate a records file against schema contracts
insidellms schema validate --name ResultRecord --input <records.jsonl>
```

---

## CI integration

For CI policy gates, run harness on baseline and candidate configurations and enforce a diff check:

```yaml
- name: Baseline
  run: insidellms harness ci/harness.yaml --run-dir runs/baseline --overwrite

- name: Candidate
  run: insidellms harness ci/harness.yaml --run-dir runs/candidate --overwrite

- name: Behavioural gate
  run: insidellms diff runs/baseline runs/candidate --fail-on-changes
```

You can also use the project GitHub Action published from this repository (`action.yml`).

---

## Configuration and examples

- Sample CI harness config: [`ci/harness.yaml`](ci/harness.yaml)
- Runnable examples: [`examples/`](examples/)
- CLI reference docs: [`wiki/reference/CLI.md`](wiki/reference/CLI.md)

Simple Python usage:

```python
from insideLLMs import LogicProbe, OpenAIModel, run_probe

model = OpenAIModel(model_name="gpt-4o-mini")
probe = LogicProbe()
result = run_probe(model, probe, ["What is 2+2?"])
print(result)
```

---


### Optional advanced modes

- Active adversarial evaluation: `--active-red-team`
- Drift sensitivity gate: `--fail-on-trajectory-drift`
- Shadow capture middleware helper: `shadow.fastapi`
- Reusable action reference: `dr-gareth-roberts/insideLLMs@v1`

## Documentation map

- Start here: [`wiki/index.md`](wiki/index.md)
- Getting started: [`wiki/getting-started/index.md`](wiki/getting-started/index.md)
- Concepts: [`wiki/concepts/index.md`](wiki/concepts/index.md)
- Tutorials: [`wiki/tutorials/index.md`](wiki/tutorials/index.md)
- Advanced topics: [`wiki/advanced/index.md`](wiki/advanced/index.md)
- Config reference: [`wiki/reference/Configuration.md`](wiki/reference/Configuration.md)
- API surface: [`API_REFERENCE.md`](API_REFERENCE.md)
- Architecture overview: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Contribution guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## Contributing

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a PR.

Useful local checks:

```bash
make check-fast   # lint + format-check + fast tests
make check        # lint + format-check + typecheck + full test suite
make docs-audit   # markdown + docs coverage + wiki link checks
```

---

## License

MIT. See [`LICENSE`](LICENSE).
