# CLAUDE.md

Project guidance for Claude Code when working in this repository.

## Quick Reference

```bash
# Install (editable, dev extras + providers)
pip install -e ".[dev,providers,nlp,visualization]"

# Or minimal (core only, no provider SDKs)
pip install -e ".[dev]"

# Lint & format
make lint              # ruff check .
make format            # ruff format . + ruff check --fix .
make format-check      # ruff format --check .

# Type check
make typecheck         # mypy insideLLMs

# Test
make test              # pytest (full suite)
make test-fast         # pytest -m "not slow and not integration"
pytest tests/test_runner.py -k test_<name>   # single test

# All checks (lint + format-check + typecheck + test)
make check

# Determinism verification (offline, diff-gating)
make golden-path

# CLI
python -m insideLLMs.cli <subcommand>
```

## Project Overview

**insideLLMs** is a cross-model behavioural probe harness for LLM evaluation. It catches behavioural regressions by comparing run artifacts across model versions. Core philosophy: same inputs produce byte-for-byte identical artifacts.

## Architecture

```
CLI (insideLLMs/cli/)
  └─> Runner (insideLLMs/runtime/runner.py)
        ├─> Registry (insideLLMs/registry.py)
        ├─> Datasets (insideLLMs/dataset_utils.py)
        ├─> Probes (insideLLMs/probes/)
        │     └─> Models (insideLLMs/models/)
        │           └─> Providers (openai / anthropic / transformers / local)
        ├─> Artifacts (records.jsonl / manifest.json / summary.json / report.html)
        └─> Infra (caching / rate_limiting / cost_tracking) [optional]
```

**Core modules (~38 top-level files):**
- `insideLLMs/cli/` — CLI entrypoint and subcommands (`run`, `harness`, `report`, `diff`, `schema`, `doctor`)
- `insideLLMs/runtime/runner.py` — Canonical execution engine (`ProbeRunner`, `AsyncProbeRunner`)
- `insideLLMs/registry.py` — Model/probe/dataset registration and lookup
- `insideLLMs/schemas/` — Versioned output schemas and validation
- `insideLLMs/results.py`, `export.py`, `visualization.py` — Aggregation and report generation
- `insideLLMs/models/` — Provider adapters (OpenAI, Anthropic, HuggingFace, Cohere, Gemini, local)
- `insideLLMs/probes/` — Probe implementations (logic, bias, attack, code, instruction, agent, factuality)

**Contrib modules (~39 files in `insideLLMs/contrib/`):**
- Standalone research/analysis modules NOT part of the core pipeline
- Includes: agents, hitl, deployment, routing, synthesis, calibration, hallucination, etc.
- Still importable via lazy loading: `from insideLLMs import ReActAgent`
- Tests live in `tests/contrib/`

**Data flow:** config → runner/harness → `records.jsonl` → `summary.json` / `report.html` → `diff.json`

## Determinism Rules

Determinism is a hard requirement for the artifact spine. When modifying code that touches artifacts:

- Emit canonical JSON/JSONL with **stable key ordering** and consistent separators
- Derive timestamps from `run_id`, never wall-clock time
- Persist volatile fields as `null` (`latency_ms`, `manifest.json:command`)
- Content-address local datasets with `dataset_hash=sha256:<file-bytes>`
- Keep all lists/sets in stable order when they affect artifact output

Determinism can be toggled via config but defaults to strict:
```yaml
determinism:
  strict_serialization: true       # default
  deterministic_artifacts: true    # follows strict_serialization by default
```

## CI Constraints

CI runs in `.github/workflows/ci.yml`:

- **Lint job (Python 3.12):** Ruff lint + Ruff format check
- **Typecheck job (Python 3.12):** Mypy (standard + strict on security modules)
- **Test job (Python 3.10, 3.11, 3.12 matrix):** Full pytest with `--cov-fail-under=80`
- **Determinism job:** `make golden-path` (needs lint + typecheck to pass first)
- **Contracts job:** `pytest -m contract` (stability contract tests)
- Coverage threshold is **80%** (will be raised incrementally)
- Tests write artifacts under `INSIDELLMS_RUN_ROOT` when applicable
- Ruff line-length is **100** characters (`pyproject.toml`)
- Ruff selects: `E`, `W`, `F`, `I` (pycodestyle, pyflakes, isort)

## Style & Conventions

- Python 3.10+ (no walrus operators or newer-only syntax beyond 3.10)
- Ruff for linting and formatting (not black/isort directly, though they're in dev deps)
- Mypy for type checking (strict on core modules: runtime/, cli/, schemas/; lenient on contrib/)
- `asyncio_mode = "auto"` in pytest (no need for `@pytest.mark.asyncio`)
- Test markers: `@pytest.mark.slow`, `@pytest.mark.integration`

## Dependencies

Core has only `pyyaml` as a required dependency. Everything else is optional:

```bash
pip install insidellms                    # core only (pyyaml)
pip install insidellms[openai]            # + OpenAI SDK
pip install insidellms[anthropic]         # + Anthropic SDK
pip install insidellms[huggingface]       # + transformers + huggingface-hub
pip install insidellms[providers]         # all three providers
pip install insidellms[signing]           # + tuf + oras
pip install insidellms[crypto]            # + cryptography
pip install insidellms[all]              # everything
```

All provider imports are lazy-guarded — the core package imports without provider SDKs installed.

## Common Development Patterns

**Adding a CLI subcommand:** Wire in `insideLLMs/cli/` (`_parsing.py` + `commands/`), route into `insideLLMs/runtime/runner.py`.

**Adding a model provider:** Create adapter in `insideLLMs/models/`, register with `model_registry`.

**Adding a probe:** Implement in `insideLLMs/probes/`, register with `probe_registry`. Keep outputs compatible with versioned schemas.

**Changing schemas/artifacts:** Update `insideLLMs/schemas/`, ensure `insidellms schema validate` stays aligned. Treat schema changes as part of the CI diff surface.
