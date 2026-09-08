# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

insideLLMs is a Python CLI tool/library for LLM behavioural regression testing. It produces deterministic, diffable artefacts and can gate CI pipelines on behavioural changes. No Docker, databases, or external services are needed for development.

### Development commands

Standard commands are in the `Makefile` and documented in `CONTRIBUTING.md`. Key ones:

- `make check` — full quality gates (lint + format-check + typecheck + test)
- `make check-fast` — quick pre-commit (lint + format-check + test-fast)
- `make golden-path` — offline harness + diff using DummyModel (no API keys needed)
- `make lint` / `make format-check` / `make typecheck` / `make test`

### Non-obvious caveats

- **PATH**: Scripts installed by `pip install -e ".[dev]"` go to `~/.local/bin`. Ensure `export PATH="$HOME/.local/bin:$PATH"` is in your shell profile, or invoke tools via `python3 -m pytest`, `python3 -m ruff`, etc.
- **`python` vs `python3`**: The system provides `python3` but not `python`. Use `python3` explicitly, or use Makefile targets (which default to `PYTHON=python3` and can be overridden).
- **Green-by-default suite**: a fresh `pip install -e ".[dev]"` passes the full test suite. Tests needing optional dependencies skip (do not fail) when the extra is absent — expect ~350 skips in a bare `[dev]` env. `jinja2` (pandas Styler) is declared in the `visualization` extra.
- **`insidellms doctor`**: Reports missing optional dependencies as warnings. In `[dev]`-only environments, expect `nltk`/visualization-related warnings but no hard failure.
- **`insidellms validate`**: Pydantic is a core dependency. Configuration and artifact validation work in a base install; run and harness share the versioned configuration schema.
- **Golden-path is the best offline demo**: `make golden-path` runs a full harness→diff cycle using the built-in `DummyModel` with no API keys or network access required.

## Repository Map

A full codemap is available at `codemap.md` in the project root.

Before working on any task, read `codemap.md` to understand:
- Project architecture and entry points
- Directory responsibilities and design patterns
- Data flow and integration points between modules

For deep work on a specific folder, also read that folder's `codemap.md`.
