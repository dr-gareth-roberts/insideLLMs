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
- **`python` vs `python3`**: The system provides `python3` but not `python`. Use `python3` explicitly, or the Makefile targets which use `python -m pytest` (which may need a `python` symlink or alias).
- **Test failures with `[dev]` only**: ~164 tests fail because they exercise optional provider integrations (OpenAI, Anthropic, HuggingFace) and optional packages (`pydantic`, `nltk`). These are not regressions — the core suite of 6300+ tests passes. Install `pip install -e ".[all]"` to reduce failures (requires provider API keys for full integration tests).
- **`insidellms doctor`**: Requires `nltk` (part of `[nlp]` extra, not `[dev]`). It will error if nltk is missing. Use `insidellms schema list` or the golden-path for a quick sanity check instead.
- **`insidellms validate`**: Requires `pydantic` (not in `[dev]`). The harness and diff commands work without it.
- **Golden-path is the best offline demo**: `make golden-path` runs a full harness→diff cycle using the built-in `DummyModel` with no API keys or network access required.
