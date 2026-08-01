# AGENTS.md

## Cursor Cloud specific instructions

### Setup (run once per cloud environment)

```bash
bash scripts/cloud-setup.sh
source .venv/bin/activate
```

Default install is lean: editable `.[dev]` + `pydantic`, no provider keys, ends with import smoke + architecture tests + `make golden-path`.

Optional overlays (comma-separated):

```bash
CLOUD_EXTRAS=providers bash scripts/cloud-setup.sh
CLOUD_EXTRAS=nlp bash scripts/cloud-setup.sh
CLOUD_EXTRAS=all bash scripts/cloud-setup.sh   # last resort; slow
```

Do not default to `.[all]` in cloud setup — nlp/transformers bloat cold start.

### Project overview

insideLLMs is a Python CLI tool/library for LLM behavioural regression testing. It produces deterministic, diffable artefacts and can gate CI pipelines on behavioural changes. No Docker, databases, or external services are needed for development.

### Development commands

Standard commands are in the `Makefile` and documented in `CONTRIBUTING.md`. Key ones:

- `make check` — full quality gates (lint + format-check + typecheck + test)
- `make check-fast` — quick pre-commit (lint + format-check + test-fast)
- `make golden-path` — offline harness + diff using DummyModel (no API keys needed)
- `make lint` / `make format-check` / `make typecheck` / `make test`

### Non-obvious caveats

- **PATH**: Scripts installed by `pip install -e ".[dev]"` go to `~/.local/bin`. Ensure `export PATH="$HOME/.local/bin:$PATH"` is in your shell profile, or invoke tools via `python3 -m pytest`, `python3 -m ruff`, etc. `scripts/cloud-setup.sh` exports this for the session.
- **`python` vs `python3`**: The system provides `python3` but not `python`. Use `python3` explicitly, or use Makefile targets (which default to `PYTHON=python3` and can be overridden). The setup script links `python` into `~/.local/bin` when missing.
- **Test failures with `[dev]` only**: ~164 tests fail because they exercise optional provider integrations (OpenAI, Anthropic, HuggingFace) and optional packages (`nltk`). These are not regressions — the core suite of 6300+ tests passes. Use `CLOUD_EXTRAS=providers` / `nlp` only when the task needs them (full integration still needs API keys).
- **`insidellms doctor`**: Reports missing optional dependencies as warnings. In `[dev]`-only environments, expect `nltk`-related warnings but no hard failure.
- **`insidellms validate`**: Requires `pydantic`. Cloud setup installs it; a bare `.[dev]` install does not.
- **Golden-path is the best offline demo**: `make golden-path` runs a full harness→diff cycle using the built-in `DummyModel` with no API keys or network access required.
