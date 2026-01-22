# Repository Guidelines

## Project Structure & Module Organization

- `insideLLMs/`: main Python package (library + CLI entrypoint in `insideLLMs/cli.py`).
- `tests/`: pytest suite (`test_*.py`) and shared fixtures in `tests/conftest.py`.
- `examples/`: runnable scripts and example configs (start with `examples/example_quickstart.py`).
- `ci/`: small deterministic harness inputs used for CI diff-gating (`ci/harness.yaml`, `ci/harness_dataset.jsonl`).
- `docs/`, `wiki/`: documentation and planning notes; user-facing docs are primarily Wiki-linked from `README.md`.
- `benchmarks/`, `data/`: benchmark/data assets and artefacts used by examples and experiments.

## Build, Test, and Development Commands

Create an environment and install editable deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,nlp,visualization]"
```

Common checks (mirrors CI):

```bash
ruff check .
ruff format --check .
mypy insideLLMs
pytest
```

Run the CLI locally (write outputs to a temp run dir):

```bash
insidellms harness examples/harness.yaml --run-dir .tmp/runs/dev
insidellms report .tmp/runs/dev
```

## Coding Style & Naming Conventions

- Python 3.10+, 4-space indentation, and Ruff formatting (line length: 100).
- Prefer type hints on public APIs; keep config surfaces explicit (often via dataclasses).
- Use `snake_case` for modules/functions, `PascalCase` for types, and keep imports sorted via Ruff.
- When changing the run/report/diff spine, preserve determinism (stable ordering; avoid raw timestamps/UUIDs in artefacts).

## Testing Guidelines

- Framework: `pytest` + `pytest-asyncio`; markers include `slow` and `integration`.
  - Example: `pytest -m "not slow and not integration"`.
- Coverage is enforced in CI (currently `--cov-fail-under=80` on `insideLLMs`).
- If tests create run artefacts, prefer an isolated root: `INSIDELLMS_RUN_ROOT=.tmp/insidellms_runs`.

## Commit & Pull Request Guidelines

- Follow the prevalent conventional-commit style: `feat(scope): ...`, `fix: ...`, `test: ...`, `docs: ...`, `chore: ...`.
- Keep commits atomic and PRs focused; add/adjust tests for behavior changes.
- PRs should follow `.github/PULL_REQUEST_TEMPLATE.md`: clear description, linked issue (if any), test notes, and screenshots for report/UI changes.

## Security & Configuration Tips

- Never commit credentials; configure providers via environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
- For vulnerability reports, follow `SECURITY.md` (no public issues for security findings).
