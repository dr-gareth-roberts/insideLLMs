You are an experienced, pragmatic software engineering AI agent. Do not over-engineer a solution when a simple one is possible. Keep edits minimal. If you want an exception to ANY rule, you MUST stop and get permission first.

# insideLLMs contributor guide

## Project overview

insideLLMs is a typed Python 3.10+ library and `insidellms` CLI for cross-model behavioural probes, deterministic run artefacts, and diff-gated regression testing. It is packaged with setuptools; the base runtime dependencies are PyYAML and Pydantic, while model providers and feature areas are optional extras. Core development needs no Docker, database, network service, or API key.

The repository also contains `compliance_intelligence/`, a standalone FastAPI/LangGraph demonstration with its own requirements. Do not treat it as part of the main package’s dependency graph or test suite.

## Reference

- `pyproject.toml` — package metadata, optional extras, Ruff, MyPy, pytest, and coverage configuration. The console entry point is `insideLLMs.cli:main`.
- `insideLLMs/cli/__init__.py` — CLI entry point and command dispatch. Command implementations belong in `insideLLMs/cli/commands/`.
- `insideLLMs/runtime/runner.py` — compatibility-preserving public runner exports. Keep implementation changes in the focused `runtime/_*.py` modules when appropriate; underscore-prefixed modules are internal.
- `insideLLMs/models/`, `probes/`, and `datasets/` — model, probe, and dataset implementations; `registry.py` supplies extension registries.
- `insideLLMs/schemas/` and `docs/STABILITY*.md` — versioned artifact contracts and compatibility policy.
- `architecture/` and `docs/ARCHITECTURE_GUARDS.md` — the enforced layer matrix (`layers.json`), generated import/API evidence, and the owned, expiring import exceptions checked by `make architecture`.
- `tests/` mirrors features with `test_*.py`; `tests/conftest.py` defines shared fixtures. `data/` and `ci/` contain harness fixtures/configuration.
- `docs/` contains contributor and contract documentation; user-facing GitHub Pages Markdown is under `wiki/`. Use `DOCUMENTATION_INDEX.md` to locate broader docs.
- `ARCHITECTURE.md` describes the current runner → probe → model flow and points at the guard documentation for the dependency rules.

The primary flow is Python API or CLI → runner → probe → model/provider → results and deterministic artifacts. Registries and dataset loaders support config-driven execution. Stable surfaces include CLI semantics, schema-governed artifacts, and registry extension APIs; consult `docs/STABILITY_MATRIX.md` before changing them.

## Essential commands

Use `python3` (the Makefile defaults `PYTHON` to it). Set up an isolated environment and install either the small development set or all extras:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"      # normal development
# python3 -m pip install -e ".[all]"    # provider/feature work
pre-commit install
```

| Need | Command |
| --- | --- |
| Install/build for local development | `python3 -m pip install -e ".[dev]"` |
| Format (changes files) | `make format` |
| Check formatting / lint | `make format-check` / `make lint` |
| Type-check | `make typecheck`; use `make typecheck-strict` for `injection.py` and `safety.py` |
| Fast test pass | `make check-fast` |
| Full local gates | `make check` |
| Focused tests | `make test-determinism`, `make test-contract`, `make test-adapter`, or `make test-performance` |
| Architecture guards | `make architecture` (check evidence, layers, expiry); `make architecture-update` (regenerate evidence) |
| Offline end-to-end verification | `make golden-path`; `make clean-install-golden-path` for a wheel-based isolated run |
| Installed-distribution smoke | `make package-smoke PYTHON=/path/to/clean-venv/bin/python` |
| Documentation audit | `make docs-audit` |
| Type reports | `make typecheck-report` or `make typecheck-coverage` |

There is no general Makefile `clean` target. Remove only known generated paths when needed (for example, `rm -rf .tmp mypy-report mypy-coverage`); do not delete fixture or artifact directories blindly. There is no main-package development server. To run the separate Compliance Intelligence UI, install its requirements and run it from its own directory:

```bash
cd compliance_intelligence
python3 -m pip install -r requirements.txt
python3 run_server.py  # http://localhost:8000
```

The available repository scripts are `scripts/checks.sh`, `scripts/audit_docs.py`, `scripts/check_wiki_links.py`, `scripts/check_secrets.py`, `scripts/architecture_evidence.py`, `scripts/smoke_installed_package.py`, `scripts/clean_install_golden_path.sh`, plus GitHub-action helpers. Prefer their corresponding Make targets; run a script directly only when its purpose requires it.

## Non-obvious caveats

- **PATH**: Scripts installed by `pip install -e ".[dev]"` go to `~/.local/bin`. Ensure `export PATH="$HOME/.local/bin:$PATH"` is in your shell profile, or invoke tools via `python3 -m pytest`, `python3 -m ruff`, etc.
- **`python` vs `python3`**: The system provides `python3` but not `python`. Use `python3` explicitly, or use Makefile targets (which default to `PYTHON=python3` and can be overridden).
- **Green-by-default suite**: a fresh `pip install -e ".[dev]"` passes the full test suite. Tests needing optional dependencies skip (do not fail) when the extra is absent — expect ~350 skips in a bare `[dev]` env. `jinja2` (pandas Styler) is declared in the `visualization` extra.
- **`insidellms doctor`**: Reports missing optional dependencies as warnings. In `[dev]`-only environments, expect `nltk`/visualization-related warnings but no hard failure.
- **`insidellms validate`**: Pydantic is a core dependency. Configuration and artifact validation work in a base install; run and harness share the versioned configuration schema.
- **Golden-path is the best offline demo**: `make golden-path` runs a full harness→diff cycle using the built-in `DummyModel` with no API keys or network access required.

## Patterns and quality rules

- Keep public imports on stable paths such as `insideLLMs.runtime.runner` and `insideLLMs.cli.commands.*`; do not add new external dependencies on `runtime._*` or `cli._*` internals.
- Respect the layer matrix in `architecture/layers.json`. A new cross-layer import needs either a fix or an owned, justified, expiring entry in `architecture/import_exceptions.json`; never widen the matrix to make the guard pass.
- Preserve deterministic artifact behavior. Canonical outputs include `manifest.json`, `records.jsonl`, `summary.json`, and `diff.json`; schema or semantic changes require the versioning/deprecation process in `docs/STABILITY.md`, not a silent rewrite.
- Add or update focused regression tests with behavior changes. Pytest is strict about registered markers; use only the configured markers: `slow`, `integration`, `determinism`, `contract`, `adapter`, and `performance`.
- Use Ruff for formatting and imports (100-character target line length) and MyPy for types. Do not use Black or isort as separate formatters despite their presence in development dependencies.
- For CLI, probe, or model changes, update the appropriate wiki/API material and run `make docs-audit`. Do not leave catalog or CLI documentation stale.
- Use `make golden-path` for an API-key-free harness→diff validation. `insidellms doctor` reports missing optional dependencies as warnings; do not mistake those warnings for failures.

A `[dev]` environment does not exercise optional-provider, NLP, or visualization tests; they skip rather than fail. Install the required extras for the code you change; provider integration tests may additionally require credentials. Report environment-related failures precisely rather than hiding them.

## Commit and pull request guidelines

Before committing, run the narrowest relevant tests plus `make lint`, `make format-check`, and `make typecheck`; use `make check` when the environment supports the full suite. Run `make typecheck-strict` for security-module changes, `make docs-audit` for CLI/probe/model documentation changes, `make architecture` for any import or root-facade change, and `make golden-path` for harness/diff behavior. Install and honor pre-commit hooks, which also check secrets, YAML/JSON, file endings, and private keys.

Use Conventional Commit subjects: `type(scope): description`. Existing history uses forms such as `fix: ...`, `chore(deps): ...`, and `docs(agents): ...`; valid types include `feat`, `fix`, `docs`, `style`, `refactor`, `test`, and `chore`. Keep commits atomic and do not mix unrelated refactors with functional changes.

Keep each PR focused. Complete `.github/PULL_REQUEST_TEMPLATE.md`: describe the change, link any issue, select a change type, list validation, and note documentation. For a user-facing stable-surface change, update `CHANGELOG.md` under **Unreleased**, add migration/deprecation notes where needed, and update contract tests plus `docs/STABILITY*.md`. Do not claim all tests pass when optional dependencies or credentials prevent that; state the exact command and blocker.

## Repository Map

A full codemap is available at `codemap.md` in the project root.

Before working on any task, read `codemap.md` to understand:
- Project architecture and entry points
- Directory responsibilities and design patterns
- Data flow and integration points between modules

For deep work on a specific folder, also read that folder's `codemap.md`.
