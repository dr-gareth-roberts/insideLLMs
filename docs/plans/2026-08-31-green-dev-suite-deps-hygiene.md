# Green Dev Suite, Dependency Diet, Repo Hygiene — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a fresh `pip install -e ".[dev]"` checkout fully green (`make check` passes end-to-end), remove two dead dependencies, and clear tracked repo residue — with docs updated to match reality.

**Architecture:** Three independent workstreams sharing one verification harness: (1) fix the single `[dev]`-failing test (a `unittest.mock.patch.object(create=True)` bug interacting with PEP 562 lazy module imports) and declare the missing `jinja2` dependency; (2) delete unused `sentry-sdk` (mandatory dep) and `tuf` (signing extra) from `pyproject.toml` and re-sync the stale `requirements-dev.txt` mirror; (3) `git rm` CLI-output residue, gitignore its regeneration paths, and add a public-API parity gate to `scripts/audit_docs.py` so `API_REFERENCE.md` cannot silently drift.

**Tech Stack:** Python 3.10–3.14, pytest 9, unittest.mock, pip/venv, Keep-a-Changelog.

**Spec:** This plan (the "spec" is the evidence section below — measured, not assumed).

## Measured Evidence (2026-08-31, this machine)

- **Bare `[dev]` venv (`.venv-dev`, Python 3.14.7, pytest 9.1.1):** full suite `1 failed, 7140 passed, 353 skipped` (147s). Sole failure: `tests/test_coverage_w7_0008_slice20.py::test_config_loader_model_probe_paths — ModuleNotFoundError: No module named 'openai'`.
- **Rich venv (`.venv`, Python 3.13.13, near-`[all]`, jinja2 present):** `7492 passed, 72 skipped, 0 failed` (240s).
- `ruff check .` and `ruff format --check .` pass in `[dev]`; `mypy insideLLMs` passes in `[dev]` (235 files, no issues).
- **Root cause of the 1 failure:** `mock.patch.object(models_mod, "OpenAIModel", _Stub, create=True)` — mock's `get_original()` falls back to `getattr(target, name)` when the attribute is absent from `__dict__`, which fires `insideLLMs/models/__init__.py:530` PEP 562 `__getattr__` → `importlib.import_module("insideLLMs.models.openai")` → `from openai import ...` (`insideLLMs/models/openai.py:48`) → `ModuleNotFoundError` in core-only installs. Verified by standalone reproduction.
- **sentry-sdk:** declared mandatory (`pyproject.toml:43`); zero `import sentry_sdk`/`from sentry_sdk` anywhere in the repo. Installed into fresh `[dev]` venvs for nothing.
- **tuf:** declared in `signing` extra (`pyproject.toml:64`); zero imports. `insideLLMs/datasets/tuf_client.py` is deliberately fail-closed (refuses without `allow_mock=True`); its tests never import tuf unguarded. `insidellms doctor` probes tuf via `importlib.util.find_spec` with a standalone `pip install "tuf>=3.0.0"` hint — unaffected by extra removal.
- **jinja2:** `pandas.Styler.to_html()` requires `jinja2>=3.1.5` (AGENTS.md documents 5 residual `[all]` failures); `jinja2` is in no extra in `pyproject.toml`. The visualization extra (which provides pandas) must declare it.
- **requirements-dev.txt drift:** claims to mirror pyproject but (a) carries a `# Documentation (synced with pyproject.toml [docs] optional dependencies)` sphinx section — pyproject has **no `[docs]` extra** and nothing in the repo builds with sphinx (no `conf.py`; `pages.yml` installs no Python deps); (b) omits `pydantic>=2.0.0` which **is** in `[dev]`. Dependabot has been filing phantom sphinx update PRs (#45, #66).
- **Tracked residue (git ls-files confirmed):** `FIXES_APPLIED.md` (26KB one-time audit report, referenced only by itself), `experiment.yaml`, `test_config.yaml` (outputs of `insidellms init` run at repo root — both point at `data/questions.jsonl`), `log.txt`, `trace_export.json` (outputs of `contrib/debugging.py` docstring examples), `trace.json`, `fingerprint.json` (0-byte output stubs). Nothing references any of them (code, tests, docs audit, wiki).
- **`MONSTER_LOOP.md` stays** (revision from the earlier proposal): it is referenced by tracked, append-only process records (`.loop/BACKLOG.json`, `.loop/LOG.md`, `.goals/*/goal.md`). Deleting it would corrupt the referential integrity of an active audit process. `FIXES_APPLIED.md` has no such references.
- **`.venv`/`.venv-dev` are untracked AND not gitignored** — they show as `git status` noise.
- CI (`ci.yml`): test job installs `[dev,openai,anthropic,nlp]` on 3.10/3.11/3.12, runs `-m "not slow and not integration and not performance"` with `--cov-fail-under=88`. Guards added in this plan only skip when a dependency is absent, so CI coverage is unaffected. `docs-audit` is not gated in any workflow (make target only).
- `uv.lock` is untracked and gitignored — no lock maintenance needed.
- `tests/test_docs_audit_contract.py` (contract-marked) runs `scripts/audit_docs.py` end-to-end and asserts exit 0.

## Global Constraints

- No new dependencies. `jinja2>=3.1.5` is declaring an existing undeclared transitive requirement (pandas Styler), not adding a new one.
- Do not weaken CI: no changes to `ci.yml` installs, marker filters, or coverage thresholds.
- Follow existing conventions: in-test guard styles already in `tests/`, Keep-a-Changelog sections, `audit_docs.py` parity-check pattern, root-scoped `.gitignore` entries (precedent: `/repo_summary.txt`, `diff.schema.json`).
- Do not touch `MONSTER_LOOP.md`, `.loop/`, `.goals/`, the user's `.venv`, or untracked user data (`copilot-debug-logs*`).
- **Commits only when the user asks** (user rule). This plan's commit steps are for a future executor with commit authority; in-session implementation stages changes and reports.
- Out of scope (noted, not fixed): `insideLLMs/models/__init__.py:__getattr__` lets `ModuleNotFoundError` escape instead of raising `AttributeError`, breaking `getattr(m, name, default)` semantics on core-only installs. Changing public import-error semantics is a separate decision.
- Scratch venvs (`.venv-dev`, `.venv-vis`, `.venv-check`) are created inside the repo for verification and deleted in Task 7. The user's `.venv` is never modified.

---

### Task 1: Fix the lazy-module patch bug in `test_config_loader_model_probe_paths`

**Files:**
- Modify: `tests/test_coverage_w7_0008_slice20.py:127-148`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: a `[dev]`-green full suite (Task 7 depends on this).

- [x] **Step 1: Replace the `patch.object(..., create=True)` block with direct `setattr`/`try/finally`**

`patch.object(..., create=True)` is the bug (see Evidence). Plain `setattr` never triggers the PEP 562 `__getattr__`, and `try/finally` restores the module state in both envs. Replace lines 139-141:

```python
    # Force registry miss → direct import path (434-437)
    import insideLLMs.models as models_mod

    with patch.object(cl.model_registry, "get", side_effect=NotFoundError("x")):
        with patch.object(models_mod, "OpenAIModel", _Stub, create=True):
            m = cl._create_model_from_config({"type": "openai", "args": {"api_key": "k"}})
            assert isinstance(m, _Stub)
```

with:

```python
    # Force registry miss → direct import path (434-437).
    # NOTE: use plain setattr, not patch.object(create=True) — mock's original-attribute
    # lookup falls back to getattr(), which fires the package's PEP 562 lazy
    # __getattr__ and imports the real provider SDK (ModuleNotFoundError on
    # core-only installs).
    import insideLLMs.models as models_mod

    with patch.object(cl.model_registry, "get", side_effect=NotFoundError("x")):
        setattr(models_mod, "OpenAIModel", _Stub)
        try:
            m = cl._create_model_from_config({"type": "openai", "args": {"api_key": "k"}})
            assert isinstance(m, _Stub)
        finally:
            del models_mod.OpenAIModel
```

The probe half of the test (lines 143-148) is untouched — it exercises the probe fallback and needs no SDK.

- [x] **Step 2: Verify the test passes in bare `[dev]` (`.venv-dev`) — it must run, not skip**

Run: `.venv-dev/bin/python -m pytest tests/test_coverage_w7_0008_slice20.py -q`
Expected: `12 passed` (whole file; the previously-failing test now passes because the stub class short-circuits the lazy import).

- [x] **Step 3: Verify no regression in the rich env**

Run: `.venv/bin/python -m pytest tests/test_coverage_w7_0008_slice20.py -q`
Expected: all passed.

- [x] **Step 4: Verify the full `[dev]` suite is green**

Run: `.venv-dev/bin/python -m pytest --tb=short -q`
Expected: `0 failed` (≈7141 passed, ≈353 skipped).

- [ ] **Step 5: Commit (executor with commit authority only)**

```bash
git add tests/test_coverage_w7_0008_slice20.py
git commit -m "fix(tests): avoid patch.object(create=True) on lazy model imports

mock's get_original() getattr fallback fires insideLLMs.models PEP 562
__getattr__ and imports the openai SDK, failing core-only [dev] installs.
Use plain setattr/try-finally instead."
```

---

### Task 2: Declare `jinja2` in the visualization extra

**Files:**
- Modify: `pyproject.toml:75-80` (visualization extra)
- Modify: `requirements-dev.txt:16-20` (visualization section)

**Interfaces:**
- Produces: `pip install ".[visualization]"` provides everything `pandas.Styler` needs; `[all]` therefore fully green.

- [x] **Step 1: Add jinja2 to the visualization extra in `pyproject.toml`**

```toml
visualization = [
    "matplotlib>=3.7.0",
    "pillow>=12.1.1",
    "pandas>=2.0.0",
    "seaborn>=0.12.0",
    "jinja2>=3.1.5",
]
```

(`jinja2>=3.1.5` is the pandas Styler floor documented in AGENTS.md.)

- [x] **Step 2: Mirror in `requirements-dev.txt`**

```text
# Visualization (synced with pyproject.toml [visualization] optional dependencies)
matplotlib>=3.7.0
pillow>=12.1.1
pandas>=2.0.0
seaborn>=0.12.0
jinja2>=3.1.5
```

- [x] **Step 3: Verify end-to-end in a fresh env**

Run:
```bash
python3 -m venv .venv-vis
.venv-vis/bin/pip install -q -e ".[dev,visualization]"
.venv-vis/bin/python -c "import jinja2; print(jinja2.__version__)"
.venv-vis/bin/python -m pytest tests/test_visualization_coverage.py tests/test_visualization.py -q
```
Expected: jinja2 version prints; both test files fully pass (the Styler-dependent `TestExperimentExplorer` cases run rather than skip).

- [ ] **Step 4: Commit (executor with commit authority only)**

```bash
git add pyproject.toml requirements-dev.txt
git commit -m "build: declare jinja2>=3.1.5 in visualization extra for pandas Styler"
```

---

### Task 3: Dependency diet — remove `sentry-sdk` (mandatory) and `tuf` (signing extra); re-sync `requirements-dev.txt`

**Files:**
- Modify: `pyproject.toml:41-44` (dependencies), `pyproject.toml:62-65` (signing extra)
- Modify: `requirements-dev.txt` (add pydantic to dev section; delete phantom sphinx section)

**Interfaces:**
- Produces: mandatory deps = `pyyaml>=6.0` only; `signing` extra = `oras>=0.2.0` only.

- [x] **Step 1: Remove sentry-sdk from mandatory dependencies (`pyproject.toml:41-44`)**

```toml
dependencies = [
    "pyyaml>=6.0",
]
```

Zero imports repo-wide (see Evidence); it currently forces a telemetry SDK into every install.

- [x] **Step 2: Remove tuf from the signing extra (`pyproject.toml:62-65`)**

```toml
signing = [
    "oras>=0.2.0",
]
```

`tuf_client.py` is fail-closed by design; doctor's `ultimate:tuf` check probes `find_spec("tuf")` independently and keeps its standalone `pip install "tuf>=3.0.0"` hint — no code change needed.

- [x] **Step 3: Re-sync `requirements-dev.txt`**

Add to the dev-tools section (mirrors `[dev]`, which includes `pydantic>=2.0.0`):

```text
pydantic>=2.0.0
```

Delete the entire phantom section (lines 33-37):

```text
# Documentation (synced with pyproject.toml [docs] optional dependencies)
sphinx>=8.1.3
sphinx-autodoc-typehints>=3.0.1
furo>=2025.12.19
sphinx-autobuild>=2024.10.3
```

There is no `[docs]` extra in pyproject, no sphinx build anywhere in the repo, and no workflow installing these. Removing them also stops dependabot's phantom update PRs.

- [x] **Step 4: Verify in a fresh env**

Run:
```bash
python3 -m venv .venv-check
.venv-check/bin/pip install -q -e ".[dev]"
.venv-check/bin/python -c "import importlib.util; assert importlib.util.find_spec('sentry_sdk') is None; print('sentry absent: OK')"
.venv-check/bin/python -c "
import importlib.metadata as md
reqs = [r for r in md.requires('insidellms') if 'extra == \"signing\"' in r]
print(reqs)
assert not any('tuf' in r for r in reqs), 'tuf still in signing extra'
"
.venv-check/bin/python -m insidellms.cli doctor >/dev/null; echo "doctor rc=$?"
```
Expected: sentry absent; signing extra metadata lists only oras; doctor exits 0 (warnings about missing optional deps are by design).

- [x] **Step 5: Verify the suite is still green in the fresh env (no hidden sentry/pydantic consumer)**

Run: `.venv-check/bin/python -m pytest --tb=no -q`
Expected: `0 failed` (≈7141 passed, ≈353 skipped — same shape as the pre-change `[dev]` run).

- [ ] **Step 6: Commit (executor with commit authority only)**

```bash
git add pyproject.toml requirements-dev.txt
git commit -m "build: drop unused sentry-sdk and tuf dependencies; sync requirements-dev

sentry-sdk was a mandatory dep never imported anywhere; tuf backs a
deliberately unimplemented, fail-closed client. requirements-dev.txt
mirrored a nonexistent [docs] extra (sphinx) and missed pydantic."
```

---

### Task 4: Remove tracked residue and gitignore regeneration paths

**Files:**
- Delete (git rm): `FIXES_APPLIED.md`, `experiment.yaml`, `test_config.yaml`, `log.txt`, `trace.json`, `trace_export.json`, `fingerprint.json`
- Modify: `.gitignore` (append section)

**Interfaces:**
- Produces: clean `git status`; future `insidellms init` / docstring-example runs at repo root never re-add noise.

- [x] **Step 1: git rm the residue**

```bash
git rm FIXES_APPLIED.md experiment.yaml test_config.yaml log.txt trace.json trace_export.json fingerprint.json
```

`MONSTER_LOOP.md` is deliberately kept (see Evidence). `FIXES_APPLIED.md` content remains recoverable from git history.

- [x] **Step 2: Append root-scoped ignore patterns to `.gitignore`**

```text

# Local CLI/demo output at repo root (insidellms init, docstring examples)
/experiment.yaml
/test_config.yaml
/trace.json
/fingerprint.json
/trace_export.json
/log.txt

# Virtual environments
/.venv/
/.venv-*/
```

Root-scoped (leading `/`) so legit files elsewhere — e.g. `examples/experiment.yaml` (referenced by `examples/README.md`) — stay trackable. The venv patterns stop `git status` noise from scratch envs.

- [x] **Step 3: Verify**

Run:
```bash
git check-ignore -v experiment.yaml test_config.yaml trace.json log.txt .venv .venv-dev
python3 scripts/audit_docs.py
```
Expected: every path matched by a rule; docs audit passes (none of the deleted files are audit targets or cross-referenced).

- [ ] **Step 4: Commit (executor with commit authority only)**

```bash
git add .gitignore
git commit -m "chore: remove tracked CLI-output residue; ignore its regeneration

Keeps MONSTER_LOOP.md (referenced by tracked .loop/.goals process
records); drops FIXES_APPLIED.md (one-time audit report, unreferenced)
and root-level init/trace/debug output stubs."
```

---

### Task 5: Public-API parity gate for `API_REFERENCE.md`

**Files:**
- Modify: `scripts/audit_docs.py` (new helper + parity loop in `main()`)
- Modify: `tests/test_docs_audit_contract.py` (unit test for the helper)
- Modify: `API_REFERENCE.md` (add missing entries; see Step 4)

**Interfaces:**
- Consumes: `insideLLMs.__all__` (eager exports) and `insideLLMs._LAZY_IMPORTS` (PEP 562 lazy names) — both module-level on the package.
- Produces: `audit_docs.py` fails with `API_REFERENCE.md missing documentation for: <Name>` for any public export absent from the reference — the same pattern as the existing CLI/probe/model parity checks.

- [x] **Step 1: Write the failing unit test in `tests/test_docs_audit_contract.py`**

```python
def test_exported_api_names_are_public_and_present() -> None:
    from scripts.audit_docs import _get_exported_api_names

    names = _get_exported_api_names()

    assert names  # non-empty
    assert all(not n.startswith("_") for n in names)  # public names only
    api_reference = (Path(__file__).resolve().parents[1] / "API_REFERENCE.md").read_text(
        encoding="utf-8"
    )
    missing = sorted(n for n in names if f"`{n}`" not in api_reference)
    assert not missing, f"API_REFERENCE.md missing: {missing}"
```

- [x] **Step 2: Run it to verify it fails**

Run: `.venv-dev/bin/python -m pytest tests/test_docs_audit_contract.py -q`
Expected: FAIL — `ImportError: cannot import name '_get_exported_api_names'`.

- [x] **Step 3: Implement the helper and parity check in `scripts/audit_docs.py`**

Add near the probe/model helpers (same style):

```python
# =========================================================================
# Public API Parity Checks
# =========================================================================


def _get_exported_api_names() -> set[str]:
    """All public names exported by the insideLLMs package (eager + lazy)."""
    import insideLLMs

    exported = set(getattr(insideLLMs, "__all__", ()))
    exported |= set(getattr(insideLLMs, "_LAZY_IMPORTS", {}))
    return {name for name in exported if not name.startswith("_")}
```

And in `main()`, after the Model Parity Check block:

```python
    # =========================================================================
    # Public API Parity Check
    # =========================================================================
    exported_api = _get_exported_api_names()
    for name in sorted(_missing_tokens(api_reference, exported_api)):
        failures.append(f"API_REFERENCE.md missing documentation for: {name}")
```

(`api_reference` is already read at `main()`'s line 249; `_missing_tokens` checks bare-token presence, matching the file's existing convention.)

- [x] **Step 4: Run the audit and fix the fallout**

Run: `.venv-dev/bin/python -m pytest tests/test_docs_audit_contract.py -q`
Expected: FAIL with a concrete `missing` list.

For each missing name: add an entry to `API_REFERENCE.md` if it is genuinely public API (preferred — that is the point of the gate), or, only for names that are deprecated aliases or not user-facing, skip that name in `_get_exported_api_names()` with an inline comment naming the reason. Iterate until the audit passes.

- [x] **Step 5: Verify the gate detects drift (mutation check)**

Temporarily delete one documented name from `API_REFERENCE.md`, run `python3 scripts/audit_docs.py`, confirm exit 1 with the `missing documentation for:` message, then restore.

- [x] **Step 6: Run the full docs audit**

Run: `.venv-dev/bin/python -m pytest tests/test_docs_audit_contract.py -q && .venv-dev/bin/python scripts/audit_docs.py`
Expected: both pass (`Documentation audit passed.`).

- [ ] **Step 7: Commit (executor with commit authority only)**

```bash
git add scripts/audit_docs.py tests/test_docs_audit_contract.py API_REFERENCE.md
git commit -m "docs: add public-API parity gate for API_REFERENCE.md

Every insideLLMs export (eager __all__ + lazy imports) must appear in
API_REFERENCE.md, mirroring the CLI/probe/model parity checks."
```

---

### Task 6: Truth-sync AGENTS.md caveats and CHANGELOG

**Files:**
- Modify: `AGENTS.md` ("Non-obvious caveats" section)
- Modify: `CHANGELOG.md` (`[Unreleased]` Added/Fixed/Changed/Removed)

**Interfaces:** none.

- [x] **Step 1: Replace the two stale caveats bullets in `AGENTS.md`**

Delete the `**Test failures with `[dev]` only**` and `**Residual failures with `[all]`**` bullets (their ~164/~5 counts are stale — measured reality: bare `[dev]` full suite is green after Task 1) and replace with:

```markdown
- **Green-by-default suite**: a fresh `pip install -e ".[dev]"` passes the full test suite (optional-dependency tests skip, not fail — ~350 skips in a bare `[dev]` env). Provider SDK tests that need the real package are guarded; `jinja2` is declared in the `visualization` extra for pandas Styler.
```

Keep the PATH, `python`/`python3`, doctor, validate, and golden-path bullets unchanged.

- [x] **Step 2: Add CHANGELOG entries under `[Unreleased]`**

Under the existing `### Fixed`: the `patch.object(create=True)` lazy-import test fix (core-only installs fail the suite).
Under `### Changed`: `jinja2>=3.1.5` added to `visualization` extra.
Under `### Removed`: `sentry-sdk` mandatory dependency (never imported); `tuf` from `signing` extra (client is fail-closed by design); tracked root-level CLI-output residue; phantom sphinx section in `requirements-dev.txt`.
Under `### Added`: public-API parity gate in `scripts/audit_docs.py` for `API_REFERENCE.md`.

- [x] **Step 3: Verify**

Run: `.venv-dev/bin/python scripts/audit_docs.py && .venv-dev/bin/python -m pytest tests/test_docs_audit_contract.py -q`
Expected: pass (neither file is an audit target, but re-run to prove no collateral).

- [ ] **Step 4: Commit (executor with commit authority only)**

```bash
git add AGENTS.md CHANGELOG.md
git commit -m "docs: sync AGENTS.md caveats and CHANGELOG with measured suite state"
```

---

### Task 7: Final verification and cleanup

**Files:** none modified; verification + scratch-env cleanup + codemap state refresh.

- [x] **Step 1: Fresh-env full gate**

```bash
python3 -m venv .venv-final
.venv-final/bin/pip install -q -e ".[dev]"
.venv-final/bin/python -m ruff check . && .venv-final/bin/python -m ruff format --check .
.venv-final/bin/python -m mypy insideLLMs
.venv-final/bin/python -m pytest --tb=short -q
make golden-path PYTHON=.venv-final/bin/python
```

Expected: lint clean, 560 files formatted, mypy no issues, **0 failed** full suite, golden-path harness→diff cycle exits 0.

- [x] **Step 2: Clean up scratch envs**

```bash
rm -rf .venv-dev .venv-vis .venv-check .venv-final
```

(The user's `.venv` is untouched.)

- [x] **Step 3: Refresh codemap state for the changed files**

```bash
node ~/.config/opencode/skills/codemap/scripts/codemap.mjs changes --root ./
node ~/.config/opencode/skills/codemap/scripts/codemap.mjs update --root ./
```

Expected changes: `pyproject.toml` (modified), `requirements-dev.txt` (modified), `scripts/audit_docs.py` (modified). Update `codemap.md` (root atlas) line naming `sentry-sdk` as a mandatory dep, and `scripts/codemap.md` if its audit description is affected.

- [x] **Step 4: Report**

Summarize with measured before/after: `[dev]` suite `1 failed → 0 failed`; deps removed; files deleted; docs-audit parity count.

---

## Self-Review

- **Spec coverage:** Item 1 = Tasks 1+2 (+AGENTS truth-sync in 6); item 2 = Task 3 (+mirror sync); item 6 = Tasks 4+5. ✔
- **Placeholder scan:** no TBDs; every step has exact content or a concrete discovery loop (Task 5 Step 4 is an evidence-driven fix loop with a defined exit condition). ✔
- **Type consistency:** `_get_exported_api_names() -> set[str]` used identically in audit and test; `_Stub`/`models_mod` names match the existing test body. ✔
- **Scope revisions vs the original proposal (documented):** (1) "164 failures" was stale — actual work is one test bug + one missing dep; (2) `MONSTER_LOOP.md` is kept — tracked process records reference it; (3) API_REFERENCE gets a parity gate, not regeneration — the 72KB is hand-curated with examples, the repo's established pattern is parity checking, and curation is worth preserving.
