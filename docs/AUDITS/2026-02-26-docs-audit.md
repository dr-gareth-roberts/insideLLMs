# Documentation Audit (2026-02-26)

## Scope (Phase 1)

This audit focused on user-facing, high-traffic docs that directly affect first-run success and CI adoption:

- `README.md`
- `wiki/reference/CLI.md`
- `QUICK_REFERENCE.md`
- `DOCUMENTATION_INDEX.md`
- `CONTRIBUTING.md`

## What Was Audited

### Core feature parity checks

- `insidellms harness` profile/explain docs
- `insidellms harness` active red-team mode docs
- `insidellms diff` judge mode docs
- `insidellms diff` interactive snapshot docs
- `insidellms diff --fail-on-trajectory-drift` docs
- FastAPI production shadow capture docs (`insideLLMs.shadow.fastapi`)
- GitHub Action + PR comment workflow docs
- VS Code/Cursor extension scaffold discoverability

### Documentation reliability checks

- Added `scripts/audit_docs.py` to enforce key token/section coverage across CLI reference and README.
- Added `make docs-audit` target to run:
  - `python scripts/audit_docs.py`
  - `python scripts/check_wiki_links.py`

## Changes Applied

- Updated `wiki/reference/CLI.md`:
  - Added harness options for active red-team flags.
  - Added diff options for judge flags and trajectory drift gate.
  - Added diff example commands for judge and trajectory workflows.
  - Added exit code `5` for trajectory drift gating.
- Updated `QUICK_REFERENCE.md` with advanced harness/diff command examples.
- Updated `DOCUMENTATION_INDEX.md` to include:
  - `docs/GITHUB_ACTION.md`
  - `extensions/vscode-insidellms/README.md`
- Updated `CONTRIBUTING.md` to surface `make docs-audit`.
- Added `scripts/audit_docs.py` for repeatable docs drift detection.

## Phase 2 Completion

Phase 2 items were completed in this pass:

1. `API_REFERENCE.md` aligned with new runtime/CLI surfaces:
   - `insideLLMs.diffing` public facade
   - `DiffGatePolicy` and diff schema validation path
   - `insidellms harness` profile + active red-team options
   - `insidellms diff` judge/interactive/trajectory-gate options
   - `insidellms doctor --capabilities` and `insidellms init`
2. Added production shadow capture guide:
   - `wiki/guides/Production-Shadow-Capture.md`
3. Added VS Code/Cursor extension workflow guide:
   - `wiki/guides/IDE-Extension-Workflow.md`
4. Expanded docs audit coverage:
   - Added parser smoke checks for representative command examples.
   - Added stale-option checks for removed CLI flags in long-form docs.
   - Added API reference and docs index token parity checks for new surfaces.
