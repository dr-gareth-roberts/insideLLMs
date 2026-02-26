# Audit Remaining Cleanup — Design

**Date**: 2026-02-26
**Status**: Approved
**Target version**: 0.2.0

## Context

The codebase audit (see `docs/AUDIT_FINDINGS.md`) resolved 11 issues but left 5
items requiring product decisions. This design covers all 5 in a single pass.

The project is pre-1.0 with no published PyPI release. There are no external
consumers of the deprecated import paths, so breaking changes carry no real cost.

## Changes

### 1. Delete 6 deprecated shim modules

Remove these re-export shims and repoint all consumers to canonical modules:

| Shim (delete) | Canonical module | Consumers to repoint |
|----------------|-----------------|----------------------|
| `cache.py` | `caching_unified.py` (becomes `caching.py`, see §2) | `tests/test_cache.py` (delete) |
| `caching.py` | `caching_unified.py` (becomes `caching.py`, see §2) | `tests/test_caching.py` (delete) |
| `runner.py` | `runtime/runner.py` | Docs only (QUICK_REFERENCE, API_REFERENCE) |
| `comparison.py` | `analysis/comparison.py` | `tests/test_comparison.py` |
| `statistics.py` | `analysis/statistics.py` | `tests/test_statistics.py`, `cli/commands/report.py` |
| `trace_config.py` | `trace/trace_config.py` | `__init__.py`, `probes/agent_probe.py`, `tests/test_trace_config.py`, `tests/test_agent_probe.py` |

`prompt_testing.py` was misclassified as deprecated in the audit — it is a
standalone module with its own implementation, not a shim. It stays.

Also remove ruff per-file-ignores for deleted files from `pyproject.toml`
(`insideLLMs/cache.py`, `insideLLMs/caching.py`).

### 2. Rename `caching_unified.py` → `caching.py`

Once the old `caching.py` shim is deleted, rename the canonical module into the
freed filename. This is the name users would naturally expect.

| File (old) | File (new) |
|------------|-----------|
| `insideLLMs/caching_unified.py` | `insideLLMs/caching.py` |
| `tests/test_caching_unified.py` | `tests/test_caching.py` |

Update imports in:
- `insideLLMs/semantic_cache.py` (`from insideLLMs.caching_unified` → `from insideLLMs.caching`)
- `insideLLMs/__init__.py` (if it re-exports caching symbols)
- Any other internal references

The `pyproject.toml` ruff per-file-ignore for `insideLLMs/caching.py` remains
valid (same filename, new content with top-level import guards).

### 3. Verify and fix ORAS package name

`pyproject.toml` declares `"oras>=0.2.0"`. The code does
`from oras import client as oras_client`. Verify on PyPI whether the correct
package name is `oras` or `oras-py`. Fix the dependency declaration and/or
import if they don't match.

### 4. Clean up policy engine

- Remove the unused `policy_yaml_path` parameter from `run_policy()` in
  `insideLLMs/policy/engine.py` and update its docstring to accurately describe
  the hardcoded checks it performs.
- Update callers of `run_policy()` that pass `policy_yaml_path` (search
  codebase for all callsites).
- Add a module-level note in `insideLLMs/policy/rules/__init__.py` explaining
  that no built-in rules ship with the package.

### 5. Version bump to 0.2.0

- Update `version` in `pyproject.toml` from `"0.1.0"` to `"0.2.0"`.
- Add a CHANGELOG entry under `## [0.2.0]` documenting:
  - **Removed**: 6 deprecated shim modules (list the old import paths)
  - **Changed**: `caching_unified` renamed to `caching`
  - **Changed**: `run_policy()` no longer accepts `policy_yaml_path`
  - **Fixed**: ORAS dependency name (if it needed fixing)

## Out of scope

- `prompt_testing.py` — not deprecated, no change needed.
- Implementing YAML-based policy rules — future feature, separate brainstorm.
- Any new functionality — this is strictly cleanup.

## Risks

- **Import breakage for docs/examples**: Any code sample referencing the old
  shim paths needs updating. Grep for all old import paths in `*.md`, `*.yaml`,
  and `examples/` files.
- **Caching rename collision**: The `pyproject.toml` ruff ignore for
  `insideLLMs/caching.py` was for the old shim's `E402`. Verify the renamed
  canonical module still needs it (it likely does — it has conditional imports).

## Verification

After all changes:

```bash
python -m pytest tests/ -q
ruff check .
ruff format --check .
```

All tests must pass. Zero ruff violations. No import errors.
