# Audit Remaining Cleanup — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove 6 deprecated shim modules, rename `caching_unified.py` → `caching.py`, clean up the policy engine, and bump to v0.2.0.

**Architecture:** Delete deprecated shim files, repoint all consumers to canonical module paths, rename the caching module into the freed filename, remove unused policy parameter, update version and changelog.

**Tech Stack:** Python, pytest, ruff

---

### Task 1: Delete `cache.py` shim and its test

**Files:**
- Delete: `insideLLMs/cache.py`
- Delete: `tests/test_cache.py`

**Step 1: Delete the files**

```bash
rm insideLLMs/cache.py tests/test_cache.py
```

**Step 2: Run tests to confirm nothing else breaks**

Run: `python -m pytest tests/ -q --ignore=tests/test_nlp_classification_full.py --ignore=tests/test_nlp_keyword_extraction_full.py -x 2>&1 | tail -5`
Expected: Collection error — `__init__.py` lazy imports reference `insideLLMs.cache`. This is expected; we fix it in Task 7.

**Step 3: Commit**

```bash
git add -A && git commit -m "remove: delete deprecated cache.py shim and its test"
```

---

### Task 2: Delete `caching.py` shim and its test

**Files:**
- Delete: `insideLLMs/caching.py`
- Delete: `tests/test_caching.py`

**Step 1: Delete the files**

```bash
rm insideLLMs/caching.py tests/test_caching.py
```

**Step 2: Commit**

```bash
git add -A && git commit -m "remove: delete deprecated caching.py shim and its test"
```

---

### Task 3: Rename `caching_unified.py` → `caching.py`

**Files:**
- Rename: `insideLLMs/caching_unified.py` → `insideLLMs/caching.py`
- Rename: `tests/test_caching_unified.py` → `tests/test_caching.py`
- Modify: `tests/test_caching_unified_branch_coverage.py:10` — update import
- Modify: `insideLLMs/semantic_cache.py:179` — update import

**Step 1: Rename the files**

```bash
git mv insideLLMs/caching_unified.py insideLLMs/caching.py
git mv tests/test_caching_unified.py tests/test_caching.py
```

**Step 2: Update import in `tests/test_caching_unified_branch_coverage.py`**

Change line 10:
```python
# Old
from insideLLMs.caching_unified import (
# New
from insideLLMs.caching import (
```

**Step 3: Update import in `insideLLMs/semantic_cache.py`**

Change line 179:
```python
# Old
from insideLLMs.caching_unified import CacheEntryMixin
# New
from insideLLMs.caching import CacheEntryMixin
```

**Step 4: Update all internal docstring references in `insideLLMs/caching.py`**

In the renamed `insideLLMs/caching.py` (formerly `caching_unified.py`), find and replace all occurrences of `insideLLMs.caching_unified` with `insideLLMs.caching` in docstrings. Use:
```bash
rg "insideLLMs\.caching_unified" insideLLMs/caching.py
```
Replace each occurrence. There are ~22 docstring references.

**Step 5: Commit**

```bash
git add -A && git commit -m "refactor: rename caching_unified.py to caching.py"
```

---

### Task 4: Delete `runner.py` shim and its test

**Files:**
- Delete: `insideLLMs/runner.py`
- Delete: `tests/test_runner_shim_coverage.py`

**Step 1: Delete the files**

```bash
rm insideLLMs/runner.py tests/test_runner_shim_coverage.py
```

**Step 2: Commit**

```bash
git add -A && git commit -m "remove: delete deprecated runner.py shim and its test"
```

---

### Task 5: Delete `comparison.py` shim and repoint its test

**Files:**
- Delete: `insideLLMs/comparison.py`
- Modify: `tests/test_comparison.py:5` — update import

**Step 1: Read `tests/test_comparison.py` line 5 to see the exact import**

**Step 2: Update the import**

Change line 5:
```python
# Old
from insideLLMs.comparison import (
# New
from insideLLMs.analysis.comparison import (
```

**Step 3: Delete the shim**

```bash
rm insideLLMs/comparison.py
```

**Step 4: Run the test to verify**

Run: `python -m pytest tests/test_comparison.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add -A && git commit -m "remove: delete deprecated comparison.py shim, repoint test"
```

---

### Task 6: Delete `statistics.py` shim and repoint its 3 consumers

**Files:**
- Delete: `insideLLMs/statistics.py`
- Modify: `tests/test_statistics.py:5` — update import
- Modify: `insideLLMs/cli/commands/report.py:7` — update import
- Modify: `insideLLMs/results.py:1850` — update import

**Step 1: Update `tests/test_statistics.py` line 5**

```python
# Old
from insideLLMs.statistics import (
# New
from insideLLMs.analysis.statistics import (
```

**Step 2: Update `insideLLMs/cli/commands/report.py` line 7**

```python
# Old
from insideLLMs.statistics import generate_summary_report
# New
from insideLLMs.analysis.statistics import generate_summary_report
```

**Step 3: Update `insideLLMs/results.py` line 1850**

```python
# Old
    from insideLLMs.statistics import (
        generate_summary_report,
    )
# New
    from insideLLMs.analysis.statistics import (
        generate_summary_report,
    )
```

**Step 4: Delete the shim**

```bash
rm insideLLMs/statistics.py
```

**Step 5: Run the test to verify**

Run: `python -m pytest tests/test_statistics.py -q`
Expected: PASS

**Step 6: Commit**

```bash
git add -A && git commit -m "remove: delete deprecated statistics.py shim, repoint 3 consumers"
```

---

### Task 7: Delete `trace_config.py` shim and repoint its 4 consumers

**Files:**
- Delete: `insideLLMs/trace_config.py`
- Modify: `insideLLMs/__init__.py:439` — update import
- Modify: `insideLLMs/probes/agent_probe.py:207` — update import
- Modify: `tests/test_trace_config.py:5` — update import
- Modify: `tests/test_agent_probe.py:12` — update import

**Step 1: Update `insideLLMs/__init__.py` line 439**

```python
# Old
from insideLLMs.trace_config import (
# New
from insideLLMs.trace.trace_config import (
```

**Step 2: Update `insideLLMs/probes/agent_probe.py` line 207**

```python
# Old
from insideLLMs.trace_config import (
# New
from insideLLMs.trace.trace_config import (
```

**Step 3: Update `tests/test_trace_config.py` line 5**

```python
# Old
from insideLLMs.trace_config import (
# New
from insideLLMs.trace.trace_config import (
```

**Step 4: Update `tests/test_agent_probe.py` line 12**

```python
# Old
from insideLLMs.trace_config import OnViolationMode, TraceConfig, load_trace_config
# New
from insideLLMs.trace.trace_config import OnViolationMode, TraceConfig, load_trace_config
```

**Step 5: Delete the shim**

```bash
rm insideLLMs/trace_config.py
```

**Step 6: Run the tests to verify**

Run: `python -m pytest tests/test_trace_config.py tests/test_agent_probe.py -q`
Expected: PASS

**Step 7: Commit**

```bash
git add -A && git commit -m "remove: delete deprecated trace_config.py shim, repoint 4 consumers"
```

---

### Task 8: Update `__init__.py` lazy imports and `pyproject.toml`

**Files:**
- Modify: `insideLLMs/__init__.py:642-648` — change lazy import paths from `insideLLMs.cache` to `insideLLMs.caching`
- Modify: `insideLLMs/__init__.py:685-687` — change lazy import paths from `insideLLMs.statistics` to `insideLLMs.analysis.statistics`
- Modify: `pyproject.toml:139-140` — remove per-file-ignores for deleted `cache.py`; keep ignore for `caching.py` (now the renamed canonical module)

**Step 1: Update caching lazy imports in `__init__.py` lines 642-648**

```python
# Old
"InMemoryCache": "insideLLMs.cache",
"DiskCache": "insideLLMs.cache",
"CachedModel": "insideLLMs.cache",
"cached": "insideLLMs.cache",
"BaseCache": "insideLLMs.cache",
"CacheEntry": "insideLLMs.cache",
"CacheStats": "insideLLMs.cache",
# New
"InMemoryCache": "insideLLMs.caching",
"DiskCache": "insideLLMs.caching",
"CachedModel": "insideLLMs.caching",
"cached": "insideLLMs.caching",
"BaseCache": "insideLLMs.caching",
"CacheEntry": "insideLLMs.caching",
"CacheStats": "insideLLMs.caching",
```

**Step 2: Update statistics lazy imports in `__init__.py` lines 685-687**

```python
# Old
"descriptive_statistics": "insideLLMs.statistics",
"compare_experiments": "insideLLMs.statistics",
"confidence_interval": "insideLLMs.statistics",
# New
"descriptive_statistics": "insideLLMs.analysis.statistics",
"compare_experiments": "insideLLMs.analysis.statistics",
"confidence_interval": "insideLLMs.analysis.statistics",
```

**Step 3: Update `pyproject.toml` per-file-ignores**

Remove the line for `cache.py` (deleted). Keep `caching.py` (now the canonical module):

```toml
# Old
"insideLLMs/cache.py" = ["E402"]
"insideLLMs/caching.py" = ["E402"]
# New
"insideLLMs/caching.py" = ["E402"]
```

**Step 4: Run full test suite**

Run: `python -m pytest tests/ -q --ignore=tests/test_nlp_classification_full.py --ignore=tests/test_nlp_keyword_extraction_full.py 2>&1 | tail -5`
Expected: All pass, 0 failures

**Step 5: Run ruff**

Run: `ruff check . && ruff format --check .`
Expected: No violations

**Step 6: Commit**

```bash
git add -A && git commit -m "refactor: update __init__.py lazy imports and pyproject.toml for removed shims"
```

---

### Task 9: Clean up policy engine

**Files:**
- Modify: `insideLLMs/policy/engine.py:13-22` — remove `policy_yaml_path` parameter and update docstring
- Modify: `insideLLMs/policy/rules/__init__.py` — add note about no built-in rules

**Step 1: Update `run_policy()` signature and docstring in `insideLLMs/policy/engine.py`**

```python
# Old (line 13)
def run_policy(run_dir: Path | str, policy_yaml_path: Path | str | None = None) -> dict[str, Any]:
    """Load policy, run checks (attestations exist, signatures verify, Merkle recompute, etc.), return verdict.

    Args:
        run_dir: Path to the run directory.
        policy_yaml_path: Optional path to policy YAML; if present, its digest is included.

    Returns:
        Verdict dict with keys: passed, reasons, checks.
    """

# New (line 13)
def run_policy(run_dir: Path | str) -> dict[str, Any]:
    """Run artifact-completeness checks on a run directory and return a verdict.

    Checks for required artifacts (manifest.json, records.jsonl), core
    attestations (00-07), integrity roots, and SCITT receipts.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Verdict dict with keys: passed (bool), reasons (list[str]), checks (dict[str, bool]).
    """
```

**Step 2: Update `insideLLMs/policy/rules/__init__.py`**

```python
# Old
"""Policy rules (attestation_present, signature_verified, merkle_recompute, etc.)."""

# New
"""Policy rules.

No built-in rules ship with insideLLMs. The policy engine
(``insideLLMs.policy.engine.run_policy``) performs hardcoded
artifact-completeness checks. User-configurable rules may be
added in a future version.
"""
```

**Step 3: Run policy tests to verify**

Run: `python -m pytest tests/test_policy_engine.py -q`
Expected: PASS (no callers pass `policy_yaml_path`)

**Step 4: Commit**

```bash
git add -A && git commit -m "refactor: remove unused policy_yaml_path parameter, document policy rules"
```

---

### Task 10: Version bump to 0.2.0 and changelog

**Files:**
- Modify: `pyproject.toml:3` — version `"0.1.0"` → `"0.2.0"`
- Modify: `CHANGELOG.md` — add `[0.2.0]` section

**Step 1: Update version in `pyproject.toml`**

```toml
# Old
version = "0.1.0"
# New
version = "0.2.0"
```

**Step 2: Add changelog entry in `CHANGELOG.md`**

Insert after the `## [Unreleased]` section (before `## [0.1.0]`):

```markdown
## [0.2.0] - 2026-02-26

### Removed
- Deprecated import shims: `insideLLMs.cache`, `insideLLMs.caching`,
  `insideLLMs.runner`, `insideLLMs.comparison`, `insideLLMs.statistics`,
  `insideLLMs.trace_config`. Import from canonical paths instead
  (`insideLLMs.caching`, `insideLLMs.runtime.runner`,
  `insideLLMs.analysis.comparison`, `insideLLMs.analysis.statistics`,
  `insideLLMs.trace.trace_config`).

### Changed
- Renamed `insideLLMs.caching_unified` to `insideLLMs.caching`.
- `run_policy()` no longer accepts a `policy_yaml_path` parameter (was unused).
```

**Step 3: Run full test suite one final time**

Run: `python -m pytest tests/ -q --ignore=tests/test_nlp_classification_full.py --ignore=tests/test_nlp_keyword_extraction_full.py 2>&1 | tail -5`
Expected: All pass, 0 failures

**Step 4: Run ruff one final time**

Run: `ruff check . && ruff format --check .`
Expected: No violations

**Step 5: Commit**

```bash
git add -A && git commit -m "release: bump version to 0.2.0"
```

---

### Task 11: Update documentation references

**Files:**
- Modify: `QUICK_REFERENCE.md` — replace `insideLLMs.runner` references with `insideLLMs.runtime.runner`
- Modify: `API_REFERENCE.md` — replace `insideLLMs.runner` references with `insideLLMs.runtime.runner`
- Modify: `AGENTS.md` — remove mention of back-compat shim for runner
- Modify: `docs/AUDIT_FINDINGS.md` — update remaining items table to mark items as done

**Step 1: Search and update `QUICK_REFERENCE.md`**

```bash
rg "insideLLMs\.runner" QUICK_REFERENCE.md
```
Replace all `insideLLMs.runner` with `insideLLMs.runtime.runner` (except where text discusses the removal itself).

**Step 2: Search and update `API_REFERENCE.md`**

```bash
rg "insideLLMs\.runner[^_]" API_REFERENCE.md
```
Replace all `insideLLMs.runner` with `insideLLMs.runtime.runner`.

**Step 3: Update `AGENTS.md`**

Remove or update the line mentioning the runner back-compat shim (line ~40).

**Step 4: Update `docs/AUDIT_FINDINGS.md`**

Mark the deprecated module sunset plan, test migration, and caching naming items as completed in the "Remaining Items for Future Work" table. Mark ORAS as verified (no change needed).

**Step 5: Commit**

```bash
git add -A && git commit -m "docs: update references for removed shims and v0.2.0 changes"
```
