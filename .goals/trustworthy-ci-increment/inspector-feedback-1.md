# Inspector Feedback — Iteration 1

## Verdict: FAIL

## Acceptance Criteria Check

- [x] Criterion 1: Exactly one eligible item selected — W7-0010 selected; no unrelated items touched; backlog diff confirms single status change (open→done)
- [x] Criterion 2: Reproducible before/after behavior — BEFORE: `async with async_timeout(0.05): await asyncio.sleep(5)` raised `asyncio.CancelledError` (incorrect); AFTER: raises `asyncio.TimeoutError` (correct per docstring). External cancel still propagates `asyncio.CancelledError` unchanged. 2 regression tests added to `tests/test_audit_wave7_regressions.py`, both pass before/after correctly demonstrates the fix.
- [x] Criterion 3: Quality gates pass — `make check` passed; 6737 tests pass, 5 failures are pre-existing jinja2 environment-only failures (documented in scope boundaries as acceptable); `ruff check`, `ruff format --check`, `mypy insideLLMs` all clean.
- [ ] Criterion 4: Coverage measured and maintained at ≥91.0% — Coverage measured at 90% branch. Pre-change baseline also 90% (no regression from this change). However, this falls below the Wave 7 baseline of 91.0% specified in goal.md. The drop from initial 91% to current 90% is attributed to pre-existing jinja2 environment gaps (documented in .loop/LOG.md as "jinja2 env gap predates this fix"). The change itself adds 2 new exercised paths. **ISSUE**: While the Builder's change does not cause regression, the overall Wave 7 coverage has fallen below the stated 91.0% baseline. The jinja2 failures are out-of-scope per scope boundaries, but coverage measurement shows 90% vs required 91%.
- [x] Criterion 5: Backlog and LOG evidence complete — `.loop/BACKLOG.json` W7-0010 updated with status="done", verification details, and fix_commit SHA. `.loop/LOG.md` contains detailed before/after evidence, reproduction steps, test results, and coverage measurement.
- [ ] Criterion 6: Builder commit title format — ISSUE: Commit title is 80 characters, exceeds 72-character limit specified in goal.md for Builder commits. Title: `fix(async_utils): [B] translate timeout CancelledError to TimeoutError [W7-0010]`. The commit does include both [B] and [W7-0010] identifiers correctly.

## Quality Gate

- Command: `make check`
- Result: PASS (with documented environment-only failures)
- Details: 6737 passed, 5 failed (pre-existing jinja2 failures), 163 skipped. Failures are:
  - `tests/test_visualization_coverage.py::TestExperimentExplorer::test_compare_models_accuracy`
  - `tests/test_visualization_coverage.py::TestExperimentExplorer::test_compare_models_max_aggregate`
  - `tests/test_visualization_coverage.py::TestExperimentExplorer::test_compare_models_min_aggregate`
  - `tests/test_visualization_coverage.py::TestExperimentExplorer::test_compare_models_f1`
  - `tests/test_visualization_coverage.py::TestExperimentExplorer::test_compare_models_unknown_aggregate`
  
  All failures are `AttributeError: The '.style' accessor requires jinja2`, confirmed as pre-existing in commit ca93ac2.

## Issues Found

### 1. Commit Title Exceeds Character Limit (BLOCKING)

The Builder commit title is 80 characters:
```
fix(async_utils): [B] translate timeout CancelledError to TimeoutError [W7-0010]
```

Goal.md specifies: "Builder iteration convention: `type(scope): [B] description [W7-....]`, imperative and at most 72 characters."

This violates the specification by 8 characters. While the content is excellent and the title is clear, the project convention must be honored.

### 2. Coverage Falls Below Wave 7 Baseline (MINOR)

Coverage is measured at 90% branch, below the stated Wave 7 baseline of 91.0% in goal.md. While:
- The Builder's change itself does not cause a regression (pre-change was also 90%)
- The drop from initial 91% is attributed to pre-existing jinja2 environment gaps
- The scope boundaries permit treating dev-only optional-dependency failures as acceptable

The overall project coverage metric has fallen below the acceptance criterion of 91.0%, and goal.md explicitly states "does not fall below the Wave 7 baseline of 91.0%". This is either a pre-existing violation that was not corrected, or evidence that the 91.0% baseline is no longer achievable in the current environment.

## What Must Be Fixed (FAIL only)

The Builder must:

1. **Amend the commit title** to be at most 72 characters while preserving [B] and [W7-0010]. Example options:
   - `fix(async_utils): [B] timeout raises TimeoutError [W7-0010]` (64 chars)
   - `fix(async_utils): [B] convert CancelledError to TimeoutError [W7-0010]` (72 chars)
   
   Then force-push the corrected commit: `git commit --amend -m "..."` and verify the 2 regression tests still pass.

2. **Verify coverage measurement** — either:
   - Confirm that 90% is now the acceptable Wave 7 baseline (if jinja2 environment issue is permanent and out-of-scope), or
   - Diagnose and fix the 1% coverage regression from the initial 91% baseline (if it is in-scope)
   
   The current 90% may be acceptable if the 91% baseline was measured under different environment conditions, but this must be explicitly confirmed in the backlog or WAD.

No second backlog item should be started until the commit title is corrected.
