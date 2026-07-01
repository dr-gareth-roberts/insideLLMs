# MONSTER_LOOP run log — insideLLMs

Append-only. Never edit history.

## [2026-07-01T00:00Z] Wave 7 — A0 seed

Environment: `pip install -e ".[all]"` succeeded (tools newer than CI pins:
ruff 0.15.20, mypy 2.1.0, pytest 9.1.1; installed `vulture` 2.16 separately).

A1 baseline scan (core `insideLLMs/**`):
- `ruff check .` — clean.
- `ruff format --check .` — clean (473 files).
- `mypy insideLLMs` — clean (0 errors, 217 source files) even under mypy 2.1.
- `vulture insideLLMs --min-confidence 80` — 17 hits, all false-positive or
  documented-intentional (see W7-0005, `wontfix`).
- `pytest --collect-only` — 6947 tests, no collection errors.
- Coverage baseline: **91%** branch (`pytest -m "not slow and not integration"
  --cov`). The 22 failures + 12 errors are env-only (nlp nltk-corpora + tuf
  pyo3), non-regressions per `AGENTS.md`.

Static tooling is clean, so the seed came from a fan-out Workflow (14
subsystem readers hunting concrete bugs / docs-drift / inefficiency, each
running python to reproduce). 67 unique candidate findings harvested
(5 HIGH, 44 MEDIUM, 16 LOW, 2 INFO; 18 bug, 49 docs_drift) → seeded as `open`
leads in `BACKLOG.json` (re-verified individually at A3 before any fix). Plus
manual findings: CHANGELOG visualization-deprecation drift (W7-0002), the §9
trace-shim removal-version escalation (W7-0003), residual safety.py doctest
drifts (W7-0004), and the vulture wontfix note (W7-0005).

Prior "done" claims re-verified against code: the v0.2.0 shim removals
(`cache`/`caching_unified`/`runner`/`comparison`/`statistics`/`trace_config`)
are genuinely gone — that claim holds. The §9 trace shims are not — logged as
W7-0003 (escalated).

## [2026-07-01T00:00Z] W7-0001 — verified

category: bug (security detector false-negative) | file: insideLLMs/safety.py
before: `re.compile(r"\b\d+(?:\.\d+)?%\b").findall("95% of users")` → `[]`;
  `SafetyHallucinationIndicatorDetector.analyze("Studies show 75% ...")
  ["indicators"]["has_specific_claims"]` → `False`, while the module docstring
  (lines 1306, 1436) asserts `True`. Trailing `\b` after non-word `%` can never
  be satisfied by a normal percentage, so percentages never counted toward the
  hallucination `risk_score`.
after: removed trailing `\b` → `r"\b\d+(?:\.\d+)?%"`. `findall("95% of users")`
  → `["95%"]`; `has_specific_claims` → `True`. Added
  `tests/test_audit_wave7_regressions.py` (3 tests, pass). `tests/test_safety.py`
  47 pass. safety.py doctest failures 7→5 (fixed lines 1306 & 1436, none added).
  `ruff check`/`ruff format --check` clean; `make typecheck-strict` clean; full
  `mypy insideLLMs` clean (217 files); full fast suite 6834 passed, all failures
  confined to the known env-only nlp/tuf set. Coverage ≥ 91% baseline (change
  only adds exercised paths).
commit: dd0235f
