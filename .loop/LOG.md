# MONSTER_LOOP Wave 7 — Run Log

## [2026-07-20T07:15Z] W7-0001 — verified
category: docs_drift | files: CHANGELOG.md, docs/ARTIFACT_CONTRACT.md
before: `rg 'results\.jsonl' CHANGELOG.md` returned no matches; ARTIFACT_CONTRACT L55 promised "timeline will be announced in changelog" with no corresponding entry
after: CHANGELOG [Unreleased] ### Deprecated documents v0.3.0 removal; ARTIFACT_CONTRACT cross-references announcement; `make check-fast` green (6728 passed, 162 skipped)
commit: 588a7af
notes: Alias still emitted (W7-0002 escalated separately — Stable surface removal needs product sign-off)

## [2026-07-20T07:20Z] W7-0003 — verified
category: docs_drift | files: docs/IMPORT_PATHS.md, insideLLMs/visualization.py, tests/test_audit_wave7_regressions.py
before: IMPORT_PATHS.md L46 `Indefinite support`; visualization.py L173-174 `is not deprecated` / `indefinitely`; CHANGELOG Unreleased L24 `removed in v2.0.0`
after: IMPORT_PATHS and shim docstring state `Deprecated; removal in v2.0.0`; regression test locks consistency; `make check` 6736 passed, coverage 90%
commit: (pending)
notes: Docs-only; CHANGELOG already had the product decision — fixed drift in matrix and module docstring

