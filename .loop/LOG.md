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
commit: 678406b
notes: Docs-only; CHANGELOG already had the product decision — fixed drift in matrix and module docstring

## [2026-07-20T07:25Z] W7-0003 — verified (warning path + cov gate)
category: docs_drift | files: insideLLMs/visualization.py, docs/IMPORT_PATHS.md, tests/test_audit_wave7_regressions.py
before: docs aligned to v2.0.0 in 678406b, but CHANGELOG/MIGRATION said "Deprecation warnings issued" while shim emitted none; `pytest tests/test_audit_wave7_regressions.py --cov=insideLLMs.visualization` → module-not-imported / no data collected (policy test only read file as text)
after: shim issues DeprecationWarning (stacklevel=1) then sys.modules alias; wave7 tests import module; `--cov=insideLLMs.visualization` → 100% (5 stmts, 0 miss); `make check-fast` green (6730 passed, 162 skipped)
commit: 52a5729
notes: CLI/report already use canonical analysis.visualization; Stable surfaces untouched. Full-repo 100% coverage logged as W7-0008.

## [2026-07-20T07:55Z] W7-0008 — in_progress (stricter bar + slice1)
category: structure | files: tests/test_coverage_w7_0008_slice1.py, .loop/BACKLOG.json
bar: True 100% with omit-list shrink toward empty; no new omit/pragma; no vanity deletes
before: measured TOTAL **90%** (19698 stmts, 1558 miss, branch-aware); omit list unchanged (providers/nlp/publish/crypto/tuf/integrations/contrib)
slice1 after (focused): encryption, cosign, slsa_provenance, builders, receipt, shadow → **100%** (412 stmts, 0 miss, 0 partial)
omit: unchanged (only shrink allowed without escalation)
next: caching.py (~263 miss), CLI doctor/diff/init, async_runner; then shrink omit starting with crypto/openrouter/publish

## [2026-07-20T08:10Z] W7-0008 — slice2 (crypto omit shrink + CLI/caching)
category: structure | files: pyproject.toml, insideLLMs/crypto/canonical.py, tests/test_coverage_w7_0008_slice2.py
before: measured TOTAL **90%**; crypto/* omitted
after (focused): `insideLLMs.crypto` + `cli.commands.doctor` → **100%**; caching improved in-slice
omit shrink: removed `insideLLMs/crypto/*` (only shrink; no new omit/pragma)
dead-code: removed unreachable `digest_bytes` raise after SUPPORTED_ALGOS guard (A3-proven dead)
commit: 286cb47
after TOTAL: **93%** (19789 stmts / 1001 miss)

## [2026-07-20T08:35Z] W7-0008 — slice3 (omit shrink + interactive/CLI)
category: structure | files: pyproject.toml, tests/test_coverage_w7_0008_slice3.py, tests/test_langchain_integration_coverage.py
before TOTAL: **93%**; omit still had openrouter/publish/tuf/integrations
after TOTAL: **94%** (20004 stmts / 934 miss)
omit shrink: removed openrouter, publish/*, tuf_client, integrations/* (now measured @100% focused)
focused 100%: diffing_interactive, optimize_prompt, tuf_client, publish, integrations, openrouter
check-fast: green (6762 passed)
commit: 1d41664
