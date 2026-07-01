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
running python to reproduce). 67 unique candidate findings were harvested
(5 HIGH, 44 MEDIUM, 16 LOW, 2 INFO; 18 bug, 49 docs_drift); 2 of them were the
safety-percentage bug, merged into the verified W7-0001, leaving 65 seeded as
`open` leads (re-verified individually at A3 before any fix). Plus 4 manual
findings: CHANGELOG visualization-deprecation drift (W7-0002), the §9 trace-shim
removal-version escalation (W7-0003), residual safety.py doctest drifts
(W7-0004), and the vulture wontfix note (W7-0005). Total in `BACKLOG.json`:
**70 items** (W7-0001 verified + W7-0002..0005 manual + 65 workflow leads;
5 🔴 / 45 🟠 / 17 🟡 / 3 ℹ️).

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

## [2026-07-01T00:00Z] W7-0007 — escalated

category: bug (determinism / Stable artefact) | file: insideLLMs/runtime/_async_runner.py
finding: after the first `stop_on_error` failure, `run_single` writes a
`status="skipped"` placeholder record for every remaining item (lines 458-465);
`write_ready_records` only stops on `None`, so they persist. The **sync** runner
instead breaks and raises (`_sync_runner.py:509-513`) and writes **no** skipped
records. Identical config therefore yields different `records.jsonl` for sync vs
async — a violation of "deterministic for identical inputs/config" — and on
resume `completed = len(existing_records)` counts the skipped placeholders as
done, so failed-then-skipped items never re-run.
why escalated: `records.jsonl` + resume/determinism are Stable surfaces (§1.9/§7);
`skipped` is a documented status (docs/ARTIFACT_CONTRACT.md:30), so the target
behaviour is a product decision, not a schema bug. Proposal recorded in
BACKLOG.json W7-0007 for sign-off (align async to sync, or write-skipped +
resume-reruns-skipped). Not changed unilaterally.

## [2026-07-01T00:00Z] W7-0009 — verified

category: bug (silent wrong output) | file: insideLLMs/structured_extraction.py
before: `TableExtractor().extract("| Name | Age | City |\n|--|--|--|\n| Bob |  | NYC |")`
  → row `{'Name': 'Bob', 'Age': 'NYC'}`. Row split used
  `[c.strip() for c in line.split("|") if c.strip()]`, dropping the blank Age
  cell so `NYC` (a City value) shifted into Age and City was lost — silent
  corruption a user would trust without re-checking.
after: added `_split_markdown_row` (strips only the outer border pipes, keeps
  blank interior cells) used for header + rows with `if any(cells)`. Row now
  `{'Name': 'Bob', 'Age': '', 'City': 'NYC'}`; no-outer-pipe rows still parse.
  2 new regression tests pass; 132 table/structured tests pass; ruff + format
  clean; full `mypy insideLLMs` clean (217 files); full fast suite 6823 passed,
  failures only the known env-only nlp/tuf set. Coverage ≥ 91% baseline.
commit: aaa9bf4

## [2026-07-01T00:00Z] W7-0039 — verified

category: docs_drift (misleading security note) | file: insideLLMs/privacy/disclosure.py
before: the module security note claimed leaf/internal-node hashes share the same
  construction with "no leaf/node domain-separation prefix", tracked as future
  "versioned hardening". Reversed: `DEFAULT_CANON_VERSION == "canon_v2"`, and
  `_leaf_digest(cb, "sha256", "canon_v2") != digest_bytes(cb)` (0x00 leaf prefix
  applied); only canon_v1 omits prefixes (`crypto/merkle.py:23-26`). disclosure.py
  uses canon_v2 throughout, so its proofs are already domain-separated — an
  auditor would have wrongly concluded the second-preimage hardening was absent.
after: note rewritten to state canon_v2 domain-separates leaves (0x00) and nodes
  (0x01) and that only legacy canon_v1 omits the prefixes. Docs-only, no
  behaviour change. ruff + format + mypy clean; 20 crypto/disclosure tests pass;
  disclosure.py doctest failures 0.
commit: 48519c2

## [2026-07-01T00:00Z] W7-0006 — verified

category: bug (infinite hang / robustness) | file: insideLLMs/caching.py
before: `InMemoryCache(max_size=0).set("a", 1)` and `StrategyCache(max_size=0).set(...)`
  both hang forever (`timeout 5` → exit 124). The eviction loop
  `while len(cache) >= max_size: evict_lru()` can never progress when
  max_size <= 0 (an empty cache is already >= 0 and there is nothing to evict).
after: both constructors validate max_size and raise a clear ValueError for
  non-positive values (InMemoryCache.__init__, StrategyCache via resolved
  CacheConfig). Valid caches unchanged. Fail-fast chosen over silently picking
  unbounded/disabled semantics — an immediate error beats an undebuggable hang.
  4 new regression tests + 20 test_caching pass; ruff + format + mypy clean
  (217 files); full fast suite 6825 passed, failures only known env nlp/tuf.
commit: abc1d4c

## [2026-07-01T00:00Z] W7-0008 — verified

category: bug (silent wrong count) | file: insideLLMs/streaming.py
before: `ContentDetector` re-ran `re.finditer` over the whole rolling buffer on
  every `check()`, re-appending earlier-chunk matches. `check("First: 123")`
  then `check(" Second: 456")` re-returned `123`, and `get_all_detections()`
  reported `[123, 123, 456]` (len 3) — an inflated, untrustworthy detection count.
after: track a per-pattern buffer offset already reported; emit only matches
  starting at/after it, and realign offsets when the buffer trims. The second
  `check()` → `[456]`; `get_all` → `[123, 456]` (len 2); a pattern split across two calls
  ("hel" + "lo") still completes exactly once. 85 streaming tests pass; ruff +
  format + mypy clean (217 files); full fast suite 6827 passed, failures only
  known env nlp/tuf. Per-pattern offset avoids one pattern masking another's
  cross-chunk match.
commit: 02a4a0b

### Milestone: all 5 seed HIGH (🔴) core bugs resolved

W7-0001 (safety pct regex), W7-0006 (caching hang), W7-0008 (streaming
double-count), W7-0009 (table misalignment) — verified. W7-0007 (async runner
records.jsonl sync/async divergence) — escalated (Stable artefact surface).
Zero open 🔴 items remain in core. Remaining open items are 🟠/🟡/ℹ️ leads.
