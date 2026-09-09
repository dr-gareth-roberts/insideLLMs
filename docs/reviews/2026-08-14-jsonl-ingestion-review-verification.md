# External Review — Verification Record

**Date**: 2026-08-14
**Scope**: Independent re-verification of every claim in the external review of `insideLLMs` (10 strengths, 10 mediocrities, 10 weaknesses), against the working tree.
**Method**: Every verdict below cites `file:line` evidence reproduced with `wc -l`, `grep -n`, and direct reads. Replay commands are included so an agent or human can re-verify without trusting this document.
**Consumers**: `REVIEW_LOOP.md` (repo root) — the remediation loop that executes against these verdicts. Queue item IDs (`R-01`…`R-18`) referenced below are defined there.

---

## Verdict legend

| Symbol | Meaning |
|---|---|
| ✅ | Confirmed as stated — evidence reproduces the claim |
| 🟡 | Partially confirmed — a real gap exists, but the review misstates the mechanism, location, or scope (see Corrections Ledger) |
| ❌ | Refuted — the claim is factually wrong; acting on it as written would be a defect |

---

## Table 1 — Strengths (all confirmed; these are regression surfaces)

Every confirmed strength is a surface that remediation must not erode. A fix that weakens any row below fails the loop even if it "addresses" a criticism.

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| S1 | Deterministic artifacts: SHA-256-derived 32-char run IDs, sorted JSON keys, diffable output | ✅ | `runtime/_determinism.py:201-256` — `_deterministic_run_id_from_config_snapshot` returns `_deterministic_hash(payload, …)[:32]`; docstring: "32-character hexadecimal run ID"; `docs/DETERMINISM.md` |
| S2 | Near-zero-dependency core | ✅ | `pyproject.toml:41-43` — core `dependencies = ["pyyaml>=6.0"]` only; all providers/NLP/serving are optional extras (`pyproject.toml:45+`) |
| S3 | Offline golden path with no keys or network | ✅ | `Makefile:89` `golden-path` target; `ci/harness.yaml`; `DummyModel` at `insideLLMs/models/__init__.py:351` |
| S4 | Explicit stability contract for CLI and artifacts | ✅ | `docs/STABILITY_MATRIX.md:21-24` — Stable rows for CLI command names, flag semantics, canonical artifacts (`records.jsonl`, `manifest.json`, `summary.json`, `diff.json`), `SchemaRegistry` names/versions |
| S5 | Supply-chain stack: signing, attestation, transparency, TUF | ✅ | `signing/cosign.py`; `attestations/dsse.py`; `transparency/scitt_client.py`; `datasets/tuf_client.py`; `crypto/merkle.py:69` `merkle_root_from_items` |
| S6 | Versioned schema registry with a migration framework | ✅ | `schemas/registry.py` (1,870 ln); built-in RunManifest 1.0.0 → 1.0.1 migration path documented at `registry.py:1632` |
| S7 | LLM-as-judge scoring exists (rubric + chain-of-thought) | ✅ | `probes/judge.py:82` `JudgeScorer` ("scores responses against a rubric using chain-of-thought reasoning", `judge.py:5-6`); `judge.py:250` `JudgeScoredProbe` |
| S8 | Zero-dep text-similarity metrics library | ✅ | `nlp/similarity.py` (908 ln): `jaccard_similarity:157`, `levenshtein_distance:326`, `jaro_winkler_similarity:635`, etc.; normalized `levenshtein_similarity` at `analysis/evaluation.py:761` |
| S9 | Structure-aware PII redaction for exports | ✅ | `privacy/redaction.py:10-31` — `redact_pii` recurses dicts (including string keys), lists, tuples; delegates strings to `safety.mask_pii` (`safety.py:2274`) |
| S10 | Resumable runs with a run-directory sentinel | ✅ | `.insidellms_run` sentinel written at `runtime/_artifact_utils.py:229`, checked at `:509-516`; `_read_jsonl_records(..., truncate_incomplete=True)` (`_artifact_utils.py:299`) supports resume after interruption |

Replay: `wc -l insideLLMs/schemas/registry.py insideLLMs/nlp/similarity.py`; `grep -n "insidellms_run" insideLLMs/runtime/_artifact_utils.py`; `grep -n "dependencies" pyproject.toml`.

---

## Table 2 — Mediocrities (M1–M10)

| # | Claim | Verdict | Evidence | Queue item |
|---|---|---|---|---|
| M1 | Several probes are regex/lexicon heuristics presented without stating their heuristic nature | ✅ | Probe docstrings and the README probe table do not consistently state false-negative modes or the `JudgeScoredProbe` upgrade path (`probes/judge.py:250`) | R-18 |
| M2 | `contrib/adapters.py` is a monolith | ✅ | `wc -l` → 3,207 lines; 19 classes + 8 module-level functions in one file (`adapters.py:172-3195`) | R-05 |
| M3 | Sync and async runners diverge on `stop_on_error` | ✅ | `runtime/_async_runner.py:466-468` writes `status="skipped"` placeholders with `metadata.reason="stop_on_error"`; `grep -n "skipped" insideLLMs/runtime/_sync_runner.py` → no matches | R-02 |
| M4 | Snapshot acceptance is unusable in CI (review said non-TTY *aborts*) | 🟡 | `cli/commands/diff.py:111-114` only **warns** in non-TTY; the real gap: `prompt_accept_snapshot()` (`diff.py:236`) is the sole baseline-update path — no acceptance flag exists | R-03 |
| M5 | PII masking pattern coverage is narrow | ✅ | `safety.py:650` `PIIDetector` / `safety.py:2274` `mask_pii` lack intl phone formats, IBAN, passport, address heuristics; mechanism itself is sound (see M4-style correction: gap is coverage, not mechanism) | R-15 |
| M6 | No persistent caching (review said caches don't exist) | 🟡 | `DiskCache` exists at `caching.py:1276` (SQLite-based); `RedisCache` (`semantic_cache.py:975`) and `SemanticCache` (`semantic_cache.py:1433`) exist too. The real gap: none is wired into harness config | R-16 |
| M7 | `models/` vs `contrib/adapters` duplicated provider layers confuse users | ✅ | Both `models/openai.py`/`models/anthropic.py` and `contrib/adapters.py` `OpenAIAdapter:2236`/`AnthropicAdapter:2553` exist with no cross-referencing docstring note | folded into R-05 |
| M8 | No HuggingFace dataset loading (review claim) | 🟡 | ❌ as stated: `dataset_utils.load_hf_dataset` exists (`dataset_utils.py:229`). Real gaps: it is missing from README/API_REFERENCE dataset sections, and there is no Parquet loader | R-17 |
| M9 | Probe result claims overstate rigor (heuristics scored as ground truth) | ✅ | Same evidence base as M1; docstring/README honesty pass required, not new scoring code | R-18 |
| M10 | CLI has sprawled: overlapping commands | ✅ | 22 command modules in `cli/commands/` (25 entries minus `__init__.py`, `__pycache__`, `_run_common.py`); `run`/`quicktest`/`benchmark`/`harness`/`compare` overlap; names+flags are Stable rows (`docs/STABILITY_MATRIX.md:21-22`) | R-08 |

Replay: `ls insideLLMs/cli/commands/`; `grep -n "skipped" insideLLMs/runtime/_sync_runner.py insideLLMs/runtime/_async_runner.py`; `grep -n "def load_hf_dataset" insideLLMs/dataset_utils.py`.

---

## Table 3 — Weaknesses (W1–W10)

| # | Claim | Verdict | Evidence | Queue item |
|---|---|---|---|---|
| W1 | Oversized modules impede maintenance (review named `runtime/runner.py` a 1,600-line monolith) | 🟡 | ❌ for `runner.py`: it is a **227-line facade** (`wc -l`). ✅ for the pattern elsewhere: `contrib/adapters.py` 3,207 ln, `caching.py` 3,501 ln, `schemas/registry.py` 1,870 ln | R-05/R-06/R-07 |
| W2 | Determinism guarantees are silent about nondeterministic hosted providers | ✅ | Determinism scope (harness artifacts, not model outputs) is not stated in README/ARCHITECTURE.md; no multi-sample variance mode exists | R-14 |
| W3 | Exact-match diffing is brittle to benign formatting drift | 🟡 | Gap confirmed: `grep -rn "similarity\|SequenceMatcher\|levenshtein" insideLLMs/runtime insideLLMs/cli` → empty. But the metrics already exist (S8) — the fix is wiring, not new metric code | R-04 |
| W4 | Provider failure modes (429/503/timeout/truncated stream) are untested | ✅ | No offline recorded-fixture tests against `models/openai.py`, `models/anthropic.py`, `retry.py`, or `contrib/adapters` providers | R-12 |
| W5 | JSONL ingestion materializes entire files in memory | ✅ | `cli/_record_utils.py:94-108` (`_read_jsonl_records` → `records.append` at `:107`) and `runtime/_artifact_utils.py:299` (list accumulation at `:375`) return full `list[dict]`; consumers: `cli/commands/diff.py:92-94` loads both runs; `_async_runner.py:379` resume path | R-01 |
| W6 | No distributed execution | ✅ | Confirmed absent — **explicitly deferred**: out of loop scope; a design proposal, not an increment | deferred |
| W7 | Transparency-log receipts are not verified (review said signing doesn't exist) | 🟡 | ❌ as stated: cosign/SCITT/DSSE/TUF all exist (S5). ✅ for the residue: `transparency/scitt_client.py:1-9` docstring — "does NOT perform cryptographic receipt verification: … no COSE countersignature check, no Merkle inclusion-proof check" | R-13 |
| W8 | No LLM-judge evaluation (review claim) | 🟡 | ❌ as stated: `probes/judge.py` exists (S7). ✅ for the residue: no position-swap consistency checks, no k-judge consensus, and the diff gate uses only the deterministic `judge_diff_report` (`runtime/diffing.py:900`) | R-11 |
| W9 | Error records flatten all provider failure detail to a string | ✅ | `runtime/_sync_runner.py:585,597` — `"error": str(e)`; `ProbeExecutionError` (`exceptions.py:956`) already carries `original_error`/`sample_index` in-process, but artifacts drop it | R-09 |
| W10 | Schema migration framework is thin (extensible in name only) | ✅ | `schemas/registry.py:1632` — one concrete cross-version path (RunManifest 1.0.0 → 1.0.1) plus legacy/identity normalizations | R-10 |

Replay: `wc -l insideLLMs/runtime/runner.py insideLLMs/caching.py`; `sed -n '580,600p' insideLLMs/runtime/_sync_runner.py`; `head -9 insideLLMs/transparency/scitt_client.py`.

---

## Quick-win spot-checks (QW1–QW5)

| QW | Proposal | Spot-check result |
|---|---|---|
| QW1 | Streaming JSONL generator | Viable: both readers are self-contained (`_record_utils.py:94`, `_artifact_utils.py:299`); a shared `iter_jsonl_records` with thin list wrappers preserves compatibility → R-01 |
| QW2 | Similarity tolerance in diff gate | Viable but **must reuse** existing metrics (S8); zero usage today under `runtime/`+`cli/`; Stable-flag escalation required → R-04 |
| QW3 | `stop_on_error` parity | Confirmed one-sided: placeholders exist only in `_async_runner.py:466-468`; `records.jsonl` is Stable → direction escalation required → R-02 |
| QW4 | Split `contrib/adapters.py` | Inventory verified (19 classes, 8 functions, `adapters.py:172-3195`); re-export shim keeps `from insideLLMs.contrib.adapters import X` unchanged → R-05 |
| QW5 | CLI consolidation | 22 modules confirmed; `_run_common.py` already exists as the shared-wiring landing spot; deprecation-first path mandatory (Stable rows) → R-08 |

## Tribunal spot-checks

Deep replays performed where the review's credibility was weakest:

1. **`runner.py` "monolith"** — `wc -l insideLLMs/runtime/runner.py` → 227. The review inflated this ~7×. Refuted; recorded as Corrections Ledger #1.
2. **"No caching layer"** — `grep -n "class DiskCache" insideLLMs/caching.py` → `:1276`, an SQLite-backed persistent cache with eviction, TTL, export/import. Refuted as stated; the wiring gap stands (M6).
3. **"No judge"** — `probes/judge.py` read end-to-end (395 ln): rubric CoT scorer + `JudgeScoredProbe`. Refuted as stated; calibration residue stands (W8).
4. **Non-TTY diff behavior** — `cli/commands/diff.py:111-114` issues `print_warning(...)` and proceeds; it does not abort. The review's "aborts" claim refuted; acceptance-flag gap stands (M4).
5. **SCITT honesty** — the module docstring itself (`scitt_client.py:1-9`) already discloses the verification gap; the code is honest, the review merely rediscovered the disclosure. Residue (real verification) stands (W7).

---

## Corrections Ledger (verified-false review claims — never "fix" these as stated)

1. `runtime/runner.py` is a 227-line facade, not a 1,600-line monolith — do not split it.
2. `DiskCache`/`RedisCache`/`SemanticCache` exist — wire, don't rewrite.
3. `probes/judge.py` exists (rubric CoT judge) — extend, never duplicate.
4. `dataset_utils.load_hf_dataset` exists — the HF gap is documentation.
5. `insidellms diff --interactive` warns (not aborts) without a TTY — the gap is an acceptance flag.
6. `privacy/redaction.py` is structure-aware recursion — the gap is pattern coverage, not mechanism.
7. Sigstore/cosign signing, SCITT submission, DSSE, TUF all exist — the gap is receipt *verification* + TSA.
8. Run IDs are 32-char; the sentinel is `.insidellms_run`.
9. Diff gating has no `strict|lenient|judge` modes — it's `fail_on_*` flags + separate deterministic judge triage.
10. Similarity metrics exist in `nlp/similarity.py` — R-04 wires them; it must not add new metric code or `difflib` dependencies.
11. Normalized `levenshtein_similarity` lives in `analysis/evaluation.py:761`, not `nlp/similarity.py` (which provides `levenshtein_distance:326` and `jaccard_similarity:157`) — R-04 must import from the modules that actually define them.
12. `RedisCache` (`semantic_cache.py:975`) and `SemanticCache` (`semantic_cache.py:1433`) live in `semantic_cache.py`, not `caching.py`; `caching.py` hosts `DiskCache` (`caching.py:1276`) — cache-wiring work (R-16) must cite the correct modules.

---

## Verification Addendum

- **`caching.py` is itself a 3,501-line monolith** (`wc -l insideLLMs/caching.py` → 3501). The review missed this while inventing a `runner.py` monolith; it enters the queue as R-06 with the same shim-split pattern as R-05.
- **Wave numbering**: `ls tests/test_audit_wave*.py` → waves 2, 3, 6, 7 exist. The next free integer is **8**; REVIEW_LOOP items are `R8-XX` in `.loop/BACKLOG.json`, and the regression-test module is `tests/test_audit_wave8_regressions.py`. Do not renumber old waves.
- **Backlog compatibility**: `.loop/BACKLOG.json` (wave 7) carries `baseline_coverage_pct: 91.0` and the item schema (`id`, `category`, `severity`, `location`, `found_by`, `description`, `evidence`, `status`, `escalation_reason`, `fix_commit`, `verification`, `notes`) that REVIEW_LOOP seeding must reuse.
- **Commit format**: wave-7 entries record fixes as `fix(scope): summary [W7-XXXX]`; REVIEW_LOOP uses `fix(scope): summary [R8-XX]`.

---

## Criticism → queue mapping (totality check)

| Criticism | Disposition |
|---|---|
| M1, M9 | R-18 (heuristic-probe honesty pass) |
| M2 | R-05 (adapters split; absorbs M7) |
| M3 | R-02 (`stop_on_error` parity) |
| M4 | R-03 (non-interactive snapshot acceptance) |
| M5 | R-15 (PII masking coverage) |
| M6 | R-16 (persistent cache wiring) |
| M7 | fold-in note on R-05 (module docstring: provider glue here vs `models/`) |
| M8 | R-17 (dataset loader extensions + docs) |
| M10 | R-08 (CLI consolidation, deprecation-first) |
| W1 | R-05 / R-06 / R-07 (structural decomposition; `runner.py` part refuted) |
| W2 | R-14 (nondeterministic-provider strategy, doc + escalated feature) |
| W3 | R-04 (opt-in similarity tolerance) |
| W4 | R-12 (provider failure-mode fixtures) |
| W5 | R-01 (streaming JSONL ingestion) |
| W6 | **Deferred** — distributed execution is out of loop scope (design proposal only) |
| W7 | R-13 (SCITT receipt verification + RFC 3161 option) |
| W8 | R-11 (judge calibration + gate integration) |
| W9 | R-09 (structured provider error telemetry) |
| W10 | R-10 (schema migration hardening) |

All 20 criticisms map to a queue item, a fold-in note, or an explicit deferral. No orphans.
