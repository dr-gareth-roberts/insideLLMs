# REVIEW_LOOP

Resumable remediation loop for the verified external review of `insideLLMs`
(see docs/REVIEW_VERIFICATION.md). Address every confirmed criticism — one
verified increment per turn — without eroding the strengths the review
confirmed: deterministic artifacts, zero-dep core, offline golden path,
stable CLI/schema surfaces.

## Trust goal

Same bar as MONSTER_LOOP.md §"Trust goal": every change must make the tool's
outputs more trustworthy without a human re-checking them. Two additions:
(a) fix only *verified* problems — the Corrections Ledger (§5) lists review
claims that are factually wrong; acting on them as written is a defect, not
progress. (b) The review's confirmed strengths are regression surfaces: a
remediation that weakens determinism, offline capability, or stability
guarantees fails this loop even if it "addresses" a criticism.

## 0. Precedence and continuity

- `AGENTS.md` is authoritative for environment and commands. Read it first, every time.
- `MONSTER_LOOP.md` §1 non-negotiables are inherited wholesale. Where this
  file is silent on process, MONSTER_LOOP.md governs.
- `docs/REVIEW_VERIFICATION.md` is the evidence base. Every queue item cites
  it. Evidence goes stale: re-verify before fixing (§3 R3).
- `docs/STABILITY_MATRIX.md`: anything touching a Stable row — CLI command
  names/flags, `records.jsonl`/`manifest.json`/`summary.json`/`diff.json`
  fields, `SchemaRegistry` names/versions, determinism flags — requires the
  deprecation-first path in §6. No silent changes, ever.
- Wave number: run `ls tests/test_audit_wave*.py`, take the next free integer
  `<n>` (currently 8). Item IDs are `R<n>-XX`. Don't renumber old waves.

## 1. Non-negotiables

1. **One verified increment per turn.** Pick one item, fix it, prove it, stop.
2. **Golden-path parity.** `make golden-path` must produce identical artifacts
   before/after every increment, unless the item explicitly declares an
   artifact change — then: schema version bump + migration + docs + changelog.
3. **Determinism by default.** New tolerance/telemetry behaviors are opt-in
   flags; default invocations of `run`/`harness`/`diff` never change silently.
4. **No new required dependencies.** Optional extras only, with escalation (§6).
5. **Reuse existing assets first.** `nlp/similarity.py`, `probes/judge.py`,
   `DiskCache`, `load_hf_dataset`, `signing/cosign.py`, `transparency/`
   already exist. The review's biggest errors were assuming they didn't (§5).
6. **Prove it.** `make check` green per increment; regression tests land in
   `tests/test_audit_wave<n>_regressions.py` or feature test modules.
   Coverage must not drop below the 91.0 baseline in `.loop/BACKLOG.json`.
7. **No docstring padding.** Docs change only when behavior changes or docs are wrong.

## 2. State

.loop/BACKLOG.json   — add §4 items with R<n>- prefix on first run (same item schema as wave 7)
.loop/LOG.md         — append one entry per increment

## 3. The rail (R0–R7)

**R0 — Sync.** Read `AGENTS.md`, this file, `docs/REVIEW_VERIFICATION.md`,
`.loop/BACKLOG.json`, tail of `.loop/LOG.md`.

**R1 — Seed** (first run only). Mirror §4 into `.loop/BACKLOG.json` as pending
items with evidence strings copied verbatim.

**R2 — Select.** Lowest tier, then queue order. Skip items flagged
`escalation` until the escalation is answered in `.loop/LOG.md`.

**R3 — Re-verify.** Reproduce the item's evidence with its listed command. If
it no longer reproduces, mark the item `stale` with proof, log, and move on.

**R4 — Fix.** Smallest correct diff. No unrequested refactors; new findings
become backlog items, not scope creep.

**R5 — Verify.** Item acceptance checks + `make check`; plus
`make golden-path` for anything under `runtime/`, `cli/`, `schemas/`, `crypto/`.

**R6 — Record.** Update backlog, append log, commit
`fix(scope): summary [R<n>-XX]`.

**R7 — Stop.**

## 4. Work queue

### Tier 1 — Quick wins (high leverage, small diffs)

**R-01 Streaming JSONL ingestion** *(review W5, QW1)*
- Evidence: `cli/_record_utils.py:94-108` and `runtime/_artifact_utils.py:299`
  accumulate full `list[dict]`; consumers: `cli/commands/diff.py:92-94`
  (loads both runs), `_async_runner.py:379` (resume), validate path.
  Re-verify: `grep -n "records.append" insideLLMs/cli/_record_utils.py insideLLMs/runtime/_artifact_utils.py`
- Do: add a single shared `iter_jsonl_records(path) -> Iterator[dict]`
  generator in `runtime/_artifact_utils.py` (CLI re-exports it); convert diff
  to a key-indexed streaming comparison and resume/validate to iteration;
  keep the list readers as thin wrappers for compatibility.
- Accept: memory regression test (tracemalloc) shows bounded peak on a
  synthetic 100k-record diff; `diff.json` for golden-path runs byte-identical
  to pre-change; `make check` green.

**R-02 Sync/async `stop_on_error` parity** *(review M3, QW3)* — `escalation: direction`
- Evidence: `_async_runner.py:466-468` writes `status="skipped"` placeholder
  records with `metadata.reason="stop_on_error"`; `_sync_runner.py` writes
  none. Re-verify: `grep -n "skipped" insideLLMs/runtime/_sync_runner.py insideLLMs/runtime/_async_runner.py`
- Do: pick one canonical behavior (default recommendation: sync semantics —
  no placeholders — with an explicit `emit_skipped_placeholders` opt-in) and
  apply to both runners. `records.jsonl` is a Stable artifact: document the
  change, add changelog + schema note.
- Accept: fault-injection test (failing DummyModel) produces identical
  `records.jsonl` from sync and async runners for the same config; both
  placeholder and no-placeholder paths covered.

**R-03 Non-interactive snapshot acceptance** *(review M4 residue)*
- Evidence: `cli/commands/diff.py:111-114` only warns in non-TTY;
  `prompt_accept_snapshot()` (`diff.py:236`) is the sole baseline-update path.
- Do: add `--accept-snapshot` flag (and `INSIDELLMS_ACCEPT_SNAPSHOT=1` env
  var) that runs the same review output then applies
  `copy_candidate_artifacts_to_baseline` without prompting; interactive flow
  unchanged; mutually exclusive with `--format json`.
- Accept: e2e test in a non-TTY subprocess updates the baseline with the
  flag and leaves it untouched without; help text documents both.

**R-04 Opt-in similarity tolerance in the diff gate** *(review W3, QW2)* — `escalation: stable-flag`
- Evidence: zero similarity usage under `runtime/`+`cli/`
  (`grep -rn "similarity\|SequenceMatcher\|levenshtein" insideLLMs/runtime insideLLMs/cli`
  → empty); assets exist in `nlp/similarity.py` (levenshtein_similarity,
  jaccard_similarity — zero-dep).
- Do: add `--similarity-threshold FLOAT` (default `None` = exact matching,
  preserving Stable semantics). When set, text-output changes scoring ≥
  threshold via `nlp.similarity` are classified into a new additive
  `formatting_drift` section instead of `regressions`/`other_changes`;
  additive `diff.json` field → schema minor bump + migration + docs.
- Accept: fixture pair differing only in whitespace/punctuation passes at
  `0.95` while a genuinely wrong answer still fails; default runs
  byte-identical to pre-change; exit-code matrix tested.

### Tier 2 — Structural decomposition (behavior-preserving)

**R-05 Split `contrib/adapters.py` (3,207 ln) into a package** *(review M2/W1, QW4; absorbs M7)*
- Inventory (verified): enums `Provider`/`AdapterStatus`/`ModelCapability`;
  data `AdapterModelInfo`/`AdapterConfig`/`GenerationParams`/`GenerationResult`/
  `HealthCheckResult`; `BaseAdapter`; providers `MockAdapter`/`OpenAIAdapter`/
  `AnthropicAdapter`; `ModelRegistry`/`ProviderDetector`; `AdapterFactory`;
  `FallbackChain`; `AdapterPool`/`ConnectionInfo`/`ConnectionMonitor`; 8
  module-level factory/list functions.
- Do: `contrib/adapters/{__init__,enums,config,results,base,registry,factory,pool,fallback}.py`
  + `providers/{mock,openai,anthropic}.py`; `__init__.py` re-exports the full
  public surface so `from insideLLMs.contrib.adapters import X` is unchanged.
  Add a module docstring note pointing provider glue here vs `models/`
  (addresses duplicated-layer confusion, M7).
- Accept: import-compatibility test enumerating all public names; `mypy`
  clean; no new module > 900 lines; zero behavior diffs (`make check`).

**R-06 Split `caching.py` (3,501 ln — verification addendum)** — same shim
pattern: `caching/{__init__,core,entries,strategies,disk,memoize,warming,dedup,async_adapter}.py`;
`insideLLMs.caching` re-exports; property tests for eviction invariants ride along.

**R-07 Split `schemas/registry.py` (1,870 ln)** — `escalation: stable-surface`
- `SchemaRegistry` names/versions are a Stable row. Split into
  `schemas/registry/{__init__,models_v1_0_0,models_v1_0_1,migration,export}.py`
  with byte-identical public API and JSON Schema output
  (`insidellms schema dump` golden test before/after).

**R-08 CLI consolidation, deprecation-first** *(review M10, QW5)* — `escalation: stable-surface`
- Evidence: 22 command modules; `run`/`quicktest`/`benchmark`/`harness`/
  `compare` overlap; STABILITY_MATRIX:21-22 marks names+flags Stable.
- Do: establish `harness` + `diff` as primary in README/help/docs; convert
  `quicktest`/`benchmark`/`compare` into thin delegating aliases emitting
  `DeprecationWarning` + pointer to the canonical verb; no removals; shared
  argument wiring moves to `_run_common.py`; changelog + migration note.
- Accept: every legacy invocation still works and warns exactly once; CLI
  smoke tests cover alias parity; help output snapshot updated.

### Tier 3 — Evaluation & trust upgrades

**R-09 Structured provider error telemetry** *(review W9)* — `escalation: stable-artifact`
- Evidence: `_sync_runner.py:585,597` flatten to `"error": str(e)`;
  `ProbeExecutionError` (`exceptions.py:956`) already carries
  `original_error`/`sample_index` in-process but artifacts drop it.
- Do: additive optional `error_detail` object on error records —
  `{exception_type, provider_status, retry_after, attempts, request_id}` —
  populated from model/adapter retry paths; schema minor bump + migration;
  both runners emit identically (ties into R-02).
- Accept: fault-injection tests (429-with-retry-after, 503, timeout) show
  populated fields; `insidellms validate` passes old and new records.

**R-10 Schema migration hardening** *(review W10; rides on R-04/R-09 bumps)*
- Do: implement the real forward migrations for the fields added in R-04/R-09;
  property-based round-trip tests (old→new→validate); document migration
  authoring in `schemas/` docs so the extensible framework has ≥3 concrete paths.

**R-11 Judge calibration + gate integration** *(review W8 residue)* — `escalation: stable-flag, network`
- Evidence: `probes/judge.py` has rubric CoT scoring but no position-swap,
  no consensus; diff gate has only deterministic `judge_diff_report`.
- Do: extend `probes/judge.py` with pairwise position-swap consistency checks
  and optional k-judge majority; add opt-in `insidellms diff --judge-model …`
  semantic triage that annotates (never gates by default) the diff report;
  deterministic replay via cached judge transcripts for tests (DummyModel-backed).
- Accept: position-bias unit tests (A/B swap agreement); offline-deterministic
  test path; docs explain judge-vs-deterministic triage.

**R-12 Provider failure-mode fixtures** *(review W4)*
- Do: offline recorded-fixture tests (stdlib `http.server`/socket stubs — no
  new required deps) for 429/503/timeout/truncated-stream against
  `models/openai.py`, `models/anthropic.py`, `retry.py`, and
  `contrib/adapters` providers; wire into `make test-fast` (no keys, no network).
- Accept: each failure mode asserts retry/backoff/error-record behavior;
  suite runs offline in < a few seconds.

**R-13 SCITT receipt verification + RFC 3161 option** *(review W7 residue)* — `escalation: new-optional-dep`
- Evidence: `transparency/scitt_client.py` docstring: "does NOT perform
  cryptographic receipt verification… no COSE countersignature check, no
  Merkle inclusion-proof check".
- Do: implement real receipt verification behind an optional extra (COSE/CBOR
  lib — escalate name/license first); keep `receipt_looks_well_formed` as the
  no-dep fallback; optional RFC 3161 timestamp of `manifest.json` Merkle root;
  update `SECURITY.md` trust-model section to state exactly what is and isn't
  verified (honest-docs requirement).
- Accept: verify-path unit tests with known-good/known-tampered receipts;
  docs corrected.

**R-14 Nondeterministic-provider strategy** *(review W2 residue)* — split doc/feature
- Doc part (no escalation): README + ARCHITECTURE.md state the determinism
  scope precisely (harness artifacts, not model outputs) and recommend
  workflows for hosted APIs (pin seeds where supported, similarity gate R-04,
  judge triage R-11).
- Feature part (`escalation: scope`): opt-in `--samples N` stability mode
  emitting per-item variance metrics into an additive report section.

### Tier 4 — Targeted residues

**R-15 PII masking coverage** *(review M5 residue)* — extend
`safety.mask_pii`/`PIIDetector` patterns (intl phone formats, IBAN, passport,
address heuristics) + adversarial fixtures; document known limitations
honestly instead of overclaiming.

**R-16 Persistent cache wiring** *(review M6 residue)* — expose existing
`DiskCache` through harness config (`cache: {backend: disk, dir: …}`),
default **off**; document determinism interaction (cache hits must not alter
artifacts); integration test across two CLI invocations.

**R-17 Dataset loader extensions + docs** *(review M8 residue)* — optional
Parquet via lazy `pyarrow` extra (escalate dep) or documented CSV/JSONL
conversion recipe; fix the docs gap: `load_hf_dataset` exists and must be
listed in README/API_REFERENCE dataset sections.

**R-18 Heuristic-probe honesty pass** *(review M1 + M9)* — audit probe
docstrings/README probe table so every regex/lexicon probe states its
heuristic nature, false-negative modes, and the `JudgeScoredProbe` upgrade
path; align with the doc-drift bar from MONSTER_LOOP (no padding — only
correction of misleading claims).

## 5. Corrections ledger (verified-false review claims — never "fix" these as stated)

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

## 6. Escalation

Stop and ask (record question + answer in `.loop/LOG.md`) before:
- Any Stable-surface change (STABILITY_MATRIX): R-02, R-04, R-07, R-08, R-09, R-11 — present the deprecation/migration plan first.
- Any new dependency, even optional (R-13, R-17): name, license, size, import-guard plan.
- R-14 feature part (multi-sample mode) — scope approval.
- Distributed execution (review W6) is **explicitly deferred**: out of this
  loop's scope. If demanded, it is a design proposal, not a loop increment.

## 7. Definition of done

- Tier 1 + Tier 2 items landed and verified (or marked `stale` with proof).
- Tier 3 + Tier 4 items landed, or escalated-and-descoped with a logged rationale.
- `make check` green; `make golden-path` deterministic; coverage ≥ 91.0 baseline.
- CHANGELOG + docs updated for every behavior change; zero silent Stable-surface changes.
- Closing `.loop/LOG.md` entry: per-item disposition table (fixed / deferred / stale) mapping back to review criticisms M1-M10, W1-W10.
