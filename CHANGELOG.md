# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `diff --html PATH` writes a deterministic, self-contained HTML diff report
  (summary counts, per-section tables, inline before/after) from the current
  DiffReport payload; it works with either `--format` and leaves the exit-code
  policy unchanged.
- Output schema 1.0.2 carries per-item scores and primary metrics. Historical
  1.0.0/1.0.1 contracts remain readable; migration does not invent missing scores.
- `diff --fail-on-any-difference` gates improvements and trace/trajectory changes
  as well as regressions. Malformed, missing, or incomparable primary scores
  cannot pass a regression gate.
- Clean wheel and source-distribution release smoke checks exercise all starter
  templates and a real correct-to-incorrect scoring regression.
- Public-API parity gate: `scripts/audit_docs.py` now fails `make docs-audit` when
  the exact Public API Index rows differ from the package's eager `__all__` plus
  PEP 562 lazy-import map; prose mentions no longer satisfy the gate.
- Declared `jinja2>=3.1.5` in the `visualization` extra (required by the pandas Styler;
  previously an undeclared transitive dependency)
- Full CI pipeline: lint, typecheck, test (Python 3.10/3.11/3.12 matrix), determinism, and contract jobs
- Optional dependency groups: `huggingface`, `signing`, `crypto`, `providers`
- International PII detection support (EU, UK regions)
- Rate limiting integration via `RunConfig`
- Schema version compatibility checking in diff command


### Fixed

- Async fail-fast resume now retries only scheduler-proven undispatched suffix
  items, rejects ambiguous legacy skipped histories before mutation, and
  atomically preserves the exact attempted record prefix.
- Attack, Bias and Agent aggregate scoring now validates both live typed outputs
  and persisted mapping outputs during resume without changing serialization.
- LaTeX export now escapes literal headers and cells after truncation while
  preserving Unicode and the existing row cap.
- Satellite analysis endpoints now offload synchronous work behind a shared
  process-local four-call cap, return HTTP 503 on overflow, and retain admission
  across request cancellation until the underlying work completes.
- Welch and paired t-tests now use two-sided Student-t probabilities with the
  appropriate degrees of freedom instead of a normal approximation. Historical
  p-values and significance decisions need recomputation; stored evidence is not
  rewritten. Fewer than two observations return insufficient data; invalid alpha,
  nonfinite samples and unequal pair lengths raise `ValueError`. Valid zero-variance
  samples use a documented degenerate convention. Confidence-interval approximations
  are unchanged.
- Scored probes now evaluate held-out `reference_answer`/`reference` values in
  sync, async, batch, and harness runs, persist scores, and preserve them on resume.
  Unlabelled successful executions no longer imply measured accuracy.
- Empty, incomplete, or failed runs return failure from `run`/`harness`; CI checks
  both persisted runs before diffing. Duplicate identities and invalid numeric
  scores fail comparison instead of producing misleading passes.
- Configuration credentials are scrubbed before snapshots, metadata, attestations,
  tracking parameters, and run-identity hashing. Live credentials remain available
  to providers. Existing saved artefacts are not rewritten.
- PR evaluation uses read-only permissions and no persisted checkout credentials.
  Sticky comments use a separate trusted workflow; inline action commenting is
  deprecated and its token input is removed.
- `init`, `validate`, and runtime loading share configuration schema version 1,
  relative dataset path handling, and required Pydantic validation. Unsupported
  settings are rejected. Old typed builders use explicit compatibility conversion.
- Single-run `max_examples`, dataset info statistics, and the Python harness
  workflow's report arguments now work through their real execution paths.
- Encrypted JSONL exports now stage plaintext in a private sibling file and
  publish only after Fernet encryption succeeds; failures preserve an existing
  destination or leave an absent destination absent.
- JSONL encryption and decryption preserve source POSIX permissions instead of
  replacing mode-0600 files with mode-0644 files.
- Distributed checkpoint saves serialize first and atomically replace the old
  checkpoint, so invalid payloads and write failures cannot destroy resumable state.
- Resume recovery preserves a syntactically valid final JSONL record that lacks
  a trailing newline and normalizes it before appending new records.
- Defensive delimiter and input-marking strategies escape exact boundary strings
  supplied by user input, preventing forged structural closing markers.
- Response profiling preserves explicit zero token counts and estimates only
  when token counts are omitted.
- **Core-only `[dev]` installs now pass the full test suite.**
  `test_config_loader_model_probe_paths` used `patch.object(..., create=True)` on
  `insideLLMs.models`; mock's original-attribute lookup falls back to `getattr()`,
  which fired the package's PEP 562 lazy `__getattr__` and imported the real
  `openai` SDK (`ModuleNotFoundError` without the extra). Replaced with plain
  `setattr`/`try-finally` so the config-loader fallback branch is covered on
  every install.
- **Policy verdicts no longer pass when a SCITT receipt is missing.**
  `run_policy` handled receipt-with-attestation and receipt-without-attestation
  but left attestation-without-receipt unhandled: no `scitt_*` check was
  recorded and the verdict stayed `passed=true`, so a run with an incomplete
  transparency record read as compliant. Every asymmetry now fails closed.
- **Chat dispatch honours raising stubs, on both the sync and async paths.**
  `Model.chat` and `AsyncModel.achat` are concrete stubs that raise
  `NotImplementedError`, so `hasattr()` reported a capability the model lacked.
  On the async side this made the documented run-in-executor fallback
  unreachable in `ModelPipeline.achat`, `Middleware.aprocess_chat` and
  `TraceMiddleware.aprocess_chat`. On the sync side it made the
  `ModelError("No chat implementation available")` guard in
  `Middleware.process_chat`, `TraceMiddleware.process_chat` and
  `ModelPipeline.chat` dead code, so callers wrapping the pipeline in
  `except ModelError` received an uncaught `NotImplementedError`. Adds
  `can_chat`, `can_chat_async` and `can_generate_async` to complete the
  `can_stream`/`can_stream_async` family.
- **Generate-only models work again in structured output, RAG and receipts.**
  The same presence check disabled documented fallbacks in three more callers:
  `StructuredOutputGenerator.generate` (which documents *"chat() or generate()"*
  support and repeated the raise once per retry attempt), `RAGChain.query_with_chat`
  (which documents accepting a model implementing only `generate`), and
  `ReceiptMiddleware.aprocess_chat` (whose run-in-executor fallback was
  unreachable). In each case the dead branch was a working alternative path, not
  merely a different error.
- **Escalation time budgets now bound the `confidence` callback.** Scoring ran
  outside the deadline, so a model-backed confidence callback could overrun
  `max_seconds` without limit (measured: 0.40s against a 0.05s budget).
  Synchronous confidence callbacks remain supported; a scorer raising its own
  `TimeoutError` still propagates instead of being relabelled as exhaustion; and
  a candidate the budget left unscored is dropped rather than admitted with a
  fabricated confidence.
- **Tool timeout provenance is recorded rather than inferred.** `execute_tool`
  told its own per-attempt deadline apart from a tool's transport timeout by
  measuring elapsed time in the handler. A tool that stalls the event loop
  prevents the deadline callback from firing while pushing that measurement past
  the limit, so a genuine retryable transport fault was denied its retry. Our
  deadline is still never retried, so a declared timeout is never multiplied by
  `max_transport_attempts`.
- **`best_of_n` judge calls appear in the trace as well as in `Spend`.**
  `judge_model_calls` was added to `Spend.calls` with no corresponding trace
  event, so a consumer summing `TraceEvent.calls` under-counted model calls by
  exactly that amount whenever a model-backed judge ran.
- **Provider timeouts are no longer relabelled as budget exhaustion.** The
  earlier fix only distinguished "no budget armed"; with a budget armed, a
  callback raising its own `TimeoutError` was still reported as exhaustion even
  when the whole budget remained. `escalation` additionally *swallowed* the
  error, returning a cheaper earlier answer with `stop_reason=BUDGET`. All four
  strategy modules now test whether the deadline genuinely elapsed.
- **Generation dispatch preserves accounting and resolves awaitables.** A model
  whose `generate()` is `async def` had the coroutine object returned as its
  answer text; and `agenerate` was preferred over `generate_with_metadata`,
  zeroing token/latency `Spend` for models offering both (notably
  `InferenceClient.from_model_config` pipelines), which made matched-compute
  reports compare zeros.

### Changed (breaking)
- **`compose_cached_prompt(model_id=...)` is now required** (was optional,
  defaulting to `""`). Different models sharing a stable prefix and tenant
  produced byte-identical `cache_key`s, so a shared KV/response cache could
  serve one model's completion for another model's request. Pass the model name
  plus any generation parameters that change outputs, e.g.
  `"gpt-4o-mini:temp=0"`. Callers omitting it now raise `TypeError`.

### Deprecated
- **`results.jsonl` legacy alias** (Stable artifact surface)
  - Canonical file is `records.jsonl`; harness runs still emit `results.jsonl` as a symlink or copy for backward compatibility.
  - **Removal planned in v0.3.0.** After removal, use `records.jsonl` only; update diff snapshot and CI scripts that reference the alias.
  - See `docs/ARTIFACT_CONTRACT.md` (Legacy Artifact Aliases).

### Changed
- **Inference-harness review fixes**: beam search terminates on cyclic state
  graphs; callback timeouts raised with no time budget armed are no longer
  mislabelled as budget exhaustion in search/evolution/DAG/escalation (the
  budget-armed case was only fixed later — see Fixed above); caller metadata can no longer forge spend
  accounting; self-consistency returns a real sample output (the normalization
  key moves to `provenance["winning_key"]`); NaN verifier scores rank last;
  matched-compute treats declared calls as a per-example ceiling, compares
  executors by underlying model identity, and guards zero-wall-time
  throughput; best-of-n verifies candidates and judge orders concurrently;
  unused `inference.protocols` module and `Decision` schema removed
- **Trust-surface honesty (P0)**: `datasets.tuf_client.fetch_dataset` now always
  refuses without `allow_mock=True` and labels its proof `status="mock"` /
  `verified=False` (it previously reported mock data as `status="verified"` when
  the `tuf` package was merely importable); `transparency.scitt_client.verify_receipt`
  is deprecated in favour of `receipt_looks_well_formed` (structural check only —
  no cryptographic receipt verification); `insidellms verify-signatures` now fails
  when a run directory contains no attestations instead of reporting success
- `HuggingFaceModel` now reports `supports_streaming=False`: its `stream()` is
  simulated (yields the full response as one chunk), so the capability flag no
  longer overclaims; `info().extra` gains `"streaming": "simulated"`
- Builtin benchmark datasets (13 helpers, 87 handwritten examples total) are now
  labelled smoke-test fixtures in descriptions, docstrings, and CLI output —
  results on them are not benchmark evidence
- Core dependencies slimmed: `openai`, `anthropic`, `transformers`, `huggingface-hub`, `tuf`, `oras`, `cryptography` moved to optional extras
- Disconnected modules (~85k LOC) moved to `insideLLMs/contrib/` for clearer project scope
- Mypy config tightened: re-enabled `name-defined`, `syntax`, `return-value` error codes
- **DEPRECATION**: `insideLLMs.visualization` module is deprecated
  - Use `insideLLMs.analysis.visualization` instead
  - Compatibility layer will be removed in v2.0.0
  - See migration guide below

### Fixed
- Removed duplicate "Verifiable Evaluation Commands" and "Schema Validation Commands" sections from README
- Delimiter escape vulnerability in prompt injection defense
- Async determinism with concurrent execution
- Config loading error messages now show line numbers and context
- Production-quality audit (waves 1–4): resolved ~50 verified bugs, silent
  failures, determinism violations, robustness issues, and docs inaccuracies.
  Highlights:
  - **Determinism**: the sync batch path no longer writes wall-clock `latency_ms`
    into `records.jsonl`; cache keys, set serialization, and trace fingerprints
    are now order-stable; `ModelComparator.generate_report` no longer embeds
    `datetime.now()` by default.
  - **Bugs**: `LogicProbe` no longer scores negated answers as correct; the
    instruction probes (`InstructionFollowingProbe`, `MultiStepTaskProbe`,
    `ConstraintComplianceProbe`) now report meaningful accuracy; `CachedModel`
    no longer crashes when wrapping `StrategyCache`/`PromptCache`;
    `TokenEstimator.split_to_chunks` no longer infinite-loops on large overlaps;
    `RetryHandler` reports accurate attempt counts.
  - **Robustness**: `ModelWrapper` now transparently delegates `chat`/`stream`/
    `batch_generate`; `redact_pii` scrubs string dict keys; `ensure_nltk` accepts
    any iterable.

### Changed (behavior — review before upgrading)
> Several audit fixes alter runtime/output behavior. Most affect only previously
> incorrect cases, but the following are behavior changes to be aware of:
- **BREAKING** `TokenBucketRateLimiter.acquire`/`acquire_async` now raise
  `ValueError` when `tokens > capacity` (previously slept then returned `False`).
- **BREAKING (output)** The instruction probes now emit `status="success"` for
  evaluated-but-non-compliant results (with compliance in `metadata["is_correct"]`)
  instead of `status="error"`. This changes `records.jsonl` `status` values and
  lowers `error_rate` for those probes. Records remain schema-valid and runs
  remain deterministic (verified).
- **BREAKING (output)** `AnthropicModel.chat` now routes `system` messages to the
  Anthropic `system=` parameter and forwards previously-dropped kwargs
  (`top_p`, `top_k`, `stop_sequences`, `system`); generated outputs may differ.
- `insidellms init` now refuses to overwrite an existing config unless
  `--overwrite` is passed.
- Runtime validation helpers now reject unsupported validation modes instead of
  silently treating them as strict validation. Supported values are `strict`,
  `lenient`, and `warn`.

### Removed
- `sentry-sdk` mandatory dependency: never imported anywhere in the codebase, it forced a
  telemetry SDK into every install. Core dependencies are now `pyyaml` only.
- `tuf` from the `signing` extra: the dataset TUF client is deliberately unimplemented
  and fail-closed (`fetch_dataset` refuses without `allow_mock=True`); the package was
  never imported. `insidellms doctor` still probes for it via `find_spec` with its
  standalone install hint.
- Phantom sphinx section from `requirements-dev.txt` (referenced a nonexistent
  `[docs]` extra; nothing in the repo builds with sphinx) and added the missing
  `pydantic` mirror entry.
- Tracked repo-root residue: `FIXES_APPLIED.md`, `experiment.yaml`, `test_config.yaml`,
  `log.txt`, `trace.json`, `trace_export.json`, `fingerprint.json` (CLI/demo output
  committed by accident; regeneration paths are now root-scoped in `.gitignore`).

### Security
- Added delimiter escape sanitization to prevent injection bypass
- Enhanced API key exposure prevention in documentation
- Added pre-commit hook for secret detection
- `insidellms export --encrypt` now validates the encryption key/format **before**
  writing any output, so a failed precondition no longer leaves plaintext on disk.
- The calculator agent tool's arithmetic evaluator now bounds exponentiation,
  preventing a CPU/memory exhaustion DoS (e.g. `10**10**10`).
- The PyPI release workflow now runs the full quality gate (lint, typecheck,
  tests, determinism) before building/publishing.
- Selective-disclosure Merkle inclusion proofs (`insideLLMs.privacy.disclosure`)
  are now **direction-aware and self-verifiable** via a new
  `verify_inclusion_proof()`.
- **BREAKING (artifact format)** Merkle trees now **domain-separate leaf vs.
  internal-node hashes** (second-preimage hardening) under a new default
  canonicalization version **`canon_v2`**. `canon_v1` remains supported for
  reading artifacts produced before this change. Because this changes every
  Merkle root (including the artifact-spine `records_merkle_root`), runs produced
  with this version are not byte-comparable to pre-`canon_v2` artifacts; the
  `canon_version` field on each root manifest records which scheme was used.
  Same-version runs remain byte-for-byte deterministic.

## Migration Guide

### Visualization Module (v0.2.0 → v2.0.0)

**Old code (deprecated):**
```python
from insideLLMs.visualization import text_bar_chart
```

**New code:**
```python
from insideLLMs.analysis.visualization import text_bar_chart
```

**Timeline:**
- v0.2.0 (current): Deprecation warnings issued, old imports still work
- Until v2.0.0: Deprecation warnings continue, both import paths keep working
- v2.0.0: Old import path removed, must use new path

**Automated migration:**
```bash
# Find all deprecated imports
grep -r "from insideLLMs.visualization import" .

# Replace with new import (GNU sed)
find . -name "*.py" -exec sed -i 's/from insideLLMs\.visualization import/from insideLLMs.analysis.visualization import/g' {} +

# Replace with new import (macOS sed)
find . -name "*.py" -exec sed -i '' 's/from insideLLMs\.visualization import/from insideLLMs.analysis.visualization import/g' {} +
```

## [0.2.0] - 2026-02-26

### Added
- Deterministic artifact pipeline with SHA-256 run IDs
- Schema versioning (v1.0.0, v1.0.1) with Pydantic-based validation
- `insidellms diff` for cross-run comparison with CI diff-gating
- `insidellms doctor` readiness checker
- `insidellms schema validate` for artifact payload validation
- DSSE attestation and cosign signing workflows
- Golden-path determinism verification (`make golden-path`)
- Provider adapters: OpenAI, Anthropic, HuggingFace, Gemini, Cohere, Ollama, llama.cpp, vLLM
- Probe categories: logic, factuality, bias, attack, agent, code, instruction
- `AsyncProbeRunner` for concurrent execution
- Registry/plugin system with lazy loading
- Trace configuration for deterministic CI enforcement

### Changed
- Consolidated caching under `insideLLMs.caching`; replace imports from the removed
  `insideLLMs.cache` and `insideLLMs.caching_unified` modules.

## [0.1.0] - 2024-11-10

### Added
- Initial release
- Core probe execution pipeline
- `ProbeRunner` with config-driven execution
- Basic model adapters (OpenAI, DummyModel)
- JSONL record output format

[Unreleased]: https://github.com/dr-gareth-roberts/insideLLMs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dr-gareth-roberts/insideLLMs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dr-gareth-roberts/insideLLMs/releases/tag/v0.1.0
