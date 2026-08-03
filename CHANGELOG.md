# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Full CI pipeline: lint, typecheck, test (Python 3.10/3.11/3.12 matrix), determinism, and contract jobs
- Optional dependency groups: `huggingface`, `signing`, `crypto`, `providers`
- International PII detection support (EU, UK regions)
- Rate limiting integration via `RunConfig`
- Schema version compatibility checking in diff command
- Delimiter escape protection in defensive prompt builder

### Fixed
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

### Visualization Module (v1.1.0 → v2.0.0)

**Old code (deprecated):**
```python
from insideLLMs.visualization import text_bar_chart
```

**New code:**
```python
from insideLLMs.analysis.visualization import text_bar_chart
```

**Timeline:**
- v1.1.0 (current): Deprecation warnings issued, old imports still work
- v1.2.0: Continued deprecation warnings
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

## [0.2.0] - 2025-01-15

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

## [0.1.0] - 2024-09-01

### Added
- Initial release
- Core probe execution pipeline
- `ProbeRunner` with config-driven execution
- Basic model adapters (OpenAI, DummyModel)
- JSONL record output format

[Unreleased]: https://github.com/dr-gareth-roberts/insideLLMs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dr-gareth-roberts/insideLLMs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dr-gareth-roberts/insideLLMs/releases/tag/v0.1.0
