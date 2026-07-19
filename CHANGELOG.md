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

### Deprecated
- **`results.jsonl` legacy alias** (Stable artifact surface)
  - Canonical file is `records.jsonl`; harness runs still emit `results.jsonl` as a symlink or copy for backward compatibility.
  - **Removal planned in v0.3.0.** After removal, use `records.jsonl` only; update diff snapshot and CI scripts that reference the alias.
  - See `docs/ARTIFACT_CONTRACT.md` (Legacy Artifact Aliases).

### Changed
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
