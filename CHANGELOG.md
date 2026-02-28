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

### Security
- Added delimiter escape sanitization to prevent injection bypass
- Enhanced API key exposure prevention in documentation
- Added pre-commit hook for secret detection

## Migration Guide

### Visualization Module (v1.1.0 → v2.0.0)

**Old code (deprecated):**
```python
from insideLLMs.visualization import TraceVisualizer
```

**New code:**
```python
from insideLLMs.analysis.visualization import TraceVisualizer
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
