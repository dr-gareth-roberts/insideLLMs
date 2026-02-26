# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- International PII detection support (EU, UK regions)
- Rate limiting integration via `RunConfig`
- Schema version compatibility checking in diff command
- Delimiter escape protection in defensive prompt builder
- Comprehensive API key security documentation and pre-commit hooks

### Changed
- **DEPRECATION**: `insideLLMs.visualization` module is deprecated
  - Use `insideLLMs.analysis.visualization` instead
  - Compatibility layer will be removed in v2.0.0
  - See migration guide below

### Fixed
- Delimiter escape vulnerability in prompt injection defense
- Async determinism with concurrent execution
- Config loading error messages now show line numbers and context

### Security
- Added delimiter escape sanitization to prevent injection bypass
- Enhanced API key exposure prevention in documentation
- Added pre-commit hook for secret detection

### Docs changelog
- Aligned `API_REFERENCE.md` with current runtime/CLI surfaces, including `insideLLMs.diffing`,
  `DiffGatePolicy`, DiffReport schema-validation flow, `insideLLMs.shadow.fastapi`, and updated
  `harness`/`diff`/`doctor`/`init` command coverage.
- Audited and corrected long-form docs to match current CLI flags and behavior (notably
  `wiki/reference/CLI.md`, CI integration, determinism, troubleshooting, and related guides).
- Added dedicated guides for production shadow capture and VS Code/Cursor extension workflows:
  `wiki/guides/Production-Shadow-Capture.md` and `wiki/guides/IDE-Extension-Workflow.md`.
- Expanded `scripts/audit_docs.py` to enforce parser smoke checks, stale-option detection, and
  API/docs-index parity checks for the new surfaces.

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

## [1.0.0] - 2024-01-15

### Added
- Initial stable release
- Deterministic probe runner
- Multi-provider model support
- Diff-gating for CI/CD
- Comprehensive probe library

[Unreleased]: https://github.com/yourusername/insideLLMs/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/insideLLMs/releases/tag/v1.0.0
