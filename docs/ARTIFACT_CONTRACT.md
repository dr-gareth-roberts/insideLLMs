## Artifact Contract

This page is the field-level contract for deterministic run artifacts.
Use it as the source of truth when updating artifact emission, validation, or CI diff-gating.

For broader guidance, see `docs/DETERMINISM.md`.

### Canonical Run Artifact Set

Every run directory is expected to include:

- `records.jsonl` (canonical record stream)
- `manifest.json` (run-level metadata)
- `config.resolved.yaml` (resolved config snapshot)
- `summary.json` (aggregated metrics)
- `report.html` (human-readable report)

### Determinism and Volatility Rules

| Artifact | Contract | Deterministic | Volatile Rules |
|---|---|---|---|
| `records.jsonl` | One JSON object per execution item, stable ordering | Yes | `latency_ms` is intentionally persisted as `null` |
| `manifest.json` | Run-level metadata and schema references | Yes | `command` is intentionally persisted as `null`; `python_version`/`platform` may be `null` in deterministic mode |
| `config.resolved.yaml` | Fully resolved effective config for replay/repro | Yes | None |
| `summary.json` | Aggregated summaries derived from records | Yes (for fixed records) | Generation timestamp is deterministic in standard run flow |
| `report.html` | Human-facing summary report | Yes (for fixed records) | Should not include wall-clock or host-specific unstable fields |

### `records.jsonl` Field Expectations

- Status values follow schema enum (`success`, `error`, `timeout`, `rate_limited`, `skipped`)
- `custom.replicate_key` identifies logical item identity for diffing
- `custom.output_fingerprint` is used for structured output drift checks
- Timeout diagnostics (when applicable) are persisted via:
  - `status: "timeout"`
  - `error_type`
  - `custom.timeout` (boolean)
  - `custom.timeout_seconds` (numeric, when known)

### `manifest.json` Field Expectations

- Required counters:
  - `record_count`
  - `success_count`
  - `error_count`
- Extended status diagnostics are persisted in `custom`:
  - `custom.status_counts`
  - `custom.timeout_count`
- `records_file` must reference canonical `records.jsonl`
- `schemas` must include active schema bindings for emitted artifacts

### Legacy Artifact Aliases

| Alias | Canonical | Status | Deprecation |
|-------|-----------|--------|-------------|
| `results.jsonl` | `records.jsonl` | Deprecated | Emitted for backward compatibility (symlink/copy). Prefer `records.jsonl`. Removal planned for 0.3.x; timeline will be announced in changelog. |

### Compatibility Policy

- New fields should prefer `custom` namespaces unless schema versioning is explicitly updated.
- Existing canonical fields must not change semantics without schema/version migration.
- Any planned alias/shim/removal should be documented in changelog and migration docs.

