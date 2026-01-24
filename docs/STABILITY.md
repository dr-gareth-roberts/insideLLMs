## Stability & Versioning

insideLLMs aims to be CI-friendly: deterministic runs, stable artifacts, and explicit contracts.

### What is considered stable

- **CLI interface**: command names and flag semantics are treated as user-facing. Breaking changes
  should be avoided; if needed, deprecate first.
- **Serialized outputs**: `manifest.json`, `records.jsonl`, `summary.json`, `diff.json` are validated
  against strict, versioned schemas (`insideLLMs.schemas.SchemaRegistry`).
- **Custom trace bundle**: `ResultRecord.custom["trace"]` uses a strict schema and a stable key order
  (`insideLLMs.custom.trace@1`). New keys are append-only within each object; semantic changes require
  a schema bump.

### Versioning rules

- **Library version** follows SemVer.
- **Output schemas** are versioned independently (SemVer for the main output contract; named versions
  for custom bundles like `insideLLMs.custom.trace@1`).
- **Additive changes** (new optional fields) require a schema version bump.
- **Breaking changes** (removed/renamed fields, changed meaning) require a major schema bump.

### Determinism scope

The run → records → report → diff pipeline is deterministic for the same inputs and configuration.
Provider APIs, external services, and model stochasticity are outside this guarantee unless explicitly
controlled in configuration.

In canonical run artifacts, determinism is enforced by:

- Canonical JSON emission (stable key order and separators) for `records.jsonl`, `manifest.json`, `summary.json`, and `diff.json`.
- Omitting volatile runtime fields from the diff surface (e.g., `latency_ms` is persisted as `null`, and `manifest.json:command` is `null`).
- Content-addressing local file datasets via `dataset_hash=sha256:<...>` (when applicable) so dataset changes affect `run_id`.
