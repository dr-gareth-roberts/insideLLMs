## Stability & Versioning

insideLLMs aims to be CI-friendly: deterministic runs, stable artefacts, and explicit contracts.

For a surface-by-surface contract table, see `docs/STABILITY_MATRIX.md`.

### What is considered stable

- **CLI interface**: command names and flag semantics are treated as user-facing. Breaking changes
  should be avoided; if needed, deprecate first.
- **Serialized outputs**: `manifest.json`, `records.jsonl`, `summary.json`, `diff.json` are validated
  against strict, versioned schemas (`insideLLMs.schemas.SchemaRegistry`).
- **Custom trace bundle**: `ResultRecord.custom["trace"]` uses a strict schema and a stable key order
  (`insideLLMs.custom.trace@1`). New keys are append-only within each object; semantic changes require
  a schema bump.

### Stability matrix

The detailed compatibility contract (Stable / Experimental / Internal) is maintained in:

- `docs/STABILITY_MATRIX.md`

Use the matrix during PR review when changing CLI behavior, schema-governed artifacts,
or registry extension points.

### Versioning rules

- **Library version** follows SemVer.
- **Output schemas** are versioned independently (SemVer for the main output contract; named versions
  for custom bundles like `insideLLMs.custom.trace@1`).
- **Additive changes** (new optional fields) require a schema version bump.
- **Breaking changes** (removed/renamed fields, changed meaning) require a major schema bump.

### Deprecation lifecycle

For user-facing stable surfaces (CLI semantics, schema-governed artifacts, and registry extension APIs),
use this lifecycle:

1. **Deprecate in minor release `X.Y`**
   - Keep behavior working.
   - Add release notes and migration guidance.
   - Prefer compatibility aliases when possible.
2. **Warn during deprecation window**
   - Emit clear warning messages in CLI/docs where practical.
   - Keep changelog entries explicit about timeline.
3. **Remove in next major release `Z.0`**
   - Only remove after at least one documented minor-release deprecation window.
   - Include final migration notes and replacement examples.

Minimum policy for every deprecation:

- Changelog entry under **Unreleased**.
- Update `docs/STABILITY_MATRIX.md` if contract scope changes.
- Add or update compatibility tests when behavior is user-visible.

### Determinism scope

The run → records → report → diff pipeline is deterministic for the same inputs and configuration.
Provider APIs, external services, and model stochasticity are outside this guarantee unless explicitly
controlled in configuration.

In canonical run artefacts, determinism is enforced by:

- Canonical JSON emission (stable key order and separators) for `records.jsonl`, `manifest.json`, `summary.json`, and `diff.json`.
- Omitting volatile runtime fields from the diff surface (e.g., `latency_ms` is persisted as `null`, and `manifest.json:command` is `null`).
- Content-addressing local file datasets via `dataset_hash=sha256:<...>` (when applicable) so dataset changes affect `run_id`.
