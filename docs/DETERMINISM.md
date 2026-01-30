## Determinism

insideLLMs is designed so the “run → records → report → diff” spine can be used for CI diff-gating.
For the same inputs and configuration, run directories are intended to be byte-for-byte identical.
This assumes the model responses themselves are identical (e.g., deterministic sampling settings,
cached responses, or provider-side determinism).

### Canonical run artefacts

A run directory contains (at minimum):

- `records.jsonl`: canonical record stream (one JSON object per line)
- `manifest.json`: run metadata (`RunManifest` schema)
- `config.resolved.yaml`: normalized config snapshot used for the run
- `summary.json`: aggregates
- `report.html`: human-readable report

### What we do to stay deterministic

- **Canonical JSON emission**: on-disk JSON/JSONL is written with a stable key order and separators.
- **Deterministic time spine**: run/item timestamps are derived from `run_id` (not wall-clock).
- **Stable ordering**: any ordering that could affect artefacts (e.g., plugin discovery, filesystem listings)
  is sorted.
- **Content-address datasets**: for local file datasets (`format: csv|jsonl`), if a hash is not provided,
  insideLLMs computes `dataset_hash=sha256:<file-bytes>` and includes it in `manifest.json`. Dataset
  changes therefore change `run_id`.
- **Pinned remote datasets**: for HuggingFace datasets (`format: hf`), include a `revision` or
  explicit `dataset_hash` to keep run IDs stable when upstream datasets change.
- **Resume safety**: resumable runs validate that existing records match the current prompt inputs
  (run_id + input fingerprint), preventing mixed artefacts.

### Volatile fields (intentionally omitted)

Some values are inherently non-deterministic (wall-clock timing, exact CLI invocation, host metadata).
To keep the diff surface stable:

- `ResultRecord.latency_ms` is persisted as `null`.
- `manifest.json:command` is persisted as `null`.

If you need timing/host details, use tracing/telemetry rather than the canonical CI artefacts.

### Determinism controls (strict mode)

insideLLMs defaults to strict determinism controls when emitting canonical run artefacts.
If you want a more permissive mode (or want host metadata persisted), you can disable them.

Config (`config.yaml` / `harness.yaml`):

```yaml
determinism:
  strict_serialization: false
  deterministic_artifacts: false
```

CLI equivalents:

```bash
insidellms run config.yaml --no-strict-serialization --no-deterministic-artifacts
insidellms harness harness.yaml --no-strict-serialization --no-deterministic-artifacts
```

What these do:

- `strict_serialization`: fail fast when hashing/fingerprinting would fall back to
  non-deterministic stringification (for example, exotic objects or ambiguous dict keys).
- `deterministic_artifacts`: neutralize host-dependent manifest fields like
  `python_version` and `platform` by persisting them as `null`.

Notes:

- If `deterministic_artifacts` is omitted, it defaults to the value of
  `strict_serialization`.
- Defaults: `strict_serialization=true`, `deterministic_artifacts=None` (follows strict).
- Determinism config values must be booleans (or null/omitted). Strings like
  `"true"` are rejected to avoid silent truthiness bugs.

### Verifying determinism locally

Run the same harness twice and diff the run dirs:

```bash
insidellms harness ci/harness.yaml --run-dir .tmp/runs/base --overwrite --skip-report
insidellms harness ci/harness.yaml --run-dir .tmp/runs/head --overwrite --skip-report
insidellms diff .tmp/runs/base .tmp/runs/head --fail-on-changes
```

Notes:

- `report.html` is deterministic when generated from deterministic records; `--skip-report` is
  optional and is mainly for speed in CI runs.
