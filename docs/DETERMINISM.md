## Determinism

insideLLMs is designed so the “run → records → report → diff” spine can be used for CI diff-gating.
For the same inputs and configuration, run directories are intended to be byte-for-byte identical.

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
- **Resume safety**: resumable runs validate that existing records match the current prompt inputs
  (run_id + input fingerprint), preventing mixed artefacts.

### Volatile fields (intentionally omitted)

Some values are inherently non-deterministic (wall-clock timing, exact CLI invocation, host metadata).
To keep the diff surface stable:

- `ResultRecord.latency_ms` is persisted as `null`.
- `manifest.json:command` is persisted as `null`.

If you need timing/host details, use tracing/telemetry rather than the canonical CI artefacts.

### Verifying determinism locally

Run the same harness twice and diff the run dirs:

```bash
insidellms harness ci/harness.yaml --run-dir .tmp/runs/base --overwrite --skip-report
insidellms harness ci/harness.yaml --run-dir .tmp/runs/head --overwrite --skip-report
insidellms diff .tmp/runs/base .tmp/runs/head --fail-on-changes
```
