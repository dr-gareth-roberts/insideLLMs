# Tracing and Fingerprinting

This page explains how trace bundles and output fingerprinting appear in `records.jsonl` and how
`insidellms diff` uses them for CI gating.

Trace data is opt-in: most runs will not include `custom.trace` unless you emit it explicitly
(e.g., via `TraceRecorder` + `trace_to_custom_field`).

## Trace bundles (`ResultRecord.custom["trace"]`)

A trace bundle is a structured summary (and optionally a full event list) of a record’s execution
trace. When present, it is stored under `ResultRecord.custom["trace"]` and validated against the
`insideLLMs.custom.trace@1` schema.

Key points:

- **Schema name**: `insideLLMs.custom.trace@1` (stable key ordering; append-only fields)
- **Location**: `ResultRecord.custom["trace"]`
- **Emitter**: `trace_to_custom_field` (plus whatever instrumentation populates the events)

If you do not emit a trace bundle, trace-aware diff flags are effectively no-ops.

### Where fingerprints and violations live

`insidellms diff` looks in two places:

- **Structured bundle**:
  - `custom.trace.fingerprint.value` (expects raw 64-hex or `sha256:<hex>`)
  - `custom.trace.violations` (TraceViolation schema)
- **Legacy flat fields**:
  - `custom.trace_fingerprint`
  - `custom.trace_violations`

If none are present, trace drift and trace violations are not evaluated.

## Trace drift vs. trace violations

**Trace drift**: for a matching record (same model/probe/example), drift is reported when both
baseline and candidate have trace fingerprints and they differ. This can catch behavioural changes
even if the final output text stays the same.

- Diff flag: `--fail-on-trace-drift`

**Trace violations increase**: violations come from trace contract validation (e.g., missing tool
result, invalid tool payload). Diff reports an increase when the candidate has a larger violation
count than baseline for the same record.

- Diff flag: `--fail-on-trace-violations`

## Output fingerprinting for structured outputs

`insidellms diff` compares outputs in this order:

1) If `_output_text` exists (e.g. `output_text` field, or a dict output containing `output_text` or
   `text`), diff compares only the extracted text.
2) Otherwise diff compares structured fingerprints (short 12-hex SHA-256 over canonical JSON).

The runner stores `custom.output_fingerprint` for non-string outputs and the diff uses it when no
ignore list is provided.

### Ignoring volatile output fields

Use `--output-fingerprint-ignore` to drop keys before fingerprinting structured outputs:

```bash
insidellms diff .tmp/runs/base .tmp/runs/head \
  --output-fingerprint-ignore timestamp,request_id
```

Notes:

- Keys are case-insensitive.
- The ignore list applies to any matching key at any depth (not path-based).
- When an ignore list is present, the diff recomputes fingerprints from output data (it will not use
  `custom.output_fingerprint` unless `output` is missing).

## CI gating recommendations

- **Strict determinism**: emit trace bundles and use `--fail-on-trace-drift`.
- **Contract enforcement**: use `--fail-on-trace-violations` to block regressions in trace
  correctness.
- **Structured outputs with noisy fields**: use `--output-fingerprint-ignore`.

Example CI sequence:

```bash
insidellms harness ci/harness.yaml --run-dir .tmp/runs/base --overwrite --skip-report
insidellms harness ci/harness.yaml --run-dir .tmp/runs/head --overwrite --skip-report
insidellms diff .tmp/runs/base .tmp/runs/head \
  --fail-on-changes \
  --fail-on-trace-drift \
  --output-fingerprint-ignore timestamp,request_id
```
