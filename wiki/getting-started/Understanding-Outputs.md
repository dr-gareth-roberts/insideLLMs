---
title: Understanding Outputs
parent: Getting Started
nav_order: 4
---

# Understanding Outputs

**What insideLLMs creates and why.**

## Artefacts by Command

| File | Created by | Purpose |
|------|------------|---------|
| `config.resolved.yaml` | `run`, `harness` | Resolved config snapshot |
| `records.jsonl` | `run`, `harness` | Every input/output pair (canonical) |
| `manifest.json` | `run`, `harness` | Run metadata and fingerprints |
| `summary.json` | `harness`, `report` | Aggregated metrics |
| `report.html` | `harness`, `report` | Standalone visual report |
| `results.jsonl` | `harness` | Legacy alias for `records.jsonl` |
| `diff.json` | `diff --format json --output diff.json` | Optional saved diff report |

`run` creates the three canonical artifacts. `harness` also creates a summary
and, unless `--skip-report` is used, an HTML report. Running `report` against an
existing run directory rebuilds `summary.json` and `report.html` there.

## records.jsonl

One JSON line per result:

```jsonl
{"example_id": "0", "input": {"question": "What is 2 + 2?"}, "output": "4", "status": "success"}
{"example_id": "1", "input": {"question": "Is the sky blue?"}, "output": "Yes", "status": "success"}
```

Key fields:
- `run_id` - Deterministic hash (same inputs = same ID)
- `example_id` - Input identifier
- `input` - Original data
- `output` - Model response
- `status` - `success` or `error`

## summary.json

Aggregated stats:

```json
{
  "models": {
    "gpt-4o": {"success_rate": 0.98, "example_count": 100}
  }
}
```

## report.html

Standalone HTML comparison. Open in browser. No server needed.

## Diff Output

```bash
# Print a human-readable diff to the terminal
insidellms diff baseline/ candidate/

# Save a JSON diff explicitly
insidellms diff baseline/ candidate/ --format json --output diff.json
```

Selected fields from the JSON report look like this:

```json
{
  "schema_version": "1.0.1",
  "baseline": "baseline",
  "candidate": "candidate",
  "counts": {
    "common": 100,
    "only_baseline": 0,
    "only_candidate": 0,
    "regressions": 0,
    "improvements": 0,
    "other_changes": 0,
    "trace_drifts": 0,
    "trace_violation_increases": 0,
    "trajectory_drifts": 0
  },
  "regressions": [],
  "improvements": [],
  "changes": []
}
```

The complete document also includes run IDs, duplicate counts, records found on
only one side, and trace/trajectory finding arrays. Each populated finding
contains a record identity; regression, improvement, and other-change entries
also include baseline/candidate summaries.

For CI:

```bash
insidellms diff baseline/ candidate/ --fail-on-changes
# Exit 2 for gated regressions/other changes/one-sided records
```

A plain diff is informational: it exits 0 even when differences are present.
`--fail-on-changes` does not fail for improvements alone or trace/trajectory-only
findings. Use their dedicated flags: trace gates return codes 3 and 4, and the
trajectory-drift gate returns code 5. A parsed command/setup error normally
returns 1; argparse usage errors also return 2 before the diff command runs.

## What Deterministic Means

insideLLMs stabilizes artifact naming, ordering, fingerprints, and selected
metadata when the resolved config, inputs, and model responses are the same.
It does not make a hosted model deterministic: provider responses may still
change because of sampling, model updates, or service behaviour.

Canonical artifacts deliberately omit or normalize volatile runtime data. For
example, timestamps may be synthesized and latency may be `null`; do not treat
those fields as wall-clock observability data.

Enables:

- CI diff-gating (block regressions)
- Reproducibility of the evaluation record and artifact contract
- Caching (skip computed results)

Run directories contain original prompts and model outputs. Treat them as
potentially sensitive. Export-time PII redaction changes only the exported copy,
not `records.jsonl` in the run directory.

## Next

[CI Integration Tutorial →](../tutorials/CI-Integration.md) Block regressions in CI.
