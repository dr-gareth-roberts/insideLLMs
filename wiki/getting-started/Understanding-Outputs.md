---
title: Understanding Outputs
parent: Getting Started
nav_order: 4
---

# Understanding Outputs

**What insideLLMs creates and why.**

## Artefacts

| File | Purpose |
|------|----------|
| `records.jsonl` | Every input/output pair (canonical) |
| `manifest.json` | Run metadata |
| `summary.json` | Aggregated metrics |
| `report.html` | Visual comparison |
| `diff.json` | Change detection (via `insidellms diff`) |

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

## diff.json

```bash
insidellms diff baseline/ candidate/
```

```json
{
  "changes": [
    {"example_id": "42", "field": "output",
     "baseline": "The answer is 4", "candidate": "The answer is four"}
  ],
  "summary": {"total_examples": 100, "changed": 3}
}
```

For CI:
```bash
insidellms diff baseline/ candidate/ --fail-on-changes
# Exit 1 if changes detected
```

## Why Deterministic?

Same inputs → byte-for-byte identical outputs.

Enables:
- CI diff-gating (block regressions)
- Reproducibility (re-run = same artefacts)
- Caching (skip computed results)

## Next

[CI Integration Tutorial →](../tutorials/CI-Integration.md) Block regressions in CI.
