# Golden Path: 5-Minute Quickstart

This walkthrough runs end-to-end without API keys (uses `DummyModel`) and demonstrates
the complete diff workflow you'll use in production and CI.

## What You'll Learn

1. Run a deterministic evaluation
2. Generate baseline artifacts
3. Diff two runs to detect changes
4. Generate an HTML diff report

**Time required:** ~5 minutes
**API keys required:** None

---

## 1. Install

```bash
git clone https://github.com/dr-gareth-roberts/insideLLMs.git
cd insideLLMs
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Verify the installation:

```bash
insidellms doctor
```

---

## 2. Create a Baseline

Run the built-in harness with DummyModel (deterministic, offline):

```bash
insidellms harness ci/harness.yaml --run-dir .tmp/runs/baseline --overwrite
```

This produces deterministic artifacts:

```
.tmp/runs/baseline/
├── manifest.json         # Run metadata (content-addressed)
├── records.jsonl         # Every input/output pair (canonical)
├── config.resolved.yaml  # Normalized configuration
├── summary.json          # Aggregated metrics
└── report.html           # Human-readable report
```

View the report:

```bash
open .tmp/runs/baseline/report.html  # macOS
# or: xdg-open .tmp/runs/baseline/report.html  # Linux
```

---

## 3. Create a Candidate (Identical Run)

Run the same harness again:

```bash
insidellms harness ci/harness.yaml --run-dir .tmp/runs/candidate --overwrite
```

---

## 4. Diff the Runs

Now compare them:

```bash
insidellms diff .tmp/runs/baseline .tmp/runs/candidate
```

You should see:

```
Behavioural Diff

  X compared | 0 regressions | 0 improvements | 0 changes | X unchanged

  Baseline:  .tmp/runs/baseline
  Candidate: .tmp/runs/candidate

No behavioural differences detected
```

**This is the key insight:** Identical inputs + identical model = identical outputs.

---

## 5. Generate an HTML Diff Report

```bash
insidellms diff .tmp/runs/baseline .tmp/runs/candidate --html diff.html
open diff.html
```

The HTML report provides:
- Side-by-side comparison of responses
- Filters for regressions, improvements, changes
- Search by model, probe, or example ID
- Export to JSON

---

## 6. CI Gate (Fail on Changes)

In CI, you'd use the fail flags:

```bash
insidellms diff .tmp/runs/baseline .tmp/runs/candidate --fail-on-changes
echo "Exit code: $?"  # Should be 0 (no changes)
```

Exit codes:
- `0`: No differences detected
- `2`: Regressions or changes detected (with `--fail-on-regressions` or `--fail-on-changes`)

---

## 7. Validate Schemas (Optional)

Verify the artifacts conform to the expected schemas:

```bash
insidellms schema validate --name ResultRecord --input .tmp/runs/baseline/records.jsonl
insidellms schema validate --name RunManifest --input .tmp/runs/baseline/manifest.json
```

---

## Next Steps

### Use Real Models

```bash
# Initialize a config for OpenAI
insidellms init --model openai --model-name gpt-4o

# Set your API key
export OPENAI_API_KEY="sk-..."

# Run
insidellms harness experiment.yaml --run-dir ./my-baseline
```

### Add to CI

See [CI Integration Guide](ci-integration.md) for GitHub Actions, GitLab CI, and Jenkins examples.

### Compare Models

```yaml
# harness.yaml - compare GPT-4o vs Claude
models:
  - type: openai
    args: {model_name: gpt-4o}
  - type: anthropic
    args: {model_name: claude-3-5-sonnet-20241022}

probes:
  - type: logic
  - type: safety

dataset:
  format: jsonl
  path: data/prompts.jsonl
```

```bash
insidellms harness harness.yaml --run-dir ./comparison
insidellms report ./comparison
```

---

## Troubleshooting

### "records.jsonl not found"

Ensure the run directory exists and the harness completed successfully:

```bash
ls -la .tmp/runs/baseline/
```

### Non-zero diff on identical runs

Check that:
1. You're using DummyModel (deterministic)
2. No external factors changed (timestamps are neutralized by default)
3. Dataset hasn't been modified between runs

```bash
# Verify determinism
insidellms harness ci/harness.yaml --run-dir .tmp/run1 --overwrite
insidellms harness ci/harness.yaml --run-dir .tmp/run2 --overwrite
diff .tmp/run1/records.jsonl .tmp/run2/records.jsonl
# Should show no differences
```

### Reset and try again

```bash
rm -rf .tmp/runs
insidellms harness ci/harness.yaml --run-dir .tmp/runs/baseline --overwrite
```
