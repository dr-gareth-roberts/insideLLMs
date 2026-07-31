# Diff two behavioural runs

This offline example compares a known-good baseline with a candidate run using
`insidellms diff`. Both runs use `DummyModel`; different canned responses
simulate a behavioural change without API keys or network access.

Run the commands from the repository root after installing insideLLMs:

```bash
python3 -m pip install -e .
```

## 1. Create comparable run artefacts

The configurations use the same probe and dataset. Stable inputs give
corresponding records the same identity, while the different
`canned_response` values produce an output change.

```bash
rm -rf .tmp/diff-example

insidellms run examples/diff/baseline.yaml \
  --run-dir .tmp/diff-example/baseline

insidellms run examples/diff/candidate.yaml \
  --run-dir .tmp/diff-example/candidate
```

Each run directory now contains canonical `manifest.json` and `records.jsonl`
artefacts. The diff command matches records by model, probe, and example
identity before comparing outputs, metrics, traces, and record presence.

## 2. Inspect the diff

```bash
insidellms diff \
  .tmp/diff-example/baseline \
  .tmp/diff-example/candidate
```

The report identifies the changed output. An ordinary diff is informational,
so it exits with status `0` even when differences exist.

## 3. Turn the diff into a CI gate

Use `--fail-on-changes` when regressions, output changes, or added/removed
records should fail a build:

```bash
insidellms diff \
  .tmp/diff-example/baseline \
  .tmp/diff-example/candidate \
  --fail-on-changes
```

This example exits with status `2` because the candidate response changed.
Choose the gate that matches your policy:

| Option | Fails when |
|---|---|
| `--fail-on-regressions` | A measured score regresses |
| `--fail-on-changes` | Any regression, output change, or missing/new record exists |
| `--fail-on-trace-violations` | Trace violations increase |
| `--fail-on-trace-drift` | Trace fingerprints differ |
| `--fail-on-trajectory-drift` | Agent or tool trajectories differ |

## 4. Save machine-readable output

JSON output is useful for CI annotations and downstream analysis:

```bash
insidellms diff \
  .tmp/diff-example/baseline \
  .tmp/diff-example/candidate \
  --format json \
  --output .tmp/diff-example/diff.json

python3 -m json.tool .tmp/diff-example/diff.json
```

Add a gate option to the same command if producing the report should also fail
the CI step.

## Using this with real models

Use the same provider and model settings in both configurations, or compare two
model versions during an upgrade. Keep the probe and dataset stable, store the
approved baseline artefacts, and generate only the candidate run in CI. Review
intentional changes before replacing the baseline.
