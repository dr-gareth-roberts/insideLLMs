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
`canned_response` values produce a measured accuracy regression. Each dataset
item supplies a held-out `reference_answer`; the runner evaluates the output
and writes `scores.accuracy` and `primary_metric: accuracy` to the record.

```bash
DIFF_EXAMPLE_DIR=$(mktemp -d)

insidellms run examples/diff/baseline.yaml \
  --run-dir "$DIFF_EXAMPLE_DIR/baseline"

insidellms run examples/diff/candidate.yaml \
  --run-dir "$DIFF_EXAMPLE_DIR/candidate"
```

Each run directory now contains canonical `manifest.json` and `records.jsonl`
artefacts. The diff command matches records by model, probe, and example
identity before comparing outputs, metrics, traces, and record presence.

## 2. Inspect the diff

```bash
insidellms diff \
  "$DIFF_EXAMPLE_DIR/baseline" \
  "$DIFF_EXAMPLE_DIR/candidate"
```

The report identifies accuracy dropping from `1.0` (Paris) to `0.0` (Lyon).
An ordinary diff is informational,
so it exits with status `0` even when differences exist.

## 3. Turn the diff into a CI gate

Use `--fail-on-regressions` to fail a build when measured scores fall:

```bash
insidellms diff \
  "$DIFF_EXAMPLE_DIR/baseline" \
  "$DIFF_EXAMPLE_DIR/candidate" \
  --fail-on-regressions
```

This example exits with status `2` because the candidate answer is incorrect.
Choose the gate that matches your policy:

| Option | Fails when |
|---|---|
| `--fail-on-regressions` | A measured score regresses |
| `--fail-on-changes` | A regression, neutral/other output change, or missing/new record exists; improvements alone do not fail |
| `--fail-on-trace-violations` | Trace violations increase |
| `--fail-on-trace-drift` | Trace fingerprints differ |
| `--fail-on-trajectory-drift` | Agent or tool trajectories differ |

## 4. Save machine-readable output

JSON output is useful for CI annotations and downstream analysis:

```bash
insidellms diff \
  "$DIFF_EXAMPLE_DIR/baseline" \
  "$DIFF_EXAMPLE_DIR/candidate" \
  --format json \
  --output "$DIFF_EXAMPLE_DIR/diff.json"

python3 -m json.tool "$DIFF_EXAMPLE_DIR/diff.json"
```

Add a gate option to the same command if producing the report should also fail
the CI step.

For a shareable, human-readable artefact, add `--html`:

```bash
insidellms diff \
  "$DIFF_EXAMPLE_DIR/baseline" \
  "$DIFF_EXAMPLE_DIR/candidate" \
  --html "$DIFF_EXAMPLE_DIR/diff.html"
```

The HTML report is self-contained (inline CSS, no scripts, no external assets)
and deterministic: the same diff renders byte-identical output. It works with
either `--format` and leaves the exit code unchanged.

## Scored dataset contract

For `ScoredProbe` subclasses, use `reference_answer` for the expected answer or
probe-specific evaluation criteria. `reference` is an alias; specifying both
is an item error. The runner strips these fields before calling `probe.run`,
but keeps the original input in the artefact and passes it to `evaluate_single`.

Omitting both fields leaves the item unscored: execution success is still
recorded, but aggregate accuracy is `null`. Use an explicit `reference_answer:
null` for evaluators that do not require a reference. Evaluation must return an
`is_correct` boolean and may supply finite numeric `scores`, a `score`, and a
`primary_metric` naming a score. Every evaluated item gets `scores.accuracy`;
the primary metric defaults to `score` when supplied, otherwise `accuracy`.
Evaluator exceptions and invalid metrics become item errors, not passing scores.

Scoring applies to sync, async, direct probe batches and harness runs. Resuming a
run restores evaluation metadata and scores without calling the evaluator again.
Historical labelled runs without persisted evaluation cannot be resumed: rerun
them into a new directory to establish a scored baseline.

## Using this with real models

Use the same provider and model settings in both configurations, or compare two
model versions during an upgrade. Keep the probe and dataset stable, store the
approved baseline artefacts, and generate only the candidate run in CI. Review
intentional changes before replacing the baseline.
