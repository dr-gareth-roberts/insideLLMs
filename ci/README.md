# CI Harness

This directory contains the minimal harness used for CI diff-gating. It runs fully offline with no API keys.

## Contents

- `harness.yaml` — Harness config (DummyModel + probes that accept dict-or-string inputs)
- `harness_dataset.jsonl` — Small deterministic dataset (`question`, `prompt`, `task` fields)

## Extending the CI Harness

To add probes while preserving offline determinism:

1. **Use DummyModel only** — The harness uses `type: dummy`; do not add real model providers.
2. **Use compatible probes** — Probes must accept dict-or-string inputs. The current set (`logic`, `attack`, `instruction_following`, `code_generation`) all work with the dataset fields (`question`, `prompt`, `task`).
3. **Add to `harness.yaml`** — Add a probe entry under `probes:` with `type` and optional `args`:

   ```yaml
   probes:
     - type: logic
       args: {}
     - type: bias
       args: {}
   ```

4. **Verify dataset compatibility** — Ensure `harness_dataset.jsonl` includes fields your probe expects. Logic uses `question`; attack uses `prompt`; instruction_following and code_generation use `task`. If adding a probe that needs a new field, extend the dataset accordingly.
5. **Keep `max_examples` low** — The harness uses `max_examples: 3` for fast CI. Avoid increasing it unless necessary.

## Running Locally

```bash
insidellms harness ci/harness.yaml --run-dir .tmp/ci_harness
```

`run` and `harness` now exit 1 when any record is an error, timeout, skipped, or
unknown status, when no successful records exist, or when the number of records
does not match the expected evaluation count. Completed collection retains its
artifacts so item failures can be inspected. With `runner.stop_on_error: true`,
a harness can abort before aggregate records and the manifest are written;
initialization failures can do the same. These paths still exit nonzero.
`run_completed` means execution finished; `custom.health.healthy`
records whether it finished with a complete set of successful executions. A model
answer can execute successfully and still receive a failing behavioural score.

For a strict snapshot gate, use:

```bash
insidellms diff .tmp/baseline .tmp/candidate --fail-on-any-difference
```

This exits 2 for every detected difference, including improvements, score changes,
missing/extra records, and trace/trajectory drift. The older `--fail-on-changes`
flag retains its narrower behaviour: regressions, output/metric changes, and
missing/extra records, excluding improvements and trace-only differences.
Duplicate record identities are invalid input and return an error; the diff never
silently selects one duplicate. Repeated examples must have distinct replicate keys.
Non-finite scores (`NaN` or infinity) are invalid on either side. A regression gate
also exits 1 for missing or incomparable metric evidence, such as removing the
reference answers or changing the primary metric. It cannot report a pass merely
because a previous numeric score disappeared. Ungated diffs still report these
comparability changes for inspection.
Declared primary metrics must name an existing finite numeric score: null,
boolean, string, or missing primary values are malformed even when identical in
both runs. Legacy `scores.score` follows the same rule. Explicitly unscored
records use an empty score mapping and no primary metric.

## GitHub Actions and PR comments

The evaluation job runs candidate code and must have only `contents: read`, no
secrets, and `persist-credentials: false` on checkout. The action independently
checks both manifests and record streams before comparing them, including when
an older baseline harness returns zero for an unhealthy run. Completed artifact
directories and a health/diff report remain available on failure; an aborted
harness may have only a partial directory, which the health gate rejects.

The action's `fail-on-changes: "true"` input now selects the strict
`--fail-on-any-difference` gate. The repository enables it by default.

Inline PR comments have been removed from the composite action. The deprecated
`post-pr-comment` and `comment-on-forks` inputs produce a migration warning when
enabled; they do not grant commenting capability. Remove the former `github-token`
input and write permissions from your evaluation workflow.

The repository's `diff-gate-comment.yml` preserves sticky comments in a separate
`workflow_run` job using trusted code from the default branch. It never checks out
code or executes artifact contents. It accepts only one small `pr-report.json`
with fixed integer counts, validates its PR number and head SHA against GitHub's
workflow and current PR metadata, and updates only the bot's own comment. Reports
for stale commits or runs without an unambiguous associated PR are skipped.

To adopt this split, copy both `diff-gate.yml` and `diff-gate-comment.yml`, adjusting
the trusted workflow's `workflows: [Diff Gate]` trigger if you rename evaluation.
Keep artifact names consistent. The comment workflow must be present on the
default branch before GitHub will trigger it. No provider credentials should be
exposed to untrusted pull request evaluation.

## Related Docs

- `docs/DETERMINISM.md` — Determinism guarantees
- `docs/ARTIFACT_CONTRACT.md` — Artifact field contract
- `wiki/Determinism-and-CI.md` — CI diff-gating workflows
