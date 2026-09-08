---
title: CI Integration
parent: Tutorials
nav_order: 3
---

# CI Integration Tutorial

Use deterministic run artefacts to detect behavioural changes in pull requests. This tutorial is
offline: it uses `DummyModel`, makes no provider calls, and needs no API keys.

**Time:** 30 minutes

**Prerequisites:** Python 3.10+, insideLLMs, Git, and GitHub Actions

If insideLLMs is not installed yet, follow [Quick Install](../getting-started/Quick-Install.md).
Run all local commands below from your repository root unless a command block explicitly changes
directory.

## Choose what “baseline” means

insideLLMs supports two different CI workflows. Choose deliberately; they answer different
questions.

| Workflow | Baseline | Best question to answer |
|---|---|---|
| Reusable action | A fresh, ephemeral run from the PR base commit | What behaviour changed between the base and head revisions? |
| Approved artefact | An exact run directory reviewed and stored at an immutable commit | Does the candidate still match the behaviour we approved? |

The reusable action does **not** read an approved `ci/baseline` directory. It checks out the base
revision, runs its harness, runs the candidate harness, and compares those temporary runs. Use the
approved-artefact workflow when approval must attach to exact records.

## 1. Create a portable offline harness

Generate the config from inside `ci/`:

```bash
mkdir -p ci
(
  cd ci
  insidellms init harness.yaml --template harness
)
```

This creates:

```text
ci/
├── harness.yaml
└── data/
    └── harness_dataset.jsonl
```

The location matters. `init` writes sample data relative to the current directory, while the
harness resolves the dataset path relative to the config file. Running `init` inside `ci/` keeps
the generated `data/harness_dataset.jsonl` path valid.

Add this top-level block to `ci/harness.yaml` for CI:

```yaml
runner:
  stop_on_error: true
```

By default, an item-level provider or probe exception becomes an `error` or `timeout` record, the
harness continues, and the command can still exit `0`. `runner.stop_on_error: true` aborts on the
first item error, so the harness command exits `1` instead of silently producing a partial CI run.
If partial runs are intentional, leave it `false` and inspect `manifest.json` error and timeout
counts explicitly.

Resolve the execution plan without running a model:

```bash
insidellms harness ci/harness.yaml --dry-run
```

The generated template currently reports:

```text
Models: 1
Probes: 4
Dataset examples: 3
max_examples cap: 3
Total evaluations: 12
```

Commit the config and dataset. Keep temporary runs out of version control:

```bash
git add ci/harness.yaml ci/data/harness_dataset.jsonl
git commit -m "Add offline behavioural harness"
```

## 2. Exercise the diff locally

Create two temporary runs from the same config:

```bash
insidellms harness ci/harness.yaml \
  --run-dir .tmp/runs/baseline \
  --skip-report

insidellms harness ci/harness.yaml \
  --run-dir .tmp/runs/candidate \
  --skip-report
```

Compare them without a gate first:

```bash
insidellms diff .tmp/runs/baseline .tmp/runs/candidate
```

The important part of the output is:

```text
Behavioural Diff
Baseline: .tmp/runs/baseline
Comparison: .tmp/runs/candidate
Common keys: 12
Only in baseline: 0
Only in comparison: 0
Regressions: 0
Improvements: 0
Other changes: 0
```

A plain `diff` is a report: it exits `0` even when it finds differences. Add a gate when CI must
fail, and use JSON format whenever `--output` is present:

```bash
insidellms diff .tmp/runs/baseline .tmp/runs/candidate \
  --format json \
  --output .tmp/diff.json \
  --fail-on-changes
```

`--output` is ignored for text output and emits a warning, so `--format json --output PATH` should
always be treated as a pair.

### Understand gate classifications and exit codes

| Invocation or condition | Exit |
|---|---:|
| Plain diff, including a diff that reports changes | `0` |
| `--fail-on-regressions` finds a regression | `2` |
| `--fail-on-changes` finds a regression, other change, or one-sided record | `2` |
| `--fail-on-trace-violations` finds increased violations | `3` |
| `--fail-on-trace-drift` finds trace drift | `4` |
| `--fail-on-trajectory-drift` finds trajectory drift | `5` |
| Command, config, dataset, or artefact failure | normally `1` |

There are two details worth calling out:

- Despite its name, `--fail-on-changes` currently does not fail on an improvement-only score
  change. Use the JSON classifications and choose a policy that matches your review process.
- Argument-parser usage errors can also exit `2`. Do not interpret `2` as proof of behavioural
  drift without reading the diagnostic or the generated `diff.json`.

When several enabled gates apply, exit-code precedence is `2`, then `3`, then `4`, then `5`. The
JSON report retains the separate finding counts.

## Workflow A: ephemeral base-commit versus head-commit

Use the reusable composite action when you want to regenerate both sides from Git revisions. Save
this as `.github/workflows/behavioural-base-vs-head.yml`:

> **Migration notice:** the alternate action at `.github/actions/diff-gate` is retired and now
> always exits `1` without installing dependencies, running a harness, producing a diff, or posting
> a comment. Existing input names remain accepted only so callers receive the migration error.
> Change `uses: ./.github/actions/diff-gate` to the repository-root action (`uses: ./` for a local
> checkout, or `uses: dr-gareth-roberts/insideLLMs@<reviewed-full-sha>`). Inline commenting is no
> longer available from the evaluation action; use the split, trusted `workflow_run` comment
> workflow documented below.

```yaml
name: Behavioural Base vs Head

on:
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  behavioural-diff:
    runs-on: ubuntu-latest
    steps:
      - name: Check out the PR head commit
        uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          fetch-depth: 0
          persist-credentials: false

      - name: Compare base and head runs
        id: insidellms
        uses: dr-gareth-roberts/insideLLMs@v1
        with:
          harness-config: ci/harness.yaml
          fail-on-changes: "true"

      - name: Upload the JSON diff
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: insidellms-diff-${{ github.sha }}
          path: ${{ steps.insidellms.outputs.diff-json }}
          if-no-files-found: warn
          retention-days: 7
```

For production use, pin third-party actions, including insideLLMs, to reviewed full commit SHAs.
The tags above keep the example readable.

The current action performs these steps:

1. Resolve the baseline from `baseline-ref`, then the PR base SHA, then `GITHUB_BASE_REF`, and
   finally `main`.
2. Create a detached worktree for that base commit.
3. Install and run insideLLMs from the base checkout into a temporary run directory.
4. Install and run the checked-out candidate into another temporary run directory.
5. Run `diff --format json`; by default it also adds `--fail-on-changes`.
6. Expose paths and the diff exit code as action outputs, write an allowlisted PR report for the
   trusted follow-up workflow, and propagate a non-zero gate result.

Those temporary base and candidate runs live under the GitHub runner's temporary directory. The
action does not consult a committed `ci/baseline`, and only `diff.json` is uploaded by the example.
The repository's `.github/workflows/diff-gate-comment.yml` shows the separate `workflow_run`
pattern for downloading the allowlisted report and updating a pull-request comment with narrowly
scoped write permission. Keep candidate checkout and execution in the read-only evaluation
workflow; the trusted comment workflow must never check out or execute candidate code.

### Current action scope

The current action runner executes `pip install -e` against both compared repository revisions.
That makes it suitable for this insideLLMs source repository, or another repository that contains
an installable insideLLMs source tree. If your application merely consumes the published package,
use the explicit pinned-package workflow below or adapt the installation step for your repository.

The base-vs-head action is a change detector, not a security boundary: candidate code is installed
and run, and the candidate installation supplies the final CLI process. Use the approved-artefact
workflow when the checker and baseline must come from a trusted revision.

### Simulate a behavioural change

Change the generated model block on a branch:

```yaml
models:
  - type: dummy
    args:
      canned_response: CHANGED RESPONSE
```

Re-run the local candidate with `--overwrite`, because that exact temporary directory already
exists:

```bash
insidellms harness ci/harness.yaml \
  --run-dir .tmp/runs/candidate \
  --overwrite \
  --skip-report

insidellms diff .tmp/runs/baseline .tmp/runs/candidate
```

With the generated 12-evaluation harness, this is reported as `Other changes: 12`. The plain diff
still exits `0`; `--fail-on-changes` exits `2`. It is a behavioural change, not a score regression,
so calling every non-zero gate a “regression” would be misleading.

## Workflow B: an explicitly approved immutable artefact

Use this workflow when the exact baseline records are the reviewed object. The safest pattern is:

1. Generate the baseline offline with the same pinned insideLLMs version used in CI.
2. Review the config, dataset, `records.jsonl`, `manifest.json`, and `summary.json`.
3. Merge the baseline through a protected, reviewed branch.
4. Record the resulting full 40-character commit SHA in the repository variable
   `INSIDELLMS_APPROVED_BASELINE_SHA`.
5. Never let a candidate PR silently update that variable or approve its own replacement baseline.

After the approved baseline is merged, a repository administrator should create the Actions
repository variable in GitHub settings. Use the exact commit printed by `git rev-parse HEAD` from
the reviewed branch. Restrict changes to the variable and protect the baseline branch with your
organisation's required-review rules. Workflow B fails closed if the variable is missing, is not a
full SHA, resolves to a different commit, or does not contain `ci/baseline/records.jsonl`.

For an initial baseline:

```bash
insidellms harness ci/harness.yaml \
  --run-dir ci/baseline \
  --skip-report

# Optional when the pydantic-backed validation dependency is installed:
insidellms validate ci/baseline

git add ci/harness.yaml ci/data/harness_dataset.jsonl ci/baseline
git commit -m "Approve initial behavioural baseline"
```

Do not routinely regenerate an approved directory in place with `--overwrite`. Generate a proposed
baseline elsewhere, diff it against the approved run, review the changed records, and replace the
stored directory only in an explicit baseline-approval pull request. After merge, update the
repository variable to that reviewed merge commit SHA.

The following two-job workflow keeps candidate execution away from the approved checkout and uses
a pinned published insideLLMs version for the trusted diff. Save it as
`.github/workflows/behavioural-approved-baseline.yml`:

```yaml
name: Behavioural Approved Baseline

on:
  pull_request:
    branches: [main]

permissions:
  contents: read

env:
  INSIDELLMS_VERSION: "0.2.0"

jobs:
  candidate:
    runs-on: ubuntu-latest
    steps:
      - name: Check out the PR head commit
        uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          persist-credentials: false

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install the pinned evaluator
        run: python -m pip install "insidellms==${INSIDELLMS_VERSION}"

      - name: Run the candidate harness
        run: |
          insidellms harness ci/harness.yaml \
            --run-dir candidate-run \
            --skip-report

      - name: Transfer the candidate run to the trusted gate job
        uses: actions/upload-artifact@v4
        with:
          name: insidellms-candidate-${{ github.sha }}
          path: candidate-run
          if-no-files-found: error
          retention-days: 1

  gate:
    needs: candidate
    runs-on: ubuntu-latest
    steps:
      - name: Check out the approved baseline commit
        uses: actions/checkout@v4
        with:
          ref: ${{ vars.INSIDELLMS_APPROVED_BASELINE_SHA }}
          path: approved
          persist-credentials: false

      - name: Verify the approved revision
        env:
          APPROVED_SHA: ${{ vars.INSIDELLMS_APPROVED_BASELINE_SHA }}
        run: |
          if [[ ! "${APPROVED_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
            echo "INSIDELLMS_APPROVED_BASELINE_SHA must be a full commit SHA" >&2
            exit 1
          fi
          test "$(git -C approved rev-parse HEAD)" = "${APPROVED_SHA}"
          test -f approved/ci/baseline/records.jsonl

      - name: Download the candidate run
        uses: actions/download-artifact@v4
        with:
          name: insidellms-candidate-${{ github.sha }}
          path: candidate-run

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install the trusted evaluator
        run: python -m pip install "insidellms==${INSIDELLMS_VERSION}"

      - name: Compare with the approved run
        run: |
          insidellms diff approved/ci/baseline candidate-run \
            --format json \
            --output diff.json \
            --fail-on-changes

      - name: Upload the JSON diff
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: insidellms-approved-diff-${{ github.sha }}
          path: diff.json
          if-no-files-found: warn
          retention-days: 7
```

If your harness needs application code or a custom plugin, install that code only in the candidate
job. Keep the gate job's evaluator pinned to a reviewed version. Changing the evaluator version is
a baseline migration: regenerate, review, and approve a new baseline rather than mixing tool
versions silently.

## Pull-request and fork safety

Use the ordinary `pull_request` event for workflows that check out and execute candidate code.
GitHub restricts the token and withholds repository secrets from fork pull requests. Keep it that
way:

- Do not change this workflow to `pull_request_target` and then check out or execute the untrusted
  PR head. `pull_request_target` runs in a privileged base-repository context.
- Do not pass provider keys, cloud credentials, signing identities, or other secrets to a fork
  candidate job. Route secret-backed evaluations to a separate trusted, post-merge or manually
  approved workflow.
- The reusable action never posts comments. If comments are wanted, publish only its allowlisted
  report and let the separate trusted `workflow_run` workflow validate and render it.
- Treat any candidate checkout and its build/install hooks as untrusted code, even when the harness
  itself uses `DummyModel`.

## Artefact sensitivity and retention

Run directories can contain raw prompts, model outputs, errors, dataset references, and resolved
configuration. A JSON diff can repeat changed output text or fingerprints. Do not place secrets in
YAML model arguments, and do not assume a GitHub Actions artefact is public-safe.

Upload only what reviewers need, set an explicit short `retention-days`, restrict repository
access, and apply your organisation's redaction and data-residency rules. The approved workflow
uploads a raw candidate run for one day because a separate trusted job needs it; avoid that pattern
for sensitive production traffic unless your artefact store and access policy are appropriate.

## Additional deterministic gates

Structured model outputs can exclude known volatile output keys from output fingerprinting:

```bash
insidellms diff .tmp/runs/baseline .tmp/runs/candidate \
  --format json \
  --output .tmp/diff.json \
  --output-fingerprint-ignore request_id,timestamp \
  --fail-on-changes
```

The ignore list applies to keys inside structured outputs; it is not a general ignore mechanism for
arbitrary record fields.

Trace and agent/tool trajectory gates have separate exit classes:

```bash
insidellms diff .tmp/runs/baseline .tmp/runs/candidate \
  --format json \
  --output .tmp/diff.json \
  --fail-on-trace-violations \
  --fail-on-trace-drift \
  --fail-on-trajectory-drift
```

## Troubleshooting

**`ci/data/harness_dataset.jsonl` is missing**

Generate from inside `ci/` as shown above. Creating `ci/harness.yaml` while running `init` from the
repository root puts the generated sample data in the wrong directory for that config path.

**The harness exited `0` but some records failed**

Add `runner.stop_on_error: true`, or deliberately inspect manifest status counts when partial runs
are acceptable.

**`diff` reported changes but CI passed**

A plain diff exits `0`. Add the appropriate `--fail-on-*` gate. Remember that
`--fail-on-changes` does not currently fail on improvements alone.

**The command exited `2` but no behavioural gate fired**

Read stderr. Parser usage errors also return `2`, while a successful diff gate writes a structured
JSON report when invoked with `--format json --output PATH`.

**The diff has zero common keys**

Check model, probe, and example identity. Renaming a model or changing the dataset can appear as a
large set of removals and additions instead of matched output changes.

**The reusable action ignored `ci/baseline`**

That is its current design: it regenerates ephemeral base and head runs. Use the approved-artefact
workflow when CI must compare against stored reviewed records.

## Verification checklist

- [ ] `insidellms harness ci/harness.yaml --dry-run` reports the expected plan.
- [ ] `runner.stop_on_error: true` is set, or partial-run status checks are explicit.
- [ ] The chosen workflow's baseline meaning is documented for reviewers.
- [ ] A plain local diff works before gate flags are enabled.
- [ ] Every `--output` example also sets `--format json`.
- [ ] The approved baseline can only change through an explicit reviewed process.
- [ ] Fork candidate execution receives no secrets and does not use `pull_request_target`.
- [ ] Uploaded artefacts have suitable content, access, and retention controls.

## What's next?

- [Determinism and CI](../Determinism-and-CI.md) — understand deterministic artefacts and limits
- [GitHub Action](https://github.com/dr-gareth-roberts/insideLLMs/blob/main/docs/GITHUB_ACTION.md) — action inputs, outputs, and baseline resolution
- [Tracing and Fingerprinting](../Tracing-and-Fingerprinting.md) — advanced trace gates
- [Troubleshooting](../guides/Troubleshooting.md) — common runtime and configuration failures
