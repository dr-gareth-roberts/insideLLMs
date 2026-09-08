# insideLLMs GitHub Action

Run deterministic base-vs-head behavioural diffing with read-only permissions.
Sticky PR comments use a separate trusted workflow. Until a published action
version has been verified, use a full reviewed commit SHA; the placeholder below
must be replaced before use. Development within this repository can use `uses: ./`.

## Quick Start

```yaml
name: Behavioural Diff

on:
  pull_request:
    branches: [main]

jobs:
  behavioural-diff:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          persist-credentials: false
      - uses: dr-gareth-roberts/insideLLMs@<reviewed-full-commit-sha>
        with:
          harness-config: ci/harness.yaml
```

## What It Does

1. Resolves baseline commit from `baseline-ref`, then `pull_request.base.sha`, then `GITHUB_BASE_REF`, then `main`.
2. Runs `insidellms harness` on baseline code and candidate code.
3. Checks execution status, counts, and completion in both persisted runs, including old baselines.
4. Runs `insidellms diff --format json --fail-on-any-difference` by default.
5. Fails the workflow on unhealthy runs or differences, retaining diagnostic artefacts.
6. Emits a small allowlisted report for a separate trusted PR-comment workflow.

## Inputs

- `harness-config` (default: `ci/harness.yaml`): path to harness config.
- `baseline-ref` (default: auto): explicit baseline branch/ref.
- `python-version` (default: `3.11`): runtime for action steps.
- `install-extras` (default: empty): extras for `pip install -e` (e.g., `dev,nlp`).
- `run-args` (default: empty): extra args forwarded to both harness runs.
- `diff-args` (default: empty): extra args forwarded to diff.
- `fail-on-changes` (default: `true`): include `--fail-on-any-difference`, including improvements and trace changes.
- `post-pr-comment` and `comment-on-forks` (default: `false`): deprecated; emit a migration warning when enabled but do not post comments.
- The old `github-token` input is removed. Do not pass write tokens to evaluation.

## Outputs

- `diff-json`: diff report or run-health error report path.
- `baseline-run-dir`: baseline run directory path.
- `candidate-run-dir`: candidate run directory path.
- `diff-exit-code`: `1` for unhealthy runs or invalid evidence; otherwise the diff gate exit code.
- `baseline-commit`: resolved baseline commit used for comparison.
- `is-fork-pr`: whether pull request head repo is a fork.
- `comment-status`: `disabled-inline-commenting`.
- `pr-report-json`: path to the small fixed-field report consumed by the trusted workflow.

## PR comments and fork pull requests

The evaluation workflow executes candidate code, including fork code, with
`contents: read`, no provider secrets, and checkout credential persistence disabled.
It must never use `pull_request_target` or a write-capable token.

Copy the paired [evaluation](../.github/workflows/diff-gate.yml) and
[comment](../.github/workflows/diff-gate-comment.yml) workflows to retain comments.
The trusted `workflow_run` workflow must exist on the default branch. It downloads
one size-limited report, validates fixed fields against the triggering PR and head
commit, and renders integer counts. It never extracts executable artefacts, checks
out candidate code, or runs candidate scripts. Stale or ambiguous PR reports are
skipped. See [CI setup](../ci/README.md) for migration details.

## Recommended Trigger Pattern

Use `pull_request` with `fetch-depth: 0` for base-vs-head execution. Select artefact
retention and access appropriate to your dataset: configuration credentials are
scrubbed, but model prompts and responses can still contain sensitive information.
