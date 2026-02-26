# insideLLMs GitHub Action (`dr-gareth-roberts/insideLLMs@v1`)

Use the first-class action to run deterministic base-vs-head behavioral diffing and post pull
request comments automatically.

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
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: dr-gareth-roberts/insideLLMs@v1
        with:
          harness-config: ci/harness.yaml
          comment-on-forks: "false"
```

## What It Does

1. Resolves baseline commit from `baseline-ref`, then `pull_request.base.sha`, then `GITHUB_BASE_REF`, then `main`.
2. Runs `insidellms harness` on baseline code and candidate code.
3. Runs `insidellms diff --format json`.
4. Upserts a sticky PR comment with summary counts and top changed records.
5. Fails the workflow when diff gating returns non-zero.

## Inputs

- `harness-config` (default: `ci/harness.yaml`): path to harness config.
- `baseline-ref` (default: auto): explicit baseline branch/ref.
- `python-version` (default: `3.11`): runtime for action steps.
- `install-extras` (default: empty): extras for `pip install -e` (e.g., `dev,nlp`).
- `run-args` (default: empty): extra args forwarded to both harness runs.
- `diff-args` (default: empty): extra args forwarded to diff.
- `fail-on-changes` (default: `true`): include `--fail-on-changes`.
- `post-pr-comment` (default: `true`): create/update PR comment.
- `comment-on-forks` (default: `false`): skip PR comments for forked PRs unless explicitly enabled.
- `github-token` (default: `${{ github.token }}`): token for comment permissions.

## Outputs

- `diff-json`: generated `diff.json` path.
- `baseline-run-dir`: baseline run directory path.
- `candidate-run-dir`: candidate run directory path.
- `diff-exit-code`: exit code returned by `insidellms diff`.
- `baseline-commit`: resolved baseline commit used for comparison.
- `is-fork-pr`: whether pull request head repo is a fork.
- `comment-status`: comment result (`created`, `updated`, `skipped-fork`, `skipped-permissions`, etc).

## Fork Pull Requests

- By default, forked PRs run the diff gate but skip PR comments (`comment-on-forks: false`).
- If you enable `comment-on-forks: true`, the action still degrades gracefully when token permissions are insufficient (for example, `Resource not accessible by integration`).
- A ready-to-copy workflow exists at `.github/workflows/behavioural-diff-example.yml`.

## Recommended Trigger Pattern

Use `pull_request` with `fetch-depth: 0` for safe base-vs-head execution and deterministic diffs.
This avoids running untrusted fork code in a privileged `pull_request_target` context.
