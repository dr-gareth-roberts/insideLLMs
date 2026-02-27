---
title: CI Integration
parent: Tutorials
nav_order: 3
---

# CI Integration Tutorial

**Block regressions automatically.**

**Time:** 30 minutes
**Prerequisites:** Git, GitHub Actions

---

## Step 1: Create a Baseline

First, create a deterministic baseline run using `DummyModel`:

```bash
# Create a harness config for CI
mkdir -p ci
cat > ci/harness.yaml << 'EOF'
models:
  - type: dummy
    args:
      name: baseline_model

probes:
  - type: logic

dataset:
  format: inline
  items:
    - question: "What is 2 + 2?"
    - question: "Is the sky blue?"
    - question: "What comes next: 1, 2, 3, ?"
EOF

# Run the baseline
insidellms harness ci/harness.yaml --run-dir ci/baseline --overwrite --skip-report
```

## Step 2: Verify Determinism

Run again and diff to confirm identical outputs:

```bash
insidellms harness ci/harness.yaml --run-dir ci/candidate --overwrite --skip-report
insidellms diff ci/baseline ci/candidate
```

**Expected output:**

```
Comparing runs...
Baseline: ci/baseline
Candidate: ci/candidate

Changes: 0
Status: IDENTICAL
```

## Step 3: Commit the Baseline

```bash
git add ci/
git commit -m "Add CI baseline for behavioural testing"
```

## Step 4: Create GitHub Actions Workflow

Create `.github/workflows/behavioural-test.yml`:

```yaml
name: Behavioural Tests

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
```

This action runs the harness on the PR branch and the PR base ref, generates a deterministic
`diff.json`, and upserts a sticky pull-request comment with top regressions and changes.

## Step 5: Test the Workflow

Push to trigger the workflow:

```bash
git add .github/
git commit -m "Add behavioural testing workflow"
git push
```

The workflow should pass (no changes detected).

## Step 6: Simulate a Regression

To test that the CI catches changes, modify something that affects outputs:

```bash
# Modify the harness to use a different probe
cat > ci/harness.yaml << 'EOF'
models:
  - type: dummy
    args:
      name: baseline_model
      response: "CHANGED RESPONSE"  # This changes outputs!

probes:
  - type: logic

dataset:
  format: inline
  items:
    - question: "What is 2 + 2?"
    - question: "Is the sky blue?"
    - question: "What comes next: 1, 2, 3, ?"
EOF
```

Push this change — the CI will **fail** with a diff report.

---

## Advanced Options

### Ignore Specific Fields

Some fields are intentionally volatile. Ignore them:

```bash
insidellms diff ci/baseline ci/candidate \
  --output-fingerprint-ignore latency_ms,timestamps \
  --fail-on-changes
```

### Trace and Trajectory Gates

Use dedicated diff gates for trace contracts and agent/tool trajectories:

```bash
insidellms diff ci/baseline ci/candidate \
  --fail-on-trace-drift \
  --fail-on-trace-violations
```

```bash
insidellms diff ci/baseline ci/candidate \
  --fail-on-trajectory-drift
```

### Judge-Assisted Triage

Layer deterministic judge triage on top of the same core diff computation:

```bash
insidellms diff ci/baseline ci/candidate \
  --judge \
  --judge-policy balanced \
  --judge-limit 50 \
  --fail-on-trace-violations
```

### Update Baseline

When changes are intentional:

```bash
# Run new baseline
insidellms harness ci/harness.yaml --run-dir ci/baseline --overwrite --skip-report

# Commit the update
git add ci/baseline
git commit -m "Update behavioural baseline: [describe changes]"
```

---

## Complete Workflow Example

```yaml
name: Behavioural Tests

on:
  push:
    branches: [main]
  pull_request:

env:
  INSIDELLMS_RUN_ROOT: .tmp/runs

jobs:
  behavioural-diff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - run: pip install -e ".[all]"

      - name: Run candidate
        run: |
          insidellms harness ci/harness.yaml \
            --run-dir ${{ env.INSIDELLMS_RUN_ROOT }}/candidate \
            --overwrite --skip-report

      - name: Compare to baseline
        run: |
          insidellms diff \
            ci/baseline \
            ${{ env.INSIDELLMS_RUN_ROOT }}/candidate \
            --fail-on-changes \
            --output diff-report.json

      - name: Upload diff on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: behavioural-diff-${{ github.sha }}
          path: diff-report.json
          retention-days: 7
```

---

## Verification

 Baseline committed to repository
 GitHub Actions workflow created
 CI passes with no changes
 CI fails when outputs change

---

## What's Next?

- [Determinism and CI](../Determinism-and-CI.md) - Understand why this works
- [Tracing and Fingerprinting](../Tracing-and-Fingerprinting.md) - Advanced diff features
- [Troubleshooting](../guides/Troubleshooting.md) - Common CI issues
