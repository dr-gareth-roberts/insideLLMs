# CI Integration Guide

This guide shows how to integrate insideLLMs behavioural regression testing into your CI/CD pipeline.

## Overview

The core workflow is:
1. **Baseline**: Store known-good evaluation results (typically in version control or CI artifacts)
2. **Candidate**: Run evaluations on the current branch/PR
3. **Diff**: Compare candidate against baseline, fail if regressions detected

## GitHub Actions

### Quick Start

```yaml
# .github/workflows/llm-regression.yml
name: LLM Regression Tests
on: [pull_request]

jobs:
  regression-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install insideLLMs
        run: pip install insidellms

      - name: Run candidate evaluation
        run: insidellms harness config.yaml --run-dir ./candidate
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Compare against baseline
        run: |
          insidellms diff ./baseline ./candidate \
            --html diff.html \
            --fail-on-regressions

      - name: Upload diff report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: diff-report
          path: diff.html
```

### Using the insideLLMs Diff Action

For more features (PR comments, structured outputs), use the built-in action:

```yaml
name: LLM Regression Tests
on: [pull_request]

jobs:
  regression-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run candidate evaluation
        run: |
          pip install insidellms
          insidellms harness config.yaml --run-dir ./candidate
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Diff against baseline
        uses: ./.github/actions/diff
        with:
          baseline: ./baseline
          candidate: ./candidate
          fail-on-regressions: true
          comment-on-pr: true

      - name: Upload diff report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: diff-report
          path: diff-report.html
```

### Action Inputs

| Input | Description | Default |
|-------|-------------|---------|
| `baseline` | Path to baseline run directory | (required) |
| `candidate` | Path to candidate run directory | (required) |
| `fail-on-regressions` | Fail if regressions detected | `true` |
| `fail-on-changes` | Fail if any changes detected (stricter) | `false` |
| `html-report` | Generate HTML diff report | `true` |
| `html-report-path` | Path for HTML report | `diff-report.html` |
| `comment-on-pr` | Add summary comment on PR | `true` |

### Action Outputs

| Output | Description |
|--------|-------------|
| `regressions` | Number of regressions detected |
| `improvements` | Number of improvements detected |
| `changes` | Number of other changes |
| `has-regressions` | `true` if regressions detected |
| `html-report-path` | Path to HTML report |

### Managing Baselines

#### Option 1: Commit baseline to repo

Good for small datasets or when baseline should be versioned:

```yaml
# baseline/ directory is committed to the repo
- name: Diff against committed baseline
  run: insidellms diff ./baseline ./candidate --fail-on-regressions
```

#### Option 2: Cache baseline as artifact

Good for larger datasets or frequently updating baselines:

```yaml
- name: Download baseline artifact
  uses: dawidd6/action-download-artifact@v3
  with:
    name: baseline-${{ github.base_ref }}
    path: ./baseline
    if_no_artifact_found: warn

- name: Create baseline if missing
  if: ${{ hashFiles('./baseline/records.jsonl') == '' }}
  run: insidellms harness config.yaml --run-dir ./baseline

- name: Run candidate and diff
  run: |
    insidellms harness config.yaml --run-dir ./candidate
    insidellms diff ./baseline ./candidate --fail-on-regressions

- name: Upload new baseline on main merge
  if: github.ref == 'refs/heads/main'
  uses: actions/upload-artifact@v4
  with:
    name: baseline-main
    path: ./baseline
```

#### Option 3: S3/GCS storage

For enterprise teams with artifact storage:

```yaml
- name: Download baseline from S3
  run: aws s3 sync s3://my-bucket/baselines/main ./baseline

- name: Run evaluation and diff
  run: |
    insidellms harness config.yaml --run-dir ./candidate
    insidellms diff ./baseline ./candidate --fail-on-regressions

- name: Upload new baseline on main
  if: github.ref == 'refs/heads/main'
  run: aws s3 sync ./candidate s3://my-bucket/baselines/main
```

---

## GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test
  - report

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

llm-regression:
  stage: test
  image: python:3.11
  cache:
    paths:
      - .cache/pip
  before_script:
    - pip install insidellms
  script:
    - insidellms harness config.yaml --run-dir ./candidate
    - insidellms diff ./baseline ./candidate --html diff.html --fail-on-regressions
  artifacts:
    paths:
      - diff.html
      - candidate/
    when: always
    reports:
      junit: candidate/summary.xml  # If you generate JUnit output
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
```

### GitLab with baseline artifacts

```yaml
fetch-baseline:
  stage: .pre
  script:
    - 'curl --header "PRIVATE-TOKEN: $CI_JOB_TOKEN" -o baseline.tar.gz "$CI_API_V4_URL/projects/$CI_PROJECT_ID/jobs/artifacts/main/download?job=save-baseline"'
    - tar -xzf baseline.tar.gz || mkdir -p baseline
  artifacts:
    paths:
      - baseline/

llm-regression:
  stage: test
  needs: [fetch-baseline]
  script:
    - pip install insidellms
    - insidellms harness config.yaml --run-dir ./candidate
    - insidellms diff ./baseline ./candidate --html diff.html --fail-on-regressions
  artifacts:
    paths:
      - diff.html
      - candidate/
    when: always

save-baseline:
  stage: report
  script:
    - tar -czf baseline.tar.gz candidate/
  artifacts:
    paths:
      - baseline.tar.gz
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
```

---

## Jenkins

### Jenkinsfile (Declarative)

```groovy
pipeline {
    agent {
        docker {
            image 'python:3.11'
        }
    }

    environment {
        OPENAI_API_KEY = credentials('openai-api-key')
    }

    stages {
        stage('Setup') {
            steps {
                sh 'pip install insidellms'
            }
        }

        stage('Run Evaluation') {
            steps {
                sh 'insidellms harness config.yaml --run-dir ./candidate'
            }
        }

        stage('Diff') {
            steps {
                script {
                    def exitCode = sh(
                        script: '''
                            insidellms diff ./baseline ./candidate \
                                --html diff.html \
                                --format json \
                                --output diff.json \
                                --fail-on-regressions
                        ''',
                        returnStatus: true
                    )

                    if (exitCode == 2) {
                        unstable('Regressions detected')
                    } else if (exitCode != 0) {
                        error('Diff failed')
                    }
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'diff.html, diff.json, candidate/**', fingerprint: true
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: '.',
                reportFiles: 'diff.html',
                reportName: 'LLM Diff Report'
            ])
        }
    }
}
```

### Jenkinsfile (Scripted)

```groovy
node {
    stage('Checkout') {
        checkout scm
    }

    docker.image('python:3.11').inside {
        stage('Setup') {
            sh 'pip install insidellms'
        }

        stage('Evaluate') {
            withCredentials([string(credentialsId: 'openai-api-key', variable: 'OPENAI_API_KEY')]) {
                sh 'insidellms harness config.yaml --run-dir ./candidate'
            }
        }

        stage('Diff') {
            def result = sh(
                script: 'insidellms diff ./baseline ./candidate --html diff.html --fail-on-regressions',
                returnStatus: true
            )

            archiveArtifacts 'diff.html'

            if (result == 2) {
                currentBuild.result = 'UNSTABLE'
                echo 'Regressions detected!'
            }
        }
    }
}
```

---

## CircleCI

```yaml
# .circleci/config.yml
version: 2.1

jobs:
  llm-regression:
    docker:
      - image: cimg/python:3.11
    steps:
      - checkout
      - restore_cache:
          keys:
            - pip-{{ checksum "requirements.txt" }}
      - run:
          name: Install dependencies
          command: pip install insidellms
      - save_cache:
          paths:
            - ~/.cache/pip
          key: pip-{{ checksum "requirements.txt" }}
      - run:
          name: Run evaluation
          command: insidellms harness config.yaml --run-dir ./candidate
      - run:
          name: Diff against baseline
          command: |
            insidellms diff ./baseline ./candidate \
              --html diff.html \
              --fail-on-regressions || echo "export REGRESSIONS_FOUND=true" >> $BASH_ENV
      - store_artifacts:
          path: diff.html
          destination: diff-report
      - store_artifacts:
          path: candidate
          destination: candidate-run
      - run:
          name: Check for regressions
          command: |
            if [ "$REGRESSIONS_FOUND" = "true" ]; then
              echo "Regressions detected!"
              exit 1
            fi

workflows:
  version: 2
  test:
    jobs:
      - llm-regression
```

---

## Azure Pipelines

```yaml
# azure-pipelines.yml
trigger:
  - main

pr:
  - main

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.11'
    displayName: 'Use Python 3.11'

  - script: pip install insidellms
    displayName: 'Install insideLLMs'

  - script: insidellms harness config.yaml --run-dir ./candidate
    displayName: 'Run evaluation'
    env:
      OPENAI_API_KEY: $(OPENAI_API_KEY)

  - script: |
      insidellms diff ./baseline ./candidate \
        --html $(Build.ArtifactStagingDirectory)/diff.html \
        --fail-on-regressions
    displayName: 'Diff against baseline'
    continueOnError: true

  - task: PublishBuildArtifacts@1
    inputs:
      pathtoPublish: '$(Build.ArtifactStagingDirectory)/diff.html'
      artifactName: 'diff-report'
    condition: always()
    displayName: 'Publish diff report'
```

---

## Best Practices

### 1. Use deterministic sampling

Ensure your model configurations use deterministic settings when possible:

```yaml
# config.yaml
model:
  type: openai
  args:
    model_name: gpt-4o
    temperature: 0
    seed: 42  # OpenAI supports seed for determinism
```

### 2. Cache API responses in CI

For expensive evaluations, use caching:

```yaml
- name: Cache API responses
  uses: actions/cache@v4
  with:
    path: ~/.insidellms/cache
    key: llm-cache-${{ hashFiles('config.yaml', 'data/**') }}
```

### 3. Gate critical probes differently

Use different failure thresholds for different probe types:

```yaml
- name: Safety probes (strict)
  run: insidellms diff ./baseline ./candidate --fail-on-changes
  env:
    PROBE_FILTER: safety

- name: Other probes (allow improvements)
  run: insidellms diff ./baseline ./candidate --fail-on-regressions
```

### 4. Generate JUnit reports for CI dashboards

```yaml
- name: Run with JUnit output
  run: |
    insidellms harness config.yaml --run-dir ./candidate
    # Convert summary to JUnit XML for CI integration
    python -c "
    import json
    from junit_xml import TestSuite, TestCase
    with open('candidate/summary.json') as f:
        summary = json.load(f)
    # ... generate test cases ...
    "
```

### 5. Notify on regressions

```yaml
- name: Slack notification on regression
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    payload: |
      {
        "text": "LLM regression detected in ${{ github.repository }}",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "*LLM Regression Detected*\nRepository: ${{ github.repository }}\nPR: #${{ github.event.pull_request.number }}"
            }
          }
        ]
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

---

## Troubleshooting

### "records.jsonl not found"

Ensure the baseline directory exists and contains valid run artifacts:

```bash
ls -la ./baseline/
# Should contain: records.jsonl, manifest.json, config.resolved.yaml
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success, no issues |
| 1 | General error (missing files, invalid config) |
| 2 | Regressions or changes detected (with `--fail-on-*`) |
| 3 | Trace violations increased (with `--fail-on-trace-violations`) |
| 4 | Trace drift detected (with `--fail-on-trace-drift`) |

### Non-deterministic diffs

If diffs show changes when they shouldn't:

1. Check model sampling settings (temperature, seed)
2. Verify dataset hasn't changed (check `dataset_hash` in manifest)
3. Use `--output-fingerprint-ignore` for expected-volatile fields

```bash
insidellms diff ./baseline ./candidate \
  --output-fingerprint-ignore timestamp,request_id
```
