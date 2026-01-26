## Golden Path (Offline, 5 minutes)

This walkthrough runs end-to-end without API keys (uses `DummyModel`) and produces
the same artefacts you would diff-gate in CI.

### 1) Install (minimal)

```bash
git clone https://github.com/dr-gareth-roberts/insideLLMs.git
cd insideLLMs
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Optional but recommended:

```bash
insidellms doctor
```

### 2) Run a deterministic harness (offline)

```bash
insidellms harness ci/harness.yaml --run-dir .tmp/runs/baseline --overwrite
insidellms report .tmp/runs/baseline
```

Artefacts:
- `.tmp/runs/baseline/manifest.json`
- `.tmp/runs/baseline/records.jsonl` (canonical)
- `.tmp/runs/baseline/summary.json`
- `.tmp/runs/baseline/report.html`

### 3) Diff-gate (baseline vs candidate)

Run the same harness again into a second directory:

```bash
insidellms harness ci/harness.yaml --run-dir .tmp/runs/candidate --overwrite
insidellms diff .tmp/runs/baseline .tmp/runs/candidate --fail-on-changes
```

### 4) Validate output schemas (optional)

```bash
insidellms schema validate --name ResultRecord --input .tmp/runs/baseline/records.jsonl
insidellms schema validate --name RunManifest --input .tmp/runs/baseline/manifest.json
```

