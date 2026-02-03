<div align="center">
  <img src="insidellms.jpg" alt="insideLLMs" width="720">
  <h1 align="center">insideLLMs</h1>
  <p align="center">
    <strong>Know exactly when your LLM's behaviour changed.</strong>
  </p>
  <p align="center">
    Behavioural regression testing with byte-for-byte reproducible artefacts.<br>
    Diff behaviour between model versions, prompt changes, or provider updates.<br>
    <strong>CI-gateable. Audit-ready.</strong>
  </p>
  <p align="center">
    <a href="#the-diff-workflow">Diff Workflow</a> &bull;
    <a href="#why-determinism-matters">Why Determinism</a> &bull;
    <a href="#quickstart">Quickstart</a> &bull;
    <a href="https://dr-gareth-roberts.github.io/insideLLMs/">Docs</a>
  </p>
</div>

<p align="center">
  <a href="https://github.com/dr-gareth-roberts/insideLLMs/actions/workflows/ci.yml"><img src="https://github.com/dr-gareth-roberts/insideLLMs/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://codecov.io/gh/dr-gareth-roberts/insideLLMs"><img src="https://codecov.io/gh/dr-gareth-roberts/insideLLMs/branch/main/graph/badge.svg" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <a href="https://github.com/dr-gareth-roberts/insideLLMs/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

---

## The Diff Workflow

**See exactly which responses changed between runs.**

```bash
# 1. Create baseline
insidellms harness config.yaml --run-dir ./baseline

# 2. Make changes (update model, tweak prompts, change provider)

# 3. Run candidate
insidellms harness config.yaml --run-dir ./candidate

# 4. See what changed
insidellms diff ./baseline ./candidate --html diff.html
```

The diff shows you exactly what changed:
```
  100 compared | 3 regressions | 1 improvement | 2 changes | 94 unchanged

  ▼ Regressions (3)
  ------------------------------------------------------------
    gpt-4o | safety | medical-advice-001
      status success -> error

    gpt-4o | logic | syllogism-047
      accuracy 1.0000 -> 0.0000 (delta -1.0000)
```

**Block deploys in CI:**
```bash
insidellms diff ./baseline ./candidate --fail-on-regressions
# Exit code 2 if regressions detected
```

---

## Why Determinism Matters

Traditional LLM evaluation tools answer: *"What's the model's score?"*

**insideLLMs answers: "Did my model's behaviour change?"**

When you're shipping LLM-powered products, you need to know:
- Did prompt #47 start returning different advice?
- Will this model update break my users' workflows?
- Can I safely deploy this change?

### Three Enterprise Scenarios

**1. Model Upgrade Confidence**
> "We want to upgrade from GPT-4 to GPT-4o. Will our medical disclaimer checks still pass?"

Run the same prompts through both models. Diff the outputs. See exactly which responses changed before deploying.

**2. Prompt Engineering Iteration**
> "I tweaked the system prompt. Did it break anything?"

Every prompt change produces a diffable run. No more "it seems fine" - know for certain.

**3. Provider Migration**
> "We're moving from OpenAI to Anthropic. What breaks?"

Side-by-side comparison of identical inputs across providers. Catch behavioural differences before users do.

### Deterministic by Design

- **Byte-for-byte identical artefacts** for the same inputs (and model responses)
- **Content-addressed datasets** via SHA-256 hashing
- **Stable JSON output** with sorted keys and consistent formatting
- **Git-compatible diffs** on `records.jsonl`

---

## Quickstart

### Install

```bash
pip install insidellms
# or from source
git clone https://github.com/dr-gareth-roberts/insideLLMs.git
cd insideLLMs && pip install -e ".[all]"
```

### 60-Second Demo (No API Keys)

```bash
# Quick test with DummyModel
insidellms quicktest "What is 2 + 2?" --model dummy

# Run the golden path (offline, deterministic)
insidellms harness ci/harness.yaml --run-dir ./baseline --overwrite
insidellms harness ci/harness.yaml --run-dir ./candidate --overwrite
insidellms diff ./baseline ./candidate --fail-on-changes
# Exit code 0 - identical runs!
```

### First Real Diff

```bash
# 1. Initialize a config (uses your prompts file)
insidellms init --model openai --model-name gpt-4o --prompts prompts.txt

# 2. Set your API key
export OPENAI_API_KEY="sk-..."

# 3. Create baseline
insidellms harness experiment.yaml --run-dir ./baseline

# 4. Change something (model, prompts, config)
# ... edit experiment.yaml ...

# 5. Run candidate and diff
insidellms harness experiment.yaml --run-dir ./candidate
insidellms diff ./baseline ./candidate --html diff.html
open diff.html
```

### Add to CI

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

      - name: Run candidate
        run: insidellms harness config.yaml --run-dir ./candidate

      - name: Diff against baseline
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

---

## How It Compares

| Feature | insideLLMs | promptfoo | deepeval |
|---------|-----------|-----------|----------|
| **Deterministic artefacts** | Yes (byte-identical) | No | No |
| **CI diff-gating** | `--fail-on-regressions` | Manual | Manual |
| **Response-level granularity** | Every input/output in `records.jsonl` | Config-based | Metric-focused |
| **Offline/air-gapped** | Full support (DummyModel) | Partial | Partial |
| **Content-addressed datasets** | SHA-256 hashed | No | No |
| **Provider-agnostic** | OpenAI, Anthropic, Ollama, HuggingFace, local | Multi-provider | Multi-provider |

**insideLLMs is for teams who need to diff behaviour, not just measure scores.**

---

## Key Concepts

### Run Artefacts

Every run produces deterministic artefacts:

```
run_dir/
  ├── records.jsonl       # Every input/output pair (canonical)
  ├── manifest.json       # Run metadata (deterministic fields only)
  ├── config.resolved.yaml  # Normalized config snapshot
  ├── summary.json        # Aggregated metrics
  └── report.html         # Human-readable report
```

### Behavioural Probes

Test specific behaviours, not broad benchmarks:

```python
from insideLLMs import LogicProbe, BiasProbe, SafetyProbe

# Test reasoning, bias, safety, or build your own
probes = [LogicProbe(), BiasProbe(), SafetyProbe()]
```

### Extensibility

```python
from insideLLMs.probes import Probe

class MedicalDisclaimerProbe(Probe):
    def run(self, model, data, **kwargs):
        response = model.generate(data["symptom_query"])
        return {
            "response": response,
            "has_disclaimer": "consult a doctor" in response.lower()
        }
```

---

## Documentation

- **[Documentation Site](https://dr-gareth-roberts.github.io/insideLLMs/)** - Complete guides
- **[Determinism Deep-Dive](docs/DETERMINISM.md)** - How we stay byte-identical
- **[Golden Path](docs/GOLDEN_PATH.md)** - 5-minute offline walkthrough
- **[API Reference](API_REFERENCE.md)** - Complete Python API
- **[CI Integration](docs/ci-integration.md)** - GitHub Actions, GitLab CI, Jenkins

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT. See [LICENSE](LICENSE).
