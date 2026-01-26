<div align="center">
  <img src="insidellms.jpg" alt="insideLLMs" width="720">
  <h1 align="center">insideLLMs</h1>
  <p align="center">
    <strong>Catch LLM behavioural regressions before they reach production</strong>
  </p>
  <p align="center">
    <a href="#why-insidellms">Why</a> &bull;
    <a href="#how-it-works">How It Works</a> &bull;
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

## Why insideLLMs

**Traditional LLM evaluation frameworks answer:** "What's the model's score on MMLU?"

**insideLLMs answers:** "Did my model's behaviour change between versions?"

When you're shipping LLM-powered products, you don't need leaderboard rankings. You need to know:
- Did prompt #47 start returning different advice?
- Will this model update break my users' workflows?
- Can I safely deploy this change?

insideLLMs provides **deterministic, diffable, CI-gateable behavioural testing** for LLMs.

### Built for Production Teams

- **Deterministic by design**: Same inputs produce byte-for-byte identical outputs
- **CI-native**: `insidellms diff --fail-on-changes` blocks bad deploys
- **Response-level granularity**: See exactly which prompts changed, not just aggregate metrics
- **Provider-agnostic**: OpenAI, Anthropic, local models (Ollama, llama.cpp), all through one interface

## How It Works

### 1. Define Behavioural Tests (Probes)

```python
from insideLLMs import LogicProbe, BiasProbe, SafetyProbe

# Test specific behaviours, not broad benchmarks
probes = [LogicProbe(), BiasProbe(), SafetyProbe()]
```

### 2. Run Across Models

```bash
insidellms harness config.yaml --run-dir ./baseline
```

Produces deterministic artefacts:
- `records.jsonl` - Every input/output pair (canonical)
- `summary.json` - Aggregated metrics
- `report.html` - Human-readable comparison

### 3. Detect Changes in CI

```bash
insidellms diff ./baseline ./candidate --fail-on-changes
```

Blocks the deploy if behaviour changed:
```
Changes detected:
  example_id: 47
  field: output
  baseline: "Consult a doctor for medical advice."
  candidate: "Here's what you should do..."
```

## Quickstart

### Install

```bash
git clone https://github.com/dr-gareth-roberts/insideLLMs.git
cd insideLLMs
pip install -e ".[all]"
```

### 5-Minute Test (No API Keys)

```bash
# Quick test with DummyModel
insidellms quicktest "What is 2 + 2?" --model dummy

# Run offline golden path
python examples/example_cli_golden_path.py
```

### First Real Comparison

```yaml
# harness.yaml
models:
  - type: openai
    args: {model_name: gpt-4o}
  - type: anthropic
    args: {model_name: claude-3-5-sonnet-20241022}

probes:
  - type: logic
  - type: bias

dataset:
  format: jsonl
  path: data/test.jsonl
```

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

insidellms harness harness.yaml --run-dir ./baseline
insidellms report ./baseline
```

### Add to CI

```yaml
# .github/workflows/behavioural-tests.yml
- name: Run candidate
  run: insidellms harness config.yaml --run-dir ./candidate

- name: Diff against baseline
  run: insidellms diff ./baseline ./candidate --fail-on-changes
```

## Key Features

### Deterministic Artefacts

- Run IDs are SHA-256 hashes of inputs (config + dataset)
- Timestamps derive from run IDs, not wall clocks
- JSON output has stable formatting (sorted keys, consistent separators)
- Result: `git diff` works on model behaviour

### Response-Level Granularity

`records.jsonl` preserves every input/output pair:
```jsonl
{"example_id": "47", "input": {...}, "output": "...", "status": "success"}
{"example_id": "48", "input": {...}, "output": "...", "status": "success"}
```

No more debugging aggregate metrics. See exactly what changed.

### Extensible Probes

```python
from insideLLMs.probes import Probe

class MedicalSafetyProbe(Probe):
    def run(self, model, data, **kwargs):
        response = model.generate(data["symptom_query"])
        return {
            "response": response,
            "has_disclaimer": "consult a doctor" in response.lower()
        }
```

Build domain-specific tests without forking the framework.

## Documentation

- **[Documentation Site](https://dr-gareth-roberts.github.io/insideLLMs/)** - Complete guides and reference
- **[Philosophy](https://dr-gareth-roberts.github.io/insideLLMs/Philosophy.html)** - Why insideLLMs exists
- **[Getting Started](https://dr-gareth-roberts.github.io/insideLLMs/getting-started/)** - Install and first run
- **[Tutorials](https://dr-gareth-roberts.github.io/insideLLMs/tutorials/)** - Bias testing, CI integration, custom probes
- **[API Reference](API_REFERENCE.md)** - Complete Python API
- **[Examples](examples/)** - Runnable code samples

## Use Cases

| Scenario | Solution |
|----------|----------|
| Model upgrade breaks production | Catch it in CI with `--fail-on-changes` |
| Need to compare GPT-4 vs Claude | Run harness, get side-by-side report |
| Detect bias in salary advice | Use BiasProbe with paired prompts |
| Test jailbreak resistance | Use SafetyProbe with attack patterns |
| Custom domain evaluation | Extend Probe base class |

## Comparison with Other Frameworks

| Framework | Focus | insideLLMs Difference |
|-----------|-------|----------------------|
| Eleuther lm-evaluation-harness | Benchmark scores | Behavioural regression detection |
| HELM | Holistic evaluation | CI-native, deterministic diffing |
| OpenAI Evals | Conversational tasks | Response-level granularity, provider-agnostic |

**insideLLMs is for teams shipping LLM products who need to know what changed, not just what scored well.**

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT. See [LICENSE](LICENSE).
