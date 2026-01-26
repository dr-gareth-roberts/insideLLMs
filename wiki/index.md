---
title: Home
nav_order: 1
description: Stop shipping LLM regressions. Deterministic behavioural testing for production teams.
---

# insideLLMs

**Stop shipping LLM regressions.** Deterministic behavioural testing that catches breaking changes before they reach production.

```mermaid
graph LR
    Dataset[Dataset] --> Runner[Runner]
    Model[Models] --> Runner
    Probe[Probes] --> Runner
    Runner --> Records[records.jsonl]
    Records --> Summary[summary.json]
    Records --> Report[report.html]
    Records --> Diff[diff.json]
```

---

## The Problem

You update your LLM. Prompt #47 now gives dangerous medical advice. Prompt #103 starts hallucinating. Your users notice before you do.

**Traditional eval frameworks can't help.** They tell you the model scored 87% on MMLU. They don't tell you *what changed*.

## The Solution

insideLLMs treats model behaviour like code: testable, diffable, gateable.

```bash
insidellms diff ./baseline ./candidate --fail-on-changes
```

If behaviour changed, the deploy blocks. Simple.

---

## Start Here

| Goal | Path | Time |
|------|------|------|
| **See it work** | [Quick Install](getting-started/Quick-Install.md) → [First Run](getting-started/First-Run.md) | 5 min |
| **Compare models** | [First Harness](getting-started/First-Harness.md) | 15 min |
| **Block regressions** | [CI Integration](tutorials/CI-Integration.md) | 30 min |
| **Understand the approach** | [Philosophy](Philosophy.md) | 10 min |

---

## Why Teams Choose insideLLMs

### Catch Regressions Before Production
Know exactly which prompts changed behaviour. No more debugging aggregate metrics.

### CI-Native Design
Built for `git diff` on model behaviour. Deterministic artefacts. Stable diffs. Automated gates.

### Response-Level Visibility
`records.jsonl` preserves every input/output pair. See what changed, not just that something changed.

### Provider-Agnostic
OpenAI, Anthropic, Cohere, Google, local models (Ollama, llama.cpp, vLLM). One interface.

---

## How It Works

**1. Define behavioural tests**
```yaml
probes:
  - type: logic      # Reasoning consistency
  - type: bias       # Fairness across demographics
  - type: safety     # Jailbreak resistance
```

**2. Run across models**
```bash
insidellms harness config.yaml --run-dir ./baseline
```

**3. Catch changes in CI**
```bash
insidellms diff ./baseline ./candidate --fail-on-changes
# Exit code 1 if behaviour changed
```

**Result:** Breaking changes blocked. Users protected.

---

## Documentation

| Section | Description |
|---------|-------------|
| [Philosophy](Philosophy.md) | Why insideLLMs exists and how it differs |
| [Getting Started](getting-started/index.md) | Install and run your first test |
| [Tutorials](tutorials/index.md) | Bias testing, CI integration, custom probes |
| [Concepts](concepts/index.md) | Models, probes, runners, determinism |
| [Advanced Features](advanced/index.md) | Pipeline, cost tracking, structured outputs |
| [Reference](reference/index.md) | Complete CLI and API documentation |
| [Guides](guides/index.md) | Caching, rate limiting, local models |
| [FAQ](FAQ.md) | Common questions and troubleshooting |

---

## What You Get That Others Don't

| Feature | Eleuther | HELM | OpenAI Evals | insideLLMs |
|---------|----------|------|--------------|------------|
| CI diff-gating | No | No | No | **Yes** |
| Deterministic artefacts | No | No | No | **Yes** |
| Response-level granularity | No | Partial | No | **Yes** |
| Pipeline middleware | No | No | No | **Yes** |
| Cost tracking & budgets | No | No | No | **Yes** |
| Structured output parsing | No | No | No | **Yes** |
| Agent evaluation | No | No | No | **Yes** |

**Not just benchmarks. Production infrastructure.**

---

## Community

- [GitHub Repository](https://github.com/dr-gareth-roberts/insideLLMs)
- [Report an Issue](https://github.com/dr-gareth-roberts/insideLLMs/issues)
- [Contributing Guide](CONTRIBUTING.md)
