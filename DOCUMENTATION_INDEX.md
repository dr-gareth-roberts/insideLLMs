# insideLLMs Documentation Index

Welcome to the insideLLMs documentation. Start with the Docs Site for guides and then use this
index for local references.

[Docs Site](https://dr-gareth-roberts.github.io/insideLLMs/)

```mermaid
flowchart LR
  W[Docs Site] --> R[README]
  R --> Q[Quick Reference]
  R --> A[Architecture]
  R --> API[API Reference]
  Q --> Examples[Examples]
  A --> Schemas[Schemas]
```

---

## Documentation Files

### [ARCHITECTURE.md](ARCHITECTURE.md)
**Architecture and execution flow diagrams** for the core runtime.

**Contents:**
- High-level component map
- ProbeRunner and config execution flows
- Benchmarking flow
- Extension points and supporting subsystems

**Best for:** Understanding how modules interact, onboarding contributors, system-level reasoning

---

### [API_REFERENCE.md](API_REFERENCE.md) (living reference)
**Comprehensive API documentation** covering all public interfaces.

**Contents:**
- **Core Model Classes** - Base `Model` class and implementations (`OpenAIModel`, `AnthropicModel`, `HuggingFaceModel`, `DummyModel`)
- **Core Probe Classes** - Base `Probe` class and implementations (`LogicProbe`, `BiasProbe`, `AttackProbe`, `FactualityProbe`)
- **Runner and Orchestration** - `ProbeRunner`, `AsyncProbeRunner`, `ModelBenchmark`, configuration-based execution
- **Diffing Engine** - `insideLLMs.diffing` public facade, `DiffGatePolicy`, judge composition, interactive snapshot helpers
- **Shadow Capture** - `insideLLMs.shadow.fastapi` middleware for production traffic capture
- **Core CLI Surfaces** - `harness`, `diff`, `doctor`, and `init` runtime-aligned options
- **Registry System** - `Registry` class and global registries (`model_registry`, `probe_registry`, `dataset_registry`)
- **Dataset Utilities** - Functions for loading CSV, JSONL, and HuggingFace datasets
- **NLP Utilities** - 100+ text processing functions (cleaning, tokenization, similarity, metrics, etc.)
- **Type Definitions** - All data classes and enums used throughout the library

**Best for:** Detailed reference, understanding all parameters and return types, exploring advanced features

---

### [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (6.6 KB)
**Quick reference guide** for the most commonly used APIs.

**Contents:**
- Quick start examples
- Common usage patterns
- Code snippets for frequent tasks
- Configuration examples
- Cheat sheet format

**Best for:** Quick lookups, getting started, common tasks, copy-paste examples

---

### [README.md](README.md)
**Project overview and getting started guide.**

**Contents:**
- Installation instructions
- Feature overview
- Basic usage examples
- Project structure
- Contributing guidelines

**Best for:** First-time users, understanding what the library does, installation

---

### [docs/GITHUB_ACTION.md](docs/GITHUB_ACTION.md)
**First-class GitHub Action reference** for deterministic base-vs-head harness diffing and PR comments.

**Contents:**
- Action inputs/outputs
- Forked PR edge-case behavior (`comment-on-forks`)
- Reusable workflow snippets

**Best for:** CI maintainers, release gating in GitHub PR workflows

---

### [extensions/vscode-insidellms/README.md](extensions/vscode-insidellms/README.md)
**VS Code / Cursor extension scaffold** for CodeLens-driven probe execution.

**Contents:**
- CodeLens behavior
- Extension settings
- Build instructions

**Best for:** Editor-native evaluation workflows and rapid local feedback

---

### [wiki/guides/Production-Shadow-Capture.md](wiki/guides/Production-Shadow-Capture.md)
**Long-form guide** for sampling real production traffic into canonical `records.jsonl` safely.

**Best for:** Converting production behavior into replayable regression suites

---

### [wiki/guides/IDE-Extension-Workflow.md](wiki/guides/IDE-Extension-Workflow.md)
**Long-form guide** for VS Code/Cursor extension setup, local feedback loop, and roadmap.

**Best for:** Editor-native prompt/probe iteration workflows

---

### [docs/AUDITS/2026-02-26-docs-audit.md](docs/AUDITS/2026-02-26-docs-audit.md)
**Phase 1 documentation audit report** covering high-traffic docs and CLI parity checks.

**Best for:** Tracking docs quality improvements, identifying remaining audit backlog

---

### [docs/GOLDEN_PATH.md](docs/GOLDEN_PATH.md)
**A 5-minute offline, end-to-end workflow** (DummyModel → records → report → diff) suitable for CI gating.

**Best for:** First-time users, CI diff-gating setup, verifying installation without API keys

---

### [ci/README.md](ci/README.md)
**How to extend the CI harness** with additional low-cost probes while preserving offline determinism.

**Best for:** Adding probes to CI diff-gating, understanding harness.yaml and harness_dataset.jsonl

---

### [docs/PLUGINS.md](docs/PLUGINS.md)
**How to extend insideLLMs without forking** using Python entry points.

**Best for:** Third-party integrations, adding private models/probes, sharing extensions

---

### [docs/architecture/API_TIERS.md](docs/architecture/API_TIERS.md)
**API compatibility tier map** (Tier 1 stable, Tier 2 evolving, Tier 3 internal)
for CLI, runtime, and extension surfaces.

**Best for:** PR review, release planning, avoiding accidental internal coupling

---

### [docs/architecture/PLUGIN_SDK_RFC.md](docs/architecture/PLUGIN_SDK_RFC.md)
**Draft proposal** for formal plugin SDK metadata (capabilities, compatibility,
determinism declarations) on top of current entry-point registration.

**Best for:** Plugin ecosystem planning, extension governance, CI policy discussions

---

### [docs/STABILITY.md](docs/STABILITY.md)
**Versioning and stability policy** for CLI, schemas, and determinism guarantees.

**Best for:** Contributors, CI users, reviewing breaking changes

---

### [docs/STABILITY_MATRIX.md](docs/STABILITY_MATRIX.md)
**Surface-by-surface compatibility contract matrix** (Stable / Experimental / Internal)
for CLI, artifacts, schemas, and extension points.

**Best for:** PR review, release planning, deprecation decisions, contract governance

---

### [docs/DETERMINISM.md](docs/DETERMINISM.md)
**Deterministic artifact guarantees** for CI diff-gating.

**Best for:** Reproducibility expectations, deterministic controls, local verification workflow

---

### [docs/ARTIFACT_CONTRACT.md](docs/ARTIFACT_CONTRACT.md)
**Field-level contract for canonical run artifacts** (`records.jsonl`, `manifest.json`, `summary.json`, `report.html`).

**Best for:** Contributor implementation checks, artifact compatibility reviews, schema-change impact

---

### [docs/CRYPTO_DIGEST_CONTRACT.md](docs/CRYPTO_DIGEST_CONTRACT.md)
**Digest descriptor contract** for external verifiers: `purpose`, `algo`, `canon_version` fields and Merkle root descriptors.

**Best for:** Verifying attestations, interpreting provenance digests, building external verifier tooling

---

### [docs/IMPORT_PATHS.md](docs/IMPORT_PATHS.md)
**Import path migration matrix** for canonical vs deprecated vs shim paths.

**Best for:** Updating imports after module moves, identifying canonical locations, avoiding internal modules

---

### [wiki/Experiment-Tracking.md](wiki/Experiment-Tracking.md)
**Experiment tracking backends** for logging metrics/params/artifacts (local, W&B, MLflow, TensorBoard).

**Best for:** Logging runs to dashboards, correlating metadata outside deterministic artifacts

---

### [wiki/Performance-and-Caching.md](wiki/Performance-and-Caching.md)
**Performance knobs and caching overview** (async, concurrency, rate limiting, cache layers).

**Best for:** Speeding up runs, reducing cost, understanding caching tradeoffs

---

### [wiki/Tracing-and-Fingerprinting.md](wiki/Tracing-and-Fingerprinting.md)
**Trace-aware diffing and fingerprinting** for CI gating and drift detection.

**Best for:** Understanding `insidellms diff` trace flags and structured-output fingerprints

---

### [wiki/advanced/Verifiable-Evaluation.md](wiki/advanced/Verifiable-Evaluation.md)
**Provenance-oriented workflows** for attestations, signature verification, SCITT receipts, and OCI publishing.

**Best for:** Security/compliance-aligned evaluation flows and end-to-end assurance quickstarts

---

## Quick Navigation

### I want to...

#### **Get started with the library**
→ Start with [README.md](README.md), then [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

#### **Use a specific model (OpenAI, Anthropic, HuggingFace)**
→ [API_REFERENCE.md - Core Model Classes](API_REFERENCE.md#core-model-classes)

#### **Test a model for bias, logic, or security**
→ [API_REFERENCE.md - Core Probe Classes](API_REFERENCE.md#core-probe-classes)

#### **Run benchmarks comparing multiple models**
→ [API_REFERENCE.md - ModelBenchmark](API_REFERENCE.md#modelbenchmark)

#### **Process text (clean, tokenize, analyze)**
→ [API_REFERENCE.md - NLP Utilities](API_REFERENCE.md#nlp-utilities)

#### **Load datasets from files**
→ [API_REFERENCE.md - Dataset Utilities](API_REFERENCE.md#dataset-utilities)

#### **Run experiments from configuration files**
→ [API_REFERENCE.md - Configuration-Based Execution](API_REFERENCE.md#configuration-based-execution)

#### **Run a cross-model behavioural harness**
→ [README.md](README.md#behavioural-harness), [Docs Site](https://dr-gareth-roberts.github.io/insideLLMs/)

#### **Capture and replay production traffic safely**
→ [wiki/guides/Production-Shadow-Capture.md](wiki/guides/Production-Shadow-Capture.md), [API_REFERENCE.md - Production Shadow Capture](API_REFERENCE.md#production-shadow-capture)

#### **Track runs in W&B / MLflow / TensorBoard**
→ [wiki/Experiment-Tracking.md](wiki/Experiment-Tracking.md)

#### **Speed up runs with caching / concurrency**
→ [wiki/Performance-and-Caching.md](wiki/Performance-and-Caching.md)

#### **Use trace-aware diffing**
→ [wiki/Tracing-and-Fingerprinting.md](wiki/Tracing-and-Fingerprinting.md)

#### **Use the VS Code/Cursor extension workflow**
→ [wiki/guides/IDE-Extension-Workflow.md](wiki/guides/IDE-Extension-Workflow.md), [extensions/vscode-insidellms/README.md](extensions/vscode-insidellms/README.md)

#### **Run verifiable-evaluation workflows (attest/sign/verify)**
→ [wiki/advanced/Verifiable-Evaluation.md](wiki/advanced/Verifiable-Evaluation.md), [API_REFERENCE.md - CLI Verifiable Evaluation Commands](API_REFERENCE.md#cli-verifiable-evaluation-commands)

#### **Validate JSON/JSONL artifacts against schemas**
→ [README.md - Schema Validation Commands](README.md#schema-validation-commands), [wiki/reference/CLI.md](wiki/reference/CLI.md#schema)

#### **Find a quick code example**
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

#### **Understand the architecture and runtime flows**
→ [ARCHITECTURE.md](ARCHITECTURE.md)

#### **Understand the type system**
→ [API_REFERENCE.md - Type Definitions](API_REFERENCE.md#type-definitions)

---

## Documentation by Topic

### Models

| Topic | Document | Section |
|-------|----------|---------|
| Model overview | API_REFERENCE.md | [Core Model Classes](API_REFERENCE.md#core-model-classes) |
| OpenAI models | API_REFERENCE.md | [OpenAIModel](API_REFERENCE.md#openaimodel) |
| Anthropic models | API_REFERENCE.md | [AnthropicModel](API_REFERENCE.md#anthropicmodel) |
| HuggingFace models | API_REFERENCE.md | [HuggingFaceModel](API_REFERENCE.md#huggingfacemodel) |
| Testing models | API_REFERENCE.md | [DummyModel](API_REFERENCE.md#dummymodel) |
| Quick examples | QUICK_REFERENCE.md | [Models](QUICK_REFERENCE.md#models) |

### Probes

| Topic | Document | Section |
|-------|----------|---------|
| Probe overview | API_REFERENCE.md | [Core Probe Classes](API_REFERENCE.md#core-probe-classes) |
| Logic testing | API_REFERENCE.md | [LogicProbe](API_REFERENCE.md#logicprobe) |
| Bias detection | API_REFERENCE.md | [BiasProbe](API_REFERENCE.md#biasprobe) |
| Security testing | API_REFERENCE.md | [AttackProbe](API_REFERENCE.md#attackprobe) |
| Factuality testing | API_REFERENCE.md | [FactualityProbe](API_REFERENCE.md#factualityprobe) |
| Quick examples | QUICK_REFERENCE.md | [Probes](QUICK_REFERENCE.md#probes) |

### Execution

| Topic | Document | Section |
|-------|----------|---------|
| Running probes | API_REFERENCE.md | [ProbeRunner](API_REFERENCE.md#proberunner) |
| Async execution | API_REFERENCE.md | [AsyncProbeRunner](API_REFERENCE.md#asyncproberunner) |
| Benchmarking | API_REFERENCE.md | [ModelBenchmark](API_REFERENCE.md#modelbenchmark) |
| Config files | API_REFERENCE.md | [Configuration-Based Execution](API_REFERENCE.md#configuration-based-execution) |
| Quick examples | QUICK_REFERENCE.md | [Runners](QUICK_REFERENCE.md#runners) |

### Verifiable Evaluation

| Topic | Document | Section |
|-------|----------|---------|
| Determinism guarantees | docs/DETERMINISM.md | [Determinism controls](docs/DETERMINISM.md#determinism-controls-strict-mode) |
| Artifact field contract | docs/ARTIFACT_CONTRACT.md | [records + manifest expectations](docs/ARTIFACT_CONTRACT.md) |
| Digest descriptor contract | docs/CRYPTO_DIGEST_CONTRACT.md | [purpose, algo, canon_version](docs/CRYPTO_DIGEST_CONTRACT.md) |
| Provenance quickstart | wiki/advanced/Verifiable-Evaluation.md | [Quickstart](wiki/advanced/Verifiable-Evaluation.md#quickstart-attest-sign-verify) |
| CLI command reference | wiki/reference/CLI.md | [`attest` / `sign` / `verify-signatures`](wiki/reference/CLI.md#attest) |

### Data Management

| Topic | Document | Section |
|-------|----------|---------|
| Registry system | API_REFERENCE.md | [Registry System](API_REFERENCE.md#registry-system) |
| Loading datasets | API_REFERENCE.md | [Dataset Utilities](API_REFERENCE.md#dataset-utilities) |
| Quick examples | QUICK_REFERENCE.md | [Registry](QUICK_REFERENCE.md#registry) |

### NLP Utilities

| Topic | Document | Section |
|-------|----------|---------|
| Text cleaning | API_REFERENCE.md | [Text Cleaning](API_REFERENCE.md#text-cleaning-and-normalization) |
| Tokenization | API_REFERENCE.md | [Tokenization](API_REFERENCE.md#tokenization-and-segmentation) |
| Similarity metrics | API_REFERENCE.md | [Text Similarity](API_REFERENCE.md#text-similarity) |
| Text metrics | API_REFERENCE.md | [Text Metrics](API_REFERENCE.md#text-metrics) |
| Pattern extraction | API_REFERENCE.md | [Pattern Extraction](API_REFERENCE.md#pattern-extraction) |
| Quick examples | QUICK_REFERENCE.md | [NLP Utilities](QUICK_REFERENCE.md#nlp-utilities) |

---

## Common Use Cases

### Use Case 1: Test a model for bias
1. Read [BiasProbe documentation](API_REFERENCE.md#biasprobe)
2. Check [quick example](QUICK_REFERENCE.md#using-probes)
3. See [complete example](API_REFERENCE.md#complete-usage-example)

### Use Case 2: Compare multiple models
1. Read [ModelBenchmark documentation](API_REFERENCE.md#modelbenchmark)
2. Check [benchmarking example](QUICK_REFERENCE.md#benchmarking)
3. See [configuration files](QUICK_REFERENCE.md#configuration-files)

### Use Case 3: Process and analyze text
1. Browse [NLP Utilities](API_REFERENCE.md#nlp-utilities)
2. Check [quick examples](QUICK_REFERENCE.md#nlp-utilities)
3. See `examples/example_nlp.py` for full demo

### Use Case 4: Run experiments from config
1. Read [Configuration-Based Execution](API_REFERENCE.md#configuration-based-execution)
2. Check [config example](QUICK_REFERENCE.md#configuration-files)
3. See `README.md` and `QUICK_REFERENCE.md` for sample configs

### Use Case 5: Run a cross-model behavioural harness
1. Read [README.md](README.md#behavioural-harness)
2. See the runner guide in the [Docs Site](https://dr-gareth-roberts.github.io/insideLLMs/)

---

## Search Tips

- **Looking for a specific function?** Use your editor's search (Cmd/Ctrl+F) in API_REFERENCE.md
- **Need a quick example?** Check QUICK_REFERENCE.md first
- **Want to understand a concept?** Start with the relevant section in API_REFERENCE.md
- **Need installation help?** See README.md

---

## Additional Resources

- **Examples:** See `examples/` directory for runnable code
- **Compliance Intelligence:** See `compliance_intelligence/` for a multi-agent AML/KYC demo (LangGraph; separate from core insideLLMs)
- **Tests:** See `tests/` directory for usage examples
- **Configuration:** See `README.md` and `QUICK_REFERENCE.md` for sample configurations
- **Source Code:** Browse `insideLLMs/` for implementation details

## Archival (Historical Planning Docs)

- **docs/plans/**: Historical implementation plans and migration notes. See [docs/plans/README.md](docs/plans/README.md). These are **not** current API guidance; use `docs/STABILITY_MATRIX.md` and `docs/IMPORT_PATHS.md` for up-to-date contracts.

---

## Contributing

If you find errors or have suggestions for improving the documentation:
1. Open an issue on GitHub
2. Submit a pull request with corrections
3. Contact the maintainers

---

**Last Updated:** February 26, 2026  
**Version:** 0.2.0
