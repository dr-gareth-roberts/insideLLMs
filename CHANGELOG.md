# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Stability contract matrix documentation (`docs/STABILITY_MATRIX.md`) defining Stable/Experimental/Internal surfaces and change policy
- GitHub issue templates for bug reports and feature requests
- Pull request template
- Security policy (SECURITY.md)
- Code of Conduct
- Comprehensive CI/CD with multi-Python version testing (3.10-3.12)
- Coverage reporting with Codecov integration
- Cross-model behavioural harness command with JSONL outputs, summary, and HTML report
- Versioned output schemas with optional Pydantic validation (runner/results/export) and `insidellms schema` CLI utilities

## [0.2.0] - 2026-02-26

### Removed
- Deprecated import shims: `insideLLMs.cache`, `insideLLMs.caching`,
  `insideLLMs.runner`, `insideLLMs.comparison`, `insideLLMs.statistics`,
  `insideLLMs.trace_config`. Import from canonical paths instead
  (`insideLLMs.caching`, `insideLLMs.runtime.runner`,
  `insideLLMs.analysis.comparison`, `insideLLMs.analysis.statistics`,
  `insideLLMs.trace.trace_config`).

### Changed
- Renamed `insideLLMs.caching_unified` to `insideLLMs.caching`.
- `run_policy()` no longer accepts a `policy_yaml_path` parameter (was unused).

## [0.1.0] - 2025-01-18

### Added
- **Core Framework**
  - Base `Model` and `Probe` abstractions with protocol-based design
  - `ProbeRunner` and `AsyncProbeRunner` for batch evaluation
  - Plugin registry system for models, probes, and datasets
  - Configuration-driven experiments via YAML/JSON

- **Model Integrations**
  - OpenAI (GPT-4, GPT-3.5, etc.)
  - Anthropic (Claude 3 family)
  - Google Gemini
  - Cohere (Command, Command-R)
  - HuggingFace Transformers
  - Local models: Ollama, llama.cpp, vLLM
  - `DummyModel` for testing without API calls

- **Evaluation Probes**
  - `LogicProbe` - Logical reasoning evaluation
  - `BiasProbe` - Bias detection in responses
  - `AttackProbe` - Adversarial robustness testing
  - `FactualityProbe` - Factual accuracy assessment
  - `PromptInjectionProbe` - Injection vulnerability testing
  - `JailbreakProbe` - Jailbreak attempt detection
  - Code probes: generation, explanation, debugging

- **Production Infrastructure**
  - Intelligent caching (LRU, LFU, TTL, semantic similarity)
  - Rate limiting and throttling
  - Cost tracking and budget management
  - Context window management
  - Streaming utilities

- **Safety & Security**
  - PII detection
  - Content safety analysis
  - Prompt injection detection
  - Input sanitization

- **Analysis Tools**
  - Hallucination detection
  - Reasoning chain extraction
  - Model fingerprinting
  - Calibration analysis
  - Sensitivity analysis

- **Experiment Tracking**
  - W&B, MLflow, TensorBoard integration
  - Full reproducibility with environment capture
  - Multi-backend logging

- **Visualization**
  - Interactive dashboards
  - Comparison charts
  - HTML report generation

- **CLI**
  - `insidellms run` - Run experiments from config
  - `insidellms benchmark` - Benchmark models
  - `insidellms compare` - Compare models
  - `insidellms list` - List available components

- **NLP Utilities**
  - 70+ text processing functions
  - Similarity metrics
  - Tokenization and chunking
  - Keyword extraction

### Documentation
- Comprehensive README with examples
- API Reference documentation
- Architecture diagrams (Mermaid)
- Quick Reference guide
- Contributing guidelines

[Unreleased]: https://github.com/dr-gareth-roberts/insideLLMs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dr-gareth-roberts/insideLLMs/releases/tag/v0.1.0
