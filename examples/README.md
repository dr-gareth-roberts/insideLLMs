# Examples

This directory contains runnable examples for common insideLLMs workflows. Each example demonstrates specific features and can be run independently.

## Getting Started

**Recommended order for new users:**
1. Start with `example_quickstart.py` - Learn the basics in 5 minutes
2. Try `example_cli_golden_path.py` - See the CLI workflow
3. Run `demo_diff_pipeline.py` - Understand the full diff pipeline
4. Explore other examples based on your needs

## Prerequisites

**Basic examples** (no API keys needed):
- Python 3.10+
- `pip install -e ".[dev]"` from the repository root

**Examples requiring API keys:**
- Set environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_API_KEY`
- Some examples gracefully skip providers if keys are missing

## Compatibility Matrix

| Example | Offline? | API keys required | Duration |
|---------|----------|-------------------|----------|
| `example_quickstart.py` | Yes | None | ~30 seconds |
| `example_cli_golden_path.py` | Yes | None | ~1 minute |
| `demo_diff_pipeline.py` | Yes | None | ~2 minutes |
| `example_harness_programmatic.py` | Yes | None | ~30 seconds |
| `example_models.py` | Partial | Dummy/HF work offline; OpenAI needs `OPENAI_API_KEY` | ~1 minute |
| `example_factuality.py` | Partial | Skips OpenAI/Anthropic if keys missing | ~2 minutes |
| `example_registry.py` | Yes | None | ~30 seconds |
| `example_benchmark_suite.py` | Yes | None | ~1 minute |
| `example_experiment_tracking.py` | Yes | None (local backend) | ~1 minute |
| `example_interactive_visualization.py` | Yes | None | ~1 minute |
| `example_nlp.py` | Yes | None | ~30 seconds |
| `example_langchain_langgraph.py` | No | Requires LangChain dependencies | ~1 minute |
| `example_ollama_cloud_harness_diff.py` | No | `OLLAMA_API_KEY` | ~3 minutes |
| `example_openrouter_advanced.py` | No | `OPENROUTER_API_KEY` | ~3 minutes |
| `examples/experiment.yaml` | Yes | None (DummyModel) | N/A |
| `examples/harness.yaml` | No | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | N/A |
| `examples/harness_ollama_cloud_deepseek_battery.yaml` | No | `OLLAMA_API_KEY` | N/A |
| `ci/harness.yaml` | Yes | None (DummyModel only) | N/A |

## Core Examples (Start Here)

### `example_quickstart.py` - 5-Minute Introduction

**What it demonstrates:**
- Basic model interaction (generate, chat, stream)
- Running individual probes (LogicProbe, BiasProbe)
- Using ProbeRunner for batch evaluation
- Saving and loading results

**Run it:**
```bash
python examples/example_quickstart.py
```

**Expected output:** Four sections showing model usage, probe results, batch processing, and file I/O

---

### `example_cli_golden_path.py` - End-to-End CLI Workflow

**What it demonstrates:**
- Running harness via CLI (not programmatically)
- Generating reports from run directories
- Comparing two runs with `insidellms diff`
- Verifying deterministic outputs

**Run it:**
```bash
python examples/example_cli_golden_path.py
```

**Expected output:** CLI command execution, run directories created under `.tmp/runs/`, diff showing no changes

---

### `demo_diff_pipeline.py` - Complete Diff Pipeline Walkthrough

**What it demonstrates:**
- 10 narrated sections walking through:
  1. Creating baseline and candidate runs
  2. Detecting output changes
  3. Schema validation
  4. CI gating exit codes
  5. Interactive diff exploration
  6. Report rebuilding

**Run it:**
```bash
python examples/demo_diff_pipeline.py
```

**Expected output:** Detailed walkthrough with explanations, temporary artefacts in `/tmp/`, educational output

---

## Model Integration Examples

### `example_models.py` - Multi-Provider Model Usage

**What it demonstrates:**
- Using DummyModel (offline testing)
- Using HuggingFace models (local inference)
- Using OpenAI models (API-based)
- Prompt templates with model composition

**Run it:**
```bash
python examples/example_models.py
# With OpenAI:
OPENAI_API_KEY=your_key python examples/example_models.py
```

---

### `example_ollama_cloud_harness_diff.py` - Ollama Cloud Integration

**What it demonstrates:**
- Configuring Ollama Cloud models
- Running double-harness workflow
- Comparing outputs between runs
- Cloud-hosted model integration patterns

**Run it:**
```bash
OLLAMA_API_KEY=your_key python examples/example_ollama_cloud_harness_diff.py
```

---

### `example_openrouter_advanced.py` - OpenRouter Advanced Features

**What it demonstrates:**
- OpenRouter model configuration
- Advanced harness options
- Statistical analysis of results
- Multi-model comparison via OpenRouter

**Run it:**
```bash
OPENROUTER_API_KEY=your_key python examples/example_openrouter_advanced.py
```

---

## Probe and Testing Examples

### `example_factuality.py` - Factuality Testing

**What it demonstrates:**
- Using FactualityProbe for fact-checking
- Model benchmarking across providers
- Result visualisation
- HTML report generation

**Run it:**
```bash
python examples/example_factuality.py
# With API keys:
OPENAI_API_KEY=key1 ANTHROPIC_API_KEY=key2 python examples/example_factuality.py
```

---

### `example_harness_programmatic.py` - Programmatic Harness

**What it demonstrates:**
- Running harness without CLI
- Programmatic configuration
- Direct API usage
- Custom workflow integration

**Run it:**
```bash
python examples/example_harness_programmatic.py
```

---

## Advanced Workflows

### `killer_feature_1_ci_gate.py` - CI Gating Workflow

**What it demonstrates:**
- Deterministic behavioural CI gate
- Baseline vs candidate comparison
- Exit codes for CI pipeline gating
- Regression detection in CI

**Run it:**
```bash
python examples/killer_feature_1_ci_gate.py
```

---

### `killer_feature_2_trace_guard.py` - Trace-Aware Drift Detection

**What it demonstrates:**
- Trace-aware diffing
- Drift sensitivity gates
- Violation guardrails
- Advanced fingerprinting

**Run it:**
```bash
python examples/killer_feature_2_trace_guard.py
```

---

### `killer_feature_3_model_selection.py` - Model Selection Workflow

**What it demonstrates:**
- Multi-model comparison workflow
- Statistical analysis and reporting
- Model selection criteria
- Performance benchmarking

**Run it:**
```bash
python examples/killer_feature_3_model_selection.py
```

---

## Utility Examples

### `example_registry.py` - Registry Patterns

**What it demonstrates:**
- Registering custom models and probes
- Using the registry system
- Plugin-style extensions
- Discovery and lookup patterns

**Run it:**
```bash
python examples/example_registry.py
```

---

### `example_benchmark_suite.py` - Benchmarking

**What it demonstrates:**
- Running benchmark suites
- Performance measurement
- Result aggregation
- Comparative analysis

**Run it:**
```bash
python examples/example_benchmark_suite.py
```

---

### `example_experiment_tracking.py` - Experiment Tracking

**What it demonstrates:**
- Local experiment tracking
- Integration with W&B, MLflow, TensorBoard
- Logging metrics and parameters
- Dashboard integration

**Run it:**
```bash
python examples/example_experiment_tracking.py
```

---

### `example_nlp.py` - NLP Utilities

**What it demonstrates:**
- Text processing functions
- Similarity metrics
- Tokenisation and cleaning
- 100+ NLP utilities

**Run it:**
```bash
python examples/example_nlp.py
```

---

### `example_interactive_visualization.py` - Visualisation Tools

**What it demonstrates:**
- Interactive report generation
- Data visualisation
- HTML report customisation
- Chart and graph creation

**Run it:**
```bash
python examples/example_interactive_visualization.py
```

---

### `example_plugin.py` - Plugin Development

**What it demonstrates:**
- Creating custom plugins
- Plugin registration
- Extension points
- Third-party integration

**Run it:**
```bash
python examples/example_plugin.py
```

---

### `example_langchain_langgraph.py` - LangChain/LangGraph Integration

**What it demonstrates:**
- Integration with LangChain
- LangGraph workflow patterns
- Chain composition
- Agent integration

**Requires:** LangChain dependencies

**Run it:**
```bash
pip install langchain langgraph
python examples/example_langchain_langgraph.py
```

---

## Configuration Examples

### `experiment.yaml` - Minimal Offline Experiment

A minimal configuration file for running experiments with DummyModel (no API keys required).

**Use with:**
```bash
insidellms run examples/experiment.yaml --output-dir .tmp/experiment_run
```

---

### `harness.yaml` - Multi-Model Harness

Sample configuration for running a harness across multiple model providers.

**Requires:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

**Use with:**
```bash
insidellms harness examples/harness.yaml --run-dir .tmp/multi_model_run
```

---

### `harness_ollama_cloud_deepseek_battery.yaml` - Ollama Battery Test

Configuration for running a comprehensive probe battery with Ollama Cloud's DeepSeek models.

**Requires:** `OLLAMA_API_KEY`

**Use with:**
```bash
insidellms harness examples/harness_ollama_cloud_deepseek_battery.yaml --run-dir .tmp/ollama_run
```

---

### `probe_battery.jsonl` - Multi-Probe Dataset

Sample dataset containing diverse prompts for testing multiple probe categories (logic, bias, safety, factuality).

**Use as input to harness runs** or programmatic probe execution.

---

## Helper Scripts

### `scripts/ollama_cloud_harness_diff.sh`

Bash script to run Ollama Cloud harness twice and diff the results. Automates the baseline/candidate workflow.

**Run it:**
```bash
bash scripts/ollama_cloud_harness_diff.sh
```

---

## Markdown Guides

### `ollama_cloud_benchmark.md` - Detailed Ollama Walkthrough

Comprehensive guide to running all-model benchmarks on Ollama Cloud, including:
- Setup instructions
- Model selection
- Performance analysis
- Cost considerations

### `parallel_killer_features.md` - Parallel CI Execution

Guide for splitting killer feature demos across parallel CI environments for faster execution.

---

## Troubleshooting

**Common issues:**

1. **Import errors**: Ensure you've installed dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. **Missing API keys**: Examples gracefully skip providers when keys aren't set. Set them in your environment:
   ```bash
   export OPENAI_API_KEY=your_key
   export ANTHROPIC_API_KEY=your_key
   export OLLAMA_API_KEY=your_key
   ```

3. **Path errors**: Run examples from the repository root:
   ```bash
   cd /path/to/insideLLMs
   python examples/example_quickstart.py
   ```

4. **Missing data files**: Some examples expect data files in `../data/`. Check the example source code for specific requirements.

---

## Next Steps

After exploring these examples:
- Review the [API Reference](../API_REFERENCE.md) for detailed API documentation
- Check the [Quick Reference](../QUICK_REFERENCE.md) for code snippets
- Explore the [wiki](../wiki/) for in-depth guides
- Try the [CI harness](../ci/harness.yaml) for offline diff-gating workflows
