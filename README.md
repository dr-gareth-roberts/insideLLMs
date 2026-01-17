# insideLLMs

A world-class Python library for probing, evaluating, and building production-grade LLM applications. Systematically test LLMs' zero-shot ability at unseen logic problems, propensity for bias, vulnerabilities to attacks, factual accuracy, and more.

## Features

### Core Evaluation
- **Multiple Model Support**: OpenAI, Anthropic Claude, Cohere, Google Gemini, HuggingFace, local models (llama.cpp, Ollama, vLLM)
- **Diverse Probes**: Test LLMs on logic, bias, safety, code generation, instruction following, and factual accuracy
- **Comprehensive Benchmarks**: 13+ built-in benchmark datasets covering reasoning, coding, safety, language understanding, and more
- **Ensemble Evaluation**: Run multi-model ensembles with voting strategies

### Experiment Tracking
- **Multiple Backends**: Weights & Biases, MLflow, TensorBoard, and local file-based tracking
- **Auto-tracking Decorator**: Automatically track function calls and their results
- **Multi-tracker Support**: Log to multiple backends simultaneously

### Interactive Visualization
- **Plotly Charts**: Interactive accuracy comparisons, latency distributions, radar charts, heatmaps
- **Jupyter Widgets**: `ExperimentExplorer` for interactive data exploration
- **HTML Reports**: Generate comprehensive HTML reports with embedded charts

### Production Infrastructure
- **Model Adapters**: Unified interface for all LLM providers with fallback chains
- **Context Window Management**: Smart truncation, compression, and token budget allocation
- **Prompt Caching**: Response caching with LRU/LFU/TTL strategies and similarity matching
- **Cost Tracking**: Real-time cost estimation and budget management
- **Streaming Utilities**: Process streaming responses with filters and collectors

### Development Tools
- **Prompt Chains**: Build complex workflows with sequential, parallel, and conditional steps
- **Prompt Debugging**: Trace visualization, breakpoints, and issue detection
- **Experiment Reproducibility**: Seed management, environment capture, and checkpointing
- **Prompt Testing**: Systematic testing with regression detection

## Requirements

- **Python**: 3.8 or higher
- **API Keys** (Set as environment variables):
  - `OPENAI_API_KEY`: For `OpenAIModel`
  - `ANTHROPIC_API_KEY`: For `AnthropicModel`
  - `COHERE_API_KEY`: For `CohereModel`
  - `GOOGLE_API_KEY`: For `GeminiModel`

## Installation

### Using pip

```bash
# Install the base package
pip install .

# To include NLP utilities (requires NLTK, spaCy, scikit-learn, gensim):
pip install .[nlp]

# To include visualization tools (requires matplotlib, pandas, seaborn, plotly):
pip install .[visualization]

# To include all optional dependencies:
pip install .[nlp,visualization]
```

### Using uv (Recommended)

```bash
uv pip install .
```

## Quick Start

```python
from insideLLMs import DummyModel, LogicProbe, ProbeRunner

# Initialize a model (use DummyModel for testing, or a real model)
model = DummyModel()

# Create a probe and runner
probe = LogicProbe()
runner = ProbeRunner(model, probe)

# Run the probe
results = runner.run(["If all A are B, and all B are C, then are all A also C?"])
print(results)
```

For real models, import from submodules:

```python
from insideLLMs.models import OpenAIModel, AnthropicModel

model = OpenAIModel(model_name="gpt-4")
```

## API Structure

The library uses a clean, minimal public API. Core items are available at the top level:

```python
# Core imports (available directly)
from insideLLMs import (
    # Models
    Model, DummyModel, AsyncModel, ChatMessage,
    # Probes
    Probe, LogicProbe, BiasProbe, AttackProbe, FactualityProbe, CustomProbe,
    # Runner
    ProbeRunner, AsyncProbeRunner, run_probe,
    # Types
    ModelResponse, ExperimentResult, ProbeResult,
    # Configuration
    ExperimentConfig, ModelConfig, ProbeConfig,
    # Results
    save_results_json, load_results_json,
)
```

Specialized features are in submodules:

```python
# Models (all providers)
from insideLLMs.models import OpenAIModel, AnthropicModel, HuggingFaceModel

# Evaluation metrics
from insideLLMs.evaluation import bleu_score, rouge_l, exact_match

# Caching
from insideLLMs.caching import InMemoryCache, DiskCache, PromptCache

# Safety analysis
from insideLLMs.safety import detect_pii, quick_safety_check

# Visualization
from insideLLMs.visualization import plot_accuracy_comparison, create_html_report

# Benchmarks
from insideLLMs.benchmark_datasets import list_builtin_datasets, load_builtin_dataset

# Statistics
from insideLLMs.statistics import descriptive_statistics, compare_experiments

# And many more...
```

## Comprehensive Benchmark Datasets

The library includes 13 built-in benchmark datasets covering various evaluation domains:

```python
from insideLLMs.benchmark_datasets import (
    list_builtin_datasets,
    load_builtin_dataset,
    create_comprehensive_benchmark_suite,
    DatasetCategory,
)

# See all available datasets
for ds in list_builtin_datasets():
    print(f"{ds['name']}: {ds['num_examples']} examples ({ds['category']})")

# Load a specific dataset
coding_dataset = load_builtin_dataset("coding")
for example in coding_dataset.sample(3):
    print(f"Input: {example.input_text[:50]}...")

# Create a comprehensive evaluation suite
suite = create_comprehensive_benchmark_suite(
    categories=[DatasetCategory.REASONING, DatasetCategory.MATH],
    max_examples_per_dataset=10,
)
```

**Available datasets:**
- `reasoning` - Syllogisms and logical reasoning
- `factual` - General knowledge questions
- `math` - Arithmetic and algebra
- `commonsense` - Common sense reasoning
- `coding` - Programming problems
- `safety` - Safe request handling
- `bias` - Bias evaluation scenarios
- `language` - Language understanding (sentiment, parsing)
- `instruction` - Instruction following
- `reading` - Reading comprehension
- `multi_step` - Multi-step reasoning
- `analogical` - Analogical reasoning
- `world_knowledge` - Geography, science, history

## Experiment Tracking

Track experiments with Weights & Biases, MLflow, TensorBoard, or local files:

```python
from insideLLMs.experiment_tracking import (
    LocalFileTracker,
    auto_track,
    create_tracker,
)

# Create a local tracker
tracker = LocalFileTracker(output_dir="./experiments")

# Track an experiment
with tracker:
    tracker.log_params({"model": "gpt-4", "temperature": 0.7})
    tracker.log_metrics({"accuracy": 0.95, "latency": 120})

# Or use the decorator
@auto_track(tracker, experiment_name="my-experiment")
def run_evaluation():
    return {"accuracy": 0.92}

# Use W&B, MLflow, or TensorBoard
wandb_tracker = create_tracker("wandb", project="my-project")
mlflow_tracker = create_tracker("mlflow", experiment_name="my-exp")
tensorboard_tracker = create_tracker("tensorboard", log_dir="./logs")
```

## Interactive Visualization

Create interactive visualizations with Plotly:

```python
from insideLLMs.visualization import (
    interactive_accuracy_comparison,
    interactive_latency_distribution,
    interactive_metric_radar,
    create_interactive_dashboard,
    create_interactive_html_report,
    ExperimentExplorer,
)

# Create interactive charts
fig = interactive_accuracy_comparison(experiments, color_by="model")
fig.show()

# Create a full dashboard
dashboard = create_interactive_dashboard(experiments)
dashboard.write_html("dashboard.html")

# Generate HTML report
create_interactive_html_report(
    experiments,
    save_path="report.html",
    title="LLM Evaluation Report",
    include_raw_results=True,
)

# Jupyter notebook exploration
explorer = ExperimentExplorer(experiments)
explorer.display()  # Interactive widgets
```

## Available Models

All model implementations are in `insideLLMs.models`:

```python
from insideLLMs.models import (
    OpenAIModel,      # OpenAI GPT models
    AnthropicModel,   # Anthropic Claude models
    CohereModel,      # Cohere Command models
    GeminiModel,      # Google Gemini models
    HuggingFaceModel, # HuggingFace Transformers
    LlamaCppModel,    # Local llama.cpp models
    OllamaModel,      # Ollama local models
    VLLMModel,        # vLLM inference server
    DummyModel,       # For testing and CI
)
```

## Available Probes

### Core Probes (available at top level)

```python
from insideLLMs import LogicProbe, BiasProbe, AttackProbe, FactualityProbe
```

### Additional Probes (from submodule)

```python
from insideLLMs.probes import (
    # Security probes
    PromptInjectionProbe,
    JailbreakProbe,
    # Code probes
    CodeGenerationProbe,
    CodeExplanationProbe,
    CodeDebugProbe,
    # Instruction probes
    InstructionFollowingProbe,
    MultiStepTaskProbe,
    ConstraintComplianceProbe,
)
```

## Advanced Usage

### Context Window Management

```python
from insideLLMs.context_window import (
    ContextWindow,
    PriorityLevel,
    TruncationStrategy,
)

window = ContextWindow(max_tokens=128000)
window.add_message("system", "You are helpful", priority=PriorityLevel.CRITICAL)
window.add_message("user", "Hello!")

# Auto-truncate when approaching limit
window.truncate(target_tokens=50000, strategy=TruncationStrategy.PRIORITY)
```

### Prompt Caching

```python
from insideLLMs.caching import create_prompt_cache, memoize

cache = create_prompt_cache(max_size=1000, ttl_seconds=3600)

@memoize(max_size=100)
def expensive_llm_call(prompt):
    return model.generate(prompt)
```

### Model Adapters with Fallback

```python
from insideLLMs.adapters import create_adapter, create_fallback_chain, Provider

primary = create_adapter(Provider.OPENAI, model="gpt-4")
fallback = create_adapter(Provider.ANTHROPIC, model="claude-3-opus")
chain = create_fallback_chain([primary, fallback])

result = chain.generate("Hello!")  # Auto-fallback on failure
```

### Prompt Chains

```python
from insideLLMs.chains import ChainBuilder

builder = ChainBuilder()
builder.add_llm_step("analyze", "Analyze: {input}")
builder.add_llm_step("summarize", "Summarize: {analyze_output}")
chain = builder.build()

result = chain.execute({"input": "Your text here"})
```

## Configuration-Based Execution

Run experiments using a YAML configuration file:

```yaml
# config.yaml
model:
  type: openai
  args:
    model_name: gpt-4
probe:
  type: factuality
dataset:
  format: jsonl
  path: data/factuality_questions.jsonl
```

```python
from insideLLMs.runner import run_experiment_from_config

results = run_experiment_from_config("config.yaml")
```

## Submodules Reference

| Submodule | Description |
|-----------|-------------|
| `insideLLMs.models` | Model implementations for all providers |
| `insideLLMs.probes` | Probe implementations for testing LLMs |
| `insideLLMs.evaluation` | Evaluation metrics and evaluators |
| `insideLLMs.caching` | Caching utilities (LRU, LFU, TTL, disk) |
| `insideLLMs.cache` | Simple caching (backward compatibility) |
| `insideLLMs.safety` | Safety analysis (PII, toxicity, bias) |
| `insideLLMs.adversarial` | Adversarial testing and robustness |
| `insideLLMs.knowledge` | Knowledge probing and fact verification |
| `insideLLMs.reasoning` | Chain-of-thought and reasoning analysis |
| `insideLLMs.optimization` | Prompt optimization utilities |
| `insideLLMs.statistics` | Statistical analysis tools |
| `insideLLMs.visualization` | Charts, dashboards, and reports |
| `insideLLMs.streaming` | Output streaming utilities |
| `insideLLMs.distributed` | Distributed execution |
| `insideLLMs.templates` | Prompt template library |
| `insideLLMs.benchmark_datasets` | Benchmark dataset utilities |
| `insideLLMs.experiment_tracking` | W&B, MLflow, TensorBoard integration |
| `insideLLMs.chains` | Prompt workflow orchestration |
| `insideLLMs.adapters` | Model adapter factory |
| `insideLLMs.context_window` | Context window management |
| `insideLLMs.reproducibility` | Experiment reproducibility |
| `insideLLMs.debugging` | Prompt debugging utilities |

## Project Structure

```text
insideLLMs/
├── models/              # Model wrappers (OpenAI, Anthropic, HF, local)
├── probes/              # Probe implementations (Logic, Bias, Code, etc.)
├── nlp/                 # NLP utility modules
├── __init__.py          # Clean, minimal public API
├── caching_unified.py   # Unified caching implementation
├── benchmark_datasets.py # Comprehensive benchmark datasets
├── experiment_tracking.py # W&B, MLflow, TensorBoard integration
├── visualization.py     # Static and interactive visualizations
├── runner.py            # Configuration-based experiment runner
└── ...                  # 40+ additional modules
```

## Development

```bash
git clone https://github.com/dr-gareth-roberts/insideLLMs
cd insideLLMs
pip install -r requirements-dev.txt
```

### Running Tests

```bash
pytest
```

## License

This project is licensed under the MIT License.

Copyright (c) 2024 Dr Gareth Roberts
