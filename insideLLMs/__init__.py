"""insideLLMs - A comprehensive library for probing the inner workings of large language models.

insideLLMs provides a powerful, extensible framework for systematically evaluating
and understanding LLM behavior across multiple critical dimensions. Whether you're
building production AI systems, conducting research, or ensuring model safety,
insideLLMs offers the tools you need.

Overview
--------
The library is organized around three core concepts:

1. **Models**: Unified interface for different LLM providers (OpenAI, Anthropic,
   HuggingFace, Gemini, Cohere, Ollama, and more).
2. **Probes**: Evaluation modules that test specific aspects of model behavior
   (logic, factuality, bias, safety, etc.).
3. **Runners**: Orchestration tools that execute probes against models and
   collect structured results.

Key Features
------------
- **Logic & Reasoning**: Test zero-shot reasoning on unseen logic problems
- **Bias Detection**: Measure propensity for demographic and cultural biases
- **Attack Vulnerabilities**: Test susceptibility to prompt injection and jailbreaks
- **Factual Accuracy**: Evaluate correctness of factual knowledge
- **Safety Analysis**: Detect PII, toxicity, and harmful content
- **Model Comparison**: Side-by-side benchmarking across models
- **LLM-as-a-Judge**: Use LLMs to evaluate other LLM outputs
- **Structured Output**: Parse responses into Pydantic models
- **Observability**: OpenTelemetry tracing and telemetry
- **Async Support**: Concurrent execution for high-throughput evaluation

Quick Start Examples
--------------------

**Example 1: Basic Probe Execution**

Run a logic probe against a model and get structured results:

    >>> from insideLLMs import DummyModel, LogicProbe, ProbeRunner
    >>>
    >>> # Create a model (use DummyModel for testing, OpenAIModel for production)
    >>> model = DummyModel()
    >>>
    >>> # Create a probe to test logic reasoning
    >>> probe = LogicProbe(name="basic_logic")
    >>>
    >>> # Create a runner to orchestrate execution
    >>> runner = ProbeRunner(model, probe)
    >>>
    >>> # Run the probe on test inputs
    >>> prompts = [
    ...     {"messages": [{"role": "user", "content": "What comes next: 1, 2, 3, ?"}]},
    ...     {"messages": [{"role": "user", "content": "If A > B and B > C, is A > C?"}]},
    ... ]
    >>> results = runner.run(prompts)
    >>>
    >>> # Access statistics
    >>> print(f"Success rate: {runner.success_rate:.1%}")
    Success rate: 100.0%

**Example 2: Using Real LLM Providers**

Connect to OpenAI, Anthropic, or other providers:

    >>> from insideLLMs import ProbeRunner
    >>> from insideLLMs.models import OpenAIModel, AnthropicModel
    >>> from insideLLMs.probes import FactualityProbe
    >>>
    >>> # OpenAI (requires OPENAI_API_KEY environment variable)
    >>> openai_model = OpenAIModel(model_name="gpt-4")
    >>>
    >>> # Anthropic (requires ANTHROPIC_API_KEY environment variable)
    >>> claude_model = AnthropicModel(model_name="claude-3-opus-20240229")
    >>>
    >>> # Run factuality probe
    >>> probe = FactualityProbe(name="knowledge_test")
    >>> runner = ProbeRunner(openai_model, probe)
    >>> results = runner.run(test_prompts)

**Example 3: Async Execution for High Throughput**

Process many prompts concurrently:

    >>> import asyncio
    >>> from insideLLMs import AsyncProbeRunner
    >>> from insideLLMs.models import OpenAIModel
    >>> from insideLLMs.probes import BiasProbe
    >>>
    >>> async def run_evaluation():
    ...     model = OpenAIModel(model_name="gpt-4")
    ...     probe = BiasProbe(name="gender_bias")
    ...     runner = AsyncProbeRunner(model, probe)
    ...
    ...     # Run 100 prompts with 10 concurrent requests
    ...     results = await runner.run(large_prompt_list, concurrency=10)
    ...     return results
    >>>
    >>> results = asyncio.run(run_evaluation())

**Example 4: Running from Configuration Files**

Define experiments in YAML for reproducibility:

    >>> from insideLLMs import run_harness_from_config
    >>>
    >>> # config.yaml defines models, probes, and datasets
    >>> results = run_harness_from_config("experiment_config.yaml")
    >>> print(f"Ran {len(results)} experiments")

**Example 5: Accessing Evaluation Metrics**

Use built-in metrics for scoring:

    >>> from insideLLMs.evaluation import bleu_score, rouge_l, exact_match
    >>>
    >>> reference = "The capital of France is Paris."
    >>> candidate = "Paris is the capital of France."
    >>>
    >>> print(f"BLEU: {bleu_score(reference, candidate):.3f}")
    >>> print(f"ROUGE-L: {rouge_l(reference, candidate):.3f}")

**Example 6: Safety and PII Detection**

Analyze content for safety issues:

    >>> from insideLLMs.safety import detect_pii, quick_safety_check
    >>>
    >>> text = "Contact John at john.doe@email.com or 555-123-4567"
    >>> pii_results = detect_pii(text)
    >>> print(f"Found PII types: {[p['type'] for p in pii_results]}")
    Found PII types: ['EMAIL', 'PHONE']
    >>>
    >>> safety_result = quick_safety_check("Some potentially harmful content")
    >>> print(f"Safe: {safety_result['is_safe']}")

**Example 7: LLM-as-a-Judge Evaluation**

Use one LLM to evaluate another's outputs:

    >>> from insideLLMs.evaluation import create_judge, HELPFULNESS_CRITERIA
    >>>
    >>> judge = create_judge(model_name="gpt-4", criteria=HELPFULNESS_CRITERIA)
    >>> result = judge.evaluate(
    ...     question="Explain quantum computing",
    ...     answer="Quantum computing uses qubits that can be 0 and 1 simultaneously..."
    ... )
    >>> print(f"Score: {result.score}/5, Reasoning: {result.reasoning}")

**Example 8: Structured Output Parsing**

Parse LLM responses into Pydantic models:

    >>> from pydantic import BaseModel
    >>> from insideLLMs.structured import quick_extract
    >>>
    >>> class PersonInfo(BaseModel):
    ...     name: str
    ...     age: int
    ...     occupation: str
    >>>
    >>> text = "John is a 32-year-old software engineer from NYC."
    >>> person = quick_extract(PersonInfo, text, model=model)
    >>> print(f"{person.name}, {person.age}, {person.occupation}")
    John, 32, software engineer

**Example 9: Custom Probe Implementation**

Create your own probe for specific evaluation needs:

    >>> from insideLLMs.probes.base import Probe
    >>> from insideLLMs.types import ProbeCategory
    >>>
    >>> class ToneProbe(Probe[dict]):
    ...     \"\"\"Probe that analyzes response tone.\"\"\"
    ...
    ...     default_category = ProbeCategory.CUSTOM
    ...
    ...     def run(self, model, data, **kwargs):
    ...         prompt = f"Analyze the tone of: {data}"
    ...         response = model.generate(prompt, **kwargs)
    ...         return {"input": data, "analysis": response}
    >>>
    >>> probe = ToneProbe(name="tone_analyzer")
    >>> runner = ProbeRunner(model, probe)
    >>> results = runner.run(["I love this product!", "This is unacceptable."])

**Example 10: Model Pipeline with Middleware**

Add caching, retry logic, and rate limiting:

    >>> from insideLLMs.pipeline import (
    ...     ModelPipeline,
    ...     CacheMiddleware,
    ...     RetryMiddleware,
    ...     RateLimitMiddleware,
    ... )
    >>> from insideLLMs.models import OpenAIModel
    >>>
    >>> base_model = OpenAIModel(model_name="gpt-4")
    >>> pipeline = ModelPipeline(base_model).use(
    ...     CacheMiddleware(),           # Cache repeated prompts
    ...     RetryMiddleware(max_retries=3),  # Retry on failures
    ...     RateLimitMiddleware(rpm=60),     # 60 requests per minute
    ... )
    >>> response = pipeline.generate("Hello, world!")

Available Submodules
--------------------
insideLLMs.models
    Model implementations for various providers:
    - OpenAIModel: GPT-3.5, GPT-4, and other OpenAI models
    - AnthropicModel: Claude 2, Claude 3 family
    - HuggingFaceModel: Any Transformers model
    - GeminiModel: Google Gemini models
    - CohereModel: Cohere Command models
    - OllamaModel: Local Ollama models
    - LlamaCppModel: Local llama.cpp models
    - VLLMModel: vLLM-served models
    - DummyModel: Testing without API calls

insideLLMs.probes
    Probe implementations for different evaluation dimensions:
    - LogicProbe: Logical reasoning and deduction
    - FactualityProbe: Factual accuracy and knowledge
    - BiasProbe: Demographic and cultural bias detection
    - AttackProbe: Adversarial robustness testing
    - AgentProbe: Agent/tool-use capabilities
    - CustomProbe: Build your own probes

insideLLMs.evaluation
    Evaluation metrics and LLM-as-a-Judge:
    - bleu_score, rouge_l, token_f1: Text similarity metrics
    - JudgeModel, JudgeEvaluator: LLM-based evaluation
    - ExactMatchEvaluator: Exact string matching
    - HELPFULNESS_CRITERIA, ACCURACY_CRITERIA: Pre-defined judge criteria

insideLLMs.caching
    Response caching utilities:
    - InMemoryCache: Fast in-process caching
    - DiskCache: Persistent disk-based caching
    - CachedModel: Wrap any model with caching

insideLLMs.safety
    Safety analysis tools:
    - detect_pii: Find personally identifiable information
    - quick_safety_check: Fast toxicity/safety screening
    - ContentSafetyAnalyzer: Comprehensive safety analysis

insideLLMs.structured
    Structured output parsing:
    - generate_structured: Generate and parse into Pydantic models
    - quick_extract: Simple extraction interface
    - batch_extract: Process multiple inputs

insideLLMs.observability
    Tracing and telemetry:
    - TracingConfig: Configure OpenTelemetry tracing
    - TracedModel: Wrap models with automatic tracing
    - trace_function: Decorator for function-level tracing

insideLLMs.pipeline
    Model middleware:
    - ModelPipeline: Chain middleware together
    - CacheMiddleware, RetryMiddleware, RateLimitMiddleware

insideLLMs.analysis.statistics
    Statistical analysis:
    - descriptive_statistics: Basic statistics
    - compare_experiments: Cross-experiment comparison
    - confidence_interval: Statistical intervals

insideLLMs.visualization
    Visualization and reporting:
    - plot_accuracy_comparison: Compare model accuracies
    - create_html_report: Generate HTML reports
    - text_comparison_table: Side-by-side text comparison

insideLLMs.contrib
    Research and analysis modules (agents, routing, deployment, synthesis,
    HITL, etc.). Not part of the core eval pipeline but available via
    lazy imports: ``from insideLLMs import ReActAgent``

Configuration and Results
-------------------------
The library uses structured types for configuration and results:

- ExperimentConfig: Define complete experiment setup
- ModelConfig: Configure model parameters
- ProbeConfig: Configure probe behavior
- ExperimentResult: Structured experiment outcomes
- ProbeResult: Individual probe execution results
- ProbeScore: Aggregate scoring metrics

Registry System
---------------
Models, probes, and datasets can be registered and discovered:

    >>> from insideLLMs import model_registry, probe_registry
    >>>
    >>> # List available models
    >>> print(model_registry.list_names())
    ['openai', 'anthropic', 'huggingface', ...]
    >>>
    >>> # Get a model by name
    >>> model = model_registry.get("openai", model_name="gpt-4")

Environment Variables
---------------------
- OPENAI_API_KEY: Required for OpenAI models
- ANTHROPIC_API_KEY: Required for Anthropic models
- GOOGLE_API_KEY: Required for Gemini models
- COHERE_API_KEY: Required for Cohere models
- HF_TOKEN: Optional for private HuggingFace models

See Also
--------
- insideLLMs.models.base : Base classes for model implementations
- insideLLMs.probes.base : Base classes for probe implementations
- insideLLMs.runtime.runner : Runner implementation details
- insideLLMs.types : Type definitions for results and configuration

Notes
-----
This package follows semantic versioning. The public API consists of all
symbols exported in __all__. Internal modules (prefixed with _) may change
without notice.

For production use, always pin to a specific version and test thoroughly
before upgrading.
"""

__version__ = "0.2.0"

# =============================================================================
# Core Types (Essential for all users)
# =============================================================================
# =============================================================================
# Configuration (Essential for experiment setup)
# =============================================================================
from insideLLMs.config import (
    ExperimentConfig,
    ModelConfig,
    ProbeConfig,
)

# =============================================================================
# Exceptions (Core error handling)
# =============================================================================
from insideLLMs.exceptions import (
    InsideLLMsError,
    ModelError,
    ProbeError,
)

# =============================================================================
# Core Models (Most commonly used)
# =============================================================================
from insideLLMs.models import (
    DummyModel,
    Model,
)
from insideLLMs.models.base import (
    AsyncModel,
    ChatMessage,
    ModelProtocol,
)

# =============================================================================
# Core Probes (Most commonly used)
# =============================================================================
from insideLLMs.probes import (
    AgentProbe,
    AgentProbeResult,
    AttackProbe,
    BiasProbe,
    CustomProbe,
    FactualityProbe,
    LogicProbe,
    Probe,
    ToolDefinition,
)

# =============================================================================
# Registry (For plugin system)
# =============================================================================
from insideLLMs.registry import (
    Registry,
    ensure_builtins_registered,
    model_registry,
    probe_registry,
)

# =============================================================================
# Results and Reporting (Essential for saving/loading results)
# =============================================================================
from insideLLMs.results import (
    load_results_json,
    save_results_json,
)

# =============================================================================
# Runner and Execution (Essential for running experiments)
# =============================================================================
from insideLLMs.runtime.runner import (
    AsyncProbeRunner,
    ProbeRunner,
    create_experiment_result,
    run_harness_from_config,
    run_probe,
)

# =============================================================================
# Trace Configuration (For deterministic CI enforcement)
# =============================================================================
from insideLLMs.trace.trace_config import (
    FingerprintConfig,
    NormaliserConfig,
    NormaliserKind,
    OnViolationMode,
    StoreMode,
    TraceConfig,
    TracePayloadNormaliser,
    load_trace_config,
    make_structural_v1_normaliser,
    validate_with_config,
)
from insideLLMs.types import (
    ExperimentResult,
    ModelInfo,
    ModelResponse,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
    TokenUsage,
)

from . import shadow

# Auto-register built-in models, probes, and datasets
ensure_builtins_registered()

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version
    "__version__",
    # Core Types
    "ExperimentResult",
    "ModelInfo",
    "ModelResponse",
    "ProbeCategory",
    "ProbeResult",
    "ProbeScore",
    "ResultStatus",
    "TokenUsage",
    # Core Models
    "AsyncModel",
    "ChatMessage",
    "DummyModel",
    "Model",
    "ModelProtocol",
    # Core Probes
    "AgentProbe",
    "AgentProbeResult",
    "AttackProbe",
    "BiasProbe",
    "CustomProbe",
    "FactualityProbe",
    "LogicProbe",
    "Probe",
    "ToolDefinition",
    # Trace Configuration
    "TraceConfig",
    "load_trace_config",
    "validate_with_config",
    "TracePayloadNormaliser",
    "OnViolationMode",
    "StoreMode",
    "NormaliserKind",
    "NormaliserConfig",
    "FingerprintConfig",
    "make_structural_v1_normaliser",
    # Runner
    "AsyncProbeRunner",
    "ProbeRunner",
    "create_experiment_result",
    "run_harness_from_config",
    "run_probe",
    # Results
    "load_results_json",
    "save_results_json",
    # Configuration
    "ExperimentConfig",
    "ModelConfig",
    "ProbeConfig",
    # Exceptions
    "InsideLLMsError",
    "ModelError",
    "ProbeError",
    # Registry
    "Registry",
    "ensure_builtins_registered",
    "model_registry",
    "probe_registry",
    "shadow",
]


# =============================================================================
# Lazy Loading for Heavy Submodules
# =============================================================================
def __getattr__(name: str):
    """Lazy-load submodule attributes to minimize import time and memory usage.

    This function implements Python's module-level __getattr__ (PEP 562) to
    provide lazy loading of heavy submodule components. When you access an
    attribute like `insideLLMs.OpenAIModel`, this function intercepts the
    access and imports the required submodule on-demand.

    This design pattern offers several benefits:

    1. **Fast Import Times**: The main `insideLLMs` package imports quickly
       because heavy dependencies (transformers, openai, anthropic, etc.)
       are only loaded when actually needed.

    2. **Memory Efficiency**: Unused components never consume memory.

    3. **Graceful Degradation**: Optional dependencies don't cause import
       errors until they're actually used.

    Parameters
    ----------
    name : str
        The attribute name being accessed. This is typically a class name
        like 'OpenAIModel' or a function name like 'bleu_score'.

    Returns
    -------
    Any
        The requested attribute from the appropriate submodule.

    Raises
    ------
    AttributeError
        If the requested attribute is not found in any submodule. The error
        message includes a suggestion to import from a submodule directly.

    Examples
    --------
    Accessing a lazy-loaded model class:

        >>> from insideLLMs import OpenAIModel  # Triggers lazy import
        >>> model = OpenAIModel(model_name="gpt-4")

    This is equivalent to but faster at initial import than:

        >>> from insideLLMs.models import OpenAIModel  # Direct import

    Accessing evaluation functions:

        >>> from insideLLMs import bleu_score, rouge_l
        >>> score = bleu_score("reference text", "candidate text")

    Handling missing attributes:

        >>> from insideLLMs import NonExistentThing
        AttributeError: module 'insideLLMs' has no attribute 'NonExistentThing'.
        Did you mean to import from a submodule?
        Try: from insideLLMs.<submodule> import NonExistentThing

    Notes
    -----
    The following categories of components are available via lazy loading:

    - **Models**: OpenAIModel, OpenRouterModel, AnthropicModel, HuggingFaceModel,
      GeminiModel, CohereModel, OllamaModel, VLLMModel, LlamaCppModel
    - **Probes**: PromptInjectionProbe, JailbreakProbe, CodeGenerationProbe, etc.
    - **Caching**: InMemoryCache, DiskCache, CachedModel
    - **Pipeline**: ModelPipeline, CacheMiddleware, RetryMiddleware, etc.
    - **Evaluation**: bleu_score, rouge_l, JudgeModel, create_judge, etc.
    - **Structured**: generate_structured, quick_extract, batch_extract
    - **Safety**: detect_pii, quick_safety_check, ContentSafetyAnalyzer
    - **Agents**: ReActAgent, SimpleAgent, Tool, ToolRegistry
    - **HITL**: HITLSession, ReviewQueue, FeedbackCollector
    - **Routing**: SemanticRouter, ModelPool, IntentClassifier
    - **Observability**: TracingConfig, TracedModel, trace_function
    - **Synthesis**: PromptVariator, AdversarialGenerator, DataAugmenter
    - **Deployment**: create_app, quick_deploy, ModelEndpoint

    For a complete list, see the _LAZY_IMPORTS dictionary in the source.

    See Also
    --------
    __all__ : List of directly exported symbols (not lazy-loaded)
    """
    # Map of commonly requested items to their submodule paths
    _LAZY_IMPORTS = {
        # Models
        "OpenAIModel": "insideLLMs.models",
        "OpenRouterModel": "insideLLMs.models",
        "AnthropicModel": "insideLLMs.models",
        "HuggingFaceModel": "insideLLMs.models",
        "GeminiModel": "insideLLMs.models",
        "CohereModel": "insideLLMs.models",
        "OllamaModel": "insideLLMs.models",
        "VLLMModel": "insideLLMs.models",
        "LlamaCppModel": "insideLLMs.models",
        # Probes
        "PromptInjectionProbe": "insideLLMs.probes",
        "JailbreakProbe": "insideLLMs.probes",
        "CodeGenerationProbe": "insideLLMs.probes",
        "CodeExplanationProbe": "insideLLMs.probes",
        "CodeDebugProbe": "insideLLMs.probes",
        "InstructionFollowingProbe": "insideLLMs.probes",
        "MultiStepTaskProbe": "insideLLMs.probes",
        "ConstraintComplianceProbe": "insideLLMs.probes",
        "ComparativeProbe": "insideLLMs.probes",
        "ScoredProbe": "insideLLMs.probes",
        # Caching
        "InMemoryCache": "insideLLMs.caching",
        "DiskCache": "insideLLMs.caching",
        "CachedModel": "insideLLMs.caching",
        "cached": "insideLLMs.caching",
        "BaseCache": "insideLLMs.caching",
        "CacheEntry": "insideLLMs.caching",
        "CacheStats": "insideLLMs.caching",
        # Pipeline (Model Middleware)
        "ModelPipeline": "insideLLMs.pipeline",
        "Middleware": "insideLLMs.pipeline",
        "PassthroughMiddleware": "insideLLMs.pipeline",
        "CacheMiddleware": "insideLLMs.pipeline",
        "RateLimitMiddleware": "insideLLMs.pipeline",
        "RetryMiddleware": "insideLLMs.pipeline",
        "CostTrackingMiddleware": "insideLLMs.pipeline",
        # Evaluation
        "Evaluator": "insideLLMs.evaluation",
        "ExactMatchEvaluator": "insideLLMs.evaluation",
        "bleu_score": "insideLLMs.evaluation",
        "rouge_l": "insideLLMs.evaluation",
        "token_f1": "insideLLMs.evaluation",
        "exact_match": "insideLLMs.evaluation",
        # LLM-as-a-Judge
        "JudgeModel": "insideLLMs.evaluation",
        "JudgeResult": "insideLLMs.evaluation",
        "JudgeCriterion": "insideLLMs.evaluation",
        "JudgeEvaluator": "insideLLMs.evaluation",
        "create_judge": "insideLLMs.evaluation",
        "HELPFULNESS_CRITERIA": "insideLLMs.evaluation",
        "ACCURACY_CRITERIA": "insideLLMs.evaluation",
        "SAFETY_CRITERIA": "insideLLMs.evaluation",
        "CODE_QUALITY_CRITERIA": "insideLLMs.evaluation",
        # Structured Output (Pydantic Integration)
        "generate_structured": "insideLLMs.structured",
        "quick_extract": "insideLLMs.structured",
        "StructuredResult": "insideLLMs.structured",
        "StructuredOutputGenerator": "insideLLMs.structured",
        "StructuredOutputConfig": "insideLLMs.structured",
        "batch_extract": "insideLLMs.structured",
        "results_to_html_report": "insideLLMs.structured",
        "extract_json": "insideLLMs.structured",
        "parse_json": "insideLLMs.structured",
        # Statistics
        "descriptive_statistics": "insideLLMs.analysis.statistics",
        "compare_experiments": "insideLLMs.analysis.statistics",
        "confidence_interval": "insideLLMs.analysis.statistics",
        # Safety
        "detect_pii": "insideLLMs.safety",
        "quick_safety_check": "insideLLMs.safety",
        "ContentSafetyAnalyzer": "insideLLMs.safety",
        # Visualization
        "plot_accuracy_comparison": "insideLLMs.visualization",
        "create_html_report": "insideLLMs.visualization",
        "text_comparison_table": "insideLLMs.visualization",
        # Benchmarking
        "ModelBenchmark": "insideLLMs.performance.benchmark",
        "ProbeBenchmark": "insideLLMs.performance.benchmark",
        # Synthetic Data Generation
        "PromptVariator": "insideLLMs.datasets.synthesis",
        "AdversarialGenerator": "insideLLMs.datasets.synthesis",
        "DataAugmenter": "insideLLMs.datasets.synthesis",
        "TemplateGenerator": "insideLLMs.datasets.synthesis",
        "SyntheticDataset": "insideLLMs.datasets.synthesis",
        "quick_variations": "insideLLMs.datasets.synthesis",
        "quick_adversarial": "insideLLMs.datasets.synthesis",
        "generate_test_dataset": "insideLLMs.datasets.synthesis",
        "SynthesisConfig": "insideLLMs.datasets.synthesis",
        "VariationStrategy": "insideLLMs.datasets.synthesis",
        "AdversarialType": "insideLLMs.datasets.synthesis",
        # Observability
        "TracingConfig": "insideLLMs.observability",
        "CallRecord": "insideLLMs.observability",
        "TelemetryCollector": "insideLLMs.observability",
        "TracedModel": "insideLLMs.observability",
        "instrument_model": "insideLLMs.observability",
        "trace_call": "insideLLMs.observability",
        "trace_function": "insideLLMs.observability",
        # Semantic Caching
        "SemanticCache": "insideLLMs.semantic_cache",
        "VectorCache": "insideLLMs.semantic_cache",
        "RedisCache": "insideLLMs.semantic_cache",
        "SemanticCacheModel": "insideLLMs.semantic_cache",
        "SemanticCacheConfig": "insideLLMs.semantic_cache",
        "create_semantic_cache": "insideLLMs.semantic_cache",
        "quick_semantic_cache": "insideLLMs.semantic_cache",
        "wrap_model_with_semantic_cache": "insideLLMs.semantic_cache",
        # Autonomous Agents
        "ReActAgent": "insideLLMs.agents.agents",
        "SimpleAgent": "insideLLMs.agents.agents",
        "ChainOfThoughtAgent": "insideLLMs.agents.agents",
        "Tool": "insideLLMs.agents.agents",
        "ToolRegistry": "insideLLMs.agents.agents",
        "AgentExecutor": "insideLLMs.agents.agents",
        "AgentConfig": "insideLLMs.agents.agents",
        "AgentResult": "insideLLMs.agents.agents",
        "create_react_agent": "insideLLMs.agents.agents",
        "create_simple_agent": "insideLLMs.agents.agents",
        "create_calculator_tool": "insideLLMs.agents.agents",
        "quick_agent_run": "insideLLMs.agents.agents",
        # Human-in-the-Loop
        "HITLSession": "insideLLMs.agents.hitl",
        "InteractiveSession": "insideLLMs.agents.hitl",
        "ApprovalWorkflow": "insideLLMs.agents.hitl",
        "ReviewWorkflow": "insideLLMs.agents.hitl",
        "AnnotationWorkflow": "insideLLMs.agents.hitl",
        "ReviewQueue": "insideLLMs.agents.hitl",
        "PriorityReviewQueue": "insideLLMs.agents.hitl",
        "HumanValidator": "insideLLMs.agents.hitl",
        "ConsensusValidator": "insideLLMs.agents.hitl",
        "FeedbackCollector": "insideLLMs.agents.hitl",
        "AnnotationCollector": "insideLLMs.agents.hitl",
        "HITLConfig": "insideLLMs.agents.hitl",
        "FeedbackType": "insideLLMs.agents.hitl",
        "ReviewStatus": "insideLLMs.agents.hitl",
        "Priority": "insideLLMs.agents.hitl",
        "Feedback": "insideLLMs.agents.hitl",
        "ReviewItem": "insideLLMs.agents.hitl",
        "Annotation": "insideLLMs.agents.hitl",
        "create_hitl_session": "insideLLMs.agents.hitl",
        "quick_review": "insideLLMs.agents.hitl",
        "collect_feedback": "insideLLMs.agents.hitl",
        # Model Routing
        "SemanticRouter": "insideLLMs.agents.routing",
        "Route": "insideLLMs.agents.routing",
        "RouteMatch": "insideLLMs.agents.routing",
        "ModelPool": "insideLLMs.agents.routing",
        "IntentClassifier": "insideLLMs.agents.routing",
        "RouterConfig": "insideLLMs.agents.routing",
        "RoutingStrategy": "insideLLMs.agents.routing",
        "create_router": "insideLLMs.agents.routing",
        "quick_route": "insideLLMs.agents.routing",
        # Deployment (FastAPI)
        "create_app": "insideLLMs.system.deployment",
        "DeploymentApp": "insideLLMs.system.deployment",
        "AppConfig": "insideLLMs.system.deployment",
        "ModelEndpoint": "insideLLMs.system.deployment",
        "ProbeEndpoint": "insideLLMs.system.deployment",
        "BatchEndpoint": "insideLLMs.system.deployment",
        "DeploymentConfig": "insideLLMs.system.deployment",
        "EndpointConfig": "insideLLMs.system.deployment",
        "RateLimiter": "insideLLMs.system.deployment",
        "APIKeyAuth": "insideLLMs.system.deployment",
        "MetricsCollector": "insideLLMs.system.deployment",
        "HealthChecker": "insideLLMs.system.deployment",
        "quick_deploy": "insideLLMs.system.deployment",
        "create_model_endpoint": "insideLLMs.system.deployment",
        "create_probe_endpoint": "insideLLMs.system.deployment",
    }

    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)

    raise AttributeError(
        f"module 'insideLLMs' has no attribute '{name}'. "
        f"Did you mean to import from a submodule? "
        f"Try: from insideLLMs.<submodule> import {name}"
    )
