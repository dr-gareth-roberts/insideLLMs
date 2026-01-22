"""insideLLMs - A world-class library for probing the inner workings of large language models.

This library provides tools for systematically evaluating LLMs across multiple dimensions:
- Logic & Reasoning: Zero-shot ability on unseen logic problems
- Bias Detection: Propensity for bias in responses
- Attack Vulnerabilities: Susceptibilities to prompt injection and adversarial inputs
- Factual Accuracy: Correctness of factual knowledge
- Benchmarking: Side-by-side model comparison

Quick Start:
    >>> from insideLLMs import DummyModel, LogicProbe, ProbeRunner
    >>> model = DummyModel()
    >>> probe = LogicProbe()
    >>> runner = ProbeRunner(model, probe)
    >>> results = runner.run(["What comes next: 1, 2, 3, ?"])

For more specialized features, import from submodules:
    >>> from insideLLMs.evaluation import bleu_score, rouge_l
    >>> from insideLLMs.caching import InMemoryCache, DiskCache
    >>> from insideLLMs.adversarial import RobustnessTester
    >>> from insideLLMs.safety import detect_pii, quick_safety_check

Available Submodules:
    - insideLLMs.models: Model implementations (OpenAI, Anthropic, HuggingFace, etc.)
    - insideLLMs.probes: Probe implementations (Logic, Bias, Attack, etc.)
    - insideLLMs.evaluation: Evaluation metrics and evaluators (includes LLM-as-a-Judge)
    - insideLLMs.caching: Caching utilities (InMemoryCache, DiskCache, etc.)
    - insideLLMs.safety: Safety analysis (PII detection, toxicity, etc.)
    - insideLLMs.adversarial: Adversarial testing and robustness
    - insideLLMs.knowledge: Knowledge probing and fact verification
    - insideLLMs.reasoning: Chain-of-thought and reasoning analysis
    - insideLLMs.optimization: Prompt optimization utilities
    - insideLLMs.statistics: Statistical analysis tools
    - insideLLMs.visualization: Visualization and reporting
    - insideLLMs.streaming: Output streaming utilities
    - insideLLMs.distributed: Distributed execution
    - insideLLMs.templates: Prompt template library
    - insideLLMs.benchmark_datasets: Benchmark dataset utilities
    - insideLLMs.structured: Structured output parsing with Pydantic integration
    - insideLLMs.observability: OpenTelemetry tracing and telemetry
    - insideLLMs.synthesis: Synthetic data generation and augmentation
    - insideLLMs.semantic_cache: Semantic caching with vector similarity
    - insideLLMs.agents: Autonomous agents (ReAct, Tool Use, Chain-of-Thought)
    - insideLLMs.routing: Model routing with semantic matching
    - insideLLMs.hitl: Human-in-the-loop workflows and feedback
    - insideLLMs.deployment: FastAPI deployment wrapper (optional)
"""

__version__ = "0.1.0"

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
from insideLLMs.runner import (
    AsyncProbeRunner,
    ProbeRunner,
    create_experiment_result,
    run_harness_from_config,
    run_probe,
)

# =============================================================================
# Trace Configuration (For deterministic CI enforcement)
# =============================================================================
from insideLLMs.trace_config import (
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
]


# =============================================================================
# Lazy Loading for Heavy Submodules
# =============================================================================
def __getattr__(name: str):
    """Lazy load submodules and provide helpful error messages."""
    # Map of commonly requested items to their submodule paths
    _LAZY_IMPORTS = {
        # Models
        "OpenAIModel": "insideLLMs.models",
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
        "InMemoryCache": "insideLLMs.cache",
        "DiskCache": "insideLLMs.cache",
        "CachedModel": "insideLLMs.cache",
        "cached": "insideLLMs.cache",
        "BaseCache": "insideLLMs.cache",
        "CacheEntry": "insideLLMs.cache",
        "CacheStats": "insideLLMs.cache",
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
        "descriptive_statistics": "insideLLMs.statistics",
        "compare_experiments": "insideLLMs.statistics",
        "confidence_interval": "insideLLMs.statistics",
        # Safety
        "detect_pii": "insideLLMs.safety",
        "quick_safety_check": "insideLLMs.safety",
        "ContentSafetyAnalyzer": "insideLLMs.safety",
        # Visualization
        "plot_accuracy_comparison": "insideLLMs.visualization",
        "create_html_report": "insideLLMs.visualization",
        "text_comparison_table": "insideLLMs.visualization",
        # Benchmarking
        "ModelBenchmark": "insideLLMs.benchmark",
        "ProbeBenchmark": "insideLLMs.benchmark",
        # Synthetic Data Generation
        "PromptVariator": "insideLLMs.synthesis",
        "AdversarialGenerator": "insideLLMs.synthesis",
        "DataAugmenter": "insideLLMs.synthesis",
        "TemplateGenerator": "insideLLMs.synthesis",
        "SyntheticDataset": "insideLLMs.synthesis",
        "quick_variations": "insideLLMs.synthesis",
        "quick_adversarial": "insideLLMs.synthesis",
        "generate_test_dataset": "insideLLMs.synthesis",
        "SynthesisConfig": "insideLLMs.synthesis",
        "VariationStrategy": "insideLLMs.synthesis",
        "AdversarialType": "insideLLMs.synthesis",
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
        "ReActAgent": "insideLLMs.agents",
        "SimpleAgent": "insideLLMs.agents",
        "ChainOfThoughtAgent": "insideLLMs.agents",
        "Tool": "insideLLMs.agents",
        "ToolRegistry": "insideLLMs.agents",
        "AgentExecutor": "insideLLMs.agents",
        "AgentConfig": "insideLLMs.agents",
        "AgentResult": "insideLLMs.agents",
        "create_react_agent": "insideLLMs.agents",
        "create_simple_agent": "insideLLMs.agents",
        "create_calculator_tool": "insideLLMs.agents",
        "quick_agent_run": "insideLLMs.agents",
        # Human-in-the-Loop
        "HITLSession": "insideLLMs.hitl",
        "InteractiveSession": "insideLLMs.hitl",
        "ApprovalWorkflow": "insideLLMs.hitl",
        "ReviewWorkflow": "insideLLMs.hitl",
        "AnnotationWorkflow": "insideLLMs.hitl",
        "ReviewQueue": "insideLLMs.hitl",
        "PriorityReviewQueue": "insideLLMs.hitl",
        "HumanValidator": "insideLLMs.hitl",
        "ConsensusValidator": "insideLLMs.hitl",
        "FeedbackCollector": "insideLLMs.hitl",
        "AnnotationCollector": "insideLLMs.hitl",
        "HITLConfig": "insideLLMs.hitl",
        "FeedbackType": "insideLLMs.hitl",
        "ReviewStatus": "insideLLMs.hitl",
        "Priority": "insideLLMs.hitl",
        "Feedback": "insideLLMs.hitl",
        "ReviewItem": "insideLLMs.hitl",
        "Annotation": "insideLLMs.hitl",
        "create_hitl_session": "insideLLMs.hitl",
        "quick_review": "insideLLMs.hitl",
        "collect_feedback": "insideLLMs.hitl",
        # Model Routing
        "SemanticRouter": "insideLLMs.routing",
        "Route": "insideLLMs.routing",
        "RouteMatch": "insideLLMs.routing",
        "ModelPool": "insideLLMs.routing",
        "IntentClassifier": "insideLLMs.routing",
        "RouterConfig": "insideLLMs.routing",
        "RoutingStrategy": "insideLLMs.routing",
        "create_router": "insideLLMs.routing",
        "quick_route": "insideLLMs.routing",
        # Deployment (FastAPI)
        "create_app": "insideLLMs.deployment",
        "DeploymentApp": "insideLLMs.deployment",
        "AppConfig": "insideLLMs.deployment",
        "ModelEndpoint": "insideLLMs.deployment",
        "ProbeEndpoint": "insideLLMs.deployment",
        "BatchEndpoint": "insideLLMs.deployment",
        "DeploymentConfig": "insideLLMs.deployment",
        "EndpointConfig": "insideLLMs.deployment",
        "RateLimiter": "insideLLMs.deployment",
        "APIKeyAuth": "insideLLMs.deployment",
        "MetricsCollector": "insideLLMs.deployment",
        "HealthChecker": "insideLLMs.deployment",
        "quick_deploy": "insideLLMs.deployment",
        "create_model_endpoint": "insideLLMs.deployment",
        "create_probe_endpoint": "insideLLMs.deployment",
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
