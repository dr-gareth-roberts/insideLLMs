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
    - insideLLMs.evaluation: Evaluation metrics and evaluators
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
"""

__version__ = "0.1.0"

# =============================================================================
# Core Types (Essential for all users)
# =============================================================================
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
    AttackProbe,
    BiasProbe,
    CustomProbe,
    FactualityProbe,
    LogicProbe,
    Probe,
)

# =============================================================================
# Runner and Execution (Essential for running experiments)
# =============================================================================
from insideLLMs.runner import (
    AsyncProbeRunner,
    ProbeRunner,
    create_experiment_result,
    run_probe,
)

# =============================================================================
# Results and Reporting (Essential for saving/loading results)
# =============================================================================
from insideLLMs.results import (
    load_results_json,
    save_results_json,
)

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
# Registry (For plugin system)
# =============================================================================
from insideLLMs.registry import (
    Registry,
    model_registry,
    probe_registry,
)

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
    "AttackProbe",
    "BiasProbe",
    "CustomProbe",
    "FactualityProbe",
    "LogicProbe",
    "Probe",
    # Runner
    "AsyncProbeRunner",
    "ProbeRunner",
    "create_experiment_result",
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
        # Evaluation
        "Evaluator": "insideLLMs.evaluation",
        "ExactMatchEvaluator": "insideLLMs.evaluation",
        "bleu_score": "insideLLMs.evaluation",
        "rouge_l": "insideLLMs.evaluation",
        "token_f1": "insideLLMs.evaluation",
        "exact_match": "insideLLMs.evaluation",
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
