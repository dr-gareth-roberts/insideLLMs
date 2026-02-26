"""Configuration validation and management using Pydantic.

This module provides strongly-typed configuration classes for running
experiments that probe language model capabilities. It includes:

- **ModelConfig**: Configuration for connecting to language model APIs
- **ProbeConfig**: Configuration for defining evaluation probes
- **DatasetConfig**: Configuration for data sources (files, HuggingFace, inline)
- **RunnerConfig**: Configuration for experiment execution settings
- **ExperimentConfig**: Top-level configuration combining all components

All configurations support:
- Pydantic validation with automatic type coercion
- Serialization to/from YAML and JSON
- Loading from files with auto-format detection
- Fallback implementations when Pydantic is not available

Examples
--------
Creating a minimal experiment configuration:

>>> from insideLLMs.config import (
...     ExperimentConfig, ModelConfig, ProbeConfig, DatasetConfig
... )
>>> config = ExperimentConfig(
...     name="Simple Logic Test",
...     model=ModelConfig(provider="openai", model_id="gpt-4"),
...     probe=ProbeConfig(type="logic"),
...     dataset=DatasetConfig(source="inline", data=[{"q": "2+2=?"}])
... )

Loading configuration from a YAML file:

>>> from insideLLMs.config import load_config
>>> config = load_config("experiments/my_experiment.yaml")

Creating and saving a complete configuration:

>>> from insideLLMs.config import (
...     ExperimentConfig, ModelConfig, ProbeConfig,
...     DatasetConfig, RunnerConfig, save_config_to_yaml
... )
>>> config = ExperimentConfig(
...     name="Full Experiment",
...     description="A comprehensive logic evaluation",
...     model=ModelConfig(
...         provider="anthropic",
...         model_id="claude-3-opus-20240229",
...         api_key_env="ANTHROPIC_API_KEY",
...         temperature=0.0,
...         max_tokens=2000
...     ),
...     probe=ProbeConfig(
...         type="logic",
...         name="Logical Reasoning",
...         params={"difficulty": "hard", "include_hints": False}
...     ),
...     dataset=DatasetConfig(
...         source="file",
...         path="data/logic_questions.jsonl",
...         sample_size=100,
...         shuffle=True,
...         seed=42
...     ),
...     runner=RunnerConfig(
...         concurrency=10,
...         output_dir="results/logic_test",
...         output_formats=["json", "markdown", "csv"],
...         verbose=True
...     ),
...     tags=["logic", "reasoning", "claude-3"]
... )
>>> save_config_to_yaml(config, "experiments/full_experiment.yaml")

Using the HuggingFace dataset source:

>>> dataset_config = DatasetConfig(
...     source="hf",
...     name="truthful_qa",
...     split="validation",
...     sample_size=50
... )

Validating a configuration dictionary:

>>> from insideLLMs.config import validate_config
>>> config_dict = {
...     "name": "Test",
...     "model": {"provider": "dummy", "model_id": "test-model"},
...     "probe": {"type": "factuality"},
...     "dataset": {"source": "inline", "data": [{"text": "sample"}]}
... }
>>> validated = validate_config(config_dict)

Notes
-----
When Pydantic is not available, fallback implementations are provided
that offer basic functionality without validation. The fallback classes
use simple attribute assignment and do not perform type checking.

See Also
--------
load_config : Load configuration with auto-format detection
save_config_to_yaml : Save configuration to YAML format
save_config_to_json : Save configuration to JSON format
create_example_config : Generate a complete example configuration
"""

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Union

from insideLLMs._serialization import serialize_value as _serialize_value

try:
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        field_validator,
        model_validator,
    )

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

    # Provide fallback base class
    class BaseModel:  # type: ignore[no-redef]  # Intentional: stub when Pydantic unavailable
        """Fallback BaseModel when Pydantic is not available."""

        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

        def model_dump(self) -> dict[str, Any]:
            return self.__dict__.copy()

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]  # Intentional: stub when Pydantic unavailable
        return kwargs.get("default")

    def field_validator(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]  # Intentional: stub when Pydantic unavailable
        def decorator(func: Any) -> Any:
            return func

        return decorator

    def model_validator(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]  # Intentional: stub when Pydantic unavailable
        def decorator(func: Any) -> Any:
            return func

        return decorator

    class ConfigDict:  # type: ignore[no-redef]  # Intentional: stub when Pydantic unavailable
        pass


def _require_pydantic() -> None:
    """Raise if Pydantic is not available.

    This is an internal helper used by tests and by optional features that
    depend on Pydantic. The main configuration surface provides fallbacks when
    Pydantic is not installed, but some integrations may require it.
    """
    if not PYDANTIC_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "Pydantic is required for configuration validation. Install with: pip install pydantic"
        )


class ModelProvider(str, Enum):
    """Enumeration of supported model providers for API access.

    This enum defines the available providers that can be used to connect
    to language model APIs. Each provider has specific requirements for
    authentication and configuration.

    Attributes
    ----------
    OPENAI : str
        OpenAI API (GPT-4, GPT-3.5, etc.). Requires OPENAI_API_KEY.
    ANTHROPIC : str
        Anthropic API (Claude models). Requires ANTHROPIC_API_KEY.
    HUGGINGFACE : str
        HuggingFace Inference API or local models. May require HF_TOKEN.
    DUMMY : str
        Mock provider for testing without API calls.
    CUSTOM : str
        Custom provider implementation for specialized use cases.

    Examples
    --------
    Using provider enum with ModelConfig:

    >>> from insideLLMs.config import ModelConfig, ModelProvider
    >>> config = ModelConfig(
    ...     provider=ModelProvider.OPENAI,
    ...     model_id="gpt-4-turbo"
    ... )

    Checking provider type:

    >>> config.provider == ModelProvider.OPENAI
    True
    >>> config.provider.value
    'openai'

    Using string value (auto-coerced by Pydantic):

    >>> config = ModelConfig(provider="anthropic", model_id="claude-3-opus")
    >>> config.provider
    <ModelProvider.ANTHROPIC: 'anthropic'>

    Iterating over all providers:

    >>> for provider in ModelProvider:
    ...     print(f"{provider.name}: {provider.value}")
    OPENAI: openai
    ANTHROPIC: anthropic
    HUGGINGFACE: huggingface
    DUMMY: dummy
    CUSTOM: custom
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    DUMMY = "dummy"
    CUSTOM = "custom"


class ProbeType(str, Enum):
    """Enumeration of available probe types for model evaluation.

    Probes are evaluation frameworks that test specific aspects of
    language model capabilities. Each probe type has its own methodology
    and metrics.

    Attributes
    ----------
    LOGIC : str
        Tests logical reasoning and inference capabilities.
        Evaluates syllogisms, mathematical reasoning, and deduction.
    FACTUALITY : str
        Tests factual accuracy and knowledge retrieval.
        Evaluates truthfulness and factual grounding of responses.
    BIAS : str
        Tests for various forms of bias in model outputs.
        Evaluates demographic, cultural, and ideological biases.
    ATTACK : str
        Tests robustness against adversarial inputs.
        Evaluates prompt injection and jailbreak resistance.
    CUSTOM : str
        Custom probe implementation for specialized evaluations.

    Examples
    --------
    Creating a logic probe configuration:

    >>> from insideLLMs.config import ProbeConfig, ProbeType
    >>> probe = ProbeConfig(
    ...     type=ProbeType.LOGIC,
    ...     params={"difficulty": "hard", "categories": ["syllogisms"]}
    ... )

    Creating a factuality probe with custom settings:

    >>> probe = ProbeConfig(
    ...     type=ProbeType.FACTUALITY,
    ...     name="Scientific Facts Test",
    ...     params={"domain": "science", "verify_sources": True}
    ... )

    Using string value (auto-coerced):

    >>> probe = ProbeConfig(type="bias")
    >>> probe.type
    <ProbeType.BIAS: 'bias'>

    Checking probe type:

    >>> if probe.type == ProbeType.BIAS:
    ...     print("Running bias evaluation")
    Running bias evaluation
    """

    LOGIC = "logic"
    FACTUALITY = "factuality"
    BIAS = "bias"
    ATTACK = "attack"
    CUSTOM = "custom"


if PYDANTIC_AVAILABLE:

    class ModelConfig(BaseModel):
        """Configuration for connecting to a language model API.

        This class defines all parameters needed to connect to and interact
        with a language model through various provider APIs. It supports
        OpenAI, Anthropic, HuggingFace, and custom providers.

        Attributes
        ----------
        provider : ModelProvider
            The API provider (openai, anthropic, huggingface, dummy, custom).
        model_id : str
            The specific model identifier (e.g., "gpt-4", "claude-3-opus-20240229").
        name : str, optional
            Display name for the model. Defaults to model_id if not provided.
        api_key_env : str, optional
            Environment variable name containing the API key.
        api_base : str, optional
            Custom API base URL for self-hosted or proxy endpoints.
        temperature : float
            Sampling temperature between 0.0 and 2.0. Default is 0.7.
        max_tokens : int, optional
            Maximum tokens to generate. None means use provider default.
        timeout : float
            Request timeout in seconds. Default is 30.0.
        max_retries : int
            Maximum retry attempts for failed requests. Default is 3.
        extra_params : dict
            Additional provider-specific parameters passed to the API.

        Examples
        --------
        Minimal configuration with just provider and model:

        >>> config = ModelConfig(
        ...     provider="openai",
        ...     model_id="gpt-4"
        ... )

        Full OpenAI configuration with custom settings:

        >>> config = ModelConfig(
        ...     provider="openai",
        ...     model_id="gpt-4-turbo",
        ...     name="GPT-4 Turbo (Low Temp)",
        ...     api_key_env="OPENAI_API_KEY",
        ...     temperature=0.0,
        ...     max_tokens=4096,
        ...     timeout=60.0,
        ...     max_retries=5,
        ...     extra_params={"top_p": 0.9, "presence_penalty": 0.1}
        ... )

        Anthropic Claude configuration:

        >>> config = ModelConfig(
        ...     provider="anthropic",
        ...     model_id="claude-3-opus-20240229",
        ...     api_key_env="ANTHROPIC_API_KEY",
        ...     temperature=0.5,
        ...     max_tokens=2000,
        ...     extra_params={"top_k": 40}
        ... )

        Custom endpoint configuration (e.g., Azure OpenAI or local server):

        >>> config = ModelConfig(
        ...     provider="custom",
        ...     model_id="my-fine-tuned-model",
        ...     api_base="https://my-server.example.com/v1",
        ...     api_key_env="CUSTOM_API_KEY",
        ...     timeout=120.0
        ... )

        Dummy provider for testing (no API calls):

        >>> config = ModelConfig(
        ...     provider="dummy",
        ...     model_id="test-model",
        ...     name="Test Model"
        ... )

        Notes
        -----
        The `extra` config is set to "allow", meaning additional fields
        can be passed and will be stored without validation errors.
        This enables forward compatibility with new provider features.

        See Also
        --------
        ModelProvider : Enum of supported providers
        ExperimentConfig : Parent configuration using ModelConfig
        """

        model_config = ConfigDict(extra="allow", validate_default=True, protected_namespaces=())

        provider: ModelProvider = Field(
            description="Model provider (openai, anthropic, huggingface, dummy)"
        )
        model_id: str = Field(description="Model identifier (e.g., 'gpt-4', 'claude-3-opus')")
        name: Optional[str] = Field(default=None, description="Optional display name for the model")
        api_key_env: Optional[str] = Field(
            default=None, description="Environment variable name containing API key"
        )
        api_base: Optional[str] = Field(default=None, description="Optional custom API base URL")
        temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
        max_tokens: Optional[int] = Field(
            default=None, gt=0, description="Maximum tokens to generate"
        )
        timeout: float = Field(default=30.0, gt=0, description="Request timeout in seconds")
        max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
        extra_params: dict[str, Any] = Field(
            default_factory=dict, description="Additional provider-specific parameters"
        )

        @field_validator("name", mode="before")
        @classmethod
        def set_default_name(cls, v: Optional[str], info: Any) -> str:
            if v is None:
                return info.data.get("model_id", "unknown")
            return v

    class ProbeConfig(BaseModel):
        """Configuration for a model evaluation probe.

        Probes are evaluation frameworks that assess specific capabilities
        of language models. This configuration defines the probe type,
        display settings, and probe-specific parameters.

        Attributes
        ----------
        type : ProbeType
            The type of probe to execute (logic, factuality, bias, attack, custom).
        name : str, optional
            Display name for the probe. Defaults to the probe type if not provided.
        description : str, optional
            Human-readable description of what this probe evaluates.
        params : dict
            Probe-specific parameters. Contents depend on the probe type.
        timeout_per_item : float
            Timeout in seconds for each individual probe item. Default is 30.0.
        stop_on_error : bool
            If True, stop the probe execution on first error. Default is False.

        Examples
        --------
        Basic logic probe configuration:

        >>> config = ProbeConfig(
        ...     type="logic",
        ...     name="Basic Logic Test"
        ... )

        Logic probe with difficulty and category settings:

        >>> config = ProbeConfig(
        ...     type="logic",
        ...     name="Advanced Syllogisms",
        ...     description="Tests complex syllogistic reasoning",
        ...     params={
        ...         "difficulty": "hard",
        ...         "categories": ["syllogisms", "conditionals"],
        ...         "require_explanation": True
        ...     },
        ...     timeout_per_item=60.0
        ... )

        Factuality probe for scientific knowledge:

        >>> config = ProbeConfig(
        ...     type="factuality",
        ...     name="Science Facts",
        ...     description="Evaluates scientific knowledge accuracy",
        ...     params={
        ...         "domains": ["physics", "chemistry", "biology"],
        ...         "verify_with_sources": True,
        ...         "allow_uncertainty": True
        ...     }
        ... )

        Bias detection probe:

        >>> config = ProbeConfig(
        ...     type="bias",
        ...     name="Demographic Bias Test",
        ...     params={
        ...         "bias_types": ["gender", "race", "age"],
        ...         "sensitivity": "high",
        ...         "include_counterfactuals": True
        ...     },
        ...     stop_on_error=True
        ... )

        Adversarial attack probe:

        >>> config = ProbeConfig(
        ...     type="attack",
        ...     name="Jailbreak Resistance",
        ...     params={
        ...         "attack_types": ["prompt_injection", "jailbreak"],
        ...         "intensity": "medium"
        ...     },
        ...     timeout_per_item=120.0
        ... )

        Notes
        -----
        The `params` dictionary is intentionally flexible to accommodate
        different probe types' requirements. Refer to specific probe
        implementations for their expected parameters.

        See Also
        --------
        ProbeType : Enum of available probe types
        ExperimentConfig : Parent configuration using ProbeConfig
        """

        model_config = ConfigDict(extra="allow")

        type: ProbeType = Field(description="Type of probe to use")
        name: Optional[str] = Field(default=None, description="Optional display name")
        description: Optional[str] = Field(default=None, description="Optional description")
        params: dict[str, Any] = Field(
            default_factory=dict, description="Probe-specific parameters"
        )
        timeout_per_item: float = Field(
            default=30.0, gt=0, description="Timeout per probe item in seconds"
        )
        stop_on_error: bool = Field(default=False, description="Stop execution on first error")

        @field_validator("name", mode="before")
        @classmethod
        def set_default_name(cls, v: Optional[str], info: Any) -> str:
            if v is None:
                return info.data.get("type", "unknown")
            return v

    class DatasetConfig(BaseModel):
        """Configuration for loading evaluation datasets.

        Supports three data sources: local files, HuggingFace datasets,
        and inline data. The configuration handles data loading, sampling,
        and shuffling.

        Attributes
        ----------
        source : Literal["file", "hf", "inline"]
            The data source type. Must be one of:
            - "file": Load from a local file (JSONL, JSON, CSV)
            - "hf": Load from HuggingFace datasets
            - "inline": Use data provided directly in configuration
        path : str, optional
            Path to the dataset file. Required when source="file".
        name : str, optional
            HuggingFace dataset name. Required when source="hf".
        split : str
            Dataset split to use (e.g., "train", "test", "validation").
            Default is "test".
        data : list[dict], optional
            Inline data as list of dictionaries. Required when source="inline".
        sample_size : int, optional
            Number of samples to use. None means use all available data.
        shuffle : bool
            Whether to shuffle the data before sampling. Default is False.
        seed : int, optional
            Random seed for reproducible shuffling.

        Examples
        --------
        Loading from a local JSONL file:

        >>> config = DatasetConfig(
        ...     source="file",
        ...     path="data/logic_questions.jsonl"
        ... )

        Loading from a local file with sampling:

        >>> config = DatasetConfig(
        ...     source="file",
        ...     path="data/large_dataset.jsonl",
        ...     sample_size=100,
        ...     shuffle=True,
        ...     seed=42
        ... )

        Loading from HuggingFace datasets:

        >>> config = DatasetConfig(
        ...     source="hf",
        ...     name="truthful_qa",
        ...     split="validation",
        ...     sample_size=200
        ... )

        Loading a specific HuggingFace dataset configuration:

        >>> config = DatasetConfig(
        ...     source="hf",
        ...     name="super_glue",
        ...     split="test",
        ...     sample_size=500,
        ...     shuffle=True,
        ...     seed=123
        ... )

        Using inline data for quick tests:

        >>> config = DatasetConfig(
        ...     source="inline",
        ...     data=[
        ...         {"question": "What is 2+2?", "answer": "4"},
        ...         {"question": "Is the sky blue?", "answer": "yes"},
        ...         {"question": "What comes after Monday?", "answer": "Tuesday"}
        ...     ]
        ... )

        Inline data with complex structure:

        >>> config = DatasetConfig(
        ...     source="inline",
        ...     data=[
        ...         {
        ...             "prompt": "Complete the syllogism: All A are B. All B are C.",
        ...             "expected": "All A are C",
        ...             "category": "syllogism",
        ...             "difficulty": "easy"
        ...         },
        ...         {
        ...             "prompt": "If P then Q. P is true. What follows?",
        ...             "expected": "Q is true",
        ...             "category": "modus_ponens",
        ...             "difficulty": "easy"
        ...         }
        ...     ]
        ... )

        Raises
        ------
        ValueError
            If source requirements are not met:
            - source="file" requires path
            - source="hf" requires name
            - source="inline" requires data

        Notes
        -----
        The validation ensures that the appropriate fields are provided
        based on the source type. Extra fields are allowed for flexibility.

        See Also
        --------
        ExperimentConfig : Parent configuration using DatasetConfig
        """

        model_config = ConfigDict(extra="allow")

        source: Literal["file", "hf", "inline"] = Field(description="Dataset source type")
        path: Optional[str] = Field(
            default=None, description="Path to dataset file (for file source)"
        )
        name: Optional[str] = Field(default=None, description="Dataset name (for hf source)")
        split: str = Field(default="test", description="Dataset split to use")
        data: Optional[list[dict[str, Any]]] = Field(
            default=None, description="Inline data (for inline source)"
        )
        sample_size: Optional[int] = Field(
            default=None, gt=0, description="Number of samples to use (None for all)"
        )
        shuffle: bool = Field(default=False, description="Whether to shuffle data")
        seed: Optional[int] = Field(default=None, description="Random seed for shuffling")

        @model_validator(mode="after")
        def validate_source_requirements(self) -> "DatasetConfig":
            if self.source == "file" and not self.path:
                raise ValueError("path is required for file source")
            if self.source == "hf" and not self.name:
                raise ValueError("name is required for hf source")
            if self.source == "inline" and not self.data:
                raise ValueError("data is required for inline source")
            return self

    class RunnerConfig(BaseModel):
        """Configuration for experiment execution and output settings.

        Controls how experiments are executed, including concurrency,
        output formats, and progress reporting. This configuration
        affects performance and resource usage.

        Attributes
        ----------
        concurrency : int
            Number of concurrent API requests. Higher values speed up
            execution but may hit rate limits. Default is 1.
        output_dir : str
            Directory for saving experiment results. Default is "output".
        output_formats : list[str]
            List of output formats to generate. Supported: "json", "markdown",
            "csv", "html". Default is ["json", "markdown"].
        save_intermediate : bool
            If True, save results after each probe item. Useful for long
            experiments. Default is False.
        progress_bar : bool
            If True, display a progress bar during execution. Default is True.
        verbose : bool
            If True, enable verbose logging output. Default is False.
        cache_responses : bool
            If True, cache model responses to avoid duplicate API calls.
            Default is False.

        Examples
        --------
        Default runner configuration:

        >>> config = RunnerConfig()
        >>> config.concurrency
        1
        >>> config.output_dir
        'output'

        High-throughput configuration for large experiments:

        >>> config = RunnerConfig(
        ...     concurrency=10,
        ...     output_dir="results/large_experiment",
        ...     output_formats=["json", "csv"],
        ...     save_intermediate=True,
        ...     cache_responses=True
        ... )

        Debug configuration with verbose output:

        >>> config = RunnerConfig(
        ...     concurrency=1,
        ...     output_dir="debug_output",
        ...     output_formats=["json", "markdown"],
        ...     verbose=True,
        ...     progress_bar=True
        ... )

        Minimal output configuration for CI/CD:

        >>> config = RunnerConfig(
        ...     concurrency=5,
        ...     output_dir="ci_results",
        ...     output_formats=["json"],
        ...     progress_bar=False,
        ...     verbose=False
        ... )

        Full HTML report generation:

        >>> config = RunnerConfig(
        ...     output_dir="reports/monthly",
        ...     output_formats=["json", "markdown", "csv", "html"],
        ...     save_intermediate=False
        ... )

        Notes
        -----
        When using high concurrency values, be aware of API rate limits.
        Most providers have per-minute or per-hour request limits.
        The cache_responses option can help reduce API calls when
        running similar experiments multiple times.

        See Also
        --------
        ExperimentConfig : Parent configuration using RunnerConfig
        """

        model_config = ConfigDict(extra="allow")

        concurrency: int = Field(default=1, ge=1, description="Number of concurrent executions")
        output_dir: str = Field(default="output", description="Directory for output files")
        output_formats: list[Literal["json", "markdown", "csv", "html"]] = Field(
            default=["json", "markdown"], description="Output formats to generate"
        )
        save_intermediate: bool = Field(default=False, description="Save intermediate results")
        progress_bar: bool = Field(default=True, description="Show progress bar")
        verbose: bool = Field(default=False, description="Verbose logging")
        cache_responses: bool = Field(default=False, description="Cache model responses")

    class ExperimentConfig(BaseModel):
        """Complete experiment configuration combining all components.

        This is the top-level configuration class that brings together model,
        probe, dataset, and runner configurations into a single cohesive
        experiment definition. It represents everything needed to run an
        evaluation experiment.

        Attributes
        ----------
        name : str
            Unique name for the experiment. Used in output filenames and reports.
        description : str, optional
            Human-readable description of the experiment's purpose.
        model : ModelConfig
            Configuration for the language model to evaluate.
        probe : ProbeConfig
            Configuration for the evaluation probe to run.
        dataset : DatasetConfig
            Configuration for the evaluation data source.
        runner : RunnerConfig
            Configuration for experiment execution. Uses defaults if not provided.
        tags : list[str]
            Tags for organizing and filtering experiments. Default is empty list.
        metadata : dict
            Additional metadata for custom tracking. Default is empty dict.

        Examples
        --------
        Minimal experiment configuration:

        >>> config = ExperimentConfig(
        ...     name="Quick Test",
        ...     model=ModelConfig(provider="openai", model_id="gpt-4"),
        ...     probe=ProbeConfig(type="logic"),
        ...     dataset=DatasetConfig(source="inline", data=[{"q": "2+2=?"}])
        ... )

        Complete GPT-4 logic evaluation:

        >>> config = ExperimentConfig(
        ...     name="GPT-4 Logic Evaluation",
        ...     description="Comprehensive logic reasoning test on GPT-4",
        ...     model=ModelConfig(
        ...         provider="openai",
        ...         model_id="gpt-4-turbo",
        ...         temperature=0.0,
        ...         max_tokens=1000
        ...     ),
        ...     probe=ProbeConfig(
        ...         type="logic",
        ...         name="Logic Suite",
        ...         params={"difficulty": "hard"}
        ...     ),
        ...     dataset=DatasetConfig(
        ...         source="file",
        ...         path="data/logic_benchmark.jsonl",
        ...         sample_size=500
        ...     ),
        ...     runner=RunnerConfig(
        ...         concurrency=5,
        ...         output_dir="results/gpt4_logic"
        ...     ),
        ...     tags=["gpt-4", "logic", "benchmark"],
        ...     metadata={"version": "1.0", "author": "research-team"}
        ... )

        Claude model comparison experiment:

        >>> config = ExperimentConfig(
        ...     name="Claude-3 Factuality Test",
        ...     description="Testing Claude-3 Opus factual accuracy",
        ...     model=ModelConfig(
        ...         provider="anthropic",
        ...         model_id="claude-3-opus-20240229",
        ...         api_key_env="ANTHROPIC_API_KEY",
        ...         temperature=0.0
        ...     ),
        ...     probe=ProbeConfig(
        ...         type="factuality",
        ...         params={"strict_mode": True}
        ...     ),
        ...     dataset=DatasetConfig(
        ...         source="hf",
        ...         name="truthful_qa",
        ...         split="validation"
        ...     ),
        ...     tags=["claude-3", "factuality", "truthful_qa"]
        ... )

        Bias evaluation with detailed metadata:

        >>> config = ExperimentConfig(
        ...     name="Bias Audit Q4 2024",
        ...     description="Quarterly bias evaluation for compliance",
        ...     model=ModelConfig(provider="openai", model_id="gpt-4"),
        ...     probe=ProbeConfig(
        ...         type="bias",
        ...         params={"bias_types": ["gender", "race"]}
        ...     ),
        ...     dataset=DatasetConfig(
        ...         source="file",
        ...         path="data/bias_prompts.jsonl"
        ...     ),
        ...     runner=RunnerConfig(
        ...         output_formats=["json", "markdown", "html"]
        ...     ),
        ...     tags=["bias", "compliance", "audit", "q4-2024"],
        ...     metadata={
        ...         "compliance_framework": "AI-RMF",
        ...         "reviewer": "ethics-team",
        ...         "priority": "high"
        ...     }
        ... )

        Testing configuration for development:

        >>> config = ExperimentConfig(
        ...     name="Dev Test",
        ...     model=ModelConfig(provider="dummy", model_id="test"),
        ...     probe=ProbeConfig(type="logic"),
        ...     dataset=DatasetConfig(
        ...         source="inline",
        ...         data=[{"q": "test question"}]
        ...     ),
        ...     runner=RunnerConfig(verbose=True),
        ...     tags=["dev", "test"]
        ... )

        Notes
        -----
        The runner configuration defaults to RunnerConfig() if not provided,
        which uses sensible defaults for most use cases.

        When serializing to YAML/JSON, nested configurations are properly
        serialized as nested dictionaries.

        See Also
        --------
        ModelConfig : Model connection settings
        ProbeConfig : Probe evaluation settings
        DatasetConfig : Data source settings
        RunnerConfig : Execution settings
        load_config : Load configuration from file
        save_config_to_yaml : Save configuration to YAML
        """

        model_config = ConfigDict(extra="allow")

        name: str = Field(description="Experiment name")
        description: Optional[str] = Field(default=None, description="Experiment description")
        model: ModelConfig = Field(description="Model configuration")
        probe: ProbeConfig = Field(description="Probe configuration")
        dataset: DatasetConfig = Field(description="Dataset configuration")
        runner: RunnerConfig = Field(
            default_factory=RunnerConfig, description="Runner configuration"
        )
        tags: list[str] = Field(default_factory=list, description="Tags for organizing experiments")
        metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

else:
    # Fallback implementations without Pydantic validation
    class ModelConfig(BaseModel):  # type: ignore[no-redef]  # Intentional: fallback class when Pydantic unavailable
        """Configuration for connecting to a language model API (fallback).

        This is a fallback implementation used when Pydantic is not available.
        It provides the same interface as the Pydantic version but without
        automatic validation.

        Parameters
        ----------
        provider : str
            The API provider (openai, anthropic, huggingface, dummy, custom).
        model_id : str
            The specific model identifier.
        name : str, optional
            Display name for the model. Defaults to model_id.
        api_key_env : str, optional
            Environment variable name containing the API key.
        api_base : str, optional
            Custom API base URL.
        temperature : float
            Sampling temperature. Default is 0.7.
        max_tokens : int, optional
            Maximum tokens to generate.
        timeout : float
            Request timeout in seconds. Default is 30.0.
        max_retries : int
            Maximum retry attempts. Default is 3.
        extra_params : dict, optional
            Additional provider-specific parameters.

        Examples
        --------
        >>> config = ModelConfig(provider="openai", model_id="gpt-4")
        >>> config = ModelConfig(
        ...     provider="anthropic",
        ...     model_id="claude-3-opus-20240229",
        ...     temperature=0.0,
        ...     max_tokens=2000
        ... )

        Notes
        -----
        This fallback class does not perform validation. Use the Pydantic
        version (install pydantic) for automatic type checking and validation.
        """

        def __init__(
            self,
            provider: str,
            model_id: str,
            name: Optional[str] = None,
            api_key_env: Optional[str] = None,
            api_base: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            timeout: float = 30.0,
            max_retries: int = 3,
            extra_params: Optional[dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            self.provider = provider
            self.model_id = model_id
            self.name = name or model_id
            self.api_key_env = api_key_env
            self.api_base = api_base
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.timeout = timeout
            self.max_retries = max_retries
            self.extra_params = extra_params or {}

    class ProbeConfig(BaseModel):  # type: ignore[no-redef]  # Intentional: fallback class when Pydantic unavailable
        """Configuration for a model evaluation probe (fallback).

        This is a fallback implementation used when Pydantic is not available.
        It provides the same interface as the Pydantic version but without
        automatic validation.

        Parameters
        ----------
        type : str
            The type of probe (logic, factuality, bias, attack, custom).
        name : str, optional
            Display name for the probe. Defaults to the type.
        description : str, optional
            Human-readable description of the probe.
        params : dict, optional
            Probe-specific parameters.
        timeout_per_item : float
            Timeout per probe item in seconds. Default is 30.0.
        stop_on_error : bool
            Stop on first error. Default is False.

        Examples
        --------
        >>> config = ProbeConfig(type="logic")
        >>> config = ProbeConfig(
        ...     type="factuality",
        ...     name="Knowledge Test",
        ...     params={"domain": "science"}
        ... )

        Notes
        -----
        This fallback class does not perform validation. Use the Pydantic
        version (install pydantic) for automatic type checking and validation.
        """

        def __init__(
            self,
            type: str,
            name: Optional[str] = None,
            description: Optional[str] = None,
            params: Optional[dict[str, Any]] = None,
            timeout_per_item: float = 30.0,
            stop_on_error: bool = False,
            **kwargs: Any,
        ) -> None:
            self.type = type
            self.name = name or type
            self.description = description
            self.params = params or {}
            self.timeout_per_item = timeout_per_item
            self.stop_on_error = stop_on_error

    class DatasetConfig(BaseModel):  # type: ignore[no-redef]  # Intentional: fallback class when Pydantic unavailable
        """Configuration for loading evaluation datasets (fallback).

        This is a fallback implementation used when Pydantic is not available.
        It provides the same interface as the Pydantic version but without
        automatic validation.

        Parameters
        ----------
        source : str
            Data source type: "file", "hf", or "inline".
        path : str, optional
            Path to dataset file (required for source="file").
        name : str, optional
            HuggingFace dataset name (required for source="hf").
        split : str
            Dataset split to use. Default is "test".
        data : list[dict], optional
            Inline data (required for source="inline").
        sample_size : int, optional
            Number of samples to use.
        shuffle : bool
            Whether to shuffle data. Default is False.
        seed : int, optional
            Random seed for shuffling.

        Examples
        --------
        >>> config = DatasetConfig(source="file", path="data/test.jsonl")
        >>> config = DatasetConfig(
        ...     source="inline",
        ...     data=[{"q": "test"}]
        ... )
        >>> config = DatasetConfig(
        ...     source="hf",
        ...     name="truthful_qa",
        ...     split="validation"
        ... )

        Notes
        -----
        This fallback class does not perform validation. Use the Pydantic
        version (install pydantic) for automatic type checking and validation.
        """

        def __init__(
            self,
            source: str,
            path: Optional[str] = None,
            name: Optional[str] = None,
            split: str = "test",
            data: Optional[list[dict[str, Any]]] = None,
            sample_size: Optional[int] = None,
            shuffle: bool = False,
            seed: Optional[int] = None,
            **kwargs: Any,
        ) -> None:
            self.source = source
            self.path = path
            self.name = name
            self.split = split
            self.data = data
            self.sample_size = sample_size
            self.shuffle = shuffle
            self.seed = seed

    class RunnerConfig(BaseModel):  # type: ignore[no-redef]  # Intentional: fallback class when Pydantic unavailable
        """Configuration for experiment execution settings (fallback).

        This is a fallback implementation used when Pydantic is not available.
        It provides the same interface as the Pydantic version but without
        automatic validation.

        Parameters
        ----------
        concurrency : int
            Number of concurrent API requests. Default is 1.
        output_dir : str
            Directory for output files. Default is "output".
        output_formats : list[str], optional
            Output formats to generate. Default is ["json", "markdown"].
        save_intermediate : bool
            Save intermediate results. Default is False.
        progress_bar : bool
            Show progress bar. Default is True.
        verbose : bool
            Enable verbose logging. Default is False.
        cache_responses : bool
            Cache model responses. Default is False.

        Examples
        --------
        >>> config = RunnerConfig()
        >>> config = RunnerConfig(
        ...     concurrency=5,
        ...     output_dir="results",
        ...     verbose=True
        ... )

        Notes
        -----
        This fallback class does not perform validation. Use the Pydantic
        version (install pydantic) for automatic type checking and validation.
        """

        def __init__(
            self,
            concurrency: int = 1,
            output_dir: str = "output",
            output_formats: Optional[list[str]] = None,
            save_intermediate: bool = False,
            progress_bar: bool = True,
            verbose: bool = False,
            cache_responses: bool = False,
            **kwargs: Any,
        ) -> None:
            self.concurrency = concurrency
            self.output_dir = output_dir
            self.output_formats = output_formats or ["json", "markdown"]
            self.save_intermediate = save_intermediate
            self.progress_bar = progress_bar
            self.verbose = verbose
            self.cache_responses = cache_responses

    class ExperimentConfig(BaseModel):  # type: ignore[no-redef]  # Intentional: fallback class when Pydantic unavailable
        """Complete experiment configuration combining all components (fallback).

        This is a fallback implementation used when Pydantic is not available.
        It provides the same interface as the Pydantic version but without
        automatic validation.

        Parameters
        ----------
        name : str
            Unique name for the experiment.
        model : ModelConfig
            Configuration for the language model.
        probe : ProbeConfig
            Configuration for the evaluation probe.
        dataset : DatasetConfig
            Configuration for the data source.
        description : str, optional
            Human-readable description.
        runner : RunnerConfig, optional
            Execution settings. Defaults to RunnerConfig().
        tags : list[str], optional
            Tags for organizing experiments.
        metadata : dict, optional
            Additional custom metadata.

        Examples
        --------
        >>> config = ExperimentConfig(
        ...     name="Test",
        ...     model=ModelConfig(provider="openai", model_id="gpt-4"),
        ...     probe=ProbeConfig(type="logic"),
        ...     dataset=DatasetConfig(source="inline", data=[{"q": "test"}])
        ... )
        >>> config = ExperimentConfig(
        ...     name="Full Test",
        ...     description="A complete experiment",
        ...     model=ModelConfig(provider="anthropic", model_id="claude-3-opus"),
        ...     probe=ProbeConfig(type="factuality"),
        ...     dataset=DatasetConfig(source="file", path="data/test.jsonl"),
        ...     runner=RunnerConfig(concurrency=5),
        ...     tags=["test", "factuality"]
        ... )

        Notes
        -----
        This fallback class does not perform validation. Use the Pydantic
        version (install pydantic) for automatic type checking and validation.
        """

        def __init__(
            self,
            name: str,
            model: ModelConfig,
            probe: ProbeConfig,
            dataset: DatasetConfig,
            description: Optional[str] = None,
            runner: Optional[RunnerConfig] = None,
            tags: Optional[list[str]] = None,
            metadata: Optional[dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            self.name = name
            self.description = description
            self.model = model
            self.probe = probe
            self.dataset = dataset
            self.runner = runner or RunnerConfig()
            self.tags = tags or []
            self.metadata = metadata or {}


# Configuration loading and saving utilities


def load_config_from_yaml(path: Union[str, Path]) -> ExperimentConfig:
    """Load experiment configuration from a YAML file.

    Parses a YAML configuration file and returns a validated
    ExperimentConfig object with all nested configurations.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    ExperimentConfig
        A validated experiment configuration object.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the configuration content is invalid.

    Examples
    --------
    Loading a basic configuration:

    >>> config = load_config_from_yaml("experiments/basic_test.yaml")
    >>> print(config.name)
    'Basic Test'

    Loading with Path object:

    >>> from pathlib import Path
    >>> config = load_config_from_yaml(Path("experiments") / "test.yaml")

    Example YAML file content::

        name: "GPT-4 Logic Evaluation"
        description: "Testing logical reasoning"
        model:
          provider: openai
          model_id: gpt-4
          temperature: 0.0
        probe:
          type: logic
          params:
            difficulty: medium
        dataset:
          source: file
          path: data/logic.jsonl
        runner:
          concurrency: 5

    See Also
    --------
    load_config : Auto-detect format and load
    load_config_from_json : Load from JSON format
    save_config_to_yaml : Save configuration to YAML
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for loading YAML configs. Install it with: pip install pyyaml"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return _parse_config_dict(data)


def load_config_from_json(path: Union[str, Path]) -> ExperimentConfig:
    """Load experiment configuration from a JSON file.

    Parses a JSON configuration file and returns a validated
    ExperimentConfig object with all nested configurations.

    Parameters
    ----------
    path : str or Path
        Path to the JSON configuration file.

    Returns
    -------
    ExperimentConfig
        A validated experiment configuration object.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the configuration content is invalid.
    json.JSONDecodeError
        If the file contains invalid JSON.

    Examples
    --------
    Loading a configuration:

    >>> config = load_config_from_json("experiments/test.json")
    >>> print(config.name)
    'Test Experiment'

    Loading with Path object:

    >>> from pathlib import Path
    >>> config = load_config_from_json(Path("experiments") / "test.json")

    Example JSON file content::

        {
          "name": "GPT-4 Logic Evaluation",
          "model": {
            "provider": "openai",
            "model_id": "gpt-4",
            "temperature": 0.0
          },
          "probe": {
            "type": "logic"
          },
          "dataset": {
            "source": "file",
            "path": "data/logic.jsonl"
          }
        }

    See Also
    --------
    load_config : Auto-detect format and load
    load_config_from_yaml : Load from YAML format
    save_config_to_json : Save configuration to JSON
    """
    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    return _parse_config_dict(data)


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load experiment configuration with automatic format detection.

    Automatically detects the file format based on extension and
    loads the configuration using the appropriate parser.

    Parameters
    ----------
    path : str or Path
        Path to the configuration file. Supported extensions:
        - .yaml, .yml for YAML format
        - .json for JSON format

    Returns
    -------
    ExperimentConfig
        A validated experiment configuration object.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the specified file does not exist.
    ImportError
        If PyYAML is not installed and loading a YAML file.

    Examples
    --------
    Loading YAML configuration:

    >>> config = load_config("experiments/test.yaml")

    Loading JSON configuration:

    >>> config = load_config("experiments/test.json")

    Loading with Path object:

    >>> from pathlib import Path
    >>> config = load_config(Path("experiments") / "config.yml")

    Handling unsupported formats:

    >>> try:
    ...     config = load_config("config.toml")
    ... except ValueError as e:
    ...     print(f"Error: {e}")
    Error: Unsupported configuration format: .toml

    See Also
    --------
    load_config_from_yaml : Load specifically from YAML
    load_config_from_json : Load specifically from JSON
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        return load_config_from_yaml(path)
    elif suffix == ".json":
        return load_config_from_json(path)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")


def _parse_config_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Parse a configuration dictionary into ExperimentConfig.

    Internal function that converts a raw configuration dictionary
    (from YAML or JSON) into a fully constructed ExperimentConfig
    with all nested configuration objects.

    Parameters
    ----------
    data : dict
        Raw configuration dictionary with keys: name, model, probe,
        dataset, runner (optional), tags (optional), metadata (optional).

    Returns
    -------
    ExperimentConfig
        A fully constructed experiment configuration.

    Examples
    --------
    >>> data = {
    ...     "name": "Test",
    ...     "model": {"provider": "openai", "model_id": "gpt-4"},
    ...     "probe": {"type": "logic"},
    ...     "dataset": {"source": "inline", "data": [{"q": "test"}]}
    ... }
    >>> config = _parse_config_dict(data)
    >>> config.name
    'Test'

    Notes
    -----
    This is an internal function. Use load_config, load_config_from_yaml,
    or load_config_from_json for loading configurations from files.
    """
    # Parse nested configurations
    model_data = data.get("model", {})
    probe_data = data.get("probe", {})
    dataset_data = data.get("dataset", {})
    runner_data = data.get("runner", {})

    model_config = ModelConfig(**model_data)
    probe_config = ProbeConfig(**probe_data)
    dataset_config = DatasetConfig(**dataset_data)
    runner_config = RunnerConfig(**runner_data)

    return ExperimentConfig(
        name=data.get("name", "Unnamed Experiment"),
        description=data.get("description"),
        model=model_config,
        probe=probe_config,
        dataset=dataset_config,
        runner=runner_config,
        tags=data.get("tags", []),
        metadata=data.get("metadata", {}),
    )


def save_config_to_yaml(config: ExperimentConfig, path: Union[str, Path]) -> None:
    """Save experiment configuration to a YAML file.

    Serializes an ExperimentConfig object to YAML format and writes
    it to the specified file. Creates parent directories if needed.

    Parameters
    ----------
    config : ExperimentConfig
        The experiment configuration to save.
    path : str or Path
        Output file path. Parent directories will be created if
        they don't exist.

    Raises
    ------
    ImportError
        If PyYAML is not installed.

    Examples
    --------
    Saving a configuration:

    >>> from insideLLMs.config import (
    ...     ExperimentConfig, ModelConfig, ProbeConfig,
    ...     DatasetConfig, save_config_to_yaml
    ... )
    >>> config = ExperimentConfig(
    ...     name="My Experiment",
    ...     model=ModelConfig(provider="openai", model_id="gpt-4"),
    ...     probe=ProbeConfig(type="logic"),
    ...     dataset=DatasetConfig(source="inline", data=[{"q": "test"}])
    ... )
    >>> save_config_to_yaml(config, "experiments/my_experiment.yaml")

    Saving to a nested directory (auto-created):

    >>> save_config_to_yaml(config, "results/2024/01/config.yaml")

    Using Path object:

    >>> from pathlib import Path
    >>> save_config_to_yaml(config, Path("experiments") / "test.yaml")

    See Also
    --------
    save_config_to_json : Save to JSON format
    load_config_from_yaml : Load from YAML format
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for saving YAML configs. Install it with: pip install pyyaml"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump() if PYDANTIC_AVAILABLE else _config_to_dict(config)
    # Normalize complex types (e.g., Enum) before dumping with SafeDumper.
    data = _serialize_value(data)

    # Use safe_dump with sorted keys for stable, review-friendly YAML output.
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=True, allow_unicode=True)


def save_config_to_json(config: ExperimentConfig, path: Union[str, Path]) -> None:
    """Save experiment configuration to a JSON file.

    Serializes an ExperimentConfig object to JSON format and writes
    it to the specified file. Creates parent directories if needed.

    Parameters
    ----------
    config : ExperimentConfig
        The experiment configuration to save.
    path : str or Path
        Output file path. Parent directories will be created if
        they don't exist.

    Examples
    --------
    Saving a configuration:

    >>> from insideLLMs.config import (
    ...     ExperimentConfig, ModelConfig, ProbeConfig,
    ...     DatasetConfig, save_config_to_json
    ... )
    >>> config = ExperimentConfig(
    ...     name="My Experiment",
    ...     model=ModelConfig(provider="openai", model_id="gpt-4"),
    ...     probe=ProbeConfig(type="logic"),
    ...     dataset=DatasetConfig(source="inline", data=[{"q": "test"}])
    ... )
    >>> save_config_to_json(config, "experiments/my_experiment.json")

    Saving to a nested directory (auto-created):

    >>> save_config_to_json(config, "results/2024/01/config.json")

    Using Path object:

    >>> from pathlib import Path
    >>> save_config_to_json(config, Path("experiments") / "test.json")

    See Also
    --------
    save_config_to_yaml : Save to YAML format
    load_config_from_json : Load from JSON format
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump() if PYDANTIC_AVAILABLE else _config_to_dict(config)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a configuration object to a dictionary.

    Internal function used as a fallback when Pydantic's model_dump
    is not available. Recursively converts nested configuration objects
    and handles Enum values.

    Parameters
    ----------
    config : Any
        A configuration object with attributes to convert.

    Returns
    -------
    dict
        Dictionary representation of the configuration.

    Examples
    --------
    >>> config = ModelConfig(provider="openai", model_id="gpt-4")
    >>> data = _config_to_dict(config)
    >>> data["provider"]
    'openai'

    Notes
    -----
    This is an internal function. For serialization, use save_config_to_yaml
    or save_config_to_json which handle both Pydantic and fallback cases.
    """
    if hasattr(config, "model_dump"):
        return config.model_dump()

    result = {}
    for key, value in vars(config).items():
        if hasattr(value, "__dict__") and not isinstance(value, type):
            result[key] = _config_to_dict(value)
        elif isinstance(value, Enum):
            result[key] = value.value
        else:
            result[key] = value
    return result


def create_example_config() -> ExperimentConfig:
    """Create a complete example configuration for reference.

    Generates a fully populated ExperimentConfig with example values
    for all fields. Useful for understanding configuration structure
    or as a starting template.

    Returns
    -------
    ExperimentConfig
        A complete experiment configuration with example values including:
        - OpenAI GPT-4 model configuration
        - Logic probe with medium difficulty
        - Inline dataset with sample questions
        - Runner with concurrency=5

    Examples
    --------
    Creating and inspecting an example configuration:

    >>> config = create_example_config()
    >>> print(config.name)
    'Example Experiment'
    >>> print(config.model.model_id)
    'gpt-4'
    >>> print(config.probe.type)
    <ProbeType.LOGIC: 'logic'>

    Using as a template and modifying:

    >>> config = create_example_config()
    >>> config.name = "My Custom Experiment"
    >>> config.model.model_id = "gpt-4-turbo"

    Saving the example to a file:

    >>> config = create_example_config()
    >>> save_config_to_yaml(config, "examples/template.yaml")

    Viewing the structure:

    >>> config = create_example_config()
    >>> print(config.tags)
    ['example', 'logic']

    See Also
    --------
    validate_config : Validate a configuration dictionary
    load_config : Load configuration from a file
    """
    return ExperimentConfig(
        name="Example Experiment",
        description="An example experiment configuration",
        model=ModelConfig(
            provider=ModelProvider.OPENAI if PYDANTIC_AVAILABLE else "openai",
            model_id="gpt-4",
            temperature=0.7,
            max_tokens=1000,
        ),
        probe=ProbeConfig(
            type=ProbeType.LOGIC if PYDANTIC_AVAILABLE else "logic",
            name="Logic Reasoning Test",
            params={"difficulty": "medium"},
        ),
        dataset=DatasetConfig(
            source="inline",
            data=[
                {"question": "What comes next: 1, 2, 3, ?", "answer": "4"},
                {"question": "If A > B and B > C, is A > C?", "answer": "yes"},
            ],
        ),
        runner=RunnerConfig(
            concurrency=5,
            output_dir="results",
            output_formats=["json", "markdown"],
        ),
        tags=["example", "logic"],
    )


def validate_config(config: Union[dict[str, Any], ExperimentConfig]) -> ExperimentConfig:
    """Validate and convert a configuration to ExperimentConfig.

    Takes either a raw configuration dictionary or an existing
    ExperimentConfig and ensures it is valid. If given a dictionary,
    converts it to a proper ExperimentConfig with all nested objects.

    Parameters
    ----------
    config : dict or ExperimentConfig
        Either a configuration dictionary with model, probe, dataset
        keys, or an existing ExperimentConfig object.

    Returns
    -------
    ExperimentConfig
        A validated experiment configuration. If input was already
        an ExperimentConfig, returns it unchanged.

    Raises
    ------
    ValueError
        If the configuration is invalid (missing required fields,
        invalid values, etc.).

    Examples
    --------
    Validating a configuration dictionary:

    >>> config_dict = {
    ...     "name": "Test Experiment",
    ...     "model": {"provider": "openai", "model_id": "gpt-4"},
    ...     "probe": {"type": "logic"},
    ...     "dataset": {"source": "inline", "data": [{"q": "test"}]}
    ... }
    >>> config = validate_config(config_dict)
    >>> isinstance(config, ExperimentConfig)
    True

    Validating an existing ExperimentConfig (passes through):

    >>> existing = ExperimentConfig(
    ...     name="Test",
    ...     model=ModelConfig(provider="openai", model_id="gpt-4"),
    ...     probe=ProbeConfig(type="logic"),
    ...     dataset=DatasetConfig(source="inline", data=[{"q": "test"}])
    ... )
    >>> validated = validate_config(existing)
    >>> validated is existing
    True

    Handling invalid configuration:

    >>> try:
    ...     validate_config({"name": "Bad Config"})  # Missing required fields
    ... except ValueError:
    ...     print("Invalid configuration")
    Invalid configuration

    Validating configuration loaded from external source:

    >>> import json
    >>> with open("config.json") as f:
    ...     raw_config = json.load(f)
    >>> config = validate_config(raw_config)

    See Also
    --------
    load_config : Load and validate from file
    _parse_config_dict : Internal parsing function
    """
    if isinstance(config, ExperimentConfig):
        return config

    return _parse_config_dict(config)
