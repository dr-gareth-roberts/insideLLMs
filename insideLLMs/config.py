"""Configuration validation and management using Pydantic.

This module provides strongly-typed configuration classes for:
- Model configuration
- Probe configuration
- Experiment configuration
- Runner configuration

All configurations support validation, serialization to YAML/JSON,
and can be loaded from files or environment variables.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Union

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
    class BaseModel:  # type: ignore
        """Fallback BaseModel when Pydantic is not available."""

        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

        def model_dump(self) -> dict[str, Any]:
            return self.__dict__.copy()

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        return kwargs.get("default")

    def field_validator(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        def decorator(func: Any) -> Any:
            return func

        return decorator

    def model_validator(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        def decorator(func: Any) -> Any:
            return func

        return decorator

    class ConfigDict:  # type: ignore
        pass


class ModelProvider(str, Enum):
    """Supported model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    DUMMY = "dummy"
    CUSTOM = "custom"


class ProbeType(str, Enum):
    """Available probe types."""

    LOGIC = "logic"
    FACTUALITY = "factuality"
    BIAS = "bias"
    ATTACK = "attack"
    CUSTOM = "custom"


if PYDANTIC_AVAILABLE:

    class ModelConfig(BaseModel):
        """Configuration for a language model.

        Example:
            >>> config = ModelConfig(
            ...     provider="openai",
            ...     model_id="gpt-4",
            ...     api_key_env="OPENAI_API_KEY"
            ... )
        """

        model_config = ConfigDict(extra="allow", validate_default=True)

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
        """Configuration for a probe.

        Example:
            >>> config = ProbeConfig(
            ...     type="logic",
            ...     name="Logic Test",
            ...     params={"difficulty": "hard"}
            ... )
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
        """Configuration for a dataset.

        Example:
            >>> config = DatasetConfig(
            ...     source="file",
            ...     path="data/test.jsonl"
            ... )
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
        """Configuration for the experiment runner.

        Example:
            >>> config = RunnerConfig(
            ...     concurrency=5,
            ...     output_dir="results/"
            ... )
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
        """Complete experiment configuration.

        Example:
            >>> config = ExperimentConfig(
            ...     name="GPT-4 Logic Test",
            ...     model=ModelConfig(provider="openai", model_id="gpt-4"),
            ...     probe=ProbeConfig(type="logic"),
            ...     dataset=DatasetConfig(source="inline", data=[{"q": "1+1=?"}])
            ... )
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
    class ModelConfig(BaseModel):  # type: ignore
        """Configuration for a language model (fallback without Pydantic)."""

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

    class ProbeConfig(BaseModel):  # type: ignore
        """Configuration for a probe (fallback without Pydantic)."""

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

    class DatasetConfig(BaseModel):  # type: ignore
        """Configuration for a dataset (fallback without Pydantic)."""

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

    class RunnerConfig(BaseModel):  # type: ignore
        """Configuration for the experiment runner (fallback without Pydantic)."""

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

    class ExperimentConfig(BaseModel):  # type: ignore
        """Complete experiment configuration (fallback without Pydantic)."""

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
    """Load configuration from a YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        ExperimentConfig object.

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the configuration is invalid.
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
    """Load configuration from a JSON file.

    Args:
        path: Path to JSON configuration file.

    Returns:
        ExperimentConfig object.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    return _parse_config_dict(data)


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load configuration from a file (auto-detect format).

    Args:
        path: Path to configuration file (.yaml, .yml, or .json).

    Returns:
        ExperimentConfig object.

    Raises:
        ValueError: If the file format is not supported.
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

    Args:
        data: Configuration dictionary.

    Returns:
        ExperimentConfig object.
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
    """Save configuration to a YAML file.

    Args:
        config: ExperimentConfig to save.
        path: Output file path.
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

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def save_config_to_json(config: ExperimentConfig, path: Union[str, Path]) -> None:
    """Save configuration to a JSON file.

    Args:
        config: ExperimentConfig to save.
        path: Output file path.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump() if PYDANTIC_AVAILABLE else _config_to_dict(config)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _config_to_dict(config: Any) -> dict[str, Any]:
    """Convert config object to dictionary (fallback for non-Pydantic)."""
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
    """Create an example configuration for reference.

    Returns:
        A complete ExperimentConfig with example values.
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
    """Validate a configuration and return a validated ExperimentConfig.

    Args:
        config: Configuration dictionary or ExperimentConfig.

    Returns:
        Validated ExperimentConfig.

    Raises:
        ValueError: If the configuration is invalid.
    """
    if isinstance(config, ExperimentConfig):
        return config

    return _parse_config_dict(config)
