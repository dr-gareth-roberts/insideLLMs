"""Model adapter factory module for unified LLM provider access.

This module provides a unified interface for creating and managing adapters
across multiple LLM providers with consistent APIs. It abstracts away
provider-specific implementation details, enabling seamless switching between
different LLM backends while maintaining a consistent programming interface.

Key Features
------------
- **Unified Interface**: Single API across OpenAI, Anthropic, Google, Cohere,
  Mistral, Groq, Together, Anyscale, Replicate, HuggingFace, and local models.
- **Automatic Provider Detection**: Intelligently detects the provider from
  model names (e.g., "gpt-4" -> OpenAI, "claude-3" -> Anthropic).
- **Lazy Loading**: Provider-specific dependencies are loaded only when needed,
  reducing startup time and memory footprint.
- **Connection Pooling**: Efficient connection reuse across multiple requests.
- **Rate Limiting**: Built-in support for RPM (requests per minute) and TPM
  (tokens per minute) rate limiting.
- **Fallback Chains**: Automatic failover across providers for high availability.
- **Model Aliasing**: Use convenient aliases (e.g., "claude-sonnet") that resolve
  to canonical model IDs (e.g., "claude-3-5-sonnet-20241022").
- **Health Checks**: Monitor adapter availability and latency.
- **Load Balancing**: Distribute requests across multiple adapters using
  round-robin, least-requests, or random strategies.

Module Structure
----------------
The module is organized into the following main components:

**Enums**:
    - `Provider`: Supported LLM providers
    - `AdapterStatus`: Adapter state (ready, initializing, error, etc.)
    - `ModelCapability`: Model features (chat, completion, embedding, etc.)

**Dataclasses**:
    - `AdapterModelInfo`: Model metadata (context window, pricing, etc.)
    - `AdapterConfig`: Adapter configuration (API keys, timeouts, etc.)
    - `GenerationParams`: Text generation parameters (temperature, etc.)
    - `GenerationResult`: Response from text generation
    - `HealthCheckResult`: Health check response
    - `ConnectionInfo`: Connection status information

**Adapters**:
    - `BaseAdapter`: Abstract base class for all adapters
    - `MockAdapter`: Testing adapter with configurable responses
    - `OpenAIAdapter`: OpenAI API adapter
    - `AnthropicAdapter`: Anthropic API adapter

**Utilities**:
    - `ModelRegistry`: Registry of known models and metadata
    - `ProviderDetector`: Automatic provider detection from model names
    - `AdapterFactory`: Factory for creating configured adapters
    - `FallbackChain`: Chain adapters with automatic failover
    - `AdapterPool`: Load balance across multiple adapters
    - `ConnectionMonitor`: Monitor adapter health and connectivity

Examples
--------
Basic usage with automatic provider detection:

>>> from insideLLMs.contrib.adapters import create_adapter
>>>
>>> # Create adapter for GPT-4 (auto-detects OpenAI)
>>> adapter = create_adapter("gpt-4o", api_key="sk-...")
>>> response = adapter.generate("Explain quantum computing in simple terms.")
>>> print(response.text)
'Quantum computing uses quantum bits or "qubits"...'
>>> print(f"Tokens used: {response.tokens_input} in, {response.tokens_output} out")
Tokens used: 8 in, 150 out

Using the factory for more control:

>>> from insideLLMs.contrib.adapters import AdapterFactory, Provider
>>>
>>> factory = AdapterFactory()
>>> factory.set_api_key(Provider.OPENAI, "sk-...")
>>> factory.set_api_key(Provider.ANTHROPIC, "sk-ant-...")
>>>
>>> # Create with explicit provider
>>> claude = factory.create("claude-3-opus", provider=Provider.ANTHROPIC)
>>> response = claude.generate_chat([
...     {"role": "user", "content": "Write a haiku about Python programming"}
... ])
>>> print(response.text)
'Code flows like water...'

Setting up a fallback chain for high availability:

>>> from insideLLMs.contrib.adapters import create_fallback_chain
>>>
>>> chain = create_fallback_chain(
...     model_ids=["gpt-4o", "claude-3-opus", "mistral-large"],
...     api_keys={
...         "openai": "sk-...",
...         "anthropic": "sk-ant-...",
...         "mistral": "..."
...     }
... )
>>> # Automatically falls back if primary model fails
>>> response = chain.generate("Hello, world!")

Load balancing across multiple instances:

>>> from insideLLMs.contrib.adapters import create_adapter_pool
>>>
>>> pool = create_adapter_pool(
...     model_ids=["gpt-4o", "gpt-4o", "gpt-4o"],  # Three instances
...     strategy="least_requests"
... )
>>> # Distributes requests to least-loaded adapter
>>> for prompt in prompts:
...     response = pool.generate(prompt)

Using the mock adapter for testing:

>>> from insideLLMs.contrib.adapters import create_mock_adapter
>>>
>>> mock = create_mock_adapter()
>>> mock.set_response("hello", "Hello! How can I help you?")
>>> mock.set_default_response("I don't understand.")
>>>
>>> response = mock.generate("hello world")
>>> print(response.text)
'Hello! How can I help you?'

Model information and discovery:

>>> from insideLLMs.contrib.adapters import list_models, get_model_info, Provider
>>>
>>> # List all Anthropic models
>>> models = list_models(provider="anthropic")
>>> for m in models:
...     print(f"{m.display_name}: {m.context_window} tokens")
Claude 3.5 Sonnet: 200000 tokens
Claude 3 Opus: 200000 tokens
Claude 3 Haiku: 200000 tokens
>>>
>>> # Get specific model info
>>> info = get_model_info("gpt-4o")
>>> print(f"Price: ${info.input_price_per_1k}/1K input tokens")
Price: $0.005/1K input tokens

Notes
-----
- API keys can be provided directly, stored in the factory, or read from
  environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
- All adapters are thread-safe for concurrent usage.
- The module uses exponential backoff for retries on transient failures.
- Rate limiting is applied per-adapter instance, not globally.

See Also
--------
- OpenAI API documentation: https://platform.openai.com/docs
- Anthropic API documentation: https://docs.anthropic.com
- insideLLMs.tracing : Request tracing and logging utilities
- insideLLMs.config : Configuration management

Warnings
--------
- Always handle API keys securely; never commit them to version control.
- Rate limits vary by provider and subscription tier; configure accordingly.
- Some models may be deprecated; check `AdapterModelInfo.deprecated` field.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Provider(Enum):
    """Enumeration of supported LLM providers.

    This enum defines all LLM providers that the adapter system can interface
    with. Each provider has a unique string value used for serialization and
    configuration lookup.

    Attributes
    ----------
    OPENAI : str
        OpenAI API (GPT-4, GPT-3.5, etc.). Value: "openai"
    ANTHROPIC : str
        Anthropic API (Claude family). Value: "anthropic"
    GOOGLE : str
        Google AI (Gemini, PaLM). Value: "google"
    COHERE : str
        Cohere API (Command, Embed). Value: "cohere"
    MISTRAL : str
        Mistral AI API. Value: "mistral"
    GROQ : str
        Groq inference API (fast inference for various models). Value: "groq"
    TOGETHER : str
        Together AI API (open-source model hosting). Value: "together"
    ANYSCALE : str
        Anyscale Endpoints API. Value: "anyscale"
    REPLICATE : str
        Replicate API (model hosting platform). Value: "replicate"
    HUGGINGFACE : str
        HuggingFace Inference API. Value: "huggingface"
    LOCAL : str
        Local model inference (Ollama, llama.cpp, etc.). Value: "local"
    MOCK : str
        Mock provider for testing. Value: "mock"

    Examples
    --------
    Creating an adapter with explicit provider:

    >>> from insideLLMs.contrib.adapters import AdapterFactory, Provider
    >>> factory = AdapterFactory()
    >>> adapter = factory.create("gpt-4o", provider=Provider.OPENAI)
    >>> print(adapter.provider)
    Provider.OPENAI

    Iterating over all providers:

    >>> from insideLLMs.contrib.adapters import Provider
    >>> for provider in Provider:
    ...     print(f"{provider.name}: {provider.value}")
    OPENAI: openai
    ANTHROPIC: anthropic
    ...

    Using provider value for configuration:

    >>> from insideLLMs.contrib.adapters import Provider
    >>> config = {"provider": Provider.ANTHROPIC.value}
    >>> print(config)
    {'provider': 'anthropic'}

    Checking provider from adapter:

    >>> from insideLLMs.contrib.adapters import create_adapter
    >>> adapter = create_adapter("claude-3-opus")
    >>> if adapter.provider == Provider.ANTHROPIC:
    ...     print("Using Anthropic API")
    Using Anthropic API

    See Also
    --------
    ProviderDetector : Automatic provider detection from model names
    AdapterFactory : Factory for creating provider-specific adapters
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"
    GROQ = "groq"
    TOGETHER = "together"
    ANYSCALE = "anyscale"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    MOCK = "mock"


class AdapterStatus(Enum):
    """Enumeration of adapter operational states.

    This enum represents the various states an adapter can be in during its
    lifecycle. The status is used for health monitoring, error handling, and
    determining whether an adapter is ready to accept requests.

    Attributes
    ----------
    READY : str
        Adapter is initialized and ready to handle requests. Value: "ready"
    INITIALIZING : str
        Adapter is in the process of initializing (loading client, etc.).
        Value: "initializing"
    ERROR : str
        Adapter encountered an error and cannot process requests.
        Value: "error"
    RATE_LIMITED : str
        Adapter is temporarily rate-limited by the provider.
        Value: "rate_limited"
    UNAVAILABLE : str
        Adapter or its provider is currently unavailable.
        Value: "unavailable"

    Examples
    --------
    Checking adapter readiness before making requests:

    >>> from insideLLMs.contrib.adapters import create_adapter, AdapterStatus
    >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
    >>> if adapter.status == AdapterStatus.READY:
    ...     response = adapter.generate("Hello!")
    ... else:
    ...     print(f"Adapter not ready: {adapter.status}")

    Handling different adapter states:

    >>> from insideLLMs.contrib.adapters import AdapterStatus
    >>> def handle_adapter_status(adapter):
    ...     match adapter.status:
    ...         case AdapterStatus.READY:
    ...             return "OK"
    ...         case AdapterStatus.RATE_LIMITED:
    ...             return "Wait and retry"
    ...         case AdapterStatus.ERROR:
    ...             return "Check configuration"
    ...         case _:
    ...             return "Unknown state"

    Monitoring adapter status in a pool:

    >>> from insideLLMs.contrib.adapters import AdapterPool, AdapterStatus
    >>> def get_healthy_adapters(pool):
    ...     return [a for a in pool.adapters if a.status == AdapterStatus.READY]

    See Also
    --------
    BaseAdapter.status : Property to get current adapter status
    HealthCheckResult : Detailed health check information
    ConnectionMonitor : Monitor adapter connections over time
    """

    READY = "ready"
    INITIALIZING = "initializing"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"


class ModelCapability(Enum):
    """Enumeration of model capabilities and features.

    This enum defines the various capabilities that LLM models may support.
    Capabilities are used for model discovery, validation, and feature
    detection at runtime.

    Attributes
    ----------
    CHAT : str
        Model supports multi-turn chat/conversation format. Value: "chat"
    COMPLETION : str
        Model supports single-turn text completion. Value: "completion"
    EMBEDDING : str
        Model can generate text embeddings/vectors. Value: "embedding"
    IMAGE : str
        Model supports image input (vision) or generation. Value: "image"
    AUDIO : str
        Model supports audio input or generation. Value: "audio"
    FUNCTION_CALLING : str
        Model supports function/tool calling features. Value: "function_calling"
    STREAMING : str
        Model supports streaming token-by-token responses. Value: "streaming"
    JSON_MODE : str
        Model supports structured JSON output mode. Value: "json_mode"

    Examples
    --------
    Checking if a model supports function calling:

    >>> from insideLLMs.contrib.adapters import get_model_info, ModelCapability
    >>> info = get_model_info("gpt-4o")
    >>> if ModelCapability.FUNCTION_CALLING in info.capabilities:
    ...     print("Model supports function calling!")
    Model supports function calling!

    Finding models with specific capabilities:

    >>> from insideLLMs.contrib.adapters import ModelRegistry, ModelCapability
    >>> registry = ModelRegistry()
    >>> json_models = registry.get_by_capability(ModelCapability.JSON_MODE)
    >>> for model in json_models:
    ...     print(f"{model.display_name} supports JSON mode")
    GPT-4o supports JSON mode
    GPT-4 Turbo supports JSON mode
    ...

    Filtering models by multiple capabilities:

    >>> from insideLLMs.contrib.adapters import list_models, ModelCapability
    >>> models = list_models()
    >>> streaming_chat = [
    ...     m for m in models
    ...     if ModelCapability.CHAT in m.capabilities
    ...     and ModelCapability.STREAMING in m.capabilities
    ... ]
    >>> print(f"Found {len(streaming_chat)} models with chat + streaming")

    Validating capability before using a feature:

    >>> from insideLLMs.contrib.adapters import create_adapter, ModelCapability, get_model_info
    >>> def safe_stream_generate(adapter, prompt):
    ...     info = get_model_info(adapter.model_id)
    ...     if info and ModelCapability.STREAMING in info.capabilities:
    ...         return adapter.generate(prompt, stream=True)
    ...     else:
    ...         return adapter.generate(prompt, stream=False)

    See Also
    --------
    AdapterModelInfo : Model metadata including capabilities list
    ModelRegistry.get_by_capability : Find models by capability
    GenerationParams : Parameters that may require specific capabilities
    """

    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO = "audio"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"


@dataclass
class AdapterModelInfo:
    """Comprehensive metadata about an LLM model.

    This dataclass encapsulates all relevant information about a model,
    including technical specifications, pricing, capabilities, and
    deprecation status. It is used by the ModelRegistry for model
    discovery and by adapters for configuration.

    Parameters
    ----------
    model_id : str
        Canonical identifier for the model (e.g., "gpt-4o", "claude-3-opus-20240229").
    provider : Provider
        The provider that offers this model.
    display_name : str
        Human-readable name for display purposes.
    capabilities : list[ModelCapability], optional
        List of capabilities this model supports. Default is empty list.
    context_window : int, optional
        Maximum number of tokens the model can process (input + output).
        Default is 4096.
    max_output_tokens : int, optional
        Maximum number of tokens the model can generate in a single response.
        Default is 4096.
    input_price_per_1k : float, optional
        Cost per 1,000 input tokens in USD. Default is 0.0.
    output_price_per_1k : float, optional
        Cost per 1,000 output tokens in USD. Default is 0.0.
    aliases : list[str], optional
        Alternative names that resolve to this model. Default is empty list.
    deprecated : bool, optional
        Whether this model is deprecated. Default is False.
    deprecation_date : str, optional
        ISO date string when the model will be/was deprecated. Default is None.

    Attributes
    ----------
    model_id : str
        Canonical model identifier.
    provider : Provider
        Provider enum value.
    display_name : str
        Human-readable display name.
    capabilities : list[ModelCapability]
        Supported model capabilities.
    context_window : int
        Maximum context length in tokens.
    max_output_tokens : int
        Maximum output length in tokens.
    input_price_per_1k : float
        Input token pricing (USD per 1K tokens).
    output_price_per_1k : float
        Output token pricing (USD per 1K tokens).
    aliases : list[str]
        Model aliases for convenience.
    deprecated : bool
        Deprecation status flag.
    deprecation_date : str or None
        Deprecation date if applicable.

    Examples
    --------
    Creating a model info entry:

    >>> from insideLLMs.contrib.adapters import AdapterModelInfo, Provider, ModelCapability
    >>> model = AdapterModelInfo(
    ...     model_id="gpt-4o-mini",
    ...     provider=Provider.OPENAI,
    ...     display_name="GPT-4o Mini",
    ...     capabilities=[ModelCapability.CHAT, ModelCapability.STREAMING],
    ...     context_window=128000,
    ...     max_output_tokens=16384,
    ...     input_price_per_1k=0.00015,
    ...     output_price_per_1k=0.0006,
    ...     aliases=["gpt-4o-mini-2024-07-18"]
    ... )
    >>> print(model.display_name)
    GPT-4o Mini

    Calculating estimated cost:

    >>> from insideLLMs.contrib.adapters import get_model_info
    >>> info = get_model_info("gpt-4o")
    >>> input_tokens, output_tokens = 1000, 500
    >>> cost = (input_tokens / 1000 * info.input_price_per_1k +
    ...         output_tokens / 1000 * info.output_price_per_1k)
    >>> print(f"Estimated cost: ${cost:.4f}")
    Estimated cost: $0.0125

    Checking deprecation status:

    >>> from insideLLMs.contrib.adapters import get_model_info
    >>> info = get_model_info("gpt-4-turbo")
    >>> if info.deprecated:
    ...     print(f"Warning: Model deprecated on {info.deprecation_date}")
    ... else:
    ...     print("Model is active")

    Serializing to dictionary for storage:

    >>> from insideLLMs.contrib.adapters import get_model_info
    >>> info = get_model_info("claude-3-opus")
    >>> data = info.to_dict()
    >>> print(data["context_window"])
    200000

    See Also
    --------
    ModelRegistry : Registry that stores and manages AdapterModelInfo instances
    ModelCapability : Enum of possible model capabilities
    get_model_info : Convenience function to retrieve model info
    """

    model_id: str
    provider: Provider
    display_name: str
    capabilities: list[ModelCapability] = field(default_factory=list)
    context_window: int = 4096
    max_output_tokens: int = 4096
    input_price_per_1k: float = 0.0
    output_price_per_1k: float = 0.0
    aliases: list[str] = field(default_factory=list)
    deprecated: bool = False
    deprecation_date: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert model info to a dictionary representation.

        Creates a serializable dictionary containing all model metadata.
        Provider and capabilities are converted to their string values
        for JSON compatibility.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with the following keys:
            - model_id: str
            - provider: str (enum value)
            - display_name: str
            - capabilities: list[str] (enum values)
            - context_window: int
            - max_output_tokens: int
            - input_price_per_1k: float
            - output_price_per_1k: float
            - aliases: list[str]
            - deprecated: bool

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import AdapterModelInfo, Provider, ModelCapability
        >>> model = AdapterModelInfo(
        ...     model_id="test-model",
        ...     provider=Provider.OPENAI,
        ...     display_name="Test Model",
        ...     capabilities=[ModelCapability.CHAT]
        ... )
        >>> data = model.to_dict()
        >>> print(data["provider"])
        openai
        >>> print(data["capabilities"])
        ['chat']

        Saving to JSON:

        >>> import json
        >>> from insideLLMs.contrib.adapters import get_model_info
        >>> info = get_model_info("gpt-4o")
        >>> json_str = json.dumps(info.to_dict(), indent=2)
        >>> print(json_str[:50])
        {
          "model_id": "gpt-4o",
          "provider": "openai"

        See Also
        --------
        AdapterConfig.to_dict : Similar serialization for adapter config
        GenerationResult.to_dict : Similar serialization for results
        """
        return {
            "model_id": self.model_id,
            "provider": self.provider.value,
            "display_name": self.display_name,
            "capabilities": [c.value for c in self.capabilities],
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "input_price_per_1k": self.input_price_per_1k,
            "output_price_per_1k": self.output_price_per_1k,
            "aliases": self.aliases,
            "deprecated": self.deprecated,
        }


@dataclass
class AdapterConfig:
    """Configuration settings for an LLM adapter instance.

    This dataclass contains all configuration options needed to initialize
    and operate an adapter, including authentication, connection settings,
    rate limiting, and retry behavior.

    Parameters
    ----------
    model_id : str
        The model identifier to use (e.g., "gpt-4o", "claude-3-opus-20240229").
    provider : Provider
        The LLM provider for this adapter.
    api_key : str, optional
        API key for authentication. Can also be set via environment variables.
        Default is None.
    api_base : str, optional
        Custom API base URL (for proxies, local deployments, etc.).
        Default is None (uses provider default).
    organization_id : str, optional
        Organization ID for providers that support it (e.g., OpenAI).
        Default is None.
    timeout : float, optional
        Request timeout in seconds. Default is 60.0.
    max_retries : int, optional
        Maximum number of retry attempts on transient failures. Default is 3.
    retry_delay : float, optional
        Base delay between retries in seconds (exponential backoff).
        Default is 1.0.
    rate_limit_rpm : int, optional
        Requests per minute rate limit. Default is None (no limit).
    rate_limit_tpm : int, optional
        Tokens per minute rate limit. Default is None (no limit).
    extra_params : dict[str, Any], optional
        Additional provider-specific parameters. Default is empty dict.

    Attributes
    ----------
    model_id : str
        Model identifier.
    provider : Provider
        Provider enum value.
    api_key : str or None
        Authentication key (sensitive).
    api_base : str or None
        Custom API endpoint.
    organization_id : str or None
        Organization identifier.
    timeout : float
        Request timeout in seconds.
    max_retries : int
        Retry count limit.
    retry_delay : float
        Base retry delay in seconds.
    rate_limit_rpm : int or None
        RPM rate limit.
    rate_limit_tpm : int or None
        TPM rate limit.
    extra_params : dict[str, Any]
        Additional configuration parameters.

    Examples
    --------
    Basic configuration:

    >>> from insideLLMs.contrib.adapters import AdapterConfig, Provider
    >>> config = AdapterConfig(
    ...     model_id="gpt-4o",
    ...     provider=Provider.OPENAI,
    ...     api_key="sk-..."
    ... )
    >>> print(f"Model: {config.model_id}, Timeout: {config.timeout}s")
    Model: gpt-4o, Timeout: 60.0s

    Configuration with rate limiting and custom timeout:

    >>> from insideLLMs.contrib.adapters import AdapterConfig, Provider
    >>> config = AdapterConfig(
    ...     model_id="claude-3-opus-20240229",
    ...     provider=Provider.ANTHROPIC,
    ...     api_key="sk-ant-...",
    ...     timeout=120.0,
    ...     max_retries=5,
    ...     rate_limit_rpm=60,
    ...     rate_limit_tpm=100000
    ... )
    >>> print(f"Rate limit: {config.rate_limit_rpm} RPM")
    Rate limit: 60 RPM

    Using a custom API base (e.g., Azure OpenAI):

    >>> from insideLLMs.contrib.adapters import AdapterConfig, Provider
    >>> config = AdapterConfig(
    ...     model_id="gpt-4",
    ...     provider=Provider.OPENAI,
    ...     api_key="...",
    ...     api_base="https://my-resource.openai.azure.com/",
    ...     extra_params={"api_version": "2024-02-15-preview"}
    ... )

    Creating adapter from config:

    >>> from insideLLMs.contrib.adapters import AdapterConfig, Provider, OpenAIAdapter
    >>> config = AdapterConfig(
    ...     model_id="gpt-4o",
    ...     provider=Provider.OPENAI,
    ...     api_key="sk-..."
    ... )
    >>> adapter = OpenAIAdapter(config)
    >>> print(adapter.status)
    AdapterStatus.READY

    Safe serialization (excludes API key):

    >>> from insideLLMs.contrib.adapters import AdapterConfig, Provider
    >>> config = AdapterConfig(
    ...     model_id="gpt-4o",
    ...     provider=Provider.OPENAI,
    ...     api_key="sk-secret-key"
    ... )
    >>> data = config.to_dict()
    >>> print("api_key" in data)  # API key is NOT included
    False

    See Also
    --------
    AdapterFactory : Creates adapters with merged configurations
    BaseAdapter : Uses AdapterConfig for initialization
    GenerationParams : Runtime parameters for each request
    """

    model_id: str
    provider: Provider
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    organization_id: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_rpm: Optional[int] = None
    rate_limit_tpm: Optional[int] = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary, excluding sensitive data.

        Creates a serializable dictionary containing configuration metadata.
        Sensitive fields (api_key, organization_id) are intentionally
        excluded for security.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with the following keys:
            - model_id: str
            - provider: str (enum value)
            - api_base: str or None
            - timeout: float
            - max_retries: int
            - rate_limit_rpm: int or None
            - rate_limit_tpm: int or None

        Notes
        -----
        The api_key and organization_id fields are NOT included in the
        output to prevent accidental exposure of sensitive credentials.

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import AdapterConfig, Provider
        >>> config = AdapterConfig(
        ...     model_id="gpt-4o",
        ...     provider=Provider.OPENAI,
        ...     api_key="sk-secret-key",
        ...     timeout=30.0
        ... )
        >>> data = config.to_dict()
        >>> print(data)
        {'model_id': 'gpt-4o', 'provider': 'openai', 'api_base': None, ...}
        >>> print("api_key" in data)
        False

        Logging configuration safely:

        >>> import logging
        >>> from insideLLMs.contrib.adapters import AdapterConfig, Provider
        >>> config = AdapterConfig(
        ...     model_id="claude-3-opus",
        ...     provider=Provider.ANTHROPIC,
        ...     api_key="sk-ant-secret"
        ... )
        >>> logging.info(f"Adapter config: {config.to_dict()}")

        See Also
        --------
        AdapterModelInfo.to_dict : Similar method for model info
        GenerationResult.to_dict : Similar method for results
        """
        return {
            "model_id": self.model_id,
            "provider": self.provider.value,
            "api_base": self.api_base,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "rate_limit_rpm": self.rate_limit_rpm,
            "rate_limit_tpm": self.rate_limit_tpm,
        }


@dataclass
class GenerationParams:
    """Parameters controlling text generation behavior.

    This dataclass encapsulates all parameters that influence how an LLM
    generates text, including sampling parameters, output constraints,
    and special features like streaming and function calling.

    Parameters
    ----------
    temperature : float, optional
        Controls randomness in generation. Higher values (e.g., 1.0) produce
        more creative/random output; lower values (e.g., 0.1) produce more
        deterministic output. Range: 0.0 to 2.0. Default is 0.7.
    max_tokens : int, optional
        Maximum number of tokens to generate. Default is 1024.
    top_p : float, optional
        Nucleus sampling: only consider tokens with cumulative probability
        up to this value. Range: 0.0 to 1.0. Default is 1.0 (no filtering).
    top_k : int, optional
        Only consider the top k tokens with highest probability.
        Default is None (no filtering).
    frequency_penalty : float, optional
        Penalty for repeating tokens based on their frequency in the output.
        Range: -2.0 to 2.0. Default is 0.0.
    presence_penalty : float, optional
        Penalty for repeating any token that has appeared in the output.
        Range: -2.0 to 2.0. Default is 0.0.
    stop_sequences : list[str], optional
        List of sequences that will stop generation when encountered.
        Default is empty list.
    seed : int, optional
        Random seed for reproducible generation. Default is None.
    stream : bool, optional
        Whether to stream the response token by token. Default is False.
    json_mode : bool, optional
        Whether to enforce JSON-formatted output. Default is False.
    functions : list[dict[str, Any]], optional
        Function definitions for function calling capability. Default is None.

    Attributes
    ----------
    temperature : float
        Sampling temperature.
    max_tokens : int
        Output token limit.
    top_p : float
        Nucleus sampling threshold.
    top_k : int or None
        Top-k sampling limit.
    frequency_penalty : float
        Token frequency penalty.
    presence_penalty : float
        Token presence penalty.
    stop_sequences : list[str]
        Generation stop triggers.
    seed : int or None
        Random seed for reproducibility.
    stream : bool
        Streaming mode flag.
    json_mode : bool
        JSON output mode flag.
    functions : list[dict[str, Any]] or None
        Function call definitions.

    Examples
    --------
    Default parameters for general use:

    >>> from insideLLMs.contrib.adapters import GenerationParams
    >>> params = GenerationParams()
    >>> print(f"Temperature: {params.temperature}, Max tokens: {params.max_tokens}")
    Temperature: 0.7, Max tokens: 1024

    Creative writing with high temperature:

    >>> from insideLLMs.contrib.adapters import GenerationParams, create_adapter
    >>> creative_params = GenerationParams(
    ...     temperature=1.2,
    ...     max_tokens=2000,
    ...     presence_penalty=0.5  # Encourage diversity
    ... )
    >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
    >>> response = adapter.generate("Write a poem about coding", creative_params)

    Precise, deterministic output:

    >>> from insideLLMs.contrib.adapters import GenerationParams
    >>> precise_params = GenerationParams(
    ...     temperature=0.0,  # Fully deterministic
    ...     max_tokens=500,
    ...     seed=42  # Reproducible
    ... )

    JSON output mode:

    >>> from insideLLMs.contrib.adapters import GenerationParams, create_adapter
    >>> json_params = GenerationParams(
    ...     temperature=0.2,
    ...     json_mode=True,
    ...     max_tokens=1000
    ... )
    >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
    >>> response = adapter.generate(
    ...     "List 3 programming languages with their year of creation",
    ...     json_params
    ... )
    >>> import json
    >>> data = json.loads(response.text)

    Using stop sequences:

    >>> from insideLLMs.contrib.adapters import GenerationParams
    >>> params = GenerationParams(
    ...     max_tokens=500,
    ...     stop_sequences=["\\n\\n", "END", "---"]
    ... )

    Function calling setup:

    >>> from insideLLMs.contrib.adapters import GenerationParams
    >>> functions = [
    ...     {
    ...         "name": "get_weather",
    ...         "description": "Get current weather for a location",
    ...         "parameters": {
    ...             "type": "object",
    ...             "properties": {
    ...                 "location": {"type": "string"}
    ...             },
    ...             "required": ["location"]
    ...         }
    ...     }
    ... ]
    >>> params = GenerationParams(functions=functions)

    Streaming response:

    >>> from insideLLMs.contrib.adapters import GenerationParams
    >>> stream_params = GenerationParams(
    ...     stream=True,
    ...     max_tokens=2000
    ... )

    Notes
    -----
    - Not all parameters are supported by all providers. Unsupported
      parameters may be silently ignored or cause errors.
    - `top_p` and `temperature` both control randomness; using both
      together may have unexpected effects.
    - `json_mode` requires models with JSON mode capability.

    See Also
    --------
    GenerationResult : Result returned from generation
    BaseAdapter.generate : Method that uses these parameters
    ModelCapability : Capabilities that affect parameter support
    """

    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)
    seed: Optional[int] = None
    stream: bool = False
    json_mode: bool = False
    functions: Optional[list[dict[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert generation parameters to dictionary.

        Creates a serializable dictionary containing all generation
        parameters. Useful for logging, debugging, or passing to
        APIs that accept dictionary configuration.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with all parameter keys:
            - temperature: float
            - max_tokens: int
            - top_p: float
            - top_k: int or None
            - frequency_penalty: float
            - presence_penalty: float
            - stop_sequences: list[str]
            - seed: int or None
            - stream: bool
            - json_mode: bool
            - functions: list[dict] or None

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import GenerationParams
        >>> params = GenerationParams(temperature=0.5, max_tokens=100)
        >>> data = params.to_dict()
        >>> print(data["temperature"])
        0.5
        >>> print(data["max_tokens"])
        100

        Merging with additional parameters:

        >>> from insideLLMs.contrib.adapters import GenerationParams
        >>> params = GenerationParams(temperature=0.8)
        >>> config = {**params.to_dict(), "custom_param": "value"}
        >>> print(config["temperature"])
        0.8

        Logging parameters:

        >>> import logging
        >>> from insideLLMs.contrib.adapters import GenerationParams
        >>> params = GenerationParams(temperature=0.9, json_mode=True)
        >>> logging.debug(f"Generation params: {params.to_dict()}")

        See Also
        --------
        AdapterConfig.to_dict : Similar method for adapter configuration
        GenerationResult.to_dict : Similar method for generation results
        """
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop_sequences": self.stop_sequences,
            "seed": self.seed,
            "stream": self.stream,
            "json_mode": self.json_mode,
            "functions": self.functions,
        }


@dataclass
class GenerationResult:
    """Result container for text generation responses.

    This dataclass encapsulates the complete result from an LLM generation
    request, including the generated text, token usage statistics, latency
    metrics, and any function call results.

    Parameters
    ----------
    text : str
        The generated text content from the model.
    model : str
        The model ID that generated this response.
    provider : str
        The provider name (e.g., "openai", "anthropic").
    tokens_input : int, optional
        Number of input tokens processed. Default is 0.
    tokens_output : int, optional
        Number of output tokens generated. Default is 0.
    latency_ms : float, optional
        Response latency in milliseconds. Default is 0.0.
    finish_reason : str, optional
        Reason generation stopped ("stop", "length", "function_call", etc.).
        Default is "stop".
    raw_response : dict[str, Any], optional
        Original provider response for debugging. Default is None.
    function_call : dict[str, Any], optional
        Function call result if the model invoked a function. Default is None.

    Attributes
    ----------
    text : str
        Generated text content.
    model : str
        Model identifier.
    provider : str
        Provider name.
    tokens_input : int
        Input token count.
    tokens_output : int
        Output token count.
    latency_ms : float
        Response time in milliseconds.
    finish_reason : str
        Generation termination reason.
    raw_response : dict[str, Any] or None
        Raw provider response.
    function_call : dict[str, Any] or None
        Function invocation details.

    Examples
    --------
    Basic usage after generation:

    >>> from insideLLMs.contrib.adapters import create_adapter
    >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
    >>> result = adapter.generate("What is 2+2?")
    >>> print(result.text)
    The sum of 2+2 is 4.
    >>> print(f"Tokens: {result.tokens_input} in, {result.tokens_output} out")
    Tokens: 6 in, 8 out

    Calculating cost from result:

    >>> from insideLLMs.contrib.adapters import create_adapter, get_model_info
    >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
    >>> result = adapter.generate("Explain machine learning")
    >>> info = get_model_info("gpt-4o")
    >>> cost = (result.tokens_input / 1000 * info.input_price_per_1k +
    ...         result.tokens_output / 1000 * info.output_price_per_1k)
    >>> print(f"Request cost: ${cost:.4f}")
    Request cost: $0.0085

    Checking finish reason:

    >>> from insideLLMs.contrib.adapters import create_adapter, GenerationParams
    >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
    >>> result = adapter.generate("Write a long essay", GenerationParams(max_tokens=10))
    >>> if result.finish_reason == "length":
    ...     print("Response was truncated due to token limit")
    Response was truncated due to token limit

    Handling function call results:

    >>> from insideLLMs.contrib.adapters import create_adapter, GenerationParams
    >>> functions = [{"name": "get_weather", "parameters": {...}}]
    >>> params = GenerationParams(functions=functions)
    >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
    >>> result = adapter.generate("What's the weather in Paris?", params)
    >>> if result.function_call:
    ...     print(f"Function called: {result.function_call['name']}")
    ...     print(f"Arguments: {result.function_call['arguments']}")
    Function called: get_weather
    Arguments: {"location": "Paris"}

    Measuring performance:

    >>> from insideLLMs.contrib.adapters import create_adapter
    >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
    >>> result = adapter.generate("Hello")
    >>> print(f"Response time: {result.latency_ms:.2f}ms")
    Response time: 523.45ms
    >>> throughput = result.tokens_output / (result.latency_ms / 1000)
    >>> print(f"Throughput: {throughput:.1f} tokens/second")
    Throughput: 45.2 tokens/second

    Serializing for logging:

    >>> from insideLLMs.contrib.adapters import create_mock_adapter
    >>> adapter = create_mock_adapter()
    >>> result = adapter.generate("Test")
    >>> data = result.to_dict()
    >>> print(data.keys())
    dict_keys(['text', 'model', 'provider', 'tokens_input', ...])

    See Also
    --------
    GenerationParams : Parameters that control generation
    BaseAdapter.generate : Method that returns GenerationResult
    HealthCheckResult : Similar result type for health checks
    """

    text: str
    model: str
    provider: str
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    raw_response: Optional[dict[str, Any]] = None
    function_call: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert generation result to dictionary.

        Creates a serializable dictionary containing the generation result.
        The raw_response field is excluded to keep the output clean and
        avoid serialization issues with complex provider responses.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with the following keys:
            - text: str
            - model: str
            - provider: str
            - tokens_input: int
            - tokens_output: int
            - latency_ms: float
            - finish_reason: str
            - function_call: dict or None

        Notes
        -----
        The raw_response field is intentionally excluded from the output
        as it may contain large, provider-specific data structures that
        are not easily serializable.

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> result = adapter.generate("Hello world")
        >>> data = result.to_dict()
        >>> print(data["model"])
        mock-model
        >>> print(data["provider"])
        mock

        Saving results to JSON:

        >>> import json
        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> result = adapter.generate("Test prompt")
        >>> json_str = json.dumps(result.to_dict(), indent=2)
        >>> print("text" in json_str)
        True

        Aggregating results:

        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> results = [adapter.generate(f"Prompt {i}") for i in range(3)]
        >>> total_tokens = sum(r.to_dict()["tokens_output"] for r in results)

        See Also
        --------
        AdapterConfig.to_dict : Similar method for configuration
        GenerationParams.to_dict : Similar method for parameters
        """
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
            "function_call": self.function_call,
        }


@dataclass
class HealthCheckResult:
    """Result of an adapter health check.

    This dataclass captures the outcome of a health check performed on
    an adapter, including connectivity status, response latency, and
    any error information.

    Parameters
    ----------
    healthy : bool
        Whether the adapter is healthy and able to process requests.
    latency_ms : float
        Time taken to perform the health check in milliseconds.
    error : str, optional
        Error message if the health check failed. Default is None.
    timestamp : float, optional
        Unix timestamp when the check was performed.
        Default is current time (time.time()).

    Attributes
    ----------
    healthy : bool
        Health status flag.
    latency_ms : float
        Check latency in milliseconds.
    error : str or None
        Error message if unhealthy.
    timestamp : float
        Unix timestamp of the check.

    Examples
    --------
    Performing a health check:

    >>> from insideLLMs.contrib.adapters import create_adapter
    >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
    >>> health = adapter.health_check()
    >>> print(f"Healthy: {health.healthy}, Latency: {health.latency_ms:.2f}ms")
    Healthy: True, Latency: 245.32ms

    Handling unhealthy adapter:

    >>> from insideLLMs.contrib.adapters import create_adapter
    >>> adapter = create_adapter("gpt-4o", api_key="invalid-key")
    >>> health = adapter.health_check()
    >>> if not health.healthy:
    ...     print(f"Health check failed: {health.error}")
    Health check failed: Invalid API key

    Monitoring multiple adapters:

    >>> from insideLLMs.contrib.adapters import AdapterFactory, Provider
    >>> factory = AdapterFactory()
    >>> adapters = [
    ...     factory.create("gpt-4o"),
    ...     factory.create("claude-3-opus")
    ... ]
    >>> for adapter in adapters:
    ...     health = adapter.health_check()
    ...     status = "OK" if health.healthy else "FAIL"
    ...     print(f"{adapter.model_id}: {status}")

    Using timestamp for monitoring:

    >>> import datetime
    >>> from insideLLMs.contrib.adapters import create_adapter
    >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
    >>> health = adapter.health_check()
    >>> check_time = datetime.datetime.fromtimestamp(health.timestamp)
    >>> print(f"Last check: {check_time.isoformat()}")
    Last check: 2024-01-15T10:30:45.123456

    Serializing for logging:

    >>> from insideLLMs.contrib.adapters import create_mock_adapter
    >>> adapter = create_mock_adapter()
    >>> health = adapter.health_check()
    >>> data = health.to_dict()
    >>> print(data["healthy"])
    True

    See Also
    --------
    BaseAdapter.health_check : Method that returns HealthCheckResult
    ConnectionMonitor : Aggregates health checks across adapters
    ConnectionInfo : Extended connection information
    """

    healthy: bool
    latency_ms: float
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert health check result to dictionary.

        Creates a serializable dictionary containing the health check
        outcome, suitable for logging, monitoring systems, or API responses.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with the following keys:
            - healthy: bool
            - latency_ms: float
            - error: str or None
            - timestamp: float

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> health = adapter.health_check()
        >>> data = health.to_dict()
        >>> print(data["healthy"])
        True
        >>> print("latency_ms" in data)
        True

        Sending to monitoring system:

        >>> import json
        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> health = adapter.health_check()
        >>> json_payload = json.dumps(health.to_dict())

        Aggregating health metrics:

        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> checks = [adapter.health_check() for _ in range(5)]
        >>> avg_latency = sum(c.to_dict()["latency_ms"] for c in checks) / len(checks)

        See Also
        --------
        GenerationResult.to_dict : Similar method for generation results
        ConnectionInfo.to_dict : Similar method for connection info
        """
        return {
            "healthy": self.healthy,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class BaseAdapter(ABC):
    """Abstract base class for all LLM adapters.

    This class defines the interface that all LLM adapters must implement,
    providing a consistent API across different providers. Concrete
    implementations handle provider-specific details while exposing
    a unified interface for text generation.

    Parameters
    ----------
    config : AdapterConfig
        Configuration object containing model ID, provider, API credentials,
        timeout settings, and other adapter options.

    Attributes
    ----------
    config : AdapterConfig
        The configuration used to initialize this adapter.
    model_id : str
        The model identifier (read-only property).
    provider : Provider
        The provider enum value (read-only property).
    status : AdapterStatus
        Current operational status (read-only property).

    Examples
    --------
    Creating a concrete adapter (using OpenAI as example):

    >>> from insideLLMs.contrib.adapters import AdapterConfig, Provider, OpenAIAdapter
    >>> config = AdapterConfig(
    ...     model_id="gpt-4o",
    ...     provider=Provider.OPENAI,
    ...     api_key="sk-..."
    ... )
    >>> adapter = OpenAIAdapter(config)
    >>> print(adapter.status)
    AdapterStatus.READY

    Using the factory (recommended):

    >>> from insideLLMs.contrib.adapters import create_adapter
    >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
    >>> response = adapter.generate("Hello, world!")
    >>> print(response.text)

    Implementing a custom adapter:

    >>> from insideLLMs.contrib.adapters import BaseAdapter, GenerationResult, GenerationParams
    >>> class MyCustomAdapter(BaseAdapter):
    ...     def generate(self, prompt, params=None, **kwargs):
    ...         # Custom implementation
    ...         return GenerationResult(
    ...             text="Custom response",
    ...             model=self.model_id,
    ...             provider=self.provider.value
    ...         )
    ...
    ...     def generate_chat(self, messages, params=None, **kwargs):
    ...         prompt = "\\n".join(m["content"] for m in messages)
    ...         return self.generate(prompt, params, **kwargs)

    Checking adapter statistics:

    >>> from insideLLMs.contrib.adapters import create_mock_adapter
    >>> adapter = create_mock_adapter()
    >>> _ = adapter.generate("Test 1")
    >>> _ = adapter.generate("Test 2")
    >>> stats = adapter.get_stats()
    >>> print(f"Requests: {stats['request_count']}")
    Requests: 2

    Notes
    -----
    - Subclasses must implement `generate` and `generate_chat` methods.
    - The `_record_request` and `_record_error` methods should be called
      by implementations to track statistics.
    - Health checks use a minimal generation request to verify connectivity.

    See Also
    --------
    OpenAIAdapter : Concrete implementation for OpenAI
    AnthropicAdapter : Concrete implementation for Anthropic
    MockAdapter : Testing implementation
    AdapterFactory : Factory for creating adapters
    """

    def __init__(self, config: AdapterConfig):
        """Initialize the adapter with configuration.

        Sets up the adapter's internal state including status tracking,
        request counting, and error tracking. The adapter starts in
        INITIALIZING status and should be set to READY by subclass
        implementations after successful client initialization.

        Parameters
        ----------
        config : AdapterConfig
            Configuration object containing all adapter settings including
            model ID, provider, API credentials, and operational parameters.

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import AdapterConfig, Provider, BaseAdapter
        >>> config = AdapterConfig(
        ...     model_id="test-model",
        ...     provider=Provider.MOCK,
        ...     timeout=30.0
        ... )
        >>> # Note: BaseAdapter is abstract, use concrete implementation
        >>> from insideLLMs.contrib.adapters import MockAdapter
        >>> adapter = MockAdapter(config)
        >>> print(adapter.config.timeout)
        30.0
        """
        self.config = config
        self._status = AdapterStatus.INITIALIZING
        self._last_request_time: float = 0
        self._request_count: int = 0
        self._error_count: int = 0
        self._client: Optional[Any] = None

    @property
    def model_id(self) -> str:
        """Get the model identifier.

        Returns
        -------
        str
            The model ID from the adapter configuration.

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter("my-custom-model")
        >>> print(adapter.model_id)
        my-custom-model
        """
        return self.config.model_id

    @property
    def provider(self) -> Provider:
        """Get the provider.

        Returns
        -------
        Provider
            The provider enum value from the adapter configuration.

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import create_mock_adapter, Provider
        >>> adapter = create_mock_adapter()
        >>> print(adapter.provider)
        Provider.MOCK
        >>> print(adapter.provider == Provider.MOCK)
        True
        """
        return self.config.provider

    @property
    def status(self) -> AdapterStatus:
        """Get the current operational status.

        Returns
        -------
        AdapterStatus
            Current status of the adapter (READY, INITIALIZING, ERROR, etc.).

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import create_mock_adapter, AdapterStatus
        >>> adapter = create_mock_adapter()
        >>> print(adapter.status)
        AdapterStatus.READY
        >>> print(adapter.status == AdapterStatus.READY)
        True
        """
        return self._status

    @abstractmethod
    def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate text from a prompt.

        This is the primary method for text generation. Implementations
        should send the prompt to the LLM and return the generated text
        along with usage statistics.

        Parameters
        ----------
        prompt : str
            The input text prompt to send to the model.
        params : GenerationParams, optional
            Parameters controlling generation (temperature, max_tokens, etc.).
            If None, default parameters are used.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        GenerationResult
            Result object containing generated text, token counts,
            latency, and other metadata.

        Raises
        ------
        RuntimeError
            If the API call fails or returns an error.

        Examples
        --------
        Basic generation:

        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> result = adapter.generate("What is Python?")
        >>> print(result.text)
        This is a mock response.

        With custom parameters:

        >>> from insideLLMs.contrib.adapters import create_mock_adapter, GenerationParams
        >>> adapter = create_mock_adapter()
        >>> params = GenerationParams(temperature=0.5, max_tokens=100)
        >>> result = adapter.generate("Explain AI", params)
        >>> print(result.model)
        mock-model

        Handling errors:

        >>> from insideLLMs.contrib.adapters import create_adapter
        >>> adapter = create_adapter("gpt-4o", api_key="invalid")
        >>> try:
        ...     result = adapter.generate("Hello")
        ... except RuntimeError as e:
        ...     print(f"Generation failed: {e}")

        See Also
        --------
        generate_chat : Chat-based generation with message history
        GenerationParams : Available generation parameters
        GenerationResult : Structure of the return value
        """
        pass

    @abstractmethod
    def generate_chat(
        self,
        messages: list[dict[str, str]],
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate text from chat messages.

        This method handles multi-turn conversations with message history.
        Each message should have 'role' (system/user/assistant) and
        'content' fields.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dictionaries, each containing:
            - role: str ("system", "user", or "assistant")
            - content: str (the message text)
        params : GenerationParams, optional
            Parameters controlling generation. If None, defaults are used.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        GenerationResult
            Result object containing the assistant's response and metadata.

        Raises
        ------
        RuntimeError
            If the API call fails or returns an error.

        Examples
        --------
        Simple chat:

        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> messages = [
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> result = adapter.generate_chat(messages)
        >>> print(result.text)
        This is a mock response.

        Multi-turn conversation:

        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "What is 2+2?"},
        ...     {"role": "assistant", "content": "2+2 equals 4."},
        ...     {"role": "user", "content": "What about 3+3?"}
        ... ]
        >>> result = adapter.generate_chat(messages)

        With custom parameters:

        >>> from insideLLMs.contrib.adapters import create_mock_adapter, GenerationParams
        >>> adapter = create_mock_adapter()
        >>> messages = [{"role": "user", "content": "Write a poem"}]
        >>> params = GenerationParams(temperature=1.0, max_tokens=200)
        >>> result = adapter.generate_chat(messages, params)

        See Also
        --------
        generate : Single-prompt generation
        GenerationParams : Available generation parameters
        GenerationResult : Structure of the return value
        """
        pass

    def health_check(self) -> HealthCheckResult:
        """Check if the adapter is healthy and responsive.

        Performs a minimal generation request to verify that the adapter
        can successfully communicate with the LLM provider. This is useful
        for monitoring, load balancing, and failover decisions.

        Returns
        -------
        HealthCheckResult
            Result object containing:
            - healthy: bool indicating if the check passed
            - latency_ms: time taken for the check
            - error: error message if unhealthy
            - timestamp: when the check was performed

        Examples
        --------
        Basic health check:

        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> health = adapter.health_check()
        >>> print(f"Healthy: {health.healthy}")
        Healthy: True

        Monitoring with latency:

        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> health = adapter.health_check()
        >>> if health.healthy:
        ...     print(f"Response time: {health.latency_ms:.2f}ms")
        ... else:
        ...     print(f"Check failed: {health.error}")
        Response time: ...

        Periodic health monitoring:

        >>> import time
        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> def monitor(adapter, interval=60):
        ...     health = adapter.health_check()
        ...     return {
        ...         "healthy": health.healthy,
        ...         "latency": health.latency_ms,
        ...         "checked_at": health.timestamp
        ...     }
        >>> status = monitor(adapter)
        >>> print(status["healthy"])
        True

        See Also
        --------
        HealthCheckResult : Structure of the return value
        ConnectionMonitor : Higher-level monitoring utility
        get_stats : Get adapter statistics without API call
        """
        start = time.time()
        try:
            # Simple check - generate minimal output
            self.generate("Hello", GenerationParams(max_tokens=5))
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                healthy=True,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                healthy=False,
                latency_ms=latency,
                error=str(e),
            )

    def get_stats(self) -> dict[str, Any]:
        """Get adapter usage statistics.

        Returns current statistics about adapter usage including request
        counts, error counts, and timing information. This does not make
        any API calls.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - model_id: str - The model identifier
            - provider: str - The provider name
            - status: str - Current status
            - request_count: int - Total requests made
            - error_count: int - Total errors encountered
            - last_request_time: float - Unix timestamp of last request

        Examples
        --------
        Getting statistics after some requests:

        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> _ = adapter.generate("Test 1")
        >>> _ = adapter.generate("Test 2")
        >>> stats = adapter.get_stats()
        >>> print(f"Total requests: {stats['request_count']}")
        Total requests: 2
        >>> print(f"Model: {stats['model_id']}")
        Model: mock-model

        Monitoring error rates:

        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> stats = adapter.get_stats()
        >>> error_rate = stats['error_count'] / max(stats['request_count'], 1)
        >>> print(f"Error rate: {error_rate:.2%}")
        Error rate: 0.00%

        Logging adapter metrics:

        >>> import json
        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> stats = adapter.get_stats()
        >>> print(json.dumps(stats, indent=2))
        {
          "model_id": "mock-model",
          "provider": "mock",
          ...
        }

        See Also
        --------
        health_check : Active health verification with API call
        AdapterStatus : Possible status values
        """
        return {
            "model_id": self.model_id,
            "provider": self.provider.value,
            "status": self._status.value,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "last_request_time": self._last_request_time,
        }

    def _record_request(self) -> None:
        """Record that a request was made.

        Internal method to track request statistics. Should be called
        by subclass implementations at the start of generate methods.

        This method increments the request counter and updates the
        last request timestamp.

        Examples
        --------
        Usage in a custom adapter implementation:

        >>> from insideLLMs.contrib.adapters import BaseAdapter, GenerationResult
        >>> class CustomAdapter(BaseAdapter):
        ...     def generate(self, prompt, params=None, **kwargs):
        ...         self._record_request()  # Track the request
        ...         # ... actual generation logic ...
        ...         return GenerationResult(text="...", model=self.model_id,
        ...                                 provider=self.provider.value)

        Notes
        -----
        This is an internal method intended for use by adapter implementations.
        """
        self._request_count += 1
        self._last_request_time = time.time()

    def _record_error(self) -> None:
        """Record that an error occurred.

        Internal method to track error statistics. Should be called
        by subclass implementations when an error occurs during generation.

        This method increments the error counter.

        Examples
        --------
        Usage in a custom adapter implementation:

        >>> from insideLLMs.contrib.adapters import BaseAdapter, GenerationResult
        >>> class CustomAdapter(BaseAdapter):
        ...     def generate(self, prompt, params=None, **kwargs):
        ...         self._record_request()
        ...         try:
        ...             # ... actual generation logic ...
        ...             pass
        ...         except Exception as e:
        ...             self._record_error()  # Track the error
        ...             raise

        Notes
        -----
        This is an internal method intended for use by adapter implementations.
        """
        self._error_count += 1


class MockAdapter(BaseAdapter):
    """Mock adapter for testing without making real API calls.

    This adapter provides a way to test code that uses adapters without
    incurring API costs or requiring network connectivity. It supports
    configurable responses based on prompt patterns.

    Parameters
    ----------
    config : AdapterConfig
        Configuration object. The provider should be Provider.MOCK.

    Attributes
    ----------
    config : AdapterConfig
        The adapter configuration.
    model_id : str
        The mock model identifier.
    provider : Provider
        Always Provider.MOCK.
    status : AdapterStatus
        Always AdapterStatus.READY after initialization.

    Examples
    --------
    Basic usage:

    >>> from insideLLMs.contrib.adapters import create_mock_adapter
    >>> adapter = create_mock_adapter()
    >>> result = adapter.generate("Hello!")
    >>> print(result.text)
    This is a mock response.

    Custom responses:

    >>> from insideLLMs.contrib.adapters import create_mock_adapter
    >>> adapter = create_mock_adapter()
    >>> adapter.set_response("weather", "It's sunny today!")
    >>> adapter.set_response("time", "It's 3:00 PM.")
    >>> print(adapter.generate("What's the weather?").text)
    It's sunny today!
    >>> print(adapter.generate("What time is it?").text)
    It's 3:00 PM.

    Setting default response:

    >>> from insideLLMs.contrib.adapters import create_mock_adapter
    >>> adapter = create_mock_adapter()
    >>> adapter.set_default_response("Custom default response")
    >>> result = adapter.generate("Random prompt")
    >>> print(result.text)
    Custom default response

    Testing with chat format:

    >>> from insideLLMs.contrib.adapters import create_mock_adapter
    >>> adapter = create_mock_adapter()
    >>> messages = [
    ...     {"role": "system", "content": "You are helpful."},
    ...     {"role": "user", "content": "Hello!"}
    ... ]
    >>> result = adapter.generate_chat(messages)
    >>> print(result.provider)
    mock

    Using in unit tests:

    >>> from insideLLMs.contrib.adapters import create_mock_adapter
    >>> def test_my_function():
    ...     adapter = create_mock_adapter()
    ...     adapter.set_response("test", "Expected output")
    ...     result = adapter.generate("test query")
    ...     assert result.text == "Expected output"

    See Also
    --------
    create_mock_adapter : Convenience function to create MockAdapter
    BaseAdapter : Parent class defining the interface
    AdapterFactory : Factory that can create mock adapters
    """

    def __init__(self, config: AdapterConfig):
        """Initialize the mock adapter.

        Sets up the mock adapter with an immediately READY status and
        initializes the response mapping dictionary.

        Parameters
        ----------
        config : AdapterConfig
            Configuration object for the adapter.

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import AdapterConfig, Provider, MockAdapter
        >>> config = AdapterConfig(model_id="test", provider=Provider.MOCK)
        >>> adapter = MockAdapter(config)
        >>> print(adapter.status)
        AdapterStatus.READY
        """
        super().__init__(config)
        self._status = AdapterStatus.READY
        self._responses: dict[str, str] = {}
        self._default_response = "This is a mock response."

    def set_response(self, prompt_pattern: str, response: str) -> None:
        """Set a response for prompts containing a pattern.

        When a prompt contains the specified pattern, the mock adapter
        will return the associated response instead of the default.
        Patterns are checked in the order they were added.

        Parameters
        ----------
        prompt_pattern : str
            A substring to match in incoming prompts.
        response : str
            The response to return when the pattern is matched.

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> adapter.set_response("hello", "Hi there!")
        >>> adapter.set_response("bye", "Goodbye!")
        >>> print(adapter.generate("say hello").text)
        Hi there!
        >>> print(adapter.generate("bye now").text)
        Goodbye!

        Pattern matching is case-sensitive:

        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> adapter.set_response("Hello", "Matched!")
        >>> print(adapter.generate("Hello world").text)
        Matched!
        >>> print(adapter.generate("hello world").text)  # No match
        This is a mock response.

        See Also
        --------
        set_default_response : Set response for non-matching prompts
        """
        self._responses[prompt_pattern] = response

    def set_default_response(self, response: str) -> None:
        """Set the default response for non-matching prompts.

        This response is returned when no pattern from `set_response`
        matches the incoming prompt.

        Parameters
        ----------
        response : str
            The default response text.

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> adapter.set_default_response("I don't understand.")
        >>> result = adapter.generate("random query")
        >>> print(result.text)
        I don't understand.

        See Also
        --------
        set_response : Set responses for specific patterns
        """
        self._default_response = response

    def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate a mock response for a prompt.

        Checks configured patterns against the prompt and returns the
        matching response, or the default response if no pattern matches.
        Token counts are estimated by splitting on whitespace.

        Parameters
        ----------
        prompt : str
            The input prompt text.
        params : GenerationParams, optional
            Generation parameters. Only max_tokens is used to truncate
            the response. Default is None.
        **kwargs
            Ignored for mock adapter.

        Returns
        -------
        GenerationResult
            Mock result with text, estimated token counts, and 10ms latency.

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> result = adapter.generate("What is Python?")
        >>> print(result.text)
        This is a mock response.
        >>> print(result.latency_ms)
        10.0

        With max_tokens limit:

        >>> from insideLLMs.contrib.adapters import create_mock_adapter, GenerationParams
        >>> adapter = create_mock_adapter()
        >>> adapter.set_default_response("This is a longer response text")
        >>> params = GenerationParams(max_tokens=10)
        >>> result = adapter.generate("test", params)
        >>> print(len(result.text) <= 10)
        True

        See Also
        --------
        generate_chat : Chat-based generation
        set_response : Configure pattern-based responses
        """
        self._record_request()

        # Check for matching pattern
        response = self._default_response
        for pattern, resp in self._responses.items():
            if pattern in prompt:
                response = resp
                break

        params = params or GenerationParams()

        return GenerationResult(
            text=response[: params.max_tokens],
            model=self.model_id,
            provider=self.provider.value,
            tokens_input=len(prompt.split()),
            tokens_output=len(response.split()),
            latency_ms=10.0,
        )

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate a mock response for chat messages.

        Converts chat messages to a single prompt by concatenating
        role and content, then delegates to the generate method.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dictionaries with 'role' and 'content' keys.
        params : GenerationParams, optional
            Generation parameters passed to generate.
        **kwargs
            Additional arguments passed to generate.

        Returns
        -------
        GenerationResult
            Mock result from the generate method.

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> messages = [
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> result = adapter.generate_chat(messages)
        >>> print(result.provider)
        mock

        Multi-turn conversation:

        >>> from insideLLMs.contrib.adapters import create_mock_adapter
        >>> adapter = create_mock_adapter()
        >>> adapter.set_response("favorite color", "Blue!")
        >>> messages = [
        ...     {"role": "user", "content": "What is your favorite color?"}
        ... ]
        >>> result = adapter.generate_chat(messages)
        >>> print(result.text)
        Blue!

        See Also
        --------
        generate : Single-prompt generation
        """
        # Convert messages to prompt
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return self.generate(prompt, params, **kwargs)


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI API models (GPT-4, GPT-3.5, etc.).

    This adapter provides integration with OpenAI's chat completions API,
    supporting all GPT models including GPT-4o, GPT-4 Turbo, and GPT-3.5 Turbo.
    It handles client initialization, request formatting, and response parsing.

    Parameters
    ----------
    config : AdapterConfig
        Configuration object containing:
        - model_id: OpenAI model name (e.g., "gpt-4o", "gpt-3.5-turbo")
        - api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        - api_base: Custom API base URL (optional, for Azure/proxies)
        - organization_id: OpenAI organization ID (optional)
        - timeout: Request timeout in seconds

    Attributes
    ----------
    config : AdapterConfig
        The adapter configuration.
    model_id : str
        The OpenAI model identifier.
    provider : Provider
        Always Provider.OPENAI.
    status : AdapterStatus
        READY if client initialized successfully, ERROR otherwise.

    Examples
    --------
    Using the factory (recommended):

    >>> from insideLLMs.contrib.adapters import create_adapter
    >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
    >>> result = adapter.generate("Explain recursion in programming")
    >>> print(result.text[:50])
    Recursion is a programming technique where...

    Direct instantiation:

    >>> from insideLLMs.contrib.adapters import AdapterConfig, Provider, OpenAIAdapter
    >>> config = AdapterConfig(
    ...     model_id="gpt-4o",
    ...     provider=Provider.OPENAI,
    ...     api_key="sk-...",
    ...     timeout=30.0
    ... )
    >>> adapter = OpenAIAdapter(config)
    >>> if adapter.status == AdapterStatus.READY:
    ...     result = adapter.generate("Hello!")

    Chat with conversation history:

    >>> from insideLLMs.contrib.adapters import create_adapter
    >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
    >>> messages = [
    ...     {"role": "system", "content": "You are a Python expert."},
    ...     {"role": "user", "content": "How do I read a file?"}
    ... ]
    >>> result = adapter.generate_chat(messages)

    Using Azure OpenAI:

    >>> from insideLLMs.contrib.adapters import AdapterConfig, Provider, OpenAIAdapter
    >>> config = AdapterConfig(
    ...     model_id="gpt-4",
    ...     provider=Provider.OPENAI,
    ...     api_key="your-azure-key",
    ...     api_base="https://your-resource.openai.azure.com/"
    ... )
    >>> adapter = OpenAIAdapter(config)

    Notes
    -----
    - Requires the `openai` package to be installed.
    - The adapter uses the chat completions API for all requests.
    - Single prompts are automatically wrapped in a user message.

    See Also
    --------
    AnthropicAdapter : Similar adapter for Anthropic's Claude
    AdapterFactory : Factory for creating adapters
    create_adapter : Convenience function with auto-detection
    """

    def __init__(self, config: AdapterConfig):
        """Initialize the OpenAI adapter.

        Creates the OpenAI client with the provided configuration. If
        initialization fails (missing package or invalid credentials),
        the adapter status is set to ERROR.

        Parameters
        ----------
        config : AdapterConfig
            Configuration object with API credentials and settings.

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import AdapterConfig, Provider, OpenAIAdapter
        >>> config = AdapterConfig(
        ...     model_id="gpt-4o",
        ...     provider=Provider.OPENAI,
        ...     api_key="sk-..."
        ... )
        >>> adapter = OpenAIAdapter(config)
        >>> print(adapter.model_id)
        gpt-4o
        """
        super().__init__(config)
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client.

        Creates an OpenAI client instance using the configuration settings.
        Sets adapter status to READY on success or ERROR on failure.

        This is called automatically during __init__ and should not
        typically be called directly.

        Raises
        ------
        None
            Exceptions are caught and result in ERROR status.

        Notes
        -----
        Failure reasons include:
        - ImportError: openai package not installed
        - Invalid API key or configuration
        - Network connectivity issues
        """
        try:
            import openai

            self._client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                organization=self.config.organization_id,
                timeout=self.config.timeout,
            )
            self._status = AdapterStatus.READY
        except ImportError:
            self._status = AdapterStatus.ERROR
        except Exception:
            self._status = AdapterStatus.ERROR

    def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate text from a prompt using OpenAI.

        Wraps the prompt in a user message and delegates to generate_chat.
        This provides a simple interface for single-turn generation.

        Parameters
        ----------
        prompt : str
            The input text prompt.
        params : GenerationParams, optional
            Generation parameters (temperature, max_tokens, etc.).
        **kwargs
            Additional arguments passed to generate_chat.

        Returns
        -------
        GenerationResult
            Result containing generated text, token usage, and latency.

        Raises
        ------
        RuntimeError
            If the OpenAI API call fails.

        Examples
        --------
        >>> from insideLLMs.contrib.adapters import create_adapter
        >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
        >>> result = adapter.generate("What is Python?")
        >>> print(result.provider)
        openai

        With custom temperature:

        >>> from insideLLMs.contrib.adapters import create_adapter, GenerationParams
        >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
        >>> params = GenerationParams(temperature=0.0, max_tokens=100)
        >>> result = adapter.generate("2+2=", params)

        See Also
        --------
        generate_chat : Chat-based generation with message history
        """
        messages = [{"role": "user", "content": prompt}]
        return self.generate_chat(messages, params, **kwargs)

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate text from chat messages using OpenAI.

        Sends the message history to OpenAI's chat completions API and
        returns the assistant's response along with usage statistics.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dictionaries with 'role' and 'content' keys.
            Roles can be "system", "user", or "assistant".
        params : GenerationParams, optional
            Generation parameters including:
            - temperature: Sampling temperature (0.0 to 2.0)
            - max_tokens: Maximum output tokens
            - top_p: Nucleus sampling threshold
            - frequency_penalty: Token frequency penalty
            - presence_penalty: Token presence penalty
            - stop_sequences: Stop sequences
            - seed: Random seed for reproducibility
        **kwargs
            Additional arguments (currently unused).

        Returns
        -------
        GenerationResult
            Result object containing:
            - text: The generated response
            - model: Model ID used
            - provider: "openai"
            - tokens_input: Prompt token count
            - tokens_output: Completion token count
            - latency_ms: Request duration
            - finish_reason: Why generation stopped

        Raises
        ------
        RuntimeError
            If the API call fails with details in the error message.

        Examples
        --------
        Simple conversation:

        >>> from insideLLMs.contrib.adapters import create_adapter
        >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
        >>> messages = [{"role": "user", "content": "Hi!"}]
        >>> result = adapter.generate_chat(messages)
        >>> print(result.finish_reason)
        stop

        With system prompt:

        >>> from insideLLMs.contrib.adapters import create_adapter
        >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
        >>> messages = [
        ...     {"role": "system", "content": "Respond in French only."},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> result = adapter.generate_chat(messages)

        Tracking token usage:

        >>> from insideLLMs.contrib.adapters import create_adapter
        >>> adapter = create_adapter("gpt-4o", api_key="sk-...")
        >>> messages = [{"role": "user", "content": "Count to 10"}]
        >>> result = adapter.generate_chat(messages)
        >>> print(f"Used {result.tokens_input + result.tokens_output} total tokens")

        See Also
        --------
        generate : Simple single-prompt generation
        GenerationParams : Available generation parameters
        """
        self._record_request()
        params = params or GenerationParams()

        start = time.time()

        try:
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
                frequency_penalty=params.frequency_penalty,
                presence_penalty=params.presence_penalty,
                stop=params.stop_sequences or None,
                seed=params.seed,
            )

            latency = (time.time() - start) * 1000

            return GenerationResult(
                text=response.choices[0].message.content or "",
                model=self.model_id,
                provider=Provider.OPENAI.value,
                tokens_input=response.usage.prompt_tokens if response.usage else 0,
                tokens_output=response.usage.completion_tokens if response.usage else 0,
                latency_ms=latency,
                finish_reason=response.choices[0].finish_reason or "stop",
            )
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"OpenAI API error: {e}")


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic models."""

    def __init__(self, config: AdapterConfig):
        """Initialize Anthropic adapter."""
        super().__init__(config)
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout,
            )
            self._status = AdapterStatus.READY
        except ImportError:
            self._status = AdapterStatus.ERROR
        except Exception:
            self._status = AdapterStatus.ERROR

    def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate using Anthropic."""
        messages = [{"role": "user", "content": prompt}]
        return self.generate_chat(messages, params, **kwargs)

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate using Anthropic chat."""
        self._record_request()
        params = params or GenerationParams()

        start = time.time()

        try:
            response = self._client.messages.create(
                model=self.model_id,
                messages=messages,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                stop_sequences=params.stop_sequences or None,
            )

            latency = (time.time() - start) * 1000

            text = ""
            if response.content:
                text = response.content[0].text

            return GenerationResult(
                text=text,
                model=self.model_id,
                provider=Provider.ANTHROPIC.value,
                tokens_input=response.usage.input_tokens,
                tokens_output=response.usage.output_tokens,
                latency_ms=latency,
                finish_reason=response.stop_reason or "stop",
            )
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"Anthropic API error: {e}")


class ModelRegistry:
    """Registry of known models and their metadata."""

    def __init__(self):
        """Initialize registry."""
        self._models: dict[str, AdapterModelInfo] = {}
        self._aliases: dict[str, str] = {}
        self._setup_default_models()

    def _setup_default_models(self) -> None:
        """Setup default model definitions."""
        # OpenAI models
        self.register(
            AdapterModelInfo(
                model_id="gpt-4o",
                provider=Provider.OPENAI,
                display_name="GPT-4o",
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                    ModelCapability.JSON_MODE,
                ],
                context_window=128000,
                max_output_tokens=16384,
                input_price_per_1k=0.005,
                output_price_per_1k=0.015,
                aliases=["gpt-4o-2024-08-06"],
            )
        )

        self.register(
            AdapterModelInfo(
                model_id="gpt-4-turbo",
                provider=Provider.OPENAI,
                display_name="GPT-4 Turbo",
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                    ModelCapability.JSON_MODE,
                ],
                context_window=128000,
                max_output_tokens=4096,
                input_price_per_1k=0.01,
                output_price_per_1k=0.03,
                aliases=["gpt-4-turbo-preview", "gpt-4-1106-preview"],
            )
        )

        self.register(
            AdapterModelInfo(
                model_id="gpt-3.5-turbo",
                provider=Provider.OPENAI,
                display_name="GPT-3.5 Turbo",
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING,
                    ModelCapability.JSON_MODE,
                ],
                context_window=16385,
                max_output_tokens=4096,
                input_price_per_1k=0.0005,
                output_price_per_1k=0.0015,
            )
        )

        # Anthropic models
        self.register(
            AdapterModelInfo(
                model_id="claude-3-5-sonnet-20241022",
                provider=Provider.ANTHROPIC,
                display_name="Claude 3.5 Sonnet",
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.STREAMING,
                ],
                context_window=200000,
                max_output_tokens=8192,
                input_price_per_1k=0.003,
                output_price_per_1k=0.015,
                aliases=["claude-3.5-sonnet", "claude-sonnet"],
            )
        )

        self.register(
            AdapterModelInfo(
                model_id="claude-3-opus-20240229",
                provider=Provider.ANTHROPIC,
                display_name="Claude 3 Opus",
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.STREAMING,
                ],
                context_window=200000,
                max_output_tokens=4096,
                input_price_per_1k=0.015,
                output_price_per_1k=0.075,
                aliases=["claude-3-opus", "claude-opus"],
            )
        )

        self.register(
            AdapterModelInfo(
                model_id="claude-3-haiku-20240307",
                provider=Provider.ANTHROPIC,
                display_name="Claude 3 Haiku",
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.STREAMING,
                ],
                context_window=200000,
                max_output_tokens=4096,
                input_price_per_1k=0.00025,
                output_price_per_1k=0.00125,
                aliases=["claude-3-haiku", "claude-haiku"],
            )
        )

        # Mistral models
        self.register(
            AdapterModelInfo(
                model_id="mistral-large-latest",
                provider=Provider.MISTRAL,
                display_name="Mistral Large",
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.STREAMING,
                    ModelCapability.FUNCTION_CALLING,
                ],
                context_window=128000,
                max_output_tokens=4096,
                input_price_per_1k=0.002,
                output_price_per_1k=0.006,
                aliases=["mistral-large"],
            )
        )

        self.register(
            AdapterModelInfo(
                model_id="mistral-small-latest",
                provider=Provider.MISTRAL,
                display_name="Mistral Small",
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.STREAMING,
                ],
                context_window=32000,
                max_output_tokens=4096,
                input_price_per_1k=0.0002,
                output_price_per_1k=0.0006,
                aliases=["mistral-small"],
            )
        )

    def register(self, model_info: AdapterModelInfo) -> None:
        """Register a model."""
        self._models[model_info.model_id] = model_info

        # Register aliases
        for alias in model_info.aliases:
            self._aliases[alias] = model_info.model_id

    def get(self, model_id: str) -> Optional[AdapterModelInfo]:
        """Get model info by ID or alias."""
        # Check direct lookup
        if model_id in self._models:
            return self._models[model_id]

        # Check aliases
        resolved_id = self._aliases.get(model_id)
        if resolved_id:
            return self._models.get(resolved_id)

        return None

    def resolve_alias(self, model_id: str) -> str:
        """Resolve model alias to canonical ID."""
        return self._aliases.get(model_id, model_id)

    def list_models(self, provider: Optional[Provider] = None) -> list[AdapterModelInfo]:
        """List all models, optionally filtered by provider."""
        models = list(self._models.values())
        if provider:
            models = [m for m in models if m.provider == provider]
        return models

    def get_by_capability(self, capability: ModelCapability) -> list[AdapterModelInfo]:
        """Get models with a specific capability."""
        return [m for m in self._models.values() if capability in m.capabilities]


class ProviderDetector:
    """Detect provider from model name."""

    # Model prefix patterns
    PATTERNS = {
        Provider.OPENAI: ["gpt-", "text-davinci", "text-embedding"],
        Provider.ANTHROPIC: ["claude-"],
        Provider.GOOGLE: ["gemini-", "palm-"],
        Provider.COHERE: ["command-", "embed-"],
        Provider.MISTRAL: ["mistral-", "mixtral-"],
        Provider.GROQ: ["llama-", "gemma-"],  # Groq hosts various models
    }

    @classmethod
    def detect(cls, model_id: str) -> Optional[Provider]:
        """Detect provider from model ID."""
        model_lower = model_id.lower()

        for provider, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                if model_lower.startswith(pattern):
                    return provider

        return None


class AdapterFactory:
    """Factory for creating model adapters."""

    # Adapter classes by provider
    ADAPTER_CLASSES: dict[Provider, type[BaseAdapter]] = {
        Provider.OPENAI: OpenAIAdapter,
        Provider.ANTHROPIC: AnthropicAdapter,
        Provider.MOCK: MockAdapter,
    }

    def __init__(self):
        """Initialize factory."""
        self._registry = ModelRegistry()
        self._adapters: dict[str, BaseAdapter] = {}
        self._api_keys: dict[Provider, str] = {}
        self._default_configs: dict[Provider, dict[str, Any]] = {}

    def set_api_key(self, provider: Provider, api_key: str) -> None:
        """Set API key for a provider."""
        self._api_keys[provider] = api_key

    def set_default_config(
        self,
        provider: Provider,
        **kwargs,
    ) -> None:
        """Set default configuration for a provider."""
        self._default_configs[provider] = kwargs

    def create(
        self,
        model_id: str,
        provider: Optional[Provider] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> BaseAdapter:
        """Create an adapter for a model.

        Args:
            model_id: Model identifier
            provider: Provider (auto-detected if not specified)
            api_key: API key (uses stored key if not specified)
            **kwargs: Additional config options

        Returns:
            Configured adapter
        """
        # Resolve alias
        resolved_id = self._registry.resolve_alias(model_id)

        # Detect provider if not specified
        if provider is None:
            # Try registry first
            model_info = self._registry.get(resolved_id)
            provider = model_info.provider if model_info else ProviderDetector.detect(resolved_id)

            if provider is None:
                raise ValueError(f"Could not detect provider for model: {model_id}")

        # Get API key
        if api_key is None:
            api_key = self._api_keys.get(provider)

        # Get adapter class
        adapter_class = self.ADAPTER_CLASSES.get(provider)
        if adapter_class is None:
            raise ValueError(f"No adapter available for provider: {provider}")

        # Merge configs
        default_config = self._default_configs.get(provider, {})
        merged_config = {**default_config, **kwargs}

        # Create config
        config = AdapterConfig(
            model_id=resolved_id,
            provider=provider,
            api_key=api_key,
            **merged_config,
        )

        # Create adapter
        adapter = adapter_class(config)

        # Cache adapter
        cache_key = f"{provider.value}:{resolved_id}"
        self._adapters[cache_key] = adapter

        return adapter

    def get_cached(self, model_id: str, provider: Provider) -> Optional[BaseAdapter]:
        """Get cached adapter if available."""
        cache_key = f"{provider.value}:{model_id}"
        return self._adapters.get(cache_key)

    def list_available(self) -> list[Provider]:
        """List providers with available adapters."""
        return list(self.ADAPTER_CLASSES.keys())

    def get_model_info(self, model_id: str) -> Optional[AdapterModelInfo]:
        """Get model information."""
        return self._registry.get(model_id)

    def list_models(self, provider: Optional[Provider] = None) -> list[AdapterModelInfo]:
        """List known models."""
        return self._registry.list_models(provider)


class FallbackChain:
    """Chain of adapters with fallback logic."""

    def __init__(
        self,
        adapters: list[BaseAdapter],
        fallback_on_error: bool = True,
        fallback_on_rate_limit: bool = True,
    ):
        """Initialize fallback chain.

        Args:
            adapters: List of adapters in priority order
            fallback_on_error: Whether to fallback on errors
            fallback_on_rate_limit: Whether to fallback on rate limits
        """
        self.adapters = adapters
        self.fallback_on_error = fallback_on_error
        self.fallback_on_rate_limit = fallback_on_rate_limit
        self._fallback_count = 0

    def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate with fallback logic."""
        last_error = None

        for adapter in self.adapters:
            try:
                return adapter.generate(prompt, params, **kwargs)
            except Exception as e:
                last_error = e
                self._fallback_count += 1

                # Check if we should fallback
                if not self.fallback_on_error:
                    raise

        # All adapters failed
        raise RuntimeError(f"All adapters failed. Last error: {last_error}")

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate chat with fallback logic."""
        last_error = None

        for adapter in self.adapters:
            try:
                return adapter.generate_chat(messages, params, **kwargs)
            except Exception as e:
                last_error = e
                self._fallback_count += 1

                if not self.fallback_on_error:
                    raise

        raise RuntimeError(f"All adapters failed. Last error: {last_error}")

    def get_stats(self) -> dict[str, Any]:
        """Get chain statistics."""
        return {
            "adapter_count": len(self.adapters),
            "fallback_count": self._fallback_count,
            "adapters": [a.get_stats() for a in self.adapters],
        }


class AdapterPool:
    """Pool of adapters for load balancing."""

    def __init__(
        self,
        adapters: list[BaseAdapter],
        strategy: str = "round_robin",
    ):
        """Initialize adapter pool.

        Args:
            adapters: List of adapters
            strategy: Selection strategy ('round_robin', 'least_requests', 'random')
        """
        self.adapters = adapters
        self.strategy = strategy
        self._current_index = 0
        self._request_counts: dict[str, int] = {}

    def get_adapter(self) -> BaseAdapter:
        """Get next adapter based on strategy."""
        if not self.adapters:
            raise ValueError("No adapters in pool")

        if self.strategy == "round_robin":
            adapter = self.adapters[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.adapters)
            return adapter

        elif self.strategy == "least_requests":
            return min(self.adapters, key=lambda a: a._request_count)

        elif self.strategy == "random":
            import random

            return random.choice(self.adapters)

        return self.adapters[0]

    def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate using pool."""
        adapter = self.get_adapter()
        return adapter.generate(prompt, params, **kwargs)


@dataclass
class ConnectionInfo:
    """Connection information for an adapter."""

    adapter_id: str
    provider: Provider
    connected: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "adapter_id": self.adapter_id,
            "provider": self.provider.value,
            "connected": self.connected,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "checked_at": self.checked_at,
        }


class ConnectionMonitor:
    """Monitor adapter connections."""

    def __init__(self):
        """Initialize monitor."""
        self._connections: dict[str, ConnectionInfo] = {}

    def check_adapter(self, adapter: BaseAdapter) -> ConnectionInfo:
        """Check adapter connection."""
        adapter_id = f"{adapter.provider.value}:{adapter.model_id}"

        health = adapter.health_check()

        info = ConnectionInfo(
            adapter_id=adapter_id,
            provider=adapter.provider,
            connected=health.healthy,
            latency_ms=health.latency_ms,
            error=health.error,
        )

        self._connections[adapter_id] = info
        return info

    def get_status(self) -> dict[str, ConnectionInfo]:
        """Get all connection statuses."""
        return dict(self._connections)


# Convenience functions
def create_adapter(
    model_id: str,
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseAdapter:
    """Create an adapter for a model."""
    factory = AdapterFactory()
    return factory.create(model_id, api_key=api_key, **kwargs)


def create_mock_adapter(model_id: str = "mock-model") -> MockAdapter:
    """Create a mock adapter for testing."""
    config = AdapterConfig(model_id=model_id, provider=Provider.MOCK)
    return MockAdapter(config)


def create_fallback_chain(
    model_ids: list[str],
    api_keys: Optional[dict[str, str]] = None,
) -> FallbackChain:
    """Create a fallback chain from model IDs."""
    factory = AdapterFactory()

    if api_keys:
        for provider_str, key in api_keys.items():
            provider = Provider(provider_str)
            factory.set_api_key(provider, key)

    adapters = [factory.create(model_id) for model_id in model_ids]
    return FallbackChain(adapters)


def create_adapter_pool(
    model_ids: list[str],
    strategy: str = "round_robin",
) -> AdapterPool:
    """Create an adapter pool from model IDs."""
    factory = AdapterFactory()
    adapters = [factory.create(model_id) for model_id in model_ids]
    return AdapterPool(adapters, strategy=strategy)


def detect_provider(model_id: str) -> Optional[Provider]:
    """Detect provider from model ID."""
    return ProviderDetector.detect(model_id)


def list_providers() -> list[Provider]:
    """List all supported providers."""
    return list(Provider)


def list_models(provider: Optional[str] = None) -> list[AdapterModelInfo]:
    """List known models."""
    registry = ModelRegistry()
    p = Provider(provider) if provider else None
    return registry.list_models(p)


def get_model_info(model_id: str) -> Optional[AdapterModelInfo]:
    """Get model information."""
    registry = ModelRegistry()
    return registry.get(model_id)


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import ModelInfo from this module. The canonical name
# is AdapterModelInfo.
ModelInfo = AdapterModelInfo
