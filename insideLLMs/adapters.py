"""Model adapter factory module.

This module provides a unified interface for creating and managing
adapters across multiple LLM providers with consistent APIs.

Key Features:
- Unified interface for multiple providers (OpenAI, Anthropic, Google, etc.)
- Automatic provider detection from model names
- Lazy loading of provider-specific dependencies
- Connection pooling and rate limiting
- Fallback chains across providers
- Model aliasing and versioning
- Health checks and availability monitoring

Example:
    >>> from insideLLMs.adapters import AdapterFactory, create_adapter
    >>>
    >>> adapter = create_adapter("gpt-4")  # Auto-detects OpenAI
    >>> response = adapter.generate("Hello, world!")
    >>>
    >>> # Or use factory for more control
    >>> factory = AdapterFactory()
    >>> adapter = factory.create("claude-3-opus", provider="anthropic")
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Provider(Enum):
    """Supported LLM providers."""

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
    """Status of an adapter."""

    READY = "ready"
    INITIALIZING = "initializing"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"


class ModelCapability(Enum):
    """Model capabilities."""

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
    """Information about a model."""

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
        """Convert to dictionary."""
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
    """Configuration for an adapter."""

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
        """Convert to dictionary (excluding sensitive data)."""
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
    """Parameters for text generation."""

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
        """Convert to dictionary."""
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
    """Result from text generation."""

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
        """Convert to dictionary."""
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
    """Result of a health check."""

    healthy: bool
    latency_ms: float
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "healthy": self.healthy,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class BaseAdapter(ABC):
    """Abstract base class for LLM adapters."""

    def __init__(self, config: AdapterConfig):
        """Initialize adapter.

        Args:
            config: Adapter configuration
        """
        self.config = config
        self._status = AdapterStatus.INITIALIZING
        self._last_request_time: float = 0
        self._request_count: int = 0
        self._error_count: int = 0
        self._client: Optional[Any] = None

    @property
    def model_id(self) -> str:
        """Get model ID."""
        return self.config.model_id

    @property
    def provider(self) -> Provider:
        """Get provider."""
        return self.config.provider

    @property
    def status(self) -> AdapterStatus:
        """Get current status."""
        return self._status

    @abstractmethod
    def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            params: Generation parameters
            **kwargs: Additional arguments

        Returns:
            Generation result
        """
        pass

    @abstractmethod
    def generate_chat(
        self,
        messages: list[dict[str, str]],
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate from chat messages.

        Args:
            messages: List of messages with 'role' and 'content'
            params: Generation parameters
            **kwargs: Additional arguments

        Returns:
            Generation result
        """
        pass

    def health_check(self) -> HealthCheckResult:
        """Check adapter health."""
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
        """Get adapter statistics."""
        return {
            "model_id": self.model_id,
            "provider": self.provider.value,
            "status": self._status.value,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "last_request_time": self._last_request_time,
        }

    def _record_request(self) -> None:
        """Record a request."""
        self._request_count += 1
        self._last_request_time = time.time()

    def _record_error(self) -> None:
        """Record an error."""
        self._error_count += 1


class MockAdapter(BaseAdapter):
    """Mock adapter for testing."""

    def __init__(self, config: AdapterConfig):
        """Initialize mock adapter."""
        super().__init__(config)
        self._status = AdapterStatus.READY
        self._responses: dict[str, str] = {}
        self._default_response = "This is a mock response."

    def set_response(self, prompt_pattern: str, response: str) -> None:
        """Set response for a prompt pattern."""
        self._responses[prompt_pattern] = response

    def set_default_response(self, response: str) -> None:
        """Set default response."""
        self._default_response = response

    def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate mock response."""
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
        """Generate mock chat response."""
        # Convert messages to prompt
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return self.generate(prompt, params, **kwargs)


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI models."""

    def __init__(self, config: AdapterConfig):
        """Initialize OpenAI adapter."""
        super().__init__(config)
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
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
        """Generate using OpenAI completion."""
        messages = [{"role": "user", "content": prompt}]
        return self.generate_chat(messages, params, **kwargs)

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate using OpenAI chat."""
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
