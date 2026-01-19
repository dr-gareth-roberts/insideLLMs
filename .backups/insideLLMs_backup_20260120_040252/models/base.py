"""Base class and protocols for language model wrappers.

This module defines the abstract interface that all model implementations
must follow, along with async variants for parallel execution.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import (
    Any,
    Optional,
    Protocol,
    TypedDict,
    runtime_checkable,
)

from insideLLMs.types import ModelInfo, ModelResponse


class ChatMessage(TypedDict, total=False):
    """A single message in a chat conversation."""

    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str]


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol defining the interface for language models.

    Use this for type hints when you want to accept any model-like object.
    """

    name: str

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the model given a prompt."""
        ...

    def info(self) -> dict[str, Any]:
        """Return model metadata/info as a dict."""
        ...


@runtime_checkable
class ChatModelProtocol(ModelProtocol, Protocol):
    """Protocol for models that support multi-turn chat."""

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Engage in a multi-turn chat."""
        ...


@runtime_checkable
class StreamingModelProtocol(ModelProtocol, Protocol):
    """Protocol for models that support streaming responses."""

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the response from the model."""
        ...


@runtime_checkable
class AsyncModelProtocol(Protocol):
    """Protocol for models that support async operations."""

    name: str

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate a response."""
        ...


class Model(ABC):
    """Base class for all language models.

    Provides a unified interface for interacting with different LLM providers.
    Subclasses must implement the abstract methods for their specific API.

    Attributes:
        name: Human-readable name for this model instance.
        model_id: The specific model identifier (e.g., "gpt-4", "claude-3-opus").

    Example:
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> response = model.generate("What is 2+2?")
        >>> print(response)
        "2+2 equals 4."
    """

    def __init__(self, name: str, model_id: Optional[str] = None):
        """Initialize the model.

        Args:
            name: Human-readable name for this model instance.
            model_id: The specific model identifier. If not provided,
                     defaults to the name.
        """
        self.name = name
        self.model_id = model_id or name
        self._call_count = 0
        self._total_tokens = 0

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the model given a prompt.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments specific to the model provider
                     (e.g., temperature, max_tokens).

        Returns:
            The model's text response.

        Raises:
            NotImplementedError: If not implemented by subclass.
            Exception: Provider-specific errors (API errors, rate limits, etc.)
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_with_metadata(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response with full metadata.

        Like generate(), but returns a ModelResponse with additional information
        like token usage and latency.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments specific to the model provider.

        Returns:
            A ModelResponse containing the response and metadata.
        """
        import time

        start = time.perf_counter()
        content = self.generate(prompt, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        return ModelResponse(
            content=content,
            model=self.model_id,
            latency_ms=latency_ms,
        )

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Engage in a multi-turn chat conversation.

        Args:
            messages: A list of message dicts with 'role' and 'content' keys.
                     Roles are typically "system", "user", or "assistant".
            **kwargs: Additional arguments specific to the model provider.

        Returns:
            The model's text response.

        Raises:
            NotImplementedError: If the model doesn't support chat.
        """
        raise NotImplementedError("This model does not support chat.")

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the response from the model as it is generated.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments specific to the model provider.

        Yields:
            Chunks of the response as they are generated.

        Raises:
            NotImplementedError: If the model doesn't support streaming.
        """
        raise NotImplementedError("This model does not support streaming.")

    def info(self) -> ModelInfo:
        """Return model metadata/info.

        Returns:
            A ModelInfo object containing model details.
        """
        return ModelInfo(
            name=self.name,
            provider=self.__class__.__name__.replace("Model", ""),
            model_id=self.model_id,
            supports_streaming=hasattr(self, "_supports_streaming") and self._supports_streaming,
            supports_chat=hasattr(self, "_supports_chat") and self._supports_chat,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model_id={self.model_id!r})"


class AsyncModel(Model):
    """Base class for models with async support.

    Extends the base Model class with async methods for concurrent execution.
    Useful for running multiple model calls in parallel.

    Example:
        >>> async def run_batch():
        ...     model = AsyncOpenAIModel(model_name="gpt-4")
        ...     results = await asyncio.gather(*[
        ...         model.agenerate(prompt) for prompt in prompts
        ...     ])
        ...     return results
    """

    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate a response from the model.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments specific to the model provider.

        Returns:
            The model's text response.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def agenerate_with_metadata(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Asynchronously generate a response with full metadata.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments specific to the model provider.

        Returns:
            A ModelResponse containing the response and metadata.
        """
        import time

        start = time.perf_counter()
        content = await self.agenerate(prompt, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        return ModelResponse(
            content=content,
            model=self.model_id,
            latency_ms=latency_ms,
        )

    async def achat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Asynchronously engage in a multi-turn chat.

        Args:
            messages: A list of message dicts with 'role' and 'content' keys.
            **kwargs: Additional arguments specific to the model provider.

        Returns:
            The model's text response.
        """
        raise NotImplementedError("This model does not support async chat.")

    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously stream the response from the model.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments specific to the model provider.

        Yields:
            Chunks of the response as they are generated.
        """
        raise NotImplementedError("This model does not support async streaming.")
        yield  # Make this a generator


class ModelWrapper:
    """Wrapper that adds common functionality to any model.

    Provides features like retry logic, rate limiting, and response caching
    that work with any Model implementation.

    Example:
        >>> base_model = OpenAIModel(model_name="gpt-4")
        >>> model = ModelWrapper(base_model, max_retries=3)
        >>> response = model.generate("What is 2+2?")
    """

    def __init__(
        self,
        model: Model,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_responses: bool = False,
    ):
        """Initialize the wrapper.

        Args:
            model: The underlying model to wrap.
            max_retries: Maximum number of retry attempts on failure.
            retry_delay: Delay in seconds between retries.
            cache_responses: Whether to cache responses for identical prompts.
        """
        self._model = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._cache_responses = cache_responses
        self._cache: dict[str, str] = {}

    @property
    def name(self) -> str:
        return self._model.name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate with retry logic and optional caching."""
        import time

        cache_key = f"{prompt}:{sorted(kwargs.items())}"

        if self._cache_responses and cache_key in self._cache:
            return self._cache[cache_key]

        last_error = None
        for attempt in range(self._max_retries):
            try:
                result = self._model.generate(prompt, **kwargs)
                if self._cache_responses:
                    self._cache[cache_key] = result
                return result
            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))

        raise last_error or RuntimeError("Max retries exceeded")

    def info(self) -> ModelInfo:
        return self._model.info()

    def __repr__(self) -> str:
        return f"ModelWrapper({self._model!r}, max_retries={self._max_retries})"
