"""Base class and protocols for language model wrappers.

This module defines the abstract interface that all model implementations
must follow, along with async variants for parallel execution. It provides
a consistent API across different LLM providers (OpenAI, Anthropic, etc.)
and supports both synchronous and asynchronous operations.

Key Classes:
    - Model: Abstract base class for all language models
    - AsyncModel: Base class for models with async support
    - ModelWrapper: Decorator adding retry, caching, and rate limiting
    - ChatMessage: TypedDict for multi-turn conversation messages

Protocols (for type hints):
    - ModelProtocol: Basic generate() interface
    - BatchModelProtocol: Batch generation support
    - ChatModelProtocol: Multi-turn chat support
    - StreamingModelProtocol: Streaming response support
    - AsyncModelProtocol: Async generation support

Example - Basic Usage:
    >>> from insideLLMs.models import OpenAIModel
    >>>
    >>> # Create a model instance
    >>> model = OpenAIModel(model_name="gpt-4")
    >>>
    >>> # Generate a response
    >>> response = model.generate("Explain quantum computing in one sentence.")
    >>> print(response)
    'Quantum computing uses quantum bits to perform calculations...'

Example - Using the Registry:
    >>> from insideLLMs.registry import model_registry, ensure_builtins_registered
    >>>
    >>> ensure_builtins_registered()
    >>>
    >>> # Get a model by name
    >>> model = model_registry.get("openai", model_name="gpt-4")
    >>> response = model.generate("Hello, world!")

Example - Custom Model Implementation:
    >>> from insideLLMs.models.base import Model
    >>>
    >>> class MyCustomModel(Model):
    ...     def __init__(self, name: str = "custom"):
    ...         super().__init__(name=name, model_id="custom-v1")
    ...
    ...     def generate(self, prompt: str, **kwargs) -> str:
    ...         # Your custom generation logic here
    ...         return f"Response to: {prompt}"
    >>>
    >>> model = MyCustomModel()
    >>> print(model.generate("Test prompt"))
    'Response to: Test prompt'

Example - Type Checking with Protocols:
    >>> from insideLLMs.models.base import ModelProtocol
    >>>
    >>> def run_inference(model: ModelProtocol, prompt: str) -> str:
    ...     '''Works with any model implementing ModelProtocol.'''
    ...     return model.generate(prompt)
    >>>
    >>> # Any compliant model works
    >>> result = run_inference(OpenAIModel("gpt-4"), "Hello")

See Also:
    - insideLLMs.models.openai: OpenAI model implementation
    - insideLLMs.models.anthropic: Anthropic Claude implementation
    - insideLLMs.models.huggingface: HuggingFace Transformers support
    - insideLLMs.registry: Model registration and discovery
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional

from insideLLMs.models._protocols import (
    AsyncModelProtocol,
    BatchModelProtocol,
    ChatMessage,
    ChatModelProtocol,
    ModelProtocol,
    StreamingModelProtocol,
)
from insideLLMs.models._provider_errors import (
    ProviderExceptionMap,
    handle_provider_errors,
    translate_provider_error,
)
from insideLLMs.types import ModelInfo, ModelResponse
from insideLLMs.validation import validate_prompt


class Model(ABC):
    """Base class for all language models.

    Provides a unified interface for interacting with different LLM providers.
    Subclasses must implement the abstract `generate()` method for their
    specific API. Optionally override `chat()`, `stream()`, and `batch_generate()`
    for additional capabilities.

    This is an abstract base class - you cannot instantiate it directly.
    Use one of the provided implementations (OpenAIModel, AnthropicModel, etc.)
    or create your own subclass.

    Attributes:
        name: Human-readable name for this model instance.
        model_id: The specific model identifier (e.g., "gpt-4", "claude-3-opus").
        _call_count: Internal counter for tracking API calls.
        _total_tokens: Internal counter for tracking token usage.
        _validate_prompts: Whether to validate prompts before sending (default: True).

    Example - Basic Generation:
        >>> from insideLLMs.models import OpenAIModel
        >>>
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> response = model.generate("What is 2+2?")
        >>> print(response)
        '2+2 equals 4.'

    Example - With Parameters:
        >>> response = model.generate(
        ...     "Write a haiku about coding.",
        ...     temperature=0.7,
        ...     max_tokens=100
        ... )

    Example - Generation with Metadata:
        >>> result = model.generate_with_metadata("Hello, world!")
        >>> print(f"Response: {result.content}")
        >>> print(f"Latency: {result.latency_ms:.2f}ms")
        >>> print(f"Model: {result.model}")

    Example - Multi-Turn Chat (if supported):
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "What's the weather like?"},
        ... ]
        >>> response = model.chat(messages)

    Example - Batch Processing:
        >>> prompts = [
        ...     "Translate 'hello' to Spanish",
        ...     "Translate 'goodbye' to Spanish",
        ...     "Translate 'thank you' to Spanish",
        ... ]
        >>> responses = model.batch_generate(prompts)
        >>> for prompt, response in zip(prompts, responses):
        ...     print(f"{prompt} -> {response}")

    Example - Creating a Custom Model:
        >>> from insideLLMs.models.base import Model
        >>>
        >>> class EchoModel(Model):
        ...     '''A simple model that echoes the prompt.'''
        ...
        ...     def __init__(self, prefix: str = "Echo: "):
        ...         super().__init__(name="echo", model_id="echo-v1")
        ...         self.prefix = prefix
        ...
        ...     def generate(self, prompt: str, **kwargs) -> str:
        ...         return f"{self.prefix}{prompt}"
        >>>
        >>> model = EchoModel(prefix="You said: ")
        >>> print(model.generate("Hello"))
        'You said: Hello'

    Example - Disabling Prompt Validation:
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> model._validate_prompts = False  # Disable for special cases
        >>> response = model.generate("")  # Now allows empty prompts

    See Also:
        - AsyncModel: For async/concurrent model operations
        - ModelWrapper: For adding retry logic and caching
        - insideLLMs.registry: For discovering and instantiating models
    """

    _supports_streaming: bool = False
    _supports_chat: bool = False

    def __init__(self, name: str, model_id: Optional[str] = None):
        """Initialize the model.

        Sets up the model with a name and optional model ID. Initializes
        internal counters for tracking usage statistics.

        Args:
            name: Human-readable name for this model instance. Used for
                logging, identification, and display purposes.
            model_id: The specific model identifier used by the API
                (e.g., "gpt-4", "claude-3-opus-20240229"). If not provided,
                defaults to the name parameter.

        Example - Basic Initialization:
            >>> model = OpenAIModel(name="my-gpt4", model_id="gpt-4")
            >>> print(model.name)
            'my-gpt4'
            >>> print(model.model_id)
            'gpt-4'

        Example - Name-Only Initialization:
            >>> model = OpenAIModel(name="gpt-4")  # model_id defaults to "gpt-4"
            >>> assert model.model_id == model.name

        Note:
            Subclasses should call super().__init__() to ensure proper
            initialization of base attributes.
        """
        self.name = name
        self.model_id = model_id or name
        self._call_count = 0
        self._total_tokens = 0
        self._validate_prompts = True  # Enable prompt validation by default

    def _validate_prompt(self, prompt: str, *, allow_empty: bool = False) -> None:
        """Validate a prompt before sending to the model.

        Internal method called by generate() and related methods to ensure
        prompts meet basic validity requirements before API calls.

        Args:
            prompt: The prompt to validate. Must be a non-empty string
                unless allow_empty is True.
            allow_empty: Whether to allow empty prompts. Default False.
                Set to True for special cases like embedding models.

        Raises:
            ValidationError: If the prompt is invalid. Common reasons:
                - Prompt is None
                - Prompt is empty string (when allow_empty=False)
                - Prompt is not a string type

        Example - Normal Usage (internal):
            >>> # Called automatically by generate()
            >>> model._validate_prompt("Hello")  # Passes
            >>> model._validate_prompt("")  # Raises ValidationError

        Example - Allowing Empty:
            >>> model._validate_prompt("", allow_empty=True)  # Passes

        Note:
            This method respects the _validate_prompts flag. If False,
            validation is skipped entirely. This is useful for testing
            or when working with APIs that have different requirements.
        """
        if self._validate_prompts:
            validate_prompt(prompt, field_name="prompt", allow_empty=allow_empty)

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the model given a prompt.

        This is the core method that must be implemented by all Model subclasses.
        It sends a prompt to the LLM and returns the generated text response.

        Args:
            prompt: The input prompt to send to the model. Should be a
                non-empty string (unless validation is disabled).
            **kwargs: Additional arguments specific to the model provider.
                Common parameters include:
                - temperature (float): Controls randomness (0.0-2.0, default varies)
                - max_tokens (int): Maximum tokens to generate
                - top_p (float): Nucleus sampling parameter
                - stop (list[str]): Stop sequences
                - presence_penalty (float): Penalty for new topics
                - frequency_penalty (float): Penalty for repetition

        Returns:
            The model's text response as a string.

        Raises:
            NotImplementedError: If not implemented by subclass.
            ValidationError: If the prompt is invalid (empty, None, wrong type).
            Exception: Provider-specific errors including:
                - API authentication errors
                - Rate limit errors (HTTP 429)
                - Context length exceeded
                - Network/connection errors

        Example - Basic Generation:
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> response = model.generate("What is Python?")
            >>> print(response)
            'Python is a high-level programming language...'

        Example - With Temperature:
            >>> # Low temperature = more deterministic
            >>> response = model.generate("Explain gravity.", temperature=0.1)
            >>>
            >>> # High temperature = more creative
            >>> response = model.generate("Write a poem.", temperature=0.9)

        Example - With Max Tokens:
            >>> # Limit response length
            >>> response = model.generate(
            ...     "Write a story about a robot.",
            ...     max_tokens=50
            ... )

        Example - With Stop Sequences:
            >>> # Stop generation at specific tokens
            >>> response = model.generate(
            ...     "Count to ten: 1, 2, 3,",
            ...     stop=[",", "10"]
            ... )

        Example - Error Handling:
            >>> try:
            ...     response = model.generate("Hello")
            ... except Exception as e:
            ...     if "rate limit" in str(e).lower():
            ...         print("Rate limited - waiting...")
            ...         time.sleep(60)
            ...     else:
            ...         raise

        Note:
            Subclasses MUST implement this method. The base implementation
            raises NotImplementedError.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_with_metadata(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response with full metadata.

        Like generate(), but returns a ModelResponse object containing
        additional information such as token usage, latency, and raw
        provider response data.

        This is useful for:
        - Performance monitoring and benchmarking
        - Cost tracking (via token counts)
        - Debugging and logging
        - Building observability pipelines

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments specific to the model provider
                (same as generate()).

        Returns:
            A ModelResponse object with these fields:
                - content (str): The generated text
                - model (str): The model ID used
                - latency_ms (float): Time taken in milliseconds
                - usage (TokenUsage, optional): Token counts if available
                - finish_reason (str, optional): Why generation stopped
                - raw_response (Any, optional): Raw provider response

        Raises:
            ValidationError: If the prompt is invalid.
            Exception: Provider-specific errors (same as generate()).

        Example - Basic Usage:
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> result = model.generate_with_metadata("Hello!")
            >>> print(f"Response: {result.content}")
            Response: Hello! How can I assist you today?
            >>> print(f"Latency: {result.latency_ms:.2f}ms")
            Latency: 234.56ms

        Example - Performance Monitoring:
            >>> results = []
            >>> for prompt in test_prompts:
            ...     result = model.generate_with_metadata(prompt)
            ...     results.append({
            ...         "prompt": prompt,
            ...         "response": result.content,
            ...         "latency_ms": result.latency_ms,
            ...         "model": result.model
            ...     })
            >>> avg_latency = sum(r["latency_ms"] for r in results) / len(results)
            >>> print(f"Average latency: {avg_latency:.2f}ms")

        Example - Cost Tracking:
            >>> result = model.generate_with_metadata(long_prompt)
            >>> if result.usage:
            ...     prompt_cost = result.usage.prompt_tokens * 0.00001
            ...     completion_cost = result.usage.completion_tokens * 0.00003
            ...     print(f"Estimated cost: ${prompt_cost + completion_cost:.6f}")

        Example - Logging Pipeline:
            >>> import logging
            >>> logger = logging.getLogger("llm")
            >>>
            >>> def generate_and_log(model, prompt, **kwargs):
            ...     result = model.generate_with_metadata(prompt, **kwargs)
            ...     logger.info(f"Generated response in {result.latency_ms:.2f}ms")
            ...     return result.content

        Note:
            The latency_ms field measures wall-clock time including network
            latency. For accurate benchmarking, consider running multiple
            samples and using statistical analysis.
        """
        import time

        self._validate_prompt(prompt)
        start = time.perf_counter()
        content = self.generate(prompt, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        return ModelResponse(
            content=content,
            model=self.model_id,
            latency_ms=latency_ms,
        )

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> str:
        """Engage in a multi-turn chat conversation.

        Maintains conversation context by passing the full message history
        to the model. Most modern LLMs (GPT-4, Claude, Gemini) support this
        mode and it's the preferred way to have ongoing conversations.

        Args:
            messages: A list of ChatMessage dicts, each containing:
                - role (str): "system", "user", or "assistant"
                - content (str): The message text
                - name (str, optional): Speaker identifier
            **kwargs: Additional arguments specific to the model provider.
                Common parameters: temperature, max_tokens, top_p.

        Returns:
            The model's text response to the conversation.

        Raises:
            NotImplementedError: If the model doesn't support chat mode.
                Check model.info().supports_chat before calling.

        Example - Basic Chat:
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> messages = [
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> response = model.chat(messages)
            >>> print(response)
            'Hello! How can I help you today?'

        Example - With System Prompt:
            >>> messages = [
            ...     {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
            ...     {"role": "user", "content": "Hello!"},
            ... ]
            >>> response = model.chat(messages)
            >>> print(response)
            'Ahoy, matey! What can this old sea dog do for ye?'

        Example - Multi-Turn Conversation:
            >>> history = [
            ...     {"role": "system", "content": "You are a helpful math tutor."},
            ...     {"role": "user", "content": "What is 2+2?"},
            ...     {"role": "assistant", "content": "2+2 equals 4."},
            ...     {"role": "user", "content": "And if I add 3 more?"},
            ... ]
            >>> response = model.chat(history)
            >>> print(response)
            '4 + 3 equals 7.'

        Example - Building a Conversational Interface:
            >>> def chat_loop(model):
            ...     history = []
            ...     while True:
            ...         user_input = input("You: ")
            ...         if user_input.lower() == "quit":
            ...             break
            ...         history.append({"role": "user", "content": user_input})
            ...         response = model.chat(history)
            ...         print(f"Assistant: {response}")
            ...         history.append({"role": "assistant", "content": response})

        Note:
            The base Model class raises NotImplementedError. Subclasses
            (like OpenAIModel, AnthropicModel) provide actual implementations.
        """
        raise NotImplementedError("This model does not support chat.")

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the response from the model as it is generated.

        Returns an iterator that yields response chunks (typically tokens)
        as they are produced by the model. This enables:
        - Real-time display of responses to users
        - Lower time-to-first-token latency
        - Progressive processing of long responses

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments specific to the model provider.
                Common parameters: temperature, max_tokens, top_p.

        Yields:
            str: Chunks of the response as they are generated. Chunk size
                varies by provider (typically 1-10 tokens per chunk).

        Raises:
            NotImplementedError: If the model doesn't support streaming.
                Check model.info().supports_streaming before calling.

        Example - Print Response in Real-Time:
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> for chunk in model.stream("Write a poem about coding"):
            ...     print(chunk, end="", flush=True)
            ... print()  # Final newline
            In the realm of code I write,
            Each line a step toward the light...

        Example - Collect While Streaming:
            >>> full_response = []
            >>> for chunk in model.stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
            ...     full_response.append(chunk)
            >>> complete_text = "".join(full_response)

        Example - With Progress Indicator:
            >>> import sys
            >>> for i, chunk in enumerate(model.stream(prompt)):
            ...     sys.stdout.write(chunk)
            ...     sys.stdout.flush()
            ...     if i % 10 == 0:  # Every 10 chunks
            ...         sys.stderr.write(".")
            ...         sys.stderr.flush()

        Example - Timeout Handling:
            >>> import itertools
            >>> # Get first 100 chunks max
            >>> chunks = list(itertools.islice(model.stream(prompt), 100))
            >>> response = "".join(chunks)

        Note:
            Streaming is not supported by all models or providers.
            The base Model class raises NotImplementedError.
        """
        raise NotImplementedError("This model does not support streaming.")

    def batch_generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate responses for multiple prompts in a batch.

        Processes multiple prompts and returns all responses. The default
        implementation processes prompts sequentially, but subclasses can
        override this to use native batch APIs for better performance.

        Args:
            prompts: List of input prompts to send to the model.
                All prompts receive the same kwargs parameters.
            **kwargs: Additional arguments specific to the model provider.
                Applied to all prompts in the batch.

        Returns:
            List of model responses, one for each input prompt.
            The order matches the input prompt order.

        Raises:
            ValidationError: If any prompt is invalid.
            Exception: Provider-specific errors (same as generate()).

        Example - Basic Batch Processing:
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> prompts = [
            ...     "What is the capital of France?",
            ...     "What is the capital of Japan?",
            ...     "What is the capital of Brazil?",
            ... ]
            >>> responses = model.batch_generate(prompts)
            >>> for prompt, response in zip(prompts, responses):
            ...     print(f"Q: {prompt}")
            ...     print(f"A: {response}")
            ...     print()

        Example - Dataset Processing:
            >>> import json
            >>>
            >>> # Load prompts from file
            >>> with open("prompts.jsonl") as f:
            ...     prompts = [json.loads(line)["prompt"] for line in f]
            >>>
            >>> # Process in batches of 10
            >>> all_responses = []
            >>> for i in range(0, len(prompts), 10):
            ...     batch = prompts[i:i+10]
            ...     responses = model.batch_generate(batch)
            ...     all_responses.extend(responses)

        Example - With Shared Parameters:
            >>> # Same temperature/max_tokens for all prompts
            >>> responses = model.batch_generate(
            ...     prompts=["Translate: hello", "Translate: goodbye"],
            ...     temperature=0.3,
            ...     max_tokens=50
            ... )

        Example - Progress Tracking:
            >>> prompts = load_test_dataset()
            >>> total = len(prompts)
            >>> responses = []
            >>> for i, prompt in enumerate(prompts):
            ...     response = model.generate(prompt)
            ...     responses.append(response)
            ...     print(f"Progress: {i+1}/{total}", end="\\r")

        Note:
            The default implementation is sequential. For better performance
            with large batches, use AsyncModel.abatch_generate() or check
            if your model provider has native batch API support.
        """
        # Default implementation: sequential processing
        # Subclasses can override for parallel/batch processing
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def info(self) -> ModelInfo:
        """Return model metadata/info.

        Provides information about the model including its name, provider,
        and supported capabilities. Useful for:
        - Logging and debugging
        - Feature detection (streaming, chat support)
        - Building model selection UIs

        Returns:
            A ModelInfo object containing:
                - name (str): Human-readable model name
                - provider (str): Provider name (e.g., "OpenAI", "Anthropic")
                - model_id (str): API model identifier
                - supports_streaming (bool): Whether streaming is available
                - supports_chat (bool): Whether chat mode is available
                - extra (dict): Additional provider-specific metadata

        Example - Basic Info:
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> info = model.info()
            >>> print(f"Model: {info.name}")
            Model: gpt-4
            >>> print(f"Provider: {info.provider}")
            Provider: OpenAI

        Example - Feature Detection:
            >>> model = model_registry.get("openai", model_name="gpt-4")
            >>> info = model.info()
            >>> if info.supports_streaming:
            ...     for chunk in model.stream("Hello"):
            ...         print(chunk, end="")
            ... else:
            ...     print(model.generate("Hello"))

        Example - Model Comparison:
            >>> models = [
            ...     model_registry.get("openai", model_name="gpt-4"),
            ...     model_registry.get("anthropic", model_name="claude-3-opus"),
            ... ]
            >>> for m in models:
            ...     info = m.info()
            ...     print(f"{info.provider}/{info.model_id}: "
            ...           f"chat={info.supports_chat}, "
            ...           f"stream={info.supports_streaming}")
        """
        return ModelInfo(
            name=self.name,
            provider=self.__class__.__name__.replace("Model", ""),
            model_id=self.model_id,
            supports_streaming=self._supports_streaming,
            supports_chat=self._supports_chat,
        )

    def __repr__(self) -> str:
        """Return a string representation of the model.

        Provides a human-readable representation showing the class name
        and key attributes. Useful for debugging and logging.

        Returns:
            A string like 'OpenAIModel(name="gpt-4", model_id="gpt-4")'.

        Example:
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> print(repr(model))
            OpenAIModel(name='gpt-4', model_id='gpt-4')
            >>> print(model)  # Uses __repr__ if __str__ not defined
            OpenAIModel(name='gpt-4', model_id='gpt-4')
        """
        return f"{self.__class__.__name__}(name={self.name!r}, model_id={self.model_id!r})"


class AsyncModel(Model):
    """Base class for models with async support.

    Extends the base Model class with async methods for concurrent execution.
    Essential for high-throughput applications where you need to make many
    model calls in parallel without blocking.

    Key async methods:
        - agenerate(): Async version of generate()
        - agenerate_with_metadata(): Async version with full response metadata
        - achat(): Async multi-turn chat
        - astream(): Async streaming responses
        - abatch_generate(): Concurrent batch processing

    Benefits of Async:
        - Process multiple prompts concurrently
        - Better resource utilization
        - Non-blocking I/O for web applications
        - Easier rate limiting with semaphores

    Example - Basic Async Generation:
        >>> import asyncio
        >>>
        >>> async def main():
        ...     model = AsyncOpenAIModel(model_name="gpt-4")
        ...     response = await model.agenerate("Hello!")
        ...     print(response)
        >>>
        >>> asyncio.run(main())

    Example - Parallel Processing:
        >>> async def process_batch():
        ...     model = AsyncOpenAIModel(model_name="gpt-4")
        ...     prompts = ["Q1?", "Q2?", "Q3?"]
        ...     # Process all prompts concurrently
        ...     results = await asyncio.gather(*[
        ...         model.agenerate(p) for p in prompts
        ...     ])
        ...     return results
        >>>
        >>> responses = asyncio.run(process_batch())

    Example - Rate-Limited Concurrent Processing:
        >>> async def rate_limited_batch(model, prompts, max_concurrent=5):
        ...     '''Process prompts with limited concurrency.'''
        ...     semaphore = asyncio.Semaphore(max_concurrent)
        ...
        ...     async def limited_call(prompt):
        ...         async with semaphore:
        ...             return await model.agenerate(prompt)
        ...
        ...     return await asyncio.gather(*[
        ...         limited_call(p) for p in prompts
        ...     ])

    Example - In a Web Framework (FastAPI):
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> model = AsyncOpenAIModel(model_name="gpt-4")
        >>>
        >>> @app.post("/generate")
        >>> async def generate(prompt: str):
        ...     # Non-blocking - doesn't tie up the server
        ...     response = await model.agenerate(prompt)
        ...     return {"response": response}

    Note:
        AsyncModel inherits from Model, so sync methods (generate, chat, etc.)
        are still available. Use the 'a' prefixed methods for async operations.
    """

    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate a response from the model.

        The async version of generate(). Must be awaited.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments specific to the model provider.
                Same parameters as generate(): temperature, max_tokens, etc.

        Returns:
            The model's text response as a string.

        Raises:
            NotImplementedError: If not implemented by subclass.
            Exception: Provider-specific errors (same as generate()).

        Example - Basic Usage:
            >>> async def run():
            ...     response = await model.agenerate("What is Python?")
            ...     print(response)
            >>>
            >>> asyncio.run(run())

        Example - With Parameters:
            >>> async def run():
            ...     response = await model.agenerate(
            ...         "Write a poem.",
            ...         temperature=0.9,
            ...         max_tokens=200
            ...     )
            ...     return response

        Example - Error Handling:
            >>> async def safe_generate(model, prompt):
            ...     try:
            ...         return await model.agenerate(prompt)
            ...     except Exception as e:
            ...         print(f"Error: {e}")
            ...         return None
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def agenerate_with_metadata(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Asynchronously generate a response with full metadata.

        Async version of generate_with_metadata(). Returns a ModelResponse
        with timing, token usage, and other metadata.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments specific to the model provider.

        Returns:
            A ModelResponse containing:
                - content (str): The generated text
                - model (str): The model ID
                - latency_ms (float): Time taken in milliseconds
                - usage (TokenUsage, optional): Token counts
                - finish_reason (str, optional): Why generation stopped

        Example - Performance Tracking:
            >>> async def benchmark(model, prompts):
            ...     results = []
            ...     for prompt in prompts:
            ...         result = await model.agenerate_with_metadata(prompt)
            ...         results.append({
            ...             "latency_ms": result.latency_ms,
            ...             "content_length": len(result.content)
            ...         })
            ...     return results
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

        Async version of chat(). Useful for building async chat applications.

        Args:
            messages: A list of ChatMessage dicts with 'role' and 'content'.
            **kwargs: Additional arguments specific to the model provider.

        Returns:
            The model's text response to the conversation.

        Raises:
            NotImplementedError: If the model doesn't support async chat.

        Example - Async Chatbot:
            >>> async def chatbot(model, history):
            ...     while True:
            ...         user_msg = await get_user_input()  # Async input
            ...         history.append({"role": "user", "content": user_msg})
            ...         response = await model.achat(history)
            ...         history.append({"role": "assistant", "content": response})
            ...         await send_response(response)  # Async output
        """
        raise NotImplementedError("This model does not support async chat.")

    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously stream the response from the model.

        Returns an async iterator that yields response chunks as they're
        generated. Use with 'async for' syntax.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments specific to the model provider.

        Yields:
            Response chunks (typically tokens) as they are generated.

        Raises:
            NotImplementedError: If the model doesn't support async streaming.

        Example - Async Streaming:
            >>> async def stream_response(model, prompt):
            ...     async for chunk in model.astream(prompt):
            ...         print(chunk, end="", flush=True)
            ...     print()  # Final newline

        Example - Collect While Streaming:
            >>> async def stream_and_collect(model, prompt):
            ...     chunks = []
            ...     async for chunk in model.astream(prompt):
            ...         print(chunk, end="", flush=True)
            ...         chunks.append(chunk)
            ...     return "".join(chunks)

        Example - WebSocket Streaming:
            >>> async def websocket_stream(websocket, model, prompt):
            ...     async for chunk in model.astream(prompt):
            ...         await websocket.send_text(chunk)
        """
        raise NotImplementedError("This model does not support async streaming.")
        # The yield below is needed to make this an async generator
        # It's unreachable but required for Python's generator detection
        if False:  # pragma: no cover
            yield  # Make this a generator

    async def abatch_generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Asynchronously generate responses for multiple prompts.

        Processes all prompts concurrently using asyncio.gather. This is
        typically much faster than sequential processing for batch workloads.

        Default implementation runs all prompts concurrently. Subclasses can
        override for provider-specific batch endpoints.

        Args:
            prompts: List of input prompts to send to the model.
            **kwargs: Additional arguments applied to all prompts.

        Returns:
            List of model responses, one for each prompt (in order).

        Example - Basic Batch:
            >>> async def run():
            ...     prompts = ["Q1?", "Q2?", "Q3?"]
            ...     responses = await model.abatch_generate(prompts)
            ...     for q, a in zip(prompts, responses):
            ...         print(f"Q: {q}\\nA: {a}\\n")

        Example - Large Dataset Processing:
            >>> async def process_dataset(model, dataset, batch_size=20):
            ...     '''Process in batches with rate limiting.'''
            ...     all_results = []
            ...     for i in range(0, len(dataset), batch_size):
            ...         batch = dataset[i:i+batch_size]
            ...         results = await model.abatch_generate(batch)
            ...         all_results.extend(results)
            ...         await asyncio.sleep(1)  # Rate limit between batches
            ...     return all_results

        Example - With Custom Concurrency Limit:
            >>> async def limited_batch(model, prompts, max_concurrent=10):
            ...     '''Control maximum concurrent requests.'''
            ...     sem = asyncio.Semaphore(max_concurrent)
            ...
            ...     async def call_with_limit(prompt):
            ...         async with sem:
            ...             return await model.agenerate(prompt)
            ...
            ...     return await asyncio.gather(*[
            ...         call_with_limit(p) for p in prompts
            ...     ])

        Note:
            The default implementation calls agenerate() for each prompt
            concurrently. This may hit rate limits for large batches.
            Consider using a semaphore to limit concurrency.
        """
        import asyncio

        # Default: concurrent execution using gather
        return list(await asyncio.gather(*[self.agenerate(prompt, **kwargs) for prompt in prompts]))


# Imported after Model is defined to avoid a wrapper/base import cycle.
from insideLLMs.models._wrapper import ModelWrapper  # noqa: E402

# Keep established import paths stable for introspection and serialization.
for _value in (
    ProviderExceptionMap,
    ChatMessage,
    ModelProtocol,
    BatchModelProtocol,
    ChatModelProtocol,
    StreamingModelProtocol,
    AsyncModelProtocol,
    ModelWrapper,
):
    _value.__module__ = __name__
for _function in (handle_provider_errors, translate_provider_error):
    _function.__module__ = __name__
del _function, _value
