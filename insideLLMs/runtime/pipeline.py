"""Model pipeline with composable middleware for advanced model capabilities.

This module provides a flexible middleware-based architecture for enhancing
language model interactions. It implements the chain of responsibility pattern,
allowing capabilities like caching, rate limiting, retry logic, cost tracking,
and tracing to be composed in a modular and reusable way.

The pipeline architecture separates cross-cutting concerns from the core model
logic, making it easy to add, remove, or reorder capabilities without modifying
the underlying model code. Both synchronous and asynchronous execution patterns
are fully supported.

Key Components
--------------
Middleware : ABC
    Base class for all middleware. Subclass this to create custom middleware
    that can intercept, modify, or short-circuit requests and responses.

PassthroughMiddleware : Middleware
    A no-op middleware that passes requests unchanged. Useful as a base class
    for middleware that only needs to observe or log requests.

TraceMiddleware : PassthroughMiddleware
    Records execution traces for debugging, validation, and reproducibility.
    Captures generate, chat, and stream operations with timing information.

CacheMiddleware : Middleware
    Caches responses based on prompt and parameters to avoid redundant API
    calls. Implements LRU eviction and optional TTL expiration.

RateLimitMiddleware : Middleware
    Enforces request rate limits using a token bucket algorithm. Supports
    configurable rates and burst sizes for both sync and async operations.

RetryMiddleware : Middleware
    Automatically retries failed requests with exponential backoff and jitter.
    Handles transient errors like rate limits and timeouts gracefully.

CostTrackingMiddleware : Middleware
    Tracks API costs and token usage for monitoring and budgeting. Provides
    estimated costs based on known model pricing.

ModelPipeline : Model
    The main pipeline class that composes a base model with middleware. Acts
    as a drop-in replacement for any Model while adding middleware capabilities.

AsyncModelPipeline : ModelPipeline
    An async-optimized pipeline with additional features for concurrent batch
    processing, progress callbacks, and streaming results.

Architecture
------------
Middleware is executed in a chain pattern:

    Request → [M1 → M2 → M3 → ... → Model]
    Response ← [M1 ← M2 ← M3 ← ... ← Model]

Each middleware can:
- Modify the request before passing it on
- Short-circuit the chain (e.g., cache hit, rate limit exceeded)
- Modify the response before returning
- Add side effects (logging, tracing, cost tracking)

Thread Safety
-------------
All async middleware implementations use asyncio.Lock for thread-safe access
to shared state. Synchronous methods are not thread-safe by default and should
be protected externally if used from multiple threads.

Examples
--------
Basic synchronous usage with caching and retry:

    >>> from insideLLMs.models import OpenAIModel
    >>> from insideLLMs.runtime.pipeline import (
    ...     ModelPipeline,
    ...     CacheMiddleware,
    ...     RetryMiddleware,
    ... )
    >>>
    >>> # Create a base model
    >>> base_model = OpenAIModel(model_name="gpt-4")
    >>>
    >>> # Build pipeline with middleware
    >>> pipeline = ModelPipeline(
    ...     base_model,
    ...     middlewares=[
    ...         CacheMiddleware(cache_size=500, ttl_seconds=3600),
    ...         RetryMiddleware(max_retries=3, initial_delay=1.0),
    ...     ],
    ... )
    >>>
    >>> # Use like any other model
    >>> response = pipeline.generate("Explain the transformer architecture")
    >>> print(response[:100])
    The transformer architecture, introduced in "Attention Is All You Need"...
    >>>
    >>> # Second call hits cache
    >>> response2 = pipeline.generate("Explain the transformer architecture")
    >>> print(pipeline.info()["cache_hits"])
    1

Full production pipeline with all middleware:

    >>> from insideLLMs.models import AnthropicModel
    >>> from insideLLMs.runtime.pipeline import (
    ...     ModelPipeline,
    ...     CacheMiddleware,
    ...     RateLimitMiddleware,
    ...     RetryMiddleware,
    ...     CostTrackingMiddleware,
    ...     TraceMiddleware,
    ... )
    >>>
    >>> base_model = AnthropicModel(model_name="claude-3-sonnet")
    >>>
    >>> # Order matters: outer middleware execute first on request
    >>> pipeline = ModelPipeline(
    ...     base_model,
    ...     middlewares=[
    ...         TraceMiddleware(run_id="session_001"),  # Trace everything
    ...         CacheMiddleware(cache_size=1000),       # Check cache early
    ...         RateLimitMiddleware(requests_per_minute=60),  # Rate limit
    ...         RetryMiddleware(max_retries=3),         # Retry on failure
    ...         CostTrackingMiddleware(),               # Track costs
    ...     ],
    ... )
    >>>
    >>> response = pipeline.generate("What is machine learning?")
    >>>
    >>> # Get comprehensive stats
    >>> info = pipeline.info()
    >>> print(f"Cache hit rate: {info['cache_hit_rate']:.2%}")
    >>> print(f"Total cost: ${info['cost_stats']['estimated_cost_usd']:.4f}")

Async batch processing with progress tracking:

    >>> import asyncio
    >>> from insideLLMs.runtime.pipeline import AsyncModelPipeline
    >>>
    >>> async def process_questions():
    ...     pipeline = AsyncModelPipeline(
    ...         base_model,
    ...         middlewares=[
    ...             CacheMiddleware(),
    ...             RateLimitMiddleware(requests_per_minute=120),
    ...         ],
    ...     )
    ...
    ...     questions = [
    ...         "What is Python?",
    ...         "Explain recursion",
    ...         "What are decorators?",
    ...         "Define a closure",
    ...     ]
    ...
    ...     # Process with progress callback
    ...     def on_progress(completed, total):
    ...         print(f"Progress: {completed}/{total}")
    ...
    ...     results = await pipeline.agenerate_with_callback(
    ...         questions,
    ...         max_concurrency=5,
    ...         on_progress=on_progress,
    ...     )
    ...     return results
    >>>
    >>> # Run async code
    >>> results = asyncio.run(process_questions())
    Progress: 1/4
    Progress: 2/4
    Progress: 3/4
    Progress: 4/4

Streaming results as they complete:

    >>> async def stream_results():
    ...     pipeline = AsyncModelPipeline(base_model, middlewares=[])
    ...     prompts = ["Q1", "Q2", "Q3"]
    ...
    ...     async for idx, result in pipeline.agenerate_stream_results(prompts):
    ...         print(f"Completed prompt {idx}: {result[:50]}...")

Custom middleware example:

    >>> from insideLLMs.runtime.pipeline import Middleware
    >>>
    >>> class LoggingMiddleware(Middleware):
    ...     '''Middleware that logs all requests and responses.'''
    ...
    ...     def __init__(self, logger):
    ...         super().__init__()
    ...         self.logger = logger
    ...
    ...     def process_generate(self, prompt: str, **kwargs) -> str:
    ...         self.logger.info(f"Request: {prompt[:50]}...")
    ...
    ...         # Delegate to next middleware or model
    ...         if self.next_middleware:
    ...             response = self.next_middleware.process_generate(prompt, **kwargs)
    ...         elif self.model:
    ...             response = self.model.generate(prompt, **kwargs)
    ...         else:
    ...             raise ModelError("No model available")
    ...
    ...         self.logger.info(f"Response: {response[:50]}...")
    ...         return response
    >>>
    >>> # Use custom middleware
    >>> import logging
    >>> logger = logging.getLogger("model")
    >>> pipeline = ModelPipeline(
    ...     base_model,
    ...     middlewares=[LoggingMiddleware(logger), CacheMiddleware()],
    ... )

Notes
-----
- Middleware order matters: outer middleware (first in list) execute first on
  requests and last on responses. Place caching early to avoid unnecessary
  processing, and place tracing at the outermost position to capture everything.

- The pipeline implements the Model interface, so it can be used anywhere a
  Model is expected, including as the base model of another pipeline (for
  complex middleware compositions).

- Cost tracking uses approximate token counts (4 characters ≈ 1 token) and
  may not match exact API billing. Use official API usage data for precise
  billing information.

- Cache keys are deterministic hashes of prompt and parameters, so identical
  requests will always hit the cache (within TTL).

See Also
--------
insideLLMs.models.base : Base model classes and protocols
insideLLMs.tracing : Trace recording utilities
insideLLMs.exceptions : Exception classes used by middleware
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Callable, Optional

from insideLLMs.exceptions import ModelError, RateLimitError, TimeoutError
from insideLLMs.models.base import (
    AsyncModelProtocol,
    ChatMessage,
    Model,
    ModelProtocol,
)

if TYPE_CHECKING:
    from insideLLMs.tracing import TraceRecorder


class Middleware(ABC):
    """Abstract base class for model pipeline middleware.

    Middleware provides a way to intercept, modify, or short-circuit model
    requests and responses. Common use cases include caching, rate limiting,
    retry logic, logging, tracing, and cost tracking.

    Middleware is organized in a chain: each middleware can process the request,
    optionally delegate to the next middleware or the base model, and then
    process the response before returning.

    To create custom middleware, subclass this class and implement at minimum
    the `process_generate` method. Override other methods as needed for chat,
    streaming, and async operations.

    Parameters
    ----------
    None
        The base class takes no parameters. Subclasses may define their own.

    Attributes
    ----------
    next_middleware : Optional[Middleware]
        The next middleware in the chain. Set automatically by ModelPipeline
        when the middleware is added to a pipeline.
    model : Optional[ModelProtocol]
        Reference to the base model. Set automatically by ModelPipeline.
        Use this to delegate requests when at the end of the middleware chain.

    Examples
    --------
    Creating a simple logging middleware:

        >>> class LoggingMiddleware(Middleware):
        ...     '''Logs all requests and responses.'''
        ...
        ...     def __init__(self, prefix: str = ""):
        ...         super().__init__()
        ...         self.prefix = prefix
        ...
        ...     def process_generate(self, prompt: str, **kwargs) -> str:
        ...         print(f"{self.prefix}Request: {prompt[:50]}...")
        ...
        ...         # Always delegate to next in chain
        ...         if self.next_middleware:
        ...             response = self.next_middleware.process_generate(
        ...                 prompt, **kwargs
        ...             )
        ...         elif self.model:
        ...             response = self.model.generate(prompt, **kwargs)
        ...         else:
        ...             raise ModelError("No model available")
        ...
        ...         print(f"{self.prefix}Response: {response[:50]}...")
        ...         return response
        >>>
        >>> # Use in pipeline
        >>> pipeline = ModelPipeline(model, middlewares=[LoggingMiddleware("[LOG] ")])
        >>> response = pipeline.generate("Hello")
        [LOG] Request: Hello...
        [LOG] Response: Hi there! How can I help you today?...

    Creating middleware that modifies requests:

        >>> class PromptPrefixMiddleware(Middleware):
        ...     '''Adds a prefix to all prompts.'''
        ...
        ...     def __init__(self, prefix: str):
        ...         super().__init__()
        ...         self.prefix = prefix
        ...
        ...     def process_generate(self, prompt: str, **kwargs) -> str:
        ...         modified_prompt = f"{self.prefix}\\n\\n{prompt}"
        ...
        ...         if self.next_middleware:
        ...             return self.next_middleware.process_generate(
        ...                 modified_prompt, **kwargs
        ...             )
        ...         elif self.model:
        ...             return self.model.generate(modified_prompt, **kwargs)
        ...         raise ModelError("No model available")
        >>>
        >>> # Add system context to all prompts
        >>> prefix_mw = PromptPrefixMiddleware(
        ...     "You are a helpful coding assistant. Be concise."
        ... )
        >>> pipeline = ModelPipeline(model, middlewares=[prefix_mw])

    Creating middleware that short-circuits the chain:

        >>> class BlocklistMiddleware(Middleware):
        ...     '''Blocks prompts containing forbidden words.'''
        ...
        ...     def __init__(self, blocklist: list[str]):
        ...         super().__init__()
        ...         self.blocklist = [w.lower() for w in blocklist]
        ...
        ...     def process_generate(self, prompt: str, **kwargs) -> str:
        ...         prompt_lower = prompt.lower()
        ...         for word in self.blocklist:
        ...             if word in prompt_lower:
        ...                 return "I cannot process this request."
        ...
        ...         # Safe prompt, continue chain
        ...         if self.next_middleware:
        ...             return self.next_middleware.process_generate(
        ...                 prompt, **kwargs
        ...             )
        ...         elif self.model:
        ...             return self.model.generate(prompt, **kwargs)
        ...         raise ModelError("No model available")

    See Also
    --------
    PassthroughMiddleware : Base class for observation-only middleware
    ModelPipeline : The pipeline that chains middleware together
    """

    def __init__(self) -> None:
        """Initialize the middleware base class.

        Sets up the chain references (next_middleware and model) to None.
        These are populated automatically when the middleware is added to
        a ModelPipeline.

        Examples
        --------
        >>> class MyMiddleware(Middleware):
        ...     def __init__(self, config: dict):
        ...         super().__init__()  # Always call parent __init__
        ...         self.config = config
        ...
        ...     def process_generate(self, prompt: str, **kwargs) -> str:
        ...         # Implementation here
        ...         pass
        """
        self.next_middleware: Optional["Middleware"] = None
        self.model: Optional[ModelProtocol] = None

    @abstractmethod
    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Process a text generation request.

        This is the primary method that all middleware must implement. It
        receives the prompt and any additional generation parameters, and
        should return the generated response string.

        Implementations typically follow this pattern:
        1. Optionally modify the prompt or kwargs
        2. Optionally short-circuit (return early without calling next)
        3. Delegate to next_middleware or model
        4. Optionally modify the response
        5. Return the response

        Args
        ----
        prompt : str
            The input prompt for text generation. May be a simple question,
            a detailed instruction, or a complex prompt template.
        **kwargs : Any
            Additional generation parameters passed through the chain.
            Common kwargs include:
            - temperature (float): Sampling temperature (0.0-2.0)
            - max_tokens (int): Maximum tokens to generate
            - stop (list[str]): Stop sequences
            - top_p (float): Nucleus sampling parameter

        Returns
        -------
        str
            The generated text response from the model or from middleware
            that short-circuits the chain (e.g., cached responses).

        Raises
        ------
        ModelError
            If generation fails or no model/middleware is available to
            handle the request.

        Examples
        --------
        Basic implementation that passes through unchanged:

            >>> def process_generate(self, prompt: str, **kwargs) -> str:
            ...     if self.next_middleware:
            ...         return self.next_middleware.process_generate(prompt, **kwargs)
            ...     elif self.model:
            ...         return self.model.generate(prompt, **kwargs)
            ...     raise ModelError("No model available in pipeline")

        Implementation with request modification:

            >>> def process_generate(self, prompt: str, **kwargs) -> str:
            ...     # Force lower temperature for deterministic output
            ...     kwargs['temperature'] = 0.0
            ...
            ...     if self.next_middleware:
            ...         return self.next_middleware.process_generate(prompt, **kwargs)
            ...     elif self.model:
            ...         return self.model.generate(prompt, **kwargs)
            ...     raise ModelError("No model available")

        Implementation with response modification:

            >>> def process_generate(self, prompt: str, **kwargs) -> str:
            ...     if self.next_middleware:
            ...         response = self.next_middleware.process_generate(
            ...             prompt, **kwargs
            ...         )
            ...     elif self.model:
            ...         response = self.model.generate(prompt, **kwargs)
            ...     else:
            ...         raise ModelError("No model available")
            ...
            ...     # Post-process: strip whitespace
            ...     return response.strip()

        Implementation that short-circuits:

            >>> def process_generate(self, prompt: str, **kwargs) -> str:
            ...     # Check cache first
            ...     cached = self.cache.get(prompt)
            ...     if cached:
            ...         return cached
            ...
            ...     # Cache miss, continue chain
            ...     if self.next_middleware:
            ...         response = self.next_middleware.process_generate(
            ...             prompt, **kwargs
            ...         )
            ...     elif self.model:
            ...         response = self.model.generate(prompt, **kwargs)
            ...     else:
            ...         raise ModelError("No model available")
            ...
            ...     self.cache[prompt] = response
            ...     return response
        """
        raise NotImplementedError

    def process_chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Process a multi-turn chat request.

        Handles conversation-style interactions with a list of messages
        representing the chat history. The default implementation delegates
        to the next middleware or model.

        Override this method if your middleware needs to handle chat
        differently from text generation (e.g., for conversation-aware
        caching or logging).

        Args
        ----
        messages : list[ChatMessage]
            The conversation history as a list of ChatMessage objects.
            Each message has a 'role' (e.g., 'user', 'assistant', 'system')
            and 'content' (the message text).
        **kwargs : Any
            Additional chat parameters. Common kwargs include:
            - temperature (float): Sampling temperature
            - max_tokens (int): Maximum tokens to generate
            - stop (list[str]): Stop sequences

        Returns
        -------
        str
            The assistant's response to the conversation.

        Raises
        ------
        ModelError
            If chat fails or no chat implementation is available.

        Examples
        --------
        Using chat through middleware:

            >>> messages = [
            ...     ChatMessage(role="system", content="You are helpful."),
            ...     ChatMessage(role="user", content="What is Python?"),
            ... ]
            >>> response = middleware.process_chat(messages, temperature=0.7)
            >>> print(response)
            Python is a high-level programming language...

        Custom implementation that logs conversations:

            >>> def process_chat(self, messages: list[ChatMessage], **kwargs) -> str:
            ...     # Log the conversation
            ...     for msg in messages:
            ...         self.logger.debug(f"{msg.role}: {msg.content[:50]}...")
            ...
            ...     # Delegate to chain
            ...     if self.next_middleware:
            ...         response = self.next_middleware.process_chat(
            ...             messages, **kwargs
            ...         )
            ...     elif self.model and hasattr(self.model, "chat"):
            ...         response = self.model.chat(messages, **kwargs)
            ...     else:
            ...         raise ModelError("No chat implementation available")
            ...
            ...     self.logger.debug(f"assistant: {response[:50]}...")
            ...     return response
        """
        # Default implementation delegates to next middleware or model
        if self.next_middleware:
            return self.next_middleware.process_chat(messages, **kwargs)
        if self.model and hasattr(self.model, "chat"):
            return self.model.chat(messages, **kwargs)
        raise ModelError("No chat implementation available")

    def process_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Process a streaming text generation request.

        Returns an iterator that yields response chunks as they are
        generated. This enables real-time display of model output and
        is useful for long responses.

        The default implementation delegates to the next middleware or
        model's streaming interface. Override for streaming-aware
        middleware behavior.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters. Common kwargs include:
            - temperature (float): Sampling temperature
            - max_tokens (int): Maximum tokens to generate

        Yields
        ------
        str
            Response chunks as they are generated. Chunks are typically
            words or word fragments, depending on the model's tokenization.

        Raises
        ------
        ModelError
            If streaming fails or no streaming implementation is available.

        Examples
        --------
        Basic streaming usage:

            >>> for chunk in middleware.process_stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
            Once upon a time...

        Collecting streamed output:

            >>> chunks = list(middleware.process_stream("Explain AI"))
            >>> full_response = "".join(chunks)
            >>> print(full_response)

        Custom implementation that tracks chunks:

            >>> def process_stream(self, prompt: str, **kwargs) -> Iterator[str]:
            ...     chunk_count = 0
            ...
            ...     if self.next_middleware:
            ...         stream = self.next_middleware.process_stream(prompt, **kwargs)
            ...     elif self.model and hasattr(self.model, "stream"):
            ...         stream = self.model.stream(prompt, **kwargs)
            ...     else:
            ...         raise ModelError("No streaming implementation")
            ...
            ...     for chunk in stream:
            ...         chunk_count += 1
            ...         yield chunk
            ...
            ...     self.logger.info(f"Streamed {chunk_count} chunks")
        """
        # Default implementation delegates to next middleware or model
        if self.next_middleware:
            yield from self.next_middleware.process_stream(prompt, **kwargs)
        elif self.model and hasattr(self.model, "stream"):
            yield from self.model.stream(prompt, **kwargs)
        else:
            raise ModelError("No streaming implementation available")

    # Async methods - default implementations delegate to sync or use executor

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously process a text generation request.

        The async counterpart to process_generate. The default implementation
        delegates to the next middleware's async method or runs the model's
        sync method in an executor if no async implementation is available.

        For true async behavior (e.g., using aiohttp for API calls), override
        this method in your middleware subclass.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters.

        Returns
        -------
        str
            The generated text response.

        Raises
        ------
        ModelError
            If generation fails or no model is available.

        Examples
        --------
        Using async generation:

            >>> async def main():
            ...     response = await middleware.aprocess_generate(
            ...         "Explain quantum computing",
            ...         temperature=0.7
            ...     )
            ...     print(response)
            >>> asyncio.run(main())

        Custom async implementation with true async I/O:

            >>> async def aprocess_generate(self, prompt: str, **kwargs) -> str:
            ...     # Async HTTP call
            ...     async with aiohttp.ClientSession() as session:
            ...         async with session.post(self.api_url, json={
            ...             "prompt": prompt, **kwargs
            ...         }) as resp:
            ...             data = await resp.json()
            ...             return data["response"]

        Implementation that wraps sync code:

            >>> async def aprocess_generate(self, prompt: str, **kwargs) -> str:
            ...     # Expensive sync operation
            ...     loop = asyncio.get_running_loop()
            ...     result = await loop.run_in_executor(
            ...         None,
            ...         lambda: self.sync_process(prompt, **kwargs)
            ...     )
            ...     return result
        """
        if self.next_middleware:
            return await self.next_middleware.aprocess_generate(prompt, **kwargs)
        if self.model:
            if isinstance(self.model, AsyncModelProtocol):
                return await self.model.agenerate(prompt, **kwargs)
            # Fall back to running sync method in executor
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self.model.generate(prompt, **kwargs))
        raise ModelError("No model available in pipeline")

    async def aprocess_chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Asynchronously process a multi-turn chat request.

        The async counterpart to process_chat. Handles conversation-style
        interactions asynchronously.

        Args
        ----
        messages : list[ChatMessage]
            The conversation history as a list of ChatMessage objects.
        **kwargs : Any
            Additional chat parameters.

        Returns
        -------
        str
            The assistant's response to the conversation.

        Raises
        ------
        ModelError
            If chat fails or no chat implementation is available.

        Examples
        --------
        Async chat usage:

            >>> async def chat_example():
            ...     messages = [
            ...         ChatMessage(role="user", content="Hello!"),
            ...     ]
            ...     response = await middleware.aprocess_chat(messages)
            ...     print(response)
            >>> asyncio.run(chat_example())
            Hi! How can I help you today?

        Processing multiple conversations concurrently:

            >>> async def process_conversations(conversations):
            ...     tasks = [
            ...         middleware.aprocess_chat(msgs)
            ...         for msgs in conversations
            ...     ]
            ...     return await asyncio.gather(*tasks)
        """
        if self.next_middleware:
            return await self.next_middleware.aprocess_chat(messages, **kwargs)
        if self.model:
            if hasattr(self.model, "achat"):
                return await self.model.achat(messages, **kwargs)
            if hasattr(self.model, "chat"):
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: self.model.chat(messages, **kwargs))
        raise ModelError("No chat implementation available")

    async def aprocess_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously process a streaming text generation request.

        The async counterpart to process_stream. Returns an async iterator
        that yields response chunks as they are generated.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters.

        Yields
        ------
        str
            Response chunks as they are generated asynchronously.

        Raises
        ------
        ModelError
            If streaming fails or no streaming implementation is available.

        Examples
        --------
        Async streaming usage:

            >>> async def stream_example():
            ...     async for chunk in middleware.aprocess_stream("Tell a joke"):
            ...         print(chunk, end="", flush=True)
            >>> asyncio.run(stream_example())
            Why did the programmer quit? Because they didn't get arrays!

        Collecting async streamed output:

            >>> async def collect_stream():
            ...     chunks = []
            ...     async for chunk in middleware.aprocess_stream("Explain AI"):
            ...         chunks.append(chunk)
            ...     return "".join(chunks)

        Custom implementation with progress tracking:

            >>> async def aprocess_stream(self, prompt: str, **kwargs):
            ...     chunk_count = 0
            ...     async for chunk in super().aprocess_stream(prompt, **kwargs):
            ...         chunk_count += 1
            ...         if chunk_count % 10 == 0:
            ...             self.logger.debug(f"Streamed {chunk_count} chunks")
            ...         yield chunk
        """
        if self.next_middleware:
            async for chunk in self.next_middleware.aprocess_stream(prompt, **kwargs):
                yield chunk
        elif self.model:
            if hasattr(self.model, "astream"):
                async for chunk in self.model.astream(prompt, **kwargs):
                    yield chunk
            elif hasattr(self.model, "stream"):
                loop = asyncio.get_running_loop()
                sync_iter = await loop.run_in_executor(
                    None, lambda: list(self.model.stream(prompt, **kwargs))
                )
                for chunk in sync_iter:
                    yield chunk
            else:
                raise ModelError("No streaming implementation available")
        else:
            raise ModelError("No streaming implementation available")


class PassthroughMiddleware(Middleware):
    """Middleware that passes requests through unchanged.

    A concrete implementation of Middleware that simply delegates all requests
    to the next middleware or model without modification. This is useful as a
    base class for middleware that only needs to observe requests (logging,
    metrics collection) without altering them.

    This class provides complete implementations of all abstract methods,
    making it easier to create middleware that only overrides specific
    operations while inheriting sensible defaults for the rest.

    Parameters
    ----------
    None
        PassthroughMiddleware takes no parameters.

    Attributes
    ----------
    next_middleware : Optional[Middleware]
        Inherited from Middleware. The next middleware in the chain.
    model : Optional[ModelProtocol]
        Inherited from Middleware. Reference to the base model.

    Examples
    --------
    Using PassthroughMiddleware directly (no-op middleware):

        >>> # Useful for testing or as a placeholder
        >>> pipeline = ModelPipeline(
        ...     model,
        ...     middlewares=[PassthroughMiddleware(), CacheMiddleware()]
        ... )

    Subclassing for observation-only middleware:

        >>> class MetricsMiddleware(PassthroughMiddleware):
        ...     '''Collects metrics without modifying requests.'''
        ...
        ...     def __init__(self, metrics_client):
        ...         super().__init__()
        ...         self.metrics = metrics_client
        ...         self.request_count = 0
        ...
        ...     def process_generate(self, prompt: str, **kwargs) -> str:
        ...         # Record metrics before delegating
        ...         self.request_count += 1
        ...         self.metrics.increment("model.requests")
        ...
        ...         start_time = time.time()
        ...         # Use parent's passthrough behavior
        ...         response = super().process_generate(prompt, **kwargs)
        ...
        ...         # Record latency after response
        ...         latency = time.time() - start_time
        ...         self.metrics.timing("model.latency", latency)
        ...
        ...         return response

    Creating timing middleware:

        >>> class TimingMiddleware(PassthroughMiddleware):
        ...     '''Records request timing information.'''
        ...
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.timings = []
        ...
        ...     def process_generate(self, prompt: str, **kwargs) -> str:
        ...         start = time.perf_counter()
        ...         response = super().process_generate(prompt, **kwargs)
        ...         elapsed = time.perf_counter() - start
        ...
        ...         self.timings.append({
        ...             "prompt_length": len(prompt),
        ...             "response_length": len(response),
        ...             "elapsed_seconds": elapsed,
        ...         })
        ...         return response
        ...
        ...     @property
        ...     def avg_latency(self) -> float:
        ...         if not self.timings:
        ...             return 0.0
        ...         return sum(t["elapsed_seconds"] for t in self.timings) / len(self.timings)

    See Also
    --------
    Middleware : The abstract base class
    TraceMiddleware : A PassthroughMiddleware subclass for execution tracing
    """

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Pass the request through to the next middleware or model unchanged.

        This implementation simply delegates to the next middleware in the
        chain, or directly to the model if this is the last middleware.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters, passed through unchanged.

        Returns
        -------
        str
            The generated text response from downstream.

        Raises
        ------
        ModelError
            If no model is available in the pipeline.

        Examples
        --------
        Direct usage:

            >>> middleware = PassthroughMiddleware()
            >>> middleware.model = some_model
            >>> response = middleware.process_generate("Hello")
            >>> print(response)
            Hi there!

        In subclass with pre/post processing:

            >>> def process_generate(self, prompt: str, **kwargs) -> str:
            ...     print(f"Before: {len(prompt)} chars")
            ...     response = super().process_generate(prompt, **kwargs)
            ...     print(f"After: {len(response)} chars")
            ...     return response
        """
        if self.next_middleware:
            return self.next_middleware.process_generate(prompt, **kwargs)
        if self.model:
            return self.model.generate(prompt, **kwargs)
        raise ModelError("No model available in pipeline")

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Async pass the request through to the next middleware or model.

        The asynchronous counterpart to process_generate. Delegates to the
        next async middleware or wraps the sync model in an executor.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters, passed through unchanged.

        Returns
        -------
        str
            The generated text response from downstream.

        Raises
        ------
        ModelError
            If no model is available in the pipeline.

        Examples
        --------
        Direct async usage:

            >>> async def example():
            ...     middleware = PassthroughMiddleware()
            ...     middleware.model = some_async_model
            ...     response = await middleware.aprocess_generate("Hello")
            ...     return response

        In async subclass:

            >>> async def aprocess_generate(self, prompt: str, **kwargs) -> str:
            ...     self.logger.info("Starting async request")
            ...     response = await super().aprocess_generate(prompt, **kwargs)
            ...     self.logger.info("Completed async request")
            ...     return response
        """
        if self.next_middleware:
            return await self.next_middleware.aprocess_generate(prompt, **kwargs)
        if self.model:
            if isinstance(self.model, AsyncModelProtocol):
                return await self.model.agenerate(prompt, **kwargs)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self.model.generate(prompt, **kwargs))
        raise ModelError("No model available in pipeline")


class TraceMiddleware(PassthroughMiddleware):
    """Middleware for capturing detailed execution traces.

    Records trace events for generate, chat, and stream operations, providing
    a complete record of model interactions for debugging, validation, and
    reproducibility. Uses the TraceRecorder from insideLLMs.tracing with
    deterministic event sequencing (monotonic sequence numbers instead of
    wall-clock time).

    Traces capture:
    - Request start events with prompt and parameters
    - Response end events with full output
    - Stream chunk events for streaming operations
    - Error events with exception details
    - Chat events for multi-turn conversations

    The recorder instance is accessible after execution to retrieve trace data,
    which can be stored in ResultRecord.custom for analysis or exported.

    Parameters
    ----------
    run_id : str, optional
        Identifier for the current evaluation run. Used to group traces
        from the same run. Defaults to None.
    example_id : str, optional
        Identifier for the specific example being processed. Useful when
        running evaluations with multiple test cases. Defaults to None.

    Attributes
    ----------
    recorder : TraceRecorder
        The trace recorder instance. Access this after execution to retrieve
        captured trace events.
    _RESERVED_KWARGS : set[str]
        Class attribute containing kwargs that are reserved for tracing and
        should not be passed to model providers. Includes: _trace,
        _trace_recorder, _run_id, _example_id.

    Examples
    --------
    Basic tracing usage:

        >>> from insideLLMs.runtime.pipeline import TraceMiddleware, ModelPipeline
        >>> from insideLLMs.models import OpenAIModel
        >>>
        >>> # Create trace middleware
        >>> trace_mw = TraceMiddleware(run_id="eval_001", example_id="test_1")
        >>>
        >>> # Build pipeline with tracing
        >>> model = OpenAIModel("gpt-4")
        >>> pipeline = ModelPipeline(model, middlewares=[trace_mw])
        >>>
        >>> # Execute request
        >>> response = pipeline.generate("What is 2+2?")
        >>>
        >>> # Examine trace events
        >>> events = trace_mw.recorder.events
        >>> print(f"Captured {len(events)} events")
        Captured 2 events
        >>> for event in events:
        ...     print(f"  {event.kind}: {event.data.keys()}")
        GENERATE_START: dict_keys(['prompt', 'params'])
        GENERATE_END: dict_keys(['response'])

    Tracing with error handling:

        >>> trace_mw = TraceMiddleware(run_id="error_test")
        >>> pipeline = ModelPipeline(unreliable_model, middlewares=[trace_mw])
        >>>
        >>> try:
        ...     response = pipeline.generate("Test prompt")
        ... except ModelError:
        ...     # Error was captured in trace
        ...     error_events = [
        ...         e for e in trace_mw.recorder.events
        ...         if e.kind.name == "ERROR"
        ...     ]
        ...     print(f"Captured {len(error_events)} error(s)")
        ...     print(f"Error: {error_events[0].data['message']}")

    Tracing streaming operations:

        >>> trace_mw = TraceMiddleware(run_id="stream_test")
        >>> pipeline = ModelPipeline(model, middlewares=[trace_mw])
        >>>
        >>> # Stream and collect response
        >>> chunks = []
        >>> for chunk in pipeline.stream("Tell me a story"):
        ...     chunks.append(chunk)
        >>>
        >>> # Examine stream trace
        >>> events = trace_mw.recorder.events
        >>> stream_start = events[0]
        >>> chunk_events = [e for e in events if "CHUNK" in e.kind.name]
        >>> stream_end = events[-1]
        >>>
        >>> print(f"Streamed {len(chunk_events)} chunks")
        >>> print(f"Full response: {stream_end.data['full_response'][:50]}...")

    Resetting for multiple evaluations:

        >>> trace_mw = TraceMiddleware()
        >>> pipeline = ModelPipeline(model, middlewares=[trace_mw])
        >>>
        >>> results = []
        >>> for i, prompt in enumerate(test_prompts):
        ...     # Reset for each example
        ...     trace_mw.reset(run_id="batch_run", example_id=f"example_{i}")
        ...
        ...     response = pipeline.generate(prompt)
        ...
        ...     # Store trace with result
        ...     results.append({
        ...         "prompt": prompt,
        ...         "response": response,
        ...         "trace": trace_mw.recorder.events.copy(),
        ...     })

    Combining with other middleware:

        >>> # Place TraceMiddleware first to capture everything
        >>> pipeline = ModelPipeline(
        ...     model,
        ...     middlewares=[
        ...         TraceMiddleware(run_id="production"),  # Traces all
        ...         CacheMiddleware(),                      # Cache hits traced
        ...         RetryMiddleware(max_retries=3),        # Retries traced
        ...         RateLimitMiddleware(),                 # Rate limits traced
        ...     ],
        ... )

    See Also
    --------
    insideLLMs.tracing.TraceRecorder : The underlying trace recorder
    insideLLMs.tracing.TraceEvent : Individual trace event structure
    insideLLMs.tracing.TraceEventKind : Enumeration of trace event types
    PassthroughMiddleware : The parent class
    """

    # Reserved kwargs that should not leak to providers
    _RESERVED_KWARGS = {"_trace", "_trace_recorder", "_run_id", "_example_id"}

    def __init__(
        self,
        run_id: Optional[str] = None,
        example_id: Optional[str] = None,
    ) -> None:
        """Initialize the trace middleware with optional context identifiers.

        Creates a new TraceRecorder instance that will capture all subsequent
        model operations until reset() is called.

        Args
        ----
        run_id : str, optional
            Identifier for the current evaluation run. Use this to group
            traces from multiple examples in the same run. Common patterns:
            - "eval_2024_01_15_001" (date-based)
            - "experiment_alpha" (experiment name)
            - UUID strings for unique identification
        example_id : str, optional
            Identifier for the specific example or test case. Useful for:
            - Linking traces to dataset examples
            - Debugging specific failing cases
            - Organizing traces in multi-example evaluations

        Examples
        --------
        Basic initialization:

            >>> trace_mw = TraceMiddleware()
            >>> print(trace_mw.recorder.run_id)
            None

        With run context:

            >>> trace_mw = TraceMiddleware(
            ...     run_id="evaluation_2024_01",
            ...     example_id="math_problem_42"
            ... )
            >>> print(trace_mw.recorder.run_id)
            evaluation_2024_01

        In a loop:

            >>> for idx, example in enumerate(dataset):
            ...     trace_mw = TraceMiddleware(
            ...         run_id="batch_001",
            ...         example_id=example["id"]
            ...     )
            ...     # ... use trace_mw
        """
        super().__init__()
        # Lazy import to avoid circular dependencies
        from insideLLMs.tracing import TraceRecorder

        self._recorder = TraceRecorder(run_id=run_id, example_id=example_id)

    @property
    def recorder(self) -> "TraceRecorder":
        """Get the trace recorder instance.

        The recorder contains all captured trace events from model operations.
        Access this property after executing requests to retrieve trace data.

        Returns
        -------
        TraceRecorder
            The trace recorder instance containing captured events.

        Examples
        --------
        Accessing events:

            >>> trace_mw = TraceMiddleware()
            >>> # ... execute requests ...
            >>> recorder = trace_mw.recorder
            >>> print(f"Events: {len(recorder.events)}")
            >>> print(f"Run ID: {recorder.run_id}")

        Exporting trace data:

            >>> events = trace_mw.recorder.events
            >>> trace_data = [
            ...     {"kind": e.kind.name, "seq": e.sequence, "data": e.data}
            ...     for e in events
            ... ]
            >>> import json
            >>> json.dump(trace_data, open("trace.json", "w"))
        """
        return self._recorder

    def reset(
        self,
        run_id: Optional[str] = None,
        example_id: Optional[str] = None,
    ) -> None:
        """Reset the recorder for a new execution context.

        Creates a fresh TraceRecorder, discarding all previously captured
        events. Use this when processing multiple examples to keep traces
        separate, or when reusing the same pipeline instance for different
        evaluation runs.

        Args
        ----
        run_id : str, optional
            New run identifier for the reset recorder. If None, the recorder
            will have no run_id set.
        example_id : str, optional
            New example identifier for the reset recorder. If None, the
            recorder will have no example_id set.

        Examples
        --------
        Resetting between examples:

            >>> trace_mw = TraceMiddleware(run_id="my_run")
            >>> pipeline = ModelPipeline(model, middlewares=[trace_mw])
            >>>
            >>> for idx, example in enumerate(examples):
            ...     trace_mw.reset(
            ...         run_id="my_run",
            ...         example_id=f"ex_{idx}"
            ...     )
            ...     response = pipeline.generate(example["prompt"])
            ...     # Process trace_mw.recorder.events

        Complete reset with new context:

            >>> trace_mw.reset()  # Clear all context
            >>> trace_mw.reset(run_id="new_run")  # New run only
        """
        from insideLLMs.tracing import TraceRecorder

        self._recorder = TraceRecorder(run_id=run_id, example_id=example_id)

    def _strip_reserved_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Remove reserved trace kwargs before passing to model providers.

        Internal method that filters out kwargs that are reserved for
        tracing purposes and should not be forwarded to the underlying
        model's API calls. This prevents trace-related parameters from
        causing errors or unexpected behavior in model providers.

        Args
        ----
        kwargs : dict[str, Any]
            The keyword arguments to filter.

        Returns
        -------
        dict[str, Any]
            A new dictionary with reserved kwargs removed.

        Examples
        --------
        Internal usage (typically not called directly):

            >>> kwargs = {
            ...     "temperature": 0.7,
            ...     "_trace": True,
            ...     "_run_id": "test",
            ...     "max_tokens": 100
            ... }
            >>> clean = trace_mw._strip_reserved_kwargs(kwargs)
            >>> print(clean)
            {'temperature': 0.7, 'max_tokens': 100}
        """
        return {k: v for k, v in kwargs.items() if k not in self._RESERVED_KWARGS}

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Process a text generation request with full trace recording.

        Records GENERATE_START before execution, GENERATE_END after successful
        completion, or ERROR if an exception occurs. Reserved kwargs are
        stripped before passing to downstream middleware or the model.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters. Reserved kwargs (_trace,
            _trace_recorder, _run_id, _example_id) are automatically
            stripped before forwarding.

        Returns
        -------
        str
            The generated text response.

        Raises
        ------
        ModelError
            If generation fails. The error is recorded in the trace before
            being re-raised.

        Examples
        --------
        Basic traced generation:

            >>> trace_mw = TraceMiddleware(run_id="test")
            >>> trace_mw.model = model
            >>> response = trace_mw.process_generate("What is AI?")
            >>> print(len(trace_mw.recorder.events))
            2  # GENERATE_START and GENERATE_END

        Examining trace after generation:

            >>> events = trace_mw.recorder.events
            >>> start_event = events[0]
            >>> print(start_event.kind.name)
            GENERATE_START
            >>> print(start_event.data["prompt"])
            What is AI?
        """
        # Strip reserved kwargs
        clean_kwargs = self._strip_reserved_kwargs(kwargs)

        # Record start
        self._recorder.record_generate_start(prompt, **clean_kwargs)

        try:
            # Delegate to next middleware or model
            if self.next_middleware:
                response = self.next_middleware.process_generate(prompt, **clean_kwargs)
            elif self.model:
                response = self.model.generate(prompt, **clean_kwargs)
            else:
                raise ModelError("No model available in pipeline")

            # Record end
            self._recorder.record_generate_end(response)
            return response

        except Exception as e:
            # Record error
            self._recorder.record_error(
                str(e),
                error_type=type(e).__name__,
            )
            raise

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously process a text generation request with trace recording.

        The async counterpart to process_generate. Records the same trace
        events (GENERATE_START, GENERATE_END, ERROR) for async operations.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters. Reserved kwargs are stripped.

        Returns
        -------
        str
            The generated text response.

        Raises
        ------
        ModelError
            If generation fails. The error is recorded before re-raising.

        Examples
        --------
        Async traced generation:

            >>> async def example():
            ...     trace_mw = TraceMiddleware(run_id="async_test")
            ...     trace_mw.model = async_model
            ...     response = await trace_mw.aprocess_generate("Explain ML")
            ...     return trace_mw.recorder.events
            >>>
            >>> events = asyncio.run(example())
            >>> print(len(events))
            2
        """
        # Strip reserved kwargs
        clean_kwargs = self._strip_reserved_kwargs(kwargs)

        # Record start
        self._recorder.record_generate_start(prompt, **clean_kwargs)

        try:
            # Delegate to next middleware or model
            if self.next_middleware:
                response = await self.next_middleware.aprocess_generate(prompt, **clean_kwargs)
            elif self.model:
                if isinstance(self.model, AsyncModelProtocol):
                    response = await self.model.agenerate(prompt, **clean_kwargs)
                else:
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(
                        None, lambda: self.model.generate(prompt, **clean_kwargs)
                    )
            else:
                raise ModelError("No model available in pipeline")

            # Record end
            self._recorder.record_generate_end(response)
            return response

        except Exception as e:
            # Record error
            self._recorder.record_error(
                str(e),
                error_type=type(e).__name__,
            )
            raise

    def process_chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Process a chat request with trace recording.

        Records CHAT_START with message count and parameters before execution,
        CHAT_END with the response after successful completion, or ERROR if
        an exception occurs.

        Args
        ----
        messages : list[ChatMessage]
            The conversation history as a list of ChatMessage objects.
        **kwargs : Any
            Additional chat parameters. Reserved kwargs are stripped.

        Returns
        -------
        str
            The assistant's response to the conversation.

        Raises
        ------
        ModelError
            If chat fails. The error is recorded before re-raising.

        Examples
        --------
        Traced chat:

            >>> trace_mw = TraceMiddleware(run_id="chat_test")
            >>> trace_mw.model = chat_model
            >>> messages = [ChatMessage(role="user", content="Hello!")]
            >>> response = trace_mw.process_chat(messages)
            >>>
            >>> # Examine chat trace
            >>> start = trace_mw.recorder.events[0]
            >>> print(start.data["message_count"])
            1
        """
        from insideLLMs.tracing import TraceEventKind

        clean_kwargs = self._strip_reserved_kwargs(kwargs)

        # Record chat start
        self._recorder.record(
            TraceEventKind.CHAT_START,
            {"message_count": len(messages), "params": clean_kwargs},
        )

        try:
            # Delegate
            if self.next_middleware:
                response = self.next_middleware.process_chat(messages, **clean_kwargs)
            elif self.model and hasattr(self.model, "chat"):
                response = self.model.chat(messages, **clean_kwargs)
            else:
                raise ModelError("No chat implementation available")

            # Record chat end
            self._recorder.record(
                TraceEventKind.CHAT_END,
                {"response": response},
            )
            return response

        except Exception as e:
            self._recorder.record_error(str(e), error_type=type(e).__name__)
            raise

    async def aprocess_chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Asynchronously process a chat request with trace recording.

        The async counterpart to process_chat. Records CHAT_START, CHAT_END,
        and ERROR events for async chat operations.

        Args
        ----
        messages : list[ChatMessage]
            The conversation history as a list of ChatMessage objects.
        **kwargs : Any
            Additional chat parameters. Reserved kwargs are stripped.

        Returns
        -------
        str
            The assistant's response to the conversation.

        Raises
        ------
        ModelError
            If chat fails. The error is recorded before re-raising.

        Examples
        --------
        Async traced chat:

            >>> async def chat_example():
            ...     trace_mw = TraceMiddleware()
            ...     trace_mw.model = async_chat_model
            ...     messages = [ChatMessage(role="user", content="Hi!")]
            ...     response = await trace_mw.aprocess_chat(messages)
            ...     return response, trace_mw.recorder.events
        """
        from insideLLMs.tracing import TraceEventKind

        clean_kwargs = self._strip_reserved_kwargs(kwargs)

        # Record chat start
        self._recorder.record(
            TraceEventKind.CHAT_START,
            {"message_count": len(messages), "params": clean_kwargs},
        )

        try:
            # Delegate
            if self.next_middleware:
                response = await self.next_middleware.aprocess_chat(messages, **clean_kwargs)
            elif self.model:
                if hasattr(self.model, "achat"):
                    response = await self.model.achat(messages, **clean_kwargs)
                elif hasattr(self.model, "chat"):
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(
                        None, lambda: self.model.chat(messages, **clean_kwargs)
                    )
                else:
                    raise ModelError("No chat implementation available")
            else:
                raise ModelError("No chat implementation available")

            # Record chat end
            self._recorder.record(
                TraceEventKind.CHAT_END,
                {"response": response},
            )
            return response

        except Exception as e:
            self._recorder.record_error(str(e), error_type=type(e).__name__)
            raise

    def process_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Process a streaming request with detailed chunk-level tracing.

        Records STREAM_START before streaming begins, STREAM_CHUNK for each
        chunk received (with chunk index), and STREAM_END with the full
        accumulated response and chunk count. If an error occurs, an ERROR
        event is recorded.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters. Reserved kwargs are stripped.

        Yields
        ------
        str
            Response chunks as they are generated.

        Raises
        ------
        ModelError
            If streaming fails. The error is recorded before re-raising.

        Examples
        --------
        Traced streaming:

            >>> trace_mw = TraceMiddleware(run_id="stream_test")
            >>> trace_mw.model = streaming_model
            >>>
            >>> chunks = []
            >>> for chunk in trace_mw.process_stream("Tell a story"):
            ...     chunks.append(chunk)
            >>>
            >>> # Examine trace
            >>> events = trace_mw.recorder.events
            >>> print(events[0].kind.name)  # STREAM_START
            STREAM_START
            >>> chunk_events = [e for e in events if "CHUNK" in e.kind.name]
            >>> print(f"Recorded {len(chunk_events)} chunks")
            >>> print(events[-1].data["chunk_count"])  # STREAM_END
        """
        clean_kwargs = self._strip_reserved_kwargs(kwargs)

        # Record stream start
        self._recorder.record_stream_start(prompt, **clean_kwargs)

        chunk_index = 0
        accumulated = []

        try:
            # Delegate to next middleware or model
            if self.next_middleware:
                stream = self.next_middleware.process_stream(prompt, **clean_kwargs)
            elif self.model and hasattr(self.model, "stream"):
                stream = self.model.stream(prompt, **clean_kwargs)
            else:
                raise ModelError("No streaming implementation available")

            for chunk in stream:
                # Record each chunk
                self._recorder.record_stream_chunk(chunk, chunk_index)
                accumulated.append(chunk)
                chunk_index += 1
                yield chunk

            # Record stream end
            self._recorder.record_stream_end(
                full_response="".join(accumulated),
                chunk_count=chunk_index,
            )

        except Exception as e:
            self._recorder.record_error(str(e), error_type=type(e).__name__)
            raise

    async def aprocess_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously process a streaming request with chunk-level tracing.

        The async counterpart to process_stream. Records STREAM_START,
        STREAM_CHUNK for each chunk, and STREAM_END events for async
        streaming operations.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters. Reserved kwargs are stripped.

        Yields
        ------
        str
            Response chunks as they are generated asynchronously.

        Raises
        ------
        ModelError
            If streaming fails. The error is recorded before re-raising.

        Examples
        --------
        Async traced streaming:

            >>> async def stream_example():
            ...     trace_mw = TraceMiddleware()
            ...     trace_mw.model = async_streaming_model
            ...
            ...     chunks = []
            ...     async for chunk in trace_mw.aprocess_stream("Write a poem"):
            ...         chunks.append(chunk)
            ...
            ...     return "".join(chunks), trace_mw.recorder.events
            >>>
            >>> response, events = asyncio.run(stream_example())
            >>> print(f"Total events: {len(events)}")
        """
        clean_kwargs = self._strip_reserved_kwargs(kwargs)

        # Record stream start
        self._recorder.record_stream_start(prompt, **clean_kwargs)

        chunk_index = 0
        accumulated = []

        try:
            # Delegate to next middleware or model
            if self.next_middleware:
                async for chunk in self.next_middleware.aprocess_stream(prompt, **clean_kwargs):
                    self._recorder.record_stream_chunk(chunk, chunk_index)
                    accumulated.append(chunk)
                    chunk_index += 1
                    yield chunk
            elif self.model:
                if hasattr(self.model, "astream"):
                    async for chunk in self.model.astream(prompt, **clean_kwargs):
                        self._recorder.record_stream_chunk(chunk, chunk_index)
                        accumulated.append(chunk)
                        chunk_index += 1
                        yield chunk
                elif hasattr(self.model, "stream"):
                    loop = asyncio.get_running_loop()
                    sync_chunks = await loop.run_in_executor(
                        None, lambda: list(self.model.stream(prompt, **clean_kwargs))
                    )
                    for chunk in sync_chunks:
                        self._recorder.record_stream_chunk(chunk, chunk_index)
                        accumulated.append(chunk)
                        chunk_index += 1
                        yield chunk
                else:
                    raise ModelError("No streaming implementation available")
            else:
                raise ModelError("No streaming implementation available")

            # Record stream end
            self._recorder.record_stream_end(
                full_response="".join(accumulated),
                chunk_count=chunk_index,
            )

        except Exception as e:
            self._recorder.record_error(str(e), error_type=type(e).__name__)
            raise


class CacheMiddleware(Middleware):
    """Middleware for caching model responses to avoid redundant API calls.

    Implements an in-memory cache with LRU (Least Recently Used) eviction and
    optional TTL (Time-To-Live) expiration. Cache keys are deterministic SHA-256
    hashes of the prompt and all parameters, ensuring identical requests always
    hit the cache.

    Caching is particularly useful for:
    - Evaluation runs where the same prompts may be processed multiple times
    - Development and testing to avoid repeated API calls
    - Cost reduction in scenarios with repeated queries
    - Improving latency for frequently requested responses

    Parameters
    ----------
    cache_size : int, default=1000
        Maximum number of entries to store in the cache. When this limit is
        reached, the oldest entry (by insertion order) is evicted to make
        room for new entries.
    ttl_seconds : float, optional
        Time-to-live for cache entries in seconds. Entries older than this
        are considered expired and will not be returned (though they remain
        in storage until evicted or replaced). If None (default), entries
        never expire based on time.

    Attributes
    ----------
    cache : dict[str, tuple[str, float]]
        The cache storage mapping cache keys to (response, timestamp) tuples.
    cache_size : int
        The configured maximum cache size.
    ttl_seconds : float or None
        The configured TTL, or None if no expiration.
    hits : int
        Counter for cache hits (requests served from cache).
    misses : int
        Counter for cache misses (requests forwarded to model).
    hit_rate : float
        Property that calculates the cache hit rate as hits / (hits + misses).

    Examples
    --------
    Basic caching:

        >>> from insideLLMs.runtime.pipeline import CacheMiddleware, ModelPipeline
        >>>
        >>> # Create cache with 500 entries max
        >>> cache_mw = CacheMiddleware(cache_size=500)
        >>> pipeline = ModelPipeline(model, middlewares=[cache_mw])
        >>>
        >>> # First call: cache miss, calls model
        >>> response1 = pipeline.generate("What is Python?")
        >>> print(f"Hits: {cache_mw.hits}, Misses: {cache_mw.misses}")
        Hits: 0, Misses: 1
        >>>
        >>> # Second identical call: cache hit, returns cached response
        >>> response2 = pipeline.generate("What is Python?")
        >>> print(f"Hits: {cache_mw.hits}, Misses: {cache_mw.misses}")
        Hits: 1, Misses: 1
        >>> assert response1 == response2  # Same response

    Caching with TTL expiration:

        >>> # Cache entries expire after 1 hour
        >>> cache_mw = CacheMiddleware(cache_size=1000, ttl_seconds=3600)
        >>> pipeline = ModelPipeline(model, middlewares=[cache_mw])
        >>>
        >>> response = pipeline.generate("Explain AI")
        >>> # After 1 hour, this entry will be considered expired

    Cache key includes parameters:

        >>> cache_mw = CacheMiddleware()
        >>> pipeline = ModelPipeline(model, middlewares=[cache_mw])
        >>>
        >>> # Different parameters = different cache keys
        >>> r1 = pipeline.generate("Hello", temperature=0.7)
        >>> r2 = pipeline.generate("Hello", temperature=0.9)
        >>> print(f"Misses: {cache_mw.misses}")
        Misses: 2  # Both are cache misses (different params)

    Monitoring cache performance:

        >>> cache_mw = CacheMiddleware(cache_size=100)
        >>> pipeline = ModelPipeline(model, middlewares=[cache_mw])
        >>>
        >>> # Run evaluation
        >>> for prompt in test_prompts:
        ...     pipeline.generate(prompt)
        >>>
        >>> # Check cache statistics
        >>> print(f"Hit rate: {cache_mw.hit_rate:.2%}")
        >>> print(f"Cache size: {len(cache_mw.cache)}")
        >>> info = pipeline.info()
        >>> print(f"Stats from pipeline: {info['cache_hit_rate']:.2%}")

    Async caching:

        >>> async def cached_batch():
        ...     cache_mw = CacheMiddleware()
        ...     pipeline = ModelPipeline(model, middlewares=[cache_mw])
        ...
        ...     # Run same prompts twice
        ...     prompts = ["Q1", "Q2", "Q3"]
        ...     await pipeline.abatch_generate(prompts)
        ...     await pipeline.abatch_generate(prompts)  # All cache hits
        ...
        ...     print(f"Final hit rate: {cache_mw.hit_rate:.2%}")
        ...     return cache_mw.hits, cache_mw.misses

    Notes
    -----
    - Cache keys are SHA-256 hashes of JSON-serialized prompt and parameters,
      ensuring deterministic and collision-resistant key generation.
    - The cache uses insertion-order dict semantics for LRU eviction (oldest
      entries are evicted first).
    - Expired entries are only removed when accessed; there is no background
      cleanup process.
    - The cache is not shared across pipeline instances; each CacheMiddleware
      has its own independent cache.
    - Async operations use an asyncio.Lock for thread-safe cache access.

    See Also
    --------
    Middleware : The base middleware class
    ModelPipeline : The pipeline that uses this middleware
    """

    def __init__(self, cache_size: int = 1000, ttl_seconds: Optional[float] = None) -> None:
        """Initialize the cache middleware with size and TTL configuration.

        Args
        ----
        cache_size : int, default=1000
            Maximum number of entries to store. When exceeded, the oldest
            entry is evicted using LRU policy.
        ttl_seconds : float, optional
            Time-to-live for entries in seconds. If None, entries never
            expire based on time.

        Examples
        --------
        Default configuration (1000 entries, no expiration):

            >>> cache_mw = CacheMiddleware()

        Limited cache with 1-hour TTL:

            >>> cache_mw = CacheMiddleware(cache_size=100, ttl_seconds=3600)

        Large cache for batch processing:

            >>> cache_mw = CacheMiddleware(cache_size=10000)
        """
        super().__init__()
        if cache_size < 1:
            raise ValueError("cache_size must be >= 1")
        self.cache: dict[str, tuple[str, float]] = {}
        self.cache_size = cache_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _cache_key(self, prompt: str, **kwargs: Any) -> str:
        """Generate a deterministic cache key from prompt and parameters.

        Creates a SHA-256 hash of the JSON-serialized prompt and kwargs,
        ensuring that identical inputs always produce the same cache key.

        Args
        ----
        prompt : str
            The input prompt.
        **kwargs : Any
            Additional generation parameters to include in the key.

        Returns
        -------
        str
            A 64-character hexadecimal SHA-256 hash string.

        Examples
        --------
        Key generation:

            >>> key1 = cache_mw._cache_key("Hello", temperature=0.7)
            >>> key2 = cache_mw._cache_key("Hello", temperature=0.7)
            >>> assert key1 == key2  # Same inputs = same key

            >>> key3 = cache_mw._cache_key("Hello", temperature=0.9)
            >>> assert key1 != key3  # Different params = different key
        """
        import hashlib
        import json

        key_data = {"prompt": prompt, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cache entry has expired based on its timestamp.

        Args
        ----
        timestamp : float
            The time.time() value when the entry was cached.

        Returns
        -------
        bool
            True if the entry is expired, False otherwise. Always returns
            False if ttl_seconds is None.

        Examples
        --------
        Checking expiration:

            >>> cache_mw = CacheMiddleware(ttl_seconds=60)
            >>> old_timestamp = time.time() - 120  # 2 minutes ago
            >>> cache_mw._is_expired(old_timestamp)
            True

            >>> recent_timestamp = time.time() - 30  # 30 seconds ago
            >>> cache_mw._is_expired(recent_timestamp)
            False
        """
        if self.ttl_seconds is None:
            return False
        return (time.time() - timestamp) > self.ttl_seconds

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Process a generate request with cache lookup and storage.

        First checks if a valid (non-expired) cached response exists for
        the given prompt and parameters. If found, returns the cached
        response immediately (cache hit). Otherwise, delegates to the
        next middleware or model, stores the response in cache, and
        returns it (cache miss).

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters. These are included in the
            cache key calculation.

        Returns
        -------
        str
            The generated (or cached) response.

        Raises
        ------
        ModelError
            If generation fails and no cached response is available.

        Examples
        --------
        Cache behavior:

            >>> cache_mw = CacheMiddleware()
            >>> cache_mw.model = model
            >>>
            >>> # First call: cache miss
            >>> r1 = cache_mw.process_generate("What is AI?")
            >>> print(cache_mw.hits, cache_mw.misses)
            0 1
            >>>
            >>> # Second call: cache hit
            >>> r2 = cache_mw.process_generate("What is AI?")
            >>> print(cache_mw.hits, cache_mw.misses)
            1 1
            >>> assert r1 == r2

        With different parameters:

            >>> r3 = cache_mw.process_generate("What is AI?", temperature=0.5)
            >>> print(cache_mw.misses)
            2  # Different params = new cache entry
        """
        key = self._cache_key(prompt, **kwargs)

        # Check cache
        cached = self.cache.pop(key, None)
        if cached is not None:
            response, timestamp = cached
            if not self._is_expired(timestamp):
                self.hits += 1
                # True LRU: move most-recently-used key to the end.
                self.cache[key] = (response, timestamp)
                return response

        # Cache miss - generate and cache
        self.misses += 1
        if self.next_middleware:
            response = self.next_middleware.process_generate(prompt, **kwargs)
        elif self.model:
            response = self.model.generate(prompt, **kwargs)
        else:
            raise ModelError("No model available in pipeline")

        # Store in cache with LRU eviction
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = (response, time.time())
        return response

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously process a generate request with cache support.

        The async counterpart to process_generate. Uses an asyncio.Lock
        for thread-safe access to the cache when checking and storing
        entries.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters.

        Returns
        -------
        str
            The generated (or cached) response.

        Raises
        ------
        ModelError
            If generation fails and no cached response is available.

        Examples
        --------
        Async caching:

            >>> async def example():
            ...     cache_mw = CacheMiddleware()
            ...     cache_mw.model = async_model
            ...
            ...     # Concurrent requests for same prompt
            ...     tasks = [
            ...         cache_mw.aprocess_generate("Question?")
            ...         for _ in range(5)
            ...     ]
            ...     results = await asyncio.gather(*tasks)
            ...
            ...     # First one was a miss, rest were hits
            ...     print(f"Hits: {cache_mw.hits}, Misses: {cache_mw.misses}")
            ...     return results
        """
        key = self._cache_key(prompt, **kwargs)

        # Check cache (async-safe since cache is just a dict read)
        async with self._get_lock():
            cached = self.cache.pop(key, None)
            if cached is not None:
                response, timestamp = cached
                if not self._is_expired(timestamp):
                    self.hits += 1
                    self.cache[key] = (response, timestamp)
                    return response

            self.misses += 1

        # Cache miss - generate and cache
        if self.next_middleware:
            response = await self.next_middleware.aprocess_generate(prompt, **kwargs)
        elif self.model:
            if isinstance(self.model, AsyncModelProtocol):
                response = await self.model.agenerate(prompt, **kwargs)
            else:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.model.generate(prompt, **kwargs)
                )
        else:
            raise ModelError("No model available in pipeline")

        # Store in cache with LRU eviction
        async with self._get_lock():
            if len(self.cache) >= self.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = (response, time.time())

        return response

    def _get_lock(self) -> asyncio.Lock:
        """Get or create an asyncio.Lock for thread-safe cache access.

        Lazily creates the lock on first access. This is necessary because
        asyncio.Lock cannot be created outside of an async context in some
        Python versions.

        Returns
        -------
        asyncio.Lock
            The lock instance for synchronizing cache access.

        Examples
        --------
        Internal usage (typically not called directly):

            >>> async def safe_operation():
            ...     cache_mw = CacheMiddleware()
            ...     async with cache_mw._get_lock():
            ...         # Thread-safe cache operations
            ...         pass
        """
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def hit_rate(self) -> float:
        """Calculate the cache hit rate.

        Computes the ratio of cache hits to total requests (hits + misses).

        Returns
        -------
        float
            The hit rate as a value between 0.0 and 1.0. Returns 0.0 if
            no requests have been processed yet.

        Examples
        --------
        Monitoring hit rate:

            >>> cache_mw = CacheMiddleware()
            >>> # After processing requests...
            >>> print(f"Hit rate: {cache_mw.hit_rate:.2%}")
            Hit rate: 75.00%

            >>> # With no requests yet
            >>> empty_cache = CacheMiddleware()
            >>> print(empty_cache.hit_rate)
            0.0
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class RateLimitMiddleware(Middleware):
    """Middleware for rate limiting model requests using a token bucket algorithm.

    Enforces a maximum request rate to prevent exceeding API rate limits or
    to control costs. The token bucket algorithm allows for configurable burst
    behavior: requests can be made at a higher rate temporarily if tokens have
    accumulated during idle periods.

    The algorithm works as follows:
    1. Tokens are added to the bucket at a steady rate (requests_per_minute / 60)
    2. Each request consumes one token
    3. If no tokens are available, the request waits until one becomes available
    4. The bucket has a maximum capacity (burst_size) to limit bursts

    Parameters
    ----------
    requests_per_minute : int, default=60
        Maximum sustained request rate. Internally converted to requests
        per second for finer-grained control.
    burst_size : int, optional
        Maximum number of requests that can be made in a burst when tokens
        have accumulated. Defaults to requests_per_minute (allowing a full
        minute's worth of requests to be made immediately after idle time).

    Attributes
    ----------
    rate : float
        The token refill rate in tokens per second.
    burst_size : int
        The maximum number of tokens the bucket can hold.
    tokens : float
        Current number of available tokens.
    last_update : float
        Timestamp of the last token update.

    Examples
    --------
    Basic rate limiting:

        >>> from insideLLMs.runtime.pipeline import RateLimitMiddleware, ModelPipeline
        >>>
        >>> # Allow 60 requests per minute (1 per second)
        >>> rate_mw = RateLimitMiddleware(requests_per_minute=60)
        >>> pipeline = ModelPipeline(model, middlewares=[rate_mw])
        >>>
        >>> # This will process at ~1 request per second
        >>> for prompt in prompts:
        ...     response = pipeline.generate(prompt)  # May wait if too fast

    High burst allowance:

        >>> # Allow 30/minute but permit bursts of up to 10 requests
        >>> rate_mw = RateLimitMiddleware(
        ...     requests_per_minute=30,
        ...     burst_size=10
        ... )
        >>>
        >>> # First 10 requests execute immediately
        >>> # Subsequent requests are spaced at 2 seconds apart

    Strict rate limiting (no bursts):

        >>> # Exactly 1 request per second, no accumulation
        >>> rate_mw = RateLimitMiddleware(
        ...     requests_per_minute=60,
        ...     burst_size=1
        ... )

    Async rate limiting with concurrent requests:

        >>> async def rate_limited_batch():
        ...     rate_mw = RateLimitMiddleware(requests_per_minute=120)
        ...     pipeline = AsyncModelPipeline(
        ...         model,
        ...         middlewares=[rate_mw]
        ...     )
        ...
        ...     # These will be rate limited even with concurrency
        ...     results = await pipeline.abatch_generate(
        ...         prompts,
        ...         max_concurrency=10
        ...     )
        ...     return results

    Combining with other middleware:

        >>> # Place rate limiting before retry to count retries correctly
        >>> pipeline = ModelPipeline(
        ...     model,
        ...     middlewares=[
        ...         CacheMiddleware(),  # Cache hits bypass rate limit
        ...         RateLimitMiddleware(requests_per_minute=60),
        ...         RetryMiddleware(max_retries=3),  # Retries consume tokens
        ...     ],
        ... )

    Notes
    -----
    - The token bucket algorithm provides smooth rate limiting with controlled
      burst behavior, unlike simple time-window approaches.
    - Async operations use an asyncio.Lock to ensure thread-safe token access
      across concurrent requests.
    - The middleware waits (sleeps) when rate limited rather than raising an
      exception, ensuring requests eventually complete.
    - Place CacheMiddleware before RateLimitMiddleware so cache hits don't
      consume rate limit tokens.

    See Also
    --------
    Middleware : The base middleware class
    RetryMiddleware : Often used together with rate limiting
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: Optional[int] = None,
    ) -> None:
        """Initialize the rate limiter with rate and burst configuration.

        Args
        ----
        requests_per_minute : int, default=60
            Maximum sustained request rate. This is the average rate that
            can be maintained over time.
        burst_size : int, optional
            Maximum burst capacity. If not specified, defaults to
            requests_per_minute, allowing accumulated tokens from idle
            time to be used in a burst.

        Examples
        --------
        Standard rate limiting:

            >>> rate_mw = RateLimitMiddleware(requests_per_minute=60)

        Strict rate limiting (no bursts):

            >>> rate_mw = RateLimitMiddleware(
            ...     requests_per_minute=30,
            ...     burst_size=1
            ... )

        High burst tolerance:

            >>> rate_mw = RateLimitMiddleware(
            ...     requests_per_minute=10,
            ...     burst_size=50
            ... )
        """
        super().__init__()
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0")
        if burst_size is not None and burst_size <= 0:
            raise ValueError("burst_size must be > 0")
        self.rate = requests_per_minute / 60.0  # requests per second
        self.burst_size = burst_size if burst_size is not None else requests_per_minute
        self.tokens = float(self.burst_size)
        self.last_update = time.time()

    def _acquire_token(self) -> None:
        """Acquire a token from the bucket, waiting if necessary.

        Updates the token count based on elapsed time, then either
        consumes a token immediately (if available) or waits until
        a token becomes available.

        This method modifies the token state and may block (sleep)
        if rate limited.

        Examples
        --------
        Internal usage (typically not called directly):

            >>> rate_mw = RateLimitMiddleware(requests_per_minute=60)
            >>> rate_mw._acquire_token()  # May wait if no tokens
        """
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
        self.last_update = now

        if self.tokens < 1.0:
            # Wait for token to be available
            wait_time = (1.0 - self.tokens) / self.rate
            time.sleep(wait_time)
            self.tokens = 1.0
            self.last_update = time.time()

        self.tokens -= 1.0

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Process a generate request with rate limiting.

        Acquires a rate limit token (waiting if necessary) before
        delegating to the next middleware or model.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters.

        Returns
        -------
        str
            The generated response.

        Raises
        ------
        ModelError
            If generation fails.

        Examples
        --------
        Rate-limited generation:

            >>> rate_mw = RateLimitMiddleware(requests_per_minute=60)
            >>> rate_mw.model = model
            >>>
            >>> # First request: immediate (tokens available)
            >>> r1 = rate_mw.process_generate("Q1")
            >>>
            >>> # Rapid follow-up: may wait for token
            >>> r2 = rate_mw.process_generate("Q2")
        """
        self._acquire_token()

        if self.next_middleware:
            return self.next_middleware.process_generate(prompt, **kwargs)
        if self.model:
            return self.model.generate(prompt, **kwargs)
        raise ModelError("No model available in pipeline")

    async def _aacquire_token(self) -> None:
        """Asynchronously acquire a token, waiting if necessary.

        The async counterpart to _acquire_token. Uses asyncio.sleep for
        non-blocking waits and an asyncio.Lock for thread-safe token
        access across concurrent requests.

        Examples
        --------
        Internal async usage:

            >>> async def example():
            ...     rate_mw = RateLimitMiddleware(requests_per_minute=60)
            ...     await rate_mw._aacquire_token()
        """
        async with self._get_lock():
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 1.0
                self.last_update = time.time()

            self.tokens -= 1.0

    def _get_lock(self) -> asyncio.Lock:
        """Get or create an asyncio.Lock for thread-safe token access.

        Lazily creates the lock on first access to avoid issues with
        asyncio.Lock creation outside of async contexts.

        Returns
        -------
        asyncio.Lock
            The lock instance for synchronizing token access.
        """
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()
        return self._lock

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously process a generate request with rate limiting.

        The async counterpart to process_generate. Acquires a token using
        async-safe methods before delegating to the next middleware or model.

        Args
        ----
        prompt : str
            The input prompt for text generation.
        **kwargs : Any
            Additional generation parameters.

        Returns
        -------
        str
            The generated response.

        Raises
        ------
        ModelError
            If generation fails.

        Examples
        --------
        Async rate-limited generation:

            >>> async def example():
            ...     rate_mw = RateLimitMiddleware(requests_per_minute=120)
            ...     rate_mw.model = async_model
            ...
            ...     # Concurrent but rate-limited
            ...     tasks = [
            ...         rate_mw.aprocess_generate(f"Question {i}")
            ...         for i in range(10)
            ...     ]
            ...     results = await asyncio.gather(*tasks)
            ...     return results
        """
        await self._aacquire_token()

        if self.next_middleware:
            return await self.next_middleware.aprocess_generate(prompt, **kwargs)
        if self.model:
            if isinstance(self.model, AsyncModelProtocol):
                return await self.model.agenerate(prompt, **kwargs)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self.model.generate(prompt, **kwargs))
        raise ModelError("No model available in pipeline")


class RetryMiddleware(Middleware):
    """Middleware for retrying failed requests with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exponential_base: Base for exponential backoff.
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        """Initialize the retry middleware."""
        super().__init__()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retry_count = 0
        self.total_retries = 0

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        import random

        delay = min(self.initial_delay * (self.exponential_base**attempt), self.max_delay)
        # Add jitter
        jitter = random.uniform(0, delay * 0.1)
        return delay + jitter

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Retry on failure with exponential backoff."""
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                if self.next_middleware:
                    return self.next_middleware.process_generate(prompt, **kwargs)
                if self.model:
                    return self.model.generate(prompt, **kwargs)
                raise ModelError("No model available in pipeline")
            except (RateLimitError, TimeoutError, ModelError) as e:
                last_error = e

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.retry_count += 1
                    self.total_retries += 1
                    time.sleep(delay)
                    continue

                # Max retries exceeded
                break

        # All retries failed
        raise ModelError(
            f"Failed after {self.max_retries} retries", details={"original_error": str(last_error)}
        )

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Async retry on failure with exponential backoff."""
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                if self.next_middleware:
                    return await self.next_middleware.aprocess_generate(prompt, **kwargs)
                if self.model:
                    if isinstance(self.model, AsyncModelProtocol):
                        return await self.model.agenerate(prompt, **kwargs)
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        None, lambda: self.model.generate(prompt, **kwargs)
                    )
                raise ModelError("No model available in pipeline")
            except (RateLimitError, TimeoutError, ModelError) as e:
                last_error = e

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.retry_count += 1
                    self.total_retries += 1
                    await asyncio.sleep(delay)
                    continue

                # Max retries exceeded
                break

        # All retries failed
        raise ModelError(
            f"Failed after {self.max_retries} retries", details={"original_error": str(last_error)}
        )


class CostTrackingMiddleware(Middleware):
    """Middleware for tracking API costs and token usage.

    Tracks requests, tokens, and estimated costs for model usage.
    """

    # Approximate costs per 1K tokens (as of 2024, subject to change)
    COST_PER_1K_TOKENS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    def __init__(self) -> None:
        """Initialize cost tracking."""
        super().__init__()
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.estimated_cost = 0.0

    def _estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on model and token counts."""
        # Try to match model name to known pricing
        model_key = None
        for key in self.COST_PER_1K_TOKENS:
            if key in model_name.lower():
                model_key = key
                break

        if not model_key:
            return 0.0  # Unknown model

        pricing = self.COST_PER_1K_TOKENS[model_key]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Track costs for generation."""
        # Generate response
        if self.next_middleware:
            response = self.next_middleware.process_generate(prompt, **kwargs)
        elif self.model:
            response = self.model.generate(prompt, **kwargs)
        else:
            raise ModelError("No model available in pipeline")

        # Track usage
        self.total_requests += 1

        # Rough token estimation (4 chars ≈ 1 token)
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Estimate cost
        if self.model:
            model_name = getattr(self.model, "model_id", "unknown")
            cost = self._estimate_cost(model_name, input_tokens, output_tokens)
            self.estimated_cost += cost

        return response

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Async track costs for generation."""
        # Generate response
        if self.next_middleware:
            response = await self.next_middleware.aprocess_generate(prompt, **kwargs)
        elif self.model:
            if isinstance(self.model, AsyncModelProtocol):
                response = await self.model.agenerate(prompt, **kwargs)
            else:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.model.generate(prompt, **kwargs)
                )
        else:
            raise ModelError("No model available in pipeline")

        # Track usage (thread-safe via lock)
        async with self._get_lock():
            self.total_requests += 1

            # Rough token estimation (4 chars ≈ 1 token)
            input_tokens = len(prompt) // 4
            output_tokens = len(response) // 4

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            # Estimate cost
            if self.model:
                model_name = getattr(self.model, "model_id", "unknown")
                cost = self._estimate_cost(model_name, input_tokens, output_tokens)
                self.estimated_cost += cost

        return response

    def _get_lock(self) -> asyncio.Lock:
        """Get or create an async lock for thread-safe tracking."""
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()
        return self._lock

    def get_stats(self) -> dict[str, Any]:
        """Get cost tracking statistics."""
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost, 4),
        }


class ModelPipeline(Model):
    """A model wrapper that composes multiple middleware for enhanced capabilities.

    The pipeline executes middleware in order for requests and reverse order
    for responses, following the chain of responsibility pattern.

    Args:
        base_model: The underlying model to wrap.
        middlewares: List of middleware to apply (in order).
        name: Optional name for the pipeline (defaults to base model name).

    Example:
        >>> pipeline = ModelPipeline(
        ...     OpenAIModel("gpt-4"),
        ...     middlewares=[
        ...         CacheMiddleware(cache_size=500),
        ...         RateLimitMiddleware(requests_per_minute=60),
        ...         RetryMiddleware(max_retries=3),
        ...         CostTrackingMiddleware(),
        ...     ],
        ... )
        >>> response = pipeline.generate("Hello, world!")
    """

    def __init__(
        self,
        base_model: ModelProtocol,
        middlewares: Optional[list[Middleware]] = None,
        name: Optional[str] = None,
    ):
        """Initialize the pipeline."""
        self.base_model = base_model
        self.middlewares = middlewares or []

        # Chain middleware together
        prev: Optional[Middleware] = None
        for middleware in self.middlewares:
            middleware.model = base_model
            if prev:
                prev.next_middleware = middleware
            prev = middleware

        # Initialize base Model
        pipeline_name = name or f"{base_model.name}_pipeline"
        model_id = getattr(base_model, "model_id", base_model.name)
        super().__init__(name=pipeline_name, model_id=model_id)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate through the middleware pipeline."""
        if self.middlewares:
            return self.middlewares[0].process_generate(prompt, **kwargs)
        return self.base_model.generate(prompt, **kwargs)

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Chat through the middleware pipeline."""
        if self.middlewares:
            return self.middlewares[0].process_chat(messages, **kwargs)
        if hasattr(self.base_model, "chat"):
            return self.base_model.chat(messages, **kwargs)
        raise ModelError("Base model does not support chat")

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream through the middleware pipeline."""
        if self.middlewares:
            yield from self.middlewares[0].process_stream(prompt, **kwargs)
        elif hasattr(self.base_model, "stream"):
            yield from self.base_model.stream(prompt, **kwargs)
        else:
            raise ModelError("Base model does not support streaming")

    # Async methods

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate through the middleware pipeline."""
        if self.middlewares:
            return await self.middlewares[0].aprocess_generate(prompt, **kwargs)
        if isinstance(self.base_model, AsyncModelProtocol):
            return await self.base_model.agenerate(prompt, **kwargs)
        # Fall back to executor for sync model
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.base_model.generate(prompt, **kwargs))

    async def achat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Asynchronously chat through the middleware pipeline."""
        if self.middlewares:
            return await self.middlewares[0].aprocess_chat(messages, **kwargs)
        if hasattr(self.base_model, "achat"):
            return await self.base_model.achat(messages, **kwargs)
        if hasattr(self.base_model, "chat"):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self.base_model.chat(messages, **kwargs)
            )
        raise ModelError("Base model does not support chat")

    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously stream through the middleware pipeline."""
        if self.middlewares:
            async for chunk in self.middlewares[0].aprocess_stream(prompt, **kwargs):
                yield chunk
        elif hasattr(self.base_model, "astream"):
            async for chunk in self.base_model.astream(prompt, **kwargs):
                yield chunk
        elif hasattr(self.base_model, "stream"):
            loop = asyncio.get_running_loop()
            chunks = await loop.run_in_executor(
                None, lambda: list(self.base_model.stream(prompt, **kwargs))
            )
            for chunk in chunks:
                yield chunk
        else:
            raise ModelError("Base model does not support streaming")

    async def abatch_generate(
        self,
        prompts: list[str],
        *,
        max_concurrency: int = 10,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """Generate responses for multiple prompts concurrently.

        Args:
            prompts: List of prompts to process.
            max_concurrency: Maximum concurrent requests.
            return_exceptions: If True, include exceptions in results as strings.
            **kwargs: Additional arguments for generation.

        Returns:
            List of responses in same order as prompts.

        Example:
            >>> results = await pipeline.abatch_generate(
            ...     ["Q1", "Q2", "Q3"],
            ...     max_concurrency=5
            ... )
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[Any] = [None] * len(prompts)

        async def process(index: int, prompt: str) -> None:
            async with semaphore:
                try:
                    results[index] = await self.agenerate(prompt, **kwargs)
                except Exception as e:
                    if return_exceptions:
                        results[index] = f"Error: {e}"
                    else:
                        raise

        tasks = [asyncio.create_task(process(i, prompt)) for i, prompt in enumerate(prompts)]
        await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        return results

    def info(self) -> dict[str, Any]:
        """Get pipeline information including middleware stats."""
        from dataclasses import asdict, is_dataclass

        base_info = self.base_model.info()

        # Convert ModelInfo dataclass to dict if needed
        if is_dataclass(base_info):
            base_info_dict = asdict(base_info)
        else:
            base_info_dict = base_info

        pipeline_info = {
            **base_info_dict,
            "pipeline": True,
            "middleware_count": len(self.middlewares),
            "middlewares": [type(m).__name__ for m in self.middlewares],
        }

        # Add middleware-specific stats
        for middleware in self.middlewares:
            if isinstance(middleware, CacheMiddleware):
                pipeline_info["cache_hit_rate"] = middleware.hit_rate
                pipeline_info["cache_hits"] = middleware.hits
                pipeline_info["cache_misses"] = middleware.misses
            elif isinstance(middleware, RetryMiddleware):
                pipeline_info["total_retries"] = middleware.total_retries
            elif isinstance(middleware, CostTrackingMiddleware):
                pipeline_info["cost_stats"] = middleware.get_stats()

        return pipeline_info


class AsyncModelPipeline(ModelPipeline):
    """Async-first model pipeline optimized for concurrent workloads.

    Extends ModelPipeline with additional async features like batch processing
    with progress tracking and advanced concurrency control.

    Example:
        >>> async def main():
        ...     pipeline = AsyncModelPipeline(
        ...         base_model,
        ...         middlewares=[CacheMiddleware(), RateLimitMiddleware()],
        ...     )
        ...     # Single request
        ...     response = await pipeline.agenerate("Hello")
        ...
        ...     # Batch processing with progress
        ...     async for result in pipeline.agenerate_stream_results(prompts):
        ...         print(f"Got result: {result}")
    """

    async def agenerate_with_callback(
        self,
        prompts: list[str],
        *,
        max_concurrency: int = 10,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_result: Optional[Callable[[int, str], None]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate with progress and result callbacks.

        Args:
            prompts: List of prompts to process.
            max_concurrency: Maximum concurrent requests.
            on_progress: Callback(completed, total) for progress updates.
            on_result: Callback(index, result) when each result completes.
            **kwargs: Additional arguments for generation.

        Returns:
            List of responses in order.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[Any] = [None] * len(prompts)
        completed = 0
        total = len(prompts)

        async def process(index: int, prompt: str) -> None:
            nonlocal completed
            async with semaphore:
                try:
                    result = await self.agenerate(prompt, **kwargs)
                    results[index] = result
                    if on_result:
                        on_result(index, result)
                except Exception as e:
                    results[index] = f"Error: {e}"
                    if on_result:
                        on_result(index, f"Error: {e}")
                finally:
                    completed += 1
                    if on_progress:
                        on_progress(completed, total)

        tasks = [asyncio.create_task(process(i, prompt)) for i, prompt in enumerate(prompts)]
        await asyncio.gather(*tasks)
        return results

    async def agenerate_stream_results(
        self,
        prompts: list[str],
        *,
        max_concurrency: int = 10,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[int, str]]:
        """Generate and yield results as they complete.

        Yields results in completion order (not input order).

        Args:
            prompts: List of prompts to process.
            max_concurrency: Maximum concurrent requests.
            **kwargs: Additional arguments for generation.

        Yields:
            Tuples of (index, result) as each completes.

        Example:
            >>> async for idx, result in pipeline.agenerate_stream_results(prompts):
            ...     print(f"Prompt {idx}: {result[:50]}...")
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()

        async def process(index: int, prompt: str) -> None:
            async with semaphore:
                try:
                    result = await self.agenerate(prompt, **kwargs)
                    await queue.put((index, result))
                except Exception as e:
                    await queue.put((index, f"Error: {e}"))

        tasks = [asyncio.create_task(process(i, prompt)) for i, prompt in enumerate(prompts)]

        # Yield results as they complete
        for _ in range(len(prompts)):
            result = await queue.get()
            yield result

        # Ensure all tasks complete
        await asyncio.gather(*tasks)

    async def amap(
        self,
        prompts: list[str],
        *,
        max_concurrency: int = 10,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> list[tuple[str, Optional[str], Optional[Exception]]]:
        """Map prompts to responses with detailed error handling.

        Args:
            prompts: List of prompts to process.
            max_concurrency: Maximum concurrent requests.
            timeout: Optional timeout per request in seconds.
            **kwargs: Additional arguments for generation.

        Returns:
            List of (prompt, response, error) tuples.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[tuple[str, Optional[str], Optional[Exception]]] = []

        async def process(prompt: str) -> tuple[str, Optional[str], Optional[Exception]]:
            async with semaphore:
                try:
                    if timeout:
                        result = await asyncio.wait_for(
                            self.agenerate(prompt, **kwargs), timeout=timeout
                        )
                    else:
                        result = await self.agenerate(prompt, **kwargs)
                    return (prompt, result, None)
                except Exception as e:
                    return (prompt, None, e)

        tasks = [asyncio.create_task(process(prompt)) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return list(results)
