"""Middleware base classes."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional

from insideLLMs.exceptions import ModelError
from insideLLMs.models.base import AsyncModelProtocol, ChatMessage, ModelProtocol


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
