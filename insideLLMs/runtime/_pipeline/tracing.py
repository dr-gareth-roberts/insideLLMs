"""Execution tracing middleware."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Optional

from insideLLMs.exceptions import ModelError
from insideLLMs.models.base import AsyncModelProtocol, ChatMessage
from insideLLMs.runtime._pipeline.middleware import PassthroughMiddleware

if TYPE_CHECKING:
    from insideLLMs.tracing import TraceRecorder


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
