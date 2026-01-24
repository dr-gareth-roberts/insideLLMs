"""Output streaming utilities for LLM responses.

This module provides tools for handling, processing, and analyzing
streaming output from language models, including:
- Stream buffer management
- Token-level processing
- Streaming metrics collection
- Content filtering during streaming
- Aggregation and transformation

The streaming utilities are designed for real-time processing of LLM outputs,
enabling applications to display partial results, apply content filters,
collect performance metrics, and analyze token-by-token generation patterns.

Examples
--------
Basic stream collection:

>>> from insideLLMs.streaming import StreamCollector, StreamEventType
>>> collector = StreamCollector()
>>> collector.start()
>>> for token in ["Hello", " ", "world", "!"]:
...     chunk = collector.collect_chunk(token)
...     print(f"Received: {chunk.content}")
>>> summary = collector.complete()
>>> print(f"Full content: {summary.full_content}")
>>> print(f"Tokens per second: {summary.metrics.tokens_per_second:.2f}")

Using a stream processor with filters:

>>> from insideLLMs.streaming import StreamProcessor, FilterAction
>>> processor = StreamProcessor()
>>> processor.add_pattern_filter(
...     "pii_filter",
...     r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",
...     FilterAction.REPLACE,
...     "[EMAIL REDACTED]"
... )
>>> processor.add_transformer(lambda x: x.strip())

Iterating over a stream with processing:

>>> from insideLLMs.streaming import StreamIterator
>>> source = iter(["chunk1", "chunk2", "chunk3"])
>>> stream_iter = StreamIterator(source)
>>> for chunk in stream_iter:
...     print(f"Processed: {chunk.content}")
>>> summary = stream_iter.get_summary()

Async streaming with collection:

>>> import asyncio
>>> from insideLLMs.streaming import AsyncStreamIterator
>>> async def example():
...     async def async_source():
...         for item in ["async", " ", "stream"]:
...             yield item
...     stream_iter = AsyncStreamIterator(async_source())
...     async for chunk in stream_iter:
...         print(chunk.content)
>>> # asyncio.run(example())

Classes
-------
StreamEventType
    Enum defining types of streaming events (TOKEN, CHUNK, START, END, etc.)
StreamState
    Enum defining stream states (IDLE, STREAMING, PAUSED, COMPLETED, ERROR)
FilterAction
    Enum defining filter actions (ALLOW, BLOCK, REPLACE, WARN)
StreamToken
    Dataclass representing a single token with metadata
StreamChunk
    Dataclass representing a chunk of streamed content
StreamMetrics
    Dataclass for collecting streaming performance metrics
FilterResult
    Dataclass representing the result of content filtering
StreamSummary
    Dataclass containing a summary of completed stream
StreamBuffer
    Buffer for managing and storing streamed content
StreamProcessor
    Process streaming content with transformations and filters
StreamAggregator
    Aggregate streaming content from multiple sources
StreamCollector
    Collect and analyze streaming output with metrics
StreamIterator
    Synchronous iterator over streaming content
AsyncStreamIterator
    Asynchronous iterator over streaming content
ContentDetector
    Detect specific content patterns in streams
StreamingWindowAnalyzer
    Analyze streaming content with sliding windows
StreamRateLimiter
    Rate limit streaming output with token bucket algorithm
"""

import re
import time
from collections import deque
from collections.abc import AsyncIterator, Generator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
)

T = TypeVar("T")


class StreamEventType(Enum):
    """Types of streaming events emitted during LLM response generation.

    StreamEventType categorizes the different kinds of events that can occur
    during streaming, enabling handlers to respond appropriately to each
    event type.

    Attributes
    ----------
    TOKEN : str
        A single token emission event.
    CHUNK : str
        A chunk of content (may contain multiple tokens).
    START : str
        Stream initialization event.
    END : str
        Stream completion event.
    ERROR : str
        Error occurred during streaming.
    METADATA : str
        Metadata update event (e.g., usage statistics).
    TOOL_CALL : str
        Tool/function call event in the stream.
    CONTENT_BLOCK : str
        Content block boundary event.

    Examples
    --------
    Handling different event types:

    >>> from insideLLMs.streaming import StreamEventType, StreamChunk
    >>> import time
    >>> chunk = StreamChunk(
    ...     content="Hello",
    ...     event_type=StreamEventType.TOKEN,
    ...     timestamp=time.time(),
    ...     index=0
    ... )
    >>> if chunk.event_type == StreamEventType.TOKEN:
    ...     print(f"Token received: {chunk.content}")

    Filtering by event type:

    >>> chunks = [
    ...     StreamChunk("", StreamEventType.START, time.time(), 0),
    ...     StreamChunk("Hello", StreamEventType.TOKEN, time.time(), 1),
    ...     StreamChunk("", StreamEventType.END, time.time(), 2),
    ... ]
    >>> tokens = [c for c in chunks if c.event_type == StreamEventType.TOKEN]
    >>> len(tokens)
    1

    Creating event type from string:

    >>> event_type = StreamEventType("token")
    >>> event_type == StreamEventType.TOKEN
    True

    Checking event type value:

    >>> StreamEventType.TOOL_CALL.value
    'tool_call'
    """

    TOKEN = "token"
    CHUNK = "chunk"
    START = "start"
    END = "end"
    ERROR = "error"
    METADATA = "metadata"
    TOOL_CALL = "tool_call"
    CONTENT_BLOCK = "content_block"


class StreamState(Enum):
    """State of a stream during its lifecycle.

    StreamState represents the current operational state of a streaming
    connection or collection process. The state transitions typically follow:
    IDLE -> STREAMING -> COMPLETED (or ERROR).

    Attributes
    ----------
    IDLE : str
        Stream is initialized but not yet started.
    STREAMING : str
        Stream is actively receiving content.
    PAUSED : str
        Stream is temporarily paused.
    COMPLETED : str
        Stream has finished successfully.
    ERROR : str
        Stream encountered an error.

    Examples
    --------
    Checking stream state:

    >>> from insideLLMs.streaming import StreamCollector, StreamState
    >>> collector = StreamCollector()
    >>> collector.state == StreamState.IDLE
    True

    State transitions during collection:

    >>> collector = StreamCollector()
    >>> collector.state
    <StreamState.IDLE: 'idle'>
    >>> collector.start()
    >>> collector.state
    <StreamState.STREAMING: 'streaming'>
    >>> summary = collector.complete()
    >>> collector.state
    <StreamState.COMPLETED: 'completed'>

    Using state in conditional logic:

    >>> collector = StreamCollector()
    >>> if collector.state == StreamState.IDLE:
    ...     collector.start()
    >>> collector.state == StreamState.STREAMING
    True

    State value access:

    >>> StreamState.STREAMING.value
    'streaming'
    """

    IDLE = "idle"
    STREAMING = "streaming"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class FilterAction(Enum):
    """Actions for content filtering during stream processing.

    FilterAction defines the possible responses when a content filter
    detects a pattern match. These actions determine how the StreamProcessor
    handles filtered content.

    Attributes
    ----------
    ALLOW : str
        Allow the content to pass through unchanged.
    BLOCK : str
        Block the content entirely (replace with empty string).
    REPLACE : str
        Replace matched content with a specified replacement string.
    WARN : str
        Allow content but log a warning about the match.

    Examples
    --------
    Setting up a blocking filter for sensitive content:

    >>> from insideLLMs.streaming import StreamProcessor, FilterAction
    >>> processor = StreamProcessor()
    >>> processor.add_pattern_filter(
    ...     "block_secrets",
    ...     r"(?i)password:\\s*\\S+",
    ...     FilterAction.BLOCK
    ... )

    Setting up a replacement filter for PII:

    >>> processor = StreamProcessor()
    >>> processor.add_pattern_filter(
    ...     "redact_ssn",
    ...     r"\\d{3}-\\d{2}-\\d{4}",
    ...     FilterAction.REPLACE,
    ...     "[SSN REDACTED]"
    ... )

    Setting up a warning filter for monitoring:

    >>> processor = StreamProcessor()
    >>> processor.add_pattern_filter(
    ...     "detect_profanity",
    ...     r"\\b(badword1|badword2)\\b",
    ...     FilterAction.WARN
    ... )

    Checking filter action value:

    >>> FilterAction.REPLACE.value
    'replace'
    """

    ALLOW = "allow"
    BLOCK = "block"
    REPLACE = "replace"
    WARN = "warn"


@dataclass
class StreamToken:
    """A single token from a stream with associated metadata.

    StreamToken represents the smallest unit of streamed content, typically
    corresponding to a single token from the language model's vocabulary.
    It includes timing information, optional probability data, and custom
    metadata for analysis and debugging.

    Attributes
    ----------
    text : str
        The text content of the token.
    index : int
        The position of this token in the stream (0-indexed).
    timestamp : float
        Unix timestamp when the token was received.
    logprob : Optional[float]
        Log probability of the token, if available from the model.
    token_id : Optional[int]
        The vocabulary ID of the token, if available.
    metadata : dict[str, Any]
        Additional custom metadata associated with the token.

    Examples
    --------
    Creating a basic token:

    >>> import time
    >>> from insideLLMs.streaming import StreamToken
    >>> token = StreamToken(
    ...     text="Hello",
    ...     index=0,
    ...     timestamp=time.time()
    ... )
    >>> token.text
    'Hello'

    Creating a token with log probability:

    >>> token = StreamToken(
    ...     text="world",
    ...     index=1,
    ...     timestamp=time.time(),
    ...     logprob=-0.5,
    ...     token_id=1234
    ... )
    >>> token.logprob
    -0.5

    Adding custom metadata:

    >>> token = StreamToken(
    ...     text="!",
    ...     index=2,
    ...     timestamp=time.time(),
    ...     metadata={"source": "gpt-4", "is_final": True}
    ... )
    >>> token.metadata["source"]
    'gpt-4'

    Converting to dictionary for serialization:

    >>> token = StreamToken("test", 0, 1234567890.0)
    >>> token_dict = token.to_dict()
    >>> "text" in token_dict and "timestamp" in token_dict
    True
    """

    text: str
    index: int
    timestamp: float
    logprob: Optional[float] = None
    token_id: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the token to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all token attributes with their values.
            Suitable for JSON serialization or logging.

        Examples
        --------
        >>> import time
        >>> token = StreamToken("Hello", 0, time.time(), logprob=-0.1)
        >>> d = token.to_dict()
        >>> d["text"]
        'Hello'
        >>> "logprob" in d
        True
        """
        return {
            "text": self.text,
            "index": self.index,
            "timestamp": self.timestamp,
            "logprob": self.logprob,
            "token_id": self.token_id,
            "metadata": self.metadata,
        }


@dataclass
class StreamChunk:
    """A chunk of streamed content that may contain multiple tokens.

    StreamChunk represents a larger unit of streamed content than a single
    token. Chunks are commonly used when the streaming API delivers content
    in batches or when grouping tokens for efficiency. Each chunk has an
    event type indicating its role in the stream.

    Attributes
    ----------
    content : str
        The text content of the chunk.
    event_type : StreamEventType
        The type of streaming event this chunk represents.
    timestamp : float
        Unix timestamp when the chunk was received.
    index : int
        The position of this chunk in the stream (0-indexed).
    tokens : list[StreamToken]
        Individual tokens that make up this chunk, if available.
    metadata : dict[str, Any]
        Additional custom metadata associated with the chunk.

    Examples
    --------
    Creating a simple content chunk:

    >>> import time
    >>> from insideLLMs.streaming import StreamChunk, StreamEventType
    >>> chunk = StreamChunk(
    ...     content="Hello world",
    ...     event_type=StreamEventType.CHUNK,
    ...     timestamp=time.time(),
    ...     index=0
    ... )
    >>> chunk.content
    'Hello world'

    Creating a chunk with individual tokens:

    >>> from insideLLMs.streaming import StreamToken
    >>> tokens = [
    ...     StreamToken("Hello", 0, time.time()),
    ...     StreamToken(" world", 1, time.time())
    ... ]
    >>> chunk = StreamChunk(
    ...     content="Hello world",
    ...     event_type=StreamEventType.CHUNK,
    ...     timestamp=time.time(),
    ...     index=0,
    ...     tokens=tokens
    ... )
    >>> len(chunk.tokens)
    2

    Creating event markers (start/end):

    >>> start_chunk = StreamChunk(
    ...     content="",
    ...     event_type=StreamEventType.START,
    ...     timestamp=time.time(),
    ...     index=0,
    ...     metadata={"model": "gpt-4", "stream_id": "abc123"}
    ... )
    >>> start_chunk.event_type == StreamEventType.START
    True

    Converting to dictionary:

    >>> chunk = StreamChunk("test", StreamEventType.TOKEN, 1234567890.0, 0)
    >>> d = chunk.to_dict()
    >>> d["event_type"]
    'token'
    """

    content: str
    event_type: StreamEventType
    timestamp: float
    index: int
    tokens: list[StreamToken] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the chunk to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all chunk attributes. The event_type is
            converted to its string value, and tokens are recursively
            converted to dictionaries.

        Examples
        --------
        >>> import time
        >>> chunk = StreamChunk("Hello", StreamEventType.TOKEN, time.time(), 0)
        >>> d = chunk.to_dict()
        >>> d["content"]
        'Hello'
        >>> isinstance(d["tokens"], list)
        True
        """
        return {
            "content": self.content,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "index": self.index,
            "tokens": [t.to_dict() for t in self.tokens],
            "metadata": self.metadata,
        }


@dataclass
class StreamMetrics:
    """Metrics collected during streaming for performance analysis.

    StreamMetrics tracks various performance indicators during stream
    collection, including timing, throughput, and error counts. These
    metrics are useful for monitoring LLM response performance and
    identifying potential issues.

    Attributes
    ----------
    start_time : float
        Unix timestamp when streaming started.
    end_time : Optional[float]
        Unix timestamp when streaming ended, or None if still in progress.
    total_tokens : int
        Total number of tokens received.
    total_chunks : int
        Total number of chunks received.
    total_characters : int
        Total number of characters received.
    inter_token_times : list[float]
        List of time intervals between consecutive tokens (in seconds).
    error_count : int
        Number of errors encountered during streaming.

    Examples
    --------
    Creating metrics for a stream:

    >>> import time
    >>> from insideLLMs.streaming import StreamMetrics
    >>> metrics = StreamMetrics(start_time=time.time())
    >>> metrics.total_tokens = 100
    >>> metrics.total_chunks = 10

    Calculating throughput:

    >>> metrics = StreamMetrics(start_time=1000.0)
    >>> metrics.end_time = 1010.0  # 10 seconds later
    >>> metrics.total_tokens = 500
    >>> metrics.tokens_per_second
    50.0

    Analyzing inter-token timing:

    >>> metrics = StreamMetrics(start_time=time.time())
    >>> metrics.inter_token_times = [0.02, 0.03, 0.025, 0.022]
    >>> round(metrics.mean_inter_token_time, 4)
    0.0242

    Converting to dictionary for logging:

    >>> metrics = StreamMetrics(start_time=1000.0)
    >>> metrics.end_time = 1005.0
    >>> metrics.total_tokens = 250
    >>> d = metrics.to_dict()
    >>> d["duration"]
    5.0
    """

    start_time: float
    end_time: Optional[float] = None
    total_tokens: int = 0
    total_chunks: int = 0
    total_characters: int = 0
    inter_token_times: list[float] = field(default_factory=list)
    error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all metrics including computed properties
            like duration, tokens_per_second, and mean_inter_token_time.

        Examples
        --------
        >>> metrics = StreamMetrics(start_time=1000.0)
        >>> metrics.end_time = 1010.0
        >>> metrics.total_tokens = 100
        >>> d = metrics.to_dict()
        >>> d["total_tokens"]
        100
        >>> d["duration"]
        10.0
        """
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_tokens": self.total_tokens,
            "total_chunks": self.total_chunks,
            "total_characters": self.total_characters,
            "duration": self.duration,
            "tokens_per_second": self.tokens_per_second,
            "mean_inter_token_time": self.mean_inter_token_time,
            "error_count": self.error_count,
        }

    @property
    def duration(self) -> float:
        """Total duration of the stream in seconds.

        If the stream is still in progress (end_time is None), returns
        the time elapsed since start_time.

        Returns
        -------
        float
            Duration in seconds.

        Examples
        --------
        >>> metrics = StreamMetrics(start_time=1000.0)
        >>> metrics.end_time = 1005.0
        >>> metrics.duration
        5.0
        """
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def tokens_per_second(self) -> float:
        """Calculate the token generation rate.

        Returns
        -------
        float
            Tokens per second. Returns 0.0 if duration is zero or negative.

        Examples
        --------
        >>> metrics = StreamMetrics(start_time=1000.0)
        >>> metrics.end_time = 1002.0
        >>> metrics.total_tokens = 100
        >>> metrics.tokens_per_second
        50.0
        """
        duration = self.duration
        if duration <= 0:
            return 0.0
        return self.total_tokens / duration

    @property
    def mean_inter_token_time(self) -> float:
        """Calculate the mean time between consecutive tokens.

        Returns
        -------
        float
            Mean inter-token time in seconds. Returns 0.0 if no timing
            data is available.

        Examples
        --------
        >>> metrics = StreamMetrics(start_time=1000.0)
        >>> metrics.inter_token_times = [0.01, 0.02, 0.03]
        >>> metrics.mean_inter_token_time
        0.02
        """
        if not self.inter_token_times:
            return 0.0
        return sum(self.inter_token_times) / len(self.inter_token_times)


@dataclass
class FilterResult:
    """Result of content filtering applied to streamed content.

    FilterResult captures the outcome of applying a content filter to
    a piece of text, including what action was taken, the original and
    filtered content, and diagnostic information about the filter rule.

    Attributes
    ----------
    action : FilterAction
        The action that was taken (ALLOW, BLOCK, REPLACE, or WARN).
    original : str
        The original content before filtering.
    filtered : str
        The content after filtering (may be same as original for ALLOW/WARN).
    reason : Optional[str]
        Human-readable explanation of why the action was taken.
    rule_triggered : Optional[str]
        Name of the filter rule that triggered this result.

    Examples
    --------
    Result when content is blocked:

    >>> from insideLLMs.streaming import FilterResult, FilterAction
    >>> result = FilterResult(
    ...     action=FilterAction.BLOCK,
    ...     original="secret password: abc123",
    ...     filtered="",
    ...     reason="Contains sensitive credentials",
    ...     rule_triggered="block_passwords"
    ... )
    >>> result.action == FilterAction.BLOCK
    True

    Result when content is replaced:

    >>> result = FilterResult(
    ...     action=FilterAction.REPLACE,
    ...     original="Contact: user@example.com",
    ...     filtered="Contact: [EMAIL REDACTED]",
    ...     reason="PII detected",
    ...     rule_triggered="email_filter"
    ... )
    >>> result.filtered
    'Contact: [EMAIL REDACTED]'

    Result for allowed content:

    >>> result = FilterResult(
    ...     action=FilterAction.ALLOW,
    ...     original="Hello world",
    ...     filtered="Hello world"
    ... )
    >>> result.original == result.filtered
    True

    Converting to dictionary for logging:

    >>> result = FilterResult(FilterAction.WARN, "test", "test", "Suspicious")
    >>> d = result.to_dict()
    >>> d["action"]
    'warn'
    """

    action: FilterAction
    original: str
    filtered: str
    reason: Optional[str] = None
    rule_triggered: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the filter result to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all filter result fields with the action
            converted to its string value.

        Examples
        --------
        >>> result = FilterResult(
        ...     FilterAction.REPLACE,
        ...     "original text",
        ...     "filtered text",
        ...     "Matched pattern",
        ...     "my_rule"
        ... )
        >>> d = result.to_dict()
        >>> d["action"]
        'replace'
        >>> d["rule_triggered"]
        'my_rule'
        """
        return {
            "action": self.action.value,
            "original": self.original,
            "filtered": self.filtered,
            "reason": self.reason,
            "rule_triggered": self.rule_triggered,
        }


@dataclass
class StreamSummary:
    """Summary of a completed stream with aggregated data and metrics.

    StreamSummary provides a comprehensive overview of a streaming session
    after completion, including the full content, performance metrics,
    counts, errors, and any filtering that occurred.

    Attributes
    ----------
    full_content : str
        The complete aggregated content from the stream.
    metrics : StreamMetrics
        Performance metrics collected during streaming.
    chunk_count : int
        Total number of chunks received.
    token_count : int
        Total number of tokens received.
    state : StreamState
        Final state of the stream (typically COMPLETED or ERROR).
    errors : list[str]
        List of error messages encountered during streaming.
    filtered_count : int
        Number of chunks or tokens that were filtered.
    metadata : dict[str, Any]
        Additional custom metadata about the stream.

    Examples
    --------
    Accessing summary after collection:

    >>> from insideLLMs.streaming import StreamCollector
    >>> collector = StreamCollector()
    >>> collector.start()
    >>> collector.collect_chunk("Hello ")
    StreamChunk(content='Hello ', ...)
    >>> collector.collect_chunk("world!")
    StreamChunk(content='world!', ...)
    >>> summary = collector.complete()
    >>> summary.full_content
    'Hello world!'

    Examining metrics from summary:

    >>> summary.metrics.total_chunks
    2
    >>> summary.chunk_count
    2

    Checking for errors:

    >>> if summary.errors:
    ...     print(f"Errors occurred: {summary.errors}")
    >>> summary.state
    <StreamState.COMPLETED: 'completed'>

    Converting to dictionary for serialization:

    >>> from insideLLMs.streaming import StreamMetrics, StreamState
    >>> import time
    >>> metrics = StreamMetrics(start_time=1000.0, end_time=1005.0)
    >>> summary = StreamSummary(
    ...     full_content="test content",
    ...     metrics=metrics,
    ...     chunk_count=5,
    ...     token_count=10,
    ...     state=StreamState.COMPLETED
    ... )
    >>> d = summary.to_dict()
    >>> d["chunk_count"]
    5
    """

    full_content: str
    metrics: StreamMetrics
    chunk_count: int
    token_count: int
    state: StreamState
    errors: list[str] = field(default_factory=list)
    filtered_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the summary to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all summary fields with nested objects
            (metrics, state) also converted to their dictionary/string forms.

        Examples
        --------
        >>> import time
        >>> metrics = StreamMetrics(start_time=1000.0, end_time=1010.0)
        >>> summary = StreamSummary(
        ...     full_content="Hello",
        ...     metrics=metrics,
        ...     chunk_count=1,
        ...     token_count=1,
        ...     state=StreamState.COMPLETED
        ... )
        >>> d = summary.to_dict()
        >>> d["state"]
        'completed'
        >>> "metrics" in d
        True
        """
        return {
            "full_content": self.full_content,
            "metrics": self.metrics.to_dict(),
            "chunk_count": self.chunk_count,
            "token_count": self.token_count,
            "state": self.state.value,
            "errors": self.errors,
            "filtered_count": self.filtered_count,
            "metadata": self.metadata,
        }


class StreamBuffer:
    """Buffer for managing and storing streamed content with automatic compaction.

    StreamBuffer provides efficient storage for streaming content with automatic
    memory management. When the buffer exceeds its maximum size, older content
    is automatically removed to make room for new content.

    Parameters
    ----------
    max_size : int, optional
        Maximum buffer size in characters. Default is 1,000,000 (1MB of text).

    Attributes
    ----------
    max_size : int
        Maximum allowed buffer size in characters.
    chunks : list[StreamChunk]
        List of chunks stored in the buffer.
    tokens : list[StreamToken]
        List of individual tokens stored in the buffer.

    Examples
    --------
    Basic buffer usage:

    >>> import time
    >>> from insideLLMs.streaming import StreamBuffer, StreamChunk, StreamEventType
    >>> buffer = StreamBuffer(max_size=1000)
    >>> chunk = StreamChunk("Hello ", StreamEventType.CHUNK, time.time(), 0)
    >>> buffer.add_chunk(chunk)
    >>> buffer.get_content()
    'Hello '

    Adding multiple chunks:

    >>> buffer = StreamBuffer()
    >>> for i, text in enumerate(["Hello", " ", "world", "!"]):
    ...     chunk = StreamChunk(text, StreamEventType.CHUNK, time.time(), i)
    ...     buffer.add_chunk(chunk)
    >>> buffer.get_content()
    'Hello world!'
    >>> buffer.chunk_count
    4

    Getting recent content:

    >>> buffer = StreamBuffer()
    >>> buffer.add_chunk(StreamChunk("Long text here", StreamEventType.CHUNK, time.time(), 0))
    >>> buffer.get_recent(4)
    'here'

    Automatic compaction when buffer is full:

    >>> buffer = StreamBuffer(max_size=20)
    >>> buffer.add_chunk(StreamChunk("A" * 15, StreamEventType.CHUNK, time.time(), 0))
    >>> buffer.add_chunk(StreamChunk("B" * 10, StreamEventType.CHUNK, time.time(), 1))
    >>> buffer.size <= 20  # Compacted to fit
    True
    """

    def __init__(self, max_size: int = 1000000):
        """Initialize the stream buffer.

        Parameters
        ----------
        max_size : int, optional
            Maximum buffer size in characters. Default is 1,000,000.

        Examples
        --------
        >>> buffer = StreamBuffer()  # Default 1MB
        >>> buffer.max_size
        1000000
        >>> small_buffer = StreamBuffer(max_size=100)
        >>> small_buffer.max_size
        100
        """
        self.max_size = max_size
        self.chunks: list[StreamChunk] = []
        self.tokens: list[StreamToken] = []
        self._content_buffer: list[str] = []
        self._total_chars = 0

    def add_chunk(self, chunk: StreamChunk) -> None:
        """Add a chunk to the buffer, compacting if necessary.

        Parameters
        ----------
        chunk : StreamChunk
            The chunk to add to the buffer.

        Examples
        --------
        >>> import time
        >>> buffer = StreamBuffer()
        >>> chunk = StreamChunk("test", StreamEventType.CHUNK, time.time(), 0)
        >>> buffer.add_chunk(chunk)
        >>> buffer.chunk_count
        1
        """
        if self._total_chars + len(chunk.content) > self.max_size:
            # Remove old chunks to make room
            self._compact()

        self.chunks.append(chunk)
        self._content_buffer.append(chunk.content)
        self._total_chars += len(chunk.content)

        for token in chunk.tokens:
            self.tokens.append(token)

    def add_token(self, token: StreamToken) -> None:
        """Add a single token to the buffer, compacting if necessary.

        Parameters
        ----------
        token : StreamToken
            The token to add to the buffer.

        Examples
        --------
        >>> import time
        >>> buffer = StreamBuffer()
        >>> token = StreamToken("Hello", 0, time.time())
        >>> buffer.add_token(token)
        >>> buffer.token_count
        1
        >>> buffer.get_content()
        'Hello'
        """
        if self._total_chars + len(token.text) > self.max_size:
            self._compact()

        self.tokens.append(token)
        self._content_buffer.append(token.text)
        self._total_chars += len(token.text)

    def _compact(self) -> None:
        """Remove old content to stay within size limit.

        Removes approximately the first half of chunks to make room for
        new content while preserving recent data.
        """
        # Remove first half of chunks
        mid = len(self.chunks) // 2
        removed_chars = sum(len(c.content) for c in self.chunks[:mid])
        self.chunks = self.chunks[mid:]
        self._total_chars -= removed_chars
        self._rebuild_content_buffer()

    def _rebuild_content_buffer(self) -> None:
        """Rebuild content buffer from chunks after compaction."""
        self._content_buffer = [c.content for c in self.chunks]

    def get_content(self) -> str:
        """Get the full buffered content as a single string.

        Returns
        -------
        str
            All buffered content concatenated together.

        Examples
        --------
        >>> import time
        >>> buffer = StreamBuffer()
        >>> buffer.add_chunk(StreamChunk("Hello", StreamEventType.CHUNK, time.time(), 0))
        >>> buffer.add_chunk(StreamChunk(" world", StreamEventType.CHUNK, time.time(), 1))
        >>> buffer.get_content()
        'Hello world'
        """
        return "".join(self._content_buffer)

    def get_recent(self, n_chars: int) -> str:
        """Get the most recent n characters from the buffer.

        Parameters
        ----------
        n_chars : int
            Number of characters to retrieve from the end.

        Returns
        -------
        str
            The last n_chars characters, or all content if less available.

        Examples
        --------
        >>> import time
        >>> buffer = StreamBuffer()
        >>> buffer.add_chunk(StreamChunk("Hello world", StreamEventType.CHUNK, time.time(), 0))
        >>> buffer.get_recent(5)
        'world'
        >>> buffer.get_recent(100)  # More than available
        'Hello world'
        """
        content = self.get_content()
        return content[-n_chars:] if len(content) > n_chars else content

    def clear(self) -> None:
        """Clear all content from the buffer.

        Examples
        --------
        >>> import time
        >>> buffer = StreamBuffer()
        >>> buffer.add_chunk(StreamChunk("test", StreamEventType.CHUNK, time.time(), 0))
        >>> buffer.size > 0
        True
        >>> buffer.clear()
        >>> buffer.size
        0
        >>> buffer.chunk_count
        0
        """
        self.chunks.clear()
        self.tokens.clear()
        self._content_buffer.clear()
        self._total_chars = 0

    @property
    def size(self) -> int:
        """Current buffer size in characters.

        Returns
        -------
        int
            Total number of characters currently in the buffer.

        Examples
        --------
        >>> import time
        >>> buffer = StreamBuffer()
        >>> buffer.add_chunk(StreamChunk("Hello", StreamEventType.CHUNK, time.time(), 0))
        >>> buffer.size
        5
        """
        return self._total_chars

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently in the buffer.

        Returns
        -------
        int
            Count of chunks stored.

        Examples
        --------
        >>> import time
        >>> buffer = StreamBuffer()
        >>> buffer.add_chunk(StreamChunk("a", StreamEventType.CHUNK, time.time(), 0))
        >>> buffer.add_chunk(StreamChunk("b", StreamEventType.CHUNK, time.time(), 1))
        >>> buffer.chunk_count
        2
        """
        return len(self.chunks)

    @property
    def token_count(self) -> int:
        """Number of tokens currently in the buffer.

        Returns
        -------
        int
            Count of tokens stored.

        Examples
        --------
        >>> import time
        >>> buffer = StreamBuffer()
        >>> buffer.add_token(StreamToken("a", 0, time.time()))
        >>> buffer.add_token(StreamToken("b", 1, time.time()))
        >>> buffer.token_count
        2
        """
        return len(self.tokens)


class StreamProcessor:
    """Process streaming content with transformations, filters, and callbacks.

    StreamProcessor provides a pipeline for processing streamed content,
    allowing you to apply transformations (modify content), filters (block
    or replace content based on patterns), and callbacks (side effects like
    logging or display).

    Processing order:
    1. Transformers (in order added)
    2. Pattern filters (in order added)
    3. Custom filter functions (in order added)
    4. Callbacks (in order added)

    Attributes
    ----------
    transformers : list[Callable[[str], str]]
        List of transformer functions that modify content.
    filters : list[Callable[[str], FilterResult]]
        List of custom filter functions.
    callbacks : list[Callable[[StreamChunk], None]]
        List of callback functions called after processing.
    filter_patterns : dict[str, tuple[str, FilterAction, str]]
        Dictionary of named pattern filters.

    Examples
    --------
    Basic processor with transformer:

    >>> from insideLLMs.streaming import StreamProcessor, StreamChunk, StreamEventType
    >>> import time
    >>> processor = StreamProcessor()
    >>> processor.add_transformer(str.upper)
    >>> chunk = StreamChunk("hello", StreamEventType.CHUNK, time.time(), 0)
    >>> processed, _ = processor.process_chunk(chunk)
    >>> processed.content
    'HELLO'

    Adding a pattern filter:

    >>> processor = StreamProcessor()
    >>> processor.add_pattern_filter(
    ...     "redact_email",
    ...     r"[\\w.-]+@[\\w.-]+",
    ...     FilterAction.REPLACE,
    ...     "[REDACTED]"
    ... )
    >>> chunk = StreamChunk("Contact: test@example.com", StreamEventType.CHUNK, time.time(), 0)
    >>> processed, results = processor.process_chunk(chunk)
    >>> processed.content
    'Contact: [REDACTED]'

    Using callbacks for logging:

    >>> logs = []
    >>> processor = StreamProcessor()
    >>> processor.add_callback(lambda c: logs.append(c.content))
    >>> chunk = StreamChunk("test", StreamEventType.CHUNK, time.time(), 0)
    >>> _ = processor.process_chunk(chunk)
    >>> logs
    ['test']

    Chaining multiple transformers:

    >>> processor = StreamProcessor()
    >>> processor.add_transformer(str.strip)
    >>> processor.add_transformer(str.lower)
    >>> chunk = StreamChunk("  HELLO WORLD  ", StreamEventType.CHUNK, time.time(), 0)
    >>> processed, _ = processor.process_chunk(chunk)
    >>> processed.content
    'hello world'
    """

    def __init__(self):
        """Initialize an empty stream processor.

        Examples
        --------
        >>> processor = StreamProcessor()
        >>> len(processor.transformers)
        0
        >>> len(processor.filters)
        0
        """
        self.transformers: list[Callable[[str], str]] = []
        self.filters: list[Callable[[str], FilterResult]] = []
        self.callbacks: list[Callable[[StreamChunk], None]] = []
        self.filter_patterns: dict[str, tuple[str, FilterAction, str]] = {}

    def add_transformer(self, transformer: Callable[[str], str]) -> None:
        """Add a content transformer function.

        Transformers are applied in order and receive the content string,
        returning the transformed string.

        Parameters
        ----------
        transformer : Callable[[str], str]
            Function that takes a string and returns a transformed string.

        Examples
        --------
        >>> processor = StreamProcessor()
        >>> processor.add_transformer(str.upper)
        >>> processor.add_transformer(lambda s: s.replace("A", "@"))
        >>> len(processor.transformers)
        2
        """
        self.transformers.append(transformer)

    def add_filter(self, filter_fn: Callable[[str], FilterResult]) -> None:
        """Add a custom content filter function.

        Custom filters receive content and return a FilterResult indicating
        what action to take.

        Parameters
        ----------
        filter_fn : Callable[[str], FilterResult]
            Function that takes content and returns a FilterResult.

        Examples
        --------
        >>> def my_filter(content):
        ...     if "bad" in content:
        ...         return FilterResult(FilterAction.BLOCK, content, "", "Contains 'bad'")
        ...     return FilterResult(FilterAction.ALLOW, content, content)
        >>> processor = StreamProcessor()
        >>> processor.add_filter(my_filter)
        >>> len(processor.filters)
        1
        """
        self.filters.append(filter_fn)

    def add_callback(self, callback: Callable[[StreamChunk], None]) -> None:
        """Add a callback function called after processing each chunk.

        Callbacks are useful for side effects like logging, display updates,
        or metrics collection.

        Parameters
        ----------
        callback : Callable[[StreamChunk], None]
            Function that receives the processed chunk.

        Examples
        --------
        >>> processed_chunks = []
        >>> processor = StreamProcessor()
        >>> processor.add_callback(lambda c: processed_chunks.append(c))
        >>> len(processor.callbacks)
        1
        """
        self.callbacks.append(callback)

    def add_pattern_filter(
        self,
        name: str,
        pattern: str,
        action: FilterAction,
        replacement: str = "",
    ) -> None:
        """Add a regex-based pattern filter.

        Pattern filters are applied before custom filter functions and
        can block, replace, or warn about content matching the pattern.

        Parameters
        ----------
        name : str
            Unique name for this filter (used in FilterResult).
        pattern : str
            Regular expression pattern to match.
        action : FilterAction
            Action to take when pattern matches.
        replacement : str, optional
            Replacement string when action is REPLACE. Default is "".

        Examples
        --------
        Block content with passwords:

        >>> processor = StreamProcessor()
        >>> processor.add_pattern_filter(
        ...     "block_passwords",
        ...     r"password:\\s*\\S+",
        ...     FilterAction.BLOCK
        ... )

        Replace phone numbers:

        >>> processor = StreamProcessor()
        >>> processor.add_pattern_filter(
        ...     "redact_phone",
        ...     r"\\d{3}-\\d{3}-\\d{4}",
        ...     FilterAction.REPLACE,
        ...     "[PHONE REDACTED]"
        ... )

        Warn about suspicious content:

        >>> processor = StreamProcessor()
        >>> processor.add_pattern_filter(
        ...     "warn_sql",
        ...     r"(?i)(select|insert|delete|update).*from",
        ...     FilterAction.WARN
        ... )
        """
        self.filter_patterns[name] = (pattern, action, replacement)

    def process_chunk(self, chunk: StreamChunk) -> tuple[StreamChunk, list[FilterResult]]:
        """Process a chunk through all transformers, filters, and callbacks.

        Parameters
        ----------
        chunk : StreamChunk
            The chunk to process.

        Returns
        -------
        tuple[StreamChunk, list[FilterResult]]
            A tuple containing the processed chunk and a list of filter
            results from any filters that matched.

        Examples
        --------
        >>> import time
        >>> processor = StreamProcessor()
        >>> processor.add_transformer(str.upper)
        >>> chunk = StreamChunk("hello", StreamEventType.CHUNK, time.time(), 0)
        >>> processed, results = processor.process_chunk(chunk)
        >>> processed.content
        'HELLO'
        >>> processed.metadata.get("processed")
        True

        With a blocking filter:

        >>> processor = StreamProcessor()
        >>> processor.add_pattern_filter("block_test", "secret", FilterAction.BLOCK)
        >>> chunk = StreamChunk("my secret data", StreamEventType.CHUNK, time.time(), 0)
        >>> processed, results = processor.process_chunk(chunk)
        >>> processed.content
        ''
        >>> results[0].action == FilterAction.BLOCK
        True
        """
        content = chunk.content
        filter_results = []

        # Apply transformers
        for transformer in self.transformers:
            content = transformer(content)

        # Apply pattern filters
        for name, (pattern, action, replacement) in self.filter_patterns.items():
            if re.search(pattern, content):
                if action == FilterAction.BLOCK:
                    filter_results.append(
                        FilterResult(
                            action=action,
                            original=content,
                            filtered="",
                            reason="Blocked by pattern",
                            rule_triggered=name,
                        )
                    )
                    content = ""
                elif action == FilterAction.REPLACE:
                    new_content = re.sub(pattern, replacement, content)
                    filter_results.append(
                        FilterResult(
                            action=action,
                            original=content,
                            filtered=new_content,
                            reason="Replaced by pattern",
                            rule_triggered=name,
                        )
                    )
                    content = new_content
                elif action == FilterAction.WARN:
                    filter_results.append(
                        FilterResult(
                            action=action,
                            original=content,
                            filtered=content,
                            reason="Warning: pattern matched",
                            rule_triggered=name,
                        )
                    )

        # Apply custom filters
        for filter_fn in self.filters:
            result = filter_fn(content)
            filter_results.append(result)
            if result.action == FilterAction.BLOCK:
                content = ""
            elif result.action == FilterAction.REPLACE:
                content = result.filtered

        # Create processed chunk
        processed_chunk = StreamChunk(
            content=content,
            event_type=chunk.event_type,
            timestamp=chunk.timestamp,
            index=chunk.index,
            tokens=chunk.tokens,
            metadata={**chunk.metadata, "processed": True},
        )

        # Call callbacks
        for callback in self.callbacks:
            callback(processed_chunk)

        return processed_chunk, filter_results


class StreamAggregator:
    """Aggregate streaming content from multiple sources with named buffers.

    StreamAggregator manages multiple stream buffers, allowing you to collect
    content from different sources (identified by stream IDs) and aggregate,
    merge, or retrieve them independently.

    Parameters
    ----------
    strategy : str, optional
        Aggregation strategy. Default is "concatenate".

    Attributes
    ----------
    strategy : str
        The aggregation strategy being used.
    buffers : dict[str, StreamBuffer]
        Dictionary mapping stream IDs to their buffers.
    default_buffer : StreamBuffer
        The default buffer for unspecified streams.

    Examples
    --------
    Basic aggregation with single stream:

    >>> import time
    >>> from insideLLMs.streaming import StreamAggregator, StreamChunk, StreamEventType
    >>> aggregator = StreamAggregator()
    >>> aggregator.add(StreamChunk("Hello ", StreamEventType.CHUNK, time.time(), 0))
    >>> aggregator.add(StreamChunk("world!", StreamEventType.CHUNK, time.time(), 1))
    >>> aggregator.get_aggregated()
    'Hello world!'

    Multiple named streams:

    >>> aggregator = StreamAggregator()
    >>> aggregator.add(StreamChunk("Stream A content", StreamEventType.CHUNK, time.time(), 0), "stream_a")
    >>> aggregator.add(StreamChunk("Stream B content", StreamEventType.CHUNK, time.time(), 0), "stream_b")
    >>> aggregator.get_aggregated("stream_a")
    'Stream A content'
    >>> aggregator.get_aggregated("stream_b")
    'Stream B content'

    Merging multiple streams:

    >>> aggregator = StreamAggregator()
    >>> aggregator.add(StreamChunk("Part 1", StreamEventType.CHUNK, time.time(), 0), "a")
    >>> aggregator.add(StreamChunk("Part 2", StreamEventType.CHUNK, time.time(), 0), "b")
    >>> aggregator.merge_streams(["a", "b"], separator=" | ")
    'Part 1 | Part 2'

    Getting all streams at once:

    >>> aggregator = StreamAggregator()
    >>> aggregator.add(StreamChunk("A", StreamEventType.CHUNK, time.time(), 0), "x")
    >>> aggregator.add(StreamChunk("B", StreamEventType.CHUNK, time.time(), 0), "y")
    >>> all_content = aggregator.get_all_aggregated()
    >>> "x" in all_content and "y" in all_content
    True
    """

    def __init__(self, strategy: str = "concatenate"):
        """Initialize the stream aggregator.

        Parameters
        ----------
        strategy : str, optional
            Aggregation strategy. Default is "concatenate".

        Examples
        --------
        >>> aggregator = StreamAggregator()
        >>> aggregator.strategy
        'concatenate'
        >>> aggregator = StreamAggregator(strategy="custom")
        >>> aggregator.strategy
        'custom'
        """
        self.strategy = strategy
        self.buffers: dict[str, StreamBuffer] = {}
        self.default_buffer = StreamBuffer()

    def add(self, chunk: StreamChunk, stream_id: str = "default") -> None:
        """Add a chunk to the appropriate buffer by stream ID.

        Parameters
        ----------
        chunk : StreamChunk
            The chunk to add.
        stream_id : str, optional
            Identifier for the stream. Default is "default".

        Examples
        --------
        >>> import time
        >>> aggregator = StreamAggregator()
        >>> chunk = StreamChunk("test", StreamEventType.CHUNK, time.time(), 0)
        >>> aggregator.add(chunk, "my_stream")
        >>> aggregator.get_aggregated("my_stream")
        'test'
        """
        if stream_id not in self.buffers:
            self.buffers[stream_id] = StreamBuffer()
        self.buffers[stream_id].add_chunk(chunk)

    def get_aggregated(self, stream_id: str = "default") -> str:
        """Get aggregated content for a specific stream.

        Parameters
        ----------
        stream_id : str, optional
            Identifier for the stream. Default is "default".

        Returns
        -------
        str
            The aggregated content, or empty string if stream doesn't exist.

        Examples
        --------
        >>> aggregator = StreamAggregator()
        >>> aggregator.get_aggregated("nonexistent")
        ''
        """
        buffer = self.buffers.get(stream_id)
        if not buffer:
            return ""
        return buffer.get_content()

    def get_all_aggregated(self) -> dict[str, str]:
        """Get aggregated content for all streams.

        Returns
        -------
        dict[str, str]
            Dictionary mapping stream IDs to their aggregated content.

        Examples
        --------
        >>> import time
        >>> aggregator = StreamAggregator()
        >>> aggregator.add(StreamChunk("a", StreamEventType.CHUNK, time.time(), 0), "s1")
        >>> aggregator.add(StreamChunk("b", StreamEventType.CHUNK, time.time(), 0), "s2")
        >>> result = aggregator.get_all_aggregated()
        >>> result["s1"]
        'a'
        >>> result["s2"]
        'b'
        """
        return {stream_id: buffer.get_content() for stream_id, buffer in self.buffers.items()}

    def merge_streams(self, stream_ids: list[str], separator: str = "\n") -> str:
        """Merge multiple streams into a single string.

        Parameters
        ----------
        stream_ids : list[str]
            List of stream IDs to merge, in order.
        separator : str, optional
            String to place between merged streams. Default is newline.

        Returns
        -------
        str
            Merged content from all specified streams.

        Examples
        --------
        >>> import time
        >>> aggregator = StreamAggregator()
        >>> aggregator.add(StreamChunk("First", StreamEventType.CHUNK, time.time(), 0), "a")
        >>> aggregator.add(StreamChunk("Second", StreamEventType.CHUNK, time.time(), 0), "b")
        >>> aggregator.merge_streams(["a", "b"])
        'First\\nSecond'
        >>> aggregator.merge_streams(["a", "b"], separator=", ")
        'First, Second'
        """
        contents = []
        for stream_id in stream_ids:
            content = self.get_aggregated(stream_id)
            if content:
                contents.append(content)
        return separator.join(contents)

    def clear(self, stream_id: Optional[str] = None) -> None:
        """Clear buffer(s).

        Parameters
        ----------
        stream_id : Optional[str], optional
            If provided, clear only that stream. If None, clear all streams.

        Examples
        --------
        Clear a specific stream:

        >>> import time
        >>> aggregator = StreamAggregator()
        >>> aggregator.add(StreamChunk("a", StreamEventType.CHUNK, time.time(), 0), "s1")
        >>> aggregator.add(StreamChunk("b", StreamEventType.CHUNK, time.time(), 0), "s2")
        >>> aggregator.clear("s1")
        >>> aggregator.get_aggregated("s1")
        ''
        >>> aggregator.get_aggregated("s2")
        'b'

        Clear all streams:

        >>> aggregator.clear()
        >>> aggregator.get_aggregated("s2")
        ''
        """
        if stream_id:
            if stream_id in self.buffers:
                self.buffers[stream_id].clear()
        else:
            for buffer in self.buffers.values():
                buffer.clear()


class StreamCollector:
    """Collect and analyze streaming output with metrics and optional processing.

    StreamCollector is the primary class for collecting streamed content from
    LLMs. It handles buffering, metrics collection, error tracking, and optional
    content processing through a StreamProcessor.

    Attributes
    ----------
    buffer : StreamBuffer
        Buffer storing the collected content.
    metrics : StreamMetrics
        Performance metrics for the stream.
    processor : Optional[StreamProcessor]
        Optional processor for transforming/filtering content.
    state : StreamState
        Current state of the collector.
    errors : list[str]
        List of error messages encountered.
    filtered_count : int
        Count of chunks/tokens that were filtered.

    Examples
    --------
    Basic stream collection:

    >>> from insideLLMs.streaming import StreamCollector
    >>> collector = StreamCollector()
    >>> collector.start()
    >>> for text in ["Hello", " ", "world", "!"]:
    ...     chunk = collector.collect_chunk(text)
    >>> summary = collector.complete()
    >>> summary.full_content
    'Hello world!'

    Collection with metrics:

    >>> collector = StreamCollector()
    >>> collector.start()
    >>> collector.collect_chunk("Fast token")
    StreamChunk(content='Fast token', ...)
    >>> collector.collect_chunk("Another token")
    StreamChunk(content='Another token', ...)
    >>> metrics = collector.get_metrics()
    >>> metrics.total_chunks
    2

    Using with a processor:

    >>> from insideLLMs.streaming import StreamProcessor
    >>> processor = StreamProcessor()
    >>> processor.add_transformer(str.upper)
    >>> collector = StreamCollector()
    >>> collector.set_processor(processor)
    >>> collector.start()
    >>> chunk = collector.collect_chunk("hello")
    >>> chunk.content
    'HELLO'

    Error tracking:

    >>> collector = StreamCollector()
    >>> collector.start()
    >>> collector.collect_chunk("partial data")
    StreamChunk(content='partial data', ...)
    >>> collector.record_error("Connection timeout")
    >>> summary = collector.complete()
    >>> summary.errors
    ['Connection timeout']
    """

    def __init__(self):
        """Initialize a new stream collector.

        Examples
        --------
        >>> collector = StreamCollector()
        >>> collector.state == StreamState.IDLE
        True
        >>> collector.buffer.size
        0
        """
        self.buffer = StreamBuffer()
        self.metrics = StreamMetrics(start_time=time.time())
        self.processor: Optional[StreamProcessor] = None
        self.state = StreamState.IDLE
        self._last_token_time: Optional[float] = None
        self.errors: list[str] = []
        self.filtered_count = 0

    def set_processor(self, processor: StreamProcessor) -> None:
        """Set the stream processor for content transformation and filtering.

        Parameters
        ----------
        processor : StreamProcessor
            The processor to use for all collected content.

        Examples
        --------
        >>> collector = StreamCollector()
        >>> processor = StreamProcessor()
        >>> processor.add_transformer(str.lower)
        >>> collector.set_processor(processor)
        >>> collector.processor is not None
        True
        """
        self.processor = processor

    def start(self) -> None:
        """Start collecting, initializing metrics and state.

        This method resets the metrics and sets the state to STREAMING.
        It is called automatically by collect_chunk/collect_token if
        not already started.

        Examples
        --------
        >>> collector = StreamCollector()
        >>> collector.state
        <StreamState.IDLE: 'idle'>
        >>> collector.start()
        >>> collector.state
        <StreamState.STREAMING: 'streaming'>
        """
        self.state = StreamState.STREAMING
        self.metrics = StreamMetrics(start_time=time.time())
        self._last_token_time = time.time()

    def collect_chunk(
        self, content: str, event_type: StreamEventType = StreamEventType.CHUNK
    ) -> StreamChunk:
        """Collect a chunk of content, optionally processing it.

        Parameters
        ----------
        content : str
            The content to collect.
        event_type : StreamEventType, optional
            The type of event. Default is CHUNK.

        Returns
        -------
        StreamChunk
            The created (and possibly processed) chunk.

        Examples
        --------
        >>> collector = StreamCollector()
        >>> chunk = collector.collect_chunk("Hello")
        >>> chunk.content
        'Hello'
        >>> chunk.index
        0

        With specific event type:

        >>> chunk = collector.collect_chunk("", StreamEventType.END)
        >>> chunk.event_type
        <StreamEventType.END: 'end'>

        Auto-starts if not started:

        >>> collector = StreamCollector()
        >>> collector.state
        <StreamState.IDLE: 'idle'>
        >>> _ = collector.collect_chunk("test")
        >>> collector.state
        <StreamState.STREAMING: 'streaming'>
        """
        if self.state == StreamState.IDLE:
            self.start()

        current_time = time.time()

        chunk = StreamChunk(
            content=content,
            event_type=event_type,
            timestamp=current_time,
            index=self.metrics.total_chunks,
        )

        # Process if processor is set
        if self.processor:
            chunk, filter_results = self.processor.process_chunk(chunk)
            self.filtered_count += sum(
                1 for r in filter_results if r.action in [FilterAction.BLOCK, FilterAction.REPLACE]
            )

        # Update metrics
        if self._last_token_time:
            inter_time = current_time - self._last_token_time
            self.metrics.inter_token_times.append(inter_time)

        self._last_token_time = current_time
        self.metrics.total_chunks += 1
        self.metrics.total_characters += len(chunk.content)

        # Add to buffer
        self.buffer.add_chunk(chunk)

        return chunk

    def collect_token(self, text: str, logprob: Optional[float] = None) -> StreamToken:
        """Collect a single token with optional log probability.

        Parameters
        ----------
        text : str
            The token text.
        logprob : Optional[float], optional
            Log probability of the token from the model.

        Returns
        -------
        StreamToken
            The created token.

        Examples
        --------
        >>> collector = StreamCollector()
        >>> token = collector.collect_token("Hello")
        >>> token.text
        'Hello'
        >>> token.index
        0

        With log probability:

        >>> token = collector.collect_token("world", logprob=-0.5)
        >>> token.logprob
        -0.5

        Metrics are updated:

        >>> collector = StreamCollector()
        >>> _ = collector.collect_token("a")
        >>> _ = collector.collect_token("b")
        >>> collector.metrics.total_tokens
        2
        """
        if self.state == StreamState.IDLE:
            self.start()

        current_time = time.time()

        token = StreamToken(
            text=text,
            index=self.metrics.total_tokens,
            timestamp=current_time,
            logprob=logprob,
        )

        # Update metrics
        if self._last_token_time:
            inter_time = current_time - self._last_token_time
            self.metrics.inter_token_times.append(inter_time)

        self._last_token_time = current_time
        self.metrics.total_tokens += 1
        self.metrics.total_characters += len(text)

        # Add to buffer
        self.buffer.add_token(token)

        return token

    def record_error(self, error: str) -> None:
        """Record an error that occurred during streaming.

        Parameters
        ----------
        error : str
            Error message to record.

        Examples
        --------
        >>> collector = StreamCollector()
        >>> collector.start()
        >>> collector.record_error("Network timeout")
        >>> collector.record_error("Retry failed")
        >>> len(collector.errors)
        2
        >>> collector.metrics.error_count
        2
        """
        self.errors.append(error)
        self.metrics.error_count += 1

    def complete(self) -> StreamSummary:
        """Complete collection and return a comprehensive summary.

        Returns
        -------
        StreamSummary
            Summary containing full content, metrics, and state information.

        Examples
        --------
        >>> collector = StreamCollector()
        >>> collector.start()
        >>> collector.collect_chunk("test content")
        StreamChunk(content='test content', ...)
        >>> summary = collector.complete()
        >>> summary.full_content
        'test content'
        >>> summary.state
        <StreamState.COMPLETED: 'completed'>
        >>> summary.metrics.end_time is not None
        True
        """
        self.metrics.end_time = time.time()
        self.state = StreamState.COMPLETED

        return StreamSummary(
            full_content=self.buffer.get_content(),
            metrics=self.metrics,
            chunk_count=self.buffer.chunk_count,
            token_count=self.buffer.token_count,
            state=self.state,
            errors=self.errors,
            filtered_count=self.filtered_count,
        )

    def get_current_content(self) -> str:
        """Get the current collected content without completing.

        Returns
        -------
        str
            All content collected so far.

        Examples
        --------
        >>> collector = StreamCollector()
        >>> collector.start()
        >>> collector.collect_chunk("Hello ")
        StreamChunk(content='Hello ', ...)
        >>> collector.get_current_content()
        'Hello '
        >>> collector.collect_chunk("world")
        StreamChunk(content='world', ...)
        >>> collector.get_current_content()
        'Hello world'
        """
        return self.buffer.get_content()

    def get_metrics(self) -> StreamMetrics:
        """Get the current metrics without completing.

        Returns
        -------
        StreamMetrics
            Current performance metrics.

        Examples
        --------
        >>> collector = StreamCollector()
        >>> collector.start()
        >>> collector.collect_chunk("test")
        StreamChunk(content='test', ...)
        >>> metrics = collector.get_metrics()
        >>> metrics.total_chunks
        1
        >>> metrics.total_characters
        4
        """
        return self.metrics


class StreamIterator:
    """Synchronous iterator over streaming content with automatic collection.

    StreamIterator wraps a string iterator (e.g., from an LLM streaming API)
    and provides processed StreamChunk objects while automatically collecting
    metrics and managing the stream lifecycle.

    Parameters
    ----------
    source : Iterator[str]
        Source iterator yielding string content.
    processor : Optional[StreamProcessor], optional
        Processor for transforming/filtering content.
    collector : Optional[StreamCollector], optional
        Collector for storing content and metrics. Creates one if not provided.

    Attributes
    ----------
    source : Iterator[str]
        The source iterator.
    processor : Optional[StreamProcessor]
        The configured processor.
    collector : StreamCollector
        The collector managing content and metrics.

    Examples
    --------
    Basic iteration:

    >>> from insideLLMs.streaming import StreamIterator
    >>> source = iter(["Hello", " ", "world", "!"])
    >>> stream_iter = StreamIterator(source)
    >>> for chunk in stream_iter:
    ...     print(chunk.content, end="")
    Hello world!

    With processing:

    >>> source = iter(["hello", " ", "world"])
    >>> processor = StreamProcessor()
    >>> processor.add_transformer(str.upper)
    >>> stream_iter = StreamIterator(source, processor=processor)
    >>> [chunk.content for chunk in stream_iter]
    ['HELLO', ' ', 'WORLD']

    Getting summary after iteration:

    >>> source = iter(["a", "b", "c"])
    >>> stream_iter = StreamIterator(source)
    >>> for _ in stream_iter:
    ...     pass
    >>> summary = stream_iter.get_summary()
    >>> summary.chunk_count
    3

    Manual iteration with next():

    >>> source = iter(["first", "second"])
    >>> stream_iter = StreamIterator(source)
    >>> it = iter(stream_iter)
    >>> next(it).content
    'first'
    >>> next(it).content
    'second'
    """

    def __init__(
        self,
        source: Iterator[str],
        processor: Optional[StreamProcessor] = None,
        collector: Optional[StreamCollector] = None,
    ):
        """Initialize the stream iterator.

        Parameters
        ----------
        source : Iterator[str]
            Source iterator yielding string content.
        processor : Optional[StreamProcessor], optional
            Processor for transforming/filtering content.
        collector : Optional[StreamCollector], optional
            Collector for storing content and metrics.

        Examples
        --------
        >>> source = iter(["test"])
        >>> stream_iter = StreamIterator(source)
        >>> stream_iter.collector is not None
        True
        """
        self.source = source
        self.processor = processor
        self.collector = collector or StreamCollector()
        if processor:
            self.collector.set_processor(processor)
        self._index = 0

    def __iter__(self) -> "StreamIterator":
        """Return the iterator and start collection.

        Returns
        -------
        StreamIterator
            Self, for use in for loops.
        """
        self.collector.start()
        return self

    def __next__(self) -> StreamChunk:
        """Get the next chunk from the stream.

        Returns
        -------
        StreamChunk
            The next processed chunk.

        Raises
        ------
        StopIteration
            When the source iterator is exhausted.
        """
        try:
            content = next(self.source)
            chunk = self.collector.collect_chunk(content)
            self._index += 1
            return chunk
        except StopIteration:
            self.collector.complete()
            raise

    def get_summary(self) -> StreamSummary:
        """Get the collection summary.

        Returns
        -------
        StreamSummary
            Summary of the collected stream.

        Examples
        --------
        >>> source = iter(["a", "b"])
        >>> stream_iter = StreamIterator(source)
        >>> list(stream_iter)  # Exhaust the iterator
        [StreamChunk(...), StreamChunk(...)]
        >>> summary = stream_iter.get_summary()
        >>> summary.full_content
        'ab'
        """
        return self.collector.complete()


class AsyncStreamIterator:
    """Asynchronous iterator over streaming content with automatic collection.

    AsyncStreamIterator wraps an async string iterator (e.g., from an async
    LLM streaming API) and provides processed StreamChunk objects while
    automatically collecting metrics and managing the stream lifecycle.

    Parameters
    ----------
    source : AsyncIterator[str]
        Async source iterator yielding string content.
    processor : Optional[StreamProcessor], optional
        Processor for transforming/filtering content.
    collector : Optional[StreamCollector], optional
        Collector for storing content and metrics. Creates one if not provided.

    Attributes
    ----------
    source : AsyncIterator[str]
        The async source iterator.
    processor : Optional[StreamProcessor]
        The configured processor.
    collector : StreamCollector
        The collector managing content and metrics.

    Examples
    --------
    Basic async iteration:

    >>> import asyncio
    >>> from insideLLMs.streaming import AsyncStreamIterator
    >>> async def async_source():
    ...     for item in ["Hello", " ", "world"]:
    ...         yield item
    >>> async def main():
    ...     stream_iter = AsyncStreamIterator(async_source())
    ...     async for chunk in stream_iter:
    ...         print(chunk.content, end="")
    >>> # asyncio.run(main())  # Prints: Hello world

    With processing:

    >>> async def example_with_processor():
    ...     async def source():
    ...         for item in ["hello", "world"]:
    ...             yield item
    ...     processor = StreamProcessor()
    ...     processor.add_transformer(str.upper)
    ...     stream_iter = AsyncStreamIterator(source(), processor=processor)
    ...     results = []
    ...     async for chunk in stream_iter:
    ...         results.append(chunk.content)
    ...     return results
    >>> # asyncio.run(example_with_processor())  # Returns: ['HELLO', 'WORLD']

    Getting summary after iteration:

    >>> async def example_summary():
    ...     async def source():
    ...         yield "test"
    ...     stream_iter = AsyncStreamIterator(source())
    ...     async for _ in stream_iter:
    ...         pass
    ...     return stream_iter.get_summary()
    >>> # summary = asyncio.run(example_summary())

    Simulating LLM streaming response:

    >>> async def mock_llm_stream():
    ...     tokens = ["The", " answer", " is", " 42"]
    ...     for token in tokens:
    ...         yield token
    >>> async def process_response():
    ...     stream_iter = AsyncStreamIterator(mock_llm_stream())
    ...     async for chunk in stream_iter:
    ...         print(chunk.content, end="", flush=True)
    ...     print()
    ...     return stream_iter.get_summary()
    >>> # asyncio.run(process_response())
    """

    def __init__(
        self,
        source: AsyncIterator[str],
        processor: Optional[StreamProcessor] = None,
        collector: Optional[StreamCollector] = None,
    ):
        """Initialize the async stream iterator.

        Parameters
        ----------
        source : AsyncIterator[str]
            Async source iterator yielding string content.
        processor : Optional[StreamProcessor], optional
            Processor for transforming/filtering content.
        collector : Optional[StreamCollector], optional
            Collector for storing content and metrics.

        Examples
        --------
        >>> async def source():
        ...     yield "test"
        >>> stream_iter = AsyncStreamIterator(source())
        >>> stream_iter.collector is not None
        True
        """
        self.source = source
        self.processor = processor
        self.collector = collector or StreamCollector()
        if processor:
            self.collector.set_processor(processor)
        self._index = 0

    def __aiter__(self) -> "AsyncStreamIterator":
        """Return the async iterator and start collection.

        Returns
        -------
        AsyncStreamIterator
            Self, for use in async for loops.
        """
        self.collector.start()
        return self

    async def __anext__(self) -> StreamChunk:
        """Get the next chunk from the async stream.

        Returns
        -------
        StreamChunk
            The next processed chunk.

        Raises
        ------
        StopAsyncIteration
            When the source iterator is exhausted.
        """
        try:
            content = await self.source.__anext__()
            chunk = self.collector.collect_chunk(content)
            self._index += 1
            return chunk
        except StopAsyncIteration:
            self.collector.complete()
            raise

    def get_summary(self) -> StreamSummary:
        """Get the collection summary.

        Returns
        -------
        StreamSummary
            Summary of the collected stream.

        Examples
        --------
        >>> async def example():
        ...     async def source():
        ...         yield "a"
        ...         yield "b"
        ...     stream_iter = AsyncStreamIterator(source())
        ...     async for _ in stream_iter:
        ...         pass
        ...     return stream_iter.get_summary().full_content
        >>> # asyncio.run(example())  # Returns: 'ab'
        """
        return self.collector.complete()


class ContentDetector:
    """Detect specific content patterns in streams using regex matching.

    ContentDetector monitors streaming content for predefined patterns,
    such as code blocks, URLs, or sensitive information. It maintains
    a rolling buffer to detect patterns that may span multiple chunks.

    Attributes
    ----------
    patterns : dict[str, str]
        Dictionary mapping pattern names to regex patterns.
    detections : list[dict[str, Any]]
        List of all pattern detections found.

    Examples
    --------
    Detecting URLs in a stream:

    >>> from insideLLMs.streaming import ContentDetector
    >>> detector = ContentDetector()
    >>> detector.add_pattern("url", r"https?://[\\w.-]+(?:/[\\w./-]*)?")
    >>> detector.check("Check out https://example.com for more info")
    [{'pattern_name': 'url', 'pattern': ..., 'match': 'https://example.com', ...}]

    Detecting code blocks:

    >>> detector = ContentDetector()
    >>> detector.add_pattern("code_block", r"```[\\w]*\\n[\\s\\S]*?```")
    >>> content = "Here is code:\\n```python\\nprint('hello')\\n```"
    >>> detections = detector.check(content)
    >>> len(detections) > 0
    True

    Monitoring for sensitive content:

    >>> detector = ContentDetector()
    >>> detector.add_pattern("api_key", r"(?i)api[_-]?key[:\\s]*[a-zA-Z0-9_-]{20,}")
    >>> detector.add_pattern("email", r"[\\w.-]+@[\\w.-]+\\.[a-z]{2,}")
    >>> detector.check("Contact admin@example.com")
    [{'pattern_name': 'email', ...}]

    Accumulating detections over time:

    >>> detector = ContentDetector()
    >>> detector.add_pattern("number", r"\\d+")
    >>> detector.check("First: 123")
    [{'pattern_name': 'number', 'match': '123', ...}]
    >>> detector.check(" Second: 456")
    [...]
    >>> len(detector.get_all_detections()) >= 2
    True
    """

    def __init__(self):
        """Initialize the content detector with empty patterns and buffer.

        Examples
        --------
        >>> detector = ContentDetector()
        >>> len(detector.patterns)
        0
        >>> len(detector.detections)
        0
        """
        self.patterns: dict[str, str] = {}
        self.detections: list[dict[str, Any]] = []
        self._buffer = ""
        self._buffer_size = 10000

    def add_pattern(self, name: str, pattern: str) -> None:
        """Add a regex pattern to detect.

        Parameters
        ----------
        name : str
            Unique name for the pattern (used in detection results).
        pattern : str
            Regular expression pattern to match.

        Examples
        --------
        >>> detector = ContentDetector()
        >>> detector.add_pattern("phone", r"\\d{3}-\\d{3}-\\d{4}")
        >>> detector.add_pattern("email", r"[\\w.-]+@[\\w.-]+")
        >>> len(detector.patterns)
        2
        """
        self.patterns[name] = pattern

    def check(self, content: str) -> list[dict[str, Any]]:
        """Check content for registered patterns.

        The content is added to a rolling buffer, allowing detection
        of patterns that span multiple check() calls.

        Parameters
        ----------
        content : str
            New content to check for patterns.

        Returns
        -------
        list[dict[str, Any]]
            List of detection results from this check, each containing:
            - pattern_name: Name of the matched pattern
            - pattern: The regex pattern that matched
            - match: The matched text
            - start: Start position in buffer
            - end: End position in buffer
            - timestamp: When the detection occurred

        Examples
        --------
        >>> detector = ContentDetector()
        >>> detector.add_pattern("word", r"\\bhello\\b")
        >>> results = detector.check("Say hello world")
        >>> results[0]["match"]
        'hello'
        >>> results[0]["pattern_name"]
        'word'
        """
        # Add to buffer
        self._buffer += content
        if len(self._buffer) > self._buffer_size:
            self._buffer = self._buffer[-self._buffer_size :]

        detected = []
        for name, pattern in self.patterns.items():
            matches = list(re.finditer(pattern, self._buffer))
            for match in matches:
                detection = {
                    "pattern_name": name,
                    "pattern": pattern,
                    "match": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "timestamp": time.time(),
                }
                detected.append(detection)
                self.detections.append(detection)

        return detected

    def get_all_detections(self) -> list[dict[str, Any]]:
        """Get all detections accumulated so far.

        Returns
        -------
        list[dict[str, Any]]
            List of all detection results from all check() calls.

        Examples
        --------
        >>> detector = ContentDetector()
        >>> detector.add_pattern("num", r"\\d+")
        >>> detector.check("a1b2c3")
        [...]
        >>> all_detections = detector.get_all_detections()
        >>> len(all_detections)
        3
        """
        return self.detections

    def clear(self) -> None:
        """Clear the buffer and all detections.

        Examples
        --------
        >>> detector = ContentDetector()
        >>> detector.add_pattern("test", r"test")
        >>> detector.check("test content")
        [...]
        >>> len(detector.detections) > 0
        True
        >>> detector.clear()
        >>> len(detector.detections)
        0
        """
        self._buffer = ""
        self.detections.clear()


class StreamingWindowAnalyzer:
    """Analyze streaming content with sliding windows for real-time metrics.

    StreamingWindowAnalyzer maintains a fixed-size window of recent tokens
    and runs custom analyzer functions on the window contents. This is useful
    for computing rolling statistics, detecting patterns, or monitoring
    content characteristics as the stream progresses.

    Parameters
    ----------
    window_size : int, optional
        Maximum number of tokens to keep in the window. Default is 100.

    Attributes
    ----------
    window_size : int
        The configured window size.
    window : deque
        The sliding window of tokens.
    analyzers : list[Callable[[list[str]], dict[str, Any]]]
        List of analyzer functions to run on each update.

    Examples
    --------
    Basic usage with custom analyzer:

    >>> from insideLLMs.streaming import StreamingWindowAnalyzer
    >>> analyzer = StreamingWindowAnalyzer(window_size=10)
    >>> def count_words(tokens):
    ...     return {"word_count": len(tokens)}
    >>> analyzer.add_analyzer(count_words)
    >>> for word in ["hello", "world"]:
    ...     result = analyzer.add_token(word)
    >>> result["analyzer_0"]["word_count"]
    2

    Computing rolling statistics:

    >>> analyzer = StreamingWindowAnalyzer(window_size=5)
    >>> def avg_length(tokens):
    ...     if not tokens:
    ...         return {"avg_length": 0}
    ...     return {"avg_length": sum(len(t) for t in tokens) / len(tokens)}
    >>> analyzer.add_analyzer(avg_length)
    >>> for word in ["hi", "hello", "world"]:
    ...     result = analyzer.add_token(word)
    >>> result["analyzer_0"]["avg_length"]
    4.0

    Detecting repetition:

    >>> analyzer = StreamingWindowAnalyzer(window_size=10)
    >>> def detect_repetition(tokens):
    ...     if len(tokens) < 2:
    ...         return {"has_repetition": False}
    ...     return {"has_repetition": tokens[-1] == tokens[-2]}
    >>> analyzer.add_analyzer(detect_repetition)
    >>> analyzer.add_token("the")
    {'analyzer_0': {'has_repetition': False}}
    >>> analyzer.add_token("the")
    {'analyzer_0': {'has_repetition': True}}

    Getting window content:

    >>> analyzer = StreamingWindowAnalyzer(window_size=5)
    >>> for word in ["Hello", " ", "world"]:
    ...     analyzer.add_token(word)
    {...}
    >>> analyzer.get_window_content()
    'Hello world'
    """

    def __init__(self, window_size: int = 100):
        """Initialize the sliding window analyzer.

        Parameters
        ----------
        window_size : int, optional
            Maximum number of tokens to keep. Default is 100.

        Examples
        --------
        >>> analyzer = StreamingWindowAnalyzer()
        >>> analyzer.window_size
        100
        >>> analyzer = StreamingWindowAnalyzer(window_size=50)
        >>> analyzer.window_size
        50
        """
        self.window_size = window_size
        self.window: deque = deque(maxlen=window_size)
        self.analyzers: list[Callable[[list[str]], dict[str, Any]]] = []

    def add_analyzer(self, analyzer: Callable[[list[str]], dict[str, Any]]) -> None:
        """Add a window analyzer function.

        Analyzer functions receive the current window contents as a list
        and should return a dictionary of computed values.

        Parameters
        ----------
        analyzer : Callable[[list[str]], dict[str, Any]]
            Function that takes a list of tokens and returns analysis results.

        Examples
        --------
        >>> analyzer = StreamingWindowAnalyzer()
        >>> analyzer.add_analyzer(lambda tokens: {"count": len(tokens)})
        >>> analyzer.add_analyzer(lambda tokens: {"joined": "".join(tokens)})
        >>> len(analyzer.analyzers)
        2
        """
        self.analyzers.append(analyzer)

    def add_token(self, token: str) -> dict[str, Any]:
        """Add a token to the window and run all analyzers.

        Parameters
        ----------
        token : str
            The token to add.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping analyzer names (analyzer_0, analyzer_1, etc.)
            to their results.

        Examples
        --------
        >>> analyzer = StreamingWindowAnalyzer()
        >>> analyzer.add_analyzer(lambda t: {"len": len(t)})
        >>> result = analyzer.add_token("hello")
        >>> result["analyzer_0"]["len"]
        1
        >>> result = analyzer.add_token("world")
        >>> result["analyzer_0"]["len"]
        2
        """
        self.window.append(token)

        results = {}
        for i, analyzer in enumerate(self.analyzers):
            result = analyzer(list(self.window))
            results[f"analyzer_{i}"] = result

        return results

    def get_window_content(self) -> str:
        """Get the current window content as a concatenated string.

        Returns
        -------
        str
            All tokens in the window joined together.

        Examples
        --------
        >>> analyzer = StreamingWindowAnalyzer(window_size=5)
        >>> analyzer.add_token("Hello")
        {}
        >>> analyzer.add_token(" ")
        {}
        >>> analyzer.add_token("world")
        {}
        >>> analyzer.get_window_content()
        'Hello world'
        """
        return "".join(self.window)

    def clear(self) -> None:
        """Clear all tokens from the window.

        Examples
        --------
        >>> analyzer = StreamingWindowAnalyzer()
        >>> analyzer.add_token("test")
        {}
        >>> len(analyzer.window)
        1
        >>> analyzer.clear()
        >>> len(analyzer.window)
        0
        """
        self.window.clear()


class StreamRateLimiter:
    """Rate limit streaming output using a token bucket algorithm.

    StreamRateLimiter controls the rate at which tokens are emitted,
    useful for simulating realistic streaming output or preventing
    overwhelming downstream consumers. It uses a token bucket algorithm
    that allows bursts while maintaining an average rate.

    Parameters
    ----------
    tokens_per_second : float, optional
        Average rate of token emission. Default is 50.0.
    burst_size : int, optional
        Maximum tokens that can be emitted in a burst. Default is 10.

    Attributes
    ----------
    tokens_per_second : float
        Configured tokens per second rate.
    burst_size : int
        Maximum burst size.

    Examples
    --------
    Basic rate limiting:

    >>> from insideLLMs.streaming import StreamRateLimiter
    >>> import time
    >>> limiter = StreamRateLimiter(tokens_per_second=10.0, burst_size=5)
    >>> # First few tokens are instant (burst)
    >>> for i in range(5):
    ...     wait = limiter.acquire()
    ...     # wait should be 0.0 for burst tokens

    Simulating streaming display:

    >>> limiter = StreamRateLimiter(tokens_per_second=20.0)
    >>> tokens = ["Hello", " ", "world", "!"]
    >>> for token in tokens:
    ...     limiter.wait_and_acquire()  # Blocks if needed
    ...     print(token, end="", flush=True)
    Hello world!

    Checking wait time without waiting:

    >>> limiter = StreamRateLimiter(tokens_per_second=10.0, burst_size=2)
    >>> limiter.acquire()  # First token, instant
    0.0
    >>> limiter.acquire()  # Second token, instant (within burst)
    0.0
    >>> wait = limiter.acquire()  # Third token, may need to wait
    >>> wait >= 0  # Returns wait time needed
    True

    High-speed streaming:

    >>> limiter = StreamRateLimiter(tokens_per_second=1000.0, burst_size=100)
    >>> # Good for fast LLM responses with large burst allowance
    >>> wait = limiter.acquire()
    >>> wait == 0.0  # Within burst
    True
    """

    def __init__(
        self,
        tokens_per_second: float = 50.0,
        burst_size: int = 10,
    ):
        """Initialize the rate limiter.

        Parameters
        ----------
        tokens_per_second : float, optional
            Average rate of token emission. Default is 50.0.
        burst_size : int, optional
            Maximum tokens that can be emitted in a burst. Default is 10.

        Examples
        --------
        >>> limiter = StreamRateLimiter()
        >>> limiter.tokens_per_second
        50.0
        >>> limiter.burst_size
        10
        >>> limiter = StreamRateLimiter(tokens_per_second=100.0, burst_size=20)
        >>> limiter.tokens_per_second
        100.0
        """
        self.tokens_per_second = tokens_per_second
        self.burst_size = burst_size
        self._tokens_available = float(burst_size)
        self._last_time = time.time()

    def acquire(self) -> float:
        """Acquire permission to emit a token, returning wait time needed.

        This method does not block. It returns 0.0 if a token can be
        emitted immediately, or the number of seconds to wait if rate
        limited.

        Returns
        -------
        float
            Seconds to wait before emitting, or 0.0 if immediate.

        Examples
        --------
        >>> limiter = StreamRateLimiter(tokens_per_second=10.0, burst_size=2)
        >>> limiter.acquire()  # First token
        0.0
        >>> limiter.acquire()  # Second token
        0.0
        >>> wait = limiter.acquire()  # Third token
        >>> wait > 0  # Need to wait
        True
        """
        current_time = time.time()
        time_passed = current_time - self._last_time
        self._last_time = current_time

        # Replenish tokens
        self._tokens_available += time_passed * self.tokens_per_second
        self._tokens_available = min(self._tokens_available, float(self.burst_size))

        if self._tokens_available >= 1.0:
            self._tokens_available -= 1.0
            return 0.0
        else:
            wait_time = (1.0 - self._tokens_available) / self.tokens_per_second
            return wait_time

    def wait_and_acquire(self) -> None:
        """Wait if necessary and then acquire permission to emit.

        This method blocks until a token can be emitted, then acquires
        the permission. Use this for simple rate-limited loops.

        Examples
        --------
        >>> limiter = StreamRateLimiter(tokens_per_second=100.0)
        >>> import time
        >>> start = time.time()
        >>> for _ in range(5):
        ...     limiter.wait_and_acquire()
        >>> elapsed = time.time() - start
        >>> elapsed < 1.0  # Should be fast with burst
        True
        """
        wait_time = self.acquire()
        if wait_time > 0:
            time.sleep(wait_time)


# Convenience functions


def create_stream_collector() -> StreamCollector:
    """Create and return a new StreamCollector instance.

    This is a factory function providing a simple way to create a
    stream collector without importing the class directly.

    Returns
    -------
    StreamCollector
        A new, initialized stream collector.

    Examples
    --------
    >>> from insideLLMs.streaming import create_stream_collector
    >>> collector = create_stream_collector()
    >>> collector.start()
    >>> collector.collect_chunk("Hello")
    StreamChunk(content='Hello', ...)
    >>> summary = collector.complete()

    Using in a context where you need quick collection:

    >>> collector = create_stream_collector()
    >>> for token in ["a", "b", "c"]:
    ...     collector.collect_chunk(token)
    >>> collector.get_current_content()
    'abc'
    """
    return StreamCollector()


def create_stream_processor() -> StreamProcessor:
    """Create and return a new StreamProcessor instance.

    This is a factory function providing a simple way to create a
    stream processor without importing the class directly.

    Returns
    -------
    StreamProcessor
        A new, empty stream processor.

    Examples
    --------
    >>> from insideLLMs.streaming import create_stream_processor
    >>> processor = create_stream_processor()
    >>> processor.add_transformer(str.upper)
    >>> processor.add_pattern_filter("test", "bad", FilterAction.BLOCK)

    Quick setup with chaining:

    >>> processor = create_stream_processor()
    >>> processor.add_transformer(str.strip)
    >>> processor.add_transformer(str.lower)
    >>> len(processor.transformers)
    2
    """
    return StreamProcessor()


def collect_stream(
    stream: Iterator[str],
    processor: Optional[StreamProcessor] = None,
) -> StreamSummary:
    """Collect an entire stream and return a summary.

    This convenience function consumes the entire stream, optionally
    processing each chunk, and returns a comprehensive summary.

    Parameters
    ----------
    stream : Iterator[str]
        Iterator yielding string content.
    processor : Optional[StreamProcessor], optional
        Processor for transforming/filtering content.

    Returns
    -------
    StreamSummary
        Summary containing full content, metrics, and state.

    Examples
    --------
    Basic collection:

    >>> from insideLLMs.streaming import collect_stream
    >>> stream = iter(["Hello", " ", "world", "!"])
    >>> summary = collect_stream(stream)
    >>> summary.full_content
    'Hello world!'

    With processing:

    >>> processor = StreamProcessor()
    >>> processor.add_transformer(str.upper)
    >>> stream = iter(["hello", " ", "world"])
    >>> summary = collect_stream(stream, processor)
    >>> summary.full_content
    'HELLO WORLD'

    Getting metrics:

    >>> stream = iter(["a", "b", "c"])
    >>> summary = collect_stream(stream)
    >>> summary.chunk_count
    3

    Simulating LLM response collection:

    >>> def mock_llm():
    ...     yield "The answer"
    ...     yield " is "
    ...     yield "42."
    >>> summary = collect_stream(mock_llm())
    >>> summary.full_content
    'The answer is 42.'
    """
    collector = StreamCollector()
    if processor:
        collector.set_processor(processor)

    collector.start()
    for content in stream:
        collector.collect_chunk(content)

    return collector.complete()


def iterate_stream(
    stream: Iterator[str],
    processor: Optional[StreamProcessor] = None,
) -> Generator[StreamChunk, None, None]:
    """Iterate over a stream with processing, yielding StreamChunks.

    This generator function wraps a string iterator and yields processed
    StreamChunk objects, allowing real-time handling of streamed content.

    Parameters
    ----------
    stream : Iterator[str]
        Iterator yielding string content.
    processor : Optional[StreamProcessor], optional
        Processor for transforming/filtering content.

    Yields
    ------
    StreamChunk
        Processed chunk for each input string.

    Examples
    --------
    Basic iteration:

    >>> from insideLLMs.streaming import iterate_stream
    >>> stream = iter(["Hello", " ", "world"])
    >>> for chunk in iterate_stream(stream):
    ...     print(chunk.content, end="")
    Hello world

    With processing:

    >>> processor = StreamProcessor()
    >>> processor.add_transformer(str.upper)
    >>> for chunk in iterate_stream(iter(["hi"]), processor):
    ...     print(chunk.content)
    HI

    Real-time display simulation:

    >>> import sys
    >>> for chunk in iterate_stream(iter(["Loading", ".", ".", "."])):
    ...     print(chunk.content, end="", flush=True)
    Loading...

    Collecting chunks into a list:

    >>> chunks = list(iterate_stream(iter(["a", "b", "c"])))
    >>> len(chunks)
    3
    """
    collector = StreamCollector()
    if processor:
        collector.set_processor(processor)

    collector.start()
    for content in stream:
        yield collector.collect_chunk(content)

    collector.complete()


def add_content_filter(
    processor: StreamProcessor,
    patterns: dict[str, str],
    action: FilterAction = FilterAction.REPLACE,
    replacement: str = "[FILTERED]",
) -> None:
    """Add multiple content filters to a processor at once.

    This convenience function adds multiple regex pattern filters to
    a processor, all with the same action and replacement string.

    Parameters
    ----------
    processor : StreamProcessor
        The processor to add filters to.
    patterns : dict[str, str]
        Dictionary mapping filter names to regex patterns.
    action : FilterAction, optional
        Action to take on match. Default is REPLACE.
    replacement : str, optional
        Replacement text for REPLACE action. Default is "[FILTERED]".

    Examples
    --------
    Adding PII filters:

    >>> from insideLLMs.streaming import add_content_filter, StreamProcessor, FilterAction
    >>> processor = StreamProcessor()
    >>> add_content_filter(processor, {
    ...     "email": r"[\\w.-]+@[\\w.-]+\\.[a-z]+",
    ...     "phone": r"\\d{3}-\\d{3}-\\d{4}",
    ...     "ssn": r"\\d{3}-\\d{2}-\\d{4}"
    ... })
    >>> len(processor.filter_patterns)
    3

    With blocking action:

    >>> processor = StreamProcessor()
    >>> add_content_filter(
    ...     processor,
    ...     {"profanity": r"\\b(bad|words)\\b"},
    ...     action=FilterAction.BLOCK
    ... )

    Custom replacement:

    >>> processor = StreamProcessor()
    >>> add_content_filter(
    ...     processor,
    ...     {"secret": r"password:\\s*\\S+"},
    ...     replacement="password: [REDACTED]"
    ... )
    """
    for name, pattern in patterns.items():
        processor.add_pattern_filter(name, pattern, action, replacement)


def measure_stream_speed(
    stream: Iterator[str],
    sample_size: int = 100,
) -> dict[str, float]:
    """Measure the speed characteristics of a stream.

    This function collects a sample of the stream and returns
    speed metrics including tokens per second and inter-token timing.

    Parameters
    ----------
    stream : Iterator[str]
        Iterator yielding string content.
    sample_size : int, optional
        Number of chunks to sample. Default is 100.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - tokens_per_second: Rate of token generation
        - mean_inter_token_time: Average time between tokens
        - total_duration: Total time for sample
        - sample_size: Actual number of samples collected

    Examples
    --------
    Measuring LLM response speed:

    >>> from insideLLMs.streaming import measure_stream_speed
    >>> def mock_stream():
    ...     for i in range(50):
    ...         yield f"token{i}"
    >>> metrics = measure_stream_speed(mock_stream(), sample_size=50)
    >>> "tokens_per_second" in metrics
    True
    >>> "mean_inter_token_time" in metrics
    True

    Small sample:

    >>> stream = iter(["a", "b", "c"])
    >>> metrics = measure_stream_speed(stream, sample_size=10)
    >>> metrics["sample_size"]
    3

    Comparing stream speeds:

    >>> fast_stream = iter(["x"] * 100)
    >>> metrics = measure_stream_speed(fast_stream, sample_size=100)
    >>> metrics["sample_size"]
    100
    """
    collector = StreamCollector()
    collector.start()

    count = 0
    for content in stream:
        collector.collect_chunk(content)
        count += 1
        if count >= sample_size:
            break

    summary = collector.complete()

    return {
        "tokens_per_second": summary.metrics.tokens_per_second,
        "mean_inter_token_time": summary.metrics.mean_inter_token_time,
        "total_duration": summary.metrics.duration,
        "sample_size": count,
    }


def detect_in_stream(
    stream: Iterator[str],
    patterns: dict[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    """Detect patterns in a stream and return content with all detections.

    This convenience function consumes a stream while monitoring for
    specified patterns, returning both the full content and all detections.

    Parameters
    ----------
    stream : Iterator[str]
        Iterator yielding string content.
    patterns : dict[str, str]
        Dictionary mapping pattern names to regex patterns.

    Returns
    -------
    tuple[str, list[dict[str, Any]]]
        Tuple of (full_content, list_of_detections).
        Each detection contains pattern_name, match, start, end, etc.

    Examples
    --------
    Detecting URLs:

    >>> from insideLLMs.streaming import detect_in_stream
    >>> stream = iter(["Check ", "https://example.com", " for info"])
    >>> content, detections = detect_in_stream(stream, {
    ...     "url": r"https?://[\\w.-]+"
    ... })
    >>> content
    'Check https://example.com for info'
    >>> len(detections) > 0
    True

    Multiple pattern types:

    >>> stream = iter(["Email: test@example.com, Phone: 555-123-4567"])
    >>> content, detections = detect_in_stream(stream, {
    ...     "email": r"[\\w.-]+@[\\w.-]+",
    ...     "phone": r"\\d{3}-\\d{3}-\\d{4}"
    ... })
    >>> len(detections)
    2

    No matches:

    >>> stream = iter(["plain text without patterns"])
    >>> content, detections = detect_in_stream(stream, {"number": r"\\d+"})
    >>> detections
    []

    Detecting code blocks:

    >>> stream = iter(["Here is code: ", "```python\\nprint('hi')\\n```"])
    >>> content, detections = detect_in_stream(stream, {
    ...     "code": r"```[\\w]*"
    ... })
    >>> len(detections) > 0
    True
    """
    detector = ContentDetector()
    for name, pattern in patterns.items():
        detector.add_pattern(name, pattern)

    content_parts = []
    for chunk in stream:
        content_parts.append(chunk)
        detector.check(chunk)

    return "".join(content_parts), detector.get_all_detections()
