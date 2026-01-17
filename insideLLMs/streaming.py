"""Output streaming utilities for LLM responses.

This module provides tools for handling, processing, and analyzing
streaming output from language models, including:
- Stream buffer management
- Token-level processing
- Streaming metrics collection
- Content filtering during streaming
- Aggregation and transformation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, AsyncIterator, Callable, Dict, Generator, Iterator,
    List, Optional, Tuple, TypeVar, Union
)
from datetime import datetime
import time
import re
from collections import deque


T = TypeVar('T')


class StreamEventType(Enum):
    """Types of streaming events."""
    TOKEN = "token"
    CHUNK = "chunk"
    START = "start"
    END = "end"
    ERROR = "error"
    METADATA = "metadata"
    TOOL_CALL = "tool_call"
    CONTENT_BLOCK = "content_block"


class StreamState(Enum):
    """State of a stream."""
    IDLE = "idle"
    STREAMING = "streaming"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class FilterAction(Enum):
    """Actions for content filtering."""
    ALLOW = "allow"
    BLOCK = "block"
    REPLACE = "replace"
    WARN = "warn"


@dataclass
class StreamToken:
    """A single token from a stream."""
    text: str
    index: int
    timestamp: float
    logprob: Optional[float] = None
    token_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    """A chunk of streamed content (may contain multiple tokens)."""
    content: str
    event_type: StreamEventType
    timestamp: float
    index: int
    tokens: List[StreamToken] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    """Metrics collected during streaming."""
    start_time: float
    end_time: Optional[float] = None
    total_tokens: int = 0
    total_chunks: int = 0
    total_characters: int = 0
    inter_token_times: List[float] = field(default_factory=list)
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
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
        """Total duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        duration = self.duration
        if duration <= 0:
            return 0.0
        return self.total_tokens / duration

    @property
    def mean_inter_token_time(self) -> float:
        """Mean time between tokens."""
        if not self.inter_token_times:
            return 0.0
        return sum(self.inter_token_times) / len(self.inter_token_times)


@dataclass
class FilterResult:
    """Result of content filtering."""
    action: FilterAction
    original: str
    filtered: str
    reason: Optional[str] = None
    rule_triggered: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "original": self.original,
            "filtered": self.filtered,
            "reason": self.reason,
            "rule_triggered": self.rule_triggered,
        }


@dataclass
class StreamSummary:
    """Summary of a completed stream."""
    full_content: str
    metrics: StreamMetrics
    chunk_count: int
    token_count: int
    state: StreamState
    errors: List[str] = field(default_factory=list)
    filtered_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    """Buffer for managing streamed content."""

    def __init__(self, max_size: int = 1000000):
        self.max_size = max_size
        self.chunks: List[StreamChunk] = []
        self.tokens: List[StreamToken] = []
        self._content_buffer: List[str] = []
        self._total_chars = 0

    def add_chunk(self, chunk: StreamChunk) -> None:
        """Add a chunk to the buffer."""
        if self._total_chars + len(chunk.content) > self.max_size:
            # Remove old chunks to make room
            self._compact()

        self.chunks.append(chunk)
        self._content_buffer.append(chunk.content)
        self._total_chars += len(chunk.content)

        for token in chunk.tokens:
            self.tokens.append(token)

    def add_token(self, token: StreamToken) -> None:
        """Add a single token to the buffer."""
        if self._total_chars + len(token.text) > self.max_size:
            self._compact()

        self.tokens.append(token)
        self._content_buffer.append(token.text)
        self._total_chars += len(token.text)

    def _compact(self) -> None:
        """Remove old content to stay within size limit."""
        # Remove first half of chunks
        mid = len(self.chunks) // 2
        removed_chars = sum(len(c.content) for c in self.chunks[:mid])
        self.chunks = self.chunks[mid:]
        self._total_chars -= removed_chars
        self._rebuild_content_buffer()

    def _rebuild_content_buffer(self) -> None:
        """Rebuild content buffer from chunks."""
        self._content_buffer = [c.content for c in self.chunks]

    def get_content(self) -> str:
        """Get the full buffered content."""
        return "".join(self._content_buffer)

    def get_recent(self, n_chars: int) -> str:
        """Get the most recent n characters."""
        content = self.get_content()
        return content[-n_chars:] if len(content) > n_chars else content

    def clear(self) -> None:
        """Clear the buffer."""
        self.chunks.clear()
        self.tokens.clear()
        self._content_buffer.clear()
        self._total_chars = 0

    @property
    def size(self) -> int:
        """Current buffer size in characters."""
        return self._total_chars

    @property
    def chunk_count(self) -> int:
        """Number of chunks in buffer."""
        return len(self.chunks)

    @property
    def token_count(self) -> int:
        """Number of tokens in buffer."""
        return len(self.tokens)


class StreamProcessor:
    """Process streaming content with transformations and filters."""

    def __init__(self):
        self.transformers: List[Callable[[str], str]] = []
        self.filters: List[Callable[[str], FilterResult]] = []
        self.callbacks: List[Callable[[StreamChunk], None]] = []
        self.filter_patterns: Dict[str, Tuple[str, FilterAction, str]] = {}

    def add_transformer(self, transformer: Callable[[str], str]) -> None:
        """Add a content transformer."""
        self.transformers.append(transformer)

    def add_filter(self, filter_fn: Callable[[str], FilterResult]) -> None:
        """Add a content filter."""
        self.filters.append(filter_fn)

    def add_callback(self, callback: Callable[[StreamChunk], None]) -> None:
        """Add a callback for each chunk."""
        self.callbacks.append(callback)

    def add_pattern_filter(
        self,
        name: str,
        pattern: str,
        action: FilterAction,
        replacement: str = "",
    ) -> None:
        """Add a regex-based pattern filter."""
        self.filter_patterns[name] = (pattern, action, replacement)

    def process_chunk(self, chunk: StreamChunk) -> Tuple[StreamChunk, List[FilterResult]]:
        """Process a chunk through all transformers and filters."""
        content = chunk.content
        filter_results = []

        # Apply transformers
        for transformer in self.transformers:
            content = transformer(content)

        # Apply pattern filters
        for name, (pattern, action, replacement) in self.filter_patterns.items():
            if re.search(pattern, content):
                if action == FilterAction.BLOCK:
                    filter_results.append(FilterResult(
                        action=action,
                        original=content,
                        filtered="",
                        reason="Blocked by pattern",
                        rule_triggered=name,
                    ))
                    content = ""
                elif action == FilterAction.REPLACE:
                    new_content = re.sub(pattern, replacement, content)
                    filter_results.append(FilterResult(
                        action=action,
                        original=content,
                        filtered=new_content,
                        reason="Replaced by pattern",
                        rule_triggered=name,
                    ))
                    content = new_content
                elif action == FilterAction.WARN:
                    filter_results.append(FilterResult(
                        action=action,
                        original=content,
                        filtered=content,
                        reason="Warning: pattern matched",
                        rule_triggered=name,
                    ))

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
    """Aggregate streaming content with various strategies."""

    def __init__(self, strategy: str = "concatenate"):
        self.strategy = strategy
        self.buffers: Dict[str, StreamBuffer] = {}
        self.default_buffer = StreamBuffer()

    def add(self, chunk: StreamChunk, stream_id: str = "default") -> None:
        """Add a chunk to the appropriate buffer."""
        if stream_id not in self.buffers:
            self.buffers[stream_id] = StreamBuffer()
        self.buffers[stream_id].add_chunk(chunk)

    def get_aggregated(self, stream_id: str = "default") -> str:
        """Get aggregated content for a stream."""
        buffer = self.buffers.get(stream_id)
        if not buffer:
            return ""
        return buffer.get_content()

    def get_all_aggregated(self) -> Dict[str, str]:
        """Get aggregated content for all streams."""
        return {
            stream_id: buffer.get_content()
            for stream_id, buffer in self.buffers.items()
        }

    def merge_streams(self, stream_ids: List[str], separator: str = "\n") -> str:
        """Merge multiple streams into one."""
        contents = []
        for stream_id in stream_ids:
            content = self.get_aggregated(stream_id)
            if content:
                contents.append(content)
        return separator.join(contents)

    def clear(self, stream_id: Optional[str] = None) -> None:
        """Clear buffer(s)."""
        if stream_id:
            if stream_id in self.buffers:
                self.buffers[stream_id].clear()
        else:
            for buffer in self.buffers.values():
                buffer.clear()


class StreamCollector:
    """Collect and analyze streaming output."""

    def __init__(self):
        self.buffer = StreamBuffer()
        self.metrics = StreamMetrics(start_time=time.time())
        self.processor: Optional[StreamProcessor] = None
        self.state = StreamState.IDLE
        self._last_token_time: Optional[float] = None
        self.errors: List[str] = []
        self.filtered_count = 0

    def set_processor(self, processor: StreamProcessor) -> None:
        """Set the stream processor."""
        self.processor = processor

    def start(self) -> None:
        """Start collecting."""
        self.state = StreamState.STREAMING
        self.metrics = StreamMetrics(start_time=time.time())
        self._last_token_time = time.time()

    def collect_chunk(self, content: str, event_type: StreamEventType = StreamEventType.CHUNK) -> StreamChunk:
        """Collect a chunk of content."""
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
        """Collect a single token."""
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
        """Record an error during streaming."""
        self.errors.append(error)
        self.metrics.error_count += 1

    def complete(self) -> StreamSummary:
        """Complete collection and return summary."""
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
        """Get current collected content."""
        return self.buffer.get_content()

    def get_metrics(self) -> StreamMetrics:
        """Get current metrics."""
        return self.metrics


class StreamIterator:
    """Iterate over streaming content with processing."""

    def __init__(
        self,
        source: Iterator[str],
        processor: Optional[StreamProcessor] = None,
        collector: Optional[StreamCollector] = None,
    ):
        self.source = source
        self.processor = processor
        self.collector = collector or StreamCollector()
        if processor:
            self.collector.set_processor(processor)
        self._index = 0

    def __iter__(self) -> "StreamIterator":
        self.collector.start()
        return self

    def __next__(self) -> StreamChunk:
        try:
            content = next(self.source)
            chunk = self.collector.collect_chunk(content)
            self._index += 1
            return chunk
        except StopIteration:
            self.collector.complete()
            raise

    def get_summary(self) -> StreamSummary:
        """Get collection summary."""
        return self.collector.complete()


class AsyncStreamIterator:
    """Async iterator over streaming content."""

    def __init__(
        self,
        source: AsyncIterator[str],
        processor: Optional[StreamProcessor] = None,
        collector: Optional[StreamCollector] = None,
    ):
        self.source = source
        self.processor = processor
        self.collector = collector or StreamCollector()
        if processor:
            self.collector.set_processor(processor)
        self._index = 0

    def __aiter__(self) -> "AsyncStreamIterator":
        self.collector.start()
        return self

    async def __anext__(self) -> StreamChunk:
        try:
            content = await self.source.__anext__()
            chunk = self.collector.collect_chunk(content)
            self._index += 1
            return chunk
        except StopAsyncIteration:
            self.collector.complete()
            raise

    def get_summary(self) -> StreamSummary:
        """Get collection summary."""
        return self.collector.complete()


class ContentDetector:
    """Detect specific content patterns in streams."""

    def __init__(self):
        self.patterns: Dict[str, str] = {}
        self.detections: List[Dict[str, Any]] = []
        self._buffer = ""
        self._buffer_size = 10000

    def add_pattern(self, name: str, pattern: str) -> None:
        """Add a pattern to detect."""
        self.patterns[name] = pattern

    def check(self, content: str) -> List[Dict[str, Any]]:
        """Check content for patterns."""
        # Add to buffer
        self._buffer += content
        if len(self._buffer) > self._buffer_size:
            self._buffer = self._buffer[-self._buffer_size:]

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

    def get_all_detections(self) -> List[Dict[str, Any]]:
        """Get all detections."""
        return self.detections

    def clear(self) -> None:
        """Clear buffer and detections."""
        self._buffer = ""
        self.detections.clear()


class StreamingWindowAnalyzer:
    """Analyze streaming content with sliding windows."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.window: deque = deque(maxlen=window_size)
        self.analyzers: List[Callable[[List[str]], Dict[str, Any]]] = []

    def add_analyzer(self, analyzer: Callable[[List[str]], Dict[str, Any]]) -> None:
        """Add a window analyzer function."""
        self.analyzers.append(analyzer)

    def add_token(self, token: str) -> Dict[str, Any]:
        """Add a token and run analysis."""
        self.window.append(token)

        results = {}
        for i, analyzer in enumerate(self.analyzers):
            result = analyzer(list(self.window))
            results[f"analyzer_{i}"] = result

        return results

    def get_window_content(self) -> str:
        """Get current window content as string."""
        return "".join(self.window)

    def clear(self) -> None:
        """Clear the window."""
        self.window.clear()


class StreamRateLimiter:
    """Rate limit streaming output."""

    def __init__(
        self,
        tokens_per_second: float = 50.0,
        burst_size: int = 10,
    ):
        self.tokens_per_second = tokens_per_second
        self.burst_size = burst_size
        self._tokens_available = float(burst_size)
        self._last_time = time.time()

    def acquire(self) -> float:
        """Acquire permission to emit a token, return wait time."""
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
        """Wait if necessary and acquire permission."""
        wait_time = self.acquire()
        if wait_time > 0:
            time.sleep(wait_time)


# Convenience functions

def create_stream_collector() -> StreamCollector:
    """Create a new stream collector."""
    return StreamCollector()


def create_stream_processor() -> StreamProcessor:
    """Create a new stream processor."""
    return StreamProcessor()


def collect_stream(
    stream: Iterator[str],
    processor: Optional[StreamProcessor] = None,
) -> StreamSummary:
    """Collect an entire stream and return summary."""
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
    """Iterate over a stream with processing."""
    collector = StreamCollector()
    if processor:
        collector.set_processor(processor)

    collector.start()
    for content in stream:
        yield collector.collect_chunk(content)

    collector.complete()


def add_content_filter(
    processor: StreamProcessor,
    patterns: Dict[str, str],
    action: FilterAction = FilterAction.REPLACE,
    replacement: str = "[FILTERED]",
) -> None:
    """Add content filters to a processor."""
    for name, pattern in patterns.items():
        processor.add_pattern_filter(name, pattern, action, replacement)


def measure_stream_speed(
    stream: Iterator[str],
    sample_size: int = 100,
) -> Dict[str, float]:
    """Measure the speed of a stream."""
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
    patterns: Dict[str, str],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Detect patterns in a stream and return content with detections."""
    detector = ContentDetector()
    for name, pattern in patterns.items():
        detector.add_pattern(name, pattern)

    content_parts = []
    for chunk in stream:
        content_parts.append(chunk)
        detector.check(chunk)

    return "".join(content_parts), detector.get_all_detections()
