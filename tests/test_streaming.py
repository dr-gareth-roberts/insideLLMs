"""Tests for the streaming module."""

import pytest
import time
from insideLLMs.streaming import (
    # Enums
    StreamEventType,
    StreamState,
    FilterAction,
    # Dataclasses
    StreamToken,
    StreamChunk,
    StreamMetrics,
    FilterResult,
    StreamSummary,
    # Classes
    StreamBuffer,
    StreamProcessor,
    StreamAggregator,
    StreamCollector,
    StreamIterator,
    ContentDetector,
    StreamingWindowAnalyzer,
    StreamRateLimiter,
    # Convenience functions
    create_stream_collector,
    create_stream_processor,
    collect_stream,
    iterate_stream,
    add_content_filter,
    measure_stream_speed,
    detect_in_stream,
)


# ============================================================================
# Enum Tests
# ============================================================================

class TestStreamEnums:
    """Test streaming-related enums."""

    def test_stream_event_type_values(self):
        assert StreamEventType.TOKEN.value == "token"
        assert StreamEventType.CHUNK.value == "chunk"
        assert StreamEventType.START.value == "start"
        assert StreamEventType.END.value == "end"
        assert StreamEventType.ERROR.value == "error"

    def test_stream_state_values(self):
        assert StreamState.IDLE.value == "idle"
        assert StreamState.STREAMING.value == "streaming"
        assert StreamState.PAUSED.value == "paused"
        assert StreamState.COMPLETED.value == "completed"
        assert StreamState.ERROR.value == "error"

    def test_filter_action_values(self):
        assert FilterAction.ALLOW.value == "allow"
        assert FilterAction.BLOCK.value == "block"
        assert FilterAction.REPLACE.value == "replace"
        assert FilterAction.WARN.value == "warn"


# ============================================================================
# Dataclass Tests
# ============================================================================

class TestStreamToken:
    """Test StreamToken dataclass."""

    def test_creation(self):
        token = StreamToken(
            text="hello",
            index=0,
            timestamp=1000.0,
        )
        assert token.text == "hello"
        assert token.index == 0

    def test_with_logprob(self):
        token = StreamToken(
            text="world",
            index=1,
            timestamp=1001.0,
            logprob=-0.5,
            token_id=12345,
        )
        assert token.logprob == -0.5
        assert token.token_id == 12345

    def test_to_dict(self):
        token = StreamToken(
            text="test",
            index=0,
            timestamp=1000.0,
        )
        d = token.to_dict()
        assert d["text"] == "test"
        assert "timestamp" in d


class TestStreamChunk:
    """Test StreamChunk dataclass."""

    def test_creation(self):
        chunk = StreamChunk(
            content="Hello world",
            event_type=StreamEventType.CHUNK,
            timestamp=1000.0,
            index=0,
        )
        assert chunk.content == "Hello world"
        assert chunk.event_type == StreamEventType.CHUNK

    def test_with_tokens(self):
        tokens = [
            StreamToken("Hello", 0, 1000.0),
            StreamToken(" world", 1, 1001.0),
        ]
        chunk = StreamChunk(
            content="Hello world",
            event_type=StreamEventType.CHUNK,
            timestamp=1000.0,
            index=0,
            tokens=tokens,
        )
        assert len(chunk.tokens) == 2

    def test_to_dict(self):
        chunk = StreamChunk(
            content="test",
            event_type=StreamEventType.TOKEN,
            timestamp=1000.0,
            index=0,
        )
        d = chunk.to_dict()
        assert d["event_type"] == "token"


class TestStreamMetrics:
    """Test StreamMetrics dataclass."""

    def test_creation(self):
        metrics = StreamMetrics(start_time=1000.0)
        assert metrics.start_time == 1000.0
        assert metrics.total_tokens == 0

    def test_duration(self):
        metrics = StreamMetrics(
            start_time=1000.0,
            end_time=1010.0,
        )
        assert metrics.duration == 10.0

    def test_tokens_per_second(self):
        metrics = StreamMetrics(
            start_time=1000.0,
            end_time=1010.0,
            total_tokens=100,
        )
        assert metrics.tokens_per_second == 10.0

    def test_mean_inter_token_time(self):
        metrics = StreamMetrics(
            start_time=0.0,
            inter_token_times=[0.1, 0.2, 0.3],
        )
        assert abs(metrics.mean_inter_token_time - 0.2) < 0.001

    def test_to_dict(self):
        metrics = StreamMetrics(
            start_time=1000.0,
            end_time=1010.0,
            total_tokens=50,
        )
        d = metrics.to_dict()
        assert d["tokens_per_second"] == 5.0


class TestFilterResult:
    """Test FilterResult dataclass."""

    def test_creation(self):
        result = FilterResult(
            action=FilterAction.REPLACE,
            original="bad word",
            filtered="[FILTERED]",
            reason="Content filtered",
        )
        assert result.action == FilterAction.REPLACE
        assert result.filtered == "[FILTERED]"

    def test_to_dict(self):
        result = FilterResult(
            action=FilterAction.BLOCK,
            original="blocked",
            filtered="",
            rule_triggered="profanity",
        )
        d = result.to_dict()
        assert d["action"] == "block"
        assert d["rule_triggered"] == "profanity"


class TestStreamSummary:
    """Test StreamSummary dataclass."""

    def test_creation(self):
        metrics = StreamMetrics(start_time=1000.0, end_time=1010.0)
        summary = StreamSummary(
            full_content="Hello world",
            metrics=metrics,
            chunk_count=2,
            token_count=2,
            state=StreamState.COMPLETED,
        )
        assert summary.full_content == "Hello world"
        assert summary.state == StreamState.COMPLETED


# ============================================================================
# StreamBuffer Tests
# ============================================================================

class TestStreamBuffer:
    """Test StreamBuffer class."""

    def test_add_chunk(self):
        buffer = StreamBuffer()
        chunk = StreamChunk("Hello", StreamEventType.CHUNK, 1000.0, 0)
        buffer.add_chunk(chunk)

        assert buffer.chunk_count == 1
        assert buffer.get_content() == "Hello"

    def test_add_multiple_chunks(self):
        buffer = StreamBuffer()
        buffer.add_chunk(StreamChunk("Hello", StreamEventType.CHUNK, 1000.0, 0))
        buffer.add_chunk(StreamChunk(" ", StreamEventType.CHUNK, 1001.0, 1))
        buffer.add_chunk(StreamChunk("world", StreamEventType.CHUNK, 1002.0, 2))

        assert buffer.get_content() == "Hello world"
        assert buffer.chunk_count == 3

    def test_add_token(self):
        buffer = StreamBuffer()
        buffer.add_token(StreamToken("Hello", 0, 1000.0))
        buffer.add_token(StreamToken(" world", 1, 1001.0))

        assert buffer.token_count == 2
        assert buffer.get_content() == "Hello world"

    def test_get_recent(self):
        buffer = StreamBuffer()
        buffer.add_chunk(StreamChunk("Hello world", StreamEventType.CHUNK, 1000.0, 0))

        recent = buffer.get_recent(5)
        assert recent == "world"

    def test_clear(self):
        buffer = StreamBuffer()
        buffer.add_chunk(StreamChunk("test", StreamEventType.CHUNK, 1000.0, 0))
        buffer.clear()

        assert buffer.size == 0
        assert buffer.get_content() == ""

    def test_max_size_compaction(self):
        buffer = StreamBuffer(max_size=100)

        # Add content that exceeds max size
        for i in range(20):
            buffer.add_chunk(StreamChunk("x" * 10, StreamEventType.CHUNK, 1000.0 + i, i))

        # Buffer should have compacted
        assert buffer.size <= 100


# ============================================================================
# StreamProcessor Tests
# ============================================================================

class TestStreamProcessor:
    """Test StreamProcessor class."""

    def test_add_transformer(self):
        processor = StreamProcessor()
        processor.add_transformer(str.upper)

        chunk = StreamChunk("hello", StreamEventType.CHUNK, 1000.0, 0)
        processed, _ = processor.process_chunk(chunk)

        assert processed.content == "HELLO"

    def test_multiple_transformers(self):
        processor = StreamProcessor()
        processor.add_transformer(str.strip)
        processor.add_transformer(str.upper)

        chunk = StreamChunk("  hello  ", StreamEventType.CHUNK, 1000.0, 0)
        processed, _ = processor.process_chunk(chunk)

        assert processed.content == "HELLO"

    def test_pattern_filter_replace(self):
        processor = StreamProcessor()
        processor.add_pattern_filter("bad_word", r"bad", FilterAction.REPLACE, "[GOOD]")

        chunk = StreamChunk("this is bad content", StreamEventType.CHUNK, 1000.0, 0)
        processed, results = processor.process_chunk(chunk)

        assert "bad" not in processed.content
        assert "[GOOD]" in processed.content
        assert len(results) == 1

    def test_pattern_filter_block(self):
        processor = StreamProcessor()
        processor.add_pattern_filter("block_test", r"block_me", FilterAction.BLOCK, "")

        chunk = StreamChunk("please block_me now", StreamEventType.CHUNK, 1000.0, 0)
        processed, results = processor.process_chunk(chunk)

        assert processed.content == ""
        assert results[0].action == FilterAction.BLOCK

    def test_pattern_filter_warn(self):
        processor = StreamProcessor()
        processor.add_pattern_filter("warning", r"caution", FilterAction.WARN, "")

        chunk = StreamChunk("proceed with caution", StreamEventType.CHUNK, 1000.0, 0)
        processed, results = processor.process_chunk(chunk)

        assert processed.content == "proceed with caution"  # Not modified
        assert results[0].action == FilterAction.WARN

    def test_callback(self):
        processor = StreamProcessor()
        received_chunks = []
        processor.add_callback(lambda c: received_chunks.append(c))

        chunk = StreamChunk("test", StreamEventType.CHUNK, 1000.0, 0)
        processor.process_chunk(chunk)

        assert len(received_chunks) == 1


# ============================================================================
# StreamAggregator Tests
# ============================================================================

class TestStreamAggregator:
    """Test StreamAggregator class."""

    def test_add_and_get(self):
        aggregator = StreamAggregator()
        aggregator.add(StreamChunk("Hello", StreamEventType.CHUNK, 1000.0, 0))
        aggregator.add(StreamChunk(" world", StreamEventType.CHUNK, 1001.0, 1))

        assert aggregator.get_aggregated() == "Hello world"

    def test_multiple_streams(self):
        aggregator = StreamAggregator()
        aggregator.add(StreamChunk("Stream1", StreamEventType.CHUNK, 1000.0, 0), "s1")
        aggregator.add(StreamChunk("Stream2", StreamEventType.CHUNK, 1000.0, 0), "s2")

        assert aggregator.get_aggregated("s1") == "Stream1"
        assert aggregator.get_aggregated("s2") == "Stream2"

    def test_get_all_aggregated(self):
        aggregator = StreamAggregator()
        aggregator.add(StreamChunk("A", StreamEventType.CHUNK, 1000.0, 0), "a")
        aggregator.add(StreamChunk("B", StreamEventType.CHUNK, 1000.0, 0), "b")

        all_content = aggregator.get_all_aggregated()
        assert "a" in all_content
        assert "b" in all_content

    def test_merge_streams(self):
        aggregator = StreamAggregator()
        aggregator.add(StreamChunk("First", StreamEventType.CHUNK, 1000.0, 0), "s1")
        aggregator.add(StreamChunk("Second", StreamEventType.CHUNK, 1001.0, 0), "s2")

        merged = aggregator.merge_streams(["s1", "s2"], separator=" | ")
        assert merged == "First | Second"

    def test_clear(self):
        aggregator = StreamAggregator()
        aggregator.add(StreamChunk("test", StreamEventType.CHUNK, 1000.0, 0), "s1")
        aggregator.clear("s1")

        assert aggregator.get_aggregated("s1") == ""


# ============================================================================
# StreamCollector Tests
# ============================================================================

class TestStreamCollector:
    """Test StreamCollector class."""

    def test_collect_chunk(self):
        collector = StreamCollector()
        collector.start()
        chunk = collector.collect_chunk("Hello")

        assert chunk.content == "Hello"
        assert collector.metrics.total_chunks == 1

    def test_collect_multiple_chunks(self):
        collector = StreamCollector()
        collector.start()
        collector.collect_chunk("Hello")
        collector.collect_chunk(" world")

        assert collector.get_current_content() == "Hello world"
        assert collector.metrics.total_chunks == 2

    def test_collect_token(self):
        collector = StreamCollector()
        collector.start()
        token = collector.collect_token("Hello", logprob=-0.5)

        assert token.text == "Hello"
        assert token.logprob == -0.5
        assert collector.metrics.total_tokens == 1

    def test_complete(self):
        collector = StreamCollector()
        collector.start()
        collector.collect_chunk("Test content")
        summary = collector.complete()

        assert summary.state == StreamState.COMPLETED
        assert summary.full_content == "Test content"
        assert summary.metrics.end_time is not None

    def test_with_processor(self):
        processor = StreamProcessor()
        processor.add_transformer(str.upper)

        collector = StreamCollector()
        collector.set_processor(processor)
        collector.start()
        chunk = collector.collect_chunk("hello")

        assert chunk.content == "HELLO"

    def test_record_error(self):
        collector = StreamCollector()
        collector.start()
        collector.record_error("Connection lost")
        summary = collector.complete()

        assert len(summary.errors) == 1
        assert summary.metrics.error_count == 1

    def test_auto_start(self):
        collector = StreamCollector()
        # Don't call start()
        collector.collect_chunk("Auto started")

        assert collector.state == StreamState.STREAMING


# ============================================================================
# StreamIterator Tests
# ============================================================================

class TestStreamIterator:
    """Test StreamIterator class."""

    def test_iteration(self):
        source = iter(["Hello", " ", "world"])
        iterator = StreamIterator(source)

        chunks = list(iterator)
        assert len(chunks) == 3
        assert "".join(c.content for c in chunks) == "Hello world"

    def test_with_processor(self):
        source = iter(["hello", " ", "world"])
        processor = StreamProcessor()
        processor.add_transformer(str.upper)

        iterator = StreamIterator(source, processor)
        chunks = list(iterator)

        assert chunks[0].content == "HELLO"

    def test_get_summary(self):
        source = iter(["test", "content"])
        iterator = StreamIterator(source)

        # Consume iterator
        list(iterator)

        summary = iterator.get_summary()
        assert summary.state == StreamState.COMPLETED


# ============================================================================
# ContentDetector Tests
# ============================================================================

class TestContentDetector:
    """Test ContentDetector class."""

    def test_add_pattern(self):
        detector = ContentDetector()
        detector.add_pattern("email", r"\S+@\S+\.\S+")
        assert "email" in detector.patterns

    def test_detect_pattern(self):
        detector = ContentDetector()
        detector.add_pattern("email", r"\S+@\S+\.\S+")

        detections = detector.check("Contact us at test@example.com")

        assert len(detections) == 1
        assert detections[0]["pattern_name"] == "email"
        assert "test@example.com" in detections[0]["match"]

    def test_multiple_patterns(self):
        detector = ContentDetector()
        detector.add_pattern("email", r"\S+@\S+\.\S+")
        detector.add_pattern("phone", r"\d{3}-\d{3}-\d{4}")

        detector.check("Email: test@test.com, Phone: 123-456-7890")
        detections = detector.get_all_detections()

        pattern_names = [d["pattern_name"] for d in detections]
        assert "email" in pattern_names
        assert "phone" in pattern_names

    def test_clear(self):
        detector = ContentDetector()
        detector.add_pattern("test", r"test")
        detector.check("test content")
        detector.clear()

        assert len(detector.detections) == 0


# ============================================================================
# StreamingWindowAnalyzer Tests
# ============================================================================

class TestStreamingWindowAnalyzer:
    """Test StreamingWindowAnalyzer class."""

    def test_add_token(self):
        analyzer = StreamingWindowAnalyzer(window_size=5)
        analyzer.add_token("hello")

        assert analyzer.get_window_content() == "hello"

    def test_window_size_limit(self):
        analyzer = StreamingWindowAnalyzer(window_size=3)
        analyzer.add_token("a")
        analyzer.add_token("b")
        analyzer.add_token("c")
        analyzer.add_token("d")

        # Only last 3 should remain
        assert analyzer.get_window_content() == "bcd"

    def test_analyzer_callback(self):
        analyzer = StreamingWindowAnalyzer(window_size=5)

        def count_analyzer(tokens):
            return {"count": len(tokens)}

        analyzer.add_analyzer(count_analyzer)
        result = analyzer.add_token("x")

        assert result["analyzer_0"]["count"] == 1

    def test_clear(self):
        analyzer = StreamingWindowAnalyzer()
        analyzer.add_token("test")
        analyzer.clear()

        assert analyzer.get_window_content() == ""


# ============================================================================
# StreamRateLimiter Tests
# ============================================================================

class TestStreamRateLimiter:
    """Test StreamRateLimiter class."""

    def test_acquire_immediate(self):
        limiter = StreamRateLimiter(tokens_per_second=1000, burst_size=10)
        wait_time = limiter.acquire()
        assert wait_time == 0.0

    def test_burst_size(self):
        limiter = StreamRateLimiter(tokens_per_second=1.0, burst_size=3)

        # First 3 should be immediate (burst)
        for _ in range(3):
            wait_time = limiter.acquire()
            assert wait_time == 0.0

        # 4th should require waiting
        wait_time = limiter.acquire()
        assert wait_time > 0

    def test_replenishment(self):
        limiter = StreamRateLimiter(tokens_per_second=1000.0, burst_size=1)

        # Use the burst token
        limiter.acquire()

        # Wait a bit for replenishment
        time.sleep(0.01)

        # Should be able to acquire again
        wait_time = limiter.acquire()
        assert wait_time == 0.0


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_create_stream_collector(self):
        collector = create_stream_collector()
        assert isinstance(collector, StreamCollector)

    def test_create_stream_processor(self):
        processor = create_stream_processor()
        assert isinstance(processor, StreamProcessor)

    def test_collect_stream(self):
        source = iter(["Hello", " ", "world", "!"])
        summary = collect_stream(source)

        assert summary.full_content == "Hello world!"
        assert summary.chunk_count == 4

    def test_collect_stream_with_processor(self):
        source = iter(["hello", " ", "world"])
        processor = StreamProcessor()
        processor.add_transformer(str.upper)

        summary = collect_stream(source, processor)
        assert summary.full_content == "HELLO WORLD"

    def test_iterate_stream(self):
        source = iter(["a", "b", "c"])
        chunks = list(iterate_stream(source))

        assert len(chunks) == 3
        assert all(isinstance(c, StreamChunk) for c in chunks)

    def test_add_content_filter(self):
        processor = StreamProcessor()
        patterns = {"secret": r"secret\d+"}
        add_content_filter(processor, patterns, FilterAction.REPLACE, "[REDACTED]")

        chunk = StreamChunk("my secret123 code", StreamEventType.CHUNK, 1000.0, 0)
        processed, _ = processor.process_chunk(chunk)

        assert "secret123" not in processed.content
        assert "[REDACTED]" in processed.content

    def test_measure_stream_speed(self):
        # Create a simple stream
        def slow_stream():
            for i in range(10):
                yield f"token{i}"

        speed = measure_stream_speed(slow_stream(), sample_size=10)

        assert "tokens_per_second" in speed
        assert speed["sample_size"] == 10

    def test_detect_in_stream(self):
        source = iter(["Email: ", "test@example.com", " end"])
        patterns = {"email": r"\S+@\S+\.\S+"}

        content, detections = detect_in_stream(source, patterns)

        assert "test@example.com" in content
        assert len(detections) > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for streaming workflows."""

    def test_full_streaming_workflow(self):
        # Set up processor with filter
        processor = StreamProcessor()
        processor.add_pattern_filter("pii", r"\d{3}-\d{2}-\d{4}", FilterAction.REPLACE, "[SSN]")
        processor.add_transformer(str.strip)

        # Set up collector
        collector = StreamCollector()
        collector.set_processor(processor)

        # Simulate streaming
        stream_data = [
            "User info: ",
            "Name: John, ",
            "SSN: 123-45-6789, ",
            "Age: 30",
        ]

        collector.start()
        for data in stream_data:
            collector.collect_chunk(data)
        summary = collector.complete()

        # Verify filtering worked
        assert "123-45-6789" not in summary.full_content
        assert "[SSN]" in summary.full_content
        assert summary.state == StreamState.COMPLETED

    def test_multi_stream_aggregation(self):
        aggregator = StreamAggregator()

        # Simulate multiple concurrent streams
        for i in range(3):
            stream_id = f"stream_{i}"
            for j in range(3):
                chunk = StreamChunk(
                    f"[{stream_id}:{j}]",
                    StreamEventType.CHUNK,
                    1000.0 + j,
                    j,
                )
                aggregator.add(chunk, stream_id)

        # Verify each stream
        for i in range(3):
            content = aggregator.get_aggregated(f"stream_{i}")
            assert f"stream_{i}" in content

        # Merge all streams
        merged = aggregator.merge_streams(["stream_0", "stream_1", "stream_2"])
        assert "stream_0" in merged
        assert "stream_2" in merged

    def test_windowed_analysis(self):
        analyzer = StreamingWindowAnalyzer(window_size=10)

        # Add custom analyzer
        def word_count(tokens):
            text = "".join(tokens)
            return {"word_count": len(text.split())}

        analyzer.add_analyzer(word_count)

        # Add tokens
        for token in ["Hello", " ", "world", " ", "how", " ", "are", " ", "you"]:
            result = analyzer.add_token(token)

        # Check analysis
        assert result["analyzer_0"]["word_count"] > 0


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_stream(self):
        source = iter([])
        summary = collect_stream(source)

        assert summary.full_content == ""
        assert summary.chunk_count == 0

    def test_empty_chunk(self):
        collector = StreamCollector()
        collector.start()
        chunk = collector.collect_chunk("")

        assert chunk.content == ""

    def test_very_long_content(self):
        collector = StreamCollector()
        collector.start()

        # Add large content
        large_content = "x" * 100000
        collector.collect_chunk(large_content)

        summary = collector.complete()
        assert len(summary.full_content) == 100000

    def test_metrics_no_tokens(self):
        metrics = StreamMetrics(start_time=0.0)
        assert metrics.tokens_per_second == 0.0
        assert metrics.mean_inter_token_time == 0.0

    def test_filter_result_no_match(self):
        processor = StreamProcessor()
        processor.add_pattern_filter("test", r"nomatch", FilterAction.REPLACE, "")

        chunk = StreamChunk("hello world", StreamEventType.CHUNK, 1000.0, 0)
        processed, results = processor.process_chunk(chunk)

        # No matches, no filter results
        assert processed.content == "hello world"
        assert len(results) == 0

    def test_rate_limiter_zero_rate(self):
        # Very low rate
        limiter = StreamRateLimiter(tokens_per_second=0.001, burst_size=1)
        limiter.acquire()  # Use burst

        wait_time = limiter.acquire()
        assert wait_time > 0
