"""Additional branch coverage for streaming internals."""

from __future__ import annotations

import pytest

from insideLLMs.streaming import (
    AsyncStreamIterator,
    ContentDetector,
    FilterAction,
    FilterResult,
    StreamAggregator,
    StreamBuffer,
    StreamChunk,
    StreamCollector,
    StreamEventType,
    StreamMetrics,
    StreamProcessor,
    StreamRateLimiter,
    StreamState,
    StreamSummary,
    StreamToken,
    iterate_stream,
    measure_stream_speed,
)


def test_stream_metrics_negative_duration_and_summary_to_dict():
    metrics = StreamMetrics(start_time=10.0, end_time=5.0, total_tokens=100)
    assert metrics.tokens_per_second == 0.0

    summary = StreamSummary(
        full_content="payload",
        metrics=metrics,
        chunk_count=1,
        token_count=2,
        state=StreamState.COMPLETED,
        errors=["err"],
        filtered_count=1,
        metadata={"k": "v"},
    )
    data = summary.to_dict()
    assert data["state"] == "completed"
    assert data["metadata"] == {"k": "v"}
    assert data["metrics"]["tokens_per_second"] == 0.0


def test_stream_buffer_compaction_and_chunk_token_copy_paths():
    buffer = StreamBuffer(max_size=4)
    tokenized_chunk = StreamChunk(
        content="ab",
        event_type=StreamEventType.CHUNK,
        timestamp=1.0,
        index=0,
        tokens=[StreamToken("x", 0, 1.0), StreamToken("y", 1, 1.0)],
    )
    buffer.add_chunk(tokenized_chunk)
    assert [t.text for t in buffer.tokens] == ["x", "y"]

    # Force chunk compaction branch.
    buffer.add_chunk(StreamChunk("xyz", StreamEventType.CHUNK, 2.0, 1))

    token_buffer = StreamBuffer(max_size=3)
    token_buffer.add_token(StreamToken("abc", 0, 1.0))
    # Force add_token compaction branch.
    token_buffer.add_token(StreamToken("d", 1, 2.0))
    assert token_buffer.token_count == 2


def test_stream_processor_custom_filter_addition_and_actions():
    processor = StreamProcessor()
    processor.add_filter(
        lambda content: FilterResult(
            action=FilterAction.REPLACE,
            original=content,
            filtered=content.replace("secret", "[REDACTED]"),
            reason="redact",
            rule_triggered="custom_replace",
        )
    )
    processor.add_filter(
        lambda content: FilterResult(
            action=FilterAction.BLOCK if "ban" in content else FilterAction.ALLOW,
            original=content,
            filtered="" if "ban" in content else content,
            reason="block banned term" if "ban" in content else "allow",
            rule_triggered="custom_block",
        )
    )

    chunk = StreamChunk("secret ban data", StreamEventType.CHUNK, 1.0, 0)
    processed, results = processor.process_chunk(chunk)
    assert processed.content == ""
    assert len(results) == 2
    assert results[0].action == FilterAction.REPLACE
    assert results[1].action == FilterAction.BLOCK


def test_stream_aggregator_missing_stream_merge_and_clear_all():
    aggregator = StreamAggregator()
    aggregator.add(StreamChunk("first", StreamEventType.CHUNK, 1.0, 0), "s1")
    aggregator.add(StreamChunk("second", StreamEventType.CHUNK, 1.0, 0), "s2")

    assert aggregator.get_aggregated("missing") == ""
    merged = aggregator.merge_streams(["missing", "s1"], separator="|")
    assert merged == "first"

    aggregator.clear()
    assert aggregator.get_aggregated("s1") == ""
    assert aggregator.get_aggregated("s2") == ""


def test_stream_collector_collect_token_auto_start_and_metrics_access():
    collector = StreamCollector()
    assert collector.state == StreamState.IDLE
    collector.collect_token("x")
    assert collector.state == StreamState.STREAMING

    # Exercise false branch of inter-token-time update checks.
    collector.start()
    collector._last_token_time = 0.0
    collector.collect_chunk("chunk")
    collector._last_token_time = 0.0
    collector.collect_token("y")

    metrics = collector.get_metrics()
    assert metrics.total_chunks >= 1
    assert metrics.total_tokens >= 1


@pytest.mark.asyncio
async def test_async_stream_iterator_processor_and_stop_iteration_paths():
    async def source():
        yield "a"
        yield "b"

    processor = StreamProcessor()
    processor.add_transformer(str.upper)

    iterator = AsyncStreamIterator(source(), processor=processor)
    assert iterator.processor is processor

    aiter = iterator.__aiter__()
    first = await aiter.__anext__()
    second = await aiter.__anext__()
    assert first.content == "A"
    assert second.content == "B"

    with pytest.raises(StopAsyncIteration):
        await aiter.__anext__()

    summary = iterator.get_summary()
    assert summary.full_content == "AB"


def test_content_detector_buffer_truncation_branch():
    detector = ContentDetector()
    detector.add_pattern("digits", r"\d+")
    detector._buffer_size = 5

    detector.check("12345")
    detections = detector.check("67890")
    assert detector._buffer == "67890"
    assert len(detections) >= 1


def test_rate_limiter_wait_and_acquire_sleep_branch(monkeypatch):
    limiter = StreamRateLimiter(tokens_per_second=10.0, burst_size=1)
    monkeypatch.setattr(limiter, "acquire", lambda: 0.02)

    slept: list[float] = []
    monkeypatch.setattr("insideLLMs.streaming.time.sleep", lambda value: slept.append(value))

    limiter.wait_and_acquire()
    assert slept == [0.02]


def test_iterate_stream_processor_branch_and_measure_speed_break_paths():
    processor = StreamProcessor()
    processor.add_transformer(str.upper)
    chunks = list(iterate_stream(iter(["hi"]), processor=processor))
    assert chunks[0].content == "HI"

    metrics = measure_stream_speed(iter(["a", "b", "c", "d"]), sample_size=2)
    assert metrics["sample_size"] == 2

    # Also cover empty stream path through the loop.
    empty_metrics = measure_stream_speed(iter([]), sample_size=3)
    assert empty_metrics["sample_size"] == 0
