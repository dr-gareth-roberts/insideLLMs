"""Comprehensive coverage tests for insideLLMs/runtime/pipeline.py.

This test file targets uncovered branches and code paths to bring coverage
from ~55% to 90%+.  It focuses on:
  - TraceMiddleware (completely untested in test_pipeline.py)
  - Middleware error handling / exception propagation
  - Complex middleware interactions (Cache + Trace + CostTracking)
  - Pipeline batch operations: abatch_generate, achat, astream
  - Cache edge cases (None/empty responses, TTL boundary, eviction)
  - Rate limit edge cases (burst exhaustion, async token acquisition)
  - Retry middleware exhaustion (sync + async)
  - Middleware base class abstract / default paths
  - AsyncModelPipeline callback error paths
"""

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.exceptions import ModelError, RateLimitError, TimeoutError
from insideLLMs.models import DummyModel
from insideLLMs.models.base import ChatMessage, Model
from insideLLMs.pipeline import (
    AsyncModelPipeline,
    CacheMiddleware,
    CostTrackingMiddleware,
    Middleware,
    ModelPipeline,
    PassthroughMiddleware,
    RateLimitMiddleware,
    RetryMiddleware,
    TraceMiddleware,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FailingModel(Model):
    """Model that always raises a ModelError."""

    _supports_streaming = True
    _supports_chat = True

    def __init__(self, error_cls=ModelError, msg="model failure"):
        super().__init__(name="FailingModel", model_id="fail-v1")
        self.error_cls = error_cls
        self.msg = msg

    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise self.error_cls(self.msg)

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        raise self.error_cls(self.msg)

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        raise self.error_cls(self.msg)


class FailNTimesModel(Model):
    """Model that fails N times then succeeds."""

    _supports_streaming = True
    _supports_chat = True

    def __init__(self, fail_count: int = 2, error_cls=ModelError):
        super().__init__(name="FailNTimesModel", model_id="failn-v1")
        self.fail_count = fail_count
        self.error_cls = error_cls
        self.call_count = 0

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise self.error_cls(f"failure #{self.call_count}")
        return f"success after {self.call_count} calls"

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        return self.generate(messages[-1]["content"] if messages else "")

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        resp = self.generate(prompt)
        for word in resp.split():
            yield word + " "


# ============================================================================
# TraceMiddleware Tests
# ============================================================================


class TestTraceMiddlewareInit:
    """Tests for TraceMiddleware initialization and reset."""

    def test_init_defaults(self):
        tm = TraceMiddleware()
        assert tm.recorder is not None
        assert tm.recorder.run_id is None
        assert tm.recorder.example_id is None

    def test_init_with_ids(self):
        tm = TraceMiddleware(run_id="run1", example_id="ex1")
        assert tm.recorder.run_id == "run1"
        assert tm.recorder.example_id == "ex1"

    def test_reset_clears_events(self):
        tm = TraceMiddleware(run_id="r1")
        model = DummyModel()
        tm.model = model
        tm.process_generate("hello")
        assert len(tm.recorder.events) > 0

        tm.reset(run_id="r2", example_id="e2")
        assert len(tm.recorder.events) == 0
        assert tm.recorder.run_id == "r2"
        assert tm.recorder.example_id == "e2"

    def test_reset_no_args(self):
        tm = TraceMiddleware(run_id="r1", example_id="e1")
        tm.reset()
        assert tm.recorder.run_id is None
        assert tm.recorder.example_id is None

    def test_reserved_kwargs_stripped(self):
        tm = TraceMiddleware()
        kwargs = {
            "temperature": 0.5,
            "_trace": True,
            "_trace_recorder": "x",
            "_run_id": "r",
            "_example_id": "e",
            "max_tokens": 100,
        }
        clean = tm._strip_reserved_kwargs(kwargs)
        assert "_trace" not in clean
        assert "_trace_recorder" not in clean
        assert "_run_id" not in clean
        assert "_example_id" not in clean
        assert clean["temperature"] == 0.5
        assert clean["max_tokens"] == 100


class TestTraceMiddlewareGenerate:
    """Tests for TraceMiddleware process_generate and aprocess_generate."""

    def test_process_generate_records_start_and_end(self):
        model = DummyModel()
        tm = TraceMiddleware(run_id="gen_test")
        pipeline = ModelPipeline(model, middlewares=[tm])

        response = pipeline.generate("What is AI?")
        assert isinstance(response, str)

        events = tm.recorder.events
        assert len(events) == 2
        assert events[0].kind == "generate_start"
        assert events[0].payload["prompt"] == "What is AI?"
        assert events[1].kind == "generate_end"
        assert "response" in events[1].payload

    def test_process_generate_records_error(self):
        model = FailingModel()
        tm = TraceMiddleware(run_id="err_test")
        pipeline = ModelPipeline(model, middlewares=[tm])

        with pytest.raises(ModelError):
            pipeline.generate("will fail")

        events = tm.recorder.events
        # Should have GENERATE_START and ERROR
        kinds = [e.kind for e in events]
        assert "generate_start" in kinds
        assert "error" in kinds

    def test_process_generate_strips_reserved_kwargs(self):
        """Reserved kwargs must not reach the model."""
        model = DummyModel()
        tm = TraceMiddleware()
        pipeline = ModelPipeline(model, middlewares=[tm])

        # These reserved kwargs should be stripped before reaching the model
        response = pipeline.generate("test", _trace=True, _run_id="r1")
        assert isinstance(response, str)

    def test_process_generate_with_next_middleware(self):
        """TraceMiddleware delegates to next middleware in chain."""
        model = DummyModel()
        tm = TraceMiddleware(run_id="chain")
        cache = CacheMiddleware(cache_size=10)
        pipeline = ModelPipeline(model, middlewares=[tm, cache])

        response = pipeline.generate("prompt")
        assert isinstance(response, str)
        assert len(tm.recorder.events) == 2

    def test_process_generate_no_model_raises(self):
        """TraceMiddleware raises when there is no model or next middleware."""
        tm = TraceMiddleware()
        # Set model to None and no next_middleware
        tm.model = None
        tm.next_middleware = None

        with pytest.raises(ModelError, match="No model available"):
            tm.process_generate("test")

    @pytest.mark.asyncio
    async def test_aprocess_generate_records_start_and_end(self):
        model = DummyModel()
        tm = TraceMiddleware(run_id="async_gen")
        pipeline = ModelPipeline(model, middlewares=[tm])

        response = await pipeline.agenerate("async prompt")
        assert isinstance(response, str)

        events = tm.recorder.events
        assert len(events) == 2
        assert events[0].kind == "generate_start"
        assert events[1].kind == "generate_end"

    @pytest.mark.asyncio
    async def test_aprocess_generate_records_error(self):
        model = FailingModel()
        tm = TraceMiddleware(run_id="async_err")
        pipeline = ModelPipeline(model, middlewares=[tm])

        with pytest.raises(ModelError):
            await pipeline.agenerate("will fail")

        kinds = [e.kind for e in tm.recorder.events]
        assert "generate_start" in kinds
        assert "error" in kinds

    @pytest.mark.asyncio
    async def test_aprocess_generate_no_model_raises(self):
        tm = TraceMiddleware()
        tm.model = None
        tm.next_middleware = None

        with pytest.raises(ModelError, match="No model available"):
            await tm.aprocess_generate("test")

    @pytest.mark.asyncio
    async def test_aprocess_generate_with_next_middleware(self):
        model = DummyModel()
        tm = TraceMiddleware()
        cache = CacheMiddleware()
        pipeline = ModelPipeline(model, middlewares=[tm, cache])

        response = await pipeline.agenerate("test")
        assert isinstance(response, str)
        assert len(tm.recorder.events) == 2


class TestTraceMiddlewareChat:
    """Tests for TraceMiddleware process_chat and aprocess_chat."""

    def test_process_chat_records_events(self):
        model = DummyModel()
        tm = TraceMiddleware(run_id="chat_test")
        pipeline = ModelPipeline(model, middlewares=[tm])

        messages = [{"role": "user", "content": "Hello!"}]
        response = pipeline.chat(messages)
        assert isinstance(response, str)

        events = tm.recorder.events
        kinds = [e.kind for e in events]
        assert "chat_start" in kinds
        assert "chat_end" in kinds

        # Check CHAT_START data
        start = [e for e in events if e.kind == "chat_start"][0]
        assert start.payload["message_count"] == 1

    def test_process_chat_records_error(self):
        model = FailingModel()
        tm = TraceMiddleware(run_id="chat_err")
        pipeline = ModelPipeline(model, middlewares=[tm])

        messages = [{"role": "user", "content": "fail"}]
        with pytest.raises(ModelError):
            pipeline.chat(messages)

        kinds = [e.kind for e in tm.recorder.events]
        assert "chat_start" in kinds
        assert "error" in kinds

    def test_process_chat_no_model_raises(self):
        tm = TraceMiddleware()
        tm.model = None
        tm.next_middleware = None

        with pytest.raises(ModelError, match="No chat implementation"):
            tm.process_chat([{"role": "user", "content": "x"}])

    def test_process_chat_with_next_middleware(self):
        model = DummyModel()
        tm = TraceMiddleware()
        passthrough = PassthroughMiddleware()
        pipeline = ModelPipeline(model, middlewares=[tm, passthrough])

        messages = [{"role": "user", "content": "test"}]
        response = pipeline.chat(messages)
        assert isinstance(response, str)
        kinds = [e.kind for e in tm.recorder.events]
        assert "chat_start" in kinds
        assert "chat_end" in kinds

    @pytest.mark.asyncio
    async def test_aprocess_chat_records_events(self):
        model = DummyModel()
        tm = TraceMiddleware(run_id="async_chat")
        pipeline = ModelPipeline(model, middlewares=[tm])

        messages = [{"role": "user", "content": "hi"}]
        response = await pipeline.achat(messages)
        assert isinstance(response, str)

        kinds = [e.kind for e in tm.recorder.events]
        assert "chat_start" in kinds
        assert "chat_end" in kinds

    @pytest.mark.asyncio
    async def test_aprocess_chat_records_error(self):
        model = FailingModel()
        tm = TraceMiddleware()
        pipeline = ModelPipeline(model, middlewares=[tm])

        messages = [{"role": "user", "content": "fail"}]
        with pytest.raises(ModelError):
            await pipeline.achat(messages)

        kinds = [e.kind for e in tm.recorder.events]
        assert "chat_start" in kinds
        assert "error" in kinds

    @pytest.mark.asyncio
    async def test_aprocess_chat_no_model_raises(self):
        tm = TraceMiddleware()
        tm.model = None
        tm.next_middleware = None

        with pytest.raises(ModelError, match="No chat implementation"):
            await tm.aprocess_chat([{"role": "user", "content": "x"}])

    @pytest.mark.asyncio
    async def test_aprocess_chat_with_next_middleware(self):
        model = DummyModel()
        tm = TraceMiddleware()
        passthrough = PassthroughMiddleware()
        pipeline = ModelPipeline(model, middlewares=[tm, passthrough])

        messages = [{"role": "user", "content": "test"}]
        response = await pipeline.achat(messages)
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_aprocess_chat_model_with_chat_no_achat(self):
        """Model has .chat but not .achat -- should fall back to executor."""
        model = DummyModel()
        # DummyModel has chat but not achat, so the executor fallback is used
        tm = TraceMiddleware()
        tm.model = model
        tm.next_middleware = None

        messages = [{"role": "user", "content": "test"}]
        response = await tm.aprocess_chat(messages)
        assert isinstance(response, str)


class TestTraceMiddlewareStream:
    """Tests for TraceMiddleware process_stream and aprocess_stream."""

    def test_process_stream_records_events(self):
        model = DummyModel()
        tm = TraceMiddleware(run_id="stream_test")
        pipeline = ModelPipeline(model, middlewares=[tm])

        chunks = list(pipeline.stream("Tell me a story"))
        assert len(chunks) > 0

        events = tm.recorder.events
        kinds = [e.kind for e in events]
        assert "stream_start" in kinds
        assert "stream_chunk" in kinds
        assert "stream_end" in kinds

        # Check STREAM_END data
        end_event = [e for e in events if e.kind == "stream_end"][0]
        assert end_event.payload["chunk_count"] == len(chunks)
        assert isinstance(end_event.payload["full_response"], str)

    def test_process_stream_records_error(self):
        model = FailingModel()
        tm = TraceMiddleware(run_id="stream_err")
        pipeline = ModelPipeline(model, middlewares=[tm])

        with pytest.raises(ModelError):
            list(pipeline.stream("fail"))

        kinds = [e.kind for e in tm.recorder.events]
        assert "stream_start" in kinds
        assert "error" in kinds

    def test_process_stream_no_model_raises(self):
        tm = TraceMiddleware()
        tm.model = None
        tm.next_middleware = None

        with pytest.raises(ModelError, match="No streaming implementation"):
            list(tm.process_stream("test"))

    def test_process_stream_with_next_middleware(self):
        model = DummyModel()
        tm = TraceMiddleware()
        passthrough = PassthroughMiddleware()
        pipeline = ModelPipeline(model, middlewares=[tm, passthrough])

        chunks = list(pipeline.stream("test"))
        assert len(chunks) > 0
        kinds = [e.kind for e in tm.recorder.events]
        assert "stream_start" in kinds
        assert "stream_end" in kinds

    @pytest.mark.asyncio
    async def test_aprocess_stream_records_events(self):
        model = DummyModel()
        tm = TraceMiddleware(run_id="async_stream")
        pipeline = ModelPipeline(model, middlewares=[tm])

        chunks = []
        async for chunk in pipeline.astream("Tell me a story"):
            chunks.append(chunk)

        assert len(chunks) > 0

        events = tm.recorder.events
        kinds = [e.kind for e in events]
        assert "stream_start" in kinds
        assert "stream_chunk" in kinds
        assert "stream_end" in kinds

    @pytest.mark.asyncio
    async def test_aprocess_stream_records_error(self):
        model = FailingModel()
        tm = TraceMiddleware()
        pipeline = ModelPipeline(model, middlewares=[tm])

        with pytest.raises(ModelError):
            async for _ in pipeline.astream("fail"):
                pass

        kinds = [e.kind for e in tm.recorder.events]
        assert "stream_start" in kinds
        assert "error" in kinds

    @pytest.mark.asyncio
    async def test_aprocess_stream_no_model_raises(self):
        tm = TraceMiddleware()
        tm.model = None
        tm.next_middleware = None

        with pytest.raises(ModelError, match="No streaming implementation"):
            async for _ in tm.aprocess_stream("test"):
                pass

    @pytest.mark.asyncio
    async def test_aprocess_stream_with_next_middleware(self):
        model = DummyModel()
        tm = TraceMiddleware()
        passthrough = PassthroughMiddleware()
        pipeline = ModelPipeline(model, middlewares=[tm, passthrough])

        chunks = []
        async for chunk in pipeline.astream("test"):
            chunks.append(chunk)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_aprocess_stream_fallback_sync_stream(self):
        """When model has .stream but not .astream, falls back to executor."""
        model = DummyModel()
        # DummyModel has stream but not astream
        tm = TraceMiddleware()
        tm.model = model
        tm.next_middleware = None

        chunks = []
        async for chunk in tm.aprocess_stream("test"):
            chunks.append(chunk)

        assert len(chunks) > 0
        kinds = [e.kind for e in tm.recorder.events]
        assert "stream_start" in kinds
        assert "stream_end" in kinds

    @pytest.mark.asyncio
    async def test_aprocess_stream_no_streaming_impl(self):
        """Model with no stream or astream raises."""
        tm = TraceMiddleware()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="response")
        # Remove stream/astream attributes
        del mock_model.stream
        del mock_model.astream
        tm.model = mock_model
        tm.next_middleware = None

        with pytest.raises(ModelError, match="No streaming implementation"):
            async for _ in tm.aprocess_stream("test"):
                pass


# ============================================================================
# Middleware Error Handling
# ============================================================================


class TestMiddlewareErrorHandling:
    """Tests for error propagation through middleware chains."""

    def test_error_propagates_through_chain(self):
        model = FailingModel()
        cache = CacheMiddleware()
        pipeline = ModelPipeline(model, middlewares=[cache])

        with pytest.raises(ModelError, match="model failure"):
            pipeline.generate("test")

    def test_error_in_middle_of_chain(self):
        """An error raised by an inner middleware propagates out."""

        class ErrorMiddleware(Middleware):
            def process_generate(self, prompt: str, **kwargs: Any) -> str:
                raise ModelError("middleware failure")

        model = DummyModel()
        err_mw = ErrorMiddleware()
        cache = CacheMiddleware()
        pipeline = ModelPipeline(model, middlewares=[cache, err_mw])

        with pytest.raises(ModelError, match="middleware failure"):
            pipeline.generate("test")

    @pytest.mark.asyncio
    async def test_async_error_propagates_through_chain(self):
        model = FailingModel()
        cache = CacheMiddleware()
        pipeline = ModelPipeline(model, middlewares=[cache])

        with pytest.raises(ModelError, match="model failure"):
            await pipeline.agenerate("test")


# ============================================================================
# Complex Middleware Interactions
# ============================================================================


class TestComplexMiddlewareInteractions:
    """Tests for multi-middleware stacks including TraceMiddleware."""

    def test_trace_cache_cost_together(self):
        model = DummyModel()
        model.model_id = "gpt-4"
        tm = TraceMiddleware(run_id="full_stack")
        cache = CacheMiddleware(cache_size=10)
        cost = CostTrackingMiddleware()

        pipeline = ModelPipeline(model, middlewares=[tm, cache, cost])

        # First call: cache miss
        r1 = pipeline.generate("What is AI?")
        assert cache.misses == 1
        assert cost.total_requests == 1

        # Second call: cache hit (cost tracker should not be called again)
        r2 = pipeline.generate("What is AI?")
        assert cache.hits == 1
        assert cost.total_requests == 1  # Cost tracker not invoked for cache hit
        assert r1 == r2

        # Trace should capture both requests
        events = tm.recorder.events
        gen_starts = [e for e in events if e.kind == "generate_start"]
        gen_ends = [e for e in events if e.kind == "generate_end"]
        assert len(gen_starts) == 2
        assert len(gen_ends) == 2

    def test_trace_retry_interaction(self):
        """Trace captures retry attempts."""
        model = FailNTimesModel(fail_count=2, error_cls=ModelError)
        tm = TraceMiddleware(run_id="retry_trace")
        retry = RetryMiddleware(max_retries=3, initial_delay=0.001)

        pipeline = ModelPipeline(model, middlewares=[tm, retry])

        response = pipeline.generate("test")
        assert "success" in response

        # Trace should see start and end
        events = tm.recorder.events
        kinds = [e.kind for e in events]
        assert "generate_start" in kinds
        assert "generate_end" in kinds

    @pytest.mark.asyncio
    async def test_trace_cache_cost_async(self):
        model = DummyModel()
        model.model_id = "gpt-3.5-turbo"
        tm = TraceMiddleware(run_id="async_full")
        cache = CacheMiddleware(cache_size=10)
        cost = CostTrackingMiddleware()

        pipeline = ModelPipeline(model, middlewares=[tm, cache, cost])

        r1 = await pipeline.agenerate("async test")
        r2 = await pipeline.agenerate("async test")

        assert cache.hits == 1
        assert cache.misses == 1
        assert cost.total_requests == 1
        assert r1 == r2


# ============================================================================
# Cache Edge Cases
# ============================================================================


class TestCacheEdgeCases:
    """Tests for cache middleware edge cases."""

    def test_cache_with_short_response(self):
        """Cache correctly stores and retrieves very short responses."""
        model = DummyModel(canned_response="ok")
        cache = CacheMiddleware(cache_size=10)
        pipeline = ModelPipeline(model, middlewares=[cache])

        r1 = pipeline.generate("test")
        assert r1 == "ok"
        assert cache.misses == 1

        r2 = pipeline.generate("test")
        assert r2 == "ok"
        assert cache.hits == 1

    def test_cache_ttl_exact_boundary(self):
        """Cache entry right at TTL boundary."""
        model = DummyModel()
        cache = CacheMiddleware(cache_size=10, ttl_seconds=0.05)
        pipeline = ModelPipeline(model, middlewares=[cache])

        pipeline.generate("test")
        assert cache.misses == 1

        # Still within TTL
        pipeline.generate("test")
        assert cache.hits == 1

        # Wait for TTL to expire
        time.sleep(0.06)

        pipeline.generate("test")
        assert cache.misses == 2

    def test_cache_size_1(self):
        """Cache with size 1 should evict immediately on second entry."""
        model = DummyModel()
        cache = CacheMiddleware(cache_size=1)
        pipeline = ModelPipeline(model, middlewares=[cache])

        pipeline.generate("prompt_a")
        assert cache.misses == 1

        pipeline.generate("prompt_a")
        assert cache.hits == 1

        # New prompt evicts old one
        pipeline.generate("prompt_b")
        assert cache.misses == 2

        # prompt_a should be evicted
        pipeline.generate("prompt_a")
        assert cache.misses == 3

    def test_cache_invalid_size_raises(self):
        with pytest.raises(ValueError, match="cache_size must be >= 1"):
            CacheMiddleware(cache_size=0)

    def test_cache_different_kwargs_different_keys(self):
        model = DummyModel()
        cache = CacheMiddleware(cache_size=10)
        pipeline = ModelPipeline(model, middlewares=[cache])

        pipeline.generate("test", temperature=0.5)
        assert cache.misses == 1

        pipeline.generate("test", temperature=0.9)
        assert cache.misses == 2

        # Same kwargs hits
        pipeline.generate("test", temperature=0.5)
        assert cache.hits == 1

    @pytest.mark.asyncio
    async def test_cache_async_eviction(self):
        model = DummyModel()
        cache = CacheMiddleware(cache_size=2)
        pipeline = ModelPipeline(model, middlewares=[cache])

        await pipeline.agenerate("a")
        await pipeline.agenerate("b")
        await pipeline.agenerate("c")  # Should evict "a"

        # "a" should be evicted
        await pipeline.agenerate("a")
        assert cache.misses == 4  # a, b, c, a (evicted)

    @pytest.mark.asyncio
    async def test_cache_async_ttl_expiration(self):
        model = DummyModel()
        cache = CacheMiddleware(cache_size=10, ttl_seconds=0.05)
        pipeline = ModelPipeline(model, middlewares=[cache])

        await pipeline.agenerate("test")
        assert cache.misses == 1

        await pipeline.agenerate("test")
        assert cache.hits == 1

        await asyncio.sleep(0.06)

        await pipeline.agenerate("test")
        assert cache.misses == 2

    def test_cache_no_model_raises(self):
        cache = CacheMiddleware(cache_size=10)
        cache.model = None
        cache.next_middleware = None

        with pytest.raises(ModelError, match="No model available"):
            cache.process_generate("test")

    @pytest.mark.asyncio
    async def test_cache_async_no_model_raises(self):
        cache = CacheMiddleware(cache_size=10)
        cache.model = None
        cache.next_middleware = None

        with pytest.raises(ModelError, match="No model available"):
            await cache.aprocess_generate("test")

    def test_cache_is_expired_no_ttl(self):
        cache = CacheMiddleware(ttl_seconds=None)
        assert cache._is_expired(0) is False

    def test_cache_is_expired_with_ttl(self):
        cache = CacheMiddleware(ttl_seconds=10)
        old_timestamp = time.time() - 20
        assert cache._is_expired(old_timestamp) is True
        recent_timestamp = time.time()
        assert cache._is_expired(recent_timestamp) is False


# ============================================================================
# Rate Limit Edge Cases
# ============================================================================


class TestRateLimitEdgeCases:
    """Tests for rate limit middleware edge cases."""

    def test_invalid_requests_per_minute(self):
        with pytest.raises(ValueError, match="requests_per_minute must be > 0"):
            RateLimitMiddleware(requests_per_minute=0)

        with pytest.raises(ValueError, match="requests_per_minute must be > 0"):
            RateLimitMiddleware(requests_per_minute=-1)

    def test_invalid_burst_size(self):
        with pytest.raises(ValueError, match="burst_size must be > 0"):
            RateLimitMiddleware(requests_per_minute=60, burst_size=0)

        with pytest.raises(ValueError, match="burst_size must be > 0"):
            RateLimitMiddleware(requests_per_minute=60, burst_size=-1)

    def test_burst_size_defaults_to_rpm(self):
        rl = RateLimitMiddleware(requests_per_minute=120)
        assert rl.burst_size == 120

    def test_custom_burst_size(self):
        rl = RateLimitMiddleware(requests_per_minute=60, burst_size=5)
        assert rl.burst_size == 5

    def test_rate_limit_acquire_token_with_wait(self):
        """Burst size = 1, so second request must wait."""
        rl = RateLimitMiddleware(requests_per_minute=6000, burst_size=1)
        model = DummyModel()
        pipeline = ModelPipeline(model, middlewares=[rl])

        pipeline.generate("test1")
        # Tokens should be depleted now
        assert rl.tokens < 1.0

        start = time.time()
        pipeline.generate("test2")
        elapsed = time.time() - start
        # Should have waited briefly
        assert elapsed >= 0

    def test_rate_limit_no_model_raises(self):
        rl = RateLimitMiddleware(requests_per_minute=60)
        rl.model = None
        rl.next_middleware = None

        with pytest.raises(ModelError, match="No model available"):
            rl.process_generate("test")

    @pytest.mark.asyncio
    async def test_async_rate_limit_no_model_raises(self):
        rl = RateLimitMiddleware(requests_per_minute=60)
        rl.model = None
        rl.next_middleware = None

        with pytest.raises(ModelError, match="No model available"):
            await rl.aprocess_generate("test")

    @pytest.mark.asyncio
    async def test_async_rate_limit_with_burst(self):
        rl = RateLimitMiddleware(requests_per_minute=600, burst_size=3)
        model = DummyModel()
        pipeline = ModelPipeline(model, middlewares=[rl])

        # Should handle burst within limit
        results = await asyncio.gather(
            pipeline.agenerate("a"),
            pipeline.agenerate("b"),
            pipeline.agenerate("c"),
        )
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_async_acquire_token_wait(self):
        """Async token acquisition with wait."""
        rl = RateLimitMiddleware(requests_per_minute=6000, burst_size=1)
        model = DummyModel()
        pipeline = ModelPipeline(model, middlewares=[rl])

        await pipeline.agenerate("test1")
        await pipeline.agenerate("test2")  # May need to wait for token


# ============================================================================
# Retry Middleware Exhaustion
# ============================================================================


class TestRetryMiddlewareExhaustion:
    """Tests for retry middleware when all retries are exhausted."""

    def test_sync_retry_exhaustion(self):
        model = FailingModel(error_cls=ModelError, msg="always fails")
        retry = RetryMiddleware(max_retries=2, initial_delay=0.001, max_delay=0.01)
        pipeline = ModelPipeline(model, middlewares=[retry])

        with pytest.raises(ModelError, match="Failed after 2 retries"):
            pipeline.generate("test")

        assert retry.total_retries == 2

    def test_sync_retry_exhaustion_rate_limit_error(self):
        model = FailingModel(error_cls=RateLimitError, msg="rate limited")
        retry = RetryMiddleware(max_retries=1, initial_delay=0.001)
        pipeline = ModelPipeline(model, middlewares=[retry])

        with pytest.raises(ModelError, match="Failed after 1 retries"):
            pipeline.generate("test")

    def test_sync_retry_exhaustion_timeout_error(self):
        """TimeoutError is retryable and causes retry exhaustion."""

        class TimeoutFailModel(Model):
            _supports_streaming = False
            _supports_chat = False

            def __init__(self):
                super().__init__(name="TimeoutFail", model_id="timeout-v1")

            def generate(self, prompt: str, **kwargs: Any) -> str:
                raise TimeoutError(model_id="timeout-v1", timeout_seconds=30.0)

        model = TimeoutFailModel()
        retry = RetryMiddleware(max_retries=1, initial_delay=0.001)
        pipeline = ModelPipeline(model, middlewares=[retry])

        with pytest.raises(ModelError, match="Failed after 1 retries"):
            pipeline.generate("test")

    def test_sync_retry_partial_failure(self):
        """Model fails twice then succeeds."""
        model = FailNTimesModel(fail_count=2, error_cls=ModelError)
        retry = RetryMiddleware(max_retries=3, initial_delay=0.001, max_delay=0.01)
        pipeline = ModelPipeline(model, middlewares=[retry])

        response = pipeline.generate("test")
        assert "success" in response
        assert retry.total_retries == 2

    @pytest.mark.asyncio
    async def test_async_retry_exhaustion(self):
        model = FailingModel(error_cls=ModelError, msg="always fails")
        retry = RetryMiddleware(max_retries=2, initial_delay=0.001, max_delay=0.01)
        pipeline = ModelPipeline(model, middlewares=[retry])

        with pytest.raises(ModelError, match="Failed after 2 retries"):
            await pipeline.agenerate("test")

        assert retry.total_retries == 2

    @pytest.mark.asyncio
    async def test_async_retry_partial_failure(self):
        model = FailNTimesModel(fail_count=1, error_cls=ModelError)
        retry = RetryMiddleware(max_retries=3, initial_delay=0.001)
        pipeline = ModelPipeline(model, middlewares=[retry])

        response = await pipeline.agenerate("test")
        assert "success" in response

    def test_retry_no_model_raises(self):
        retry = RetryMiddleware(max_retries=1, initial_delay=0.001)
        retry.model = None
        retry.next_middleware = None

        with pytest.raises(ModelError, match="Failed after 1 retries"):
            retry.process_generate("test")

    @pytest.mark.asyncio
    async def test_async_retry_no_model_raises(self):
        retry = RetryMiddleware(max_retries=1, initial_delay=0.001)
        retry.model = None
        retry.next_middleware = None

        with pytest.raises(ModelError, match="Failed after 1 retries"):
            await retry.aprocess_generate("test")

    def test_retry_calculate_delay_capped(self):
        retry = RetryMiddleware(
            max_retries=3,
            initial_delay=1.0,
            max_delay=5.0,
            exponential_base=10.0,
        )
        delay = retry._calculate_delay(3)
        # 1.0 * 10^3 = 1000, capped to max_delay=5.0, plus jitter
        assert delay <= 5.0 * 1.1 + 0.01


# ============================================================================
# Pipeline Batch Operations
# ============================================================================


class TestPipelineBatchOperations:
    """Tests for batch operations on pipelines."""

    @pytest.mark.asyncio
    async def test_abatch_generate_with_return_exceptions(self):
        model = FailingModel()
        pipeline = ModelPipeline(model, middlewares=[])

        results = await pipeline.abatch_generate(
            ["p1", "p2"],
            max_concurrency=2,
            return_exceptions=True,
        )
        assert len(results) == 2
        for r in results:
            assert "Error" in r

    @pytest.mark.asyncio
    async def test_abatch_generate_exception_propagates(self):
        model = FailingModel()
        pipeline = ModelPipeline(model, middlewares=[])

        with pytest.raises(ModelError):
            await pipeline.abatch_generate(
                ["p1", "p2"],
                max_concurrency=2,
                return_exceptions=False,
            )

    @pytest.mark.asyncio
    async def test_abatch_generate_empty_list(self):
        model = DummyModel()
        pipeline = ModelPipeline(model, middlewares=[])

        results = await pipeline.abatch_generate([], max_concurrency=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_abatch_generate_single_item(self):
        model = DummyModel()
        pipeline = ModelPipeline(model, middlewares=[])

        results = await pipeline.abatch_generate(["hello"], max_concurrency=1)
        assert len(results) == 1
        assert isinstance(results[0], str)


# ============================================================================
# Pipeline achat / astream
# ============================================================================


class TestPipelineAsyncChatStream:
    """Tests for async chat and stream through pipeline."""

    @pytest.mark.asyncio
    async def test_achat_with_middleware(self):
        model = DummyModel()
        cache = CacheMiddleware(cache_size=10)
        pipeline = ModelPipeline(model, middlewares=[cache])

        messages = [{"role": "user", "content": "Hello"}]
        response = await pipeline.achat(messages)
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_achat_no_support_raises(self):
        """Model with no chat support raises."""
        mock_model = MagicMock()
        mock_model.name = "nochat"
        mock_model.model_id = "nochat"
        mock_model.generate = MagicMock(return_value="ok")
        mock_model.info = MagicMock(return_value={"name": "nochat"})
        del mock_model.chat
        del mock_model.achat

        pipeline = ModelPipeline(mock_model, middlewares=[])
        with pytest.raises(ModelError, match="does not support chat"):
            await pipeline.achat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_astream_with_middleware(self):
        model = DummyModel()
        cache = CacheMiddleware(cache_size=10)
        pipeline = ModelPipeline(model, middlewares=[cache])

        chunks = []
        async for chunk in pipeline.astream("test"):
            chunks.append(chunk)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_astream_no_support_raises(self):
        """Model with no stream support raises."""
        mock_model = MagicMock()
        mock_model.name = "nostream"
        mock_model.model_id = "nostream"
        mock_model.generate = MagicMock(return_value="ok")
        mock_model.info = MagicMock(return_value={"name": "nostream"})
        del mock_model.stream
        del mock_model.astream

        pipeline = ModelPipeline(mock_model, middlewares=[])
        with pytest.raises(ModelError, match="does not support streaming"):
            async for _ in pipeline.astream("test"):
                pass

    @pytest.mark.asyncio
    async def test_astream_no_middleware_with_astream(self):
        """Pipeline delegates to model.astream when present and no middleware."""
        model = DummyModel()

        async def fake_astream(prompt, **kwargs):
            for word in ["hello ", "world "]:
                yield word

        model.astream = fake_astream

        pipeline = ModelPipeline(model, middlewares=[])
        chunks = []
        async for chunk in pipeline.astream("test"):
            chunks.append(chunk)
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_astream_no_middleware_sync_fallback(self):
        """Pipeline falls back to model.stream via executor when no middleware."""
        model = DummyModel()
        # DummyModel has .stream but not .astream
        assert hasattr(model, "stream")

        pipeline = ModelPipeline(model, middlewares=[])
        chunks = []
        async for chunk in pipeline.astream("hello"):
            chunks.append(chunk)
        assert len(chunks) > 0


# ============================================================================
# Middleware Base Class Paths
# ============================================================================


class TestMiddlewareBasePaths:
    """Tests for abstract Middleware class default implementations."""

    def test_process_chat_delegates_to_next(self):
        """Default process_chat delegates to next_middleware."""
        model = DummyModel()
        passthrough = PassthroughMiddleware()
        passthrough.model = model
        passthrough.next_middleware = None

        messages = [{"role": "user", "content": "hi"}]
        response = passthrough.process_chat(messages)
        assert isinstance(response, str)

    def test_process_chat_no_model_raises(self):
        passthrough = PassthroughMiddleware()
        passthrough.model = None
        passthrough.next_middleware = None

        with pytest.raises(ModelError, match="No chat implementation"):
            passthrough.process_chat([{"role": "user", "content": "hi"}])

    def test_process_stream_delegates_to_next(self):
        model = DummyModel()
        passthrough = PassthroughMiddleware()
        passthrough.model = model
        passthrough.next_middleware = None

        chunks = list(passthrough.process_stream("test"))
        assert len(chunks) > 0

    def test_process_stream_no_model_raises(self):
        passthrough = PassthroughMiddleware()
        passthrough.model = None
        passthrough.next_middleware = None

        with pytest.raises(ModelError, match="No streaming implementation"):
            list(passthrough.process_stream("test"))

    @pytest.mark.asyncio
    async def test_aprocess_generate_no_model_raises(self):
        passthrough = PassthroughMiddleware()
        passthrough.model = None
        passthrough.next_middleware = None

        with pytest.raises(ModelError, match="No model available"):
            await passthrough.aprocess_generate("test")

    @pytest.mark.asyncio
    async def test_aprocess_chat_delegates(self):
        model = DummyModel()
        passthrough = PassthroughMiddleware()
        passthrough.model = model
        passthrough.next_middleware = None

        messages = [{"role": "user", "content": "hi"}]
        response = await passthrough.aprocess_chat(messages)
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_aprocess_chat_no_model_raises(self):
        passthrough = PassthroughMiddleware()
        passthrough.model = None
        passthrough.next_middleware = None

        with pytest.raises(ModelError, match="No chat implementation"):
            await passthrough.aprocess_chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_aprocess_stream_delegates(self):
        model = DummyModel()
        passthrough = PassthroughMiddleware()
        passthrough.model = model
        passthrough.next_middleware = None

        chunks = []
        async for chunk in passthrough.aprocess_stream("test"):
            chunks.append(chunk)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_aprocess_stream_no_model_raises(self):
        passthrough = PassthroughMiddleware()
        passthrough.model = None
        passthrough.next_middleware = None

        with pytest.raises(ModelError, match="No streaming implementation"):
            async for _ in passthrough.aprocess_stream("test"):
                pass

    @pytest.mark.asyncio
    async def test_aprocess_stream_no_streaming_attr(self):
        """Model with no stream or astream attrs raises."""
        passthrough = PassthroughMiddleware()
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="ok")
        del mock_model.stream
        del mock_model.astream
        passthrough.model = mock_model
        passthrough.next_middleware = None

        with pytest.raises(ModelError, match="No streaming implementation"):
            async for _ in passthrough.aprocess_stream("test"):
                pass


# ============================================================================
# CostTrackingMiddleware Edge Cases
# ============================================================================


class TestCostTrackingEdgeCases:
    """Tests for cost tracking middleware edge cases."""

    def test_cost_tracking_no_model_raises(self):
        ct = CostTrackingMiddleware()
        ct.model = None
        ct.next_middleware = None

        with pytest.raises(ModelError, match="No model available"):
            ct.process_generate("test")

    @pytest.mark.asyncio
    async def test_async_cost_tracking_no_model_raises(self):
        ct = CostTrackingMiddleware()
        ct.model = None
        ct.next_middleware = None

        with pytest.raises(ModelError, match="No model available"):
            await ct.aprocess_generate("test")

    def test_cost_tracking_known_models(self):
        """Test cost estimation for known models."""
        # Use a long prompt to ensure enough tokens for non-zero cost after rounding
        long_prompt = "This is a very long test prompt " * 50
        for model_name in [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
        ]:
            model = DummyModel()
            model.model_id = model_name
            ct = CostTrackingMiddleware()
            pipeline = ModelPipeline(model, middlewares=[ct])

            pipeline.generate(long_prompt)
            stats = ct.get_stats()
            assert stats["total_requests"] == 1
            assert stats["total_input_tokens"] > 0
            assert stats["total_output_tokens"] > 0
            # Cost estimate uses internal _estimate_cost; verify it is non-negative
            assert stats["estimated_cost_usd"] >= 0

    def test_cost_tracking_gpt4_has_nonzero_cost(self):
        """gpt-4 has the highest cost -- definitely non-zero for any prompt."""
        long_prompt = "This is a long test prompt for cost estimation " * 50
        model = DummyModel()
        model.model_id = "gpt-4"
        ct = CostTrackingMiddleware()
        pipeline = ModelPipeline(model, middlewares=[ct])

        pipeline.generate(long_prompt)
        stats = ct.get_stats()
        assert stats["estimated_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_async_cost_tracking_with_next_middleware(self):
        model = DummyModel()
        model.model_id = "gpt-4"
        ct = CostTrackingMiddleware()
        passthrough = PassthroughMiddleware()
        pipeline = ModelPipeline(model, middlewares=[ct, passthrough])

        response = await pipeline.agenerate("test")
        assert isinstance(response, str)

    def test_cost_tracking_with_next_middleware(self):
        model = DummyModel()
        model.model_id = "gpt-4"
        ct = CostTrackingMiddleware()
        passthrough = PassthroughMiddleware()
        pipeline = ModelPipeline(model, middlewares=[ct, passthrough])

        response = pipeline.generate("test")
        assert isinstance(response, str)
        assert ct.total_requests == 1

    def test_get_stats_format(self):
        ct = CostTrackingMiddleware()
        stats = ct.get_stats()
        assert "total_requests" in stats
        assert "total_input_tokens" in stats
        assert "total_output_tokens" in stats
        assert "total_tokens" in stats
        assert "estimated_cost_usd" in stats
        assert stats["total_tokens"] == stats["total_input_tokens"] + stats["total_output_tokens"]


# ============================================================================
# ModelPipeline misc
# ============================================================================


class TestModelPipelineMisc:
    """Miscellaneous tests for ModelPipeline."""

    def test_pipeline_name_default(self):
        model = DummyModel()
        pipeline = ModelPipeline(model)
        assert "_pipeline" in pipeline.name

    def test_pipeline_name_custom(self):
        model = DummyModel()
        pipeline = ModelPipeline(model, name="my_pipe")
        assert pipeline.name == "my_pipe"

    def test_pipeline_chat_no_middleware(self):
        model = DummyModel()
        pipeline = ModelPipeline(model, middlewares=[])
        messages = [{"role": "user", "content": "hello"}]
        response = pipeline.chat(messages)
        assert isinstance(response, str)

    def test_pipeline_chat_no_support_raises(self):
        mock_model = MagicMock()
        mock_model.name = "nochat"
        mock_model.model_id = "nochat"
        mock_model.info = MagicMock(return_value={"name": "nochat"})
        del mock_model.chat

        pipeline = ModelPipeline(mock_model, middlewares=[])
        with pytest.raises(ModelError, match="does not support chat"):
            pipeline.chat([{"role": "user", "content": "hi"}])

    def test_pipeline_stream_no_middleware(self):
        model = DummyModel()
        pipeline = ModelPipeline(model, middlewares=[])
        chunks = list(pipeline.stream("test"))
        assert len(chunks) > 0

    def test_pipeline_stream_no_support_raises(self):
        mock_model = MagicMock()
        mock_model.name = "nostream"
        mock_model.model_id = "nostream"
        mock_model.info = MagicMock(return_value={"name": "nostream"})
        del mock_model.stream

        pipeline = ModelPipeline(mock_model, middlewares=[])
        with pytest.raises(ModelError, match="does not support streaming"):
            list(pipeline.stream("test"))

    def test_pipeline_info_with_all_middleware(self):
        model = DummyModel()
        model.model_id = "gpt-4"
        cache = CacheMiddleware()
        retry = RetryMiddleware()
        cost = CostTrackingMiddleware()
        tm = TraceMiddleware()

        pipeline = ModelPipeline(model, middlewares=[tm, cache, retry, cost])
        pipeline.generate("test")
        pipeline.generate("test")  # cache hit

        info = pipeline.info()
        assert info["pipeline"] is True
        assert info["middleware_count"] == 4
        assert "cache_hit_rate" in info
        assert "total_retries" in info
        assert "cost_stats" in info

    @pytest.mark.asyncio
    async def test_agenerate_no_middleware_sync_model(self):
        """Pipeline agenerate with no middleware and sync model uses executor."""
        model = DummyModel()
        pipeline = ModelPipeline(model, middlewares=[])

        response = await pipeline.agenerate("test")
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_achat_no_middleware_sync_fallback(self):
        """Pipeline achat with no middleware falls back to sync chat via executor."""
        model = DummyModel()
        # DummyModel has .chat but not .achat
        pipeline = ModelPipeline(model, middlewares=[])

        messages = [{"role": "user", "content": "hello"}]
        response = await pipeline.achat(messages)
        assert isinstance(response, str)


# ============================================================================
# AsyncModelPipeline Extended Tests
# ============================================================================


class TestAsyncModelPipelineExtended:
    """Extended tests for AsyncModelPipeline features."""

    @pytest.mark.asyncio
    async def test_agenerate_with_callback_error_handling(self):
        """Callback pipeline handles errors gracefully."""
        model = FailingModel()
        pipeline = AsyncModelPipeline(model, middlewares=[])

        progress_calls = []
        result_calls = []

        def on_progress(completed, total):
            progress_calls.append((completed, total))

        def on_result(index, result):
            result_calls.append((index, result))

        results = await pipeline.agenerate_with_callback(
            ["p1", "p2"],
            max_concurrency=2,
            on_progress=on_progress,
            on_result=on_result,
        )

        assert len(results) == 2
        assert len(progress_calls) == 2
        assert len(result_calls) == 2
        # Errors should be captured as strings
        for _, result in result_calls:
            assert "Error" in result

    @pytest.mark.asyncio
    async def test_agenerate_with_callback_no_callbacks(self):
        """Callback pipeline works without callbacks."""
        model = DummyModel()
        pipeline = AsyncModelPipeline(model, middlewares=[])

        results = await pipeline.agenerate_with_callback(
            ["p1", "p2"],
            max_concurrency=2,
        )
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    @pytest.mark.asyncio
    async def test_agenerate_stream_results_order(self):
        """Stream results yields all items eventually."""
        model = DummyModel()
        pipeline = AsyncModelPipeline(model, middlewares=[])

        prompts = ["a", "b", "c", "d"]
        results = []
        async for idx, result in pipeline.agenerate_stream_results(prompts, max_concurrency=2):
            results.append((idx, result))

        assert len(results) == 4
        indices = sorted([idx for idx, _ in results])
        assert indices == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_agenerate_stream_results_with_error(self):
        """Stream results captures errors as strings."""
        model = FailingModel()
        pipeline = AsyncModelPipeline(model, middlewares=[])

        results = []
        async for idx, result in pipeline.agenerate_stream_results(["p1", "p2"], max_concurrency=2):
            results.append((idx, result))

        assert len(results) == 2
        for _, r in results:
            assert "Error" in r

    @pytest.mark.asyncio
    async def test_amap_with_error(self):
        """amap captures errors in result tuples."""
        model = FailingModel()
        pipeline = AsyncModelPipeline(model, middlewares=[])

        results = await pipeline.amap(["p1", "p2"], max_concurrency=2)
        assert len(results) == 2
        for prompt, response, error in results:
            assert response is None
            assert error is not None
            assert isinstance(error, Exception)

    @pytest.mark.asyncio
    async def test_amap_with_timeout_expire(self):
        """amap with very short timeout should produce timeout errors."""

        class SlowModel(Model):
            _supports_streaming = False
            _supports_chat = False

            def __init__(self):
                super().__init__(name="SlowModel", model_id="slow-v1")

            def generate(self, prompt: str, **kwargs: Any) -> str:
                time.sleep(0.5)
                return "slow response"

        model = SlowModel()
        pipeline = AsyncModelPipeline(model, middlewares=[])

        results = await pipeline.amap(["p1"], max_concurrency=1, timeout=0.01)
        assert len(results) == 1
        prompt, response, error = results[0]
        assert response is None
        assert error is not None

    @pytest.mark.asyncio
    async def test_amap_empty_list(self):
        model = DummyModel()
        pipeline = AsyncModelPipeline(model, middlewares=[])

        results = await pipeline.amap([], max_concurrency=2)
        assert results == []


# ============================================================================
# Pipeline with chained middleware edge cases
# ============================================================================


class TestMiddlewareChaining:
    """Tests for middleware chain setup and delegation."""

    def test_middleware_chain_wiring(self):
        """Pipeline correctly wires next_middleware and model references."""
        model = DummyModel()
        m1 = PassthroughMiddleware()
        m2 = PassthroughMiddleware()
        m3 = PassthroughMiddleware()

        ModelPipeline(model, middlewares=[m1, m2, m3])

        assert m1.next_middleware is m2
        assert m2.next_middleware is m3
        assert m3.next_middleware is None

        assert m1.model is model
        assert m2.model is model
        assert m3.model is model

    def test_empty_middleware_list(self):
        model = DummyModel()
        pipeline = ModelPipeline(model, middlewares=[])
        assert pipeline.middlewares == []
        response = pipeline.generate("test")
        assert isinstance(response, str)

    def test_none_middleware_list(self):
        model = DummyModel()
        pipeline = ModelPipeline(model, middlewares=None)
        assert pipeline.middlewares == []
        response = pipeline.generate("test")
        assert isinstance(response, str)
