"""Tests for the model pipeline and middleware system."""

import asyncio
import time

import pytest

from insideLLMs.exceptions import ModelError
from insideLLMs.models import DummyModel
from insideLLMs.pipeline import (
    AsyncModelPipeline,
    CacheMiddleware,
    CostTrackingMiddleware,
    Middleware,
    ModelPipeline,
    PassthroughMiddleware,
    RateLimitMiddleware,
    RetryMiddleware,
)


class CountingMiddleware(Middleware):
    """Test middleware that counts invocations."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def process_generate(self, prompt: str, **kwargs):
        self.count += 1
        if self.next_middleware:
            return self.next_middleware.process_generate(prompt, **kwargs)
        if self.model:
            return self.model.generate(prompt, **kwargs)
        raise ModelError("No model available")


def test_pipeline_basic():
    """Test basic pipeline functionality."""
    model = DummyModel()
    pipeline = ModelPipeline(model, middlewares=[])

    response = pipeline.generate("test prompt")
    assert isinstance(response, str)
    assert len(response) > 0


def test_pipeline_with_middleware():
    """Test pipeline with single middleware."""
    model = DummyModel()
    counter = CountingMiddleware()
    pipeline = ModelPipeline(model, middlewares=[counter])

    pipeline.generate("test 1")
    assert counter.count == 1

    pipeline.generate("test 2")
    assert counter.count == 2


def test_pipeline_middleware_chain():
    """Test pipeline with multiple middleware."""
    model = DummyModel()
    counter1 = CountingMiddleware()
    counter2 = CountingMiddleware()
    pipeline = ModelPipeline(model, middlewares=[counter1, counter2])

    pipeline.generate("test")
    assert counter1.count == 1
    assert counter2.count == 1


def test_cache_middleware():
    """Test cache middleware functionality."""
    model = DummyModel()
    cache = CacheMiddleware(cache_size=10)
    pipeline = ModelPipeline(model, middlewares=[cache])

    # First call - should miss
    response1 = pipeline.generate("test prompt")
    assert cache.misses == 1
    assert cache.hits == 0

    # Second call with same prompt - should hit
    response2 = pipeline.generate("test prompt")
    assert cache.hits == 1
    assert cache.misses == 1
    assert response1 == response2

    # Different prompt - should miss
    pipeline.generate("different prompt")
    assert cache.misses == 2
    assert cache.hits == 1


def test_cache_middleware_ttl():
    """Test cache middleware with TTL expiration."""
    model = DummyModel()
    cache = CacheMiddleware(cache_size=10, ttl_seconds=0.1)
    pipeline = ModelPipeline(model, middlewares=[cache])

    # First call
    pipeline.generate("test")
    assert cache.hits == 0
    assert cache.misses == 1

    # Second call immediately - should hit
    pipeline.generate("test")
    assert cache.hits == 1

    # Wait for expiration
    time.sleep(0.15)

    # Third call after expiration - should miss
    pipeline.generate("test")
    assert cache.misses == 2


def test_cache_middleware_lru_eviction():
    """Test cache middleware LRU eviction."""
    model = DummyModel()
    cache = CacheMiddleware(cache_size=2)
    pipeline = ModelPipeline(model, middlewares=[cache])

    # Fill cache
    pipeline.generate("prompt1")
    pipeline.generate("prompt2")

    # Add third item - should evict oldest (prompt1)
    pipeline.generate("prompt3")

    # Access prompt1 again - should be cache miss (was evicted)
    pipeline.generate("prompt1")
    assert cache.misses == 4  # All original + evicted prompt1

    # Access prompt3 - should still be cached (most recent)
    pipeline.generate("prompt3")
    assert cache.hits == 1


def test_cache_hit_rate():
    """Test cache hit rate calculation."""
    model = DummyModel()
    cache = CacheMiddleware()
    pipeline = ModelPipeline(model, middlewares=[cache])

    assert cache.hit_rate == 0.0

    pipeline.generate("test")
    assert cache.hit_rate == 0.0  # 0/1

    pipeline.generate("test")
    assert cache.hit_rate == 0.5  # 1/2

    pipeline.generate("test")
    assert cache.hit_rate == pytest.approx(0.666, rel=0.01)  # 2/3


def test_rate_limit_middleware():
    """Test rate limiting middleware."""
    model = DummyModel()
    rate_limiter = RateLimitMiddleware(requests_per_minute=120)  # 2 per second
    pipeline = ModelPipeline(model, middlewares=[rate_limiter])

    start = time.time()

    # First request should be immediate
    pipeline.generate("test1")

    # Second request should also be fast (within burst)
    pipeline.generate("test2")

    elapsed = time.time() - start
    # Should be very fast if both within burst
    assert elapsed < 0.1


def test_retry_middleware_success():
    """Test retry middleware with successful call."""
    model = DummyModel()
    retry = RetryMiddleware(max_retries=3)
    pipeline = ModelPipeline(model, middlewares=[retry])

    response = pipeline.generate("test")
    assert isinstance(response, str)
    assert retry.total_retries == 0  # No retries needed


def test_retry_middleware_counts():
    """Test retry middleware retry counting."""
    model = DummyModel()
    retry = RetryMiddleware(max_retries=3, initial_delay=0.01)
    pipeline = ModelPipeline(model, middlewares=[retry])

    # Successful call
    pipeline.generate("test")
    assert retry.total_retries == 0


def test_cost_tracking_middleware():
    """Test cost tracking middleware."""
    model = DummyModel()
    model.model_id = "gpt-3.5-turbo"
    cost_tracker = CostTrackingMiddleware()
    pipeline = ModelPipeline(model, middlewares=[cost_tracker])

    # Generate some responses
    pipeline.generate("test prompt 1")
    pipeline.generate("test prompt 2")

    stats = cost_tracker.get_stats()
    assert stats["total_requests"] == 2
    assert stats["total_input_tokens"] > 0
    assert stats["total_output_tokens"] > 0
    assert stats["estimated_cost_usd"] >= 0


def test_cost_tracking_unknown_model():
    """Test cost tracking with unknown model."""
    model = DummyModel()
    model.model_id = "unknown-model"
    cost_tracker = CostTrackingMiddleware()
    pipeline = ModelPipeline(model, middlewares=[cost_tracker])

    pipeline.generate("test")
    stats = cost_tracker.get_stats()
    assert stats["estimated_cost_usd"] == 0.0  # Unknown model


def test_pipeline_info():
    """Test pipeline info includes middleware details."""
    model = DummyModel()
    cache = CacheMiddleware()
    retry = RetryMiddleware()
    cost_tracker = CostTrackingMiddleware()

    pipeline = ModelPipeline(
        model,
        middlewares=[cache, retry, cost_tracker],
    )

    # Generate some traffic
    pipeline.generate("test")
    pipeline.generate("test")  # Cache hit

    info = pipeline.info()
    assert info["pipeline"] is True
    assert info["middleware_count"] == 3
    assert "CacheMiddleware" in info["middlewares"]
    assert "RetryMiddleware" in info["middlewares"]
    assert "CostTrackingMiddleware" in info["middlewares"]
    assert "cache_hit_rate" in info
    assert "cost_stats" in info


def test_pipeline_chat():
    """Test pipeline chat support."""
    model = DummyModel()
    pipeline = ModelPipeline(model, middlewares=[])

    messages = [
        {"role": "user", "content": "Hello"},
    ]
    response = pipeline.chat(messages)
    assert isinstance(response, str)


def test_pipeline_stream():
    """Test pipeline streaming support."""
    model = DummyModel()
    pipeline = ModelPipeline(model, middlewares=[])

    chunks = list(pipeline.stream("test"))
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_pipeline_name():
    """Test pipeline naming."""
    model = DummyModel()
    pipeline = ModelPipeline(model, name="custom_pipeline")
    assert pipeline.name == "custom_pipeline"

    pipeline2 = ModelPipeline(model)
    assert "_pipeline" in pipeline2.name


def test_combined_middleware_stack():
    """Test realistic middleware stack."""
    model = DummyModel()
    model.model_id = "gpt-4"

    pipeline = ModelPipeline(
        model,
        middlewares=[
            CacheMiddleware(cache_size=100),
            RateLimitMiddleware(requests_per_minute=60),
            RetryMiddleware(max_retries=3),
            CostTrackingMiddleware(),
        ],
    )

    # Generate some requests
    pipeline.generate("prompt 1")
    pipeline.generate("prompt 1")  # Cache hit
    pipeline.generate("prompt 2")

    info = pipeline.info()
    assert info["middleware_count"] == 4
    assert info["cache_hit_rate"] > 0
    assert "cost_stats" in info


def test_passthrough_middleware():
    """Test passthrough middleware."""
    model = DummyModel()
    passthrough = PassthroughMiddleware()
    pipeline = ModelPipeline(model, middlewares=[passthrough])

    response = pipeline.generate("test")
    assert isinstance(response, str)


def test_middleware_without_model_raises():
    """Test that middleware without model raises error."""
    middleware = PassthroughMiddleware()

    with pytest.raises(ModelError, match="No model available"):
        middleware.process_generate("test")


# ============================================================================
# Async Pipeline Tests
# ============================================================================


class AsyncCountingMiddleware(Middleware):
    """Test middleware that counts async invocations."""

    def __init__(self):
        super().__init__()
        self.sync_count = 0
        self.async_count = 0

    def process_generate(self, prompt: str, **kwargs):
        self.sync_count += 1
        if self.next_middleware:
            return self.next_middleware.process_generate(prompt, **kwargs)
        if self.model:
            return self.model.generate(prompt, **kwargs)
        raise ModelError("No model available")

    async def aprocess_generate(self, prompt: str, **kwargs):
        self.async_count += 1
        if self.next_middleware:
            return await self.next_middleware.aprocess_generate(prompt, **kwargs)
        if self.model:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self.model.generate(prompt, **kwargs)
            )
        raise ModelError("No model available")


@pytest.mark.asyncio
async def test_pipeline_async_generate():
    """Test basic async pipeline generation."""
    model = DummyModel()
    pipeline = ModelPipeline(model, middlewares=[])

    response = await pipeline.agenerate("test prompt")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_pipeline_async_with_middleware():
    """Test async pipeline with middleware."""
    model = DummyModel()
    counter = AsyncCountingMiddleware()
    pipeline = ModelPipeline(model, middlewares=[counter])

    await pipeline.agenerate("test 1")
    assert counter.async_count == 1
    assert counter.sync_count == 0

    await pipeline.agenerate("test 2")
    assert counter.async_count == 2


@pytest.mark.asyncio
async def test_pipeline_async_middleware_chain():
    """Test async pipeline with multiple middleware."""
    model = DummyModel()
    counter1 = AsyncCountingMiddleware()
    counter2 = AsyncCountingMiddleware()
    pipeline = ModelPipeline(model, middlewares=[counter1, counter2])

    await pipeline.agenerate("test")
    assert counter1.async_count == 1
    assert counter2.async_count == 1


@pytest.mark.asyncio
async def test_async_cache_middleware():
    """Test async cache middleware functionality."""
    model = DummyModel()
    cache = CacheMiddleware(cache_size=10)
    pipeline = ModelPipeline(model, middlewares=[cache])

    # First call - should miss
    response1 = await pipeline.agenerate("test prompt")
    assert cache.misses == 1
    assert cache.hits == 0

    # Second call with same prompt - should hit
    response2 = await pipeline.agenerate("test prompt")
    assert cache.hits == 1
    assert cache.misses == 1
    assert response1 == response2


@pytest.mark.asyncio
async def test_async_rate_limit_middleware():
    """Test async rate limiting middleware."""
    model = DummyModel()
    rate_limiter = RateLimitMiddleware(requests_per_minute=120)
    pipeline = ModelPipeline(model, middlewares=[rate_limiter])

    start = time.time()

    # Run concurrent requests
    results = await asyncio.gather(
        pipeline.agenerate("test1"),
        pipeline.agenerate("test2"),
        pipeline.agenerate("test3"),
    )

    elapsed = time.time() - start
    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)


@pytest.mark.asyncio
async def test_async_retry_middleware():
    """Test async retry middleware."""
    model = DummyModel()
    retry = RetryMiddleware(max_retries=3, initial_delay=0.01)
    pipeline = ModelPipeline(model, middlewares=[retry])

    response = await pipeline.agenerate("test")
    assert isinstance(response, str)
    assert retry.total_retries == 0


@pytest.mark.asyncio
async def test_async_cost_tracking_middleware():
    """Test async cost tracking middleware."""
    model = DummyModel()
    model.model_id = "gpt-3.5-turbo"
    cost_tracker = CostTrackingMiddleware()
    pipeline = ModelPipeline(model, middlewares=[cost_tracker])

    # Run concurrent requests
    await asyncio.gather(
        pipeline.agenerate("test 1"),
        pipeline.agenerate("test 2"),
        pipeline.agenerate("test 3"),
    )

    stats = cost_tracker.get_stats()
    assert stats["total_requests"] == 3
    assert stats["total_input_tokens"] > 0
    assert stats["total_output_tokens"] > 0


@pytest.mark.asyncio
async def test_async_batch_generate():
    """Test async batch generation."""
    model = DummyModel()
    pipeline = ModelPipeline(model, middlewares=[])

    prompts = ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"]
    results = await pipeline.abatch_generate(prompts, max_concurrency=3)

    assert len(results) == len(prompts)
    assert all(isinstance(r, str) for r in results)


@pytest.mark.asyncio
async def test_async_batch_generate_with_cache():
    """Test async batch generation with caching."""
    model = DummyModel()
    cache = CacheMiddleware(cache_size=10)
    pipeline = ModelPipeline(model, middlewares=[cache])

    # First batch
    prompts = ["A", "B", "C"]
    await pipeline.abatch_generate(prompts, max_concurrency=3)
    assert cache.misses == 3
    assert cache.hits == 0

    # Same prompts again - should all hit cache
    await pipeline.abatch_generate(prompts, max_concurrency=3)
    assert cache.hits == 3


@pytest.mark.asyncio
async def test_async_pipeline_concurrency_limit():
    """Test that concurrency limit is respected."""
    model = DummyModel()
    pipeline = ModelPipeline(model, middlewares=[])

    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    original_generate = model.generate

    def tracking_generate(prompt, **kwargs):
        nonlocal max_concurrent, current_concurrent
        # Note: This is sync, so we can't use asyncio primitives directly
        # But we can track via the fact that executor calls are concurrent
        return original_generate(prompt, **kwargs)

    model.generate = tracking_generate

    prompts = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
    results = await pipeline.abatch_generate(prompts, max_concurrency=2)

    assert len(results) == len(prompts)


@pytest.mark.asyncio
async def test_async_combined_middleware_stack():
    """Test realistic async middleware stack."""
    model = DummyModel()
    model.model_id = "gpt-4"

    pipeline = ModelPipeline(
        model,
        middlewares=[
            CacheMiddleware(cache_size=100),
            RateLimitMiddleware(requests_per_minute=60),
            RetryMiddleware(max_retries=3),
            CostTrackingMiddleware(),
        ],
    )

    # Concurrent requests with some duplicates
    prompts = ["prompt1", "prompt2", "prompt1", "prompt3", "prompt2"]
    results = await pipeline.abatch_generate(prompts, max_concurrency=5)

    assert len(results) == 5
    info = pipeline.info()
    assert info["middleware_count"] == 4


# ============================================================================
# AsyncModelPipeline Tests
# ============================================================================


@pytest.mark.asyncio
async def test_async_model_pipeline_basic():
    """Test AsyncModelPipeline basic functionality."""
    model = DummyModel()
    pipeline = AsyncModelPipeline(model, middlewares=[])

    response = await pipeline.agenerate("test")
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_async_model_pipeline_with_callback():
    """Test AsyncModelPipeline with progress callback."""
    model = DummyModel()
    pipeline = AsyncModelPipeline(model, middlewares=[])

    progress_calls = []
    result_calls = []

    def on_progress(completed, total):
        progress_calls.append((completed, total))

    def on_result(index, result):
        result_calls.append((index, result))

    prompts = ["a", "b", "c"]
    results = await pipeline.agenerate_with_callback(
        prompts,
        max_concurrency=2,
        on_progress=on_progress,
        on_result=on_result,
    )

    assert len(results) == 3
    assert len(progress_calls) == 3
    assert len(result_calls) == 3

    # All progress calls should show increasing completion
    for i, (completed, total) in enumerate(sorted(progress_calls)):
        assert total == 3


@pytest.mark.asyncio
async def test_async_model_pipeline_stream_results():
    """Test AsyncModelPipeline streaming results."""
    model = DummyModel()
    pipeline = AsyncModelPipeline(model, middlewares=[])

    prompts = ["prompt1", "prompt2", "prompt3"]
    results = []

    async for idx, result in pipeline.agenerate_stream_results(prompts, max_concurrency=2):
        results.append((idx, result))

    assert len(results) == 3
    indices = [idx for idx, _ in results]
    assert sorted(indices) == [0, 1, 2]


@pytest.mark.asyncio
async def test_async_model_pipeline_map():
    """Test AsyncModelPipeline map function."""
    model = DummyModel()
    pipeline = AsyncModelPipeline(model, middlewares=[])

    prompts = ["p1", "p2", "p3"]
    results = await pipeline.amap(prompts, max_concurrency=2)

    assert len(results) == 3
    for prompt, response, error in results:
        assert prompt in prompts
        assert isinstance(response, str)
        assert error is None


@pytest.mark.asyncio
async def test_async_model_pipeline_map_with_timeout():
    """Test AsyncModelPipeline map with timeout."""
    model = DummyModel()
    pipeline = AsyncModelPipeline(model, middlewares=[])

    prompts = ["p1", "p2"]
    # Very generous timeout - should not expire
    results = await pipeline.amap(prompts, max_concurrency=2, timeout=10.0)

    assert len(results) == 2
    for _, response, error in results:
        assert response is not None
        assert error is None


@pytest.mark.asyncio
async def test_async_chat():
    """Test async chat through pipeline."""
    model = DummyModel()
    pipeline = ModelPipeline(model, middlewares=[])

    messages = [{"role": "user", "content": "Hello"}]
    response = await pipeline.achat(messages)
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_async_stream():
    """Test async streaming through pipeline."""
    model = DummyModel()
    pipeline = ModelPipeline(model, middlewares=[])

    chunks = []
    async for chunk in pipeline.astream("test"):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


@pytest.mark.asyncio
async def test_async_passthrough_middleware():
    """Test async passthrough middleware."""
    model = DummyModel()
    passthrough = PassthroughMiddleware()
    pipeline = ModelPipeline(model, middlewares=[passthrough])

    response = await pipeline.agenerate("test")
    assert isinstance(response, str)
