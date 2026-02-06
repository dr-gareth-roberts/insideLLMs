"""Additional branch coverage for runtime.pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from insideLLMs.exceptions import ModelError
from insideLLMs.pipeline import (
    AsyncModelPipeline,
    CacheMiddleware,
    CostTrackingMiddleware,
    Middleware,
    PassthroughMiddleware,
    RateLimitMiddleware,
    RetryMiddleware,
    TraceMiddleware,
)


class SyncModel:
    name = "sync-model"
    model_id = "sync-model"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return f"sync:{prompt}"

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return f"sync-chat:{len(messages)}"

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        yield "s1"
        yield "s2"

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "model_id": self.model_id}


class AsyncCapableModel(SyncModel):
    name = "async-model"
    model_id = "async-model"

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        return f"async:{prompt}"

    async def achat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return f"async-chat:{len(messages)}"

    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        for chunk in ("a1", "a2"):
            yield chunk


class ChatlessModel:
    name = "chatless"
    model_id = "chatless"

    # Intentionally no chat/achat methods for error path.
    def generate(self, prompt: str, **kwargs: Any) -> str:
        return f"chatless:{prompt}"


class NextDelegate:
    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        return f"next-gen:{prompt}"

    def process_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return f"next-chat:{len(messages)}"

    def process_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        yield "n1"
        yield "n2"

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        return f"next-agen:{prompt}"

    async def aprocess_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return f"next-achat:{len(messages)}"

    async def aprocess_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        for chunk in ("na1", "na2"):
            yield chunk


class MinimalMiddleware(Middleware):
    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        if self.next_middleware:
            return self.next_middleware.process_generate(prompt, **kwargs)
        if self.model:
            return self.model.generate(prompt, **kwargs)
        raise ModelError("No model available in pipeline")


def test_middleware_default_chat_and_stream_delegate_paths():
    mw = MinimalMiddleware()
    mw.next_middleware = NextDelegate()  # type: ignore[assignment]
    assert mw.process_chat([{"role": "user", "content": "hi"}]) == "next-chat:1"
    assert list(mw.process_stream("prompt")) == ["n1", "n2"]


@pytest.mark.asyncio
async def test_middleware_async_delegate_and_model_fallback_paths():
    mw = MinimalMiddleware()
    mw.next_middleware = NextDelegate()  # type: ignore[assignment]
    assert await mw.aprocess_generate("p") == "next-agen:p"
    assert await mw.aprocess_chat([{"role": "user", "content": "x"}]) == "next-achat:1"
    assert [c async for c in mw.aprocess_stream("p")] == ["na1", "na2"]

    async_model = AsyncCapableModel()
    mw2 = MinimalMiddleware()
    mw2.model = async_model  # type: ignore[assignment]
    assert await mw2.aprocess_generate("x") == "async:x"
    assert await mw2.aprocess_chat([{"role": "user", "content": "x"}]) == "async-chat:1"
    assert [c async for c in mw2.aprocess_stream("x")] == ["a1", "a2"]

    # Sync-model fallback branches in async wrappers.
    mw3 = MinimalMiddleware()
    mw3.model = SyncModel()  # type: ignore[assignment]
    assert await mw3.aprocess_generate("x") == "sync:x"
    assert await mw3.aprocess_chat([{"role": "user", "content": "x"}]) == "sync-chat:1"
    assert [c async for c in mw3.aprocess_stream("x")] == ["s1", "s2"]

    mw4 = MinimalMiddleware()
    with pytest.raises(ModelError, match="No model available in pipeline"):
        await mw4.aprocess_generate("x")


@pytest.mark.asyncio
async def test_passthrough_and_trace_async_model_specific_branches():
    passthrough = PassthroughMiddleware()
    passthrough.next_middleware = NextDelegate()  # type: ignore[assignment]
    assert passthrough.process_generate("q") == "next-gen:q"
    assert await passthrough.aprocess_generate("q") == "next-agen:q"

    passthrough2 = PassthroughMiddleware()
    passthrough2.model = AsyncCapableModel()  # type: ignore[assignment]
    assert await passthrough2.aprocess_generate("q2") == "async:q2"

    trace = TraceMiddleware()
    trace.model = AsyncCapableModel()  # type: ignore[assignment]
    assert await trace.aprocess_generate("tg") == "async:tg"
    assert await trace.aprocess_chat([{"role": "user", "content": "hello"}]) == "async-chat:1"
    assert [c async for c in trace.aprocess_stream("ts")] == ["a1", "a2"]

    trace_chatless = TraceMiddleware()
    trace_chatless.model = ChatlessModel()  # type: ignore[assignment]
    with pytest.raises(ModelError, match="No chat implementation available"):
        await trace_chatless.aprocess_chat([{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_async_model_branches_for_cache_rate_retry_and_cost_middleware():
    async_model = AsyncCapableModel()

    cache = CacheMiddleware(cache_size=2)
    cache.model = async_model  # type: ignore[assignment]
    assert await cache.aprocess_generate("cache-key") == "async:cache-key"

    rate = RateLimitMiddleware(requests_per_minute=6000, burst_size=10)
    rate.model = async_model  # type: ignore[assignment]
    assert await rate.aprocess_generate("rate") == "async:rate"

    retry = RetryMiddleware(max_retries=0)
    retry.model = async_model  # type: ignore[assignment]
    assert await retry.aprocess_generate("retry") == "async:retry"

    cost = CostTrackingMiddleware()
    cost.model = async_model  # type: ignore[assignment]
    assert await cost.aprocess_generate("cost") == "async:cost"
    assert cost.total_requests == 1


@pytest.mark.asyncio
async def test_async_model_pipeline_base_async_and_info_dict_paths():
    pipeline = AsyncModelPipeline(AsyncCapableModel())
    assert await pipeline.agenerate("p") == "async:p"
    assert await pipeline.achat([{"role": "user", "content": "x"}]) == "async-chat:1"
    assert [c async for c in pipeline.astream("p")] == ["a1", "a2"]

    info = pipeline.info()
    assert info["pipeline"] is True
    assert info["model_id"] == "async-model"


@pytest.mark.asyncio
async def test_async_pipeline_progress_error_callback_branch():
    class FailingAsyncModel(AsyncCapableModel):
        async def agenerate(self, prompt: str, **kwargs: Any) -> str:
            raise ValueError("boom")

    pipeline = AsyncModelPipeline(FailingAsyncModel())
    seen: list[str] = []

    def on_result(_idx: int, result: str) -> None:
        seen.append(result)

    results = await pipeline.agenerate_with_callback(["a"], on_result=on_result)
    assert results[0].startswith("Error: boom")
    assert seen[0].startswith("Error: boom")
