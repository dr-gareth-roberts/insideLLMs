"""Tests for batch generation functionality."""

import asyncio

import pytest

from insideLLMs.models import DummyModel
from insideLLMs.models.base import AsyncModel, BatchModelProtocol


def test_batch_generate_basic():
    """Test basic batch generation."""
    model = DummyModel()
    prompts = ["prompt 1", "prompt 2", "prompt 3"]

    responses = model.batch_generate(prompts)

    assert len(responses) == 3
    assert all(isinstance(r, str) for r in responses)
    assert "prompt 1" in responses[0]
    assert "prompt 2" in responses[1]
    assert "prompt 3" in responses[2]


def test_batch_generate_empty():
    """Test batch generation with empty list."""
    model = DummyModel()
    responses = model.batch_generate([])

    assert responses == []


def test_batch_generate_single():
    """Test batch generation with single prompt."""
    model = DummyModel()
    responses = model.batch_generate(["test"])

    assert len(responses) == 1
    assert "test" in responses[0]


def test_batch_generate_with_kwargs():
    """Test batch generation with additional kwargs."""
    model = DummyModel()
    prompts = ["test 1", "test 2"]

    # Should not raise even with extra kwargs
    responses = model.batch_generate(prompts, temperature=0.7, max_tokens=100)

    assert len(responses) == 2


def test_batch_protocol_check():
    """Test that DummyModel implements BatchModelProtocol."""
    model = DummyModel()

    assert isinstance(model, BatchModelProtocol)
    assert hasattr(model, "batch_generate")
    assert callable(model.batch_generate)


class AsyncDummyModel(AsyncModel):
    """Async version of DummyModel for testing."""

    def __init__(self, name: str = "AsyncDummyModel"):
        super().__init__(name=name, model_id="async-dummy-v1")

    def generate(self, prompt: str, **kwargs) -> str:
        """Sync generate (required by base class)."""
        return f"[AsyncDummy] {prompt}"

    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Async generate."""
        await asyncio.sleep(0.01)  # Simulate async work
        return f"[AsyncDummy] {prompt}"


@pytest.mark.asyncio
async def test_async_batch_generate():
    """Test async batch generation."""
    model = AsyncDummyModel()
    prompts = ["prompt 1", "prompt 2", "prompt 3"]

    responses = await model.abatch_generate(prompts)

    assert len(responses) == 3
    assert all("[AsyncDummy]" in r for r in responses)


@pytest.mark.asyncio
async def test_async_batch_generate_concurrent():
    """Test that async batch generation runs concurrently."""
    import time

    model = AsyncDummyModel()
    prompts = ["test"] * 10  # 10 prompts

    start = time.time()
    responses = await model.abatch_generate(prompts)
    elapsed = time.time() - start

    assert len(responses) == 10
    # If sequential, would take 10 * 0.01 = 0.1s
    # Concurrent should be much faster
    assert elapsed < 0.05  # Should complete in ~0.01s


@pytest.mark.asyncio
async def test_async_batch_generate_empty():
    """Test async batch generation with empty list."""
    model = AsyncDummyModel()
    responses = await model.abatch_generate([])

    assert responses == []


def test_batch_generation_order_preserved():
    """Test that batch generation preserves order."""
    model = DummyModel()
    prompts = [f"prompt {i}" for i in range(10)]

    responses = model.batch_generate(prompts)

    for i, response in enumerate(responses):
        assert f"prompt {i}" in response


def test_dummy_model_batch_canned_response():
    """Test DummyModel batch generation with canned response."""
    model = DummyModel(canned_response="Always this")
    prompts = ["test 1", "test 2", "test 3"]

    responses = model.batch_generate(prompts)

    assert all(r == "Always this" for r in responses)


def test_batch_generate_large():
    """Test batch generation with larger batch size."""
    model = DummyModel()
    prompts = [f"prompt {i}" for i in range(100)]

    responses = model.batch_generate(prompts)

    assert len(responses) == 100
    assert all(isinstance(r, str) for r in responses)
