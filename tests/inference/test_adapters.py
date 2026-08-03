import asyncio

import pytest

from insideLLMs.inference import InferenceRequest
from insideLLMs.inference.adapters import ModelProposer
from insideLLMs.types import ModelResponse, TokenUsage


class AsyncRecordingModel:
    name = "async-test"
    model_id = "async/test-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def agenerate_with_metadata(self, prompt: str, **kwargs: object) -> ModelResponse:
        self.calls.append((prompt, kwargs))
        return ModelResponse(
            content="answer",
            model=self.model_id,
            latency_ms=12.5,
            usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )


async def test_model_proposer_bridges_async_model_with_auditable_metadata() -> None:
    model = AsyncRecordingModel()
    proposer = ModelProposer(model, generation_kwargs={"temperature": 0.2})
    request = InferenceRequest(prompt="question", metadata={"request_id": "req-1"})

    first = await proposer.sample(request, 2)
    second = await proposer.sample(request, 2)

    assert [candidate.output for candidate in first] == ["answer", "answer"]
    assert first[0].id != first[1].id
    assert [candidate.id for candidate in first] == [candidate.id for candidate in second]
    assert first[0].metadata == {
        "model": "async/test-v1",
        "sample_index": 0,
        "latency_ms": 12.5,
        "prompt_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "request_id": "req-1",
    }
    assert model.calls == [("question", {"temperature": 0.2})] * 4


class SyncRecordingModel:
    name = "sync-test"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.calls.append(prompt)
        return prompt.upper()


async def test_model_proposer_supports_minimal_sync_model_protocol() -> None:
    model = SyncRecordingModel()

    candidates = await ModelProposer(model).sample(InferenceRequest(prompt="hello"), 1)

    assert candidates[0].output == "HELLO"
    assert candidates[0].metadata["model"] == "sync-test"
    assert model.calls == ["hello"]


async def test_model_proposer_rejects_non_positive_sample_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        await ModelProposer(SyncRecordingModel()).sample(InferenceRequest(prompt="hello"), 0)


async def test_candidate_identity_changes_when_stochastic_output_changes() -> None:
    model = AsyncRecordingModel()
    proposer = ModelProposer(model)
    request = InferenceRequest(prompt="question")

    first = await proposer.sample_one(request, 0)
    model.agenerate_with_metadata = lambda prompt, **kwargs: ModelResponse(  # type: ignore[method-assign]
        content="different answer",
        model=model.model_id,
    )
    second = await proposer.sample_one(request, 0)

    assert first.output != second.output
    assert first.id != second.id


class PartiallyFailingAsyncModel:
    name = "partially-failing"
    model_id = "async/failure-v1"

    def __init__(self) -> None:
        self.started = 0
        self.cancelled = 0
        self.all_started = asyncio.Event()

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        index = self.started
        self.started += 1
        if self.started == 3:
            self.all_started.set()
        await self.all_started.wait()
        if index == 0:
            raise RuntimeError("provider failure")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        raise AssertionError("blocking sample unexpectedly resumed")


async def test_parallel_sampling_cancels_and_drains_siblings_after_failure() -> None:
    model = PartiallyFailingAsyncModel()
    proposer = ModelProposer(model)

    with pytest.raises(RuntimeError, match="provider failure"):
        await proposer.sample(InferenceRequest(prompt="question"), 3)

    assert model.started == 3
    assert model.cancelled == 2
