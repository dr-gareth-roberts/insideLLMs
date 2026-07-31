"""Adapters between canonical insideLLMs models and inference protocols."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Mapping
from typing import Any

from insideLLMs.types import ModelResponse

from .schemas import Candidate, InferenceRequest


class ModelProposer:
    """Expose a canonical model-like object as an inference candidate proposer."""

    def __init__(
        self,
        model: object,
        *,
        generation_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self.model = model
        self.generation_kwargs = dict(generation_kwargs or {})

    async def sample(self, request: InferenceRequest, n: int) -> list[Candidate]:
        """Generate *n* candidates in stable sample-index order."""
        if n <= 0:
            raise ValueError("sample count must be positive")
        if not self._has_async_generation():
            # Worker-thread cancellation cannot stop an already-running sync call.
            return [await self.sample_one(request, index) for index in range(n)]

        tasks = [
            asyncio.create_task(self.sample_one(request, index), name=f"model-sample-{index}")
            for index in range(n)
        ]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def _has_async_generation(self) -> bool:
        return any(
            inspect.iscoroutinefunction(getattr(self.model, method_name, None))
            for method_name in ("agenerate_with_metadata", "agenerate")
        )

    async def sample_one(self, request: InferenceRequest, sample_index: int) -> Candidate:
        """Generate one candidate with an explicit deterministic sample index."""

        if sample_index < 0:
            raise ValueError("sample index must be non-negative")
        response = await self._generate(request.prompt)
        return self._candidate(request, response, sample_index)

    async def _generate(self, prompt: str) -> str | ModelResponse:
        async_metadata = getattr(self.model, "agenerate_with_metadata", None)
        if callable(async_metadata):
            return await _resolve(async_metadata(prompt, **self.generation_kwargs))

        async_generate = getattr(self.model, "agenerate", None)
        if callable(async_generate):
            return await _resolve(async_generate(prompt, **self.generation_kwargs))

        sync_metadata = getattr(self.model, "generate_with_metadata", None)
        if callable(sync_metadata):
            return await asyncio.to_thread(sync_metadata, prompt, **self.generation_kwargs)

        sync_generate = getattr(self.model, "generate", None)
        if callable(sync_generate):
            return await asyncio.to_thread(sync_generate, prompt, **self.generation_kwargs)

        raise TypeError("model must implement generate() or agenerate()")

    def _candidate(
        self,
        request: InferenceRequest,
        response: str | ModelResponse,
        sample_index: int,
    ) -> Candidate:
        model_name = _model_name(self.model, response)
        metadata: dict[str, Any] = {
            **request.metadata,
            "model": model_name,
            "sample_index": sample_index,
        }
        if isinstance(response, ModelResponse):
            output = response.content
            if response.latency_ms is not None:
                metadata["latency_ms"] = response.latency_ms
            if response.usage is not None:
                metadata.update(
                    prompt_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
        else:
            output = str(response)

        identity = json.dumps(
            [request.prompt, model_name, self.generation_kwargs, sample_index, output],
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        ).encode()
        candidate_id = hashlib.sha256(identity).hexdigest()[:16]
        return Candidate(id=candidate_id, output=output, metadata=metadata)


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _model_name(model: object, response: str | ModelResponse) -> str:
    if isinstance(response, ModelResponse) and response.model:
        return response.model
    return str(getattr(model, "model_id", getattr(model, "name", type(model).__name__)))
