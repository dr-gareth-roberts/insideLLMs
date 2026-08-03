"""Adapters between canonical insideLLMs models and inference protocols."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from insideLLMs.types import ModelResponse

from ._callbacks import is_async_callable
from ._callbacks import resolve as _resolve
from .schemas import Candidate, InferenceRequest

# Accounting keys the proposer owns; caller-supplied request metadata must not
# be able to forge (or crash) spend accounting through them.
_RESERVED_METADATA_KEYS = frozenset(
    {"model", "sample_index", "latency_ms", "prompt_tokens", "output_tokens", "total_tokens"}
)


def _stable_json_default(value: object) -> str:
    # repr() of arbitrary objects embeds memory addresses, which would make
    # candidate IDs differ across runs; a type-based token stays deterministic.
    return f"<unserializable:{type(value).__module__}.{type(value).__qualname__}>"


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

    # Metadata-bearing sources come first: token/latency accounting is the point
    # of the auditable result envelope, and preferring a metadata-less
    # ``agenerate`` over ``generate_with_metadata`` silently zeroes Spend for
    # every model that offers both (notably the pipelines built by
    # ``InferenceClient.from_model_config``).
    _GENERATOR_NAMES = (
        "agenerate_with_metadata",
        "generate_with_metadata",
        "agenerate",
        "generate",
    )

    def _select_generator(self) -> Any:
        """Return the generation method ``_generate`` will actually call."""
        for method_name in self._GENERATOR_NAMES:
            method = getattr(self.model, method_name, None)
            if callable(method):
                return method
        return None

    def _has_async_generation(self) -> bool:
        # Derived from the same selection ``_generate`` performs, so the two can
        # never disagree about whether sampling may run concurrently.
        return is_async_callable(self._select_generator())

    async def sample_one(self, request: InferenceRequest, sample_index: int) -> Candidate:
        """Generate one candidate with an explicit deterministic sample index."""

        if sample_index < 0:
            raise ValueError("sample index must be non-negative")
        response = await self._generate(request.prompt)
        return self._candidate(request, response, sample_index)

    async def _generate(self, prompt: str) -> str | ModelResponse:
        method = self._select_generator()
        if method is None:
            raise TypeError("model must implement generate() or agenerate()")

        if is_async_callable(method):
            return await _resolve(method(prompt, **self.generation_kwargs))

        # A synchronously-named method may still be ``async def`` (or return an
        # awaitable). Handing that to a worker thread yields the coroutine
        # object itself as the answer text, so resolve whatever comes back.
        return await _resolve(await asyncio.to_thread(method, prompt, **self.generation_kwargs))

    def _candidate(
        self,
        request: InferenceRequest,
        response: str | ModelResponse,
        sample_index: int,
    ) -> Candidate:
        model_name = _model_name(self.model, response)
        metadata: dict[str, Any] = {
            **{
                key: value
                for key, value in request.metadata.items()
                if key not in _RESERVED_METADATA_KEYS
            },
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
            default=_stable_json_default,
        ).encode()
        candidate_id = hashlib.sha256(identity).hexdigest()[:16]
        return Candidate(id=candidate_id, output=output, metadata=metadata)


def _model_name(model: object, response: str | ModelResponse) -> str:
    if isinstance(response, ModelResponse) and response.model:
        return response.model
    return str(getattr(model, "model_id", getattr(model, "name", type(model).__name__)))
