"""Blessed user-facing entry point for model-backed inference strategies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from time import perf_counter

from .adapters import ModelProposer
from .best_of_n import JudgeCallback, VerifierSpec, select_best
from .schemas import Candidate, InferenceRequest, InferenceResult, Spend, StopReason, TraceEvent
from .self_consistency import sample_consistent


class InferenceClient:
    """Run inference strategies against a canonical insideLLMs model."""

    def __init__(
        self,
        model: object,
        *,
        generation_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self.proposer = ModelProposer(model, generation_kwargs=generation_kwargs)

    @classmethod
    def from_model_config(
        cls,
        config: Mapping[str, object],
        *,
        generation_kwargs: Mapping[str, object] | None = None,
    ) -> InferenceClient:
        """Build through the canonical model registry and middleware config path."""

        from insideLLMs.runtime._config_loader import _create_model_from_config

        model = _create_model_from_config(dict(config), prefer_async_pipeline=True)
        return cls(model, generation_kwargs=generation_kwargs)

    @property
    def model(self) -> object:
        return self.proposer.model

    async def generate(self, request: str | InferenceRequest) -> InferenceResult:
        """Run one model call and return the shared auditable result envelope."""

        normalized_request = (
            request if isinstance(request, InferenceRequest) else InferenceRequest(prompt=request)
        )
        candidate = (await self.proposer.sample(normalized_request, 1))[0]
        return _one_shot_result(candidate)

    async def generate_many(
        self,
        request: str | InferenceRequest,
        *,
        n: int,
    ) -> tuple[InferenceResult, ...]:
        """Generate *n* independent one-shot results through the same proposer path."""

        normalized_request = (
            request if isinstance(request, InferenceRequest) else InferenceRequest(prompt=request)
        )
        candidates = await self.proposer.sample(normalized_request, n)
        return tuple(_one_shot_result(candidate) for candidate in candidates)

    async def self_consistency(
        self,
        request: str | InferenceRequest,
        *,
        max_samples: int,
        normalize: Callable[[Candidate], str | None] | None = None,
    ) -> InferenceResult:
        """Sample sequentially until the normalized modal answer is uncatchable."""

        normalized_request = (
            request if isinstance(request, InferenceRequest) else InferenceRequest(prompt=request)
        )
        normalizer = normalize or _default_vote_normalizer
        started = perf_counter()
        result = await sample_consistent(
            normalized_request,
            sample=self.proposer.sample_one,
            normalize=normalizer,
            max_samples=max_samples,
        )
        return _with_model_spend(
            result,
            concurrent=False,
            observed_elapsed=perf_counter() - started,
            strategy="self-consistency",
        )

    async def best_of_n(
        self,
        request: str | InferenceRequest,
        *,
        n: int,
        verifiers: Sequence[VerifierSpec],
        top_k: int = 1,
        judge: JudgeCallback | None = None,
        normalize: Callable[[Candidate], str | None] | None = None,
    ) -> InferenceResult:
        """Generate candidates and select only after ordered verification."""

        normalized_request = (
            request if isinstance(request, InferenceRequest) else InferenceRequest(prompt=request)
        )
        started = perf_counter()
        result = await select_best(
            normalized_request,
            generate=self.proposer.sample,
            n=n,
            verifiers=verifiers,
            top_k=top_k,
            judge=judge,
            normalize=normalize,
        )
        return _with_model_spend(
            result,
            concurrent=True,
            observed_elapsed=perf_counter() - started,
            strategy="best-of-n",
        )


def _default_vote_normalizer(candidate: Candidate) -> str | None:
    """Group votes with the same normalization every evaluator uses.

    Delegates to :func:`insideLLMs.analysis.evaluation.normalize_text` (lazily,
    to avoid an import cycle through the analysis package) so vote grouping and
    evaluation scoring agree on which answers are equal.
    """

    from insideLLMs.analysis.evaluation import normalize_text

    return normalize_text(candidate.normalized_answer or candidate.output)


def _one_shot_result(candidate: Candidate) -> InferenceResult:
    input_tokens = int(candidate.metadata.get("prompt_tokens", 0))
    output_tokens = int(candidate.metadata.get("output_tokens", 0))
    latency_seconds = float(candidate.metadata.get("latency_ms", 0.0)) / 1000
    model_name = str(candidate.metadata["model"])
    return InferenceResult(
        answer=candidate.output,
        confidence=1.0,
        candidates=(candidate,),
        trace=(
            TraceEvent(
                id=candidate.id,
                kind="model-generation",
                calls=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=latency_seconds,
                metadata={"model": model_name},
            ),
        ),
        spend=Spend(
            calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_seconds=latency_seconds,
        ),
        stop_reason=StopReason.COMPLETED,
        provenance={"strategy": "one-shot", "model": model_name},
    )


def _with_model_spend(
    result: InferenceResult,
    *,
    concurrent: bool,
    observed_elapsed: float,
    strategy: str,
) -> InferenceResult:
    input_tokens = sum(int(item.metadata.get("prompt_tokens", 0)) for item in result.candidates)
    output_tokens = sum(int(item.metadata.get("output_tokens", 0)) for item in result.candidates)
    latencies = [float(item.metadata.get("latency_ms", 0.0)) / 1000 for item in result.candidates]
    generation_elapsed = (max(latencies) if concurrent else sum(latencies)) if latencies else 0.0
    elapsed_seconds = max(generation_elapsed, observed_elapsed)
    candidate_metadata = {item.id: item.metadata for item in result.candidates}
    trace = []
    for event in result.trace:
        if event.kind == "candidate-generation":
            trace.append(
                replace(
                    event,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=generation_elapsed,
                )
            )
            continue
        metadata = candidate_metadata.get(event.id)
        if metadata is None:
            trace.append(event)
            continue
        trace.append(
            replace(
                event,
                input_tokens=int(metadata.get("prompt_tokens", 0)),
                output_tokens=int(metadata.get("output_tokens", 0)),
                latency_seconds=float(metadata.get("latency_ms", 0.0)) / 1000,
            )
        )
    return replace(
        result,
        trace=tuple(trace),
        spend=replace(
            result.spend,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_seconds=elapsed_seconds,
        ),
        provenance={
            **result.provenance,
            "strategy": strategy,
            "model": _result_model(result),
        },
    )


def _result_model(result: InferenceResult) -> str | tuple[str, ...]:
    models = tuple(
        sorted({str(item.metadata.get("model", "unknown")) for item in result.candidates})
    )
    return models[0] if len(models) == 1 else models
