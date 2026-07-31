"""Structured generation with deterministic validation and bounded repair."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Awaitable

from ._callbacks import resolve
from .schemas import (
    Candidate,
    InferenceRequest,
    InferenceResult,
    Spend,
    StopReason,
    TraceEvent,
    Verification,
)

ValidatorCallback = Callable[[Candidate], Verification | Awaitable[Verification]]


async def generate_validated(
    request: InferenceRequest,
    *,
    generate: Callable[[InferenceRequest], str | Awaitable[str]],
    validate: ValidatorCallback | Sequence[ValidatorCallback],
    repair: Callable[[Candidate, Verification], str | Awaitable[str]] | None = None,
    max_repairs: int = 1,
) -> InferenceResult:
    """Generate, objectively validate, and repair only failed output."""

    if max_repairs < 0:
        raise ValueError("max_repairs must be non-negative")
    validators = (validate,) if callable(validate) else tuple(validate)
    if not validators:
        raise ValueError("at least one validator is required")
    original = await resolve(generate(request))
    candidates = [Candidate(id="candidate-0", output=original)]

    async def verify_candidate(candidate: Candidate) -> list[Verification]:
        results: list[Verification] = []
        for validator in validators:
            result = await resolve(validator(candidate))
            results.append(result)
            if not result.passed:
                break
        return results

    candidate_verifications = [await verify_candidate(candidates[0])]
    verifications = list(candidate_verifications[0])
    calls = 1

    while (
        not candidate_verifications[-1][-1].passed and repair is not None and calls <= max_repairs
    ):
        revised = await resolve(repair(candidates[-1], candidate_verifications[-1][-1]))
        candidate = Candidate(
            id=f"candidate-{len(candidates)}",
            output=revised,
            parent_id=candidates[-1].id,
        )
        candidates.append(candidate)
        current_verifications = await verify_candidate(candidate)
        candidate_verifications.append(current_verifications)
        verifications.extend(current_verifications)
        calls += 1

    final = candidates[-1]
    trace = tuple(
        TraceEvent(
            id=candidate.id,
            kind="generation" if index == 0 else "structured-repair",
            parent_ids=(candidate.parent_id,) if candidate.parent_id else (),
            calls=1,
            metadata={
                "verification_passed": candidate_verifications[index][-1].passed,
                "validator_ids": tuple(item.verifier_id for item in candidate_verifications[index]),
            },
        )
        for index, candidate in enumerate(candidates)
    )
    return InferenceResult(
        answer=final.output,
        confidence=candidate_verifications[-1][-1].score,
        candidates=tuple(candidates),
        verifications=tuple(verifications),
        trace=trace,
        spend=Spend(calls=calls),
        stop_reason=(
            StopReason.VERIFIED if candidate_verifications[-1][-1].passed else StopReason.EXHAUSTED
        ),
        provenance={"original_output": original, "selected_candidate_id": final.id},
    )
