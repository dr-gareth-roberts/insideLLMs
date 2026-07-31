"""Tool-grounded critique and repair with monotonic acceptance."""

from __future__ import annotations

from collections.abc import Callable
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


async def repair_with_evidence(
    request: InferenceRequest,
    original: Candidate,
    *,
    verify: Callable[[Candidate], Verification | Awaitable[Verification]],
    revise: Callable[[Candidate, object, int], str | Awaitable[str]],
    max_rounds: int = 1,
) -> InferenceResult:
    """Revise only verified failures and never replace an answer with a lower score."""

    if max_rounds < 0:
        raise ValueError("max_rounds must be non-negative")
    selected = original
    candidates = [original]
    verifications = [await resolve(verify(original))]
    selected_verification = verifications[0]
    accepted: list[str] = []
    wrong_to_right = 0
    right_to_wrong = 0
    repair_calls = 0

    for round_index in range(max_rounds):
        if selected_verification.passed:
            break
        output = await resolve(revise(selected, selected_verification.evidence, round_index))
        repair_calls += 1
        revision = Candidate(
            id=f"repair-{round_index + 1}",
            output=output,
            parent_id=selected.id,
        )
        revision_verification = await resolve(verify(revision))
        candidates.append(revision)
        verifications.append(revision_verification)
        if (revision_verification.passed, revision_verification.score) > (
            selected_verification.passed,
            selected_verification.score,
        ):
            if not selected_verification.passed and revision_verification.passed:
                wrong_to_right += 1
            if selected_verification.passed and not revision_verification.passed:
                right_to_wrong += 1
            selected = revision
            selected_verification = revision_verification
            accepted.append(revision.id)

    return InferenceResult(
        answer=selected.output,
        confidence=selected_verification.score,
        candidates=tuple(candidates),
        verifications=tuple(verifications),
        trace=tuple(
            TraceEvent(
                id=candidate.id,
                kind="original" if index == 0 else "grounded-repair",
                parent_ids=(candidate.parent_id,) if candidate.parent_id else (),
                calls=1,
                metadata={
                    "score": verifications[index].score,
                    "accepted": candidate.id == original.id or candidate.id in accepted,
                },
            )
            for index, candidate in enumerate(candidates)
        ),
        spend=Spend(calls=1 + repair_calls),
        stop_reason=(StopReason.VERIFIED if selected_verification.passed else StopReason.EXHAUSTED),
        provenance={
            "original_output": original.output,
            "accepted_revision_ids": tuple(accepted),
            "wrong_to_right": wrong_to_right,
            "right_to_wrong": right_to_wrong,
        },
    )
