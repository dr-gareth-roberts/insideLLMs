"""Self-consistency sampling with sequential, provably safe early stopping."""

from __future__ import annotations

from collections import Counter
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
)


async def sample_consistent(
    request: InferenceRequest,
    *,
    sample: Callable[[InferenceRequest, int], Candidate | Awaitable[Candidate]],
    normalize: Callable[[Candidate], str | None],
    max_samples: int,
) -> InferenceResult:
    """Sample until exhausted or the modal answer cannot be caught."""

    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    candidates: list[Candidate] = []
    counts: Counter[str] = Counter()
    first_output_by_key: dict[str, str] = {}
    stop_reason = StopReason.EXHAUSTED

    for index in range(max_samples):
        candidate = await resolve(sample(request, index))
        key = normalize(candidate)
        candidates.append(candidate)
        if key is not None:
            counts[key] += 1
            first_output_by_key.setdefault(key, candidate.output)
        ordered = counts.most_common(2)
        leader_count = ordered[0][1] if ordered else 0
        runner_count = ordered[1][1] if len(ordered) > 1 else 0
        remaining = max_samples - len(candidates)
        if leader_count > runner_count + remaining:
            stop_reason = StopReason.AGREEMENT
            break

    if counts:
        winning_key, votes = counts.most_common(1)[0]
        # Return a real sample output, not the normalization key: the default
        # normalizer casefolds, and the key must never masquerade as an answer.
        answer = first_output_by_key[winning_key]
    else:
        winning_key, votes = None, 0
        answer = candidates[0].output
    return InferenceResult(
        answer=answer,
        confidence=votes / len(candidates),
        candidates=tuple(candidates),
        trace=tuple(
            TraceEvent(
                id=candidate.id,
                kind="self-consistency-sample",
                calls=1,
                metadata={"normalized_answer": normalize(candidate)},
            )
            for candidate in candidates
        ),
        spend=Spend(calls=len(candidates)),
        stop_reason=stop_reason,
        provenance={
            "vote_counts": dict(counts),
            "winning_key": winning_key,
            "max_samples": max_samples,
        },
    )
