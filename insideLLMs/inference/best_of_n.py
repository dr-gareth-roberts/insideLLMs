"""Best-of-N selection led by deterministic, fail-closed verification."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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

VerifierCallback = Callable[[Candidate], Verification | Awaitable[Verification]]
JudgeCallback = Callable[
    [InferenceRequest, tuple[str, ...]], Sequence[float] | Awaitable[Sequence[float]]
]


class NoVerifiedCandidateError(RuntimeError):
    """Raised when every candidate fails a hard verifier."""


@dataclass(frozen=True)
class VerifierSpec:
    id: str
    verify: VerifierCallback
    hard: bool = False


async def select_best(
    request: InferenceRequest,
    *,
    generate: Callable[
        [InferenceRequest, int], Sequence[Candidate] | Awaitable[Sequence[Candidate]]
    ],
    n: int,
    verifiers: Sequence[VerifierSpec],
    top_k: int = 1,
    judge: JudgeCallback | None = None,
    normalize: Callable[[Candidate], str | None] | None = None,
) -> InferenceResult:
    """Select by ordered verifiers, using an optional two-order blinded judge last."""

    if n < 1 or top_k < 1:
        raise ValueError("n and top_k must be positive")
    if not verifiers:
        raise ValueError("at least one verifier is required")
    ordered_verifiers = tuple(spec for spec in verifiers if spec.hard) + tuple(
        spec for spec in verifiers if not spec.hard
    )
    candidates = list(await resolve(generate(request, n)))
    if not candidates:
        raise ValueError("generate returned no candidates")
    candidate_ids = [candidate.id for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate ids must be unique")

    all_verifications: list[Verification] = []
    totals: dict[str, float] = {}
    passed_ids: set[str] = set()
    for candidate in candidates:
        total = 0.0
        passed = True
        for spec in ordered_verifiers:
            verification = await resolve(spec.verify(candidate))
            all_verifications.append(verification)
            total += verification.score
            if spec.hard and not verification.passed:
                passed = False
                break
        totals[candidate.id] = total
        if passed:
            passed_ids.add(candidate.id)

    eligible = [candidate for candidate in candidates if candidate.id in passed_ids]
    if not eligible:
        raise NoVerifiedCandidateError("no candidate passed hard verification")
    if judge is not None and len(eligible) > 1:
        forward = list(await resolve(judge(request, tuple(item.output for item in eligible))))
        reverse_items = list(reversed(eligible))
        reverse = list(await resolve(judge(request, tuple(item.output for item in reverse_items))))
        if len(forward) != len(eligible) or len(reverse) != len(eligible):
            raise ValueError("judge must return exactly one score per candidate")
        reverse_by_id = {item.id: score for item, score in zip(reverse_items, reverse)}
        for item, score in zip(eligible, forward):
            totals[item.id] += (score + reverse_by_id[item.id]) / 2

    ranked = sorted(eligible, key=lambda item: (-totals[item.id], item.id))
    top_candidates = ranked[:top_k]
    vote_counts: Counter[str] = Counter()
    winner = ranked[0]
    if normalize is not None:
        for candidate in top_candidates:
            normalized = normalize(candidate)
            if normalized is not None:
                vote_counts[normalized] += 1
        if vote_counts:
            winning_key = max(
                vote_counts,
                key=lambda key: (
                    vote_counts[key],
                    -next(
                        index
                        for index, candidate in enumerate(top_candidates)
                        if normalize(candidate) == key
                    ),
                ),
            )
            winner = next(
                candidate for candidate in top_candidates if normalize(candidate) == winning_key
            )
    return InferenceResult(
        answer=winner.output,
        confidence=totals[winner.id],
        candidates=tuple(candidates),
        verifications=tuple(all_verifications),
        trace=(
            TraceEvent(id="best-of-n-generate", kind="candidate-generation", calls=n),
            *(
                TraceEvent(
                    id=f"best-of-n-{candidate.id}",
                    kind="candidate-verification",
                    parent_ids=("best-of-n-generate",),
                    metadata={"score": totals[candidate.id]},
                )
                for candidate in candidates
            ),
        ),
        spend=Spend(calls=n + (2 if judge is not None and len(eligible) > 1 else 0)),
        stop_reason=StopReason.VERIFIED if passed_ids else StopReason.EXHAUSTED,
        provenance={
            "pass_at_n": bool(passed_ids),
            "oracle_best_score": max(totals.values()),
            "top_k_ids": tuple(item.id for item in top_candidates),
            "top_k_vote_counts": dict(vote_counts),
            "verifier_order": tuple(spec.id for spec in ordered_verifiers),
            "judge_order_debiased": judge is not None,
        },
    )
