"""Best-of-N selection led by deterministic, fail-closed verification."""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Awaitable, Literal

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


def rank_candidates(
    candidates: Sequence[Candidate],
    *,
    score: Callable[[Candidate], float],
    tie_break: Literal["candidate_id", "input_order"] = "candidate_id",
) -> tuple[tuple[Candidate, float], ...]:
    """Score once, then rank by ID or legacy input-order ``max`` semantics."""

    if tie_break not in {"candidate_id", "input_order"}:
        raise ValueError(f"unknown tie-break policy: {tie_break}")
    candidate_ids = [candidate.id for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate ids must be unique")
    scored = tuple((candidate, score(candidate)) for candidate in candidates)
    if tie_break == "candidate_id":
        return tuple(sorted(scored, key=lambda item: (-_orderable(item[1]), item[0].id)))

    remaining = list(scored)
    ranked: list[tuple[Candidate, float]] = []
    while remaining:
        winner_index = max(range(len(remaining)), key=lambda index: _orderable(remaining[index][1]))
        ranked.append(remaining.pop(winner_index))
    return tuple(ranked)


def _orderable(score: float) -> float:
    """Rank NaN scores last: NaN comparisons would make the sort order arbitrary."""

    return float("-inf") if math.isnan(score) else score


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

    async def _verify(candidate: Candidate) -> tuple[list[Verification], float, bool]:
        verifications: list[Verification] = []
        total = 0.0
        passed = True
        for spec in ordered_verifiers:
            verification = await resolve(spec.verify(candidate))
            verifications.append(verification)
            total += verification.score
            if spec.hard and not verification.passed:
                passed = False
                break
        return verifications, total, passed

    # Candidates verify concurrently; the hard-verifier short-circuit only
    # requires ordering within a single candidate's verifier chain.
    verified = await asyncio.gather(*(_verify(candidate) for candidate in candidates))
    all_verifications: list[Verification] = []
    totals: dict[str, float] = {}
    passed_ids: set[str] = set()
    for candidate, (verifications, total, passed) in zip(candidates, verified):
        all_verifications.extend(verifications)
        totals[candidate.id] = total
        if passed:
            passed_ids.add(candidate.id)

    eligible = [candidate for candidate in candidates if candidate.id in passed_ids]
    if not eligible:
        raise NoVerifiedCandidateError("no candidate passed hard verification")
    if judge is not None and len(eligible) > 1:
        reverse_items = list(reversed(eligible))
        forward_raw, reverse_raw = await asyncio.gather(
            resolve(judge(request, tuple(item.output for item in eligible))),
            resolve(judge(request, tuple(item.output for item in reverse_items))),
        )
        forward = list(forward_raw)
        reverse = list(reverse_raw)
        if len(forward) != len(eligible) or len(reverse) != len(eligible):
            raise ValueError("judge must return exactly one score per candidate")
        reverse_by_id = {item.id: score for item, score in zip(reverse_items, reverse)}
        for item, score in zip(eligible, forward):
            totals[item.id] += (score + reverse_by_id[item.id]) / 2

    ranked = [
        candidate
        for candidate, _ in rank_candidates(
            eligible,
            score=lambda candidate: totals[candidate.id],
        )
    ]
    top_candidates = ranked[:top_k]
    vote_counts: Counter[str] = Counter()
    winner = ranked[0]
    if normalize is not None:
        # Normalize once per candidate: a costly or stateful callback must not
        # be able to disagree with itself between counting and winner lookup.
        normalized_by_index = [normalize(candidate) for candidate in top_candidates]
        for normalized in normalized_by_index:
            if normalized is not None:
                vote_counts[normalized] += 1
        if vote_counts:
            winning_key = max(
                vote_counts,
                key=lambda key: (
                    vote_counts[key],
                    -normalized_by_index.index(key),
                ),
            )
            winner = top_candidates[normalized_by_index.index(winning_key)]
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
        # Reaching here means at least one candidate passed hard verification;
        # the empty case already raised NoVerifiedCandidateError.
        stop_reason=StopReason.VERIFIED,
        provenance={
            # A measured count, not the constant True a bool(passed_ids) would be.
            "verified_candidates": len(eligible),
            # Only eligible candidates: a hard-failed candidate's total is a
            # partial sum over fewer verifiers and is not on a comparable scale.
            "oracle_best_score": max(totals[candidate.id] for candidate in eligible),
            "top_k_ids": tuple(item.id for item in top_candidates),
            "top_k_vote_counts": dict(vote_counts),
            "verifier_order": tuple(spec.id for spec in ordered_verifiers),
            "verifier_invocations": len(all_verifications),
            "judge_order_debiased": judge is not None,
        },
    )
