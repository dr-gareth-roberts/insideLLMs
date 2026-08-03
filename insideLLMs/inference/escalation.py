"""Confidence and disagreement driven adaptive-compute routing."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Awaitable

from ._callbacks import budget_elapsed, invoke_with_timeout, is_async_callable
from .schemas import (
    Budget,
    Candidate,
    InferenceRequest,
    InferenceResult,
    Spend,
    StopReason,
    TraceEvent,
)

StepCallback = Callable[[InferenceRequest, Candidate | None], Candidate | Awaitable[Candidate]]


@dataclass(frozen=True)
class EscalationStep:
    id: str
    run: StepCallback
    minimum_confidence: float
    cost: float = 0.0


async def escalate_adaptively(
    request: InferenceRequest,
    *,
    steps: Sequence[EscalationStep],
    confidence: Callable[[Candidate], float],
    budget: Budget = Budget(),
) -> InferenceResult:
    """Start cheaply and run stronger actions only while uncertainty remains."""

    if not steps:
        raise ValueError("at least one escalation step is required")
    if budget.max_calls is not None and budget.max_calls < 1:
        raise ValueError("escalation budget must permit at least one call")
    if budget.max_seconds is not None and budget.max_seconds <= 0:
        raise ValueError("escalation budget must permit positive time")
    if budget.max_seconds is not None and any(not is_async_callable(step.run) for step in steps):
        raise ValueError("escalation time budgets require asynchronous step callbacks")
    started = time.monotonic()
    candidates: list[Candidate] = []
    scores: list[float] = []
    trace: list[TraceEvent] = []
    actions: list[str] = []
    total_cost = 0.0
    stop_reason = StopReason.EXHAUSTED

    for step in steps:
        if budget.max_calls is not None and len(candidates) >= budget.max_calls:
            stop_reason = StopReason.BUDGET
            break
        remaining = (
            None
            if budget.max_seconds is None
            else budget.max_seconds - (time.monotonic() - started)
        )
        if remaining is not None and remaining <= 0:
            if not candidates:
                raise TimeoutError("escalation time budget exhausted before first step")
            stop_reason = StopReason.BUDGET
            break
        try:
            candidate = await invoke_with_timeout(
                step.run,
                request,
                candidates[-1] if candidates else None,
                timeout=remaining,
            )
        except TimeoutError:
            if not budget_elapsed(started, budget.max_seconds) or not candidates:
                # Either the step's own error (the deadline did not fire) or
                # there is no completed work to return. Salvaging a cheaper
                # answer here would silently swallow a real provider timeout and
                # report it as StopReason.BUDGET.
                raise
            stop_reason = StopReason.BUDGET
            break
        score = confidence(candidate)
        scores.append(score)
        candidates.append(candidate)
        actions.append(step.id)
        total_cost += step.cost
        trace.append(
            TraceEvent(
                id=f"escalation-{len(trace)}",
                kind=step.id,
                parent_ids=(trace[-1].id,) if trace else (),
                calls=1,
                cost=step.cost,
                metadata={
                    "confidence": score,
                    "uncertainty": 1.0 - score,
                    "threshold": step.minimum_confidence,
                },
            )
        )
        if score >= step.minimum_confidence:
            stop_reason = StopReason.VERIFIED
            break

    selected = candidates[-1]
    # Reuse the loop's score: re-invoking a stateful/model-backed confidence
    # callback could contradict the stop decision and hide an extra call.
    selected_confidence = scores[-1]
    return InferenceResult(
        answer=selected.output,
        confidence=selected_confidence,
        candidates=tuple(candidates),
        trace=tuple(trace),
        spend=Spend(calls=len(candidates), cost=round(total_cost, 12)),
        stop_reason=stop_reason,
        provenance={
            "actions": tuple(actions),
            "risk_coverage": tuple(
                (event.metadata["uncertainty"], index + 1) for index, event in enumerate(trace)
            ),
        },
    )
