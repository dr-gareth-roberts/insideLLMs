"""Confidence and disagreement driven adaptive-compute routing."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Awaitable

from ._callbacks import invoke_with_timeout
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
    if budget.max_seconds is not None and any(
        not inspect.iscoroutinefunction(step.run) for step in steps
    ):
        raise ValueError("escalation time budgets require asynchronous step callbacks")
    started = time.monotonic()
    candidates: list[Candidate] = []
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
        candidate = await invoke_with_timeout(
            step.run,
            request,
            candidates[-1] if candidates else None,
            timeout=remaining,
        )
        score = confidence(candidate)
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
    selected_confidence = confidence(selected)
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
