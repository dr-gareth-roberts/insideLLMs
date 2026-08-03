"""Budgeted beam search over states scored by objective feedback."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Awaitable, TypeVar

from ._callbacks import budget_elapsed, invoke_with_timeout, is_async_callable
from .schemas import Budget, InferenceResult, Spend, StopReason, TraceEvent

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")

# FIFO cap on the transposition cache so unbounded searches cannot pin memory.
_TRANSPOSITION_CACHE_LIMIT = 10_000


class SearchBudgetExceeded(RuntimeError):
    """Raised when the initial state cannot be scored inside the time budget."""


async def beam_search(
    initial_state: StateT,
    *,
    propose: Callable[[StateT], Sequence[ActionT] | Awaitable[Sequence[ActionT]]],
    transition: Callable[[StateT, ActionT], StateT | Awaitable[StateT]],
    value: Callable[[StateT], float | Awaitable[float]],
    key: Callable[[StateT], str],
    is_terminal: Callable[[StateT], bool],
    beam_width: int,
    budget: Budget,
    token_cost: Callable[[StateT], int] = lambda state: len(str(state)),
) -> InferenceResult:
    """Run deterministic beam search; an objective value callback is mandatory."""

    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if budget.max_seconds is not None and any(
        not is_async_callable(callback) for callback in (propose, transition, value)
    ):
        raise ValueError("search time budgets require asynchronous callbacks")
    started = time.monotonic()
    initial_tokens = token_cost(initial_state)
    if (
        (budget.max_calls is not None and budget.max_calls < 1)
        or (budget.max_evaluations is not None and budget.max_evaluations < 1)
        or (budget.max_nodes is not None and budget.max_nodes < 1)
        or (budget.max_tokens is not None and initial_tokens > budget.max_tokens)
    ):
        raise SearchBudgetExceeded("initial state exceeds search budget")

    calls = 0

    async def call(callback: Callable, *args: object):
        nonlocal calls
        if budget.max_calls is not None and calls >= budget.max_calls:
            raise SearchBudgetExceeded("search call budget exhausted")
        remaining = (
            None
            if budget.max_seconds is None
            else budget.max_seconds - (time.monotonic() - started)
        )
        if remaining is not None and remaining <= 0:
            raise SearchBudgetExceeded("search time budget exhausted")
        calls += 1
        try:
            return await invoke_with_timeout(callback, *args, timeout=remaining)
        except TimeoutError as error:
            if not budget_elapsed(started, budget.max_seconds):
                # The deadline did not fire: this is the callback's own error.
                raise
            raise SearchBudgetExceeded("search time budget exhausted") from error

    initial_score = await call(value, initial_state)
    evaluations = 1
    nodes = 1
    tokens = initial_tokens
    cache: dict[str, tuple[StateT, float]] = {key(initial_state): (initial_state, initial_score)}
    trace = [
        TraceEvent(
            id=key(initial_state),
            kind="search-state",
            calls=1,
            metadata={"value": initial_score, "generation": 0},
        )
    ]
    beam = [(initial_state, initial_score)]
    best = beam[0]
    generations = 0
    transposition_hits = 0
    stagnant_beams: set[frozenset[str]] = set()
    stop_reason = StopReason.VERIFIED if is_terminal(initial_state) else StopReason.EXHAUSTED

    while stop_reason is not StopReason.VERIFIED:
        if budget.max_generations is not None and generations >= budget.max_generations:
            stop_reason = StopReason.BUDGET
            break
        if budget.max_seconds is not None and time.monotonic() - started >= budget.max_seconds:
            stop_reason = StopReason.BUDGET
            break
        next_states: list[tuple[StateT, float]] = []
        next_keys: set[str] = set()
        budget_hit = False
        progressed = False
        for state, _ in beam:
            try:
                actions = await call(propose, state)
            except SearchBudgetExceeded:
                budget_hit = True
                break
            for action in actions:
                try:
                    child = await call(transition, state, action)
                except SearchBudgetExceeded:
                    budget_hit = True
                    break
                child_key = key(child)
                if child_key in cache:
                    transposition_hits += 1
                    if child_key not in next_keys:
                        next_states.append(cache[child_key])
                        next_keys.add(child_key)
                    continue
                child_tokens = token_cost(child)
                if (
                    (budget.max_evaluations is not None and evaluations >= budget.max_evaluations)
                    or (budget.max_nodes is not None and nodes >= budget.max_nodes)
                    or (budget.max_tokens is not None and tokens + child_tokens > budget.max_tokens)
                    or (
                        budget.max_seconds is not None
                        and time.monotonic() - started >= budget.max_seconds
                    )
                ):
                    # Break rather than continue: these counters are monotonic,
                    # so once a limit is hit no later child in this state can be
                    # admitted, and continuing still pays for every remaining
                    # action's (potentially model-backed) transition call inside
                    # the generation the budget was meant to stop.
                    budget_hit = True
                    break
                try:
                    score = await call(value, child)
                except SearchBudgetExceeded:
                    budget_hit = True
                    break
                evaluations += 1
                nodes += 1
                tokens += child_tokens
                progressed = True
                if len(cache) >= _TRANSPOSITION_CACHE_LIMIT:
                    del cache[next(iter(cache))]
                cache[child_key] = (child, score)
                next_states.append((child, score))
                next_keys.add(child_key)
                trace.append(
                    TraceEvent(
                        id=child_key,
                        kind="search-state",
                        parent_ids=(key(state),),
                        calls=1,
                        metadata={"value": score, "generation": generations + 1},
                    )
                )
            if budget_hit:
                break
        generations += 1
        if not next_states:
            stop_reason = StopReason.BUDGET if budget_hit else StopReason.EXHAUSTED
            break
        if budget_hit and not progressed:
            # Only cached states survived a budget-limited generation: no future
            # generation can evaluate anything new, so the search must not spin
            # forever re-appending transposition hits.
            stop_reason = StopReason.BUDGET
            break
        next_states.sort(key=lambda item: (-item[1], key(item[0])))
        beam = next_states[:beam_width]
        if (-beam[0][1], key(beam[0][0])) < (-best[1], key(best[0])):
            best = beam[0]
        terminal = next((item for item in beam if is_terminal(item[0])), None)
        if terminal is not None:
            best = terminal
            stop_reason = StopReason.VERIFIED
            continue
        # Cycle guard: a generation that evaluated nothing new and produced a
        # beam already seen since the last progress can only repeat forever
        # (transposition hits keep the beam populated on cyclic state graphs).
        if progressed:
            stagnant_beams.clear()
        else:
            signature = frozenset(key(state) for state, _ in beam)
            if signature in stagnant_beams:
                stop_reason = StopReason.BUDGET if budget_hit else StopReason.EXHAUSTED
                break
            stagnant_beams.add(signature)

    elapsed = time.monotonic() - started
    return InferenceResult(
        answer=best[0],
        confidence=best[1],
        spend=Spend(
            calls=calls,
            output_tokens=tokens,
            evaluations=evaluations,
            elapsed_seconds=elapsed,
        ),
        trace=tuple(trace),
        stop_reason=stop_reason,
        provenance={
            "generations": generations,
            "nodes": nodes,
            "transposition_hits": transposition_hits,
            "best_state_key": key(best[0]),
        },
    )
