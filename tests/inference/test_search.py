import pytest

from insideLLMs.inference import Budget, StopReason
from insideLLMs.inference.search import SearchBudgetExceeded, beam_search


async def test_beam_search_uses_objective_feedback_budgets_and_transposition_cache() -> None:
    evaluated: list[int] = []

    def value(state: int) -> float:
        evaluated.append(state)
        return -abs(3 - state)

    result = await beam_search(
        0,
        propose=lambda state: (1, 1),
        transition=lambda state, action: state + action,
        value=value,
        key=str,
        is_terminal=lambda state: state == 3,
        beam_width=2,
        budget=Budget(max_evaluations=4, max_generations=3),
    )

    assert result.answer == 3
    assert result.stop_reason is StopReason.VERIFIED
    assert evaluated == [0, 1, 2, 3]
    assert result.provenance["transposition_hits"] == 3
    assert result.spend.evaluations == 4


async def test_beam_search_preflights_initial_state_against_zero_budgets() -> None:
    evaluated = False

    def value(state: int) -> float:
        nonlocal evaluated
        evaluated = True
        return 1.0

    with pytest.raises(SearchBudgetExceeded, match="initial state"):
        await beam_search(
            0,
            propose=lambda state: (),
            transition=lambda state, action: state,
            value=value,
            key=str,
            is_terminal=lambda state: False,
            beam_width=1,
            budget=Budget(max_evaluations=0, max_nodes=0, max_tokens=0),
        )

    assert evaluated is False


async def test_beam_search_preflights_initial_call_budget() -> None:
    evaluated = False

    def value(state: int) -> float:
        nonlocal evaluated
        evaluated = True
        return 1.0

    with pytest.raises(SearchBudgetExceeded, match="initial state"):
        await beam_search(
            0,
            propose=lambda state: (),
            transition=lambda state, action: state,
            value=value,
            key=str,
            is_terminal=lambda state: False,
            beam_width=1,
            budget=Budget(max_calls=0),
        )

    assert evaluated is False


async def test_beam_search_call_budget_counts_propose_and_transition_callbacks() -> None:
    proposed = 0
    transitioned = 0

    def propose(state: int) -> tuple[str, ...]:
        nonlocal proposed
        proposed += 1
        return ("next",)

    def transition(state: int, action: str) -> int:
        nonlocal transitioned
        transitioned += 1
        return state + 1

    result = await beam_search(
        0,
        propose=propose,
        transition=transition,
        value=float,
        key=str,
        is_terminal=lambda state: False,
        beam_width=1,
        budget=Budget(max_calls=2),
    )

    assert proposed == 1
    assert transitioned == 0
    assert result.spend.calls == 2
    assert result.stop_reason is StopReason.BUDGET


async def test_beam_search_equal_scores_use_state_key_tie_break() -> None:
    result = await beam_search(
        "z",
        propose=lambda state: ("a",) if state == "z" else (),
        transition=lambda state, action: action,
        value=lambda state: 1.0,
        key=str,
        is_terminal=lambda state: False,
        beam_width=1,
        budget=Budget(max_generations=1),
    )

    assert result.answer == "a"
    assert result.provenance["best_state_key"] == "a"


async def test_timed_search_rejects_sync_callbacks_before_evaluation() -> None:
    evaluated = False

    def value(state: int) -> float:
        nonlocal evaluated
        evaluated = True
        return 1.0

    with pytest.raises(ValueError, match="asynchronous"):
        await beam_search(
            0,
            propose=lambda state: (),
            transition=lambda state, action: state,
            value=value,
            key=str,
            is_terminal=lambda state: False,
            beam_width=1,
            budget=Budget(max_seconds=0.1),
        )

    assert evaluated is False
