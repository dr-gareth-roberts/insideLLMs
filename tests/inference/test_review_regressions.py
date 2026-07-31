"""Regression pins for the applied code-review findings."""

import asyncio
import math

import pytest

from insideLLMs.inference import (
    Budget,
    Candidate,
    InferenceRequest,
    StopReason,
    ToolAction,
    ToolLimits,
    ToolPolicyError,
    beam_search,
    execute_tool,
    run_sync,
)
from insideLLMs.inference.adapters import ModelProposer
from insideLLMs.inference.best_of_n import VerifierSpec, rank_candidates, select_best
from insideLLMs.inference.client import InferenceClient
from insideLLMs.inference.dag import PlanNode, execute_dag
from insideLLMs.inference.escalation import EscalationStep, escalate_adaptively
from insideLLMs.inference.schemas import Verification


async def test_beam_search_terminates_on_cyclic_state_graph() -> None:
    """Cached transposition hits must not keep a budget-exhausted search alive."""

    result = await asyncio.wait_for(
        beam_search(
            0,
            propose=lambda state: ("stay",),
            transition=lambda state, action: state,
            value=lambda state: 0.5,
            key=str,
            is_terminal=lambda state: False,
            beam_width=1,
            budget=Budget(max_evaluations=10),
        ),
        timeout=5.0,
    )
    assert result.stop_reason in (StopReason.BUDGET, StopReason.EXHAUSTED)


async def test_beam_search_reraises_callback_timeout_without_time_budget() -> None:
    def value(state: object) -> float:
        raise TimeoutError("network timeout from callback")

    with pytest.raises(TimeoutError, match="network timeout"):
        await beam_search(
            0,
            propose=lambda state: ("go",),
            transition=lambda state, action: state + 1,
            value=value,
            key=str,
            is_terminal=lambda state: False,
            beam_width=1,
            budget=Budget(max_evaluations=3),
        )


async def test_execute_dag_reraises_callback_timeout_without_time_budget() -> None:
    async def execute(node: PlanNode, dependencies: dict[str, object]) -> object:
        raise TimeoutError("provider timeout")

    with pytest.raises(TimeoutError, match="provider timeout"):
        await execute_dag(
            (PlanNode("a"),),
            execute=execute,
            reduce=lambda observations: observations,
            budget=Budget(),
        )


async def test_execute_dag_accepts_async_callable_object_with_time_budget() -> None:
    class AsyncExecutor:
        async def __call__(self, node: PlanNode, dependencies: dict[str, object]) -> object:
            return node.id

    async def reduce(observations: dict[str, object]) -> object:
        return sorted(observations)

    result = await execute_dag(
        (PlanNode("a"), PlanNode("b")),
        execute=AsyncExecutor(),
        reduce=reduce,
        budget=Budget(max_seconds=5.0),
    )
    assert result.answer == ["a", "b"]


async def test_execute_dag_rejects_sync_reduce_under_time_budget() -> None:
    """A sync reduce would keep running in its worker thread past the deadline."""

    async def execute(node: PlanNode, dependencies: dict[str, object]) -> object:
        return node.id

    with pytest.raises(ValueError, match="asynchronous reduce"):
        await execute_dag(
            (PlanNode("a"),),
            execute=execute,
            reduce=lambda observations: observations,
            budget=Budget(max_seconds=5.0),
        )


def test_rank_candidates_places_nan_scores_last() -> None:
    candidates = (
        Candidate(id="a", output="a"),
        Candidate(id="b", output="b"),
    )
    scores = {"a": float("nan"), "b": 1.0}
    ranked = rank_candidates(candidates, score=lambda candidate: scores[candidate.id])
    assert [candidate.id for candidate, _ in ranked] == ["b", "a"]


async def test_select_best_oracle_score_covers_only_eligible_candidates() -> None:
    def generate(request: InferenceRequest, n: int) -> tuple[Candidate, ...]:
        return (
            Candidate(id="x", output="x"),
            Candidate(id="y", output="y"),
        )

    def hard_a(candidate: Candidate) -> Verification:
        score = 5.0 if candidate.id == "x" else 1.0
        return Verification(verifier_id="hardA", passed=True, score=score)

    def hard_b(candidate: Candidate) -> Verification:
        return Verification(verifier_id="hardB", passed=candidate.id == "y", score=0.5)

    result = await select_best(
        InferenceRequest(prompt="q"),
        generate=generate,
        n=2,
        verifiers=(
            VerifierSpec(id="hardA", verify=hard_a, hard=True),
            VerifierSpec(id="hardB", verify=hard_b, hard=True),
        ),
    )
    assert result.answer == "y"
    assert result.provenance["oracle_best_score"] == pytest.approx(1.5)
    assert result.provenance["verifier_invocations"] == 4


async def test_client_generate_ignores_forged_accounting_metadata() -> None:
    class PlainModel:
        name = "plain"

        async def agenerate(self, prompt: str, **kwargs: object) -> str:
            return "out"

    client = InferenceClient(PlainModel())
    result = await client.generate(
        InferenceRequest(
            prompt="hi",
            metadata={"prompt_tokens": 500, "latency_ms": "n/a", "keep": "me"},
        )
    )
    assert result.spend.input_tokens == 0
    assert result.spend.elapsed_seconds == 0.0
    assert result.candidates[0].metadata["keep"] == "me"


def test_candidate_ids_stable_for_non_json_generation_kwargs() -> None:
    class Weird:
        pass

    proposer_a = ModelProposer(object(), generation_kwargs={"processor": Weird()})
    proposer_b = ModelProposer(object(), generation_kwargs={"processor": Weird()})
    request = InferenceRequest(prompt="p")
    candidate_a = proposer_a._candidate(request, "out", 0)
    candidate_b = proposer_b._candidate(request, "out", 0)
    assert candidate_a.id == candidate_b.id


async def test_escalation_scores_each_candidate_once_and_salvages_timeout() -> None:
    scored: list[str] = []

    async def cheap(request: InferenceRequest, previous: Candidate | None) -> Candidate:
        return Candidate(id="cheap", output="cheap-answer")

    async def slow(request: InferenceRequest, previous: Candidate | None) -> Candidate:
        await asyncio.sleep(10)
        return Candidate(id="slow", output="slow-answer")

    def confidence(candidate: Candidate) -> float:
        scored.append(candidate.id)
        return 0.1

    result = await escalate_adaptively(
        InferenceRequest(prompt="q"),
        steps=(
            EscalationStep(id="cheap", run=cheap, minimum_confidence=0.9),
            EscalationStep(id="slow", run=slow, minimum_confidence=0.9),
        ),
        confidence=confidence,
        budget=Budget(max_seconds=0.3),
    )
    assert result.answer == "cheap-answer"
    assert result.stop_reason is StopReason.BUDGET
    # The confidence callback ran exactly once per completed candidate.
    assert scored == ["cheap"]


async def test_execute_tool_rejects_sync_policy_under_timeout() -> None:
    async def tool(arguments: dict[str, object]) -> str:
        return "ok"

    def sync_policy(action: ToolAction) -> bool:
        return True

    with pytest.raises(ToolPolicyError, match="asynchronous policy"):
        await execute_tool(
            ToolAction(tool="t", arguments={}),
            tools={"t": tool},
            allowed_tools={"t"},
            limits=ToolLimits(timeout_seconds=0.5),
            policy=sync_policy,
        )


def test_run_sync_accepts_non_coroutine_awaitables() -> None:
    class BoxedAwaitable:
        def __await__(self):
            async def _inner() -> int:
                return 42

            return _inner().__await__()

    assert run_sync(BoxedAwaitable()) == 42


def test_nan_confidence_never_selected_over_finite_scores() -> None:
    values = [float("nan"), 2.0, 1.0]
    candidates = tuple(
        Candidate(id=f"c{index}", output=str(value)) for index, value in enumerate(values)
    )
    scores = {f"c{index}": value for index, value in enumerate(values)}
    ranked = rank_candidates(candidates, score=lambda candidate: scores[candidate.id])
    assert not math.isnan(ranked[0][1])
    # NaN maps to -inf, so it must land last rather than anywhere in the middle.
    assert [candidate.id for candidate, _ in ranked] == ["c1", "c2", "c0"]


def test_streaming_dispatch_idioms_agree_on_simulated_streaming() -> None:
    """can_stream() and supports_streaming answer distinct, non-contradictory questions."""
    from insideLLMs.models.base import can_stream, can_stream_async

    class SimulatedStreamModel:
        """Stands in for HuggingFaceModel: stream() works but is not native."""

        _supports_streaming = False

        def generate(self, prompt: str, **kwargs: object) -> str:
            return "whole response"

        def stream(self, prompt: str, **kwargs: object):
            yield self.generate(prompt)

    model = SimulatedStreamModel()
    # The capability flag stays honest: no native, incremental streaming...
    assert model._supports_streaming is False
    # ...while the predicate that actually gates a stream() call says yes, so a
    # capability-gated caller and a presence-gated caller no longer contradict.
    assert can_stream(model) is True
    assert can_stream_async(model) is False
    assert list(model.stream("q")) == ["whole response"]


def test_huggingface_reports_simulated_streaming_consistently() -> None:
    pytest.importorskip("transformers")
    from insideLLMs.models.base import can_stream
    from insideLLMs.models.huggingface import HuggingFaceModel

    assert HuggingFaceModel._supports_streaming is False
    # stream() is still present and callable, so the pipeline path keeps working.
    assert can_stream(HuggingFaceModel)


async def test_shared_timeout_shim_raises_builtin_timeout_error() -> None:
    """The pre-3.11 asyncio.TimeoutError split is normalized in the shared layer."""
    from insideLLMs.async_utils import wait_for

    async def slow() -> None:
        await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        await wait_for(slow(), 0.01)

    # The normalized type is what RetryConfig.retryable_exceptions lists.
    from insideLLMs.retry import RetryConfig

    assert TimeoutError in RetryConfig.retryable_exceptions


async def test_async_timeout_context_manager_raises_builtin_timeout_error() -> None:
    from insideLLMs.async_utils import async_timeout

    with pytest.raises(TimeoutError):
        async with async_timeout(0.01):
            await asyncio.sleep(10)


async def test_inference_timeout_delegates_to_shared_shim() -> None:
    from insideLLMs.inference._callbacks import invoke_with_timeout

    async def slow(_arg: object) -> None:
        await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        await invoke_with_timeout(slow, "x", timeout=0.01)
