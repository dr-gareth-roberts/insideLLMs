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


def test_capability_predicates_reject_inherited_raising_stubs() -> None:
    """Model.stream and AsyncModel.astream are stubs that only raise.

    Reporting them as capabilities sends callers down a branch that cannot
    work; only genuine overrides count.
    """
    from insideLLMs.models.base import AsyncModel, Model, can_stream, can_stream_async

    class NoStreamAtAll(Model):
        def generate(self, prompt: str, **kwargs: object) -> str:
            return "out"

    class SyncStreamOnly(AsyncModel):
        def generate(self, prompt: str, **kwargs: object) -> str:
            return "out"

        async def agenerate(self, prompt: str, **kwargs: object) -> str:
            return "out"

        def stream(self, prompt: str, **kwargs: object):
            yield "chunk"

    class RealAsyncStream(AsyncModel):
        def generate(self, prompt: str, **kwargs: object) -> str:
            return "out"

        async def agenerate(self, prompt: str, **kwargs: object) -> str:
            return "out"

        async def astream(self, prompt: str, **kwargs: object):
            yield "async-chunk"

    # Inheriting the raising stub is not a capability.
    assert can_stream(NoStreamAtAll(name="none")) is False
    assert can_stream_async(SyncStreamOnly(name="sync")) is False
    # Genuine overrides still register.
    assert can_stream(SyncStreamOnly(name="sync")) is True
    assert can_stream_async(RealAsyncStream(name="real")) is True


async def test_pipeline_astream_falls_back_to_sync_stream() -> None:
    """An AsyncModel with only stream() must not hit the raising astream stub."""
    from insideLLMs.models.base import AsyncModel
    from insideLLMs.runtime.pipeline import ModelPipeline

    class SyncStreamOnly(AsyncModel):
        def generate(self, prompt: str, **kwargs: object) -> str:
            return "out"

        async def agenerate(self, prompt: str, **kwargs: object) -> str:
            return "out"

        def stream(self, prompt: str, **kwargs: object):
            yield "chunk"

    pipeline = ModelPipeline(SyncStreamOnly(name="sync"))
    assert [chunk async for chunk in pipeline.astream("hi")] == ["chunk"]


async def test_pipeline_achat_falls_back_to_sync_chat() -> None:
    """An AsyncModel with only chat() must not hit the raising achat stub.

    Peer of test_pipeline_astream_falls_back_to_sync_stream. The streaming
    dispatch was converted to can_stream_async, but ModelPipeline.achat and the
    middleware chat paths kept ``hasattr(model, "achat")``, which is True for
    every AsyncModel because ``achat`` is a concrete raising stub — so the
    documented run-in-executor fallback was unreachable.
    """
    from insideLLMs.models.base import AsyncModel, can_chat, can_chat_async
    from insideLLMs.runtime.pipeline import ModelPipeline

    class SyncChatOnly(AsyncModel):
        def generate(self, prompt: str, **kwargs: object) -> str:
            return "out"

        async def agenerate(self, prompt: str, **kwargs: object) -> str:
            return "async-out"

        def chat(self, messages: list, **kwargs: object) -> str:
            return "sync-chat"

    class RealAsyncChat(AsyncModel):
        def generate(self, prompt: str, **kwargs: object) -> str:
            return "out"

        async def agenerate(self, prompt: str, **kwargs: object) -> str:
            return "async-out"

        async def achat(self, messages: list, **kwargs: object) -> str:
            return "async-chat"

    sync_only = SyncChatOnly(name="sync")
    # Inheriting the raising stub is not a capability.
    assert can_chat_async(sync_only) is False
    assert can_chat(sync_only) is True
    assert can_chat_async(RealAsyncChat(name="real")) is True

    messages = [{"role": "user", "content": "hi"}]
    assert await ModelPipeline(sync_only).achat(messages) == "sync-chat"
    assert await ModelPipeline(RealAsyncChat(name="real")).achat(messages) == "async-chat"


async def test_middleware_aprocess_chat_falls_back_to_sync_chat() -> None:
    """The middleware chat path shares the achat stub-blindness fix."""
    from insideLLMs.models.base import AsyncModel
    from insideLLMs.runtime.pipeline import Middleware, ModelPipeline

    class SyncChatOnly(AsyncModel):
        def generate(self, prompt: str, **kwargs: object) -> str:
            return "out"

        async def agenerate(self, prompt: str, **kwargs: object) -> str:
            return "async-out"

        def chat(self, messages: list, **kwargs: object) -> str:
            return "sync-chat"

    class Passthrough(Middleware):
        def process_generate(self, prompt: str, **kwargs: object) -> str:
            return self.model.generate(prompt, **kwargs)

    pipeline = ModelPipeline(SyncChatOnly(name="sync"), middlewares=[Passthrough()])
    assert await pipeline.achat([{"role": "user", "content": "hi"}]) == "sync-chat"


async def test_proposer_resolves_async_def_generate() -> None:
    """A model whose ``generate`` is ``async def`` must not leak the coroutine.

    Regression: sync-named methods were handed straight to ``asyncio.to_thread``
    without resolving the result, so the coroutine object became the answer text
    ('<coroutine object ...>') plus a never-awaited RuntimeWarning.
    """
    from insideLLMs.inference import InferenceClient

    class AsyncDefGenerate:
        name = "adg"

        async def generate(self, prompt: str, **kwargs: object) -> str:
            return "real-answer"

    result = await InferenceClient(AsyncDefGenerate()).generate("q")
    assert result.answer == "real-answer"


async def test_proposer_prefers_metadata_over_metadataless_async() -> None:
    """Metadata-bearing generation wins, so Spend is not silently zeroed.

    Regression: ``agenerate`` was dispatched before ``generate_with_metadata``,
    so any model offering both (notably ``from_model_config`` pipelines) lost
    all token and latency accounting in the auditable envelope.
    """
    from insideLLMs.inference import InferenceClient
    from insideLLMs.types import ModelResponse, TokenUsage

    class BothPaths:
        name = "both"

        async def agenerate(self, prompt: str, **kwargs: object) -> str:
            return "async-no-metadata"

        def generate_with_metadata(self, prompt: str, **kwargs: object) -> ModelResponse:
            return ModelResponse(
                content="with-metadata",
                model="m1",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
                latency_ms=5.0,
            )

    result = await InferenceClient(BothPaths()).generate("q")
    assert result.answer == "with-metadata"
    assert result.spend.input_tokens == 10
    assert result.spend.output_tokens == 3
    assert result.spend.elapsed_seconds > 0


def test_proposer_async_detection_mirrors_dispatch() -> None:
    """``_has_async_generation`` must agree with the method ``_generate`` picks."""
    from insideLLMs.inference.adapters import ModelProposer

    class SyncMetadataPlusAsyncGenerate:
        name = "mixed"

        async def agenerate(self, prompt: str, **kwargs: object) -> str:
            return "async"

        def generate_with_metadata(self, prompt: str, **kwargs: object) -> str:
            return "sync-metadata"

    proposer = ModelProposer(SyncMetadataPlusAsyncGenerate())
    selected = proposer._select_generator()
    assert selected.__name__ == "generate_with_metadata"
    # Selection is sync, so sampling must not claim the concurrent path.
    assert proposer._has_async_generation() is False


async def test_execute_dag_propagates_callback_timeout_with_budget_remaining() -> None:
    """A provider timeout must not be relabelled while the budget is nearly full.

    Regression: the guard only asked "was a time budget armed?" (``remaining is
    None``), so with max_seconds=300 a callback raising its own TimeoutError at
    t=0 surfaced as DagBudgetExceeded('DAG time budget exhausted'), hiding the
    real fault behind a false budget verdict.
    """
    from insideLLMs.inference.dag import DagBudgetExceeded

    async def execute(node: PlanNode, dependencies: dict[str, object]) -> object:
        raise TimeoutError("provider-side HTTP timeout")

    async def reduce(observations: dict[str, object]) -> object:
        return observations

    with pytest.raises(TimeoutError, match="provider-side HTTP timeout"):
        await execute_dag(
            (PlanNode("a"),),
            execute=execute,
            reduce=reduce,
            budget=Budget(max_seconds=300),
        )

    # And a genuinely elapsed deadline still reports budget exhaustion.
    async def slow(node: PlanNode, dependencies: dict[str, object]) -> object:
        await asyncio.sleep(10)

    with pytest.raises(DagBudgetExceeded):
        await execute_dag(
            (PlanNode("a"),),
            execute=slow,
            reduce=reduce,
            budget=Budget(max_seconds=0.2),
        )


async def test_escalation_does_not_swallow_real_timeout() -> None:
    """A step's own timeout must surface, not be salvaged as budget exhaustion.

    Regression: escalate_adaptively returned the previous cheaper candidate with
    stop_reason=BUDGET whenever any TimeoutError arrived and a budget was armed,
    so a provider outage silently downgraded the answer and reported the wrong
    reason with almost the entire budget unused.
    """

    async def cheap(request: InferenceRequest, previous: Candidate | None) -> Candidate:
        return Candidate(id="cheap", output="cheap-answer")

    async def pricey(request: InferenceRequest, previous: Candidate | None) -> Candidate:
        raise TimeoutError("provider-side timeout on expensive model")

    def confidence(candidate: Candidate) -> float:
        return 0.1

    with pytest.raises(TimeoutError, match="provider-side timeout"):
        await escalate_adaptively(
            InferenceRequest(prompt="q"),
            steps=(
                EscalationStep(id="c", run=cheap, minimum_confidence=0.9),
                EscalationStep(id="p", run=pricey, minimum_confidence=0.9),
            ),
            confidence=confidence,
            budget=Budget(max_seconds=300),
        )


async def test_beam_search_propagates_callback_timeout_with_budget_remaining() -> None:
    """Peer of the DAG case for search.py."""

    async def value(state: object) -> float:
        raise TimeoutError("scorer provider timeout")

    async def propose(state: object) -> tuple[str, ...]:
        return ("go",)

    async def transition(state: object, action: str) -> object:
        return state

    with pytest.raises(TimeoutError, match="scorer provider timeout"):
        await beam_search(
            0,
            propose=propose,
            transition=transition,
            value=value,
            key=str,
            is_terminal=lambda state: False,
            beam_width=1,
            budget=Budget(max_seconds=300),
        )


async def test_self_consistency_junk_outputs_do_not_form_a_voting_bloc() -> None:
    """Empty normalizations must abstain, not cluster into one winning key.

    Regression: normalize_text strips punctuation and articles, so "!!!", "..."
    and "?!" all normalized to "". Returning that as a vote key made three
    unrelated junk answers a single bloc that outvoted the genuine modal answer.
    """

    class JunkModel:
        name = "junk"

        def __init__(self) -> None:
            self._outputs = iter(["4", "!!!", "...", "4", "?!"])

        async def agenerate(self, prompt: str, **kwargs: object) -> str:
            return next(self._outputs)

    result = await InferenceClient(JunkModel()).self_consistency("q", max_samples=5)
    assert result.answer == "4"
    assert "" not in result.provenance["vote_counts"]


def test_default_vote_normalizer_distinguishes_empty_from_unset() -> None:
    """An explicit empty normalized_answer must not fall back to raw output."""
    from insideLLMs.inference.client import _default_vote_normalizer

    # Explicitly normalized to empty -> abstain rather than reusing the output.
    assert _default_vote_normalizer(Candidate(id="a", output="junk", normalized_answer="")) is None
    # Unset -> derive from the output as before.
    assert _default_vote_normalizer(Candidate(id="b", output="Four")) == "four"


async def test_select_best_counts_actual_generations_and_real_judge_runs() -> None:
    """Spend.calls reflects produced candidates; a local judge can declare zero.

    Regression: calls came from the requested ``n`` (so an over-sampling
    generator slipped past the matched-compute ceiling) and always charged 2 for
    a judge, raising a spurious ComputeMismatchError for purely local judges.
    """

    def verify(candidate: Candidate) -> Verification:
        return Verification(verifier_id="v", passed=True, score=1.0)

    def over_sampling(request: InferenceRequest, n: int) -> tuple[Candidate, ...]:
        return tuple(Candidate(id=f"c{i}", output=f"o{i}") for i in range(n + 3))

    result = await select_best(
        InferenceRequest(prompt="q"),
        generate=over_sampling,
        n=2,
        verifiers=(VerifierSpec(id="v", verify=verify),),
    )
    assert len(result.candidates) == 5
    assert result.spend.calls == 5

    def local_judge(request: InferenceRequest, outputs: tuple[str, ...]) -> tuple[float, ...]:
        return tuple(float(len(output)) for output in outputs)

    local = await select_best(
        InferenceRequest(prompt="q"),
        generate=lambda r, n: tuple(Candidate(id=f"c{i}", output="x" * (i + 1)) for i in range(n)),
        n=2,
        verifiers=(VerifierSpec(id="v", verify=verify),),
        judge=local_judge,
        judge_model_calls=0,
    )
    assert local.spend.calls == 2
    assert local.provenance["judge_order_debiased"] is True

    # A judge that never runs must not be reported as having debiased anything.
    single = await select_best(
        InferenceRequest(prompt="q"),
        generate=lambda r, n: (Candidate(id="only", output="x"),),
        n=1,
        verifiers=(VerifierSpec(id="v", verify=verify),),
        judge=local_judge,
    )
    assert single.spend.calls == 1
    assert single.provenance["judge_order_debiased"] is False


async def test_execute_tool_does_not_retry_its_own_deadline() -> None:
    """The declared timeout must bound the wall clock, not be multiplied by retries.

    Regression: the retry clause caught TimeoutError, so an idempotent action
    with timeout_seconds=T and max_transport_attempts=N took roughly N*T plus
    backoff before failing.
    """

    async def hang(arguments: dict[str, object]) -> str:
        await asyncio.sleep(30)
        return "never"

    started = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError):
        await execute_tool(
            ToolAction(tool="t", arguments={}, idempotent=True),
            tools={"t": hang},
            allowed_tools={"t"},
            limits=ToolLimits(timeout_seconds=0.3, max_transport_attempts=3),
        )
    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 0.9, f"deadline was retried: {elapsed:.2f}s for a 0.3s timeout"


async def test_execute_tool_does_not_retry_deterministic_os_errors() -> None:
    """FileNotFoundError and friends can never succeed on retry."""
    calls = 0

    async def missing(arguments: dict[str, object]) -> str:
        nonlocal calls
        calls += 1
        raise FileNotFoundError("no such file")

    with pytest.raises(FileNotFoundError):
        await execute_tool(
            ToolAction(tool="t", arguments={}, idempotent=True),
            tools={"t": missing},
            allowed_tools={"t"},
            limits=ToolLimits(max_transport_attempts=3),
        )
    assert calls == 1


async def test_execute_tool_still_retries_connection_errors() -> None:
    """Genuine transport faults remain retryable."""
    calls = 0

    async def flaky(arguments: dict[str, object]) -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("connection reset")
        return "ok"

    observation = await execute_tool(
        ToolAction(tool="t", arguments={}, idempotent=True),
        tools={"t": flaky},
        allowed_tools={"t"},
        limits=ToolLimits(max_transport_attempts=3),
    )
    assert observation.output == "ok"
    assert observation.transport_attempts == 3


async def test_output_too_large_signals_the_tool_already_ran() -> None:
    """A post-execution violation must be distinguishable from a pre-flight one.

    Regression: exceeding max_output_characters raised bare ToolPolicyError,
    whose docstring promises the action was rejected *before* execution — so a
    caller could retry a non-idempotent action and duplicate its side effect.
    """
    from insideLLMs.inference import ToolOutputTooLarge

    ran = 0

    async def tool(arguments: dict[str, object]) -> str:
        nonlocal ran
        ran += 1
        return "x" * 100

    with pytest.raises(ToolOutputTooLarge):
        await execute_tool(
            ToolAction(tool="t", arguments={}),
            tools={"t": tool},
            allowed_tools={"t"},
            limits=ToolLimits(max_output_characters=10),
        )
    assert ran == 1
    # Still a ToolPolicyError, so existing handlers keep working.
    assert issubclass(ToolOutputTooLarge, ToolPolicyError)


async def test_evolution_parent_pool_has_no_duplicate_elites() -> None:
    """Elites must not appear twice in the selection view.

    Regression: parent_pool concatenated population with next_population, which
    is seeded from population's elites, so every elite was listed twice —
    doubling its draw probability in the default tournament and handing custom
    selectors a population containing duplicate candidate ids.
    """
    from insideLLMs.inference.evolution import EvolutionConfig, evolve_artifacts

    observed: list[tuple[int, int]] = []

    def selector(population: tuple, rng: object) -> object:
        ids = [candidate.id for candidate in population]
        observed.append((len(ids), len(set(ids))))
        return population[0]

    await evolve_artifacts(
        ["a", "b", "c"],
        evaluate=lambda text: float(len(text)),
        mutate=lambda candidate, rng: candidate.artifact_text + "x",
        select_parent=selector,
        config=EvolutionConfig(population_size=3, elite_count=2, max_generations=2, seed=0),
    )
    assert observed, "selector was never invoked"
    assert all(total == unique for total, unique in observed), observed


async def test_escalation_accepts_async_confidence_callback() -> None:
    """An async confidence must be awaited, not returned as a coroutine.

    Regression: confidence() was called bare, so an async callback produced a
    coroutine that crashed at ``1.0 - score`` after the step's model call had
    already been paid for.
    """

    async def cheap(request: InferenceRequest, previous: Candidate | None) -> Candidate:
        return Candidate(id="c", output="ans")

    async def async_confidence(candidate: Candidate) -> float:
        return 0.95

    result = await escalate_adaptively(
        InferenceRequest(prompt="q"),
        steps=(EscalationStep(id="c", run=cheap, minimum_confidence=0.9),),
        confidence=async_confidence,
    )
    assert result.answer == "ans"
    assert result.confidence == 0.95
    assert result.stop_reason is StopReason.VERIFIED

    # A synchronous confidence remains supported, including under a time budget:
    # it is typically a cheap local heuristic, and invoke() accepts either form.
    sync_result = await escalate_adaptively(
        InferenceRequest(prompt="q"),
        steps=(EscalationStep(id="c", run=cheap, minimum_confidence=0.9),),
        confidence=lambda candidate: 0.95,
        budget=Budget(max_seconds=5.0),
    )
    assert sync_result.confidence == 0.95


async def test_beam_search_stops_transitioning_once_budget_is_hit() -> None:
    """Budget exhaustion must break the action loop, not continue through it.

    Regression: the budget branch used ``continue``, so every remaining action of
    the current state still invoked the (potentially model-backed) transition
    callback even though the monotonic counters guaranteed no further child
    could be admitted.
    """
    transitions = 0

    async def propose(state: object) -> tuple[int, ...]:
        return tuple(range(20))

    async def transition(state: object, action: int) -> object:
        nonlocal transitions
        transitions += 1
        return (state[0] + 1, action)

    async def value(state: object) -> float:
        return 0.5

    result = await beam_search(
        (0, 0),
        propose=propose,
        transition=transition,
        value=value,
        key=str,
        is_terminal=lambda state: False,
        beam_width=1,
        budget=Budget(max_evaluations=3),
    )
    assert result.stop_reason is StopReason.BUDGET
    # Previously 40 for this shape: 20 actions per state across two generations.
    assert transitions <= 8, f"budget overrun: {transitions} transition calls"


async def test_select_best_cancels_sibling_verifications_on_failure() -> None:
    """A failed verifier must not leave sibling model calls running unobserved.

    Regression: bare asyncio.gather propagated the first exception immediately
    while the other verification chains kept executing, so provider calls
    continued after the caller had already raised and any second failure
    surfaced only as a "Task exception was never retrieved" warning at GC.
    """
    completed_after_raise: list[str] = []

    async def verify(candidate: Candidate) -> Verification:
        if candidate.id == "c1":
            raise ConnectionError("verifier boom")
        await asyncio.sleep(0.2)
        completed_after_raise.append(candidate.id)
        return Verification(verifier_id="v", passed=True, score=1.0)

    def generate(request: InferenceRequest, n: int) -> tuple[Candidate, ...]:
        return tuple(Candidate(id=f"c{i}", output=f"o{i}") for i in range(6))

    with pytest.raises(ConnectionError, match="verifier boom"):
        await select_best(
            InferenceRequest(prompt="q"),
            generate=generate,
            n=6,
            verifiers=(VerifierSpec(id="v", verify=verify),),
        )
    await asyncio.sleep(0.4)
    assert completed_after_raise == [], f"orphaned verifications: {completed_after_raise}"
