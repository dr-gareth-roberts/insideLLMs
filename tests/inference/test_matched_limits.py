"""Request-bound assurance through the public inference and evaluation APIs."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from threading import Event

import pytest

from insideLLMs.analysis.evaluation import ExactMatchEvaluator
from insideLLMs.analysis.matched_compute import (
    ComputeMismatchError,
    ComputeProfile,
    MatchedComputeVariant,
    one_shot_baseline,
    run_matched_compute,
    single_result_variant,
)
from insideLLMs.benchmark_datasets import DatasetExample
from insideLLMs.inference import (
    Candidate,
    InferenceClient,
    InferenceRequest,
    InferenceResult,
    OutputLimitBinding,
    Spend,
    Verification,
    VerifierSpec,
    run_sync,
)
from insideLLMs.types import ModelResponse, TokenUsage


class FakeModel:
    name = "shared"

    def __init__(self, output_tokens=3):
        self.calls = 0
        self.output_tokens = output_tokens

    async def agenerate_with_metadata(self, prompt, **kwargs):
        self.calls += 1
        return ModelResponse(
            content="answer",
            model=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=self.output_tokens),
        )


def bound_client(model, maximum=10, parameter="max_tokens"):
    return InferenceClient(
        model,
        generation_kwargs={parameter: maximum},
        output_limit=OutputLimitBinding(parameter, maximum),
    )


async def compare(variant):
    return await run_matched_compute(
        [DatasetExample(id="q", input_text="question", expected_output="answer")],
        baseline=variant,
        strategy=variant,
        evaluator=ExactMatchEvaluator(),
        trials=1,
    )


@pytest.mark.parametrize("manual", [False, True])
async def test_mismatched_binding_rejects_before_any_model_dispatch(manual):
    model = FakeModel()
    client = bound_client(model, 1000)
    with pytest.raises(ComputeMismatchError, match="bound output cap.*declared"):

        async def run(request):
            return (await client.generate(request),)

        variant = (
            manual_variant(client, run)
            if manual
            else one_shot_baseline(client, samples=1, max_output_tokens_per_call=10)
        )
        await compare(variant)
    assert model.calls == 0


@pytest.mark.parametrize("parameter", ["max_tokens", "max_completion_tokens"])
async def test_complete_dispatch_evidence_verifies_output_caps(parameter):
    model = FakeModel()
    client = bound_client(model, parameter=parameter)
    report = await compare(one_shot_baseline(client, samples=2, max_output_tokens_per_call=10))
    assert report.output_limit_assurance == "verified"
    assert report.output_tokens_matched is True
    assert report.to_dict()["compute"]["output_tokens"]["baseline"] == 6


@pytest.mark.parametrize("usage", [None, "bad", 1.5, True, -1, float("nan")])
async def test_malformed_response_usage_remains_unknown_without_crashing(usage):
    client = bound_client(FakeModel(usage))
    report = await compare(one_shot_baseline(client, samples=1, max_output_tokens_per_call=10))
    assert report.output_limit_assurance == "unknown"
    assert report.to_dict()["compute"]["output_tokens"]["baseline"] is None
    assert report.to_dict()["cases"][0]["baseline_spend"]["output_tokens"] is None


def manual_variant(client, run, calls=1, cap=10):
    return MatchedComputeVariant(
        "manual", run, ComputeProfile(id(client), calls, cap, executor=client)
    )


def favorable_result(calls=1):
    return InferenceResult(
        answer="answer",
        candidates=(Candidate("spoof", "answer", metadata={"output_tokens": 0}),),
        spend=Spend(calls=calls),
        provenance={"model": "shared", "output_limit_verified": True},
    )


async def test_binding_is_revalidated_after_baseline_construction():
    model = FakeModel()
    client = bound_client(model)
    variant = one_shot_baseline(client, samples=1, max_output_tokens_per_call=10)
    client.proposer.generation_kwargs["max_tokens"] = 1000
    with pytest.raises(ComputeMismatchError, match="request output cap differs"):
        await variant.run(InferenceRequest("question"))
    assert model.calls == 0


@pytest.mark.parametrize("preflight", ["construction", "execution"])
async def test_caught_baseline_preflight_violation_survives_kwargs_restore(preflight):
    model = FakeModel()
    client = bound_client(model)
    baseline = one_shot_baseline(client, samples=1, max_output_tokens_per_call=10)

    async def run(request):
        client.proposer.generation_kwargs["max_tokens"] = 1000
        with pytest.raises(ComputeMismatchError, match="request output cap differs"):
            if preflight == "construction":
                one_shot_baseline(client, samples=1, max_output_tokens_per_call=10)
            else:
                await baseline.run(request)
        client.proposer.generation_kwargs["max_tokens"] = 10
        return (await client.generate(request),)

    with pytest.raises(ComputeMismatchError, match="request output cap differs"):
        await compare(manual_variant(client, run))
    assert model.calls == 1


async def test_each_dispatch_validates_a_fresh_copy_of_generation_kwargs():
    class MutatingModel(FakeModel):
        async def agenerate_with_metadata(self, prompt, **kwargs):
            response = await super().agenerate_with_metadata(prompt, **kwargs)
            client.proposer.generation_kwargs["max_tokens"] = 1000
            return response

    model = MutatingModel()
    client = bound_client(model)
    with pytest.raises(ComputeMismatchError, match="request output cap differs"):
        await compare(one_shot_baseline(client, samples=2, max_output_tokens_per_call=10))
    assert model.calls == 1


async def test_conflicting_aliases_reject_without_dispatch():
    model = FakeModel()
    client = bound_client(model)
    client.proposer.generation_kwargs["max_completion_tokens"] = 10
    with pytest.raises(ComputeMismatchError, match="exactly one"):
        await client.generate("question")
    assert model.calls == 0


@pytest.mark.parametrize("kwargs", [{}, {"max_tokens": 10}, {"provider_options": {"limit": 10}}])
async def test_missing_binding_never_verifies_even_when_usage_is_known(kwargs):
    client = InferenceClient(FakeModel(), generation_kwargs=kwargs)
    report = await compare(one_shot_baseline(client, samples=1, max_output_tokens_per_call=10))
    assert report.output_limit_assurance == "unknown"


@pytest.mark.parametrize("usage", [None, {}, {"completion_tokens": 1}])
async def test_unsupported_response_usage_shape_is_unknown(usage):
    class UnsupportedUsageModel(FakeModel):
        async def agenerate_with_metadata(self, prompt, **kwargs):
            response = await super().agenerate_with_metadata(prompt, **kwargs)
            response.usage = usage
            return response

    client = bound_client(UnsupportedUsageModel())
    report = await compare(one_shot_baseline(client, samples=1, max_output_tokens_per_call=10))
    assert report.output_limit_assurance == "unknown"
    assert report.to_dict()["compute"]["output_tokens"]["baseline"] is None


async def test_explicit_zero_response_usage_can_verify():
    client = bound_client(FakeModel(0))
    report = await compare(one_shot_baseline(client, samples=1, max_output_tokens_per_call=10))
    assert report.output_limit_assurance == "verified"
    assert report.to_dict()["compute"]["output_tokens"]["baseline"] == 0


async def test_missing_response_usage_cannot_be_spoofed_by_returned_candidate_metadata():
    client = bound_client(FakeModel(None))

    async def run(request):
        await client.generate(replace(request, metadata={"output_tokens": 0}))
        return (favorable_result(),)

    report = await compare(manual_variant(client, run))
    assert report.output_limit_assurance == "unknown"
    assert report.to_dict()["compute"]["output_tokens"]["baseline"] is None


async def test_two_excessive_calls_cannot_hide_under_small_reported_total():
    model = FakeModel(11)
    client = bound_client(model)

    async def run(request):
        for _ in range(2):
            try:
                await client.generate(request)
            except ComputeMismatchError:
                pass
        return (favorable_result(calls=2),)

    with pytest.raises(ComputeMismatchError, match="observed dispatch output tokens exceed"):
        await compare(manual_variant(client, run, calls=2))
    assert model.calls == 2


async def test_caught_extra_call_rejection_cannot_be_erased_by_favorable_result():
    model = FakeModel()
    client = bound_client(model)

    async def run(request):
        result = await client.generate(request)
        with pytest.raises(ComputeMismatchError, match="model-call ceiling"):
            await client.generate(request)
        return (result,)

    with pytest.raises(ComputeMismatchError, match="model-call ceiling"):
        await compare(manual_variant(client, run))
    assert model.calls == 1


async def test_multiple_clients_share_scope_and_account_for_model_judge_output():
    model = FakeModel()
    generator, judge_client = bound_client(model), bound_client(model)

    async def judge(request, outputs):
        await judge_client.generate("judge")
        return [1.0 for _ in outputs]

    variant = single_result_variant(
        "judged",
        lambda request: generator.best_of_n(
            request,
            n=2,
            verifiers=(VerifierSpec("local", lambda _: Verification("local", 1.0, True)),),
            judge=judge,
        ),
        client=generator,
        generated_calls=4,
        max_output_tokens_per_call=10,
    )
    report = await compare(variant)
    assert report.output_limit_assurance == "verified"
    assert report.to_dict()["compute"]["output_tokens"]["baseline"] == 12
    assert report.baseline_spend.calls == 4


@pytest.mark.parametrize("reported_calls", [1, 2])
@pytest.mark.parametrize("synchronous", [False, True])
async def test_instrumented_verifier_requires_honest_call_count_and_includes_output(
    reported_calls, synchronous
):
    model = FakeModel()
    generator, verifier_client = bound_client(model), bound_client(model)

    async def verify(candidate):
        await verifier_client.generate("verify")
        return Verification("model-verifier", score=1.0, passed=True)

    def sync_verify(candidate):
        return run_sync(verify(candidate))

    async def verify_from_worker(candidate):
        return await asyncio.to_thread(sync_verify, candidate)

    async def run(request):
        result = await generator.best_of_n(
            request,
            n=1,
            verifiers=(
                VerifierSpec("model-verifier", verify_from_worker if synchronous else verify),
            ),
        )
        return (replace(result, spend=replace(result.spend, calls=reported_calls)),)

    report = await compare(manual_variant(generator, run, calls=2))
    assert report.output_limit_assurance == ("verified" if reported_calls == 2 else "unknown")
    assert report.to_dict()["compute"]["output_tokens"]["baseline"] == (
        6 if reported_calls == 2 else None
    )


async def test_another_unbound_client_marks_scope_unknown_without_hiding_bound_violations():
    model = FakeModel(11)
    client, unbound = bound_client(model), InferenceClient(model)

    async def run(request):
        for caller in (unbound, client):
            try:
                await caller.generate(request)
            except ComputeMismatchError:
                pass
        return (favorable_result(calls=2),)

    with pytest.raises(ComputeMismatchError, match="observed dispatch output tokens exceed"):
        await compare(manual_variant(client, run, calls=2))
    assert model.calls == 2


async def test_concurrent_evaluations_have_separate_caps_counts_and_usage():
    class YieldingModel(FakeModel):
        async def agenerate_with_metadata(self, prompt, **kwargs):
            await asyncio.sleep(0)
            self.output_tokens = kwargs["max_tokens"]
            return await super().agenerate_with_metadata(prompt, **kwargs)

    model = YieldingModel()

    async def evaluate(cap):
        client = bound_client(model, cap)
        return await compare(one_shot_baseline(client, samples=2, max_output_tokens_per_call=cap))

    reports = await asyncio.gather(evaluate(10), evaluate(20))
    assert [report.output_limit_assurance for report in reports] == ["verified", "verified"]
    assert [report.baseline_spend.output_tokens for report in reports] == [20, 40]


@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("caught", [False, True])
async def test_provider_exception_leaves_incomplete_evidence_and_resets_scope(synchronous, caught):
    class FailingModel:
        name = "shared"
        calls = 0

        def response(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("provider failed")
            return ModelResponse("answer", model=self.name, usage=TokenUsage(completion_tokens=3))

    model = FailingModel()
    if synchronous:
        model.generate_with_metadata = model.response
    else:

        async def generate(prompt, **kwargs):
            return model.response(prompt, **kwargs)

        model.agenerate_with_metadata = generate
    client = bound_client(model)

    async def run(request):
        try:
            return (await client.generate(request),)
        except RuntimeError:
            if not caught:
                raise
            return (favorable_result(),)

    if caught:
        report = await compare(manual_variant(client, run))
        assert report.output_limit_assurance == "unknown"
        assert report.to_dict()["compute"]["output_tokens"]["baseline"] is None
    else:
        with pytest.raises(RuntimeError, match="provider failed"):
            await compare(manual_variant(client, run))
    clean = await compare(one_shot_baseline(client, samples=1, max_output_tokens_per_call=10))
    assert clean.output_limit_assurance == "verified"


async def test_delayed_sync_worker_cannot_settle_into_next_variant_scope():
    entered, release, finished = Event(), Event(), Event()

    class DelayedModel:
        name = "shared"
        calls = 0

        def generate_with_metadata(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 1:
                entered.set()
                assert release.wait(2)
                finished.set()
                return ModelResponse(
                    "late", model=self.name, usage=TokenUsage(completion_tokens=99)
                )
            release.set()
            assert finished.wait(2)
            return ModelResponse("answer", model=self.name, usage=TokenUsage(completion_tokens=3))

    model = DelayedModel()
    client = bound_client(model)

    async def run(request):
        task = asyncio.create_task(client.generate(request))
        try:
            if model.calls == 0:
                assert await asyncio.to_thread(entered.wait, 2)
                await asyncio.wait_for(asyncio.shield(task), timeout=0.01)
            return (await task,)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return (favorable_result(),)

    try:
        report = await compare(manual_variant(client, run))
        assert report.output_limit_assurance == "unknown"
        assert report.to_dict()["compute"]["output_tokens"]["baseline"] is None
        assert report.to_dict()["compute"]["output_tokens"]["strategy"] == 3
        clean = await compare(one_shot_baseline(client, samples=1, max_output_tokens_per_call=10))
        assert clean.output_limit_assurance == "verified"
    finally:
        release.set()


async def test_evaluation_cancellation_resets_collector_before_next_run():
    entered = asyncio.Event()

    class CancellableModel(FakeModel):
        async def agenerate_with_metadata(self, prompt, **kwargs):
            if not entered.is_set():
                entered.set()
                await asyncio.Event().wait()
            return await super().agenerate_with_metadata(prompt, **kwargs)

    client = bound_client(CancellableModel())
    variant = one_shot_baseline(client, samples=1, max_output_tokens_per_call=10)
    task = asyncio.create_task(compare(variant))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    report = await compare(variant)
    assert report.output_limit_assurance == "verified"


async def test_unobserved_foreign_model_call_is_unknown_when_honestly_counted():
    model = FakeModel()
    client = bound_client(model)

    async def run(request):
        result = await client.generate(request)
        await model.agenerate_with_metadata("foreign")
        return (replace(result, spend=replace(result.spend, calls=2)),)

    report = await compare(manual_variant(client, run, calls=2))
    assert report.output_limit_assurance == "unknown"
    assert report.to_dict()["compute"]["output_tokens"]["baseline"] is None


async def test_model_backed_judge_with_unobserved_calls_cannot_verify():
    client = bound_client(FakeModel())

    async def run(request):
        return await client.best_of_n(
            request,
            n=2,
            verifiers=(VerifierSpec("local", lambda _: Verification("local", 1.0, True)),),
            judge=lambda request, outputs: [1.0 for _ in outputs],
        )

    variant = single_result_variant(
        "unobserved-judge", run, client=client, generated_calls=4, max_output_tokens_per_call=10
    )
    report = await compare(variant)
    assert report.output_limit_assurance == "unknown"


@pytest.mark.parametrize("parameter", ["max_output_tokens", "options", ""])
def test_binding_rejects_unsupported_provider_parameter_shapes(parameter):
    with pytest.raises(ValueError, match="max_tokens or max_completion_tokens"):
        OutputLimitBinding(parameter, 10)


@pytest.mark.parametrize("maximum", [0, -1, 1.5, True, float("inf")])
def test_binding_requires_a_positive_integer_cap(maximum):
    with pytest.raises(ValueError, match="positive integer"):
        OutputLimitBinding("max_tokens", maximum)


async def test_from_model_config_propagates_request_binding():
    client = InferenceClient.from_model_config(
        {"type": "dummy", "args": {"canned_response": "answer"}},
        generation_kwargs={"max_tokens": 1000},
        output_limit=OutputLimitBinding("max_tokens", 10),
    )
    with pytest.raises(ComputeMismatchError, match="request output cap differs"):
        await client.generate("question")
