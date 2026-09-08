from __future__ import annotations

import pytest

from insideLLMs.analysis.evaluation import ExactMatchEvaluator
from insideLLMs.analysis.matched_compute import (
    ComputeMismatchError,
    ComputeProfile,
    MatchedComputeReport,
    MatchedComputeVariant,
    one_shot_baseline,
    run_matched_compute,
)
from insideLLMs.benchmark_datasets import DatasetExample
from insideLLMs.inference import InferenceClient, InferenceResult, OutputLimitBinding, Spend
from insideLLMs.types import ModelResponse, TokenUsage


async def test_equal_profiles_do_not_verify_different_client_request_caps() -> None:
    """A profile declaration cannot certify the cap actually sent to a shared model."""

    class CapAwareModel:
        name = "cap-aware"
        calls = 0

        async def agenerate_with_metadata(self, prompt: str, **kwargs: object) -> ModelResponse:
            self.calls += 1
            output_tokens = int(kwargs["max_tokens"])
            return ModelResponse(
                content="answer",
                model=self.name,
                usage=TokenUsage(
                    prompt_tokens=1,
                    completion_tokens=output_tokens,
                    total_tokens=output_tokens + 1,
                ),
            )

    model = CapAwareModel()
    baseline_client = InferenceClient(
        model,
        generation_kwargs={"max_tokens": 10},
        output_limit=OutputLimitBinding("max_tokens", 10),
    )
    strategy_client = InferenceClient(
        model,
        generation_kwargs={"max_tokens": 1000},
        output_limit=OutputLimitBinding("max_tokens", 1000),
    )
    with pytest.raises(ComputeMismatchError, match="bound output cap differs from declared"):
        await run_matched_compute(
            [DatasetExample(id="q", input_text="question", expected_output="answer")],
            baseline=one_shot_baseline(
                baseline_client,
                samples=1,
                max_output_tokens_per_call=10,
            ),
            strategy=one_shot_baseline(
                strategy_client,
                samples=1,
                max_output_tokens_per_call=10,
            ),
            evaluator=ExactMatchEvaluator(),
            trials=1,
        )
    assert model.calls == 0


async def test_observed_output_overflow_is_rejected_before_reporting() -> None:
    async def overflow(_request):
        return (
            InferenceResult(
                answer="answer",
                spend=Spend(calls=1, output_tokens=11),
                provenance={"model": "shared"},
            ),
        )

    variant = MatchedComputeVariant(
        "overflow",
        overflow,
        ComputeProfile(executor_id=1, generated_calls=1, max_output_tokens_per_call=10),
    )
    with pytest.raises(
        ComputeMismatchError,
        match="observed output tokens exceed declared call caps",
    ):
        await run_matched_compute(
            [DatasetExample(id="q", input_text="question", expected_output="answer")],
            baseline=variant,
            strategy=variant,
            evaluator=ExactMatchEvaluator(),
            trials=1,
        )


@pytest.mark.parametrize(
    ("spend", "message"),
    [
        (Spend(calls=-1), "non-negative"),
        (Spend(calls=0, output_tokens=1), "zero calls"),
        (Spend(calls=1, output_tokens=float("inf")), "finite"),
    ],
)
async def test_invalid_individual_spend_is_rejected_before_aggregation(
    spend: Spend, message: str
) -> None:
    async def invalid(_request):
        # The second item can cancel a bad first item if validation happens only
        # after aggregation.
        return (
            InferenceResult(answer="answer", spend=spend, provenance={"model": "shared"}),
            InferenceResult(
                answer="answer",
                spend=Spend(calls=1, output_tokens=0),
                provenance={"model": "shared"},
            ),
        )

    variant = MatchedComputeVariant(
        "invalid",
        invalid,
        ComputeProfile(executor_id=1, generated_calls=2, max_output_tokens_per_call=10),
    )
    with pytest.raises(ComputeMismatchError, match=message):
        await run_matched_compute(
            [DatasetExample(id="q", input_text="question", expected_output="answer")],
            baseline=variant,
            strategy=variant,
            evaluator=ExactMatchEvaluator(),
            trials=1,
        )


@pytest.mark.parametrize("field", ["generated_calls", "max_output_tokens_per_call"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_compute_profile_rejects_nonfinite_or_nonpositive_limits(field: str, value: float) -> None:
    values = {"executor_id": 1, "generated_calls": 1, "max_output_tokens_per_call": 10}
    values[field] = value
    with pytest.raises(ValueError, match="finite positive"):
        ComputeProfile(**values)


async def test_serialization_distinguishes_genuine_zero_from_missing_usage() -> None:
    class UsageModel:
        name = "usage-model"

        def __init__(self, usage: TokenUsage | None) -> None:
            self.usage = usage

        async def agenerate_with_metadata(self, prompt: str, **kwargs: object) -> ModelResponse:
            return ModelResponse(content="answer", model=self.name, usage=self.usage)

    zero_model = UsageModel(TokenUsage(prompt_tokens=1, completion_tokens=0, total_tokens=1))
    missing_model = UsageModel(None)

    async def artifact_for(model: UsageModel) -> dict[str, object]:
        client = InferenceClient(model)
        variant = one_shot_baseline(client, samples=1, max_output_tokens_per_call=10)
        report = await run_matched_compute(
            [DatasetExample(id="q", input_text="question", expected_output="answer")],
            baseline=variant,
            strategy=variant,
            evaluator=ExactMatchEvaluator(),
            trials=1,
        )
        return report.to_dict()

    zero = await artifact_for(zero_model)
    missing = await artifact_for(missing_model)

    assert zero["compute"]["output_tokens"]["baseline"] == 0
    assert zero["cases"][0]["baseline_spend"]["output_tokens"] == 0
    assert missing["compute"]["output_tokens"]["baseline"] is None
    assert missing["cases"][0]["baseline_spend"]["output_tokens"] is None


async def test_mixed_canonical_candidate_usage_makes_total_and_ratio_unknown() -> None:
    class MixedUsageModel:
        name = "mixed-usage"

        def __init__(self) -> None:
            self.calls = 0

        async def agenerate_with_metadata(self, prompt: str, **kwargs: object) -> ModelResponse:
            usage = (
                TokenUsage(prompt_tokens=1, completion_tokens=3, total_tokens=4)
                if self.calls % 2 == 0
                else None
            )
            self.calls += 1
            return ModelResponse(content="answer", model=self.name, usage=usage)

    client = InferenceClient(MixedUsageModel())
    variant = one_shot_baseline(client, samples=2, max_output_tokens_per_call=10)
    report = await run_matched_compute(
        [DatasetExample(id="q", input_text="question", expected_output="answer")],
        baseline=variant,
        strategy=variant,
        evaluator=ExactMatchEvaluator(),
        trials=1,
    )
    artifact = report.to_dict()

    assert report.output_token_ratio is None
    assert artifact["compute"]["output_tokens"]["baseline"] is None
    assert artifact["compute"]["output_tokens"]["strategy_to_baseline_ratio"] is None
    assert artifact["cases"][0]["baseline_spend"]["output_tokens"] is None


async def test_identical_declared_limits_keep_unequal_actual_usage_separate() -> None:
    class UsageModel:
        name = "usage-model"

        async def agenerate_with_metadata(self, prompt: str, **kwargs: object) -> ModelResponse:
            output_tokens = int(kwargs["reported_output_tokens"])
            return ModelResponse(
                content="answer",
                model=self.name,
                usage=TokenUsage(
                    prompt_tokens=1,
                    completion_tokens=output_tokens,
                    total_tokens=output_tokens + 1,
                ),
            )

    model = UsageModel()
    baseline = one_shot_baseline(
        InferenceClient(model, generation_kwargs={"reported_output_tokens": 3}),
        samples=1,
        max_output_tokens_per_call=10,
    )
    strategy = one_shot_baseline(
        InferenceClient(model, generation_kwargs={"reported_output_tokens": 7}),
        samples=1,
        max_output_tokens_per_call=10,
    )
    report = await run_matched_compute(
        [DatasetExample(id="q", input_text="question", expected_output="answer")],
        baseline=baseline,
        strategy=strategy,
        evaluator=ExactMatchEvaluator(),
        trials=1,
    )
    artifact = report.to_dict()

    assert report.declared_limits_match is True
    assert artifact["matching"]["declared_limits_match"] is True
    assert artifact["matching"]["output_limit_assurance"] == "unknown"
    assert artifact["matching"]["max_output_tokens"] is False
    assert artifact["compute"]["output_tokens"]["baseline"] == 3
    assert artifact["compute"]["output_tokens"]["strategy"] == 7


async def test_custom_spend_and_provenance_cannot_self_certify_output_caps() -> None:
    async def custom(_request):
        return (
            InferenceResult(
                answer="answer",
                spend=Spend(calls=1, output_tokens=1),
                provenance={"model": "shared", "output_limit_verified": True},
            ),
        )

    variant = MatchedComputeVariant(
        "custom",
        custom,
        ComputeProfile(executor_id=1, generated_calls=1, max_output_tokens_per_call=10),
    )
    report = await run_matched_compute(
        [DatasetExample(id="q", input_text="question", expected_output="answer")],
        baseline=variant,
        strategy=variant,
        evaluator=ExactMatchEvaluator(),
        trials=1,
    )

    assert report.output_limit_assurance == "unknown"
    assert report.output_tokens_matched is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("calls", 0.5),
        ("input_tokens", 0.5),
        ("output_tokens", 0.5),
        ("evaluations", 0.5),
    ],
)
async def test_spend_counters_reject_fractional_values(field: str, value: float) -> None:
    values = {"calls": 1, "input_tokens": 0, "output_tokens": 0, "evaluations": 0}
    values[field] = value

    async def invalid(_request):
        return (
            InferenceResult(
                answer="answer",
                spend=Spend(**values),
                provenance={"model": "shared"},
            ),
        )

    variant = MatchedComputeVariant(
        "invalid",
        invalid,
        ComputeProfile(executor_id=1, generated_calls=1, max_output_tokens_per_call=10),
    )
    with pytest.raises(ComputeMismatchError, match=f"{field} must be an integer"):
        await run_matched_compute(
            [DatasetExample(id="q", input_text="question", expected_output="answer")],
            baseline=variant,
            strategy=variant,
            evaluator=ExactMatchEvaluator(),
            trials=1,
        )


@pytest.mark.parametrize("field", ["generated_calls", "max_output_tokens_per_call"])
def test_compute_profile_rejects_fractional_limits(field: str) -> None:
    values = {"executor_id": 1, "generated_calls": 1, "max_output_tokens_per_call": 10}
    values[field] = 1.5
    with pytest.raises(ValueError, match="positive integer"):
        ComputeProfile(**values)


def test_report_ninth_positional_argument_remains_regressions_mapping() -> None:
    profile = ComputeProfile(executor_id=1, generated_calls=1, max_output_tokens_per_call=10)
    report = MatchedComputeReport(
        "baseline",
        "strategy",
        profile,
        profile,
        (),
        Spend(),
        Spend(),
        True,
        {"legacy": 2},
    )

    assert report.regressions_by_subset == {"legacy": 2}
