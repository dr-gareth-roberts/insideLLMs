import json
import subprocess
import sys
from pathlib import Path

import pytest

from insideLLMs.analysis.evaluation import EvaluationResult, Evaluator, ExactMatchEvaluator
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
    InferenceClient,
    InferenceResult,
    Spend,
    Verification,
    VerifierSpec,
)


def _result(answer: str, *, calls: int, tokens: int, elapsed: float) -> InferenceResult:
    return InferenceResult(
        answer=answer,
        spend=Spend(
            calls=calls,
            input_tokens=tokens,
            output_tokens=1,
            elapsed_seconds=elapsed,
        ),
        provenance={"model": "test-model"},
    )


def _variant(
    name: str,
    run,
    *,
    calls: int,
    max_output_tokens: int = 8,
    executor_id: int = 1,
    cost_available: bool = False,
) -> MatchedComputeVariant:
    return MatchedComputeVariant(
        name,
        run,
        compute=ComputeProfile(
            executor_id=executor_id,
            generated_calls=calls,
            max_output_tokens_per_call=max_output_tokens,
            cost_available=cost_available,
        ),
    )


def test_matched_compute_is_exported_from_analysis_package() -> None:
    from insideLLMs.analysis import run_matched_compute as exported

    assert exported is run_matched_compute


def test_matched_compute_wildcard_export_is_limited_to_public_api() -> None:
    from insideLLMs.analysis import matched_compute

    assert "InferenceClient" not in matched_compute.__all__
    assert "Counter" not in matched_compute.__all__
    assert "run_matched_compute" in matched_compute.__all__


def test_matched_compute_example_emits_auditable_json() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "examples.matched_compute_evaluation"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["matching"] == {
        "calls": True,
        "models": True,
        "max_output_tokens": True,
        "order_balanced": True,
    }
    assert payload["compute"]["calls"] == {"baseline": 4, "strategy": 4}
    assert payload["limitations"]["evidence_scope"] == "offline-smoke-only"


async def test_matched_compute_reports_quality_oracle_spend_and_regressions() -> None:
    example = DatasetExample(
        id="math-1",
        input_text="2 + 2 = ?",
        expected_output="4",
        category="math",
    )

    async def baseline(_request):
        return (
            _result("4", calls=1, tokens=4, elapsed=0.1),
            _result("5", calls=1, tokens=4, elapsed=0.2),
        )

    async def strategy(_request):
        return (_result("4", calls=2, tokens=8, elapsed=0.25),)

    report = await run_matched_compute(
        [example],
        baseline=_variant("one-shot", baseline, calls=2),
        strategy=_variant("best-of-2", strategy, calls=2),
        evaluator=ExactMatchEvaluator(),
        trials=2,
    )

    assert report.mean_baseline_score == 0.5
    assert report.mean_strategy_score == 1.0
    assert report.mean_score_delta == 0.5
    assert report.regression_count == 0
    assert report.regressions_by_subset == {}
    assert report.baseline_pass_at_n == 1.0
    assert report.strategy_pass_rate == 1.0
    assert report.mean_baseline_oracle_score == 1.0
    assert report.baseline_spend.calls == 4
    assert report.strategy_spend.calls == 4
    assert report.baseline_spend.input_tokens == 16
    assert report.strategy_spend.input_tokens == 16
    assert report.calls_matched
    assert report.models_matched
    assert report.output_tokens_matched
    assert report.order_balanced
    assert report.input_token_ratio == 1.0
    assert report.output_token_ratio == 0.5
    assert report.baseline_cost is None
    assert report.strategy_cost is None
    assert not report.cost_comparable
    assert report.ttft_seconds is None
    assert len(report.cases) == 2
    payload = report.to_dict()
    assert payload["variants"] == {"baseline": "one-shot", "strategy": "best-of-2"}
    assert payload["quality"]["mean_score_delta"] == 0.5
    assert payload["compute"]["calls"] == {"baseline": 4, "strategy": 4}
    assert payload["compute"]["cost"] == {
        "baseline": None,
        "strategy": None,
        "comparable": False,
    }
    assert payload["provider_seconds"]["comparable"] is False
    assert payload["limitations"]["ttft_seconds"] is None
    assert payload["cases"][0]["baseline_spend"]["input_tokens"] == 8
    assert payload["cases"][0]["strategy_spend"]["input_tokens"] == 8
    assert payload["cases"][0]["baseline_passed"] == [True, False]
    assert payload["cases"][0]["wall_seconds"]["baseline"] > 0
    json.dumps(payload, allow_nan=False)


async def test_matched_compute_rejects_different_call_budgets() -> None:
    example = DatasetExample(id="q1", input_text="answer", expected_output="answer")

    async def baseline(_request):
        return (_result("answer", calls=1, tokens=1, elapsed=0.1),)

    async def strategy(_request):
        return (_result("answer", calls=2, tokens=1, elapsed=0.1),)

    with pytest.raises(ComputeMismatchError, match="declared"):
        await run_matched_compute(
            [example],
            baseline=_variant("one-shot", baseline, calls=1),
            strategy=_variant("strategy", strategy, calls=1),
            evaluator=ExactMatchEvaluator(),
            trials=1,
        )


async def test_matched_compute_rejects_undeclared_judge_calls() -> None:
    async def baseline(_request):
        return (
            _result("answer", calls=1, tokens=1, elapsed=0.1),
            _result("answer", calls=1, tokens=1, elapsed=0.1),
        )

    async def strategy(_request):
        # Two generations plus two undeclared model-backed judge calls.
        return (_result("answer", calls=4, tokens=1, elapsed=0.1),)

    with pytest.raises(ComputeMismatchError, match="declared"):
        await run_matched_compute(
            [DatasetExample(id="q", input_text="answer", expected_output="answer")],
            baseline=_variant("baseline", baseline, calls=2),
            strategy=_variant("strategy", strategy, calls=2),
            evaluator=ExactMatchEvaluator(),
            trials=1,
        )


async def test_matched_compute_rejects_different_executors() -> None:
    async def variant(_request):
        return (_result("answer", calls=1, tokens=1, elapsed=0.1),)

    with pytest.raises(ComputeMismatchError, match="model executor"):
        await run_matched_compute(
            [DatasetExample(id="q", input_text="answer", expected_output="answer")],
            baseline=_variant("baseline", variant, calls=1, executor_id=1),
            strategy=_variant("strategy", variant, calls=1, executor_id=2),
            evaluator=ExactMatchEvaluator(),
            trials=1,
        )


async def test_matched_compute_rejects_different_output_token_budgets() -> None:
    async def variant(_request):
        return (_result("answer", calls=2, tokens=1, elapsed=0.1),)

    with pytest.raises(ComputeMismatchError, match="output-token"):
        await run_matched_compute(
            [DatasetExample(id="q", input_text="answer", expected_output="answer")],
            baseline=_variant("baseline", variant, calls=2, max_output_tokens=8),
            strategy=_variant("strategy", variant, calls=2, max_output_tokens=16),
            evaluator=ExactMatchEvaluator(),
            trials=1,
        )


async def test_matched_compute_rejects_missing_model_provenance() -> None:
    async def variant(_request):
        return (InferenceResult(answer="answer", spend=Spend(calls=1)),)

    with pytest.raises(ComputeMismatchError, match="model provenance"):
        await run_matched_compute(
            [DatasetExample(id="q", input_text="answer", expected_output="answer")],
            baseline=_variant("baseline", variant, calls=1),
            strategy=_variant("strategy", variant, calls=1),
            evaluator=ExactMatchEvaluator(),
            trials=1,
        )


async def test_matched_compute_rejects_non_finite_scores() -> None:
    class NanEvaluator(Evaluator):
        def evaluate(self, prediction: str, reference: str, **kwargs: object) -> EvaluationResult:
            return EvaluationResult(score=float("nan"), passed=False, metric_name="nan")

    async def variant(_request):
        return (_result("answer", calls=1, tokens=1, elapsed=0.1),)

    with pytest.raises(ValueError, match="non-finite score"):
        await run_matched_compute(
            [DatasetExample(id="q", input_text="answer", expected_output="answer")],
            baseline=_variant("baseline", variant, calls=1),
            strategy=_variant("strategy", variant, calls=1),
            evaluator=NanEvaluator(),
            trials=1,
        )


async def test_matched_compute_preflights_all_references_before_model_calls() -> None:
    calls = 0

    async def variant(_request):
        nonlocal calls
        calls += 1
        return (_result("answer", calls=1, tokens=1, elapsed=0.1),)

    with pytest.raises(ValueError, match="missing.*no expected output"):
        await run_matched_compute(
            [
                DatasetExample(id="ready", input_text="answer", expected_output="answer"),
                DatasetExample(id="missing", input_text="unknown", expected_output=None),
            ],
            baseline=_variant("one-shot", variant, calls=1),
            strategy=_variant("strategy", variant, calls=1),
            evaluator=ExactMatchEvaluator(),
            trials=1,
        )

    assert calls == 0


async def test_matched_compute_alternates_variant_order_and_reports_throughput() -> None:
    order: list[str] = []

    async def baseline(_request):
        order.append("baseline")
        return (_result("answer", calls=1, tokens=1, elapsed=0.1),)

    async def strategy(_request):
        order.append("strategy")
        return (_result("answer", calls=1, tokens=1, elapsed=0.1),)

    report = await run_matched_compute(
        [DatasetExample(id="q", input_text="answer", expected_output="answer")],
        baseline=_variant("baseline", baseline, calls=1),
        strategy=_variant("strategy", strategy, calls=1),
        evaluator=ExactMatchEvaluator(),
        trials=2,
    )

    assert order == ["baseline", "strategy", "strategy", "baseline"]
    assert report.order_balanced
    assert report.baseline_wall_seconds > 0
    assert report.strategy_wall_seconds > 0
    assert report.baseline_wall_p50_seconds > 0
    assert report.baseline_wall_p95_seconds >= report.baseline_wall_p50_seconds
    assert report.strategy_wall_p50_seconds > 0
    assert report.strategy_wall_p95_seconds >= report.strategy_wall_p50_seconds
    assert report.baseline_throughput_calls_per_second > 0
    assert report.strategy_throughput_calls_per_second > 0
    payload = report.to_dict()
    assert payload["throughput_calls_per_second"]["baseline"] > 0


async def test_matched_compute_flags_unbalanced_order_for_odd_trials() -> None:
    async def variant(_request):
        return (_result("answer", calls=1, tokens=1, elapsed=0.1),)

    report = await run_matched_compute(
        [DatasetExample(id="q", input_text="answer", expected_output="answer")],
        baseline=_variant("baseline", variant, calls=1),
        strategy=_variant("strategy", variant, calls=1),
        evaluator=ExactMatchEvaluator(),
        trials=3,
    )

    assert not report.order_balanced
    assert report.to_dict()["limitations"]["order_balanced"] is False


async def test_matched_compute_runs_canonical_model_backed_variants() -> None:
    class SequenceModel:
        name = "sequence-model"

        def __init__(self) -> None:
            self.outputs = iter(("4", "5", "4", "5"))

        async def agenerate(self, prompt: str, **kwargs: object) -> str:
            return next(self.outputs)

    client = InferenceClient(SequenceModel())
    verifier = VerifierSpec(
        "exact",
        lambda candidate: Verification(
            "exact",
            score=float(candidate.output == "4"),
            passed=True,
        ),
    )
    report = await run_matched_compute(
        [DatasetExample(id="math", input_text="2 + 2", expected_output="4")],
        baseline=one_shot_baseline(client, samples=2, max_output_tokens_per_call=16),
        strategy=single_result_variant(
            "best-of-2",
            lambda request: client.best_of_n(request, n=2, verifiers=(verifier,)),
            client=client,
            generated_calls=2,
            max_output_tokens_per_call=16,
        ),
        evaluator=ExactMatchEvaluator(),
        trials=1,
    )

    assert report.mean_baseline_score == 0.5
    assert report.mean_strategy_score == 1.0
    assert report.baseline_spend.calls == report.strategy_spend.calls == 2
    assert report.output_tokens_matched
