"""Matched-compute evaluation for model-backed inference strategies."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

from insideLLMs.inference.client import InferenceClient
from insideLLMs.inference.schemas import InferenceRequest, InferenceResult, Spend

from .evaluation import Evaluator
from .statistics import calculate_mean, calculate_percentile

__all__ = [
    "ComputeMismatchError",
    "ComputeProfile",
    "EvaluationExample",
    "MatchedComputeCase",
    "MatchedComputeReport",
    "MatchedComputeVariant",
    "one_shot_baseline",
    "run_matched_compute",
    "single_result_variant",
]


class EvaluationExample(Protocol):
    """Dataset fields required by the matched-compute runner."""

    id: str
    input_text: str
    expected_output: str | None
    category: str | None


VariantRunner = Callable[[InferenceRequest], Awaitable[Sequence[InferenceResult]]]
SingleResultRunner = Callable[[InferenceRequest], Awaitable[InferenceResult]]


class ComputeMismatchError(ValueError):
    """Raised when paired variants did not consume comparable model compute."""


@dataclass(frozen=True)
class ComputeProfile:
    """Declared generation context used to establish compute equivalence.

    ``executor`` retains the model-backed executor (normally the client) so two
    variants cannot silently run different models; identity is compared through
    the executor's underlying ``model`` when it exposes one, so two clients
    wrapping the same model still match. ``executor_id`` remains as a fallback
    for callers that construct profiles directly — note that a bare ``id()``
    can alias after garbage collection, so prefer passing ``executor``.
    ``generated_calls`` is the per-example CEILING of expensive model calls the
    variant may make, including any model-backed judge or verifier calls;
    observing more than declared fails, observing fewer (e.g. a judge skipped
    when only one candidate survives hard verification) is allowed and visible
    in the per-case observed calls. ``cost_available`` records whether the
    provider actually reports cost, so a zero total is never mistaken for a
    genuine zero.
    """

    executor_id: int
    generated_calls: int
    max_output_tokens_per_call: int
    cost_available: bool = False
    executor: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.generated_calls < 1:
            raise ValueError("generated_calls must be positive")
        if self.max_output_tokens_per_call < 1:
            raise ValueError("max_output_tokens_per_call must be positive")

    @property
    def max_output_tokens(self) -> int:
        return self.generated_calls * self.max_output_tokens_per_call


@dataclass(frozen=True)
class MatchedComputeVariant:
    """One named execution path participating in a comparison."""

    name: str
    run: VariantRunner
    compute: ComputeProfile


def one_shot_baseline(
    client: InferenceClient,
    *,
    samples: int,
    max_output_tokens_per_call: int,
    cost_available: bool = False,
) -> MatchedComputeVariant:
    """Create an N-sample one-shot baseline through the canonical client."""

    if samples < 1:
        raise ValueError("samples must be positive")

    async def run(request: InferenceRequest) -> Sequence[InferenceResult]:
        return await client.generate_many(request, n=samples)

    return MatchedComputeVariant(
        name=f"one-shot-{samples}",
        run=run,
        compute=ComputeProfile(
            executor_id=id(client),
            generated_calls=samples,
            max_output_tokens_per_call=max_output_tokens_per_call,
            cost_available=cost_available,
            executor=client,
        ),
    )


def single_result_variant(
    name: str,
    run: SingleResultRunner,
    *,
    client: InferenceClient,
    generated_calls: int,
    max_output_tokens_per_call: int,
    cost_available: bool = False,
) -> MatchedComputeVariant:
    """Adapt one strategy result to the matched-compute batch contract."""

    async def run_batch(request: InferenceRequest) -> Sequence[InferenceResult]:
        return (await run(request),)

    return MatchedComputeVariant(
        name=name,
        run=run_batch,
        compute=ComputeProfile(
            executor_id=id(client),
            generated_calls=generated_calls,
            max_output_tokens_per_call=max_output_tokens_per_call,
            cost_available=cost_available,
            executor=client,
        ),
    )


@dataclass(frozen=True)
class MatchedComputeCase:
    """One paired example/trial comparison."""

    example_id: str
    trial: int
    subset: str
    baseline_scores: tuple[float, ...]
    strategy_scores: tuple[float, ...]
    baseline_passed: tuple[bool, ...]
    strategy_passed: tuple[bool, ...]
    baseline_spend: Spend
    strategy_spend: Spend
    baseline_models: tuple[str, ...]
    strategy_models: tuple[str, ...]
    baseline_wall_seconds: float
    strategy_wall_seconds: float

    @property
    def baseline_score(self) -> float:
        return sum(self.baseline_scores) / len(self.baseline_scores)

    @property
    def strategy_score(self) -> float:
        return sum(self.strategy_scores) / len(self.strategy_scores)

    @property
    def score_delta(self) -> float:
        return self.strategy_score - self.baseline_score

    @property
    def regression(self) -> bool:
        return self.score_delta < 0

    @property
    def baseline_oracle_score(self) -> float:
        return max(self.baseline_scores)

    @property
    def baseline_pass_at_n(self) -> bool:
        return any(self.baseline_passed)

    @property
    def strategy_passed_any(self) -> bool:
        return any(self.strategy_passed)

    @property
    def calls_matched(self) -> bool:
        """True when both variants actually spent the same number of calls.

        This is an observation, not the compute-equivalence gate: a variant may
        legitimately spend fewer calls than its declared ceiling (for example a
        judge skipped when only one candidate survives hard verification), and
        the run is still valid. The gate is ``_check_case``, which rejects only
        observed calls ABOVE the declared ceiling.
        """

        return self.baseline_spend.calls == self.strategy_spend.calls

    @property
    def models_matched(self) -> bool:
        return self.baseline_models == self.strategy_models


@dataclass(frozen=True)
class MatchedComputeReport:
    """Aggregate evidence across repeated paired trials."""

    baseline_name: str
    strategy_name: str
    baseline_compute: ComputeProfile
    strategy_compute: ComputeProfile
    cases: tuple[MatchedComputeCase, ...]
    baseline_spend: Spend
    strategy_spend: Spend
    order_balanced: bool
    regressions_by_subset: dict[str, int] = field(default_factory=dict)

    @property
    def mean_baseline_score(self) -> float:
        return _mean(tuple(case.baseline_score for case in self.cases))

    @property
    def mean_strategy_score(self) -> float:
        return _mean(tuple(case.strategy_score for case in self.cases))

    @property
    def mean_score_delta(self) -> float:
        return self.mean_strategy_score - self.mean_baseline_score

    @property
    def regression_count(self) -> int:
        return sum(case.regression for case in self.cases)

    @property
    def baseline_pass_at_n(self) -> float:
        return _mean(tuple(float(case.baseline_pass_at_n) for case in self.cases))

    @property
    def strategy_pass_rate(self) -> float:
        return _mean(tuple(float(case.strategy_passed_any) for case in self.cases))

    @property
    def mean_baseline_oracle_score(self) -> float:
        return _mean(tuple(case.baseline_oracle_score for case in self.cases))

    @property
    def calls_matched(self) -> bool:
        """True when every case spent equal calls on both sides (see the case property).

        Unused budget on one side makes this False without invalidating the
        comparison; exceeding a declared ceiling raises during the run instead.
        """

        return all(case.calls_matched for case in self.cases)

    @property
    def models_matched(self) -> bool:
        return all(case.models_matched for case in self.cases)

    @property
    def output_tokens_matched(self) -> bool:
        return self.baseline_compute.max_output_tokens == self.strategy_compute.max_output_tokens

    def _wall_values(self, side: str) -> tuple[float, ...]:
        return tuple(getattr(case, f"{side}_wall_seconds") for case in self.cases)

    def _throughput(self, side: str) -> float | None:
        wall = sum(self._wall_values(side))
        spend: Spend = getattr(self, f"{side}_spend")
        return spend.calls / wall if wall else None

    @property
    def baseline_wall_seconds(self) -> float:
        return sum(self._wall_values("baseline"))

    @property
    def strategy_wall_seconds(self) -> float:
        return sum(self._wall_values("strategy"))

    @property
    def baseline_wall_p50_seconds(self) -> float:
        return _percentile(self._wall_values("baseline"), 0.5)

    @property
    def baseline_wall_p95_seconds(self) -> float:
        return _percentile(self._wall_values("baseline"), 0.95)

    @property
    def strategy_wall_p50_seconds(self) -> float:
        return _percentile(self._wall_values("strategy"), 0.5)

    @property
    def strategy_wall_p95_seconds(self) -> float:
        return _percentile(self._wall_values("strategy"), 0.95)

    @property
    def baseline_throughput_calls_per_second(self) -> float | None:
        return self._throughput("baseline")

    @property
    def strategy_throughput_calls_per_second(self) -> float | None:
        return self._throughput("strategy")

    @property
    def baseline_provider_seconds(self) -> float:
        """Total provider-reported time; a diagnostic, not a cross-variant statistic."""

        return self.baseline_spend.elapsed_seconds

    @property
    def strategy_provider_seconds(self) -> float:
        """Total provider-reported time; a diagnostic, not a cross-variant statistic."""

        return self.strategy_spend.elapsed_seconds

    @property
    def input_token_ratio(self) -> float | None:
        return _ratio(self.strategy_spend.input_tokens, self.baseline_spend.input_tokens)

    @property
    def output_token_ratio(self) -> float | None:
        return _ratio(self.strategy_spend.output_tokens, self.baseline_spend.output_tokens)

    @property
    def baseline_cost(self) -> float | None:
        return self.baseline_spend.cost if self.baseline_compute.cost_available else None

    @property
    def strategy_cost(self) -> float | None:
        return self.strategy_spend.cost if self.strategy_compute.cost_available else None

    @property
    def cost_comparable(self) -> bool:
        return self.baseline_compute.cost_available and self.strategy_compute.cost_available

    @property
    def ttft_seconds(self) -> None:
        """TTFT is unavailable until the canonical result envelope records streaming timing."""

        return None

    def to_dict(self) -> dict[str, object]:
        """Return a stable, strict-JSON-safe evidence artifact."""

        return {
            "variants": {"baseline": self.baseline_name, "strategy": self.strategy_name},
            "quality": {
                "mean_baseline_score": self.mean_baseline_score,
                "mean_strategy_score": self.mean_strategy_score,
                "mean_score_delta": self.mean_score_delta,
                "baseline_pass_at_n": self.baseline_pass_at_n,
                "strategy_pass_rate": self.strategy_pass_rate,
                "mean_baseline_oracle_score": self.mean_baseline_oracle_score,
            },
            "regressions": {
                "count": self.regression_count,
                # Copy: callers routinely mutate the returned artifact, which
                # must not write through to this frozen report.
                "by_subset": dict(self.regressions_by_subset),
            },
            "compute": {
                "calls": {
                    "baseline": self.baseline_spend.calls,
                    "strategy": self.strategy_spend.calls,
                },
                "declared_calls_per_example": {
                    "baseline": self.baseline_compute.generated_calls,
                    "strategy": self.strategy_compute.generated_calls,
                },
                "max_output_tokens_per_example": {
                    "baseline": self.baseline_compute.max_output_tokens,
                    "strategy": self.strategy_compute.max_output_tokens,
                },
                "input_tokens": {
                    "baseline": self.baseline_spend.input_tokens,
                    "strategy": self.strategy_spend.input_tokens,
                    "strategy_to_baseline_ratio": self.input_token_ratio,
                },
                "output_tokens": {
                    "baseline": self.baseline_spend.output_tokens,
                    "strategy": self.strategy_spend.output_tokens,
                    "strategy_to_baseline_ratio": self.output_token_ratio,
                },
                "cost": {
                    "baseline": self.baseline_cost,
                    "strategy": self.strategy_cost,
                    "comparable": self.cost_comparable,
                },
            },
            "wall_seconds": {
                "baseline": {
                    "total": self.baseline_wall_seconds,
                    "p50": self.baseline_wall_p50_seconds,
                    "p95": self.baseline_wall_p95_seconds,
                },
                "strategy": {
                    "total": self.strategy_wall_seconds,
                    "p50": self.strategy_wall_p50_seconds,
                    "p95": self.strategy_wall_p95_seconds,
                },
            },
            "provider_seconds": {
                "baseline": self.baseline_provider_seconds,
                "strategy": self.strategy_provider_seconds,
                "comparable": False,
            },
            "throughput_calls_per_second": {
                "baseline": self.baseline_throughput_calls_per_second,
                "strategy": self.strategy_throughput_calls_per_second,
            },
            "matching": {
                "calls": self.calls_matched,
                "models": self.models_matched,
                "max_output_tokens": self.output_tokens_matched,
                "order_balanced": self.order_balanced,
            },
            "limitations": {
                "ttft_seconds": self.ttft_seconds,
                "cost_comparable": self.cost_comparable,
                "provider_seconds_comparable": False,
                "order_balanced": self.order_balanced,
            },
            "cases": [
                {
                    "example_id": case.example_id,
                    "trial": case.trial,
                    "subset": case.subset,
                    "baseline_scores": list(case.baseline_scores),
                    "strategy_scores": list(case.strategy_scores),
                    "baseline_passed": list(case.baseline_passed),
                    "strategy_passed": list(case.strategy_passed),
                    "score_delta": case.score_delta,
                    "regression": case.regression,
                    "baseline_calls": case.baseline_spend.calls,
                    "strategy_calls": case.strategy_spend.calls,
                    "baseline_spend": _spend_dict(case.baseline_spend),
                    "strategy_spend": _spend_dict(case.strategy_spend),
                    "wall_seconds": {
                        "baseline": case.baseline_wall_seconds,
                        "strategy": case.strategy_wall_seconds,
                    },
                    "baseline_models": list(case.baseline_models),
                    "strategy_models": list(case.strategy_models),
                }
                for case in self.cases
            ],
        }


async def run_matched_compute(
    examples: Sequence[EvaluationExample],
    *,
    baseline: MatchedComputeVariant,
    strategy: MatchedComputeVariant,
    evaluator: Evaluator,
    trials: int,
) -> MatchedComputeReport:
    """Run paired repeated trials and summarize quality and actual spend."""

    if not examples:
        raise ValueError("at least one evaluation example is required")
    if trials < 1:
        raise ValueError("trials must be positive")
    for example in examples:
        if example.expected_output is None:
            raise ValueError(f"example {example.id!r} has no expected output")
    _check_compute_profiles(baseline.compute, strategy.compute)

    cases: list[MatchedComputeCase] = []
    for trial in range(trials):
        for example in examples:
            reference = example.expected_output
            assert reference is not None
            request = InferenceRequest(
                prompt=example.input_text,
                metadata={"example_id": example.id, "trial": trial},
            )
            if trial % 2 == 0:
                baseline_results, baseline_wall_seconds = await _timed_run(baseline, request)
                strategy_results, strategy_wall_seconds = await _timed_run(strategy, request)
            else:
                strategy_results, strategy_wall_seconds = await _timed_run(strategy, request)
                baseline_results, baseline_wall_seconds = await _timed_run(baseline, request)
            if not baseline_results or not strategy_results:
                raise ValueError("variants must return at least one inference result")

            baseline_evaluations = tuple(
                evaluator.evaluate(str(result.answer), reference) for result in baseline_results
            )
            strategy_evaluations = tuple(
                evaluator.evaluate(str(result.answer), reference) for result in strategy_results
            )
            case = MatchedComputeCase(
                example_id=example.id,
                trial=trial,
                subset=example.category or "uncategorized",
                baseline_scores=_finite_scores(
                    baseline_evaluations, example.id, trial, baseline.name
                ),
                strategy_scores=_finite_scores(
                    strategy_evaluations, example.id, trial, strategy.name
                ),
                baseline_passed=tuple(bool(result.passed) for result in baseline_evaluations),
                strategy_passed=tuple(bool(result.passed) for result in strategy_evaluations),
                baseline_spend=_sum_spend(result.spend for result in baseline_results),
                strategy_spend=_sum_spend(result.spend for result in strategy_results),
                baseline_models=_models(baseline_results),
                strategy_models=_models(strategy_results),
                baseline_wall_seconds=baseline_wall_seconds,
                strategy_wall_seconds=strategy_wall_seconds,
            )
            _check_case(case, baseline, strategy)
            cases.append(case)

    regressions = Counter(case.subset for case in cases if case.regression)
    return MatchedComputeReport(
        baseline_name=baseline.name,
        strategy_name=strategy.name,
        baseline_compute=baseline.compute,
        strategy_compute=strategy.compute,
        cases=tuple(cases),
        baseline_spend=_sum_spend(case.baseline_spend for case in cases),
        strategy_spend=_sum_spend(case.strategy_spend for case in cases),
        order_balanced=trials % 2 == 0,
        regressions_by_subset=dict(sorted(regressions.items())),
    )


def _executor_identity(profile: ComputeProfile) -> object | None:
    if profile.executor is None:
        return None
    # Compare through the underlying model when the executor exposes one, so
    # two clients wrapping the same model count as the same executor.
    return getattr(profile.executor, "model", profile.executor)


def _check_compute_profiles(baseline: ComputeProfile, strategy: ComputeProfile) -> None:
    baseline_executor = _executor_identity(baseline)
    strategy_executor = _executor_identity(strategy)
    if baseline_executor is not None and strategy_executor is not None:
        if baseline_executor is not strategy_executor:
            raise ComputeMismatchError(
                "variants must share one model executor: "
                f"{baseline_executor!r} is not {strategy_executor!r}"
            )
    elif baseline.executor_id != strategy.executor_id:
        raise ComputeMismatchError(
            "variants must share one model executor: "
            f"{baseline.executor_id} != {strategy.executor_id}"
        )
    if baseline.generated_calls != strategy.generated_calls:
        raise ComputeMismatchError(
            f"declared model calls differ: {baseline.generated_calls} != {strategy.generated_calls}"
        )
    if baseline.max_output_tokens_per_call != strategy.max_output_tokens_per_call:
        raise ComputeMismatchError(
            "declared max output-token budget per call differs: "
            f"{baseline.max_output_tokens_per_call} != {strategy.max_output_tokens_per_call}"
        )


def _check_case(
    case: MatchedComputeCase,
    baseline: MatchedComputeVariant,
    strategy: MatchedComputeVariant,
) -> None:
    where = f"example {case.example_id!r}, trial {case.trial}"
    for variant, observed in (
        (baseline, case.baseline_spend.calls),
        (strategy, case.strategy_spend.calls),
    ):
        if observed > variant.compute.generated_calls:
            raise ComputeMismatchError(
                f"variant {variant.name!r} made {observed} model calls for {where} but declared "
                f"at most {variant.compute.generated_calls}; declare every model-backed judge "
                "or verifier call in its ComputeProfile"
            )
    if not case.baseline_models or not case.strategy_models:
        raise ComputeMismatchError(f"model provenance is missing for {where}")
    if not case.models_matched:
        raise ComputeMismatchError(
            f"models differ for {where}: {case.baseline_models!r} != {case.strategy_models!r}"
        )


def _finite_scores(
    evaluations: Iterable[object],
    example_id: str,
    trial: int,
    variant_name: str,
) -> tuple[float, ...]:
    scores: list[float] = []
    for evaluation in evaluations:
        score = float(getattr(evaluation, "score"))
        if not math.isfinite(score):
            raise ValueError(
                f"variant {variant_name!r} produced a non-finite score {score!r} for "
                f"example {example_id!r}, trial {trial}"
            )
        scores.append(score)
    return tuple(scores)


def _spend_dict(spend: Spend) -> dict[str, int | float]:
    return {
        "calls": spend.calls,
        "input_tokens": spend.input_tokens,
        "output_tokens": spend.output_tokens,
        "elapsed_seconds": spend.elapsed_seconds,
        "cost": spend.cost,
        "evaluations": spend.evaluations,
    }


def _sum_spend(spends: Iterable[Spend]) -> Spend:
    items = tuple(spends)
    return Spend(
        calls=sum(item.calls for item in items),
        input_tokens=sum(item.input_tokens for item in items),
        output_tokens=sum(item.output_tokens for item in items),
        elapsed_seconds=sum(item.elapsed_seconds for item in items),
        cost=sum(item.cost for item in items),
        evaluations=sum(item.evaluations for item in items),
    )


async def _timed_run(
    variant: MatchedComputeVariant,
    request: InferenceRequest,
) -> tuple[tuple[InferenceResult, ...], float]:
    started = perf_counter()
    results = tuple(await variant.run(request))
    return results, perf_counter() - started


def _models(results: Sequence[InferenceResult]) -> tuple[str, ...]:
    models: set[str] = set()
    for result in results:
        model = result.provenance.get("model")
        if isinstance(model, (tuple, list, set)):
            models.update(str(item) for item in model if str(item))
        elif model is not None and str(model):
            models.add(str(model))
    return tuple(sorted(models))


def _mean(values: tuple[float, ...]) -> float:
    return calculate_mean(list(values)) if values else 0.0


def _percentile(values: tuple[float, ...], probability: float) -> float:
    if not values:
        return 0.0
    return calculate_percentile(list(values), probability * 100)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
