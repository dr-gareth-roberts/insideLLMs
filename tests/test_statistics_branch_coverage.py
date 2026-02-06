"""Additional branch coverage for analysis.statistics."""

from __future__ import annotations

import math

import pytest

import insideLLMs.analysis.statistics as stats
from insideLLMs.types import (
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)


def _exp_result(model_name: str, score: ProbeScore | None) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=f"exp-{model_name}",
        model_info=ModelInfo(name=model_name, provider="test", model_id=f"{model_name}-id"),
        probe_name="probe",
        probe_category=ProbeCategory.LOGIC,
        results=[
            ProbeResult(
                input="in",
                output="out",
                status=ResultStatus.SUCCESS,
                latency_ms=12.0,
            )
        ],
        score=score,
    )


def test_cv_returns_inf_when_mean_is_zero():
    desc = stats.descriptive_statistics([-1.0, 1.0])
    assert desc.mean == 0.0
    assert desc.cv == float("inf")


def test_z_and_t_helpers_edge_paths():
    assert stats._z_score(0.0) == 0.0
    assert stats._t_score(0.95, 101) == stats._z_score(0.95)


def test_percentile_and_moments_edge_cases():
    assert stats.calculate_percentile([], 50) == 0.0
    assert stats.calculate_skewness([1.0, 1.0]) == 0.0
    assert stats.calculate_skewness([2.0, 2.0, 2.0]) == 0.0
    assert stats.calculate_kurtosis([1.0, 2.0, 3.0]) == 0.0
    assert stats.calculate_kurtosis([3.0, 3.0, 3.0, 3.0]) == 0.0


def test_confidence_interval_bootstrap_and_empty_paths():
    empty_ci = stats.confidence_interval([], method="t")
    assert empty_ci.point_estimate == 0.0
    assert empty_ci.lower == 0.0
    assert empty_ci.upper == 0.0

    boot_ci = stats.confidence_interval([1.0, 2.0, 3.0], method="bootstrap", confidence_level=0.9)
    assert boot_ci.method == "bootstrap"
    assert boot_ci.lower <= boot_ci.point_estimate <= boot_ci.upper


def test_bootstrap_confidence_interval_empty_returns_zero_ci():
    ci = stats.bootstrap_confidence_interval([], lambda x: sum(x), confidence_level=0.95, seed=7)
    assert ci.method == "bootstrap"
    assert ci.point_estimate == 0.0
    assert ci.lower == 0.0
    assert ci.upper == 0.0


def test_cohens_d_empty_groups_returns_zero():
    assert stats.cohens_d([], [1.0, 2.0]) == 0.0
    assert stats.cohens_d([1.0, 2.0], []) == 0.0


def test_paired_t_test_empty_and_non_constant_differences():
    empty = stats.paired_t_test([], [])
    assert empty.p_value == 1.0
    assert not empty.significant

    varied = stats.paired_t_test([1.0, 2.0, 4.0], [1.0, 1.0, 2.0])
    assert varied.test_name == "Paired t-test"
    assert varied.effect_size is not None
    assert "t=" in varied.conclusion


def test_mann_whitney_additional_effect_interpretations():
    empty = stats.mann_whitney_u([], [1.0, 2.0])
    assert empty.p_value == 1.0
    assert not empty.significant

    separated = stats.mann_whitney_u([1.0, 1.0, 1.0], [9.0, 9.0, 9.0])
    assert separated.effect_size_interpretation in {"group 1 higher", "group 2 higher"}

    negligible = stats.mann_whitney_u([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert negligible.effect_size_interpretation == "negligible difference"


def test_metric_extraction_branches_for_all_supported_metrics():
    scored = _exp_result(
        "m1",
        ProbeScore(
            accuracy=0.8,
            precision=0.7,
            recall=0.6,
            f1_score=0.65,
            mean_latency_ms=42.0,
            error_rate=0.2,
            custom_metrics={"custom_metric": 9.0},
        ),
    )
    unscored = _exp_result("m2", None)
    results = [scored, unscored]

    assert stats.extract_metric_from_results(results, "accuracy") == [0.8]
    assert stats.extract_metric_from_results(results, "precision") == [0.7]
    assert stats.extract_metric_from_results(results, "recall") == [0.6]
    assert stats.extract_metric_from_results(results, "f1_score") == [0.65]
    assert stats.extract_metric_from_results(results, "error_rate") == [0.2]
    assert stats.extract_metric_from_results(results, "mean_latency_ms") == [42.0]
    assert stats.extract_metric_from_results(results, "custom_metric") == [9.0]


def test_aggregate_and_compare_experiment_branch_paths():
    a = [_exp_result("a", ProbeScore(accuracy=0.8)), _exp_result("a", ProbeScore(accuracy=0.9))]
    b = [_exp_result("b", ProbeScore(accuracy=0.6)), _exp_result("b", ProbeScore(accuracy=0.7))]

    aggregated = stats.aggregate_experiment_results(a, metric="accuracy", confidence_level=0.9)
    assert aggregated.metric_name == "accuracy"
    assert aggregated.raw_results == a

    paired = stats.compare_experiments(a, b, metric="accuracy", test="paired")
    assert paired.test_result.test_name == "Paired t-test"
    assert paired.difference_ci is not None

    mw = stats.compare_experiments(a, b, metric="accuracy", test="mannwhitney")
    assert mw.test_result.test_name == "Mann-Whitney U"

    empty = stats.compare_experiments([], [], metric="accuracy")
    assert empty.difference_ci is None

    with pytest.raises(ValueError, match="Unknown test"):
        stats.compare_experiments(a, b, metric="accuracy", test="bogus")


def test_multiple_comparison_correction_empty_and_unknown():
    corrected, significant = stats.multiple_comparison_correction([], method="bonferroni")
    assert corrected == []
    assert significant == []

    with pytest.raises(ValueError, match="Unknown correction method"):
        stats.multiple_comparison_correction([0.01, 0.02], method="invalid")


def test_power_and_sample_size_edge_paths():
    assert stats.power_analysis(effect_size=0.5, n=1) == 0.0
    assert stats.required_sample_size(effect_size=0.0) == math.inf


def test_generate_summary_report_empty_results():
    assert stats.generate_summary_report([]) == {"error": "No results to analyze"}
