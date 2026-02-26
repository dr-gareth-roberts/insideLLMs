"""Tests for statistical analysis utilities."""

import pytest

from insideLLMs.analysis.statistics import (
    ConfidenceInterval,
    HypothesisTestResult,
    bootstrap_confidence_interval,
    calculate_kurtosis,
    calculate_mean,
    calculate_median,
    calculate_percentile,
    calculate_skewness,
    calculate_std,
    calculate_variance,
    cohens_d,
    compare_experiments,
    confidence_interval,
    descriptive_statistics,
    interpret_cohens_d,
    mann_whitney_u,
    multiple_comparison_correction,
    paired_t_test,
    power_analysis,
    required_sample_size,
    welchs_t_test,
)
from insideLLMs.types import (
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)


class TestBasicStatistics:
    """Tests for basic statistical functions."""

    def test_calculate_mean_basic(self):
        """Test mean calculation."""
        assert calculate_mean([1, 2, 3, 4, 5]) == 3.0
        assert calculate_mean([10]) == 10.0
        assert calculate_mean([]) == 0.0

    def test_calculate_mean_floats(self):
        """Test mean with floats."""
        result = calculate_mean([1.5, 2.5, 3.5])
        assert abs(result - 2.5) < 0.0001

    def test_calculate_variance(self):
        """Test variance calculation."""
        # Sample variance of [2, 4, 4, 4, 5, 5, 7, 9]
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        var = calculate_variance(data)
        assert abs(var - 4.571) < 0.01

    def test_calculate_variance_single(self):
        """Test variance with single value."""
        assert calculate_variance([5]) == 0.0
        assert calculate_variance([]) == 0.0

    def test_calculate_std(self):
        """Test standard deviation calculation."""
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        std = calculate_std(data)
        assert abs(std - 2.138) < 0.01

    def test_calculate_median_odd(self):
        """Test median with odd number of elements."""
        assert calculate_median([1, 3, 5, 7, 9]) == 5

    def test_calculate_median_even(self):
        """Test median with even number of elements."""
        assert calculate_median([1, 2, 3, 4]) == 2.5

    def test_calculate_median_empty(self):
        """Test median with empty list."""
        assert calculate_median([]) == 0.0

    def test_calculate_percentile(self):
        """Test percentile calculation."""
        data = list(range(1, 101))  # 1 to 100
        assert abs(calculate_percentile(data, 25) - 25.75) < 0.1
        assert abs(calculate_percentile(data, 50) - 50.5) < 0.1
        assert abs(calculate_percentile(data, 75) - 75.25) < 0.1

    def test_calculate_skewness_symmetric(self):
        """Test skewness of symmetric distribution."""
        # Symmetric data should have skewness near 0
        data = [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]
        skew = calculate_skewness(data)
        assert abs(skew) < 0.1

    def test_calculate_skewness_right(self):
        """Test skewness of right-skewed distribution."""
        data = [1, 1, 1, 2, 2, 3, 10]
        skew = calculate_skewness(data)
        assert skew > 0  # Right-skewed

    def test_calculate_kurtosis_normal(self):
        """Test kurtosis approaches 0 for normal-like distribution."""
        import random

        random.seed(42)
        # Large sample from approximately normal distribution
        data = [random.gauss(0, 1) for _ in range(1000)]
        kurt = calculate_kurtosis(data)
        # Should be close to 0 (excess kurtosis)
        assert abs(kurt) < 0.5


class TestDescriptiveStatistics:
    """Tests for descriptive statistics calculation."""

    def test_descriptive_statistics_basic(self):
        """Test basic descriptive statistics."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        stats = descriptive_statistics(data)

        assert stats.n == 10
        assert stats.mean == 5.5
        assert stats.min_val == 1
        assert stats.max_val == 10
        assert stats.median == 5.5

    def test_descriptive_statistics_iqr(self):
        """Test IQR calculation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        stats = descriptive_statistics(data)

        assert stats.iqr == stats.q3 - stats.q1

    def test_descriptive_statistics_empty(self):
        """Test with empty data."""
        stats = descriptive_statistics([])
        assert stats.n == 0
        assert stats.mean == 0.0

    def test_descriptive_statistics_properties(self):
        """Test computed properties."""
        data = [1, 5, 10]
        stats = descriptive_statistics(data)

        assert stats.range == 9
        assert stats.cv > 0  # Coefficient of variation


class TestConfidenceInterval:
    """Tests for confidence interval calculations."""

    def test_confidence_interval_z(self):
        """Test z-based confidence interval."""
        # Large sample (>100) uses z
        data = list(range(1, 201))
        ci = confidence_interval(data, 0.95, method="z")

        assert ci.confidence_level == 0.95
        assert ci.method == "z"
        assert ci.lower < ci.point_estimate < ci.upper

    def test_confidence_interval_t(self):
        """Test t-based confidence interval."""
        data = [1, 2, 3, 4, 5]
        ci = confidence_interval(data, 0.95, method="t")

        assert ci.point_estimate == 3.0
        assert ci.lower < 3.0 < ci.upper
        # T intervals are wider than z for small samples
        assert ci.width > 0

    def test_confidence_interval_contains(self):
        """Test contains method."""
        ci = ConfidenceInterval(
            point_estimate=5.0,
            lower=4.0,
            upper=6.0,
            confidence_level=0.95,
        )

        assert ci.contains(5.0)
        assert ci.contains(4.5)
        assert not ci.contains(3.0)
        assert not ci.contains(7.0)

    def test_confidence_interval_margin(self):
        """Test margin of error calculation."""
        ci = ConfidenceInterval(
            point_estimate=10.0,
            lower=8.0,
            upper=12.0,
            confidence_level=0.95,
        )

        assert ci.margin_of_error == 2.0
        assert ci.width == 4.0

    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ci = bootstrap_confidence_interval(
            data, calculate_mean, confidence_level=0.95, n_bootstrap=500, seed=42
        )

        assert ci.method == "bootstrap"
        # Bootstrap should contain the sample mean
        assert ci.lower < 5.5 < ci.upper


class TestHypothesisTesting:
    """Tests for hypothesis tests."""

    def test_welchs_t_test_different(self):
        """Test Welch's t-test with different means."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [10, 11, 12, 13, 14]

        result = welchs_t_test(group1, group2)

        assert result.test_name == "Welch's t-test"
        assert result.p_value < 0.05
        assert result.significant
        assert result.effect_size is not None

    def test_welchs_t_test_same(self):
        """Test Welch's t-test with similar means."""
        group1 = [5, 5, 5, 5, 5]
        group2 = [5, 5, 5, 5, 5]

        result = welchs_t_test(group1, group2)

        assert not result.significant
        assert result.p_value > 0.05

    def test_welchs_t_test_empty(self):
        """Test with empty groups."""
        result = welchs_t_test([], [1, 2, 3])
        assert not result.significant
        assert "Insufficient" in result.conclusion

    def test_paired_t_test_different(self):
        """Test paired t-test with different paired means."""
        before = [1, 2, 3, 4, 5]
        after = [6, 7, 8, 9, 10]

        result = paired_t_test(before, after)

        assert result.test_name == "Paired t-test"
        assert result.significant
        assert result.p_value < 0.05

    def test_paired_t_test_same(self):
        """Test paired t-test with identical values."""
        values1 = [1, 2, 3, 4, 5]
        values2 = [1, 2, 3, 4, 5]

        result = paired_t_test(values1, values2)
        assert not result.significant

    def test_paired_t_test_unequal_length(self):
        """Test paired t-test raises with unequal lengths."""
        with pytest.raises(ValueError):
            paired_t_test([1, 2, 3], [1, 2])

    def test_mann_whitney_different(self):
        """Test Mann-Whitney U with different distributions."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [10, 11, 12, 13, 14]

        result = mann_whitney_u(group1, group2)

        assert result.test_name == "Mann-Whitney U"
        assert result.significant
        assert result.statistic == 0  # Perfect separation

    def test_mann_whitney_same(self):
        """Test Mann-Whitney U with same distribution."""
        group1 = [1, 3, 5, 7, 9]
        group2 = [2, 4, 6, 8, 10]

        result = mann_whitney_u(group1, group2)
        # Values are interleaved, so not significant
        assert not result.significant


class TestEffectSize:
    """Tests for effect size calculations."""

    def test_cohens_d_large(self):
        """Test Cohen's d with large effect."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [10, 11, 12, 13, 14]

        d = cohens_d(group1, group2)
        assert abs(d) > 0.8  # Large effect

    def test_cohens_d_small(self):
        """Test Cohen's d with small effect."""
        # Need larger variance to get small effect size
        group1 = [4.0, 5.0, 6.0, 5.0, 5.0]
        group2 = [4.1, 5.1, 6.1, 5.1, 5.1]

        d = cohens_d(group1, group2)
        assert abs(d) < 0.5  # Small effect

    def test_cohens_d_zero(self):
        """Test Cohen's d with same means."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [1, 2, 3, 4, 5]

        d = cohens_d(group1, group2)
        assert d == 0.0

    def test_interpret_cohens_d(self):
        """Test effect size interpretation."""
        assert interpret_cohens_d(0.1) == "negligible"
        assert interpret_cohens_d(0.3) == "small"
        assert interpret_cohens_d(0.6) == "medium"
        assert interpret_cohens_d(1.0) == "large"
        assert interpret_cohens_d(-0.5) == "medium"  # Absolute value


class TestMultipleComparisons:
    """Tests for multiple comparison corrections."""

    def test_bonferroni_correction(self):
        """Test Bonferroni correction."""
        p_values = [0.005, 0.02, 0.03, 0.04, 0.05]
        corrected, significant = multiple_comparison_correction(
            p_values, method="bonferroni", alpha=0.05
        )

        # All p-values multiplied by 5
        assert corrected[0] == 0.025  # 0.005 * 5
        assert corrected[1] == 0.10  # 0.02 * 5
        # Only first one significant at 0.05/5 = 0.01
        assert significant[0]  # 0.005 < 0.01
        assert not significant[1]  # 0.02 >= 0.01

    def test_holm_correction(self):
        """Test Holm-Bonferroni correction."""
        p_values = [0.01, 0.04, 0.03, 0.02, 0.05]
        corrected, significant = multiple_comparison_correction(p_values, method="holm", alpha=0.05)

        # Holm is less conservative than Bonferroni
        assert len(corrected) == 5
        assert sum(significant) >= sum(
            [p < 0.05 / 5 for p in p_values]
        )  # At least as many as Bonferroni

    def test_fdr_correction(self):
        """Test FDR (Benjamini-Hochberg) correction."""
        p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
        corrected, significant = multiple_comparison_correction(
            p_values, method="fdr_bh", alpha=0.05
        )

        # FDR is typically more powerful
        assert len(corrected) == 10
        assert corrected[0] <= 0.05  # Smallest p-value should be significant


class TestPowerAnalysis:
    """Tests for power analysis functions."""

    def test_power_analysis_large_effect(self):
        """Test power with large effect size."""
        power = power_analysis(effect_size=0.8, n=30)
        # Large effect, moderate sample should have high power
        assert power > 0.7

    def test_power_analysis_small_effect(self):
        """Test power with small effect size."""
        power = power_analysis(effect_size=0.2, n=30)
        # Small effect needs larger sample
        assert power < 0.5

    def test_required_sample_size(self):
        """Test sample size calculation."""
        n = required_sample_size(effect_size=0.5, power=0.8)
        # Medium effect, 80% power should need ~60-70 per group
        assert 50 < n < 100

    def test_required_sample_size_large_effect(self):
        """Test sample size for large effect."""
        n = required_sample_size(effect_size=0.8, power=0.8)
        # Large effect needs smaller sample
        assert n < 50


class TestExperimentAnalysis:
    """Tests for experiment result analysis."""

    def _create_experiment_result(
        self, model_name: str, accuracy: float, n_results: int = 10
    ) -> ExperimentResult:
        """Helper to create experiment results."""
        return ExperimentResult(
            experiment_id=f"test_{model_name}",
            model_info=ModelInfo(
                name=model_name,
                provider="test",
                model_id=f"{model_name}-v1",
            ),
            probe_name="test_probe",
            probe_category=ProbeCategory.LOGIC,
            results=[
                ProbeResult(
                    input=f"test_{i}",
                    output=f"output_{i}",
                    status=ResultStatus.SUCCESS,
                    latency_ms=100.0 + i,
                )
                for i in range(n_results)
            ],
            score=ProbeScore(
                accuracy=accuracy,
                precision=accuracy * 0.9,
                recall=accuracy * 0.95,
            ),
        )

    def test_compare_experiments(self):
        """Test comparing two sets of experiments."""
        results_a = [self._create_experiment_result("model_a", 0.8 + i * 0.01) for i in range(5)]
        results_b = [self._create_experiment_result("model_b", 0.6 + i * 0.01) for i in range(5)]

        comparison = compare_experiments(
            results_a,
            results_b,
            metric="accuracy",
            name_a="Model A",
            name_b="Model B",
        )

        assert comparison.group_a_name == "Model A"
        assert comparison.group_b_name == "Model B"
        assert comparison.difference > 0  # Model A is better
        assert comparison.test_result is not None

    def test_compare_experiments_same(self):
        """Test comparing identical experiments."""
        results = [self._create_experiment_result("model", 0.8) for _ in range(5)]

        comparison = compare_experiments(results, results, metric="accuracy")

        assert comparison.difference == 0
        assert not comparison.test_result.significant


class TestDataclassRepr:
    """Tests for dataclass representations."""

    def test_confidence_interval_repr(self):
        """Test ConfidenceInterval string representation."""
        ci = ConfidenceInterval(
            point_estimate=0.85,
            lower=0.80,
            upper=0.90,
            confidence_level=0.95,
        )
        repr_str = repr(ci)
        assert "0.85" in repr_str
        assert "95%" in repr_str

    def test_hypothesis_test_result_repr(self):
        """Test HypothesisTestResult string representation."""
        result = HypothesisTestResult(
            test_name="t-test",
            statistic=2.5,
            p_value=0.02,
            significant=True,
        )
        repr_str = repr(result)
        assert "t-test" in repr_str
        assert "significant" in repr_str
