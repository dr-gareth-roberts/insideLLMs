"""Statistical analysis utilities for experiment results.

This module provides tools for analyzing experiment results including:
- Confidence intervals
- Significance testing
- Aggregation across multiple runs
- Effect size calculations
- Bootstrap resampling
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from insideLLMs.types import (
    ExperimentResult,
)


@dataclass
class ConfidenceInterval:
    """A confidence interval with lower and upper bounds."""

    point_estimate: float
    lower: float
    upper: float
    confidence_level: float = 0.95
    method: str = "normal"

    @property
    def margin_of_error(self) -> float:
        """Half-width of the confidence interval."""
        return (self.upper - self.lower) / 2

    @property
    def width(self) -> float:
        """Full width of the confidence interval."""
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        """Check if a value falls within the interval."""
        return self.lower <= value <= self.upper

    def __repr__(self) -> str:
        return (
            f"ConfidenceInterval({self.point_estimate:.4f} "
            f"[{self.lower:.4f}, {self.upper:.4f}], "
            f"{self.confidence_level * 100:.0f}%)"
        )


@dataclass
class HypothesisTestResult:
    """Result of a statistical hypothesis test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    effect_size: Optional[float] = None
    effect_size_interpretation: Optional[str] = None
    conclusion: str = ""

    def __repr__(self) -> str:
        sig = "significant" if self.significant else "not significant"
        return f"HypothesisTestResult({self.test_name}: p={self.p_value:.4f}, {sig})"


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a sample."""

    n: int
    mean: float
    std: float
    variance: float
    min_val: float
    max_val: float
    median: float
    q1: float
    q3: float
    iqr: float
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

    @property
    def range(self) -> float:
        """Range of the data."""
        return self.max_val - self.min_val

    @property
    def cv(self) -> float:
        """Coefficient of variation."""
        if self.mean == 0:
            return float("inf")
        return self.std / abs(self.mean)


@dataclass
class AggregatedResults:
    """Aggregated results from multiple experiment runs."""

    metric_name: str
    values: list[float]
    stats: DescriptiveStats
    ci: ConfidenceInterval
    raw_results: list[ExperimentResult] = field(default_factory=list)


@dataclass
class StatisticalComparisonResult:
    """Result of comparing two groups/experiments."""

    group_a_name: str
    group_b_name: str
    metric_name: str
    group_a_stats: DescriptiveStats
    group_b_stats: DescriptiveStats
    test_result: HypothesisTestResult
    difference: float
    difference_ci: Optional[ConfidenceInterval] = None


# Statistical helper functions


def _z_score(confidence_level: float) -> float:
    """Get z-score for a given confidence level using normal approximation."""
    # Common z-scores for standard confidence levels
    z_table = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
        0.999: 3.291,
    }
    if confidence_level in z_table:
        return z_table[confidence_level]

    # Approximation using inverse error function
    # For two-tailed test: z = sqrt(2) * erfinv(confidence_level)
    # Using Winitzki's approximation for erfinv
    p = confidence_level
    a = 0.147  # Constant for approximation

    def erfinv_approx(x: float) -> float:
        """Approximate inverse error function."""
        if x == 0:
            return 0
        sign = 1 if x > 0 else -1
        x = abs(x)
        ln_term = math.log(1 - x * x)
        term1 = 2 / (math.pi * a) + ln_term / 2
        term2 = ln_term / a
        return sign * math.sqrt(math.sqrt(term1 * term1 - term2) - term1)

    return math.sqrt(2) * erfinv_approx(p)


def _t_score(confidence_level: float, df: int) -> float:
    """Approximate t-score for given confidence level and degrees of freedom.

    Uses a reasonable approximation when scipy is not available.
    """
    # For large df, t approaches z
    if df > 100:
        return _z_score(confidence_level)

    # Approximation for smaller df
    z = _z_score(confidence_level)
    g1 = (z**3 + z) / 4
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96
    g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384

    return z + g1 / df + g2 / (df**2) + g3 / (df**3)


def calculate_mean(values: list[float]) -> float:
    """Calculate arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_variance(values: list[float], ddof: int = 1) -> float:
    """Calculate sample variance.

    Args:
        values: List of numeric values.
        ddof: Delta degrees of freedom (1 for sample, 0 for population).

    Returns:
        Sample variance.
    """
    if len(values) < 2:
        return 0.0
    mean = calculate_mean(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    return sum(squared_diffs) / (len(values) - ddof)


def calculate_std(values: list[float], ddof: int = 1) -> float:
    """Calculate sample standard deviation."""
    return math.sqrt(calculate_variance(values, ddof))


def calculate_median(values: list[float]) -> float:
    """Calculate median."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return sorted_values[mid]


def calculate_percentile(values: list[float], percentile: float) -> float:
    """Calculate a specific percentile.

    Args:
        values: List of numeric values.
        percentile: Percentile to calculate (0-100).

    Returns:
        The value at the given percentile.
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    k = (percentile / 100) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def calculate_skewness(values: list[float]) -> float:
    """Calculate Fisher-Pearson coefficient of skewness."""
    if len(values) < 3:
        return 0.0
    n = len(values)
    mean = calculate_mean(values)
    std = calculate_std(values, ddof=0)
    if std == 0:
        return 0.0
    m3 = sum((x - mean) ** 3 for x in values) / n
    return m3 / (std**3)


def calculate_kurtosis(values: list[float]) -> float:
    """Calculate excess kurtosis (Fisher's definition)."""
    if len(values) < 4:
        return 0.0
    n = len(values)
    mean = calculate_mean(values)
    std = calculate_std(values, ddof=0)
    if std == 0:
        return 0.0
    m4 = sum((x - mean) ** 4 for x in values) / n
    return m4 / (std**4) - 3


def descriptive_statistics(values: list[float]) -> DescriptiveStats:
    """Calculate comprehensive descriptive statistics.

    Args:
        values: List of numeric values.

    Returns:
        DescriptiveStats object with all statistics.
    """
    if not values:
        return DescriptiveStats(
            n=0,
            mean=0.0,
            std=0.0,
            variance=0.0,
            min_val=0.0,
            max_val=0.0,
            median=0.0,
            q1=0.0,
            q3=0.0,
            iqr=0.0,
        )

    q1 = calculate_percentile(values, 25)
    q3 = calculate_percentile(values, 75)

    return DescriptiveStats(
        n=len(values),
        mean=calculate_mean(values),
        std=calculate_std(values),
        variance=calculate_variance(values),
        min_val=min(values),
        max_val=max(values),
        median=calculate_median(values),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        skewness=calculate_skewness(values) if len(values) >= 3 else None,
        kurtosis=calculate_kurtosis(values) if len(values) >= 4 else None,
    )


def confidence_interval(
    values: list[float],
    confidence_level: float = 0.95,
    method: str = "t",
) -> ConfidenceInterval:
    """Calculate confidence interval for the mean.

    Args:
        values: List of numeric values.
        confidence_level: Confidence level (default 0.95 for 95% CI).
        method: "t" for t-distribution, "z" for normal, "bootstrap" for bootstrap.

    Returns:
        ConfidenceInterval object.
    """
    if not values:
        return ConfidenceInterval(
            point_estimate=0.0,
            lower=0.0,
            upper=0.0,
            confidence_level=confidence_level,
            method=method,
        )

    n = len(values)
    mean = calculate_mean(values)
    std = calculate_std(values)
    se = std / math.sqrt(n) if n > 0 else 0

    if method == "bootstrap":
        return bootstrap_confidence_interval(values, lambda x: calculate_mean(x), confidence_level)
    elif method == "z" or n > 100:
        z = _z_score(confidence_level)
        margin = z * se
        return ConfidenceInterval(
            point_estimate=mean,
            lower=mean - margin,
            upper=mean + margin,
            confidence_level=confidence_level,
            method="z",
        )
    else:
        # Use t-distribution for small samples
        df = n - 1
        t = _t_score(confidence_level, df)
        margin = t * se
        return ConfidenceInterval(
            point_estimate=mean,
            lower=mean - margin,
            upper=mean + margin,
            confidence_level=confidence_level,
            method="t",
        )


def bootstrap_confidence_interval(
    values: list[float],
    statistic_fn: Callable[[list[float]], float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> ConfidenceInterval:
    """Calculate bootstrap confidence interval.

    Args:
        values: List of numeric values.
        statistic_fn: Function that computes the statistic of interest.
        confidence_level: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed for reproducibility.

    Returns:
        ConfidenceInterval object.
    """
    import random

    if seed is not None:
        random.seed(seed)

    if not values:
        return ConfidenceInterval(
            point_estimate=0.0,
            lower=0.0,
            upper=0.0,
            confidence_level=confidence_level,
            method="bootstrap",
        )

    n = len(values)
    point_estimate = statistic_fn(values)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = [random.choice(values) for _ in range(n)]
        bootstrap_stats.append(statistic_fn(sample))

    bootstrap_stats.sort()
    alpha = 1 - confidence_level
    lower_idx = int((alpha / 2) * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap) - 1

    return ConfidenceInterval(
        point_estimate=point_estimate,
        lower=bootstrap_stats[lower_idx],
        upper=bootstrap_stats[upper_idx],
        confidence_level=confidence_level,
        method="bootstrap",
    )


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Calculate Cohen's d effect size.

    Args:
        group1: First group of values.
        group2: Second group of values.

    Returns:
        Cohen's d effect size.
    """
    if not group1 or not group2:
        return 0.0

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = calculate_mean(group1), calculate_mean(group2)
    var1, var2 = calculate_variance(group1), calculate_variance(group2)

    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value.

    Returns:
        Interpretation string.
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def welchs_t_test(
    group1: list[float],
    group2: list[float],
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Perform Welch's t-test for independent samples.

    Welch's t-test is more robust than Student's t-test when
    variances are unequal.

    Args:
        group1: First group of values.
        group2: Second group of values.
        alpha: Significance level.

    Returns:
        HypothesisTestResult object.
    """
    if not group1 or not group2:
        return HypothesisTestResult(
            test_name="Welch's t-test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            conclusion="Insufficient data for test",
        )

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = calculate_mean(group1), calculate_mean(group2)
    var1, var2 = calculate_variance(group1), calculate_variance(group2)

    # Standard error of difference
    se = math.sqrt(var1 / n1 + var2 / n2) if (var1 / n1 + var2 / n2) > 0 else 1e-10

    # t-statistic
    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var1 / n1 + var2 / n2) ** 2
    denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    df = num / denom if denom > 0 else 1  # noqa: F841 - degrees of freedom for reference

    # Approximate p-value (two-tailed)
    # Using normal approximation for simplicity
    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

    effect_size = cohens_d(group1, group2)
    effect_interp = interpret_cohens_d(effect_size)

    significant = p_value < alpha

    conclusion = (
        f"The difference between groups is {'statistically significant' if significant else 'not statistically significant'} "
        f"(t={t_stat:.3f}, p={p_value:.4f}). "
        f"Effect size: {effect_interp} (d={effect_size:.3f})."
    )

    return HypothesisTestResult(
        test_name="Welch's t-test",
        statistic=t_stat,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        effect_size_interpretation=effect_interp,
        conclusion=conclusion,
    )


def _normal_cdf(x: float) -> float:
    """Approximate cumulative distribution function for standard normal."""
    # Using error function approximation
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def paired_t_test(
    values1: list[float],
    values2: list[float],
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Perform paired t-test.

    Args:
        values1: First set of paired values.
        values2: Second set of paired values.
        alpha: Significance level.

    Returns:
        HypothesisTestResult object.
    """
    if len(values1) != len(values2):
        raise ValueError("Paired samples must have equal length")

    if not values1:
        return HypothesisTestResult(
            test_name="Paired t-test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            conclusion="Insufficient data for test",
        )

    # Calculate differences
    differences = [v1 - v2 for v1, v2 in zip(values1, values2)]

    n = len(differences)
    mean_diff = calculate_mean(differences)
    std_diff = calculate_std(differences)
    se = std_diff / math.sqrt(n)

    if se == 0:
        return HypothesisTestResult(
            test_name="Paired t-test",
            statistic=float("inf") if mean_diff != 0 else 0.0,
            p_value=0.0 if mean_diff != 0 else 1.0,
            significant=mean_diff != 0,
            alpha=alpha,
            conclusion="Zero variance in differences",
        )

    t_stat = mean_diff / se
    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

    # Effect size (Cohen's d for paired samples)
    effect_size = mean_diff / std_diff if std_diff != 0 else 0.0
    effect_interp = interpret_cohens_d(effect_size)

    significant = p_value < alpha

    conclusion = (
        f"The mean difference is {'statistically significant' if significant else 'not statistically significant'} "
        f"(t={t_stat:.3f}, p={p_value:.4f}). "
        f"Effect size: {effect_interp} (d={effect_size:.3f})."
    )

    return HypothesisTestResult(
        test_name="Paired t-test",
        statistic=t_stat,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        effect_size_interpretation=effect_interp,
        conclusion=conclusion,
    )


def mann_whitney_u(
    group1: list[float],
    group2: list[float],
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Perform Mann-Whitney U test (non-parametric alternative to t-test).

    Args:
        group1: First group of values.
        group2: Second group of values.
        alpha: Significance level.

    Returns:
        HypothesisTestResult object.
    """
    if not group1 or not group2:
        return HypothesisTestResult(
            test_name="Mann-Whitney U",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            conclusion="Insufficient data for test",
        )

    n1, n2 = len(group1), len(group2)

    # Combine and rank
    combined = [(v, 1) for v in group1] + [(v, 2) for v in group2]
    combined.sort(key=lambda x: x[0])

    # Assign ranks (handling ties by averaging)
    ranks = []
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks.append((avg_rank, combined[k][1]))
        i = j

    # Sum of ranks for group 1
    r1 = sum(rank for rank, group in ranks if group == 1)

    # U statistics
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u_stat = min(u1, u2)

    # Normal approximation for large samples
    mean_u = n1 * n2 / 2
    std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    z_stat = 0.0 if std_u == 0 else (u_stat - mean_u) / std_u

    p_value = 2 * (1 - _normal_cdf(abs(z_stat)))

    # Effect size: rank-biserial correlation
    effect_size = 1 - 2 * u_stat / (n1 * n2)
    if effect_size < -0.1:
        effect_interp = "group 2 higher"
    elif effect_size > 0.1:
        effect_interp = "group 1 higher"
    else:
        effect_interp = "negligible difference"

    significant = p_value < alpha

    conclusion = (
        f"The difference between groups is {'statistically significant' if significant else 'not statistically significant'} "
        f"(U={u_stat:.1f}, z={z_stat:.3f}, p={p_value:.4f})."
    )

    return HypothesisTestResult(
        test_name="Mann-Whitney U",
        statistic=u_stat,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        effect_size_interpretation=effect_interp,
        conclusion=conclusion,
    )


# Experiment analysis functions


def extract_metric_from_results(
    results: list[ExperimentResult],
    metric: str = "accuracy",
) -> list[float]:
    """Extract a specific metric from experiment results.

    Args:
        results: List of experiment results.
        metric: Metric name to extract (accuracy, precision, recall, f1_score, etc.).

    Returns:
        List of metric values.
    """
    values = []
    for result in results:
        if result.score is not None:
            if metric == "accuracy" and result.score.accuracy is not None:
                values.append(result.score.accuracy)
            elif metric == "precision" and result.score.precision is not None:
                values.append(result.score.precision)
            elif metric == "recall" and result.score.recall is not None:
                values.append(result.score.recall)
            elif metric == "f1_score" and result.score.f1_score is not None:
                values.append(result.score.f1_score)
            elif metric == "error_rate":
                values.append(result.score.error_rate)
            elif metric == "mean_latency_ms" and result.score.mean_latency_ms is not None:
                values.append(result.score.mean_latency_ms)
            elif metric in result.score.custom_metrics:
                values.append(result.score.custom_metrics[metric])
    return values


def extract_latencies(results: list[ExperimentResult]) -> list[float]:
    """Extract all latencies from experiment results.

    Args:
        results: List of experiment results.

    Returns:
        List of latency values in milliseconds.
    """
    latencies = []
    for result in results:
        for probe_result in result.results:
            if probe_result.latency_ms is not None:
                latencies.append(probe_result.latency_ms)
    return latencies


def extract_success_rates(results: list[ExperimentResult]) -> list[float]:
    """Extract success rates from experiment results.

    Args:
        results: List of experiment results.

    Returns:
        List of success rate values (0-1).
    """
    return [result.success_rate for result in results]


def aggregate_experiment_results(
    results: list[ExperimentResult],
    metric: str = "accuracy",
    confidence_level: float = 0.95,
) -> AggregatedResults:
    """Aggregate results from multiple experiment runs.

    Args:
        results: List of experiment results.
        metric: Metric to aggregate.
        confidence_level: Confidence level for CI.

    Returns:
        AggregatedResults object with statistics and CI.
    """
    values = extract_metric_from_results(results, metric)

    stats = descriptive_statistics(values)
    ci = confidence_interval(values, confidence_level)

    return AggregatedResults(
        metric_name=metric,
        values=values,
        stats=stats,
        ci=ci,
        raw_results=results,
    )


def compare_experiments(
    results_a: list[ExperimentResult],
    results_b: list[ExperimentResult],
    metric: str = "accuracy",
    test: str = "welch",
    alpha: float = 0.05,
    name_a: str = "Group A",
    name_b: str = "Group B",
) -> StatisticalComparisonResult:
    """Compare two sets of experiment results.

    Args:
        results_a: First set of experiment results.
        results_b: Second set of experiment results.
        metric: Metric to compare.
        test: Statistical test ("welch", "paired", "mannwhitney").
        alpha: Significance level.
        name_a: Name for first group.
        name_b: Name for second group.

    Returns:
        StatisticalComparisonResult object.
    """
    values_a = extract_metric_from_results(results_a, metric)
    values_b = extract_metric_from_results(results_b, metric)

    stats_a = descriptive_statistics(values_a)
    stats_b = descriptive_statistics(values_b)

    if test == "welch":
        test_result = welchs_t_test(values_a, values_b, alpha)
    elif test == "paired":
        test_result = paired_t_test(values_a, values_b, alpha)
    elif test == "mannwhitney":
        test_result = mann_whitney_u(values_a, values_b, alpha)
    else:
        raise ValueError(f"Unknown test: {test}")

    difference = stats_a.mean - stats_b.mean

    # CI for the difference (using pooled SE for simplicity)
    if values_a and values_b:
        se_diff = math.sqrt(stats_a.variance / len(values_a) + stats_b.variance / len(values_b))
        z = _z_score(1 - alpha)
        diff_ci = ConfidenceInterval(
            point_estimate=difference,
            lower=difference - z * se_diff,
            upper=difference + z * se_diff,
            confidence_level=1 - alpha,
            method="z",
        )
    else:
        diff_ci = None

    return StatisticalComparisonResult(
        group_a_name=name_a,
        group_b_name=name_b,
        metric_name=metric,
        group_a_stats=stats_a,
        group_b_stats=stats_b,
        test_result=test_result,
        difference=difference,
        difference_ci=diff_ci,
    )


def multiple_comparison_correction(
    p_values: list[float],
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> tuple[list[float], list[bool]]:
    """Apply multiple comparison correction to p-values.

    Args:
        p_values: List of p-values to correct.
        method: Correction method ("bonferroni", "holm", "fdr_bh").
        alpha: Family-wise error rate.

    Returns:
        Tuple of (corrected p-values, significant flags).
    """
    n = len(p_values)
    if n == 0:
        return [], []

    if method == "bonferroni":
        # Simple Bonferroni correction
        corrected = [min(p * n, 1.0) for p in p_values]
        significant = [p < alpha / n for p in p_values]

    elif method == "holm":
        # Holm-Bonferroni step-down method
        indexed = [(p, i) for i, p in enumerate(p_values)]
        indexed.sort(key=lambda x: x[0])

        corrected = [0.0] * n
        significant = [False] * n
        max_corrected = 0.0

        for rank, (p, orig_idx) in enumerate(indexed):
            k = n - rank
            adj_p = p * k
            max_corrected = max(max_corrected, adj_p)
            corrected[orig_idx] = min(max_corrected, 1.0)
            significant[orig_idx] = p < alpha / k

    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR correction
        indexed = [(p, i) for i, p in enumerate(p_values)]
        indexed.sort(key=lambda x: x[0])

        corrected = [0.0] * n
        significant = [False] * n
        min_corrected = 1.0

        for rank in range(n - 1, -1, -1):
            p, orig_idx = indexed[rank]
            k = rank + 1
            adj_p = p * n / k
            min_corrected = min(min_corrected, adj_p)
            corrected[orig_idx] = min(min_corrected, 1.0)
            significant[orig_idx] = corrected[orig_idx] < alpha

    else:
        raise ValueError(f"Unknown correction method: {method}")

    return corrected, significant


def power_analysis(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
) -> float:
    """Calculate statistical power for a two-sample t-test.

    Args:
        effect_size: Expected Cohen's d effect size.
        n: Sample size per group.
        alpha: Significance level.

    Returns:
        Statistical power (0-1).
    """
    if n <= 1:
        return 0.0

    # Non-centrality parameter
    ncp = effect_size * math.sqrt(n / 2)

    # Critical value
    z_crit = _z_score(1 - alpha / 2)

    # Power = P(Z > z_crit - ncp) + P(Z < -z_crit - ncp)
    power = 1 - _normal_cdf(z_crit - ncp) + _normal_cdf(-z_crit - ncp)

    return power


def required_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """Calculate required sample size per group to achieve desired power.

    Args:
        effect_size: Expected Cohen's d effect size.
        power: Desired statistical power.
        alpha: Significance level.

    Returns:
        Required sample size per group.
    """
    if effect_size == 0:
        return float("inf")

    # Binary search for sample size
    low, high = 2, 10000
    target_power = power

    while low < high:
        mid = (low + high) // 2
        current_power = power_analysis(effect_size, mid, alpha)

        if current_power < target_power:
            low = mid + 1
        else:
            high = mid

    return low


def generate_summary_report(
    results: list[ExperimentResult],
    include_ci: bool = True,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Generate a comprehensive summary report of experiment results.

    Args:
        results: List of experiment results.
        include_ci: Whether to include confidence intervals.
        confidence_level: Confidence level for intervals.

    Returns:
        Dictionary with summary statistics and analysis.
    """
    if not results:
        return {"error": "No results to analyze"}

    # Group by model
    by_model: dict[str, list[ExperimentResult]] = {}
    for r in results:
        model_name = r.model_info.name
        if model_name not in by_model:
            by_model[model_name] = []
        by_model[model_name].append(r)

    # Group by probe
    by_probe: dict[str, list[ExperimentResult]] = {}
    for r in results:
        if r.probe_name not in by_probe:
            by_probe[r.probe_name] = []
        by_probe[r.probe_name].append(r)

    report = {
        "total_experiments": len(results),
        "unique_models": list(by_model.keys()),
        "unique_probes": list(by_probe.keys()),
        "by_model": {},
        "by_probe": {},
        "overall": {},
    }

    # Analyze by model
    for model_name, model_results in by_model.items():
        success_rates = extract_success_rates(model_results)
        latencies = extract_latencies(model_results)

        model_report = {
            "n_experiments": len(model_results),
            "success_rate": descriptive_statistics(success_rates).__dict__,
        }

        if latencies:
            model_report["latency_ms"] = descriptive_statistics(latencies).__dict__

        if include_ci and len(success_rates) > 1:
            ci = confidence_interval(success_rates, confidence_level)
            model_report["success_rate_ci"] = {
                "point_estimate": ci.point_estimate,
                "lower": ci.lower,
                "upper": ci.upper,
                "confidence_level": ci.confidence_level,
            }

        report["by_model"][model_name] = model_report

    # Analyze by probe
    for probe_name, probe_results in by_probe.items():
        success_rates = extract_success_rates(probe_results)

        probe_report = {
            "n_experiments": len(probe_results),
            "success_rate": descriptive_statistics(success_rates).__dict__,
        }

        # Extract accuracy if available
        accuracies = extract_metric_from_results(probe_results, "accuracy")
        if accuracies:
            probe_report["accuracy"] = descriptive_statistics(accuracies).__dict__
            if include_ci and len(accuracies) > 1:
                ci = confidence_interval(accuracies, confidence_level)
                probe_report["accuracy_ci"] = {
                    "point_estimate": ci.point_estimate,
                    "lower": ci.lower,
                    "upper": ci.upper,
                    "confidence_level": ci.confidence_level,
                }

        report["by_probe"][probe_name] = probe_report

    # Overall statistics
    all_success_rates = extract_success_rates(results)
    all_latencies = extract_latencies(results)

    report["overall"] = {
        "success_rate": descriptive_statistics(all_success_rates).__dict__,
    }

    if all_latencies:
        report["overall"]["latency_ms"] = descriptive_statistics(all_latencies).__dict__

    return report
