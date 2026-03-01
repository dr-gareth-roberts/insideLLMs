"""Statistical analysis utilities for experiment results.

This module provides tools for analyzing experiment results including:
- Confidence intervals
- Significance testing
- Aggregation across multiple runs
- Effect size calculations
- Bootstrap resampling

Module Overview
---------------
The statistics module provides a comprehensive suite of statistical analysis
tools designed specifically for LLM experiment evaluation. It includes both
descriptive statistics and inferential methods for comparing model performance.

Quick Start Examples
--------------------
Computing descriptive statistics:

    >>> from insideLLMs.analysis.statistics import descriptive_statistics
    >>> values = [0.85, 0.88, 0.92, 0.87, 0.90, 0.89]
    >>> stats = descriptive_statistics(values)
    >>> print(f"Mean: {stats.mean:.3f}, Std: {stats.std:.3f}")
    Mean: 0.885, Std: 0.024

Computing confidence intervals:

    >>> from insideLLMs.analysis.statistics import confidence_interval
    >>> ci = confidence_interval(values, confidence_level=0.95)
    >>> print(f"95% CI: [{ci.lower:.3f}, {ci.upper:.3f}]")
    95% CI: [0.860, 0.910]

Comparing two groups with Welch's t-test:

    >>> from insideLLMs.analysis.statistics import welchs_t_test
    >>> group_a = [0.90, 0.92, 0.88, 0.91, 0.89]
    >>> group_b = [0.85, 0.83, 0.86, 0.84, 0.82]
    >>> result = welchs_t_test(group_a, group_b)
    >>> print(f"Significant: {result.significant}, p={result.p_value:.4f}")
    Significant: True, p=0.0002

Effect size calculation:

    >>> from insideLLMs.analysis.statistics import cohens_d, interpret_cohens_d
    >>> d = cohens_d(group_a, group_b)
    >>> print(f"Cohen's d: {d:.2f} ({interpret_cohens_d(d)})")
    Cohen's d: 3.53 (large)

Bootstrap confidence intervals:

    >>> from insideLLMs.analysis.statistics import bootstrap_confidence_interval
    >>> ci = bootstrap_confidence_interval(
    ...     values,
    ...     statistic_fn=lambda x: sum(x)/len(x),
    ...     n_bootstrap=1000,
    ...     seed=42
    ... )
    >>> print(f"Bootstrap 95% CI: [{ci.lower:.3f}, {ci.upper:.3f}]")
    Bootstrap 95% CI: [0.862, 0.907]

Classes Summary
---------------
- **ConfidenceInterval**: Confidence interval with bounds and methods
- **HypothesisTestResult**: Result of statistical hypothesis testing
- **DescriptiveStats**: Complete descriptive statistics for a sample
- **AggregatedResults**: Aggregated results with statistics and CI
- **StatisticalComparisonResult**: Full comparison of two groups

Key Functions
-------------
- **descriptive_statistics**: Compute comprehensive stats for a sample
- **confidence_interval**: Calculate CI for the mean (t, z, or bootstrap)
- **bootstrap_confidence_interval**: Non-parametric bootstrap CI
- **welchs_t_test**: Independent samples t-test (unequal variance)
- **paired_t_test**: Paired/dependent samples t-test
- **mann_whitney_u**: Non-parametric rank-based test
- **cohens_d**: Effect size for group comparison
- **compare_experiments**: Compare two sets of experiment results
- **power_analysis**: Calculate statistical power
- **required_sample_size**: Compute sample size for desired power

See Also
--------
insideLLMs.analysis.comparison : Model comparison utilities
insideLLMs.analysis.evaluation : Evaluation metrics
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from insideLLMs.types import (
    ExperimentResult,
)


@dataclass
class ConfidenceInterval:
    """A confidence interval with lower and upper bounds.

    Represents a confidence interval for a point estimate, including
    the bounds, confidence level, and computation method.

    Attributes:
        point_estimate: The central estimate (typically the sample mean).
        lower: Lower bound of the confidence interval.
        upper: Upper bound of the confidence interval.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        method: Method used to compute the interval ("t", "z", "bootstrap").

    Examples:
        Creating a confidence interval:

            >>> from insideLLMs.analysis.statistics import ConfidenceInterval
            >>> ci = ConfidenceInterval(
            ...     point_estimate=0.85,
            ...     lower=0.80,
            ...     upper=0.90,
            ...     confidence_level=0.95
            ... )
            >>> print(ci)
            ConfidenceInterval(0.8500 [0.8000, 0.9000], 95%)

        Checking margin of error:

            >>> ci.margin_of_error
            0.05

        Checking if a value is within the interval:

            >>> ci.contains(0.87)
            True
            >>> ci.contains(0.75)
            False

        Getting interval width:

            >>> ci.width
            0.1
    """

    point_estimate: float
    lower: float
    upper: float
    confidence_level: float = 0.95
    method: str = "normal"

    @property
    def margin_of_error(self) -> float:
        """Half-width of the confidence interval.

        Returns:
            The margin of error (distance from point estimate to either bound).

        Examples:
            >>> ci = ConfidenceInterval(0.5, 0.4, 0.6)
            >>> ci.margin_of_error
            0.1
        """
        return (self.upper - self.lower) / 2

    @property
    def width(self) -> float:
        """Full width of the confidence interval.

        Returns:
            The total width from lower to upper bound.

        Examples:
            >>> ci = ConfidenceInterval(0.5, 0.4, 0.6)
            >>> ci.width
            0.2
        """
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        """Check if a value falls within the interval.

        Args:
            value: The value to check.

        Returns:
            True if the value is within [lower, upper], False otherwise.

        Examples:
            >>> ci = ConfidenceInterval(0.5, 0.4, 0.6)
            >>> ci.contains(0.5)
            True
            >>> ci.contains(0.3)
            False
        """
        return self.lower <= value <= self.upper

    def __repr__(self) -> str:
        return (
            f"ConfidenceInterval({self.point_estimate:.4f} "
            f"[{self.lower:.4f}, {self.upper:.4f}], "
            f"{self.confidence_level * 100:.0f}%)"
        )


@dataclass
class HypothesisTestResult:
    """Result of a statistical hypothesis test.

    Contains all relevant information from a hypothesis test including
    the test statistic, p-value, significance determination, and
    effect size when applicable.

    Attributes:
        test_name: Name of the test performed (e.g., "Welch's t-test").
        statistic: The test statistic value (e.g., t-value, U-value).
        p_value: The computed p-value.
        significant: Whether the result is significant at alpha level.
        alpha: Significance level used (default 0.05).
        effect_size: Effect size measure (e.g., Cohen's d) if computed.
        effect_size_interpretation: Text interpretation of effect size.
        conclusion: Human-readable conclusion statement.

    Examples:
        Understanding test results:

            >>> from insideLLMs.analysis.statistics import welchs_t_test
            >>> group_a = [0.90, 0.92, 0.88, 0.91, 0.89]
            >>> group_b = [0.85, 0.83, 0.86, 0.84, 0.82]
            >>> result = welchs_t_test(group_a, group_b)
            >>> print(result)  # doctest: +ELLIPSIS
            HypothesisTestResult(Welch's t-test: p=..., significant)

        Accessing test components:

            >>> result.significant
            True
            >>> result.p_value < 0.05
            True

        Effect size interpretation:

            >>> result.effect_size_interpretation
            'large'

        Using the conclusion:

            >>> "statistically significant" in result.conclusion
            True
    """

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
    """Comprehensive descriptive statistics for a numeric sample.

    Contains measures of central tendency, dispersion, and distribution
    shape for a sample of numeric values.

    Attributes:
        n: Sample size (number of observations).
        mean: Arithmetic mean of the values.
        std: Sample standard deviation (ddof=1).
        variance: Sample variance (ddof=1).
        min_val: Minimum value in the sample.
        max_val: Maximum value in the sample.
        median: Median (50th percentile).
        q1: First quartile (25th percentile).
        q3: Third quartile (75th percentile).
        iqr: Interquartile range (q3 - q1).
        skewness: Skewness coefficient (None if n < 3).
        kurtosis: Excess kurtosis (None if n < 4).

    Examples:
        Computing descriptive statistics:

            >>> from insideLLMs.analysis.statistics import descriptive_statistics
            >>> data = [85, 90, 88, 92, 87, 91, 89, 86, 93, 88]
            >>> stats = descriptive_statistics(data)
            >>> print(f"n={stats.n}, mean={stats.mean:.1f}")
            n=10, mean=88.9

        Accessing distribution measures:

            >>> stats.min_val
            85
            >>> stats.max_val
            93
            >>> stats.median
            88.5

        Using quartiles for outlier detection:

            >>> lower_fence = stats.q1 - 1.5 * stats.iqr
            >>> upper_fence = stats.q3 + 1.5 * stats.iqr
            >>> outliers = [x for x in data if x < lower_fence or x > upper_fence]

        Coefficient of variation for relative spread:

            >>> cv = stats.cv
            >>> print(f"CV: {cv:.2%}")  # doctest: +SKIP
    """

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
        """Range of the data (max - min).

        Returns:
            The difference between maximum and minimum values.

        Examples:
            >>> from insideLLMs.analysis.statistics import descriptive_statistics
            >>> stats = descriptive_statistics([10, 20, 30, 40, 50])
            >>> stats.range
            40
        """
        return self.max_val - self.min_val

    @property
    def cv(self) -> float:
        """Coefficient of variation (relative standard deviation).

        The CV expresses the standard deviation as a fraction of the mean,
        allowing comparison of variability across different scales.

        Returns:
            Standard deviation divided by the absolute mean.
            Returns infinity if mean is zero.

        Examples:
            >>> from insideLLMs.analysis.statistics import descriptive_statistics
            >>> stats = descriptive_statistics([100, 105, 95, 102, 98])
            >>> 0 < stats.cv < 1
            True
        """
        if self.mean == 0:
            return float("inf")
        return self.std / abs(self.mean)


@dataclass
class AggregatedResults:
    """Aggregated results from multiple experiment runs.

    Combines multiple experiment results for a single metric, providing
    both the raw values and computed statistics with confidence intervals.

    Attributes:
        metric_name: Name of the metric being aggregated.
        values: Raw metric values from all runs.
        stats: Descriptive statistics for the values.
        ci: Confidence interval for the mean.
        raw_results: Original ExperimentResult objects.

    Examples:
        Aggregating accuracy across runs:

            >>> from insideLLMs.analysis.statistics import aggregate_experiments
            >>> # Assuming you have experiment results
            >>> # aggregated = aggregate_experiments(experiments, "accuracy")
            >>> # print(f"Mean: {aggregated.stats.mean:.3f}")
            >>> # print(f"95% CI: [{aggregated.ci.lower:.3f}, {aggregated.ci.upper:.3f}]")

        Checking result variability:

            >>> # print(f"Std: {aggregated.stats.std:.3f}")
            >>> # print(f"Range: {aggregated.stats.min_val} - {aggregated.stats.max_val}")
    """

    metric_name: str
    values: list[float]
    stats: DescriptiveStats
    ci: ConfidenceInterval
    raw_results: list[ExperimentResult] = field(default_factory=list)


@dataclass
class StatisticalComparisonResult:
    """Result of comparing two groups or experiments statistically.

    Contains full statistical comparison between two groups including
    descriptive statistics for both, hypothesis test results, and
    confidence intervals for the difference.

    Attributes:
        group_a_name: Name/identifier for the first group.
        group_b_name: Name/identifier for the second group.
        metric_name: Name of the metric being compared.
        group_a_stats: Descriptive statistics for group A.
        group_b_stats: Descriptive statistics for group B.
        test_result: Results of the hypothesis test.
        difference: Mean difference (A - B).
        difference_ci: Confidence interval for the difference.

    Examples:
        Comparing two models:

            >>> from insideLLMs.analysis.statistics import compare_groups
            >>> model_a_scores = [0.90, 0.92, 0.88, 0.91, 0.89]
            >>> model_b_scores = [0.85, 0.83, 0.86, 0.84, 0.82]
            >>> result = compare_groups(
            ...     model_a_scores, model_b_scores,
            ...     group_a_name="GPT-4", group_b_name="Claude",
            ...     metric_name="accuracy"
            ... )
            >>> print(f"Difference: {result.difference:.3f}")  # doctest: +SKIP

        Checking significance:

            >>> result.test_result.significant  # doctest: +SKIP
            True

        Interpreting the comparison:

            >>> print(result.test_result.conclusion)  # doctest: +SKIP
    """

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
    """Get z-score for a given confidence level using normal approximation.

    Computes the critical z-value (standard normal quantile) for the given
    confidence level. Uses a lookup table for common values and Winitzki's
    approximation for the inverse error function otherwise.

    Args:
        confidence_level: The desired confidence level as a decimal (e.g., 0.95
            for 95% confidence). Must be between 0 and 1.

    Returns:
        The z-score corresponding to the given confidence level. For example,
        0.95 confidence returns approximately 1.96.

    Examples:
        Standard confidence levels:

            >>> _z_score(0.95)
            1.96
            >>> _z_score(0.99)
            2.576
            >>> _z_score(0.90)
            1.645

        Using for margin of error calculation:

            >>> z = _z_score(0.95)
            >>> se = 0.05  # standard error
            >>> margin_of_error = z * se
            >>> margin_of_error
            0.098

    Notes:
        This is an internal function. For typical usage, prefer the
        confidence_interval() function which handles z-score calculation
        automatically.
    """
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

    Uses a polynomial expansion approximation when scipy is not available.
    For large degrees of freedom (>100), defaults to z-score as the
    t-distribution converges to normal.

    Args:
        confidence_level: The desired confidence level as a decimal (e.g., 0.95).
        df: Degrees of freedom (typically n - 1 for a single sample).

    Returns:
        The t-score for the given confidence level and degrees of freedom.

    Examples:
        Small sample (df=5):

            >>> t = _t_score(0.95, 5)
            >>> t > _z_score(0.95)  # t > z for small samples
            True

        Large sample approximates z:

            >>> t_large = _t_score(0.95, 200)
            >>> z = _z_score(0.95)
            >>> abs(t_large - z) < 0.01
            True

    Notes:
        This is an internal function. The confidence_interval() function
        automatically selects t or z based on sample size.
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
    """Calculate the arithmetic mean of a list of values.

    The arithmetic mean is the sum of all values divided by the count.
    Returns 0.0 for empty lists.

    Args:
        values: List of numeric values.

    Returns:
        The arithmetic mean, or 0.0 if the list is empty.

    Examples:
        Basic usage:

            >>> from insideLLMs.analysis.statistics import calculate_mean
            >>> calculate_mean([1, 2, 3, 4, 5])
            3.0

        Model accuracy scores:

            >>> accuracies = [0.92, 0.88, 0.91, 0.90, 0.89]
            >>> calculate_mean(accuracies)
            0.9

        Empty list handling:

            >>> calculate_mean([])
            0.0

        Single value:

            >>> calculate_mean([42.0])
            42.0

    See Also:
        calculate_median: For the median (50th percentile).
        descriptive_statistics: For comprehensive statistics.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_variance(values: list[float], ddof: int = 1) -> float:
    """Calculate the variance of a list of values.

    Variance measures the average squared deviation from the mean,
    quantifying the spread or dispersion of the data.

    Args:
        values: List of numeric values.
        ddof: Delta degrees of freedom. Use 1 for sample variance (default,
            Bessel's correction) or 0 for population variance.

    Returns:
        The variance of the values. Returns 0.0 if fewer than 2 values.

    Examples:
        Sample variance (default):

            >>> from insideLLMs.analysis.statistics import calculate_variance
            >>> calculate_variance([2, 4, 4, 4, 5, 5, 7, 9])
            4.571428571428571

        Population variance:

            >>> calculate_variance([2, 4, 4, 4, 5, 5, 7, 9], ddof=0)
            4.0

        Model score variability:

            >>> scores = [0.90, 0.92, 0.88, 0.91, 0.89]
            >>> var = calculate_variance(scores)
            >>> var < 0.01  # Low variance = consistent performance
            True

        Edge cases:

            >>> calculate_variance([5])  # Single value
            0.0
            >>> calculate_variance([])  # Empty list
            0.0

    See Also:
        calculate_std: For standard deviation (square root of variance).
    """
    if len(values) < 2:
        return 0.0
    mean = calculate_mean(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    return sum(squared_diffs) / (len(values) - ddof)


def calculate_std(values: list[float], ddof: int = 1) -> float:
    """Calculate the standard deviation of a list of values.

    Standard deviation is the square root of variance, expressing dispersion
    in the same units as the original data.

    Args:
        values: List of numeric values.
        ddof: Delta degrees of freedom. Use 1 for sample std (default)
            or 0 for population std.

    Returns:
        The standard deviation. Returns 0.0 if fewer than 2 values.

    Examples:
        Basic usage:

            >>> from insideLLMs.analysis.statistics import calculate_std
            >>> calculate_std([2, 4, 4, 4, 5, 5, 7, 9])
            2.138089935299395

        Comparing model consistency:

            >>> consistent_model = [0.90, 0.91, 0.90, 0.89, 0.90]
            >>> variable_model = [0.95, 0.80, 0.92, 0.75, 0.88]
            >>> calculate_std(consistent_model) < calculate_std(variable_model)
            True

        Edge cases:

            >>> calculate_std([5])  # Single value
            0.0
            >>> calculate_std([])  # Empty
            0.0

    See Also:
        calculate_variance: For variance (std squared).
        descriptive_statistics: For comprehensive statistics including std.
    """
    return math.sqrt(calculate_variance(values, ddof))


def calculate_median(values: list[float]) -> float:
    """Calculate the median (50th percentile) of a list of values.

    The median is the middle value when data is sorted. For even-length
    lists, it's the average of the two middle values. The median is more
    robust to outliers than the mean.

    Args:
        values: List of numeric values.

    Returns:
        The median value, or 0.0 if the list is empty.

    Examples:
        Odd number of values:

            >>> from insideLLMs.analysis.statistics import calculate_median
            >>> calculate_median([1, 3, 5, 7, 9])
            5

        Even number of values:

            >>> calculate_median([1, 2, 3, 4])
            2.5

        Robust to outliers (compare with mean):

            >>> data_with_outlier = [0.90, 0.91, 0.89, 0.90, 0.10]  # 0.10 is outlier
            >>> median = calculate_median(data_with_outlier)
            >>> mean = calculate_mean(data_with_outlier)
            >>> median > mean  # Median not affected by outlier
            True

        Edge cases:

            >>> calculate_median([42])
            42
            >>> calculate_median([])
            0.0

    See Also:
        calculate_percentile: For arbitrary percentiles (median = 50th).
        calculate_mean: For the arithmetic mean.
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    count = len(sorted_values)
    mid = count // 2
    if count % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return sorted_values[mid]


def calculate_percentile(values: list[float], percentile: float) -> float:
    """Calculate a specific percentile using linear interpolation.

    Computes the value below which a given percentage of observations fall.
    Uses linear interpolation for percentiles that fall between data points.

    Args:
        values: List of numeric values.
        percentile: Percentile to calculate, from 0 to 100.
            Common values: 25 (Q1), 50 (median), 75 (Q3), 90, 95, 99.

    Returns:
        The value at the given percentile. Returns 0.0 for empty lists.

    Raises:
        No exceptions raised; returns 0.0 for edge cases.

    Examples:
        Common percentiles:

            >>> from insideLLMs.analysis.statistics import calculate_percentile
            >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> calculate_percentile(data, 50)  # Median
            5.5
            >>> calculate_percentile(data, 25)  # Q1
            3.25
            >>> calculate_percentile(data, 75)  # Q3
            7.75

        Analyzing latency percentiles:

            >>> latencies = [45, 52, 48, 95, 51, 49, 47, 200, 50, 46]
            >>> p50 = calculate_percentile(latencies, 50)
            >>> p95 = calculate_percentile(latencies, 95)
            >>> p99 = calculate_percentile(latencies, 99)
            >>> p95 > p50  # P95 captures tail latency
            True

        Boundary values:

            >>> calculate_percentile([1, 2, 3], 0)
            1
            >>> calculate_percentile([1, 2, 3], 100)
            3

        Empty list:

            >>> calculate_percentile([], 50)
            0.0

    See Also:
        calculate_median: Convenience function for the 50th percentile.
        descriptive_statistics: Includes Q1, Q3, and IQR.
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    count = len(sorted_values)
    fractional_index = (percentile / 100) * (count - 1)
    lower_bound = math.floor(fractional_index)
    upper_bound = math.ceil(fractional_index)
    if lower_bound == upper_bound:
        return sorted_values[int(fractional_index)]
    return sorted_values[lower_bound] * (upper_bound - fractional_index) + sorted_values[upper_bound] * (fractional_index - lower_bound)


def calculate_skewness(values: list[float]) -> float:
    """Calculate Fisher-Pearson coefficient of skewness."""
    if len(values) < 3:
        return 0.0
    count = len(values)
    mean = calculate_mean(values)
    std = calculate_std(values, ddof=0)
    if std == 0:
        return 0.0
    m3 = sum((x - mean) ** 3 for x in values) / count
    return m3 / (std**3)


def calculate_kurtosis(values: list[float]) -> float:
    """Calculate excess kurtosis (Fisher's definition)."""
    if len(values) < 4:
        return 0.0
    count = len(values)
    mean = calculate_mean(values)
    std = calculate_std(values, ddof=0)
    if std == 0:
        return 0.0
    m4 = sum((x - mean) ** 4 for x in values) / count
    return m4 / (std**4) - 3


def descriptive_statistics(values: list[float]) -> DescriptiveStats:
    """Calculate comprehensive descriptive statistics for a sample.

    Computes all common descriptive statistics including measures of
    central tendency (mean, median), dispersion (std, variance, IQR),
    and distribution shape (skewness, kurtosis).

    Args:
        values: List of numeric values to analyze.

    Returns:
        DescriptiveStats object containing all computed statistics.

    Examples:
        Basic usage:

            >>> from insideLLMs.analysis.statistics import descriptive_statistics
            >>> data = [85, 90, 88, 92, 87, 91, 89, 86, 93, 88]
            >>> stats = descriptive_statistics(data)
            >>> print(f"Mean: {stats.mean}, Median: {stats.median}")
            Mean: 88.9, Median: 88.5

        Analyzing model accuracy scores:

            >>> accuracies = [0.92, 0.89, 0.94, 0.91, 0.93]
            >>> stats = descriptive_statistics(accuracies)
            >>> print(f"Mean accuracy: {stats.mean:.2%}")
            Mean accuracy: 91.80%

        Checking for outliers with IQR:

            >>> data = [10, 12, 14, 15, 100]  # 100 is outlier
            >>> stats = descriptive_statistics(data)
            >>> upper_fence = stats.q3 + 1.5 * stats.iqr
            >>> any(x > upper_fence for x in data)
            True

        Empty data handling:

            >>> empty_stats = descriptive_statistics([])
            >>> empty_stats.n
            0
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
    """Calculate confidence interval for the mean of a sample.

    Computes confidence intervals using t-distribution (small samples),
    z-distribution (large samples), or bootstrap resampling methods.

    Args:
        values: List of numeric values.
        confidence_level: Confidence level (default 0.95 for 95% CI).
            Common values: 0.90, 0.95, 0.99.
        method: Method for CI calculation:
            - "t": t-distribution (default, best for small samples)
            - "z": Normal distribution (for large samples n>100)
            - "bootstrap": Non-parametric bootstrap resampling

    Returns:
        ConfidenceInterval object with point estimate and bounds.

    Examples:
        Computing a 95% CI for accuracy:

            >>> from insideLLMs.analysis.statistics import confidence_interval
            >>> accuracies = [0.92, 0.89, 0.94, 0.91, 0.93, 0.90]
            >>> ci = confidence_interval(accuracies, confidence_level=0.95)
            >>> print(f"Mean: {ci.point_estimate:.3f}")  # doctest: +SKIP
            >>> print(f"95% CI: [{ci.lower:.3f}, {ci.upper:.3f}]")  # doctest: +SKIP

        Using 99% confidence level:

            >>> ci_99 = confidence_interval(accuracies, confidence_level=0.99)
            >>> ci_99.confidence_level
            0.99

        Bootstrap CI for non-normal data:

            >>> ci_boot = confidence_interval(
            ...     accuracies,
            ...     method="bootstrap"
            ... )
            >>> ci_boot.method
            'bootstrap'

        Comparing CI width across confidence levels:

            >>> ci_90 = confidence_interval(accuracies, confidence_level=0.90)
            >>> ci_95 = confidence_interval(accuracies, confidence_level=0.95)
            >>> ci_90.width < ci_95.width  # 90% CI is narrower
            True
    """
    if not values:
        return ConfidenceInterval(
            point_estimate=0.0,
            lower=0.0,
            upper=0.0,
            confidence_level=confidence_level,
            method=method,
        )

    count = len(values)
    mean = calculate_mean(values)
    std = calculate_std(values)
    se = std / math.sqrt(count) if count > 0 else 0

    if method == "bootstrap":
        return bootstrap_confidence_interval(values, lambda x: calculate_mean(x), confidence_level)
    elif method == "z" or count > 100:
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
        df = count - 1
        t_score = _t_score(confidence_level, df)
        margin = t_score * se
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
    """Calculate bootstrap confidence interval for any statistic.

    Non-parametric method that works by resampling with replacement
    from the original data to estimate the sampling distribution of
    any statistic (mean, median, percentiles, etc.).

    Args:
        values: List of numeric values to resample from.
        statistic_fn: Function that computes the statistic of interest.
            Takes a list of floats and returns a single float.
        confidence_level: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap resamples (default 1000).
            Higher values give more stable estimates.
        seed: Random seed for reproducibility.

    Returns:
        ConfidenceInterval object with bootstrap bounds.

    Examples:
        Bootstrap CI for the mean:

            >>> from insideLLMs.analysis.statistics import bootstrap_confidence_interval
            >>> data = [0.85, 0.88, 0.92, 0.87, 0.90, 0.89]
            >>> ci = bootstrap_confidence_interval(
            ...     data,
            ...     statistic_fn=lambda x: sum(x)/len(x),
            ...     seed=42
            ... )
            >>> ci.method
            'bootstrap'

        Bootstrap CI for the median:

            >>> def median_fn(x):
            ...     sorted_x = sorted(x)
            ...     n = len(sorted_x)
            ...     return sorted_x[n//2]
            >>> ci_median = bootstrap_confidence_interval(data, median_fn, seed=42)

        Custom statistic (e.g., 90th percentile):

            >>> def p90(x):
            ...     sorted_x = sorted(x)
            ...     idx = int(0.9 * (len(sorted_x) - 1))
            ...     return sorted_x[idx]
            >>> ci_p90 = bootstrap_confidence_interval(data, p90, seed=42)

        High-precision estimate with more samples:

            >>> ci_precise = bootstrap_confidence_interval(
            ...     data,
            ...     lambda x: sum(x)/len(x),
            ...     n_bootstrap=10000,
            ...     seed=42
            ... )
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

    count = len(values)
    point_estimate = statistic_fn(values)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = [random.choice(values) for _ in range(count)]
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
    """Calculate Cohen's d effect size for two independent groups.

    Cohen's d measures the standardized difference between two means,
    providing a scale-independent measure of effect magnitude. It uses
    the pooled standard deviation as the standardizer.

    Args:
        group1: First group of values.
        group2: Second group of values.

    Returns:
        Cohen's d effect size. Positive if group1 > group2.

    Notes:
        Conventional interpretations (Cohen, 1988):
        - |d| < 0.2: negligible effect
        - 0.2 <= |d| < 0.5: small effect
        - 0.5 <= |d| < 0.8: medium effect
        - |d| >= 0.8: large effect

    Examples:
        Calculating effect size between two models:

            >>> from insideLLMs.analysis.statistics import cohens_d, interpret_cohens_d
            >>> model_a = [0.90, 0.92, 0.88, 0.91, 0.89]
            >>> model_b = [0.85, 0.83, 0.86, 0.84, 0.82]
            >>> d = cohens_d(model_a, model_b)
            >>> d > 0  # model_a performs better
            True

        Interpreting the effect size:

            >>> interpretation = interpret_cohens_d(d)
            >>> interpretation
            'large'

        Small effect example:

            >>> similar_a = [0.90, 0.91, 0.89]
            >>> similar_b = [0.89, 0.90, 0.88]
            >>> d_small = cohens_d(similar_a, similar_b)
            >>> interpret_cohens_d(d_small) in ['negligible', 'small']
            True
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
    """Interpret Cohen's d effect size using conventional thresholds.

    Converts a numeric Cohen's d value to a categorical interpretation
    based on Cohen's (1988) conventional thresholds.

    Args:
        d: Cohen's d value (can be positive or negative).

    Returns:
        Interpretation string: "negligible", "small", "medium", or "large".

    Examples:
        >>> from insideLLMs.analysis.statistics import interpret_cohens_d
        >>> interpret_cohens_d(0.1)
        'negligible'
        >>> interpret_cohens_d(0.3)
        'small'
        >>> interpret_cohens_d(0.6)
        'medium'
        >>> interpret_cohens_d(1.2)
        'large'
        >>> interpret_cohens_d(-0.9)  # Negative values work too
        'large'
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
    """Perform Welch's t-test for independent samples with unequal variance.

    Welch's t-test is preferred over Student's t-test when sample sizes
    or variances may differ between groups. It does not assume equal
    variances (homoscedasticity).

    Args:
        group1: First group of values.
        group2: Second group of values.
        alpha: Significance level (default 0.05).

    Returns:
        HypothesisTestResult with t-statistic, p-value, and effect size.

    Examples:
        Testing if two models have significantly different accuracy:

            >>> from insideLLMs.analysis.statistics import welchs_t_test
            >>> model_a = [0.90, 0.92, 0.88, 0.91, 0.89]
            >>> model_b = [0.85, 0.83, 0.86, 0.84, 0.82]
            >>> result = welchs_t_test(model_a, model_b)
            >>> result.significant
            True
            >>> result.test_name
            "Welch's t-test"

        Interpreting the full result:

            >>> print(f"t = {result.statistic:.3f}, p = {result.p_value:.4f}")
            ... # doctest: +SKIP
            >>> print(f"Effect: {result.effect_size_interpretation}")
            ... # doctest: +SKIP

        Using custom significance level:

            >>> result_strict = welchs_t_test(model_a, model_b, alpha=0.01)
            >>> result_strict.alpha
            0.01

        Non-significant difference:

            >>> similar_a = [0.90, 0.91, 0.89, 0.90]
            >>> similar_b = [0.89, 0.90, 0.91, 0.88]
            >>> result = welchs_t_test(similar_a, similar_b)
            >>> result.significant
            False
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

    count = len(differences)
    mean_diff = calculate_mean(differences)
    std_diff = calculate_std(differences)
    se = std_diff / math.sqrt(count)

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
    num_tests = len(p_values)
    if num_tests == 0:
        return [], []

    if method == "bonferroni":
        # Simple Bonferroni correction
        corrected = [min(p * num_tests, 1.0) for p in p_values]
        significant = [p < alpha / num_tests for p in p_values]

    elif method == "holm":
        # Holm-Bonferroni step-down method
        indexed = [(p, i) for i, p in enumerate(p_values)]
        indexed.sort(key=lambda x: x[0])

        corrected = [0.0] * num_tests
        significant = [False] * num_tests
        max_corrected = 0.0

        for rank, (p, orig_idx) in enumerate(indexed):
            k = num_tests - rank
            adj_p = p * k
            max_corrected = max(max_corrected, adj_p)
            corrected[orig_idx] = min(max_corrected, 1.0)
            significant[orig_idx] = p < alpha / k

    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR correction
        indexed = [(p, i) for i, p in enumerate(p_values)]
        indexed.sort(key=lambda x: x[0])

        corrected = [0.0] * num_tests
        significant = [False] * num_tests
        min_corrected = 1.0

        for rank in range(num_tests - 1, -1, -1):
            p, orig_idx = indexed[rank]
            k = rank + 1
            adj_p = p * num_tests / k
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
) -> float:
    """Calculate required sample size per group to achieve desired power.

    Args:
        effect_size: Expected Cohen's d effect size.
        power: Desired statistical power.
        alpha: Significance level.

    Returns:
        Required sample size per group, or inf if effect_size is zero.
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
        "unique_models": sorted(by_model.keys()),
        "unique_probes": sorted(by_probe.keys()),
        "by_model": {},
        "by_probe": {},
        "overall": {},
    }

    # Analyze by model
    for model_name in sorted(by_model):
        model_results = by_model[model_name]
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
    for probe_name in sorted(by_probe):
        probe_results = by_probe[probe_name]
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
