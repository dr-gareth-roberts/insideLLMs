"""Compatibility shim for insideLLMs.analysis.statistics.

This module provides backward-compatible access to statistical analysis
utilities that have been reorganized into the ``insideLLMs.analysis.statistics``
submodule. All functionality is re-exported from the canonical location.

Module Overview
---------------
This shim module exists to maintain backward compatibility with code that
imports directly from ``insideLLMs.statistics``. New code should prefer
importing from ``insideLLMs.analysis.statistics`` directly.

The module re-exports all public symbols from the statistics module including:

**Data Classes**:
    - ConfidenceInterval: Confidence interval with bounds and methods
    - HypothesisTestResult: Result of statistical hypothesis testing
    - DescriptiveStats: Comprehensive descriptive statistics
    - AggregatedResults: Aggregated results with statistics and CI
    - StatisticalComparisonResult: Full statistical comparison of groups

**Core Statistical Functions**:
    - descriptive_statistics: Compute comprehensive stats for a sample
    - confidence_interval: Calculate CI for the mean
    - bootstrap_confidence_interval: Non-parametric bootstrap CI
    - calculate_mean, calculate_std, calculate_variance: Basic statistics
    - calculate_median, calculate_percentile: Quantile functions
    - calculate_skewness, calculate_kurtosis: Distribution shape

**Hypothesis Testing**:
    - welchs_t_test: Independent samples t-test (unequal variance)
    - paired_t_test: Paired/dependent samples t-test
    - mann_whitney_u: Non-parametric rank-based test

**Effect Size and Power**:
    - cohens_d: Effect size for group comparison
    - interpret_cohens_d: Text interpretation of effect size
    - power_analysis: Calculate statistical power
    - required_sample_size: Compute sample size for desired power

**Experiment Analysis**:
    - extract_metric_from_results: Extract metrics from experiments
    - aggregate_experiment_results: Aggregate multiple runs
    - compare_experiments: Compare two experiment sets
    - generate_summary_report: Create comprehensive report

Quick Start Examples
--------------------
Basic descriptive statistics (via compatibility import):

    >>> from insideLLMs.statistics import descriptive_statistics
    >>> scores = [0.85, 0.88, 0.92, 0.87, 0.90]
    >>> stats = descriptive_statistics(scores)
    >>> print(f"Mean: {stats.mean:.3f}, Std: {stats.std:.3f}")
    Mean: 0.884, Std: 0.026

Computing confidence intervals:

    >>> from insideLLMs.statistics import confidence_interval
    >>> ci = confidence_interval(scores, confidence_level=0.95)
    >>> print(f"95% CI: [{ci.lower:.3f}, {ci.upper:.3f}]")  # doctest: +SKIP

Hypothesis testing:

    >>> from insideLLMs.statistics import welchs_t_test
    >>> group_a = [0.90, 0.92, 0.88, 0.91, 0.89]
    >>> group_b = [0.85, 0.83, 0.86, 0.84, 0.82]
    >>> result = welchs_t_test(group_a, group_b)
    >>> print(f"Significant: {result.significant}")  # doctest: +SKIP

Migration Guide
---------------
To migrate to the new import location, simply change:

    # Old (still works, but deprecated)
    from insideLLMs.statistics import descriptive_statistics

    # New (recommended)
    from insideLLMs.analysis.statistics import descriptive_statistics

Both import paths provide identical functionality.

Notes
-----
- This module uses wildcard imports to ensure all public symbols are available
- The noqa comments suppress linter warnings about wildcard imports
- Performance is identical to importing from the canonical location

See Also
--------
insideLLMs.analysis.statistics : The canonical location for statistical utilities
insideLLMs.analysis.comparison : Model comparison utilities
insideLLMs.analysis.evaluation : Evaluation metrics and scoring

Deprecated Since
----------------
Version 0.5.0: Direct imports from this module are deprecated. Use
``insideLLMs.analysis.statistics`` instead for new code.
"""

from insideLLMs.analysis.statistics import *  # noqa: F401,F403
