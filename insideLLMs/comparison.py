"""Compatibility shim for backward-compatible imports from insideLLMs.comparison.

This module provides backward compatibility for code that imports from the
legacy ``insideLLMs.comparison`` path. All functionality has been moved to
``insideLLMs.analysis.comparison``, which is the canonical import location.

Module Overview
---------------
This shim re-exports all public symbols from ``insideLLMs.analysis.comparison``
to support existing code that uses the old import path. New code should import
directly from the canonical location.

Re-exported Classes
-------------------
The following classes are available through this compatibility shim:

- **ComparisonMetric**: Enum of standard metrics (accuracy, latency, etc.)
- **MetricValue**: Single metric measurement with timestamp and metadata
- **MetricSummary**: Summary statistics for a metric (mean, std, percentiles)
- **ModelProfile**: Complete performance profile for an LLM model
- **ModelComparisonResult**: Result of comparing multiple models
- **ModelComparator**: Main comparison engine for ranking models
- **LatencyProfile**: Detailed latency breakdown (first token, throughput)
- **CostEstimate**: Cost calculation for token usage
- **ModelCostComparator**: Compare costs across models with pricing data
- **QualityMetrics**: Quality scores for model outputs
- **PerformanceTracker**: Track metrics during experiment execution

Re-exported Functions
---------------------
- **create_comparison_table**: Create markdown comparison table from profiles
- **rank_models**: Rank models by a specific metric

Migration Guide
---------------
To migrate from the legacy import path to the canonical location:

**Before (deprecated):**

    >>> from insideLLMs.comparison import ModelComparator, ModelProfile

**After (recommended):**

    >>> from insideLLMs.analysis.comparison import ModelComparator, ModelProfile

Both import paths will work, but the canonical path under ``insideLLMs.analysis``
is preferred for new code and provides better IDE support and documentation.

Examples
--------
Basic model comparison using this compatibility shim:

    >>> from insideLLMs.comparison import ModelComparator, ModelProfile
    >>>
    >>> # Create profiles for two models
    >>> profile_a = ModelProfile(model_name="gpt-4", model_id="gpt-4-turbo")
    >>> profile_a.add_metric("accuracy", [0.92, 0.94, 0.91, 0.93], unit="%")
    >>> profile_a.add_metric("latency", [150.0, 145.0, 160.0, 155.0], unit="ms")
    >>>
    >>> profile_b = ModelProfile(model_name="claude-3", model_id="claude-3-opus")
    >>> profile_b.add_metric("accuracy", [0.95, 0.93, 0.94, 0.96], unit="%")
    >>> profile_b.add_metric("latency", [120.0, 115.0, 125.0, 118.0], unit="ms")
    >>>
    >>> # Compare models
    >>> comparator = ModelComparator()
    >>> comparator.add_profile(profile_a).add_profile(profile_b)  # doctest: +ELLIPSIS
    <insideLLMs.analysis.comparison.ModelComparator object at ...>
    >>> result = comparator.compare()
    >>> print(result.winner)
    claude-3

Using the PerformanceTracker:

    >>> from insideLLMs.comparison import PerformanceTracker
    >>> tracker = PerformanceTracker("gpt-4")
    >>> tracker.record_latency(150.5)
    >>> tracker.record_latency(145.2)
    >>> tracker.record_success(True)
    >>> tracker.record_tokens(100, 50)
    >>> summary = tracker.get_summary()
    >>> "latency" in summary.metrics
    True

Cost comparison across models:

    >>> from insideLLMs.comparison import ModelCostComparator
    >>> calc = ModelCostComparator()
    >>> calc.set_pricing("custom-model", input_per_1k=0.02, output_per_1k=0.04)
    ... # doctest: +ELLIPSIS
    <insideLLMs.analysis.comparison.ModelCostComparator object at ...>
    >>> cost = calc.estimate("custom-model", input_tokens=1000, output_tokens=500)
    >>> cost.total_cost
    0.04

Creating comparison tables:

    >>> from insideLLMs.comparison import create_comparison_table, ModelProfile
    >>> profiles = [
    ...     ModelProfile(model_name="model-a"),
    ...     ModelProfile(model_name="model-b"),
    ... ]
    >>> profiles[0].add_metric("accuracy", [0.90, 0.92])
    >>> profiles[1].add_metric("accuracy", [0.95, 0.94])
    >>> table = create_comparison_table(profiles)
    >>> "model-a" in table and "model-b" in table
    True

Notes
-----
- This module exists solely for backward compatibility
- No new functionality should be added to this module
- All bug fixes and features should be implemented in the canonical module
- This shim may be deprecated in a future major version release

See Also
--------
insideLLMs.analysis.comparison : Canonical module with full implementation
insideLLMs.analysis.statistics : Statistical analysis functions
insideLLMs.analysis.evaluation : Evaluation metrics and evaluators

.. deprecated::
    Direct imports from ``insideLLMs.comparison`` are deprecated.
    Use ``insideLLMs.analysis.comparison`` instead.
"""

from insideLLMs.analysis.comparison import *  # noqa: F401,F403
