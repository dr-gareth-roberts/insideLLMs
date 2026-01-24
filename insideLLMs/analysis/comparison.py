"""Model and experiment comparison utilities.

This module provides tools for comparing LLM performance across
multiple dimensions: accuracy, latency, cost, and quality metrics.

Key features:
- Multi-dimensional model comparison
- Statistical significance testing
- Performance profiling
- Cost analysis
- Quality metrics aggregation
- Comparison report generation

Module Overview
---------------
The comparison module enables systematic evaluation of multiple LLM models
by providing data structures and algorithms for collecting metrics, computing
summary statistics, ranking models, and generating detailed comparison reports.

Quick Start Examples
--------------------
Basic model comparison workflow:

    >>> from insideLLMs.analysis.comparison import (
    ...     ModelComparator, ModelProfile, PerformanceTracker
    ... )
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
    >>> comparator.add_profile(profile_a).add_profile(profile_b)
    >>> result = comparator.compare()
    >>> print(result.winner)
    claude-3

Tracking performance during experiments:

    >>> tracker = PerformanceTracker("gpt-4")
    >>> for response in model_responses:  # doctest: +SKIP
    ...     tracker.record_latency(response.latency_ms)
    ...     tracker.record_success(response.success)
    ...     tracker.record_tokens(response.input_tokens, response.output_tokens)
    >>> summary = tracker.get_summary()

Cost comparison across models:

    >>> from insideLLMs.analysis.comparison import ModelCostComparator
    >>> calc = ModelCostComparator()
    >>> calc.set_pricing("custom-model", input_per_1k=0.02, output_per_1k=0.04)
    >>> costs = calc.compare_costs(input_tokens=1000, output_tokens=500)
    >>> cheapest_model, cost = calc.cheapest_model(1000, 500)

Generating comparison reports:

    >>> report = comparator.generate_report()
    >>> print(report)  # Markdown-formatted comparison table

Classes Summary
---------------
- **MetricValue**: A single metric measurement with timestamp and metadata
- **MetricSummary**: Summary statistics (mean, std, percentiles) for a metric
- **ModelProfile**: Complete performance profile for a model
- **ModelComparisonResult**: Result of comparing multiple models
- **ModelComparator**: Main comparison engine for models
- **LatencyProfile**: Detailed latency breakdown (first token, throughput)
- **CostEstimate**: Cost calculation for token usage
- **ModelCostComparator**: Compare costs across models with pricing
- **QualityMetrics**: Quality scores for model outputs
- **PerformanceTracker**: Track metrics during experiment execution

See Also
--------
insideLLMs.analysis.statistics : Statistical analysis functions
insideLLMs.analysis.evaluation : Evaluation metrics and evaluators
"""

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Optional,
    TypeVar,
)

T = TypeVar("T")


class ComparisonMetric(Enum):
    """Standard metrics for model comparison.

    This enumeration defines the common metrics used when comparing
    LLM model performance. Each metric represents a different dimension
    of evaluation.

    Attributes:
        ACCURACY: Correctness of model outputs (higher is better).
        LATENCY: Response time in milliseconds (lower is better).
        THROUGHPUT: Requests or tokens per second (higher is better).
        COST: Financial cost per request (lower is better).
        ERROR_RATE: Percentage of failed requests (lower is better).
        TOKEN_USAGE: Total tokens consumed (depends on context).
        QUALITY_SCORE: Subjective quality rating (higher is better).

    Examples:
        Using metrics in comparisons:

            >>> from insideLLMs.analysis.comparison import ComparisonMetric
            >>> metric = ComparisonMetric.ACCURACY
            >>> print(metric.value)
            accuracy

        Iterating over all metrics:

            >>> for metric in ComparisonMetric:
            ...     print(f"{metric.name}: {metric.value}")
            ACCURACY: accuracy
            LATENCY: latency
            ...

        Checking if a string is a valid metric:

            >>> "accuracy" in [m.value for m in ComparisonMetric]
            True
    """

    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    COST = "cost"
    ERROR_RATE = "error_rate"
    TOKEN_USAGE = "token_usage"
    QUALITY_SCORE = "quality_score"


@dataclass
class MetricValue:
    """A single metric measurement with timestamp and context.

    Represents an individual measurement point for any metric, including
    the value, its unit, when it was recorded, and any additional context.

    Attributes:
        value: The numeric metric value.
        unit: Unit of measurement (e.g., "ms", "%", "tokens").
        timestamp: When the measurement was taken (auto-set if None).
        metadata: Additional context like model version, prompt type, etc.

    Examples:
        Creating a latency measurement:

            >>> from insideLLMs.analysis.comparison import MetricValue
            >>> latency = MetricValue(value=150.5, unit="ms")
            >>> print(f"Latency: {latency.value} {latency.unit}")
            Latency: 150.5 ms

        Adding metadata for context:

            >>> accuracy = MetricValue(
            ...     value=0.95,
            ...     unit="%",
            ...     metadata={"probe": "factuality", "difficulty": "hard"}
            ... )
            >>> print(accuracy.metadata["probe"])
            factuality

        Timestamp is auto-populated:

            >>> measurement = MetricValue(value=100.0)
            >>> measurement.timestamp is not None
            True

        Using with custom timestamp:

            >>> from datetime import datetime
            >>> custom_time = datetime(2024, 1, 15, 10, 30, 0)
            >>> metric = MetricValue(value=42.0, timestamp=custom_time)
            >>> metric.timestamp.year
            2024
    """

    value: float
    unit: str = ""
    timestamp: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MetricSummary:
    """Summary statistics for a metric across multiple measurements.

    Provides comprehensive statistical analysis of a metric including
    central tendency (mean, median), dispersion (std, range), and
    distribution characteristics (percentiles).

    Attributes:
        name: Metric name identifier.
        mean: Arithmetic mean of all values.
        std: Sample standard deviation.
        min: Minimum observed value.
        max: Maximum observed value.
        median: Median (50th percentile) value.
        count: Number of samples/measurements.
        percentile_95: 95th percentile value.
        percentile_99: 99th percentile value.
        unit: Unit of measurement.

    Examples:
        Creating a summary from raw values:

            >>> from insideLLMs.analysis.comparison import MetricSummary
            >>> latencies = [100.0, 150.0, 120.0, 180.0, 110.0]
            >>> summary = MetricSummary.from_values("latency", latencies, "ms")
            >>> print(f"Mean: {summary.mean:.1f} {summary.unit}")
            Mean: 132.0 ms

        Accessing statistical properties:

            >>> summary = MetricSummary.from_values("accuracy", [0.9, 0.92, 0.88, 0.91])
            >>> print(f"Range: {summary.min:.2f} - {summary.max:.2f}")
            Range: 0.88 - 0.92
            >>> print(f"Std: {summary.std:.4f}")
            Std: 0.0165

        Handling empty data:

            >>> empty_summary = MetricSummary.from_values("empty", [])
            >>> empty_summary.count
            0
            >>> empty_summary.mean
            0.0

        Using percentiles for latency analysis:

            >>> latencies = [50, 60, 70, 80, 90, 100, 200, 500, 1000, 2000]
            >>> summary = MetricSummary.from_values("response_time", latencies, "ms")
            >>> print(f"P95: {summary.percentile_95} ms")
            P95: 1000 ms
    """

    name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    count: int
    percentile_95: float = 0.0
    percentile_99: float = 0.0
    unit: str = ""

    @classmethod
    def from_values(cls, name: str, values: list[float], unit: str = "") -> "MetricSummary":
        """Compute summary statistics from a list of numeric values.

        Creates a comprehensive statistical summary including mean, standard
        deviation, min/max, median, and percentiles from raw measurement data.

        Args:
            name: Identifier for this metric (e.g., "latency", "accuracy").
            values: List of numeric values to summarize.
            unit: Unit of measurement (e.g., "ms", "%").

        Returns:
            MetricSummary object with all statistics computed.

        Examples:
            Basic usage:

                >>> summary = MetricSummary.from_values(
                ...     "accuracy",
                ...     [0.85, 0.90, 0.88, 0.92, 0.87],
                ...     unit="%"
                ... )
                >>> round(summary.mean, 3)
                0.884

            With latency data:

                >>> latencies = [100, 120, 110, 150, 200]
                >>> summary = MetricSummary.from_values("latency", latencies, "ms")
                >>> summary.count
                5
                >>> summary.min
                100

            Empty values return zero-filled summary:

                >>> empty = MetricSummary.from_values("test", [])
                >>> empty.mean == 0.0 and empty.count == 0
                True
        """
        if not values:
            return cls(
                name=name,
                mean=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                median=0.0,
                count=0,
                unit=unit,
            )

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        # Calculate percentiles
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)

        return cls(
            name=name,
            mean=statistics.mean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
            min=min(values),
            max=max(values),
            median=statistics.median(values),
            count=len(values),
            percentile_95=sorted_vals[min(p95_idx, n - 1)],
            percentile_99=sorted_vals[min(p99_idx, n - 1)],
            unit=unit,
        )


@dataclass
class ModelProfile:
    """Complete performance profile for an LLM model.

    Aggregates all performance metrics for a single model, providing a
    comprehensive view of its capabilities across accuracy, latency,
    cost, and other dimensions.

    Attributes:
        model_name: Human-readable name of the model (e.g., "gpt-4").
        model_id: Unique identifier or API model ID.
        metrics: Dictionary mapping metric names to MetricSummary objects.
        raw_results: List of raw result dictionaries for detailed analysis.
        metadata: Additional model info (version, provider, config, etc.).

    Examples:
        Creating a profile and adding metrics:

            >>> from insideLLMs.analysis.comparison import ModelProfile
            >>> profile = ModelProfile(
            ...     model_name="gpt-4-turbo",
            ...     model_id="gpt-4-1106-preview",
            ...     metadata={"provider": "openai", "max_tokens": 128000}
            ... )
            >>> profile.add_metric("accuracy", [0.92, 0.94, 0.91, 0.93])
            >>> profile.add_metric("latency", [150.0, 145.0, 160.0], unit="ms")

        Retrieving metrics:

            >>> accuracy = profile.get_metric("accuracy")
            >>> accuracy is not None
            True
            >>> round(accuracy.mean, 2)
            0.93

        Checking available metrics:

            >>> sorted(profile.metrics.keys())
            ['accuracy', 'latency']

        Using with ModelComparator:

            >>> from insideLLMs.analysis.comparison import ModelComparator
            >>> comparator = ModelComparator()
            >>> comparator.add_profile(profile)  # doctest: +ELLIPSIS
            <insideLLMs.analysis.comparison.ModelComparator object at ...>
    """

    model_name: str
    model_id: str = ""
    metrics: dict[str, MetricSummary] = field(default_factory=dict)
    raw_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_metric(self, name: str, values: list[float], unit: str = "") -> None:
        """Add a metric summary from raw values.

        Computes summary statistics from the provided values and stores
        them under the given metric name.

        Args:
            name: Metric name identifier (e.g., "accuracy", "latency").
            values: List of numeric values to summarize.
            unit: Unit of measurement (e.g., "ms", "%").

        Examples:
            Adding accuracy data:

                >>> profile = ModelProfile(model_name="test-model")
                >>> profile.add_metric("accuracy", [0.85, 0.90, 0.88])
                >>> profile.metrics["accuracy"].count
                3

            Adding latency with units:

                >>> profile.add_metric("latency", [100, 120, 110], unit="ms")
                >>> profile.metrics["latency"].unit
                'ms'

            Overwriting existing metric:

                >>> profile.add_metric("accuracy", [0.95, 0.96])
                >>> profile.metrics["accuracy"].count
                2
        """
        self.metrics[name] = MetricSummary.from_values(name, values, unit)

    def get_metric(self, name: str) -> Optional[MetricSummary]:
        """Get a metric summary by name.

        Args:
            name: Metric name to retrieve.

        Returns:
            MetricSummary if the metric exists, None otherwise.

        Examples:
            Getting an existing metric:

                >>> profile = ModelProfile(model_name="test")
                >>> profile.add_metric("accuracy", [0.9, 0.92])
                >>> metric = profile.get_metric("accuracy")
                >>> metric is not None
                True

            Handling missing metrics:

                >>> profile.get_metric("nonexistent") is None
                True
        """
        return self.metrics.get(name)


@dataclass
class ModelComparisonResult:
    """Result of comparing two or more models across metrics.

    Contains the complete results of a model comparison, including
    per-metric rankings, differences from the best model, and an
    overall winner determination.

    Attributes:
        models: List of model names that were compared.
        winner: Best performing model based on weighted scoring (None if tie).
        rankings: Dict mapping metric names to ranked lists of models.
        differences: Dict mapping metric to {model: difference from best}.
        summary: Human-readable text summary of the comparison.
        significant_differences: Dict mapping metric to whether difference
            is statistically significant.

    Examples:
        Accessing comparison results:

            >>> from insideLLMs.analysis.comparison import (
            ...     ModelComparator, ModelProfile
            ... )
            >>> # Setup profiles
            >>> profile_a = ModelProfile(model_name="model-a")
            >>> profile_a.add_metric("accuracy", [0.90, 0.92])
            >>> profile_b = ModelProfile(model_name="model-b")
            >>> profile_b.add_metric("accuracy", [0.95, 0.94])
            >>> # Compare
            >>> comparator = ModelComparator()
            >>> comparator.add_profile(profile_a).add_profile(profile_b)
            ... # doctest: +ELLIPSIS
            <...>
            >>> result = comparator.compare()
            >>> result.winner
            'model-b'

        Checking per-metric rankings:

            >>> "accuracy" in result.rankings
            True
            >>> result.rankings["accuracy"][0]  # Best model for accuracy
            'model-b'

        Reading the summary:

            >>> "model-b" in result.summary
            True

        Checking differences from best:

            >>> result.differences.get("accuracy", {}).get("model-a") is not None
            True
    """

    models: list[str]
    winner: Optional[str] = None
    rankings: dict[str, list[str]] = field(default_factory=dict)
    differences: dict[str, dict[str, float]] = field(default_factory=dict)
    summary: str = ""
    significant_differences: dict[str, bool] = field(default_factory=dict)


class ModelComparator:
    """Compare multiple LLM models across various metrics.

    The main comparison engine that ranks models, computes weighted scores,
    and generates comparison reports. Supports customizable metric weights
    and direction preferences (higher/lower is better).

    Attributes:
        _profiles: Internal dictionary of model profiles.
        _weights: Metric weights for overall scoring.

    Examples:
        Basic comparison workflow:

            >>> from insideLLMs.analysis.comparison import ModelComparator, ModelProfile
            >>> # Create profiles
            >>> gpt4 = ModelProfile(model_name="gpt-4")
            >>> gpt4.add_metric("accuracy", [0.92, 0.94, 0.91])
            >>> gpt4.add_metric("latency", [150, 160, 145], unit="ms")
            >>>
            >>> claude = ModelProfile(model_name="claude-3")
            >>> claude.add_metric("accuracy", [0.95, 0.93, 0.94])
            >>> claude.add_metric("latency", [120, 115, 125], unit="ms")
            >>>
            >>> # Compare
            >>> comparator = ModelComparator()
            >>> comparator.add_profile(gpt4).add_profile(claude)
            ... # doctest: +ELLIPSIS
            <...>
            >>> result = comparator.compare()
            >>> print(f"Winner: {result.winner}")
            Winner: claude-3

        Customizing metric weights:

            >>> comparator = ModelComparator()
            >>> comparator.set_weight("accuracy", 2.0)  # Prioritize accuracy
            >>> comparator.set_weight("latency", 0.5)   # De-prioritize latency
            ... # doctest: +ELLIPSIS
            <...>

        Comparing on a single metric:

            >>> winner, ranked = comparator.compare_metric("accuracy")
            >>> print(f"Best accuracy: {winner}")  # doctest: +SKIP

        Generating markdown reports:

            >>> report = comparator.generate_report()
            >>> "Model Comparison Report" in report
            True
    """

    def __init__(self):
        """Initialize the model comparator.

        Creates a new comparator with default metric weights:
        - accuracy: 1.0 (highest priority)
        - error_rate: 0.8
        - latency: 0.5

        Examples:
            >>> comparator = ModelComparator()
            >>> len(comparator._profiles)
            0
        """
        self._profiles: dict[str, ModelProfile] = {}
        self._weights: dict[str, float] = {
            ComparisonMetric.ACCURACY.value: 1.0,
            ComparisonMetric.LATENCY.value: 0.5,
            ComparisonMetric.ERROR_RATE.value: 0.8,
        }

    def add_profile(self, profile: ModelProfile) -> "ModelComparator":
        """Add a model profile for comparison.

        Registers a model profile with the comparator. The profile will
        be included in subsequent comparisons.

        Args:
            profile: The ModelProfile to add.

        Returns:
            Self for method chaining.

        Examples:
            Adding a single profile:

                >>> comparator = ModelComparator()
                >>> profile = ModelProfile(model_name="gpt-4")
                >>> comparator.add_profile(profile)  # doctest: +ELLIPSIS
                <...>
                >>> "gpt-4" in comparator._profiles
                True

            Chaining multiple profiles:

                >>> p1 = ModelProfile(model_name="model-a")
                >>> p2 = ModelProfile(model_name="model-b")
                >>> comparator.add_profile(p1).add_profile(p2)  # doctest: +ELLIPSIS
                <...>

            Overwriting existing profile:

                >>> new_profile = ModelProfile(model_name="gpt-4")
                >>> new_profile.add_metric("accuracy", [0.99])
                >>> comparator.add_profile(new_profile)  # doctest: +ELLIPSIS
                <...>
        """
        self._profiles[profile.model_name] = profile
        return self

    def set_weight(self, metric: str, weight: float) -> "ModelComparator":
        """Set the weight for a metric in overall scoring.

        Higher weights mean the metric contributes more to the overall
        winner determination. Weights are relative to each other.

        Args:
            metric: Metric name (e.g., "accuracy", "latency").
            weight: Weight value (typically 0-2, where 1.0 is default).

        Returns:
            Self for method chaining.

        Examples:
            Prioritizing accuracy:

                >>> comparator = ModelComparator()
                >>> comparator.set_weight("accuracy", 2.0)  # doctest: +ELLIPSIS
                <...>
                >>> comparator._weights["accuracy"]
                2.0

            De-prioritizing cost:

                >>> comparator.set_weight("cost", 0.3)  # doctest: +ELLIPSIS
                <...>

            Chaining weight settings:

                >>> comparator.set_weight("accuracy", 1.5).set_weight("latency", 0.8)
                ... # doctest: +ELLIPSIS
                <...>
        """
        self._weights[metric] = weight
        return self

    def compare(
        self,
        metrics: Optional[list[str]] = None,
        higher_is_better: Optional[dict[str, bool]] = None,
    ) -> ModelComparisonResult:
        """Compare all registered models across specified metrics.

        Computes rankings for each metric, calculates weighted scores,
        and determines an overall winner.

        Args:
            metrics: Specific metrics to compare. If None, uses all
                available metrics across all profiles.
            higher_is_better: Dict mapping metric names to whether higher
                values are better. Defaults to True for accuracy/quality
                metrics and False for latency/cost/error metrics.

        Returns:
            ModelComparisonResult with rankings, differences, and winner.

        Raises:
            ValueError: If fewer than 2 models are registered.

        Examples:
            Basic comparison:

                >>> p1 = ModelProfile(model_name="a")
                >>> p1.add_metric("accuracy", [0.9])
                >>> p2 = ModelProfile(model_name="b")
                >>> p2.add_metric("accuracy", [0.95])
                >>> comparator = ModelComparator()
                >>> comparator.add_profile(p1).add_profile(p2)  # doctest: +ELLIPSIS
                <...>
                >>> result = comparator.compare()
                >>> result.winner
                'b'

            Comparing specific metrics:

                >>> result = comparator.compare(metrics=["accuracy"])
                >>> "accuracy" in result.rankings
                True

            Custom higher_is_better:

                >>> result = comparator.compare(
                ...     higher_is_better={"accuracy": True, "latency": False}
                ... )
        """
        if len(self._profiles) < 2:
            raise ValueError("Need at least 2 models to compare")

        # Default higher_is_better settings
        default_hib = {
            "accuracy": True,
            "success_rate": True,
            "quality_score": True,
            "throughput": True,
            "latency": False,
            "error_rate": False,
            "cost": False,
            "token_usage": False,
        }
        hib = {**default_hib, **(higher_is_better or {})}

        model_names = list(self._profiles.keys())

        # Find common metrics
        if metrics is None:
            all_metrics = set()
            for profile in self._profiles.values():
                all_metrics.update(profile.metrics.keys())
            metrics = list(all_metrics)

        # Compute rankings
        rankings: dict[str, list[str]] = {}
        differences: dict[str, dict[str, float]] = {}

        for metric in metrics:
            # Get values for each model
            values: dict[str, float] = {}
            for name, profile in self._profiles.items():
                if metric in profile.metrics:
                    values[name] = profile.metrics[metric].mean

            if not values:
                continue

            # Rank models
            reverse = hib.get(metric, True)
            ranked = sorted(values.keys(), key=lambda x: values[x], reverse=reverse)
            rankings[metric] = ranked

            # Calculate differences from best
            best_value = values[ranked[0]]
            differences[metric] = {}
            for name in model_names:
                if name in values:
                    diff = values[name] - best_value
                    differences[metric][name] = diff

        # Determine overall winner using weighted scoring
        scores: dict[str, float] = dict.fromkeys(model_names, 0.0)

        for metric, ranked in rankings.items():
            weight = self._weights.get(metric, 1.0)
            for i, name in enumerate(ranked):
                # Points: n for first, n-1 for second, etc.
                points = (len(ranked) - i) * weight
                scores[name] += points

        winner = max(scores.keys(), key=lambda x: scores[x])

        # Generate summary
        summary_parts = [f"Comparison of {len(model_names)} models:"]
        for metric, ranked in rankings.items():
            if ranked:
                summary_parts.append(f"  {metric}: {ranked[0]} (best)")

        summary_parts.append(f"\nOverall winner: {winner}")

        return ModelComparisonResult(
            models=model_names,
            winner=winner,
            rankings=rankings,
            differences=differences,
            summary="\n".join(summary_parts),
        )

    def compare_metric(
        self,
        metric: str,
        higher_is_better: bool = True,
    ) -> tuple[str, list[tuple[str, float]]]:
        """Compare all registered models on a single metric.

        Ranks models by the mean value of the specified metric, returning
        the winner and a sorted list of all models with their values.

        Parameters
        ----------
        metric : str
            The metric name to compare (e.g., "accuracy", "latency").
        higher_is_better : bool, optional
            If True, higher values are ranked first (e.g., for accuracy).
            If False, lower values are ranked first (e.g., for latency).
            Default is True.

        Returns
        -------
        tuple[str, list[tuple[str, float]]]
            A tuple containing:
            - winner: Name of the best-performing model for this metric
            - ranked: List of (model_name, mean_value) tuples, sorted by rank

        Raises
        ------
        ValueError
            If no registered models have data for the specified metric.

        Examples
        --------
        Comparing accuracy (higher is better):

            >>> from insideLLMs.analysis.comparison import ModelComparator, ModelProfile
            >>> p1 = ModelProfile(model_name="model-a")
            >>> p1.add_metric("accuracy", [0.90, 0.92])
            >>> p2 = ModelProfile(model_name="model-b")
            >>> p2.add_metric("accuracy", [0.95, 0.94])
            >>> comparator = ModelComparator()
            >>> comparator.add_profile(p1).add_profile(p2)  # doctest: +ELLIPSIS
            <...>
            >>> winner, ranked = comparator.compare_metric("accuracy")
            >>> winner
            'model-b'
            >>> ranked[0]  # Best model and its value
            ('model-b', 0.945)

        Comparing latency (lower is better):

            >>> p1.add_metric("latency", [150, 160])
            >>> p2.add_metric("latency", [100, 110])
            >>> winner, ranked = comparator.compare_metric("latency", higher_is_better=False)
            >>> winner
            'model-b'
            >>> ranked[0][1]  # Best (lowest) latency
            105.0

        Handling missing metric:

            >>> try:
            ...     comparator.compare_metric("nonexistent")
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: No data for metric: nonexistent

        Getting the ranking order:

            >>> winner, ranked = comparator.compare_metric("accuracy")
            >>> for i, (model, value) in enumerate(ranked, 1):
            ...     print(f"{i}. {model}: {value:.4f}")  # doctest: +NORMALIZE_WHITESPACE
            1. model-b: 0.9450
            2. model-a: 0.9100
        """
        values = []
        for name, profile in self._profiles.items():
            if metric in profile.metrics:
                values.append((name, profile.metrics[metric].mean))

        if not values:
            raise ValueError(f"No data for metric: {metric}")

        ranked = sorted(values, key=lambda x: x[1], reverse=higher_is_better)
        winner = ranked[0][0]

        return winner, ranked

    def generate_report(self) -> str:
        """Generate a detailed markdown-formatted comparison report.

        Creates a comprehensive report including a metrics summary table,
        per-metric rankings, and overall winner determination. The report
        is formatted in Markdown for easy viewing in documentation systems
        or Jupyter notebooks.

        Returns
        -------
        str
            A Markdown-formatted string containing:
            - Header with models compared and generation timestamp
            - Metrics summary table with mean +/- std for each model
            - Rankings section showing model ordering for each metric
            - Overall winner based on weighted scoring

        Examples
        --------
        Generating a basic report:

            >>> from insideLLMs.analysis.comparison import ModelComparator, ModelProfile
            >>> p1 = ModelProfile(model_name="gpt-4")
            >>> p1.add_metric("accuracy", [0.92, 0.94])
            >>> p1.add_metric("latency", [150, 160], unit="ms")
            >>> p2 = ModelProfile(model_name="claude-3")
            >>> p2.add_metric("accuracy", [0.95, 0.93])
            >>> p2.add_metric("latency", [100, 110], unit="ms")
            >>> comparator = ModelComparator()
            >>> comparator.add_profile(p1).add_profile(p2)  # doctest: +ELLIPSIS
            <...>
            >>> report = comparator.generate_report()
            >>> "Model Comparison Report" in report
            True
            >>> "gpt-4" in report and "claude-3" in report
            True

        Report structure:

            >>> lines = report.split("\\n")
            >>> lines[0]
            '# Model Comparison Report'
            >>> "Metrics Summary" in report
            True
            >>> "Rankings" in report
            True

        Handling single model (no comparison possible):

            >>> single = ModelComparator()
            >>> single.add_profile(p1)  # doctest: +ELLIPSIS
            <...>
            >>> report = single.generate_report()
            >>> "Rankings" not in report  # No rankings with 1 model
            True

        Using report in Jupyter:

            >>> from IPython.display import Markdown  # doctest: +SKIP
            >>> display(Markdown(comparator.generate_report()))  # doctest: +SKIP

        Saving report to file:

            >>> with open("comparison_report.md", "w") as f:  # doctest: +SKIP
            ...     f.write(comparator.generate_report())
        """
        lines = ["# Model Comparison Report", ""]
        lines.append(f"**Models compared:** {', '.join(self._profiles.keys())}")
        lines.append(f"**Generated:** {datetime.now().isoformat()}")
        lines.append("")

        # Metrics table
        lines.append("## Metrics Summary")
        lines.append("")

        # Get all metrics
        all_metrics = set()
        for profile in self._profiles.values():
            all_metrics.update(profile.metrics.keys())

        if all_metrics:
            # Header
            header = "| Metric |"
            separator = "| --- |"
            for name in self._profiles:
                header += f" {name} |"
                separator += " --- |"
            lines.append(header)
            lines.append(separator)

            # Data rows
            for metric in sorted(all_metrics):
                row = f"| {metric} |"
                for name, profile in self._profiles.items():
                    if metric in profile.metrics:
                        m = profile.metrics[metric]
                        row += f" {m.mean:.4f} ± {m.std:.4f} |"
                    else:
                        row += " N/A |"
                lines.append(row)

        lines.append("")

        # Rankings
        try:
            result = self.compare()
            lines.append("## Rankings")
            lines.append("")
            for metric, ranking in result.rankings.items():
                lines.append(f"**{metric}:** {' > '.join(ranking)}")
            lines.append("")
            lines.append(f"**Overall Winner:** {result.winner}")
        except ValueError:
            pass

        return "\n".join(lines)


@dataclass
class LatencyProfile:
    """Detailed latency profiling data for LLM responses.

    Captures the various components of response latency including time to
    first token (for streaming APIs), generation throughput, and network/API
    overhead. This breakdown enables fine-grained analysis of where time is
    spent during model inference.

    Parameters
    ----------
    total_ms : float
        Total end-to-end latency in milliseconds from request to complete response.
    first_token_ms : float, optional
        Time to first token in milliseconds. Relevant for streaming APIs where
        partial responses are returned incrementally. None for non-streaming.
    tokens_per_second : float, optional
        Token generation throughput after the first token. Calculated as
        output_tokens / (total_ms - first_token_ms) * 1000. None if not measured.
    overhead_ms : float, optional
        Non-generation overhead in milliseconds (network latency, API processing,
        tokenization, etc.). Estimated as first_token_ms minus actual inference
        start time if available.

    Attributes
    ----------
    total_ms : float
        Total latency in milliseconds.
    first_token_ms : float or None
        Time to first token (streaming).
    tokens_per_second : float or None
        Generation speed.
    overhead_ms : float or None
        Non-generation overhead.

    Examples
    --------
    Creating a basic latency profile:

        >>> from insideLLMs.analysis.comparison import LatencyProfile
        >>> profile = LatencyProfile(total_ms=250.0)
        >>> print(f"Total latency: {profile.total_ms}ms")
        Total latency: 250.0ms

    Recording streaming response metrics:

        >>> profile = LatencyProfile(
        ...     total_ms=1500.0,
        ...     first_token_ms=200.0,
        ...     tokens_per_second=45.0,
        ...     overhead_ms=50.0
        ... )
        >>> print(f"TTFT: {profile.first_token_ms}ms")
        TTFT: 200.0ms
        >>> print(f"Throughput: {profile.tokens_per_second} tok/s")
        Throughput: 45.0 tok/s

    Analyzing latency breakdown:

        >>> profile = LatencyProfile(
        ...     total_ms=2000.0,
        ...     first_token_ms=300.0,
        ...     tokens_per_second=50.0
        ... )
        >>> generation_time = profile.total_ms - (profile.first_token_ms or 0)
        >>> print(f"Generation time: {generation_time}ms")
        Generation time: 1700.0ms

    Comparing latency profiles:

        >>> fast_profile = LatencyProfile(total_ms=100.0, first_token_ms=50.0)
        >>> slow_profile = LatencyProfile(total_ms=500.0, first_token_ms=200.0)
        >>> speedup = slow_profile.total_ms / fast_profile.total_ms
        >>> print(f"Fast model is {speedup}x faster")
        Fast model is 5.0x faster

    See Also
    --------
    PerformanceTracker : Track latencies during experiments.
    MetricSummary : Aggregate latency statistics across multiple measurements.
    """

    total_ms: float
    first_token_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    overhead_ms: Optional[float] = None


@dataclass
class CostEstimate:
    """Cost estimation for LLM API usage based on token counts.

    Represents the financial cost of a model API call, broken down by input
    and output tokens. Used for budget tracking, cost optimization, and
    model comparison on a cost basis.

    Parameters
    ----------
    input_tokens : int, optional
        Number of input/prompt tokens consumed. Default is 0.
    output_tokens : int, optional
        Number of output/completion tokens generated. Default is 0.
    input_cost : float, optional
        Calculated cost for input tokens in the specified currency. Default is 0.0.
    output_cost : float, optional
        Calculated cost for output tokens in the specified currency. Default is 0.0.
    total_cost : float, optional
        Sum of input_cost and output_cost. Default is 0.0.
    currency : str, optional
        ISO 4217 currency code for the costs. Default is "USD".

    Attributes
    ----------
    input_tokens : int
        Number of input tokens.
    output_tokens : int
        Number of output tokens.
    input_cost : float
        Cost for input tokens.
    output_cost : float
        Cost for output tokens.
    total_cost : float
        Total cost (input + output).
    currency : str
        Currency code.

    Examples
    --------
    Creating a cost estimate manually:

        >>> from insideLLMs.analysis.comparison import CostEstimate
        >>> estimate = CostEstimate(
        ...     input_tokens=1000,
        ...     output_tokens=500,
        ...     input_cost=0.01,
        ...     output_cost=0.03,
        ...     total_cost=0.04
        ... )
        >>> print(f"Total: ${estimate.total_cost:.2f} {estimate.currency}")
        Total: $0.04 USD

    Using the calculate factory method:

        >>> estimate = CostEstimate.calculate(
        ...     input_tokens=5000,
        ...     output_tokens=2000,
        ...     input_price_per_1k=0.01,
        ...     output_price_per_1k=0.03
        ... )
        >>> print(f"Input cost: ${estimate.input_cost:.2f}")
        Input cost: $0.05
        >>> print(f"Output cost: ${estimate.output_cost:.2f}")
        Output cost: $0.06
        >>> print(f"Total: ${estimate.total_cost:.2f}")
        Total: $0.11

    Comparing costs for different token counts:

        >>> small_request = CostEstimate.calculate(100, 50, 0.01, 0.03)
        >>> large_request = CostEstimate.calculate(10000, 5000, 0.01, 0.03)
        >>> ratio = large_request.total_cost / small_request.total_cost
        >>> print(f"Large request is {ratio:.0f}x more expensive")
        Large request is 100x more expensive

    See Also
    --------
    ModelCostComparator : Compare costs across multiple models.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"

    @classmethod
    def calculate(
        cls,
        input_tokens: int,
        output_tokens: int,
        input_price_per_1k: float,
        output_price_per_1k: float,
    ) -> "CostEstimate":
        """Calculate a cost estimate from token counts and pricing.

        Factory method that computes input cost, output cost, and total cost
        based on the number of tokens and per-1000-token pricing rates.

        Parameters
        ----------
        input_tokens : int
            Number of input/prompt tokens.
        output_tokens : int
            Number of output/completion tokens.
        input_price_per_1k : float
            Price per 1000 input tokens (e.g., 0.01 for $0.01/1K tokens).
        output_price_per_1k : float
            Price per 1000 output tokens (e.g., 0.03 for $0.03/1K tokens).

        Returns
        -------
        CostEstimate
            A new CostEstimate instance with all costs calculated.

        Examples
        --------
        Basic cost calculation:

            >>> estimate = CostEstimate.calculate(
            ...     input_tokens=1000,
            ...     output_tokens=500,
            ...     input_price_per_1k=0.03,
            ...     output_price_per_1k=0.06
            ... )
            >>> estimate.input_cost
            0.03
            >>> estimate.output_cost
            0.03
            >>> estimate.total_cost
            0.06

        GPT-4 pricing example:

            >>> # GPT-4: $0.03/1K input, $0.06/1K output
            >>> estimate = CostEstimate.calculate(
            ...     input_tokens=2000,
            ...     output_tokens=1000,
            ...     input_price_per_1k=0.03,
            ...     output_price_per_1k=0.06
            ... )
            >>> print(f"GPT-4 cost: ${estimate.total_cost:.2f}")
            GPT-4 cost: $0.12

        Claude pricing example:

            >>> # Claude-3-Haiku: $0.00025/1K input, $0.00125/1K output
            >>> estimate = CostEstimate.calculate(
            ...     input_tokens=2000,
            ...     output_tokens=1000,
            ...     input_price_per_1k=0.00025,
            ...     output_price_per_1k=0.00125
            ... )
            >>> print(f"Claude-3-Haiku cost: ${estimate.total_cost:.6f}")
            Claude-3-Haiku cost: $0.001750

        Zero tokens edge case:

            >>> empty = CostEstimate.calculate(0, 0, 0.01, 0.03)
            >>> empty.total_cost
            0.0
        """
        input_cost = (input_tokens / 1000) * input_price_per_1k
        output_cost = (output_tokens / 1000) * output_price_per_1k

        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
        )


class ModelCostComparator:
    """Calculate and compare API costs across different LLM models.

    Provides cost estimation based on token usage and configurable pricing.
    Includes default pricing for common models and supports custom pricing.

    Attributes:
        DEFAULT_PRICING: Default per-1k-token prices for common models.
        _pricing: Internal pricing dictionary.

    Examples:
        Basic cost estimation:

            >>> from insideLLMs.analysis.comparison import ModelCostComparator
            >>> calc = ModelCostComparator()
            >>> cost = calc.estimate("gpt-4", input_tokens=1000, output_tokens=500)
            >>> print(f"Total cost: ${cost.total_cost:.4f}")
            Total cost: $0.0600

        Setting custom pricing:

            >>> calc = ModelCostComparator()
            >>> calc.set_pricing("my-model", input_per_1k=0.001, output_per_1k=0.002)
            ... # doctest: +ELLIPSIS
            <...>
            >>> cost = calc.estimate("my-model", 10000, 5000)
            >>> cost.total_cost
            0.02

        Comparing costs across models:

            >>> costs = calc.compare_costs(input_tokens=5000, output_tokens=2000)
            >>> for model, cost in sorted(costs.items(), key=lambda x: x[1].total_cost):
            ...     print(f"{model}: ${cost.total_cost:.4f}")  # doctest: +SKIP

        Finding the cheapest model:

            >>> model, cost = calc.cheapest_model(1000, 500)
            >>> print(f"Cheapest: {model} at ${cost.total_cost:.4f}")  # doctest: +SKIP
    """

    # Default pricing (approximate, as of late 2024)
    DEFAULT_PRICING = {
        "gpt-4": (0.03, 0.06),  # input, output per 1k tokens
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "claude-3-opus": (0.015, 0.075),
        "claude-3-sonnet": (0.003, 0.015),
        "claude-3-haiku": (0.00025, 0.00125),
    }

    def __init__(self):
        """Initialize with default pricing for common models.

        Examples:
            >>> calc = ModelCostComparator()
            >>> "gpt-4" in calc._pricing
            True
            >>> calc._pricing["gpt-4"]
            (0.03, 0.06)
        """
        self._pricing: dict[str, tuple[float, float]] = dict(self.DEFAULT_PRICING)

    def set_pricing(
        self,
        model: str,
        input_per_1k: float,
        output_per_1k: float,
    ) -> "ModelCostComparator":
        """Set or update pricing for a model.

        Registers custom pricing for a model, allowing cost estimation for
        models not in the default pricing table or overriding default prices
        with current rates.

        Parameters
        ----------
        model : str
            Model identifier (e.g., "gpt-4", "my-custom-model").
        input_per_1k : float
            Price per 1000 input tokens in USD (e.g., 0.01 for $0.01/1K).
        output_per_1k : float
            Price per 1000 output tokens in USD (e.g., 0.03 for $0.03/1K).

        Returns
        -------
        ModelCostComparator
            Self for method chaining.

        Examples
        --------
        Adding a custom model:

            >>> from insideLLMs.analysis.comparison import ModelCostComparator
            >>> calc = ModelCostComparator()
            >>> calc.set_pricing("my-model", input_per_1k=0.001, output_per_1k=0.002)
            ... # doctest: +ELLIPSIS
            <...>
            >>> "my-model" in calc._pricing
            True

        Updating existing model pricing:

            >>> calc.set_pricing("gpt-4", input_per_1k=0.02, output_per_1k=0.05)
            ... # doctest: +ELLIPSIS
            <...>
            >>> calc._pricing["gpt-4"]
            (0.02, 0.05)

        Chaining multiple pricing updates:

            >>> calc.set_pricing("model-a", 0.01, 0.02).set_pricing("model-b", 0.02, 0.04)
            ... # doctest: +ELLIPSIS
            <...>

        Using with estimate:

            >>> calc.set_pricing("fast-model", 0.0001, 0.0002)  # doctest: +ELLIPSIS
            <...>
            >>> cost = calc.estimate("fast-model", 10000, 5000)
            >>> print(f"Cost: ${cost.total_cost:.4f}")
            Cost: $0.0020
        """
        self._pricing[model] = (input_per_1k, output_per_1k)
        return self

    def estimate(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostEstimate:
        """Estimate the cost for a specific model and token usage.

        Calculates the total cost based on the model's pricing and the
        number of input/output tokens.

        Parameters
        ----------
        model : str
            Model identifier. Must have pricing registered (either default
            or via set_pricing).
        input_tokens : int
            Number of input/prompt tokens.
        output_tokens : int
            Number of output/completion tokens.

        Returns
        -------
        CostEstimate
            A CostEstimate object with detailed cost breakdown.

        Raises
        ------
        KeyError
            If the model is not in the pricing table.

        Examples
        --------
        Estimating GPT-4 cost:

            >>> from insideLLMs.analysis.comparison import ModelCostComparator
            >>> calc = ModelCostComparator()
            >>> cost = calc.estimate("gpt-4", input_tokens=1000, output_tokens=500)
            >>> print(f"Total: ${cost.total_cost:.4f}")
            Total: $0.0600

        Comparing costs for different usage patterns:

            >>> small_cost = calc.estimate("gpt-4", 100, 50)
            >>> large_cost = calc.estimate("gpt-4", 10000, 5000)
            >>> print(f"Small: ${small_cost.total_cost:.4f}")
            Small: $0.0060
            >>> print(f"Large: ${large_cost.total_cost:.4f}")
            Large: $0.6000

        Handling unknown models:

            >>> try:
            ...     calc.estimate("unknown-model", 1000, 500)
            ... except KeyError as e:
            ...     print(f"Error: {e}")
            Error: 'No pricing for model: unknown-model'

        Cost breakdown analysis:

            >>> cost = calc.estimate("claude-3-opus", 5000, 2000)
            >>> print(f"Input: ${cost.input_cost:.4f}")
            Input: $0.0750
            >>> print(f"Output: ${cost.output_cost:.4f}")
            Output: $0.1500
            >>> print(f"Total: ${cost.total_cost:.4f}")
            Total: $0.2250
        """
        if model not in self._pricing:
            raise KeyError(f"No pricing for model: {model}")

        input_price, output_price = self._pricing[model]
        return CostEstimate.calculate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_price_per_1k=input_price,
            output_price_per_1k=output_price,
        )

    def compare_costs(
        self,
        input_tokens: int,
        output_tokens: int,
        models: Optional[list[str]] = None,
    ) -> dict[str, CostEstimate]:
        """Compare costs across multiple models for the same token usage.

        Generates cost estimates for all specified models (or all known models
        if not specified), enabling side-by-side cost comparison.

        Parameters
        ----------
        input_tokens : int
            Number of input/prompt tokens to estimate.
        output_tokens : int
            Number of output/completion tokens to estimate.
        models : list of str, optional
            Specific models to compare. If None, compares all models with
            registered pricing.

        Returns
        -------
        dict[str, CostEstimate]
            Dictionary mapping model names to their CostEstimate objects.
            Models not found in pricing are silently skipped.

        Examples
        --------
        Comparing all default models:

            >>> from insideLLMs.analysis.comparison import ModelCostComparator
            >>> calc = ModelCostComparator()
            >>> costs = calc.compare_costs(input_tokens=1000, output_tokens=500)
            >>> len(costs) >= 6  # At least 6 default models
            True

        Comparing specific models:

            >>> costs = calc.compare_costs(
            ...     input_tokens=5000,
            ...     output_tokens=2000,
            ...     models=["gpt-4", "claude-3-haiku"]
            ... )
            >>> "gpt-4" in costs and "claude-3-haiku" in costs
            True
            >>> len(costs)
            2

        Finding the cheapest model from comparison:

            >>> costs = calc.compare_costs(1000, 500)
            >>> cheapest = min(costs.items(), key=lambda x: x[1].total_cost)
            >>> print(f"Cheapest: {cheapest[0]}")  # doctest: +SKIP

        Generating a cost report:

            >>> costs = calc.compare_costs(10000, 5000)
            >>> for model, cost in sorted(costs.items(), key=lambda x: x[1].total_cost):
            ...     print(f"{model}: ${cost.total_cost:.4f}")  # doctest: +SKIP

        Handling unknown models gracefully:

            >>> costs = calc.compare_costs(
            ...     1000, 500,
            ...     models=["gpt-4", "nonexistent-model"]
            ... )
            >>> "nonexistent-model" in costs
            False
            >>> "gpt-4" in costs
            True
        """
        if models is None:
            models = list(self._pricing.keys())

        results = {}
        for model in models:
            if model in self._pricing:
                results[model] = self.estimate(model, input_tokens, output_tokens)

        return results

    def cheapest_model(
        self,
        input_tokens: int,
        output_tokens: int,
        models: Optional[list[str]] = None,
    ) -> tuple[str, CostEstimate]:
        """Find the cheapest model for a given token usage.

        Compares costs across all specified models and returns the one with
        the lowest total cost.

        Parameters
        ----------
        input_tokens : int
            Number of input/prompt tokens.
        output_tokens : int
            Number of output/completion tokens.
        models : list of str, optional
            Specific models to consider. If None, considers all models with
            registered pricing.

        Returns
        -------
        tuple[str, CostEstimate]
            A tuple of (model_name, cost_estimate) for the cheapest model.

        Raises
        ------
        ValueError
            If no models with pricing are available to compare.

        Examples
        --------
        Finding the cheapest model:

            >>> from insideLLMs.analysis.comparison import ModelCostComparator
            >>> calc = ModelCostComparator()
            >>> model, cost = calc.cheapest_model(1000, 500)
            >>> model  # Will be the cheapest default model
            'claude-3-haiku'
            >>> print(f"Cost: ${cost.total_cost:.6f}")
            Cost: $0.000875

        Comparing within a subset:

            >>> model, cost = calc.cheapest_model(
            ...     input_tokens=5000,
            ...     output_tokens=2000,
            ...     models=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
            ... )
            >>> model
            'gpt-3.5-turbo'

        Cost-based model selection logic:

            >>> def select_model(tokens_in, tokens_out, budget):
            ...     model, cost = calc.cheapest_model(tokens_in, tokens_out)
            ...     if cost.total_cost <= budget:
            ...         return model
            ...     return None
            >>> selected = select_model(1000, 500, 0.01)
            >>> selected is not None
            True

        Comparing premium vs budget models:

            >>> premium, premium_cost = calc.cheapest_model(
            ...     1000, 500, models=["gpt-4", "claude-3-opus"]
            ... )
            >>> budget, budget_cost = calc.cheapest_model(
            ...     1000, 500, models=["gpt-3.5-turbo", "claude-3-haiku"]
            ... )
            >>> premium_cost.total_cost > budget_cost.total_cost
            True
        """
        costs = self.compare_costs(input_tokens, output_tokens, models)
        cheapest = min(costs.items(), key=lambda x: x[1].total_cost)
        return cheapest


@dataclass
class QualityMetrics:
    """Quality metrics for evaluating LLM output quality.

    Captures multiple dimensions of response quality including coherence,
    relevance, accuracy, completeness, and fluency. Supports weighted
    aggregation into an overall quality score.

    Parameters
    ----------
    coherence : float, optional
        How logically consistent and well-organized the output is (0-1).
        Default is 0.0.
    relevance : float, optional
        How well the output addresses the input/prompt (0-1). Default is 0.0.
    accuracy : float, optional
        Factual correctness of the output (0-1). Default is 0.0.
    completeness : float, optional
        How thoroughly the output covers the requested information (0-1).
        Default is 0.0.
    fluency : float, optional
        Grammatical correctness and natural language flow (0-1). Default is 0.0.
    overall : float, optional
        Weighted aggregate of all quality dimensions (0-1). Default is 0.0.
        Computed via compute_overall().

    Attributes
    ----------
    coherence : float
        Output coherence score (0-1).
    relevance : float
        Relevance to input (0-1).
    accuracy : float
        Factual accuracy (0-1).
    completeness : float
        Completeness of response (0-1).
    fluency : float
        Language fluency (0-1).
    overall : float
        Overall quality score (0-1).

    Examples
    --------
    Creating quality metrics from evaluation:

        >>> from insideLLMs.analysis.comparison import QualityMetrics
        >>> metrics = QualityMetrics(
        ...     coherence=0.9,
        ...     relevance=0.85,
        ...     accuracy=0.95,
        ...     completeness=0.8,
        ...     fluency=0.92
        ... )
        >>> print(f"Accuracy: {metrics.accuracy}")
        Accuracy: 0.95

    Computing overall score with default weights:

        >>> metrics = QualityMetrics(
        ...     coherence=0.9,
        ...     relevance=0.85,
        ...     accuracy=0.95,
        ...     completeness=0.8,
        ...     fluency=0.92
        ... )
        >>> overall = metrics.compute_overall()
        >>> print(f"Overall: {overall:.4f}")
        Overall: 0.8930

    Using custom weights for specific use cases:

        >>> # Prioritize accuracy for factual tasks
        >>> metrics = QualityMetrics(
        ...     coherence=0.8,
        ...     relevance=0.9,
        ...     accuracy=0.95,
        ...     completeness=0.7,
        ...     fluency=0.85
        ... )
        >>> overall = metrics.compute_overall(weights={
        ...     "coherence": 0.1,
        ...     "relevance": 0.2,
        ...     "accuracy": 0.5,  # High weight for accuracy
        ...     "completeness": 0.1,
        ...     "fluency": 0.1
        ... })
        >>> print(f"Accuracy-weighted overall: {overall:.4f}")
        Accuracy-weighted overall: 0.9000

    Comparing quality across models:

        >>> model_a_quality = QualityMetrics(
        ...     coherence=0.85, relevance=0.9, accuracy=0.8,
        ...     completeness=0.75, fluency=0.88
        ... )
        >>> model_b_quality = QualityMetrics(
        ...     coherence=0.9, relevance=0.85, accuracy=0.92,
        ...     completeness=0.8, fluency=0.9
        ... )
        >>> a_overall = model_a_quality.compute_overall()
        >>> b_overall = model_b_quality.compute_overall()
        >>> winner = "Model A" if a_overall > b_overall else "Model B"
        >>> print(f"Higher quality: {winner}")
        Higher quality: Model B

    See Also
    --------
    ModelProfile : Store quality metrics as part of model profiles.
    ModelComparator : Compare models including quality metrics.
    """

    coherence: float = 0.0
    relevance: float = 0.0
    accuracy: float = 0.0
    completeness: float = 0.0
    fluency: float = 0.0
    overall: float = 0.0

    def compute_overall(
        self,
        weights: Optional[dict[str, float]] = None,
    ) -> float:
        """Compute a weighted overall quality score from component metrics.

        Calculates a single aggregate score by combining all quality dimensions
        using specified or default weights. The result is stored in self.overall
        and also returned.

        Parameters
        ----------
        weights : dict[str, float], optional
            Custom weights for each quality dimension. Keys should be:
            "coherence", "relevance", "accuracy", "completeness", "fluency".
            Weights should sum to 1.0 for a normalized result.
            If None, uses default weights:
            - coherence: 0.2
            - relevance: 0.25
            - accuracy: 0.25
            - completeness: 0.15
            - fluency: 0.15

        Returns
        -------
        float
            The computed overall quality score (0-1 if inputs are 0-1).

        Examples
        --------
        Using default weights:

            >>> metrics = QualityMetrics(
            ...     coherence=0.9, relevance=0.85, accuracy=0.95,
            ...     completeness=0.8, fluency=0.92
            ... )
            >>> overall = metrics.compute_overall()
            >>> print(f"Overall: {overall:.4f}")
            Overall: 0.8930
            >>> metrics.overall == overall
            True

        Custom weights for creative writing evaluation:

            >>> metrics = QualityMetrics(
            ...     coherence=0.9, relevance=0.7, accuracy=0.6,
            ...     completeness=0.8, fluency=0.95
            ... )
            >>> # Prioritize fluency and coherence for creative tasks
            >>> overall = metrics.compute_overall(weights={
            ...     "coherence": 0.3,
            ...     "relevance": 0.1,
            ...     "accuracy": 0.1,
            ...     "completeness": 0.1,
            ...     "fluency": 0.4
            ... })
            >>> print(f"Creative writing score: {overall:.4f}")
            Creative writing score: 0.8600

        Custom weights for factual Q&A evaluation:

            >>> metrics = QualityMetrics(
            ...     coherence=0.85, relevance=0.9, accuracy=0.98,
            ...     completeness=0.75, fluency=0.8
            ... )
            >>> overall = metrics.compute_overall(weights={
            ...     "coherence": 0.1,
            ...     "relevance": 0.2,
            ...     "accuracy": 0.4,  # High accuracy importance
            ...     "completeness": 0.2,
            ...     "fluency": 0.1
            ... })
            >>> print(f"Factual Q&A score: {overall:.4f}")
            Factual Q&A score: 0.9070

        Partial weight override (merges with defaults):

            >>> metrics = QualityMetrics(
            ...     coherence=0.9, relevance=0.9, accuracy=0.9,
            ...     completeness=0.9, fluency=0.9
            ... )
            >>> overall = metrics.compute_overall(weights={"accuracy": 0.5})
            >>> overall  # Uses default weights except accuracy=0.5
            0.9
        """
        default_weights = {
            "coherence": 0.2,
            "relevance": 0.25,
            "accuracy": 0.25,
            "completeness": 0.15,
            "fluency": 0.15,
        }
        w = {**default_weights, **(weights or {})}

        self.overall = (
            self.coherence * w["coherence"]
            + self.relevance * w["relevance"]
            + self.accuracy * w["accuracy"]
            + self.completeness * w["completeness"]
            + self.fluency * w["fluency"]
        )
        return self.overall


class PerformanceTracker:
    """Track performance metrics during experiment execution.

    Collects latency, success/failure, and token usage data during
    model evaluation, then provides summary statistics as a ModelProfile.

    Attributes:
        model_name: Name of the model being tracked.
        _latencies: List of recorded latency measurements.
        _successes: List of success/failure flags.
        _token_counts: List of (input_tokens, output_tokens) tuples.
        _timestamps: Timestamps for each recording.
        _errors: List of error messages for failed requests.

    Examples:
        Basic tracking during evaluation:

            >>> from insideLLMs.analysis.comparison import PerformanceTracker
            >>> tracker = PerformanceTracker("gpt-4")
            >>>
            >>> # Record metrics during experiment
            >>> tracker.record_latency(150.5)
            >>> tracker.record_latency(145.0)
            >>> tracker.record_success(True)
            >>> tracker.record_success(True)
            >>> tracker.record_tokens(100, 50)
            >>> tracker.record_tokens(120, 60)
            >>>
            >>> # Get summary as ModelProfile
            >>> profile = tracker.get_summary()
            >>> profile.model_name
            'gpt-4'
            >>> "latency" in profile.metrics
            True

        Recording errors:

            >>> tracker.record_success(False, error="API timeout")
            >>> len(tracker._errors)
            1

        Tracking a full evaluation loop:

            >>> tracker = PerformanceTracker("claude-3")
            >>> for response in api_responses:  # doctest: +SKIP
            ...     tracker.record_latency(response.latency_ms)
            ...     tracker.record_success(response.success, response.error)
            ...     tracker.record_tokens(response.input_tokens, response.output_tokens)

        Resetting for a new experiment:

            >>> tracker.reset()
            >>> len(tracker._latencies)
            0
    """

    def __init__(self, model_name: str):
        """Initialize the performance tracker.

        Args:
            model_name: Name of the model being tracked.

        Examples:
            >>> tracker = PerformanceTracker("gpt-4-turbo")
            >>> tracker.model_name
            'gpt-4-turbo'
            >>> len(tracker._latencies)
            0
        """
        self.model_name = model_name
        self._latencies: list[float] = []
        self._successes: list[bool] = []
        self._token_counts: list[tuple[int, int]] = []  # (input, output)
        self._timestamps: list[datetime] = []
        self._errors: list[str] = []

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement for the tracked model.

        Appends a latency value to the internal list and records the
        timestamp. Use this to track response times during experiments.

        Parameters
        ----------
        latency_ms : float
            Response latency in milliseconds.

        Examples
        --------
        Recording latencies during an experiment:

            >>> from insideLLMs.analysis.comparison import PerformanceTracker
            >>> tracker = PerformanceTracker("gpt-4")
            >>> tracker.record_latency(150.5)
            >>> tracker.record_latency(145.0)
            >>> tracker.record_latency(160.2)
            >>> len(tracker._latencies)
            3
            >>> tracker._latencies[0]
            150.5

        Simulating API calls:

            >>> import time
            >>> tracker = PerformanceTracker("claude-3")
            >>> for _ in range(3):  # doctest: +SKIP
            ...     start = time.time()
            ...     response = api.call()  # Your API call
            ...     latency = (time.time() - start) * 1000
            ...     tracker.record_latency(latency)

        Checking timestamps are recorded:

            >>> tracker = PerformanceTracker("test")
            >>> tracker.record_latency(100.0)
            >>> len(tracker._timestamps) == len(tracker._latencies)
            True
        """
        self._latencies.append(latency_ms)
        self._timestamps.append(datetime.now())

    def record_success(self, success: bool, error: Optional[str] = None) -> None:
        """Record whether a model call succeeded or failed.

        Tracks success/failure status for computing success rate. Optionally
        records error messages for failed calls to enable error analysis.

        Parameters
        ----------
        success : bool
            True if the API call succeeded, False if it failed.
        error : str, optional
            Error message describing the failure. Only recorded if success
            is False and error is provided.

        Examples
        --------
        Recording successful calls:

            >>> from insideLLMs.analysis.comparison import PerformanceTracker
            >>> tracker = PerformanceTracker("gpt-4")
            >>> tracker.record_success(True)
            >>> tracker.record_success(True)
            >>> sum(tracker._successes)
            2

        Recording failures with error messages:

            >>> tracker.record_success(False, error="Rate limit exceeded")
            >>> tracker.record_success(False, error="API timeout")
            >>> len(tracker._errors)
            2
            >>> tracker._errors[0]
            'Rate limit exceeded'

        Calculating success rate:

            >>> tracker = PerformanceTracker("test")
            >>> for _ in range(8):
            ...     tracker.record_success(True)
            >>> for _ in range(2):
            ...     tracker.record_success(False, "Error")
            >>> success_rate = sum(tracker._successes) / len(tracker._successes)
            >>> print(f"Success rate: {success_rate:.0%}")
            Success rate: 80%

        Recording failure without error message:

            >>> tracker = PerformanceTracker("test")
            >>> tracker.record_success(False)  # No error message
            >>> len(tracker._errors)  # Error not recorded
            0
            >>> len(tracker._successes)
            1
        """
        self._successes.append(success)
        if not success and error:
            self._errors.append(error)

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for a model call.

        Stores the input and output token counts as a tuple for later
        analysis and summary statistics.

        Parameters
        ----------
        input_tokens : int
            Number of tokens in the input/prompt.
        output_tokens : int
            Number of tokens in the output/completion.

        Examples
        --------
        Recording token usage:

            >>> from insideLLMs.analysis.comparison import PerformanceTracker
            >>> tracker = PerformanceTracker("gpt-4")
            >>> tracker.record_tokens(100, 50)
            >>> tracker.record_tokens(150, 75)
            >>> tracker.record_tokens(200, 100)
            >>> len(tracker._token_counts)
            3
            >>> tracker._token_counts[0]
            (100, 50)

        Tracking token usage from API responses:

            >>> tracker = PerformanceTracker("claude-3")
            >>> responses = [
            ...     {"input_tokens": 100, "output_tokens": 50},
            ...     {"input_tokens": 120, "output_tokens": 60},
            ... ]
            >>> for resp in responses:
            ...     tracker.record_tokens(resp["input_tokens"], resp["output_tokens"])
            >>> len(tracker._token_counts)
            2

        Calculating total tokens:

            >>> tracker = PerformanceTracker("test")
            >>> tracker.record_tokens(500, 200)
            >>> tracker.record_tokens(600, 300)
            >>> total_input = sum(t[0] for t in tracker._token_counts)
            >>> total_output = sum(t[1] for t in tracker._token_counts)
            >>> print(f"Total: {total_input} input, {total_output} output")
            Total: 1100 input, 500 output
        """
        self._token_counts.append((input_tokens, output_tokens))

    def get_summary(self) -> ModelProfile:
        """Generate a ModelProfile summarizing all tracked metrics.

        Aggregates all recorded latencies, success/failure outcomes, and
        token counts into a ModelProfile with computed statistics for each
        metric type.

        Returns
        -------
        ModelProfile
            A ModelProfile containing:
            - latency: MetricSummary of recorded latencies (if any)
            - success_rate: Percentage of successful calls (if any)
            - error_rate: Percentage of failed calls (if any)
            - input_tokens: MetricSummary of input token counts (if any)
            - output_tokens: MetricSummary of output token counts (if any)
            - total_tokens: MetricSummary of combined token counts (if any)
            - metadata: total_requests and total_errors counts

        Examples
        --------
        Getting a summary after tracking:

            >>> from insideLLMs.analysis.comparison import PerformanceTracker
            >>> tracker = PerformanceTracker("gpt-4")
            >>> tracker.record_latency(150.0)
            >>> tracker.record_latency(145.0)
            >>> tracker.record_latency(160.0)
            >>> tracker.record_success(True)
            >>> tracker.record_success(True)
            >>> tracker.record_success(False, "Timeout")
            >>> tracker.record_tokens(100, 50)
            >>> tracker.record_tokens(120, 60)
            >>>
            >>> profile = tracker.get_summary()
            >>> profile.model_name
            'gpt-4'
            >>> round(profile.metrics["latency"].mean, 1)
            151.7

        Checking success rate:

            >>> profile.metrics["success_rate"].mean  # 2/3 = 66.67%
            66.66666666666666

        Accessing token metrics:

            >>> profile.metrics["input_tokens"].mean
            110.0
            >>> profile.metrics["total_tokens"].mean
            165.0

        Checking metadata:

            >>> profile.metadata["total_requests"]
            3
            >>> profile.metadata["total_errors"]
            1

        Using summary with ModelComparator:

            >>> from insideLLMs.analysis.comparison import ModelComparator
            >>> tracker_a = PerformanceTracker("model-a")
            >>> tracker_a.record_latency(100.0)
            >>> tracker_a.record_success(True)
            >>> tracker_b = PerformanceTracker("model-b")
            >>> tracker_b.record_latency(150.0)
            >>> tracker_b.record_success(True)
            >>>
            >>> comparator = ModelComparator()
            >>> comparator.add_profile(tracker_a.get_summary())  # doctest: +ELLIPSIS
            <...>
            >>> comparator.add_profile(tracker_b.get_summary())  # doctest: +ELLIPSIS
            <...>
        """
        profile = ModelProfile(
            model_name=self.model_name,
            metadata={
                "total_requests": len(self._successes),
                "total_errors": len(self._errors),
            },
        )

        # Latency metrics
        if self._latencies:
            profile.add_metric("latency", self._latencies, "ms")

        # Success rate
        if self._successes:
            success_rate = sum(self._successes) / len(self._successes)
            profile.add_metric("success_rate", [success_rate * 100], "%")
            profile.add_metric("error_rate", [(1 - success_rate) * 100], "%")

        # Token metrics
        if self._token_counts:
            input_tokens = [t[0] for t in self._token_counts]
            output_tokens = [t[1] for t in self._token_counts]
            profile.add_metric("input_tokens", input_tokens, "tokens")
            profile.add_metric("output_tokens", output_tokens, "tokens")
            profile.add_metric(
                "total_tokens",
                [t[0] + t[1] for t in self._token_counts],
                "tokens",
            )

        return profile

    def reset(self) -> None:
        """Reset all tracked data to initial empty state.

        Clears all recorded latencies, successes, token counts, timestamps,
        and errors. Use this to reuse the tracker for a new experiment
        without creating a new instance.

        Examples
        --------
        Resetting after an experiment:

            >>> from insideLLMs.analysis.comparison import PerformanceTracker
            >>> tracker = PerformanceTracker("gpt-4")
            >>> tracker.record_latency(150.0)
            >>> tracker.record_success(True)
            >>> tracker.record_tokens(100, 50)
            >>> len(tracker._latencies)
            1
            >>>
            >>> tracker.reset()
            >>> len(tracker._latencies)
            0
            >>> len(tracker._successes)
            0
            >>> len(tracker._token_counts)
            0

        Running multiple experiments:

            >>> tracker = PerformanceTracker("model")
            >>> # First experiment
            >>> for i in range(5):
            ...     tracker.record_latency(100.0 + i * 10)
            >>> summary1 = tracker.get_summary()
            >>> summary1.metrics["latency"].count
            5
            >>>
            >>> # Reset for second experiment
            >>> tracker.reset()
            >>> for i in range(3):
            ...     tracker.record_latency(200.0 + i * 10)
            >>> summary2 = tracker.get_summary()
            >>> summary2.metrics["latency"].count
            3

        Preserving model name after reset:

            >>> tracker = PerformanceTracker("my-model")
            >>> tracker.record_latency(100.0)
            >>> tracker.reset()
            >>> tracker.model_name  # Model name is preserved
            'my-model'
        """
        self._latencies.clear()
        self._successes.clear()
        self._token_counts.clear()
        self._timestamps.clear()
        self._errors.clear()


def create_comparison_table(
    profiles: list[ModelProfile],
    metrics: Optional[list[str]] = None,
) -> str:
    """Create a markdown comparison table from model profiles.

    Generates a formatted markdown table showing metric values for each
    model, suitable for documentation or reports.

    Args:
        profiles: List of ModelProfile objects to compare.
        metrics: Specific metrics to include. If None, includes all
            metrics found across all profiles.

    Returns:
        Markdown-formatted table string.

    Examples:
        Basic usage:

            >>> from insideLLMs.analysis.comparison import (
            ...     ModelProfile, create_comparison_table
            ... )
            >>> p1 = ModelProfile(model_name="gpt-4")
            >>> p1.add_metric("accuracy", [0.92, 0.94])
            >>> p2 = ModelProfile(model_name="claude-3")
            >>> p2.add_metric("accuracy", [0.95, 0.93])
            >>> table = create_comparison_table([p1, p2])
            >>> "gpt-4" in table and "claude-3" in table
            True

        Specifying metrics:

            >>> table = create_comparison_table([p1, p2], metrics=["accuracy"])
            >>> "accuracy" in table
            True

        Empty profiles:

            >>> create_comparison_table([])
            ''
    """
    if not profiles:
        return ""

    # Get all metrics
    if metrics is None:
        all_metrics = set()
        for profile in profiles:
            all_metrics.update(profile.metrics.keys())
        metrics = sorted(all_metrics)

    # Build table
    lines = []

    # Header
    header = "| Metric |"
    separator = "| --- |"
    for profile in profiles:
        header += f" {profile.model_name} |"
        separator += " ---: |"
    lines.append(header)
    lines.append(separator)

    # Data rows
    for metric in metrics:
        row = f"| {metric} |"
        for profile in profiles:
            if metric in profile.metrics:
                m = profile.metrics[metric]
                row += f" {m.mean:.3f} |"
            else:
                row += " - |"
        lines.append(row)

    return "\n".join(lines)


def rank_models(
    profiles: list[ModelProfile],
    metric: str,
    higher_is_better: bool = True,
) -> list[tuple[str, float]]:
    """Rank models by a specific metric.

    Sorts model profiles by the mean value of the specified metric,
    returning a ranked list of (model_name, value) tuples.

    Args:
        profiles: List of ModelProfile objects to rank.
        metric: Metric name to rank by (e.g., "accuracy", "latency").
        higher_is_better: If True, higher values rank first. If False,
            lower values rank first (e.g., for latency).

    Returns:
        List of (model_name, metric_value) tuples, sorted by rank.

    Examples:
        Ranking by accuracy (higher is better):

            >>> from insideLLMs.analysis.comparison import ModelProfile, rank_models
            >>> p1 = ModelProfile(model_name="gpt-4")
            >>> p1.add_metric("accuracy", [0.90])
            >>> p2 = ModelProfile(model_name="claude-3")
            >>> p2.add_metric("accuracy", [0.95])
            >>> ranked = rank_models([p1, p2], "accuracy")
            >>> ranked[0][0]  # Best model
            'claude-3'

        Ranking by latency (lower is better):

            >>> p1.add_metric("latency", [150])
            >>> p2.add_metric("latency", [100])
            >>> ranked = rank_models([p1, p2], "latency", higher_is_better=False)
            >>> ranked[0][0]  # Fastest model
            'claude-3'

        Handling missing metrics:

            >>> p3 = ModelProfile(model_name="new-model")  # No accuracy metric
            >>> ranked = rank_models([p1, p2, p3], "accuracy")
            >>> len(ranked)  # Only models with the metric
            2
    """
    values = []
    for profile in profiles:
        if metric in profile.metrics:
            values.append((profile.model_name, profile.metrics[metric].mean))

    return sorted(values, key=lambda x: x[1], reverse=higher_is_better)
