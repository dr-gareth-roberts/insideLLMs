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
"""

import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)


T = TypeVar("T")


class ComparisonMetric(Enum):
    """Standard metrics for model comparison."""

    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    COST = "cost"
    ERROR_RATE = "error_rate"
    TOKEN_USAGE = "token_usage"
    QUALITY_SCORE = "quality_score"


@dataclass
class MetricValue:
    """A single metric measurement.

    Attributes:
        value: The metric value.
        unit: Unit of measurement.
        timestamp: When measured.
        metadata: Additional context.
    """

    value: float
    unit: str = ""
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MetricSummary:
    """Summary statistics for a metric.

    Attributes:
        name: Metric name.
        mean: Mean value.
        std: Standard deviation.
        min: Minimum value.
        max: Maximum value.
        median: Median value.
        count: Number of samples.
        percentile_95: 95th percentile.
        percentile_99: 99th percentile.
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
    def from_values(cls, name: str, values: List[float], unit: str = "") -> "MetricSummary":
        """Compute summary from list of values.

        Args:
            name: Metric name.
            values: List of values.
            unit: Unit of measurement.

        Returns:
            MetricSummary object.
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
    """Performance profile for a model.

    Attributes:
        model_name: Name of the model.
        model_id: Unique identifier.
        metrics: Dictionary of metric summaries.
        raw_results: Raw result data.
        metadata: Additional model info.
    """

    model_name: str
    model_id: str = ""
    metrics: Dict[str, MetricSummary] = field(default_factory=dict)
    raw_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_metric(self, name: str, values: List[float], unit: str = "") -> None:
        """Add a metric summary.

        Args:
            name: Metric name.
            values: List of values.
            unit: Unit of measurement.
        """
        self.metrics[name] = MetricSummary.from_values(name, values, unit)

    def get_metric(self, name: str) -> Optional[MetricSummary]:
        """Get a metric summary by name.

        Args:
            name: Metric name.

        Returns:
            MetricSummary or None.
        """
        return self.metrics.get(name)


@dataclass
class ComparisonResult:
    """Result of comparing two or more models.

    Attributes:
        models: List of model names compared.
        winner: Best performing model (if clear winner).
        rankings: Rankings by each metric.
        differences: Statistical differences.
        summary: Text summary.
    """

    models: List[str]
    winner: Optional[str] = None
    rankings: Dict[str, List[str]] = field(default_factory=dict)
    differences: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: str = ""
    significant_differences: Dict[str, bool] = field(default_factory=dict)


class ModelComparator:
    """Compare multiple models across various metrics.

    Example:
        >>> comparator = ModelComparator()
        >>> comparator.add_profile(profile_a)
        >>> comparator.add_profile(profile_b)
        >>> result = comparator.compare()
    """

    def __init__(self):
        """Initialize comparator."""
        self._profiles: Dict[str, ModelProfile] = {}
        self._weights: Dict[str, float] = {
            ComparisonMetric.ACCURACY.value: 1.0,
            ComparisonMetric.LATENCY.value: 0.5,
            ComparisonMetric.ERROR_RATE.value: 0.8,
        }

    def add_profile(self, profile: ModelProfile) -> "ModelComparator":
        """Add a model profile.

        Args:
            profile: The model profile.

        Returns:
            Self for chaining.
        """
        self._profiles[profile.model_name] = profile
        return self

    def set_weight(self, metric: str, weight: float) -> "ModelComparator":
        """Set weight for a metric in overall scoring.

        Args:
            metric: Metric name.
            weight: Weight value (0-1).

        Returns:
            Self for chaining.
        """
        self._weights[metric] = weight
        return self

    def compare(
        self,
        metrics: Optional[List[str]] = None,
        higher_is_better: Optional[Dict[str, bool]] = None,
    ) -> ComparisonResult:
        """Compare all models.

        Args:
            metrics: Specific metrics to compare (default: all available).
            higher_is_better: Dict mapping metric to whether higher is better.

        Returns:
            ComparisonResult object.
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
        rankings: Dict[str, List[str]] = {}
        differences: Dict[str, Dict[str, float]] = {}

        for metric in metrics:
            # Get values for each model
            values: Dict[str, float] = {}
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
        scores: Dict[str, float] = {name: 0.0 for name in model_names}

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

        return ComparisonResult(
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
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Compare models on a single metric.

        Args:
            metric: Metric name.
            higher_is_better: Whether higher values are better.

        Returns:
            Tuple of (winner, ranked list of (model, value)).
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
        """Generate a detailed comparison report.

        Returns:
            Markdown-formatted report.
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
            for name in self._profiles.keys():
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
    """Detailed latency profiling data.

    Attributes:
        total_ms: Total latency in milliseconds.
        first_token_ms: Time to first token (streaming).
        tokens_per_second: Generation speed.
        overhead_ms: Non-generation overhead.
    """

    total_ms: float
    first_token_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    overhead_ms: Optional[float] = None


@dataclass
class CostEstimate:
    """Cost estimation for model usage.

    Attributes:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        input_cost: Cost for input tokens.
        output_cost: Cost for output tokens.
        total_cost: Total cost.
        currency: Currency code.
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
        """Calculate cost estimate.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            input_price_per_1k: Price per 1000 input tokens.
            output_price_per_1k: Price per 1000 output tokens.

        Returns:
            CostEstimate object.
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


class CostCalculator:
    """Calculate and compare costs across models.

    Example:
        >>> calc = CostCalculator()
        >>> calc.set_pricing("gpt-4", 0.03, 0.06)
        >>> calc.set_pricing("claude-3", 0.015, 0.075)
        >>> cost = calc.estimate("gpt-4", 1000, 500)
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
        """Initialize with default pricing."""
        self._pricing: Dict[str, Tuple[float, float]] = dict(self.DEFAULT_PRICING)

    def set_pricing(
        self,
        model: str,
        input_per_1k: float,
        output_per_1k: float,
    ) -> "CostCalculator":
        """Set pricing for a model.

        Args:
            model: Model name.
            input_per_1k: Price per 1000 input tokens.
            output_per_1k: Price per 1000 output tokens.

        Returns:
            Self for chaining.
        """
        self._pricing[model] = (input_per_1k, output_per_1k)
        return self

    def estimate(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostEstimate:
        """Estimate cost for a model.

        Args:
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            CostEstimate object.

        Raises:
            KeyError: If model pricing not found.
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
        models: Optional[List[str]] = None,
    ) -> Dict[str, CostEstimate]:
        """Compare costs across models.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            models: Models to compare (default: all known).

        Returns:
            Dict mapping model to cost estimate.
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
        models: Optional[List[str]] = None,
    ) -> Tuple[str, CostEstimate]:
        """Find the cheapest model.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            models: Models to consider.

        Returns:
            Tuple of (model_name, cost_estimate).
        """
        costs = self.compare_costs(input_tokens, output_tokens, models)
        cheapest = min(costs.items(), key=lambda x: x[1].total_cost)
        return cheapest


@dataclass
class QualityMetrics:
    """Quality metrics for model outputs.

    Attributes:
        coherence: Output coherence score (0-1).
        relevance: Relevance to input (0-1).
        accuracy: Factual accuracy (0-1).
        completeness: Completeness of response (0-1).
        fluency: Language fluency (0-1).
        overall: Overall quality score (0-1).
    """

    coherence: float = 0.0
    relevance: float = 0.0
    accuracy: float = 0.0
    completeness: float = 0.0
    fluency: float = 0.0
    overall: float = 0.0

    def compute_overall(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute overall score from components.

        Args:
            weights: Optional weights for each component.

        Returns:
            Overall score.
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
    """Track performance metrics over time.

    Example:
        >>> tracker = PerformanceTracker("gpt-4")
        >>> tracker.record_latency(150.5)
        >>> tracker.record_success(True)
        >>> summary = tracker.get_summary()
    """

    def __init__(self, model_name: str):
        """Initialize tracker.

        Args:
            model_name: Name of the model being tracked.
        """
        self.model_name = model_name
        self._latencies: List[float] = []
        self._successes: List[bool] = []
        self._token_counts: List[Tuple[int, int]] = []  # (input, output)
        self._timestamps: List[datetime] = []
        self._errors: List[str] = []

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement.

        Args:
            latency_ms: Latency in milliseconds.
        """
        self._latencies.append(latency_ms)
        self._timestamps.append(datetime.now())

    def record_success(self, success: bool, error: Optional[str] = None) -> None:
        """Record a success/failure.

        Args:
            success: Whether the call succeeded.
            error: Error message if failed.
        """
        self._successes.append(success)
        if not success and error:
            self._errors.append(error)

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
        """
        self._token_counts.append((input_tokens, output_tokens))

    def get_summary(self) -> ModelProfile:
        """Get performance summary as ModelProfile.

        Returns:
            ModelProfile with metrics.
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
        """Reset all tracked data."""
        self._latencies.clear()
        self._successes.clear()
        self._token_counts.clear()
        self._timestamps.clear()
        self._errors.clear()


def create_comparison_table(
    profiles: List[ModelProfile],
    metrics: Optional[List[str]] = None,
) -> str:
    """Create a markdown comparison table.

    Args:
        profiles: List of model profiles.
        metrics: Specific metrics to include.

    Returns:
        Markdown table string.
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
    profiles: List[ModelProfile],
    metric: str,
    higher_is_better: bool = True,
) -> List[Tuple[str, float]]:
    """Rank models by a specific metric.

    Args:
        profiles: List of model profiles.
        metric: Metric to rank by.
        higher_is_better: Whether higher values are better.

    Returns:
        List of (model_name, value) tuples, ranked.
    """
    values = []
    for profile in profiles:
        if metric in profile.metrics:
            values.append((profile.model_name, profile.metrics[metric].mean))

    return sorted(values, key=lambda x: x[1], reverse=higher_is_better)
