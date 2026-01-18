"""Response latency and throughput profiling for LLM evaluation.

This module provides comprehensive tools for measuring and analyzing
the performance characteristics of language model responses:

- Latency measurement (time to first token, total response time)
- Throughput analysis (tokens per second)
- Statistical profiling (percentiles, distributions)
- Performance comparison across models
- Bottleneck detection and optimization hints
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class LatencyMetric(Enum):
    """Types of latency metrics."""

    TIME_TO_FIRST_TOKEN = "time_to_first_token"
    TOTAL_RESPONSE_TIME = "total_response_time"
    INTER_TOKEN_LATENCY = "inter_token_latency"
    PROCESSING_TIME = "processing_time"
    NETWORK_LATENCY = "network_latency"


class ThroughputMetric(Enum):
    """Types of throughput metrics."""

    TOKENS_PER_SECOND = "tokens_per_second"
    CHARACTERS_PER_SECOND = "characters_per_second"
    REQUESTS_PER_MINUTE = "requests_per_minute"
    WORDS_PER_SECOND = "words_per_second"


class PerformanceLevel(Enum):
    """Performance level classification."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class LatencyMeasurement:
    """A single latency measurement."""

    metric: LatencyMetric
    value_ms: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric.value,
            "value_ms": self.value_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ThroughputMeasurement:
    """A single throughput measurement."""

    metric: ThroughputMetric
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class LatencyStats:
    """Statistical summary of latency measurements."""

    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    sample_count: int

    @property
    def coefficient_of_variation(self) -> float:
        """Calculate coefficient of variation (std/mean)."""
        if self.mean_ms == 0:
            return 0.0
        return self.std_ms / self.mean_ms

    @property
    def is_consistent(self) -> bool:
        """Check if latency is consistent (low variation)."""
        return self.coefficient_of_variation < 0.3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "sample_count": self.sample_count,
            "coefficient_of_variation": self.coefficient_of_variation,
            "is_consistent": self.is_consistent,
        }


@dataclass
class ThroughputStats:
    """Statistical summary of throughput measurements."""

    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    sample_count: int
    unit: str

    @property
    def sustained_rate(self) -> float:
        """Estimate sustainable throughput (p10 approximation)."""
        # Conservative estimate using lower bound
        return max(0, self.mean - self.std)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "sample_count": self.sample_count,
            "unit": self.unit,
            "sustained_rate": self.sustained_rate,
        }


@dataclass
class ResponseProfile:
    """Complete profile for a single response."""

    prompt_tokens: int
    completion_tokens: int
    time_to_first_token_ms: float | None
    total_time_ms: float
    tokens_per_second: float
    timestamp: float = field(default_factory=time.time)
    model_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def inter_token_latency_ms(self) -> float:
        """Calculate average inter-token latency."""
        if self.completion_tokens <= 1:
            return 0.0
        if self.time_to_first_token_ms is None:
            return self.total_time_ms / self.completion_tokens
        generation_time = self.total_time_ms - self.time_to_first_token_ms
        return generation_time / max(1, self.completion_tokens - 1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "total_time_ms": self.total_time_ms,
            "tokens_per_second": self.tokens_per_second,
            "inter_token_latency_ms": self.inter_token_latency_ms,
            "timestamp": self.timestamp,
            "model_id": self.model_id,
            "metadata": self.metadata,
        }


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""

    model_id: str
    latency_stats: dict[LatencyMetric, LatencyStats]
    throughput_stats: dict[ThroughputMetric, ThroughputStats]
    response_profiles: list[ResponseProfile]
    performance_level: PerformanceLevel
    bottlenecks: list[str]
    recommendations: list[str]

    @property
    def overall_score(self) -> float:
        """Calculate overall performance score (0-1)."""
        scores = []

        # Latency score (lower is better)
        if LatencyMetric.TOTAL_RESPONSE_TIME in self.latency_stats:
            lat = self.latency_stats[LatencyMetric.TOTAL_RESPONSE_TIME]
            # Score based on mean latency (1000ms = 0.5 score)
            lat_score = max(0, 1 - lat.mean_ms / 2000)
            scores.append(lat_score)

        # Throughput score (higher is better)
        if ThroughputMetric.TOKENS_PER_SECOND in self.throughput_stats:
            tps = self.throughput_stats[ThroughputMetric.TOKENS_PER_SECOND]
            # Score based on tokens/sec (50 tps = 0.5 score)
            tps_score = min(1, tps.mean / 100)
            scores.append(tps_score)

        # Consistency score
        if scores:
            return statistics.mean(scores)
        return 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "latency_stats": {k.value: v.to_dict() for k, v in self.latency_stats.items()},
            "throughput_stats": {k.value: v.to_dict() for k, v in self.throughput_stats.items()},
            "n_profiles": len(self.response_profiles),
            "performance_level": self.performance_level.value,
            "overall_score": self.overall_score,
            "bottlenecks": self.bottlenecks,
            "recommendations": self.recommendations,
        }


class LatencyTimer:
    """High-precision latency timer for measuring response times."""

    def __init__(self):
        """Initialize timer."""
        self._start_time: float | None = None
        self._first_token_time: float | None = None
        self._end_time: float | None = None
        self._token_times: list[float] = []

    def start(self) -> LatencyTimer:
        """Start the timer."""
        self._start_time = time.perf_counter()
        return self

    def mark_first_token(self) -> None:
        """Mark when first token is received."""
        if self._first_token_time is None:
            self._first_token_time = time.perf_counter()

    def mark_token(self) -> None:
        """Mark when a token is received."""
        self._token_times.append(time.perf_counter())

    def stop(self) -> None:
        """Stop the timer."""
        self._end_time = time.perf_counter()

    @property
    def time_to_first_token_ms(self) -> float | None:
        """Get time to first token in milliseconds."""
        if self._start_time is None or self._first_token_time is None:
            return None
        return (self._first_token_time - self._start_time) * 1000

    @property
    def total_time_ms(self) -> float | None:
        """Get total time in milliseconds."""
        if self._start_time is None:
            return None
        end = self._end_time or time.perf_counter()
        return (end - self._start_time) * 1000

    @property
    def inter_token_latencies_ms(self) -> list[float]:
        """Get inter-token latencies in milliseconds."""
        if len(self._token_times) < 2:
            return []
        latencies = []
        for i in range(1, len(self._token_times)):
            latency = (self._token_times[i] - self._token_times[i - 1]) * 1000
            latencies.append(latency)
        return latencies

    def __enter__(self) -> LatencyTimer:
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class LatencyProfiler:
    """Profile latency characteristics of model responses."""

    def __init__(self, model_id: str | None = None):
        """Initialize profiler.

        Args:
            model_id: Optional model identifier
        """
        self.model_id = model_id
        self._measurements: dict[LatencyMetric, list[float]] = defaultdict(list)

    def record(
        self,
        metric: LatencyMetric,
        value_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> LatencyMeasurement:
        """Record a latency measurement.

        Args:
            metric: Type of latency metric
            value_ms: Latency value in milliseconds
            metadata: Optional metadata

        Returns:
            LatencyMeasurement object
        """
        self._measurements[metric].append(value_ms)
        return LatencyMeasurement(
            metric=metric,
            value_ms=value_ms,
            metadata=metadata or {},
        )

    def get_stats(self, metric: LatencyMetric) -> LatencyStats:
        """Get statistics for a specific metric.

        Args:
            metric: Latency metric type

        Returns:
            LatencyStats object
        """
        values = self._measurements.get(metric, [])
        if not values:
            return LatencyStats(
                mean_ms=0,
                median_ms=0,
                std_ms=0,
                min_ms=0,
                max_ms=0,
                p50_ms=0,
                p90_ms=0,
                p95_ms=0,
                p99_ms=0,
                sample_count=0,
            )

        sorted_values = sorted(values)
        n = len(sorted_values)

        return LatencyStats(
            mean_ms=statistics.mean(values),
            median_ms=statistics.median(values),
            std_ms=statistics.stdev(values) if n > 1 else 0,
            min_ms=min(values),
            max_ms=max(values),
            p50_ms=self._percentile(sorted_values, 50),
            p90_ms=self._percentile(sorted_values, 90),
            p95_ms=self._percentile(sorted_values, 95),
            p99_ms=self._percentile(sorted_values, 99),
            sample_count=n,
        )

    def get_all_stats(self) -> dict[LatencyMetric, LatencyStats]:
        """Get statistics for all recorded metrics.

        Returns:
            Dictionary of metric to stats
        """
        return {metric: self.get_stats(metric) for metric in self._measurements}

    @staticmethod
    def _percentile(sorted_values: list[float], p: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        n = len(sorted_values)
        idx = (p / 100) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        weight = idx - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    def clear(self) -> None:
        """Clear all measurements."""
        self._measurements.clear()


class ThroughputProfiler:
    """Profile throughput characteristics of model responses."""

    def __init__(self, model_id: str | None = None):
        """Initialize profiler.

        Args:
            model_id: Optional model identifier
        """
        self.model_id = model_id
        self._measurements: dict[ThroughputMetric, list[float]] = defaultdict(list)
        self._request_times: list[float] = []

    def record(
        self,
        metric: ThroughputMetric,
        value: float,
        metadata: dict[str, Any] | None = None,
    ) -> ThroughputMeasurement:
        """Record a throughput measurement.

        Args:
            metric: Type of throughput metric
            value: Throughput value
            metadata: Optional metadata

        Returns:
            ThroughputMeasurement object
        """
        self._measurements[metric].append(value)
        return ThroughputMeasurement(
            metric=metric,
            value=value,
            unit=self._get_unit(metric),
            metadata=metadata or {},
        )

    def record_request(self) -> None:
        """Record a request timestamp for RPM calculation."""
        self._request_times.append(time.time())

    def get_stats(self, metric: ThroughputMetric) -> ThroughputStats:
        """Get statistics for a specific metric.

        Args:
            metric: Throughput metric type

        Returns:
            ThroughputStats object
        """
        values = self._measurements.get(metric, [])
        if not values:
            return ThroughputStats(
                mean=0,
                median=0,
                std=0,
                min_val=0,
                max_val=0,
                sample_count=0,
                unit=self._get_unit(metric),
            )

        return ThroughputStats(
            mean=statistics.mean(values),
            median=statistics.median(values),
            std=statistics.stdev(values) if len(values) > 1 else 0,
            min_val=min(values),
            max_val=max(values),
            sample_count=len(values),
            unit=self._get_unit(metric),
        )

    def calculate_rpm(self, window_seconds: float = 60.0) -> float:
        """Calculate requests per minute over a time window.

        Args:
            window_seconds: Time window in seconds

        Returns:
            Requests per minute
        """
        if not self._request_times:
            return 0.0

        now = time.time()
        cutoff = now - window_seconds
        recent_requests = [t for t in self._request_times if t > cutoff]

        if len(recent_requests) < 2:
            return len(recent_requests) * (60.0 / window_seconds)

        # Calculate actual rate
        time_span = recent_requests[-1] - recent_requests[0]
        if time_span == 0:
            return 0.0
        return (len(recent_requests) - 1) / time_span * 60

    def get_all_stats(self) -> dict[ThroughputMetric, ThroughputStats]:
        """Get statistics for all recorded metrics.

        Returns:
            Dictionary of metric to stats
        """
        return {metric: self.get_stats(metric) for metric in self._measurements}

    @staticmethod
    def _get_unit(metric: ThroughputMetric) -> str:
        """Get unit string for metric."""
        units = {
            ThroughputMetric.TOKENS_PER_SECOND: "tokens/s",
            ThroughputMetric.CHARACTERS_PER_SECOND: "chars/s",
            ThroughputMetric.REQUESTS_PER_MINUTE: "req/min",
            ThroughputMetric.WORDS_PER_SECOND: "words/s",
        }
        return units.get(metric, "unit")

    def clear(self) -> None:
        """Clear all measurements."""
        self._measurements.clear()
        self._request_times.clear()


class ResponseProfiler:
    """Profile complete response characteristics."""

    def __init__(
        self,
        model_id: str | None = None,
        token_estimator: Callable[[str], int] | None = None,
    ):
        """Initialize profiler.

        Args:
            model_id: Optional model identifier
            token_estimator: Function to estimate token count from text
        """
        self.model_id = model_id
        self._token_estimator = token_estimator or self._default_token_estimator
        self._profiles: list[ResponseProfile] = []
        self._latency_profiler = LatencyProfiler(model_id)
        self._throughput_profiler = ThroughputProfiler(model_id)

    @staticmethod
    def _default_token_estimator(text: str) -> int:
        """Simple token estimation (words * 1.3)."""
        return int(len(text.split()) * 1.3)

    def profile_response(
        self,
        prompt: str,
        response: str,
        total_time_ms: float,
        time_to_first_token_ms: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResponseProfile:
        """Profile a single response.

        Args:
            prompt: The input prompt
            response: The model response
            total_time_ms: Total response time in milliseconds
            time_to_first_token_ms: Time to first token in milliseconds
            prompt_tokens: Number of prompt tokens (estimated if not provided)
            completion_tokens: Number of completion tokens (estimated if not provided)
            metadata: Optional metadata

        Returns:
            ResponseProfile object
        """
        prompt_toks = prompt_tokens or self._token_estimator(prompt)
        completion_toks = completion_tokens or self._token_estimator(response)

        # Calculate tokens per second
        tokens_per_second = completion_toks / total_time_ms * 1000 if total_time_ms > 0 else 0.0

        profile = ResponseProfile(
            prompt_tokens=prompt_toks,
            completion_tokens=completion_toks,
            time_to_first_token_ms=time_to_first_token_ms,
            total_time_ms=total_time_ms,
            tokens_per_second=tokens_per_second,
            model_id=self.model_id,
            metadata=metadata or {},
        )

        self._profiles.append(profile)

        # Record to sub-profilers
        self._latency_profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, total_time_ms)
        if time_to_first_token_ms is not None:
            self._latency_profiler.record(LatencyMetric.TIME_TO_FIRST_TOKEN, time_to_first_token_ms)
        self._throughput_profiler.record(ThroughputMetric.TOKENS_PER_SECOND, tokens_per_second)
        self._throughput_profiler.record_request()

        return profile

    def generate_report(self) -> PerformanceReport:
        """Generate comprehensive performance report.

        Returns:
            PerformanceReport object
        """
        latency_stats = self._latency_profiler.get_all_stats()
        throughput_stats = self._throughput_profiler.get_all_stats()

        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(latency_stats, throughput_stats)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            latency_stats, throughput_stats, bottlenecks
        )

        # Classify performance level
        performance_level = self._classify_performance(latency_stats, throughput_stats)

        return PerformanceReport(
            model_id=self.model_id or "unknown",
            latency_stats=latency_stats,
            throughput_stats=throughput_stats,
            response_profiles=self._profiles.copy(),
            performance_level=performance_level,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )

    def _detect_bottlenecks(
        self,
        latency_stats: dict[LatencyMetric, LatencyStats],
        throughput_stats: dict[ThroughputMetric, ThroughputStats],
    ) -> list[str]:
        """Detect performance bottlenecks."""
        bottlenecks = []

        # Check TTFT
        if LatencyMetric.TIME_TO_FIRST_TOKEN in latency_stats:
            ttft = latency_stats[LatencyMetric.TIME_TO_FIRST_TOKEN]
            if ttft.mean_ms > 500:
                bottlenecks.append(f"High time to first token (mean: {ttft.mean_ms:.0f}ms)")
            if ttft.p99_ms > 2000:
                bottlenecks.append(f"TTFT tail latency issues (p99: {ttft.p99_ms:.0f}ms)")

        # Check total latency
        if LatencyMetric.TOTAL_RESPONSE_TIME in latency_stats:
            total = latency_stats[LatencyMetric.TOTAL_RESPONSE_TIME]
            if not total.is_consistent:
                bottlenecks.append(
                    f"Inconsistent response times (CV: {total.coefficient_of_variation:.2f})"
                )

        # Check throughput
        if ThroughputMetric.TOKENS_PER_SECOND in throughput_stats:
            tps = throughput_stats[ThroughputMetric.TOKENS_PER_SECOND]
            if tps.mean < 20:
                bottlenecks.append(f"Low throughput (mean: {tps.mean:.1f} tokens/s)")

        return bottlenecks

    def _generate_recommendations(
        self,
        latency_stats: dict[LatencyMetric, LatencyStats],
        throughput_stats: dict[ThroughputMetric, ThroughputStats],
        bottlenecks: list[str],
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if any("time to first token" in b.lower() for b in bottlenecks):
            recommendations.append("Consider using streaming to improve perceived responsiveness")
            recommendations.append("Check network latency and consider edge deployment")

        if any("inconsistent" in b.lower() for b in bottlenecks):
            recommendations.append("Implement request batching to stabilize latency")
            recommendations.append("Consider using dedicated inference endpoints")

        if any("low throughput" in b.lower() for b in bottlenecks):
            recommendations.append("Optimize prompt length to improve generation speed")
            recommendations.append("Consider using a faster model variant")

        if not recommendations:
            recommendations.append("Performance is within acceptable parameters")

        return recommendations

    def _classify_performance(
        self,
        latency_stats: dict[LatencyMetric, LatencyStats],
        throughput_stats: dict[ThroughputMetric, ThroughputStats],
    ) -> PerformanceLevel:
        """Classify overall performance level."""
        scores = []

        if LatencyMetric.TOTAL_RESPONSE_TIME in latency_stats:
            lat = latency_stats[LatencyMetric.TOTAL_RESPONSE_TIME]
            if lat.mean_ms < 200:
                scores.append(5)  # excellent
            elif lat.mean_ms < 500:
                scores.append(4)  # good
            elif lat.mean_ms < 1000:
                scores.append(3)  # acceptable
            elif lat.mean_ms < 2000:
                scores.append(2)  # poor
            else:
                scores.append(1)  # critical

        if ThroughputMetric.TOKENS_PER_SECOND in throughput_stats:
            tps = throughput_stats[ThroughputMetric.TOKENS_PER_SECOND]
            if tps.mean >= 100:
                scores.append(5)
            elif tps.mean >= 50:
                scores.append(4)
            elif tps.mean >= 25:
                scores.append(3)
            elif tps.mean >= 10:
                scores.append(2)
            else:
                scores.append(1)

        if not scores:
            return PerformanceLevel.ACCEPTABLE

        avg_score = statistics.mean(scores)
        if avg_score >= 4.5:
            return PerformanceLevel.EXCELLENT
        elif avg_score >= 3.5:
            return PerformanceLevel.GOOD
        elif avg_score >= 2.5:
            return PerformanceLevel.ACCEPTABLE
        elif avg_score >= 1.5:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL

    def clear(self) -> None:
        """Clear all profiles."""
        self._profiles.clear()
        self._latency_profiler.clear()
        self._throughput_profiler.clear()


class PerformanceComparator:
    """Compare performance across multiple models or configurations."""

    def __init__(self):
        """Initialize comparator."""
        self._reports: dict[str, PerformanceReport] = {}

    def add_report(self, name: str, report: PerformanceReport) -> None:
        """Add a performance report for comparison.

        Args:
            name: Identifier for this configuration
            report: PerformanceReport to compare
        """
        self._reports[name] = report

    def compare_latency(
        self, metric: LatencyMetric = LatencyMetric.TOTAL_RESPONSE_TIME
    ) -> dict[str, dict[str, Any]]:
        """Compare latency across all reports.

        Args:
            metric: Latency metric to compare

        Returns:
            Dictionary with comparison results
        """
        results = {}
        for name, report in self._reports.items():
            if metric in report.latency_stats:
                stats = report.latency_stats[metric]
                results[name] = {
                    "mean_ms": stats.mean_ms,
                    "median_ms": stats.median_ms,
                    "p95_ms": stats.p95_ms,
                    "p99_ms": stats.p99_ms,
                    "consistency": stats.is_consistent,
                }

        # Add ranking
        if results:
            sorted_by_mean = sorted(results.items(), key=lambda x: x[1]["mean_ms"])
            for rank, (name, _) in enumerate(sorted_by_mean, 1):
                results[name]["rank"] = rank

        return results

    def compare_throughput(
        self, metric: ThroughputMetric = ThroughputMetric.TOKENS_PER_SECOND
    ) -> dict[str, dict[str, Any]]:
        """Compare throughput across all reports.

        Args:
            metric: Throughput metric to compare

        Returns:
            Dictionary with comparison results
        """
        results = {}
        for name, report in self._reports.items():
            if metric in report.throughput_stats:
                stats = report.throughput_stats[metric]
                results[name] = {
                    "mean": stats.mean,
                    "median": stats.median,
                    "sustained_rate": stats.sustained_rate,
                    "unit": stats.unit,
                }

        # Add ranking (higher is better for throughput)
        if results:
            sorted_by_mean = sorted(results.items(), key=lambda x: x[1]["mean"], reverse=True)
            for rank, (name, _) in enumerate(sorted_by_mean, 1):
                results[name]["rank"] = rank

        return results

    def get_best_performer(self, by: str = "latency") -> tuple[str, PerformanceReport]:
        """Get the best performing configuration.

        Args:
            by: Criterion ("latency" or "throughput")

        Returns:
            Tuple of (name, report) for best performer
        """
        if not self._reports:
            raise ValueError("No reports to compare")

        if by == "latency":
            comparison = self.compare_latency()
            if not comparison:
                return list(self._reports.items())[0]
            best_name = min(comparison.items(), key=lambda x: x[1]["mean_ms"])[0]
        else:
            comparison = self.compare_throughput()
            if not comparison:
                return list(self._reports.items())[0]
            best_name = max(comparison.items(), key=lambda x: x[1]["mean"])[0]

        return best_name, self._reports[best_name]

    def generate_comparison_report(self) -> dict[str, Any]:
        """Generate comprehensive comparison report.

        Returns:
            Dictionary with full comparison data
        """
        return {
            "n_configurations": len(self._reports),
            "configurations": list(self._reports.keys()),
            "latency_comparison": self.compare_latency(),
            "throughput_comparison": self.compare_throughput(),
            "performance_levels": {
                name: report.performance_level.value for name, report in self._reports.items()
            },
            "overall_scores": {
                name: report.overall_score for name, report in self._reports.items()
            },
        }


class StreamingProfiler:
    """Profile streaming response characteristics."""

    def __init__(self, model_id: str | None = None):
        """Initialize profiler.

        Args:
            model_id: Optional model identifier
        """
        self.model_id = model_id
        self._timer: LatencyTimer | None = None
        self._tokens_received: int = 0
        self._chunk_sizes: list[int] = []

    def start_stream(self) -> None:
        """Start profiling a streaming response."""
        self._timer = LatencyTimer()
        self._timer.start()
        self._tokens_received = 0
        self._chunk_sizes.clear()

    def record_chunk(self, chunk: str, is_first: bool = False) -> None:
        """Record a streaming chunk.

        Args:
            chunk: The received chunk
            is_first: Whether this is the first chunk
        """
        if self._timer is None:
            return

        if is_first:
            self._timer.mark_first_token()

        self._timer.mark_token()
        chunk_tokens = len(chunk.split())  # Approximate
        self._tokens_received += chunk_tokens
        self._chunk_sizes.append(len(chunk))

    def end_stream(self) -> dict[str, Any]:
        """End stream profiling and return metrics.

        Returns:
            Dictionary with streaming metrics
        """
        if self._timer is None:
            return {}

        self._timer.stop()

        inter_token_latencies = self._timer.inter_token_latencies_ms

        return {
            "time_to_first_token_ms": self._timer.time_to_first_token_ms,
            "total_time_ms": self._timer.total_time_ms,
            "tokens_received": self._tokens_received,
            "n_chunks": len(self._chunk_sizes),
            "avg_chunk_size": (statistics.mean(self._chunk_sizes) if self._chunk_sizes else 0),
            "avg_inter_token_latency_ms": (
                statistics.mean(inter_token_latencies) if inter_token_latencies else 0
            ),
            "max_inter_token_latency_ms": (
                max(inter_token_latencies) if inter_token_latencies else 0
            ),
        }


# Convenience functions


def create_timer() -> LatencyTimer:
    """Create a new latency timer.

    Returns:
        LatencyTimer instance
    """
    return LatencyTimer()


def measure_latency(
    func: Callable[[], Any],
    warmup_runs: int = 0,
    measurement_runs: int = 1,
) -> LatencyStats:
    """Measure latency of a function.

    Args:
        func: Function to measure
        warmup_runs: Number of warmup runs (not counted)
        measurement_runs: Number of measurement runs

    Returns:
        LatencyStats object
    """
    # Warmup
    for _ in range(warmup_runs):
        func()

    # Measure
    measurements = []
    for _ in range(measurement_runs):
        timer = LatencyTimer()
        with timer:
            func()
        if timer.total_time_ms is not None:
            measurements.append(timer.total_time_ms)

    if not measurements:
        return LatencyStats(
            mean_ms=0,
            median_ms=0,
            std_ms=0,
            min_ms=0,
            max_ms=0,
            p50_ms=0,
            p90_ms=0,
            p95_ms=0,
            p99_ms=0,
            sample_count=0,
        )

    sorted_vals = sorted(measurements)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        idx = (p / 100) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        weight = idx - lower
        return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight

    return LatencyStats(
        mean_ms=statistics.mean(measurements),
        median_ms=statistics.median(measurements),
        std_ms=statistics.stdev(measurements) if n > 1 else 0,
        min_ms=min(measurements),
        max_ms=max(measurements),
        p50_ms=percentile(50),
        p90_ms=percentile(90),
        p95_ms=percentile(95),
        p99_ms=percentile(99),
        sample_count=n,
    )


def calculate_throughput(
    tokens: int,
    time_ms: float,
) -> float:
    """Calculate throughput in tokens per second.

    Args:
        tokens: Number of tokens
        time_ms: Time in milliseconds

    Returns:
        Tokens per second
    """
    if time_ms <= 0:
        return 0.0
    return (tokens / time_ms) * 1000


def profile_response(
    prompt: str,
    response: str,
    time_ms: float,
    model_id: str | None = None,
) -> ResponseProfile:
    """Quick profile of a response.

    Args:
        prompt: Input prompt
        response: Model response
        time_ms: Total time in milliseconds
        model_id: Optional model identifier

    Returns:
        ResponseProfile object
    """
    profiler = ResponseProfiler(model_id)
    return profiler.profile_response(prompt, response, time_ms)


def quick_performance_check(
    response_times_ms: list[float],
    tokens_per_response: list[int] | None = None,
) -> dict[str, Any]:
    """Quick performance check from response times.

    Args:
        response_times_ms: List of response times in milliseconds
        tokens_per_response: Optional list of token counts per response

    Returns:
        Dictionary with performance summary
    """
    if not response_times_ms:
        return {
            "n_responses": 0,
            "latency": None,
            "throughput": None,
            "performance_level": "unknown",
        }

    # Calculate latency stats
    sorted_times = sorted(response_times_ms)
    n = len(sorted_times)

    def percentile(p: float) -> float:
        idx = (p / 100) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        weight = idx - lower
        return sorted_times[lower] * (1 - weight) + sorted_times[upper] * weight

    latency = {
        "mean_ms": statistics.mean(response_times_ms),
        "median_ms": statistics.median(response_times_ms),
        "p95_ms": percentile(95),
        "p99_ms": percentile(99),
    }

    # Calculate throughput if token counts provided
    throughput = None
    if tokens_per_response and len(tokens_per_response) == len(response_times_ms):
        tps_values = [
            calculate_throughput(t, ms) for t, ms in zip(tokens_per_response, response_times_ms)
        ]
        throughput = {
            "mean_tps": statistics.mean(tps_values),
            "median_tps": statistics.median(tps_values),
        }

    # Classify performance
    if latency["mean_ms"] < 200:
        level = "excellent"
    elif latency["mean_ms"] < 500:
        level = "good"
    elif latency["mean_ms"] < 1000:
        level = "acceptable"
    elif latency["mean_ms"] < 2000:
        level = "poor"
    else:
        level = "critical"

    return {
        "n_responses": n,
        "latency": latency,
        "throughput": throughput,
        "performance_level": level,
    }
