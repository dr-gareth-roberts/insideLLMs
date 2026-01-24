"""Response latency and throughput profiling for LLM evaluation.

This module provides comprehensive tools for measuring and analyzing
the performance characteristics of language model responses. It enables
developers to identify bottlenecks, compare model performance, and
optimize LLM-powered applications for production workloads.

Key Features
------------
- **Latency Measurement**: Time to first token (TTFT), total response time,
  inter-token latency, and network latency tracking.
- **Throughput Analysis**: Tokens per second, characters per second,
  requests per minute, and words per second metrics.
- **Statistical Profiling**: Mean, median, standard deviation, and
  percentile calculations (p50, p90, p95, p99) for all metrics.
- **Performance Comparison**: Side-by-side comparison of multiple models
  or configurations with automatic ranking.
- **Streaming Support**: Specialized profiling for streaming responses
  with chunk-level metrics.
- **Bottleneck Detection**: Automatic identification of performance issues
  with actionable recommendations.

Classes
-------
LatencyMetric
    Enumeration of latency metric types.
ThroughputMetric
    Enumeration of throughput metric types.
PerformanceLevel
    Classification levels for overall performance.
LatencyMeasurement
    Single latency measurement data container.
ThroughputMeasurement
    Single throughput measurement data container.
LatencyStats
    Statistical summary of latency measurements.
ThroughputStats
    Statistical summary of throughput measurements.
ResponseProfile
    Complete profile for a single LLM response.
PerformanceReport
    Comprehensive performance analysis report.
LatencyTimer
    High-precision timer for measuring response times.
LatencyProfiler
    Profiler for collecting and analyzing latency data.
ThroughputProfiler
    Profiler for collecting and analyzing throughput data.
ResponseProfiler
    Complete response profiling with latency and throughput.
PerformanceComparator
    Compare performance across multiple configurations.
StreamingProfiler
    Specialized profiler for streaming responses.

Examples
--------
Basic latency measurement using the timer context manager:

>>> from insideLLMs.latency import LatencyTimer
>>> import time
>>> timer = LatencyTimer()
>>> with timer:
...     time.sleep(0.1)  # Simulate API call
>>> print(f"Elapsed: {timer.total_time_ms:.1f}ms")
Elapsed: 100.0ms

Profiling multiple responses and generating a report:

>>> from insideLLMs.latency import ResponseProfiler
>>> profiler = ResponseProfiler(model_id="gpt-4")
>>> for i in range(5):
...     profile = profiler.profile_response(
...         prompt="What is AI?",
...         response="AI is artificial intelligence...",
...         total_time_ms=250 + i * 50,
...         time_to_first_token_ms=50 + i * 10
...     )
>>> report = profiler.generate_report()
>>> print(f"Performance: {report.performance_level.value}")
Performance: good

Comparing multiple models:

>>> from insideLLMs.latency import PerformanceComparator, ResponseProfiler
>>> comparator = PerformanceComparator()
>>> for model in ["gpt-4", "claude-3"]:
...     profiler = ResponseProfiler(model_id=model)
...     # ... record responses ...
...     comparator.add_report(model, profiler.generate_report())
>>> best_name, best_report = comparator.get_best_performer(by="latency")
>>> print(f"Best model: {best_name}")
Best model: claude-3

Quick performance check from raw timing data:

>>> from insideLLMs.latency import quick_performance_check
>>> times = [150.0, 200.0, 180.0, 220.0, 190.0]
>>> tokens = [50, 60, 55, 65, 58]
>>> result = quick_performance_check(times, tokens)
>>> print(f"Level: {result['performance_level']}")
Level: good

Measuring function latency with warmup:

>>> from insideLLMs.latency import measure_latency
>>> def expensive_operation():
...     return sum(i**2 for i in range(10000))
>>> stats = measure_latency(expensive_operation, warmup_runs=2, measurement_runs=10)
>>> print(f"Mean: {stats.mean_ms:.2f}ms, p99: {stats.p99_ms:.2f}ms")
Mean: 1.23ms, p99: 1.45ms

Notes
-----
- All time measurements use `time.perf_counter()` for high precision.
- Token estimation uses a simple heuristic (words * 1.3) when exact
  counts are not provided. For production use, provide actual token counts.
- Performance classifications are based on typical LLM response times:
  - Excellent: < 200ms mean latency
  - Good: < 500ms mean latency
  - Acceptable: < 1000ms mean latency
  - Poor: < 2000ms mean latency
  - Critical: >= 2000ms mean latency
- The `coefficient_of_variation` (CV) metric indicates consistency:
  CV < 0.3 is considered consistent performance.

See Also
--------
insideLLMs.trace : Request tracing and debugging utilities.
insideLLMs.metrics : General metrics collection framework.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class LatencyMetric(Enum):
    """Enumeration of latency metric types for LLM response profiling.

    This enum defines the various latency measurements that can be tracked
    when profiling language model responses. Each metric captures a different
    aspect of the response timing characteristics.

    Attributes
    ----------
    TIME_TO_FIRST_TOKEN : str
        Time from request sent to first token received. Critical for
        perceived responsiveness in streaming applications.
    TOTAL_RESPONSE_TIME : str
        Total time from request to complete response. Includes all
        processing, generation, and network time.
    INTER_TOKEN_LATENCY : str
        Average time between consecutive tokens during generation.
        Important for smooth streaming user experiences.
    PROCESSING_TIME : str
        Server-side processing time excluding network latency.
        Useful for isolating model performance from network issues.
    NETWORK_LATENCY : str
        Round-trip network time. Helps identify network bottlenecks
        vs model performance issues.

    Examples
    --------
    Using metrics with the LatencyProfiler:

    >>> from insideLLMs.latency import LatencyProfiler, LatencyMetric
    >>> profiler = LatencyProfiler(model_id="gpt-4")
    >>> profiler.record(LatencyMetric.TIME_TO_FIRST_TOKEN, 45.0)
    LatencyMeasurement(metric=<LatencyMetric.TIME_TO_FIRST_TOKEN: ...>, ...)
    >>> profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, 850.0)
    LatencyMeasurement(metric=<LatencyMetric.TOTAL_RESPONSE_TIME: ...>, ...)

    Iterating over all metric types:

    >>> for metric in LatencyMetric:
    ...     print(f"{metric.name}: {metric.value}")
    TIME_TO_FIRST_TOKEN: time_to_first_token
    TOTAL_RESPONSE_TIME: total_response_time
    INTER_TOKEN_LATENCY: inter_token_latency
    PROCESSING_TIME: processing_time
    NETWORK_LATENCY: network_latency

    Checking metric type in conditional logic:

    >>> metric = LatencyMetric.TIME_TO_FIRST_TOKEN
    >>> if metric == LatencyMetric.TIME_TO_FIRST_TOKEN:
    ...     print("Tracking TTFT for streaming optimization")
    Tracking TTFT for streaming optimization

    See Also
    --------
    ThroughputMetric : Complementary throughput metrics.
    LatencyProfiler : Profiler that uses these metrics.
    """

    TIME_TO_FIRST_TOKEN = "time_to_first_token"
    TOTAL_RESPONSE_TIME = "total_response_time"
    INTER_TOKEN_LATENCY = "inter_token_latency"
    PROCESSING_TIME = "processing_time"
    NETWORK_LATENCY = "network_latency"


class ThroughputMetric(Enum):
    """Enumeration of throughput metric types for LLM performance analysis.

    This enum defines the various throughput measurements that can be tracked
    when profiling language model responses. Throughput metrics measure the
    rate of output generation rather than timing.

    Attributes
    ----------
    TOKENS_PER_SECOND : str
        Rate of token generation. The primary metric for comparing model
        performance across different providers and configurations.
    CHARACTERS_PER_SECOND : str
        Rate of character generation. Useful when token counts are not
        available or for comparing across different tokenizers.
    REQUESTS_PER_MINUTE : str
        Rate of completed requests. Important for capacity planning
        and rate limit management.
    WORDS_PER_SECOND : str
        Rate of word generation. Provides human-interpretable output
        speed, useful for content generation applications.

    Examples
    --------
    Recording throughput measurements:

    >>> from insideLLMs.latency import ThroughputProfiler, ThroughputMetric
    >>> profiler = ThroughputProfiler(model_id="claude-3")
    >>> profiler.record(ThroughputMetric.TOKENS_PER_SECOND, 75.5)
    ThroughputMeasurement(metric=<ThroughputMetric.TOKENS_PER_SECOND: ...>, ...)
    >>> profiler.record(ThroughputMetric.CHARACTERS_PER_SECOND, 320.0)
    ThroughputMeasurement(metric=<ThroughputMetric.CHARACTERS_PER_SECOND: ...>, ...)

    Calculating throughput from response data:

    >>> from insideLLMs.latency import calculate_throughput
    >>> tokens = 150
    >>> time_ms = 2000.0
    >>> tps = calculate_throughput(tokens, time_ms)
    >>> print(f"Throughput: {tps:.1f} tokens/s")
    Throughput: 75.0 tokens/s

    Getting the unit string for display:

    >>> metric = ThroughputMetric.TOKENS_PER_SECOND
    >>> print(f"Measuring: {metric.value}")
    Measuring: tokens_per_second

    See Also
    --------
    LatencyMetric : Complementary latency metrics.
    ThroughputProfiler : Profiler that uses these metrics.
    calculate_throughput : Utility function for throughput calculation.
    """

    TOKENS_PER_SECOND = "tokens_per_second"
    CHARACTERS_PER_SECOND = "characters_per_second"
    REQUESTS_PER_MINUTE = "requests_per_minute"
    WORDS_PER_SECOND = "words_per_second"


class PerformanceLevel(Enum):
    """Performance level classification for LLM response characteristics.

    This enum provides a standardized way to classify overall model performance
    based on latency and throughput metrics. Classifications are designed to
    align with typical user experience expectations for LLM applications.

    Attributes
    ----------
    EXCELLENT : str
        Outstanding performance. Mean latency < 200ms, throughput >= 100 tps.
        Suitable for real-time interactive applications.
    GOOD : str
        Above average performance. Mean latency < 500ms, throughput >= 50 tps.
        Suitable for most production applications.
    ACCEPTABLE : str
        Meets baseline requirements. Mean latency < 1000ms, throughput >= 25 tps.
        May require optimization for latency-sensitive use cases.
    POOR : str
        Below expectations. Mean latency < 2000ms, throughput >= 10 tps.
        Requires investigation and optimization.
    CRITICAL : str
        Severely degraded performance. Mean latency >= 2000ms or very low
        throughput. Immediate attention required.

    Examples
    --------
    Getting performance level from a report:

    >>> from insideLLMs.latency import ResponseProfiler
    >>> profiler = ResponseProfiler(model_id="gpt-4")
    >>> for _ in range(5):
    ...     profiler.profile_response("Hello", "Hi there!", total_time_ms=150.0)
    >>> report = profiler.generate_report()
    >>> print(f"Level: {report.performance_level.value}")
    Level: excellent

    Comparing performance levels:

    >>> level = PerformanceLevel.GOOD
    >>> if level in [PerformanceLevel.EXCELLENT, PerformanceLevel.GOOD]:
    ...     print("Performance is satisfactory")
    Performance is satisfactory

    Using performance level for alerting:

    >>> from insideLLMs.latency import PerformanceLevel
    >>> def check_performance(level: PerformanceLevel) -> str:
    ...     if level == PerformanceLevel.CRITICAL:
    ...         return "ALERT: Critical performance degradation!"
    ...     elif level == PerformanceLevel.POOR:
    ...         return "WARNING: Performance below expectations"
    ...     return "OK"
    >>> check_performance(PerformanceLevel.CRITICAL)
    'ALERT: Critical performance degradation!'

    See Also
    --------
    PerformanceReport : Report that includes performance level.
    ResponseProfiler : Profiler that generates performance classification.
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class LatencyMeasurement:
    """A single latency measurement with timestamp and metadata.

    This dataclass represents one recorded latency observation. It captures
    the metric type, measured value, when the measurement was taken, and
    any additional context through metadata.

    Parameters
    ----------
    metric : LatencyMetric
        The type of latency being measured (e.g., TIME_TO_FIRST_TOKEN).
    value_ms : float
        The measured latency value in milliseconds.
    timestamp : float, optional
        Unix timestamp when measurement was recorded. Defaults to current time.
    metadata : dict[str, Any], optional
        Additional context such as request ID, model parameters, or tags.

    Attributes
    ----------
    metric : LatencyMetric
        The latency metric type for this measurement.
    value_ms : float
        The latency value in milliseconds.
    timestamp : float
        Unix timestamp of the measurement.
    metadata : dict[str, Any]
        Additional contextual information.

    Examples
    --------
    Creating a basic measurement:

    >>> from insideLLMs.latency import LatencyMeasurement, LatencyMetric
    >>> measurement = LatencyMeasurement(
    ...     metric=LatencyMetric.TIME_TO_FIRST_TOKEN,
    ...     value_ms=45.5
    ... )
    >>> print(f"{measurement.metric.value}: {measurement.value_ms}ms")
    time_to_first_token: 45.5ms

    Creating a measurement with metadata:

    >>> measurement = LatencyMeasurement(
    ...     metric=LatencyMetric.TOTAL_RESPONSE_TIME,
    ...     value_ms=850.0,
    ...     metadata={
    ...         "request_id": "req_abc123",
    ...         "model": "gpt-4",
    ...         "prompt_tokens": 150
    ...     }
    ... )
    >>> print(measurement.metadata["request_id"])
    req_abc123

    Converting to dictionary for serialization:

    >>> data = measurement.to_dict()
    >>> print(data["metric"])
    total_response_time
    >>> print(data["value_ms"])
    850.0

    Using measurements in analysis:

    >>> measurements = [
    ...     LatencyMeasurement(LatencyMetric.TOTAL_RESPONSE_TIME, 200.0),
    ...     LatencyMeasurement(LatencyMetric.TOTAL_RESPONSE_TIME, 250.0),
    ...     LatencyMeasurement(LatencyMetric.TOTAL_RESPONSE_TIME, 180.0),
    ... ]
    >>> avg = sum(m.value_ms for m in measurements) / len(measurements)
    >>> print(f"Average: {avg:.1f}ms")
    Average: 210.0ms

    See Also
    --------
    LatencyMetric : Available latency metric types.
    LatencyProfiler : Records and aggregates latency measurements.
    """

    metric: LatencyMetric
    value_ms: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the measurement to a dictionary for serialization.

        Returns a dictionary representation suitable for JSON serialization,
        logging, or storage in databases.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - metric: The metric type as a string value
            - value_ms: The latency value in milliseconds
            - timestamp: Unix timestamp of the measurement
            - metadata: Any additional contextual data

        Examples
        --------
        >>> measurement = LatencyMeasurement(
        ...     metric=LatencyMetric.TIME_TO_FIRST_TOKEN,
        ...     value_ms=45.5,
        ...     metadata={"model": "gpt-4"}
        ... )
        >>> data = measurement.to_dict()
        >>> print(data)
        {'metric': 'time_to_first_token', 'value_ms': 45.5, ...}

        Using with JSON serialization:

        >>> import json
        >>> json_str = json.dumps(measurement.to_dict())
        >>> print("time_to_first_token" in json_str)
        True
        """
        return {
            "metric": self.metric.value,
            "value_ms": self.value_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ThroughputMeasurement:
    """A single throughput measurement with unit and metadata.

    This dataclass represents one recorded throughput observation. It captures
    the metric type, measured value, unit of measurement, timestamp, and
    any additional context through metadata.

    Parameters
    ----------
    metric : ThroughputMetric
        The type of throughput being measured (e.g., TOKENS_PER_SECOND).
    value : float
        The measured throughput value.
    unit : str
        The unit of measurement (e.g., "tokens/s", "chars/s").
    timestamp : float, optional
        Unix timestamp when measurement was recorded. Defaults to current time.
    metadata : dict[str, Any], optional
        Additional context such as request ID, batch size, or configuration.

    Attributes
    ----------
    metric : ThroughputMetric
        The throughput metric type for this measurement.
    value : float
        The throughput value.
    unit : str
        The unit of measurement.
    timestamp : float
        Unix timestamp of the measurement.
    metadata : dict[str, Any]
        Additional contextual information.

    Examples
    --------
    Creating a tokens per second measurement:

    >>> from insideLLMs.latency import ThroughputMeasurement, ThroughputMetric
    >>> measurement = ThroughputMeasurement(
    ...     metric=ThroughputMetric.TOKENS_PER_SECOND,
    ...     value=75.5,
    ...     unit="tokens/s"
    ... )
    >>> print(f"{measurement.value} {measurement.unit}")
    75.5 tokens/s

    Creating a measurement with metadata:

    >>> measurement = ThroughputMeasurement(
    ...     metric=ThroughputMetric.CHARACTERS_PER_SECOND,
    ...     value=320.0,
    ...     unit="chars/s",
    ...     metadata={
    ...         "batch_size": 1,
    ...         "model": "claude-3",
    ...         "temperature": 0.7
    ...     }
    ... )
    >>> print(measurement.metadata["model"])
    claude-3

    Converting to dictionary for logging:

    >>> data = measurement.to_dict()
    >>> print(f"{data['metric']}: {data['value']} {data['unit']}")
    characters_per_second: 320.0 chars/s

    Comparing throughput measurements:

    >>> measurements = [
    ...     ThroughputMeasurement(ThroughputMetric.TOKENS_PER_SECOND, 70.0, "tps"),
    ...     ThroughputMeasurement(ThroughputMetric.TOKENS_PER_SECOND, 85.0, "tps"),
    ...     ThroughputMeasurement(ThroughputMetric.TOKENS_PER_SECOND, 75.0, "tps"),
    ... ]
    >>> max_tps = max(m.value for m in measurements)
    >>> print(f"Peak throughput: {max_tps} tps")
    Peak throughput: 85.0 tps

    See Also
    --------
    ThroughputMetric : Available throughput metric types.
    ThroughputProfiler : Records and aggregates throughput measurements.
    """

    metric: ThroughputMetric
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the measurement to a dictionary for serialization.

        Returns a dictionary representation suitable for JSON serialization,
        logging, or storage in databases.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - metric: The metric type as a string value
            - value: The throughput value
            - unit: The unit of measurement
            - timestamp: Unix timestamp of the measurement
            - metadata: Any additional contextual data

        Examples
        --------
        >>> measurement = ThroughputMeasurement(
        ...     metric=ThroughputMetric.TOKENS_PER_SECOND,
        ...     value=75.5,
        ...     unit="tokens/s"
        ... )
        >>> data = measurement.to_dict()
        >>> print(data["metric"], data["value"])
        tokens_per_second 75.5

        Using with JSON for API response:

        >>> import json
        >>> json_str = json.dumps(measurement.to_dict())
        >>> print("tokens_per_second" in json_str)
        True
        """
        return {
            "metric": self.metric.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class LatencyStats:
    """Statistical summary of latency measurements.

    This dataclass provides a comprehensive statistical overview of collected
    latency measurements, including central tendency, variability, and
    percentile metrics. It is typically generated by a LatencyProfiler after
    multiple measurements have been recorded.

    Parameters
    ----------
    mean_ms : float
        Arithmetic mean of all latency measurements in milliseconds.
    median_ms : float
        Median (50th percentile) latency in milliseconds.
    std_ms : float
        Standard deviation of latency measurements in milliseconds.
    min_ms : float
        Minimum observed latency in milliseconds.
    max_ms : float
        Maximum observed latency in milliseconds.
    p50_ms : float
        50th percentile latency in milliseconds (same as median).
    p90_ms : float
        90th percentile latency in milliseconds.
    p95_ms : float
        95th percentile latency in milliseconds.
    p99_ms : float
        99th percentile latency in milliseconds (tail latency).
    sample_count : int
        Number of measurements used to calculate statistics.

    Attributes
    ----------
    coefficient_of_variation : float
        Ratio of standard deviation to mean (std/mean). Lower values
        indicate more consistent performance.
    is_consistent : bool
        True if coefficient_of_variation < 0.3, indicating stable latency.

    Examples
    --------
    Getting stats from a profiler:

    >>> from insideLLMs.latency import LatencyProfiler, LatencyMetric
    >>> profiler = LatencyProfiler()
    >>> for latency in [100.0, 120.0, 95.0, 150.0, 110.0]:
    ...     profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, latency)
    >>> stats = profiler.get_stats(LatencyMetric.TOTAL_RESPONSE_TIME)
    >>> print(f"Mean: {stats.mean_ms:.1f}ms, p99: {stats.p99_ms:.1f}ms")
    Mean: 115.0ms, p99: 147.5ms

    Checking consistency:

    >>> print(f"CV: {stats.coefficient_of_variation:.2f}")
    CV: 0.18
    >>> print(f"Consistent: {stats.is_consistent}")
    Consistent: True

    Creating stats directly for testing:

    >>> stats = LatencyStats(
    ...     mean_ms=250.0,
    ...     median_ms=240.0,
    ...     std_ms=50.0,
    ...     min_ms=180.0,
    ...     max_ms=400.0,
    ...     p50_ms=240.0,
    ...     p90_ms=320.0,
    ...     p95_ms=350.0,
    ...     p99_ms=390.0,
    ...     sample_count=100
    ... )
    >>> print(f"Spread: {stats.max_ms - stats.min_ms}ms")
    Spread: 220.0ms

    Analyzing tail latency vs median:

    >>> tail_ratio = stats.p99_ms / stats.median_ms
    >>> print(f"p99/p50 ratio: {tail_ratio:.2f}x")
    p99/p50 ratio: 1.62x

    Converting to dictionary for reporting:

    >>> data = stats.to_dict()
    >>> print(f"Sample size: {data['sample_count']}")
    Sample size: 100

    See Also
    --------
    LatencyProfiler : Generates these statistics from measurements.
    ThroughputStats : Equivalent statistics for throughput metrics.
    """

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
        """Calculate the coefficient of variation (CV) for latency measurements.

        The coefficient of variation is the ratio of standard deviation to mean,
        providing a normalized measure of dispersion. It is useful for comparing
        variability across different measurement scales.

        Returns
        -------
        float
            The coefficient of variation (std_ms / mean_ms). Returns 0.0 if
            mean_ms is zero to avoid division by zero.

        Examples
        --------
        Low CV indicates consistent performance:

        >>> stats = LatencyStats(
        ...     mean_ms=100.0, median_ms=100.0, std_ms=10.0,
        ...     min_ms=80.0, max_ms=120.0, p50_ms=100.0,
        ...     p90_ms=115.0, p95_ms=118.0, p99_ms=120.0, sample_count=50
        ... )
        >>> print(f"CV: {stats.coefficient_of_variation:.2f}")
        CV: 0.10

        High CV indicates variable performance:

        >>> stats = LatencyStats(
        ...     mean_ms=100.0, median_ms=100.0, std_ms=50.0,
        ...     min_ms=30.0, max_ms=200.0, p50_ms=100.0,
        ...     p90_ms=160.0, p95_ms=180.0, p99_ms=195.0, sample_count=50
        ... )
        >>> print(f"CV: {stats.coefficient_of_variation:.2f}")
        CV: 0.50
        """
        if self.mean_ms == 0:
            return 0.0
        return self.std_ms / self.mean_ms

    @property
    def is_consistent(self) -> bool:
        """Check if latency measurements show consistent performance.

        Consistency is determined by the coefficient of variation. A CV below
        0.3 (30%) indicates that latency is relatively stable and predictable.

        Returns
        -------
        bool
            True if coefficient_of_variation < 0.3, False otherwise.

        Examples
        --------
        Consistent performance (CV < 0.3):

        >>> stats = LatencyStats(
        ...     mean_ms=100.0, median_ms=100.0, std_ms=20.0,
        ...     min_ms=70.0, max_ms=130.0, p50_ms=100.0,
        ...     p90_ms=125.0, p95_ms=128.0, p99_ms=130.0, sample_count=50
        ... )
        >>> print(stats.is_consistent)
        True

        Inconsistent performance (CV >= 0.3):

        >>> stats = LatencyStats(
        ...     mean_ms=100.0, median_ms=100.0, std_ms=40.0,
        ...     min_ms=40.0, max_ms=180.0, p50_ms=100.0,
        ...     p90_ms=150.0, p95_ms=165.0, p99_ms=175.0, sample_count=50
        ... )
        >>> print(stats.is_consistent)
        False
        """
        return self.coefficient_of_variation < 0.3

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to a dictionary for serialization.

        Returns a comprehensive dictionary including all statistical measures
        and computed properties like coefficient_of_variation and is_consistent.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all statistical fields plus:
            - coefficient_of_variation: Normalized measure of dispersion
            - is_consistent: Boolean indicating stable performance

        Examples
        --------
        >>> stats = LatencyStats(
        ...     mean_ms=150.0, median_ms=145.0, std_ms=25.0,
        ...     min_ms=100.0, max_ms=220.0, p50_ms=145.0,
        ...     p90_ms=185.0, p95_ms=200.0, p99_ms=215.0, sample_count=100
        ... )
        >>> data = stats.to_dict()
        >>> print(f"Mean: {data['mean_ms']}ms")
        Mean: 150.0ms
        >>> print(f"Consistent: {data['is_consistent']}")
        Consistent: True

        Using for JSON export:

        >>> import json
        >>> json_str = json.dumps(stats.to_dict())
        >>> print("coefficient_of_variation" in json_str)
        True
        """
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
    """Statistical summary of throughput measurements.

    This dataclass provides a comprehensive statistical overview of collected
    throughput measurements. It includes central tendency, variability metrics,
    and a sustained rate estimate useful for capacity planning.

    Parameters
    ----------
    mean : float
        Arithmetic mean of all throughput measurements.
    median : float
        Median (50th percentile) throughput.
    std : float
        Standard deviation of throughput measurements.
    min_val : float
        Minimum observed throughput.
    max_val : float
        Maximum observed throughput (peak performance).
    sample_count : int
        Number of measurements used to calculate statistics.
    unit : str
        Unit of measurement (e.g., "tokens/s", "chars/s").

    Attributes
    ----------
    sustained_rate : float
        Conservative estimate of sustainable throughput, calculated as
        mean - std. Useful for capacity planning and SLA guarantees.

    Examples
    --------
    Getting stats from a throughput profiler:

    >>> from insideLLMs.latency import ThroughputProfiler, ThroughputMetric
    >>> profiler = ThroughputProfiler()
    >>> for tps in [70.0, 75.0, 80.0, 65.0, 72.0]:
    ...     profiler.record(ThroughputMetric.TOKENS_PER_SECOND, tps)
    >>> stats = profiler.get_stats(ThroughputMetric.TOKENS_PER_SECOND)
    >>> print(f"Mean: {stats.mean:.1f} {stats.unit}")
    Mean: 72.4 tokens/s

    Using sustained rate for capacity planning:

    >>> print(f"Sustained rate: {stats.sustained_rate:.1f} {stats.unit}")
    Sustained rate: 66.8 tokens/s

    Creating stats directly for testing:

    >>> stats = ThroughputStats(
    ...     mean=75.0,
    ...     median=74.0,
    ...     std=8.0,
    ...     min_val=60.0,
    ...     max_val=95.0,
    ...     sample_count=100,
    ...     unit="tokens/s"
    ... )
    >>> print(f"Range: {stats.min_val} - {stats.max_val} {stats.unit}")
    Range: 60.0 - 95.0 tokens/s

    Analyzing peak vs sustained performance:

    >>> headroom = stats.max_val - stats.sustained_rate
    >>> print(f"Performance headroom: {headroom:.1f} {stats.unit}")
    Performance headroom: 28.0 tokens/s

    Converting to dictionary for reporting:

    >>> data = stats.to_dict()
    >>> print(f"Sample size: {data['sample_count']}, Unit: {data['unit']}")
    Sample size: 100, Unit: tokens/s

    See Also
    --------
    ThroughputProfiler : Generates these statistics from measurements.
    LatencyStats : Equivalent statistics for latency metrics.
    """

    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    sample_count: int
    unit: str

    @property
    def sustained_rate(self) -> float:
        """Estimate sustainable throughput for capacity planning.

        Provides a conservative throughput estimate (approximately p10) by
        subtracting one standard deviation from the mean. This value represents
        the throughput rate that can likely be sustained under normal conditions.

        Returns
        -------
        float
            Estimated sustainable throughput (mean - std), minimum 0.0.
            Use this value for SLA guarantees and capacity planning.

        Examples
        --------
        Normal variability:

        >>> stats = ThroughputStats(
        ...     mean=80.0, median=79.0, std=10.0,
        ...     min_val=55.0, max_val=100.0, sample_count=50, unit="tokens/s"
        ... )
        >>> print(f"Sustained: {stats.sustained_rate:.1f} tokens/s")
        Sustained: 70.0 tokens/s

        High variability reduces sustained rate:

        >>> stats = ThroughputStats(
        ...     mean=80.0, median=79.0, std=30.0,
        ...     min_val=30.0, max_val=130.0, sample_count=50, unit="tokens/s"
        ... )
        >>> print(f"Sustained: {stats.sustained_rate:.1f} tokens/s")
        Sustained: 50.0 tokens/s

        Very high variability (capped at 0):

        >>> stats = ThroughputStats(
        ...     mean=50.0, median=45.0, std=60.0,
        ...     min_val=5.0, max_val=150.0, sample_count=50, unit="tokens/s"
        ... )
        >>> print(f"Sustained: {stats.sustained_rate:.1f} tokens/s")
        Sustained: 0.0 tokens/s
        """
        # Conservative estimate using lower bound
        return max(0, self.mean - self.std)

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to a dictionary for serialization.

        Returns a comprehensive dictionary including all statistical measures
        and the computed sustained_rate property.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - mean, median, std: Central tendency and variability
            - min, max: Range of observed values
            - sample_count: Number of measurements
            - unit: Unit of measurement
            - sustained_rate: Estimated sustainable throughput

        Examples
        --------
        >>> stats = ThroughputStats(
        ...     mean=75.0, median=74.0, std=8.0,
        ...     min_val=60.0, max_val=95.0, sample_count=100, unit="tokens/s"
        ... )
        >>> data = stats.to_dict()
        >>> print(f"Mean: {data['mean']} {data['unit']}")
        Mean: 75.0 tokens/s
        >>> print(f"Sustained: {data['sustained_rate']} {data['unit']}")
        Sustained: 67.0 tokens/s

        Using for JSON export:

        >>> import json
        >>> json_str = json.dumps(stats.to_dict())
        >>> print("sustained_rate" in json_str)
        True
        """
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
    """Complete profile for a single LLM response.

    This dataclass captures all performance-related information for a single
    model response, including token counts, timing metrics, and throughput.
    ResponseProfile objects are created by ResponseProfiler and aggregated
    into PerformanceReport objects.

    Parameters
    ----------
    prompt_tokens : int
        Number of tokens in the input prompt.
    completion_tokens : int
        Number of tokens in the model's response.
    time_to_first_token_ms : float | None
        Time from request to first token received, in milliseconds.
        None if not measured (e.g., non-streaming response).
    total_time_ms : float
        Total response time from request to completion, in milliseconds.
    tokens_per_second : float
        Generation throughput calculated as completion_tokens / time.
    timestamp : float, optional
        Unix timestamp when the response was received. Defaults to current time.
    model_id : str | None, optional
        Identifier for the model that generated the response.
    metadata : dict[str, Any], optional
        Additional context such as temperature, max_tokens, or request ID.

    Attributes
    ----------
    inter_token_latency_ms : float
        Calculated average time between consecutive tokens during generation.

    Examples
    --------
    Creating a profile for a streaming response:

    >>> from insideLLMs.latency import ResponseProfile
    >>> profile = ResponseProfile(
    ...     prompt_tokens=50,
    ...     completion_tokens=150,
    ...     time_to_first_token_ms=45.0,
    ...     total_time_ms=800.0,
    ...     tokens_per_second=187.5,
    ...     model_id="gpt-4"
    ... )
    >>> print(f"TTFT: {profile.time_to_first_token_ms}ms")
    TTFT: 45.0ms
    >>> print(f"Inter-token: {profile.inter_token_latency_ms:.2f}ms")
    Inter-token: 5.07ms

    Creating a profile for a non-streaming response:

    >>> profile = ResponseProfile(
    ...     prompt_tokens=100,
    ...     completion_tokens=200,
    ...     time_to_first_token_ms=None,
    ...     total_time_ms=1500.0,
    ...     tokens_per_second=133.3
    ... )
    >>> print(f"Total time: {profile.total_time_ms}ms")
    Total time: 1500.0ms

    Using metadata for additional context:

    >>> profile = ResponseProfile(
    ...     prompt_tokens=75,
    ...     completion_tokens=100,
    ...     time_to_first_token_ms=50.0,
    ...     total_time_ms=600.0,
    ...     tokens_per_second=166.7,
    ...     metadata={
    ...         "temperature": 0.7,
    ...         "max_tokens": 500,
    ...         "request_id": "req_xyz789"
    ...     }
    ... )
    >>> print(profile.metadata["temperature"])
    0.7

    Converting to dictionary for storage:

    >>> data = profile.to_dict()
    >>> print(data["completion_tokens"])
    100

    Analyzing efficiency:

    >>> efficiency = profile.completion_tokens / profile.total_time_ms * 1000
    >>> print(f"Efficiency: {efficiency:.1f} tokens/s")
    Efficiency: 166.7 tokens/s

    See Also
    --------
    ResponseProfiler : Creates and aggregates ResponseProfile objects.
    PerformanceReport : Contains aggregated profiles and statistics.
    """

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
        """Calculate average inter-token latency during generation.

        Computes the average time between consecutive tokens. For streaming
        responses, this excludes the time to first token. For non-streaming
        responses, it divides total time by token count.

        Returns
        -------
        float
            Average inter-token latency in milliseconds. Returns 0.0 if
            completion_tokens <= 1 (no inter-token intervals).

        Examples
        --------
        Streaming response with TTFT:

        >>> profile = ResponseProfile(
        ...     prompt_tokens=50, completion_tokens=100,
        ...     time_to_first_token_ms=40.0, total_time_ms=1000.0,
        ...     tokens_per_second=100.0
        ... )
        >>> # Generation time = 1000 - 40 = 960ms for 99 intervals
        >>> print(f"Inter-token: {profile.inter_token_latency_ms:.2f}ms")
        Inter-token: 9.70ms

        Non-streaming response (no TTFT):

        >>> profile = ResponseProfile(
        ...     prompt_tokens=50, completion_tokens=100,
        ...     time_to_first_token_ms=None, total_time_ms=1000.0,
        ...     tokens_per_second=100.0
        ... )
        >>> print(f"Inter-token: {profile.inter_token_latency_ms:.2f}ms")
        Inter-token: 10.00ms

        Single token response:

        >>> profile = ResponseProfile(
        ...     prompt_tokens=50, completion_tokens=1,
        ...     time_to_first_token_ms=30.0, total_time_ms=30.0,
        ...     tokens_per_second=33.3
        ... )
        >>> print(f"Inter-token: {profile.inter_token_latency_ms:.1f}ms")
        Inter-token: 0.0ms
        """
        if self.completion_tokens <= 1:
            return 0.0
        if self.time_to_first_token_ms is None:
            return self.total_time_ms / self.completion_tokens
        generation_time = self.total_time_ms - self.time_to_first_token_ms
        return generation_time / max(1, self.completion_tokens - 1)

    def to_dict(self) -> dict[str, Any]:
        """Convert the profile to a dictionary for serialization.

        Returns a complete dictionary representation including all fields
        and the computed inter_token_latency_ms property.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - Token counts (prompt_tokens, completion_tokens)
            - Timing metrics (time_to_first_token_ms, total_time_ms)
            - Throughput (tokens_per_second)
            - Computed inter_token_latency_ms
            - Timestamp, model_id, and metadata

        Examples
        --------
        >>> profile = ResponseProfile(
        ...     prompt_tokens=50, completion_tokens=100,
        ...     time_to_first_token_ms=40.0, total_time_ms=800.0,
        ...     tokens_per_second=125.0, model_id="claude-3"
        ... )
        >>> data = profile.to_dict()
        >>> print(f"Model: {data['model_id']}")
        Model: claude-3
        >>> print(f"Tokens: {data['prompt_tokens']} + {data['completion_tokens']}")
        Tokens: 50 + 100

        Using for JSON logging:

        >>> import json
        >>> log_entry = json.dumps(profile.to_dict())
        >>> print("tokens_per_second" in log_entry)
        True
        """
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
    """Comprehensive performance report for LLM evaluation.

    This dataclass aggregates all performance analysis results including
    statistical summaries, identified bottlenecks, and actionable
    recommendations. It is generated by ResponseProfiler after collecting
    multiple response profiles.

    Parameters
    ----------
    model_id : str
        Identifier for the model being profiled.
    latency_stats : dict[LatencyMetric, LatencyStats]
        Statistical summaries for each latency metric type.
    throughput_stats : dict[ThroughputMetric, ThroughputStats]
        Statistical summaries for each throughput metric type.
    response_profiles : list[ResponseProfile]
        Individual response profiles collected during profiling.
    performance_level : PerformanceLevel
        Overall performance classification (EXCELLENT to CRITICAL).
    bottlenecks : list[str]
        Identified performance issues with descriptions.
    recommendations : list[str]
        Actionable suggestions for performance improvement.

    Attributes
    ----------
    overall_score : float
        Computed score from 0.0 (worst) to 1.0 (best) based on
        latency and throughput metrics.

    Examples
    --------
    Generating a report from a profiler:

    >>> from insideLLMs.latency import ResponseProfiler
    >>> profiler = ResponseProfiler(model_id="gpt-4")
    >>> for i in range(10):
    ...     profiler.profile_response(
    ...         prompt="Test prompt",
    ...         response="Test response " * 20,
    ...         total_time_ms=300 + i * 20,
    ...         time_to_first_token_ms=40 + i * 5
    ...     )
    >>> report = profiler.generate_report()
    >>> print(f"Model: {report.model_id}")
    Model: gpt-4
    >>> print(f"Level: {report.performance_level.value}")
    Level: good

    Analyzing bottlenecks:

    >>> if report.bottlenecks:
    ...     print("Issues found:")
    ...     for issue in report.bottlenecks:
    ...         print(f"  - {issue}")
    >>> else:
    ...     print("No bottlenecks detected")
    No bottlenecks detected

    Getting recommendations:

    >>> for rec in report.recommendations:
    ...     print(f"Recommendation: {rec}")
    Recommendation: Performance is within acceptable parameters

    Using the overall score:

    >>> score = report.overall_score
    >>> if score >= 0.8:
    ...     print("Excellent performance!")
    ... elif score >= 0.5:
    ...     print("Good performance")
    ... else:
    ...     print("Needs improvement")
    Good performance

    Converting to dictionary for export:

    >>> data = report.to_dict()
    >>> print(f"Profiles collected: {data['n_profiles']}")
    Profiles collected: 10

    Comparing with another model:

    >>> from insideLLMs.latency import PerformanceComparator
    >>> comparator = PerformanceComparator()
    >>> comparator.add_report("gpt-4", report)
    >>> # Add more reports and compare...

    See Also
    --------
    ResponseProfiler : Generates PerformanceReport objects.
    PerformanceComparator : Compares multiple PerformanceReports.
    PerformanceLevel : Performance classification enum.
    """

    model_id: str
    latency_stats: dict[LatencyMetric, LatencyStats]
    throughput_stats: dict[ThroughputMetric, ThroughputStats]
    response_profiles: list[ResponseProfile]
    performance_level: PerformanceLevel
    bottlenecks: list[str]
    recommendations: list[str]

    @property
    def overall_score(self) -> float:
        """Calculate overall performance score from 0.0 to 1.0.

        Combines latency and throughput metrics into a single score:
        - Latency score: 1.0 at 0ms, 0.5 at 1000ms, 0.0 at 2000ms+
        - Throughput score: 0.0 at 0 tps, 0.5 at 50 tps, 1.0 at 100+ tps

        The final score is the mean of available component scores.

        Returns
        -------
        float
            Performance score from 0.0 (worst) to 1.0 (best).
            Returns 0.5 if no metrics are available.

        Examples
        --------
        Fast responses with high throughput:

        >>> # With 200ms latency and 80 tps:
        >>> # lat_score = 1 - 200/2000 = 0.9
        >>> # tps_score = 80/100 = 0.8
        >>> # overall = (0.9 + 0.8) / 2 = 0.85

        Slow responses:

        >>> # With 1500ms latency and 30 tps:
        >>> # lat_score = 1 - 1500/2000 = 0.25
        >>> # tps_score = 30/100 = 0.3
        >>> # overall = (0.25 + 0.3) / 2 = 0.275
        """
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
        """Convert the report to a dictionary for serialization.

        Returns a comprehensive dictionary suitable for JSON export,
        logging, or storage. Note that individual response_profiles
        are not included; only the count is provided.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - model_id: Model identifier
            - latency_stats: Nested dict of latency statistics
            - throughput_stats: Nested dict of throughput statistics
            - n_profiles: Number of response profiles collected
            - performance_level: Classification string
            - overall_score: Computed performance score
            - bottlenecks: List of identified issues
            - recommendations: List of improvement suggestions

        Examples
        --------
        >>> from insideLLMs.latency import ResponseProfiler
        >>> profiler = ResponseProfiler(model_id="claude-3")
        >>> profiler.profile_response("Hi", "Hello!", total_time_ms=150.0)
        >>> report = profiler.generate_report()
        >>> data = report.to_dict()
        >>> print(data["model_id"])
        claude-3
        >>> print(type(data["latency_stats"]))
        <class 'dict'>

        Using for JSON export:

        >>> import json
        >>> json_str = json.dumps(data)
        >>> print("overall_score" in json_str)
        True
        """
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
    """High-precision latency timer for measuring LLM response times.

    LatencyTimer provides sub-millisecond precision timing using
    `time.perf_counter()`. It supports measuring time to first token (TTFT),
    total response time, and inter-token latencies for streaming responses.

    The timer can be used directly via start()/stop() methods or as a
    context manager for automatic timing of code blocks.

    Attributes
    ----------
    time_to_first_token_ms : float | None
        Time from start to first token mark, in milliseconds.
    total_time_ms : float | None
        Total elapsed time from start, in milliseconds.
    inter_token_latencies_ms : list[float]
        List of inter-token latencies in milliseconds.

    Examples
    --------
    Basic timing with context manager:

    >>> import time
    >>> from insideLLMs.latency import LatencyTimer
    >>> timer = LatencyTimer()
    >>> with timer:
    ...     time.sleep(0.1)  # Simulate work
    >>> print(f"Elapsed: {timer.total_time_ms:.0f}ms")
    Elapsed: 100ms

    Manual start/stop for more control:

    >>> timer = LatencyTimer()
    >>> timer.start()
    <insideLLMs.latency.LatencyTimer object at ...>
    >>> # ... make API call ...
    >>> timer.stop()
    >>> print(f"Total: {timer.total_time_ms}ms")
    Total: ...

    Measuring time to first token in streaming:

    >>> timer = LatencyTimer()
    >>> timer.start()
    <insideLLMs.latency.LatencyTimer object at ...>
    >>> # Simulate receiving first token
    >>> timer.mark_first_token()
    >>> # Simulate receiving more tokens
    >>> timer.mark_token()
    >>> timer.mark_token()
    >>> timer.stop()
    >>> print(f"TTFT: {timer.time_to_first_token_ms}ms")
    TTFT: ...

    Tracking inter-token latencies:

    >>> timer = LatencyTimer()
    >>> timer.start()
    <insideLLMs.latency.LatencyTimer object at ...>
    >>> for _ in range(5):
    ...     time.sleep(0.01)  # Simulate token arrival
    ...     timer.mark_token()
    >>> timer.stop()
    >>> latencies = timer.inter_token_latencies_ms
    >>> print(f"Avg inter-token: {sum(latencies)/len(latencies):.0f}ms")
    Avg inter-token: 10ms

    Using with LLM API calls:

    >>> def call_llm_streaming(prompt):
    ...     timer = LatencyTimer()
    ...     timer.start()
    ...     # response = client.completions.create(prompt=prompt, stream=True)
    ...     # for i, chunk in enumerate(response):
    ...     #     if i == 0:
    ...     #         timer.mark_first_token()
    ...     #     timer.mark_token()
    ...     timer.stop()
    ...     return timer

    See Also
    --------
    LatencyProfiler : Aggregates multiple timer measurements.
    StreamingProfiler : Specialized profiler for streaming responses.
    measure_latency : Convenience function using LatencyTimer.
    """

    def __init__(self):
        """Initialize a new LatencyTimer instance.

        Creates a timer in the stopped state with no recorded times.
        Call start() or use as a context manager to begin timing.

        Examples
        --------
        >>> from insideLLMs.latency import LatencyTimer
        >>> timer = LatencyTimer()
        >>> timer.total_time_ms is None
        True
        >>> timer.start()
        <insideLLMs.latency.LatencyTimer object at ...>
        >>> timer.total_time_ms is not None
        True
        """
        self._start_time: float | None = None
        self._first_token_time: float | None = None
        self._end_time: float | None = None
        self._token_times: list[float] = []

    def start(self) -> LatencyTimer:
        """Start the timer and record the start time.

        Uses `time.perf_counter()` for high-precision timing.
        Returns self to enable method chaining.

        Returns
        -------
        LatencyTimer
            Self, for method chaining.

        Examples
        --------
        >>> from insideLLMs.latency import LatencyTimer
        >>> timer = LatencyTimer().start()
        >>> timer.total_time_ms is not None
        True

        Method chaining:

        >>> timer = LatencyTimer()
        >>> result = timer.start()
        >>> result is timer
        True
        """
        self._start_time = time.perf_counter()
        return self

    def mark_first_token(self) -> None:
        """Mark the time when the first token is received.

        This method should be called exactly once when the first token
        arrives in a streaming response. Subsequent calls are ignored
        to preserve the original TTFT measurement.

        Examples
        --------
        >>> from insideLLMs.latency import LatencyTimer
        >>> import time
        >>> timer = LatencyTimer().start()
        >>> time.sleep(0.05)
        >>> timer.mark_first_token()
        >>> print(f"TTFT: {timer.time_to_first_token_ms:.0f}ms")
        TTFT: 50ms

        Only first call is recorded:

        >>> timer.mark_first_token()  # Ignored
        >>> print(f"TTFT unchanged: {timer.time_to_first_token_ms:.0f}ms")
        TTFT unchanged: 50ms
        """
        if self._first_token_time is None:
            self._first_token_time = time.perf_counter()

    def mark_token(self) -> None:
        """Mark the time when a token is received.

        Call this method for each token received in a streaming response.
        The timestamps are used to calculate inter-token latencies.

        Examples
        --------
        >>> from insideLLMs.latency import LatencyTimer
        >>> import time
        >>> timer = LatencyTimer().start()
        >>> for _ in range(3):
        ...     time.sleep(0.01)
        ...     timer.mark_token()
        >>> timer.stop()
        >>> print(f"Tokens marked: {len(timer.inter_token_latencies_ms) + 1}")
        Tokens marked: 3
        """
        self._token_times.append(time.perf_counter())

    def stop(self) -> None:
        """Stop the timer and record the end time.

        After calling stop(), total_time_ms will return the final
        elapsed time instead of the current running time.

        Examples
        --------
        >>> from insideLLMs.latency import LatencyTimer
        >>> import time
        >>> timer = LatencyTimer().start()
        >>> time.sleep(0.1)
        >>> timer.stop()
        >>> final_time = timer.total_time_ms
        >>> time.sleep(0.1)  # More time passes
        >>> timer.total_time_ms == final_time  # Time is frozen
        True
        """
        self._end_time = time.perf_counter()

    @property
    def time_to_first_token_ms(self) -> float | None:
        """Get the time to first token in milliseconds.

        Returns the time elapsed between start() and mark_first_token().
        Returns None if either method was not called.

        Returns
        -------
        float | None
            Time to first token in milliseconds, or None if not measured.

        Examples
        --------
        >>> from insideLLMs.latency import LatencyTimer
        >>> import time
        >>> timer = LatencyTimer().start()
        >>> timer.time_to_first_token_ms is None
        True
        >>> time.sleep(0.05)
        >>> timer.mark_first_token()
        >>> print(f"TTFT: {timer.time_to_first_token_ms:.0f}ms")
        TTFT: 50ms
        """
        if self._start_time is None or self._first_token_time is None:
            return None
        return (self._first_token_time - self._start_time) * 1000

    @property
    def total_time_ms(self) -> float | None:
        """Get the total elapsed time in milliseconds.

        If stop() has been called, returns the time between start() and
        stop(). Otherwise, returns the time since start() (running timer).

        Returns
        -------
        float | None
            Total elapsed time in milliseconds, or None if not started.

        Examples
        --------
        Running timer (not stopped):

        >>> from insideLLMs.latency import LatencyTimer
        >>> import time
        >>> timer = LatencyTimer().start()
        >>> time.sleep(0.05)
        >>> print(f"Running: {timer.total_time_ms:.0f}ms")
        Running: 50ms
        >>> time.sleep(0.05)
        >>> print(f"Still running: {timer.total_time_ms:.0f}ms")
        Still running: 100ms

        Stopped timer (frozen):

        >>> timer.stop()
        >>> frozen = timer.total_time_ms
        >>> time.sleep(0.1)
        >>> timer.total_time_ms == frozen
        True
        """
        if self._start_time is None:
            return None
        end = self._end_time or time.perf_counter()
        return (end - self._start_time) * 1000

    @property
    def inter_token_latencies_ms(self) -> list[float]:
        """Get the inter-token latencies in milliseconds.

        Calculates the time between consecutive mark_token() calls.
        Requires at least 2 token marks to return any latencies.

        Returns
        -------
        list[float]
            List of inter-token latencies in milliseconds. Empty if
            fewer than 2 tokens were marked.

        Examples
        --------
        >>> from insideLLMs.latency import LatencyTimer
        >>> import time
        >>> timer = LatencyTimer().start()
        >>> for _ in range(4):
        ...     time.sleep(0.01)
        ...     timer.mark_token()
        >>> timer.stop()
        >>> latencies = timer.inter_token_latencies_ms
        >>> print(f"Number of intervals: {len(latencies)}")
        Number of intervals: 3
        >>> all(9 < lat < 15 for lat in latencies)  # ~10ms each
        True

        Not enough tokens:

        >>> timer2 = LatencyTimer().start()
        >>> timer2.mark_token()
        >>> timer2.stop()
        >>> timer2.inter_token_latencies_ms
        []
        """
        if len(self._token_times) < 2:
            return []
        latencies = []
        for i in range(1, len(self._token_times)):
            latency = (self._token_times[i] - self._token_times[i - 1]) * 1000
            latencies.append(latency)
        return latencies

    def __enter__(self) -> LatencyTimer:
        """Enter the context manager and start the timer.

        Returns
        -------
        LatencyTimer
            Self, with timer started.

        Examples
        --------
        >>> from insideLLMs.latency import LatencyTimer
        >>> import time
        >>> with LatencyTimer() as timer:
        ...     time.sleep(0.05)
        >>> print(f"Elapsed: {timer.total_time_ms:.0f}ms")
        Elapsed: 50ms
        """
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and stop the timer.

        Parameters
        ----------
        exc_type : type | None
            Exception type if an exception was raised.
        exc_val : BaseException | None
            Exception instance if an exception was raised.
        exc_tb : TracebackType | None
            Traceback if an exception was raised.

        Examples
        --------
        Timer stops even on exception:

        >>> from insideLLMs.latency import LatencyTimer
        >>> timer = LatencyTimer()
        >>> try:
        ...     with timer:
        ...         raise ValueError("test")
        ... except ValueError:
        ...     pass
        >>> timer.total_time_ms is not None  # Timer was stopped
        True
        """
        self.stop()


class LatencyProfiler:
    """Profile latency characteristics of LLM model responses.

    LatencyProfiler collects, stores, and analyzes latency measurements
    across multiple requests. It supports different latency metric types
    and provides statistical summaries including percentiles.

    Parameters
    ----------
    model_id : str | None, optional
        Identifier for the model being profiled. Used for labeling
        in reports and comparisons.

    Attributes
    ----------
    model_id : str | None
        The model identifier provided at initialization.

    Examples
    --------
    Basic latency profiling:

    >>> from insideLLMs.latency import LatencyProfiler, LatencyMetric
    >>> profiler = LatencyProfiler(model_id="gpt-4")
    >>> for latency in [100.0, 150.0, 120.0, 180.0, 130.0]:
    ...     profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, latency)
    >>> stats = profiler.get_stats(LatencyMetric.TOTAL_RESPONSE_TIME)
    >>> print(f"Mean: {stats.mean_ms:.1f}ms")
    Mean: 136.0ms

    Recording multiple metric types:

    >>> profiler = LatencyProfiler(model_id="claude-3")
    >>> profiler.record(LatencyMetric.TIME_TO_FIRST_TOKEN, 45.0)
    LatencyMeasurement(...)
    >>> profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, 850.0)
    LatencyMeasurement(...)
    >>> profiler.record(LatencyMetric.INTER_TOKEN_LATENCY, 8.5)
    LatencyMeasurement(...)

    Getting all statistics at once:

    >>> all_stats = profiler.get_all_stats()
    >>> for metric, stats in all_stats.items():
    ...     print(f"{metric.value}: {stats.mean_ms:.1f}ms")
    time_to_first_token: 45.0ms
    total_response_time: 850.0ms
    inter_token_latency: 8.5ms

    Recording with metadata:

    >>> measurement = profiler.record(
    ...     LatencyMetric.TOTAL_RESPONSE_TIME,
    ...     value_ms=500.0,
    ...     metadata={"request_id": "req_123", "prompt_tokens": 50}
    ... )
    >>> print(measurement.metadata["request_id"])
    req_123

    Clearing and restarting:

    >>> profiler.clear()
    >>> profiler.get_stats(LatencyMetric.TOTAL_RESPONSE_TIME).sample_count
    0

    See Also
    --------
    ThroughputProfiler : Complementary throughput profiling.
    ResponseProfiler : Combined latency and throughput profiling.
    LatencyStats : Statistics returned by get_stats().
    """

    def __init__(self, model_id: str | None = None):
        """Initialize a new LatencyProfiler instance.

        Parameters
        ----------
        model_id : str | None, optional
            Identifier for the model being profiled. Useful for
            labeling in reports and multi-model comparisons.

        Examples
        --------
        >>> from insideLLMs.latency import LatencyProfiler
        >>> profiler = LatencyProfiler(model_id="gpt-4-turbo")
        >>> print(profiler.model_id)
        gpt-4-turbo

        Without model ID:

        >>> profiler = LatencyProfiler()
        >>> print(profiler.model_id)
        None
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

        Adds a new measurement to the profiler's collection for the
        specified metric type. Returns a LatencyMeasurement object
        that can be used for immediate inspection or logging.

        Parameters
        ----------
        metric : LatencyMetric
            The type of latency being measured.
        value_ms : float
            The latency value in milliseconds.
        metadata : dict[str, Any] | None, optional
            Additional context such as request ID, model parameters,
            or custom tags.

        Returns
        -------
        LatencyMeasurement
            A measurement object containing the recorded data.

        Examples
        --------
        Recording basic measurements:

        >>> from insideLLMs.latency import LatencyProfiler, LatencyMetric
        >>> profiler = LatencyProfiler()
        >>> m = profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, 250.0)
        >>> print(f"Recorded: {m.value_ms}ms")
        Recorded: 250.0ms

        Recording with metadata:

        >>> m = profiler.record(
        ...     LatencyMetric.TIME_TO_FIRST_TOKEN,
        ...     value_ms=45.0,
        ...     metadata={
        ...         "request_id": "req_abc123",
        ...         "stream": True,
        ...         "model_version": "2024-01"
        ...     }
        ... )
        >>> print(m.metadata["stream"])
        True

        Recording multiple measurements:

        >>> for latency in [100, 120, 95, 110]:
        ...     profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, float(latency))
        >>> stats = profiler.get_stats(LatencyMetric.TOTAL_RESPONSE_TIME)
        >>> print(f"Sample count: {stats.sample_count}")
        Sample count: 5
        """
        self._measurements[metric].append(value_ms)
        return LatencyMeasurement(
            metric=metric,
            value_ms=value_ms,
            metadata=metadata or {},
        )

    def get_stats(self, metric: LatencyMetric) -> LatencyStats:
        """Get statistical summary for a specific metric.

        Calculates comprehensive statistics for all measurements of the
        specified metric type, including mean, median, standard deviation,
        and various percentiles.

        Parameters
        ----------
        metric : LatencyMetric
            The latency metric type to summarize.

        Returns
        -------
        LatencyStats
            Statistical summary of the measurements. Returns zero-valued
            stats if no measurements exist for the metric.

        Examples
        --------
        Getting statistics for recorded measurements:

        >>> from insideLLMs.latency import LatencyProfiler, LatencyMetric
        >>> profiler = LatencyProfiler()
        >>> for latency in [100.0, 120.0, 90.0, 150.0, 110.0]:
        ...     profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, latency)
        >>> stats = profiler.get_stats(LatencyMetric.TOTAL_RESPONSE_TIME)
        >>> print(f"Mean: {stats.mean_ms:.1f}ms")
        Mean: 114.0ms
        >>> print(f"p99: {stats.p99_ms:.1f}ms")
        p99: 147.5ms

        Empty metric returns zero stats:

        >>> stats = profiler.get_stats(LatencyMetric.NETWORK_LATENCY)
        >>> print(stats.sample_count)
        0
        >>> print(stats.mean_ms)
        0
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
        """Get statistics for all recorded metric types.

        Calculates statistical summaries for every metric type that
        has at least one recorded measurement.

        Returns
        -------
        dict[LatencyMetric, LatencyStats]
            Dictionary mapping each recorded metric type to its
            statistical summary.

        Examples
        --------
        >>> from insideLLMs.latency import LatencyProfiler, LatencyMetric
        >>> profiler = LatencyProfiler()
        >>> profiler.record(LatencyMetric.TIME_TO_FIRST_TOKEN, 50.0)
        LatencyMeasurement(...)
        >>> profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, 500.0)
        LatencyMeasurement(...)
        >>> all_stats = profiler.get_all_stats()
        >>> print(len(all_stats))
        2
        >>> for metric, stats in all_stats.items():
        ...     print(f"{metric.name}: {stats.mean_ms}ms")
        TIME_TO_FIRST_TOKEN: 50.0ms
        TOTAL_RESPONSE_TIME: 500.0ms
        """
        return {metric: self.get_stats(metric) for metric in self._measurements}

    @staticmethod
    def _percentile(sorted_values: list[float], p: float) -> float:
        """Calculate percentile value using linear interpolation.

        Parameters
        ----------
        sorted_values : list[float]
            Pre-sorted list of values.
        p : float
            Percentile to calculate (0-100).

        Returns
        -------
        float
            The interpolated percentile value, or 0.0 if list is empty.

        Examples
        --------
        >>> LatencyProfiler._percentile([10, 20, 30, 40, 50], 50)
        30.0
        >>> LatencyProfiler._percentile([10, 20, 30, 40, 50], 90)
        46.0
        """
        if not sorted_values:
            return 0.0
        n = len(sorted_values)
        idx = (p / 100) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        weight = idx - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    def clear(self) -> None:
        """Clear all recorded measurements.

        Removes all measurements for all metric types. The profiler
        can be reused after clearing.

        Examples
        --------
        >>> from insideLLMs.latency import LatencyProfiler, LatencyMetric
        >>> profiler = LatencyProfiler()
        >>> profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, 100.0)
        LatencyMeasurement(...)
        >>> profiler.get_stats(LatencyMetric.TOTAL_RESPONSE_TIME).sample_count
        1
        >>> profiler.clear()
        >>> profiler.get_stats(LatencyMetric.TOTAL_RESPONSE_TIME).sample_count
        0
        """
        self._measurements.clear()


class ThroughputProfiler:
    """Profile throughput characteristics of LLM model responses.

    ThroughputProfiler collects, stores, and analyzes throughput measurements
    across multiple requests. It supports different throughput metric types,
    tracks request timestamps for RPM calculation, and provides statistical
    summaries.

    Parameters
    ----------
    model_id : str | None, optional
        Identifier for the model being profiled. Used for labeling
        in reports and comparisons.

    Attributes
    ----------
    model_id : str | None
        The model identifier provided at initialization.

    Examples
    --------
    Basic throughput profiling:

    >>> from insideLLMs.latency import ThroughputProfiler, ThroughputMetric
    >>> profiler = ThroughputProfiler(model_id="gpt-4")
    >>> for tps in [70.0, 75.0, 80.0, 65.0, 72.0]:
    ...     profiler.record(ThroughputMetric.TOKENS_PER_SECOND, tps)
    >>> stats = profiler.get_stats(ThroughputMetric.TOKENS_PER_SECOND)
    >>> print(f"Mean: {stats.mean:.1f} {stats.unit}")
    Mean: 72.4 tokens/s

    Tracking requests per minute:

    >>> import time
    >>> profiler = ThroughputProfiler()
    >>> for _ in range(5):
    ...     profiler.record_request()
    ...     time.sleep(0.1)  # Simulate request interval
    >>> rpm = profiler.calculate_rpm(window_seconds=60.0)
    >>> print(f"RPM: {rpm:.1f}")
    RPM: ...

    Recording with metadata:

    >>> measurement = profiler.record(
    ...     ThroughputMetric.TOKENS_PER_SECOND,
    ...     value=85.5,
    ...     metadata={"batch_size": 1, "max_tokens": 500}
    ... )
    >>> print(measurement.unit)
    tokens/s

    Getting all statistics at once:

    >>> all_stats = profiler.get_all_stats()
    >>> for metric, stats in all_stats.items():
    ...     print(f"{metric.value}: {stats.mean:.1f} {stats.unit}")

    See Also
    --------
    LatencyProfiler : Complementary latency profiling.
    ResponseProfiler : Combined latency and throughput profiling.
    ThroughputStats : Statistics returned by get_stats().
    """

    def __init__(self, model_id: str | None = None):
        """Initialize a new ThroughputProfiler instance.

        Parameters
        ----------
        model_id : str | None, optional
            Identifier for the model being profiled. Useful for
            labeling in reports and multi-model comparisons.

        Examples
        --------
        >>> from insideLLMs.latency import ThroughputProfiler
        >>> profiler = ThroughputProfiler(model_id="claude-3-opus")
        >>> print(profiler.model_id)
        claude-3-opus
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

        Adds a new measurement to the profiler's collection for the
        specified metric type. Returns a ThroughputMeasurement object
        that can be used for immediate inspection or logging.

        Parameters
        ----------
        metric : ThroughputMetric
            The type of throughput being measured.
        value : float
            The throughput value.
        metadata : dict[str, Any] | None, optional
            Additional context such as batch size, model parameters,
            or custom tags.

        Returns
        -------
        ThroughputMeasurement
            A measurement object containing the recorded data.

        Examples
        --------
        >>> from insideLLMs.latency import ThroughputProfiler, ThroughputMetric
        >>> profiler = ThroughputProfiler()
        >>> m = profiler.record(ThroughputMetric.TOKENS_PER_SECOND, 75.5)
        >>> print(f"{m.value} {m.unit}")
        75.5 tokens/s

        Recording with metadata:

        >>> m = profiler.record(
        ...     ThroughputMetric.CHARACTERS_PER_SECOND,
        ...     value=320.0,
        ...     metadata={"model": "gpt-4", "temperature": 0.7}
        ... )
        >>> print(m.metadata["model"])
        gpt-4
        """
        self._measurements[metric].append(value)
        return ThroughputMeasurement(
            metric=metric,
            value=value,
            unit=self._get_unit(metric),
            metadata=metadata or {},
        )

    def record_request(self) -> None:
        """Record a request timestamp for RPM calculation.

        Call this method each time a request is made to track
        requests per minute (RPM). Uses wall-clock time.

        Examples
        --------
        >>> from insideLLMs.latency import ThroughputProfiler
        >>> profiler = ThroughputProfiler()
        >>> profiler.record_request()
        >>> profiler.record_request()
        >>> profiler.record_request()
        >>> rpm = profiler.calculate_rpm(window_seconds=60.0)
        """
        self._request_times.append(time.time())

    def get_stats(self, metric: ThroughputMetric) -> ThroughputStats:
        """Get statistical summary for a specific metric.

        Calculates comprehensive statistics for all measurements of the
        specified metric type, including mean, median, and standard deviation.

        Parameters
        ----------
        metric : ThroughputMetric
            The throughput metric type to summarize.

        Returns
        -------
        ThroughputStats
            Statistical summary of the measurements. Returns zero-valued
            stats if no measurements exist for the metric.

        Examples
        --------
        >>> from insideLLMs.latency import ThroughputProfiler, ThroughputMetric
        >>> profiler = ThroughputProfiler()
        >>> for tps in [60.0, 70.0, 80.0, 90.0, 75.0]:
        ...     profiler.record(ThroughputMetric.TOKENS_PER_SECOND, tps)
        >>> stats = profiler.get_stats(ThroughputMetric.TOKENS_PER_SECOND)
        >>> print(f"Mean: {stats.mean:.1f}, Sustained: {stats.sustained_rate:.1f}")
        Mean: 75.0, Sustained: 63.8

        Empty metric returns zero stats:

        >>> stats = profiler.get_stats(ThroughputMetric.WORDS_PER_SECOND)
        >>> print(stats.sample_count)
        0
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

        Computes the request rate based on timestamps recorded via
        record_request(). Uses a sliding window to calculate the rate.

        Parameters
        ----------
        window_seconds : float, optional
            Time window to consider in seconds. Default is 60.0 (one minute).

        Returns
        -------
        float
            Estimated requests per minute based on recent activity.
            Returns 0.0 if no requests have been recorded.

        Examples
        --------
        >>> import time
        >>> from insideLLMs.latency import ThroughputProfiler
        >>> profiler = ThroughputProfiler()
        >>> for _ in range(10):
        ...     profiler.record_request()
        ...     time.sleep(0.1)  # 10 requests over ~1 second
        >>> rpm = profiler.calculate_rpm(window_seconds=60.0)
        >>> print(f"Estimated RPM: {rpm:.0f}")
        Estimated RPM: ...

        Using a shorter window:

        >>> rpm = profiler.calculate_rpm(window_seconds=10.0)
        >>> # Higher rate since more concentrated
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
        """Get statistics for all recorded metric types.

        Calculates statistical summaries for every metric type that
        has at least one recorded measurement.

        Returns
        -------
        dict[ThroughputMetric, ThroughputStats]
            Dictionary mapping each recorded metric type to its
            statistical summary.

        Examples
        --------
        >>> from insideLLMs.latency import ThroughputProfiler, ThroughputMetric
        >>> profiler = ThroughputProfiler()
        >>> profiler.record(ThroughputMetric.TOKENS_PER_SECOND, 75.0)
        ThroughputMeasurement(...)
        >>> profiler.record(ThroughputMetric.CHARACTERS_PER_SECOND, 300.0)
        ThroughputMeasurement(...)
        >>> all_stats = profiler.get_all_stats()
        >>> print(len(all_stats))
        2
        """
        return {metric: self.get_stats(metric) for metric in self._measurements}

    @staticmethod
    def _get_unit(metric: ThroughputMetric) -> str:
        """Get the unit string for a throughput metric.

        Parameters
        ----------
        metric : ThroughputMetric
            The metric to get the unit for.

        Returns
        -------
        str
            Human-readable unit string (e.g., "tokens/s", "chars/s").

        Examples
        --------
        >>> ThroughputProfiler._get_unit(ThroughputMetric.TOKENS_PER_SECOND)
        'tokens/s'
        >>> ThroughputProfiler._get_unit(ThroughputMetric.REQUESTS_PER_MINUTE)
        'req/min'
        """
        units = {
            ThroughputMetric.TOKENS_PER_SECOND: "tokens/s",
            ThroughputMetric.CHARACTERS_PER_SECOND: "chars/s",
            ThroughputMetric.REQUESTS_PER_MINUTE: "req/min",
            ThroughputMetric.WORDS_PER_SECOND: "words/s",
        }
        return units.get(metric, "unit")

    def clear(self) -> None:
        """Clear all recorded measurements and request timestamps.

        Removes all measurements for all metric types and clears
        the request timestamp history. The profiler can be reused
        after clearing.

        Examples
        --------
        >>> from insideLLMs.latency import ThroughputProfiler, ThroughputMetric
        >>> profiler = ThroughputProfiler()
        >>> profiler.record(ThroughputMetric.TOKENS_PER_SECOND, 75.0)
        ThroughputMeasurement(...)
        >>> profiler.record_request()
        >>> profiler.clear()
        >>> profiler.get_stats(ThroughputMetric.TOKENS_PER_SECOND).sample_count
        0
        """
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
