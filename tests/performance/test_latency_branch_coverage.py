"""Additional branch coverage for latency module edge paths."""

from __future__ import annotations

import pytest

import insideLLMs.performance.latency as latency_module
from insideLLMs.performance.latency import (
    LatencyMetric,
    LatencyProfiler,
    LatencyStats,
    LatencyTimer,
    PerformanceComparator,
    PerformanceLevel,
    PerformanceReport,
    ResponseProfiler,
    ThroughputMetric,
    ThroughputProfiler,
    ThroughputStats,
    measure_latency,
    quick_performance_check,
)

pytestmark = pytest.mark.performance


def _latency_stats(mean: float, std: float, p99: float) -> LatencyStats:
    return LatencyStats(
        mean_ms=mean,
        median_ms=mean,
        std_ms=std,
        min_ms=mean,
        max_ms=mean,
        p50_ms=mean,
        p90_ms=mean,
        p95_ms=mean,
        p99_ms=p99,
        sample_count=5,
    )


def _throughput_stats(mean: float) -> ThroughputStats:
    return ThroughputStats(
        mean=mean,
        median=mean,
        std=1.0,
        min_val=mean,
        max_val=mean,
        sample_count=5,
        unit="tokens/s",
    )


def _report(
    latency_stats: dict[LatencyMetric, LatencyStats],
    throughput_stats: dict[ThroughputMetric, ThroughputStats],
) -> PerformanceReport:
    return PerformanceReport(
        model_id="m",
        latency_stats=latency_stats,
        throughput_stats=throughput_stats,
        response_profiles=[],
        performance_level=PerformanceLevel.GOOD,
        bottlenecks=[],
        recommendations=[],
    )


def test_latency_timer_and_profiler_empty_edge_paths():
    timer = LatencyTimer().start()
    timer.mark_first_token()
    first_ttft = timer.time_to_first_token_ms
    timer.mark_first_token()  # Should be ignored on second call.
    assert timer.time_to_first_token_ms == first_ttft

    timer_single = LatencyTimer().start()
    timer_single.mark_token()
    timer_single.stop()
    assert timer_single.inter_token_latencies_ms == []

    assert LatencyProfiler._percentile([], 90) == 0.0
    profiler = LatencyProfiler()
    profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, 1.0)
    profiler.clear()
    assert profiler.get_stats(LatencyMetric.TOTAL_RESPONSE_TIME).sample_count == 0


def test_throughput_profiler_calculate_rpm_branches(monkeypatch):
    profiler = ThroughputProfiler()
    assert profiler.calculate_rpm(window_seconds=60.0) == 0.0

    monkeypatch.setattr("insideLLMs.performance.latency.time.time", lambda: 100.0)

    profiler._request_times = [99.0]
    assert profiler.calculate_rpm(window_seconds=60.0) == 1.0

    profiler._request_times = [100.0, 100.0]
    assert profiler.calculate_rpm(window_seconds=60.0) == 0.0


def test_response_profiler_bottlenecks_and_recommendations():
    profiler = ResponseProfiler(model_id="m")

    latency_stats = {
        LatencyMetric.TIME_TO_FIRST_TOKEN: _latency_stats(mean=700.0, std=10.0, p99=2500.0),
        LatencyMetric.TOTAL_RESPONSE_TIME: _latency_stats(mean=1200.0, std=500.0, p99=1400.0),
    }
    throughput_stats = {
        ThroughputMetric.TOKENS_PER_SECOND: _throughput_stats(mean=15.0),
    }

    bottlenecks = profiler._detect_bottlenecks(latency_stats, throughput_stats)
    assert any("tail latency" in b for b in bottlenecks)
    assert any("Inconsistent response times" in b for b in bottlenecks)
    assert any("Low throughput" in b for b in bottlenecks)

    recs = profiler._generate_recommendations(latency_stats, throughput_stats, bottlenecks)
    assert any("request batching" in r for r in recs)
    assert any("dedicated inference endpoints" in r for r in recs)


def test_response_profiler_classification_thresholds():
    profiler = ResponseProfiler(model_id="m")

    # Latency thresholds.
    assert (
        profiler._classify_performance(
            {LatencyMetric.TOTAL_RESPONSE_TIME: _latency_stats(300.0, 10.0, 320.0)},
            {},
        )
        == PerformanceLevel.GOOD
    )
    assert (
        profiler._classify_performance(
            {LatencyMetric.TOTAL_RESPONSE_TIME: _latency_stats(700.0, 10.0, 720.0)},
            {},
        )
        == PerformanceLevel.ACCEPTABLE
    )
    assert (
        profiler._classify_performance(
            {LatencyMetric.TOTAL_RESPONSE_TIME: _latency_stats(1500.0, 10.0, 1520.0)},
            {},
        )
        == PerformanceLevel.POOR
    )

    # Throughput thresholds.
    assert (
        profiler._classify_performance(
            {},
            {ThroughputMetric.TOKENS_PER_SECOND: _throughput_stats(60.0)},
        )
        == PerformanceLevel.GOOD
    )
    assert (
        profiler._classify_performance(
            {},
            {ThroughputMetric.TOKENS_PER_SECOND: _throughput_stats(30.0)},
        )
        == PerformanceLevel.ACCEPTABLE
    )
    assert (
        profiler._classify_performance(
            {},
            {ThroughputMetric.TOKENS_PER_SECOND: _throughput_stats(15.0)},
        )
        == PerformanceLevel.POOR
    )


def test_performance_comparator_fallback_when_metric_missing():
    latency_only = PerformanceComparator()
    latency_only.add_report(
        "a", _report({}, {ThroughputMetric.TOKENS_PER_SECOND: _throughput_stats(50.0)})
    )
    name_a, _ = latency_only.get_best_performer(by="latency")
    assert name_a == "a"

    throughput_only = PerformanceComparator()
    throughput_only.add_report(
        "b",
        _report({LatencyMetric.TOTAL_RESPONSE_TIME: _latency_stats(300.0, 10.0, 320.0)}, {}),
    )
    name_b, _ = throughput_only.get_best_performer(by="throughput")
    assert name_b == "b"


def test_measure_latency_returns_zero_stats_when_timer_records_none(monkeypatch):
    class NullTimer:
        def __enter__(self) -> "NullTimer":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        @property
        def total_time_ms(self):
            return None

    monkeypatch.setattr(latency_module, "LatencyTimer", NullTimer)
    stats = measure_latency(lambda: None, measurement_runs=2)
    assert stats.sample_count == 0
    assert stats.mean_ms == 0


def test_quick_performance_check_threshold_branches_and_throughput():
    assert quick_performance_check([100.0, 120.0])["performance_level"] == "excellent"
    assert quick_performance_check([300.0, 350.0])["performance_level"] == "good"
    assert quick_performance_check([700.0, 900.0])["performance_level"] == "acceptable"
    assert quick_performance_check([1500.0, 1700.0])["performance_level"] == "poor"
    assert quick_performance_check([2500.0, 2700.0])["performance_level"] == "critical"

    with_throughput = quick_performance_check([200.0, 400.0], tokens_per_response=[100, 200])
    assert with_throughput["throughput"] is not None
    assert with_throughput["latency"]["mean_ms"] == 300.0
