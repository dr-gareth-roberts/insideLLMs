"""Tests for response latency and throughput profiling."""

import time

import pytest

from insideLLMs.latency import (
    LatencyMeasurement,
    LatencyMetric,
    LatencyProfiler,
    LatencyStats,
    LatencyTimer,
    PerformanceComparator,
    PerformanceLevel,
    PerformanceReport,
    ResponseProfile,
    ResponseProfiler,
    StreamingProfiler,
    ThroughputMeasurement,
    ThroughputMetric,
    ThroughputProfiler,
    ThroughputStats,
    calculate_throughput,
    create_timer,
    measure_latency,
    profile_response,
    quick_performance_check,
)

pytestmark = pytest.mark.performance


class TestLatencyMetric:
    """Tests for LatencyMetric enum."""

    def test_all_metrics_exist(self):
        """Test all expected metrics exist."""
        assert LatencyMetric.TIME_TO_FIRST_TOKEN
        assert LatencyMetric.TOTAL_RESPONSE_TIME
        assert LatencyMetric.INTER_TOKEN_LATENCY
        assert LatencyMetric.PROCESSING_TIME
        assert LatencyMetric.NETWORK_LATENCY


class TestThroughputMetric:
    """Tests for ThroughputMetric enum."""

    def test_all_metrics_exist(self):
        """Test all expected metrics exist."""
        assert ThroughputMetric.TOKENS_PER_SECOND
        assert ThroughputMetric.CHARACTERS_PER_SECOND
        assert ThroughputMetric.REQUESTS_PER_MINUTE
        assert ThroughputMetric.WORDS_PER_SECOND


class TestPerformanceLevel:
    """Tests for PerformanceLevel enum."""

    def test_all_levels_exist(self):
        """Test all expected levels exist."""
        assert PerformanceLevel.EXCELLENT
        assert PerformanceLevel.GOOD
        assert PerformanceLevel.ACCEPTABLE
        assert PerformanceLevel.POOR
        assert PerformanceLevel.CRITICAL


class TestLatencyMeasurement:
    """Tests for LatencyMeasurement dataclass."""

    def test_basic_creation(self):
        """Test basic measurement creation."""
        measurement = LatencyMeasurement(
            metric=LatencyMetric.TOTAL_RESPONSE_TIME,
            value_ms=100.0,
        )
        assert measurement.value_ms == 100.0
        assert measurement.metric == LatencyMetric.TOTAL_RESPONSE_TIME

    def test_to_dict(self):
        """Test conversion to dictionary."""
        measurement = LatencyMeasurement(
            metric=LatencyMetric.TIME_TO_FIRST_TOKEN,
            value_ms=50.0,
            metadata={"model": "test"},
        )
        d = measurement.to_dict()
        assert d["metric"] == "time_to_first_token"
        assert d["value_ms"] == 50.0
        assert d["metadata"]["model"] == "test"


class TestThroughputMeasurement:
    """Tests for ThroughputMeasurement dataclass."""

    def test_basic_creation(self):
        """Test basic measurement creation."""
        measurement = ThroughputMeasurement(
            metric=ThroughputMetric.TOKENS_PER_SECOND,
            value=50.0,
            unit="tokens/s",
        )
        assert measurement.value == 50.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        measurement = ThroughputMeasurement(
            metric=ThroughputMetric.TOKENS_PER_SECOND,
            value=100.0,
            unit="tokens/s",
        )
        d = measurement.to_dict()
        assert d["metric"] == "tokens_per_second"
        assert d["value"] == 100.0


class TestLatencyStats:
    """Tests for LatencyStats dataclass."""

    def test_coefficient_of_variation(self):
        """Test coefficient of variation calculation."""
        stats = LatencyStats(
            mean_ms=100,
            median_ms=100,
            std_ms=20,
            min_ms=80,
            max_ms=120,
            p50_ms=100,
            p90_ms=115,
            p95_ms=118,
            p99_ms=120,
            sample_count=100,
        )
        assert stats.coefficient_of_variation == 0.2

    def test_is_consistent(self):
        """Test consistency check."""
        consistent = LatencyStats(
            mean_ms=100,
            median_ms=100,
            std_ms=10,  # Low std
            min_ms=90,
            max_ms=110,
            p50_ms=100,
            p90_ms=108,
            p95_ms=109,
            p99_ms=110,
            sample_count=100,
        )
        assert consistent.is_consistent

        inconsistent = LatencyStats(
            mean_ms=100,
            median_ms=100,
            std_ms=50,  # High std
            min_ms=50,
            max_ms=150,
            p50_ms=100,
            p90_ms=140,
            p95_ms=145,
            p99_ms=150,
            sample_count=100,
        )
        assert not inconsistent.is_consistent

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = LatencyStats(
            mean_ms=100,
            median_ms=100,
            std_ms=20,
            min_ms=80,
            max_ms=120,
            p50_ms=100,
            p90_ms=115,
            p95_ms=118,
            p99_ms=120,
            sample_count=100,
        )
        d = stats.to_dict()
        assert d["mean_ms"] == 100
        assert "coefficient_of_variation" in d


class TestThroughputStats:
    """Tests for ThroughputStats dataclass."""

    def test_sustained_rate(self):
        """Test sustained rate calculation."""
        stats = ThroughputStats(
            mean=50,
            median=50,
            std=10,
            min_val=40,
            max_val=60,
            sample_count=100,
            unit="tokens/s",
        )
        assert stats.sustained_rate == 40  # mean - std

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = ThroughputStats(
            mean=50,
            median=50,
            std=10,
            min_val=40,
            max_val=60,
            sample_count=100,
            unit="tokens/s",
        )
        d = stats.to_dict()
        assert d["mean"] == 50
        assert d["unit"] == "tokens/s"


class TestResponseProfile:
    """Tests for ResponseProfile dataclass."""

    def test_inter_token_latency(self):
        """Test inter-token latency calculation."""
        profile = ResponseProfile(
            prompt_tokens=10,
            completion_tokens=100,
            time_to_first_token_ms=50,
            total_time_ms=550,  # 50ms TTFT + 500ms generation
            tokens_per_second=200,
        )
        # (550 - 50) / (100 - 1) ≈ 5.05ms
        assert profile.inter_token_latency_ms == pytest.approx(5.05, rel=0.1)

    def test_inter_token_latency_without_ttft(self):
        """Test inter-token latency without TTFT."""
        profile = ResponseProfile(
            prompt_tokens=10,
            completion_tokens=100,
            time_to_first_token_ms=None,
            total_time_ms=500,
            tokens_per_second=200,
        )
        assert profile.inter_token_latency_ms == 5.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        profile = ResponseProfile(
            prompt_tokens=10,
            completion_tokens=100,
            time_to_first_token_ms=50,
            total_time_ms=500,
            tokens_per_second=200,
            model_id="test-model",
        )
        d = profile.to_dict()
        assert d["completion_tokens"] == 100
        assert d["model_id"] == "test-model"


class TestLatencyTimer:
    """Tests for LatencyTimer."""

    def test_basic_timing(self):
        """Test basic timing functionality."""
        timer = LatencyTimer()
        timer.start()
        time.sleep(0.01)  # 10ms
        timer.stop()
        assert timer.total_time_ms is not None
        assert timer.total_time_ms >= 10

    def test_context_manager(self):
        """Test context manager usage."""
        with LatencyTimer() as timer:
            time.sleep(0.01)
        assert timer.total_time_ms is not None
        assert timer.total_time_ms >= 10

    def test_first_token_marking(self):
        """Test first token marking."""
        timer = LatencyTimer()
        timer.start()
        time.sleep(0.005)
        timer.mark_first_token()
        time.sleep(0.005)
        timer.stop()

        assert timer.time_to_first_token_ms is not None
        assert timer.time_to_first_token_ms >= 5
        assert timer.total_time_ms is not None
        assert timer.total_time_ms > timer.time_to_first_token_ms

    def test_token_marking(self):
        """Test token time marking."""
        timer = LatencyTimer()
        timer.start()
        timer.mark_token()
        time.sleep(0.005)
        timer.mark_token()
        time.sleep(0.005)
        timer.mark_token()
        timer.stop()

        latencies = timer.inter_token_latencies_ms
        assert len(latencies) == 2
        assert all(l >= 5 for l in latencies)


class TestLatencyProfiler:
    """Tests for LatencyProfiler."""

    def test_record_measurement(self):
        """Test recording a measurement."""
        profiler = LatencyProfiler()
        measurement = profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, 100.0)
        assert measurement.value_ms == 100.0

    def test_get_stats(self):
        """Test getting statistics."""
        profiler = LatencyProfiler()
        for i in range(10):
            profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, 100 + i)

        stats = profiler.get_stats(LatencyMetric.TOTAL_RESPONSE_TIME)
        assert stats.sample_count == 10
        assert stats.mean_ms == 104.5
        assert stats.min_ms == 100
        assert stats.max_ms == 109

    def test_get_stats_empty(self):
        """Test getting stats with no data."""
        profiler = LatencyProfiler()
        stats = profiler.get_stats(LatencyMetric.TOTAL_RESPONSE_TIME)
        assert stats.sample_count == 0
        assert stats.mean_ms == 0

    def test_get_all_stats(self):
        """Test getting all statistics."""
        profiler = LatencyProfiler()
        profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, 100)
        profiler.record(LatencyMetric.TIME_TO_FIRST_TOKEN, 50)

        all_stats = profiler.get_all_stats()
        assert LatencyMetric.TOTAL_RESPONSE_TIME in all_stats
        assert LatencyMetric.TIME_TO_FIRST_TOKEN in all_stats

    def test_clear(self):
        """Test clearing measurements."""
        profiler = LatencyProfiler()
        profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, 100)
        profiler.clear()
        stats = profiler.get_stats(LatencyMetric.TOTAL_RESPONSE_TIME)
        assert stats.sample_count == 0


class TestThroughputProfiler:
    """Tests for ThroughputProfiler."""

    def test_record_measurement(self):
        """Test recording a measurement."""
        profiler = ThroughputProfiler()
        measurement = profiler.record(ThroughputMetric.TOKENS_PER_SECOND, 50.0)
        assert measurement.value == 50.0
        assert measurement.unit == "tokens/s"

    def test_get_stats(self):
        """Test getting statistics."""
        profiler = ThroughputProfiler()
        for i in range(10):
            profiler.record(ThroughputMetric.TOKENS_PER_SECOND, 50 + i)

        stats = profiler.get_stats(ThroughputMetric.TOKENS_PER_SECOND)
        assert stats.sample_count == 10
        assert stats.mean == 54.5

    def test_rpm_calculation(self):
        """Test requests per minute calculation."""
        profiler = ThroughputProfiler()
        # Record 10 requests
        for _ in range(10):
            profiler.record_request()

        rpm = profiler.calculate_rpm(window_seconds=60)
        # All requests happened nearly instantly, so RPM should be low
        # but not zero
        assert rpm >= 0

    def test_clear(self):
        """Test clearing measurements."""
        profiler = ThroughputProfiler()
        profiler.record(ThroughputMetric.TOKENS_PER_SECOND, 50)
        profiler.record_request()
        profiler.clear()
        stats = profiler.get_stats(ThroughputMetric.TOKENS_PER_SECOND)
        assert stats.sample_count == 0


class TestResponseProfiler:
    """Tests for ResponseProfiler."""

    def test_profile_response(self):
        """Test profiling a response."""
        profiler = ResponseProfiler(model_id="test-model")
        profile = profiler.profile_response(
            prompt="Hello",
            response="World, how are you today?",
            total_time_ms=100,
            time_to_first_token_ms=20,
        )

        assert profile.prompt_tokens > 0
        assert profile.completion_tokens > 0
        assert profile.total_time_ms == 100
        assert profile.tokens_per_second > 0

    def test_generate_report(self):
        """Test generating a performance report."""
        profiler = ResponseProfiler(model_id="test-model")

        # Add some profiles
        for i in range(5):
            profiler.profile_response(
                prompt="Test prompt",
                response="Test response " * 10,
                total_time_ms=100 + i * 10,
                time_to_first_token_ms=20 + i * 2,
            )

        report = profiler.generate_report()

        assert report.model_id == "test-model"
        assert len(report.response_profiles) == 5
        assert LatencyMetric.TOTAL_RESPONSE_TIME in report.latency_stats
        assert report.performance_level in PerformanceLevel

    def test_bottleneck_detection(self):
        """Test bottleneck detection."""
        profiler = ResponseProfiler()

        # Add profiles with high TTFT
        for _ in range(5):
            profiler.profile_response(
                prompt="Test",
                response="Response",
                total_time_ms=2000,
                time_to_first_token_ms=1000,  # High TTFT
            )

        report = profiler.generate_report()
        assert len(report.bottlenecks) > 0

    def test_recommendations_generated(self):
        """Test that recommendations are generated."""
        profiler = ResponseProfiler()
        profiler.profile_response(
            prompt="Test",
            response="Response",
            total_time_ms=5000,  # Very slow
            time_to_first_token_ms=2000,
        )

        report = profiler.generate_report()
        assert len(report.recommendations) > 0

    def test_clear(self):
        """Test clearing profiles."""
        profiler = ResponseProfiler()
        profiler.profile_response(
            prompt="Test",
            response="Response",
            total_time_ms=100,
        )
        profiler.clear()
        report = profiler.generate_report()
        assert len(report.response_profiles) == 0


class TestPerformanceComparator:
    """Tests for PerformanceComparator."""

    def _create_report(
        self, model_id: str, mean_latency: float, mean_throughput: float
    ) -> PerformanceReport:
        """Helper to create a performance report."""
        return PerformanceReport(
            model_id=model_id,
            latency_stats={
                LatencyMetric.TOTAL_RESPONSE_TIME: LatencyStats(
                    mean_ms=mean_latency,
                    median_ms=mean_latency,
                    std_ms=mean_latency * 0.1,
                    min_ms=mean_latency * 0.8,
                    max_ms=mean_latency * 1.2,
                    p50_ms=mean_latency,
                    p90_ms=mean_latency * 1.1,
                    p95_ms=mean_latency * 1.15,
                    p99_ms=mean_latency * 1.2,
                    sample_count=100,
                )
            },
            throughput_stats={
                ThroughputMetric.TOKENS_PER_SECOND: ThroughputStats(
                    mean=mean_throughput,
                    median=mean_throughput,
                    std=mean_throughput * 0.1,
                    min_val=mean_throughput * 0.8,
                    max_val=mean_throughput * 1.2,
                    sample_count=100,
                    unit="tokens/s",
                )
            },
            response_profiles=[],
            performance_level=PerformanceLevel.GOOD,
            bottlenecks=[],
            recommendations=[],
        )

    def test_compare_latency(self):
        """Test latency comparison."""
        comparator = PerformanceComparator()
        comparator.add_report("model_a", self._create_report("model_a", 100, 50))
        comparator.add_report("model_b", self._create_report("model_b", 200, 30))

        comparison = comparator.compare_latency()
        assert "model_a" in comparison
        assert "model_b" in comparison
        assert comparison["model_a"]["rank"] == 1  # Lower latency = better
        assert comparison["model_b"]["rank"] == 2

    def test_compare_throughput(self):
        """Test throughput comparison."""
        comparator = PerformanceComparator()
        comparator.add_report("model_a", self._create_report("model_a", 100, 50))
        comparator.add_report("model_b", self._create_report("model_b", 200, 30))

        comparison = comparator.compare_throughput()
        assert comparison["model_a"]["rank"] == 1  # Higher throughput = better
        assert comparison["model_b"]["rank"] == 2

    def test_get_best_performer_latency(self):
        """Test getting best performer by latency."""
        comparator = PerformanceComparator()
        comparator.add_report("model_a", self._create_report("model_a", 100, 50))
        comparator.add_report("model_b", self._create_report("model_b", 200, 30))

        name, report = comparator.get_best_performer(by="latency")
        assert name == "model_a"

    def test_get_best_performer_throughput(self):
        """Test getting best performer by throughput."""
        comparator = PerformanceComparator()
        comparator.add_report("model_a", self._create_report("model_a", 100, 50))
        comparator.add_report("model_b", self._create_report("model_b", 200, 30))

        name, report = comparator.get_best_performer(by="throughput")
        assert name == "model_a"

    def test_generate_comparison_report(self):
        """Test generating comparison report."""
        comparator = PerformanceComparator()
        comparator.add_report("model_a", self._create_report("model_a", 100, 50))
        comparator.add_report("model_b", self._create_report("model_b", 200, 30))

        report = comparator.generate_comparison_report()
        assert report["n_configurations"] == 2
        assert "latency_comparison" in report
        assert "throughput_comparison" in report


class TestStreamingProfiler:
    """Tests for StreamingProfiler."""

    def test_stream_profiling(self):
        """Test streaming response profiling."""
        profiler = StreamingProfiler()
        profiler.start_stream()

        # Simulate streaming chunks
        profiler.record_chunk("Hello", is_first=True)
        time.sleep(0.005)
        profiler.record_chunk("world")
        time.sleep(0.005)
        profiler.record_chunk("!")

        metrics = profiler.end_stream()

        assert metrics["time_to_first_token_ms"] is not None
        assert metrics["total_time_ms"] is not None
        assert metrics["n_chunks"] == 3
        assert metrics["tokens_received"] > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_timer(self):
        """Test create_timer function."""
        timer = create_timer()
        assert isinstance(timer, LatencyTimer)

    def test_measure_latency(self):
        """Test measure_latency function."""

        def slow_func():
            time.sleep(0.01)

        stats = measure_latency(slow_func, warmup_runs=1, measurement_runs=3)
        assert stats.sample_count == 3
        assert stats.mean_ms >= 10

    def test_calculate_throughput(self):
        """Test calculate_throughput function."""
        # 100 tokens in 1000ms = 100 tokens/s
        tps = calculate_throughput(100, 1000)
        assert tps == 100

    def test_calculate_throughput_zero_time(self):
        """Test calculate_throughput with zero time."""
        tps = calculate_throughput(100, 0)
        assert tps == 0

    def test_profile_response(self):
        """Test profile_response function."""
        profile = profile_response(
            prompt="Hello",
            response="World",
            time_ms=100,
            model_id="test",
        )
        assert isinstance(profile, ResponseProfile)
        assert profile.total_time_ms == 100

    def test_quick_performance_check(self):
        """Test quick_performance_check function."""
        result = quick_performance_check(
            response_times_ms=[100, 150, 200, 250, 300],
            tokens_per_response=[50, 60, 70, 80, 90],
        )

        assert result["n_responses"] == 5
        assert result["latency"]["mean_ms"] == 200
        assert result["throughput"] is not None
        assert result["performance_level"] in [
            "excellent",
            "good",
            "acceptable",
            "poor",
            "critical",
        ]

    def test_quick_performance_check_empty(self):
        """Test quick_performance_check with empty data."""
        result = quick_performance_check([])
        assert result["n_responses"] == 0
        assert result["latency"] is None

    def test_quick_performance_check_no_tokens(self):
        """Test quick_performance_check without token counts."""
        result = quick_performance_check([100, 200, 300])
        assert result["n_responses"] == 3
        assert result["throughput"] is None


class TestPerformanceReport:
    """Tests for PerformanceReport."""

    def test_overall_score(self):
        """Test overall score calculation."""
        report = PerformanceReport(
            model_id="test",
            latency_stats={
                LatencyMetric.TOTAL_RESPONSE_TIME: LatencyStats(
                    mean_ms=100,
                    median_ms=100,
                    std_ms=10,
                    min_ms=90,
                    max_ms=110,
                    p50_ms=100,
                    p90_ms=108,
                    p95_ms=109,
                    p99_ms=110,
                    sample_count=100,
                )
            },
            throughput_stats={
                ThroughputMetric.TOKENS_PER_SECOND: ThroughputStats(
                    mean=50,
                    median=50,
                    std=5,
                    min_val=45,
                    max_val=55,
                    sample_count=100,
                    unit="tokens/s",
                )
            },
            response_profiles=[],
            performance_level=PerformanceLevel.GOOD,
            bottlenecks=[],
            recommendations=[],
        )

        score = report.overall_score
        assert 0 <= score <= 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        report = PerformanceReport(
            model_id="test",
            latency_stats={},
            throughput_stats={},
            response_profiles=[],
            performance_level=PerformanceLevel.GOOD,
            bottlenecks=["test bottleneck"],
            recommendations=["test recommendation"],
        )

        d = report.to_dict()
        assert d["model_id"] == "test"
        assert d["performance_level"] == "good"
        assert "overall_score" in d


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_measurement(self):
        """Test with single measurement."""
        profiler = LatencyProfiler()
        profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, 100)
        stats = profiler.get_stats(LatencyMetric.TOTAL_RESPONSE_TIME)
        assert stats.sample_count == 1
        assert stats.std_ms == 0

    def test_percentile_single_value(self):
        """Test percentile calculation with single value."""
        profiler = LatencyProfiler()
        profiler.record(LatencyMetric.TOTAL_RESPONSE_TIME, 100)
        stats = profiler.get_stats(LatencyMetric.TOTAL_RESPONSE_TIME)
        assert stats.p99_ms == 100

    def test_timer_without_start(self):
        """Test timer operations without start."""
        timer = LatencyTimer()
        assert timer.total_time_ms is None
        assert timer.time_to_first_token_ms is None

    def test_streaming_without_start(self):
        """Test streaming profiler without start."""
        profiler = StreamingProfiler()
        profiler.record_chunk("test")  # Should not crash
        metrics = profiler.end_stream()
        assert metrics == {}

    def test_empty_comparator(self):
        """Test comparator with no reports."""
        comparator = PerformanceComparator()
        with pytest.raises(ValueError):
            comparator.get_best_performer()

    def test_zero_latency(self):
        """Test with zero latency measurements."""
        stats = LatencyStats(
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
        assert stats.coefficient_of_variation == 0

    def test_zero_tokens_profile(self):
        """Test profile with zero tokens."""
        profile = ResponseProfile(
            prompt_tokens=0,
            completion_tokens=0,
            time_to_first_token_ms=None,
            total_time_ms=100,
            tokens_per_second=0,
        )
        assert profile.inter_token_latency_ms == 0
