"""Tests for model comparison utilities."""

import pytest

from insideLLMs.comparison import (
    ComparisonMetric,
    ComparisonResult,
    CostCalculator,
    CostEstimate,
    LatencyProfile,
    MetricSummary,
    MetricValue,
    ModelComparator,
    ModelProfile,
    PerformanceTracker,
    QualityMetrics,
    create_comparison_table,
    rank_models,
)


class TestMetricValue:
    """Tests for MetricValue."""

    def test_basic_creation(self):
        """Test basic metric value creation."""
        mv = MetricValue(value=0.95, unit="%")
        assert mv.value == 0.95
        assert mv.unit == "%"
        assert mv.timestamp is not None

    def test_with_metadata(self):
        """Test metric value with metadata."""
        mv = MetricValue(value=100, metadata={"source": "test"})
        assert mv.metadata["source"] == "test"


class TestMetricSummary:
    """Tests for MetricSummary."""

    def test_from_values_basic(self):
        """Test creating summary from values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        summary = MetricSummary.from_values("test", values)

        assert summary.name == "test"
        assert summary.mean == 3.0
        assert summary.min == 1.0
        assert summary.max == 5.0
        assert summary.median == 3.0
        assert summary.count == 5

    def test_from_values_empty(self):
        """Test creating summary from empty values."""
        summary = MetricSummary.from_values("test", [])

        assert summary.count == 0
        assert summary.mean == 0.0

    def test_from_values_single(self):
        """Test creating summary from single value."""
        summary = MetricSummary.from_values("test", [5.0])

        assert summary.mean == 5.0
        assert summary.std == 0.0
        assert summary.count == 1

    def test_percentiles(self):
        """Test percentile calculations."""
        values = list(range(1, 101))  # 1 to 100
        summary = MetricSummary.from_values("test", values)

        assert summary.percentile_95 >= 90
        assert summary.percentile_99 >= 95


class TestModelProfile:
    """Tests for ModelProfile."""

    def test_basic_creation(self):
        """Test basic profile creation."""
        profile = ModelProfile(model_name="test-model")
        assert profile.model_name == "test-model"
        assert len(profile.metrics) == 0

    def test_add_metric(self):
        """Test adding metrics."""
        profile = ModelProfile(model_name="test-model")
        profile.add_metric("accuracy", [0.9, 0.92, 0.88], "%")

        assert "accuracy" in profile.metrics
        assert profile.metrics["accuracy"].mean == 0.9

    def test_get_metric(self):
        """Test getting metrics."""
        profile = ModelProfile(model_name="test-model")
        profile.add_metric("accuracy", [0.9], "%")

        metric = profile.get_metric("accuracy")
        assert metric is not None
        assert metric.mean == 0.9

        missing = profile.get_metric("nonexistent")
        assert missing is None


class TestModelComparator:
    """Tests for ModelComparator."""

    def test_add_profile(self):
        """Test adding profiles."""
        comparator = ModelComparator()

        profile = ModelProfile(model_name="model-a")
        profile.add_metric("accuracy", [0.9])

        comparator.add_profile(profile)
        assert "model-a" in comparator._profiles

    def test_compare_requires_two_models(self):
        """Test that compare requires at least 2 models."""
        comparator = ModelComparator()
        profile = ModelProfile(model_name="model-a")
        comparator.add_profile(profile)

        with pytest.raises(ValueError):
            comparator.compare()

    def test_compare_basic(self):
        """Test basic comparison."""
        comparator = ModelComparator()

        profile_a = ModelProfile(model_name="model-a")
        profile_a.add_metric("accuracy", [0.9])
        profile_a.add_metric("latency", [100.0])

        profile_b = ModelProfile(model_name="model-b")
        profile_b.add_metric("accuracy", [0.85])
        profile_b.add_metric("latency", [80.0])

        comparator.add_profile(profile_a)
        comparator.add_profile(profile_b)

        result = comparator.compare()

        assert len(result.models) == 2
        assert result.winner is not None
        assert "accuracy" in result.rankings
        assert "latency" in result.rankings

    def test_compare_rankings(self):
        """Test that rankings are correct."""
        comparator = ModelComparator()

        profile_a = ModelProfile(model_name="model-a")
        profile_a.add_metric("accuracy", [0.95])

        profile_b = ModelProfile(model_name="model-b")
        profile_b.add_metric("accuracy", [0.85])

        comparator.add_profile(profile_a)
        comparator.add_profile(profile_b)

        result = comparator.compare()

        # model-a should rank first for accuracy (higher is better)
        assert result.rankings["accuracy"][0] == "model-a"

    def test_compare_latency_lower_is_better(self):
        """Test that latency ranking is correct (lower is better)."""
        comparator = ModelComparator()

        profile_a = ModelProfile(model_name="model-a")
        profile_a.add_metric("latency", [200.0])

        profile_b = ModelProfile(model_name="model-b")
        profile_b.add_metric("latency", [100.0])

        comparator.add_profile(profile_a)
        comparator.add_profile(profile_b)

        result = comparator.compare()

        # model-b should rank first for latency (lower is better)
        assert result.rankings["latency"][0] == "model-b"

    def test_compare_metric_single(self):
        """Test comparing single metric."""
        comparator = ModelComparator()

        profile_a = ModelProfile(model_name="model-a")
        profile_a.add_metric("accuracy", [0.95])

        profile_b = ModelProfile(model_name="model-b")
        profile_b.add_metric("accuracy", [0.85])

        comparator.add_profile(profile_a)
        comparator.add_profile(profile_b)

        winner, ranked = comparator.compare_metric("accuracy")

        assert winner == "model-a"
        assert ranked[0][0] == "model-a"
        assert ranked[0][1] == 0.95

    def test_set_weight(self):
        """Test setting metric weights."""
        comparator = ModelComparator()
        comparator.set_weight("accuracy", 2.0)

        assert comparator._weights["accuracy"] == 2.0

    def test_generate_report(self):
        """Test report generation."""
        comparator = ModelComparator()

        profile_a = ModelProfile(model_name="model-a")
        profile_a.add_metric("accuracy", [0.9])

        profile_b = ModelProfile(model_name="model-b")
        profile_b.add_metric("accuracy", [0.85])

        comparator.add_profile(profile_a)
        comparator.add_profile(profile_b)

        report = comparator.generate_report()

        assert "Model Comparison Report" in report
        assert "model-a" in report
        assert "model-b" in report
        assert "accuracy" in report


class TestCostEstimate:
    """Tests for CostEstimate."""

    def test_calculate(self):
        """Test cost calculation."""
        estimate = CostEstimate.calculate(
            input_tokens=1000,
            output_tokens=500,
            input_price_per_1k=0.01,
            output_price_per_1k=0.02,
        )

        assert estimate.input_tokens == 1000
        assert estimate.output_tokens == 500
        assert estimate.input_cost == 0.01
        assert estimate.output_cost == 0.01
        assert estimate.total_cost == 0.02


class TestCostCalculator:
    """Tests for CostCalculator."""

    def test_set_pricing(self):
        """Test setting custom pricing."""
        calc = CostCalculator()
        calc.set_pricing("custom-model", 0.005, 0.01)

        assert "custom-model" in calc._pricing

    def test_estimate(self):
        """Test cost estimation."""
        calc = CostCalculator()
        calc.set_pricing("test-model", 0.01, 0.02)

        estimate = calc.estimate("test-model", 2000, 1000)

        assert estimate.total_cost == 0.04  # 0.02 + 0.02

    def test_estimate_unknown_model(self):
        """Test estimation for unknown model."""
        calc = CostCalculator()

        with pytest.raises(KeyError):
            calc.estimate("unknown-model", 1000, 500)

    def test_compare_costs(self):
        """Test comparing costs across models."""
        calc = CostCalculator()
        calc.set_pricing("model-a", 0.01, 0.02)
        calc.set_pricing("model-b", 0.005, 0.01)

        costs = calc.compare_costs(1000, 500, ["model-a", "model-b"])

        assert "model-a" in costs
        assert "model-b" in costs
        assert costs["model-b"].total_cost < costs["model-a"].total_cost

    def test_cheapest_model(self):
        """Test finding cheapest model."""
        calc = CostCalculator()
        calc.set_pricing("expensive", 0.1, 0.2)
        calc.set_pricing("cheap", 0.001, 0.002)

        name, estimate = calc.cheapest_model(1000, 500, ["expensive", "cheap"])

        assert name == "cheap"


class TestQualityMetrics:
    """Tests for QualityMetrics."""

    def test_compute_overall(self):
        """Test overall score computation."""
        metrics = QualityMetrics(
            coherence=0.9,
            relevance=0.85,
            accuracy=0.95,
            completeness=0.8,
            fluency=0.9,
        )

        overall = metrics.compute_overall()

        assert 0 <= overall <= 1
        assert metrics.overall == overall

    def test_compute_overall_custom_weights(self):
        """Test overall score with custom weights."""
        metrics = QualityMetrics(
            coherence=1.0,
            relevance=0.0,
            accuracy=0.0,
            completeness=0.0,
            fluency=0.0,
        )

        # Weight coherence heavily
        overall = metrics.compute_overall({"coherence": 1.0, "relevance": 0, "accuracy": 0, "completeness": 0, "fluency": 0})

        assert overall == 1.0


class TestPerformanceTracker:
    """Tests for PerformanceTracker."""

    def test_record_latency(self):
        """Test recording latency."""
        tracker = PerformanceTracker("test-model")
        tracker.record_latency(100.5)
        tracker.record_latency(150.0)

        assert len(tracker._latencies) == 2

    def test_record_success(self):
        """Test recording success/failure."""
        tracker = PerformanceTracker("test-model")
        tracker.record_success(True)
        tracker.record_success(False, "Error occurred")

        assert len(tracker._successes) == 2
        assert len(tracker._errors) == 1

    def test_record_tokens(self):
        """Test recording token usage."""
        tracker = PerformanceTracker("test-model")
        tracker.record_tokens(100, 50)

        assert len(tracker._token_counts) == 1

    def test_get_summary(self):
        """Test getting summary as ModelProfile."""
        tracker = PerformanceTracker("test-model")
        tracker.record_latency(100.0)
        tracker.record_latency(200.0)
        tracker.record_success(True)
        tracker.record_success(True)
        tracker.record_tokens(100, 50)

        summary = tracker.get_summary()

        assert summary.model_name == "test-model"
        assert "latency" in summary.metrics
        assert "success_rate" in summary.metrics

    def test_reset(self):
        """Test resetting tracker."""
        tracker = PerformanceTracker("test-model")
        tracker.record_latency(100.0)
        tracker.record_success(True)

        tracker.reset()

        assert len(tracker._latencies) == 0
        assert len(tracker._successes) == 0


class TestLatencyProfile:
    """Tests for LatencyProfile."""

    def test_basic_creation(self):
        """Test basic creation."""
        profile = LatencyProfile(
            total_ms=150.0,
            first_token_ms=50.0,
            tokens_per_second=20.0,
        )

        assert profile.total_ms == 150.0
        assert profile.first_token_ms == 50.0
        assert profile.tokens_per_second == 20.0


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_comparison_table(self):
        """Test creating comparison table."""
        profile_a = ModelProfile(model_name="model-a")
        profile_a.add_metric("accuracy", [0.9])

        profile_b = ModelProfile(model_name="model-b")
        profile_b.add_metric("accuracy", [0.85])

        table = create_comparison_table([profile_a, profile_b])

        assert "model-a" in table
        assert "model-b" in table
        assert "accuracy" in table
        assert "|" in table  # Markdown table separator

    def test_create_comparison_table_empty(self):
        """Test creating table with no profiles."""
        table = create_comparison_table([])
        assert table == ""

    def test_rank_models(self):
        """Test ranking models."""
        profile_a = ModelProfile(model_name="model-a")
        profile_a.add_metric("accuracy", [0.95])

        profile_b = ModelProfile(model_name="model-b")
        profile_b.add_metric("accuracy", [0.85])

        profile_c = ModelProfile(model_name="model-c")
        profile_c.add_metric("accuracy", [0.90])

        ranked = rank_models([profile_a, profile_b, profile_c], "accuracy")

        assert ranked[0][0] == "model-a"  # Highest
        assert ranked[1][0] == "model-c"  # Middle
        assert ranked[2][0] == "model-b"  # Lowest

    def test_rank_models_lower_is_better(self):
        """Test ranking with lower is better."""
        profile_a = ModelProfile(model_name="model-a")
        profile_a.add_metric("latency", [200.0])

        profile_b = ModelProfile(model_name="model-b")
        profile_b.add_metric("latency", [100.0])

        ranked = rank_models(
            [profile_a, profile_b],
            "latency",
            higher_is_better=False,
        )

        assert ranked[0][0] == "model-b"  # Lower is better


class TestComparisonResult:
    """Tests for ComparisonResult."""

    def test_basic_creation(self):
        """Test basic creation."""
        result = ComparisonResult(
            models=["model-a", "model-b"],
            winner="model-a",
        )

        assert len(result.models) == 2
        assert result.winner == "model-a"


class TestComparisonMetric:
    """Tests for ComparisonMetric enum."""

    def test_all_metrics_exist(self):
        """Test all standard metrics exist."""
        assert ComparisonMetric.ACCURACY.value == "accuracy"
        assert ComparisonMetric.LATENCY.value == "latency"
        assert ComparisonMetric.THROUGHPUT.value == "throughput"
        assert ComparisonMetric.COST.value == "cost"
        assert ComparisonMetric.ERROR_RATE.value == "error_rate"
