"""Tests for the cost tracking module."""

from datetime import datetime, timedelta

from insideLLMs.cost_tracking import (
    AlertLevel,
    Budget,
    BudgetAlert,
    BudgetManager,
    CostCalculator,
    CostCategory,
    CostForecast,
    CostForecaster,
    CostReporter,
    CostSummary,
    # Dataclasses
    ModelPricing,
    # Classes
    PricingRegistry,
    # Enums
    PricingTier,
    TimeGranularity,
    UsageRecord,
    UsageTracker,
    # Convenience functions
    calculate_cost,
    compare_model_costs,
    create_budget_manager,
    create_usage_tracker,
    estimate_request_cost,
    get_cheapest_model,
    quick_cost_estimate,
)

# ============================================================================
# Enum Tests
# ============================================================================


class TestCostEnums:
    """Test cost-related enums."""

    def test_pricing_tier_values(self):
        assert PricingTier.FREE.value == "free"
        assert PricingTier.PAY_AS_YOU_GO.value == "pay_as_you_go"
        assert PricingTier.STANDARD.value == "standard"
        assert PricingTier.ENTERPRISE.value == "enterprise"
        assert PricingTier.CUSTOM.value == "custom"

    def test_cost_category_values(self):
        assert CostCategory.INPUT_TOKENS.value == "input_tokens"
        assert CostCategory.OUTPUT_TOKENS.value == "output_tokens"
        assert CostCategory.EMBEDDING.value == "embedding"
        assert CostCategory.FINE_TUNING.value == "fine_tuning"

    def test_alert_level_values(self):
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"

    def test_time_granularity_values(self):
        assert TimeGranularity.MINUTE.value == "minute"
        assert TimeGranularity.HOUR.value == "hour"
        assert TimeGranularity.DAY.value == "day"
        assert TimeGranularity.WEEK.value == "week"
        assert TimeGranularity.MONTH.value == "month"


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestModelPricing:
    """Test ModelPricing dataclass."""

    def test_creation(self):
        pricing = ModelPricing(
            model_name="gpt-4",
            input_cost_per_1k=0.03,
            output_cost_per_1k=0.06,
        )
        assert pricing.model_name == "gpt-4"
        assert pricing.input_cost_per_1k == 0.03
        assert pricing.output_cost_per_1k == 0.06
        assert pricing.currency == "USD"

    def test_to_dict(self):
        pricing = ModelPricing(
            model_name="test-model",
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.02,
            context_window=8192,
        )
        d = pricing.to_dict()
        assert d["model_name"] == "test-model"
        assert d["context_window"] == 8192

    def test_from_dict(self):
        data = {
            "model_name": "test",
            "input_cost_per_1k": 0.005,
            "output_cost_per_1k": 0.015,
            "tier": "enterprise",
        }
        pricing = ModelPricing.from_dict(data)
        assert pricing.model_name == "test"
        assert pricing.tier == PricingTier.ENTERPRISE


class TestUsageRecord:
    """Test UsageRecord dataclass."""

    def test_creation(self):
        record = UsageRecord(
            timestamp=datetime.now(),
            model_name="gpt-4",
            input_tokens=100,
            output_tokens=50,
            cost=0.0045,
        )
        assert record.model_name == "gpt-4"
        assert record.input_tokens == 100

    def test_total_tokens(self):
        record = UsageRecord(
            timestamp=datetime.now(),
            model_name="test",
            input_tokens=100,
            output_tokens=50,
            cost=0.01,
        )
        assert record.total_tokens == 150

    def test_to_dict(self):
        record = UsageRecord(
            timestamp=datetime.now(),
            model_name="test",
            input_tokens=100,
            output_tokens=50,
            cost=0.01,
            request_id="req-123",
        )
        d = record.to_dict()
        assert d["request_id"] == "req-123"


class TestBudget:
    """Test Budget dataclass."""

    def test_creation(self):
        budget = Budget(
            name="daily_budget",
            limit=10.0,
            period=TimeGranularity.DAY,
        )
        assert budget.name == "daily_budget"
        assert budget.limit == 10.0
        assert budget.alert_threshold == 0.8

    def test_to_dict(self):
        budget = Budget(
            name="test",
            limit=5.0,
            period=TimeGranularity.WEEK,
            hard_limit=True,
        )
        d = budget.to_dict()
        assert d["period"] == "week"
        assert d["hard_limit"] is True


class TestBudgetAlert:
    """Test BudgetAlert dataclass."""

    def test_creation(self):
        alert = BudgetAlert(
            level=AlertLevel.WARNING,
            budget_name="test",
            message="Budget warning",
            current_spend=8.0,
            budget_limit=10.0,
            percentage_used=0.8,
        )
        assert alert.level == AlertLevel.WARNING
        assert alert.percentage_used == 0.8

    def test_to_dict(self):
        alert = BudgetAlert(
            level=AlertLevel.CRITICAL,
            budget_name="test",
            message="Over budget!",
            current_spend=12.0,
            budget_limit=10.0,
            percentage_used=1.2,
        )
        d = alert.to_dict()
        assert d["level"] == "critical"


class TestCostSummary:
    """Test CostSummary dataclass."""

    def test_creation(self):
        summary = CostSummary(
            period_start=datetime.now() - timedelta(days=1),
            period_end=datetime.now(),
            total_cost=25.0,
            total_input_tokens=10000,
            total_output_tokens=5000,
            total_requests=50,
            by_model={"gpt-4": 20.0, "gpt-3.5": 5.0},
            by_category={"input_tokens": 15.0, "output_tokens": 10.0},
            average_cost_per_request=0.5,
        )
        assert summary.total_cost == 25.0
        assert summary.total_requests == 50


class TestCostForecast:
    """Test CostForecast dataclass."""

    def test_creation(self):
        forecast = CostForecast(
            forecast_period=TimeGranularity.MONTH,
            projected_cost=100.0,
            confidence_low=80.0,
            confidence_high=120.0,
            based_on_days=30,
            trend="increasing",
            projected_monthly=100.0,
        )
        assert forecast.projected_cost == 100.0
        assert forecast.trend == "increasing"


# ============================================================================
# PricingRegistry Tests
# ============================================================================


class TestPricingRegistry:
    """Test PricingRegistry class."""

    def test_default_models_loaded(self):
        registry = PricingRegistry()
        models = registry.list_models()
        assert len(models) > 0
        assert "gpt-4" in models
        assert "claude-3-opus" in models

    def test_register_custom_model(self):
        registry = PricingRegistry()
        pricing = ModelPricing("custom-model", 0.001, 0.002)
        registry.register(pricing)

        assert "custom-model" in registry.list_models()
        retrieved = registry.get("custom-model")
        assert retrieved.input_cost_per_1k == 0.001

    def test_get_nonexistent(self):
        registry = PricingRegistry()
        assert registry.get("nonexistent") is None

    def test_get_or_default(self):
        registry = PricingRegistry()
        pricing = registry.get_or_default("nonexistent")
        # Should return a default expensive model
        assert pricing.model_name == "nonexistent"
        assert pricing.input_cost_per_1k > 0

    def test_update_pricing(self):
        registry = PricingRegistry()
        registry.update_pricing("gpt-4", 0.05, 0.10)

        pricing = registry.get("gpt-4")
        assert pricing.input_cost_per_1k == 0.05
        assert pricing.output_cost_per_1k == 0.10


# ============================================================================
# CostCalculator Tests
# ============================================================================


class TestCostCalculator:
    """Test CostCalculator class."""

    def test_calculate_cost(self):
        calculator = CostCalculator()
        # gpt-4: $0.03/1k input, $0.06/1k output
        cost = calculator.calculate_cost("gpt-4", 1000, 500)
        expected = 0.03 + 0.03  # 1k input + 0.5k output
        assert abs(cost - expected) < 0.0001

    def test_calculate_batch_cost(self):
        calculator = CostCalculator()
        requests = [(1000, 500), (2000, 1000)]
        cost = calculator.calculate_batch_cost("gpt-4", requests)
        # First: 0.03 + 0.03 = 0.06
        # Second: 0.06 + 0.06 = 0.12
        expected = 0.18
        assert abs(cost - expected) < 0.0001

    def test_estimate_cost(self):
        calculator = CostCalculator()
        # 400 chars ~ 100 tokens
        cost = calculator.estimate_cost("gpt-4", "x" * 400, 100)
        assert cost > 0

    def test_compare_models(self):
        calculator = CostCalculator()
        comparison = calculator.compare_models(1000, 500, ["gpt-4", "gpt-3.5-turbo"])

        assert "gpt-4" in comparison
        assert "gpt-3.5-turbo" in comparison
        # GPT-4 should be more expensive
        assert comparison["gpt-4"] > comparison["gpt-3.5-turbo"]

    def test_get_cheapest_model(self):
        calculator = CostCalculator()
        cheapest, cost = calculator.get_cheapest_model(1000, 500)

        assert cheapest is not None
        assert cost > 0


# ============================================================================
# UsageTracker Tests
# ============================================================================


class TestUsageTracker:
    """Test UsageTracker class."""

    def test_record_usage(self):
        tracker = UsageTracker()
        record = tracker.record_usage("gpt-4", 1000, 500)

        assert record.model_name == "gpt-4"
        assert record.cost > 0
        assert len(tracker.records) == 1

    def test_get_records_in_range(self):
        tracker = UsageTracker()

        # Add records at different times
        now = datetime.now()
        tracker.record_usage("gpt-4", 100, 50, timestamp=now - timedelta(hours=2))
        tracker.record_usage("gpt-4", 100, 50, timestamp=now - timedelta(hours=1))
        tracker.record_usage("gpt-4", 100, 50, timestamp=now)

        # Get records from last 90 minutes
        records = tracker.get_records_in_range(now - timedelta(minutes=90))
        assert len(records) == 2

    def test_get_total_cost(self):
        tracker = UsageTracker()
        tracker.record_usage("gpt-4", 1000, 500)
        tracker.record_usage("gpt-4", 1000, 500)

        total = tracker.get_total_cost()
        # Two identical requests
        assert total == tracker.records[0].cost * 2

    def test_get_total_tokens(self):
        tracker = UsageTracker()
        tracker.record_usage("gpt-4", 1000, 500)
        tracker.record_usage("gpt-4", 2000, 1000)

        input_total, output_total = tracker.get_total_tokens()
        assert input_total == 3000
        assert output_total == 1500

    def test_get_cost_by_model(self):
        tracker = UsageTracker()
        tracker.record_usage("gpt-4", 1000, 500)
        tracker.record_usage("gpt-3.5-turbo", 1000, 500)
        tracker.record_usage("gpt-4", 1000, 500)

        by_model = tracker.get_cost_by_model()
        assert "gpt-4" in by_model
        assert "gpt-3.5-turbo" in by_model

    def test_get_summary(self):
        tracker = UsageTracker()
        now = datetime.now()

        tracker.record_usage("gpt-4", 1000, 500, timestamp=now)
        tracker.record_usage("gpt-3.5-turbo", 2000, 1000, timestamp=now)

        summary = tracker.get_summary(now - timedelta(hours=1))
        assert summary.total_requests == 2
        assert summary.total_input_tokens == 3000
        assert len(summary.by_model) == 2

    def test_clear_old_records(self):
        tracker = UsageTracker()
        now = datetime.now()

        tracker.record_usage("gpt-4", 100, 50, timestamp=now - timedelta(days=10))
        tracker.record_usage("gpt-4", 100, 50, timestamp=now - timedelta(days=5))
        tracker.record_usage("gpt-4", 100, 50, timestamp=now)

        removed = tracker.clear_old_records(now - timedelta(days=7))
        assert removed == 1
        assert len(tracker.records) == 2


# ============================================================================
# BudgetManager Tests
# ============================================================================


class TestBudgetManager:
    """Test BudgetManager class."""

    def test_create_budget(self):
        manager = BudgetManager()
        budget = manager.create_budget("daily", 10.0, TimeGranularity.DAY)

        assert budget.name == "daily"
        assert budget.limit == 10.0
        assert "daily" in manager.budgets

    def test_get_budget(self):
        manager = BudgetManager()
        manager.create_budget("test", 5.0, TimeGranularity.HOUR)

        budget = manager.get_budget("test")
        assert budget is not None
        assert budget.limit == 5.0

    def test_delete_budget(self):
        manager = BudgetManager()
        manager.create_budget("test", 5.0, TimeGranularity.DAY)

        assert manager.delete_budget("test") is True
        assert manager.get_budget("test") is None
        assert manager.delete_budget("test") is False

    def test_check_budget_under(self):
        tracker = UsageTracker()
        manager = BudgetManager(tracker)
        manager.create_budget("test", 1.0, TimeGranularity.DAY)

        # Small usage shouldn't trigger alert
        tracker.record_usage("gpt-3.5-turbo", 100, 50)

        alert = manager.check_budget("test")
        assert alert is None

    def test_check_budget_warning(self):
        tracker = UsageTracker()
        manager = BudgetManager(tracker)
        manager.create_budget(
            "test",
            0.001,
            TimeGranularity.DAY,
            alert_threshold=0.5,  # Lower threshold for testing
        )

        # This should exceed 50% of $0.001
        tracker.record_usage("gpt-4", 100, 50)

        alert = manager.check_budget("test")
        assert alert is not None
        assert alert.level in [AlertLevel.WARNING, AlertLevel.CRITICAL]

    def test_can_make_request_soft_limit(self):
        tracker = UsageTracker()
        manager = BudgetManager(tracker)
        manager.create_budget("test", 0.01, TimeGranularity.DAY, hard_limit=False)

        tracker.record_usage("gpt-4", 1000, 500)

        # Soft limit should allow request
        can_request, reason = manager.can_make_request("test", 1.0)
        assert can_request is True

    def test_can_make_request_hard_limit(self):
        tracker = UsageTracker()
        manager = BudgetManager(tracker)
        manager.create_budget("test", 0.01, TimeGranularity.DAY, hard_limit=True)

        tracker.record_usage("gpt-4", 1000, 500)

        # Hard limit should block expensive request
        can_request, reason = manager.can_make_request("test", 1.0)
        assert can_request is False
        assert "exceed" in reason.lower()

    def test_alert_callback(self):
        tracker = UsageTracker()
        manager = BudgetManager(tracker)
        manager.create_budget("test", 0.0001, TimeGranularity.DAY)

        alerts_received = []
        manager.add_alert_callback(lambda a: alerts_received.append(a))

        tracker.record_usage("gpt-4", 1000, 500)
        manager.check_budget("test")

        assert len(alerts_received) > 0

    def test_get_budget_status(self):
        tracker = UsageTracker()
        manager = BudgetManager(tracker)
        manager.create_budget("test", 10.0, TimeGranularity.DAY)

        status = manager.get_budget_status("test")
        assert status["name"] == "test"
        assert status["limit"] == 10.0
        assert "percentage_used" in status


# ============================================================================
# CostForecaster Tests
# ============================================================================


class TestCostForecaster:
    """Test CostForecaster class."""

    def test_forecast_empty(self):
        tracker = UsageTracker()
        forecaster = CostForecaster(tracker)

        forecast = forecaster.forecast()
        assert forecast.projected_cost == 0.0
        assert forecast.trend == "stable"

    def test_forecast_with_data(self):
        tracker = UsageTracker()
        now = datetime.now()

        # Add usage data over past week
        for i in range(7):
            timestamp = now - timedelta(days=i)
            tracker.record_usage("gpt-4", 1000, 500, timestamp=timestamp)

        forecaster = CostForecaster(tracker)
        forecast = forecaster.forecast(TimeGranularity.WEEK, lookback_days=7)

        assert forecast.projected_cost > 0
        assert forecast.based_on_days > 0

    def test_estimate_monthly_at_rate(self):
        tracker = UsageTracker()
        forecaster = CostForecaster(tracker)

        monthly_estimate = forecaster.estimate_monthly_at_rate(
            daily_requests=100,
            avg_input_tokens=500,
            avg_output_tokens=200,
            model_name="gpt-4",
        )

        assert monthly_estimate > 0


# ============================================================================
# CostReporter Tests
# ============================================================================


class TestCostReporter:
    """Test CostReporter class."""

    def test_generate_daily_report(self):
        tracker = UsageTracker()
        tracker.record_usage("gpt-4", 1000, 500)
        tracker.record_usage("gpt-3.5-turbo", 500, 200)

        reporter = CostReporter(tracker)
        report = reporter.generate_daily_report()

        assert "date" in report
        assert "summary" in report
        assert "top_models" in report

    def test_generate_weekly_report(self):
        tracker = UsageTracker()
        now = datetime.now()

        for i in range(7):
            tracker.record_usage("gpt-4", 1000, 500, timestamp=now - timedelta(days=i))

        reporter = CostReporter(tracker)
        report = reporter.generate_weekly_report()

        assert "week_start" in report
        assert "daily_breakdown" in report
        assert len(report["daily_breakdown"]) == 7

    def test_generate_monthly_report(self):
        tracker = UsageTracker()
        tracker.record_usage("gpt-4", 1000, 500)

        reporter = CostReporter(tracker)
        report = reporter.generate_monthly_report()

        assert "year" in report
        assert "month" in report
        assert "forecast" in report

    def test_format_report_text(self):
        tracker = UsageTracker()
        tracker.record_usage("gpt-4", 1000, 500)

        reporter = CostReporter(tracker)
        report = reporter.generate_daily_report()
        text = reporter.format_report_text(report)

        assert "Total Cost" in text
        assert "Total Requests" in text


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_calculate_cost(self):
        cost = calculate_cost("gpt-4", 1000, 500)
        assert cost > 0

    def test_estimate_request_cost(self):
        cost = estimate_request_cost("gpt-4", "Hello, how are you?", 100)
        assert cost > 0

    def test_compare_model_costs(self):
        comparison = compare_model_costs(1000, 500)
        assert len(comparison) > 0
        assert all(c > 0 for c in comparison.values())

    def test_get_cheapest_model(self):
        model, cost = get_cheapest_model(1000, 500)
        assert model is not None
        assert cost > 0

    def test_create_usage_tracker(self):
        tracker = create_usage_tracker()
        assert isinstance(tracker, UsageTracker)

    def test_create_budget_manager(self):
        manager = create_budget_manager()
        assert isinstance(manager, BudgetManager)

    def test_quick_cost_estimate(self):
        estimate = quick_cost_estimate("gpt-4", 100)

        assert estimate["model"] == "gpt-4"
        assert estimate["num_requests"] == 100
        assert estimate["total_cost"] > 0
        assert estimate["monthly_estimate"] == estimate["total_cost"] * 30


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for cost tracking workflows."""

    def test_full_tracking_workflow(self):
        # Create tracker and manager
        tracker = UsageTracker()
        manager = BudgetManager(tracker)

        # Set up budget
        manager.create_budget("daily", 1.0, TimeGranularity.DAY)

        # Record some usage
        for _ in range(5):
            tracker.record_usage("gpt-4", 1000, 500)

        # Check budget
        manager.check_budget("daily")

        # Get status
        status = manager.get_budget_status("daily")
        assert status["current_spend"] > 0

        # Generate report
        reporter = CostReporter(tracker)
        report = reporter.generate_daily_report()
        assert report["summary"]["total_requests"] == 5

    def test_cost_optimization_workflow(self):
        # Compare models to find cheapest
        costs = compare_model_costs(5000, 2000)

        # Get cheapest
        cheapest, cost = get_cheapest_model(5000, 2000)

        # Verify it's actually the cheapest
        assert all(cost <= c for c in costs.values())

    def test_forecasting_workflow(self):
        tracker = UsageTracker()
        now = datetime.now()

        # Simulate a week of increasing usage
        for i in range(7):
            num_requests = 10 + i * 2  # Increasing usage
            for _ in range(num_requests):
                tracker.record_usage("gpt-4", 500, 200, timestamp=now - timedelta(days=6 - i))

        forecaster = CostForecaster(tracker)
        forecast = forecaster.forecast(TimeGranularity.WEEK, lookback_days=7)

        assert forecast.projected_cost > 0
        assert forecast.based_on_days == 7


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_tokens(self):
        cost = calculate_cost("gpt-4", 0, 0)
        assert cost == 0.0

    def test_very_large_usage(self):
        cost = calculate_cost("gpt-4", 1000000, 500000)
        assert cost > 0

    def test_unknown_model(self):
        cost = calculate_cost("unknown-model-xyz", 1000, 500)
        # Should use default pricing
        assert cost > 0

    def test_empty_tracker_summary(self):
        tracker = UsageTracker()
        now = datetime.now()

        summary = tracker.get_summary(now - timedelta(days=1))
        assert summary.total_cost == 0
        assert summary.total_requests == 0
        assert summary.average_cost_per_request == 0

    def test_budget_nonexistent(self):
        manager = BudgetManager()
        status = manager.get_budget_status("nonexistent")
        assert status is None

        alert = manager.check_budget("nonexistent")
        assert alert is None

    def test_forecast_insufficient_data(self):
        tracker = UsageTracker()
        tracker.record_usage("gpt-4", 100, 50)

        forecaster = CostForecaster(tracker)
        forecast = forecaster.forecast(lookback_days=30)

        # Should still return a forecast
        assert forecast.based_on_days == 1
