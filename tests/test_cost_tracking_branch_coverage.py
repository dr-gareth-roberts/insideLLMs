"""Additional branch coverage for cost_tracking."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import insideLLMs.cost_tracking as cost_tracking
from insideLLMs.cost_tracking import (
    BudgetManager,
    CostForecaster,
    CostReporter,
    TimeGranularity,
    UsageRecord,
    UsageTracker,
)


def test_budget_period_start_branches(monkeypatch: pytest.MonkeyPatch):
    fixed_now = datetime(2026, 2, 6, 15, 47, 33)  # Friday

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return fixed_now

    monkeypatch.setattr(cost_tracking, "datetime", FixedDateTime)
    manager = BudgetManager()

    minute_budget = manager.create_budget("m", 1.0, TimeGranularity.MINUTE)
    hour_budget = manager.create_budget("h", 1.0, TimeGranularity.HOUR)
    day_budget = manager.create_budget("d", 1.0, TimeGranularity.DAY)
    week_budget = manager.create_budget("w", 1.0, TimeGranularity.WEEK)
    month_budget = manager.create_budget("mo", 1.0, TimeGranularity.MONTH)

    assert manager._get_period_start(minute_budget) == fixed_now.replace(second=0, microsecond=0)
    assert manager._get_period_start(hour_budget) == fixed_now.replace(
        minute=0, second=0, microsecond=0
    )
    assert manager._get_period_start(day_budget) == fixed_now.replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    week_start = manager._get_period_start(week_budget)
    assert week_start.weekday() == 0
    assert week_start.hour == 0 and week_start.minute == 0 and week_start.second == 0

    month_start = manager._get_period_start(month_budget)
    assert month_start.day == 1
    assert month_start.hour == 0 and month_start.minute == 0

    # Force fallback branch.
    month_budget.period = "unknown"  # type: ignore[assignment]
    fallback = manager._get_period_start(month_budget)
    assert fallback.hour == 0 and fallback.minute == 0 and fallback.second == 0


def test_check_all_budgets_and_can_make_request_missing_budget():
    tracker = UsageTracker()
    manager = BudgetManager(tracker)
    manager.create_budget(
        "warn",
        0.001,
        TimeGranularity.DAY,
        alert_threshold=0.1,
        critical_threshold=10.0,
    )
    manager.create_budget("quiet", 999.0, TimeGranularity.DAY)

    tracker.record_usage("gpt-4", 1000, 500)
    alerts = manager.check_all_budgets()
    assert any(a.budget_name == "warn" for a in alerts)

    allowed, reason = manager.can_make_request("missing-budget", estimated_cost=123.0)
    assert allowed is True
    assert reason is None


def test_usage_tracker_filtered_totals_branches():
    tracker = UsageTracker()
    now = datetime.now()
    tracker.record_usage("gpt-4", 100, 20, timestamp=now - timedelta(days=2))
    tracker.record_usage("gpt-4", 50, 10, timestamp=now - timedelta(hours=2))
    tracker.record_usage("gpt-3.5-turbo", 40, 5, timestamp=now - timedelta(hours=1))

    input_tokens, output_tokens = tracker.get_total_tokens(start=now - timedelta(hours=3))
    assert input_tokens == 90
    assert output_tokens == 15

    by_model = tracker.get_cost_by_model(start=now - timedelta(hours=3))
    assert set(by_model) == {"gpt-4", "gpt-3.5-turbo"}


def test_forecaster_truthy_empty_iterable_and_period_paths():
    class TruthyEmpty:
        def __bool__(self) -> bool:
            return True

        def __iter__(self):
            return iter(())

    class FakeTracker:
        def get_records_in_range(self, _start, _end):
            return TruthyEmpty()

    forecaster = CostForecaster(FakeTracker())  # type: ignore[arg-type]
    emptyish = forecaster.forecast()
    assert emptyish.projected_cost == 0.0
    assert emptyish.based_on_days == 0

    tracker = UsageTracker()
    now = datetime.now()
    # Decreasing day-over-day costs should trigger "decreasing" trend.
    for i, cost in enumerate([10.0, 9.0, 8.0, 7.0, 2.0, 2.0, 1.0, 1.0]):
        tracker.records.append(
            UsageRecord(
                timestamp=now - timedelta(days=7 - i),
                model_name="synthetic",
                input_tokens=0,
                output_tokens=0,
                cost=cost,
            )
        )

    trend_day = CostForecaster(tracker).forecast(period=TimeGranularity.DAY, lookback_days=30)
    assert trend_day.trend == "decreasing"
    assert trend_day.projected_cost > 0

    one_day_tracker = UsageTracker()
    one_day_tracker.records.append(
        UsageRecord(
            timestamp=now,
            model_name="synthetic",
            input_tokens=0,
            output_tokens=0,
            cost=3.0,
        )
    )
    one_day = CostForecaster(one_day_tracker).forecast(period="other", lookback_days=1)  # type: ignore[arg-type]
    assert one_day.projected_cost > 0
    assert one_day.confidence_high > one_day.projected_cost


def test_reporter_format_text_weekly_monthly_and_forecast_branches():
    reporter = CostReporter(UsageTracker())
    weekly_report = {
        "week_start": "2026-02-02",
        "week_end": "2026-02-08",
        "summary": {"total_cost": 1.2, "total_requests": 3, "total_input_tokens": 10, "total_output_tokens": 5},
        "top_models": [{"name": "gpt-4", "cost": 1.2}],
        "forecast": {
            "forecast_period": "week",
            "projected_cost": 2.0,
            "confidence_low": 1.5,
            "confidence_high": 2.5,
            "trend": "stable",
        },
    }
    weekly_text = reporter.format_report_text(weekly_report)
    assert "Weekly Cost Report" in weekly_text
    assert "Forecast (week)" in weekly_text

    monthly_report = {
        "year": 2026,
        "month": 2,
        "summary": {"total_cost": 3.0, "total_requests": 8, "total_input_tokens": 100, "total_output_tokens": 50},
        "top_models": [],
    }
    monthly_text = reporter.format_report_text(monthly_report)
    assert "Monthly Cost Report - 2026/02" in monthly_text
