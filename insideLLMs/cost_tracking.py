"""Cost estimation and budget tracking for LLM API usage.

This module provides tools for tracking, estimating, and managing
costs associated with LLM API calls, including:
- Token-based cost calculation
- Budget management and alerts
- Cost forecasting
- Usage analytics and reporting
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional


class PricingTier(Enum):
    """Pricing tiers for different API access levels."""

    FREE = "free"
    PAY_AS_YOU_GO = "pay_as_you_go"
    STANDARD = "standard"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class CostCategory(Enum):
    """Categories of costs."""

    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    EMBEDDING = "embedding"
    FINE_TUNING = "fine_tuning"
    STORAGE = "storage"
    API_CALLS = "api_calls"
    OTHER = "other"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class TimeGranularity(Enum):
    """Time granularity for reporting."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    model_name: str
    input_cost_per_1k: float  # Cost per 1000 input tokens
    output_cost_per_1k: float  # Cost per 1000 output tokens
    currency: str = "USD"
    tier: PricingTier = PricingTier.PAY_AS_YOU_GO
    embedding_cost_per_1k: Optional[float] = None
    context_window: Optional[int] = None
    effective_date: Optional[datetime] = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_cost_per_1k": self.input_cost_per_1k,
            "output_cost_per_1k": self.output_cost_per_1k,
            "currency": self.currency,
            "tier": self.tier.value,
            "embedding_cost_per_1k": self.embedding_cost_per_1k,
            "context_window": self.context_window,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelPricing":
        return cls(
            model_name=data["model_name"],
            input_cost_per_1k=data["input_cost_per_1k"],
            output_cost_per_1k=data["output_cost_per_1k"],
            currency=data.get("currency", "USD"),
            tier=PricingTier(data.get("tier", "pay_as_you_go")),
            embedding_cost_per_1k=data.get("embedding_cost_per_1k"),
            context_window=data.get("context_window"),
            effective_date=datetime.fromisoformat(data["effective_date"])
            if data.get("effective_date")
            else None,
            notes=data.get("notes", ""),
        )


@dataclass
class UsageRecord:
    """Record of a single API usage event."""

    timestamp: datetime
    model_name: str
    input_tokens: int
    output_tokens: int
    cost: float
    request_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


@dataclass
class Budget:
    """Budget configuration."""

    name: str
    limit: float
    period: TimeGranularity
    currency: str = "USD"
    alert_threshold: float = 0.8  # Alert at 80% of budget
    critical_threshold: float = 0.95  # Critical at 95%
    hard_limit: bool = False  # Whether to block requests when exceeded
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "limit": self.limit,
            "period": self.period.value,
            "currency": self.currency,
            "alert_threshold": self.alert_threshold,
            "critical_threshold": self.critical_threshold,
            "hard_limit": self.hard_limit,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class BudgetAlert:
    """An alert triggered by budget conditions."""

    level: AlertLevel
    budget_name: str
    message: str
    current_spend: float
    budget_limit: float
    percentage_used: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "budget_name": self.budget_name,
            "message": self.message,
            "current_spend": self.current_spend,
            "budget_limit": self.budget_limit,
            "percentage_used": self.percentage_used,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CostSummary:
    """Summary of costs over a period."""

    period_start: datetime
    period_end: datetime
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    total_requests: int
    by_model: dict[str, float]
    by_category: dict[str, float]
    average_cost_per_request: float
    currency: str = "USD"

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_requests": self.total_requests,
            "by_model": self.by_model,
            "by_category": self.by_category,
            "average_cost_per_request": self.average_cost_per_request,
            "currency": self.currency,
        }


@dataclass
class CostForecast:
    """Forecast of future costs."""

    forecast_period: TimeGranularity
    projected_cost: float
    confidence_low: float
    confidence_high: float
    based_on_days: int
    trend: str  # "increasing", "decreasing", "stable"
    projected_monthly: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast_period": self.forecast_period.value,
            "projected_cost": self.projected_cost,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "based_on_days": self.based_on_days,
            "trend": self.trend,
            "projected_monthly": self.projected_monthly,
            "metadata": self.metadata,
        }


class PricingRegistry:
    """Registry of model pricing information."""

    def __init__(self):
        self.models: dict[str, ModelPricing] = {}
        self._initialize_default_pricing()

    def _initialize_default_pricing(self) -> None:
        """Initialize with common model pricing (as of knowledge cutoff)."""
        # OpenAI models
        default_models = [
            ModelPricing("gpt-4", 0.03, 0.06, context_window=8192),
            ModelPricing("gpt-4-turbo", 0.01, 0.03, context_window=128000),
            ModelPricing("gpt-4o", 0.005, 0.015, context_window=128000),
            ModelPricing("gpt-4o-mini", 0.00015, 0.0006, context_window=128000),
            ModelPricing("gpt-3.5-turbo", 0.0005, 0.0015, context_window=16385),
            # Anthropic models
            ModelPricing("claude-3-opus", 0.015, 0.075, context_window=200000),
            ModelPricing("claude-3-sonnet", 0.003, 0.015, context_window=200000),
            ModelPricing("claude-3-haiku", 0.00025, 0.00125, context_window=200000),
            ModelPricing("claude-3.5-sonnet", 0.003, 0.015, context_window=200000),
            # Google models
            ModelPricing("gemini-1.5-pro", 0.00125, 0.005, context_window=1000000),
            ModelPricing("gemini-1.5-flash", 0.000075, 0.0003, context_window=1000000),
        ]

        for model in default_models:
            self.register(model)

    def register(self, pricing: ModelPricing) -> None:
        """Register model pricing."""
        self.models[pricing.model_name] = pricing

    def get(self, model_name: str) -> Optional[ModelPricing]:
        """Get pricing for a model."""
        return self.models.get(model_name)

    def get_or_default(
        self, model_name: str, default_pricing: Optional[ModelPricing] = None
    ) -> ModelPricing:
        """Get pricing or return a default."""
        pricing = self.get(model_name)
        if pricing:
            return pricing

        if default_pricing:
            return default_pricing

        # Return a generic expensive default to avoid underestimating
        return ModelPricing(model_name, 0.01, 0.03)

    def list_models(self) -> list[str]:
        """List all registered models."""
        return list(self.models.keys())

    def update_pricing(self, model_name: str, input_cost: float, output_cost: float) -> None:
        """Update pricing for an existing model."""
        if model_name in self.models:
            self.models[model_name].input_cost_per_1k = input_cost
            self.models[model_name].output_cost_per_1k = output_cost
            self.models[model_name].effective_date = datetime.now()


class CostCalculator:
    """Calculate costs for API usage."""

    def __init__(self, pricing_registry: Optional[PricingRegistry] = None):
        self.registry = pricing_registry or PricingRegistry()

    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for a single request."""
        pricing = self.registry.get_or_default(model_name)

        input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k

        return input_cost + output_cost

    def calculate_batch_cost(
        self,
        model_name: str,
        requests: list[tuple[int, int]],  # List of (input_tokens, output_tokens)
    ) -> float:
        """Calculate cost for a batch of requests."""
        return sum(self.calculate_cost(model_name, inp, out) for inp, out in requests)

    def estimate_cost(
        self,
        model_name: str,
        prompt: str,
        estimated_output_tokens: int = 500,
        chars_per_token: float = 4.0,
    ) -> float:
        """Estimate cost for a prompt before making a request."""
        estimated_input_tokens = int(len(prompt) / chars_per_token)
        return self.calculate_cost(model_name, estimated_input_tokens, estimated_output_tokens)

    def compare_models(
        self,
        input_tokens: int,
        output_tokens: int,
        model_names: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """Compare costs across models for the same usage."""
        if model_names is None:
            model_names = self.registry.list_models()

        return {
            name: self.calculate_cost(name, input_tokens, output_tokens) for name in model_names
        }

    def get_cheapest_model(
        self,
        input_tokens: int,
        output_tokens: int,
        model_names: Optional[list[str]] = None,
    ) -> tuple[str, float]:
        """Find the cheapest model for given usage."""
        costs = self.compare_models(input_tokens, output_tokens, model_names)
        cheapest = min(costs.items(), key=lambda x: x[1])
        return cheapest


class UsageTracker:
    """Track API usage over time."""

    def __init__(self, calculator: Optional[CostCalculator] = None):
        self.calculator = calculator or CostCalculator()
        self.records: list[UsageRecord] = []

    def record_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        request_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> UsageRecord:
        """Record a usage event."""
        cost = self.calculator.calculate_cost(model_name, input_tokens, output_tokens)

        record = UsageRecord(
            timestamp=timestamp or datetime.now(),
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            request_id=request_id,
            metadata=metadata or {},
        )

        self.records.append(record)
        return record

    def get_records_in_range(
        self,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> list[UsageRecord]:
        """Get records within a time range."""
        end = end or datetime.now()
        return [r for r in self.records if start <= r.timestamp <= end]

    def get_total_cost(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> float:
        """Get total cost in a period."""
        if start is None:
            records = self.records
        else:
            records = self.get_records_in_range(start, end or datetime.now())

        return sum(r.cost for r in records)

    def get_total_tokens(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> tuple[int, int]:
        """Get total input and output tokens."""
        if start is None:
            records = self.records
        else:
            records = self.get_records_in_range(start, end or datetime.now())

        input_total = sum(r.input_tokens for r in records)
        output_total = sum(r.output_tokens for r in records)
        return input_total, output_total

    def get_cost_by_model(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> dict[str, float]:
        """Get cost breakdown by model."""
        if start is None:
            records = self.records
        else:
            records = self.get_records_in_range(start, end or datetime.now())

        by_model: dict[str, float] = defaultdict(float)
        for r in records:
            by_model[r.model_name] += r.cost

        return dict(by_model)

    def get_summary(
        self,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> CostSummary:
        """Get a cost summary for a period."""
        end = end or datetime.now()
        records = self.get_records_in_range(start, end)

        total_cost = sum(r.cost for r in records)
        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)

        by_model: dict[str, float] = defaultdict(float)
        for r in records:
            by_model[r.model_name] += r.cost

        # Calculate by category (input vs output costs)
        by_category = {
            CostCategory.INPUT_TOKENS.value: 0.0,
            CostCategory.OUTPUT_TOKENS.value: 0.0,
        }
        for r in records:
            pricing = self.calculator.registry.get_or_default(r.model_name)
            by_category[CostCategory.INPUT_TOKENS.value] += (
                r.input_tokens / 1000
            ) * pricing.input_cost_per_1k
            by_category[CostCategory.OUTPUT_TOKENS.value] += (
                r.output_tokens / 1000
            ) * pricing.output_cost_per_1k

        return CostSummary(
            period_start=start,
            period_end=end,
            total_cost=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_requests=len(records),
            by_model=dict(by_model),
            by_category=by_category,
            average_cost_per_request=total_cost / len(records) if records else 0.0,
        )

    def clear_old_records(self, older_than: datetime) -> int:
        """Remove records older than a given date."""
        original_count = len(self.records)
        self.records = [r for r in self.records if r.timestamp >= older_than]
        return original_count - len(self.records)


class BudgetManager:
    """Manage budgets and alerts."""

    def __init__(self, tracker: Optional[UsageTracker] = None):
        self.tracker = tracker or UsageTracker()
        self.budgets: dict[str, Budget] = {}
        self.alerts: list[BudgetAlert] = []
        self.alert_callbacks: list[Callable[[BudgetAlert], None]] = []

    def create_budget(
        self,
        name: str,
        limit: float,
        period: TimeGranularity,
        alert_threshold: float = 0.8,
        critical_threshold: float = 0.95,
        hard_limit: bool = False,
    ) -> Budget:
        """Create a new budget."""
        budget = Budget(
            name=name,
            limit=limit,
            period=period,
            alert_threshold=alert_threshold,
            critical_threshold=critical_threshold,
            hard_limit=hard_limit,
        )
        self.budgets[name] = budget
        return budget

    def get_budget(self, name: str) -> Optional[Budget]:
        """Get a budget by name."""
        return self.budgets.get(name)

    def delete_budget(self, name: str) -> bool:
        """Delete a budget."""
        if name in self.budgets:
            del self.budgets[name]
            return True
        return False

    def _get_period_start(self, budget: Budget) -> datetime:
        """Get the start of the current budget period."""
        now = datetime.now()

        if budget.period == TimeGranularity.MINUTE:
            return now.replace(second=0, microsecond=0)
        elif budget.period == TimeGranularity.HOUR:
            return now.replace(minute=0, second=0, microsecond=0)
        elif budget.period == TimeGranularity.DAY:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif budget.period == TimeGranularity.WEEK:
            # Start of week (Monday)
            days_since_monday = now.weekday()
            start = now - timedelta(days=days_since_monday)
            return start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif budget.period == TimeGranularity.MONTH:
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)

    def check_budget(self, name: str) -> Optional[BudgetAlert]:
        """Check if a budget has been exceeded."""
        budget = self.budgets.get(name)
        if not budget:
            return None

        period_start = self._get_period_start(budget)
        current_spend = self.tracker.get_total_cost(start=period_start)
        percentage = current_spend / budget.limit if budget.limit > 0 else 0.0

        if percentage >= budget.critical_threshold:
            alert = BudgetAlert(
                level=AlertLevel.CRITICAL,
                budget_name=name,
                message=f"CRITICAL: Budget '{name}' at {percentage * 100:.1f}% (${current_spend:.4f}/${budget.limit:.2f})",
                current_spend=current_spend,
                budget_limit=budget.limit,
                percentage_used=percentage,
            )
        elif percentage >= budget.alert_threshold:
            alert = BudgetAlert(
                level=AlertLevel.WARNING,
                budget_name=name,
                message=f"WARNING: Budget '{name}' at {percentage * 100:.1f}% (${current_spend:.4f}/${budget.limit:.2f})",
                current_spend=current_spend,
                budget_limit=budget.limit,
                percentage_used=percentage,
            )
        else:
            return None

        self.alerts.append(alert)
        for callback in self.alert_callbacks:
            callback(alert)

        return alert

    def check_all_budgets(self) -> list[BudgetAlert]:
        """Check all budgets and return any alerts."""
        alerts = []
        for name in self.budgets:
            alert = self.check_budget(name)
            if alert:
                alerts.append(alert)
        return alerts

    def can_make_request(
        self,
        budget_name: str,
        estimated_cost: float,
    ) -> tuple[bool, Optional[str]]:
        """Check if a request can be made within budget."""
        budget = self.budgets.get(budget_name)
        if not budget:
            return True, None

        if not budget.hard_limit:
            return True, None

        period_start = self._get_period_start(budget)
        current_spend = self.tracker.get_total_cost(start=period_start)

        if current_spend + estimated_cost > budget.limit:
            return (
                False,
                f"Request would exceed budget '{budget_name}' (${current_spend:.4f} + ${estimated_cost:.4f} > ${budget.limit:.2f})",
            )

        return True, None

    def add_alert_callback(self, callback: Callable[[BudgetAlert], None]) -> None:
        """Add a callback to be called when alerts are triggered."""
        self.alert_callbacks.append(callback)

    def get_budget_status(self, name: str) -> Optional[dict[str, Any]]:
        """Get current status of a budget."""
        budget = self.budgets.get(name)
        if not budget:
            return None

        period_start = self._get_period_start(budget)
        current_spend = self.tracker.get_total_cost(start=period_start)
        percentage = current_spend / budget.limit if budget.limit > 0 else 0.0

        return {
            "name": name,
            "limit": budget.limit,
            "current_spend": current_spend,
            "remaining": max(0, budget.limit - current_spend),
            "percentage_used": percentage,
            "period": budget.period.value,
            "period_start": period_start.isoformat(),
            "is_exceeded": percentage >= 1.0,
            "is_warning": percentage >= budget.alert_threshold,
            "is_critical": percentage >= budget.critical_threshold,
        }


class CostForecaster:
    """Forecast future costs based on historical data."""

    def __init__(self, tracker: UsageTracker):
        self.tracker = tracker

    def forecast(
        self,
        period: TimeGranularity = TimeGranularity.MONTH,
        lookback_days: int = 30,
    ) -> CostForecast:
        """Generate a cost forecast."""
        end = datetime.now()
        start = end - timedelta(days=lookback_days)

        records = self.tracker.get_records_in_range(start, end)

        if not records:
            return CostForecast(
                forecast_period=period,
                projected_cost=0.0,
                confidence_low=0.0,
                confidence_high=0.0,
                based_on_days=0,
                trend="stable",
                projected_monthly=0.0,
            )

        # Calculate daily costs
        daily_costs: dict[str, float] = defaultdict(float)
        for r in records:
            day_key = r.timestamp.strftime("%Y-%m-%d")
            daily_costs[day_key] += r.cost

        costs_list = list(daily_costs.values())
        actual_days = len(costs_list)

        if actual_days == 0:
            return CostForecast(
                forecast_period=period,
                projected_cost=0.0,
                confidence_low=0.0,
                confidence_high=0.0,
                based_on_days=0,
                trend="stable",
                projected_monthly=0.0,
            )

        avg_daily = sum(costs_list) / actual_days

        # Calculate trend
        if actual_days >= 7:
            first_half = sum(costs_list[: actual_days // 2]) / (actual_days // 2)
            second_half = sum(costs_list[actual_days // 2 :]) / (actual_days - actual_days // 2)

            if second_half > first_half * 1.1:
                trend = "increasing"
            elif second_half < first_half * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Project based on period
        if period == TimeGranularity.DAY:
            multiplier = 1
        elif period == TimeGranularity.WEEK:
            multiplier = 7
        elif period == TimeGranularity.MONTH:
            multiplier = 30
        else:
            multiplier = 1

        projected_cost = avg_daily * multiplier

        # Calculate confidence interval
        if actual_days >= 2:
            variance = sum((c - avg_daily) ** 2 for c in costs_list) / actual_days
            std_dev = math.sqrt(variance)
            margin = 1.96 * std_dev * math.sqrt(multiplier)  # 95% CI
        else:
            margin = projected_cost * 0.5  # 50% margin if not enough data

        return CostForecast(
            forecast_period=period,
            projected_cost=projected_cost,
            confidence_low=max(0, projected_cost - margin),
            confidence_high=projected_cost + margin,
            based_on_days=actual_days,
            trend=trend,
            projected_monthly=avg_daily * 30,
            metadata={
                "avg_daily_cost": avg_daily,
                "lookback_days_requested": lookback_days,
                "actual_data_days": actual_days,
            },
        )

    def estimate_monthly_at_rate(
        self,
        daily_requests: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
        model_name: str,
    ) -> float:
        """Estimate monthly cost at a given usage rate."""
        cost_per_request = self.tracker.calculator.calculate_cost(
            model_name, avg_input_tokens, avg_output_tokens
        )
        return cost_per_request * daily_requests * 30


class CostReporter:
    """Generate cost reports."""

    def __init__(self, tracker: UsageTracker, forecaster: Optional[CostForecaster] = None):
        self.tracker = tracker
        self.forecaster = forecaster or CostForecaster(tracker)

    def generate_daily_report(self, date: Optional[datetime] = None) -> dict[str, Any]:
        """Generate a daily cost report."""
        if date is None:
            date = datetime.now()

        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        summary = self.tracker.get_summary(start, end)

        return {
            "date": start.strftime("%Y-%m-%d"),
            "summary": summary.to_dict(),
            "top_models": self._get_top_n(summary.by_model, 5),
        }

    def generate_weekly_report(self, week_start: Optional[datetime] = None) -> dict[str, Any]:
        """Generate a weekly cost report."""
        if week_start is None:
            now = datetime.now()
            days_since_monday = now.weekday()
            week_start = now - timedelta(days=days_since_monday)

        start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7)

        summary = self.tracker.get_summary(start, end)

        # Daily breakdown
        daily_costs = []
        for i in range(7):
            day_start = start + timedelta(days=i)
            day_end = day_start + timedelta(days=1)
            day_cost = self.tracker.get_total_cost(day_start, day_end)
            daily_costs.append(
                {
                    "date": day_start.strftime("%Y-%m-%d"),
                    "cost": day_cost,
                }
            )

        return {
            "week_start": start.strftime("%Y-%m-%d"),
            "week_end": (end - timedelta(days=1)).strftime("%Y-%m-%d"),
            "summary": summary.to_dict(),
            "daily_breakdown": daily_costs,
            "top_models": self._get_top_n(summary.by_model, 5),
        }

    def generate_monthly_report(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> dict[str, Any]:
        """Generate a monthly cost report."""
        now = datetime.now()
        year = year or now.year
        month = month or now.month

        start = datetime(year, month, 1)
        end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)

        summary = self.tracker.get_summary(start, end)
        forecast = self.forecaster.forecast(TimeGranularity.MONTH)

        return {
            "year": year,
            "month": month,
            "summary": summary.to_dict(),
            "forecast": forecast.to_dict(),
            "top_models": self._get_top_n(summary.by_model, 5),
        }

    def _get_top_n(self, costs: dict[str, float], n: int) -> list[dict[str, Any]]:
        """Get top N items by cost."""
        sorted_items = sorted(costs.items(), key=lambda x: x[1], reverse=True)
        return [{"name": name, "cost": cost} for name, cost in sorted_items[:n]]

    def format_report_text(self, report: dict[str, Any]) -> str:
        """Format a report as text."""
        lines = []

        if "date" in report:
            lines.append(f"Daily Cost Report - {report['date']}")
        elif "week_start" in report:
            lines.append(f"Weekly Cost Report - {report['week_start']} to {report['week_end']}")
        elif "month" in report:
            lines.append(f"Monthly Cost Report - {report['year']}/{report['month']:02d}")

        lines.append("=" * 50)

        summary = report.get("summary", {})
        lines.append(f"Total Cost: ${summary.get('total_cost', 0):.4f}")
        lines.append(f"Total Requests: {summary.get('total_requests', 0)}")
        lines.append(f"Total Input Tokens: {summary.get('total_input_tokens', 0):,}")
        lines.append(f"Total Output Tokens: {summary.get('total_output_tokens', 0):,}")
        lines.append(f"Avg Cost/Request: ${summary.get('average_cost_per_request', 0):.6f}")

        if "top_models" in report and report["top_models"]:
            lines.append("\nTop Models by Cost:")
            for i, item in enumerate(report["top_models"], 1):
                lines.append(f"  {i}. {item['name']}: ${item['cost']:.4f}")

        if "forecast" in report:
            forecast = report["forecast"]
            lines.append(f"\nForecast ({forecast['forecast_period']}):")
            lines.append(f"  Projected: ${forecast['projected_cost']:.4f}")
            lines.append(
                f"  Range: ${forecast['confidence_low']:.4f} - ${forecast['confidence_high']:.4f}"
            )
            lines.append(f"  Trend: {forecast['trend']}")

        return "\n".join(lines)


# Convenience functions


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate cost for a single request."""
    calculator = CostCalculator()
    return calculator.calculate_cost(model_name, input_tokens, output_tokens)


def estimate_request_cost(
    model_name: str,
    prompt: str,
    estimated_output_tokens: int = 500,
) -> float:
    """Estimate cost for a request before making it."""
    calculator = CostCalculator()
    return calculator.estimate_cost(model_name, prompt, estimated_output_tokens)


def compare_model_costs(
    input_tokens: int,
    output_tokens: int,
    model_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compare costs across models."""
    calculator = CostCalculator()
    return calculator.compare_models(input_tokens, output_tokens, model_names)


def get_cheapest_model(
    input_tokens: int,
    output_tokens: int,
    model_names: Optional[list[str]] = None,
) -> tuple[str, float]:
    """Find the cheapest model for given usage."""
    calculator = CostCalculator()
    return calculator.get_cheapest_model(input_tokens, output_tokens, model_names)


def create_usage_tracker() -> UsageTracker:
    """Create a new usage tracker."""
    return UsageTracker()


def create_budget_manager(tracker: Optional[UsageTracker] = None) -> BudgetManager:
    """Create a new budget manager."""
    return BudgetManager(tracker)


def quick_cost_estimate(
    model_name: str,
    num_requests: int,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 200,
) -> dict[str, float]:
    """Quick estimate for a batch of requests."""
    single_cost = calculate_cost(model_name, avg_input_tokens, avg_output_tokens)
    total_cost = single_cost * num_requests

    return {
        "model": model_name,
        "num_requests": num_requests,
        "cost_per_request": single_cost,
        "total_cost": total_cost,
        "daily_estimate": total_cost,
        "monthly_estimate": total_cost * 30,
    }
