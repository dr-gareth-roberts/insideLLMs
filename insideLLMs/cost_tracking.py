"""Cost estimation and budget tracking for LLM API usage.

This module provides a comprehensive suite of tools for tracking, estimating,
and managing costs associated with Large Language Model (LLM) API calls. It
supports multiple providers (OpenAI, Anthropic, Google) with extensible pricing
configurations.

Overview
--------
The module is organized around several core components:

1. **Pricing Management** (`PricingRegistry`, `ModelPricing`)
   - Maintains pricing information for various LLM models
   - Supports custom pricing configurations and updates
   - Pre-loaded with common model pricing from major providers

2. **Cost Calculation** (`TokenCostCalculator`)
   - Calculates actual costs based on token usage
   - Estimates costs before making API requests
   - Compares costs across different models

3. **Usage Tracking** (`UsageTracker`, `UsageRecord`)
   - Records API usage events with timestamps and metadata
   - Aggregates usage data by time period, model, or category
   - Generates cost summaries and breakdowns

4. **Budget Management** (`BudgetManager`, `Budget`, `BudgetAlert`)
   - Creates and manages spending budgets with configurable periods
   - Triggers alerts at customizable thresholds (warning/critical)
   - Supports hard limits that block requests when exceeded

5. **Cost Forecasting** (`CostForecaster`, `CostForecast`)
   - Projects future costs based on historical usage patterns
   - Detects spending trends (increasing/decreasing/stable)
   - Provides confidence intervals for projections

6. **Reporting** (`CostReporter`, `CostSummary`)
   - Generates daily, weekly, and monthly cost reports
   - Formats reports for display or export
   - Integrates forecasting into monthly reports

Examples
--------
Basic cost calculation for a single request:

>>> from insideLLMs.cost_tracking import calculate_cost
>>> cost = calculate_cost("gpt-4", input_tokens=1000, output_tokens=500)
>>> print(f"Request cost: ${cost:.4f}")
Request cost: $0.0600

Estimate cost before making a request:

>>> from insideLLMs.cost_tracking import estimate_request_cost
>>> prompt = "Explain quantum computing in simple terms."
>>> estimated = estimate_request_cost("claude-3-sonnet", prompt, estimated_output_tokens=300)
>>> print(f"Estimated cost: ${estimated:.4f}")
Estimated cost: $0.0048

Compare costs across multiple models:

>>> from insideLLMs.cost_tracking import compare_model_costs
>>> costs = compare_model_costs(
...     input_tokens=2000,
...     output_tokens=1000,
...     model_names=["gpt-4", "gpt-4-turbo", "claude-3-sonnet"]
... )
>>> for model, cost in sorted(costs.items(), key=lambda x: x[1]):
...     print(f"{model}: ${cost:.4f}")
claude-3-sonnet: $0.0210
gpt-4-turbo: $0.0500
gpt-4: $0.1200

Track usage and manage budgets:

>>> from insideLLMs.cost_tracking import create_usage_tracker, create_budget_manager
>>> from insideLLMs.cost_tracking import TimeGranularity
>>> tracker = create_usage_tracker()
>>> manager = create_budget_manager(tracker)
>>>
>>> # Create a daily budget of $10
>>> budget = manager.create_budget(
...     name="daily-api-budget",
...     limit=10.0,
...     period=TimeGranularity.DAY,
...     alert_threshold=0.8
... )
>>>
>>> # Record some usage
>>> tracker.record_usage("gpt-4", input_tokens=5000, output_tokens=2000)
>>>
>>> # Check budget status
>>> status = manager.get_budget_status("daily-api-budget")
>>> print(f"Spent: ${status['current_spend']:.4f} of ${status['limit']:.2f}")

Generate cost reports:

>>> from insideLLMs.cost_tracking import UsageTracker, CostReporter
>>> tracker = UsageTracker()
>>> # ... record usage ...
>>> reporter = CostReporter(tracker)
>>> report = reporter.generate_daily_report()
>>> print(reporter.format_report_text(report))

Notes
-----
- All costs are in USD by default; currency conversion is not built-in
- Token counts should be obtained from the LLM provider's response
- The module uses in-memory storage; for persistence, serialize records externally
- Pre-loaded pricing may become outdated; use `PricingRegistry.update_pricing()`
  to maintain current rates
- Budget periods are aligned to calendar boundaries (start of day/week/month)

See Also
--------
- OpenAI Pricing: https://openai.com/pricing
- Anthropic Pricing: https://anthropic.com/pricing
- Google AI Pricing: https://ai.google.dev/pricing
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional


class PricingTier(Enum):
    """Pricing tiers for different API access levels.

    This enumeration represents the various pricing tiers offered by LLM
    providers, which may affect the cost per token, rate limits, and
    available features.

    Attributes
    ----------
    FREE : str
        Free tier with limited usage quotas and rate limits.
        Typically used for experimentation and development.
    PAY_AS_YOU_GO : str
        Standard pay-per-use pricing without committed spend.
        Most common tier for production applications.
    STANDARD : str
        Standard subscription tier with predictable monthly costs.
        May include volume discounts.
    ENTERPRISE : str
        Enterprise tier with custom pricing, SLAs, and support.
        Typically requires annual contracts.
    CUSTOM : str
        Custom negotiated pricing tier.
        Used for special arrangements with providers.

    Examples
    --------
    Create a model pricing with a specific tier:

    >>> pricing = ModelPricing(
    ...     model_name="custom-model",
    ...     input_cost_per_1k=0.01,
    ...     output_cost_per_1k=0.02,
    ...     tier=PricingTier.ENTERPRISE
    ... )
    >>> print(pricing.tier)
    PricingTier.ENTERPRISE

    Check if a model is on the free tier:

    >>> if pricing.tier == PricingTier.FREE:
    ...     print("Limited usage available")

    Convert tier to string for serialization:

    >>> tier_value = PricingTier.PAY_AS_YOU_GO.value
    >>> print(tier_value)
    'pay_as_you_go'

    See Also
    --------
    ModelPricing : Pricing information that uses this tier enumeration.
    """

    FREE = "free"
    PAY_AS_YOU_GO = "pay_as_you_go"
    STANDARD = "standard"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class CostCategory(Enum):
    """Categories of costs for LLM API usage.

    This enumeration categorizes different types of costs that can be
    incurred when using LLM APIs. Used for detailed cost breakdowns
    and reporting.

    Attributes
    ----------
    INPUT_TOKENS : str
        Costs associated with input/prompt tokens sent to the model.
        Typically the lower-cost component of a request.
    OUTPUT_TOKENS : str
        Costs associated with output/completion tokens generated by the model.
        Usually charged at a higher rate than input tokens.
    EMBEDDING : str
        Costs for generating text embeddings.
        Used for semantic search, similarity, and RAG applications.
    FINE_TUNING : str
        Costs associated with fine-tuning or training custom models.
        Includes both training and inference on fine-tuned models.
    STORAGE : str
        Costs for storing fine-tuned models, embeddings, or data.
        May be charged per GB/month.
    API_CALLS : str
        Fixed per-call charges, if applicable.
        Some providers charge a base fee per API request.
    OTHER : str
        Miscellaneous costs not covered by other categories.
        Used for provider-specific charges.

    Examples
    --------
    Categorize costs in a summary:

    >>> costs_by_category = {
    ...     CostCategory.INPUT_TOKENS.value: 0.50,
    ...     CostCategory.OUTPUT_TOKENS.value: 1.50,
    ... }
    >>> total_token_costs = sum(costs_by_category.values())
    >>> print(f"Total token costs: ${total_token_costs:.2f}")
    Total token costs: $2.00

    Filter costs by category:

    >>> if category == CostCategory.EMBEDDING:
    ...     print("Processing embedding costs")

    Use in cost breakdown reports:

    >>> for category in CostCategory:
    ...     print(f"{category.value}: applicable to LLM usage")
    input_tokens: applicable to LLM usage
    output_tokens: applicable to LLM usage
    ...

    See Also
    --------
    CostSummary : Summary that includes cost breakdowns by category.
    """

    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    EMBEDDING = "embedding"
    FINE_TUNING = "fine_tuning"
    STORAGE = "storage"
    API_CALLS = "api_calls"
    OTHER = "other"


class AlertLevel(Enum):
    """Alert severity levels for budget notifications.

    This enumeration defines the severity levels for budget alerts,
    allowing users to configure appropriate responses based on urgency.

    Attributes
    ----------
    INFO : str
        Informational alert for routine notifications.
        No immediate action required; used for status updates.
    WARNING : str
        Warning alert indicating approaching budget thresholds.
        Typically triggered at 80% of budget limit by default.
    CRITICAL : str
        Critical alert indicating imminent or exceeded budget limits.
        Typically triggered at 95% of budget limit by default.
        May block further requests if hard limits are enabled.

    Examples
    --------
    Check alert severity and take action:

    >>> alert = manager.check_budget("monthly-budget")
    >>> if alert and alert.level == AlertLevel.CRITICAL:
    ...     print("URGENT: Budget nearly exhausted!")
    ...     notify_admin(alert)
    >>> elif alert and alert.level == AlertLevel.WARNING:
    ...     print("Warning: Approaching budget limit")

    Filter alerts by severity:

    >>> critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
    >>> print(f"Found {len(critical_alerts)} critical alerts")

    Use in logging or monitoring:

    >>> import logging
    >>> log_level = {
    ...     AlertLevel.INFO: logging.INFO,
    ...     AlertLevel.WARNING: logging.WARNING,
    ...     AlertLevel.CRITICAL: logging.CRITICAL
    ... }
    >>> logger.log(log_level[alert.level], alert.message)

    See Also
    --------
    BudgetAlert : Alert object that uses this severity level.
    Budget : Budget configuration with threshold settings.
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class TimeGranularity(Enum):
    """Time granularity for reporting and budget periods.

    This enumeration defines time periods used for budget cycles,
    cost aggregation, and reporting intervals.

    Attributes
    ----------
    MINUTE : str
        Minute-level granularity.
        Useful for real-time monitoring and rate limiting.
    HOUR : str
        Hour-level granularity.
        Suitable for high-frequency usage tracking.
    DAY : str
        Day-level granularity.
        Most common for operational budgets and daily reports.
    WEEK : str
        Week-level granularity. Weeks start on Monday.
        Used for weekly cost reviews and sprint-based budgeting.
    MONTH : str
        Month-level granularity. Months start on the 1st.
        Standard for financial reporting and monthly budgets.

    Examples
    --------
    Create a daily budget:

    >>> budget = manager.create_budget(
    ...     name="daily-limit",
    ...     limit=50.0,
    ...     period=TimeGranularity.DAY
    ... )

    Create a monthly budget for production:

    >>> production_budget = manager.create_budget(
    ...     name="production-monthly",
    ...     limit=5000.0,
    ...     period=TimeGranularity.MONTH,
    ...     hard_limit=True
    ... )

    Generate forecasts at different granularities:

    >>> daily_forecast = forecaster.forecast(period=TimeGranularity.DAY)
    >>> monthly_forecast = forecaster.forecast(period=TimeGranularity.MONTH)
    >>> print(f"Daily: ${daily_forecast.projected_cost:.2f}")
    >>> print(f"Monthly: ${monthly_forecast.projected_cost:.2f}")

    See Also
    --------
    Budget : Budget configuration that uses time granularity for periods.
    CostForecast : Forecast that projects costs for a given time period.
    CostReporter : Reporter with daily, weekly, and monthly report methods.
    """

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class ModelPricing:
    """Pricing information for a specific LLM model.

    This dataclass encapsulates all pricing-related information for an LLM
    model, including token costs, context limits, and metadata. Used by
    PricingRegistry to maintain a catalog of model pricing.

    Parameters
    ----------
    model_name : str
        The identifier for the model (e.g., "gpt-4", "claude-3-sonnet").
        Should match the name used in API calls.
    input_cost_per_1k : float
        Cost in currency units per 1,000 input tokens.
        This is the cost for tokens in the prompt/context.
    output_cost_per_1k : float
        Cost in currency units per 1,000 output tokens.
        This is the cost for tokens generated by the model.
    currency : str, optional
        Currency code for the costs. Default is "USD".
    tier : PricingTier, optional
        The pricing tier for this model. Default is PAY_AS_YOU_GO.
    embedding_cost_per_1k : float or None, optional
        Cost per 1,000 tokens for embedding operations, if applicable.
        None if the model doesn't support embeddings.
    context_window : int or None, optional
        Maximum context window size in tokens.
        None if unknown or variable.
    effective_date : datetime or None, optional
        Date when this pricing became effective.
        Useful for tracking pricing history.
    notes : str, optional
        Additional notes about the pricing (e.g., volume discounts).

    Attributes
    ----------
    model_name : str
        The model identifier.
    input_cost_per_1k : float
        Input token cost per 1,000 tokens.
    output_cost_per_1k : float
        Output token cost per 1,000 tokens.
    currency : str
        Currency code (default: "USD").
    tier : PricingTier
        Pricing tier enumeration value.
    embedding_cost_per_1k : float or None
        Embedding cost per 1,000 tokens.
    context_window : int or None
        Maximum context window size.
    effective_date : datetime or None
        When pricing became effective.
    notes : str
        Additional pricing notes.

    Examples
    --------
    Create pricing for a new model:

    >>> pricing = ModelPricing(
    ...     model_name="gpt-4-turbo",
    ...     input_cost_per_1k=0.01,
    ...     output_cost_per_1k=0.03,
    ...     context_window=128000
    ... )
    >>> print(f"{pricing.model_name}: ${pricing.input_cost_per_1k}/1k input")
    gpt-4-turbo: $0.01/1k input

    Create pricing with full details:

    >>> from datetime import datetime
    >>> enterprise_pricing = ModelPricing(
    ...     model_name="gpt-4-enterprise",
    ...     input_cost_per_1k=0.008,
    ...     output_cost_per_1k=0.024,
    ...     currency="USD",
    ...     tier=PricingTier.ENTERPRISE,
    ...     context_window=128000,
    ...     effective_date=datetime(2024, 1, 1),
    ...     notes="20% enterprise discount applied"
    ... )

    Serialize pricing for storage:

    >>> data = pricing.to_dict()
    >>> print(data["model_name"])
    'gpt-4-turbo'

    Deserialize pricing from storage:

    >>> restored = ModelPricing.from_dict(data)
    >>> assert restored.model_name == pricing.model_name

    See Also
    --------
    PricingRegistry : Registry that manages collections of ModelPricing.
    PricingTier : Enumeration of pricing tier levels.
    """

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
        """Convert the pricing information to a dictionary.

        Serializes all pricing attributes to a dictionary format suitable
        for JSON serialization, database storage, or API responses.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all pricing attributes with the following keys:
            - model_name: str
            - input_cost_per_1k: float
            - output_cost_per_1k: float
            - currency: str
            - tier: str (enum value)
            - embedding_cost_per_1k: float or None
            - context_window: int or None
            - effective_date: str (ISO format) or None
            - notes: str

        Examples
        --------
        Serialize pricing for JSON storage:

        >>> pricing = ModelPricing("gpt-4", 0.03, 0.06)
        >>> data = pricing.to_dict()
        >>> import json
        >>> json_str = json.dumps(data)
        >>> print(data["model_name"])
        'gpt-4'

        Use in API responses:

        >>> pricing = ModelPricing("claude-3-sonnet", 0.003, 0.015)
        >>> response = {"pricing": pricing.to_dict(), "status": "active"}

        See Also
        --------
        from_dict : Class method to reconstruct from dictionary.
        """
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
        """Create a ModelPricing instance from a dictionary.

        Deserializes pricing information from a dictionary, typically
        loaded from JSON, a database, or an API response.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing pricing information with required keys:
            - model_name: str
            - input_cost_per_1k: float
            - output_cost_per_1k: float

            And optional keys:
            - currency: str (default: "USD")
            - tier: str (default: "pay_as_you_go")
            - embedding_cost_per_1k: float
            - context_window: int
            - effective_date: str (ISO format)
            - notes: str

        Returns
        -------
        ModelPricing
            A new ModelPricing instance populated with the data.

        Raises
        ------
        KeyError
            If required keys (model_name, input_cost_per_1k, output_cost_per_1k)
            are missing from the data dictionary.
        ValueError
            If tier value is not a valid PricingTier enum value.

        Examples
        --------
        Load pricing from JSON:

        >>> import json
        >>> json_str = '{"model_name": "gpt-4", "input_cost_per_1k": 0.03, "output_cost_per_1k": 0.06}'
        >>> data = json.loads(json_str)
        >>> pricing = ModelPricing.from_dict(data)
        >>> print(pricing.model_name)
        'gpt-4'

        Load pricing with all fields:

        >>> data = {
        ...     "model_name": "claude-3-opus",
        ...     "input_cost_per_1k": 0.015,
        ...     "output_cost_per_1k": 0.075,
        ...     "tier": "enterprise",
        ...     "context_window": 200000,
        ...     "effective_date": "2024-01-15T00:00:00"
        ... }
        >>> pricing = ModelPricing.from_dict(data)
        >>> print(pricing.tier)
        PricingTier.ENTERPRISE

        Round-trip serialization:

        >>> original = ModelPricing("gpt-4", 0.03, 0.06)
        >>> restored = ModelPricing.from_dict(original.to_dict())
        >>> assert original.model_name == restored.model_name

        See Also
        --------
        to_dict : Instance method to serialize to dictionary.
        """
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
    """Record of a single API usage event.

    This dataclass represents a single LLM API request with its associated
    token counts, costs, and metadata. Used by UsageTracker to maintain
    a history of all API calls.

    Parameters
    ----------
    timestamp : datetime
        When the API request was made.
    model_name : str
        The model used for the request (e.g., "gpt-4", "claude-3-sonnet").
    input_tokens : int
        Number of tokens in the input/prompt.
    output_tokens : int
        Number of tokens in the model's response.
    cost : float
        Calculated cost for this request in the configured currency.
    request_id : str or None, optional
        Unique identifier for the request, if available from the API.
        Useful for debugging and audit trails.
    metadata : dict[str, Any], optional
        Additional metadata about the request (e.g., user ID, session ID,
        application context, response latency).

    Attributes
    ----------
    timestamp : datetime
        When the request was made.
    model_name : str
        The model identifier.
    input_tokens : int
        Input token count.
    output_tokens : int
        Output token count.
    cost : float
        Calculated cost for this request.
    request_id : str or None
        Request identifier.
    metadata : dict[str, Any]
        Additional request metadata.
    total_tokens : int
        Property returning sum of input and output tokens.

    Examples
    --------
    Create a basic usage record:

    >>> from datetime import datetime
    >>> record = UsageRecord(
    ...     timestamp=datetime.now(),
    ...     model_name="gpt-4",
    ...     input_tokens=500,
    ...     output_tokens=200,
    ...     cost=0.027
    ... )
    >>> print(f"Total tokens: {record.total_tokens}")
    Total tokens: 700

    Create a record with metadata for tracking:

    >>> record = UsageRecord(
    ...     timestamp=datetime.now(),
    ...     model_name="claude-3-sonnet",
    ...     input_tokens=1000,
    ...     output_tokens=500,
    ...     cost=0.0105,
    ...     request_id="req_abc123",
    ...     metadata={
    ...         "user_id": "user_456",
    ...         "session_id": "sess_789",
    ...         "feature": "chat_completion",
    ...         "latency_ms": 1250
    ...     }
    ... )
    >>> print(record.metadata["feature"])
    'chat_completion'

    Serialize for storage:

    >>> data = record.to_dict()
    >>> print(data["model_name"])
    'claude-3-sonnet'

    See Also
    --------
    UsageTracker : Manager class that creates and stores UsageRecords.
    TokenCostCalculator : Calculator used to compute the cost field.
    """

    timestamp: datetime
    model_name: str
    input_tokens: int
    output_tokens: int
    cost: float
    request_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the usage record to a dictionary.

        Serializes all record attributes to a dictionary format suitable
        for JSON serialization, database storage, or analytics pipelines.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all record attributes:
            - timestamp: str (ISO format)
            - model_name: str
            - input_tokens: int
            - output_tokens: int
            - cost: float
            - request_id: str or None
            - metadata: dict

        Examples
        --------
        Serialize for JSON storage:

        >>> record = UsageRecord(
        ...     timestamp=datetime.now(),
        ...     model_name="gpt-4",
        ...     input_tokens=100,
        ...     output_tokens=50,
        ...     cost=0.006
        ... )
        >>> data = record.to_dict()
        >>> import json
        >>> json_str = json.dumps(data)

        Export to analytics system:

        >>> records = [r.to_dict() for r in tracker.records]
        >>> send_to_analytics(records)
        """
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
        """Total tokens used in this request.

        Returns the sum of input and output tokens, useful for
        understanding overall token consumption regardless of direction.

        Returns
        -------
        int
            Sum of input_tokens and output_tokens.

        Examples
        --------
        Calculate total token usage:

        >>> record = UsageRecord(
        ...     timestamp=datetime.now(),
        ...     model_name="gpt-4",
        ...     input_tokens=1500,
        ...     output_tokens=500,
        ...     cost=0.075
        ... )
        >>> print(f"Total: {record.total_tokens} tokens")
        Total: 2000 tokens

        Use in aggregation:

        >>> total = sum(r.total_tokens for r in tracker.records)
        >>> print(f"All requests used {total} tokens")
        """
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


class TokenCostCalculator:
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

    def __init__(self, calculator: Optional[TokenCostCalculator] = None):
        self.calculator = calculator or TokenCostCalculator()
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
    calculator = TokenCostCalculator()
    return calculator.calculate_cost(model_name, input_tokens, output_tokens)


def estimate_request_cost(
    model_name: str,
    prompt: str,
    estimated_output_tokens: int = 500,
) -> float:
    """Estimate cost for a request before making it."""
    calculator = TokenCostCalculator()
    return calculator.estimate_cost(model_name, prompt, estimated_output_tokens)


def compare_model_costs(
    input_tokens: int,
    output_tokens: int,
    model_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compare costs across models."""
    calculator = TokenCostCalculator()
    return calculator.compare_models(input_tokens, output_tokens, model_names)


def get_cheapest_model(
    input_tokens: int,
    output_tokens: int,
    model_names: Optional[list[str]] = None,
) -> tuple[str, float]:
    """Find the cheapest model for given usage."""
    calculator = TokenCostCalculator()
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


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import CostCalculator. The canonical name is
# TokenCostCalculator.
CostCalculator = TokenCostCalculator
