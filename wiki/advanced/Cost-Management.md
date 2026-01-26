---
title: Cost Management
parent: Advanced Features
nav_order: 2
---

# Cost Management

**Track spending. Set budgets. Forecast costs.**

## The Problem

API bills surprise you. You don't know which models cost what. No way to set spending limits.

## The Solution

Built-in cost tracking with budget management and forecasting.

```python
from insideLLMs.cost_tracking import BudgetManager, calculate_cost

# Set monthly budget
budget = BudgetManager(limit=1000.0, period="monthly")

# Calculate cost before calling
estimated = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
print(f"Estimated: ${estimated:.4f}")

# Check budget
if budget.can_spend(estimated):
    response = model.generate(prompt)
    budget.record_spend(estimated)
```

## Features

### Cost Calculation

```python
from insideLLMs.cost_tracking import calculate_cost

# Calculate for specific model
cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
# $0.0150 (based on current pricing)

# Compare models
gpt4_cost = calculate_cost("gpt-4o", 1000, 500)
mini_cost = calculate_cost("gpt-4o-mini", 1000, 500)
print(f"Savings: ${gpt4_cost - mini_cost:.4f}")
```

### Budget Management

```python
from insideLLMs.cost_tracking import BudgetManager

# Create budget with alerts
budget = BudgetManager(
    limit=1000.0,
    period="monthly",
    warning_threshold=0.8,  # Alert at 80%
    critical_threshold=0.95  # Alert at 95%
)

# Check before spending
if budget.can_spend(cost):
    response = model.generate(prompt)
    budget.record_spend(cost)
else:
    print("Budget exceeded!")

# Get current status
status = budget.get_status()
print(f"Spent: ${status['spent']:.2f} / ${status['limit']:.2f}")
print(f"Remaining: ${status['remaining']:.2f}")
```

### Usage Tracking

```python
from insideLLMs.cost_tracking import UsageTracker

tracker = UsageTracker()

# Record usage
tracker.record(
    model="gpt-4o",
    input_tokens=1000,
    output_tokens=500,
    cost=0.015
)

# Get summary
summary = tracker.get_summary(period="daily")
print(f"Today's cost: ${summary['total_cost']:.2f}")
print(f"Total tokens: {summary['total_tokens']}")

# Breakdown by model
by_model = tracker.get_breakdown_by_model()
for model, stats in by_model.items():
    print(f"{model}: ${stats['cost']:.2f} ({stats['requests']} requests)")
```

### Cost Forecasting

```python
from insideLLMs.cost_tracking import CostForecaster

forecaster = CostForecaster(tracker)

# Project next month
forecast = forecaster.forecast_monthly()
print(f"Projected cost: ${forecast.projected_cost:.2f}")
print(f"Confidence: {forecast.confidence_interval}")
print(f"Trend: {forecast.trend}")  # increasing/decreasing/stable
```

## Integration with Pipeline

```python
from insideLLMs.pipeline import ModelPipeline, CostTrackingMiddleware

pipeline = ModelPipeline(model)
pipeline.add_middleware(CostTrackingMiddleware(
    budget_limit=1000.0,
    alert_threshold=0.8
))

# Automatic cost tracking
response = pipeline.generate(prompt)

# Check spend
print(f"Total cost: ${pipeline.get_total_cost():.4f}")
```

## Configuration

```yaml
# In harness config
cost_tracking:
  enabled: true
  budget:
    limit: 1000.0
    period: monthly
    alert_thresholds:
      warning: 0.8
      critical: 0.95
```

## Pricing Data

Built-in pricing for:
- OpenAI (GPT-4, GPT-4o, GPT-3.5)
- Anthropic (Claude 3 Opus, Sonnet, Haiku)
- Google (Gemini Pro, Ultra)
- Cohere (Command, Command-Light)

Custom pricing:
```python
from insideLLMs.cost_tracking import PricingRegistry

registry = PricingRegistry()
registry.add_model(
    model_id="custom-model",
    input_price_per_1k=0.01,
    output_price_per_1k=0.03
)
```

## Reports

```python
from insideLLMs.cost_tracking import CostReporter

reporter = CostReporter(tracker)

# Daily report
daily = reporter.generate_daily_report()
print(daily)

# Monthly report with forecast
monthly = reporter.generate_monthly_report(include_forecast=True)
print(monthly)
```

## Why This Matters

**Without cost management:**
- Surprise API bills
- No visibility into spending
- Can't optimise model selection
- No budget controls

**With cost management:**
- Know costs before calling
- Set hard limits
- Track spending by model/time
- Forecast future costs
- Optimise model selection based on cost/performance

## See Also

- [Pipeline Architecture](Pipeline-Architecture.md) - Integrate cost tracking with other middleware
- [Rate Limiting](../guides/Rate-Limiting.md) - Combine with rate limits
