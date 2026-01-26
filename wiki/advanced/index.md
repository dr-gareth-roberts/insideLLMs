---
title: Advanced Features
nav_order: 6
has_children: true
---

# Advanced Features

**Power features that differentiate insideLLMs from benchmark frameworks.**

## What's Here

| Feature | What It Does |
|---------|--------------|
| [Pipeline Architecture](Pipeline-Architecture.md) | Composable middleware for caching, retry, cost tracking |
| [Cost Management](Cost-Management.md) | Budget limits, usage tracking, cost forecasting |
| [Structured Outputs](Structured-Outputs.md) | Extract Pydantic models from LLM responses |
| [Agent Evaluation](Agent-Evaluation.md) | Test tool-using agents with trace integration |
| [Retry Strategies](Retry-Strategies.md) | Circuit breakers, exponential backoff, error handling |

## Why These Matter

Traditional eval frameworks give you benchmark scores. insideLLMs gives you production-grade infrastructure:

**Pipeline middleware** - Wrap models with cross-cutting concerns without changing code.

**Cost management** - Set budgets, track spending, forecast costs. Don't get surprised by API bills.

**Structured outputs** - Parse JSON from LLM responses reliably. No more manual string parsing.

**Agent evaluation** - Test tool-using agents systematically. Trace tool calls and decisions.

**Retry strategies** - Handle transient failures gracefully. Circuit breakers prevent cascade failures.

## When to Use Advanced Features

| You Need... | Use This |
|-------------|----------|
| Reduce API costs | Pipeline + Cost Management |
| Handle rate limits gracefully | Pipeline + Retry Strategies |
| Extract structured data | Structured Outputs |
| Test tool-using agents | Agent Evaluation |
| Production-grade reliability | All of the above |

## These Are Differentiators

Eleuther, HELM, OpenAI Evals don't have these. insideLLMs does.

If you're shipping LLM products to production, you need more than benchmark scores. You need infrastructure.
