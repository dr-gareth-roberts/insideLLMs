---
title: Agent Evaluation
parent: Advanced Features
nav_order: 4
---

# Agent Evaluation

**Test tool-using agents systematically.**

## The Problem

Your LLM agent uses tools (search, calculator, database). How do you test it systematically?

Traditional probes test text generation. Agents need tool execution testing.

## The Solution

AgentProbe with trace integration.

```python
from insideLLMs.probes import AgentProbe, ToolDefinition

# Define available tools
tools = [
    ToolDefinition(
        name="calculator",
        description="Perform mathematical calculations",
        parameters={"expression": "string"}
    ),
    ToolDefinition(
        name="search",
        description="Search the web",
        parameters={"query": "string"}
    )
]

# Create agent probe
probe = AgentProbe(tools=tools)

# Test agent
result = probe.run(model, {
    "task": "What is 15% of the GDP of France?",
    "expected_tools": ["search", "calculator"]
})

print(result["tools_used"])      # ["search", "calculator"]
print(result["correct_sequence"]) # True
print(result["final_answer"])    # "..."
```

## Tool Definition

```python
from insideLLMs.probes import ToolDefinition

calculator = ToolDefinition(
    name="calculator",
    description="Evaluate mathematical expressions",
    parameters={
        "expression": {
            "type": "string",
            "description": "Math expression to evaluate"
        }
    },
    implementation=lambda expr: eval(expr)  # In production, use safe eval
)
```

## Testing Tool Selection

```python
# Test if agent chooses correct tools
test_cases = [
    {
        "task": "What is 25 * 17?",
        "expected_tools": ["calculator"],
        "expected_sequence": ["calculator"]
    },
    {
        "task": "Who won the 2024 Olympics 100m?",
        "expected_tools": ["search"],
        "expected_sequence": ["search"]
    },
    {
        "task": "What is the population of Tokyo divided by 2?",
        "expected_tools": ["search", "calculator"],
        "expected_sequence": ["search", "calculator"]
    }
]

for case in test_cases:
    result = probe.run(model, case)
    assert result["tools_used"] == case["expected_tools"]
```

## Trace Integration

AgentProbe automatically captures tool execution traces:

```json
{
  "task": "What is 15% of France's GDP?",
  "trace": {
    "steps": [
      {
        "step": 1,
        "tool": "search",
        "input": {"query": "France GDP 2024"},
        "output": "France GDP: $3.05 trillion"
      },
      {
        "step": 2,
        "tool": "calculator",
        "input": {"expression": "3.05 * 0.15"},
        "output": "0.4575"
      }
    ],
    "final_answer": "Approximately $457.5 billion"
  }
}
```

## Multi-Step Agent Testing

```python
# Test complex multi-step workflows
result = probe.run(model, {
    "task": "Book a flight from London to Paris, then find a hotel",
    "expected_tools": ["flight_search", "flight_booking", "hotel_search"],
    "max_steps": 5
})

# Verify execution
assert len(result["trace"]["steps"]) <= 5
assert "flight_booking" in result["tools_used"]
assert result["task_completed"]
```

## Scoring Agent Performance

```python
from insideLLMs.probes import AgentProbe

probe = AgentProbe(tools=tools)

# Run on dataset
results = probe.run_batch(model, agent_test_dataset)

# Score
score = probe.score(results)
print(f"Tool selection accuracy: {score.value:.1%}")
print(f"Correct sequences: {score.details['correct_sequences']}")
print(f"Average steps: {score.details['avg_steps']}")
```

## Configuration

```yaml
# In harness config
probes:
  - type: agent
    args:
      tools:
        - name: calculator
          description: Evaluate math expressions
        - name: search
          description: Search the web
      max_steps: 5
      trace_enabled: true
```

## Real-World Example

```python
# Test customer service agent
tools = [
    ToolDefinition(name="check_order", ...),
    ToolDefinition(name="process_refund", ...),
    ToolDefinition(name="send_email", ...)
]

probe = AgentProbe(tools=tools)

test_cases = [
    {
        "task": "Customer wants to return order #12345",
        "expected_tools": ["check_order", "process_refund", "send_email"],
        "expected_outcome": "refund_processed"
    }
]

results = probe.run_batch(model, test_cases)

# Verify agent behaviour
for result in results:
    if not result["task_completed"]:
        print(f"Failed: {result['task']}")
        print(f"Tools used: {result['tools_used']}")
        print(f"Expected: {result['expected_tools']}")
```

## Why This Matters

**Without AgentProbe:**
- Manual testing of tool-using agents
- No systematic verification
- Can't track tool selection patterns
- Difficult to catch regressions

**With AgentProbe:**
- Systematic agent testing
- Trace tool execution
- Verify correct tool selection
- Catch agent behaviour regressions

## See Also

- [Probes Catalog](../reference/Probes-Catalog.md) - Other probe types
- [Custom Probe Tutorial](../tutorials/Custom-Probe.md) - Build custom probes
- [Tracing and Fingerprinting](../Tracing-and-Fingerprinting.md) - Trace details
