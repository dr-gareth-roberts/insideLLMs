---
title: Structured Outputs
parent: Advanced Features
nav_order: 3
---

# Structured Outputs

**Extract Pydantic models from LLM responses. Reliably.**

## The Problem

LLMs return text. You need structured data. Manual JSON parsing is fragile.

```python
# Fragile
response = model.generate("Extract person info: John, 30, engineer")
data = json.loads(response)  # Fails if LLM adds explanation
```

## The Solution

Automatic extraction with validation.

```python
from pydantic import BaseModel
from insideLLMs.structured import generate_structured

class Person(BaseModel):
    name: str
    age: int
    occupation: str

result = generate_structured(
    model,
    Person,
    "Extract: John, 30, engineer"
)

print(result.data.name)  # "John"
print(result.data.age)   # 30
```

## How It Works

1. **Generate JSON schema** from Pydantic model
2. **Prompt LLM** with schema and instructions
3. **Extract JSON** from response (handles markdown, code blocks, mixed text)
4. **Validate** and instantiate Pydantic model
5. **Return** typed result with metadata

## Basic Usage

```python
from pydantic import BaseModel
from insideLLMs.structured import generate_structured
from insideLLMs.models import OpenAIModel

class Product(BaseModel):
    name: str
    price: float
    category: str
    in_stock: bool

model = OpenAIModel(model_name="gpt-4o")

result = generate_structured(
    model,
    Product,
    "Extract product info: iPhone 15 Pro, $999, Electronics, available"
)

print(result.data.name)      # "iPhone 15 Pro"
print(result.data.price)     # 999.0
print(result.data.in_stock)  # True
```

## Nested Models

```python
from pydantic import BaseModel
from typing import List

class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    addresses: List[Address]

result = generate_structured(
    model,
    Person,
    "John, 30, lives at 123 Main St, London, UK and 456 Oak Ave, Paris, France"
)

print(result.data.addresses[0].city)  # "London"
```

## Batch Processing

```python
from insideLLMs.structured import StructuredOutputGenerator

generator = StructuredOutputGenerator(model, Person)

# Process multiple inputs
inputs = [
    "Alice, 25, teacher",
    "Bob, 35, engineer",
    "Carol, 28, designer"
]

results = generator.generate_batch(inputs, concurrency=3)

for result in results:
    print(f"{result.data.name}: {result.data.occupation}")
```

## Retry on Validation Failure

```python
result = generate_structured(
    model,
    Person,
    prompt,
    max_retries=3,  # Retry if JSON invalid
    retry_prompt="The previous response was invalid. Please return valid JSON."
)
```

## Export Formats

```python
# JSON
result.to_json()

# Dict
result.to_dict()

# DataFrame (for batch results)
df = generator.to_dataframe(results)
```

## Error Handling

```python
from insideLLMs.structured import StructuredOutputError

try:
    result = generate_structured(model, Person, prompt)
except StructuredOutputError as e:
    print(f"Extraction failed: {e}")
    print(f"Raw response: {e.raw_response}")
    print(f"Validation errors: {e.validation_errors}")
```

## Configuration

```yaml
# In probe config
structured_output:
  enabled: true
  schema: Person
  max_retries: 3
  strict_validation: true
```

## Advanced: Custom Instructions

```python
result = generate_structured(
    model,
    Person,
    prompt,
    instructions="Extract person information. Use 'Unknown' for missing fields.",
    examples=[
        {"input": "John, 30", "output": {"name": "John", "age": 30, "occupation": "Unknown"}}
    ]
)
```

## Why This Matters

**Without structured outputs:**
- Manual JSON parsing
- Fragile string manipulation
- No type safety
- Validation errors at runtime

**With structured outputs:**
- Automatic extraction
- Type-safe results
- Validation before use
- Retry on failure

## Comparison

| Approach | Reliability | Type Safety | Effort |
|----------|-------------|-------------|--------|
| Manual parsing | Low | No | High |
| `json.loads()` | Medium | No | Medium |
| insideLLMs structured | High | Yes | Low |

## See Also

- [Pipeline Architecture](Pipeline-Architecture.md) - Combine with middleware
- [Custom Probe Tutorial](../tutorials/Custom-Probe.md) - Use in probes
