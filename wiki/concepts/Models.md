---
title: Models
parent: Concepts
nav_order: 1
---

# Models

**Unified interface for all LLM providers.**

## Interface

```python
from collections.abc import Sequence
from insideLLMs.models.base import ChatMessage
from insideLLMs.types import ModelInfo

class Model:
    def generate(self, prompt: str, **kwargs) -> str:
        """Text completion."""

    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> str:
        """Multi-turn conversation."""

    def info(self) -> ModelInfo:
        """Model metadata."""
```

One interface. All providers.

## Why It Matters

Write probes once. Test across:
- OpenAI, Anthropic, Google, Cohere
- Ollama, llama.cpp, vLLM
- Custom implementations

## Creating Models

### From Registry

```python
from insideLLMs.registry import model_registry, ensure_builtins_registered

ensure_builtins_registered()

# Get by type
model = model_registry.get("openai", model_name="gpt-4o")
```

### Direct Instantiation

```python
from insideLLMs.models import OpenAIModel, AnthropicModel, DummyModel

openai = OpenAIModel(model_name="gpt-4o", temperature=0.7)
claude = AnthropicModel(model_name="claude-3-5-sonnet-20241022")
dummy = DummyModel(canned_response="test response")
```

### From Config

```yaml
model:
  type: openai
  args:
    model_name: gpt-4o
    temperature: 0.7
```

## Model Info

Every model provides metadata:

```python
info = model.info()
# ModelInfo(
#   name="gpt-4o",
#   provider="OpenAI",
#   model_id="gpt-4o",
#   supports_streaming=True,
#   supports_chat=True,
# )
```

This info is included in run artifacts for reproducibility.

## Providers

| Provider | Local | Streaming | Chat |
|----------|-------|-----------|------|
| OpenAI |  |  |  |
| Anthropic |  |  |  |
| Google/Gemini |  |  |  |
| Cohere |  |  |  |
| HuggingFace | / |  |  |
| Ollama |  |  |  |
| vLLM |  |  |  |
| llama.cpp |  |  |  |

## Custom Models

Implement the interface for custom integrations:

```python
from insideLLMs.models.base import Model

class MyModel(Model):
    def __init__(self, endpoint: str, **kwargs):
        super().__init__(name="my_model", **kwargs)
        self.endpoint = endpoint

    def generate(self, prompt: str, **kwargs) -> str:
        # Your API call here
        return response
```

## See Also

- [Models Catalog](../reference/Models-Catalog.md) - All available models
- [Providers and Models](../Providers-and-Models.md) - Detailed provider setup
- [Custom Probe Tutorial](../tutorials/Custom-Probe.md) - Using models in probes
