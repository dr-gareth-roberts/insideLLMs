---
title: Models Catalog
parent: Reference
nav_order: 4
---

# Models Catalog

Complete reference for all supported model providers.

## Overview

| Provider | Model Type | API Key Required |
|----------|------------|------------------|
| [OpenAI](#openai) | `openai` | Yes |
| [Anthropic](#anthropic) | `anthropic` | Yes |
| [Google/Gemini](#google-gemini) | `gemini` | Yes |
| [Cohere](#cohere) | `cohere` | Yes |
| [HuggingFace](#huggingface) | `huggingface` | Optional |
| [Ollama](#ollama) | `ollama` | No |
| [vLLM](#vllm) | `vllm` | No |
| [llama.cpp](#llamacpp) | `llamacpp` | No |
| [DummyModel](#dummymodel) | `dummy` | No |

---

## OpenAI

OpenAI models (GPT-4, GPT-3.5, etc.)

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_ORG_ID="org-..."  # Optional
```

### Config

```yaml
models:
  - type: openai
    args:
      model_name: gpt-4o
      temperature: 0.7
      max_tokens: 1000
```

### Available Models

| Model | Description |
|-------|-------------|
| `gpt-4o` | Latest GPT-4 Omni |
| `gpt-4o-mini` | Smaller, faster GPT-4 |
| `gpt-4-turbo` | GPT-4 Turbo |
| `gpt-3.5-turbo` | Fast and affordable |

### Python

```python
from insideLLMs.models import OpenAIModel

model = OpenAIModel(
    model_name="gpt-4o",
    temperature=0.7,
    max_tokens=1000
)

response = model.generate("Hello, world!")
```

### Common Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model_name` | str | `"gpt-4o-mini"` | Model identifier |
| `temperature` | float | `1.0` | Sampling temperature |
| `max_tokens` | int | `None` | Max response tokens |
| `top_p` | float | `1.0` | Nucleus sampling |
| `timeout` | int | `60` | Request timeout |

---

## Anthropic

Anthropic Claude models.

### Environment Variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Config

```yaml
models:
  - type: anthropic
    args:
      model_name: claude-3-5-sonnet-20241022
      max_tokens: 1000
```

### Available Models

| Model | Description |
|-------|-------------|
| `claude-3-5-sonnet-20241022` | Latest Sonnet |
| `claude-3-opus-20240229` | Most capable |
| `claude-3-haiku-20240307` | Fastest |

### Python

```python
from insideLLMs.models import AnthropicModel

model = AnthropicModel(
    model_name="claude-3-5-sonnet-20241022",
    max_tokens=1000
)
```

---

## Google Gemini

Google's Gemini models.

### Environment Variables

```bash
export GOOGLE_API_KEY="..."
```

### Config

```yaml
models:
  - type: gemini
    args:
      model_name: gemini-pro
```

### Available Models

| Model | Description |
|-------|-------------|
| `gemini-pro` | General purpose |
| `gemini-pro-vision` | Multimodal |

---

## Cohere

Cohere Command models.

### Environment Variables

```bash
export CO_API_KEY="..."
# or
export COHERE_API_KEY="..."
```

### Config

```yaml
models:
  - type: cohere
    args:
      model_name: command
```

---

## HuggingFace

HuggingFace Transformers models (local or API).

### Environment Variables

```bash
export HUGGINGFACEHUB_API_TOKEN="hf_..."  # Optional for private models
```

### Config (Local)

```yaml
models:
  - type: huggingface
    args:
      model_name: meta-llama/Llama-2-7b-chat-hf
      device: cuda  # or cpu, mps
```

### Config (API)

```yaml
models:
  - type: huggingface
    args:
      model_name: meta-llama/Llama-2-7b-chat-hf
      use_api: true
```

### Python

```python
from insideLLMs.models import HuggingFaceModel

model = HuggingFaceModel(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device="cuda"
)
```

### Common Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model_name` | str | Required | HF model identifier |
| `device` | str | `"auto"` | cuda, cpu, mps |
| `use_api` | bool | `False` | Use HF Inference API |
| `torch_dtype` | str | `"auto"` | float16, bfloat16, float32 |

---

## Ollama

Local models via Ollama.

### Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3
```

### Config

```yaml
models:
  - type: ollama
    args:
      model_name: llama3
      base_url: http://localhost:11434
```

### Available Models

Any model available via `ollama pull`:

| Model | Command |
|-------|---------|
| Llama 3 | `ollama pull llama3` |
| Mistral | `ollama pull mistral` |
| CodeLlama | `ollama pull codellama` |
| Gemma | `ollama pull gemma` |

### Python

```python
from insideLLMs.models import OllamaModel

model = OllamaModel(
    model_name="llama3",
    base_url="http://localhost:11434"
)
```

---

## vLLM

High-performance local inference with vLLM.

### Setup

```bash
pip install vllm
```

### Config

```yaml
models:
  - type: vllm
    args:
      model_name: meta-llama/Llama-2-7b-chat-hf
      tensor_parallel_size: 1
```

---

## llama.cpp

CPU-optimized local inference.

### Setup

```bash
pip install llama-cpp-python
```

### Config

```yaml
models:
  - type: llamacpp
    args:
      model_path: /path/to/model.gguf
      n_ctx: 2048
```

---

## DummyModel

Testing model that returns fixed responses.

### Config

```yaml
models:
  - type: dummy
    args:
      name: test_model
      response: "This is a test response."
```

### Python

```python
from insideLLMs.models import DummyModel

model = DummyModel(name="test", response="Fixed response")
```

### Use Cases

- **Testing**: Verify framework behavior without API costs
- **CI/CD**: Deterministic baseline runs
- **Development**: Build and debug probes

---

## Using the Registry

Get models by name:

```python
from insideLLMs.registry import model_registry, ensure_builtins_registered

ensure_builtins_registered()

# Get a model
model = model_registry.get("openai", model_name="gpt-4o")

# List available models
print(model_registry.list())
# ['openai', 'anthropic', 'gemini', 'cohere', 'huggingface', 'ollama', 'dummy', ...]
```

---

## Common Interface

All models implement the same interface:

```python
# Text generation
response = model.generate("prompt", temperature=0.7)

# Chat/multi-turn
response = model.chat([
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
])

# Streaming
for chunk in model.stream("prompt"):
    print(chunk, end="")

# Model info
info = model.info()
# {"name": "gpt-4o", "provider": "openai", "model_id": "gpt-4o", ...}
```

---

## Creating Custom Models

See [API Reference](API.md) for the full Model interface.

Basic structure:

```python
from insideLLMs.models.base import Model

class MyModel(Model):
    def __init__(self, name: str = "my_model", **kwargs):
        super().__init__(name=name, **kwargs)
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Your implementation
        return "response"
    
    def info(self) -> dict:
        return {"name": self.name, "provider": "custom"}
```
