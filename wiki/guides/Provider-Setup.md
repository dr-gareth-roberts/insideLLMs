# Provider Setup Guide

This guide walks you through setting up different model providers with insideLLMs.

## Quick Start

**For offline testing (no API keys needed):**
```python
from insideLLMs import DummyModel

model = DummyModel()
response = model.generate("Hello, world!")
print(response)
```

**For production use**, configure one or more of the providers below.

---

## OpenAI

### Prerequisites
- OpenAI account: [https://platform.openai.com/signup](https://platform.openai.com/signup)
- API key from: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### Setup

**1. Install dependencies:**
```bash
pip install "insidellms[providers]"
# Or if already installed:
pip install openai
```

**2. Set environment variable:**
```bash
export OPENAI_API_KEY=your_api_key_here
```

**Or in Python:**
```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

**3. Use in code:**
```python
from insideLLMs import OpenAIModel

# With default model (gpt-4o-mini)
model = OpenAIModel()

# With specific model
model = OpenAIModel(model_name="gpt-4o")

# With custom temperature
model = OpenAIModel(
    model_name="gpt-4o",
    temperature=0.7,
    max_tokens=1000
)

# Generate response
response = model.generate("What is 2+2?")
print(response)
```

### Available Models
- `gpt-4o` - Latest GPT-4 Omni model
- `gpt-4o-mini` - Cost-effective GPT-4 Omni (default)
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-3.5-turbo` - Legacy GPT-3.5

### Rate Limits
- Free tier: Very limited (for testing only)
- Paid tier: Varies by model (see [OpenAI pricing](https://openai.com/pricing))
- Use `insideLLMs.rate_limiting` for automatic rate limit handling

### Troubleshooting

**Error: "Incorrect API key provided"**
- Check your API key is correct
- Ensure `OPENAI_API_KEY` environment variable is set
- Verify your API key hasn't been revoked

**Error: "Rate limit exceeded"**
```python
from insideLLMs import OpenAIModel, RateLimiter

rate_limiter = RateLimiter(requests_per_minute=10)
model = OpenAIModel(rate_limiter=rate_limiter)
```

**Error: "Insufficient quota"**
- Add credits to your OpenAI account
- Check your usage at [https://platform.openai.com/usage](https://platform.openai.com/usage)

---

## Anthropic (Claude)

### Prerequisites
- Anthropic account: [https://console.anthropic.com/](https://console.anthropic.com/)
- API key from: [https://console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)

### Setup

**1. Install dependencies:**
```bash
pip install "insidellms[providers]"
# Or:
pip install anthropic
```

**2. Set environment variable:**
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

**3. Use in code:**
```python
from insideLLMs import AnthropicModel

# With default model (claude-3-5-sonnet)
model = AnthropicModel()

# With specific model
model = AnthropicModel(model_name="claude-3-opus-20240229")

# With custom parameters
model = AnthropicModel(
    model_name="claude-3-5-sonnet-20241022",
    temperature=0.7,
    max_tokens=2000
)

response = model.generate("Explain quantum computing")
print(response)
```

### Available Models
- `claude-3-5-sonnet-20241022` - Latest Claude 3.5 Sonnet (default)
- `claude-3-opus-20240229` - Most capable Claude 3 model
- `claude-3-sonnet-20240229` - Balanced performance
- `claude-3-haiku-20240307` - Fastest, most cost-effective

### Rate Limits
- Varies by tier and model
- See [Anthropic pricing](https://www.anthropic.com/pricing)

### Troubleshooting

**Error: "Invalid API key"**
- Verify your API key at [https://console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
- Ensure it starts with `sk-ant-`

**Error: "Model not found"**
- Check you're using the full model identifier (e.g., `claude-3-5-sonnet-20241022`)
- Some models require specific access permissions

---

## Ollama (Local Models)

### Prerequisites
- Ollama installed: [https://ollama.ai/](https://ollama.ai/)
- Or Ollama Cloud account for hosted models

### Setup (Local)

**1. Install Ollama:**
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai/download
```

**2. Pull a model:**
```bash
ollama pull llama3.2
ollama pull mistral
ollama pull qwen2.5
```

**3. Start Ollama server:**
```bash
ollama serve
# Server runs on http://localhost:11434 by default
```

**4. Use in code:**
```python
from insideLLMs import OllamaModel

# Local Ollama instance
model = OllamaModel(
    model_name="llama3.2",
    base_url="http://localhost:11434"
)

response = model.generate("Write a haiku about coding")
print(response)
```

### Setup (Ollama Cloud)

**1. Get API key from Ollama Cloud**

**2. Set environment variable:**
```bash
export OLLAMA_API_KEY=your_api_key_here
```

**3. Use in code:**
```python
from insideLLMs import OllamaModel

model = OllamaModel(
    model_name="deepseek-r1:7b",
    base_url="https://api.ollama.cloud",
    api_key=os.getenv("OLLAMA_API_KEY")
)

response = model.generate("Explain machine learning")
print(response)
```

### Popular Models
- `llama3.2` - Meta's Llama 3.2
- `mistral` - Mistral AI's flagship model
- `qwen2.5` - Alibaba's Qwen 2.5
- `deepseek-r1:7b` - DeepSeek reasoning model

### Troubleshooting

**Error: "Connection refused"**
- Ensure Ollama server is running: `ollama serve`
- Check server URL (default: `http://localhost:11434`)

**Error: "Model not found"**
- Pull the model first: `ollama pull <model_name>`
- List available models: `ollama list`

---

## HuggingFace

### Prerequisites
- HuggingFace account (optional, for gated models)
- HuggingFace token (optional): [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Setup

**1. Install dependencies:**
```bash
pip install "insidellms[huggingface]"
# Or:
pip install transformers torch
```

**2. (Optional) Set HF token for gated models:**
```bash
export HF_TOKEN=your_token_here
```

**3. Use in code:**
```python
from insideLLMs import HuggingFaceModel

# Small model for testing
model = HuggingFaceModel(model_name="gpt2")

# Larger instruction-tuned model
model = HuggingFaceModel(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    device="cuda",  # or "cpu"
    max_length=512
)

response = model.generate("What is Python?", max_length=100)
print(response)
```

### Popular Models
- `gpt2` - Small, fast, good for testing
- `meta-llama/Llama-3.2-3B-Instruct` - Llama 3.2 3B (requires acceptance)
- `google/flan-t5-base` - T5-based instruction model
- `mistralai/Mistral-7B-Instruct-v0.2` - Mistral 7B

### System Requirements
- **CPU models**: 8GB+ RAM recommended
- **GPU models**: CUDA-capable GPU, 8GB+ VRAM for 7B models
- Use `device="cpu"` for CPU-only inference

### Troubleshooting

**Error: "Model requires authentication"**
- Create HuggingFace account
- Accept model license on model page
- Set `HF_TOKEN` environment variable

**Error: "CUDA out of memory"**
```python
# Use smaller model or CPU
model = HuggingFaceModel(model_name="gpt2", device="cpu")

# Or use quantization (requires bitsandbytes)
model = HuggingFaceModel(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    load_in_8bit=True
)
```

---

## OpenRouter

### Prerequisites
- OpenRouter account: [https://openrouter.ai/](https://openrouter.ai/)
- API key from: [https://openrouter.ai/keys](https://openrouter.ai/keys)

### Setup

**1. Set environment variable:**
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

**2. Use in code:**
```python
from insideLLMs import OpenRouterModel

model = OpenRouterModel(
    model_name="anthropic/claude-3-5-sonnet",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

response = model.generate("Summarise machine learning")
print(response)
```

### Available Models
OpenRouter provides access to 100+ models from multiple providers:
- `anthropic/claude-3-5-sonnet`
- `openai/gpt-4o`
- `google/gemini-2.0-flash-exp`
- `meta-llama/llama-3.2-90b-vision-instruct`

See full list: [https://openrouter.ai/models](https://openrouter.ai/models)

---

## Gemini (Google)

### Prerequisites
- Google Cloud account
- Gemini API key: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

### Setup

**1. Install dependencies:**
```bash
pip install "insidellms[providers]"
# Or:
pip install google-generativeai
```

**2. Set environment variable:**
```bash
export GOOGLE_API_KEY=your_api_key_here
```

**3. Use in code:**
```python
from insideLLMs import GeminiModel

model = GeminiModel(
    model_name="gemini-2.0-flash-exp",
    api_key=os.getenv("GOOGLE_API_KEY")
)

response = model.generate("Explain neural networks")
print(response)
```

### Available Models
- `gemini-2.0-flash-exp` - Latest Gemini 2.0 Flash (experimental)
- `gemini-1.5-pro` - Gemini 1.5 Pro
- `gemini-1.5-flash` - Fast and efficient

---

## Provider Comparison

| Provider | Best For | Cost | Setup Difficulty | Local Option |
|----------|----------|------|------------------|--------------|
| **OpenAI** | Production quality, reliability | $$$ | Easy | No |
| **Anthropic** | Long context, reasoning | $$$ | Easy | No |
| **Ollama** | Privacy, local inference | Free | Medium | Yes |
| **HuggingFace** | Customisation, research | Free (compute costs) | Medium-Hard | Yes |
| **OpenRouter** | Multi-model access, flexibility | Variable | Easy | No |
| **Gemini** | Google ecosystem integration | $$ | Easy | No |

---

## Multi-Provider Configuration

**Using multiple providers in a single harness:**

```python
from insideLLMs import (
    OpenAIModel,
    AnthropicModel,
    OllamaModel,
    ProbeRunner,
    LogicProbe
)

models = [
    OpenAIModel(model_name="gpt-4o-mini"),
    AnthropicModel(model_name="claude-3-5-sonnet-20241022"),
    OllamaModel(model_name="llama3.2", base_url="http://localhost:11434")
]

probe = LogicProbe()
test_data = ["What is 5 + 3?", "If A implies B, and A is true, what about B?"]

for model in models:
    print(f"\n=== Testing {model.model_name} ===")
    runner = ProbeRunner(model, probe)
    results = runner.run(test_data)
    for result in results:
        print(f"  {result}")
```

---

## Environment Variable Summary

| Provider | Variable | Example |
|----------|----------|---------|
| OpenAI | `OPENAI_API_KEY` | `sk-proj-...` |
| Anthropic | `ANTHROPIC_API_KEY` | `sk-ant-...` |
| Ollama Cloud | `OLLAMA_API_KEY` | `ollama_...` |
| HuggingFace | `HF_TOKEN` | `hf_...` |
| OpenRouter | `OPENROUTER_API_KEY` | `sk-or-...` |
| Gemini | `GOOGLE_API_KEY` | `AIza...` |

**Setting multiple at once:**
```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
export OLLAMA_API_KEY=your_ollama_key
```

**Or use a `.env` file:**
```bash
# .env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
OLLAMA_API_KEY=your_ollama_key
```

Then load with `python-dotenv`:
```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env file

# Now environment variables are set
from insideLLMs import OpenAIModel
model = OpenAIModel()  # Uses OPENAI_API_KEY from .env
```

---

## Next Steps

- Review [Model Catalog](../reference/Models-Catalog.md) for complete model reference
- See [Rate Limiting](Rate-Limiting.md) for managing API quotas
- Check [Cost Management](../advanced/Cost-Management.md) for budget tracking
- Explore [Local Models](Local-Models.md) for privacy-focused deployments
