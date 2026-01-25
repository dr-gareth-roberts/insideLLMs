# Providers and Models

insideLLMs uses a single `Model` interface (`generate`, `chat`, `stream`) across hosted providers and
local runners. Most users interact with models via the registry using `model.type` in config files.

## Supported Model Types

Use these names in configs (`model.type` or `models[].type`):

- `openai`: OpenAI API via `openai` SDK
- `anthropic`: Anthropic Claude API via `anthropic` SDK
- `gemini`: Google Gemini API via `google-generativeai` SDK
- `cohere`: Cohere API via `cohere` SDK
- `huggingface`: Local inference via `transformers` pipeline
- `ollama`: Local Ollama server via `ollama` Python client
- `llamacpp`: Local GGUF inference via `llama-cpp-python`
- `vllm`: OpenAI-compatible vLLM server (via `openai` SDK with `base_url`)
- `dummy`: Offline deterministic model for tests/CI

Discover what’s registered:

```bash
insidellms list models
insidellms info model openai
```

## Hosted Providers (API Keys)

Set provider keys via environment variables (or pass `api_key` in `model.args`):

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google/Gemini: `GOOGLE_API_KEY`
- Cohere: `CO_API_KEY` (or `COHERE_API_KEY`)

Validate your environment:

```bash
insidellms doctor
```

### Example: OpenAI (run)

```yaml
model:
  type: openai
  args:
    model_name: gpt-4o
probe:
  type: logic
  args: {}
dataset:
  format: jsonl
  path: data/questions.jsonl
  input_field: question
max_examples: 20
```

### Example: Azure / OpenAI-Compatible Endpoints

Use `base_url` in `model.args`:

```yaml
model:
  type: openai
  args:
    model_name: gpt-4o
    base_url: https://your-resource.openai.azure.com/
    api_key: ${AZURE_OPENAI_KEY}
```

## Local Models

### Dummy (offline)

`dummy` requires no dependencies or keys and is recommended for CI and quick smoke tests:

```bash
insidellms quicktest "What is 2 + 2?" --model dummy
```

### Hugging Face (transformers)

Run a local Transformers model via `transformers`:

```yaml
model:
  type: huggingface
  args:
    model_name: gpt2
    device: -1  # CPU; use 0 for GPU 0
```

### Ollama

Connect to a running Ollama server (default `http://localhost:11434`):

```yaml
model:
  type: ollama
  args:
    model_name: llama3.2
    base_url: http://localhost:11434
```

### llama.cpp (GGUF)

Run a local GGUF model via `llama-cpp-python`:

```yaml
model:
  type: llamacpp
  args:
    model_path: /models/llama-3-8b.Q4_K_M.gguf
    n_ctx: 4096
    n_gpu_layers: -1
```

### vLLM (server)

Connect to an OpenAI-compatible vLLM server:

```yaml
model:
  type: vllm
  args:
    model_name: meta-llama/Llama-3.1-8B-Instruct
    base_url: http://localhost:8000
    api_key: dummy-key  # optional; depends on your server
```

## See Also

- [Configuration](Configuration)
- [Probes and Models](Probes-and-Models)
- [Troubleshooting](Troubleshooting)
