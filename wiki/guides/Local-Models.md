---
title: Local Models
parent: Guides
nav_order: 4
---

# Local Models

Run models locally without API keys using Ollama, llama.cpp, or vLLM.

## Why Local Models?

- **Privacy**: Data never leaves your machine
- **Cost**: No per-token charges
- **Offline**: Works without internet
- **Customization**: Fine-tuned or custom models

## Ollama

The easiest way to run local models.

### Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start the server (runs in background)
ollama serve

# Pull a model
ollama pull llama3
ollama pull mistral
ollama pull codellama
```

### Config

```yaml
models:
  - type: ollama
    args:
      model_name: llama3
      base_url: http://localhost:11434
```

### Python

```python
from insideLLMs.models import OllamaModel

model = OllamaModel(
    model_name="llama3",
    base_url="http://localhost:11434"
)

response = model.generate("Hello!")
```

### Available Models

| Model | Size | Use Case |
|-------|------|----------|
| `llama3` | 8B | General purpose |
| `llama3:70b` | 70B | High quality |
| `mistral` | 7B | Fast, good quality |
| `codellama` | 7B-34B | Code generation |
| `gemma` | 2B-7B | Lightweight |

See full list: `ollama list`

### GPU Acceleration

Ollama automatically uses GPU if available:

```bash
# Check GPU usage
ollama run llama3 --verbose
```

---

## llama.cpp

CPU-optimized inference with GGUF models.

### Setup

```bash
pip install llama-cpp-python

# For GPU support (optional)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

### Download Models

Get GGUF models from HuggingFace:
- [TheBloke's models](https://huggingface.co/TheBloke)
- Look for files ending in `.gguf`

### Config

```yaml
models:
  - type: llamacpp
    args:
      model_path: /path/to/model.gguf
      n_ctx: 2048
      n_gpu_layers: 0  # 0 for CPU, -1 for all GPU
```

### Python

```python
from insideLLMs.models import LlamaCppModel

model = LlamaCppModel(
    model_path="/path/to/llama-2-7b.Q4_K_M.gguf",
    n_ctx=2048
)
```

### Quantization Levels

| Suffix | Bits | Size | Quality |
|--------|------|------|---------|
| `Q2_K` | 2 | Smallest | Lowest |
| `Q4_K_M` | 4 | Medium | Good |
| `Q5_K_M` | 5 | Larger | Better |
| `Q8_0` | 8 | Large | Best |

---

## vLLM

High-performance inference with PagedAttention.

### Setup

```bash
pip install vllm
```

Requires GPU with CUDA support.

### Config

```yaml
models:
  - type: vllm
    args:
      model_name: meta-llama/Llama-2-7b-chat-hf
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.9
```

### Python

```python
from insideLLMs.models import VLLMModel

model = VLLMModel(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1
)
```

### Multi-GPU

```yaml
models:
  - type: vllm
    args:
      model_name: meta-llama/Llama-2-70b-chat-hf
      tensor_parallel_size: 4  # Use 4 GPUs
```

---

## HuggingFace Transformers

Direct use of HuggingFace models.

### Setup

```bash
pip install transformers torch accelerate
```

### Config

```yaml
models:
  - type: huggingface
    args:
      model_name: meta-llama/Llama-2-7b-chat-hf
      device: cuda  # or cpu, mps
      torch_dtype: float16
```

### Python

```python
from insideLLMs.models import HuggingFaceModel

model = HuggingFaceModel(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device="cuda"
)
```

---

## Comparison

| Method | Setup | Speed | Memory | GPU Required |
|--------|-------|-------|--------|--------------|
| Ollama | Easy | Good | Medium | Optional |
| llama.cpp | Medium | Good | Low | Optional |
| vLLM | Complex | Best | High | Yes |
| HuggingFace | Medium | Medium | High | Recommended |

## Memory Requirements

| Model Size | RAM (CPU) | VRAM (GPU) |
|------------|-----------|------------|
| 7B | 8GB | 6GB |
| 13B | 16GB | 10GB |
| 70B | 64GB | 40GB+ |

Quantized models use less memory (Q4 ≈ 50% of above).

---

## Comparing Local to Hosted

```yaml
models:
  - type: ollama
    args:
      model_name: llama3
  - type: openai
    args:
      model_name: gpt-4o-mini

probes:
  - type: logic

dataset:
  format: jsonl
  path: data/test.jsonl
```

```bash
insidellms harness comparison.yaml
```

---

## Troubleshooting

### Ollama: "connection refused"

```bash
# Start the server
ollama serve
```

### llama.cpp: "model too large"

Use a smaller quantization or set `n_gpu_layers: -1` for GPU offloading.

### vLLM: "CUDA out of memory"

```yaml
args:
  gpu_memory_utilization: 0.7  # Lower this
```

### Slow performance

1. Enable GPU if available
2. Use quantized models
3. Reduce context length (`n_ctx`)

---

## See Also

- [Models Catalog](../reference/Models-Catalog.md) - All model configurations
- [Model Comparison Tutorial](../tutorials/Model-Comparison.md) - Compare models
