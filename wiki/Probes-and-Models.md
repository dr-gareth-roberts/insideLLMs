# Probes and Models

## Models

All models share a single interface (`generate`, `chat`, `stream`). You can use:

- Hosted providers (OpenAI, Anthropic, Cohere, Gemini)
- Local runners (Ollama, llama.cpp, vLLM)
- DummyModel for offline testing

For provider setup (API keys, local runners), see [Providers and Models](Providers-and-Models).

Example:

```python
from insideLLMs.models import OpenAIModel

model = OpenAIModel(model_name="gpt-4o")
print(model.generate("Hello"))
```

## Probes

Probes target a specific behaviour (logic, bias, safety, factuality, code).
They take a model and an input, and return the model output (plus optional scoring).

Example:

```python
from insideLLMs import LogicProbe, ProbeRunner

probe = LogicProbe()
runner = ProbeRunner(model, probe)
results = runner.run(["What comes next: 1, 4, 9, 16, ?"])
```

### Common Probe Types

- `logic`: basic reasoning prompts (dict keys: `problem` / `question`)
- `bias`: bias indicators (dict keys vary; usually `question`/`prompt`)
- `attack`: prompt-injection/jailbreak style prompts (dict keys: `prompt` / `attack`)
- `instruction_following`: task + constraints (dict keys: `task` / `instruction`)
- `code_generation`: code synthesis tasks (dict keys: `task` / `description`)

## Registry

Discover and construct components by name:

```python
from insideLLMs import model_registry, probe_registry

model = model_registry.get("openai", model_name="gpt-4o")
probe = probe_registry.get("logic")
```

## Custom Implementations

```python
from insideLLMs import Model, Probe

class MyModel(Model):
    def generate(self, prompt: str, **kwargs) -> str:
        return "custom response"

class MyProbe(Probe[str]):
    def run(self, model, data, **kwargs) -> str:
        return model.generate(str(data))
```

## See Also

- [Providers and Models](Providers-and-Models)
- [Configuration](Configuration)
- [Examples](Examples)
