"""Static builtin adapter declarations; never import SDKs or construct clients.

These declarations describe adapter operations and prerequisites. They do not
establish live availability, credentials validity, or enforceable billing limits.
Runtime dispatch continues to use the ``can_*`` predicates in ``models.base``.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

OperationSupport = Literal["native", "simulated", "unsupported"]


@dataclass(frozen=True)
class DependencySpec:
    """An importable module and the distribution that supplies it."""

    module: str
    distribution: str


@dataclass(frozen=True)
class ProviderCapabilities:
    """Declared adapter behavior, independent of instance dispatch predicates."""

    generate: OperationSupport = "native"
    chat: OperationSupport = "native"
    stream: OperationSupport = "native"
    batch_generate: OperationSupport = "simulated"
    agenerate: OperationSupport = "unsupported"
    achat: OperationSupport = "unsupported"
    astream: OperationSupport = "unsupported"


@dataclass(frozen=True)
class ProviderSpec:
    """Immutable registration metadata, never forwarded as constructor kwargs.

    Every credential group must have at least one nonempty environment variable.
    Explicit constructor credentials and endpoint overrides are not evaluated by
    doctor. External requirements are declared, never probed.
    """

    name: str
    module: str
    class_name: str
    dependencies: tuple[DependencySpec, ...] = ()
    credential_alternatives: tuple[tuple[str, ...], ...] = ()
    optional_credentials: tuple[str, ...] = ()
    external_requirements: tuple[str, ...] = ()
    capabilities: ProviderCapabilities = ProviderCapabilities()
    budget_support: Literal["unknown"] = "unknown"


_SPECS = (
    ProviderSpec(
        "dummy",
        "insideLLMs.models",
        "DummyModel",
        capabilities=ProviderCapabilities(stream="simulated"),
    ),
    ProviderSpec(
        "openai",
        "insideLLMs.models.openai",
        "OpenAIModel",
        dependencies=(DependencySpec("openai", "openai"),),
        credential_alternatives=(("OPENAI_API_KEY",),),
        external_requirements=("Reachable configured OpenAI-compatible endpoint.",),
    ),
    ProviderSpec(
        "openrouter",
        "insideLLMs.models.openrouter",
        "OpenRouterModel",
        dependencies=(DependencySpec("openai", "openai"),),
        credential_alternatives=(("OPENROUTER_API_KEY",),),
        external_requirements=("Reachable OpenRouter endpoint.",),
    ),
    ProviderSpec(
        "anthropic",
        "insideLLMs.models.anthropic",
        "AnthropicModel",
        dependencies=(DependencySpec("anthropic", "anthropic"),),
        credential_alternatives=(("ANTHROPIC_API_KEY",),),
        external_requirements=("Reachable Anthropic endpoint.",),
    ),
    ProviderSpec(
        "gemini",
        "insideLLMs.models.gemini",
        "GeminiModel",
        dependencies=(DependencySpec("google.generativeai", "google-generativeai"),),
        credential_alternatives=(("GOOGLE_API_KEY",),),
        external_requirements=("Reachable Gemini endpoint.",),
    ),
    ProviderSpec(
        "cohere",
        "insideLLMs.models.cohere",
        "CohereModel",
        dependencies=(DependencySpec("cohere", "cohere"),),
        credential_alternatives=(("CO_API_KEY", "COHERE_API_KEY"),),
        external_requirements=("Reachable Cohere endpoint.",),
    ),
    ProviderSpec(
        "huggingface",
        "insideLLMs.models.huggingface",
        "HuggingFaceModel",
        dependencies=(
            DependencySpec("transformers", "transformers"),
            DependencySpec("torch", "torch"),
        ),
        external_requirements=("Compatible model weights, tokenizer and compute resources.",),
        capabilities=ProviderCapabilities(chat="simulated", stream="simulated"),
    ),
    ProviderSpec(
        "llamacpp",
        "insideLLMs.models.local",
        "LlamaCppModel",
        dependencies=(DependencySpec("llama_cpp", "llama-cpp-python"),),
        external_requirements=("A readable compatible GGUF model and compute resources.",),
    ),
    ProviderSpec(
        "ollama",
        "insideLLMs.models.local",
        "OllamaModel",
        dependencies=(DependencySpec("ollama", "ollama"),),
        optional_credentials=("OLLAMA_API_KEY",),
        external_requirements=("Reachable Ollama service with the requested model.",),
    ),
    ProviderSpec(
        "vllm",
        "insideLLMs.models.local",
        "VLLMModel",
        dependencies=(DependencySpec("openai", "openai"),),
        external_requirements=("Reachable vLLM endpoint with the requested model.",),
    ),
)

PROVIDER_CATALOGUE: Mapping[str, ProviderSpec] = MappingProxyType(
    {spec.name: spec for spec in _SPECS}
)

__all__ = [
    "DependencySpec",
    "OperationSupport",
    "ProviderCapabilities",
    "ProviderSpec",
    "PROVIDER_CATALOGUE",
]
