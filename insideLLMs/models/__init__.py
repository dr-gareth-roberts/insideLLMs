"""Model wrappers for various LLM providers.

This module provides a unified interface for interacting with different
LLM providers including OpenAI, Anthropic, HuggingFace, Google Gemini,
Cohere, and local models (Ollama, llama.cpp, vLLM).

Heavy dependencies (like HuggingFace transformers) are lazily loaded
to keep import times fast. This means you can import from this package
without incurring the cost of loading large ML frameworks until you
actually instantiate a model that requires them.

Available Models
----------------
The following model implementations are available:

**Cloud API Providers:**

OpenAIModel
    Wrapper for OpenAI's GPT models (GPT-4, GPT-4-turbo, GPT-3.5-turbo).
    Supports text generation, chat, and streaming.
    Requires: ``openai>=1.0.0``, ``OPENAI_API_KEY`` environment variable.

AnthropicModel
    Wrapper for Anthropic's Claude models (Claude 3 Opus, Sonnet, Haiku).
    Supports text generation, chat, and streaming.
    Requires: ``anthropic``, ``ANTHROPIC_API_KEY`` environment variable.

HuggingFaceModel
    Wrapper for HuggingFace Transformers models for local inference.
    Supports any model compatible with the transformers library.
    Requires: ``transformers``, ``torch`` or ``tensorflow``.

GeminiModel
    Wrapper for Google's Gemini models (Gemini Pro, Gemini Ultra).
    Supports text generation and chat.
    Requires: ``google-generativeai``, ``GOOGLE_API_KEY`` environment variable.

CohereModel
    Wrapper for Cohere's language models (Command, Command-Light).
    Supports text generation and embeddings.
    Requires: ``cohere``, ``COHERE_API_KEY`` environment variable.

**Local Model Runners:**

LlamaCppModel
    Wrapper for llama.cpp via llama-cpp-python for running GGUF models locally.
    Supports Llama, Mistral, Phi, and other GGUF-format models.
    Ideal for complete data privacy and offline usage.
    Requires: ``llama-cpp-python``.

OllamaModel
    Wrapper for Ollama for easy local model management and inference.
    Supports pulling, running, and managing models via Ollama.
    Requires: Ollama installed and running (``ollama serve``).

VLLMModel
    Wrapper for vLLM high-performance inference server.
    Ideal for high-throughput production inference.
    Requires: vLLM server running with desired model.

**Base Classes and Utilities:**

Model
    Abstract base class for all model implementations. Defines the core
    interface (generate, chat, stream, batch_generate).

AsyncModel
    Base class for models with async support. Adds agenerate, achat,
    astream, and abatch_generate methods.

ModelWrapper
    Decorator class adding retry logic and optional caching to any model.

ChatMessage
    TypedDict for representing messages in multi-turn conversations.

ModelProtocol
    Protocol for type hints when accepting any model-like object.

BatchModelProtocol
    Protocol for models supporting batch generation.

**Testing:**

DummyModel
    A simple model for testing that echoes prompts or returns canned responses.
    No external dependencies required.

Examples
--------
**Basic Import and Usage:**

Import specific models directly:

    >>> from insideLLMs.models import OpenAIModel, AnthropicModel
    >>>
    >>> # Create an OpenAI model
    >>> openai_model = OpenAIModel(model_name="gpt-4")
    >>> response = openai_model.generate("What is Python?")
    >>> print(response)

**Using Multiple Providers:**

Switch between providers with a consistent interface:

    >>> from insideLLMs.models import OpenAIModel, AnthropicModel, GeminiModel
    >>>
    >>> # All models share the same interface
    >>> models = [
    ...     OpenAIModel(model_name="gpt-4"),
    ...     AnthropicModel(model_name="claude-3-opus-20240229"),
    ...     GeminiModel(model_name="gemini-pro"),
    ... ]
    >>>
    >>> prompt = "Explain machine learning in one sentence."
    >>> for model in models:
    ...     print(f"{model.name}: {model.generate(prompt)}")

**Multi-Turn Chat Conversations:**

Use chat mode for conversational interactions:

    >>> from insideLLMs.models import OpenAIModel
    >>>
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> messages = [
    ...     {"role": "system", "content": "You are a helpful coding assistant."},
    ...     {"role": "user", "content": "How do I read a file in Python?"},
    ... ]
    >>> response = model.chat(messages)
    >>> print(response)
    >>>
    >>> # Continue the conversation
    >>> messages.append({"role": "assistant", "content": response})
    >>> messages.append({"role": "user", "content": "How about writing to a file?"})
    >>> response = model.chat(messages)

**Streaming Responses:**

Stream responses for real-time display:

    >>> from insideLLMs.models import AnthropicModel
    >>>
    >>> model = AnthropicModel(model_name="claude-3-sonnet-20240229")
    >>> for chunk in model.stream("Write a haiku about programming"):
    ...     print(chunk, end="", flush=True)
    >>> print()  # Final newline

**Local Model Usage with Ollama:**

Run models locally for privacy and offline usage:

    >>> from insideLLMs.models import OllamaModel
    >>>
    >>> model = OllamaModel(model_name="llama3.2")
    >>> response = model.generate("What are the benefits of local LLMs?")
    >>> print(response)

**Local Model Usage with llama.cpp:**

Run GGUF models directly:

    >>> from insideLLMs.models import LlamaCppModel
    >>>
    >>> model = LlamaCppModel(
    ...     model_path="/path/to/llama-3-8b.Q4_K_M.gguf",
    ...     n_ctx=4096,
    ...     n_gpu_layers=-1  # Use all available GPU layers
    ... )
    >>> response = model.generate("Explain quantum computing.")

**Batch Processing:**

Process multiple prompts efficiently:

    >>> from insideLLMs.models import OpenAIModel
    >>>
    >>> model = OpenAIModel(model_name="gpt-3.5-turbo")
    >>> prompts = [
    ...     "What is the capital of France?",
    ...     "What is the capital of Japan?",
    ...     "What is the capital of Brazil?",
    ... ]
    >>> responses = model.batch_generate(prompts)
    >>> for prompt, response in zip(prompts, responses):
    ...     print(f"Q: {prompt}")
    ...     print(f"A: {response}")

**Adding Retry Logic with ModelWrapper:**

Make models more robust for production:

    >>> from insideLLMs.models import OpenAIModel
    >>> from insideLLMs.models.base import ModelWrapper
    >>>
    >>> base_model = OpenAIModel(model_name="gpt-4")
    >>> robust_model = ModelWrapper(
    ...     base_model,
    ...     max_retries=5,
    ...     retry_delay=2.0,
    ...     cache_responses=True  # Cache identical requests
    ... )
    >>> response = robust_model.generate("Hello!")

**Using the DummyModel for Testing:**

Test code without making real API calls:

    >>> from insideLLMs.models import DummyModel
    >>>
    >>> # Echo mode (default)
    >>> model = DummyModel()
    >>> print(model.generate("Test prompt"))
    [DummyModel] You said: Test prompt
    >>>
    >>> # Canned response mode
    >>> model = DummyModel(canned_response="Fixed response for testing")
    >>> print(model.generate("Any prompt"))
    Fixed response for testing

**Type Hints with Protocols:**

Use protocols for flexible type annotations:

    >>> from insideLLMs.models.base import ModelProtocol, ChatModelProtocol
    >>>
    >>> def evaluate_model(model: ModelProtocol, prompts: list[str]) -> list[str]:
    ...     '''Works with any model implementing ModelProtocol.'''
    ...     return [model.generate(p) for p in prompts]
    >>>
    >>> def run_chat(model: ChatModelProtocol, messages: list[dict]) -> str:
    ...     '''Requires a model that supports chat.'''
    ...     return model.chat(messages)

**Checking Model Capabilities:**

Inspect what a model supports:

    >>> from insideLLMs.models import OpenAIModel
    >>>
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> info = model.info()
    >>> print(f"Provider: {info.provider}")
    Provider: OpenAI
    >>> print(f"Supports streaming: {info.supports_streaming}")
    Supports streaming: True
    >>> print(f"Supports chat: {info.supports_chat}")
    Supports chat: True

**Async Operations (with AsyncModel):**

Use async for concurrent processing:

    >>> import asyncio
    >>> from insideLLMs.models import OpenAIModel  # If it extends AsyncModel
    >>>
    >>> async def process_many(model, prompts):
    ...     '''Process prompts concurrently.'''
    ...     results = await asyncio.gather(*[
    ...         model.agenerate(p) for p in prompts
    ...     ])
    ...     return results
    >>>
    >>> # Run with asyncio
    >>> # results = asyncio.run(process_many(model, my_prompts))

**Using HuggingFace Models Locally:**

Run transformer models without API calls:

    >>> from insideLLMs.models import HuggingFaceModel
    >>>
    >>> model = HuggingFaceModel(
    ...     model_name="meta-llama/Llama-2-7b-chat-hf",
    ...     device="cuda"  # or "cpu", "mps"
    ... )
    >>> response = model.generate("Hello, how are you?")

**Using vLLM for High-Throughput:**

Connect to a vLLM inference server:

    >>> from insideLLMs.models import VLLMModel
    >>>
    >>> model = VLLMModel(
    ...     model_name="meta-llama/Llama-3.1-8B-Instruct",
    ...     base_url="http://localhost:8000"
    ... )
    >>> response = model.generate("Summarize the benefits of vLLM.")

Notes
-----
- All models share a common interface defined by the Model base class
- Lazy loading is used for heavy dependencies to keep import times fast
- Environment variables are used for API keys by default (can be overridden)
- Error handling is unified through the insideLLMs.exceptions module
- Models are registered in the model registry for discovery (see insideLLMs.registry)

See Also
--------
insideLLMs.models.base : Base classes, protocols, and ModelWrapper
insideLLMs.models.openai : OpenAI model implementation details
insideLLMs.models.anthropic : Anthropic Claude model implementation details
insideLLMs.models.huggingface : HuggingFace Transformers implementation
insideLLMs.models.gemini : Google Gemini model implementation
insideLLMs.models.cohere : Cohere model implementation
insideLLMs.models.local : Local model implementations (Ollama, llama.cpp, vLLM)
insideLLMs.registry : Model registration and discovery
insideLLMs.exceptions : Exception hierarchy for error handling
"""

from collections.abc import Iterator
from typing import Any

# Import lightweight core components directly
from insideLLMs.models.base import (
    AsyncModel,
    BatchModelProtocol,
    ChatMessage,
    Model,
    ModelProtocol,
    ModelWrapper,
)
from insideLLMs.types import ModelInfo

# Note: __all__ is defined at the end of this file to include lazy-loaded models


class DummyModel(Model):
    """A simple model for testing that echoes the prompt or returns canned responses.

    This model is useful for:
    - Unit testing without API calls
    - Development and debugging
    - CI/CD pipelines

    Example:
        >>> model = DummyModel()
        >>> result = model.generate("Hello, world!")
        >>> print(result)
        "[DummyModel] You said: Hello, world!"
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        name: str = "DummyModel",
        response_prefix: str = "[DummyModel]",
        echo: bool = True,
        canned_response: str = None,
    ):
        """Initialize the dummy model.

        Args:
            name: Name for this model instance.
            response_prefix: Prefix to add to responses.
            echo: Whether to echo the input in the response.
            canned_response: If set, always return this response instead of echoing.
        """
        super().__init__(name=name, model_id="dummy-v1")
        self.response_prefix = response_prefix
        self.echo = echo
        self.canned_response = canned_response

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response (echo or canned).

        Args:
            prompt: The input prompt.
            **kwargs: Ignored.

        Returns:
            The echoed or canned response.
        """
        if self.canned_response:
            return self.canned_response
        return f"{self.response_prefix} You said: {prompt}"

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Simulate a chat response.

        Args:
            messages: List of chat messages.
            **kwargs: Ignored.

        Returns:
            A response based on the last message.
        """
        last_message = messages[-1]["content"] if messages else ""
        if self.canned_response:
            return self.canned_response
        return f"{self.response_prefix} Last message: {last_message}"

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the response word by word.

        Args:
            prompt: The input prompt.
            **kwargs: Ignored.

        Yields:
            Words from the response one at a time.
        """
        response = self.generate(prompt)
        for word in response.split():
            yield word + " "

    def info(self) -> ModelInfo:
        """Return model information.

        Returns:
            ModelInfo with dummy model details.
        """
        return ModelInfo(
            name=self.name,
            provider="Dummy",
            model_id=self.model_id,
            supports_streaming=True,
            supports_chat=True,
            extra={"description": "A dummy model for testing purposes."},
        )


# Lazy loading for heavy model implementations
_LAZY_MODEL_IMPORTS = {
    "OpenAIModel": "insideLLMs.models.openai",
    "AnthropicModel": "insideLLMs.models.anthropic",
    "HuggingFaceModel": "insideLLMs.models.huggingface",
    "GeminiModel": "insideLLMs.models.gemini",
    "CohereModel": "insideLLMs.models.cohere",
    "LlamaCppModel": "insideLLMs.models.local",
    "OllamaModel": "insideLLMs.models.local",
    "VLLMModel": "insideLLMs.models.local",
}


def __getattr__(name: str):
    """Lazy load model classes to avoid importing heavy dependencies upfront.

    This function implements PEP 562 module-level __getattr__ for lazy loading
    of model classes. When you access a model class like ``OpenAIModel`` from
    this module, this function dynamically imports it from its actual location.

    This approach provides several benefits:

    1. Fast initial import - ``import insideLLMs.models`` is lightweight
    2. On-demand loading - Heavy dependencies (torch, transformers) only load
       when you actually use a model that requires them
    3. Transparent API - Users can import directly from ``insideLLMs.models``
       without knowing the internal module structure

    Parameters
    ----------
    name : str
        The name of the attribute being accessed (e.g., "OpenAIModel").

    Returns
    -------
    type
        The requested model class.

    Raises
    ------
    AttributeError
        If the requested name is not a valid attribute of this module.

    Examples
    --------
    These imports are equivalent but use lazy loading:

        >>> # Lazy import via __getattr__
        >>> from insideLLMs.models import OpenAIModel
        >>>
        >>> # Direct import (loads immediately)
        >>> from insideLLMs.models.openai import OpenAIModel

    The lazy approach is preferred for scripts that may use different models
    depending on configuration:

        >>> from insideLLMs.models import OpenAIModel, AnthropicModel
        >>>
        >>> # Only the chosen model's dependencies are loaded
        >>> if config.provider == "openai":
        ...     model = OpenAIModel(model_name=config.model)
        ... else:
        ...     model = AnthropicModel(model_name=config.model)

    Notes
    -----
    The following classes are lazily loaded:

    - OpenAIModel (from insideLLMs.models.openai)
    - AnthropicModel (from insideLLMs.models.anthropic)
    - HuggingFaceModel (from insideLLMs.models.huggingface)
    - GeminiModel (from insideLLMs.models.gemini)
    - CohereModel (from insideLLMs.models.cohere)
    - LlamaCppModel (from insideLLMs.models.local)
    - OllamaModel (from insideLLMs.models.local)
    - VLLMModel (from insideLLMs.models.local)
    """
    if name in _LAZY_MODEL_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_MODEL_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'insideLLMs.models' has no attribute '{name}'")


__all__ = [
    # Base classes
    "Model",
    "AsyncModel",
    "BatchModelProtocol",
    "ModelProtocol",
    "ModelWrapper",
    "ChatMessage",
    # API Provider implementations (lazy loaded)
    "OpenAIModel",
    "AnthropicModel",
    "HuggingFaceModel",
    "GeminiModel",
    "CohereModel",
    # Local model implementations (lazy loaded)
    "LlamaCppModel",
    "OllamaModel",
    "VLLMModel",
    # Testing
    "DummyModel",
]
