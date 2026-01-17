"""Model wrappers for various LLM providers.

This module provides a unified interface for interacting with different
LLM providers including OpenAI, Anthropic, HuggingFace, Google Gemini,
Cohere, and local models (Ollama, llama.cpp, vLLM).
"""

from typing import Any, Dict, Iterator, List

from insideLLMs.models.anthropic import AnthropicModel
from insideLLMs.models.base import AsyncModel, ChatMessage, Model, ModelProtocol, ModelWrapper
from insideLLMs.models.cohere import CohereModel
from insideLLMs.models.gemini import GeminiModel
from insideLLMs.models.huggingface import HuggingFaceModel
from insideLLMs.models.local import LlamaCppModel, OllamaModel, VLLMModel
from insideLLMs.models.openai import OpenAIModel
from insideLLMs.types import ModelInfo


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

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> str:
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


__all__ = [
    # Base classes
    "Model",
    "AsyncModel",
    "ModelProtocol",
    "ModelWrapper",
    "ChatMessage",
    # API Provider implementations
    "OpenAIModel",
    "AnthropicModel",
    "HuggingFaceModel",
    "GeminiModel",
    "CohereModel",
    # Local model implementations
    "LlamaCppModel",
    "OllamaModel",
    "VLLMModel",
    # Testing
    "DummyModel",
]
