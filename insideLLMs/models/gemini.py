"""Google Gemini model implementation.

This module provides a wrapper for Google's Gemini models via the
google-generativeai SDK.
"""

import os
from typing import Any, Dict, Iterator, List, Optional

from insideLLMs.models.base import ChatMessage, Model
from insideLLMs.types import ModelInfo


class GeminiModel(Model):
    """Model implementation for Google's Gemini models via API.

    Supports both text generation and multi-turn chat conversations.

    Example:
        >>> model = GeminiModel(model_name="gemini-pro")
        >>> response = model.generate("What is machine learning?")
        >>> print(response)
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        name: str = "GeminiModel",
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Gemini model.

        Args:
            name: Human-readable name for this model instance.
            model_name: The Gemini model to use (e.g., "gemini-pro", "gemini-1.5-flash").
            api_key: Google AI API key. If not provided, uses GOOGLE_API_KEY env var.
            safety_settings: Optional safety settings to apply to all requests.
            generation_config: Optional default generation configuration.
        """
        super().__init__(name=name, model_id=model_name)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.safety_settings = safety_settings
        self.default_generation_config = generation_config or {}

        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._client = None
        self._model = None

    def _get_client(self):
        """Lazily initialize the Google AI client."""
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package required. "
                    "Install with: pip install google-generativeai"
                )

            genai.configure(api_key=self.api_key)
            self._client = genai
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=self.safety_settings,
                generation_config=self.default_generation_config,
            )
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the Gemini model.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments (temperature, max_output_tokens, etc.)

        Returns:
            The model's text response.
        """
        model = self._get_client()

        generation_config = {**self.default_generation_config}
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            generation_config["max_output_tokens"] = kwargs.pop("max_tokens")
        if "max_output_tokens" in kwargs:
            generation_config["max_output_tokens"] = kwargs.pop("max_output_tokens")
        if "top_p" in kwargs:
            generation_config["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            generation_config["top_k"] = kwargs.pop("top_k")

        response = model.generate_content(
            prompt,
            generation_config=generation_config if generation_config else None,
            **kwargs,
        )

        self._call_count += 1
        return response.text

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        """Engage in a multi-turn chat conversation.

        Args:
            messages: List of chat messages with role and content.
            **kwargs: Additional arguments for generation.

        Returns:
            The assistant's response.
        """
        model = self._get_client()

        # Convert to Gemini's chat format
        history = []
        current_message = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Prepend system message to first user message
                if history or current_message:
                    continue
                current_message = f"[System: {content}]\n"
            elif role == "user":
                if current_message:
                    content = current_message + content
                    current_message = None
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})

        # Start or continue chat
        chat = model.start_chat(history=history[:-1] if len(history) > 1 else [])

        # Send the last user message
        last_user_msg = history[-1]["parts"][0] if history else ""

        generation_config = {**self.default_generation_config}
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            generation_config["max_output_tokens"] = kwargs.pop("max_tokens")

        response = chat.send_message(
            last_user_msg,
            generation_config=generation_config if generation_config else None,
            **kwargs,
        )

        self._call_count += 1
        return response.text

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the response from the model.

        Args:
            prompt: The input prompt.
            **kwargs: Additional arguments for generation.

        Yields:
            Chunks of the response text.
        """
        model = self._get_client()

        generation_config = {**self.default_generation_config}
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            generation_config["max_output_tokens"] = kwargs.pop("max_tokens")

        response = model.generate_content(
            prompt,
            generation_config=generation_config if generation_config else None,
            stream=True,
            **kwargs,
        )

        self._call_count += 1
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def info(self) -> ModelInfo:
        """Return model information.

        Returns:
            ModelInfo with Gemini model details.
        """
        return ModelInfo(
            name=self.name,
            provider="Google",
            model_id=self.model_id,
            supports_streaming=True,
            supports_chat=True,
            extra={
                "model_name": self.model_name,
                "description": "Google Gemini model via google-generativeai SDK.",
            },
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens.
        """
        model = self._get_client()
        result = model.count_tokens(text)
        return result.total_tokens

    def list_models(self) -> List[str]:
        """List available Gemini models.

        Returns:
            List of model names.
        """
        self._get_client()  # Ensure client is initialized
        import google.generativeai as genai

        models = genai.list_models()
        return [
            m.name
            for m in models
            if "generateContent" in m.supported_generation_methods
        ]
