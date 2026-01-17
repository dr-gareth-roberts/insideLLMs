"""Local LLM model implementations.

This module provides wrappers for running LLMs locally, including:
- llama-cpp-python for GGUF models (Llama, Mistral, etc.)
- Ollama for easy local model management
"""

import os
from typing import Any, Dict, Iterator, List, Optional

from insideLLMs.models.base import ChatMessage, Model
from insideLLMs.types import ModelInfo


class LlamaCppModel(Model):
    """Model implementation for local LLMs using llama-cpp-python.

    Supports GGUF model files for running Llama, Mistral, and other
    models locally without requiring an API.

    Example:
        >>> model = LlamaCppModel(
        ...     model_path="/path/to/model.gguf",
        ...     n_ctx=4096
        ... )
        >>> response = model.generate("What is Python?")
        >>> print(response)
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        model_path: str,
        name: str = "LlamaCppModel",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        seed: int = -1,
        f16_kv: bool = True,
        verbose: bool = False,
        **model_kwargs: Any,
    ):
        """Initialize the llama-cpp model.

        Args:
            model_path: Path to the GGUF model file.
            name: Human-readable name for this model instance.
            n_ctx: Context window size.
            n_gpu_layers: Number of layers to offload to GPU (-1 for all).
            seed: Random seed for reproducibility.
            f16_kv: Use 16-bit floats for key/value cache.
            verbose: Enable verbose output from llama.cpp.
            **model_kwargs: Additional arguments passed to Llama constructor.
        """
        model_name = os.path.basename(model_path)
        super().__init__(name=name, model_id=model_name)

        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.f16_kv = f16_kv
        self.verbose = verbose
        self.model_kwargs = model_kwargs

        self._model = None

    def _get_model(self):
        """Lazily initialize the llama.cpp model."""
        if self._model is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python package required. "
                    "Install with: pip install llama-cpp-python"
                )

            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                seed=self.seed,
                f16_kv=self.f16_kv,
                verbose=self.verbose,
                **self.model_kwargs,
            )
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the local model.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            The model's text response.
        """
        model = self._get_model()

        # Map parameters
        params = {
            "prompt": prompt,
            "max_tokens": kwargs.pop("max_tokens", 512),
            "temperature": kwargs.pop("temperature", 0.7),
            "top_p": kwargs.pop("top_p", 0.95),
            "top_k": kwargs.pop("top_k", 40),
            "stop": kwargs.pop("stop", None),
            "echo": kwargs.pop("echo", False),
        }
        params.update(kwargs)

        response = model(**params)

        self._call_count += 1
        return response["choices"][0]["text"]

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        """Engage in a multi-turn chat conversation.

        Args:
            messages: List of chat messages with role and content.
            **kwargs: Additional arguments for generation.

        Returns:
            The assistant's response.
        """
        model = self._get_model()

        # Convert to llama.cpp chat format
        llama_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            llama_messages.append({"role": role, "content": content})

        params = {
            "messages": llama_messages,
            "max_tokens": kwargs.pop("max_tokens", 512),
            "temperature": kwargs.pop("temperature", 0.7),
            "top_p": kwargs.pop("top_p", 0.95),
        }
        params.update(kwargs)

        response = model.create_chat_completion(**params)

        self._call_count += 1
        return response["choices"][0]["message"]["content"]

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the response from the model.

        Args:
            prompt: The input prompt.
            **kwargs: Additional arguments for generation.

        Yields:
            Chunks of the response text.
        """
        model = self._get_model()

        params = {
            "prompt": prompt,
            "max_tokens": kwargs.pop("max_tokens", 512),
            "temperature": kwargs.pop("temperature", 0.7),
            "stream": True,
        }
        params.update(kwargs)

        self._call_count += 1
        for chunk in model(**params):
            text = chunk["choices"][0]["text"]
            if text:
                yield text

    def info(self) -> ModelInfo:
        """Return model information.

        Returns:
            ModelInfo with local model details.
        """
        return ModelInfo(
            name=self.name,
            provider="llama.cpp",
            model_id=self.model_id,
            supports_streaming=True,
            supports_chat=True,
            extra={
                "model_path": self.model_path,
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "description": "Local LLM via llama-cpp-python.",
            },
        )


class OllamaModel(Model):
    """Model implementation for Ollama-managed local models.

    Provides an easy interface to locally-running models managed by Ollama.

    Example:
        >>> model = OllamaModel(model_name="llama3.2")
        >>> response = model.generate("Explain recursion")
        >>> print(response)
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        model_name: str = "llama3.2",
        name: str = "OllamaModel",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        """Initialize the Ollama model.

        Args:
            model_name: The Ollama model to use (e.g., "llama3.2", "mistral").
            name: Human-readable name for this model instance.
            base_url: Base URL for the Ollama API.
            timeout: Request timeout in seconds.
        """
        super().__init__(name=name, model_id=model_name)
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazily initialize the Ollama client."""
        if self._client is None:
            try:
                import ollama
            except ImportError:
                raise ImportError(
                    "ollama package required. Install with: pip install ollama"
                )

            self._client = ollama.Client(host=self.base_url, timeout=self.timeout)
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the Ollama model.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments (temperature, etc.)

        Returns:
            The model's text response.
        """
        client = self._get_client()

        options = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")
        if "num_predict" in kwargs or "max_tokens" in kwargs:
            options["num_predict"] = kwargs.pop(
                "num_predict", kwargs.pop("max_tokens", None)
            )
        if "top_p" in kwargs:
            options["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            options["top_k"] = kwargs.pop("top_k")
        if "seed" in kwargs:
            options["seed"] = kwargs.pop("seed")

        response = client.generate(
            model=self.model_name,
            prompt=prompt,
            options=options if options else None,
            **kwargs,
        )

        self._call_count += 1
        return response["response"]

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        """Engage in a multi-turn chat conversation.

        Args:
            messages: List of chat messages with role and content.
            **kwargs: Additional arguments for generation.

        Returns:
            The assistant's response.
        """
        client = self._get_client()

        # Convert to Ollama message format
        ollama_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            ollama_messages.append({"role": role, "content": content})

        options = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")
        if "num_predict" in kwargs or "max_tokens" in kwargs:
            options["num_predict"] = kwargs.pop(
                "num_predict", kwargs.pop("max_tokens", None)
            )

        response = client.chat(
            model=self.model_name,
            messages=ollama_messages,
            options=options if options else None,
            **kwargs,
        )

        self._call_count += 1
        return response["message"]["content"]

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the response from the model.

        Args:
            prompt: The input prompt.
            **kwargs: Additional arguments for generation.

        Yields:
            Chunks of the response text.
        """
        client = self._get_client()

        options = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")
        if "num_predict" in kwargs or "max_tokens" in kwargs:
            options["num_predict"] = kwargs.pop(
                "num_predict", kwargs.pop("max_tokens", None)
            )

        self._call_count += 1
        for chunk in client.generate(
            model=self.model_name,
            prompt=prompt,
            options=options if options else None,
            stream=True,
            **kwargs,
        ):
            if chunk.get("response"):
                yield chunk["response"]

    def info(self) -> ModelInfo:
        """Return model information.

        Returns:
            ModelInfo with Ollama model details.
        """
        return ModelInfo(
            name=self.name,
            provider="Ollama",
            model_id=self.model_id,
            supports_streaming=True,
            supports_chat=True,
            extra={
                "model_name": self.model_name,
                "base_url": self.base_url,
                "description": "Local LLM via Ollama.",
            },
        )

    def pull(self) -> None:
        """Pull/download the model if not already available."""
        client = self._get_client()
        client.pull(self.model_name)

    def list_models(self) -> List[str]:
        """List available models in Ollama.

        Returns:
            List of model names.
        """
        client = self._get_client()
        models = client.list()
        return [m["name"] for m in models.get("models", [])]

    def show_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model.

        Returns:
            Dictionary with model details.
        """
        client = self._get_client()
        return client.show(self.model_name)


class VLLMModel(Model):
    """Model implementation for vLLM server.

    Connects to a vLLM server for high-performance inference.

    Example:
        >>> model = VLLMModel(
        ...     model_name="meta-llama/Llama-3.1-8B-Instruct",
        ...     base_url="http://localhost:8000"
        ... )
        >>> response = model.generate("Hello!")
        >>> print(response)
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        model_name: str,
        name: str = "VLLMModel",
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
    ):
        """Initialize the vLLM model.

        Args:
            model_name: The model name (for reference).
            name: Human-readable name for this model instance.
            base_url: Base URL for the vLLM server.
            api_key: Optional API key for authentication.
        """
        super().__init__(name=name, model_id=model_name)
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazily initialize the OpenAI-compatible client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )

            self._client = OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key=self.api_key or "dummy-key",
            )
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the vLLM server.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            The model's text response.
        """
        client = self._get_client()

        response = client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=kwargs.pop("max_tokens", 512),
            temperature=kwargs.pop("temperature", 0.7),
            **kwargs,
        )

        self._call_count += 1
        return response.choices[0].text

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        """Engage in a multi-turn chat conversation.

        Args:
            messages: List of chat messages with role and content.
            **kwargs: Additional arguments for generation.

        Returns:
            The assistant's response.
        """
        client = self._get_client()

        # Convert to OpenAI format
        openai_messages = []
        for msg in messages:
            openai_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        response = client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            max_tokens=kwargs.pop("max_tokens", 512),
            temperature=kwargs.pop("temperature", 0.7),
            **kwargs,
        )

        self._call_count += 1
        return response.choices[0].message.content

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the response from the model.

        Args:
            prompt: The input prompt.
            **kwargs: Additional arguments for generation.

        Yields:
            Chunks of the response text.
        """
        client = self._get_client()

        self._call_count += 1
        stream = client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=kwargs.pop("max_tokens", 512),
            temperature=kwargs.pop("temperature", 0.7),
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            if chunk.choices[0].text:
                yield chunk.choices[0].text

    def info(self) -> ModelInfo:
        """Return model information.

        Returns:
            ModelInfo with vLLM model details.
        """
        return ModelInfo(
            name=self.name,
            provider="vLLM",
            model_id=self.model_id,
            supports_streaming=True,
            supports_chat=True,
            extra={
                "model_name": self.model_name,
                "base_url": self.base_url,
                "description": "High-performance inference via vLLM server.",
            },
        )
