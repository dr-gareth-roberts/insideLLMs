"""Local LLM model implementations.

This module provides wrappers for running LLMs locally, including:
- llama-cpp-python for GGUF models (Llama, Mistral, etc.)
- Ollama for easy local model management
- vLLM for high-performance inference serving

Local models offer several advantages over cloud APIs:
- Complete data privacy (no data leaves your machine)
- No usage costs after initial setup
- Full control over model parameters and behavior
- Ability to run offline

Examples
--------
Using llama-cpp-python with a GGUF model:

    >>> from insideLLMs.models.local import LlamaCppModel
    >>> model = LlamaCppModel(
    ...     model_path="/models/llama-3-8b.Q4_K_M.gguf",
    ...     n_ctx=4096,
    ...     n_gpu_layers=-1  # Use all GPU layers
    ... )
    >>> response = model.generate("Explain quantum computing in simple terms")
    >>> print(response)

Using Ollama for quick model access:

    >>> from insideLLMs.models.local import OllamaModel
    >>> model = OllamaModel(model_name="llama3.2")
    >>> # Pull the model first if needed
    >>> model.pull()
    >>> response = model.generate("What is the capital of France?")
    >>> print(response)
    Paris is the capital of France...

Using vLLM for high-throughput inference:

    >>> from insideLLMs.models.local import VLLMModel
    >>> model = VLLMModel(
    ...     model_name="meta-llama/Llama-3.1-8B-Instruct",
    ...     base_url="http://localhost:8000"
    ... )
    >>> response = model.chat([
    ...     {"role": "system", "content": "You are a helpful assistant."},
    ...     {"role": "user", "content": "Hello!"}
    ... ])
    >>> print(response)

Streaming responses for real-time output:

    >>> model = OllamaModel(model_name="mistral")
    >>> for chunk in model.stream("Write a haiku about programming"):
    ...     print(chunk, end="", flush=True)
    Code flows like water
    Bugs emerge from the shadows
    Debug brings the light
"""

import os
from collections.abc import Iterator
from typing import Any, Optional

from insideLLMs.models.base import ChatMessage, Model
from insideLLMs.types import ModelInfo


class LlamaCppModel(Model):
    """Model implementation for local LLMs using llama-cpp-python.

    Supports GGUF model files for running Llama, Mistral, Phi, and other
    models locally without requiring an API. This is ideal for complete
    data privacy and offline usage scenarios.

    The model is lazily initialized on first use, so creating an instance
    is lightweight. GPU acceleration is enabled by default when available.

    Attributes
    ----------
    model_path : str
        Path to the GGUF model file.
    n_ctx : int
        Context window size (default: 4096).
    n_gpu_layers : int
        Number of layers to offload to GPU (-1 for all).
    seed : int
        Random seed for reproducibility (-1 for random).
    f16_kv : bool
        Whether to use 16-bit floats for key/value cache.
    verbose : bool
        Enable verbose output from llama.cpp.

    Examples
    --------
    Basic text generation:

        >>> model = LlamaCppModel(
        ...     model_path="/models/llama-3-8b.Q4_K_M.gguf",
        ...     n_ctx=4096
        ... )
        >>> response = model.generate("What is Python?")
        >>> print(response)
        Python is a high-level, interpreted programming language...

    Chat conversation with message history:

        >>> model = LlamaCppModel(model_path="/models/mistral-7b.gguf")
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful coding assistant."},
        ...     {"role": "user", "content": "How do I read a file in Python?"}
        ... ]
        >>> response = model.chat(messages)
        >>> print(response)

    Streaming output for real-time display:

        >>> model = LlamaCppModel(model_path="/models/phi-3-mini.gguf")
        >>> for chunk in model.stream("Write a short poem about AI"):
        ...     print(chunk, end="", flush=True)

    GPU acceleration with custom parameters:

        >>> model = LlamaCppModel(
        ...     model_path="/models/llama-3-70b.Q4_K_M.gguf",
        ...     n_ctx=8192,
        ...     n_gpu_layers=40,  # Offload 40 layers to GPU
        ...     seed=42,          # Reproducible output
        ...     verbose=True      # Show llama.cpp logs
        ... )
        >>> response = model.generate(
        ...     "Explain machine learning",
        ...     temperature=0.5,
        ...     max_tokens=1000
        ... )

    See Also
    --------
    OllamaModel : Easier model management with automatic downloads.
    VLLMModel : Higher throughput for production serving.
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

        Creates a new LlamaCppModel instance. The actual llama.cpp model
        is lazily loaded on first use to minimize memory footprint until needed.

        Parameters
        ----------
        model_path : str
            Path to the GGUF model file. Must be a valid path to an existing
            GGUF format model file (e.g., from Hugging Face or converted).
        name : str, optional
            Human-readable name for this model instance. Defaults to "LlamaCppModel".
        n_ctx : int, optional
            Context window size in tokens. Larger values allow longer conversations
            but require more memory. Defaults to 4096.
        n_gpu_layers : int, optional
            Number of layers to offload to GPU. Use -1 to offload all layers
            (recommended for best performance). Use 0 for CPU-only inference.
            Defaults to -1.
        seed : int, optional
            Random seed for reproducibility. Use -1 for random seed.
            Defaults to -1.
        f16_kv : bool, optional
            Use 16-bit floats for key/value cache. Reduces memory usage
            with minimal quality impact. Defaults to True.
        verbose : bool, optional
            Enable verbose output from llama.cpp, showing loading progress
            and inference details. Defaults to False.
        **model_kwargs : Any
            Additional keyword arguments passed directly to the Llama constructor.
            See llama-cpp-python documentation for available options.

        Raises
        ------
        ImportError
            If llama-cpp-python package is not installed (raised on first use).
        FileNotFoundError
            If model_path does not exist (raised on first use).

        Examples
        --------
        Basic initialization with default settings:

            >>> model = LlamaCppModel(model_path="/models/llama-3-8b.gguf")

        CPU-only inference (no GPU):

            >>> model = LlamaCppModel(
            ...     model_path="/models/phi-3-mini.gguf",
            ...     n_gpu_layers=0
            ... )

        Maximum context with reproducible output:

            >>> model = LlamaCppModel(
            ...     model_path="/models/mistral-7b.gguf",
            ...     n_ctx=32768,
            ...     seed=42,
            ...     name="MistralLarge"
            ... )

        Advanced configuration with additional llama.cpp options:

            >>> model = LlamaCppModel(
            ...     model_path="/models/llama-3-70b.Q4_K_M.gguf",
            ...     n_ctx=8192,
            ...     n_gpu_layers=40,
            ...     n_batch=512,        # Batch size for prompt processing
            ...     n_threads=8,        # Number of CPU threads
            ...     rope_freq_base=10000.0  # RoPE base frequency
            ... )
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
        """Lazily initialize the llama.cpp model.

        Creates and caches the Llama model instance on first call.
        Subsequent calls return the cached instance.

        Returns
        -------
        Llama
            The initialized llama-cpp-python Llama instance.

        Raises
        ------
        ImportError
            If llama-cpp-python package is not installed.
        RuntimeError
            If the model file cannot be loaded.

        Notes
        -----
        This method handles the lazy initialization pattern to avoid
        loading the model into memory until it's actually needed.
        The model remains cached for the lifetime of the instance.

        Examples
        --------
        Internal usage (typically not called directly):

            >>> model = LlamaCppModel(model_path="/models/llama.gguf")
            >>> llama = model._get_model()  # Model loaded here
            >>> llama2 = model._get_model()  # Returns cached instance
            >>> llama is llama2
            True
        """
        if self._model is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python package required. Install with: pip install llama-cpp-python"
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

        Sends a prompt to the llama.cpp model and returns the generated text.
        This is the simplest way to get a response from the model.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model. Can be any text string.
        **kwargs : Any
            Additional generation parameters:

            - max_tokens : int, default=512
                Maximum number of tokens to generate.
            - temperature : float, default=0.7
                Sampling temperature. Higher values (e.g., 1.0) make output
                more random, lower values (e.g., 0.1) make it more deterministic.
            - top_p : float, default=0.95
                Nucleus sampling probability threshold.
            - top_k : int, default=40
                Top-k sampling parameter.
            - stop : list[str] | None, default=None
                List of strings that stop generation when encountered.
            - echo : bool, default=False
                Whether to include the prompt in the output.

        Returns
        -------
        str
            The model's generated text response.

        Raises
        ------
        ImportError
            If llama-cpp-python is not installed.
        RuntimeError
            If generation fails.

        Examples
        --------
        Simple generation with defaults:

            >>> model = LlamaCppModel(model_path="/models/llama.gguf")
            >>> response = model.generate("What is 2 + 2?")
            >>> print(response)
            2 + 2 equals 4.

        Controlling temperature for creativity:

            >>> # Low temperature for factual responses
            >>> response = model.generate(
            ...     "What is the capital of Japan?",
            ...     temperature=0.1
            ... )
            >>> print(response)
            The capital of Japan is Tokyo.

            >>> # High temperature for creative writing
            >>> response = model.generate(
            ...     "Write a creative story opening:",
            ...     temperature=0.9,
            ...     max_tokens=200
            ... )

        Using stop sequences:

            >>> response = model.generate(
            ...     "List three fruits:\\n1.",
            ...     stop=["4.", "\\n\\n"],
            ...     max_tokens=100
            ... )
            >>> print(response)
             Apple
            2. Banana
            3. Orange

        Deterministic output with low temperature:

            >>> response = model.generate(
            ...     "Translate 'hello' to Spanish:",
            ...     temperature=0.0,
            ...     max_tokens=10
            ... )
            >>> print(response)
            Hola
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

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Engage in a multi-turn chat conversation.

        Sends a list of messages representing a conversation history to the
        model and returns the assistant's response. Supports system prompts,
        user messages, and assistant messages.

        Parameters
        ----------
        messages : list[ChatMessage]
            List of chat messages, each with 'role' and 'content' keys.
            Supported roles: 'system', 'user', 'assistant'.
        **kwargs : Any
            Additional generation parameters:

            - max_tokens : int, default=512
                Maximum number of tokens to generate.
            - temperature : float, default=0.7
                Sampling temperature for response variety.
            - top_p : float, default=0.95
                Nucleus sampling probability threshold.

        Returns
        -------
        str
            The assistant's response message content.

        Raises
        ------
        ImportError
            If llama-cpp-python is not installed.
        RuntimeError
            If chat completion fails.

        Examples
        --------
        Simple question and answer:

            >>> model = LlamaCppModel(model_path="/models/llama.gguf")
            >>> messages = [{"role": "user", "content": "What is Python?"}]
            >>> response = model.chat(messages)
            >>> print(response)

        With system prompt for persona:

            >>> messages = [
            ...     {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
            ...     {"role": "user", "content": "How are you today?"}
            ... ]
            >>> response = model.chat(messages)
            >>> print(response)
            Ahoy, matey! I be doin' fine on this grand day...

        Multi-turn conversation:

            >>> conversation = [
            ...     {"role": "system", "content": "You are a helpful math tutor."},
            ...     {"role": "user", "content": "What is calculus?"},
            ...     {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
            ...     {"role": "user", "content": "Can you give me an example?"}
            ... ]
            >>> response = model.chat(conversation)
            >>> print(response)

        Coding assistant example:

            >>> messages = [
            ...     {"role": "system", "content": "You are an expert Python developer."},
            ...     {"role": "user", "content": "Write a function to reverse a string."}
            ... ]
            >>> response = model.chat(messages, temperature=0.2)
            >>> print(response)
            def reverse_string(s: str) -> str:
                return s[::-1]
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

        Generates text incrementally, yielding chunks as they are produced.
        This is useful for displaying responses in real-time or for very
        long outputs where you want to start processing before completion.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        **kwargs : Any
            Additional generation parameters:

            - max_tokens : int, default=512
                Maximum number of tokens to generate.
            - temperature : float, default=0.7
                Sampling temperature for response variety.

        Yields
        ------
        str
            Chunks of the response text as they are generated.

        Raises
        ------
        ImportError
            If llama-cpp-python is not installed.
        RuntimeError
            If streaming fails.

        Examples
        --------
        Real-time display of generated text:

            >>> model = LlamaCppModel(model_path="/models/llama.gguf")
            >>> for chunk in model.stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
            Once upon a time...

        Collecting streamed output:

            >>> chunks = list(model.stream("Explain AI briefly"))
            >>> full_response = "".join(chunks)
            >>> print(full_response)

        Streaming with progress indicator:

            >>> import sys
            >>> for i, chunk in enumerate(model.stream("Write a poem")):
            ...     sys.stdout.write(chunk)
            ...     sys.stdout.flush()
            ...     if i % 10 == 0:
            ...         # Could update a progress bar here
            ...         pass

        Streaming to a file:

            >>> with open("output.txt", "w") as f:
            ...     for chunk in model.stream("Generate documentation"):
            ...         f.write(chunk)
            ...         f.flush()
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

        Provides metadata about this model instance including its configuration,
        capabilities, and identifying information.

        Returns
        -------
        ModelInfo
            A dataclass containing:
            - name: Human-readable model name
            - provider: "llama.cpp"
            - model_id: The GGUF filename
            - supports_streaming: True
            - supports_chat: True
            - extra: Dict with model_path, n_ctx, n_gpu_layers, description

        Examples
        --------
        Getting model information:

            >>> model = LlamaCppModel(
            ...     model_path="/models/llama-3-8b.gguf",
            ...     n_ctx=8192,
            ...     name="Llama3"
            ... )
            >>> info = model.info()
            >>> print(info.name)
            Llama3
            >>> print(info.provider)
            llama.cpp
            >>> print(info.extra["n_ctx"])
            8192

        Using info for logging or debugging:

            >>> info = model.info()
            >>> print(f"Using {info.name} ({info.model_id}) with {info.extra['n_ctx']} context")
            Using Llama3 (llama-3-8b.gguf) with 8192 context

        Checking capabilities:

            >>> info = model.info()
            >>> if info.supports_streaming:
            ...     for chunk in model.stream("Hello"):
            ...         print(chunk, end="")
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
    Ollama handles model downloads, updates, and lifecycle management, making
    it the simplest way to run local LLMs.

    Ollama must be installed and running (default: http://localhost:11434).
    Models are downloaded automatically on first use via the pull() method.

    Attributes
    ----------
    model_name : str
        The Ollama model identifier (e.g., "llama3.2", "mistral", "codellama").
    base_url : str
        Base URL for the Ollama API server.
    headers : dict[str, str] or None
        Optional HTTP headers to pass to the Ollama client (e.g., Authorization).
    timeout : int
        Request timeout in seconds.

    Examples
    --------
    Basic usage with default model:

        >>> model = OllamaModel(model_name="llama3.2")
        >>> response = model.generate("Explain recursion")
        >>> print(response)
        Recursion is a programming technique where a function calls itself...

    Pulling a model before first use:

        >>> model = OllamaModel(model_name="codellama:13b")
        >>> model.pull()  # Download if not present
        >>> response = model.generate("Write a Python fibonacci function")
        >>> print(response)

    Multi-turn chat conversation:

        >>> model = OllamaModel(model_name="llama3.2")
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "What's the weather like?"},
        ...     {"role": "assistant", "content": "I don't have access to weather data."},
        ...     {"role": "user", "content": "What can you help me with then?"}
        ... ]
        >>> response = model.chat(messages)

    Streaming output for interactive applications:

        >>> model = OllamaModel(model_name="mistral")
        >>> for chunk in model.stream("Write a haiku about programming"):
        ...     print(chunk, end="", flush=True)

    Listing and inspecting available models:

        >>> model = OllamaModel()
        >>> available = model.list_models()
        >>> print(available)
        ['llama3.2:latest', 'mistral:latest', 'codellama:13b']
        >>> info = model.show_model_info()
        >>> print(info['modelfile'])

    Custom Ollama server configuration:

        >>> model = OllamaModel(
        ...     model_name="llama3.2",
        ...     base_url="http://gpu-server:11434",
        ...     timeout=300
        ... )

    Ollama Cloud usage with an API key:

        >>> model = OllamaModel(
        ...     model_name="deepseek-v3.2:cloud",
        ...     base_url="https://ollama.com",
        ...     api_key=os.environ.get("OLLAMA_API_KEY"),
        ... )

    See Also
    --------
    LlamaCppModel : Direct GGUF model loading without Ollama.
    VLLMModel : High-throughput inference server.
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        model_name: str = "llama3.2",
        name: str = "OllamaModel",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        headers: Optional[dict[str, str]] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the Ollama model.

        Creates a new OllamaModel instance. The Ollama client is lazily
        initialized on first use.

        Parameters
        ----------
        model_name : str, optional
            The Ollama model identifier to use. Can include a tag
            (e.g., "llama3.2", "mistral:latest", "codellama:13b-instruct").
            Defaults to "llama3.2".
        name : str, optional
            Human-readable name for this model instance.
            Defaults to "OllamaModel".
        base_url : str, optional
            Base URL for the Ollama API server.
            Defaults to "http://localhost:11434".
        timeout : int, optional
            Request timeout in seconds. Increase for large models or
            slow hardware. Defaults to 120.
        headers : dict[str, str] or None, optional
            Optional HTTP headers to pass to the Ollama client (e.g.,
            ``{"Authorization": "Bearer ... "}``).
        api_key : str or None, optional
            Convenience API key for Ollama Cloud. If provided (or available via
            the ``OLLAMA_API_KEY`` environment variable), it is used to set the
            Authorization header unless already present in ``headers``.

        Raises
        ------
        ImportError
            If the ollama package is not installed (raised on first use).
        ConnectionError
            If the Ollama server is not reachable (raised on first use).

        Examples
        --------
        Default initialization:

            >>> model = OllamaModel()  # Uses llama3.2

        Specifying a model with tag:

            >>> model = OllamaModel(model_name="codellama:13b-python")

        Custom server configuration:

            >>> model = OllamaModel(
            ...     model_name="llama3.2",
            ...     base_url="http://192.168.1.100:11434",
            ...     timeout=300,
            ...     name="RemoteLlama"
            ... )

        Multiple model instances:

            >>> coding_model = OllamaModel(
            ...     model_name="codellama:34b",
            ...     name="CodingAssistant"
            ... )
            >>> chat_model = OllamaModel(
            ...     model_name="llama3.2",
            ...     name="ChatAssistant"
            ... )
        """
        super().__init__(name=name, model_id=model_name)
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        resolved_headers = dict(headers) if headers else {}
        resolved_api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        if resolved_api_key and "Authorization" not in resolved_headers:
            resolved_headers["Authorization"] = f"Bearer {resolved_api_key}"
        self.headers = resolved_headers or None
        self._client = None

    def _get_client(self):
        """Lazily initialize the Ollama client.

        Creates and caches the Ollama client on first call.
        Subsequent calls return the cached instance.

        Returns
        -------
        ollama.Client
            The initialized Ollama client instance.

        Raises
        ------
        ImportError
            If the ollama package is not installed.
        ConnectionError
            If the Ollama server is not reachable.

        Notes
        -----
        The client maintains a connection to the Ollama server.
        Ensure the Ollama service is running before calling this method.

        Examples
        --------
        Internal usage (typically not called directly):

            >>> model = OllamaModel(model_name="llama3.2")
            >>> client = model._get_client()  # Client created here
            >>> client2 = model._get_client()  # Returns cached instance
            >>> client is client2
            True
        """
        if self._client is None:
            try:
                import ollama
            except ImportError:
                raise ImportError("ollama package required. Install with: pip install ollama")

            client_kwargs = {"host": self.base_url, "timeout": self.timeout}
            if self.headers:
                client_kwargs["headers"] = self.headers
            self._client = ollama.Client(**client_kwargs)  # type: ignore[arg-type,assignment]
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the Ollama model.

        Sends a prompt to the Ollama model and returns the generated text.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        **kwargs : Any
            Additional generation parameters:

            - temperature : float
                Sampling temperature. Higher values increase randomness.
            - num_predict / max_tokens : int
                Maximum number of tokens to generate.
            - top_p : float
                Nucleus sampling probability threshold.
            - top_k : int
                Top-k sampling parameter.
            - seed : int
                Random seed for reproducibility.

        Returns
        -------
        str
            The model's generated text response.

        Raises
        ------
        ImportError
            If ollama package is not installed.
        ConnectionError
            If Ollama server is not reachable.
        ollama.ResponseError
            If the model is not found or generation fails.

        Examples
        --------
        Simple generation:

            >>> model = OllamaModel(model_name="llama3.2")
            >>> response = model.generate("What is machine learning?")
            >>> print(response)
            Machine learning is a subset of artificial intelligence...

        With temperature control:

            >>> # Deterministic response
            >>> response = model.generate(
            ...     "What is 2 + 2?",
            ...     temperature=0.0
            ... )
            >>> print(response)
            4

            >>> # Creative response
            >>> response = model.generate(
            ...     "Write a creative tagline for a coffee shop",
            ...     temperature=0.9
            ... )

        Limiting output length:

            >>> response = model.generate(
            ...     "Explain quantum physics",
            ...     max_tokens=100  # or num_predict=100
            ... )

        Reproducible generation:

            >>> response1 = model.generate("Hello", seed=42, temperature=0.5)
            >>> response2 = model.generate("Hello", seed=42, temperature=0.5)
            >>> response1 == response2
            True
        """
        client = self._get_client()

        options = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")
        if "num_predict" in kwargs or "max_tokens" in kwargs:
            options["num_predict"] = kwargs.pop("num_predict", kwargs.pop("max_tokens", None))
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

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Engage in a multi-turn chat conversation.

        Sends a conversation history to the model and returns the assistant's
        response. Supports system prompts, user messages, and assistant messages.

        Parameters
        ----------
        messages : list[ChatMessage]
            List of chat messages, each with 'role' and 'content' keys.
            Supported roles: 'system', 'user', 'assistant'.
        **kwargs : Any
            Additional generation parameters:

            - temperature : float
                Sampling temperature for response variety.
            - num_predict / max_tokens : int
                Maximum number of tokens to generate.

        Returns
        -------
        str
            The assistant's response message content.

        Raises
        ------
        ImportError
            If ollama package is not installed.
        ConnectionError
            If Ollama server is not reachable.

        Examples
        --------
        Simple conversation:

            >>> model = OllamaModel(model_name="llama3.2")
            >>> messages = [
            ...     {"role": "user", "content": "Hi, how are you?"}
            ... ]
            >>> response = model.chat(messages)
            >>> print(response)

        With system prompt:

            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful cooking assistant."},
            ...     {"role": "user", "content": "How do I make pasta?"}
            ... ]
            >>> response = model.chat(messages)
            >>> print(response)

        Multi-turn conversation with history:

            >>> conversation = [
            ...     {"role": "system", "content": "You are a math tutor."},
            ...     {"role": "user", "content": "What is a derivative?"},
            ...     {"role": "assistant", "content": "A derivative measures the rate of change..."},
            ...     {"role": "user", "content": "Can you give me an example?"}
            ... ]
            >>> response = model.chat(conversation, temperature=0.3)
            >>> # Add response to continue conversation
            >>> conversation.append({"role": "assistant", "content": response})

        Building an interactive chat loop:

            >>> history = [{"role": "system", "content": "You are helpful."}]
            >>> while True:
            ...     user_input = input("You: ")
            ...     if user_input.lower() == "quit":
            ...         break
            ...     history.append({"role": "user", "content": user_input})
            ...     response = model.chat(history)
            ...     print(f"Assistant: {response}")
            ...     history.append({"role": "assistant", "content": response})
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
            options["num_predict"] = kwargs.pop("num_predict", kwargs.pop("max_tokens", None))

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

        Generates text incrementally, yielding chunks as they are produced.
        Useful for real-time display in interactive applications.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        **kwargs : Any
            Additional generation parameters:

            - temperature : float
                Sampling temperature for response variety.
            - num_predict / max_tokens : int
                Maximum number of tokens to generate.

        Yields
        ------
        str
            Chunks of the response text as they are generated.

        Raises
        ------
        ImportError
            If ollama package is not installed.
        ConnectionError
            If Ollama server is not reachable.

        Examples
        --------
        Real-time display:

            >>> model = OllamaModel(model_name="llama3.2")
            >>> for chunk in model.stream("Tell me a joke"):
            ...     print(chunk, end="", flush=True)
            Why did the programmer quit his job?
            Because he didn't get arrays!

        Collecting streamed response:

            >>> chunks = []
            >>> for chunk in model.stream("Explain Python"):
            ...     chunks.append(chunk)
            ...     print(chunk, end="", flush=True)
            >>> full_response = "".join(chunks)

        Streaming with timeout handling:

            >>> import signal
            >>> def timeout_handler(signum, frame):
            ...     raise TimeoutError("Stream took too long")
            >>> signal.signal(signal.SIGALRM, timeout_handler)
            >>> signal.alarm(30)  # 30 second timeout
            >>> try:
            ...     for chunk in model.stream("Long prompt"):
            ...         print(chunk, end="")
            ... finally:
            ...     signal.alarm(0)

        Building a typing effect:

            >>> import time
            >>> for chunk in model.stream("Hello world"):
            ...     for char in chunk:
            ...         print(char, end="", flush=True)
            ...         time.sleep(0.02)  # Simulate typing
        """
        client = self._get_client()

        options = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")
        if "num_predict" in kwargs or "max_tokens" in kwargs:
            options["num_predict"] = kwargs.pop("num_predict", kwargs.pop("max_tokens", None))

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

        Provides metadata about this model instance including its configuration,
        capabilities, and identifying information.

        Returns
        -------
        ModelInfo
            A dataclass containing:
            - name: Human-readable model name
            - provider: "Ollama"
            - model_id: The Ollama model identifier
            - supports_streaming: True
            - supports_chat: True
            - extra: Dict with model_name, base_url, description

        Examples
        --------
        Getting model information:

            >>> model = OllamaModel(model_name="llama3.2", name="MyLlama")
            >>> info = model.info()
            >>> print(info.name)
            MyLlama
            >>> print(info.provider)
            Ollama
            >>> print(info.extra["base_url"])
            http://localhost:11434

        Logging model details:

            >>> info = model.info()
            >>> print(f"Connected to {info.extra['model_name']} at {info.extra['base_url']}")

        Checking capabilities before use:

            >>> info = model.info()
            >>> if info.supports_chat:
            ...     response = model.chat([{"role": "user", "content": "Hi"}])
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
        """Pull/download the model if not already available.

        Downloads the model from the Ollama library if it's not already
        present locally. This is a blocking operation that may take
        several minutes for large models.

        Raises
        ------
        ImportError
            If ollama package is not installed.
        ConnectionError
            If Ollama server is not reachable.
        ollama.ResponseError
            If the model is not found in the Ollama library.

        Notes
        -----
        Models are stored in Ollama's local cache (typically ~/.ollama/models).
        Progress is shown in the console during download.

        Examples
        --------
        Downloading a model before use:

            >>> model = OllamaModel(model_name="codellama:13b")
            >>> model.pull()  # Downloads if not present
            >>> response = model.generate("Hello")

        Setting up multiple models:

            >>> models_to_pull = ["llama3.2", "mistral", "codellama:7b"]
            >>> for model_name in models_to_pull:
            ...     model = OllamaModel(model_name=model_name)
            ...     print(f"Pulling {model_name}...")
            ...     model.pull()
            ...     print(f"{model_name} ready!")

        Checking if pull is needed:

            >>> model = OllamaModel(model_name="phi3")
            >>> available = model.list_models()
            >>> if "phi3:latest" not in available:
            ...     print("Downloading phi3...")
            ...     model.pull()
        """
        client = self._get_client()
        client.pull(self.model_name)

    def list_models(self) -> list[str]:
        """List available models in Ollama.

        Returns a list of all models currently available in the local
        Ollama installation.

        Returns
        -------
        list[str]
            List of model names with tags (e.g., "llama3.2:latest").

        Raises
        ------
        ImportError
            If ollama package is not installed.
        ConnectionError
            If Ollama server is not reachable.

        Examples
        --------
        Listing all available models:

            >>> model = OllamaModel()
            >>> available = model.list_models()
            >>> print(available)
            ['llama3.2:latest', 'mistral:latest', 'codellama:13b-instruct']

        Checking if a specific model is available:

            >>> model = OllamaModel()
            >>> models = model.list_models()
            >>> if any("codellama" in m for m in models):
            ...     print("CodeLlama is available!")
            ... else:
            ...     print("CodeLlama not found, pulling...")
            ...     OllamaModel(model_name="codellama").pull()

        Finding models by type:

            >>> model = OllamaModel()
            >>> coding_models = [m for m in model.list_models() if "code" in m.lower()]
            >>> print(f"Available coding models: {coding_models}")
        """
        client = self._get_client()
        models = client.list()
        return [m["name"] for m in models.get("models", [])]

    def show_model_info(self) -> dict[str, Any]:
        """Get detailed information about the model.

        Returns comprehensive metadata about the current model including
        its architecture, parameters, license, and template format.

        Returns
        -------
        dict[str, Any]
            Dictionary containing model details such as:
            - modelfile: The model's configuration
            - parameters: Model parameters and settings
            - template: Chat template format
            - license: Model license information
            - details: Architecture and size information

        Raises
        ------
        ImportError
            If ollama package is not installed.
        ConnectionError
            If Ollama server is not reachable.
        ollama.ResponseError
            If the model is not found.

        Examples
        --------
        Viewing model details:

            >>> model = OllamaModel(model_name="llama3.2")
            >>> info = model.show_model_info()
            >>> print(info.keys())
            dict_keys(['modelfile', 'parameters', 'template', 'details', 'license'])

        Checking model size:

            >>> model = OllamaModel(model_name="mistral")
            >>> info = model.show_model_info()
            >>> details = info.get("details", {})
            >>> print(f"Parameters: {details.get('parameter_size')}")
            Parameters: 7B

        Inspecting chat template:

            >>> model = OllamaModel(model_name="llama3.2")
            >>> info = model.show_model_info()
            >>> print(info.get("template"))
            {{ if .System }}<|start_header_id|>system<|end_header_id|>...

        Comparing model architectures:

            >>> for model_name in ["llama3.2", "mistral", "phi3"]:
            ...     model = OllamaModel(model_name=model_name)
            ...     info = model.show_model_info()
            ...     arch = info.get("details", {}).get("family", "unknown")
            ...     print(f"{model_name}: {arch}")
        """
        client = self._get_client()
        return client.show(self.model_name)


class VLLMModel(Model):
    """Model implementation for vLLM server.

    Connects to a vLLM server for high-performance inference. vLLM provides
    optimized serving with features like continuous batching, PagedAttention,
    and tensor parallelism for production-grade deployments.

    Requires a running vLLM server (typically started with `python -m vllm.entrypoints.openai.api_server`).
    Uses an OpenAI-compatible API for communication.

    Attributes
    ----------
    model_name : str
        The model identifier used by the vLLM server.
    base_url : str
        Base URL for the vLLM server (default: http://localhost:8000).
    api_key : str | None
        Optional API key for authentication.

    Examples
    --------
    Basic usage with local vLLM server:

        >>> model = VLLMModel(
        ...     model_name="meta-llama/Llama-3.1-8B-Instruct",
        ...     base_url="http://localhost:8000"
        ... )
        >>> response = model.generate("Hello!")
        >>> print(response)

    Chat conversation with vLLM:

        >>> model = VLLMModel(model_name="mistralai/Mistral-7B-Instruct-v0.2")
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Explain neural networks briefly."}
        ... ]
        >>> response = model.chat(messages)
        >>> print(response)

    Streaming for real-time output:

        >>> model = VLLMModel(model_name="meta-llama/Llama-3.1-8B-Instruct")
        >>> for chunk in model.stream("Write a short story"):
        ...     print(chunk, end="", flush=True)

    Remote vLLM server with authentication:

        >>> model = VLLMModel(
        ...     model_name="meta-llama/Llama-3.1-70B-Instruct",
        ...     base_url="https://vllm.example.com:8000",
        ...     api_key="your-api-key"
        ... )
        >>> response = model.generate("Complex question here")

    High-throughput batch processing:

        >>> model = VLLMModel(model_name="meta-llama/Llama-3.1-8B-Instruct")
        >>> prompts = ["Question 1?", "Question 2?", "Question 3?"]
        >>> responses = [model.generate(p) for p in prompts]

    See Also
    --------
    LlamaCppModel : Single-GPU inference without a server.
    OllamaModel : Simpler setup with automatic model management.
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

        Creates a new VLLMModel instance that connects to a vLLM server
        using an OpenAI-compatible API. The client is lazily initialized
        on first use.

        Parameters
        ----------
        model_name : str
            The model identifier as configured on the vLLM server.
            This should match the model loaded when starting the server.
        name : str, optional
            Human-readable name for this model instance.
            Defaults to "VLLMModel".
        base_url : str, optional
            Base URL for the vLLM server. Should include protocol and port.
            Defaults to "http://localhost:8000".
        api_key : str | None, optional
            API key for authentication. Required if the vLLM server
            is configured with authentication. Defaults to None.

        Raises
        ------
        ImportError
            If openai package is not installed (raised on first use).
        ConnectionError
            If the vLLM server is not reachable (raised on first use).

        Examples
        --------
        Basic local server connection:

            >>> model = VLLMModel(model_name="meta-llama/Llama-3.1-8B-Instruct")

        Specifying a custom port:

            >>> model = VLLMModel(
            ...     model_name="mistralai/Mistral-7B-Instruct-v0.2",
            ...     base_url="http://localhost:8080"
            ... )

        Remote server with authentication:

            >>> model = VLLMModel(
            ...     model_name="meta-llama/Llama-3.1-70B-Instruct",
            ...     base_url="https://inference.company.com:8000",
            ...     api_key="sk-your-api-key",
            ...     name="ProductionLlama"
            ... )

        Multiple model instances for different servers:

            >>> fast_model = VLLMModel(
            ...     model_name="meta-llama/Llama-3.1-8B-Instruct",
            ...     base_url="http://gpu-1:8000",
            ...     name="FastModel"
            ... )
            >>> large_model = VLLMModel(
            ...     model_name="meta-llama/Llama-3.1-70B-Instruct",
            ...     base_url="http://gpu-cluster:8000",
            ...     name="LargeModel"
            ... )
        """
        super().__init__(name=name, model_id=model_name)
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazily initialize the OpenAI-compatible client.

        Creates and caches an OpenAI client configured to connect to the
        vLLM server. Subsequent calls return the cached instance.

        Returns
        -------
        openai.OpenAI
            The initialized OpenAI client instance configured for vLLM.

        Raises
        ------
        ImportError
            If the openai package is not installed.
        ConnectionError
            If the vLLM server is not reachable.

        Notes
        -----
        vLLM exposes an OpenAI-compatible API, so we use the standard
        OpenAI Python client with a custom base URL. If no API key is
        provided, a dummy key is used since vLLM may not require one.

        Examples
        --------
        Internal usage (typically not called directly):

            >>> model = VLLMModel(model_name="llama")
            >>> client = model._get_client()  # Client created here
            >>> client2 = model._get_client()  # Returns cached instance
            >>> client is client2
            True
        """
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

            self._client = OpenAI(  # type: ignore[assignment]
                base_url=f"{self.base_url}/v1",
                api_key=self.api_key or "dummy-key",
            )
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the vLLM server.

        Sends a prompt to the vLLM server and returns the generated text.
        Uses the completions API for raw text generation.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        **kwargs : Any
            Additional generation parameters passed to the OpenAI API:

            - max_tokens : int, default=512
                Maximum number of tokens to generate.
            - temperature : float, default=0.7
                Sampling temperature. Higher values increase randomness.
            - top_p : float
                Nucleus sampling probability threshold.
            - stop : list[str]
                Stop sequences that end generation.
            - presence_penalty : float
                Penalty for token presence.
            - frequency_penalty : float
                Penalty for token frequency.

        Returns
        -------
        str
            The model's generated text response.

        Raises
        ------
        ImportError
            If openai package is not installed.
        openai.APIConnectionError
            If the vLLM server is not reachable.
        openai.APIError
            If the server returns an error.

        Examples
        --------
        Simple generation:

            >>> model = VLLMModel(model_name="meta-llama/Llama-3.1-8B-Instruct")
            >>> response = model.generate("What is Python?")
            >>> print(response)

        Controlling output length and creativity:

            >>> response = model.generate(
            ...     "Write a creative story opening",
            ...     max_tokens=200,
            ...     temperature=0.9
            ... )

        Using stop sequences:

            >>> response = model.generate(
            ...     "List the first 5 prime numbers:\\n1.",
            ...     stop=["6."],
            ...     max_tokens=100
            ... )

        Deterministic generation:

            >>> response = model.generate(
            ...     "What is 2 + 2?",
            ...     temperature=0.0,
            ...     max_tokens=10
            ... )
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

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Engage in a multi-turn chat conversation.

        Sends a conversation history to the model using the chat completions
        API and returns the assistant's response.

        Parameters
        ----------
        messages : list[ChatMessage]
            List of chat messages, each with 'role' and 'content' keys.
            Supported roles: 'system', 'user', 'assistant'.
        **kwargs : Any
            Additional generation parameters:

            - max_tokens : int, default=512
                Maximum number of tokens to generate.
            - temperature : float, default=0.7
                Sampling temperature for response variety.
            - top_p : float
                Nucleus sampling probability threshold.
            - stop : list[str]
                Stop sequences that end generation.

        Returns
        -------
        str
            The assistant's response message content.

        Raises
        ------
        ImportError
            If openai package is not installed.
        openai.APIConnectionError
            If the vLLM server is not reachable.

        Examples
        --------
        Simple conversation:

            >>> model = VLLMModel(model_name="meta-llama/Llama-3.1-8B-Instruct")
            >>> messages = [
            ...     {"role": "user", "content": "What is machine learning?"}
            ... ]
            >>> response = model.chat(messages)
            >>> print(response)

        With system prompt:

            >>> messages = [
            ...     {"role": "system", "content": "You are a Python expert."},
            ...     {"role": "user", "content": "How do I read a CSV file?"}
            ... ]
            >>> response = model.chat(messages, temperature=0.2)

        Multi-turn conversation:

            >>> conversation = [
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "What is recursion?"},
            ...     {"role": "assistant", "content": "Recursion is when a function calls itself..."},
            ...     {"role": "user", "content": "Show me an example in Python."}
            ... ]
            >>> response = model.chat(conversation)
            >>> print(response)

        Building an interactive assistant:

            >>> history = [{"role": "system", "content": "You are a code reviewer."}]
            >>> # Add user message
            >>> history.append({"role": "user", "content": "Review this: def add(a,b): return a+b"})
            >>> response = model.chat(history)
            >>> history.append({"role": "assistant", "content": response})
        """
        client = self._get_client()

        # Convert to OpenAI format
        openai_messages = []
        for msg in messages:
            openai_messages.append(
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                }
            )

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

        Generates text incrementally, yielding chunks as they are produced
        by the vLLM server. Useful for real-time display in interactive
        applications.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        **kwargs : Any
            Additional generation parameters:

            - max_tokens : int, default=512
                Maximum number of tokens to generate.
            - temperature : float, default=0.7
                Sampling temperature for response variety.

        Yields
        ------
        str
            Chunks of the response text as they are generated.

        Raises
        ------
        ImportError
            If openai package is not installed.
        openai.APIConnectionError
            If the vLLM server is not reachable.

        Examples
        --------
        Real-time display:

            >>> model = VLLMModel(model_name="meta-llama/Llama-3.1-8B-Instruct")
            >>> for chunk in model.stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)

        Collecting streamed output:

            >>> chunks = []
            >>> for chunk in model.stream("Explain vLLM"):
            ...     chunks.append(chunk)
            >>> full_response = "".join(chunks)
            >>> print(full_response)

        Streaming with progress tracking:

            >>> total_tokens = 0
            >>> for chunk in model.stream("Generate a long response"):
            ...     print(chunk, end="", flush=True)
            ...     total_tokens += len(chunk.split())
            >>> print(f"\\nGenerated approximately {total_tokens} words")

        Building a typing animation:

            >>> import time
            >>> for chunk in model.stream("Hello world", max_tokens=50):
            ...     for char in chunk:
            ...         print(char, end="", flush=True)
            ...         time.sleep(0.01)
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

        Provides metadata about this model instance including its configuration,
        capabilities, and server connection details.

        Returns
        -------
        ModelInfo
            A dataclass containing:
            - name: Human-readable model name
            - provider: "vLLM"
            - model_id: The model identifier
            - supports_streaming: True
            - supports_chat: True
            - extra: Dict with model_name, base_url, description

        Examples
        --------
        Getting model information:

            >>> model = VLLMModel(
            ...     model_name="meta-llama/Llama-3.1-8B-Instruct",
            ...     base_url="http://localhost:8000",
            ...     name="Llama8B"
            ... )
            >>> info = model.info()
            >>> print(info.name)
            Llama8B
            >>> print(info.provider)
            vLLM
            >>> print(info.extra["base_url"])
            http://localhost:8000

        Logging connection details:

            >>> info = model.info()
            >>> print(f"Connected to {info.extra['model_name']} at {info.extra['base_url']}")
            Connected to meta-llama/Llama-3.1-8B-Instruct at http://localhost:8000

        Checking capabilities:

            >>> info = model.info()
            >>> if info.supports_streaming:
            ...     for chunk in model.stream("Hello"):
            ...         print(chunk, end="")

        Comparing multiple model instances:

            >>> models = [model1, model2, model3]
            >>> for m in models:
            ...     info = m.info()
            ...     print(f"{info.name}: {info.extra['base_url']}")
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
