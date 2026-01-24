"""Cohere model implementation.

This module provides a wrapper for Cohere's language models via their API,
enabling text generation, multi-turn chat conversations, streaming responses,
text embeddings, and document reranking.

The CohereModel class supports Cohere's Command models (command-r, command-r-plus)
for generation tasks, as well as specialized embedding and reranking models.

Module Overview
---------------
The primary class is `CohereModel`, which wraps Cohere's API to provide:
    - Text generation with customizable parameters
    - Multi-turn chat conversations with conversation history
    - Real-time streaming of generated responses
    - Text embedding for semantic search and similarity
    - Document reranking for improved search relevance

Authentication
--------------
The module requires a Cohere API key, which can be provided:
    1. As a parameter: ``CohereModel(api_key="your-key")``
    2. Via environment variable: ``CO_API_KEY`` or ``COHERE_API_KEY``

Dependencies
------------
Requires the ``cohere`` package: ``pip install cohere``

Examples
--------
Basic text generation:

>>> from insideLLMs.models.cohere import CohereModel
>>> model = CohereModel(model_name="command-r-plus")
>>> response = model.generate("Explain the theory of relativity in simple terms")
>>> print(response)

Multi-turn chat conversation:

>>> model = CohereModel(model_name="command-r-plus")
>>> messages = [
...     {"role": "system", "content": "You are a helpful coding assistant."},
...     {"role": "user", "content": "How do I read a file in Python?"},
...     {"role": "assistant", "content": "Use open() with a context manager..."},
...     {"role": "user", "content": "What about writing to a file?"},
... ]
>>> response = model.chat(messages)
>>> print(response)

Streaming responses for real-time output:

>>> model = CohereModel(model_name="command-r")
>>> for chunk in model.stream("Write a short poem about the ocean"):
...     print(chunk, end="", flush=True)

Generating embeddings for semantic search:

>>> model = CohereModel()
>>> texts = ["Machine learning is fascinating", "Deep learning uses neural networks"]
>>> embeddings = model.embed(texts, input_type="search_document")
>>> print(f"Embedding dimension: {len(embeddings[0])}")

Reranking search results:

>>> model = CohereModel()
>>> query = "What is machine learning?"
>>> documents = [
...     "Machine learning is a subset of AI",
...     "Python is a programming language",
...     "ML algorithms learn from data",
... ]
>>> reranked = model.rerank(query, documents, top_n=2)
>>> for result in reranked:
...     print(f"{result['relevance_score']:.3f}: {result['document']}")

See Also
--------
insideLLMs.models.base.Model : Base class for all model implementations.
insideLLMs.types.ModelInfo : Data class for model metadata.

Notes
-----
Cohere offers different model tiers:
    - ``command-r-plus``: Most capable, best for complex tasks
    - ``command-r``: Fast and efficient for most use cases
    - ``embed-english-v3.0``: State-of-the-art embeddings
    - ``rerank-english-v3.0``: Document reranking model
"""

import os
from collections.abc import Iterator
from typing import Any, Optional

from insideLLMs.models.base import ChatMessage, Model
from insideLLMs.types import ModelInfo


class CohereModel(Model):
    """Model implementation for Cohere's language models via API.

    This class provides a unified interface to Cohere's AI models, supporting
    text generation, multi-turn chat conversations, streaming responses,
    text embeddings, and document reranking. It wraps the official Cohere
    Python SDK with a consistent API matching the insideLLMs Model interface.

    The class uses lazy initialization for the Cohere client, meaning the
    connection is only established when the first API call is made. This
    allows for configuration validation without requiring network access.

    Attributes
    ----------
    model_name : str
        The Cohere model identifier (e.g., "command-r-plus", "command-r").
    api_key : str
        The Cohere API key used for authentication.
    default_preamble : Optional[str]
        Default system preamble applied to all chat interactions.
    _supports_streaming : bool
        Class attribute indicating streaming support (True).
    _supports_chat : bool
        Class attribute indicating chat support (True).

    Parameters
    ----------
    name : str, optional
        Human-readable name for this model instance. Default is "CohereModel".
    model_name : str, optional
        The Cohere model to use. Default is "command-r-plus".
        Available options include:
        - "command-r-plus": Most capable model for complex tasks
        - "command-r": Fast model for general use cases
        - "command": Legacy model (deprecated)
    api_key : Optional[str], optional
        Cohere API key. If not provided, the class looks for the
        CO_API_KEY or COHERE_API_KEY environment variables.
    default_preamble : Optional[str], optional
        System preamble that sets the AI's behavior for all interactions.
        Can be overridden per-request.

    Raises
    ------
    ValueError
        If no API key is provided and none is found in environment variables.
    ImportError
        If the cohere package is not installed (raised on first API call).

    Examples
    --------
    Basic initialization and text generation:

    >>> model = CohereModel(model_name="command-r-plus")
    >>> response = model.generate("Explain quantum computing in simple terms")
    >>> print(response)

    Initialization with custom preamble for specialized behavior:

    >>> model = CohereModel(
    ...     model_name="command-r-plus",
    ...     default_preamble="You are a Python expert. Provide concise code examples."
    ... )
    >>> response = model.generate("How do I sort a dictionary by value?")
    >>> print(response)

    Using environment variables for API key:

    >>> import os
    >>> os.environ["CO_API_KEY"] = "your-api-key-here"
    >>> model = CohereModel()  # Uses env var automatically
    >>> response = model.generate("Hello, world!")

    Multi-turn conversation with chat history:

    >>> model = CohereModel(model_name="command-r-plus")
    >>> conversation = [
    ...     {"role": "system", "content": "You are a math tutor."},
    ...     {"role": "user", "content": "What is calculus?"},
    ...     {"role": "assistant", "content": "Calculus is the study of change..."},
    ...     {"role": "user", "content": "Can you give me a simple example?"},
    ... ]
    >>> response = model.chat(conversation)
    >>> print(response)

    Streaming response for real-time display:

    >>> model = CohereModel(model_name="command-r")
    >>> for chunk in model.stream("Tell me a story about a robot"):
    ...     print(chunk, end="", flush=True)
    ... print()  # Newline at end

    Generating embeddings for semantic similarity:

    >>> model = CohereModel()
    >>> embeddings = model.embed(
    ...     ["The cat sat on the mat", "A feline rested on the rug"],
    ...     input_type="search_document"
    ... )
    >>> # Compute cosine similarity between embeddings
    >>> import numpy as np
    >>> similarity = np.dot(embeddings[0], embeddings[1])
    >>> print(f"Semantic similarity: {similarity:.3f}")

    Reranking search results for better relevance:

    >>> model = CohereModel()
    >>> results = model.rerank(
    ...     query="best programming language for beginners",
    ...     documents=[
    ...         "Python is great for beginners due to simple syntax",
    ...         "C++ offers high performance for systems programming",
    ...         "JavaScript runs in web browsers",
    ...         "Python has excellent learning resources",
    ...     ],
    ...     top_n=2
    ... )
    >>> for r in results:
    ...     print(f"Score {r['relevance_score']:.3f}: {r['document'][:50]}...")

    See Also
    --------
    Model : Base class defining the model interface.
    ModelInfo : Data class for model metadata.

    Notes
    -----
    - The Cohere client is lazily initialized on first use
    - All generation methods increment the internal call counter
    - Chat history is automatically converted to Cohere's format
    - The embed and rerank methods use dedicated Cohere models
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        name: str = "CohereModel",
        model_name: str = "command-r-plus",
        api_key: Optional[str] = None,
        default_preamble: Optional[str] = None,
    ):
        """Initialize the Cohere model with configuration options.

        Creates a new CohereModel instance configured for text generation,
        chat, streaming, embeddings, and reranking. The Cohere client is
        lazily initialized on the first API call.

        Parameters
        ----------
        name : str, optional
            Human-readable name for this model instance, useful for logging
            and identification when using multiple models. Default is "CohereModel".
        model_name : str, optional
            The Cohere model identifier to use for generation. Default is
            "command-r-plus". Available options:

            - "command-r-plus": Most capable, best for complex reasoning
            - "command-r": Fast and efficient, good for most tasks
            - "command": Legacy model (deprecated, avoid for new projects)
            - "command-light": Lightweight model for simple tasks

        api_key : Optional[str], optional
            Cohere API key for authentication. If not provided, the method
            checks for the ``CO_API_KEY`` environment variable first, then
            falls back to ``COHERE_API_KEY``.
        default_preamble : Optional[str], optional
            A system message that sets the AI's behavior and persona for
            all chat interactions. This acts as persistent context that
            precedes the conversation. Can be overridden per-request.

        Raises
        ------
        ValueError
            If no API key is provided and neither ``CO_API_KEY`` nor
            ``COHERE_API_KEY`` environment variables are set.

        Examples
        --------
        Basic initialization with defaults:

        >>> model = CohereModel()
        >>> print(model.model_name)
        command-r-plus

        Specify a different model:

        >>> model = CohereModel(model_name="command-r")
        >>> response = model.generate("Hello!")

        Initialize with explicit API key:

        >>> model = CohereModel(
        ...     api_key="your-cohere-api-key",
        ...     model_name="command-r-plus"
        ... )

        Create a specialized assistant with a preamble:

        >>> model = CohereModel(
        ...     name="CodeAssistant",
        ...     model_name="command-r-plus",
        ...     default_preamble=(
        ...         "You are an expert Python developer. Always provide "
        ...         "working code examples with proper error handling. "
        ...         "Explain your code step by step."
        ...     )
        ... )
        >>> response = model.generate("How do I read a CSV file?")

        Using environment variables (recommended for security):

        >>> import os
        >>> os.environ["CO_API_KEY"] = "your-api-key"
        >>> model = CohereModel(name="MyModel")  # Key from env
        >>> print(model.api_key is not None)
        True

        Notes
        -----
        The API key is validated during initialization, but the Cohere client
        connection is not established until the first API call is made. This
        allows for fast initialization and configuration validation without
        network overhead.

        See Also
        --------
        _get_client : Lazily initializes the Cohere client.
        """
        super().__init__(name=name, model_id=model_name)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY")
        self.default_preamble = default_preamble

        if not self.api_key:
            raise ValueError(
                "Cohere API key required. Set CO_API_KEY or COHERE_API_KEY "
                "environment variable or pass api_key parameter."
            )

        self._client = None

    def _get_client(self):
        """Lazily initialize and return the Cohere client.

        Creates a Cohere client instance on first call and caches it for
        subsequent use. This lazy initialization pattern defers the network
        connection until it's actually needed, improving startup time and
        allowing configuration validation without requiring API access.

        Returns
        -------
        cohere.Client
            An initialized Cohere client configured with the instance's API key.

        Raises
        ------
        ImportError
            If the ``cohere`` package is not installed. The error message
            includes installation instructions.

        Examples
        --------
        The client is automatically retrieved when calling generation methods:

        >>> model = CohereModel(api_key="your-key")
        >>> # Client not created yet
        >>> response = model.generate("Hello")  # Client created here
        >>> # Subsequent calls reuse the same client

        Direct access to the client (advanced usage):

        >>> model = CohereModel(api_key="your-key")
        >>> client = model._get_client()
        >>> # Use client directly for unsupported features
        >>> response = client.tokenize(text="Hello world")

        Checking if the client has been initialized:

        >>> model = CohereModel(api_key="your-key")
        >>> print(model._client is None)  # True - not yet initialized
        True
        >>> _ = model.generate("test")
        >>> print(model._client is None)  # False - now initialized
        False

        Notes
        -----
        This is a private method (prefixed with underscore) intended for
        internal use. Direct access should be avoided in favor of the public
        API methods (generate, chat, stream, embed, rerank).

        The client is thread-safe once created, but the lazy initialization
        itself is not thread-safe. In multi-threaded scenarios, consider
        calling a generation method once before spawning threads.

        See Also
        --------
        __init__ : Sets up the API key used by the client.
        """
        if self._client is None:
            try:
                import cohere
            except ImportError:
                raise ImportError("cohere package required. Install with: pip install cohere")

            self._client = cohere.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a text response from the Cohere model.

        Sends a prompt to the Cohere API and returns the model's response.
        This method supports various generation parameters to control the
        output's creativity, length, and sampling behavior.

        Parameters
        ----------
        prompt : str
            The input text prompt to send to the model. This can be a question,
            instruction, or any text that the model should respond to.
        **kwargs : Any
            Additional generation parameters passed to Cohere's chat API.
            Commonly used parameters include:

            temperature : float, optional
                Controls randomness in generation. Range: 0.0 to 1.0.
                Lower values (e.g., 0.1) produce more focused, deterministic
                outputs. Higher values (e.g., 0.8) produce more creative,
                varied outputs. Default is model-specific.
            max_tokens : int, optional
                Maximum number of tokens to generate in the response.
                Useful for controlling response length and API costs.
            p : float, optional
                Nucleus sampling parameter (top-p). Range: 0.0 to 1.0.
                Only tokens with cumulative probability <= p are considered.
                Alternative name: ``top_p``.
            k : int, optional
                Top-k sampling parameter. Only the top k most likely tokens
                are considered for generation. Alternative name: ``top_k``.
            preamble : str, optional
                System message for this specific request. Overrides the
                ``default_preamble`` set during initialization.

        Returns
        -------
        str
            The model's generated text response.

        Raises
        ------
        cohere.CohereAPIError
            If the API request fails (e.g., invalid API key, rate limiting).
        ImportError
            If the cohere package is not installed.

        Examples
        --------
        Basic text generation:

        >>> model = CohereModel(model_name="command-r-plus")
        >>> response = model.generate("What is machine learning?")
        >>> print(response)

        Controlling creativity with temperature:

        >>> # Low temperature for factual, focused responses
        >>> response = model.generate(
        ...     "List the planets in our solar system",
        ...     temperature=0.1
        ... )
        >>> print(response)
        >>>
        >>> # High temperature for creative writing
        >>> response = model.generate(
        ...     "Write a creative story opening",
        ...     temperature=0.9
        ... )
        >>> print(response)

        Limiting response length:

        >>> response = model.generate(
        ...     "Explain quantum physics",
        ...     max_tokens=100  # Concise response
        ... )
        >>> print(response)

        Using a custom preamble for this request:

        >>> response = model.generate(
        ...     "Explain recursion",
        ...     preamble="You are a patient teacher explaining to a beginner."
        ... )
        >>> print(response)

        Combining multiple parameters:

        >>> response = model.generate(
        ...     "Generate a product name for a coffee shop",
        ...     temperature=0.8,
        ...     max_tokens=50,
        ...     top_p=0.9,
        ...     top_k=40
        ... )
        >>> print(response)

        Notes
        -----
        - This method increments the internal ``_call_count`` counter
        - The method uses Cohere's chat endpoint internally
        - Parameter names ``top_p`` and ``top_k`` are mapped to Cohere's
          ``p`` and ``k`` parameters automatically
        - If both ``default_preamble`` (from init) and ``preamble`` (in kwargs)
          are provided, the kwargs version takes precedence

        See Also
        --------
        chat : For multi-turn conversations with history.
        stream : For streaming responses in real-time.
        """
        client = self._get_client()

        # Map common parameter names
        params = {
            "model": self.model_name,
            "message": prompt,
        }

        if "temperature" in kwargs:
            params["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs.pop("max_tokens")
        if "p" in kwargs or "top_p" in kwargs:
            params["p"] = kwargs.pop("p", kwargs.pop("top_p", None))
        if "k" in kwargs or "top_k" in kwargs:
            params["k"] = kwargs.pop("k", kwargs.pop("top_k", None))

        # Add preamble if set
        if self.default_preamble:
            params["preamble"] = self.default_preamble
        if "preamble" in kwargs:
            params["preamble"] = kwargs.pop("preamble")

        params.update(kwargs)

        response = client.chat(**params)

        self._call_count += 1
        return response.text

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Engage in a multi-turn chat conversation with the Cohere model.

        Processes a conversation history and generates the next assistant
        response. The method automatically converts the standard ChatMessage
        format to Cohere's expected chat history format.

        The conversation flow is constructed as follows:
        1. System messages become the preamble (first system message wins)
        2. User/assistant message pairs are added to chat history
        3. The final user message becomes the current query

        Parameters
        ----------
        messages : list[ChatMessage]
            List of chat messages representing the conversation history.
            Each message is a dict with:

            - ``role`` : str - One of "system", "user", or "assistant"
            - ``content`` : str - The message text

            The last message should typically be from the user.
        **kwargs : Any
            Additional generation parameters passed to Cohere's chat API.
            Commonly used parameters include:

            temperature : float, optional
                Controls randomness. Range: 0.0 to 1.0.
            max_tokens : int, optional
                Maximum tokens to generate.

        Returns
        -------
        str
            The assistant's text response to the conversation.

        Raises
        ------
        cohere.CohereAPIError
            If the API request fails.
        ImportError
            If the cohere package is not installed.

        Examples
        --------
        Simple two-turn conversation:

        >>> model = CohereModel(model_name="command-r-plus")
        >>> messages = [
        ...     {"role": "user", "content": "What is Python?"},
        ... ]
        >>> response = model.chat(messages)
        >>> print(response)

        Multi-turn conversation with context:

        >>> messages = [
        ...     {"role": "user", "content": "My name is Alice."},
        ...     {"role": "assistant", "content": "Nice to meet you, Alice!"},
        ...     {"role": "user", "content": "What's my name?"},
        ... ]
        >>> response = model.chat(messages)
        >>> print(response)  # Should remember "Alice"

        Using a system message to set behavior:

        >>> messages = [
        ...     {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
        ...     {"role": "user", "content": "How are you today?"},
        ... ]
        >>> response = model.chat(messages)
        >>> print(response)  # Response in pirate speak

        Extended conversation with multiple turns:

        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful math tutor."},
        ...     {"role": "user", "content": "What is 2+2?"},
        ...     {"role": "assistant", "content": "2+2 equals 4."},
        ...     {"role": "user", "content": "What about 2+3?"},
        ...     {"role": "assistant", "content": "2+3 equals 5."},
        ...     {"role": "user", "content": "And 2+4?"},
        ... ]
        >>> response = model.chat(messages)
        >>> print(response)  # "2+4 equals 6."

        Building a conversation dynamically:

        >>> model = CohereModel()
        >>> conversation = []
        >>>
        >>> # Add system context
        >>> conversation.append({
        ...     "role": "system",
        ...     "content": "You are a travel advisor specializing in Europe."
        ... })
        >>>
        >>> # First turn
        >>> conversation.append({"role": "user", "content": "I want to visit Paris."})
        >>> response = model.chat(conversation)
        >>> conversation.append({"role": "assistant", "content": response})
        >>>
        >>> # Second turn
        >>> conversation.append({"role": "user", "content": "What about restaurants?"})
        >>> response = model.chat(conversation)
        >>> print(response)

        Notes
        -----
        - System messages set the preamble; if multiple system messages exist,
          the last one encountered takes precedence
        - If ``default_preamble`` was set during initialization and no system
          message is in the conversation, the default preamble is used
        - The method increments the internal ``_call_count`` counter
        - Cohere uses "USER" and "CHATBOT" roles internally; this method
          handles the conversion from "user"/"assistant" automatically

        See Also
        --------
        generate : For single-turn generation without conversation history.
        stream : For streaming responses in real-time.
        ChatMessage : Type definition for chat messages.
        """
        client = self._get_client()

        # Convert to Cohere's chat format
        chat_history = []
        preamble = self.default_preamble
        current_message = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                preamble = content
            elif role == "user":
                current_message = content
                # Don't add to history yet - only add completed turns
            elif role == "assistant" and current_message:
                # Add completed turn to history
                chat_history.append(
                    {
                        "role": "USER",
                        "message": current_message,
                    }
                )
                chat_history.append(
                    {
                        "role": "CHATBOT",
                        "message": content,
                    }
                )
                current_message = ""

        # The last user message becomes the current message
        if not current_message and messages:
            # Find last user message
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    current_message = msg.get("content", "")
                    break

        params = {
            "model": self.model_name,
            "message": current_message,
            "chat_history": chat_history if chat_history else None,
        }

        if preamble:
            params["preamble"] = preamble

        if "temperature" in kwargs:
            params["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs.pop("max_tokens")

        params.update(kwargs)

        response = client.chat(**params)

        self._call_count += 1
        return response.text

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the response from the Cohere model in real-time.

        Generates a response incrementally, yielding text chunks as they
        are produced by the model. This is useful for displaying responses
        to users in real-time, improving perceived responsiveness for
        long-form content.

        The streaming uses Cohere's chat_stream endpoint, which sends
        server-sent events (SSE) for each generated token or group of tokens.
        Only "text-generation" events are yielded; other event types
        (e.g., stream start/end) are filtered out.

        Parameters
        ----------
        prompt : str
            The input text prompt to send to the model.
        **kwargs : Any
            Additional generation parameters passed to Cohere's chat_stream API.
            Commonly used parameters include:

            temperature : float, optional
                Controls randomness in generation. Range: 0.0 to 1.0.
            max_tokens : int, optional
                Maximum number of tokens to generate.

        Yields
        ------
        str
            Chunks of the generated text as they become available.
            These can be concatenated to form the complete response.

        Raises
        ------
        cohere.CohereAPIError
            If the API request fails.
        ImportError
            If the cohere package is not installed.

        Examples
        --------
        Basic streaming with real-time display:

        >>> model = CohereModel(model_name="command-r")
        >>> for chunk in model.stream("Tell me a short story"):
        ...     print(chunk, end="", flush=True)
        >>> print()  # Newline at end

        Collecting streamed response into a string:

        >>> model = CohereModel()
        >>> chunks = []
        >>> for chunk in model.stream("Explain photosynthesis"):
        ...     chunks.append(chunk)
        >>> full_response = "".join(chunks)
        >>> print(full_response)

        Streaming with progress indicator:

        >>> model = CohereModel()
        >>> char_count = 0
        >>> for chunk in model.stream("Write a poem about coding"):
        ...     char_count += len(chunk)
        ...     print(f"\\rReceived {char_count} characters...", end="")
        >>> print(f"\\nComplete! Total: {char_count} characters")

        Using streaming with a custom preamble:

        >>> model = CohereModel(
        ...     default_preamble="You are a poet. Write in verse."
        ... )
        >>> for chunk in model.stream("Describe the ocean"):
        ...     print(chunk, end="", flush=True)

        Streaming with generation parameters:

        >>> model = CohereModel()
        >>> for chunk in model.stream(
        ...     "Generate a creative product description",
        ...     temperature=0.8,
        ...     max_tokens=200
        ... ):
        ...     print(chunk, end="", flush=True)

        Integration with web frameworks (conceptual):

        >>> # Flask/FastAPI streaming response example
        >>> def generate_stream():
        ...     model = CohereModel()
        ...     for chunk in model.stream("Generate content"):
        ...         yield chunk
        >>> # Use with streaming HTTP response

        Notes
        -----
        - This method increments the ``_call_count`` counter once at the
          start, not per chunk
        - The stream must be fully consumed or explicitly closed to avoid
          resource leaks
        - Streaming responses may have slightly higher latency for the first
          chunk compared to non-streaming, but provide better UX for long
          responses
        - Only "text-generation" events from Cohere's SSE stream are yielded;
          metadata events are filtered out

        See Also
        --------
        generate : For non-streaming single response.
        chat : For multi-turn conversations (non-streaming).
        """
        client = self._get_client()

        params = {
            "model": self.model_name,
            "message": prompt,
        }

        if "temperature" in kwargs:
            params["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs.pop("max_tokens")
        if self.default_preamble:
            params["preamble"] = self.default_preamble

        params.update(kwargs)

        response = client.chat_stream(**params)

        self._call_count += 1
        for event in response:
            if event.event_type == "text-generation":
                yield event.text

    def info(self) -> ModelInfo:
        """Return metadata about this Cohere model instance.

        Provides a structured summary of the model's configuration and
        capabilities, useful for logging, debugging, and runtime introspection.

        Returns
        -------
        ModelInfo
            A data class containing model metadata with the following fields:

            - ``name`` : str - The human-readable instance name
            - ``provider`` : str - Always "Cohere" for this implementation
            - ``model_id`` : str - The Cohere model identifier
            - ``supports_streaming`` : bool - Always True
            - ``supports_chat`` : bool - Always True
            - ``extra`` : dict - Additional metadata including:
                - ``model_name`` : The specific Cohere model being used
                - ``description`` : Brief description of the model

        Examples
        --------
        Getting basic model information:

        >>> model = CohereModel(name="MyAssistant", model_name="command-r-plus")
        >>> info = model.info()
        >>> print(f"Provider: {info.provider}")
        Provider: Cohere
        >>> print(f"Model: {info.model_id}")
        Model: command-r-plus

        Checking model capabilities:

        >>> model = CohereModel()
        >>> info = model.info()
        >>> if info.supports_streaming:
        ...     print("Streaming is available")
        >>> if info.supports_chat:
        ...     print("Multi-turn chat is available")

        Logging model configuration:

        >>> import json
        >>> model = CohereModel(name="ProductionModel", model_name="command-r")
        >>> info = model.info()
        >>> config = {
        ...     "name": info.name,
        ...     "provider": info.provider,
        ...     "model_id": info.model_id,
        ...     **info.extra
        ... }
        >>> print(json.dumps(config, indent=2))

        Comparing multiple models:

        >>> models = [
        ...     CohereModel(name="Fast", model_name="command-r"),
        ...     CohereModel(name="Capable", model_name="command-r-plus"),
        ... ]
        >>> for m in models:
        ...     info = m.info()
        ...     print(f"{info.name}: {info.model_id}")

        See Also
        --------
        ModelInfo : The data class returned by this method.
        """
        return ModelInfo(
            name=self.name,
            provider="Cohere",
            model_id=self.model_id,
            supports_streaming=True,
            supports_chat=True,
            extra={
                "model_name": self.model_name,
                "description": "Cohere language model via cohere SDK.",
            },
        )

    def embed(
        self,
        texts: list[str],
        input_type: str = "search_document",
        embedding_types: Optional[list[str]] = None,
    ) -> list[list[float]]:
        """Generate dense vector embeddings for the given texts.

        Converts text into numerical vector representations that capture
        semantic meaning. These embeddings can be used for semantic search,
        clustering, classification, and similarity comparisons.

        This method uses Cohere's embed-english-v3.0 model, which produces
        high-quality embeddings optimized for various retrieval and
        similarity tasks.

        Parameters
        ----------
        texts : list[str]
            List of text strings to embed. Each text should ideally be
            a coherent passage or document. The API has limits on the
            number of texts per request (typically 96) and text length.
        input_type : str, optional
            Specifies how the embeddings will be used, allowing the model
            to optimize for the use case. Default is "search_document".
            Options include:

            - "search_document": For texts being indexed for search
            - "search_query": For search queries to find relevant documents
            - "classification": For text classification tasks
            - "clustering": For clustering similar texts

        embedding_types : Optional[list[str]], optional
            Types of embeddings to return. If not specified, returns
            the default float embeddings. Options may include:

            - "float": Standard floating-point embeddings
            - "int8": Quantized 8-bit integer embeddings (smaller)
            - "uint8": Unsigned 8-bit integer embeddings
            - "binary": Binary embeddings (smallest, fastest comparison)

        Returns
        -------
        list[list[float]]
            A list of embedding vectors, one per input text. Each vector
            is a list of floats representing the semantic encoding of
            the text. The dimension depends on the embedding model
            (1024 for embed-english-v3.0).

        Raises
        ------
        cohere.CohereAPIError
            If the API request fails (e.g., too many texts, text too long).
        ImportError
            If the cohere package is not installed.

        Examples
        --------
        Basic embedding generation:

        >>> model = CohereModel()
        >>> texts = ["Hello world", "How are you?"]
        >>> embeddings = model.embed(texts)
        >>> print(f"Number of embeddings: {len(embeddings)}")
        Number of embeddings: 2
        >>> print(f"Embedding dimension: {len(embeddings[0])}")
        Embedding dimension: 1024

        Creating embeddings for semantic search (documents):

        >>> model = CohereModel()
        >>> documents = [
        ...     "Machine learning is a subset of artificial intelligence.",
        ...     "Python is a popular programming language.",
        ...     "Neural networks are inspired by the human brain.",
        ... ]
        >>> doc_embeddings = model.embed(documents, input_type="search_document")
        >>> # Store these embeddings in a vector database

        Creating embeddings for search queries:

        >>> query = "What is deep learning?"
        >>> query_embedding = model.embed([query], input_type="search_query")
        >>> # Use query_embedding[0] to find similar documents

        Computing semantic similarity between texts:

        >>> import numpy as np
        >>> model = CohereModel()
        >>> texts = [
        ...     "The cat sat on the mat",
        ...     "A feline rested on the rug",
        ...     "Python is great for data science"
        ... ]
        >>> embeddings = model.embed(texts)
        >>>
        >>> # Compute cosine similarity
        >>> def cosine_similarity(a, b):
        ...     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        >>>
        >>> sim_01 = cosine_similarity(embeddings[0], embeddings[1])
        >>> sim_02 = cosine_similarity(embeddings[0], embeddings[2])
        >>> print(f"Cat/Feline similarity: {sim_01:.3f}")  # High similarity
        >>> print(f"Cat/Python similarity: {sim_02:.3f}")  # Low similarity

        Using embeddings for text classification:

        >>> model = CohereModel()
        >>> # Create embeddings with classification-optimized settings
        >>> texts = ["Great product!", "Terrible experience", "Average quality"]
        >>> embeddings = model.embed(texts, input_type="classification")
        >>> # Use embeddings as features for a classifier

        Notes
        -----
        - This method uses "embed-english-v3.0" regardless of the model_name
          set during initialization (embedding requires a dedicated model)
        - For optimal search performance, use "search_document" for content
          being indexed and "search_query" for user queries
        - The embedding model is multilingual despite the "english" name
        - Consider batching large numbers of texts to avoid API limits

        See Also
        --------
        rerank : For reordering documents by relevance to a query.
        generate : For text generation instead of embeddings.
        """
        client = self._get_client()

        params = {
            "texts": texts,
            "model": "embed-english-v3.0",  # Cohere's embedding model
            "input_type": input_type,
        }

        if embedding_types:
            params["embedding_types"] = embedding_types

        response = client.embed(**params)

        return response.embeddings

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: Optional[int] = None,
        model: str = "rerank-english-v3.0",
    ) -> list[dict[str, Any]]:
        """Rerank documents based on their relevance to a query.

        Takes a query and a list of documents, then returns the documents
        sorted by relevance score. This is particularly useful for improving
        search results from keyword-based retrieval systems or for selecting
        the most relevant context for RAG (Retrieval-Augmented Generation).

        The reranking model is specifically trained to assess document-query
        relevance and typically provides better relevance ordering than
        embedding similarity alone.

        Parameters
        ----------
        query : str
            The search query or question to rank documents against.
            Should be a natural language query representing the user's
            information need.
        documents : list[str]
            List of document texts to rerank. These are typically
            candidates retrieved from a first-stage retrieval system
            (e.g., BM25, embedding similarity).
        top_n : Optional[int], optional
            Number of top-ranked documents to return. If not specified,
            returns all documents in ranked order. Use this to limit
            results and reduce response size.
        model : str, optional
            The Cohere reranking model to use. Default is "rerank-english-v3.0".
            Other options include:

            - "rerank-english-v3.0": Latest English model (recommended)
            - "rerank-multilingual-v3.0": Multilingual support
            - "rerank-english-v2.0": Previous version

        Returns
        -------
        list[dict[str, Any]]
            List of reranked results, sorted by relevance (highest first).
            Each result is a dictionary containing:

            - ``index`` : int - Original index in the input documents list
            - ``relevance_score`` : float - Relevance score (higher = more relevant)
            - ``document`` : str - The document text

        Raises
        ------
        cohere.CohereAPIError
            If the API request fails (e.g., too many documents).
        ImportError
            If the cohere package is not installed.

        Examples
        --------
        Basic document reranking:

        >>> model = CohereModel()
        >>> query = "What is machine learning?"
        >>> documents = [
        ...     "Python is a programming language.",
        ...     "Machine learning is a branch of AI.",
        ...     "The weather is nice today.",
        ...     "ML algorithms learn patterns from data.",
        ... ]
        >>> results = model.rerank(query, documents)
        >>> for r in results:
        ...     print(f"{r['relevance_score']:.3f}: {r['document'][:50]}...")

        Getting only the top results:

        >>> model = CohereModel()
        >>> results = model.rerank(
        ...     query="How to train neural networks?",
        ...     documents=documents,
        ...     top_n=2  # Only top 2 results
        ... )
        >>> print(f"Got {len(results)} results")
        Got 2 results

        Using rerank to improve search results:

        >>> model = CohereModel()
        >>> # Assume we have candidate documents from keyword search
        >>> keyword_results = [
        ...     "Introduction to Python programming basics",
        ...     "Machine learning with Python: A practical guide",
        ...     "Python snake care and habitat",
        ...     "Advanced Python data analysis techniques",
        ... ]
        >>> query = "How to use Python for data science"
        >>> reranked = model.rerank(query, keyword_results, top_n=2)
        >>> # Most relevant documents for data science are now first

        Building a RAG pipeline with reranking:

        >>> model = CohereModel()
        >>> # Step 1: Retrieve candidates (simulated)
        >>> candidates = retrieve_documents("What is deep learning?")  # Your retrieval
        >>> # Step 2: Rerank for better relevance
        >>> reranked = model.rerank("What is deep learning?", candidates, top_n=3)
        >>> # Step 3: Use top documents as context for generation
        >>> context = "\\n".join([r['document'] for r in reranked])
        >>> answer = model.generate(f"Context: {context}\\n\\nQuestion: What is deep learning?")

        Accessing original indices for metadata lookup:

        >>> model = CohereModel()
        >>> documents = ["Doc A content", "Doc B content", "Doc C content"]
        >>> metadata = [{"id": 1, "source": "A"}, {"id": 2, "source": "B"}, {"id": 3, "source": "C"}]
        >>> results = model.rerank("query", documents)
        >>> for r in results:
        ...     original_meta = metadata[r['index']]
        ...     print(f"Score {r['relevance_score']:.3f}: {original_meta['source']}")

        Using multilingual reranking:

        >>> model = CohereModel()
        >>> results = model.rerank(
        ...     query="What is artificial intelligence?",
        ...     documents=[
        ...         "AI is transforming industries worldwide.",
        ...         "La inteligencia artificial es fascinante.",  # Spanish
        ...         "KI veraendert die Welt.",  # German
        ...     ],
        ...     model="rerank-multilingual-v3.0"
        ... )

        Notes
        -----
        - Reranking is computationally more expensive than embedding similarity
          but typically provides better relevance ordering
        - The relevance_score is not a probability; use it for relative
          ordering, not absolute thresholds
        - For best results, provide diverse candidate documents from your
          initial retrieval stage
        - The model parameter is independent of the CohereModel's model_name
          (reranking uses specialized models)

        See Also
        --------
        embed : For generating embeddings for initial retrieval.
        generate : For generating responses using reranked context.
        """
        client = self._get_client()

        params = {
            "query": query,
            "documents": documents,
            "model": model,
        }

        if top_n:
            params["top_n"] = top_n

        response = client.rerank(**params)

        return [
            {
                "index": r.index,
                "relevance_score": r.relevance_score,
                "document": documents[r.index],
            }
            for r in response.results
        ]
