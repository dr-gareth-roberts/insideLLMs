"""Google Gemini model implementation.

This module provides a wrapper for Google's Gemini models via the
google-generativeai SDK. It offers a unified interface for text generation,
multi-turn chat conversations, and streaming responses.

The module implements the base Model interface from insideLLMs, providing
seamless integration with the rest of the library while exposing Gemini-specific
features like safety settings and generation configuration.

Module Components:
    GeminiModel: Main class for interacting with Google's Gemini API.

Supported Models:
    - gemini-1.5-flash: Fast, efficient model for most tasks
    - gemini-1.5-pro: More capable model for complex reasoning
    - gemini-pro: Previous generation model (legacy support)
    - gemini-pro-vision: Multimodal model (legacy support)

Environment Variables:
    GOOGLE_API_KEY: Required API key for authentication with Google AI.
        Can also be passed directly to GeminiModel constructor.

Examples:
    Basic text generation:

        >>> from insideLLMs.models.gemini import GeminiModel
        >>> model = GeminiModel(model_name="gemini-1.5-flash")
        >>> response = model.generate("Explain quantum computing in simple terms.")
        >>> print(response)

    Multi-turn chat conversation:

        >>> model = GeminiModel(model_name="gemini-1.5-pro")
        >>> messages = [
        ...     {"role": "user", "content": "What is the capital of France?"},
        ...     {"role": "assistant", "content": "The capital of France is Paris."},
        ...     {"role": "user", "content": "What is its population?"}
        ... ]
        >>> response = model.chat(messages)
        >>> print(response)

    Streaming responses for real-time output:

        >>> model = GeminiModel(model_name="gemini-1.5-flash")
        >>> for chunk in model.stream("Write a haiku about programming"):
        ...     print(chunk, end="", flush=True)

    Custom generation configuration:

        >>> model = GeminiModel(
        ...     model_name="gemini-1.5-pro",
        ...     generation_config={
        ...         "temperature": 0.7,
        ...         "max_output_tokens": 1024,
        ...         "top_p": 0.9
        ...     }
        ... )
        >>> response = model.generate("Generate a creative story opening.")

See Also:
    - insideLLMs.models.base.Model: Base class defining the model interface.
    - https://ai.google.dev/: Google AI documentation for Gemini models.
"""

import os
from collections.abc import Iterator
from typing import Any, Optional

from insideLLMs.models.base import ChatMessage, Model
from insideLLMs.types import ModelInfo


class GeminiModel(Model):
    """Model implementation for Google's Gemini models via API.

    This class provides a comprehensive wrapper around Google's Gemini generative
    AI models, supporting text generation, multi-turn chat conversations, and
    streaming responses. It implements the base Model interface for seamless
    integration with the insideLLMs framework.

    The GeminiModel lazily initializes the Google AI client on first use,
    allowing for efficient resource management. It supports all major Gemini
    models and provides access to advanced features like safety settings and
    custom generation configurations.

    Attributes:
        model_name (str): The Gemini model identifier being used.
        api_key (str): The Google AI API key for authentication.
        safety_settings (Optional[list[dict[str, Any]]]): Safety filter settings.
        default_generation_config (dict[str, Any]): Default generation parameters.

    Class Attributes:
        _supports_streaming (bool): True, indicating streaming support.
        _supports_chat (bool): True, indicating multi-turn chat support.

    Examples:
        Basic initialization and generation:

            >>> from insideLLMs.models.gemini import GeminiModel
            >>> model = GeminiModel(model_name="gemini-1.5-flash")
            >>> response = model.generate("What is machine learning?")
            >>> print(response)
            Machine learning is a subset of artificial intelligence...

        Using environment variable for API key:

            >>> import os
            >>> os.environ["GOOGLE_API_KEY"] = "your-api-key"
            >>> model = GeminiModel()  # Uses GOOGLE_API_KEY automatically
            >>> response = model.generate("Hello!")

        Explicit API key with custom generation config:

            >>> model = GeminiModel(
            ...     api_key="your-api-key",
            ...     model_name="gemini-1.5-pro",
            ...     generation_config={
            ...         "temperature": 0.9,
            ...         "max_output_tokens": 2048,
            ...         "top_k": 40,
            ...         "top_p": 0.95
            ...     }
            ... )

        With custom safety settings:

            >>> safety_settings = [
            ...     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ...     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ... ]
            >>> model = GeminiModel(
            ...     model_name="gemini-1.5-flash",
            ...     safety_settings=safety_settings
            ... )

        Token counting before generation:

            >>> model = GeminiModel(model_name="gemini-1.5-flash")
            >>> token_count = model.count_tokens("This is a test prompt.")
            >>> print(f"Prompt uses {token_count} tokens")

    Notes:
        - The API key is required and can be provided via constructor or
          GOOGLE_API_KEY environment variable.
        - The client is lazily initialized on first API call.
        - Safety settings persist across all calls unless overridden.
        - Generation config can be set at initialization or per-call.

    See Also:
        - Model: Base class defining the model interface.
        - ChatMessage: Type alias for chat message dictionaries.
        - ModelInfo: Named tuple for model metadata.
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        name: str = "GeminiModel",
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        safety_settings: Optional[list[dict[str, Any]]] = None,
        generation_config: Optional[dict[str, Any]] = None,
    ):
        """Initialize the Gemini model with configuration options.

        Creates a new GeminiModel instance configured for the specified model
        and settings. The actual API client is lazily initialized on first use
        to optimize resource usage.

        Args:
            name: Human-readable name for this model instance. Used for
                identification in logs and model info. Defaults to "GeminiModel".
            model_name: The Gemini model identifier to use. Common options include:
                - "gemini-1.5-flash": Fast, efficient model (default)
                - "gemini-1.5-pro": More capable model for complex tasks
                - "gemini-pro": Previous generation model
                Defaults to "gemini-1.5-flash".
            api_key: Google AI API key for authentication. If not provided,
                the constructor looks for the GOOGLE_API_KEY environment variable.
                Defaults to None.
            safety_settings: Optional list of safety setting dictionaries to apply
                to all generation requests. Each dictionary should contain:
                - "category": The harm category (e.g., "HARM_CATEGORY_HARASSMENT")
                - "threshold": The blocking threshold (e.g., "BLOCK_MEDIUM_AND_ABOVE")
                Defaults to None (uses Google's defaults).
            generation_config: Optional dictionary of default generation parameters.
                Supported keys include:
                - "temperature": Controls randomness (0.0-2.0)
                - "max_output_tokens": Maximum response length
                - "top_p": Nucleus sampling parameter
                - "top_k": Top-k sampling parameter
                - "stop_sequences": List of stop sequences
                Defaults to None (empty config).

        Raises:
            ValueError: If no API key is provided and GOOGLE_API_KEY environment
                variable is not set.

        Examples:
            Minimal initialization with environment variable:

                >>> import os
                >>> os.environ["GOOGLE_API_KEY"] = "your-api-key"
                >>> model = GeminiModel()
                >>> print(model.model_name)
                gemini-1.5-flash

            Full configuration with all options:

                >>> model = GeminiModel(
                ...     name="MyGeminiPro",
                ...     model_name="gemini-1.5-pro",
                ...     api_key="your-api-key",
                ...     safety_settings=[
                ...         {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                ...          "threshold": "BLOCK_ONLY_HIGH"}
                ...     ],
                ...     generation_config={
                ...         "temperature": 0.7,
                ...         "max_output_tokens": 1024
                ...     }
                ... )

            Using a specific model version:

                >>> model = GeminiModel(
                ...     model_name="gemini-1.5-pro-latest",
                ...     generation_config={"temperature": 0.5}
                ... )

            Error handling for missing API key:

                >>> try:
                ...     model = GeminiModel(api_key=None)
                ... except ValueError as e:
                ...     print("API key required!")
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
        """Lazily initialize and return the Google AI GenerativeModel client.

        This method implements lazy initialization pattern to defer the
        expensive SDK import and client setup until the first actual API call.
        Subsequent calls return the cached client instance.

        The method configures the google-generativeai SDK with the API key
        and creates a GenerativeModel instance with the specified model name,
        safety settings, and generation configuration.

        Returns:
            google.generativeai.GenerativeModel: The initialized model client
                ready for content generation.

        Raises:
            ImportError: If the google-generativeai package is not installed.
                The error message includes installation instructions.

        Examples:
            Internal usage (called automatically by generate/chat/stream):

                >>> model = GeminiModel(api_key="your-key")
                >>> client = model._get_client()  # First call initializes
                >>> client = model._get_client()  # Subsequent calls return cached

            The client is typically accessed indirectly:

                >>> model = GeminiModel(api_key="your-key")
                >>> # _get_client() called internally
                >>> response = model.generate("Hello!")

        Notes:
            - This is a private method; users should not call it directly.
            - The client is cached in self._model after first initialization.
            - The genai module is cached in self._client for reuse.
            - Thread safety is not guaranteed for concurrent first calls.
        """
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
        """Generate a text response from the Gemini model for a single prompt.

        Sends the provided prompt to the Gemini API and returns the complete
        generated text response. This method supports various generation
        parameters that can override the default configuration.

        Args:
            prompt: The input text prompt to send to the model. Can be a
                simple question, instruction, or complex prompt with context.
            **kwargs: Additional generation parameters. Supported options include:
                - temperature (float): Controls randomness in generation.
                    Range 0.0-2.0, where lower values are more deterministic.
                - max_tokens (int): Maximum number of tokens in the response.
                    Alias for max_output_tokens.
                - max_output_tokens (int): Maximum number of tokens in the response.
                - top_p (float): Nucleus sampling parameter (0.0-1.0).
                - top_k (int): Top-k sampling parameter.
                - stop_sequences (list[str]): Sequences that stop generation.
                Any additional kwargs are passed directly to the API.

        Returns:
            str: The complete generated text response from the model.

        Raises:
            ImportError: If google-generativeai package is not installed.
            google.generativeai.types.StopCandidateException: If content is blocked.
            google.api_core.exceptions.GoogleAPIError: For API-related errors.

        Examples:
            Simple text generation:

                >>> model = GeminiModel(api_key="your-key")
                >>> response = model.generate("What is the speed of light?")
                >>> print(response)
                The speed of light in a vacuum is approximately 299,792,458 m/s...

            With custom temperature for more creative output:

                >>> response = model.generate(
                ...     "Write a creative tagline for a coffee shop",
                ...     temperature=0.9
                ... )
                >>> print(response)

            Limiting response length:

                >>> response = model.generate(
                ...     "Explain quantum mechanics",
                ...     max_tokens=100  # or max_output_tokens=100
                ... )
                >>> print(len(response.split()))  # Approximately limited

            Using multiple generation parameters:

                >>> response = model.generate(
                ...     "Generate a product description",
                ...     temperature=0.7,
                ...     max_output_tokens=200,
                ...     top_p=0.9,
                ...     top_k=40
                ... )

            Checking call count after generation:

                >>> model = GeminiModel(api_key="your-key")
                >>> _ = model.generate("Hello")
                >>> _ = model.generate("World")
                >>> print(model.call_count)
                2

        Notes:
            - The method increments the internal call counter.
            - Parameters in kwargs override default_generation_config values.
            - Both max_tokens and max_output_tokens are accepted for convenience.
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

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Engage in a multi-turn chat conversation with the Gemini model.

        Processes a list of chat messages representing a conversation history
        and generates the next assistant response. The method automatically
        handles role conversion between the standard format and Gemini's
        expected format.

        The method handles special cases:
        - System messages are prepended to the first user message
        - User messages use role "user"
        - Assistant messages are converted to role "model"

        Args:
            messages: List of chat message dictionaries. Each message should have:
                - "role" (str): One of "system", "user", or "assistant"
                - "content" (str): The message text content
            **kwargs: Additional generation parameters. Supported options include:
                - temperature (float): Controls randomness (0.0-2.0)
                - max_tokens (int): Maximum response tokens

        Returns:
            str: The assistant's text response to the conversation.

        Raises:
            ImportError: If google-generativeai package is not installed.
            google.generativeai.types.StopCandidateException: If content is blocked.
            IndexError: If messages list is empty.

        Examples:
            Simple single-turn chat:

                >>> model = GeminiModel(api_key="your-key")
                >>> messages = [{"role": "user", "content": "Hello!"}]
                >>> response = model.chat(messages)
                >>> print(response)
                Hello! How can I help you today?

            Multi-turn conversation:

                >>> messages = [
                ...     {"role": "user", "content": "My name is Alice."},
                ...     {"role": "assistant", "content": "Nice to meet you, Alice!"},
                ...     {"role": "user", "content": "What's my name?"}
                ... ]
                >>> response = model.chat(messages)
                >>> print(response)
                Your name is Alice.

            With system message for context:

                >>> messages = [
                ...     {"role": "system", "content": "You are a helpful cooking assistant."},
                ...     {"role": "user", "content": "How do I make pasta?"}
                ... ]
                >>> response = model.chat(messages)

            Custom temperature for varied responses:

                >>> messages = [
                ...     {"role": "user", "content": "Tell me a joke."}
                ... ]
                >>> response = model.chat(messages, temperature=0.9)

            Building a conversation programmatically:

                >>> model = GeminiModel(api_key="your-key")
                >>> conversation = []
                >>> conversation.append({"role": "user", "content": "Hi!"})
                >>> response = model.chat(conversation)
                >>> conversation.append({"role": "assistant", "content": response})
                >>> conversation.append({"role": "user", "content": "How are you?"})
                >>> response = model.chat(conversation)

        Notes:
            - System messages are handled by prepending to the first user message
              with a "[System: ...]" prefix, as Gemini doesn't have native system
              message support.
            - The conversation history is sent with each call; no state is
              maintained between calls.
            - The method increments the internal call counter.
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
        """Stream the response from the Gemini model in real-time chunks.

        Generates content from the model and yields it incrementally as chunks
        become available. This is useful for providing real-time feedback to
        users, especially for longer responses.

        The streaming implementation uses the Gemini API's native streaming
        capability, which returns response chunks as they are generated by
        the model.

        Args:
            prompt: The input text prompt to send to the model.
            **kwargs: Additional generation parameters. Supported options include:
                - temperature (float): Controls randomness (0.0-2.0)
                - max_tokens (int): Maximum response tokens
                - max_output_tokens (int): Maximum response tokens (alternative)
                Additional kwargs are passed to the API.

        Yields:
            str: Text chunks of the response as they become available.
                Empty chunks are automatically filtered out.

        Raises:
            ImportError: If google-generativeai package is not installed.
            google.generativeai.types.StopCandidateException: If content is blocked.
            google.api_core.exceptions.GoogleAPIError: For API-related errors.

        Examples:
            Basic streaming with immediate output:

                >>> model = GeminiModel(api_key="your-key")
                >>> for chunk in model.stream("Tell me a story"):
                ...     print(chunk, end="", flush=True)
                Once upon a time...

            Collecting streamed response into a string:

                >>> model = GeminiModel(api_key="your-key")
                >>> chunks = list(model.stream("Explain AI"))
                >>> full_response = "".join(chunks)
                >>> print(full_response)

            Streaming with custom temperature:

                >>> for chunk in model.stream("Write a poem", temperature=0.8):
                ...     print(chunk, end="")

            Processing chunks with custom logic:

                >>> word_count = 0
                >>> for chunk in model.stream("Write an essay"):
                ...     word_count += len(chunk.split())
                ...     print(chunk, end="", flush=True)
                >>> print(f"\\nTotal words: {word_count}")

            Using streaming for progress indication:

                >>> import sys
                >>> print("Generating: ", end="")
                >>> for i, chunk in enumerate(model.stream("Long explanation")):
                ...     if i % 10 == 0:
                ...         sys.stdout.write(".")
                ...         sys.stdout.flush()
                >>> print(" Done!")

        Notes:
            - The call counter is incremented once at the start, not per chunk.
            - Empty chunks from the API are filtered and not yielded.
            - The iterator must be fully consumed or explicitly closed.
            - Streaming is more memory-efficient for long responses.
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
        """Return detailed information about this Gemini model instance.

        Provides metadata about the model including its name, provider,
        capabilities, and configuration details. This is useful for
        introspection, logging, and debugging.

        Returns:
            ModelInfo: A named tuple containing:
                - name (str): The human-readable name of this model instance
                - provider (str): Always "Google" for Gemini models
                - model_id (str): The model identifier (e.g., "gemini-1.5-flash")
                - supports_streaming (bool): True for all Gemini models
                - supports_chat (bool): True for all Gemini models
                - extra (dict): Additional metadata including:
                    - model_name: The Gemini model name
                    - description: Brief description of the model

        Examples:
            Getting model information:

                >>> model = GeminiModel(
                ...     name="MyModel",
                ...     model_name="gemini-1.5-pro",
                ...     api_key="your-key"
                ... )
                >>> info = model.info()
                >>> print(info.name)
                MyModel
                >>> print(info.provider)
                Google

            Checking model capabilities:

                >>> info = model.info()
                >>> if info.supports_streaming:
                ...     print("Streaming is available!")
                >>> if info.supports_chat:
                ...     print("Multi-turn chat is available!")

            Accessing extra metadata:

                >>> info = model.info()
                >>> print(info.extra["model_name"])
                gemini-1.5-pro
                >>> print(info.extra["description"])
                Google Gemini model via google-generativeai SDK.

            Using info for logging:

                >>> import logging
                >>> logger = logging.getLogger(__name__)
                >>> info = model.info()
                >>> logger.info(f"Using {info.provider} model: {info.model_id}")

        Notes:
            - This method does not make any API calls.
            - The info is based on the configuration provided at initialization.
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
        """Count the number of tokens in the given text using Gemini's tokenizer.

        Uses the Gemini API's token counting endpoint to determine exactly how
        many tokens a given text would consume. This is useful for managing
        context windows, estimating costs, and ensuring prompts fit within
        model limits.

        Token counts are model-specific, as different models may use different
        tokenization schemes. This method uses the tokenizer for the model
        configured in this GeminiModel instance.

        Args:
            text: The text string to count tokens for. Can be any length,
                though very long texts may take longer to process.

        Returns:
            int: The total number of tokens in the text according to the
                Gemini tokenizer.

        Raises:
            ImportError: If google-generativeai package is not installed.
            google.api_core.exceptions.GoogleAPIError: For API-related errors.

        Examples:
            Basic token counting:

                >>> model = GeminiModel(api_key="your-key")
                >>> count = model.count_tokens("Hello, world!")
                >>> print(f"Token count: {count}")
                Token count: 4

            Checking if prompt fits in context:

                >>> model = GeminiModel(api_key="your-key")
                >>> prompt = "Your long prompt here..."
                >>> tokens = model.count_tokens(prompt)
                >>> max_tokens = 30000  # Example limit
                >>> if tokens < max_tokens:
                ...     response = model.generate(prompt)
                ... else:
                ...     print(f"Prompt too long: {tokens} tokens")

            Estimating conversation token usage:

                >>> model = GeminiModel(api_key="your-key")
                >>> messages = [
                ...     "User: Hello!",
                ...     "Assistant: Hi there!",
                ...     "User: How are you?"
                ... ]
                >>> total_tokens = sum(model.count_tokens(m) for m in messages)
                >>> print(f"Conversation uses {total_tokens} tokens")

            Comparing token counts for different phrasings:

                >>> model = GeminiModel(api_key="your-key")
                >>> short = model.count_tokens("Summarize this.")
                >>> long = model.count_tokens("Please provide a brief summary of the following text.")
                >>> print(f"Short: {short}, Long: {long}")

        Notes:
            - This method makes an API call but does not count toward generation
              usage or the internal call counter.
            - Token counts include special tokens like BOS/EOS markers.
            - Results are exact for the specified model's tokenizer.
        """
        model = self._get_client()
        result = model.count_tokens(text)
        return result.total_tokens

    def list_models(self) -> list[str]:
        """List all available Gemini models that support content generation.

        Queries the Google AI API to retrieve a list of all models that
        support the generateContent method. This is useful for discovering
        available models, checking for new model releases, or validating
        model names.

        The method filters models to only include those that support content
        generation (text/chat), excluding models that only support other
        methods like embedding or vision-only tasks.

        Returns:
            list[str]: A list of model name strings (e.g., "models/gemini-1.5-pro",
                "models/gemini-1.5-flash"). The names include the "models/" prefix
                as returned by the API.

        Raises:
            ImportError: If google-generativeai package is not installed.
            google.api_core.exceptions.GoogleAPIError: For API-related errors.

        Examples:
            Listing all available models:

                >>> model = GeminiModel(api_key="your-key")
                >>> available = model.list_models()
                >>> for name in available:
                ...     print(name)
                models/gemini-1.5-pro
                models/gemini-1.5-flash
                models/gemini-pro
                ...

            Checking if a specific model exists:

                >>> model = GeminiModel(api_key="your-key")
                >>> available = model.list_models()
                >>> if "models/gemini-1.5-pro" in available:
                ...     print("Gemini 1.5 Pro is available!")

            Finding models by pattern:

                >>> model = GeminiModel(api_key="your-key")
                >>> flash_models = [m for m in model.list_models() if "flash" in m]
                >>> print(f"Found {len(flash_models)} flash models")

            Getting just the model names without prefix:

                >>> model = GeminiModel(api_key="your-key")
                >>> names = [m.replace("models/", "") for m in model.list_models()]
                >>> print(names)
                ['gemini-1.5-pro', 'gemini-1.5-flash', ...]

        Notes:
            - This method makes an API call to list models.
            - Only models supporting "generateContent" method are returned.
            - Model availability may vary by region and API key permissions.
            - The client is initialized if not already done.
        """
        self._get_client()  # Ensure client is initialized
        import google.generativeai as genai

        models = genai.list_models()
        return [m.name for m in models if "generateContent" in m.supported_generation_methods]
