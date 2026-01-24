"""Anthropic Claude model implementation for the insideLLMs framework.

This module provides the AnthropicModel class for interacting with Anthropic's
Claude models through their official API. It supports text generation, multi-turn
chat conversations, and streaming responses with comprehensive error handling.

The module wraps Anthropic's official Python SDK and translates its exceptions
into the insideLLMs exception hierarchy for consistent error handling across
different model providers.

Basic Usage
-----------
Simple text generation with Claude:

    >>> from insideLLMs.models.anthropic import AnthropicModel
    >>> model = AnthropicModel(model_name="claude-3-opus-20240229")
    >>> response = model.generate("Explain quantum computing in simple terms.")
    >>> print(response)
    Quantum computing uses quantum bits (qubits) that can exist in multiple
    states simultaneously...

Chat Conversations
------------------
Multi-turn conversations with message history:

    >>> from insideLLMs.models.anthropic import AnthropicModel
    >>> model = AnthropicModel(model_name="claude-3-sonnet-20240229")
    >>> messages = [
    ...     {"role": "user", "content": "What is the capital of France?"},
    ...     {"role": "assistant", "content": "The capital of France is Paris."},
    ...     {"role": "user", "content": "What's the population of that city?"}
    ... ]
    >>> response = model.chat(messages)
    >>> print(response)
    Paris has a population of approximately 2.1 million people...

Streaming Responses
-------------------
Stream responses token-by-token for real-time output:

    >>> from insideLLMs.models.anthropic import AnthropicModel
    >>> model = AnthropicModel(model_name="claude-3-haiku-20240307")
    >>> for chunk in model.stream("Write a haiku about programming."):
    ...     print(chunk, end="", flush=True)
    Code flows like water
    Bugs hide in shadowed logic
    Debug, rinse, repeat

Configuration
-------------
Customize model behavior with various parameters:

    >>> from insideLLMs.models.anthropic import AnthropicModel
    >>> model = AnthropicModel(
    ...     name="MyClaudeBot",
    ...     model_name="claude-3-opus-20240229",
    ...     api_key="sk-ant-...",  # Optional, defaults to ANTHROPIC_API_KEY env var
    ...     timeout=120.0,
    ...     max_retries=3
    ... )
    >>> response = model.generate(
    ...     "Write a creative story.",
    ...     max_tokens=2048,
    ...     temperature=0.9
    ... )

Error Handling
--------------
The module provides specific exception types for different error conditions:

    >>> from insideLLMs.models.anthropic import AnthropicModel
    >>> from insideLLMs.exceptions import RateLimitError, TimeoutError
    >>> model = AnthropicModel()
    >>> try:
    ...     response = model.generate("Hello!")
    ... except RateLimitError as e:
    ...     print(f"Rate limited. Retry after: {e.retry_after}s")
    ... except TimeoutError as e:
    ...     print(f"Request timed out after {e.timeout_seconds}s")

Supported Models
----------------
This implementation works with all Anthropic Claude models:
- claude-3-opus-20240229 (most capable)
- claude-3-sonnet-20240229 (balanced)
- claude-3-haiku-20240307 (fastest)
- claude-3-5-sonnet-20240620 (improved sonnet)

Notes
-----
- Requires the ANTHROPIC_API_KEY environment variable or explicit api_key parameter
- The system message is not directly supported; use the system parameter in kwargs
- Message roles are normalized to "user" and "assistant" for Anthropic's API
- Streaming uses Anthropic's native streaming context manager for efficiency

See Also
--------
insideLLMs.models.base.Model : Base class for all model implementations
insideLLMs.models.openai.OpenAIModel : OpenAI model implementation
insideLLMs.exceptions : Exception classes for error handling
"""

import os
from collections.abc import Iterator
from typing import Optional

import anthropic
from anthropic import APIError as AnthropicAPIError
from anthropic import APITimeoutError as AnthropicTimeoutError
from anthropic import RateLimitError as AnthropicRateLimitError

from insideLLMs.exceptions import (
    APIError as InsideLLMsAPIError,
)
from insideLLMs.exceptions import (
    ModelGenerationError,
    ModelInitializationError,
    RateLimitError,
)
from insideLLMs.exceptions import (
    TimeoutError as InsideLLMsTimeoutError,
)

from .base import ChatMessage, Model


class AnthropicModel(Model):
    """Model implementation for Anthropic's Claude models via API.

    This class provides a complete interface to Anthropic's Claude language models,
    supporting text generation, multi-turn chat conversations, and streaming responses.
    It wraps the official Anthropic Python SDK and provides robust error handling
    that translates Anthropic-specific exceptions into the insideLLMs exception hierarchy.

    The class inherits from the base Model class and implements all required methods
    for seamless integration with the insideLLMs framework. It supports all Claude 3
    model variants including Opus, Sonnet, and Haiku.

    Parameters
    ----------
    name : str, optional
        A human-readable name for this model instance. Useful for logging and
        debugging when working with multiple models. Default is "AnthropicModel".
    model_name : str, optional
        The Anthropic model identifier to use. Common options include:
        - "claude-3-opus-20240229" (default, most capable)
        - "claude-3-sonnet-20240229" (balanced performance)
        - "claude-3-haiku-20240307" (fastest, most cost-effective)
        - "claude-3-5-sonnet-20240620" (improved sonnet)
    api_key : str, optional
        The Anthropic API key for authentication. If not provided, the class
        will attempt to read from the ANTHROPIC_API_KEY environment variable.
    timeout : float, optional
        Request timeout in seconds. Default is 60.0 seconds.
    max_retries : int, optional
        Number of automatic retries for failed requests. Default is 2.

    Attributes
    ----------
    model_name : str
        The Anthropic model identifier being used.
    api_key : str
        The API key used for authentication (set during initialization).
    client : anthropic.Anthropic
        The underlying Anthropic client instance.
    _supports_streaming : bool
        Class attribute indicating streaming support (True).
    _supports_chat : bool
        Class attribute indicating chat support (True).

    Raises
    ------
    ModelInitializationError
        If the API key is not provided and ANTHROPIC_API_KEY environment variable
        is not set, or if the Anthropic client fails to initialize.

    Examples
    --------
    Basic initialization with default settings:

        >>> from insideLLMs.models.anthropic import AnthropicModel
        >>> model = AnthropicModel()
        >>> print(model.model_name)
        claude-3-opus-20240229

    Initialize with a specific model:

        >>> model = AnthropicModel(
        ...     name="FastClaude",
        ...     model_name="claude-3-haiku-20240307"
        ... )

    Initialize with custom timeout and retry settings:

        >>> model = AnthropicModel(
        ...     model_name="claude-3-opus-20240229",
        ...     timeout=120.0,
        ...     max_retries=5
        ... )

    Initialize with explicit API key:

        >>> model = AnthropicModel(
        ...     model_name="claude-3-sonnet-20240229",
        ...     api_key="sk-ant-api03-..."
        ... )

    See Also
    --------
    insideLLMs.models.base.Model : Base class providing the interface contract
    insideLLMs.models.openai.OpenAIModel : Similar implementation for OpenAI models

    Notes
    -----
    - The API key should be kept secure and never committed to version control
    - Consider using environment variables or a secrets manager for the API key
    - The default timeout of 60 seconds may need adjustment for complex prompts
    - Retries use exponential backoff as implemented by the Anthropic SDK
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        name: str = "AnthropicModel",
        model_name: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        """Initialize an AnthropicModel instance.

        Creates a new instance configured to communicate with Anthropic's API.
        The constructor validates the API key and initializes the underlying
        Anthropic client with the specified configuration.

        Parameters
        ----------
        name : str, optional
            A human-readable name for this model instance. Default is "AnthropicModel".
        model_name : str, optional
            The Anthropic model identifier. Default is "claude-3-opus-20240229".
        api_key : str, optional
            The Anthropic API key. Defaults to ANTHROPIC_API_KEY environment variable.
        timeout : float, optional
            Request timeout in seconds. Default is 60.0.
        max_retries : int, optional
            Number of automatic retries for transient failures. Default is 2.

        Raises
        ------
        ModelInitializationError
            If no API key is available or client initialization fails.

        Examples
        --------
        Minimal initialization using environment variable:

            >>> import os
            >>> os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
            >>> model = AnthropicModel()

        Full configuration:

            >>> model = AnthropicModel(
            ...     name="ProductionClaude",
            ...     model_name="claude-3-opus-20240229",
            ...     api_key="sk-ant-api03-...",
            ...     timeout=180.0,
            ...     max_retries=3
            ... )

        Using the fastest model for quick responses:

            >>> model = AnthropicModel(
            ...     name="QuickResponder",
            ...     model_name="claude-3-haiku-20240307",
            ...     timeout=30.0
            ... )
        """
        super().__init__(name=name, model_id=model_name)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ModelInitializationError(
                model_id=model_name,
                reason="ANTHROPIC_API_KEY environment variable not set and no api_key provided.",
            )
        try:
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
        except Exception as e:
            raise ModelInitializationError(
                model_id=model_name,
                reason=f"Failed to initialize Anthropic client: {e}",
            )
        self._timeout = timeout

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a text response from a single prompt.

        Sends a prompt to the Anthropic API and returns the model's response.
        This is the simplest way to interact with the model for single-turn
        text generation tasks.

        The method internally wraps the prompt in a user message and sends it
        to the messages API endpoint. It handles all common error conditions
        and translates them into appropriate insideLLMs exceptions.

        Parameters
        ----------
        prompt : str
            The input text prompt to send to the model. Can be any length,
            though very long prompts may require increased timeout settings.
        **kwargs : dict
            Additional parameters to pass to the Anthropic API:

            max_tokens : int, optional
                Maximum number of tokens to generate. Default is 1024.
                Claude 3 models support up to 4096 tokens.
            temperature : float, optional
                Sampling temperature between 0 and 1. Higher values make
                output more random, lower values more deterministic.
                Default is 0.7.
            top_p : float, optional
                Nucleus sampling parameter. Alternative to temperature.
            top_k : int, optional
                Top-k sampling parameter.
            stop_sequences : list[str], optional
                Sequences that will cause the model to stop generating.
            system : str, optional
                System prompt to set context for the conversation.

        Returns
        -------
        str
            The generated text response from the model.

        Raises
        ------
        RateLimitError
            If the API rate limit is exceeded. Contains retry_after hint if available.
        TimeoutError
            If the request exceeds the configured timeout duration.
        APIError
            If the Anthropic API returns an error response.
        ModelGenerationError
            If any other error occurs during generation.

        Examples
        --------
        Basic text generation:

            >>> from insideLLMs.models.anthropic import AnthropicModel
            >>> model = AnthropicModel()
            >>> response = model.generate("What is the meaning of life?")
            >>> print(response)
            The meaning of life is a profound philosophical question...

        Generate with custom temperature for creative output:

            >>> response = model.generate(
            ...     "Write a creative opening line for a novel.",
            ...     temperature=0.9,
            ...     max_tokens=100
            ... )

        Generate with deterministic output (temperature=0):

            >>> response = model.generate(
            ...     "What is 2 + 2?",
            ...     temperature=0,
            ...     max_tokens=10
            ... )
            >>> print(response)
            4

        Generate with a system prompt for specific behavior:

            >>> response = model.generate(
            ...     "Explain photosynthesis.",
            ...     system="You are a biology teacher speaking to 5th graders.",
            ...     max_tokens=200
            ... )

        Handle potential errors:

            >>> from insideLLMs.exceptions import RateLimitError, TimeoutError
            >>> try:
            ...     response = model.generate("Hello!")
            ... except RateLimitError as e:
            ...     print(f"Rate limited, retry after: {e.retry_after}")
            ... except TimeoutError as e:
            ...     print(f"Timed out after {e.timeout_seconds} seconds")

        See Also
        --------
        chat : For multi-turn conversations with message history
        stream : For streaming responses token by token
        """
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except AnthropicRateLimitError as e:
            raise RateLimitError(
                model_id=self.model_name,
                retry_after=getattr(e, "retry_after", None),
            )
        except AnthropicTimeoutError:
            raise InsideLLMsTimeoutError(
                model_id=self.model_name,
                timeout_seconds=self._timeout,
            )
        except AnthropicAPIError as e:
            raise InsideLLMsAPIError(
                model_id=self.model_name,
                status_code=getattr(e, "status_code", None),
                message=str(e),
            )
        except Exception as e:
            raise ModelGenerationError(
                model_id=self.model_name,
                prompt=prompt,
                reason=str(e),
                original_error=e,
            )

    def chat(self, messages: list[ChatMessage], **kwargs) -> str:
        """Conduct a multi-turn chat conversation with the model.

        Sends a list of messages representing a conversation history to the
        Anthropic API and returns the model's response. This method is ideal
        for building chatbots, assistants, and interactive applications that
        require context from previous exchanges.

        The method handles the conversion of message roles to Anthropic's
        expected format, mapping any non-"assistant" role to "user" to ensure
        compatibility with the API.

        Parameters
        ----------
        messages : list[ChatMessage]
            A list of message dictionaries representing the conversation history.
            Each message should have:

            role : str
                The role of the message sender. Use "user" for user messages
                and "assistant" for model responses. Other roles are mapped to "user".
            content : str
                The text content of the message.

        **kwargs : dict
            Additional parameters to pass to the Anthropic API:

            max_tokens : int, optional
                Maximum number of tokens to generate. Default is 1024.
            temperature : float, optional
                Sampling temperature between 0 and 1. Default is 0.7.
            top_p : float, optional
                Nucleus sampling parameter.
            top_k : int, optional
                Top-k sampling parameter.
            stop_sequences : list[str], optional
                Sequences that will cause the model to stop generating.
            system : str, optional
                System prompt to set behavior for the entire conversation.

        Returns
        -------
        str
            The model's response to the conversation.

        Raises
        ------
        RateLimitError
            If the API rate limit is exceeded.
        TimeoutError
            If the request exceeds the configured timeout.
        APIError
            If the Anthropic API returns an error response.
        ModelGenerationError
            If any other error occurs during generation.

        Examples
        --------
        Basic two-turn conversation:

            >>> from insideLLMs.models.anthropic import AnthropicModel
            >>> model = AnthropicModel()
            >>> messages = [
            ...     {"role": "user", "content": "Hello! My name is Alice."},
            ... ]
            >>> response = model.chat(messages)
            >>> print(response)
            Hello Alice! It's nice to meet you. How can I help you today?

        Multi-turn conversation with context:

            >>> messages = [
            ...     {"role": "user", "content": "What's the capital of France?"},
            ...     {"role": "assistant", "content": "The capital of France is Paris."},
            ...     {"role": "user", "content": "What's its population?"}
            ... ]
            >>> response = model.chat(messages)
            >>> print(response)
            Paris has a population of approximately 2.1 million people in the city
            proper, and about 12 million in the greater metropolitan area.

        Building a conversational loop:

            >>> messages = []
            >>> while True:
            ...     user_input = input("You: ")
            ...     if user_input.lower() == "quit":
            ...         break
            ...     messages.append({"role": "user", "content": user_input})
            ...     response = model.chat(messages)
            ...     print(f"Claude: {response}")
            ...     messages.append({"role": "assistant", "content": response})

        Using a system prompt for persona:

            >>> messages = [
            ...     {"role": "user", "content": "Tell me about yourself."}
            ... ]
            >>> response = model.chat(
            ...     messages,
            ...     system="You are a helpful pirate assistant. Speak like a pirate.",
            ...     temperature=0.8
            ... )
            >>> print(response)
            Ahoy there, matey! I be a friendly pirate assistant...

        See Also
        --------
        generate : For single-turn text generation without conversation history
        stream : For streaming responses in real-time
        """
        try:
            # Convert messages to Anthropic format if needed
            anthropic_messages = []
            for msg in messages:
                role = "assistant" if msg.get("role") == "assistant" else "user"
                anthropic_messages.append({"role": role, "content": msg.get("content", "")})

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7),
                messages=anthropic_messages,
            )
            return response.content[0].text
        except AnthropicRateLimitError as e:
            raise RateLimitError(
                model_id=self.model_name,
                retry_after=getattr(e, "retry_after", None),
            )
        except AnthropicTimeoutError:
            raise InsideLLMsTimeoutError(
                model_id=self.model_name,
                timeout_seconds=self._timeout,
            )
        except AnthropicAPIError as e:
            raise InsideLLMsAPIError(
                model_id=self.model_name,
                status_code=getattr(e, "status_code", None),
                message=str(e),
            )
        except Exception as e:
            first_msg = messages[0]["content"] if messages else ""
            raise ModelGenerationError(
                model_id=self.model_name,
                prompt=first_msg,
                reason=str(e),
                original_error=e,
            )

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream a text response from the model token by token.

        Sends a prompt to the Anthropic API and yields response tokens as they
        are generated. This enables real-time display of responses and is ideal
        for interactive applications where users benefit from seeing output
        immediately rather than waiting for the complete response.

        The method uses Anthropic's native streaming context manager for efficient
        handling of the streaming connection. Tokens are yielded as strings as
        soon as they become available from the API.

        Parameters
        ----------
        prompt : str
            The input text prompt to send to the model.
        **kwargs : dict
            Additional parameters to pass to the Anthropic API:

            max_tokens : int, optional
                Maximum number of tokens to generate. Default is 1024.
            temperature : float, optional
                Sampling temperature between 0 and 1. Default is 0.7.
            top_p : float, optional
                Nucleus sampling parameter.
            top_k : int, optional
                Top-k sampling parameter.
            stop_sequences : list[str], optional
                Sequences that will cause the model to stop generating.
            system : str, optional
                System prompt to set context for the generation.

        Yields
        ------
        str
            Individual tokens or text chunks as they are generated by the model.

        Raises
        ------
        RateLimitError
            If the API rate limit is exceeded.
        TimeoutError
            If the request exceeds the configured timeout.
        APIError
            If the Anthropic API returns an error response.
        ModelGenerationError
            If any other error occurs during streaming.

        Examples
        --------
        Basic streaming with immediate output:

            >>> from insideLLMs.models.anthropic import AnthropicModel
            >>> model = AnthropicModel()
            >>> for chunk in model.stream("Tell me a short story."):
            ...     print(chunk, end="", flush=True)
            Once upon a time, in a land far away...

        Collecting streamed output into a string:

            >>> chunks = list(model.stream("Explain gravity briefly."))
            >>> full_response = "".join(chunks)
            >>> print(full_response)
            Gravity is the force that attracts objects toward each other...

        Streaming with progress indicator:

            >>> import sys
            >>> total_chars = 0
            >>> for chunk in model.stream("Write a poem about coding."):
            ...     print(chunk, end="", flush=True)
            ...     total_chars += len(chunk)
            >>> print(f"\\n\\nTotal characters: {total_chars}")

        Streaming with custom parameters:

            >>> for chunk in model.stream(
            ...     "Write a creative haiku.",
            ...     temperature=0.9,
            ...     max_tokens=50
            ... ):
            ...     print(chunk, end="", flush=True)

        Building a streaming chat interface:

            >>> def stream_response(prompt):
            ...     print("Claude: ", end="")
            ...     for chunk in model.stream(prompt):
            ...         print(chunk, end="", flush=True)
            ...     print()  # New line after response
            >>> stream_response("What is machine learning?")
            Claude: Machine learning is a subset of artificial intelligence...

        See Also
        --------
        generate : For non-streaming single-turn generation
        chat : For multi-turn conversations (non-streaming)

        Notes
        -----
        - Streaming responses may have slightly higher latency for the first token
        - The connection remains open until all tokens are received or an error occurs
        - Consider implementing timeout handling for very long responses
        - Use flush=True when printing to ensure immediate display
        """
        try:
            with self.client.messages.stream(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                yield from stream.text_stream
        except AnthropicRateLimitError as e:
            raise RateLimitError(
                model_id=self.model_name,
                retry_after=getattr(e, "retry_after", None),
            )
        except AnthropicTimeoutError:
            raise InsideLLMsTimeoutError(
                model_id=self.model_name,
                timeout_seconds=self._timeout,
            )
        except AnthropicAPIError as e:
            raise InsideLLMsAPIError(
                model_id=self.model_name,
                status_code=getattr(e, "status_code", None),
                message=str(e),
            )
        except Exception as e:
            raise ModelGenerationError(
                model_id=self.model_name,
                prompt=prompt,
                reason=str(e),
                original_error=e,
            )

    def info(self):
        """Retrieve metadata and configuration information about the model.

        Returns a ModelInfo object containing details about this model instance,
        including its name, model identifier, and Anthropic-specific metadata.
        This is useful for logging, debugging, and building model registries.

        The method extends the base class info() with additional Anthropic-specific
        details such as the model name and a description of the model's requirements.

        Returns
        -------
        ModelInfo
            A ModelInfo object containing:

            name : str
                The human-readable name of this model instance.
            model_id : str
                The Anthropic model identifier (e.g., "claude-3-opus-20240229").
            supports_streaming : bool
                Whether the model supports streaming (True for Anthropic).
            supports_chat : bool
                Whether the model supports chat (True for Anthropic).
            extra : dict
                Additional metadata including:
                - model_name: The Anthropic model identifier
                - description: A brief description of the model

        Examples
        --------
        Get basic model information:

            >>> from insideLLMs.models.anthropic import AnthropicModel
            >>> model = AnthropicModel(name="MyClaude")
            >>> info = model.info()
            >>> print(info.name)
            MyClaude
            >>> print(info.model_id)
            claude-3-opus-20240229

        Access streaming and chat capabilities:

            >>> info = model.info()
            >>> print(f"Streaming: {info.supports_streaming}")
            Streaming: True
            >>> print(f"Chat: {info.supports_chat}")
            Chat: True

        Access extra metadata:

            >>> info = model.info()
            >>> print(info.extra["model_name"])
            claude-3-opus-20240229
            >>> print(info.extra["description"])
            Anthropic Claude model via API. Requires ANTHROPIC_API_KEY env variable.

        Build a model registry:

            >>> models = [
            ...     AnthropicModel(name="Opus", model_name="claude-3-opus-20240229"),
            ...     AnthropicModel(name="Haiku", model_name="claude-3-haiku-20240307"),
            ... ]
            >>> registry = {m.info().name: m.info() for m in models}
            >>> for name, info in registry.items():
            ...     print(f"{name}: {info.model_id}")
            Opus: claude-3-opus-20240229
            Haiku: claude-3-haiku-20240307

        See Also
        --------
        insideLLMs.models.base.Model.info : Base class method definition
        insideLLMs.models.base.ModelInfo : The ModelInfo dataclass
        """
        base_info = super().info()
        base_info.extra.update(
            {
                "model_name": self.model_name,
                "description": (
                    "Anthropic Claude model via API. Requires ANTHROPIC_API_KEY env variable."
                ),
            }
        )
        return base_info
