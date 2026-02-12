"""OpenAI model implementation for insideLLMs.

This module provides a Model implementation that wraps OpenAI's API for
GPT models (GPT-4, GPT-3.5-turbo, etc.). It handles authentication,
error mapping, streaming, and chat mode.

Requires:
    - openai>=1.0.0 Python package
    - OPENAI_API_KEY environment variable (or explicit api_key)

Example - Basic Usage:
    >>> from insideLLMs.models.openai import OpenAIModel
    >>>
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> response = model.generate("What is Python?")
    >>> print(response)

Example - With Custom Configuration:
    >>> model = OpenAIModel(
    ...     model_name="gpt-4-turbo",
    ...     api_key="sk-...",  # Or use OPENAI_API_KEY env var
    ...     timeout=120.0,
    ...     max_retries=5
    ... )

Example - Using Chat Mode:
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> messages = [
    ...     {"role": "system", "content": "You are a helpful assistant."},
    ...     {"role": "user", "content": "Hello!"},
    ... ]
    >>> response = model.chat(messages)

Example - Streaming Responses:
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> for chunk in model.stream("Write a poem about coding"):
    ...     print(chunk, end="", flush=True)

See Also:
    - OpenAI API documentation: https://platform.openai.com/docs/api-reference
    - insideLLMs.models.base.Model: Base class interface
"""

import os
from collections.abc import Iterator
from typing import Any, Optional

from openai import APIError, APITimeoutError, OpenAI
from openai import RateLimitError as OpenAIRateLimitError

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


class OpenAIModel(Model):
    """Model implementation for OpenAI's GPT models via API (openai>=1.0.0).

    Provides a full-featured wrapper around OpenAI's chat completions API with
    robust error handling for common failure modes (rate limits, timeouts, etc.).

    This model supports:
        - Text generation via generate()
        - Multi-turn chat via chat()
        - Streaming responses via stream()
        - All OpenAI API parameters (temperature, max_tokens, etc.)

    Attributes:
        model_name: The OpenAI model identifier (e.g., "gpt-4", "gpt-3.5-turbo").
        api_key: The OpenAI API key (from argument or environment).
        _client: The underlying openai.OpenAI client instance.
        _supports_streaming: True - streaming is supported.
        _supports_chat: True - chat mode is supported.

    Example - Basic Generation:
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> response = model.generate("Explain quantum computing.")
        >>> print(response)
        'Quantum computing is a type of computation...'

    Example - With Generation Parameters:
        >>> response = model.generate(
        ...     "Write a creative story.",
        ...     temperature=0.9,  # More creative
        ...     max_tokens=500,   # Limit length
        ...     top_p=0.95        # Nucleus sampling
        ... )

    Example - Multi-Turn Chat:
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> conversation = [
        ...     {"role": "system", "content": "You are a pirate."},
        ...     {"role": "user", "content": "Hello!"},
        ... ]
        >>> response = model.chat(conversation)
        >>> print(response)
        'Ahoy, matey! What brings ye to these waters?'

    Example - Streaming Output:
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> for chunk in model.stream("Count from 1 to 10"):
        ...     print(chunk, end="", flush=True)
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10

    Example - Error Handling:
        >>> from insideLLMs.exceptions import RateLimitError, TimeoutError
        >>>
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> try:
        ...     response = model.generate("Hello!")
        ... except RateLimitError as e:
        ...     print(f"Rate limited! Retry after: {e.retry_after}s")
        ... except TimeoutError as e:
        ...     print(f"Timed out after {e.timeout_seconds}s")

    Example - Using Azure OpenAI:
        >>> # Use base_url for Azure or other OpenAI-compatible endpoints
        >>> model = OpenAIModel(
        ...     model_name="gpt-4",
        ...     base_url="https://your-resource.openai.azure.com/",
        ...     api_key="your-azure-key"
        ... )

    Example - Via Registry:
        >>> from insideLLMs.registry import model_registry, ensure_builtins_registered
        >>> ensure_builtins_registered()
        >>> model = model_registry.get("openai", model_name="gpt-4")

    Raises:
        ModelInitializationError: If API key is missing or client init fails.
        RateLimitError: When OpenAI rate limits are hit (HTTP 429).
        TimeoutError: When requests exceed the timeout setting.
        APIError: For other OpenAI API errors.
        ModelGenerationError: For unexpected errors during generation.

    Note:
        The openai>=1.0.0 package is required. Earlier versions have a
        different API and are not supported.
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        name: str = "OpenAIModel",
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        api_key_env: str = "OPENAI_API_KEY",
        default_headers: Optional[dict[str, str]] = None,
    ):
        """Initialize an OpenAI model.

        Creates a configured OpenAI client and validates the API key.
        The client is created immediately to fail fast on configuration errors.

        Args:
            name: Human-readable name for this model instance.
                Used for logging and identification. Default: "OpenAIModel".
            model_name: The OpenAI model identifier to use.
                Common options: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo".
                Default: "gpt-3.5-turbo".
            api_key: OpenAI API key. If not provided, reads from the
                environment variable specified by ``api_key_env``.
            base_url: Override the default OpenAI API base URL.
                Useful for Azure OpenAI or proxy servers.
            organization: OpenAI organization ID for billing/access control.
            project: OpenAI project ID for usage tracking.
            timeout: Request timeout in seconds. Default: 60.0.
            max_retries: Number of automatic retries on transient errors.
                Handled by the openai library. Default: 2.
            api_key_env: Environment variable name used to fetch the API key
                when ``api_key`` is not provided. Default: OPENAI_API_KEY.
            default_headers: Optional default headers to include with each
                request. Useful for OpenAI-compatible gateways.

        Raises:
            ModelInitializationError: If API key is missing or invalid,
                or if the OpenAI client cannot be created.

        Example - Basic Initialization:
            >>> # Uses OPENAI_API_KEY from environment
            >>> model = OpenAIModel(model_name="gpt-4")

        Example - With Explicit API Key:
            >>> model = OpenAIModel(
            ...     model_name="gpt-4",
            ...     api_key="sk-..."
            ... )

        Example - With Organization:
            >>> model = OpenAIModel(
            ...     model_name="gpt-4",
            ...     organization="org-...",
            ...     project="proj-..."
            ... )

        Example - Custom Timeout:
            >>> # Longer timeout for complex prompts
            >>> model = OpenAIModel(
            ...     model_name="gpt-4",
            ...     timeout=120.0,  # 2 minutes
            ...     max_retries=5
            ... )

        Example - Azure OpenAI:
            >>> model = OpenAIModel(
            ...     model_name="gpt-4",
            ...     base_url="https://myresource.openai.azure.com/",
            ...     api_key=os.getenv("AZURE_OPENAI_KEY")
            ... )

        Note:
            The OpenAI client is created in the constructor. Any connection
            issues will surface on first use (generate/chat/stream), not here.
        """
        super().__init__(name=name, model_id=model_name)
        self.model_name = model_name
        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            raise ModelInitializationError(
                model_id=model_name,
                reason=(f"{api_key_env} environment variable not set and no api_key provided."),
            )
        safe_default_headers = dict(default_headers) if isinstance(default_headers, dict) else None
        try:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
                organization=organization,
                project=project,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=safe_default_headers,
            )
        except Exception as e:
            raise ModelInitializationError(
                model_id=model_name,
                reason=f"Failed to initialize OpenAI client: {e}",
            )
        self._base_url = base_url
        self._organization = organization
        self._project = project
        self._timeout = timeout
        self._api_key_env = api_key_env
        self._default_headers = safe_default_headers

    @staticmethod
    def _redact_headers(headers: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not headers:
            return None
        redacted: dict[str, Any] = {}
        for key, value in headers.items():
            key_str = str(key)
            if any(token in key_str.lower() for token in ("authorization", "api-key", "x-api-key")):
                redacted[key_str] = "***"
            else:
                redacted[key_str] = value
        return redacted

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the OpenAI model.

        Sends the prompt to OpenAI's chat completions API as a single user message
        and returns the model's response. Supports all standard OpenAI parameters.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional OpenAI API parameters. Common options:
                - temperature (float): Randomness (0.0-2.0, default ~1.0)
                - max_tokens (int): Maximum tokens to generate
                - top_p (float): Nucleus sampling (0.0-1.0)
                - frequency_penalty (float): Repetition penalty (-2.0 to 2.0)
                - presence_penalty (float): Topic diversity (-2.0 to 2.0)
                - stop (list[str]): Stop sequences
                - n (int): Number of completions (returns first)
                - seed (int): For deterministic outputs (beta)

        Returns:
            The model's text response.

        Raises:
            RateLimitError: When API rate limits are exceeded (HTTP 429).
            TimeoutError: When the request exceeds the timeout setting.
            APIError: For other OpenAI API errors (auth, invalid request, etc.).
            ModelGenerationError: For unexpected errors.

        Example - Basic Usage:
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> response = model.generate("What is 2+2?")
            >>> print(response)
            '2+2 equals 4.'

        Example - With Temperature:
            >>> # More creative/random
            >>> response = model.generate("Tell me a joke.", temperature=1.5)
            >>>
            >>> # More deterministic/focused
            >>> response = model.generate("Define photosynthesis.", temperature=0.1)

        Example - With Max Tokens:
            >>> # Limit response length
            >>> response = model.generate(
            ...     "Write a story about a robot.",
            ...     max_tokens=100
            ... )

        Example - With Stop Sequences:
            >>> response = model.generate(
            ...     "List three colors:",
            ...     stop=[".", "\\n"]  # Stop at period or newline
            ... )

        Example - JSON Mode (GPT-4-turbo):
            >>> response = model.generate(
            ...     "Return a JSON object with name and age",
            ...     response_format={"type": "json_object"}
            ... )
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except OpenAIRateLimitError as e:
            raise RateLimitError(
                model_id=self.model_name,
                retry_after=getattr(e, "retry_after", None),
            )
        except APITimeoutError:
            raise InsideLLMsTimeoutError(
                model_id=self.model_name,
                timeout_seconds=self._timeout,
            )
        except APIError as e:
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
        """Engage in a multi-turn chat conversation.

        Sends the full conversation history to OpenAI's chat completions API.
        This is the native mode for GPT models and enables context-aware responses.

        Args:
            messages: Conversation history as a list of ChatMessage dicts.
                Each message must have:
                - role (str): "system", "user", or "assistant"
                - content (str): The message content
                Optional: name (str) for named participants
            **kwargs: Additional OpenAI API parameters (see generate()).

        Returns:
            The model's text response to the conversation.

        Raises:
            RateLimitError: When API rate limits are exceeded.
            TimeoutError: When the request times out.
            APIError: For other API errors.
            ModelGenerationError: For unexpected errors.

        Example - Basic Chat:
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> messages = [
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> response = model.chat(messages)
            >>> print(response)
            'Hello! How can I assist you today?'

        Example - With System Prompt:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful tutor."},
            ...     {"role": "user", "content": "Explain recursion."},
            ... ]
            >>> response = model.chat(messages)

        Example - Multi-Turn Conversation:
            >>> conversation = [
            ...     {"role": "system", "content": "You answer in rhymes."},
            ...     {"role": "user", "content": "What is the weather?"},
            ...     {"role": "assistant", "content": "The sky is gray today..."},
            ...     {"role": "user", "content": "Will it rain?"},
            ... ]
            >>> response = model.chat(conversation)

        Example - Building a Chat Loop:
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> history = [
            ...     {"role": "system", "content": "You are a friendly assistant."}
            ... ]
            >>> while True:
            ...     user_input = input("You: ")
            ...     if user_input.lower() == "quit":
            ...         break
            ...     history.append({"role": "user", "content": user_input})
            ...     response = model.chat(history)
            ...     print(f"Assistant: {response}")
            ...     history.append({"role": "assistant", "content": response})
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except OpenAIRateLimitError as e:
            raise RateLimitError(
                model_id=self.model_name,
                retry_after=getattr(e, "retry_after", None),
            )
        except APITimeoutError:
            raise InsideLLMsTimeoutError(
                model_id=self.model_name,
                timeout_seconds=self._timeout,
            )
        except APIError as e:
            raise InsideLLMsAPIError(
                model_id=self.model_name,
                status_code=getattr(e, "status_code", None),
                message=str(e),
            )
        except Exception as e:
            # Get first message content for error context
            first_msg = messages[0]["content"] if messages else ""
            raise ModelGenerationError(
                model_id=self.model_name,
                prompt=first_msg,
                reason=str(e),
                original_error=e,
            )

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream the response from the model as it is generated.

        Returns a generator that yields response chunks (typically individual
        tokens or small groups) as they are produced by the model. This enables
        real-time display of responses to users.

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional OpenAI API parameters (see generate()).
                Note: The 'stream' parameter is automatically set to True.

        Yields:
            Response chunks as strings. Chunk boundaries are determined by
            the OpenAI API (usually token boundaries).

        Raises:
            RateLimitError: When API rate limits are exceeded.
            TimeoutError: When the request times out.
            APIError: For other API errors.
            ModelGenerationError: For unexpected errors.

        Example - Print in Real-Time:
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> for chunk in model.stream("Write a poem about coding"):
            ...     print(chunk, end="", flush=True)
            ... print()  # Final newline
            In lines of code we weave our art,
            Each function built with care and heart...

        Example - Collect While Streaming:
            >>> chunks = []
            >>> for chunk in model.stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
            ...     chunks.append(chunk)
            >>> full_response = "".join(chunks)

        Example - With Progress Indicator:
            >>> import sys
            >>> for i, chunk in enumerate(model.stream(prompt)):
            ...     sys.stdout.write(chunk)
            ...     if i % 50 == 0:
            ...         sys.stderr.write(".")  # Progress dot every 50 chunks

        Example - Time to First Token:
            >>> import time
            >>> start = time.time()
            >>> gen = model.stream("Hello")
            >>> first_chunk = next(gen)
            >>> ttft = time.time() - start
            >>> print(f"Time to first token: {ttft:.3f}s")

        Note:
            Streaming increases the total response time slightly but provides
            a better user experience by showing output immediately.
        """
        try:
            stream = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except OpenAIRateLimitError as e:
            raise RateLimitError(
                model_id=self.model_name,
                retry_after=getattr(e, "retry_after", None),
            )
        except APITimeoutError:
            raise InsideLLMsTimeoutError(
                model_id=self.model_name,
                timeout_seconds=self._timeout,
            )
        except APIError as e:
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
        """Return model metadata/info.

        Extends the base Model.info() with OpenAI-specific details.

        Returns:
            A ModelInfo object with additional 'extra' fields containing:
                - model_name: The OpenAI model identifier
                - description: Human-readable description
                - base_url: Custom API base URL (if set)
                - organization: OpenAI organization (if set)
                - project: OpenAI project (if set)

        Example:
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> info = model.info()
            >>> print(info.model_id)
            'gpt-4'
            >>> print(info.extra["description"])
            'OpenAI GPT model via API. Requires OPENAI_API_KEY env variable.'
        """
        base_info = super().info()
        base_info.extra.update(
            {
                "model_name": self.model_name,
                "description": (
                    "OpenAI GPT model via API. Requires an API key via api_key or api_key_env."
                ),
                "base_url": self._base_url,
                "organization": self._organization,
                "project": self._project,
                "api_key_env": self._api_key_env,
                "default_headers": self._redact_headers(self._default_headers),
            }
        )
        return base_info
