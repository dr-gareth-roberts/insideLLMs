"""Public model message types and structural protocols."""

from collections.abc import Iterator, Sequence
from typing import Any, Protocol, TypedDict, runtime_checkable


class _ChatMessageRequired(TypedDict):
    """Required fields for a chat message."""

    role: str
    content: str


class ChatMessage(_ChatMessageRequired, total=False):
    """A single message in a chat conversation.

    Represents one turn in a multi-turn chat, following the standard
    message format used by most LLM APIs (OpenAI, Anthropic, etc.).

    Attributes:
        role: The role of the message sender. Standard values are:
            - "system": System instructions that set behavior
            - "user": Messages from the human user
            - "assistant": Previous responses from the model
        content: The text content of the message.
        name: Optional name identifier for the speaker. Useful when
            simulating multi-party conversations or providing context.

    Example - Simple User Message:
        >>> message: ChatMessage = {
        ...     "role": "user",
        ...     "content": "What is the capital of France?"
        ... }

    Example - System Prompt:
        >>> system: ChatMessage = {
        ...     "role": "system",
        ...     "content": "You are a helpful geography assistant."
        ... }

    Example - Full Conversation:
        >>> conversation: list[ChatMessage] = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": "Hi! How can I help?"},
        ...     {"role": "user", "content": "What's 2+2?"},
        ... ]
        >>>
        >>> # Use with a chat-capable model
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> response = model.chat(conversation)

    Example - Named Participants:
        >>> messages: list[ChatMessage] = [
        ...     {"role": "user", "name": "Alice", "content": "I think it's blue."},
        ...     {"role": "user", "name": "Bob", "content": "I disagree, it's green."},
        ... ]

    Note:
        Not all models support the 'name' field. Check your model's
        documentation for supported message fields.
    """

    name: str


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol defining the interface for language models.

    Use this for type hints when you want to accept any model-like object.
    This enables duck typing - any object with a `name` attribute and
    `generate()` / `info()` methods is considered a valid model.

    The @runtime_checkable decorator allows isinstance() checks at runtime.

    Attributes:
        name: Human-readable identifier for the model.

    Methods:
        generate: Generate text from a prompt.
        info: Return model metadata.

    Example - Type Hints in Functions:
        >>> def evaluate_model(model: ModelProtocol, prompts: list[str]) -> list[str]:
        ...     '''Works with any model implementing the protocol.'''
        ...     return [model.generate(p) for p in prompts]
        >>>
        >>> # Works with OpenAIModel, AnthropicModel, or any custom model
        >>> results = evaluate_model(my_model, ["Hello", "Goodbye"])

    Example - Runtime Type Checking:
        >>> from insideLLMs.models.base import ModelProtocol
        >>> from insideLLMs.models import OpenAIModel
        >>>
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> isinstance(model, ModelProtocol)
        True

    Example - Creating a Compatible Custom Class:
        >>> class SimpleModel:
        ...     name = "simple"
        ...
        ...     def generate(self, prompt: str, **kwargs) -> str:
        ...         return "Hello!"
        ...
        ...     def info(self) -> dict:
        ...         return {"name": self.name}
        >>>
        >>> isinstance(SimpleModel(), ModelProtocol)
        True
    """

    name: str

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the model given a prompt.

        Args:
            prompt: The input text to send to the model.
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.).

        Returns:
            The model's text response.
        """
        ...

    def info(self) -> dict[str, Any]:
        """Return model metadata/info as a dict.

        Returns:
            Dictionary containing at minimum the model name and provider.
        """
        ...


@runtime_checkable
class BatchModelProtocol(ModelProtocol, Protocol):
    """Protocol for models that support batch generation.

    Extends ModelProtocol with batch_generate() for processing multiple
    prompts efficiently. Some providers offer native batch APIs that are
    faster and cheaper than sequential calls.

    Example - Batch Processing:
        >>> def process_dataset(
        ...     model: BatchModelProtocol,
        ...     prompts: list[str]
        ... ) -> list[str]:
        ...     '''Process prompts in batches for efficiency.'''
        ...     return model.batch_generate(prompts)
        >>>
        >>> prompts = ["Translate 'hello'", "Translate 'goodbye'"]
        >>> results = process_dataset(model, prompts)

    Example - Check Batch Support:
        >>> if isinstance(model, BatchModelProtocol):
        ...     # Use efficient batch API
        ...     results = model.batch_generate(prompts)
        ... else:
        ...     # Fall back to sequential
        ...     results = [model.generate(p) for p in prompts]
    """

    def batch_generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate responses for multiple prompts in a batch.

        Args:
            prompts: List of input prompts.
            **kwargs: Provider-specific parameters applied to all prompts.

        Returns:
            List of responses, one for each input prompt (in order).
        """
        ...


@runtime_checkable
class ChatModelProtocol(ModelProtocol, Protocol):
    """Protocol for models that support multi-turn chat.

    Extends ModelProtocol with chat() for maintaining conversation context.
    Most modern LLMs (GPT-4, Claude, etc.) support chat mode.

    Example - Multi-Turn Conversation:
        >>> def have_conversation(
        ...     model: ChatModelProtocol,
        ...     messages: list[ChatMessage]
        ... ) -> str:
        ...     '''Continue a conversation.'''
        ...     return model.chat(messages)
        >>>
        >>> history = [
        ...     {"role": "system", "content": "You are a tutor."},
        ...     {"role": "user", "content": "Explain recursion."},
        ... ]
        >>> response = have_conversation(model, history)

    Example - Building a Chatbot:
        >>> class ChatBot:
        ...     def __init__(self, model: ChatModelProtocol):
        ...         self.model = model
        ...         self.history: list[ChatMessage] = []
        ...
        ...     def send(self, message: str) -> str:
        ...         self.history.append({"role": "user", "content": message})
        ...         response = self.model.chat(self.history)
        ...         self.history.append({"role": "assistant", "content": response})
        ...         return response
    """

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> str:
        """Engage in a multi-turn chat.

        Args:
            messages: Conversation history as a list of ChatMessage dicts.
            **kwargs: Provider-specific parameters.

        Returns:
            The model's response to the conversation.
        """
        ...


@runtime_checkable
class StreamingModelProtocol(ModelProtocol, Protocol):
    """Protocol for models that support streaming responses.

    Extends ModelProtocol with stream() for receiving responses token-by-token.
    Useful for displaying real-time output to users.

    Example - Streaming to Console:
        >>> def stream_response(model: StreamingModelProtocol, prompt: str):
        ...     '''Print response as it's generated.'''
        ...     for chunk in model.stream(prompt):
        ...         print(chunk, end="", flush=True)
        ...     print()  # Final newline

    Example - Collecting Streamed Response:
        >>> def stream_and_collect(
        ...     model: StreamingModelProtocol,
        ...     prompt: str
        ... ) -> str:
        ...     '''Stream while also collecting the full response.'''
        ...     chunks = []
        ...     for chunk in model.stream(prompt):
        ...         print(chunk, end="", flush=True)
        ...         chunks.append(chunk)
        ...     return "".join(chunks)

    Example - With Timeout:
        >>> import itertools
        >>> def stream_with_limit(model: StreamingModelProtocol, prompt: str, max_chunks: int):
        ...     '''Stream up to max_chunks.'''
        ...     return list(itertools.islice(model.stream(prompt), max_chunks))
    """

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the response from the model.

        Args:
            prompt: The input prompt.
            **kwargs: Provider-specific parameters.

        Yields:
            Response chunks (typically tokens or small groups of tokens).
        """
        ...


@runtime_checkable
class AsyncModelProtocol(Protocol):
    """Protocol for models that support async operations.

    Use this for concurrent model calls with asyncio. Essential for
    high-throughput applications and parallel evaluations.

    Example - Parallel Generation:
        >>> import asyncio
        >>>
        >>> async def parallel_generate(
        ...     model: AsyncModelProtocol,
        ...     prompts: list[str]
        ... ) -> list[str]:
        ...     '''Generate responses concurrently.'''
        ...     tasks = [model.agenerate(p) for p in prompts]
        ...     return await asyncio.gather(*tasks)
        >>>
        >>> # Run 100 prompts concurrently
        >>> results = asyncio.run(parallel_generate(model, prompts))

    Example - With Semaphore Rate Limiting:
        >>> async def rate_limited_generate(
        ...     model: AsyncModelProtocol,
        ...     prompts: list[str],
        ...     max_concurrent: int = 10
        ... ) -> list[str]:
        ...     '''Limit concurrent requests to avoid rate limits.'''
        ...     sem = asyncio.Semaphore(max_concurrent)
        ...
        ...     async def limited_call(prompt: str) -> str:
        ...         async with sem:
        ...             return await model.agenerate(prompt)
        ...
        ...     return await asyncio.gather(*[limited_call(p) for p in prompts])
    """

    name: str

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate a response.

        Args:
            prompt: The input prompt.
            **kwargs: Provider-specific parameters.

        Returns:
            The model's text response.
        """
        ...
