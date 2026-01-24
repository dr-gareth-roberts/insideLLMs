"""HuggingFace Transformers model wrapper for local LLM inference.

This module provides the ``HuggingFaceModel`` class, which enables running
HuggingFace Transformers models locally within the insideLLMs framework.
It wraps the Transformers ``pipeline`` API for text generation, supporting
both single-turn and multi-turn (chat) generation modes.

Key Features:
    - Local model execution without API calls
    - Support for any HuggingFace causal language model (GPT-2, LLaMA, etc.)
    - CPU and GPU device selection
    - Chat mode via message concatenation
    - Simulated streaming (yields full response as single chunk)
    - Comprehensive error handling with custom exceptions

Model Loading:
    Models are loaded via ``AutoModelForCausalLM`` and ``AutoTokenizer``,
    which automatically download models from the HuggingFace Hub if not
    cached locally. First-time loading of large models may take significant
    time and disk space.

Device Selection:
    - Use ``device=-1`` (default) for CPU inference
    - Use ``device=0`` for GPU 0, ``device=1`` for GPU 1, etc.
    - Ensure CUDA is available if selecting GPU devices

Example - Basic Usage with GPT-2:
    >>> from insideLLMs.models.huggingface import HuggingFaceModel
    >>>
    >>> # Load the default GPT-2 model (small, ~124M params)
    >>> model = HuggingFaceModel()
    >>> response = model.generate("The meaning of life is")
    >>> print(response)
    'The meaning of life is to find your gift. The purpose of life...'

Example - Using a Specific Model:
    >>> # Load a larger model for better quality
    >>> model = HuggingFaceModel(
    ...     name="my-gpt2-medium",
    ...     model_name="gpt2-medium",
    ...     device=0  # Use GPU 0
    ... )
    >>> response = model.generate("Python is a programming language that")
    >>> print(response)

Example - Local LLaMA Model:
    >>> # Load a locally saved or Hub-hosted LLaMA model
    >>> model = HuggingFaceModel(
    ...     name="llama-7b",
    ...     model_name="meta-llama/Llama-2-7b-hf",
    ...     device=0
    ... )
    >>> response = model.generate("Explain quantum computing:", max_new_tokens=100)

Example - Chat-Style Interaction:
    >>> model = HuggingFaceModel(model_name="gpt2")
    >>> messages = [
    ...     {"role": "system", "content": "You are a helpful assistant."},
    ...     {"role": "user", "content": "What is 2+2?"},
    ... ]
    >>> response = model.chat(messages, max_new_tokens=50)
    >>> print(response)

Example - Generation with Parameters:
    >>> model = HuggingFaceModel(model_name="gpt2")
    >>> response = model.generate(
    ...     "Once upon a time",
    ...     max_new_tokens=100,
    ...     temperature=0.8,
    ...     top_p=0.9,
    ...     do_sample=True
    ... )
    >>> print(response)

Example - Error Handling:
    >>> from insideLLMs.exceptions import ModelInitializationError, ModelGenerationError
    >>>
    >>> try:
    ...     model = HuggingFaceModel(model_name="nonexistent-model-xyz")
    ... except ModelInitializationError as e:
    ...     print(f"Failed to load model: {e.details['reason']}")
    >>>
    >>> model = HuggingFaceModel(model_name="gpt2")
    >>> try:
    ...     response = model.generate("Hello", max_length=-1)  # Invalid param
    ... except ModelGenerationError as e:
    ...     print(f"Generation failed: {e.details['reason']}")

Example - Using with Registry:
    >>> from insideLLMs.registry import model_registry
    >>>
    >>> # Register for easy access
    >>> model_registry.register("local-gpt2", HuggingFaceModel)
    >>>
    >>> # Get from registry with custom parameters
    >>> model = model_registry.get("local-gpt2", model_name="gpt2-large")

Example - Model Information:
    >>> model = HuggingFaceModel(model_name="gpt2")
    >>> info = model.info()
    >>> print(f"Provider: {info.provider}")  # "HuggingFace"
    >>> print(f"Model: {info.extra['model_name']}")  # "gpt2"
    >>> print(f"Device: {info.extra['device']}")  # -1 (CPU)
    >>> print(f"Supports streaming: {info.supports_streaming}")  # True
    >>> print(f"Supports chat: {info.supports_chat}")  # True

Performance Considerations:
    - First model load downloads weights from HuggingFace Hub (one-time)
    - GPU inference is significantly faster for large models
    - Consider quantized models (e.g., GPTQ, AWQ) for memory efficiency
    - Set ``max_new_tokens`` to limit generation length and time

See Also:
    - insideLLMs.models.base.Model: The abstract base class
    - insideLLMs.models.openai: For OpenAI API models
    - insideLLMs.models.anthropic: For Anthropic Claude models
    - HuggingFace Transformers: https://huggingface.co/docs/transformers
"""

from collections.abc import Iterator

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from insideLLMs.exceptions import (
    ModelGenerationError,
    ModelInitializationError,
)

from .base import ChatMessage, Model


class HuggingFaceModel(Model):
    """Model implementation for HuggingFace Transformers models.

    This class wraps HuggingFace Transformers models for local text generation,
    enabling you to run language models directly on your machine without API
    calls. It supports any causal language model available on the HuggingFace
    Hub or stored locally.

    The implementation uses the Transformers ``pipeline`` API for text
    generation, which handles tokenization, inference, and decoding in a
    unified interface. Error handling wraps all Transformers exceptions
    into insideLLMs custom exceptions for consistent error management.

    Attributes:
        name: Human-readable name for this model instance.
        model_id: The HuggingFace model identifier (same as model_name).
        model_name: The HuggingFace model identifier (e.g., "gpt2", "meta-llama/Llama-2-7b-hf").
        device: Device index for inference (-1 for CPU, 0+ for GPU).
        tokenizer: The loaded AutoTokenizer instance.
        model: The loaded AutoModelForCausalLM instance.
        generator: The text-generation pipeline instance.
        _supports_streaming: Always True (simulated streaming support).
        _supports_chat: Always True (chat mode via message concatenation).

    Example - Quick Start:
        >>> from insideLLMs.models.huggingface import HuggingFaceModel
        >>>
        >>> # Create a model with defaults (GPT-2 on CPU)
        >>> model = HuggingFaceModel()
        >>> response = model.generate("Hello, world!")
        >>> print(response)

    Example - Custom Model on GPU:
        >>> # Load a specific model on GPU 0
        >>> model = HuggingFaceModel(
        ...     name="local-mistral",
        ...     model_name="mistralai/Mistral-7B-v0.1",
        ...     device=0
        ... )
        >>> response = model.generate("Explain machine learning:", max_new_tokens=150)

    Example - Using Different Model Sizes:
        >>> # GPT-2 variants (increasing size/quality)
        >>> gpt2_small = HuggingFaceModel(model_name="gpt2")          # 124M params
        >>> gpt2_medium = HuggingFaceModel(model_name="gpt2-medium")  # 355M params
        >>> gpt2_large = HuggingFaceModel(model_name="gpt2-large")    # 774M params
        >>> gpt2_xl = HuggingFaceModel(model_name="gpt2-xl")          # 1.5B params

    Example - Multi-Turn Chat:
        >>> model = HuggingFaceModel(model_name="gpt2")
        >>> conversation = [
        ...     {"role": "system", "content": "You are a helpful coding assistant."},
        ...     {"role": "user", "content": "How do I read a file in Python?"},
        ... ]
        >>> response = model.chat(conversation, max_new_tokens=100)
        >>> # Continue the conversation
        >>> conversation.append({"role": "assistant", "content": response})
        >>> conversation.append({"role": "user", "content": "What about writing?"})
        >>> response = model.chat(conversation, max_new_tokens=100)

    Example - Streaming Output:
        >>> model = HuggingFaceModel(model_name="gpt2")
        >>> for chunk in model.stream("Once upon a time", max_new_tokens=50):
        ...     print(chunk, end="", flush=True)  # Note: yields entire response
        >>> print()

    Example - Generation with Sampling Parameters:
        >>> model = HuggingFaceModel(model_name="gpt2")
        >>>
        >>> # Creative writing with high temperature
        >>> creative = model.generate(
        ...     "Write a poem about AI:",
        ...     max_new_tokens=100,
        ...     temperature=1.2,
        ...     top_p=0.95,
        ...     do_sample=True
        ... )
        >>>
        >>> # Deterministic output with low temperature
        >>> factual = model.generate(
        ...     "The capital of France is",
        ...     max_new_tokens=20,
        ...     temperature=0.1,
        ...     do_sample=True
        ... )

    Example - Error Handling:
        >>> from insideLLMs.exceptions import ModelInitializationError, ModelGenerationError
        >>>
        >>> # Handle initialization errors
        >>> try:
        ...     model = HuggingFaceModel(model_name="invalid/model/path")
        ... except ModelInitializationError as e:
        ...     print(f"Could not load: {e.details['model_id']}")
        ...     print(f"Reason: {e.details['reason']}")
        >>>
        >>> # Handle generation errors
        >>> model = HuggingFaceModel()
        >>> try:
        ...     response = model.generate("Test", invalid_param=True)
        ... except ModelGenerationError as e:
        ...     print(f"Generation failed: {e}")
        ...     if e.original_error:
        ...         print(f"Caused by: {e.original_error}")

    Example - Inspecting Model Info:
        >>> model = HuggingFaceModel(model_name="gpt2-medium", device=0)
        >>> info = model.info()
        >>> print(f"Name: {info.name}")
        >>> print(f"Provider: {info.provider}")  # "HuggingFace"
        >>> print(f"Model ID: {info.model_id}")  # "gpt2-medium"
        >>> print(f"Streaming: {info.supports_streaming}")  # True
        >>> print(f"Chat: {info.supports_chat}")  # True
        >>> print(f"Device: {info.extra['device']}")  # 0

    Notes:
        - Streaming is simulated: the ``stream()`` method yields the entire
          response as a single chunk since Transformers pipelines don't
          natively support token-by-token streaming.
        - Chat mode concatenates messages with newlines, which works for
          simple interactions but may not be optimal for all models.
          Consider using instruction-tuned models for better chat quality.
        - Large models require significant memory. Consider using quantized
          versions (GPTQ, AWQ) or smaller models for limited hardware.
        - First initialization downloads model weights from HuggingFace Hub
          (can be several GB for large models).

    See Also:
        - :class:`insideLLMs.models.base.Model`: The abstract base class.
        - :class:`insideLLMs.models.openai.OpenAIModel`: For OpenAI API models.
        - HuggingFace Model Hub: https://huggingface.co/models
    """

    _supports_streaming = True
    _supports_chat = True

    def __init__(
        self,
        name: str = "HuggingFaceModel",
        model_name: str = "gpt2",
        device: int = -1,
    ):
        """Initialize a HuggingFace Transformers model.

        Loads the tokenizer, model weights, and creates a text-generation
        pipeline. The model is downloaded from the HuggingFace Hub on first
        use and cached locally for subsequent loads.

        Args:
            name: Human-readable name for this model instance. Used for
                identification in logs and model info. Defaults to
                "HuggingFaceModel".
            model_name: The HuggingFace model identifier. Can be:
                - A Hub model ID (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
                - A local path to saved model weights
                - Any model compatible with AutoModelForCausalLM
                Defaults to "gpt2".
            device: The device index for inference:
                - -1: CPU inference (default, works everywhere)
                - 0: First GPU (CUDA device 0)
                - 1, 2, etc.: Additional GPUs
                Defaults to -1 (CPU).

        Raises:
            ModelInitializationError: If any of the following fail:
                - Tokenizer loading (invalid model ID, network error)
                - Model loading (invalid model ID, out of memory, network error)
                - Pipeline creation (device not available, incompatible model)
                The exception's ``details`` dict contains 'model_id' and 'reason'.

        Example - Default Initialization:
            >>> # Loads GPT-2 on CPU
            >>> model = HuggingFaceModel()
            >>> print(model.name)  # "HuggingFaceModel"
            >>> print(model.model_name)  # "gpt2"
            >>> print(model.device)  # -1

        Example - Custom Name and Model:
            >>> model = HuggingFaceModel(
            ...     name="creative-writer",
            ...     model_name="gpt2-large",
            ...     device=-1
            ... )
            >>> print(model.name)  # "creative-writer"
            >>> print(model.model_id)  # "gpt2-large"

        Example - GPU Inference:
            >>> # Requires CUDA-enabled GPU
            >>> model = HuggingFaceModel(
            ...     model_name="mistralai/Mistral-7B-v0.1",
            ...     device=0  # First GPU
            ... )

        Example - Local Model Path:
            >>> # Load from a local directory
            >>> model = HuggingFaceModel(
            ...     model_name="/path/to/my/saved/model",
            ...     device=0
            ... )

        Example - Handling Initialization Errors:
            >>> from insideLLMs.exceptions import ModelInitializationError
            >>> try:
            ...     model = HuggingFaceModel(model_name="nonexistent/model")
            ... except ModelInitializationError as e:
            ...     print(f"Failed: {e.message}")
            ...     print(f"Model ID: {e.details['model_id']}")
            ...     print(f"Reason: {e.details['reason']}")

        Note:
            The initialization process occurs in three stages:
            1. Tokenizer loading via ``AutoTokenizer.from_pretrained()``
            2. Model loading via ``AutoModelForCausalLM.from_pretrained()``
            3. Pipeline creation via ``pipeline("text-generation", ...)``

            Each stage can fail independently, and the error message will
            indicate which stage failed.
        """
        super().__init__(name=name, model_id=model_name)
        self.model_name = model_name
        self.device = device

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise ModelInitializationError(
                model_id=model_name,
                reason=f"Failed to load tokenizer: {e}",
            )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            raise ModelInitializationError(
                model_id=model_name,
                reason=f"Failed to load model: {e}",
            )

        try:
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
            )
        except Exception as e:
            raise ModelInitializationError(
                model_id=model_name,
                reason=f"Failed to create pipeline: {e}",
            )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion for the given prompt.

        Sends the prompt through the HuggingFace text-generation pipeline
        and returns the generated text. The output includes the original
        prompt followed by the model's continuation.

        Args:
            prompt: The input text to continue. The model will generate
                text that follows this prompt naturally.
            **kwargs: Additional generation parameters passed to the
                HuggingFace pipeline. Common parameters include:
                - max_new_tokens (int): Maximum number of new tokens to generate.
                    Recommended over max_length for predictable output length.
                - max_length (int): Maximum total length (prompt + generated).
                    Default varies by model (usually 50).
                - temperature (float): Sampling temperature (0.0-2.0).
                    Higher values increase randomness. Requires do_sample=True.
                - top_p (float): Nucleus sampling probability (0.0-1.0).
                    Requires do_sample=True.
                - top_k (int): Top-k sampling. Only sample from top k tokens.
                - do_sample (bool): Whether to use sampling. If False, uses
                    greedy decoding. Default is False.
                - num_return_sequences (int): Number of sequences to generate.
                    Default is 1.
                - repetition_penalty (float): Penalty for token repetition.
                - pad_token_id (int): Padding token ID (often set to eos_token_id).

        Returns:
            The generated text as a string, including the original prompt.
            To get only the new text, slice off the prompt length:
            ``response[len(prompt):]``

        Raises:
            ModelGenerationError: If text generation fails for any reason,
                including invalid parameters, out-of-memory errors, or
                internal model errors. The exception contains:
                - model_id: The HuggingFace model identifier
                - prompt_preview: First 100 chars of the prompt
                - reason: Description of what went wrong
                - original_error: The underlying exception

        Example - Basic Generation:
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>> response = model.generate("The quick brown fox")
            >>> print(response)
            'The quick brown fox jumps over the lazy dog and runs away.'

        Example - Controlling Output Length:
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>>
            >>> # Generate exactly 50 new tokens
            >>> response = model.generate(
            ...     "Once upon a time",
            ...     max_new_tokens=50
            ... )
            >>>
            >>> # Or set total length (prompt + generated)
            >>> response = model.generate(
            ...     "Once upon a time",
            ...     max_length=100
            ... )

        Example - Creative Generation with Sampling:
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>> response = model.generate(
            ...     "Write a haiku about programming:",
            ...     max_new_tokens=30,
            ...     temperature=0.9,
            ...     top_p=0.95,
            ...     do_sample=True
            ... )

        Example - Deterministic Generation:
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>>
            >>> # Greedy decoding (always same output)
            >>> response1 = model.generate("2 + 2 =", max_new_tokens=5)
            >>> response2 = model.generate("2 + 2 =", max_new_tokens=5)
            >>> assert response1 == response2  # Deterministic

        Example - Extracting Only New Text:
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>> prompt = "The answer is"
            >>> full_response = model.generate(prompt, max_new_tokens=10)
            >>> new_text = full_response[len(prompt):]
            >>> print(f"Generated: {new_text}")

        Example - Handling Errors:
            >>> from insideLLMs.exceptions import ModelGenerationError
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>> try:
            ...     response = model.generate("Test", max_length=-1)
            ... except ModelGenerationError as e:
            ...     print(f"Error: {e.details['reason']}")
            ...     print(f"Original: {e.original_error}")

        Note:
            The returned text includes the original prompt. This is the
            standard behavior for HuggingFace text-generation pipelines.
            If you need only the generated portion, slice it manually.
        """
        try:
            outputs = self.generator(prompt, **kwargs)
            return outputs[0]["generated_text"]
        except Exception as e:
            raise ModelGenerationError(
                model_id=self.model_name,
                prompt=prompt,
                reason=str(e),
                original_error=e,
            )

    def chat(self, messages: list[ChatMessage], **kwargs) -> str:
        """Engage in a multi-turn chat conversation.

        Converts a list of chat messages into a prompt by concatenating
        the message contents with newlines, then generates a response.
        This provides a simple chat interface for models that don't have
        native chat support.

        The implementation extracts the 'content' field from each message
        and joins them with newlines. Role information (system/user/assistant)
        is not explicitly formatted, so the model sees a plain text conversation.
        For better chat quality, consider using instruction-tuned models that
        understand conversational context.

        Args:
            messages: A list of ChatMessage dictionaries, each containing:
                - role (str): The speaker role ("system", "user", "assistant").
                    Note: Role is not included in the formatted prompt.
                - content (str): The message text. This is what gets included.
                - name (str, optional): Speaker name (not used in formatting).
            **kwargs: Additional generation parameters passed to the pipeline.
                Same parameters as generate(): max_new_tokens, temperature, etc.

        Returns:
            The generated response as a string. Note that the response includes
            all the concatenated input messages followed by the model's
            continuation. To extract only the new response, you'll need to
            track the length of your input.

        Raises:
            ModelGenerationError: If generation fails. The exception contains:
                - model_id: The HuggingFace model identifier
                - prompt_preview: First message content (for debugging)
                - reason: Description of what went wrong
                - original_error: The underlying exception

        Example - Simple Chat:
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>> messages = [
            ...     {"role": "user", "content": "Hello, how are you?"}
            ... ]
            >>> response = model.chat(messages, max_new_tokens=50)
            >>> print(response)

        Example - Multi-Turn Conversation:
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>> conversation = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "What is Python?"},
            ...     {"role": "assistant", "content": "Python is a programming language."},
            ...     {"role": "user", "content": "What can I do with it?"},
            ... ]
            >>> response = model.chat(conversation, max_new_tokens=100)

        Example - Building a Conversation Loop:
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>> history = [{"role": "system", "content": "You are a math tutor."}]
            >>>
            >>> def ask(question):
            ...     history.append({"role": "user", "content": question})
            ...     response = model.chat(history, max_new_tokens=50)
            ...     # Extract new content (simple heuristic)
            ...     history.append({"role": "assistant", "content": response})
            ...     return response
            >>>
            >>> ask("What is 2 + 2?")
            >>> ask("And if I add 3 more?")

        Example - With Generation Parameters:
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>> messages = [
            ...     {"role": "user", "content": "Tell me a creative story."}
            ... ]
            >>> response = model.chat(
            ...     messages,
            ...     max_new_tokens=150,
            ...     temperature=0.9,
            ...     top_p=0.95,
            ...     do_sample=True
            ... )

        Example - Error Handling:
            >>> from insideLLMs.exceptions import ModelGenerationError
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>> try:
            ...     response = model.chat([{"role": "user", "content": "Hi"}], max_length=-1)
            ... except ModelGenerationError as e:
            ...     print(f"Chat failed: {e.details['reason']}")

        Note:
            This is a simplified chat implementation that concatenates
            messages. It does not use special chat tokens or formatting
            that instruction-tuned models might expect. For models with
            native chat templates (like Llama-2-Chat), consider using
            the tokenizer's ``apply_chat_template()`` method for better
            results.

            The response includes all input messages. If you need only
            the model's new response, you'll need to parse or slice it.
        """
        try:
            # Simple chat: concatenate messages
            prompt = "\n".join([m.get("content", "") for m in messages])
            outputs = self.generator(prompt, **kwargs)
            return outputs[0]["generated_text"]
        except Exception as e:
            first_msg = messages[0]["content"] if messages else ""
            raise ModelGenerationError(
                model_id=self.model_name,
                prompt=first_msg,
                reason=str(e),
                original_error=e,
            )

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream the response from the model (simulated).

        Provides a streaming-compatible interface for the HuggingFace model.
        Since the Transformers pipeline API does not natively support
        token-by-token streaming, this method generates the complete
        response and yields it as a single chunk.

        This allows HuggingFaceModel to be used in code that expects
        streaming support, though without the latency benefits of true
        streaming. The interface is compatible with StreamingModelProtocol.

        Args:
            prompt: The input text to generate a continuation for.
            **kwargs: Additional generation parameters passed to the pipeline.
                Same parameters as generate(): max_new_tokens, temperature, etc.

        Yields:
            str: The complete generated response as a single chunk.
                Unlike true streaming which yields tokens incrementally,
                this yields the entire response at once.

        Raises:
            ModelGenerationError: If generation fails. The exception contains:
                - model_id: The HuggingFace model identifier
                - prompt_preview: First 100 chars of the prompt
                - reason: Description of what went wrong
                - original_error: The underlying exception

        Example - Basic Streaming:
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>> for chunk in model.stream("Once upon a time", max_new_tokens=50):
            ...     print(chunk, end="", flush=True)
            >>> print()  # Final newline
            Once upon a time, there was a little girl who lived in a...

        Example - Collecting Streamed Output:
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>> chunks = list(model.stream("Hello world", max_new_tokens=30))
            >>> assert len(chunks) == 1  # Only one chunk (simulated streaming)
            >>> full_response = chunks[0]

        Example - Compatible with Streaming Protocols:
            >>> from insideLLMs.models.base import StreamingModelProtocol
            >>>
            >>> def process_stream(model: StreamingModelProtocol, prompt: str):
            ...     '''Works with any streaming model.'''
            ...     result = []
            ...     for chunk in model.stream(prompt, max_new_tokens=50):
            ...         print(chunk, end="", flush=True)
            ...         result.append(chunk)
            ...     return "".join(result)
            >>>
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>> output = process_stream(model, "Test prompt")

        Example - With Progress Callback:
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>>
            >>> def on_chunk(chunk):
            ...     print(f"Received {len(chunk)} characters")
            >>>
            >>> for chunk in model.stream("Generate text", max_new_tokens=100):
            ...     on_chunk(chunk)  # Called once with full response

        Example - Error Handling:
            >>> from insideLLMs.exceptions import ModelGenerationError
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>> try:
            ...     for chunk in model.stream("Test", max_length=-1):
            ...         print(chunk)
            ... except ModelGenerationError as e:
            ...     print(f"Stream failed: {e.details['reason']}")

        Note:
            This is simulated streaming. The method waits for the entire
            generation to complete before yielding any output. For
            applications requiring true token-by-token streaming, consider
            using the Transformers TextIteratorStreamer with a separate
            thread, or switching to an API-based model (OpenAI, Anthropic)
            that supports native streaming.

            Despite being simulated, this method allows HuggingFaceModel
            to work in codebases designed for streaming interfaces without
            modification.
        """
        try:
            # Streaming not natively supported; yield the full output as one chunk
            outputs = self.generator(prompt, **kwargs)
            yield outputs[0]["generated_text"]
        except Exception as e:
            raise ModelGenerationError(
                model_id=self.model_name,
                prompt=prompt,
                reason=str(e),
                original_error=e,
            )

    def info(self):
        """Return model metadata and configuration information.

        Extends the base Model.info() method with HuggingFace-specific
        details including the model name, device configuration, and
        a description of the model type.

        Returns:
            ModelInfo: A dataclass containing:
                - name (str): The instance name (e.g., "HuggingFaceModel")
                - provider (str): Always "HuggingFace" for this class
                - model_id (str): The HuggingFace model identifier
                - supports_streaming (bool): True (simulated streaming)
                - supports_chat (bool): True (via message concatenation)
                - extra (dict): HuggingFace-specific metadata:
                    - model_name: The HuggingFace model identifier
                    - device: Device index (-1 for CPU, 0+ for GPU)
                    - description: Human-readable model description

        Example - Basic Info Retrieval:
            >>> model = HuggingFaceModel(model_name="gpt2")
            >>> info = model.info()
            >>> print(info.name)
            'HuggingFaceModel'
            >>> print(info.provider)
            'HuggingFace'
            >>> print(info.model_id)
            'gpt2'

        Example - Checking Capabilities:
            >>> model = HuggingFaceModel(model_name="gpt2-medium")
            >>> info = model.info()
            >>> if info.supports_streaming:
            ...     print("Model supports streaming (simulated)")
            >>> if info.supports_chat:
            ...     print("Model supports chat mode")

        Example - Accessing Extra Metadata:
            >>> model = HuggingFaceModel(
            ...     name="my-local-model",
            ...     model_name="gpt2-large",
            ...     device=0
            ... )
            >>> info = model.info()
            >>> print(f"HF Model: {info.extra['model_name']}")
            HF Model: gpt2-large
            >>> print(f"Device: {info.extra['device']}")
            Device: 0
            >>> print(f"Type: {info.extra['description']}")
            Type: HuggingFace Transformers model via pipeline.

        Example - Model Comparison:
            >>> models = [
            ...     HuggingFaceModel(model_name="gpt2"),
            ...     HuggingFaceModel(model_name="gpt2-medium"),
            ... ]
            >>> for m in models:
            ...     info = m.info()
            ...     print(f"{info.provider}/{info.model_id}: "
            ...           f"device={info.extra['device']}")
            HuggingFace/gpt2: device=-1
            HuggingFace/gpt2-medium: device=-1

        Example - Logging Model Configuration:
            >>> import logging
            >>> model = HuggingFaceModel(model_name="gpt2", device=0)
            >>> info = model.info()
            >>> logging.info(
            ...     "Loaded model",
            ...     extra={
            ...         "provider": info.provider,
            ...         "model_id": info.model_id,
            ...         **info.extra
            ...     }
            ... )

        Note:
            The 'extra' dict provides HuggingFace-specific information
            beyond the standard ModelInfo fields. This can be useful
            for logging, debugging, or building model management UIs.
        """
        base_info = super().info()
        base_info.extra.update(
            {
                "model_name": self.model_name,
                "device": self.device,
                "description": "HuggingFace Transformers model via pipeline.",
            }
        )
        return base_info
