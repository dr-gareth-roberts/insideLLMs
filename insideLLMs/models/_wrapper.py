"""Retrying and caching wrapper for model implementations."""

from typing import Any

from insideLLMs.models.base import Model
from insideLLMs.types import ModelInfo


class ModelWrapper:
    """Wrapper that adds common functionality to any model.

    Provides features like retry logic with exponential backoff and
    response caching that work with any Model implementation. Use this
    to make models more robust in production environments.

    Key features:
        - Automatic retry on failures with configurable backoff
        - Optional in-memory response caching for identical prompts
        - Preserves the underlying model's interface (ModelProtocol compliant)
        - Transparent - can be used wherever a Model is expected

    Attributes:
        name: Proxied from the underlying model.
        _model: The wrapped model instance.
        _max_retries: Maximum retry attempts.
        _retry_delay: Base delay between retries (seconds).
        _cache_responses: Whether caching is enabled.
        _cache: Internal response cache dictionary.

    Example - Basic Retry:
        >>> from insideLLMs.models import OpenAIModel
        >>> from insideLLMs.models.base import ModelWrapper
        >>>
        >>> base_model = OpenAIModel(model_name="gpt-4")
        >>> model = ModelWrapper(base_model, max_retries=3)
        >>>
        >>> # Now has automatic retry on API errors
        >>> response = model.generate("What is 2+2?")

    Example - With Caching:
        >>> model = ModelWrapper(
        ...     OpenAIModel(model_name="gpt-4"),
        ...     cache_responses=True
        ... )
        >>>
        >>> # First call hits the API
        >>> response1 = model.generate("Hello!")
        >>>
        >>> # Second identical call returns cached result (no API call)
        >>> response2 = model.generate("Hello!")
        >>> assert response1 == response2

    Example - Production Configuration:
        >>> # Robust production setup
        >>> model = ModelWrapper(
        ...     OpenAIModel(model_name="gpt-4"),
        ...     max_retries=5,      # More retries for resilience
        ...     retry_delay=2.0,    # Longer initial delay
        ...     cache_responses=True # Cache for cost savings
        ... )

    Example - Wrapping for Testing:
        >>> # Use wrapper to make tests more stable
        >>> def create_test_model():
        ...     base = OpenAIModel(model_name="gpt-4")
        ...     return ModelWrapper(
        ...         base,
        ...         max_retries=3,
        ...         cache_responses=True  # Faster re-runs
        ...     )

    Example - Type Compatibility:
        >>> from insideLLMs.models.base import ModelProtocol
        >>>
        >>> def run_evaluation(model: ModelProtocol, prompts: list):
        ...     '''Works with both raw models and wrapped models.'''
        ...     return [model.generate(p) for p in prompts]
        >>>
        >>> # Both work identically
        >>> raw_model = OpenAIModel(model_name="gpt-4")
        >>> wrapped_model = ModelWrapper(raw_model, max_retries=3)
        >>>
        >>> results1 = run_evaluation(raw_model, prompts)
        >>> results2 = run_evaluation(wrapped_model, prompts)

    Note:
        The retry delay uses linear backoff: delay * (attempt + 1).
        For exponential backoff, subclass and override generate().
    """

    def __init__(
        self,
        model: Model,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_responses: bool = False,
    ):
        """Initialize the wrapper.

        Creates a wrapper around an existing model that adds retry logic
        and optional caching functionality.

        Args:
            model: The underlying model to wrap. Can be any Model subclass
                (OpenAIModel, AnthropicModel, etc.) or another wrapper.
            max_retries: Maximum number of retry attempts on failure.
                Default is 3. Set to 1 for no retries.
            retry_delay: Base delay in seconds between retries.
                Actual delay is retry_delay * (attempt + 1) for linear backoff.
                Default is 1.0 second.
            cache_responses: Whether to cache responses for identical prompts.
                Default is False. When True, the same prompt+kwargs combination
                returns the cached result without making an API call.

        Example - Default Settings:
            >>> wrapper = ModelWrapper(OpenAIModel(model_name="gpt-4"))
            >>> # Uses max_retries=3, retry_delay=1.0, no caching

        Example - High Reliability:
            >>> wrapper = ModelWrapper(
            ...     model=OpenAIModel(model_name="gpt-4"),
            ...     max_retries=5,
            ...     retry_delay=2.0
            ... )

        Example - Development with Caching:
            >>> wrapper = ModelWrapper(
            ...     model=OpenAIModel(model_name="gpt-4"),
            ...     max_retries=1,       # Fail fast in dev
            ...     cache_responses=True # Don't waste API calls
            ... )

        Example - Nested Wrapping (not recommended but possible):
            >>> base = OpenAIModel(model_name="gpt-4")
            >>> with_retry = ModelWrapper(base, max_retries=3)
            >>> with_cache = ModelWrapper(with_retry, cache_responses=True)

        Note:
            The cache is in-memory and per-instance. It's not shared between
            wrapper instances and is lost when the wrapper is garbage collected.
        """
        self._model = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._cache_responses = cache_responses
        self._cache: dict[str, str] = {}

    @property
    def name(self) -> str:
        """Return the name of the underlying model.

        This property proxies to the wrapped model's name, making the
        wrapper transparent in code that checks model names.

        Returns:
            The name of the underlying model.

        Example:
            >>> base = OpenAIModel(model_name="gpt-4")
            >>> wrapper = ModelWrapper(base)
            >>> print(wrapper.name)
            'gpt-4'
            >>> assert wrapper.name == base.name
        """
        return self._model.name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate with retry logic and optional caching.

        Calls the underlying model's generate() method, with automatic
        retry on failures and optional caching of results.

        The retry logic uses linear backoff: wait time increases by
        retry_delay seconds with each attempt (1x, 2x, 3x, etc.).

        Args:
            prompt: The input prompt to send to the model.
            **kwargs: Additional arguments passed to the underlying model.

        Returns:
            The model's text response.

        Raises:
            Exception: Re-raises the last error after all retries are exhausted.
            RuntimeError: If max_retries is reached with no specific error.

        Example - Basic Usage:
            >>> model = ModelWrapper(OpenAIModel(model_name="gpt-4"), max_retries=3)
            >>> response = model.generate("Hello!")

        Example - Caching Behavior:
            >>> model = ModelWrapper(base_model, cache_responses=True)
            >>> r1 = model.generate("Hello", temperature=0.5)  # API call
            >>> r2 = model.generate("Hello", temperature=0.5)  # Cached
            >>> r3 = model.generate("Hello", temperature=0.7)  # API call (different kwargs)

        Example - Retry Timing:
            >>> # With retry_delay=1.0, max_retries=3:
            >>> # Attempt 1: immediate
            >>> # Attempt 2: wait 1 second
            >>> # Attempt 3: wait 2 seconds
            >>> # Total max wait: 3 seconds

        Note:
            Cache keys include both the prompt and kwargs, so the same
            prompt with different parameters is cached separately.
        """
        import time

        cache_key = f"{prompt}:{sorted(kwargs.items())}"

        if self._cache_responses and cache_key in self._cache:
            return self._cache[cache_key]

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                result = self._model.generate(prompt, **kwargs)
                if self._cache_responses:
                    self._cache[cache_key] = result
                return result
            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))

        if last_error is None:
            raise RuntimeError("Max retries exceeded")
        raise last_error

    def info(self) -> ModelInfo:
        """Return model metadata/info from the underlying model.

        Proxies to the wrapped model's info() method, providing
        transparency for code that inspects model capabilities.

        Returns:
            A ModelInfo object from the underlying model.

        Example:
            >>> base = OpenAIModel(model_name="gpt-4")
            >>> wrapper = ModelWrapper(base)
            >>> info = wrapper.info()
            >>> print(info.provider)
            'OpenAI'
        """
        return self._model.info()

    def __getattr__(self, name: str) -> Any:
        """Delegate attributes not defined on the wrapper to the wrapped model.

        Keeps the wrapper transparent so methods like ``chat``, ``stream`` and
        ``batch_generate`` reach the underlying model instead of raising
        AttributeError. Dunder lookups are excluded to avoid interfering with
        copy/pickle protocols and to prevent recursion before ``_model`` is set.
        """
        # Guard against recursion when `_model` is not yet set (e.g. instances
        # created via __new__/unpickle/copy without __init__): accessing
        # self._model would re-enter __getattr__ for "_model" forever.
        if name == "_model" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self._model, name)

    def __repr__(self) -> str:
        """Return a string representation of the wrapper.

        Shows the wrapper class name, the wrapped model, and key
        configuration parameters.

        Returns:
            A string like 'ModelWrapper(OpenAIModel(...), max_retries=3)'.

        Example:
            >>> base = OpenAIModel(model_name="gpt-4")
            >>> wrapper = ModelWrapper(base, max_retries=5)
            >>> print(repr(wrapper))
            ModelWrapper(OpenAIModel(name='gpt-4', model_id='gpt-4'), max_retries=5)
        """
        return f"ModelWrapper({self._model!r}, max_retries={self._max_retries})"
