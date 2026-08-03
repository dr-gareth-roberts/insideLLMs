"""Provider exception translation shared by model adapters."""

from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class ProviderExceptionMap:
    """Maps provider-specific exceptions to insideLLMs exceptions.

    This class provides a standardized way to define exception mappings for
    different LLM providers. It reduces code duplication across model
    implementations by centralizing error translation logic.

    Parameters
    ----------
    rate_limit_errors : tuple[type, ...]
        Provider exception types that indicate rate limiting.
    timeout_errors : tuple[type, ...]
        Provider exception types that indicate timeouts.
    api_errors : tuple[type, ...]
        Provider exception types for general API errors.

    Example - Creating an Exception Map:
        >>> from openai import RateLimitError, APITimeoutError, APIError
        >>> openai_map = ProviderExceptionMap(
        ...     rate_limit_errors=(RateLimitError,),
        ...     timeout_errors=(APITimeoutError,),
        ...     api_errors=(APIError,),
        ... )

    Example - Using with handle_provider_errors:
        >>> @handle_provider_errors(openai_map)
        ... def generate(self, prompt: str, **kwargs) -> str:
        ...     return self._client.chat.completions.create(...)
    """

    def __init__(
        self,
        rate_limit_errors: tuple[type[BaseException], ...] = (),
        timeout_errors: tuple[type[BaseException], ...] = (),
        api_errors: tuple[type[BaseException], ...] = (),
    ):
        self.rate_limit_errors = rate_limit_errors
        self.timeout_errors = timeout_errors
        self.api_errors = api_errors


def handle_provider_errors(
    exception_map: ProviderExceptionMap,
    *,
    get_model_id: Callable[[Any], str] | None = None,
    get_timeout: Callable[[Any], float] | None = None,
) -> Callable[[F], F]:
    """Decorator to handle provider-specific exceptions consistently.

    This decorator wraps model methods and translates provider-specific
    exceptions into the insideLLMs exception hierarchy. It eliminates
    duplicate try/except blocks across model implementations.

    Parameters
    ----------
    exception_map : ProviderExceptionMap
        Mapping of provider exceptions to insideLLMs exceptions.
    get_model_id : Callable[[Any], str], optional
        Function to extract model_id from self. Defaults to self.model_id.
    get_timeout : Callable[[Any], float], optional
        Function to extract timeout from self. Defaults to self._timeout.

    Returns
    -------
    Callable
        Decorated function with standardized error handling.

    Example - Decorating a Model Method:
        >>> from insideLLMs.models.base import handle_provider_errors, ProviderExceptionMap
        >>> from openai import RateLimitError, APITimeoutError, APIError
        >>>
        >>> OPENAI_EXCEPTIONS = ProviderExceptionMap(
        ...     rate_limit_errors=(RateLimitError,),
        ...     timeout_errors=(APITimeoutError,),
        ...     api_errors=(APIError,),
        ... )
        >>>
        >>> class MyOpenAIModel(Model):
        ...     @handle_provider_errors(OPENAI_EXCEPTIONS)
        ...     def generate(self, prompt: str, **kwargs) -> str:
        ...         response = self._client.chat.completions.create(
        ...             model=self.model_name,
        ...             messages=[{"role": "user", "content": prompt}],
        ...             **kwargs,
        ...         )
        ...         return response.choices[0].message.content or ""

    Notes
    -----
    The decorator automatically:
    - Catches rate limit errors and raises RateLimitError with retry_after
    - Catches timeout errors and raises TimeoutError with timeout_seconds
    - Catches API errors and raises APIError with status_code and message
    - Catches all other exceptions and raises ModelGenerationError

    The first positional argument after `self` is assumed to be the prompt
    for ModelGenerationError context.
    """
    # Lazy import to avoid circular dependency
    from insideLLMs.exceptions import (
        APIError as InsideLLMsAPIError,
    )
    from insideLLMs.exceptions import (
        ModelGenerationError,
        RateLimitError,
    )
    from insideLLMs.exceptions import (
        ModelTimeoutError as InsideLLMsTimeoutError,
    )

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Extract model metadata
            model_id = get_model_id(self) if get_model_id else getattr(self, "model_id", "unknown")
            timeout = get_timeout(self) if get_timeout else getattr(self, "_timeout", 60.0)

            # Get prompt for error context (first positional arg after self)
            prompt = args[0] if args else kwargs.get("prompt", "")

            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                # Avoid re-wrapping insideLLMs exceptions
                if isinstance(
                    e,
                    (
                        RateLimitError,
                        InsideLLMsTimeoutError,
                        InsideLLMsAPIError,
                        ModelGenerationError,
                    ),
                ):
                    raise
                if exception_map.rate_limit_errors and isinstance(
                    e, exception_map.rate_limit_errors
                ):
                    raise RateLimitError(
                        model_id=model_id,
                        retry_after=getattr(e, "retry_after", None),
                    ) from e
                if exception_map.timeout_errors and isinstance(e, exception_map.timeout_errors):
                    raise InsideLLMsTimeoutError(
                        model_id=model_id,
                        timeout_seconds=timeout,
                    ) from e
                if exception_map.api_errors and isinstance(e, exception_map.api_errors):
                    raise InsideLLMsAPIError(
                        model_id=model_id,
                        status_code=getattr(e, "status_code", None),
                        message=str(e),
                    ) from e
                raise ModelGenerationError(
                    model_id=model_id,
                    prompt=str(prompt)[:500] if prompt else "",
                    reason=str(e),
                    original_error=e,
                ) from e

        return wrapper  # type: ignore[return-value]

    return decorator


def translate_provider_error(
    error: Exception,
    model_id: str,
    exception_map: ProviderExceptionMap,
    *,
    timeout_seconds: float = 60.0,
    prompt: str = "",
) -> Exception:
    """Translate a provider exception to an insideLLMs exception.

    This function provides a non-decorator alternative for error translation,
    useful in contexts where the decorator pattern is not appropriate
    (e.g., generators, async methods with complex flow).

    Parameters
    ----------
    error : Exception
        The provider-specific exception to translate.
    model_id : str
        The model identifier for error context.
    exception_map : ProviderExceptionMap
        Mapping of provider exceptions.
    timeout_seconds : float, optional
        Timeout value for TimeoutError context. Default: 60.0.
    prompt : str, optional
        The prompt for ModelGenerationError context. Default: "".

    Returns
    -------
    Exception
        The translated insideLLMs exception.

    Example - Manual Error Translation:
        >>> try:
        ...     response = client.messages.create(...)
        ... except Exception as e:
        ...     raise translate_provider_error(
        ...         e,
        ...         model_id=self.model_id,
        ...         exception_map=ANTHROPIC_EXCEPTIONS,
        ...         prompt=prompt,
        ...     )

    Example - In a Generator:
        >>> def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        ...     try:
        ...         for chunk in self._client.stream(...):
        ...             yield chunk.text
        ...     except Exception as e:
        ...         raise translate_provider_error(
        ...             e,
        ...             model_id=self.model_id,
        ...             exception_map=MY_EXCEPTIONS,
        ...             prompt=prompt,
        ...         )
    """
    from insideLLMs.exceptions import (
        APIError as InsideLLMsAPIError,
    )
    from insideLLMs.exceptions import (
        ModelGenerationError,
        RateLimitError,
    )
    from insideLLMs.exceptions import (
        ModelTimeoutError as InsideLLMsTimeoutError,
    )

    # Check if already an insideLLMs exception
    if isinstance(
        error, (RateLimitError, InsideLLMsTimeoutError, InsideLLMsAPIError, ModelGenerationError)
    ):
        return error

    if isinstance(error, exception_map.rate_limit_errors):
        return RateLimitError(
            model_id=model_id,
            retry_after=getattr(error, "retry_after", None),
        )
    elif isinstance(error, exception_map.timeout_errors):
        return InsideLLMsTimeoutError(
            model_id=model_id,
            timeout_seconds=timeout_seconds,
        )
    elif isinstance(error, exception_map.api_errors):
        return InsideLLMsAPIError(
            model_id=model_id,
            status_code=getattr(error, "status_code", None),
            message=str(error),
        )
    else:
        return ModelGenerationError(
            model_id=model_id,
            prompt=str(prompt)[:500] if prompt else "",
            reason=str(error),
            original_error=error,
        )
