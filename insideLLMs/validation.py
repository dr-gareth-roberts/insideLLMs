"""Input validation utilities for insideLLMs.

This module provides systematic input validation with clear error messages
for prompts, prompt sets, and configuration values.

Example:
    >>> from insideLLMs.validation import validate_prompt, validate_prompt_set
    >>> validate_prompt("Hello, world!")  # Returns True
    >>> validate_prompt_set(["Q1", "Q2"])  # Returns True
    >>> validate_prompt("")  # Raises ValidationError
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, TypeVar, Union

T = TypeVar("T")


class ValidationError(Exception):
    """Raised when input validation fails.

    Attributes:
        message: Human-readable error description.
        field: Name of the field that failed validation.
        value: The invalid value (may be truncated for display).
        suggestions: Optional list of suggestions to fix the error.
    """

    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        value: Any = None,
        suggestions: Optional[list[str]] = None,
    ):
        self.message = message
        self.field = field
        self.value = value
        self.suggestions = suggestions or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.field:
            parts.insert(0, f"[{self.field}]")
        if self.suggestions:
            parts.append(f"Suggestions: {', '.join(self.suggestions)}")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """Result of a validation check.

    Attributes:
        valid: Whether the validation passed.
        errors: List of error messages if validation failed.
        warnings: List of warning messages (non-fatal issues).
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another validation result into this one."""
        return ValidationResult(
            valid=self.valid and other.valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


def validate_prompt(
    prompt: Any,
    *,
    field_name: str = "prompt",
    max_length: Optional[int] = None,
    min_length: int = 0,
    allow_empty: bool = False,
) -> bool:
    """Validate a single prompt.

    Args:
        prompt: The prompt to validate.
        field_name: Name of the field for error messages.
        max_length: Maximum allowed length (None for no limit).
        min_length: Minimum required length.
        allow_empty: Whether empty strings are allowed.

    Returns:
        True if valid.

    Raises:
        ValidationError: If validation fails.
    """
    if prompt is None:
        raise ValidationError(
            "Prompt cannot be None",
            field=field_name,
            value=prompt,
            suggestions=["Provide a non-None string value"],
        )

    if not isinstance(prompt, str):
        raise ValidationError(
            f"Prompt must be a string, got {type(prompt).__name__}",
            field=field_name,
            value=repr(prompt)[:100],
            suggestions=["Convert the value to a string using str()"],
        )

    if not allow_empty and len(prompt) == 0:
        raise ValidationError(
            "Prompt cannot be empty",
            field=field_name,
            value=prompt,
            suggestions=["Provide a non-empty prompt string"],
        )

    if len(prompt) < min_length:
        raise ValidationError(
            f"Prompt too short: {len(prompt)} < {min_length} characters",
            field=field_name,
            value=prompt[:50] + "..." if len(prompt) > 50 else prompt,
        )

    if max_length is not None and len(prompt) > max_length:
        raise ValidationError(
            f"Prompt too long: {len(prompt)} > {max_length} characters",
            field=field_name,
            value=prompt[:50] + "...",
            suggestions=[f"Truncate the prompt to {max_length} characters"],
        )

    return True


def validate_prompt_set(
    prompt_set: Any,
    *,
    field_name: str = "prompt_set",
    max_length: Optional[int] = None,
    allow_empty_set: bool = False,
    allow_empty_prompts: bool = False,
) -> bool:
    """Validate a list of prompts.

    Args:
        prompt_set: The prompt set to validate.
        field_name: Name of the field for error messages.
        max_length: Maximum allowed length per prompt.
        allow_empty_set: Whether an empty list is allowed.
        allow_empty_prompts: Whether empty strings are allowed in the set.

    Returns:
        True if valid.

    Raises:
        ValidationError: If validation fails.
    """
    if prompt_set is None:
        raise ValidationError(
            "Prompt set cannot be None",
            field=field_name,
            value=prompt_set,
            suggestions=["Provide a list of prompts"],
        )

    if not isinstance(prompt_set, (list, tuple)):
        raise ValidationError(
            f"Prompt set must be a list or tuple, got {type(prompt_set).__name__}",
            field=field_name,
            value=repr(prompt_set)[:100],
            suggestions=["Wrap the value in a list: [value]"],
        )

    if not allow_empty_set and len(prompt_set) == 0:
        raise ValidationError(
            "Prompt set cannot be empty",
            field=field_name,
            value=prompt_set,
            suggestions=["Provide at least one prompt"],
        )

    # Validate each prompt in the set
    for i, prompt in enumerate(prompt_set):
        # Allow dict items (common for structured prompts)
        if isinstance(prompt, dict):
            continue
        try:
            validate_prompt(
                prompt,
                field_name=f"{field_name}[{i}]",
                max_length=max_length,
                allow_empty=allow_empty_prompts,
            )
        except ValidationError:
            raise

    return True


def validate_positive_int(
    value: Any,
    *,
    field_name: str = "value",
    min_value: int = 1,
    max_value: Optional[int] = None,
) -> int:
    """Validate that a value is a positive integer.

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.
        min_value: Minimum allowed value (default 1).
        max_value: Maximum allowed value (None for no limit).

    Returns:
        The validated integer value.

    Raises:
        ValidationError: If validation fails.
    """
    if value is None:
        raise ValidationError(
            f"{field_name} cannot be None",
            field=field_name,
            value=value,
        )

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(
            f"{field_name} must be an integer, got {type(value).__name__}",
            field=field_name,
            value=value,
        )

    if value < min_value:
        raise ValidationError(
            f"{field_name} must be >= {min_value}, got {value}",
            field=field_name,
            value=value,
        )

    if max_value is not None and value > max_value:
        raise ValidationError(
            f"{field_name} must be <= {max_value}, got {value}",
            field=field_name,
            value=value,
        )

    return value


def validate_choice(
    value: Any,
    choices: Sequence[T],
    *,
    field_name: str = "value",
) -> T:
    """Validate that a value is one of the allowed choices.

    Args:
        value: The value to validate.
        choices: Sequence of allowed values.
        field_name: Name of the field for error messages.

    Returns:
        The validated value.

    Raises:
        ValidationError: If validation fails.
    """
    if value not in choices:
        raise ValidationError(
            f"{field_name} must be one of {list(choices)}, got {value!r}",
            field=field_name,
            value=value,
            suggestions=[f"Use one of: {', '.join(repr(c) for c in choices)}"],
        )
    return value


# ============================================================================
# Validation Decorators
# ============================================================================


def validates_prompt(param_name: str = "prompt", **validation_kwargs: Any) -> Callable:
    """Decorator to validate a prompt parameter.

    Args:
        param_name: Name of the parameter to validate.
        **validation_kwargs: Additional kwargs passed to validate_prompt.

    Example:
        >>> @validates_prompt("prompt")
        ... def generate(self, prompt: str) -> str:
        ...     return "response"
    """
    import functools
    import inspect

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the parameter value from args or kwargs
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            value = None
            if param_name in kwargs:
                value = kwargs[param_name]
            elif param_name in params:
                idx = params.index(param_name)
                if idx < len(args):
                    value = args[idx]

            if value is not None:
                validate_prompt(value, field_name=param_name, **validation_kwargs)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validates_prompt_set(
    param_name: str = "prompt_set", **validation_kwargs: Any
) -> Callable:
    """Decorator to validate a prompt set parameter.

    Args:
        param_name: Name of the parameter to validate.
        **validation_kwargs: Additional kwargs passed to validate_prompt_set.

    Example:
        >>> @validates_prompt_set("prompts")
        ... def run(self, prompts: list[str]) -> list[str]:
        ...     return prompts
    """
    import functools
    import inspect

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            value = None
            if param_name in kwargs:
                value = kwargs[param_name]
            elif param_name in params:
                idx = params.index(param_name)
                if idx < len(args):
                    value = args[idx]

            if value is not None:
                validate_prompt_set(value, field_name=param_name, **validation_kwargs)

            return func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "ValidationError",
    "ValidationResult",
    "validate_prompt",
    "validate_prompt_set",
    "validate_positive_int",
    "validate_choice",
    "validates_prompt",
    "validates_prompt_set",
]

