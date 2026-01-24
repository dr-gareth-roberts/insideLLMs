"""Input validation utilities for insideLLMs.

This module provides systematic input validation with clear error messages
for prompts, prompt sets, and configuration values. It includes validation
functions, result containers, custom exceptions, and decorators for
automatic parameter validation.

The validation system is designed to:
    - Provide clear, actionable error messages with suggestions
    - Support both synchronous validation functions and decorators
    - Enable composable validation through ValidationResult merging
    - Handle common edge cases (None values, wrong types, length limits)

Classes:
    ValidationError: Exception raised when validation fails, with structured
        error information including field name, invalid value, and suggestions.
    ValidationResult: Dataclass for collecting validation results with support
        for errors, warnings, and merging multiple results.

Functions:
    validate_prompt: Validate a single prompt string.
    validate_prompt_set: Validate a list/tuple of prompts.
    validate_positive_int: Validate positive integer values with bounds.
    validate_choice: Validate that a value is in an allowed set.
    validates_prompt: Decorator for automatic prompt parameter validation.
    validates_prompt_set: Decorator for automatic prompt set validation.

Examples:
    Basic prompt validation:

        >>> from insideLLMs.validation import validate_prompt
        >>> validate_prompt("Hello, world!")
        True
        >>> validate_prompt("What is AI?", max_length=100)
        True

    Validating prompt sets:

        >>> from insideLLMs.validation import validate_prompt_set
        >>> validate_prompt_set(["Question 1", "Question 2"])
        True
        >>> validate_prompt_set([{"role": "user", "content": "Hi"}])
        True

    Handling validation errors:

        >>> from insideLLMs.validation import validate_prompt, ValidationError
        >>> try:
        ...     validate_prompt("")
        ... except ValidationError as e:
        ...     print(f"Field: {e.field}, Suggestions: {e.suggestions}")
        Field: prompt, Suggestions: ['Provide a non-empty prompt string']

    Using validation decorators:

        >>> from insideLLMs.validation import validates_prompt
        >>> @validates_prompt("text")
        ... def process(text: str) -> str:
        ...     return text.upper()
        >>> process("hello")
        'HELLO'

    Composing validation results:

        >>> from insideLLMs.validation import ValidationResult
        >>> result1 = ValidationResult(valid=True, warnings=["Minor issue"])
        >>> result2 = ValidationResult(valid=False, errors=["Major problem"])
        >>> combined = result1.merge(result2)
        >>> combined.valid
        False
        >>> combined.errors
        ['Major problem']
        >>> combined.warnings
        ['Minor issue']
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, TypeVar

T = TypeVar("T")


class ValidationError(Exception):
    """Exception raised when input validation fails.

    ValidationError provides structured error information to help users
    understand what went wrong and how to fix it. It includes the error
    message, the field name that failed validation, the invalid value,
    and optional suggestions for resolution.

    The error message is automatically formatted to include all available
    context when the exception is raised or printed.

    Attributes:
        message (str): Human-readable error description explaining what
            validation rule was violated.
        field (Optional[str]): Name of the field that failed validation.
            Used to identify which parameter or input caused the error.
        value (Any): The invalid value that caused the error. May be
            truncated for display if it's too long.
        suggestions (list[str]): List of actionable suggestions to help
            the user fix the validation error.

    Examples:
        Basic validation error with message only:

            >>> raise ValidationError("Invalid input provided")
            Traceback (most recent call last):
                ...
            ValidationError: Invalid input provided

        Validation error with field name:

            >>> error = ValidationError(
            ...     "Value cannot be empty",
            ...     field="username"
            ... )
            >>> str(error)
            '[username] Value cannot be empty'

        Validation error with value and suggestions:

            >>> error = ValidationError(
            ...     "Must be a positive integer",
            ...     field="count",
            ...     value=-5,
            ...     suggestions=["Use a value >= 1", "Check your input"]
            ... )
            >>> error.field
            'count'
            >>> error.value
            -5
            >>> error.suggestions
            ['Use a value >= 1', 'Check your input']

        Accessing error attributes in exception handling:

            >>> try:
            ...     raise ValidationError(
            ...         "Prompt too long",
            ...         field="prompt",
            ...         value="x" * 1000,
            ...         suggestions=["Truncate to 500 characters"]
            ...     )
            ... except ValidationError as e:
            ...     print(f"Error in '{e.field}': {e.message}")
            ...     print(f"Suggestions: {e.suggestions}")
            Error in 'prompt': Prompt too long
            Suggestions: ['Truncate to 500 characters']
    """

    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        value: Any = None,
        suggestions: Optional[list[str]] = None,
    ) -> None:
        """Initialize a ValidationError with error details.

        Args:
            message: Human-readable error description explaining the
                validation failure.
            field: Optional name of the field that failed validation.
                If provided, it will be prepended to the error message
                in brackets (e.g., "[field_name] error message").
            value: Optional invalid value that caused the error. Useful
                for debugging but may be truncated in display.
            suggestions: Optional list of actionable suggestions to help
                fix the error. Will be appended to the formatted message.

        Examples:
            Create a simple validation error:

                >>> error = ValidationError("Invalid type")
                >>> error.message
                'Invalid type'

            Create an error with all fields:

                >>> error = ValidationError(
                ...     "Value out of range",
                ...     field="temperature",
                ...     value=2.5,
                ...     suggestions=["Use a value between 0 and 1"]
                ... )
                >>> error.field
                'temperature'
                >>> error.value
                2.5

            Create an error with multiple suggestions:

                >>> error = ValidationError(
                ...     "Unknown model",
                ...     field="model_name",
                ...     value="gpt-5",
                ...     suggestions=[
                ...         "Check available models",
                ...         "Verify spelling",
                ...         "Use 'gpt-4' instead"
                ...     ]
                ... )
                >>> len(error.suggestions)
                3
        """
        self.message = message
        self.field = field
        self.value = value
        self.suggestions = suggestions or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with field name and suggestions.

        Constructs the full error message by combining the base message
        with the field name (if provided) and any suggestions. The format
        is: "[field] message Suggestions: suggestion1, suggestion2"

        Returns:
            str: The fully formatted error message string.

        Examples:
            Message only:

                >>> error = ValidationError("Invalid input")
                >>> error._format_message()
                'Invalid input'

            With field name:

                >>> error = ValidationError("Cannot be empty", field="name")
                >>> error._format_message()
                '[name] Cannot be empty'

            With field and suggestions:

                >>> error = ValidationError(
                ...     "Too short",
                ...     field="password",
                ...     suggestions=["Use at least 8 characters"]
                ... )
                >>> error._format_message()
                '[password] Too short Suggestions: Use at least 8 characters'

            With multiple suggestions:

                >>> error = ValidationError(
                ...     "Invalid format",
                ...     suggestions=["Check syntax", "See docs"]
                ... )
                >>> error._format_message()
                'Invalid format Suggestions: Check syntax, See docs'
        """
        parts = [self.message]
        if self.field:
            parts.insert(0, f"[{self.field}]")
        if self.suggestions:
            parts.append(f"Suggestions: {', '.join(self.suggestions)}")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """Result of a validation check with support for errors and warnings.

    ValidationResult is a dataclass that collects the outcome of validation
    operations. It supports both hard errors (which cause validation to fail)
    and soft warnings (which are informational but don't fail validation).

    Multiple ValidationResult instances can be merged together to aggregate
    results from multiple validation checks. The merged result is invalid
    if any of the source results were invalid.

    The class supports boolean evaluation, making it easy to use in
    conditional statements.

    Attributes:
        valid (bool): Whether the validation passed. Defaults to True.
            Set to False when any validation rule is violated.
        errors (list[str]): List of error messages explaining validation
            failures. Empty list if validation passed.
        warnings (list[str]): List of warning messages for non-fatal issues.
            These don't affect the valid status but provide useful feedback.

    Examples:
        Creating a successful validation result:

            >>> result = ValidationResult()
            >>> result.valid
            True
            >>> bool(result)
            True
            >>> result.errors
            []

        Creating a failed validation result:

            >>> result = ValidationResult(
            ...     valid=False,
            ...     errors=["Field 'name' is required", "Field 'email' is invalid"]
            ... )
            >>> result.valid
            False
            >>> len(result.errors)
            2

        Creating a result with warnings:

            >>> result = ValidationResult(
            ...     valid=True,
            ...     warnings=["Using deprecated field 'old_name'"]
            ... )
            >>> result.valid
            True
            >>> result.warnings
            ["Using deprecated field 'old_name'"]

        Using in conditional statements:

            >>> result = ValidationResult(valid=True)
            >>> if result:
            ...     print("Validation passed!")
            Validation passed!

        Merging multiple results:

            >>> result1 = ValidationResult(valid=True, warnings=["Warning 1"])
            >>> result2 = ValidationResult(valid=True, warnings=["Warning 2"])
            >>> result3 = ValidationResult(valid=False, errors=["Error 1"])
            >>> combined = result1.merge(result2).merge(result3)
            >>> combined.valid
            False
            >>> combined.errors
            ['Error 1']
            >>> combined.warnings
            ['Warning 1', 'Warning 2']
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Return the validity status for boolean evaluation.

        Allows ValidationResult to be used directly in conditional
        statements and boolean contexts.

        Returns:
            bool: True if the validation passed, False otherwise.

        Examples:
            Using in if statements:

                >>> result = ValidationResult(valid=True)
                >>> if result:
                ...     print("Valid!")
                Valid!

            Using with not operator:

                >>> result = ValidationResult(valid=False, errors=["Error"])
                >>> if not result:
                ...     print("Invalid!")
                Invalid!

            Using in boolean expressions:

                >>> result1 = ValidationResult(valid=True)
                >>> result2 = ValidationResult(valid=False)
                >>> bool(result1) and bool(result2)
                False

            Truthiness check:

                >>> result = ValidationResult()
                >>> "valid" if result else "invalid"
                'valid'
        """
        return self.valid

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another validation result into this one.

        Creates a new ValidationResult that combines this result with
        another. The merged result is valid only if both source results
        are valid. Errors and warnings from both results are concatenated.

        This method is useful for aggregating results from multiple
        independent validation checks into a single result.

        Args:
            other: Another ValidationResult to merge with this one.

        Returns:
            ValidationResult: A new ValidationResult containing the
                combined validity status, errors, and warnings from
                both source results.

        Examples:
            Merging two valid results:

                >>> result1 = ValidationResult(valid=True)
                >>> result2 = ValidationResult(valid=True)
                >>> merged = result1.merge(result2)
                >>> merged.valid
                True

            Merging valid with invalid:

                >>> valid = ValidationResult(valid=True)
                >>> invalid = ValidationResult(valid=False, errors=["Error"])
                >>> merged = valid.merge(invalid)
                >>> merged.valid
                False
                >>> merged.errors
                ['Error']

            Combining errors from multiple results:

                >>> result1 = ValidationResult(valid=False, errors=["Error 1"])
                >>> result2 = ValidationResult(valid=False, errors=["Error 2"])
                >>> merged = result1.merge(result2)
                >>> merged.errors
                ['Error 1', 'Error 2']

            Combining warnings:

                >>> result1 = ValidationResult(warnings=["Warn 1"])
                >>> result2 = ValidationResult(warnings=["Warn 2"])
                >>> merged = result1.merge(result2)
                >>> merged.warnings
                ['Warn 1', 'Warn 2']

            Chaining multiple merges:

                >>> r1 = ValidationResult(valid=True, warnings=["W1"])
                >>> r2 = ValidationResult(valid=True, errors=[])
                >>> r3 = ValidationResult(valid=False, errors=["E1"])
                >>> final = r1.merge(r2).merge(r3)
                >>> final.valid
                False
                >>> final.warnings
                ['W1']
                >>> final.errors
                ['E1']
        """
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
    """Validate a single prompt string.

    Performs comprehensive validation on a prompt string, checking for:
    - Non-None value
    - Correct type (must be a string)
    - Non-empty content (unless allow_empty is True)
    - Minimum length requirement
    - Maximum length limit (if specified)

    This function raises ValidationError with detailed error messages
    and suggestions when validation fails, making it easy to understand
    and fix issues.

    Args:
        prompt: The prompt value to validate. Should be a string.
        field_name: Name of the field for error messages. Used to identify
            which parameter failed validation. Defaults to "prompt".
        max_length: Maximum allowed character length. None means no limit.
            Defaults to None.
        min_length: Minimum required character length. Defaults to 0.
        allow_empty: Whether empty strings ("") are allowed. Defaults to
            False, meaning empty strings will raise ValidationError.

    Returns:
        bool: True if the prompt passes all validation checks.

    Raises:
        ValidationError: If validation fails for any of the following:
            - prompt is None
            - prompt is not a string
            - prompt is empty (when allow_empty is False)
            - prompt length is less than min_length
            - prompt length exceeds max_length (when specified)

    Examples:
        Basic prompt validation:

            >>> validate_prompt("Hello, world!")
            True
            >>> validate_prompt("What is the meaning of life?")
            True

        Validating with length constraints:

            >>> validate_prompt("Hi", min_length=2)
            True
            >>> validate_prompt("Hello", max_length=10)
            True
            >>> validate_prompt("A" * 100, max_length=50)
            Traceback (most recent call last):
                ...
            ValidationError: [prompt] Prompt too long: 100 > 50 characters...

        Handling None values:

            >>> try:
            ...     validate_prompt(None)
            ... except ValidationError as e:
            ...     print(e.suggestions)
            ['Provide a non-None string value']

        Handling wrong types:

            >>> try:
            ...     validate_prompt(123)
            ... except ValidationError as e:
            ...     print(f"Error: {e.message}")
            Error: Prompt must be a string, got int

        Empty string handling:

            >>> validate_prompt("", allow_empty=True)
            True
            >>> try:
            ...     validate_prompt("")
            ... except ValidationError as e:
            ...     print(e.message)
            Prompt cannot be empty

        Custom field names for better error messages:

            >>> try:
            ...     validate_prompt("", field_name="user_query")
            ... except ValidationError as e:
            ...     print(e.field)
            user_query

        Combining constraints:

            >>> validate_prompt("Hello there!", min_length=5, max_length=100)
            True
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
    """Validate a collection of prompts (list or tuple).

    Performs comprehensive validation on a prompt set, checking:
    - Non-None value
    - Correct type (must be a list or tuple)
    - Non-empty collection (unless allow_empty_set is True)
    - Each individual prompt in the set (delegating to validate_prompt)

    Dictionary items in the prompt set are allowed and skipped during
    individual validation, as they represent structured prompts (e.g.,
    chat messages with role and content).

    This function raises ValidationError with detailed error messages
    including the index of any invalid prompt in the set.

    Args:
        prompt_set: The prompt collection to validate. Should be a list
            or tuple of strings or dictionaries.
        field_name: Name of the field for error messages. Used to identify
            which parameter failed validation. Defaults to "prompt_set".
        max_length: Maximum allowed character length for each prompt string.
            None means no limit. Defaults to None.
        allow_empty_set: Whether an empty list/tuple is allowed. Defaults
            to False, meaning empty collections will raise ValidationError.
        allow_empty_prompts: Whether individual prompts can be empty strings.
            Defaults to False.

    Returns:
        bool: True if the prompt set passes all validation checks.

    Raises:
        ValidationError: If validation fails for any of the following:
            - prompt_set is None
            - prompt_set is not a list or tuple
            - prompt_set is empty (when allow_empty_set is False)
            - Any individual prompt fails validation (with index in error)

    Examples:
        Basic prompt set validation:

            >>> validate_prompt_set(["Hello", "World"])
            True
            >>> validate_prompt_set(("Question 1", "Question 2"))
            True

        Validating structured prompts (chat format):

            >>> messages = [
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi there!"}
            ... ]
            >>> validate_prompt_set(messages)
            True

        Mixed string and dict prompts:

            >>> prompts = [
            ...     "Simple prompt",
            ...     {"role": "user", "content": "Chat message"}
            ... ]
            >>> validate_prompt_set(prompts)
            True

        Handling None values:

            >>> try:
            ...     validate_prompt_set(None)
            ... except ValidationError as e:
            ...     print(e.suggestions)
            ['Provide a list of prompts']

        Handling wrong types:

            >>> try:
            ...     validate_prompt_set("not a list")
            ... except ValidationError as e:
            ...     print(e.message)
            Prompt set must be a list or tuple, got str

        Empty set handling:

            >>> validate_prompt_set([], allow_empty_set=True)
            True
            >>> try:
            ...     validate_prompt_set([])
            ... except ValidationError as e:
            ...     print(e.message)
            Prompt set cannot be empty

        Individual prompt validation with index in error:

            >>> try:
            ...     validate_prompt_set(["Valid", "", "Also valid"])
            ... except ValidationError as e:
            ...     print(e.field)
            prompt_set[1]

        Length constraints per prompt:

            >>> validate_prompt_set(["Hi", "Hey"], max_length=10)
            True
            >>> try:
            ...     validate_prompt_set(["A" * 100], max_length=50)
            ... except ValidationError as e:
            ...     print("too long" in e.message)
            True

        Custom field names:

            >>> try:
            ...     validate_prompt_set(None, field_name="questions")
            ... except ValidationError as e:
            ...     print(e.field)
            questions
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
    """Validate that a value is a positive integer within bounds.

    Performs validation on an integer value, checking:
    - Non-None value
    - Correct type (must be an int, not a bool)
    - Value is at least min_value
    - Value is at most max_value (if specified)

    Note: Boolean values are explicitly rejected even though bool is
    a subclass of int in Python, as they represent logical values
    rather than numeric values.

    This function returns the validated value on success, making it
    useful in assignment contexts.

    Args:
        value: The value to validate. Should be an integer.
        field_name: Name of the field for error messages. Used to identify
            which parameter failed validation. Defaults to "value".
        min_value: Minimum allowed value (inclusive). Defaults to 1,
            meaning only positive integers are valid by default.
        max_value: Maximum allowed value (inclusive). None means no upper
            limit. Defaults to None.

    Returns:
        int: The validated integer value (same as input if valid).

    Raises:
        ValidationError: If validation fails for any of the following:
            - value is None
            - value is not an integer (or is a boolean)
            - value is less than min_value
            - value exceeds max_value (when specified)

    Examples:
        Basic positive integer validation:

            >>> validate_positive_int(5)
            5
            >>> validate_positive_int(100)
            100
            >>> validate_positive_int(1)
            1

        Validating with minimum value:

            >>> validate_positive_int(10, min_value=5)
            10
            >>> try:
            ...     validate_positive_int(3, min_value=5)
            ... except ValidationError as e:
            ...     print(e.message)
            value must be >= 5, got 3

        Validating with maximum value:

            >>> validate_positive_int(50, max_value=100)
            50
            >>> try:
            ...     validate_positive_int(150, max_value=100)
            ... except ValidationError as e:
            ...     print(e.message)
            value must be <= 100, got 150

        Validating within a range:

            >>> validate_positive_int(50, min_value=1, max_value=100)
            50
            >>> validate_positive_int(1, min_value=1, max_value=100)
            1
            >>> validate_positive_int(100, min_value=1, max_value=100)
            100

        Handling None values:

            >>> try:
            ...     validate_positive_int(None)
            ... except ValidationError as e:
            ...     print(e.field)
            value

        Handling wrong types:

            >>> try:
            ...     validate_positive_int("42")
            ... except ValidationError as e:
            ...     print(e.message)
            value must be an integer, got str

        Boolean rejection (bools are not valid integers here):

            >>> try:
            ...     validate_positive_int(True)
            ... except ValidationError as e:
            ...     print(e.message)
            value must be an integer, got bool

        Custom field names:

            >>> try:
            ...     validate_positive_int(-5, field_name="batch_size")
            ... except ValidationError as e:
            ...     print(e.field)
            batch_size

        Using in assignment:

            >>> count = validate_positive_int(10, field_name="count")
            >>> count
            10
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

    Checks if a value is contained within a sequence of valid choices.
    This is useful for validating enumerated values, mode selections,
    or any parameter that must be one of a predefined set of options.

    The error message includes all valid choices and provides a
    suggestion listing the allowed values.

    Args:
        value: The value to validate. Can be any type that supports
            equality comparison with the choices.
        choices: A sequence (list, tuple, etc.) of allowed values.
            The value must be equal to one of these to pass validation.
        field_name: Name of the field for error messages. Used to identify
            which parameter failed validation. Defaults to "value".

    Returns:
        T: The validated value (same as input if valid). The return type
            matches the type of elements in the choices sequence.

    Raises:
        ValidationError: If the value is not in the choices sequence.
            The error message lists all valid choices and includes
            suggestions for valid values.

    Examples:
        Validating string choices:

            >>> validate_choice("debug", ["debug", "info", "warning", "error"])
            'debug'
            >>> validate_choice("info", ("debug", "info", "warning"))
            'info'

        Validating numeric choices:

            >>> validate_choice(1, [1, 2, 3, 4, 5])
            1
            >>> validate_choice(100, [50, 100, 200])
            100

        Handling invalid choices:

            >>> try:
            ...     validate_choice("verbose", ["debug", "info", "warning"])
            ... except ValidationError as e:
            ...     print("verbose" in e.message and "debug" in e.message)
            True

        Error message includes suggestions:

            >>> try:
            ...     validate_choice("invalid", ["a", "b", "c"])
            ... except ValidationError as e:
            ...     print(e.suggestions)
            ["Use one of: 'a', 'b', 'c'"]

        Custom field names:

            >>> try:
            ...     validate_choice("unknown", ["gpt-4", "claude"], field_name="model")
            ... except ValidationError as e:
            ...     print(e.field)
            model

        Using with enums or constants:

            >>> MODES = ["train", "eval", "predict"]
            >>> validate_choice("train", MODES)
            'train'
            >>> try:
            ...     validate_choice("test", MODES)
            ... except ValidationError as e:
            ...     print("test" in str(e.value))
            True

        Using in assignment with type preservation:

            >>> level: str = validate_choice("warning", ["debug", "info", "warning"])
            >>> level
            'warning'
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
    """Decorator to automatically validate a prompt parameter before function execution.

    This decorator wraps a function to add automatic prompt validation for
    a specified parameter. The validation occurs before the function body
    executes, ensuring that invalid prompts are caught early with clear
    error messages.

    The decorator uses introspection to find the parameter value from
    either positional or keyword arguments, making it work with various
    function signatures.

    Args:
        param_name: Name of the parameter to validate. This should match
            the actual parameter name in the decorated function's signature.
            Defaults to "prompt".
        **validation_kwargs: Additional keyword arguments passed directly
            to validate_prompt(). Supported options include:
            - max_length (int): Maximum allowed prompt length
            - min_length (int): Minimum required prompt length
            - allow_empty (bool): Whether empty strings are allowed

    Returns:
        Callable: A decorator function that wraps the target function
            with prompt validation.

    Raises:
        ValidationError: If the prompt parameter fails validation.
            The error is raised before the decorated function executes.

    Examples:
        Basic usage with default parameter name:

            >>> @validates_prompt()
            ... def generate(prompt: str) -> str:
            ...     return f"Response to: {prompt}"
            >>> generate("Hello!")
            'Response to: Hello!'

        Custom parameter name:

            >>> @validates_prompt("user_input")
            ... def process(user_input: str) -> str:
            ...     return user_input.upper()
            >>> process("hello")
            'HELLO'

        With length constraints:

            >>> @validates_prompt("text", max_length=100, min_length=5)
            ... def analyze(text: str) -> int:
            ...     return len(text)
            >>> analyze("Hello world!")
            12

        Validation error example:

            >>> @validates_prompt("query")
            ... def search(query: str) -> list:
            ...     return []
            >>> try:
            ...     search("")  # Empty string
            ... except ValidationError as e:
            ...     print(e.message)
            Prompt cannot be empty

        Works with keyword arguments:

            >>> @validates_prompt("message")
            ... def send(to: str, message: str) -> bool:
            ...     return True
            >>> send("user@example.com", message="Hello!")
            True

        Works with positional arguments:

            >>> @validates_prompt("content")
            ... def save(filename: str, content: str) -> None:
            ...     pass
            >>> save("file.txt", "Some content")

        With class methods:

            >>> class Generator:
            ...     @validates_prompt("prompt")
            ...     def generate(self, prompt: str) -> str:
            ...         return "Generated"
            >>> g = Generator()
            >>> g.generate("Test prompt")
            'Generated'

        Allowing empty prompts:

            >>> @validates_prompt("text", allow_empty=True)
            ... def optional_text(text: str) -> str:
            ...     return text or "default"
            >>> optional_text("")
            'default'
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


def validates_prompt_set(param_name: str = "prompt_set", **validation_kwargs: Any) -> Callable:
    """Decorator to automatically validate a prompt set parameter before function execution.

    This decorator wraps a function to add automatic validation for a
    parameter containing a list or tuple of prompts. The validation occurs
    before the function body executes, ensuring that invalid prompt sets
    are caught early with clear error messages.

    The decorator uses introspection to find the parameter value from
    either positional or keyword arguments, making it work with various
    function signatures.

    Args:
        param_name: Name of the parameter to validate. This should match
            the actual parameter name in the decorated function's signature.
            Defaults to "prompt_set".
        **validation_kwargs: Additional keyword arguments passed directly
            to validate_prompt_set(). Supported options include:
            - max_length (int): Maximum allowed length per prompt
            - allow_empty_set (bool): Whether an empty list is allowed
            - allow_empty_prompts (bool): Whether empty strings in the set
              are allowed

    Returns:
        Callable: A decorator function that wraps the target function
            with prompt set validation.

    Raises:
        ValidationError: If the prompt set parameter fails validation.
            The error is raised before the decorated function executes.

    Examples:
        Basic usage with default parameter name:

            >>> @validates_prompt_set()
            ... def process(prompt_set: list[str]) -> int:
            ...     return len(prompt_set)
            >>> process(["Hello", "World"])
            2

        Custom parameter name:

            >>> @validates_prompt_set("questions")
            ... def answer_all(questions: list[str]) -> list[str]:
            ...     return ["Answer" for _ in questions]
            >>> answer_all(["Q1", "Q2", "Q3"])
            ['Answer', 'Answer', 'Answer']

        With length constraints per prompt:

            >>> @validates_prompt_set("inputs", max_length=50)
            ... def batch_process(inputs: list[str]) -> int:
            ...     return sum(len(s) for s in inputs)
            >>> batch_process(["Short", "Also short"])
            15

        Validation error for empty set:

            >>> @validates_prompt_set("prompts")
            ... def generate(prompts: list[str]) -> list:
            ...     return []
            >>> try:
            ...     generate([])
            ... except ValidationError as e:
            ...     print(e.message)
            Prompt set cannot be empty

        Allowing empty sets:

            >>> @validates_prompt_set("items", allow_empty_set=True)
            ... def optional_batch(items: list[str]) -> int:
            ...     return len(items)
            >>> optional_batch([])
            0

        Works with keyword arguments:

            >>> @validates_prompt_set("messages")
            ... def send_batch(channel: str, messages: list[str]) -> bool:
            ...     return True
            >>> send_batch("general", messages=["Hi", "Hello"])
            True

        Works with positional arguments:

            >>> @validates_prompt_set("texts")
            ... def analyze_batch(config: dict, texts: list[str]) -> int:
            ...     return len(texts)
            >>> analyze_batch({}, ["Text 1", "Text 2"])
            2

        With class methods:

            >>> class BatchProcessor:
            ...     @validates_prompt_set("items")
            ...     def process(self, items: list[str]) -> list:
            ...         return [item.upper() for item in items]
            >>> bp = BatchProcessor()
            >>> bp.process(["a", "b", "c"])
            ['A', 'B', 'C']

        Validating chat-style messages (dicts are allowed):

            >>> @validates_prompt_set("messages")
            ... def chat(messages: list) -> str:
            ...     return "response"
            >>> chat([{"role": "user", "content": "Hi"}])
            'response'

        Error includes index for invalid prompts:

            >>> @validates_prompt_set("prompts")
            ... def run(prompts: list[str]) -> None:
            ...     pass
            >>> try:
            ...     run(["Valid", "", "Also valid"])
            ... except ValidationError as e:
            ...     print("[1]" in e.field)
            True
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
