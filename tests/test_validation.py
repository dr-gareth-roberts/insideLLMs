"""Tests for insideLLMs/validation.py module."""

import pytest

from insideLLMs.validation import (
    ValidationError,
    ValidationResult,
    validate_choice,
    validate_positive_int,
    validate_prompt,
    validate_prompt_set,
    validates_prompt,
    validates_prompt_set,
)


class TestValidationError:
    """Tests for ValidationError class."""

    def test_basic_error(self):
        """Test basic ValidationError creation."""
        error = ValidationError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.field is None
        assert error.value is None
        assert error.suggestions == []

    def test_error_with_field(self):
        """Test ValidationError with field name."""
        error = ValidationError("Invalid value", field="prompt")
        assert "[prompt]" in str(error)
        assert error.field == "prompt"

    def test_error_with_value(self):
        """Test ValidationError with value."""
        error = ValidationError("Invalid value", field="prompt", value="test")
        assert error.value == "test"

    def test_error_with_suggestions(self):
        """Test ValidationError with suggestions."""
        error = ValidationError("Invalid value", suggestions=["Try this", "Or this"])
        assert "Suggestions:" in str(error)
        assert "Try this" in str(error)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_valid_result(self):
        """Test valid ValidationResult."""
        result = ValidationResult()
        assert result.valid is True
        assert bool(result) is True
        assert result.errors == []
        assert result.warnings == []

    def test_invalid_result(self):
        """Test invalid ValidationResult."""
        result = ValidationResult(valid=False, errors=["Error 1"])
        assert result.valid is False
        assert bool(result) is False
        assert "Error 1" in result.errors

    def test_result_with_warnings(self):
        """Test ValidationResult with warnings."""
        result = ValidationResult(valid=True, warnings=["Warning 1"])
        assert result.valid is True
        assert "Warning 1" in result.warnings

    def test_merge_results(self):
        """Test merging ValidationResults."""
        result1 = ValidationResult(valid=True, warnings=["W1"])
        result2 = ValidationResult(valid=False, errors=["E1"], warnings=["W2"])

        merged = result1.merge(result2)
        assert merged.valid is False  # One was invalid
        assert "E1" in merged.errors
        assert "W1" in merged.warnings
        assert "W2" in merged.warnings

    def test_merge_both_valid(self):
        """Test merging two valid results."""
        result1 = ValidationResult(valid=True)
        result2 = ValidationResult(valid=True)

        merged = result1.merge(result2)
        assert merged.valid is True


class TestValidatePrompt:
    """Tests for validate_prompt function."""

    def test_valid_prompt(self):
        """Test valid prompt."""
        result = validate_prompt("Hello, world!")
        assert result is True

    def test_none_prompt(self):
        """Test None prompt raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt(None)
        assert "cannot be None" in str(exc_info.value)

    def test_non_string_prompt(self):
        """Test non-string prompt raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt(123)
        assert "must be a string" in str(exc_info.value)

    def test_empty_prompt_not_allowed(self):
        """Test empty prompt raises error by default."""
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt("")
        assert "cannot be empty" in str(exc_info.value)

    def test_empty_prompt_allowed(self):
        """Test empty prompt when allowed."""
        result = validate_prompt("", allow_empty=True)
        assert result is True

    def test_prompt_too_short(self):
        """Test prompt shorter than min_length."""
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt("Hi", min_length=10)
        assert "too short" in str(exc_info.value)

    def test_prompt_too_long(self):
        """Test prompt longer than max_length."""
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt("A" * 100, max_length=50)
        assert "too long" in str(exc_info.value)

    def test_custom_field_name(self):
        """Test custom field name in error message."""
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt(None, field_name="my_prompt")
        assert "[my_prompt]" in str(exc_info.value)


class TestValidatePromptSet:
    """Tests for validate_prompt_set function."""

    def test_valid_prompt_set(self):
        """Test valid prompt set."""
        result = validate_prompt_set(["Hello", "World"])
        assert result is True

    def test_valid_tuple_prompt_set(self):
        """Test valid tuple prompt set."""
        result = validate_prompt_set(("Hello", "World"))
        assert result is True

    def test_none_prompt_set(self):
        """Test None prompt set raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt_set(None)
        assert "cannot be None" in str(exc_info.value)

    def test_non_list_prompt_set(self):
        """Test non-list prompt set raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt_set("not a list")
        assert "must be a list or tuple" in str(exc_info.value)

    def test_empty_prompt_set_not_allowed(self):
        """Test empty prompt set raises error by default."""
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt_set([])
        assert "cannot be empty" in str(exc_info.value)

    def test_empty_prompt_set_allowed(self):
        """Test empty prompt set when allowed."""
        result = validate_prompt_set([], allow_empty_set=True)
        assert result is True

    def test_invalid_prompt_in_set(self):
        """Test invalid prompt within set raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt_set(["Valid", None, "Also valid"])
        assert "cannot be None" in str(exc_info.value)

    def test_dict_items_allowed(self):
        """Test dict items are allowed in prompt set."""
        result = validate_prompt_set([{"role": "user", "content": "Hello"}])
        assert result is True

    def test_max_length_applied_to_items(self):
        """Test max_length is applied to each prompt."""
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt_set(["Short", "A" * 100], max_length=50)
        assert "too long" in str(exc_info.value)


class TestValidatePositiveInt:
    """Tests for validate_positive_int function."""

    def test_valid_positive_int(self):
        """Test valid positive integer."""
        result = validate_positive_int(5)
        assert result == 5

    def test_none_value(self):
        """Test None value raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_int(None)
        assert "cannot be None" in str(exc_info.value)

    def test_non_int_value(self):
        """Test non-integer raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_int("5")
        assert "must be an integer" in str(exc_info.value)

    def test_bool_value(self):
        """Test bool is rejected even though it's technically int."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_int(True)
        assert "must be an integer" in str(exc_info.value)

    def test_value_below_min(self):
        """Test value below min_value raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_int(0, min_value=1)
        assert "must be >= 1" in str(exc_info.value)

    def test_value_above_max(self):
        """Test value above max_value raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_int(100, max_value=50)
        assert "must be <= 50" in str(exc_info.value)

    def test_custom_min_value(self):
        """Test custom min_value."""
        result = validate_positive_int(0, min_value=0)
        assert result == 0


class TestValidateChoice:
    """Tests for validate_choice function."""

    def test_valid_choice(self):
        """Test valid choice."""
        result = validate_choice("a", ["a", "b", "c"])
        assert result == "a"

    def test_invalid_choice(self):
        """Test invalid choice raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_choice("d", ["a", "b", "c"])
        assert "must be one of" in str(exc_info.value)

    def test_choice_with_int(self):
        """Test choice with integers."""
        result = validate_choice(2, [1, 2, 3])
        assert result == 2

    def test_choice_suggestions(self):
        """Test choice error includes suggestions."""
        with pytest.raises(ValidationError) as exc_info:
            validate_choice("x", ["a", "b"])
        assert "Use one of:" in str(exc_info.value)


class TestValidatesPromptDecorator:
    """Tests for validates_prompt decorator."""

    def test_decorator_with_valid_prompt(self):
        """Test decorator passes with valid prompt."""

        @validates_prompt("prompt")
        def my_func(prompt):
            return prompt.upper()

        result = my_func("hello")
        assert result == "HELLO"

    def test_decorator_with_invalid_prompt(self):
        """Test decorator rejects invalid prompt (non-string)."""

        @validates_prompt("prompt")
        def my_func(prompt):
            return str(prompt)

        # Decorator validates when value is not None but is wrong type
        with pytest.raises(ValidationError):
            my_func(123)

    def test_decorator_with_empty_prompt(self):
        """Test decorator rejects empty prompt."""

        @validates_prompt("prompt")
        def my_func(prompt):
            return prompt

        with pytest.raises(ValidationError):
            my_func("")

    def test_decorator_with_kwargs(self):
        """Test decorator works with keyword arguments."""

        @validates_prompt("prompt")
        def my_func(prompt):
            return prompt.upper()

        result = my_func(prompt="hello")
        assert result == "HELLO"

    def test_decorator_with_validation_kwargs(self):
        """Test decorator with validation kwargs."""

        @validates_prompt("prompt", max_length=5)
        def my_func(prompt):
            return prompt

        with pytest.raises(ValidationError):
            my_func("too long string")

    def test_decorator_with_none_skipped(self):
        """Test decorator skips validation when value is None in call."""

        @validates_prompt("prompt")
        def my_func(other, prompt=None):
            return other

        # Should not raise since prompt is not provided
        result = my_func("value")
        assert result == "value"


class TestValidatesPromptSetDecorator:
    """Tests for validates_prompt_set decorator."""

    def test_decorator_with_valid_set(self):
        """Test decorator passes with valid prompt set."""

        @validates_prompt_set("prompts")
        def my_func(prompts):
            return len(prompts)

        result = my_func(["a", "b", "c"])
        assert result == 3

    def test_decorator_with_invalid_set(self):
        """Test decorator rejects invalid prompt set (non-list)."""

        @validates_prompt_set("prompts")
        def my_func(prompts):
            return len(prompts)

        with pytest.raises(ValidationError):
            my_func("not a list")

    def test_decorator_with_kwargs(self):
        """Test decorator works with keyword arguments."""

        @validates_prompt_set("prompts")
        def my_func(prompts):
            return len(prompts)

        result = my_func(prompts=["a", "b"])
        assert result == 2

    def test_decorator_with_empty_set(self):
        """Test decorator rejects empty set by default."""

        @validates_prompt_set("prompts")
        def my_func(prompts):
            return len(prompts)

        with pytest.raises(ValidationError):
            my_func([])

    def test_decorator_allows_empty_set(self):
        """Test decorator allows empty set when configured."""

        @validates_prompt_set("prompts", allow_empty_set=True)
        def my_func(prompts):
            return len(prompts)

        result = my_func([])
        assert result == 0
