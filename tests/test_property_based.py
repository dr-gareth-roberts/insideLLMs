"""Property-based tests using Hypothesis for validation, config, and serialization.

These tests verify invariants that should hold for ALL valid inputs, not just
hand-picked examples. Hypothesis generates thousands of random inputs to find
edge cases that manual testing would miss.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from insideLLMs.validation import (
    ValidationError,
    ValidationResult,
    validate_choice,
    validate_positive_int,
    validate_prompt,
    validate_prompt_set,
)

# =============================================================================
# Strategies
# =============================================================================

# Strategy for valid prompt strings (non-empty, reasonable length)
valid_prompts = st.text(min_size=1, max_size=10_000).filter(lambda s: s.strip() != "")

# Strategy for valid prompt sets
valid_prompt_sets = st.lists(valid_prompts, min_size=1, max_size=50)

# Strategy for positive integers
positive_ints = st.integers(min_value=1, max_value=10_000)


# =============================================================================
# ValidationResult Properties
# =============================================================================


class TestValidationResultProperties:
    """Property-based tests for ValidationResult merge semantics."""

    @given(
        valid1=st.booleans(),
        valid2=st.booleans(),
        errors1=st.lists(st.text(min_size=1, max_size=50), max_size=5),
        errors2=st.lists(st.text(min_size=1, max_size=50), max_size=5),
        warnings1=st.lists(st.text(min_size=1, max_size=50), max_size=5),
        warnings2=st.lists(st.text(min_size=1, max_size=50), max_size=5),
    )
    def test_merge_validity_is_conjunction(
        self, valid1, valid2, errors1, errors2, warnings1, warnings2
    ):
        """Merged validity is AND of both validities."""
        r1 = ValidationResult(valid=valid1, errors=errors1, warnings=warnings1)
        r2 = ValidationResult(valid=valid2, errors=errors2, warnings=warnings2)
        merged = r1.merge(r2)

        assert merged.valid == (valid1 and valid2)

    @given(
        errors1=st.lists(st.text(min_size=1, max_size=50), max_size=5),
        errors2=st.lists(st.text(min_size=1, max_size=50), max_size=5),
    )
    def test_merge_errors_concatenate(self, errors1, errors2):
        """Merged errors are concatenation of both error lists."""
        r1 = ValidationResult(errors=errors1)
        r2 = ValidationResult(errors=errors2)
        merged = r1.merge(r2)

        assert merged.errors == errors1 + errors2

    @given(
        warnings1=st.lists(st.text(min_size=1, max_size=50), max_size=5),
        warnings2=st.lists(st.text(min_size=1, max_size=50), max_size=5),
    )
    def test_merge_warnings_concatenate(self, warnings1, warnings2):
        """Merged warnings are concatenation of both warning lists."""
        r1 = ValidationResult(warnings=warnings1)
        r2 = ValidationResult(warnings=warnings2)
        merged = r1.merge(r2)

        assert merged.warnings == warnings1 + warnings2

    @given(
        valid=st.booleans(),
        errors=st.lists(st.text(min_size=1, max_size=50), max_size=5),
    )
    def test_merge_is_associative(self, valid, errors):
        """Merge of three results is the same regardless of grouping."""
        r1 = ValidationResult(valid=True, errors=["e1"])
        r2 = ValidationResult(valid=valid, errors=errors)
        r3 = ValidationResult(valid=True, errors=["e3"])

        left = r1.merge(r2).merge(r3)
        right = r1.merge(r2.merge(r3))

        assert left.valid == right.valid
        assert left.errors == right.errors

    @given(
        valid=st.booleans(),
        errors=st.lists(st.text(min_size=1, max_size=50), max_size=5),
    )
    def test_merge_identity(self, valid, errors):
        """Merging with an empty valid result returns the original."""
        r = ValidationResult(valid=valid, errors=errors)
        identity = ValidationResult(valid=True, errors=[], warnings=[])
        merged = r.merge(identity)

        assert merged.valid == r.valid
        assert merged.errors == r.errors

    @given(st.booleans())
    def test_bool_matches_valid(self, valid):
        """bool(result) always matches result.valid."""
        r = ValidationResult(valid=valid)
        assert bool(r) == valid


# =============================================================================
# validate_prompt Properties
# =============================================================================


class TestValidatePromptProperties:
    """Property-based tests for validate_prompt."""

    @given(prompt=valid_prompts)
    def test_valid_prompts_always_pass(self, prompt):
        """Any non-empty string passes basic validation."""
        assert validate_prompt(prompt) is True

    @given(prompt=valid_prompts, max_length=positive_ints)
    def test_length_constraint_respected(self, prompt, max_length):
        """If prompt is within max_length, it passes; otherwise it raises."""
        if len(prompt) <= max_length:
            assert validate_prompt(prompt, max_length=max_length) is True
        else:
            with pytest.raises(ValidationError):
                validate_prompt(prompt, max_length=max_length)

    @given(prompt=st.text(min_size=0, max_size=0))
    def test_empty_string_fails(self, prompt):
        """Empty strings always fail without allow_empty."""
        with pytest.raises(ValidationError):
            validate_prompt(prompt)

    @given(prompt=st.text(min_size=0, max_size=0))
    def test_empty_string_passes_with_flag(self, prompt):
        """Empty strings pass when allow_empty=True."""
        assert validate_prompt(prompt, allow_empty=True) is True

    @given(value=st.one_of(st.integers(), st.floats(), st.none(), st.binary()))
    def test_non_string_types_fail(self, value):
        """Non-string values always raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_prompt(value)


# =============================================================================
# validate_prompt_set Properties
# =============================================================================


class TestValidatePromptSetProperties:
    """Property-based tests for validate_prompt_set."""

    @given(prompts=valid_prompt_sets)
    def test_valid_prompt_sets_pass(self, prompts):
        """Any non-empty list of non-empty strings passes."""
        assert validate_prompt_set(prompts) is True

    @given(
        prompts=valid_prompt_sets,
        extra_dicts=st.lists(
            st.fixed_dictionaries({"role": st.text(), "content": st.text()}),
            max_size=3,
        ),
    )
    def test_dicts_in_prompt_set_are_allowed(self, prompts, extra_dicts):
        """Dict items mixed into prompt sets are allowed (chat format)."""
        mixed = list(prompts) + extra_dicts
        assert validate_prompt_set(mixed) is True


# =============================================================================
# validate_positive_int Properties
# =============================================================================


class TestValidatePositiveIntProperties:
    """Property-based tests for validate_positive_int."""

    @given(value=st.integers(min_value=1, max_value=100_000))
    def test_positive_ints_pass(self, value):
        """Any positive integer passes default validation."""
        assert validate_positive_int(value) == value

    @given(value=st.integers(max_value=0))
    def test_non_positive_ints_fail(self, value):
        """Zero and negative integers always fail."""
        with pytest.raises(ValidationError):
            validate_positive_int(value)

    @given(
        value=st.integers(min_value=1, max_value=100),
        min_val=st.integers(min_value=1, max_value=50),
        max_val=st.integers(min_value=51, max_value=200),
    )
    def test_bounds_are_respected(self, value, min_val, max_val):
        """Values within [min, max] pass; values outside raise."""
        if min_val <= value <= max_val:
            assert validate_positive_int(value, min_value=min_val, max_value=max_val) == value
        else:
            with pytest.raises(ValidationError):
                validate_positive_int(value, min_value=min_val, max_value=max_val)

    @given(value=st.booleans())
    def test_booleans_are_rejected(self, value):
        """Boolean values are rejected even though bool is subclass of int."""
        with pytest.raises(ValidationError):
            validate_positive_int(value)


# =============================================================================
# validate_choice Properties
# =============================================================================


class TestValidateChoiceProperties:
    """Property-based tests for validate_choice."""

    @given(
        choices=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10, unique=True),
        index=st.integers(min_value=0),
    )
    def test_valid_choices_pass(self, choices, index):
        """Any value that is in the allowed set passes."""
        assume(index < len(choices))
        value = choices[index]
        assert validate_choice(value, choices) == value

    @given(
        choices=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10, unique=True),
        wrong=st.text(min_size=1, max_size=20),
    )
    def test_invalid_choices_fail(self, choices, wrong):
        """Values not in the allowed set always raise."""
        assume(wrong not in choices)
        with pytest.raises(ValidationError):
            validate_choice(wrong, choices)


# =============================================================================
# JSON Serialization Round-Trip Properties
# =============================================================================


class TestSerializationProperties:
    """Property-based tests for data round-trip integrity."""

    @given(
        data=st.recursive(
            st.one_of(
                st.none(),
                st.booleans(),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.text(max_size=100),
            ),
            lambda children: st.one_of(
                st.lists(children, max_size=5),
                st.dictionaries(st.text(min_size=1, max_size=20), children, max_size=5),
            ),
            max_leaves=20,
        )
    )
    @settings(max_examples=200)
    def test_json_round_trip(self, data):
        """JSON serialization/deserialization round-trips for all valid JSON types."""
        serialized = json.dumps(data)
        deserialized = json.loads(serialized)
        assert deserialized == data
