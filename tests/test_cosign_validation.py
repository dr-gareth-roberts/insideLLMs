"""Tests for cosign identity_constraints validation."""

import pytest

from insideLLMs.signing.cosign import _validate_identity_constraints


def test_validate_identity_constraints_valid() -> None:
    """Test that valid identity constraints are accepted."""
    # Valid patterns should not raise
    _validate_identity_constraints("issuer=example@example.com")
    _validate_identity_constraints("subject=user@domain.org")
    _validate_identity_constraints("issuer=https://accounts.google.com")
    _validate_identity_constraints("user@example.com")
    _validate_identity_constraints("https://example.com/path")
    _validate_identity_constraints("key=value123")
    _validate_identity_constraints("test-value_123")
    _validate_identity_constraints("a:b+c=d@e.f/g-h_i")


def test_validate_identity_constraints_invalid_characters() -> None:
    """Test that invalid characters are rejected."""
    # Characters like ; $ & | < > ` should be rejected to prevent injection
    invalid_inputs = [
        "valid; rm -rf /",
        "valid && malicious",
        "valid | cat",
        "valid$(whoami)",
        "valid`whoami`",
        "valid&background",
        "valid>file",
        "valid<file",
        "valid;malicious",
        "test\nvalue",
        "test value",  # spaces
        "test'value",  # single quotes
        'test"value',  # double quotes
        "test\\value",  # backslash
        "test*value",  # asterisk
        "test?value",  # question mark
        "test[value]",  # brackets
        "test{value}",  # braces
        "test(value)",  # parentheses
    ]

    for invalid_input in invalid_inputs:
        with pytest.raises(ValueError, match="Invalid identity_constraints"):
            _validate_identity_constraints(invalid_input)
