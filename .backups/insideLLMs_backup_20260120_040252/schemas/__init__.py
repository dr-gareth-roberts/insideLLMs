"""Versioned output schemas and validation utilities.

This package provides **versioned** (SemVer) schemas for the serialized outputs
of insideLLMs.

Pydantic is intentionally treated as an optional dependency: importing this
package is safe, but using schema models/validation requires Pydantic.
"""

from __future__ import annotations

from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
from insideLLMs.schemas.exceptions import OutputValidationError
from insideLLMs.schemas.registry import SchemaRegistry
from insideLLMs.schemas.validator import OutputValidator, ValidationMode

__all__ = [
    "DEFAULT_SCHEMA_VERSION",
    "OutputValidationError",
    "SchemaRegistry",
    "OutputValidator",
    "ValidationMode",
]
