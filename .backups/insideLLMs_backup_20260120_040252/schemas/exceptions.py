"""Schema validation exceptions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutputValidationError(ValueError):
    """Raised when output validation fails in strict mode."""

    schema_name: str
    schema_version: str
    errors: list[str]

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"Output validation failed for {self.schema_name}@{self.schema_version}: "
            + "; ".join(self.errors)
        )
