"""Output validation wrapper (strict vs warn)."""

from __future__ import annotations

import warnings
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, Optional

from insideLLMs.schemas.exceptions import OutputValidationError
from insideLLMs.schemas.registry import SchemaRegistry, normalize_semver

ValidationMode = Literal["strict", "warn"]


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


class OutputValidator:
    """Validate insideLLMs outputs against versioned schemas."""

    def __init__(self, registry: Optional[SchemaRegistry] = None) -> None:
        self.registry = registry or SchemaRegistry()

    def validate(
        self,
        schema_name: str,
        data: Any,
        *,
        schema_version: str = "1.0.0",
        mode: ValidationMode = "strict",
    ) -> Any:
        """Validate and return parsed Pydantic model (or return data in warn mode)."""

        model = self.registry.get_model(schema_name, schema_version)
        version = normalize_semver(schema_version)
        try:
            if hasattr(model, "model_validate"):
                return model.model_validate(_to_plain(data))
            return model.parse_obj(_to_plain(data))
        except Exception as e:  # ValidationError (v1/v2), keep generic
            errors = [str(e)]
            if mode == "warn":
                warnings.warn(
                    f"Output validation failed for {schema_name}@{version}: {errors[0]}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return data
            raise OutputValidationError(
                schema_name=schema_name, schema_version=version, errors=errors
            ) from e
