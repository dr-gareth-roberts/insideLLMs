"""Schema registry for mapping (schema_name, version) -> Pydantic model.

The primary goal is to make serialized outputs stable and machine-checkable.
This registry provides:
  - lookup of versioned schema models
  - JSON Schema export
  - a place to register migrations between versions

Pydantic remains an optional dependency: schema models are only imported
when requested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type


def normalize_semver(version: str) -> str:
    """Normalize a version string into SemVer form.

    Accepts legacy forms like "1.0" and normalizes to "1.0.0".
    """

    if re.match(r"^[0-9]+\.[0-9]+\.[0-9]+$", version):
        return version
    if re.match(r"^[0-9]+\.[0-9]+$", version):
        return f"{version}.0"
    return version


def _require_pydantic() -> None:
    try:
        import pydantic  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Pydantic is required for output schema validation. Install with: pip install pydantic"
        ) from e


@dataclass(frozen=True)
class SchemaHandle:
    schema_name: str
    schema_version: str
    model: Type[Any]


class SchemaRegistry:
    """Registry for versioned output schemas."""

    # Canonical schema names exposed by insideLLMs.
    RUNNER_ITEM = "ProbeResult"  # per-item result from ProbeRunner.run
    RUNNER_OUTPUT = "RunnerOutput"  # wrapper for a batch of results
    RESULT_RECORD = "ResultRecord"  # JSONL record for ProbeRunner.run()
    RUN_MANIFEST = "RunManifest"  # manifest.json for run directories
    HARNESS_RECORD = "HarnessRecord"  # per-line JSONL record
    HARNESS_SUMMARY = "HarnessSummary"  # summary.json payload
    BENCHMARK_SUMMARY = "BenchmarkSummary"  # benchmark run output
    COMPARISON_REPORT = "ComparisonReport"  # comparison output
    DIFF_REPORT = "DiffReport"  # diff.json output
    EXPORT_METADATA = "ExportMetadata"  # export bundle metadata

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], SchemaHandle] = {}

    def available_versions(self, schema_name: str) -> list[str]:
        if schema_name not in {
            self.RUNNER_ITEM,
            self.RUNNER_OUTPUT,
            self.RESULT_RECORD,
            self.RUN_MANIFEST,
            self.HARNESS_RECORD,
            self.HARNESS_SUMMARY,
            self.BENCHMARK_SUMMARY,
            self.COMPARISON_REPORT,
            self.DIFF_REPORT,
            self.EXPORT_METADATA,
        }:
            return []
        return ["1.0.0", "1.0.1"]

    def get_model(self, schema_name: str, schema_version: str) -> Type[Any]:
        """Return the Pydantic model class for a schema."""

        _require_pydantic()
        v = normalize_semver(schema_version)

        cache_key = (schema_name, v)
        if cache_key in self._cache:
            return self._cache[cache_key].model

        if v == "1.0.0":
            from insideLLMs.schemas import v1_0_0

            model = v1_0_0.get_schema_model(schema_name)
        elif v == "1.0.1":
            from insideLLMs.schemas import v1_0_1

            model = v1_0_1.get_schema_model(schema_name)
        else:
            raise KeyError(f"Unknown schema version: {schema_name}@{v}")

        self._cache[cache_key] = SchemaHandle(
            schema_name=schema_name, schema_version=v, model=model
        )
        return model

    def get_json_schema(self, schema_name: str, schema_version: str) -> dict[str, Any]:
        """Generate JSON Schema for a schema model."""

        _require_pydantic()
        from insideLLMs.structured import get_json_schema

        model = self.get_model(schema_name, schema_version)
        schema = get_json_schema(model)
        schema.setdefault("title", f"{schema_name}@{normalize_semver(schema_version)}")
        return schema

    def migrate(
        self,
        schema_name: str,
        data: Any,
        from_version: str,
        to_version: str,
        *,
        custom_migration: Optional[Callable[[Any], Any]] = None,
    ) -> Any:
        """Migrate data between schema versions.

        Initial implementation supports:
          - legacy normalization ("1.0" -> "1.0.0")
          - identity migrations
        """

        f = normalize_semver(from_version)
        t = normalize_semver(to_version)
        if f == t:
            return custom_migration(data) if custom_migration else data
        raise NotImplementedError(f"No migration registered for {schema_name}: {f} -> {t}")
