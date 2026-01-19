"""Schema definitions for the output contract v1.0.1.

This version is a small extension of v1.0.0.

Change log:
- RunManifest: add `run_completed: bool` to explicitly mark successful completion.

All other schemas are identical to v1.0.0.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import Field

from insideLLMs.schemas import v1_0_0


SCHEMA_VERSION = "1.0.1"


class RunManifest(v1_0_0._BaseSchema):
    """Run directory manifest emitted alongside records.jsonl."""

    schema_version: str = Field(default=SCHEMA_VERSION)

    run_id: str
    created_at: datetime
    started_at: datetime
    completed_at: datetime
    run_completed: bool = False

    library_version: Optional[str] = None
    python_version: Optional[str] = None
    platform: Optional[str] = None
    command: Optional[list[str]] = None

    model: v1_0_0.ModelSpec
    probe: v1_0_0.ProbeSpec
    dataset: v1_0_0.DatasetSpec = Field(default_factory=v1_0_0.DatasetSpec)

    record_count: int = 0
    success_count: int = 0
    error_count: int = 0

    records_file: str = "records.jsonl"
    schemas: dict[str, Any] = Field(default_factory=dict)
    custom: dict[str, Any] = Field(default_factory=dict)


def get_schema_model(schema_name: str):
    if schema_name == "RunManifest":
        return RunManifest
    # Delegate everything else to v1.0.0 (unchanged in 1.0.1).
    return v1_0_0.get_schema_model(schema_name)
