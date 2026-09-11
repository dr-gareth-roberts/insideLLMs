"""Output contract v1.0.2: explicit per-item behavioural scores.

Historical schemas remain unchanged. Runner items gain optional ``scores``
and ``primary_metric``; the already-existing ResultRecord scoring fields keep
their contract. Unchanged schema families delegate to v1.0.1.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from insideLLMs.schemas import v1_0_0, v1_0_1

SCHEMA_VERSION = "1.0.2"


class SchemaProbeResult(v1_0_0.SchemaProbeResult):
    """Per-item execution outcome with independently measured scores."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    scores: dict[str, float] = Field(default_factory=dict)
    primary_metric: Optional[str] = None


ProbeResult = SchemaProbeResult


class RunnerOutput(v1_0_0._BaseSchema):
    """Batch wrapper whose nested items use the v1.0.2 contract."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    results: list[SchemaProbeResult]


class RunManifest(v1_0_1.RunManifest):
    schema_version: str = Field(default=SCHEMA_VERSION)


class ResultRecord(v1_0_0.ResultRecord):
    schema_version: str = Field(default=SCHEMA_VERSION)


class BenchmarkModelResult(v1_0_0.BenchmarkModelResult):
    results: list[SchemaProbeResult]


class BenchmarkProbeResult(v1_0_0.BenchmarkProbeResult):
    results: list[SchemaProbeResult]


class BenchmarkSummary(v1_0_0.BenchmarkSummary):
    schema_version: str = Field(default=SCHEMA_VERSION)
    models: Optional[list[BenchmarkModelResult]] = None
    probes: Optional[list[BenchmarkProbeResult]] = None


def get_schema_model(schema_name: str) -> type[BaseModel]:
    models: dict[str, type[BaseModel]] = {
        "ProbeResult": SchemaProbeResult,
        "RunnerOutput": RunnerOutput,
        "RunManifest": RunManifest,
        "ResultRecord": ResultRecord,
        "BenchmarkSummary": BenchmarkSummary,
    }
    if schema_name in models:
        return models[schema_name]
    return v1_0_1.get_schema_model(schema_name)
