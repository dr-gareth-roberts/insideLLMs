"""Schema definitions for the output contract v1.0.0.

These schemas mirror the public, serialized outputs produced by runner/benchmark/
comparison/export. They are intentionally strict (extra fields forbidden) to
prevent silent schema drift.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from insideLLMs.schemas.registry import _require_pydantic

_require_pydantic()

from pydantic import BaseModel, Field  # noqa: E402

try:  # Pydantic v2
    from pydantic import ConfigDict  # type: ignore

    _PYDANTIC_V2 = True
except Exception:  # pragma: no cover
    ConfigDict = None  # type: ignore
    _PYDANTIC_V2 = False


SCHEMA_VERSION = "1.0.0"


class _BaseSchema(BaseModel):
    """Base schema with strict field policy."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")
    else:  # Pydantic v1

        class Config:
            extra = "forbid"


class SchemaProbeResult(_BaseSchema):
    """Per-item output produced by ProbeRunner.run()."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    input: Any
    output: Optional[Any] = None
    status: Literal["success", "error", "timeout", "rate_limited", "skipped"]
    error: Optional[str] = None
    error_type: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# Backward compatibility alias
ProbeResult = SchemaProbeResult


class RunnerOutput(_BaseSchema):
    """Batch output wrapper for probe execution."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    results: list[ProbeResult]


class SchemaChatMessage(_BaseSchema):
    """A single chat message in a prompt."""

    role: str
    content: Any


# Backward compatibility alias
ChatMessage = SchemaChatMessage


class ModelSpec(_BaseSchema):
    """Serializable model identity and configuration."""

    model_id: str
    provider: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class ProbeSpec(_BaseSchema):
    """Serializable probe identity and version."""

    probe_id: str
    probe_version: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class DatasetSpec(_BaseSchema):
    """Serializable dataset provenance/version information."""

    dataset_id: Optional[str] = None
    dataset_version: Optional[str] = None
    dataset_hash: Optional[str] = None
    provenance: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class ResultRecord(_BaseSchema):
    """One JSONL record emitted by ProbeRunner.run() for wind-tunnel logging."""

    schema_version: str = Field(default=SCHEMA_VERSION)

    run_id: str
    started_at: datetime
    completed_at: datetime

    model: ModelSpec
    probe: ProbeSpec
    example_id: str
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)

    # Prompt / messages
    input: Optional[Any] = None
    messages: Optional[list[ChatMessage]] = None
    messages_hash: Optional[str] = None
    messages_storage: Optional[str] = None

    # Output
    output: Optional[Any] = None
    output_text: Optional[str] = None

    # Scoring + metrics
    scores: dict[str, Any] = Field(default_factory=dict)
    primary_metric: Optional[str] = None

    # Telemetry
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[float] = None

    status: Literal["success", "error", "timeout", "rate_limited", "skipped"]
    error: Optional[str] = None
    error_type: Optional[str] = None

    custom: dict[str, Any] = Field(default_factory=dict)


class RunManifest(_BaseSchema):
    """Run directory manifest emitted alongside records.jsonl."""

    schema_version: str = Field(default=SCHEMA_VERSION)

    run_id: str
    created_at: datetime
    started_at: datetime
    completed_at: datetime

    library_version: Optional[str] = None
    python_version: Optional[str] = None
    platform: Optional[str] = None
    command: Optional[list[str]] = None

    model: ModelSpec
    probe: ProbeSpec
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)

    record_count: int = 0
    success_count: int = 0
    error_count: int = 0

    records_file: str = "records.jsonl"
    schemas: dict[str, Any] = Field(default_factory=dict)
    custom: dict[str, Any] = Field(default_factory=dict)


class HarnessRecord(_BaseSchema):
    """One JSONL record emitted by run_harness_from_config()."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    experiment_id: str

    model_type: Optional[str] = None
    model_name: str
    model_id: Optional[str] = None
    model_provider: Optional[str] = None

    probe_type: Optional[str] = None
    probe_name: str
    probe_category: str

    dataset: str
    dataset_format: Optional[str] = None
    example_index: int

    input: Any
    output: Optional[Any] = None
    status: str
    error: Optional[str] = None
    error_type: Optional[str] = None
    latency_ms: Optional[float] = None

    started_at: datetime
    completed_at: datetime


class HarnessSummary(_BaseSchema):
    """The summary.json payload emitted by the harness CLI."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    generated_at: datetime
    summary: dict[str, Any]
    config: dict[str, Any]


class BenchmarkModelMetrics(_BaseSchema):
    total_time: float
    avg_time_per_item: float
    total_items: int
    success_rate: float
    error_rate: float
    mean_latency_ms: float


class BenchmarkModelResult(_BaseSchema):
    model: dict[str, Any]
    results: list[ProbeResult]
    metrics: BenchmarkModelMetrics


class BenchmarkProbeMetrics(_BaseSchema):
    total_time: float
    avg_time_per_item: float
    total_items: int
    success_rate: float


class BenchmarkProbeResult(_BaseSchema):
    probe: str
    results: list[ProbeResult]
    metrics: BenchmarkProbeMetrics


class BenchmarkSummary(_BaseSchema):
    """Output for ModelBenchmark.run() or ProbeBenchmark.run()."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    name: str
    timestamp: Optional[float] = None

    # ModelBenchmark.run()
    probe: Optional[str] = None
    models: Optional[list[BenchmarkModelResult]] = None

    # ProbeBenchmark.run()
    model: Optional[dict[str, Any]] = None
    probes: Optional[list[BenchmarkProbeResult]] = None


class ComparisonReport(_BaseSchema):
    """Comparison output contract.

    This is used by benchmark comparisons (dict with metrics/rankings) and also
    supports richer comparison structures.
    """

    schema_version: str = Field(default=SCHEMA_VERSION)
    name: str

    # Benchmark comparisons
    metrics: dict[str, Any] = Field(default_factory=dict)
    rankings: dict[str, list[str]] = Field(default_factory=dict)

    # Richer comparison structures
    models: Optional[list[str]] = None
    winner: Optional[str] = None
    differences: dict[str, Any] = Field(default_factory=dict)
    summary: Optional[str] = None
    significant_differences: dict[str, Any] = Field(default_factory=dict)


class DiffRecordKey(_BaseSchema):
    model_id: str
    probe_id: str
    item_id: str


class DiffRecordLabel(_BaseSchema):
    model: str
    probe: str
    example: str


class DiffRecordIdentity(_BaseSchema):
    record_key: DiffRecordKey
    model_id: str
    probe_id: str
    example_id: str
    replicate_key: Optional[str] = None
    label: DiffRecordLabel


class DiffOutputSummary(_BaseSchema):
    type: Literal["text", "structured"]
    preview: Optional[str] = None
    length: Optional[int] = None
    fingerprint: Optional[str] = None


class DiffRecordSummary(_BaseSchema):
    status: str
    primary_metric: Optional[str] = None
    primary_score: Optional[float] = None
    scores_keys: Optional[list[str]] = None
    output: Optional[DiffOutputSummary] = None


class MetricContext(_BaseSchema):
    primary_metric: Optional[str] = None
    scores_keys: Optional[list[str]] = None
    metric_value: Optional[Any] = None


class MetricMismatchDetails(_BaseSchema):
    baseline: MetricContext
    candidate: MetricContext
    replicate_key: Optional[str] = None


class DiffChangeEntry(DiffRecordIdentity):
    kind: str
    baseline: Optional[DiffRecordSummary] = None
    candidate: Optional[DiffRecordSummary] = None
    detail: Optional[str] = None
    metric: Optional[str] = None
    delta: Optional[float] = None
    reason: Optional[str] = None
    details: Optional[MetricMismatchDetails] = None
    baseline_missing: Optional[list[str]] = None
    candidate_missing: Optional[list[str]] = None
    baseline_fingerprint: Optional[str] = None
    candidate_fingerprint: Optional[str] = None


class DiffRunIds(_BaseSchema):
    baseline: list[str] = Field(default_factory=list)
    candidate: list[str] = Field(default_factory=list)


class DiffCounts(_BaseSchema):
    common: int
    only_baseline: int
    only_candidate: int
    regressions: int
    improvements: int
    other_changes: int


class DiffDuplicates(_BaseSchema):
    baseline: int
    candidate: int


class DiffReport(_BaseSchema):
    schema_version: str = Field(default=SCHEMA_VERSION)
    baseline: str
    candidate: str
    run_ids: DiffRunIds
    counts: DiffCounts
    duplicates: DiffDuplicates
    regressions: list[DiffChangeEntry] = Field(default_factory=list)
    improvements: list[DiffChangeEntry] = Field(default_factory=list)
    changes: list[DiffChangeEntry] = Field(default_factory=list)
    only_baseline: list[DiffRecordIdentity] = Field(default_factory=list)
    only_candidate: list[DiffRecordIdentity] = Field(default_factory=list)


class SchemaExportMetadata(_BaseSchema):
    """Metadata file included with export bundles."""

    export_time: datetime
    version: str = "1.0"
    schema_version: str = Field(default=SCHEMA_VERSION)
    source: str = "insideLLMs"
    record_count: int = 0
    format: str = ""
    compression: str = "none"
    custom: dict[str, Any] = Field(default_factory=dict)


# Backward compatibility alias
ExportMetadata = SchemaExportMetadata


_SCHEMA_MAP = {
    "ProbeResult": ProbeResult,
    "RunnerOutput": RunnerOutput,
    "ResultRecord": ResultRecord,
    "RunManifest": RunManifest,
    "HarnessRecord": HarnessRecord,
    "HarnessSummary": HarnessSummary,
    "BenchmarkSummary": BenchmarkSummary,
    "ComparisonReport": ComparisonReport,
    "DiffReport": DiffReport,
    "ExportMetadata": ExportMetadata,
}


def get_schema_model(schema_name: str):
    if schema_name not in _SCHEMA_MAP:
        raise KeyError(f"Unknown schema name: {schema_name}")
    return _SCHEMA_MAP[schema_name]
