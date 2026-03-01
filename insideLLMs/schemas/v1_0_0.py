"""Schema definitions for the output contract v1.0.0.

This module defines the Pydantic schema models for insideLLMs output serialization.
These schemas mirror the public, serialized outputs produced by runner, benchmark,
comparison, and export components. They are intentionally strict (extra fields
forbidden) to prevent silent schema drift.

The schemas in this module form the core output contract for version 1.0.0:

Core Execution Schemas:
    - :class:`SchemaProbeResult` (alias: ProbeResult): Per-item probe execution result
    - :class:`RunnerOutput`: Batch wrapper for probe execution results
    - :class:`ResultRecord`: Detailed JSONL record for wind-tunnel logging
    - :class:`RunManifest`: Run directory manifest with execution metadata

Message and Spec Schemas:
    - :class:`SchemaChatMessage` (alias: ChatMessage): Single chat message
    - :class:`ModelSpec`: Model identity and configuration
    - :class:`ProbeSpec`: Probe identity and version
    - :class:`DatasetSpec`: Dataset provenance information

Harness Schemas:
    - :class:`HarnessRecord`: Per-line JSONL harness record
    - :class:`HarnessSummary`: Summary payload from harness CLI
    - :class:`HarnessExplain`: Explainability payload from harness CLI

Benchmark Schemas:
    - :class:`BenchmarkModelMetrics`: Per-model benchmark metrics
    - :class:`BenchmarkModelResult`: Model results with metrics
    - :class:`BenchmarkProbeMetrics`: Per-probe benchmark metrics
    - :class:`BenchmarkProbeResult`: Probe results with metrics
    - :class:`BenchmarkSummary`: Complete benchmark output

Comparison and Diff Schemas:
    - :class:`ComparisonReport`: Benchmark comparison output
    - :class:`DiffReport`: Detailed diff analysis output
    - Various supporting diff schemas (DiffRecordKey, DiffChangeEntry, etc.)

Export Schema:
    - :class:`SchemaExportMetadata` (alias: ExportMetadata): Export bundle metadata

Examples:
    Creating a probe result:

        >>> from insideLLMs.schemas.v1_0_0 import ProbeResult
        >>> result = ProbeResult(
        ...     input="What is 2+2?",
        ...     output="4",
        ...     status="success",
        ...     latency_ms=150.5
        ... )
        >>> result.status
        'success'

    Creating a result record with full metadata:

        >>> from datetime import datetime
        >>> from insideLLMs.schemas.v1_0_0 import ResultRecord, ModelSpec, ProbeSpec
        >>> record = ResultRecord(
        ...     run_id="run-abc123",
        ...     started_at=datetime.now(),
        ...     completed_at=datetime.now(),
        ...     model=ModelSpec(model_id="gpt-4", provider="openai"),
        ...     probe=ProbeSpec(probe_id="factual-qa"),
        ...     example_id="ex-001",
        ...     status="success",
        ...     input="What is the capital of France?",
        ...     output="Paris"
        ... )
        >>> record.model.model_id
        'gpt-4'

    Building a benchmark summary:

        >>> from insideLLMs.schemas.v1_0_0 import (
        ...     BenchmarkSummary,
        ...     BenchmarkModelResult,
        ...     BenchmarkModelMetrics
        ... )
        >>> summary = BenchmarkSummary(
        ...     name="accuracy-benchmark",
        ...     probe="factual-qa",
        ...     models=[
        ...         BenchmarkModelResult(
        ...             model={"model_id": "gpt-4"},
        ...             results=[],
        ...             metrics=BenchmarkModelMetrics(
        ...                 total_time=10.5,
        ...                 avg_time_per_item=1.05,
        ...                 total_items=10,
        ...                 success_rate=0.9,
        ...                 error_rate=0.1,
        ...                 mean_latency_ms=100.0
        ...             )
        ...         )
        ...     ]
        ... )
        >>> summary.name
        'accuracy-benchmark'

    Using schema lookup:

        >>> from insideLLMs.schemas.v1_0_0 import get_schema_model
        >>> ProbeResultModel = get_schema_model("ProbeResult")
        >>> result = ProbeResultModel(input="test", status="success")
        >>> result.input
        'test'

Attributes:
    SCHEMA_VERSION (str): The version string for this schema module ("1.0.0").

Note:
    All schemas inherit from ``_BaseSchema`` which enforces ``extra="forbid"``
    to ensure strict validation and prevent schema drift.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from insideLLMs.schemas.registry import _require_pydantic

_require_pydantic()

from pydantic import BaseModel, Field  # noqa: E402

try:  # Pydantic v2
    from pydantic import ConfigDict  # type: ignore[attr-defined]  # ConfigDict only in Pydantic v2

    _PYDANTIC_V2 = True
except Exception:  # pragma: no cover
    ConfigDict = None  # type: ignore[assignment]  # None fallback for Pydantic v1
    _PYDANTIC_V2 = False


SCHEMA_VERSION = "1.0.0"
"""The version string for this schema module."""


class _BaseSchema(BaseModel):
    """Base schema with strict field policy.

    All schema models inherit from this base class which enforces that no
    extra fields are allowed during validation. This strict policy prevents
    silent schema drift and ensures that data conforms exactly to the
    expected structure.

    The class automatically configures Pydantic (v1 or v2) to forbid extra
    fields, providing consistent behavior across Pydantic versions.

    Examples:
        Extra fields cause validation errors:

            >>> from insideLLMs.schemas.v1_0_0 import ProbeResult
            >>> try:
            ...     result = ProbeResult(
            ...         input="test",
            ...         status="success",
            ...         unknown_field="value"  # Not allowed
            ...     )
            ... except Exception as e:
            ...     print("Validation error: extra field not allowed")
            Validation error: extra field not allowed

    Note:
        This is an internal base class. Use the specific schema classes
        (ProbeResult, ResultRecord, etc.) for your data models.
    """

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid", protected_namespaces=())
    else:  # Pydantic v1

        class Config:
            extra = "forbid"


class SchemaProbeResult(_BaseSchema):
    """Per-item output produced by ProbeRunner.run().

    This is the fundamental output schema for individual probe executions.
    Each instance represents the result of running a single probe against
    a single input, capturing the output, status, timing, and any errors.

    Attributes:
        schema_version: The schema version string, defaults to "1.0.0".
        input: The input data that was provided to the probe. Can be any
            JSON-serializable value (string, dict, list, etc.).
        output: The output produced by the probe. None if the probe failed
            before producing output.
        status: The execution status. One of:
            - "success": Probe completed successfully
            - "error": Probe encountered an error
            - "timeout": Probe exceeded time limit
            - "rate_limited": API rate limit was hit
            - "skipped": Probe was skipped (e.g., filter condition)
        error: Human-readable error message if status is not "success".
        error_type: The error type/class name for programmatic error handling.
        latency_ms: Execution time in milliseconds, if measured.
        metadata: Additional key-value metadata for custom extensions.

    Examples:
        Creating a successful result:

            >>> result = SchemaProbeResult(
            ...     input="What is 2+2?",
            ...     output="4",
            ...     status="success",
            ...     latency_ms=150.0
            ... )
            >>> result.status
            'success'
            >>> result.latency_ms
            150.0

        Creating an error result:

            >>> result = SchemaProbeResult(
            ...     input="complex query",
            ...     status="error",
            ...     error="Connection timeout",
            ...     error_type="TimeoutError"
            ... )
            >>> result.error
            'Connection timeout'

        Using with metadata:

            >>> result = SchemaProbeResult(
            ...     input={"query": "test"},
            ...     output={"answer": "response"},
            ...     status="success",
            ...     metadata={"model_version": "v1.2", "temperature": 0.7}
            ... )
            >>> result.metadata["model_version"]
            'v1.2'

        Serializing to dict:

            >>> result = SchemaProbeResult(input="test", status="success")
            >>> data = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
            >>> data["status"]
            'success'
    """

    schema_version: str = Field(default=SCHEMA_VERSION)
    input: Any
    output: Optional[Any] = None
    status: Literal["success", "error", "timeout", "rate_limited", "skipped"]
    error: Optional[str] = None
    error_type: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


#: Backward compatibility alias for SchemaProbeResult.
#: Use SchemaProbeResult for new code; this alias maintains API compatibility.
ProbeResult = SchemaProbeResult


class RunnerOutput(_BaseSchema):
    """Batch output wrapper for probe execution results.

    This schema wraps a list of ProbeResult instances, providing a container
    for batch probe execution results. It is typically used when running a
    probe against multiple inputs in a single batch operation.

    Attributes:
        schema_version: The schema version string, defaults to "1.0.0".
        results: List of individual ProbeResult instances from the batch execution.

    Examples:
        Creating a batch result wrapper:

            >>> results = RunnerOutput(
            ...     results=[
            ...         ProbeResult(input="q1", output="a1", status="success"),
            ...         ProbeResult(input="q2", output="a2", status="success"),
            ...         ProbeResult(input="q3", status="error", error="Failed"),
            ...     ]
            ... )
            >>> len(results.results)
            3
            >>> results.results[0].status
            'success'

        Counting success/error rates:

            >>> results = RunnerOutput(
            ...     results=[
            ...         ProbeResult(input="q1", status="success", output="a1"),
            ...         ProbeResult(input="q2", status="error", error="err"),
            ...     ]
            ... )
            >>> success_count = sum(1 for r in results.results if r.status == "success")
            >>> success_count
            1

        Serializing batch results:

            >>> results = RunnerOutput(
            ...     results=[ProbeResult(input="test", status="success")]
            ... )
            >>> data = results.model_dump() if hasattr(results, 'model_dump') else results.dict()
            >>> len(data["results"])
            1

    See Also:
        SchemaProbeResult: The individual result schema contained in this wrapper.
    """

    schema_version: str = Field(default=SCHEMA_VERSION)
    results: list[ProbeResult]


class SchemaChatMessage(_BaseSchema):
    """A single chat message in a chat-style prompt.

    This schema represents one message in a multi-turn conversation, following
    the common chat API format used by OpenAI, Anthropic, and other LLM providers.
    Messages are typically assembled into a list to form a complete prompt.

    Attributes:
        role: The role of the message sender. Common values include:
            - "system": System-level instructions or context
            - "user": Input from the human user
            - "assistant": Response from the AI model
            - "tool": Tool/function call results (provider-specific)
        content: The message content. Can be a simple string or structured
            content (e.g., list of content blocks with text and images).

    Examples:
        Creating a basic conversation:

            >>> messages = [
            ...     SchemaChatMessage(role="system", content="You are helpful."),
            ...     SchemaChatMessage(role="user", content="Hello!"),
            ...     SchemaChatMessage(role="assistant", content="Hi! How can I help?"),
            ... ]
            >>> messages[1].role
            'user'
            >>> messages[1].content
            'Hello!'

        Structured content (multimodal):

            >>> msg = SchemaChatMessage(
            ...     role="user",
            ...     content=[
            ...         {"type": "text", "text": "What's in this image?"},
            ...         {"type": "image_url", "url": "https://example.com/img.png"}
            ...     ]
            ... )
            >>> len(msg.content)
            2

        Creating from dict:

            >>> data = {"role": "user", "content": "test question"}
            >>> msg = SchemaChatMessage(**data)
            >>> msg.role
            'user'

    See Also:
        ResultRecord: Uses ChatMessage for storing prompt messages.
    """

    role: str
    content: Any


#: Backward compatibility alias for SchemaChatMessage.
#: Use SchemaChatMessage for new code; this alias maintains API compatibility.
ChatMessage = SchemaChatMessage


class ModelSpec(_BaseSchema):
    """Serializable model identity and configuration.

    This schema captures the essential information needed to identify and
    reproduce a model configuration. It is used in ResultRecord and RunManifest
    to document which model was used for a given execution.

    Attributes:
        model_id: The unique identifier for the model (e.g., "gpt-4",
            "claude-3-opus-20240229", "llama-2-70b").
        provider: The model provider or platform (e.g., "openai", "anthropic",
            "huggingface", "local"). Optional for self-evident model IDs.
        params: Additional model parameters such as temperature, max_tokens,
            top_p, or provider-specific settings. Defaults to empty dict.

    Examples:
        Basic model specification:

            >>> spec = ModelSpec(model_id="gpt-4", provider="openai")
            >>> spec.model_id
            'gpt-4'
            >>> spec.provider
            'openai'

        Model with parameters:

            >>> spec = ModelSpec(
            ...     model_id="gpt-4-turbo",
            ...     provider="openai",
            ...     params={"temperature": 0.7, "max_tokens": 1000}
            ... )
            >>> spec.params["temperature"]
            0.7

        Local/custom model:

            >>> spec = ModelSpec(
            ...     model_id="llama-2-70b-chat",
            ...     provider="local",
            ...     params={"quantization": "4bit", "device": "cuda:0"}
            ... )
            >>> spec.params["quantization"]
            '4bit'

        Minimal specification:

            >>> spec = ModelSpec(model_id="my-custom-model")
            >>> spec.provider is None
            True
            >>> spec.params
            {}

    See Also:
        ResultRecord: Uses ModelSpec to identify the model for each record.
        RunManifest: Uses ModelSpec in run-level metadata.
    """

    model_id: str
    provider: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class ProbeSpec(_BaseSchema):
    """Serializable probe identity and version information.

    This schema captures the identity and configuration of a probe used in
    an evaluation run. It enables reproducibility by documenting exactly which
    probe (and version) was used to generate results.

    Attributes:
        probe_id: The unique identifier for the probe (e.g., "factual-accuracy",
            "toxicity-detection", "code-correctness").
        probe_version: Optional version string for the probe implementation.
            Useful for tracking changes to probe logic over time.
        params: Additional probe-specific parameters that affect behavior,
            such as scoring thresholds or evaluation criteria.

    Examples:
        Basic probe specification:

            >>> spec = ProbeSpec(probe_id="factual-accuracy")
            >>> spec.probe_id
            'factual-accuracy'

        Versioned probe:

            >>> spec = ProbeSpec(
            ...     probe_id="toxicity-detection",
            ...     probe_version="2.1.0"
            ... )
            >>> spec.probe_version
            '2.1.0'

        Probe with custom parameters:

            >>> spec = ProbeSpec(
            ...     probe_id="code-correctness",
            ...     probe_version="1.0.0",
            ...     params={
            ...         "language": "python",
            ...         "timeout_seconds": 30,
            ...         "test_framework": "pytest"
            ...     }
            ... )
            >>> spec.params["language"]
            'python'

        Creating from configuration dict:

            >>> config = {
            ...     "probe_id": "sentiment-analysis",
            ...     "params": {"model": "vader", "threshold": 0.5}
            ... }
            >>> spec = ProbeSpec(**config)
            >>> spec.params["threshold"]
            0.5

    See Also:
        ResultRecord: Uses ProbeSpec to identify the probe for each record.
        RunManifest: Uses ProbeSpec in run-level metadata.
    """

    probe_id: str
    probe_version: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class DatasetSpec(_BaseSchema):
    """Serializable dataset provenance and version information.

    This schema captures metadata about the dataset used in a probe execution,
    enabling reproducibility and data lineage tracking. All fields are optional
    to accommodate various dataset sources and versioning schemes.

    Attributes:
        dataset_id: Optional unique identifier for the dataset (e.g.,
            "mmlu-test", "hellaswag-validation", "custom-qa-v1").
        dataset_version: Optional version string for the dataset. Can follow
            semantic versioning or date-based versioning.
        dataset_hash: Optional content hash (e.g., SHA-256) for exact dataset
            verification and deduplication.
        provenance: Optional description of dataset origin, such as a URL,
            paper citation, or generation method.
        params: Additional dataset-specific metadata such as split information,
            filtering criteria, or preprocessing steps.

    Examples:
        Basic dataset specification:

            >>> spec = DatasetSpec(
            ...     dataset_id="mmlu-test",
            ...     dataset_version="2024.01"
            ... )
            >>> spec.dataset_id
            'mmlu-test'

        Dataset with full provenance:

            >>> spec = DatasetSpec(
            ...     dataset_id="hellaswag",
            ...     dataset_version="1.0.0",
            ...     dataset_hash="sha256:abc123...",
            ...     provenance="https://huggingface.co/datasets/hellaswag"
            ... )
            >>> spec.provenance
            'https://huggingface.co/datasets/hellaswag'

        Custom dataset with parameters:

            >>> spec = DatasetSpec(
            ...     dataset_id="custom-qa",
            ...     params={
            ...         "split": "test",
            ...         "subset": "hard",
            ...         "sample_size": 1000,
            ...         "seed": 42
            ...     }
            ... )
            >>> spec.params["split"]
            'test'

        Empty/default specification:

            >>> spec = DatasetSpec()
            >>> spec.dataset_id is None
            True
            >>> spec.params
            {}

    See Also:
        ResultRecord: Uses DatasetSpec for dataset context per record.
        RunManifest: Uses DatasetSpec in run-level metadata.
    """

    dataset_id: Optional[str] = None
    dataset_version: Optional[str] = None
    dataset_hash: Optional[str] = None
    provenance: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class ResultRecord(_BaseSchema):
    """Comprehensive JSONL record emitted by ProbeRunner for wind-tunnel logging.

    This is the most detailed schema for recording probe execution results. Each
    record captures the complete context of a single probe execution including
    model/probe specifications, input/output data, timing, scoring, and custom
    metadata. Records are typically written to JSONL files for analysis.

    Attributes:
        schema_version: Schema version string, defaults to "1.0.0".
        run_id: Unique identifier for the execution run (e.g., "run-2024-01-15-abc123").
        started_at: Timestamp when this specific execution started.
        completed_at: Timestamp when execution completed (success or failure).
        model: ModelSpec identifying the model used.
        probe: ProbeSpec identifying the probe used.
        example_id: Unique identifier for this example within the dataset.
        dataset: DatasetSpec with dataset provenance information.
        input: The raw input data provided to the probe.
        messages: Optional list of ChatMessage objects if using chat format.
        messages_hash: Optional hash of messages for deduplication/caching.
        messages_storage: Optional reference to external message storage.
        output: The raw output from probe execution.
        output_text: Text representation of output for display/search.
        scores: Dictionary of computed scores/metrics for this execution.
        primary_metric: Name of the primary metric in scores dict.
        usage: Token usage and other API consumption metrics.
        latency_ms: Execution latency in milliseconds.
        status: Execution status ("success", "error", "timeout", "rate_limited", "skipped").
        error: Error message if status indicates failure.
        error_type: Error classification for programmatic handling.
        custom: Extensible dictionary for probe-specific custom data.

    Examples:
        Creating a complete result record:

            >>> from datetime import datetime
            >>> record = ResultRecord(
            ...     run_id="run-2024-01-15-abc123",
            ...     started_at=datetime(2024, 1, 15, 10, 0, 0),
            ...     completed_at=datetime(2024, 1, 15, 10, 0, 1),
            ...     model=ModelSpec(model_id="gpt-4", provider="openai"),
            ...     probe=ProbeSpec(probe_id="factual-qa"),
            ...     example_id="ex-001",
            ...     input="What is the capital of France?",
            ...     output="Paris",
            ...     output_text="Paris",
            ...     status="success",
            ...     latency_ms=850.5,
            ...     scores={"accuracy": 1.0, "confidence": 0.95},
            ...     primary_metric="accuracy"
            ... )
            >>> record.status
            'success'
            >>> record.scores["accuracy"]
            1.0

        Record with chat messages:

            >>> record = ResultRecord(
            ...     run_id="run-chat-001",
            ...     started_at=datetime.now(),
            ...     completed_at=datetime.now(),
            ...     model=ModelSpec(model_id="gpt-4"),
            ...     probe=ProbeSpec(probe_id="conversation-test"),
            ...     example_id="conv-001",
            ...     messages=[
            ...         ChatMessage(role="system", content="Be helpful."),
            ...         ChatMessage(role="user", content="Hello"),
            ...     ],
            ...     output="Hi! How can I help?",
            ...     status="success"
            ... )
            >>> len(record.messages)
            2

        Record with error:

            >>> record = ResultRecord(
            ...     run_id="run-err-001",
            ...     started_at=datetime.now(),
            ...     completed_at=datetime.now(),
            ...     model=ModelSpec(model_id="gpt-4"),
            ...     probe=ProbeSpec(probe_id="test-probe"),
            ...     example_id="ex-fail",
            ...     input="complex query",
            ...     status="error",
            ...     error="Rate limit exceeded",
            ...     error_type="RateLimitError",
            ...     latency_ms=100.0
            ... )
            >>> record.error_type
            'RateLimitError'

        Record with usage metrics:

            >>> record = ResultRecord(
            ...     run_id="run-usage-001",
            ...     started_at=datetime.now(),
            ...     completed_at=datetime.now(),
            ...     model=ModelSpec(model_id="gpt-4"),
            ...     probe=ProbeSpec(probe_id="test"),
            ...     example_id="ex-001",
            ...     status="success",
            ...     usage={
            ...         "prompt_tokens": 50,
            ...         "completion_tokens": 100,
            ...         "total_tokens": 150
            ...     }
            ... )
            >>> record.usage["total_tokens"]
            150

        Record with custom data:

            >>> record = ResultRecord(
            ...     run_id="run-custom-001",
            ...     started_at=datetime.now(),
            ...     completed_at=datetime.now(),
            ...     model=ModelSpec(model_id="custom-model"),
            ...     probe=ProbeSpec(probe_id="trace-probe"),
            ...     example_id="ex-001",
            ...     status="success",
            ...     custom={
            ...         "trace": {"steps": ["think", "act", "respond"]},
            ...         "annotations": ["verified", "high-quality"]
            ...     }
            ... )
            >>> record.custom["trace"]["steps"]
            ['think', 'act', 'respond']

    See Also:
        SchemaProbeResult: Simpler result schema without full context.
        RunManifest: Run-level manifest that accompanies result records.
        insideLLMs.schemas.custom_trace_v1: Schema for custom["trace"] data.
    """

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
    """Run directory manifest emitted alongside records.jsonl.

    This schema defines the manifest.json file that accompanies each run's
    records.jsonl file. It provides run-level metadata including environment
    information, execution statistics, and references to related schemas.

    The manifest enables discovery and understanding of run outputs without
    needing to parse the full records file.

    Attributes:
        schema_version: Schema version string, defaults to "1.0.0".
        run_id: Unique identifier for this execution run.
        created_at: Timestamp when the run directory was created.
        started_at: Timestamp when probe execution began.
        completed_at: Timestamp when all executions finished.
        library_version: insideLLMs library version (e.g., "0.5.0").
        python_version: Python interpreter version (e.g., "3.11.5").
        platform: Operating system platform (e.g., "linux", "darwin").
        command: Command line arguments used to invoke the run.
        model: ModelSpec for the model used in this run.
        probe: ProbeSpec for the probe used in this run.
        dataset: DatasetSpec with dataset information.
        record_count: Total number of records in records.jsonl.
        success_count: Number of successful executions.
        error_count: Number of failed executions.
        records_file: Filename for the JSONL records (default: "records.jsonl").
        schemas: Dictionary mapping schema names to their JSON schemas.
        custom: Extensible dictionary for run-level custom metadata.

    Examples:
        Creating a run manifest:

            >>> from datetime import datetime
            >>> manifest = RunManifest(
            ...     run_id="run-2024-01-15-abc123",
            ...     created_at=datetime(2024, 1, 15, 10, 0, 0),
            ...     started_at=datetime(2024, 1, 15, 10, 0, 1),
            ...     completed_at=datetime(2024, 1, 15, 10, 5, 0),
            ...     model=ModelSpec(model_id="gpt-4", provider="openai"),
            ...     probe=ProbeSpec(probe_id="factual-qa"),
            ...     record_count=100,
            ...     success_count=95,
            ...     error_count=5
            ... )
            >>> manifest.run_id
            'run-2024-01-15-abc123'
            >>> manifest.success_count
            95

        Manifest with environment info:

            >>> manifest = RunManifest(
            ...     run_id="run-env-001",
            ...     created_at=datetime.now(),
            ...     started_at=datetime.now(),
            ...     completed_at=datetime.now(),
            ...     model=ModelSpec(model_id="gpt-4"),
            ...     probe=ProbeSpec(probe_id="test"),
            ...     library_version="0.5.0",
            ...     python_version="3.11.5",
            ...     platform="darwin",
            ...     command=["python", "-m", "insideLLMs", "run", "--probe", "test"]
            ... )
            >>> manifest.library_version
            '0.5.0'
            >>> manifest.platform
            'darwin'

        Manifest with custom metadata:

            >>> manifest = RunManifest(
            ...     run_id="run-custom-001",
            ...     created_at=datetime.now(),
            ...     started_at=datetime.now(),
            ...     completed_at=datetime.now(),
            ...     model=ModelSpec(model_id="gpt-4"),
            ...     probe=ProbeSpec(probe_id="test"),
            ...     custom={
            ...         "experiment_name": "accuracy-study-v2",
            ...         "researcher": "team-a",
            ...         "tags": ["production", "validated"]
            ...     }
            ... )
            >>> manifest.custom["experiment_name"]
            'accuracy-study-v2'

        Computing success rate:

            >>> manifest = RunManifest(
            ...     run_id="run-stats",
            ...     created_at=datetime.now(),
            ...     started_at=datetime.now(),
            ...     completed_at=datetime.now(),
            ...     model=ModelSpec(model_id="gpt-4"),
            ...     probe=ProbeSpec(probe_id="test"),
            ...     record_count=100,
            ...     success_count=90,
            ...     error_count=10
            ... )
            >>> success_rate = manifest.success_count / manifest.record_count
            >>> success_rate
            0.9

    See Also:
        ResultRecord: Individual records referenced by this manifest.
        insideLLMs.schemas.v1_0_1.RunManifest: Extended version with run_completed field.
    """

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
    """One JSONL record emitted by run_harness_from_config().

    This schema captures the output of the harness execution framework, which
    runs configured experiments across multiple models and probes. Each record
    represents one execution with full experiment context.

    HarnessRecord is similar to ResultRecord but uses a flattened structure
    more suited to the harness configuration format.

    Attributes:
        schema_version: Schema version string, defaults to "1.0.0".
        experiment_id: Unique identifier for the experiment configuration.
        model_type: Type classification of the model (e.g., "chat", "completion").
        model_name: Human-readable name for the model.
        model_id: Technical model identifier.
        model_provider: Model provider (e.g., "openai", "anthropic").
        probe_type: Type classification of the probe.
        probe_name: Human-readable name for the probe.
        probe_category: Category grouping for the probe (e.g., "safety", "accuracy").
        dataset: Dataset identifier or name.
        dataset_format: Format of the dataset (e.g., "jsonl", "csv").
        example_index: Zero-based index of this example in the dataset.
        input: The input data provided to the probe.
        output: The output from probe execution.
        status: Execution status string.
        error: Error message if execution failed.
        error_type: Error classification for programmatic handling.
        latency_ms: Execution latency in milliseconds.
        started_at: Timestamp when execution started.
        completed_at: Timestamp when execution completed.

    Examples:
        Creating a harness record:

            >>> from datetime import datetime
            >>> record = HarnessRecord(
            ...     experiment_id="exp-2024-01-safety",
            ...     model_name="GPT-4",
            ...     model_id="gpt-4",
            ...     model_provider="openai",
            ...     probe_name="Toxicity Detection",
            ...     probe_category="safety",
            ...     dataset="toxic-prompts-v1",
            ...     example_index=42,
            ...     input="Is this content appropriate?",
            ...     output="Yes, this content is appropriate.",
            ...     status="success",
            ...     latency_ms=320.5,
            ...     started_at=datetime.now(),
            ...     completed_at=datetime.now()
            ... )
            >>> record.probe_category
            'safety'
            >>> record.example_index
            42

        Record with model type:

            >>> record = HarnessRecord(
            ...     experiment_id="exp-chat-001",
            ...     model_type="chat",
            ...     model_name="Claude 3",
            ...     model_provider="anthropic",
            ...     probe_name="Helpfulness",
            ...     probe_category="quality",
            ...     dataset="helpfulness-eval",
            ...     example_index=0,
            ...     input="Help me understand this.",
            ...     status="success",
            ...     started_at=datetime.now(),
            ...     completed_at=datetime.now()
            ... )
            >>> record.model_type
            'chat'

        Record with error:

            >>> record = HarnessRecord(
            ...     experiment_id="exp-err-001",
            ...     model_name="Test Model",
            ...     probe_name="Test Probe",
            ...     probe_category="test",
            ...     dataset="test-data",
            ...     example_index=5,
            ...     input="test input",
            ...     status="error",
            ...     error="Connection refused",
            ...     error_type="ConnectionError",
            ...     started_at=datetime.now(),
            ...     completed_at=datetime.now()
            ... )
            >>> record.error_type
            'ConnectionError'

    See Also:
        ResultRecord: Alternative record format with nested specs.
        HarnessSummary: Aggregated summary for harness runs.
    """

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
    """Summary payload emitted by the harness CLI.

    This schema captures the aggregated summary of a harness run, including
    configuration used and computed summary statistics. It is written to
    summary.json in the output directory.

    Attributes:
        schema_version: Schema version string, defaults to "1.0.0".
        generated_at: Timestamp when the summary was generated.
        summary: Dictionary containing aggregated statistics and results.
            Typical keys include success_rate, total_records, by_model, by_probe.
        config: Dictionary containing the harness configuration used for the run.
            Includes model configs, probe configs, and execution settings.

    Examples:
        Creating a harness summary:

            >>> from datetime import datetime
            >>> summary = HarnessSummary(
            ...     generated_at=datetime.now(),
            ...     summary={
            ...         "total_records": 1000,
            ...         "success_count": 950,
            ...         "error_count": 50,
            ...         "success_rate": 0.95,
            ...         "by_model": {
            ...             "gpt-4": {"success_rate": 0.97},
            ...             "gpt-3.5": {"success_rate": 0.93}
            ...         }
            ...     },
            ...     config={
            ...         "models": ["gpt-4", "gpt-3.5"],
            ...         "probes": ["factual-qa", "toxicity"],
            ...         "max_concurrent": 10
            ...     }
            ... )
            >>> summary.summary["success_rate"]
            0.95

        Accessing configuration:

            >>> summary.config["max_concurrent"]
            10

        Summary with detailed breakdown:

            >>> summary = HarnessSummary(
            ...     generated_at=datetime.now(),
            ...     summary={
            ...         "by_probe": {
            ...             "factual-qa": {"accuracy": 0.88, "count": 500},
            ...             "toxicity": {"accuracy": 0.99, "count": 500}
            ...         },
            ...         "latency_stats": {
            ...             "mean_ms": 450.5,
            ...             "p95_ms": 1200.0,
            ...             "p99_ms": 2500.0
            ...         }
            ...     },
            ...     config={"experiment_id": "exp-001"}
            ... )
            >>> summary.summary["by_probe"]["factual-qa"]["accuracy"]
            0.88

    See Also:
        HarnessRecord: Individual records that contribute to this summary.
    """

    schema_version: str = Field(default=SCHEMA_VERSION)
    generated_at: datetime
    summary: dict[str, Any]
    config: dict[str, Any]


class HarnessExplain(_BaseSchema):
    """Explainability payload emitted by harness CLI when --explain is enabled."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    kind: Literal["HarnessExplain"] = "HarnessExplain"
    generated_at: datetime
    run_id: str
    config_resolution: dict[str, Any] = Field(default_factory=dict)
    effective_config: dict[str, Any] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)
    determinism: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)


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
    baseline_trace_fingerprint: Optional[str] = None
    candidate_trace_fingerprint: Optional[str] = None
    baseline_violations: Optional[int] = None
    candidate_violations: Optional[int] = None
    candidate_violation_details: Optional[list[dict[str, Any]]] = None
    baseline_trajectory_fingerprint: Optional[str] = None
    candidate_trajectory_fingerprint: Optional[str] = None
    baseline_trajectory: Optional[dict[str, Any]] = None
    candidate_trajectory: Optional[dict[str, Any]] = None


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
    trace_drifts: int = 0
    trace_violation_increases: int = 0
    trajectory_drifts: int = 0


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
    trace_drifts: list[DiffChangeEntry] = Field(default_factory=list)
    trace_violation_increases: list[DiffChangeEntry] = Field(default_factory=list)
    trajectory_drifts: list[DiffChangeEntry] = Field(default_factory=list)


class SchemaExportMetadata(_BaseSchema):
    """Metadata file included with export bundles.

    This schema defines the metadata.json file accompanying exported data
    bundles. It provides essential information about the export including
    timing, format, compression, and content statistics.

    Attributes:
        export_time: Timestamp when the export was created.
        version: Export format version (default: "1.0").
        schema_version: Schema version for exported records.
        source: Source system identifier (default: "insideLLMs").
        record_count: Number of records in the export.
        format: Export format (e.g., "jsonl", "parquet", "csv").
        compression: Compression algorithm (default: "none").
        custom: Custom metadata for extensions.

    Examples:
        Creating export metadata:

            >>> from datetime import datetime
            >>> metadata = SchemaExportMetadata(
            ...     export_time=datetime.now(),
            ...     record_count=1000,
            ...     format="jsonl",
            ...     compression="gzip"
            ... )
            >>> metadata.format
            'jsonl'

        With custom metadata:

            >>> metadata = SchemaExportMetadata(
            ...     export_time=datetime.now(),
            ...     custom={"exported_by": "pipeline-v2"}
            ... )
            >>> metadata.source
            'insideLLMs'

    See Also:
        ResultRecord: Typical record type in exports.
    """

    export_time: datetime
    version: str = "1.0"
    schema_version: str = Field(default=SCHEMA_VERSION)
    source: str = "insideLLMs"
    record_count: int = 0
    format: str = ""
    compression: str = "none"
    custom: dict[str, Any] = Field(default_factory=dict)


#: Backward compatibility alias for SchemaExportMetadata.
ExportMetadata = SchemaExportMetadata


_SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "ProbeResult": ProbeResult,
    "RunnerOutput": RunnerOutput,
    "ResultRecord": ResultRecord,
    "RunManifest": RunManifest,
    "HarnessRecord": HarnessRecord,
    "HarnessSummary": HarnessSummary,
    "HarnessExplain": HarnessExplain,
    "BenchmarkSummary": BenchmarkSummary,
    "ComparisonReport": ComparisonReport,
    "DiffReport": DiffReport,
    "ExportMetadata": ExportMetadata,
}


def get_schema_model(schema_name: str) -> type[BaseModel]:
    """Get a schema model class by name.

    Retrieves the Pydantic model class for the specified schema name from
    this module's schema registry. This function provides the primary
    mechanism for dynamic schema lookup within the v1.0.0 module.

    Args:
        schema_name: The canonical name of the schema. Valid values are:
            - "ProbeResult": Per-item probe execution result
            - "RunnerOutput": Batch result wrapper
            - "ResultRecord": Detailed JSONL record
            - "RunManifest": Run directory manifest
            - "HarnessRecord": Harness per-line record
            - "HarnessSummary": Harness summary payload
            - "HarnessExplain": Harness explainability payload
            - "BenchmarkSummary": Benchmark output
            - "ComparisonReport": Comparison output
            - "DiffReport": Diff analysis output
            - "ExportMetadata": Export bundle metadata

    Returns:
        The Pydantic model class for the specified schema name.

    Raises:
        KeyError: If the schema name is not recognized.

    Examples:
        Getting and using a schema model:

            >>> Model = get_schema_model("ProbeResult")
            >>> result = Model(input="test", status="success")
            >>> result.status
            'success'

        Listing available schemas:

            >>> available = list(_SCHEMA_MAP.keys())
            >>> "ProbeResult" in available
            True
            >>> "ResultRecord" in available
            True

        Handling unknown schemas:

            >>> try:
            ...     get_schema_model("InvalidSchema")
            ... except KeyError:
            ...     print("Schema not found")
            Schema not found

    See Also:
        SchemaRegistry.get_model: Version-aware lookup via registry.
        _SCHEMA_MAP: The underlying schema name to class mapping.
    """
    if schema_name not in _SCHEMA_MAP:
        raise KeyError(f"Unknown schema name: {schema_name}")
    return _SCHEMA_MAP[schema_name]
