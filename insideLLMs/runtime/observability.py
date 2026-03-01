"""OpenTelemetry Observability for LLM Operations.

This module provides comprehensive observability capabilities for monitoring,
tracing, and debugging LLM (Large Language Model) operations. It offers both
a lightweight built-in telemetry system and full OpenTelemetry integration
for distributed tracing in production environments.

Overview
--------
The observability module is designed around three core concepts:

1. **Telemetry Collection**: Lightweight, dependency-free collection of call
   records including latency, token usage, success/failure rates, and custom
   metadata.

2. **Model Instrumentation**: Transparent wrappers that automatically trace
   all model operations (generate, chat, stream) without modifying existing
   code.

3. **OpenTelemetry Integration**: Optional deep integration with OpenTelemetry
   for distributed tracing, allowing correlation of LLM calls with other
   services in a microservices architecture.

Key Features
------------
- **Automatic tracing** of model calls with minimal code changes
- **Metrics collection** for latency, token usage, and success rates
- **Structured logging** with contextual information
- **Cost tracking** via token counting and attribution
- **Flexible backends** supporting Jaeger, OTLP, console, or custom exporters
- **Zero-dependency mode** for simple use cases without OpenTelemetry
- **Streaming support** with accurate timing and token counting

Architecture
------------
The module provides two tiers of functionality:

**Tier 1 - Built-in Telemetry (No Dependencies)**:
    - `TelemetryCollector`: Stores call records in memory
    - `CallRecord`: Immutable record of each model invocation
    - `trace_call`: Context manager for manual tracing
    - `TracedModel`: Wrapper adding automatic tracing to any model

**Tier 2 - OpenTelemetry Integration (Requires opentelemetry-sdk)**:
    - `setup_otel_tracing`: Configures OpenTelemetry providers
    - `OTelTracedModel`: Wrapper with distributed tracing spans

Examples
--------
Basic instrumentation with built-in telemetry:

    >>> from insideLLMs.runtime.observability import instrument_model, get_collector
    >>> from insideLLMs.models import OpenAIModel
    >>>
    >>> # Wrap any model with tracing
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> traced_model = instrument_model(model)
    >>>
    >>> # Use normally - calls are automatically recorded
    >>> response = traced_model.generate("Explain quantum computing")
    >>>
    >>> # Access telemetry data
    >>> collector = get_collector()
    >>> stats = collector.get_stats()
    >>> print(f"Total calls: {stats['total_calls']}")
    >>> print(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")

Production setup with OpenTelemetry and Jaeger:

    >>> from insideLLMs.runtime.observability import (
    ...     TracingConfig,
    ...     setup_otel_tracing,
    ...     OTelTracedModel,
    ... )
    >>> from insideLLMs.models import AnthropicModel
    >>>
    >>> # Configure distributed tracing
    >>> config = TracingConfig(
    ...     service_name="my-llm-service",
    ...     jaeger_endpoint="http://localhost:14268/api/traces",
    ...     log_prompts=False,  # Don't log potentially sensitive prompts
    ...     sample_rate=0.1,    # Sample 10% of traces in production
    ... )
    >>>
    >>> # Initialize OpenTelemetry
    >>> setup_otel_tracing(config)
    >>>
    >>> # Wrap model with OTel tracing
    >>> model = AnthropicModel(model_name="claude-3-opus")
    >>> traced_model = OTelTracedModel(model, config)
    >>>
    >>> # All calls create distributed trace spans
    >>> response = traced_model.generate("Write a haiku about Python")

Manual tracing with context manager:

    >>> from insideLLMs.runtime.observability import trace_call, get_collector
    >>>
    >>> # Trace custom operations
    >>> with trace_call("custom-processor", "embedding", "input text") as ctx:
    ...     result = compute_embedding("input text")
    ...     ctx["response"] = str(result)
    >>>
    >>> # Check the recorded call
    >>> records = get_collector().get_records(limit=1)
    >>> print(f"Operation took {records[0].latency_ms:.2f}ms")

Using callbacks for real-time monitoring:

    >>> from insideLLMs.runtime.observability import get_collector, CallRecord
    >>>
    >>> def alert_on_slow_calls(record: CallRecord) -> None:
    ...     if record.latency_ms > 5000:
    ...         print(f"SLOW CALL: {record.model_name} took {record.latency_ms}ms")
    >>>
    >>> collector = get_collector()
    >>> collector.add_callback(alert_on_slow_calls)

Exporting telemetry data:

    >>> from insideLLMs.runtime.observability import get_collector
    >>> import json
    >>>
    >>> collector = get_collector()
    >>>
    >>> # Export as JSON for analysis
    >>> json_data = collector.export_json()
    >>> with open("telemetry_export.json", "w") as f:
    ...     f.write(json_data)
    >>>
    >>> # Get filtered statistics
    >>> gpt4_stats = collector.get_stats(model_name="gpt-4")
    >>> stream_stats = collector.get_stats(operation="stream")

Notes
-----
- Token estimation uses a heuristic (approximately 4 characters per token).
  For precise counts, use the model's native tokenizer.

- When `log_prompts` or `log_responses` is enabled, be aware that sensitive
  data may be captured in traces. Consider data retention policies.

- The global collector has a default limit of 10,000 records. Older records
  are automatically pruned when this limit is exceeded.

- OpenTelemetry integration requires additional packages:
  ``pip install opentelemetry-api opentelemetry-sdk``

- For Jaeger export: ``pip install opentelemetry-exporter-jaeger``

- For OTLP export: ``pip install opentelemetry-exporter-otlp``

See Also
--------
insideLLMs.tokens : Token estimation utilities
insideLLMs.models.base : Base model interface
opentelemetry.trace : OpenTelemetry tracing API

References
----------
.. [1] OpenTelemetry Python Documentation
   https://opentelemetry.io/docs/instrumentation/python/

.. [2] Jaeger Tracing
   https://www.jaegertracing.io/
"""

import functools
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    TypeVar,
)

from insideLLMs.tokens import estimate_tokens as _canonical_estimate_tokens

if TYPE_CHECKING:
    from insideLLMs.models.base import Model
    from insideLLMs.types import ModelInfo

# Try to import OpenTelemetry
try:
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.semconv.resource import ResourceAttributes

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    metrics = None

# Setup basic logging
logger = logging.getLogger("insideLLMs.observability")

T = TypeVar("T")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TracingConfig:
    """Configuration for tracing and observability settings.

    This dataclass encapsulates all configuration options for the observability
    system, including service identification, export destinations, privacy
    settings, and sampling configuration.

    The configuration supports multiple export backends simultaneously, allowing
    you to send traces to both console (for debugging) and a production backend
    (like Jaeger or an OTLP collector) at the same time.

    Parameters
    ----------
    service_name : str, default="insideLLMs"
        Identifier for your service in distributed traces. This name appears
        in trace visualization tools and helps distinguish your LLM service
        from other services in the system.

    enabled : bool, default=True
        Master switch for tracing. When False, tracing code paths are skipped
        entirely, providing zero overhead. Useful for disabling tracing in
        specific environments without code changes.

    console_export : bool, default=False
        When True, spans are printed to stdout in a human-readable format.
        Primarily useful for local development and debugging. Not recommended
        for production due to verbosity.

    jaeger_endpoint : str or None, default=None
        Full URL to a Jaeger collector's HTTP endpoint. Typically in the form
        ``http://host:14268/api/traces``. When set, traces are batched and
        sent to Jaeger for storage and visualization.

    otlp_endpoint : str or None, default=None
        gRPC endpoint for an OpenTelemetry Protocol (OTLP) collector. Typically
        in the form ``http://host:4317``. OTLP is the vendor-neutral standard
        and works with many backends including Grafana Tempo, Honeycomb, etc.

    log_prompts : bool, default=False
        When True, the full prompt text is included in trace spans (truncated
        to 1000 characters for OTel spans). **Security Warning**: Prompts may
        contain sensitive user data. Only enable in environments where trace
        data is appropriately secured.

    log_responses : bool, default=False
        When True, the full response text is included in trace spans (truncated
        to 1000 characters for OTel spans). **Security Warning**: Responses may
        contain sensitive or proprietary information.

    sample_rate : float, default=1.0
        Fraction of traces to record, from 0.0 (none) to 1.0 (all). In high-
        throughput production systems, sampling reduces storage costs and
        overhead while still providing statistical insight. Note: This affects
        built-in telemetry; OpenTelemetry has its own sampling configuration.

    custom_attributes : dict[str, str], default={}
        Additional key-value pairs to attach to all spans. Useful for adding
        environment-specific metadata like deployment version, region, or
        customer ID for multi-tenant systems.

    Attributes
    ----------
    service_name : str
        The configured service name for trace identification.

    enabled : bool
        Whether tracing is currently enabled.

    console_export : bool
        Whether console export is enabled.

    jaeger_endpoint : str or None
        The configured Jaeger endpoint, if any.

    otlp_endpoint : str or None
        The configured OTLP endpoint, if any.

    log_prompts : bool
        Whether prompt logging is enabled.

    log_responses : bool
        Whether response logging is enabled.

    sample_rate : float
        The configured sampling rate.

    custom_attributes : dict[str, str]
        Custom attributes to add to all spans.

    Examples
    --------
    Minimal development configuration with console output:

        >>> config = TracingConfig(
        ...     service_name="my-dev-app",
        ...     console_export=True,
        ...     log_prompts=True,
        ...     log_responses=True,
        ... )

    Production configuration with Jaeger and privacy protection:

        >>> config = TracingConfig(
        ...     service_name="production-llm-service",
        ...     jaeger_endpoint="http://jaeger.internal:14268/api/traces",
        ...     log_prompts=False,  # Protect user privacy
        ...     log_responses=False,
        ...     sample_rate=0.1,  # Sample 10% to reduce costs
        ...     custom_attributes={
        ...         "deployment.environment": "production",
        ...         "deployment.version": "v2.3.1",
        ...         "cloud.region": "us-east-1",
        ...     },
        ... )

    Multi-backend configuration for observability platform migration:

        >>> config = TracingConfig(
        ...     service_name="migration-test",
        ...     jaeger_endpoint="http://old-jaeger:14268/api/traces",
        ...     otlp_endpoint="http://new-tempo:4317",
        ...     console_export=False,
        ... )

    Disabled tracing for performance-critical paths:

        >>> config = TracingConfig(enabled=False)

    See Also
    --------
    setup_otel_tracing : Function that uses this config to initialize OpenTelemetry
    OTelTracedModel : Model wrapper that uses this config for span attributes

    Notes
    -----
    - The ``sample_rate`` parameter is a hint and may not be honored by all
      tracing backends. For precise sampling control, configure sampling at
      the OpenTelemetry SDK level.

    - Custom attributes are converted to strings. For complex values, serialize
      them appropriately before adding to the dict.

    - When both ``jaeger_endpoint`` and ``otlp_endpoint`` are set, traces are
      sent to both backends. Be aware of potential cost implications.
    """

    service_name: str = "insideLLMs"
    enabled: bool = True
    console_export: bool = False
    jaeger_endpoint: Optional[str] = None
    otlp_endpoint: Optional[str] = None
    log_prompts: bool = False
    log_responses: bool = False
    sample_rate: float = 1.0
    custom_attributes: dict[str, str] = field(default_factory=dict)


# =============================================================================
# Simple Telemetry (No OpenTelemetry dependency)
# =============================================================================


@dataclass
class CallRecord:
    """Immutable record of a single model call for telemetry purposes.

    CallRecord captures all relevant metrics and metadata from a model
    invocation, including timing, token usage, success/failure status,
    and custom metadata. These records are stored by TelemetryCollector
    and can be used for monitoring, debugging, and cost analysis.

    This is a frozen-style dataclass (though not technically frozen for
    performance) that should be treated as immutable after creation.

    Parameters
    ----------
    model_name : str
        Identifier for the model that was called (e.g., "gpt-4", "claude-3-opus",
        "llama-70b"). This is typically obtained from the model's ``name``
        property.

    operation : str
        Type of operation performed. Standard values are:
        - "generate": Single-turn text generation
        - "chat": Multi-turn conversation
        - "stream": Streaming text generation
        - "embedding": Vector embedding (if supported)
        Custom operation names are also supported for specialized use cases.

    start_time : datetime
        UTC timestamp when the call began (before any network or processing).

    end_time : datetime
        UTC timestamp when the call completed (after response received or
        error occurred).

    latency_ms : float
        Total call duration in milliseconds, calculated as
        ``(end_time - start_time).total_seconds() * 1000``.

    success : bool
        True if the call completed without raising an exception, False otherwise.
        Note that a "successful" call may still return an empty or unexpected
        response.

    error : str or None, default=None
        Error message if the call failed (``success=False``). Contains the
        string representation of the exception. None for successful calls.

    prompt_tokens : int, default=0
        Estimated number of tokens in the input prompt. Uses heuristic
        estimation unless the model provides exact counts.

    completion_tokens : int, default=0
        Estimated number of tokens in the generated response. Zero for
        failed calls or calls that haven't completed.

    prompt_length : int, default=0
        Character count of the input prompt. Useful for cost estimation
        when token counts are unavailable.

    response_length : int, default=0
        Character count of the generated response. Zero for failed calls.

    metadata : dict[str, Any], default={}
        Arbitrary key-value pairs for custom tracking. Common uses include:
        - Request IDs for correlation
        - User/session identifiers
        - Model parameters (temperature, max_tokens)
        - Application-specific tags

    Attributes
    ----------
    model_name : str
        The model identifier.

    operation : str
        The operation type.

    start_time : datetime
        Call start timestamp.

    end_time : datetime
        Call end timestamp.

    latency_ms : float
        Duration in milliseconds.

    success : bool
        Whether the call succeeded.

    error : str or None
        Error message if failed.

    prompt_tokens : int
        Estimated input tokens.

    completion_tokens : int
        Estimated output tokens.

    prompt_length : int
        Input character count.

    response_length : int
        Output character count.

    metadata : dict[str, Any]
        Custom metadata.

    Examples
    --------
    Creating a record for a successful generation call:

        >>> from datetime import datetime
        >>> record = CallRecord(
        ...     model_name="gpt-4",
        ...     operation="generate",
        ...     start_time=datetime(2024, 1, 15, 10, 30, 0),
        ...     end_time=datetime(2024, 1, 15, 10, 30, 2),
        ...     latency_ms=2150.5,
        ...     success=True,
        ...     prompt_tokens=50,
        ...     completion_tokens=200,
        ...     prompt_length=180,
        ...     response_length=750,
        ...     metadata={"request_id": "req_abc123", "user_id": "user_456"},
        ... )

    Creating a record for a failed call:

        >>> record = CallRecord(
        ...     model_name="claude-3-opus",
        ...     operation="chat",
        ...     start_time=datetime(2024, 1, 15, 10, 31, 0),
        ...     end_time=datetime(2024, 1, 15, 10, 31, 30),
        ...     latency_ms=30000.0,
        ...     success=False,
        ...     error="TimeoutError: Request timed out after 30 seconds",
        ...     prompt_tokens=100,
        ...     completion_tokens=0,
        ...     prompt_length=400,
        ...     response_length=0,
        ... )

    Converting to dictionary for JSON serialization:

        >>> record_dict = record.to_dict()
        >>> print(record_dict["total_tokens"])
        100
        >>> import json
        >>> json_str = json.dumps(record_dict)

    Accessing computed properties:

        >>> record = CallRecord(
        ...     model_name="gpt-4",
        ...     operation="generate",
        ...     start_time=datetime.now(),
        ...     end_time=datetime.now(),
        ...     latency_ms=1500.0,
        ...     success=True,
        ...     prompt_tokens=100,
        ...     completion_tokens=400,
        ... )
        >>> total = record.prompt_tokens + record.completion_tokens
        >>> print(f"Total tokens: {total}")
        Total tokens: 500

    See Also
    --------
    TelemetryCollector : Stores and aggregates CallRecord instances
    trace_call : Context manager that creates CallRecords automatically

    Notes
    -----
    - Token counts are estimates unless the model provides exact values.
      The estimation uses approximately 4 characters per token.

    - The ``to_dict()`` method adds a computed ``total_tokens`` field that
      is not stored as an attribute.

    - For streaming operations, ``latency_ms`` represents the total time
      from start to final chunk, not time-to-first-token.
    """

    model_name: str
    operation: str
    start_time: datetime
    end_time: datetime
    latency_ms: float
    success: bool
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_length: int = 0
    response_length: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the CallRecord to a JSON-serializable dictionary.

        Creates a dictionary representation suitable for JSON serialization,
        logging, or export to external systems. Datetime objects are converted
        to ISO 8601 format strings, and a computed ``total_tokens`` field is
        added.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all record fields plus computed fields:
            - All instance attributes with their current values
            - ``start_time`` and ``end_time`` as ISO 8601 strings
            - ``total_tokens``: Sum of prompt_tokens and completion_tokens

        Examples
        --------
        Basic conversion:

            >>> from datetime import datetime
            >>> record = CallRecord(
            ...     model_name="gpt-4",
            ...     operation="generate",
            ...     start_time=datetime(2024, 1, 15, 10, 30, 0),
            ...     end_time=datetime(2024, 1, 15, 10, 30, 2),
            ...     latency_ms=2000.0,
            ...     success=True,
            ...     prompt_tokens=50,
            ...     completion_tokens=150,
            ... )
            >>> d = record.to_dict()
            >>> d["model_name"]
            'gpt-4'
            >>> d["total_tokens"]
            200
            >>> d["start_time"]
            '2024-01-15T10:30:00'

        Serializing to JSON:

            >>> import json
            >>> json_str = json.dumps(record.to_dict(), indent=2)
            >>> print(json_str[:50])
            {
              "model_name": "gpt-4",
              "operation":

        Using for logging:

            >>> import logging
            >>> logger = logging.getLogger(__name__)
            >>> logger.info("Model call completed", extra=record.to_dict())

        See Also
        --------
        TelemetryCollector.export_json : Exports all records as JSON
        """
        return {
            "model_name": self.model_name,
            "operation": self.operation,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error": self.error,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "prompt_length": self.prompt_length,
            "response_length": self.response_length,
            "metadata": self.metadata,
        }


class TelemetryCollector:
    """In-memory telemetry collector for model call records.

    TelemetryCollector provides a lightweight, dependency-free mechanism for
    collecting and analyzing model call telemetry. It stores CallRecord
    instances in memory and provides methods for computing aggregate statistics,
    filtering records, and exporting data.

    The collector is designed for both development debugging and production
    monitoring scenarios. For high-volume production use, consider using
    callbacks to stream records to external systems rather than relying
    solely on in-memory storage.

    Parameters
    ----------
    max_records : int, default=10000
        Maximum number of records to retain in memory. When this limit is
        exceeded, the oldest records are automatically pruned (FIFO eviction).
        Set higher for longer retention or lower to reduce memory usage.

    Attributes
    ----------
    records : list[CallRecord]
        The list of stored call records. Directly accessible for custom
        analysis, but prefer using the provided methods for filtering.

    max_records : int
        The configured maximum record limit.

    Examples
    --------
    Basic usage with manual recording:

        >>> from datetime import datetime
        >>> collector = TelemetryCollector(max_records=1000)
        >>>
        >>> # Create and record a call
        >>> record = CallRecord(
        ...     model_name="gpt-4",
        ...     operation="generate",
        ...     start_time=datetime.now(),
        ...     end_time=datetime.now(),
        ...     latency_ms=1500.0,
        ...     success=True,
        ...     prompt_tokens=100,
        ...     completion_tokens=200,
        ... )
        >>> collector.record(record)
        >>>
        >>> # Get statistics
        >>> stats = collector.get_stats()
        >>> print(f"Total calls: {stats['total_calls']}")
        Total calls: 1

    Using the global collector with instrumented models:

        >>> from insideLLMs.runtime.observability import (
        ...     get_collector,
        ...     instrument_model,
        ... )
        >>> from insideLLMs.models import OpenAIModel
        >>>
        >>> # Instrument a model (uses global collector by default)
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> traced = instrument_model(model)
        >>>
        >>> # Make some calls...
        >>> response = traced.generate("Hello!")
        >>>
        >>> # Access telemetry
        >>> collector = get_collector()
        >>> print(f"Recorded {len(collector.records)} calls")

    Setting up real-time monitoring with callbacks:

        >>> def log_to_external_system(record: CallRecord) -> None:
        ...     # Send to your monitoring system
        ...     print(f"[TELEMETRY] {record.model_name}: {record.latency_ms}ms")
        >>>
        >>> collector = TelemetryCollector()
        >>> collector.add_callback(log_to_external_system)

    Filtering and analyzing specific models:

        >>> # Get stats for a specific model
        >>> gpt4_stats = collector.get_stats(model_name="gpt-4")
        >>>
        >>> # Get stats for streaming operations only
        >>> stream_stats = collector.get_stats(operation="stream")
        >>>
        >>> # Get stats from the last hour
        >>> from datetime import datetime, timedelta
        >>> one_hour_ago = datetime.now() - timedelta(hours=1)
        >>> recent_stats = collector.get_stats(since=one_hour_ago)

    Exporting data for external analysis:

        >>> json_data = collector.export_json()
        >>> with open("telemetry.json", "w") as f:
        ...     f.write(json_data)

    See Also
    --------
    CallRecord : The record type stored by this collector
    get_collector : Returns the global collector instance
    set_collector : Sets a custom global collector
    trace_call : Context manager that records to a collector

    Notes
    -----
    - The collector is not thread-safe. For concurrent access, wrap calls
      in appropriate synchronization primitives or use separate collectors
      per thread.

    - Callbacks are invoked synchronously after each record. Long-running
      callbacks will impact tracing performance. Consider using async
      patterns for expensive operations.

    - Memory usage is approximately 200-500 bytes per record, depending on
      metadata size. With default max_records=10000, expect ~5MB memory usage.
    """

    def __init__(self, max_records: int = 10000):
        """Initialize the telemetry collector.

        Creates a new collector instance with an empty record list and
        configures the maximum retention limit.

        Parameters
        ----------
        max_records : int, default=10000
            Maximum number of CallRecord instances to retain in memory.
            When this limit is exceeded, the oldest records are removed
            to make room for new ones (FIFO eviction policy).

        Examples
        --------
        Create a collector with default settings:

            >>> collector = TelemetryCollector()
            >>> collector.max_records
            10000

        Create a collector with custom retention:

            >>> # High-retention collector for detailed analysis
            >>> detailed_collector = TelemetryCollector(max_records=100000)
            >>>
            >>> # Low-memory collector for constrained environments
            >>> light_collector = TelemetryCollector(max_records=100)

        See Also
        --------
        set_collector : Set this collector as the global instance
        """
        self.records: list[CallRecord] = []
        self.max_records = max_records
        self._callbacks: list[Callable[[CallRecord], None]] = []

    def record(self, call: CallRecord) -> None:
        """Store a call record and notify registered callbacks.

        Appends the record to the internal list, enforces the max_records
        limit by pruning old entries if necessary, and invokes all registered
        callbacks with the new record.

        Parameters
        ----------
        call : CallRecord
            The call record to store. Should be a complete record with all
            timing and metric fields populated.

        Examples
        --------
        Recording a successful call:

            >>> from datetime import datetime
            >>> collector = TelemetryCollector()
            >>> record = CallRecord(
            ...     model_name="claude-3-opus",
            ...     operation="generate",
            ...     start_time=datetime.now(),
            ...     end_time=datetime.now(),
            ...     latency_ms=2500.0,
            ...     success=True,
            ...     prompt_tokens=150,
            ...     completion_tokens=300,
            ... )
            >>> collector.record(record)
            >>> len(collector.records)
            1

        Recording a failed call:

            >>> error_record = CallRecord(
            ...     model_name="gpt-4",
            ...     operation="chat",
            ...     start_time=datetime.now(),
            ...     end_time=datetime.now(),
            ...     latency_ms=5000.0,
            ...     success=False,
            ...     error="RateLimitError: Too many requests",
            ... )
            >>> collector.record(error_record)
            >>> len(collector.records)
            2

        Automatic pruning when limit exceeded:

            >>> small_collector = TelemetryCollector(max_records=2)
            >>> for i in range(5):
            ...     rec = CallRecord(
            ...         model_name=f"model-{i}",
            ...         operation="generate",
            ...         start_time=datetime.now(),
            ...         end_time=datetime.now(),
            ...         latency_ms=100.0,
            ...         success=True,
            ...     )
            ...     small_collector.record(rec)
            >>> len(small_collector.records)
            2
            >>> small_collector.records[0].model_name
            'model-3'

        Notes
        -----
        - Callback exceptions are caught and logged, not propagated. This
          ensures that a failing callback doesn't prevent record storage.

        - Pruning removes records from the beginning of the list (oldest first)
          to maintain the most recent max_records entries.

        See Also
        --------
        add_callback : Register a callback for record events
        """
        self.records.append(call)

        # Trim if over limit
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(call)
            except Exception as e:
                logger.warning(f"Telemetry callback error: {e}")

    def add_callback(self, callback: Callable[[CallRecord], None]) -> None:
        """Register a callback to be invoked for each new record.

        Callbacks are invoked synchronously immediately after a record is
        stored. Multiple callbacks can be registered and are called in
        registration order.

        Parameters
        ----------
        callback : Callable[[CallRecord], None]
            Function that receives a CallRecord as its only argument. The
            function's return value is ignored. Exceptions are caught and
            logged but do not prevent other callbacks or record storage.

        Examples
        --------
        Simple logging callback:

            >>> def log_callback(record: CallRecord) -> None:
            ...     print(f"Call to {record.model_name}: {record.latency_ms}ms")
            >>>
            >>> collector = TelemetryCollector()
            >>> collector.add_callback(log_callback)

        Alert on slow calls:

            >>> def slow_call_alert(record: CallRecord) -> None:
            ...     if record.latency_ms > 5000:
            ...         send_alert(f"Slow call: {record.model_name}")
            >>>
            >>> collector.add_callback(slow_call_alert)

        Track error rates:

            >>> error_count = {"total": 0}
            >>> def count_errors(record: CallRecord) -> None:
            ...     if not record.success:
            ...         error_count["total"] += 1
            >>>
            >>> collector.add_callback(count_errors)

        Send to external monitoring system:

            >>> def send_to_datadog(record: CallRecord) -> None:
            ...     statsd.timing("llm.latency", record.latency_ms)
            ...     statsd.increment("llm.calls", tags=[f"model:{record.model_name}"])
            >>>
            >>> collector.add_callback(send_to_datadog)

        Notes
        -----
        - Callbacks are called synchronously. For expensive operations,
          consider using a queue and background worker pattern.

        - There is no mechanism to remove callbacks once added. Create a
          new collector if you need to change the callback configuration.

        - Callback exceptions are logged at WARNING level and do not
          propagate to the caller.

        See Also
        --------
        record : Method that invokes callbacks
        """
        self._callbacks.append(callback)

    def get_stats(
        self,
        model_name: Optional[str] = None,
        operation: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Compute aggregate statistics over stored records.

        Calculates key metrics including call counts, success rates, latency
        distributions, and token usage. Results can be filtered by model,
        operation type, or time range.

        Parameters
        ----------
        model_name : str or None, default=None
            If provided, only include records matching this model name.
            Uses exact string matching.

        operation : str or None, default=None
            If provided, only include records matching this operation type.
            Common values: "generate", "chat", "stream".

        since : datetime or None, default=None
            If provided, only include records with start_time >= this value.
            Useful for computing rolling statistics.

        Returns
        -------
        dict[str, Any]
            Dictionary containing aggregate statistics:

            - ``total_calls`` (int): Total number of matching records
            - ``success_rate`` (float): Fraction of successful calls (0.0-1.0)
            - ``successes`` (int): Count of successful calls
            - ``failures`` (int): Count of failed calls
            - ``avg_latency_ms`` (float): Mean latency in milliseconds
            - ``min_latency_ms`` (float): Minimum observed latency
            - ``max_latency_ms`` (float): Maximum observed latency
            - ``p50_latency_ms`` (float): Median latency (50th percentile)
            - ``p95_latency_ms`` (float): 95th percentile latency
            - ``total_tokens`` (int): Sum of all prompt + completion tokens
            - ``avg_tokens_per_call`` (float): Mean tokens per call

            If no records match the filters, returns a minimal dict with
            ``total_calls=0`` and zero values.

        Examples
        --------
        Get overall statistics:

            >>> collector = get_collector()
            >>> stats = collector.get_stats()
            >>> print(f"Success rate: {stats['success_rate']:.1%}")
            Success rate: 98.5%
            >>> print(f"Avg latency: {stats['avg_latency_ms']:.0f}ms")
            Avg latency: 1250ms

        Filter by model:

            >>> gpt4_stats = collector.get_stats(model_name="gpt-4")
            >>> claude_stats = collector.get_stats(model_name="claude-3-opus")
            >>> print(f"GPT-4 calls: {gpt4_stats['total_calls']}")
            >>> print(f"Claude calls: {claude_stats['total_calls']}")

        Filter by operation type:

            >>> stream_stats = collector.get_stats(operation="stream")
            >>> print(f"Streaming p95 latency: {stream_stats['p95_latency_ms']:.0f}ms")

        Get recent statistics:

            >>> from datetime import datetime, timedelta
            >>> one_hour_ago = datetime.now() - timedelta(hours=1)
            >>> recent = collector.get_stats(since=one_hour_ago)
            >>> print(f"Calls in last hour: {recent['total_calls']}")

        Combine multiple filters:

            >>> recent_gpt4_chat = collector.get_stats(
            ...     model_name="gpt-4",
            ...     operation="chat",
            ...     since=datetime.now() - timedelta(minutes=30),
            ... )

        Handle empty results:

            >>> empty_stats = collector.get_stats(model_name="nonexistent-model")
            >>> empty_stats["total_calls"]
            0
            >>> empty_stats["success_rate"]
            0.0

        Notes
        -----
        - The p95 latency requires at least 20 records for meaningful
          calculation. With fewer records, max latency is returned instead.

        - Statistics are computed on-demand by iterating over records.
          For very large record sets, consider caching or sampling.

        - Token counts are estimates unless models provide exact values.

        See Also
        --------
        get_records : Retrieve individual records for detailed analysis
        """
        filtered = self.records

        if model_name:
            filtered = [r for r in filtered if r.model_name == model_name]
        if operation:
            filtered = [r for r in filtered if r.operation == operation]
        if since:
            filtered = [r for r in filtered if r.start_time >= since]

        if not filtered:
            return {
                "total_calls": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
                "total_tokens": 0,
            }

        total = len(filtered)
        successes = sum(1 for r in filtered if r.success)
        latencies = [r.latency_ms for r in filtered]
        tokens = sum(r.prompt_tokens + r.completion_tokens for r in filtered)

        return {
            "total_calls": total,
            "success_rate": successes / total,
            "successes": successes,
            "failures": total - successes,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)]
            if len(latencies) >= 20
            else max(latencies),
            "total_tokens": tokens,
            "avg_tokens_per_call": tokens / total,
        }

    def get_records(
        self,
        limit: int = 100,
        model_name: Optional[str] = None,
    ) -> list[CallRecord]:
        """Retrieve recent call records, optionally filtered by model.

        Returns the most recent records, with optional filtering. Useful for
        debugging specific issues or examining individual call details that
        aggregate statistics don't capture.

        Parameters
        ----------
        limit : int, default=100
            Maximum number of records to return. Records are returned in
            chronological order (oldest to newest within the limit).

        model_name : str or None, default=None
            If provided, only return records matching this model name.
            Filtering is applied before the limit.

        Returns
        -------
        list[CallRecord]
            List of the most recent matching records, up to ``limit`` entries.
            Returns an empty list if no records match.

        Examples
        --------
        Get the last 10 records:

            >>> records = collector.get_records(limit=10)
            >>> for r in records:
            ...     print(f"{r.model_name}: {r.latency_ms}ms - {'OK' if r.success else 'FAIL'}")

        Get recent GPT-4 calls:

            >>> gpt4_records = collector.get_records(limit=50, model_name="gpt-4")
            >>> avg_latency = sum(r.latency_ms for r in gpt4_records) / len(gpt4_records)

        Examine failed calls:

            >>> recent = collector.get_records(limit=100)
            >>> failures = [r for r in recent if not r.success]
            >>> for f in failures:
            ...     print(f"Error in {f.operation}: {f.error}")

        Get the most recent call:

            >>> if collector.records:
            ...     latest = collector.get_records(limit=1)[0]
            ...     print(f"Last call: {latest.model_name} at {latest.end_time}")

        See Also
        --------
        get_stats : Aggregate statistics over records
        export_json : Export all records as JSON
        """
        filtered = self.records
        if model_name:
            filtered = [r for r in filtered if r.model_name == model_name]
        return filtered[-limit:]

    def clear(self) -> None:
        """Remove all stored records.

        Clears the internal record list. Registered callbacks are preserved
        and will continue to receive new records.

        Examples
        --------
        Reset collector between test runs:

            >>> collector = get_collector()
            >>> collector.clear()
            >>> assert len(collector.records) == 0

        Clear after exporting:

            >>> json_data = collector.export_json()
            >>> save_to_storage(json_data)
            >>> collector.clear()  # Free memory after export

        See Also
        --------
        export_json : Export records before clearing
        """
        self.records.clear()

    def export_json(self) -> str:
        """Export all records as a JSON-formatted string.

        Serializes all stored records to JSON format, suitable for saving
        to files, sending to external systems, or archiving. Each record
        is converted using its ``to_dict()`` method.

        Returns
        -------
        str
            JSON string containing an array of all records. The output is
            pretty-printed with 2-space indentation for readability.

        Examples
        --------
        Save to file:

            >>> json_data = collector.export_json()
            >>> with open("telemetry_export.json", "w") as f:
            ...     f.write(json_data)

        Parse for analysis:

            >>> import json
            >>> data = json.loads(collector.export_json())
            >>> high_latency = [r for r in data if r["latency_ms"] > 5000]

        Send to external API:

            >>> import requests
            >>> requests.post(
            ...     "https://api.monitoring.example.com/ingest",
            ...     headers={"Content-Type": "application/json"},
            ...     data=collector.export_json(),
            ... )

        Periodic export with clearing:

            >>> def periodic_export():
            ...     if len(collector.records) > 1000:
            ...         timestamp = datetime.now().isoformat()
            ...         with open(f"telemetry_{timestamp}.json", "w") as f:
            ...             f.write(collector.export_json())
            ...         collector.clear()

        See Also
        --------
        CallRecord.to_dict : Individual record serialization
        clear : Remove records after export
        """
        import json

        return json.dumps([r.to_dict() for r in self.records], indent=2)


# Global collector instance
_global_collector: Optional[TelemetryCollector] = None


def get_collector() -> TelemetryCollector:
    """Get or create the global telemetry collector singleton.

    Returns the shared TelemetryCollector instance used by default throughout
    the application. If no collector has been set, creates a new one with
    default settings (max_records=10000).

    This function provides a convenient way to access telemetry data from
    anywhere in your application without passing collector references through
    the call stack.

    Returns
    -------
    TelemetryCollector
        The global collector instance. Always returns the same instance
        within a process (singleton pattern).

    Examples
    --------
    Access telemetry from instrumented models:

        >>> from insideLLMs.runtime.observability import get_collector, instrument_model
        >>> from insideLLMs.models import OpenAIModel
        >>>
        >>> # Instrument model (uses global collector by default)
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> traced = instrument_model(model)
        >>>
        >>> # Later, access the telemetry
        >>> collector = get_collector()
        >>> stats = collector.get_stats()
        >>> print(f"Total API calls: {stats['total_calls']}")

    Check telemetry in a different module:

        >>> # In your monitoring module
        >>> from insideLLMs.runtime.observability import get_collector
        >>>
        >>> def check_health():
        ...     collector = get_collector()
        ...     stats = collector.get_stats()
        ...     if stats["success_rate"] < 0.95:
        ...         alert("LLM success rate below threshold")

    Access records for debugging:

        >>> collector = get_collector()
        >>> recent_failures = [
        ...     r for r in collector.get_records(limit=100)
        ...     if not r.success
        ... ]
        >>> for failure in recent_failures:
        ...     print(f"{failure.model_name}: {failure.error}")

    See Also
    --------
    set_collector : Replace the global collector with a custom instance
    TelemetryCollector : The collector class

    Notes
    -----
    - The global collector is lazily initialized on first access.

    - In multi-threaded applications, all threads share the same collector.
      The TelemetryCollector class is not thread-safe; consider using
      thread-local collectors or external synchronization if needed.

    - For testing, use ``set_collector()`` to inject a mock or isolated
      collector before tests and restore afterward.
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = TelemetryCollector()
    return _global_collector


def set_collector(collector: TelemetryCollector) -> None:
    """Replace the global telemetry collector with a custom instance.

    Allows you to configure a custom collector with specific settings
    (e.g., different max_records, pre-registered callbacks) and use it
    as the default for all instrumented models and trace_call invocations.

    Parameters
    ----------
    collector : TelemetryCollector
        The collector instance to use as the global default. This replaces
        any previously set collector.

    Examples
    --------
    Configure a high-capacity collector:

        >>> from insideLLMs.runtime.observability import (
        ...     set_collector,
        ...     TelemetryCollector,
        ... )
        >>>
        >>> # Create collector with custom settings
        >>> collector = TelemetryCollector(max_records=100000)
        >>>
        >>> # Set as global default
        >>> set_collector(collector)

    Set up collector with monitoring callbacks:

        >>> def send_to_metrics(record):
        ...     metrics.timing("llm.latency", record.latency_ms)
        ...     metrics.increment("llm.calls")
        >>>
        >>> collector = TelemetryCollector()
        >>> collector.add_callback(send_to_metrics)
        >>> set_collector(collector)

    Isolate tests with fresh collectors:

        >>> import pytest
        >>>
        >>> @pytest.fixture
        ... def clean_telemetry():
        ...     from insideLLMs.runtime.observability import (
        ...         get_collector, set_collector, TelemetryCollector
        ...     )
        ...     # Save original
        ...     original = get_collector()
        ...     # Use fresh collector for test
        ...     set_collector(TelemetryCollector())
        ...     yield
        ...     # Restore original
        ...     set_collector(original)

    Replace collector at application startup:

        >>> # In your app initialization
        >>> from insideLLMs.runtime.observability import set_collector, TelemetryCollector
        >>>
        >>> def init_telemetry():
        ...     collector = TelemetryCollector(
        ...         max_records=int(os.getenv("TELEMETRY_MAX_RECORDS", 10000))
        ...     )
        ...     # Add production callbacks
        ...     if os.getenv("ENVIRONMENT") == "production":
        ...         collector.add_callback(send_to_datadog)
        ...     set_collector(collector)

    See Also
    --------
    get_collector : Retrieve the current global collector
    TelemetryCollector : The collector class
    """
    global _global_collector
    _global_collector = collector


# =============================================================================
# Token Estimation
# =============================================================================


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate the token count for a given text string.

    Provides a rough approximation of how many tokens the text would consume
    when processed by an LLM. Uses a heuristic based on character count
    (approximately 4 characters per token for English text).

    This function is useful for cost estimation, rate limiting, and context
    window management when exact token counts are not available.

    Parameters
    ----------
    text : str
        The text to estimate tokens for. Can be any length, including empty.

    model : str, default="gpt-4"
        Model name hint. Currently ignored (kept for backward compatibility
        and potential future model-specific estimation).

    Returns
    -------
    int
        Estimated number of tokens. Returns 0 for empty strings.

    Examples
    --------
    Basic token estimation:

        >>> estimate_tokens("Hello, world!")
        4
        >>> estimate_tokens("The quick brown fox jumps over the lazy dog.")
        11

    Estimate tokens for a prompt:

        >>> prompt = "Explain the theory of relativity in simple terms."
        >>> tokens = estimate_tokens(prompt)
        >>> print(f"Estimated prompt tokens: {tokens}")
        Estimated prompt tokens: 11

    Check if content fits in context window:

        >>> MAX_CONTEXT = 8192
        >>> document = load_document()
        >>> prompt_tokens = estimate_tokens(prompt)
        >>> doc_tokens = estimate_tokens(document)
        >>> if prompt_tokens + doc_tokens > MAX_CONTEXT:
        ...     document = truncate_to_fit(document, MAX_CONTEXT - prompt_tokens)

    Estimate cost before API call:

        >>> COST_PER_1K_INPUT = 0.01
        >>> COST_PER_1K_OUTPUT = 0.03
        >>> input_tokens = estimate_tokens(prompt)
        >>> estimated_output = 500  # expected response length
        >>> estimated_cost = (
        ...     (input_tokens / 1000) * COST_PER_1K_INPUT +
        ...     (estimated_output / 1000) * COST_PER_1K_OUTPUT
        ... )

    Pre-flight check for batch processing:

        >>> prompts = ["Summarize this:", "Translate this:", "Explain this:"]
        >>> total_estimated = sum(estimate_tokens(p) for p in prompts)
        >>> print(f"Batch will use approximately {total_estimated} input tokens")

    See Also
    --------
    insideLLMs.tokens.estimate_tokens : Canonical token estimation function
    CallRecord : Uses this function for prompt_tokens and completion_tokens

    Notes
    -----
    - This is a heuristic approximation. Actual token counts vary by model
      and tokenizer. For precise counts, use the model's native tokenizer
      (e.g., tiktoken for OpenAI models).

    - The approximation works reasonably well for English text but may be
      less accurate for other languages, code, or text with many special
      characters.

    - The estimation uses approximately 4 characters per token, which is
      a common average for English text with GPT-style tokenizers.
    """
    _ = model  # kept for backward compatibility
    return _canonical_estimate_tokens(text)


# =============================================================================
# Tracing Context Manager
# =============================================================================


@contextmanager
def trace_call(
    model_name: str,
    operation: str,
    prompt: str = "",
    collector: Optional[TelemetryCollector] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Iterator[dict[str, Any]]:
    """Context manager for tracing any operation with telemetry recording.

    Provides a flexible way to add tracing to custom operations beyond the
    standard model methods. Automatically captures timing, handles errors,
    and records a CallRecord to the specified collector.

    The context manager yields a mutable dictionary where you can store the
    response text (for token counting) and any additional context. On exit,
    a complete CallRecord is created and stored.

    Parameters
    ----------
    model_name : str
        Identifier for the model or component being traced. For non-model
        operations, use a descriptive name like "embedding-service" or
        "custom-processor".

    operation : str
        Type of operation being performed. Standard values include "generate",
        "chat", "stream", "embedding". Custom values are also supported.

    prompt : str, default=""
        The input text for the operation. Used for token estimation and
        character counting. Pass empty string if not applicable.

    collector : TelemetryCollector or None, default=None
        The collector to record to. If None, uses the global collector from
        ``get_collector()``.

    metadata : dict[str, Any] or None, default=None
        Additional key-value pairs to attach to the CallRecord. Useful for
        request IDs, user identifiers, or operation-specific data.

    Yields
    ------
    dict[str, Any]
        A mutable context dictionary with the following keys:
        - ``response`` (str): Set this to the response text for token counting
        - ``error`` (str or None): Automatically set if an exception occurs
        - ``success`` (bool): Automatically set based on exception status

    Raises
    ------
    Exception
        Re-raises any exception that occurs within the context block after
        recording the failure.

    Examples
    --------
    Basic tracing of a model call:

        >>> from insideLLMs.runtime.observability import trace_call
        >>>
        >>> prompt = "Explain quantum computing"
        >>> with trace_call("gpt-4", "generate", prompt) as ctx:
        ...     response = model.generate(prompt)
        ...     ctx["response"] = response
        >>> print(response)

    Tracing with custom metadata:

        >>> metadata = {
        ...     "request_id": "req_abc123",
        ...     "user_id": "user_456",
        ...     "temperature": 0.7,
        ... }
        >>> with trace_call("claude-3", "chat", str(messages), metadata=metadata) as ctx:
        ...     response = model.chat(messages)
        ...     ctx["response"] = response

    Tracing a non-model operation:

        >>> with trace_call("embedding-service", "embed", text) as ctx:
        ...     embedding = compute_embedding(text)
        ...     ctx["response"] = str(len(embedding))  # Store embedding dimension

    Error handling (errors are recorded and re-raised):

        >>> try:
        ...     with trace_call("gpt-4", "generate", prompt) as ctx:
        ...         response = model.generate(prompt)  # Might raise
        ...         ctx["response"] = response
        ... except APIError as e:
        ...     # Error is recorded before reaching here
        ...     handle_error(e)

    Using a custom collector:

        >>> custom_collector = TelemetryCollector(max_records=100)
        >>> with trace_call("gpt-4", "generate", prompt, collector=custom_collector) as ctx:
        ...     response = model.generate(prompt)
        ...     ctx["response"] = response
        >>> stats = custom_collector.get_stats()

    Tracing streaming operations (basic approach):

        >>> with trace_call("gpt-4", "stream", prompt) as ctx:
        ...     chunks = []
        ...     for chunk in model.stream(prompt):
        ...         chunks.append(chunk)
        ...         yield chunk
        ...     ctx["response"] = "".join(chunks)

    Nested tracing for complex workflows:

        >>> with trace_call("orchestrator", "process", input_text) as outer:
        ...     # First model call
        ...     with trace_call("gpt-4", "analyze", input_text) as inner1:
        ...         analysis = model1.generate(input_text)
        ...         inner1["response"] = analysis
        ...
        ...     # Second model call based on first
        ...     with trace_call("claude-3", "summarize", analysis) as inner2:
        ...         summary = model2.generate(analysis)
        ...         inner2["response"] = summary
        ...
        ...     outer["response"] = summary

    See Also
    --------
    TracedModel : Automatic tracing wrapper for models
    TelemetryCollector : Where records are stored
    CallRecord : The record type created by this context manager

    Notes
    -----
    - Timing starts immediately when entering the context and ends in the
      finally block, so cleanup code after the yield is included in latency.

    - If you don't set ``ctx["response"]``, completion_tokens will be 0.

    - The context manager is reentrant; you can nest trace_call blocks for
      hierarchical tracing of complex workflows.

    - For async operations, consider using an async version or wrapping
      appropriately.
    """
    collector = collector or get_collector()
    metadata = metadata or {}

    start_time = datetime.now()
    ctx: dict[str, Any] = {
        "response": "",
        "error": None,
        "success": True,
    }

    try:
        yield ctx
    except Exception as e:
        ctx["error"] = str(e)
        ctx["success"] = False
        raise
    finally:
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000

        response = ctx.get("response", "")

        record = CallRecord(
            model_name=model_name,
            operation=operation,
            start_time=start_time,
            end_time=end_time,
            latency_ms=latency_ms,
            success=ctx["success"],
            error=ctx.get("error"),
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(response) if response else 0,
            prompt_length=len(prompt),
            response_length=len(response) if response else 0,
            metadata=metadata,
        )
        collector.record(record)


# =============================================================================
# Model Instrumentation
# =============================================================================


class TracedModel:
    """Transparent wrapper that adds telemetry tracing to any model.

    TracedModel wraps an existing model instance and intercepts all calls to
    ``generate()``, ``chat()``, and ``stream()``, automatically recording
    telemetry data (timing, token counts, success/failure) to a collector.

    The wrapper is transparent: it exposes the same interface as the wrapped
    model and delegates all other attributes and methods unchanged. This allows
    TracedModel to be used as a drop-in replacement anywhere a Model is expected.

    Parameters
    ----------
    model : Model
        The model instance to wrap. Must implement at least ``generate()``.
        Optional ``chat()`` and ``stream()`` methods are also wrapped if present.

    collector : TelemetryCollector or None, default=None
        The collector to record telemetry to. If None, uses the global
        collector from ``get_collector()``.

    config : TracingConfig or None, default=None
        Configuration controlling privacy settings (log_prompts, log_responses)
        and other tracing behavior. If None, uses default TracingConfig.

    Attributes
    ----------
    name : str
        The name of the wrapped model (delegated to the underlying model).

    Examples
    --------
    Basic instrumentation:

        >>> from insideLLMs.runtime.observability import TracedModel, get_collector
        >>> from insideLLMs.models import OpenAIModel
        >>>
        >>> # Wrap an existing model
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> traced = TracedModel(model)
        >>>
        >>> # Use exactly like the original
        >>> response = traced.generate("Explain recursion")
        >>>
        >>> # Check the telemetry
        >>> stats = get_collector().get_stats()
        >>> print(f"Latency: {stats['avg_latency_ms']:.0f}ms")

    With custom configuration:

        >>> from insideLLMs.runtime.observability import TracedModel, TracingConfig
        >>>
        >>> config = TracingConfig(
        ...     log_prompts=True,   # Capture prompts (dev only!)
        ...     log_responses=True,  # Capture responses
        ... )
        >>> traced = TracedModel(model, config=config)

    With isolated collector:

        >>> from insideLLMs.runtime.observability import TracedModel, TelemetryCollector
        >>>
        >>> # Create a dedicated collector for this model
        >>> my_collector = TelemetryCollector(max_records=1000)
        >>> traced = TracedModel(model, collector=my_collector)
        >>>
        >>> # Telemetry is isolated from global collector
        >>> response = traced.generate("Hello")
        >>> print(my_collector.get_stats())

    Using chat interface:

        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "What is Python?"},
        ... ]
        >>> response = traced.chat(messages)

    Using streaming:

        >>> for chunk in traced.stream("Write a story"):
        ...     print(chunk, end="", flush=True)

    Accessing underlying model properties:

        >>> # All model attributes are accessible
        >>> print(traced.name)
        >>> info = traced.info()

    See Also
    --------
    instrument_model : Convenience function to create TracedModel
    OTelTracedModel : Wrapper with OpenTelemetry distributed tracing
    TracingConfig : Configuration options for tracing

    Notes
    -----
    - Token counts are estimates based on character count. For exact counts,
      use the model's native tokenizer.

    - When ``log_prompts`` or ``log_responses`` is False (default), the prompt
      and response text are not stored in the CallRecord, but character lengths
      and token estimates are still recorded.

    - Streaming records include the full latency from first chunk to last,
      not time-to-first-token.

    - The wrapper uses ``__getattr__`` to delegate unknown attributes, so
      all model-specific methods and properties remain accessible.
    """

    def __init__(
        self,
        model: "Model",
        collector: Optional[TelemetryCollector] = None,
        config: Optional[TracingConfig] = None,
    ):
        """Initialize the traced model wrapper.

        Creates a new TracedModel that wraps the given model and records
        telemetry for all generate/chat/stream operations.

        Parameters
        ----------
        model : Model
            The model instance to wrap. Must have a ``name`` property and
            ``generate()`` method. May optionally have ``chat()`` and
            ``stream()`` methods.

        collector : TelemetryCollector or None, default=None
            The telemetry collector to record to. If None, the global
            collector from ``get_collector()`` is used.

        config : TracingConfig or None, default=None
            Configuration for tracing behavior. If None, default
            TracingConfig settings are used (no prompt/response logging).

        Examples
        --------
        Minimal initialization:

            >>> traced = TracedModel(model)

        With all options:

            >>> traced = TracedModel(
            ...     model=model,
            ...     collector=TelemetryCollector(max_records=5000),
            ...     config=TracingConfig(log_prompts=True),
            ... )

        See Also
        --------
        instrument_model : Alternative way to create TracedModel
        """
        self._model = model
        self._collector = collector or get_collector()
        self._config = config or TracingConfig()

    @property
    def name(self) -> str:
        """Get the name of the wrapped model.

        Returns
        -------
        str
            The model's name, delegated from the underlying model instance.

        Examples
        --------
            >>> traced = TracedModel(OpenAIModel(model_name="gpt-4"))
            >>> traced.name
            'gpt-4'
        """
        return self._model.name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text with automatic tracing.

        Wraps the underlying model's generate method, capturing timing,
        token estimates, and success/failure status in a CallRecord.

        Parameters
        ----------
        prompt : str
            The input prompt for text generation.

        **kwargs : Any
            Additional arguments passed through to the underlying model's
            generate method (e.g., temperature, max_tokens).

        Returns
        -------
        str
            The generated text response from the model.

        Raises
        ------
        Exception
            Any exception raised by the underlying model is recorded and
            re-raised.

        Examples
        --------
        Basic generation:

            >>> response = traced.generate("Explain machine learning")
            >>> print(response[:100])

        With model parameters:

            >>> response = traced.generate(
            ...     "Write a haiku",
            ...     temperature=0.9,
            ...     max_tokens=50,
            ... )

        Error handling:

            >>> try:
            ...     response = traced.generate(prompt)
            ... except RateLimitError:
            ...     # Error is recorded in telemetry before raising
            ...     time.sleep(60)
            ...     response = traced.generate(prompt)

        See Also
        --------
        chat : For multi-turn conversations
        stream : For streaming responses
        """
        metadata = {"kwargs": str(kwargs)} if kwargs else {}

        with trace_call(
            self._model.name,
            "generate",
            prompt if self._config.log_prompts else "",
            self._collector,
            metadata,
        ) as ctx:
            response = self._model.generate(prompt, **kwargs)
            ctx["response"] = response if self._config.log_responses else ""
            return response

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Conduct a multi-turn chat with automatic tracing.

        Wraps the underlying model's chat method, capturing timing and
        metrics. The messages list is converted to a string for token
        estimation if prompt logging is enabled.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of message dictionaries with "role" and "content" keys.
            Typical roles are "system", "user", and "assistant".

        **kwargs : Any
            Additional arguments passed through to the underlying model's
            chat method.

        Returns
        -------
        str
            The assistant's response message.

        Raises
        ------
        NotImplementedError
            If the underlying model does not have a chat method.

        Exception
            Any exception raised by the underlying model is recorded and
            re-raised.

        Examples
        --------
        Basic chat:

            >>> messages = [
            ...     {"role": "user", "content": "What is the capital of France?"}
            ... ]
            >>> response = traced.chat(messages)
            >>> print(response)
            The capital of France is Paris.

        Multi-turn conversation:

            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful tutor."},
            ...     {"role": "user", "content": "Explain photosynthesis"},
            ...     {"role": "assistant", "content": "Photosynthesis is..."},
            ...     {"role": "user", "content": "Can you simplify that?"},
            ... ]
            >>> response = traced.chat(messages)

        With parameters:

            >>> response = traced.chat(
            ...     messages,
            ...     temperature=0.3,
            ...     max_tokens=500,
            ... )

        See Also
        --------
        generate : For single-turn generation
        stream : For streaming responses
        """
        if not hasattr(self._model, "chat"):
            raise NotImplementedError("Model does not support chat")

        prompt = str(messages) if self._config.log_prompts else ""

        with trace_call(
            self._model.name,
            "chat",
            prompt,
            self._collector,
        ) as ctx:
            response = self._model.chat(messages, **kwargs)
            ctx["response"] = response if self._config.log_responses else ""
            return response

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream generated text with automatic tracing.

        Wraps the underlying model's stream method, yielding chunks as they
        arrive while collecting them for final telemetry recording. The
        complete response timing and token count are recorded after the
        stream completes.

        Parameters
        ----------
        prompt : str
            The input prompt for text generation.

        **kwargs : Any
            Additional arguments passed through to the underlying model's
            stream method.

        Yields
        ------
        str
            Text chunks as they are generated by the model.

        Raises
        ------
        NotImplementedError
            If the underlying model does not have a stream method.

        Exception
            Any exception raised during streaming is recorded and re-raised.

        Examples
        --------
        Basic streaming:

            >>> for chunk in traced.stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
            >>> print()  # Newline after stream completes

        Collecting streamed response:

            >>> chunks = list(traced.stream("Explain quantum physics"))
            >>> full_response = "".join(chunks)
            >>> print(f"Received {len(chunks)} chunks, {len(full_response)} chars")

        With progress indicator:

            >>> import sys
            >>> for i, chunk in enumerate(traced.stream(prompt)):
            ...     sys.stdout.write(chunk)
            ...     if i % 10 == 0:
            ...         sys.stdout.write(f" [{i} chunks]")
            ...     sys.stdout.flush()

        Error handling:

            >>> try:
            ...     for chunk in traced.stream(prompt):
            ...         process(chunk)
            ... except ConnectionError:
            ...     # Partial results recorded in telemetry
            ...     handle_disconnect()

        Notes
        -----
        - The CallRecord is created in the finally block, so telemetry is
          recorded even if the stream is interrupted or errors occur.

        - Latency represents total time from start to final chunk, not
          time-to-first-token.

        - Token counts are estimated from the concatenated full response.

        See Also
        --------
        generate : For non-streaming generation
        chat : For multi-turn conversations
        """
        if not hasattr(self._model, "stream"):
            raise NotImplementedError("Model does not support streaming")

        start_time = datetime.now()
        chunks = []
        error = None

        try:
            for chunk in self._model.stream(prompt, **kwargs):
                chunks.append(chunk)
                yield chunk
        except Exception as e:
            error = str(e)
            raise
        finally:
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            full_response = "".join(chunks)

            record = CallRecord(
                model_name=self._model.name,
                operation="stream",
                start_time=start_time,
                end_time=end_time,
                latency_ms=latency_ms,
                success=error is None,
                error=error,
                prompt_tokens=estimate_tokens(prompt),
                completion_tokens=estimate_tokens(full_response),
                prompt_length=len(prompt),
                response_length=len(full_response),
            )
            self._collector.record(record)

    def info(self) -> "ModelInfo":
        """Return model information from the wrapped model.

        Delegates to the underlying model's info() method.

        Returns
        -------
        ModelInfo
            Model information as returned by the wrapped model.

        Examples
        --------
            >>> info = traced.info()
            >>> print(info)
        """
        return self._model.info()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped model.

        Allows transparent access to all attributes and methods of the
        underlying model that are not explicitly defined on TracedModel.

        Parameters
        ----------
        name : str
            The attribute name to access.

        Returns
        -------
        Any
            The attribute value from the wrapped model.

        Raises
        ------
        AttributeError
            If the attribute doesn't exist on the wrapped model.

        Examples
        --------
        Accessing model-specific attributes:

            >>> # If underlying model has a 'tokenizer' attribute
            >>> tokenizer = traced.tokenizer
            >>>
            >>> # If underlying model has custom methods
            >>> traced.custom_method()
        """
        return getattr(self._model, name)


def instrument_model(
    model: "Model",
    config: Optional[TracingConfig] = None,
    collector: Optional[TelemetryCollector] = None,
) -> TracedModel:
    """Wrap a model with automatic telemetry tracing.

    This is the primary entry point for adding observability to models.
    Returns a TracedModel wrapper that records all generate/chat/stream
    operations to the telemetry collector.

    Parameters
    ----------
    model : Model
        The model instance to instrument. Must have a ``name`` property
        and ``generate()`` method at minimum.

    config : TracingConfig or None, default=None
        Configuration controlling tracing behavior, including privacy
        settings for prompt/response logging. If None, uses default
        TracingConfig (no prompt/response capture).

    collector : TelemetryCollector or None, default=None
        The collector to record telemetry to. If None, uses the global
        collector from ``get_collector()``.

    Returns
    -------
    TracedModel
        A wrapped model that automatically records telemetry for all
        supported operations (generate, chat, stream).

    Examples
    --------
    Basic instrumentation:

        >>> from insideLLMs.runtime.observability import instrument_model
        >>> from insideLLMs.models import OpenAIModel
        >>>
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> traced = instrument_model(model)
        >>>
        >>> # Use exactly like the original model
        >>> response = traced.generate("Hello, world!")

    With configuration for development debugging:

        >>> from insideLLMs.runtime.observability import instrument_model, TracingConfig
        >>>
        >>> config = TracingConfig(
        ...     log_prompts=True,
        ...     log_responses=True,
        ...     console_export=True,
        ... )
        >>> traced = instrument_model(model, config=config)

    With isolated collector for testing:

        >>> from insideLLMs.runtime.observability import (
        ...     instrument_model,
        ...     TelemetryCollector,
        ... )
        >>>
        >>> test_collector = TelemetryCollector(max_records=100)
        >>> traced = instrument_model(model, collector=test_collector)
        >>>
        >>> # Run tests
        >>> response = traced.generate("Test prompt")
        >>>
        >>> # Verify telemetry in isolated collector
        >>> assert test_collector.get_stats()["total_calls"] == 1

    Production setup with privacy protection:

        >>> config = TracingConfig(
        ...     service_name="prod-llm-api",
        ...     log_prompts=False,  # Never log user prompts
        ...     log_responses=False,
        ...     sample_rate=0.1,
        ... )
        >>> traced = instrument_model(model, config=config)

    Chaining with other wrappers:

        >>> # Instrument model, then add retry logic
        >>> traced = instrument_model(model)
        >>> resilient = add_retry_wrapper(traced, max_retries=3)

    See Also
    --------
    TracedModel : The wrapper class returned by this function
    TracingConfig : Configuration options
    get_collector : Access the global telemetry collector

    Notes
    -----
    - The returned TracedModel is a drop-in replacement for the original
      model. All attributes and methods are accessible.

    - For OpenTelemetry distributed tracing (Jaeger, Tempo, etc.), use
      ``OTelTracedModel`` instead or in addition to this wrapper.

    - The wrapper does not modify the original model instance.
    """
    return TracedModel(model, collector, config)


# =============================================================================
# OpenTelemetry Integration
# =============================================================================


def setup_otel_tracing(config: TracingConfig) -> None:
    """Set up OpenTelemetry tracing.

    Args:
        config: Tracing configuration.

    Raises:
        ImportError: If OpenTelemetry is not installed.
    """
    if not OTEL_AVAILABLE:
        raise ImportError(
            "OpenTelemetry is required for distributed tracing. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk"
        )

    resource = Resource.create(
        {
            ResourceAttributes.SERVICE_NAME: config.service_name,
            **config.custom_attributes,
        }
    )

    provider = TracerProvider(resource=resource)

    if config.console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    if config.jaeger_endpoint:
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter

            exporter = JaegerExporter(
                collector_endpoint=config.jaeger_endpoint,
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            logger.warning("Jaeger exporter not available")

    if config.otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            logger.warning("OTLP exporter not available")

    trace.set_tracer_provider(provider)


class OTelTracedModel:
    """Model wrapper with OpenTelemetry distributed tracing.

    Requires OpenTelemetry SDK to be installed.
    """

    def __init__(
        self,
        model: "Model",
        config: Optional[TracingConfig] = None,
    ):
        """Initialize the OTel traced model.

        Args:
            model: Model to wrap.
            config: Tracing configuration.
        """
        if not OTEL_AVAILABLE:
            raise ImportError(
                "OpenTelemetry is required. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk"
            )

        self._model = model
        self._config = config or TracingConfig()
        self._tracer = trace.get_tracer(__name__)

    @property
    def name(self) -> str:
        return self._model.name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate with OpenTelemetry tracing."""
        with self._tracer.start_as_current_span("llm.generate") as span:
            span.set_attribute("llm.model", self._model.name)
            span.set_attribute("llm.operation", "generate")
            span.set_attribute("llm.prompt_length", len(prompt))

            if self._config.log_prompts:
                span.set_attribute("llm.prompt", prompt[:1000])

            try:
                response = self._model.generate(prompt, **kwargs)
                span.set_attribute("llm.response_length", len(response))
                span.set_attribute("llm.success", True)

                if self._config.log_responses:
                    span.set_attribute("llm.response", response[:1000])

                return response
            except Exception as e:
                span.set_attribute("llm.success", False)
                span.set_attribute("llm.error", str(e))
                span.record_exception(e)
                raise

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Chat with OpenTelemetry tracing."""
        if not hasattr(self._model, "chat"):
            raise NotImplementedError("Model does not support chat")

        with self._tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("llm.model", self._model.name)
            span.set_attribute("llm.operation", "chat")
            span.set_attribute("llm.message_count", len(messages))

            try:
                response = self._model.chat(messages, **kwargs)
                span.set_attribute("llm.response_length", len(response))
                span.set_attribute("llm.success", True)
                return response
            except Exception as e:
                span.set_attribute("llm.success", False)
                span.set_attribute("llm.error", str(e))
                span.record_exception(e)
                raise

    def info(self) -> "ModelInfo":
        """Return model info."""
        return self._model.info()

    def __getattr__(self, name: str) -> Any:
        """Delegate to wrapped model."""
        return getattr(self._model, name)


# =============================================================================
# Convenience Decorators
# =============================================================================


def trace_function(
    operation_name: Optional[str] = None,
    include_args: bool = False,
) -> Callable:
    """Decorator to trace a function call.

    Args:
        operation_name: Name for the operation (defaults to function name).
        include_args: Whether to include function arguments in metadata.

    Returns:
        Decorator function.

    Example:
        >>> @trace_function()
        ... def my_function(x):
        ...     return x * 2
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            name = operation_name or func.__name__
            metadata = {}
            if include_args:
                metadata["args"] = str(args)
                metadata["kwargs"] = str(kwargs)

            collector = get_collector()
            start_time = datetime.now()
            error = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                end_time = datetime.now()
                latency_ms = (end_time - start_time).total_seconds() * 1000

                record = CallRecord(
                    model_name="function",
                    operation=name,
                    start_time=start_time,
                    end_time=end_time,
                    latency_ms=latency_ms,
                    success=error is None,
                    error=error,
                    metadata=metadata,
                )
                collector.record(record)

        return wrapper

    return decorator


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Configuration
    "TracingConfig",
    # Telemetry
    "CallRecord",
    "TelemetryCollector",
    "get_collector",
    "set_collector",
    # Tracing
    "trace_call",
    "trace_function",
    # Model instrumentation
    "TracedModel",
    "instrument_model",
    # OpenTelemetry
    "setup_otel_tracing",
    "OTelTracedModel",
    "OTEL_AVAILABLE",
    # Token estimation
    "estimate_tokens",
]
