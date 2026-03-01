"""
Compatibility shim for insideLLMs.runtime.observability.

This module provides backward-compatible access to the observability subsystem,
which offers comprehensive telemetry, tracing, and monitoring capabilities for
LLM operations. All functionality is re-exported from
:mod:`insideLLMs.runtime.observability`.

Overview
--------
The observability module enables you to:

- **Trace model calls**: Automatically record timing, token usage, and success/failure
  for all LLM interactions.
- **Collect metrics**: Aggregate statistics including latency percentiles, success rates,
  and token consumption.
- **Integrate with OpenTelemetry**: Export traces to distributed tracing backends like
  Jaeger, Zipkin, or any OTLP-compatible collector.
- **Debug and optimize**: Use structured logging and telemetry to identify bottlenecks
  and failures in your LLM pipelines.

Quick Start
-----------
Basic instrumentation without external dependencies:

    >>> from insideLLMs.observability import instrument_model, get_collector
    >>> from insideLLMs.models import OpenAIModel
    >>>
    >>> # Create and instrument a model
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> traced_model = instrument_model(model)
    >>>
    >>> # Use the model normally - calls are automatically traced
    >>> response = traced_model.generate("What is the capital of France?")
    >>>
    >>> # Access collected telemetry
    >>> collector = get_collector()
    >>> stats = collector.get_stats()
    >>> print(f"Total calls: {stats['total_calls']}")
    >>> print(f"Success rate: {stats['success_rate']:.2%}")
    >>> print(f"Avg latency: {stats['avg_latency_ms']:.1f}ms")

Exported Classes
----------------
TracingConfig
    Configuration dataclass for tracing behavior, including service name,
    sampling rate, and export destinations.

    Example::

        >>> from insideLLMs.observability import TracingConfig
        >>>
        >>> # Basic configuration
        >>> config = TracingConfig(
        ...     service_name="my-chatbot",
        ...     enabled=True,
        ...     log_prompts=False,  # Don't log sensitive data
        ...     log_responses=False,
        ... )
        >>>
        >>> # Configuration with Jaeger export
        >>> config = TracingConfig(
        ...     service_name="production-app",
        ...     jaeger_endpoint="http://jaeger:14268/api/traces",
        ...     sample_rate=0.1,  # Sample 10% of requests
        ...     custom_attributes={"environment": "production", "team": "ml-ops"},
        ... )
        >>>
        >>> # Configuration with OTLP export
        >>> config = TracingConfig(
        ...     service_name="analytics-service",
        ...     otlp_endpoint="http://otel-collector:4317",
        ...     console_export=True,  # Also log to console for debugging
        ... )

CallRecord
    Dataclass representing a single recorded model call with timing, tokens,
    and success information.

    Example::

        >>> from insideLLMs.observability import CallRecord
        >>> from datetime import datetime
        >>>
        >>> # Manually create a call record (typically done automatically)
        >>> record = CallRecord(
        ...     model_name="gpt-4",
        ...     operation="generate",
        ...     start_time=datetime(2024, 1, 15, 10, 30, 0),
        ...     end_time=datetime(2024, 1, 15, 10, 30, 2),
        ...     latency_ms=2000.0,
        ...     success=True,
        ...     prompt_tokens=50,
        ...     completion_tokens=150,
        ...     prompt_length=200,
        ...     response_length=600,
        ...     metadata={"temperature": "0.7"},
        ... )
        >>>
        >>> # Convert to dictionary for JSON serialization
        >>> data = record.to_dict()
        >>> print(data["total_tokens"])  # 200

TelemetryCollector
    Collector that stores CallRecord instances and computes aggregate statistics.

    Example::

        >>> from insideLLMs.observability import TelemetryCollector, CallRecord
        >>> from datetime import datetime, timedelta
        >>>
        >>> # Create a collector with custom capacity
        >>> collector = TelemetryCollector(max_records=5000)
        >>>
        >>> # Add a callback for real-time monitoring
        >>> def on_call(record: CallRecord):
        ...     if not record.success:
        ...         print(f"ALERT: {record.model_name} failed: {record.error}")
        ...     if record.latency_ms > 5000:
        ...         print(f"SLOW: {record.model_name} took {record.latency_ms}ms")
        >>>
        >>> collector.add_callback(on_call)
        >>>
        >>> # Get statistics for a specific model
        >>> stats = collector.get_stats(model_name="gpt-4")
        >>> print(f"GPT-4 p95 latency: {stats['p95_latency_ms']}ms")
        >>>
        >>> # Get statistics for the last hour
        >>> since = datetime.now() - timedelta(hours=1)
        >>> recent_stats = collector.get_stats(since=since)
        >>>
        >>> # Export all records as JSON
        >>> json_data = collector.export_json()

TracedModel
    Wrapper class that adds automatic tracing to any model without requiring
    OpenTelemetry dependencies.

    Example::

        >>> from insideLLMs.observability import TracedModel, TracingConfig
        >>> from insideLLMs.models import AnthropicModel
        >>>
        >>> model = AnthropicModel(model_name="claude-3-opus")
        >>> config = TracingConfig(
        ...     service_name="my-app",
        ...     log_prompts=True,  # Enable for debugging
        ...     log_responses=True,
        ... )
        >>>
        >>> traced = TracedModel(model, config=config)
        >>>
        >>> # All operations are traced
        >>> response = traced.generate("Explain quantum computing")
        >>> chat_response = traced.chat([
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ...     {"role": "user", "content": "How are you?"},
        ... ])
        >>>
        >>> # Streaming is also traced
        >>> for chunk in traced.stream("Tell me a story"):
        ...     print(chunk, end="")

OTelTracedModel
    Model wrapper with full OpenTelemetry distributed tracing support.
    Requires OpenTelemetry SDK to be installed.

    Example::

        >>> from insideLLMs.observability import (
        ...     OTelTracedModel,
        ...     TracingConfig,
        ...     setup_otel_tracing,
        ...     OTEL_AVAILABLE,
        ... )
        >>> from insideLLMs.models import OpenAIModel
        >>>
        >>> if OTEL_AVAILABLE:
        ...     # Configure OpenTelemetry
        ...     config = TracingConfig(
        ...         service_name="production-llm-service",
        ...         jaeger_endpoint="http://jaeger:14268/api/traces",
        ...         custom_attributes={
        ...             "deployment.environment": "production",
        ...             "service.version": "1.2.3",
        ...         },
        ...     )
        ...     setup_otel_tracing(config)
        ...
        ...     # Create traced model
        ...     model = OpenAIModel(model_name="gpt-4-turbo")
        ...     traced = OTelTracedModel(model, config)
        ...
        ...     # Calls create distributed traces
        ...     response = traced.generate("Summarize this document...")
        ... else:
        ...     print("OpenTelemetry not installed")

Exported Functions
------------------
instrument_model(model, config=None, collector=None)
    Convenience function to wrap a model with tracing.

    Example::

        >>> from insideLLMs.observability import instrument_model, TracingConfig
        >>> from insideLLMs.models import OpenAIModel
        >>>
        >>> # Simple instrumentation
        >>> model = OpenAIModel(model_name="gpt-3.5-turbo")
        >>> traced = instrument_model(model)
        >>>
        >>> # With custom configuration
        >>> config = TracingConfig(service_name="chatbot", sample_rate=0.5)
        >>> traced = instrument_model(model, config=config)
        >>>
        >>> # With custom collector
        >>> from insideLLMs.observability import TelemetryCollector
        >>> custom_collector = TelemetryCollector(max_records=1000)
        >>> traced = instrument_model(model, collector=custom_collector)

get_collector()
    Get the global TelemetryCollector instance.

    Example::

        >>> from insideLLMs.observability import get_collector
        >>>
        >>> collector = get_collector()
        >>> stats = collector.get_stats()
        >>> records = collector.get_records(limit=10)
        >>> print(f"Last 10 calls: {len(records)} records")

set_collector(collector)
    Set a custom TelemetryCollector as the global instance.

    Example::

        >>> from insideLLMs.observability import set_collector, TelemetryCollector
        >>>
        >>> # Create collector with custom settings
        >>> custom = TelemetryCollector(max_records=50000)
        >>> set_collector(custom)
        >>>
        >>> # Now all traced models will use this collector

trace_call(model_name, operation, prompt="", collector=None, metadata=None)
    Context manager for manually tracing a model call.

    Example::

        >>> from insideLLMs.observability import trace_call
        >>>
        >>> # Basic usage
        >>> with trace_call("custom-model", "inference", "Hello world") as ctx:
        ...     # Your model call here
        ...     response = "Model response"
        ...     ctx["response"] = response
        >>>
        >>> # With metadata
        >>> with trace_call(
        ...     "gpt-4",
        ...     "generate",
        ...     prompt="Translate to French",
        ...     metadata={"temperature": 0.7, "max_tokens": 100},
        ... ) as ctx:
        ...     response = model.generate("Translate to French")
        ...     ctx["response"] = response
        >>>
        >>> # Error handling
        >>> try:
        ...     with trace_call("model", "operation") as ctx:
        ...         raise ValueError("Something went wrong")
        ... except ValueError:
        ...     pass  # Error is automatically recorded

trace_function(operation_name=None, include_args=False)
    Decorator for tracing arbitrary function calls.

    Example::

        >>> from insideLLMs.observability import trace_function
        >>>
        >>> # Basic usage
        >>> @trace_function()
        ... def process_document(text: str) -> str:
        ...     # Processing logic
        ...     return text.upper()
        >>>
        >>> result = process_document("hello")
        >>>
        >>> # Custom operation name
        >>> @trace_function(operation_name="text-preprocessing")
        ... def preprocess(text: str) -> str:
        ...     return text.strip().lower()
        >>>
        >>> # Include arguments in trace metadata
        >>> @trace_function(include_args=True)
        ... def expensive_computation(x: int, y: int) -> int:
        ...     return x ** y
        >>>
        >>> result = expensive_computation(2, 10)

setup_otel_tracing(config)
    Initialize OpenTelemetry with the given configuration.

    Example::

        >>> from insideLLMs.observability import setup_otel_tracing, TracingConfig
        >>>
        >>> # Setup with Jaeger
        >>> config = TracingConfig(
        ...     service_name="llm-service",
        ...     jaeger_endpoint="http://localhost:14268/api/traces",
        ... )
        >>> setup_otel_tracing(config)
        >>>
        >>> # Setup with OTLP (OpenTelemetry Protocol)
        >>> config = TracingConfig(
        ...     service_name="llm-service",
        ...     otlp_endpoint="http://localhost:4317",
        ... )
        >>> setup_otel_tracing(config)
        >>>
        >>> # Setup with console output for debugging
        >>> config = TracingConfig(
        ...     service_name="debug-service",
        ...     console_export=True,
        ... )
        >>> setup_otel_tracing(config)

estimate_tokens(text, model="gpt-4")
    Estimate the token count for a given text string.

    Example::

        >>> from insideLLMs.observability import estimate_tokens
        >>>
        >>> text = "Hello, how are you today?"
        >>> tokens = estimate_tokens(text)
        >>> print(f"Estimated tokens: {tokens}")
        >>>
        >>> # Estimate cost based on tokens
        >>> prompt = "Explain the theory of relativity in simple terms."
        >>> response = "Einstein's theory of relativity..."
        >>> input_tokens = estimate_tokens(prompt)
        >>> output_tokens = estimate_tokens(response)
        >>> cost = (input_tokens * 0.00003) + (output_tokens * 0.00006)
        >>> print(f"Estimated cost: ${cost:.4f}")

Exported Constants
------------------
OTEL_AVAILABLE : bool
    Whether OpenTelemetry SDK is installed and available.

    Example::

        >>> from insideLLMs.observability import OTEL_AVAILABLE
        >>>
        >>> if OTEL_AVAILABLE:
        ...     from insideLLMs.observability import OTelTracedModel
        ...     print("OpenTelemetry tracing available")
        ... else:
        ...     from insideLLMs.observability import TracedModel
        ...     print("Using basic tracing (install opentelemetry-sdk for more)")

Complete Example
----------------
Here's a complete example showing how to set up observability for a production
LLM application:

    >>> from insideLLMs.observability import (
    ...     TracingConfig,
    ...     TelemetryCollector,
    ...     instrument_model,
    ...     set_collector,
    ...     get_collector,
    ...     trace_function,
    ...     OTEL_AVAILABLE,
    ... )
    >>> from insideLLMs.models import OpenAIModel
    >>> import logging
    >>>
    >>> # Configure logging
    >>> logging.basicConfig(level=logging.INFO)
    >>>
    >>> # Create a custom collector with alerting
    >>> collector = TelemetryCollector(max_records=10000)
    >>>
    >>> def alert_on_errors(record):
    ...     if not record.success:
    ...         logging.error(
    ...             f"Model call failed: {record.model_name} "
    ...             f"operation={record.operation} error={record.error}"
    ...         )
    ...     if record.latency_ms > 10000:  # 10 seconds
    ...         logging.warning(
    ...             f"Slow model call: {record.model_name} "
    ...             f"latency={record.latency_ms:.0f}ms"
    ...         )
    >>>
    >>> collector.add_callback(alert_on_errors)
    >>> set_collector(collector)
    >>>
    >>> # Configure tracing
    >>> config = TracingConfig(
    ...     service_name="my-llm-app",
    ...     log_prompts=False,  # Privacy: don't log user data
    ...     log_responses=False,
    ...     sample_rate=1.0,  # Trace all requests
    ... )
    >>>
    >>> # Instrument your models
    >>> gpt4 = instrument_model(
    ...     OpenAIModel(model_name="gpt-4"),
    ...     config=config,
    ... )
    >>> gpt35 = instrument_model(
    ...     OpenAIModel(model_name="gpt-3.5-turbo"),
    ...     config=config,
    ... )
    >>>
    >>> # Add tracing to your own functions
    >>> @trace_function(operation_name="document-processing")
    ... def process_document(doc: str) -> str:
    ...     summary = gpt35.generate(f"Summarize: {doc}")
    ...     analysis = gpt4.generate(f"Analyze: {summary}")
    ...     return analysis
    >>>
    >>> # Later, retrieve statistics
    >>> stats = get_collector().get_stats()
    >>> print(f"Total calls: {stats['total_calls']}")
    >>> print(f"Success rate: {stats['success_rate']:.1%}")
    >>> print(f"Avg latency: {stats['avg_latency_ms']:.0f}ms")
    >>> print(f"Total tokens: {stats['total_tokens']}")

Notes
-----
- The basic tracing (TracedModel) works without any external dependencies.
- For distributed tracing, install OpenTelemetry: ``pip install opentelemetry-api opentelemetry-sdk``
- For Jaeger export: ``pip install opentelemetry-exporter-jaeger``
- For OTLP export: ``pip install opentelemetry-exporter-otlp``
- Token estimation is approximate; for precise counts, use the model's tokenizer.
- When ``log_prompts`` or ``log_responses`` is True, sensitive data may be stored.

See Also
--------
insideLLMs.runtime.observability : The underlying implementation module.
insideLLMs.models : Model classes that can be instrumented.
insideLLMs.tokens : Token estimation utilities.

References
----------
- OpenTelemetry Python: https://opentelemetry.io/docs/languages/python/
- Jaeger Tracing: https://www.jaegertracing.io/
- OpenTelemetry Semantic Conventions: https://opentelemetry.io/docs/specs/semconv/
"""

import warnings as _warnings

_warnings.warn(
    "Importing from 'insideLLMs.observability' is deprecated. "
    "Use 'from insideLLMs.runtime.observability import ...' instead. "
    "This shim will be removed in v1.0.",
    DeprecationWarning,
    stacklevel=2,
)

from insideLLMs.runtime.observability import *  # noqa: E402,F401,F403

