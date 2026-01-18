"""OpenTelemetry Observability for LLM Operations.

This module provides comprehensive observability capabilities including:
- Automatic tracing of model calls
- Metrics collection (latency, token usage, success rates)
- Structured logging with context
- Cost tracking and attribution
- Easy integration with observability backends (Jaeger, Prometheus, etc.)

Example:
    >>> from insideLLMs.observability import instrument_model, TracingConfig
    >>> from insideLLMs.models import OpenAIModel
    >>>
    >>> # Basic instrumentation
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> traced_model = instrument_model(model)
    >>>
    >>> # With OpenTelemetry export
    >>> config = TracingConfig(
    ...     service_name="my-llm-app",
    ...     jaeger_endpoint="http://localhost:14268/api/traces"
    ... )
    >>> traced_model = instrument_model(model, config)
    >>> response = traced_model.generate("Hello!")  # Automatically traced
"""

import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from insideLLMs.models.base import Model

# Try to import OpenTelemetry
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
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
    """Configuration for tracing and observability.

    Attributes:
        service_name: Name of the service for tracing.
        enabled: Whether tracing is enabled.
        console_export: Export spans to console (useful for debugging).
        jaeger_endpoint: Jaeger collector endpoint URL.
        otlp_endpoint: OTLP collector endpoint URL.
        log_prompts: Whether to log full prompts (may contain sensitive data).
        log_responses: Whether to log full responses.
        sample_rate: Sampling rate (0.0 to 1.0).
        custom_attributes: Additional attributes to add to all spans.
    """

    service_name: str = "insideLLMs"
    enabled: bool = True
    console_export: bool = False
    jaeger_endpoint: Optional[str] = None
    otlp_endpoint: Optional[str] = None
    log_prompts: bool = False
    log_responses: bool = False
    sample_rate: float = 1.0
    custom_attributes: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# Simple Telemetry (No OpenTelemetry dependency)
# =============================================================================


@dataclass
class CallRecord:
    """Record of a single model call.

    Attributes:
        model_name: Name of the model.
        operation: Type of operation (generate, chat, stream).
        start_time: When the call started.
        end_time: When the call completed.
        latency_ms: Call latency in milliseconds.
        success: Whether the call succeeded.
        error: Error message if failed.
        prompt_tokens: Number of input tokens (estimated).
        completion_tokens: Number of output tokens (estimated).
        prompt_length: Length of prompt in characters.
        response_length: Length of response in characters.
        metadata: Additional metadata.
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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
    """Simple telemetry collector for model calls.

    Collects call records and computes aggregate statistics without
    requiring OpenTelemetry dependencies.
    """

    def __init__(self, max_records: int = 10000):
        """Initialize the collector.

        Args:
            max_records: Maximum number of records to keep.
        """
        self.records: List[CallRecord] = []
        self.max_records = max_records
        self._callbacks: List[Callable[[CallRecord], None]] = []

    def record(self, call: CallRecord) -> None:
        """Record a call.

        Args:
            call: The call record to store.
        """
        self.records.append(call)

        # Trim if over limit
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(call)
            except Exception as e:
                logger.warning(f"Telemetry callback error: {e}")

    def add_callback(self, callback: Callable[[CallRecord], None]) -> None:
        """Add a callback to be called for each record.

        Args:
            callback: Function to call with each CallRecord.
        """
        self._callbacks.append(callback)

    def get_stats(
        self,
        model_name: Optional[str] = None,
        operation: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get aggregate statistics.

        Args:
            model_name: Filter by model name.
            operation: Filter by operation type.
            since: Only include records after this time.

        Returns:
            Dictionary of statistics.
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
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else max(latencies),
            "total_tokens": tokens,
            "avg_tokens_per_call": tokens / total,
        }

    def get_records(
        self,
        limit: int = 100,
        model_name: Optional[str] = None,
    ) -> List[CallRecord]:
        """Get recent records.

        Args:
            limit: Maximum number of records to return.
            model_name: Filter by model name.

        Returns:
            List of recent records.
        """
        filtered = self.records
        if model_name:
            filtered = [r for r in filtered if r.model_name == model_name]
        return filtered[-limit:]

    def clear(self) -> None:
        """Clear all records."""
        self.records.clear()

    def export_json(self) -> str:
        """Export records as JSON.

        Returns:
            JSON string of all records.
        """
        import json
        return json.dumps([r.to_dict() for r in self.records], indent=2)


# Global collector instance
_global_collector: Optional[TelemetryCollector] = None


def get_collector() -> TelemetryCollector:
    """Get the global telemetry collector.

    Returns:
        Global TelemetryCollector instance.
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = TelemetryCollector()
    return _global_collector


def set_collector(collector: TelemetryCollector) -> None:
    """Set the global telemetry collector.

    Args:
        collector: Collector instance to use globally.
    """
    global _global_collector
    _global_collector = collector


# =============================================================================
# Token Estimation
# =============================================================================


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count for text.

    This is a rough estimate. For accurate counts, use the model's
    tokenizer directly.

    Args:
        text: Text to estimate tokens for.
        model: Model name (affects estimation).

    Returns:
        Estimated token count.
    """
    # Rough estimate: ~4 chars per token for English
    # This is a simplification - real tokenizers vary
    if not text:
        return 0

    # Try tiktoken if available (more accurate)
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        pass

    # Fallback: rough character-based estimate
    return max(1, len(text) // 4)


# =============================================================================
# Tracing Context Manager
# =============================================================================


@contextmanager
def trace_call(
    model_name: str,
    operation: str,
    prompt: str = "",
    collector: Optional[TelemetryCollector] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """Context manager for tracing a model call.

    Args:
        model_name: Name of the model.
        operation: Type of operation.
        prompt: The prompt being sent.
        collector: Telemetry collector to use.
        metadata: Additional metadata.

    Yields:
        Context dict for adding response information.

    Example:
        >>> with trace_call("gpt-4", "generate", prompt) as ctx:
        ...     response = model.generate(prompt)
        ...     ctx["response"] = response
    """
    collector = collector or get_collector()
    metadata = metadata or {}

    start_time = datetime.now()
    ctx: Dict[str, Any] = {
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
    """Wrapper that adds tracing to a model.

    All calls to generate(), chat(), and stream() are automatically
    traced and recorded.
    """

    def __init__(
        self,
        model: "Model",
        collector: Optional[TelemetryCollector] = None,
        config: Optional[TracingConfig] = None,
    ):
        """Initialize the traced model wrapper.

        Args:
            model: The model to wrap.
            collector: Telemetry collector (uses global if not provided).
            config: Tracing configuration.
        """
        self._model = model
        self._collector = collector or get_collector()
        self._config = config or TracingConfig()

    @property
    def name(self) -> str:
        return self._model.name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate with tracing.

        Args:
            prompt: The prompt.
            **kwargs: Model arguments.

        Returns:
            Generated response.
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

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Chat with tracing.

        Args:
            messages: Chat messages.
            **kwargs: Model arguments.

        Returns:
            Generated response.
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
        """Stream with tracing.

        Args:
            prompt: The prompt.
            **kwargs: Model arguments.

        Yields:
            Response chunks.
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

    def info(self):
        """Return model info."""
        return self._model.info()

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to wrapped model."""
        return getattr(self._model, name)


def instrument_model(
    model: "Model",
    config: Optional[TracingConfig] = None,
    collector: Optional[TelemetryCollector] = None,
) -> TracedModel:
    """Instrument a model with tracing.

    Args:
        model: Model to instrument.
        config: Tracing configuration.
        collector: Telemetry collector.

    Returns:
        TracedModel wrapper.

    Example:
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> traced = instrument_model(model)
        >>> response = traced.generate("Hello")  # Automatically traced
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

    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: config.service_name,
        **config.custom_attributes,
    })

    provider = TracerProvider(resource=resource)

    if config.console_export:
        provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )

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

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
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

    def info(self):
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
