"""Tests for OpenTelemetry Observability module."""

import json
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from insideLLMs.observability import (
    TracingConfig,
    CallRecord,
    TelemetryCollector,
    get_collector,
    set_collector,
    trace_call,
    trace_function,
    TracedModel,
    instrument_model,
    estimate_tokens,
    OTEL_AVAILABLE,
)
from insideLLMs.models import DummyModel


# =============================================================================
# Test TracingConfig
# =============================================================================


class TestTracingConfig:
    """Tests for TracingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TracingConfig()

        assert config.service_name == "insideLLMs"
        assert config.enabled is True
        assert config.console_export is False
        assert config.jaeger_endpoint is None
        assert config.log_prompts is False
        assert config.log_responses is False
        assert config.sample_rate == 1.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = TracingConfig(
            service_name="my-service",
            enabled=False,
            console_export=True,
            jaeger_endpoint="http://localhost:14268",
            log_prompts=True,
            log_responses=True,
            sample_rate=0.5,
            custom_attributes={"env": "test"},
        )

        assert config.service_name == "my-service"
        assert config.enabled is False
        assert config.console_export is True
        assert config.jaeger_endpoint == "http://localhost:14268"
        assert config.log_prompts is True
        assert config.sample_rate == 0.5
        assert config.custom_attributes == {"env": "test"}


# =============================================================================
# Test CallRecord
# =============================================================================


class TestCallRecord:
    """Tests for CallRecord dataclass."""

    def test_basic_record(self):
        """Test basic call record creation."""
        now = datetime.now()
        record = CallRecord(
            model_name="gpt-4",
            operation="generate",
            start_time=now,
            end_time=now + timedelta(milliseconds=100),
            latency_ms=100.0,
            success=True,
        )

        assert record.model_name == "gpt-4"
        assert record.operation == "generate"
        assert record.latency_ms == 100.0
        assert record.success is True
        assert record.error is None

    def test_record_with_tokens(self):
        """Test record with token counts."""
        now = datetime.now()
        record = CallRecord(
            model_name="gpt-4",
            operation="generate",
            start_time=now,
            end_time=now,
            latency_ms=50.0,
            success=True,
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert record.prompt_tokens == 100
        assert record.completion_tokens == 50

    def test_record_to_dict(self):
        """Test record serialization."""
        now = datetime.now()
        record = CallRecord(
            model_name="test-model",
            operation="chat",
            start_time=now,
            end_time=now + timedelta(seconds=1),
            latency_ms=1000.0,
            success=True,
            prompt_tokens=10,
            completion_tokens=20,
            metadata={"key": "value"},
        )

        data = record.to_dict()

        assert data["model_name"] == "test-model"
        assert data["operation"] == "chat"
        assert data["latency_ms"] == 1000.0
        assert data["success"] is True
        assert data["total_tokens"] == 30
        assert data["metadata"] == {"key": "value"}

    def test_record_with_error(self):
        """Test record with error."""
        now = datetime.now()
        record = CallRecord(
            model_name="gpt-4",
            operation="generate",
            start_time=now,
            end_time=now,
            latency_ms=10.0,
            success=False,
            error="API Error: Rate limited",
        )

        assert record.success is False
        assert record.error == "API Error: Rate limited"


# =============================================================================
# Test TelemetryCollector
# =============================================================================


class TestTelemetryCollector:
    """Tests for TelemetryCollector class."""

    def test_record_and_get_records(self):
        """Test recording and retrieving records."""
        collector = TelemetryCollector()
        now = datetime.now()

        record = CallRecord(
            model_name="test",
            operation="generate",
            start_time=now,
            end_time=now,
            latency_ms=100.0,
            success=True,
        )

        collector.record(record)
        records = collector.get_records()

        assert len(records) == 1
        assert records[0].model_name == "test"

    def test_max_records_limit(self):
        """Test that collector respects max records limit."""
        collector = TelemetryCollector(max_records=10)
        now = datetime.now()

        for i in range(20):
            record = CallRecord(
                model_name=f"model-{i}",
                operation="generate",
                start_time=now,
                end_time=now,
                latency_ms=100.0,
                success=True,
            )
            collector.record(record)

        records = collector.get_records(limit=100)
        assert len(records) == 10
        # Should keep most recent records
        assert records[-1].model_name == "model-19"

    def test_get_stats(self):
        """Test statistics calculation."""
        collector = TelemetryCollector()
        now = datetime.now()

        # Add some records
        for i in range(5):
            collector.record(CallRecord(
                model_name="test",
                operation="generate",
                start_time=now,
                end_time=now,
                latency_ms=100.0 + i * 10,  # 100, 110, 120, 130, 140
                success=True,
                prompt_tokens=10,
                completion_tokens=20,
            ))

        # Add a failure
        collector.record(CallRecord(
            model_name="test",
            operation="generate",
            start_time=now,
            end_time=now,
            latency_ms=50.0,
            success=False,
            error="Error",
        ))

        stats = collector.get_stats()

        assert stats["total_calls"] == 6
        assert stats["successes"] == 5
        assert stats["failures"] == 1
        assert stats["success_rate"] == 5 / 6
        assert stats["total_tokens"] == 150  # 5 * (10 + 20)

    def test_get_stats_filtered(self):
        """Test filtered statistics."""
        collector = TelemetryCollector()
        now = datetime.now()

        for model in ["gpt-4", "gpt-4", "claude"]:
            collector.record(CallRecord(
                model_name=model,
                operation="generate",
                start_time=now,
                end_time=now,
                latency_ms=100.0,
                success=True,
            ))

        stats = collector.get_stats(model_name="gpt-4")
        assert stats["total_calls"] == 2

        stats = collector.get_stats(model_name="claude")
        assert stats["total_calls"] == 1

    def test_callback(self):
        """Test callback functionality."""
        collector = TelemetryCollector()
        callback_records = []

        def callback(record: CallRecord):
            callback_records.append(record)

        collector.add_callback(callback)

        now = datetime.now()
        collector.record(CallRecord(
            model_name="test",
            operation="generate",
            start_time=now,
            end_time=now,
            latency_ms=100.0,
            success=True,
        ))

        assert len(callback_records) == 1
        assert callback_records[0].model_name == "test"

    def test_clear(self):
        """Test clearing records."""
        collector = TelemetryCollector()
        now = datetime.now()

        collector.record(CallRecord(
            model_name="test",
            operation="generate",
            start_time=now,
            end_time=now,
            latency_ms=100.0,
            success=True,
        ))

        assert len(collector.get_records()) == 1
        collector.clear()
        assert len(collector.get_records()) == 0

    def test_export_json(self):
        """Test JSON export."""
        collector = TelemetryCollector()
        now = datetime.now()

        collector.record(CallRecord(
            model_name="test",
            operation="generate",
            start_time=now,
            end_time=now,
            latency_ms=100.0,
            success=True,
        ))

        json_str = collector.export_json()
        data = json.loads(json_str)

        assert len(data) == 1
        assert data[0]["model_name"] == "test"


# =============================================================================
# Test Global Collector
# =============================================================================


class TestGlobalCollector:
    """Tests for global collector functions."""

    def test_get_collector_creates_instance(self):
        """Test that get_collector creates a singleton."""
        collector1 = get_collector()
        collector2 = get_collector()
        assert collector1 is collector2

    def test_set_collector(self):
        """Test setting custom collector."""
        original = get_collector()
        custom = TelemetryCollector(max_records=5)

        set_collector(custom)
        assert get_collector() is custom

        # Restore
        set_collector(original)


# =============================================================================
# Test Token Estimation
# =============================================================================


class TestTokenEstimation:
    """Tests for token estimation."""

    def test_estimate_tokens_empty(self):
        """Test empty string returns 0."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_basic(self):
        """Test basic token estimation."""
        text = "Hello, world! This is a test."
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens < len(text)  # Should be fewer tokens than chars

    def test_estimate_tokens_long_text(self):
        """Test token estimation for longer text."""
        text = "word " * 100  # 100 words
        tokens = estimate_tokens(text)
        assert tokens > 50  # Should be reasonable for 100 words


# =============================================================================
# Test trace_call Context Manager
# =============================================================================


class TestTraceCall:
    """Tests for trace_call context manager."""

    def test_trace_call_success(self):
        """Test successful call tracing."""
        collector = TelemetryCollector()

        with trace_call("test-model", "generate", "Hello", collector) as ctx:
            ctx["response"] = "Hi there!"

        records = collector.get_records()
        assert len(records) == 1
        assert records[0].model_name == "test-model"
        assert records[0].operation == "generate"
        assert records[0].success is True

    def test_trace_call_failure(self):
        """Test failed call tracing."""
        collector = TelemetryCollector()

        with pytest.raises(ValueError):
            with trace_call("test-model", "generate", "Hello", collector) as ctx:
                raise ValueError("Test error")

        records = collector.get_records()
        assert len(records) == 1
        assert records[0].success is False
        assert "Test error" in records[0].error

    def test_trace_call_with_metadata(self):
        """Test call tracing with metadata."""
        collector = TelemetryCollector()

        with trace_call(
            "test-model",
            "generate",
            "Hello",
            collector,
            metadata={"user_id": "123"}
        ) as ctx:
            ctx["response"] = "Hi"

        records = collector.get_records()
        assert records[0].metadata == {"user_id": "123"}


# =============================================================================
# Test trace_function Decorator
# =============================================================================


class TestTraceFunction:
    """Tests for trace_function decorator."""

    def test_trace_function_basic(self):
        """Test basic function tracing."""
        collector = TelemetryCollector()
        set_collector(collector)

        @trace_function()
        def my_function(x):
            return x * 2

        result = my_function(5)

        assert result == 10
        records = collector.get_records()
        assert len(records) == 1
        assert records[0].operation == "my_function"
        assert records[0].success is True

    def test_trace_function_with_name(self):
        """Test function tracing with custom name."""
        collector = TelemetryCollector()
        set_collector(collector)

        @trace_function(operation_name="custom_op")
        def another_function():
            return True

        another_function()

        records = collector.get_records()
        assert records[-1].operation == "custom_op"

    def test_trace_function_error(self):
        """Test function tracing with error."""
        collector = TelemetryCollector()
        set_collector(collector)

        @trace_function()
        def failing_function():
            raise RuntimeError("Failed!")

        with pytest.raises(RuntimeError):
            failing_function()

        records = collector.get_records()
        assert records[-1].success is False
        assert "Failed!" in records[-1].error


# =============================================================================
# Test TracedModel
# =============================================================================


class TestTracedModel:
    """Tests for TracedModel wrapper."""

    def test_traced_model_generate(self):
        """Test traced model generate."""
        collector = TelemetryCollector()
        model = DummyModel()
        traced = TracedModel(model, collector)

        response = traced.generate("Hello")

        assert "Hello" in response
        records = collector.get_records()
        assert len(records) == 1
        assert records[0].operation == "generate"
        assert records[0].success is True

    def test_traced_model_chat(self):
        """Test traced model chat."""
        collector = TelemetryCollector()
        model = DummyModel()
        traced = TracedModel(model, collector)

        messages = [{"role": "user", "content": "Hello"}]
        response = traced.chat(messages)

        assert response  # DummyModel returns something
        records = collector.get_records()
        assert len(records) == 1
        assert records[0].operation == "chat"

    def test_traced_model_stream(self):
        """Test traced model stream."""
        collector = TelemetryCollector()
        model = DummyModel()
        traced = TracedModel(model, collector)

        chunks = list(traced.stream("Hello"))

        assert len(chunks) > 0
        records = collector.get_records()
        assert len(records) == 1
        assert records[0].operation == "stream"

    def test_traced_model_preserves_attributes(self):
        """Test that traced model preserves original attributes."""
        model = DummyModel(name="TestModel")
        traced = TracedModel(model)

        assert traced.name == "TestModel"

    def test_traced_model_with_config(self):
        """Test traced model with config."""
        collector = TelemetryCollector()
        config = TracingConfig(log_prompts=True, log_responses=True)
        model = DummyModel()
        traced = TracedModel(model, collector, config)

        response = traced.generate("Secret prompt")

        # Config is stored but prompts/responses not stored in basic CallRecord
        assert traced._config.log_prompts is True


# =============================================================================
# Test instrument_model
# =============================================================================


class TestInstrumentModel:
    """Tests for instrument_model function."""

    def test_instrument_model_basic(self):
        """Test basic model instrumentation."""
        model = DummyModel()
        traced = instrument_model(model)

        assert isinstance(traced, TracedModel)
        response = traced.generate("Test")
        assert response

    def test_instrument_model_with_config(self):
        """Test instrumentation with config."""
        model = DummyModel()
        config = TracingConfig(service_name="test-service")
        traced = instrument_model(model, config=config)

        assert traced._config.service_name == "test-service"

    def test_instrument_model_with_collector(self):
        """Test instrumentation with custom collector."""
        model = DummyModel()
        collector = TelemetryCollector(max_records=5)
        traced = instrument_model(model, collector=collector)

        traced.generate("Test")
        assert len(collector.get_records()) == 1


# =============================================================================
# Test OpenTelemetry Integration
# =============================================================================


@pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
class TestOpenTelemetryIntegration:
    """Tests for OpenTelemetry integration (requires otel packages)."""

    def test_otel_available(self):
        """Test that OTEL_AVAILABLE is correct."""
        assert OTEL_AVAILABLE is True

    def test_otel_traced_model_import(self):
        """Test that OTelTracedModel can be imported."""
        from insideLLMs.observability import OTelTracedModel
        assert OTelTracedModel is not None


class TestOTelNotInstalled:
    """Tests when OpenTelemetry is not installed."""

    def test_otel_availability_flag(self):
        """Test OTEL_AVAILABLE flag is boolean."""
        assert isinstance(OTEL_AVAILABLE, bool)
