"""Tests for OpenTelemetry Observability module."""

import json
from datetime import datetime, timedelta

import pytest

from insideLLMs.models import DummyModel
from insideLLMs.observability import (
    OTEL_AVAILABLE,
    CallRecord,
    TelemetryCollector,
    TracedModel,
    TracingConfig,
    estimate_tokens,
    get_collector,
    instrument_model,
    set_collector,
    trace_call,
    trace_function,
)

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
            collector.record(
                CallRecord(
                    model_name="test",
                    operation="generate",
                    start_time=now,
                    end_time=now,
                    latency_ms=100.0 + i * 10,  # 100, 110, 120, 130, 140
                    success=True,
                    prompt_tokens=10,
                    completion_tokens=20,
                )
            )

        # Add a failure
        collector.record(
            CallRecord(
                model_name="test",
                operation="generate",
                start_time=now,
                end_time=now,
                latency_ms=50.0,
                success=False,
                error="Error",
            )
        )

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
            collector.record(
                CallRecord(
                    model_name=model,
                    operation="generate",
                    start_time=now,
                    end_time=now,
                    latency_ms=100.0,
                    success=True,
                )
            )

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
        collector.record(
            CallRecord(
                model_name="test",
                operation="generate",
                start_time=now,
                end_time=now,
                latency_ms=100.0,
                success=True,
            )
        )

        assert len(callback_records) == 1
        assert callback_records[0].model_name == "test"

    def test_clear(self):
        """Test clearing records."""
        collector = TelemetryCollector()
        now = datetime.now()

        collector.record(
            CallRecord(
                model_name="test",
                operation="generate",
                start_time=now,
                end_time=now,
                latency_ms=100.0,
                success=True,
            )
        )

        assert len(collector.get_records()) == 1
        collector.clear()
        assert len(collector.get_records()) == 0

    def test_export_json(self):
        """Test JSON export."""
        collector = TelemetryCollector()
        now = datetime.now()

        collector.record(
            CallRecord(
                model_name="test",
                operation="generate",
                start_time=now,
                end_time=now,
                latency_ms=100.0,
                success=True,
            )
        )

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
            with trace_call("test-model", "generate", "Hello", collector):
                raise ValueError("Test error")

        records = collector.get_records()
        assert len(records) == 1
        assert records[0].success is False
        assert "Test error" in records[0].error

    def test_trace_call_with_metadata(self):
        """Test call tracing with metadata."""
        collector = TelemetryCollector()

        with trace_call(
            "test-model", "generate", "Hello", collector, metadata={"user_id": "123"}
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

        traced.generate("Secret prompt")

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


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestTelemetryCollectorExtended:
    """Extended tests for TelemetryCollector to improve coverage."""

    def test_callback_error_handling(self):
        """Test that callback errors don't break the collector."""
        collector = TelemetryCollector()
        now = datetime.now()

        def failing_callback(record):
            raise ValueError("Callback failed!")

        collector.add_callback(failing_callback)

        # Should not raise despite callback error
        record = CallRecord(
            model_name="test",
            operation="generate",
            start_time=now,
            end_time=now,
            latency_ms=100.0,
            success=True,
        )
        collector.record(record)

        # Record should still be stored
        assert len(collector.get_records()) == 1

    def test_get_stats_by_operation(self):
        """Test filtering stats by operation type."""
        collector = TelemetryCollector()
        now = datetime.now()

        # Add records with different operations
        for op in ["generate", "generate", "chat"]:
            collector.record(
                CallRecord(
                    model_name="test",
                    operation=op,
                    start_time=now,
                    end_time=now,
                    latency_ms=100.0,
                    success=True,
                )
            )

        stats = collector.get_stats(operation="generate")
        assert stats["total_calls"] == 2

        stats = collector.get_stats(operation="chat")
        assert stats["total_calls"] == 1

    def test_get_stats_by_since(self):
        """Test filtering stats by time."""
        collector = TelemetryCollector()
        now = datetime.now()
        old_time = now - timedelta(hours=2)

        # Add an old record
        collector.record(
            CallRecord(
                model_name="test",
                operation="generate",
                start_time=old_time,
                end_time=old_time,
                latency_ms=100.0,
                success=True,
            )
        )

        # Add a recent record
        collector.record(
            CallRecord(
                model_name="test",
                operation="generate",
                start_time=now,
                end_time=now,
                latency_ms=100.0,
                success=True,
            )
        )

        # Filter by since
        since_time = now - timedelta(hours=1)
        stats = collector.get_stats(since=since_time)
        assert stats["total_calls"] == 1

    def test_get_stats_empty(self):
        """Test stats when collector is empty or filtered to empty."""
        collector = TelemetryCollector()

        # Empty collector
        stats = collector.get_stats()
        assert stats["total_calls"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["avg_latency_ms"] == 0.0

        # Add record but filter by non-matching model
        now = datetime.now()
        collector.record(
            CallRecord(
                model_name="gpt-4",
                operation="generate",
                start_time=now,
                end_time=now,
                latency_ms=100.0,
                success=True,
            )
        )
        stats = collector.get_stats(model_name="claude")
        assert stats["total_calls"] == 0

    def test_get_records_with_model_filter(self):
        """Test get_records with model name filter."""
        collector = TelemetryCollector()
        now = datetime.now()

        for model in ["gpt-4", "gpt-4", "claude"]:
            collector.record(
                CallRecord(
                    model_name=model,
                    operation="generate",
                    start_time=now,
                    end_time=now,
                    latency_ms=100.0,
                    success=True,
                )
            )

        records = collector.get_records(model_name="gpt-4")
        assert len(records) == 2
        assert all(r.model_name == "gpt-4" for r in records)


class TestTokenEstimationExtended:
    """Extended token estimation tests."""

    def test_estimate_tokens_with_invalid_model(self):
        """Test token estimation with unknown model falls back gracefully."""
        text = "Hello world"
        tokens = estimate_tokens(text, model="nonexistent-model-xyz")
        assert tokens > 0


class TestTracedModelExtended:
    """Extended TracedModel tests for coverage."""

    def test_traced_model_stream_error(self):
        """Test traced model stream handles errors."""
        from insideLLMs.models.base import Model

        class StreamErrorModel(Model):
            def __init__(self):
                super().__init__(name="stream-error")

            def generate(self, prompt, **kwargs):
                return "test"

            def stream(self, prompt, **kwargs):
                yield "chunk1"
                raise ValueError("Stream error")

        collector = TelemetryCollector()
        model = StreamErrorModel()
        traced = TracedModel(model, collector)

        chunks = []
        with pytest.raises(ValueError, match="Stream error"):
            for chunk in traced.stream("Hello"):
                chunks.append(chunk)

        assert chunks == ["chunk1"]
        records = collector.get_records()
        assert len(records) == 1
        assert records[0].success is False
        assert "Stream error" in records[0].error

    def test_traced_model_chat_not_supported(self):
        """Test traced model chat when model doesn't support it."""
        from unittest.mock import MagicMock

        mock_model = MagicMock(spec=[])  # Empty spec = no attributes
        mock_model.name = "test-model"

        collector = TelemetryCollector()
        traced = TracedModel(mock_model, collector)

        with pytest.raises(NotImplementedError, match="does not support chat"):
            traced.chat([{"role": "user", "content": "Hello"}])

    def test_traced_model_stream_not_supported(self):
        """Test traced model stream when model doesn't support it."""
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_model.name = "test-model"
        del mock_model.stream  # Remove stream attribute

        collector = TelemetryCollector()
        traced = TracedModel(mock_model, collector)

        with pytest.raises(NotImplementedError, match="does not support streaming"):
            list(traced.stream("Hello"))

    def test_traced_model_info(self):
        """Test traced model info method."""
        model = DummyModel()
        traced = TracedModel(model)

        info = traced.info()
        assert info is not None

    def test_traced_model_attribute_delegation(self):
        """Test that traced model delegates unknown attributes."""
        model = DummyModel()
        model.custom_attr = "custom_value"
        traced = TracedModel(model)

        assert traced.custom_attr == "custom_value"


class TestTraceFunctionExtended:
    """Extended trace_function tests."""

    def test_trace_function_with_args(self):
        """Test function tracing with include_args=True."""
        collector = TelemetryCollector()
        set_collector(collector)

        @trace_function(include_args=True)
        def function_with_args(x, y, z=10):
            return x + y + z

        result = function_with_args(1, 2, z=3)

        assert result == 6
        records = collector.get_records()
        assert len(records) >= 1
        # Check metadata contains args
        last_record = records[-1]
        assert "args" in last_record.metadata
        assert "kwargs" in last_record.metadata


class TestOpenTelemetrySetup:
    """Tests for OpenTelemetry setup functions."""

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
    def test_setup_otel_tracing(self):
        """Test setting up OpenTelemetry tracing."""
        from insideLLMs.observability import setup_otel_tracing

        config = TracingConfig(
            service_name="test-service",
            console_export=True,
        )
        # Should not raise
        setup_otel_tracing(config)

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
    def test_otel_traced_model_generate(self):
        """Test OTelTracedModel generate."""
        from insideLLMs.observability import OTelTracedModel

        model = DummyModel()
        traced = OTelTracedModel(model)

        response = traced.generate("Hello")
        assert response is not None

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
    def test_otel_traced_model_generate_with_prompts(self):
        """Test OTelTracedModel generate with log_prompts enabled."""
        from insideLLMs.observability import OTelTracedModel

        config = TracingConfig(log_prompts=True, log_responses=True)
        model = DummyModel()
        traced = OTelTracedModel(model, config)

        response = traced.generate("Hello")
        assert response is not None

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
    def test_otel_traced_model_generate_error(self):
        """Test OTelTracedModel generate error handling."""
        from insideLLMs.models.base import Model
        from insideLLMs.observability import OTelTracedModel

        class FailingModel(Model):
            def __init__(self):
                super().__init__(name="failing")

            def generate(self, prompt, **kwargs):
                raise ValueError("Generate failed")

        model = FailingModel()
        traced = OTelTracedModel(model)

        with pytest.raises(ValueError, match="Generate failed"):
            traced.generate("Hello")

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
    def test_otel_traced_model_chat(self):
        """Test OTelTracedModel chat."""
        from insideLLMs.observability import OTelTracedModel

        model = DummyModel()
        traced = OTelTracedModel(model)

        response = traced.chat([{"role": "user", "content": "Hello"}])
        assert response is not None

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
    def test_otel_traced_model_chat_not_supported(self):
        """Test OTelTracedModel chat when not supported."""
        from unittest.mock import MagicMock

        from insideLLMs.observability import OTelTracedModel

        mock_model = MagicMock()
        mock_model.name = "test-model"
        del mock_model.chat

        traced = OTelTracedModel(mock_model)

        with pytest.raises(NotImplementedError, match="does not support chat"):
            traced.chat([{"role": "user", "content": "Hello"}])

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
    def test_otel_traced_model_chat_error(self):
        """Test OTelTracedModel chat error handling."""
        from insideLLMs.models.base import Model
        from insideLLMs.observability import OTelTracedModel

        class FailingChatModel(Model):
            def __init__(self):
                super().__init__(name="failing-chat")

            def generate(self, prompt, **kwargs):
                return "test"

            def chat(self, messages, **kwargs):
                raise ValueError("Chat failed")

        model = FailingChatModel()
        traced = OTelTracedModel(model)

        with pytest.raises(ValueError, match="Chat failed"):
            traced.chat([{"role": "user", "content": "Hello"}])

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
    def test_otel_traced_model_info(self):
        """Test OTelTracedModel info method."""
        from insideLLMs.observability import OTelTracedModel

        model = DummyModel()
        traced = OTelTracedModel(model)

        info = traced.info()
        assert info is not None

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
    def test_otel_traced_model_attribute_delegation(self):
        """Test OTelTracedModel attribute delegation."""
        from insideLLMs.observability import OTelTracedModel

        model = DummyModel()
        model.custom_attr = "custom_value"
        traced = OTelTracedModel(model)

        assert traced.custom_attr == "custom_value"

    def test_setup_otel_without_package(self):
        """Test setup_otel_tracing raises when OTEL not installed."""
        # This test relies on OTEL not being installed in some envs
        # For now, just test the import error path is covered
        if not OTEL_AVAILABLE:
            from insideLLMs.observability import setup_otel_tracing

            config = TracingConfig()
            with pytest.raises(ImportError, match="OpenTelemetry is required"):
                setup_otel_tracing(config)

    def test_otel_traced_model_without_package(self):
        """Test OTelTracedModel raises when OTEL not installed."""
        if not OTEL_AVAILABLE:
            from insideLLMs.observability import OTelTracedModel

            model = DummyModel()
            with pytest.raises(ImportError, match="OpenTelemetry is required"):
                OTelTracedModel(model)
