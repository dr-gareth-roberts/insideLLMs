"""Supplemental tests to increase coverage for adapters.py and analysis/export.py.

Targets uncovered branches and statements identified by coverage analysis:
- adapters.py: OpenAI/Anthropic adapters (mocked), FallbackChain.generate_chat,
  AdapterPool random strategy, create_adapter/create_fallback_chain/create_adapter_pool
  convenience functions, ModelRegistry edge cases, AdapterFactory provider detection, etc.
- analysis/export.py: serialize_value (to_dict objects), serialize_record (fallback
  paths), _prepare_data (strings, non-iterables), file-like object exports,
  DataArchiver bzip2/decompress paths, ExportPipeline scalar inputs, stream_export
  non-JSONL, create_export_bundle with dict data and schema inference, etc.
"""

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# adapters imports
# ---------------------------------------------------------------------------
from insideLLMs.adapters import (
    AdapterConfig,
    AdapterFactory,
    AdapterPool,
    AdapterStatus,
    AnthropicAdapter,
    BaseAdapter,
    ConnectionInfo,
    ConnectionMonitor,
    FallbackChain,
    GenerationParams,
    GenerationResult,
    HealthCheckResult,
    MockAdapter,
    ModelCapability,
    ModelRegistry,
    OpenAIAdapter,
    Provider,
    ProviderDetector,
    create_adapter,
    create_adapter_pool,
    create_fallback_chain,
    create_mock_adapter,
)

# ---------------------------------------------------------------------------
# export imports
# ---------------------------------------------------------------------------
from insideLLMs.analysis.export import (
    CompressionType,
    CSVExporter,
    DataArchiver,
    DataSchema,
    ExportConfig,
    ExportFormat,
    ExportMetadata,
    ExportPipeline,
    JSONExporter,
    JSONLExporter,
    MarkdownExporter,
    SchemaField,
    create_export_bundle,
    export_to_csv,
    export_to_json,
    export_to_jsonl,
    export_to_markdown,
    get_exporter,
    serialize_record,
    serialize_value,
    stream_export,
)

# ============================================================================
# ADAPTERS COVERAGE
# ============================================================================


class TestHealthCheckResultToDict:
    """Cover HealthCheckResult.to_dict (line 1400)."""

    def test_to_dict_healthy(self):
        hc = HealthCheckResult(healthy=True, latency_ms=1.0)
        d = hc.to_dict()
        assert d["healthy"] is True
        assert d["latency_ms"] == 1.0
        assert d["error"] is None
        assert "timestamp" in d

    def test_to_dict_unhealthy(self):
        hc = HealthCheckResult(healthy=False, latency_ms=99.0, error="timeout")
        d = hc.to_dict()
        assert d["healthy"] is False
        assert d["error"] == "timeout"


class TestBaseAdapterHealthCheckError:
    """Cover health_check exception path (lines 1801-1803) and _record_error (1932)."""

    def test_health_check_returns_unhealthy_on_exception(self):
        adapter = create_mock_adapter()

        # Make generate raise to trigger the except branch in health_check
        def raise_error(*a, **kw):
            raise RuntimeError("boom")

        adapter.generate = raise_error
        result = adapter.health_check()
        assert result.healthy is False
        assert "boom" in result.error

    def test_record_error_increments_count(self):
        adapter = create_mock_adapter()
        assert adapter.get_stats()["error_count"] == 0
        adapter._record_error()
        assert adapter.get_stats()["error_count"] == 1


class TestOpenAIAdapterMocked:
    """Cover OpenAIAdapter init, generate, generate_chat with mocked openai."""

    def test_init_success(self):
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            config = AdapterConfig(model_id="gpt-4o", provider=Provider.OPENAI, api_key="sk-test")
            adapter = OpenAIAdapter(config)
            assert adapter.status == AdapterStatus.READY

    def test_init_import_error(self):
        """When openai is not importable, status = ERROR."""
        with patch("builtins.__import__", side_effect=ImportError("no openai")):
            config = AdapterConfig(model_id="gpt-4o", provider=Provider.OPENAI)
            adapter = OpenAIAdapter.__new__(OpenAIAdapter)
            adapter.config = config
            adapter._status = AdapterStatus.INITIALIZING
            adapter._last_request_time = 0
            adapter._request_count = 0
            adapter._error_count = 0
            adapter._client = None
            adapter._initialize_client()
            assert adapter.status == AdapterStatus.ERROR

    def test_generate_delegates_to_generate_chat(self):
        config = AdapterConfig(model_id="gpt-4o", provider=Provider.OPENAI, api_key="sk-test")
        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        adapter.config = config
        adapter._status = AdapterStatus.READY
        adapter._last_request_time = 0
        adapter._request_count = 0
        adapter._error_count = 0

        # Mock the _client to return a proper response
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello back"
        mock_choice.finish_reason = "stop"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 3
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        adapter._client = MagicMock()
        adapter._client.chat.completions.create.return_value = mock_response

        result = adapter.generate("Hello")
        assert result.text == "Hello back"
        assert result.provider == "openai"
        assert result.tokens_input == 5

    def test_generate_chat_error(self):
        config = AdapterConfig(model_id="gpt-4o", provider=Provider.OPENAI, api_key="sk-test")
        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        adapter.config = config
        adapter._status = AdapterStatus.READY
        adapter._last_request_time = 0
        adapter._request_count = 0
        adapter._error_count = 0

        adapter._client = MagicMock()
        adapter._client.chat.completions.create.side_effect = Exception("API error")

        with pytest.raises(RuntimeError, match="OpenAI API error"):
            adapter.generate_chat([{"role": "user", "content": "Hi"}])
        assert adapter._error_count == 1

    def test_generate_chat_no_usage(self):
        """Cover the `if response.usage else 0` branches."""
        config = AdapterConfig(model_id="gpt-4o", provider=Provider.OPENAI, api_key="sk-test")
        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        adapter.config = config
        adapter._status = AdapterStatus.READY
        adapter._last_request_time = 0
        adapter._request_count = 0
        adapter._error_count = 0

        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        adapter._client = MagicMock()
        adapter._client.chat.completions.create.return_value = mock_response

        result = adapter.generate_chat([{"role": "user", "content": "Hello"}])
        assert result.tokens_input == 0
        assert result.tokens_output == 0


class TestAnthropicAdapterMocked:
    """Cover AnthropicAdapter init, generate, generate_chat with mocked anthropic."""

    def test_init_success(self):
        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            config = AdapterConfig(
                model_id="claude-3-opus", provider=Provider.ANTHROPIC, api_key="sk-ant-test"
            )
            adapter = AnthropicAdapter(config)
            assert adapter.status == AdapterStatus.READY

    def test_init_import_error(self):
        config = AdapterConfig(
            model_id="claude-3-opus", provider=Provider.ANTHROPIC, api_key="sk-ant-test"
        )
        adapter = AnthropicAdapter.__new__(AnthropicAdapter)
        adapter.config = config
        adapter._status = AdapterStatus.INITIALIZING
        adapter._last_request_time = 0
        adapter._request_count = 0
        adapter._error_count = 0
        adapter._client = None
        with patch("builtins.__import__", side_effect=ImportError("no anthropic")):
            adapter._initialize_client()
        assert adapter.status == AdapterStatus.ERROR

    def test_generate_and_generate_chat(self):
        config = AdapterConfig(
            model_id="claude-3-opus", provider=Provider.ANTHROPIC, api_key="sk-ant-test"
        )
        adapter = AnthropicAdapter.__new__(AnthropicAdapter)
        adapter.config = config
        adapter._status = AdapterStatus.READY
        adapter._last_request_time = 0
        adapter._request_count = 0
        adapter._error_count = 0

        # Build mock response
        mock_content = MagicMock()
        mock_content.text = "Claude says hi"
        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_response.usage = mock_usage
        mock_response.stop_reason = "end_turn"

        adapter._client = MagicMock()
        adapter._client.messages.create.return_value = mock_response

        result = adapter.generate("Hello")
        assert result.text == "Claude says hi"
        assert result.provider == "anthropic"
        assert result.tokens_input == 10

    def test_generate_chat_error(self):
        config = AdapterConfig(
            model_id="claude-3-opus", provider=Provider.ANTHROPIC, api_key="sk-ant-test"
        )
        adapter = AnthropicAdapter.__new__(AnthropicAdapter)
        adapter.config = config
        adapter._status = AdapterStatus.READY
        adapter._last_request_time = 0
        adapter._request_count = 0
        adapter._error_count = 0
        adapter._client = MagicMock()
        adapter._client.messages.create.side_effect = Exception("Anthropic error")

        with pytest.raises(RuntimeError, match="Anthropic API error"):
            adapter.generate_chat([{"role": "user", "content": "Hi"}])

    def test_generate_chat_empty_content(self):
        config = AdapterConfig(
            model_id="claude-3-opus", provider=Provider.ANTHROPIC, api_key="sk-ant-test"
        )
        adapter = AnthropicAdapter.__new__(AnthropicAdapter)
        adapter.config = config
        adapter._status = AdapterStatus.READY
        adapter._last_request_time = 0
        adapter._request_count = 0
        adapter._error_count = 0

        mock_usage = MagicMock()
        mock_usage.input_tokens = 2
        mock_usage.output_tokens = 0
        mock_response = MagicMock()
        mock_response.content = []  # empty content
        mock_response.usage = mock_usage
        mock_response.stop_reason = "stop"

        adapter._client = MagicMock()
        adapter._client.messages.create.return_value = mock_response

        result = adapter.generate_chat([{"role": "user", "content": "Hi"}])
        assert result.text == ""


class TestModelRegistryEdgeCases:
    """Cover ModelRegistry.get returning None (line 2800)."""

    def test_get_nonexistent_model(self):
        registry = ModelRegistry()
        assert registry.get("nonexistent-model-xyz") is None

    def test_get_nonexistent_alias(self):
        registry = ModelRegistry()
        assert registry.get("unknown-alias-xyz") is None

    def test_resolve_alias_passthrough(self):
        registry = ModelRegistry()
        assert registry.resolve_alias("not-an-alias") == "not-an-alias"


class TestAdapterFactoryProviderDetection:
    """Cover AdapterFactory.create provider detection paths (lines 2897-2910)."""

    def test_create_auto_detect_from_registry(self):
        factory = AdapterFactory()
        # "gpt-4o" is in registry, should auto-detect OpenAI
        # OpenAI adapter will fail without real API but we can verify detection works
        # by catching the error or using mock
        factory.set_api_key(Provider.OPENAI, "sk-fake")
        with patch.object(OpenAIAdapter, "__init__", lambda self, config: None):
            # Manually set attributes since __init__ is bypassed
            factory.create("gpt-4o")

    def test_create_auto_detect_from_provider_detector(self):
        factory = AdapterFactory()
        # Use a model ID that's not in registry but has a recognizable prefix
        factory.set_api_key(Provider.OPENAI, "sk-fake")
        with patch.object(OpenAIAdapter, "__init__", lambda self, config: None):
            factory.create("gpt-99-future")

    def test_create_unknown_provider_raises(self):
        factory = AdapterFactory()
        with pytest.raises(ValueError, match="Could not detect provider"):
            factory.create("totally-unknown-model-xyz")

    def test_create_no_adapter_class_raises(self):
        factory = AdapterFactory()
        with pytest.raises(ValueError, match="No adapter available"):
            factory.create("test", provider=Provider.GOOGLE)

    def test_get_cached(self):
        factory = AdapterFactory()
        adapter = factory.create("test-model", provider=Provider.MOCK)
        # Verify caching
        cached = factory.get_cached("test-model", Provider.MOCK)
        assert cached is adapter

    def test_get_cached_miss(self):
        factory = AdapterFactory()
        assert factory.get_cached("nonexistent", Provider.MOCK) is None

    def test_create_uses_stored_api_key(self):
        factory = AdapterFactory()
        factory.set_api_key(Provider.MOCK, "stored-key")
        adapter = factory.create("test", provider=Provider.MOCK)
        assert adapter.config.api_key == "stored-key"

    def test_create_with_default_config(self):
        factory = AdapterFactory()
        factory.set_default_config(Provider.MOCK, timeout=120.0)
        adapter = factory.create("test", provider=Provider.MOCK)
        assert adapter.config.timeout == 120.0


class TestFallbackChainGenerateChat:
    """Cover FallbackChain.generate_chat (lines 3002-3014) and fallback_on_error=False."""

    def test_generate_chat_success(self):
        adapter1 = create_mock_adapter("m1")
        adapter1.set_default_response("Chat response 1")
        adapter2 = create_mock_adapter("m2")
        chain = FallbackChain([adapter1, adapter2])
        result = chain.generate_chat([{"role": "user", "content": "Hello"}])
        assert "Chat response 1" in result.text

    def test_generate_chat_fallback(self):
        class FailingAdapter(MockAdapter):
            def generate_chat(self, messages, params=None, **kwargs):
                raise RuntimeError("Chat failed")

        failing = FailingAdapter(AdapterConfig("failing", Provider.MOCK))
        working = create_mock_adapter("working")
        working.set_default_response("Fallback chat")

        chain = FallbackChain([failing, working])
        result = chain.generate_chat([{"role": "user", "content": "Test"}])
        assert "Fallback chat" in result.text

    def test_generate_chat_all_fail(self):
        class FailingAdapter(MockAdapter):
            def generate_chat(self, messages, params=None, **kwargs):
                raise RuntimeError("fail")

        chain = FallbackChain([
            FailingAdapter(AdapterConfig("f1", Provider.MOCK)),
            FailingAdapter(AdapterConfig("f2", Provider.MOCK)),
        ])
        with pytest.raises(RuntimeError, match="All adapters failed"):
            chain.generate_chat([{"role": "user", "content": "Test"}])

    def test_generate_no_fallback_raises_immediately(self):
        class FailingAdapter(MockAdapter):
            def generate(self, prompt, params=None, **kwargs):
                raise RuntimeError("immediate fail")

        chain = FallbackChain(
            [
                FailingAdapter(AdapterConfig("f1", Provider.MOCK)),
                create_mock_adapter("ok"),
            ],
            fallback_on_error=False,
        )
        with pytest.raises(RuntimeError, match="immediate fail"):
            chain.generate("test")

    def test_generate_chat_no_fallback_raises_immediately(self):
        class FailingAdapter(MockAdapter):
            def generate_chat(self, messages, params=None, **kwargs):
                raise RuntimeError("immediate chat fail")

        chain = FallbackChain(
            [
                FailingAdapter(AdapterConfig("f1", Provider.MOCK)),
                create_mock_adapter("ok"),
            ],
            fallback_on_error=False,
        )
        with pytest.raises(RuntimeError, match="immediate chat fail"):
            chain.generate_chat([{"role": "user", "content": "Hi"}])


class TestAdapterPoolRandom:
    """Cover AdapterPool random strategy (lines 3057-3062)."""

    def test_random_strategy(self):
        adapters = [create_mock_adapter(f"m{i}") for i in range(3)]
        pool = AdapterPool(adapters, strategy="random")
        # Just verify it doesn't crash and returns a valid adapter
        selected = pool.get_adapter()
        assert selected in adapters

    def test_default_fallback_strategy(self):
        """When strategy is unknown, falls through to return first adapter."""
        adapters = [create_mock_adapter("first"), create_mock_adapter("second")]
        pool = AdapterPool(adapters, strategy="unknown_strategy")
        selected = pool.get_adapter()
        assert selected.model_id == "first"


class TestConvenienceFunctions:
    """Cover create_adapter, create_fallback_chain, create_adapter_pool."""

    def test_create_adapter_mock(self):
        adapter = create_adapter("test", provider=Provider.MOCK)
        assert adapter.provider == Provider.MOCK

    def test_create_fallback_chain_with_keys(self):
        """create_fallback_chain with recognizable model IDs."""
        # Use model IDs that the ProviderDetector can recognise
        # and set corresponding API keys so the factory can create adapters.
        chain = create_fallback_chain(
            model_ids=["gpt-4o", "claude-3-opus"],
            api_keys={"openai": "sk-fake", "anthropic": "sk-ant-fake"},
        )
        assert len(chain.adapters) == 2

    def test_create_adapter_pool_recognized_models(self):
        """create_adapter_pool with recognisable model IDs."""
        # Patch OpenAIAdapter init to avoid real client creation
        with patch.object(OpenAIAdapter, "__init__", lambda self, config: None):
            pool = create_adapter_pool(
                model_ids=["gpt-4o", "gpt-4-turbo"],
                strategy="round_robin",
            )
            assert len(pool.adapters) == 2


class TestConnectionInfoAndMonitorEdgeCases:
    """Additional coverage for ConnectionInfo to_dict and monitor."""

    def test_connection_info_to_dict_with_error(self):
        info = ConnectionInfo(
            adapter_id="test:model",
            provider=Provider.OPENAI,
            connected=False,
            latency_ms=500.0,
            error="Connection refused",
        )
        d = info.to_dict()
        assert d["connected"] is False
        assert d["error"] == "Connection refused"
        assert d["latency_ms"] == 500.0

    def test_monitor_multiple_adapters(self):
        monitor = ConnectionMonitor()
        a1 = create_mock_adapter("m1")
        a2 = create_mock_adapter("m2")
        monitor.check_adapter(a1)
        monitor.check_adapter(a2)
        status = monitor.get_status()
        assert len(status) == 2


# ============================================================================
# EXPORT COVERAGE
# ============================================================================


class TestSerializeValueToDictObject:
    """Cover serialize_value hasattr(to_dict) path (line 775)."""

    def test_object_with_to_dict(self):
        class MyObj:
            def to_dict(self):
                return {"key": "value"}

        config = ExportConfig()
        result = serialize_value(MyObj(), config)
        assert result == {"key": "value"}

    def test_list_serialization(self):
        config = ExportConfig()
        result = serialize_value([1, 2, 3], config)
        assert result == [1, 2, 3]

    def test_tuple_serialization(self):
        config = ExportConfig()
        result = serialize_value((1, 2), config)
        assert result == [1, 2]

    def test_none_tsv(self):
        config = ExportConfig(format=ExportFormat.TSV, null_value="NULL")
        assert serialize_value(None, config) == "NULL"


class TestSerializeRecordFallbacks:
    """Cover serialize_record fallback paths (lines 875-882)."""

    def test_serialize_record_with_to_dict(self):
        class MyObj:
            def to_dict(self):
                return {"a": 1, "b": 2}

        config = ExportConfig()
        result = serialize_record(MyObj(), config)
        assert result == {"a": 1, "b": 2}

    def test_serialize_record_with_dict_attr(self):
        class PlainObj:
            def __init__(self):
                self.x = 10
                self.y = 20

        config = ExportConfig()
        result = serialize_record(PlainObj(), config)
        assert result["x"] == 10
        assert result["y"] == 20

    def test_serialize_record_primitive(self):
        config = ExportConfig()
        result = serialize_record(42, config)
        assert result == {"value": 42}

    def test_serialize_record_string(self):
        config = ExportConfig()
        result = serialize_record("hello", config)
        assert result == {"value": "hello"}

    def test_serialize_record_dataclass(self):
        @dataclass
        class DC:
            name: str
            val: int

        config = ExportConfig()
        result = serialize_record(DC("test", 99), config)
        assert result["name"] == "test"
        assert result["val"] == 99


class TestPrepareDataEdgeCases:
    """Cover _prepare_data string/bytes path (1126) and non-iterable (1134)."""

    def test_prepare_data_string(self):
        exporter = JSONExporter()
        result = exporter._prepare_data("hello world")
        assert len(result) == 1
        assert result[0] == {"value": "hello world"}

    def test_prepare_data_bytes(self):
        exporter = JSONExporter()
        result = exporter._prepare_data(b"binary data")
        assert len(result) == 1

    def test_prepare_data_non_iterable(self):
        exporter = JSONExporter()
        result = exporter._prepare_data(42)
        assert len(result) == 1
        assert result[0] == {"value": 42}

    def test_prepare_data_generator(self):
        exporter = JSONExporter()

        def gen():
            yield {"x": 1}
            yield {"x": 2}

        result = exporter._prepare_data(gen())
        assert len(result) == 2


class TestExporterFilelikeObjects:
    """Cover file-like object export paths (lines 1259, 1437, 1556, 1733, 1867)."""

    def test_json_export_to_stringio(self):
        exporter = JSONExporter()
        buf = io.StringIO()
        exporter.export({"key": "value"}, buf)
        content = buf.getvalue()
        assert '"key"' in content

    def test_jsonl_export_to_stringio(self):
        exporter = JSONLExporter()
        buf = io.StringIO()
        exporter.export([{"a": 1}, {"a": 2}], buf)
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 2

    def test_jsonl_export_stream_to_stringio(self):
        exporter = JSONLExporter()
        buf = io.StringIO()
        count = exporter.export_stream([{"x": i} for i in range(3)], buf)
        assert count == 3
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 3

    def test_csv_export_to_stringio(self):
        exporter = CSVExporter()
        buf = io.StringIO()
        exporter.export([{"col": "val"}], buf)
        assert "col" in buf.getvalue()

    def test_markdown_export_to_stringio(self):
        exporter = MarkdownExporter()
        buf = io.StringIO()
        exporter.export([{"name": "test"}], buf)
        content = buf.getvalue()
        assert "| name |" in content


class TestCSVExporterEdgeCases:
    """Cover CSV empty data (1779) and _flatten_record None handling (1845)."""

    def test_export_empty_list(self):
        exporter = CSVExporter()
        assert exporter.export_string([]) == ""

    def test_flatten_none_values(self):
        exporter = CSVExporter(ExportConfig(null_value="NA"))
        record = {"a": 1, "b": None}
        flat = exporter._flatten_record(record)
        assert flat["a"] == "1"
        assert flat["b"] == "NA"


class TestMarkdownExporterEdgeCases:
    """Cover Markdown empty data, dict/list cells, None cells (1880, 1902, 1904)."""

    def test_export_empty_data(self):
        exporter = MarkdownExporter()
        assert exporter.export_string([]) == ""

    def test_dict_in_cell(self):
        exporter = MarkdownExporter()
        data = [{"info": {"nested": True}}]
        result = exporter.export_string(data)
        assert '{"nested": true}' in result

    def test_list_in_cell(self):
        exporter = MarkdownExporter()
        data = [{"tags": ["a", "b"]}]
        result = exporter.export_string(data)
        assert '["a", "b"]' in result

    def test_none_in_cell(self):
        exporter = MarkdownExporter()
        data = [{"val": None}]
        result = exporter.export_string(data)
        # None becomes empty string
        assert "| val |" in result


class TestDataArchiverExtended:
    """Cover bzip2, custom output_path, decompress, and create_archive error."""

    def test_compress_bzip2(self, tmp_path):
        archiver = DataArchiver(CompressionType.BZIP2)
        input_file = tmp_path / "test.txt"
        input_file.write_text("bzip2 test data")

        output = archiver.compress_file(input_file)
        assert output.exists()
        assert output.suffix == ".bz2"

    def test_compress_custom_output_path(self, tmp_path):
        archiver = DataArchiver(CompressionType.GZIP)
        input_file = tmp_path / "test.txt"
        input_file.write_text("custom path test")
        custom_out = tmp_path / "custom.gz"

        output = archiver.compress_file(input_file, custom_out)
        assert output == custom_out
        assert output.exists()

    def test_compress_none_type(self, tmp_path):
        """CompressionType.NONE should still produce output (no extension)."""
        archiver = DataArchiver(CompressionType.NONE)
        input_file = tmp_path / "test.txt"
        input_file.write_text("no compression")
        output = archiver.compress_file(input_file)
        # NONE has no extension added, so output path = input.suffix + ""
        assert output.exists() or output == input_file

    def test_create_archive_non_zip_raises(self, tmp_path):
        archiver = DataArchiver(CompressionType.GZIP)
        f1 = tmp_path / "a.txt"
        f1.write_text("data")
        with pytest.raises(ValueError, match="Multi-file archive only supports ZIP"):
            archiver.create_archive([f1], tmp_path / "out.gz")

    def test_create_archive_with_base_path(self, tmp_path):
        archiver = DataArchiver(CompressionType.ZIP)
        subdir = tmp_path / "sub"
        subdir.mkdir()
        f1 = subdir / "a.txt"
        f2 = subdir / "b.txt"
        f1.write_text("aaa")
        f2.write_text("bbb")

        archive_path = tmp_path / "archive.zip"
        archiver.create_archive([f1, f2], archive_path, base_path=subdir)
        assert archive_path.exists()
        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
            assert "a.txt" in names
            assert "b.txt" in names

    def test_create_archive_non_deterministic(self, tmp_path):
        archiver = DataArchiver(CompressionType.ZIP)
        f1 = tmp_path / "a.txt"
        f1.write_text("data")
        archive_path = tmp_path / "ndet.zip"
        archiver.create_archive([f1], archive_path, deterministic=False)
        assert archive_path.exists()

    def test_decompress_gzip_auto_output(self, tmp_path):
        archiver = DataArchiver(CompressionType.GZIP)
        input_file = tmp_path / "test.txt"
        input_file.write_text("decompress me")
        compressed = archiver.compress_file(input_file)
        input_file.unlink()
        decompressed = archiver.decompress_file(compressed)
        assert decompressed.exists()
        assert decompressed.read_text() == "decompress me"

    def test_decompress_zip(self, tmp_path):
        archiver_zip = DataArchiver(CompressionType.ZIP)
        f1 = tmp_path / "file.txt"
        f1.write_text("zip content")
        archive_path = tmp_path / "test.zip"
        archiver_zip.create_archive([f1], archive_path)

        out_dir = tmp_path / "extracted"
        decompressed = archiver_zip.decompress_file(archive_path, out_dir)
        assert decompressed.exists()

    def test_decompress_bzip2(self, tmp_path):
        archiver = DataArchiver(CompressionType.BZIP2)
        input_file = tmp_path / "test.txt"
        input_file.write_text("bzip2 decompress")
        compressed = archiver.compress_file(input_file)
        input_file.unlink()
        decompressed = archiver.decompress_file(compressed)
        assert decompressed.exists()
        assert decompressed.read_text() == "bzip2 decompress"

    def test_decompress_unknown_suffix(self, tmp_path):
        archiver = DataArchiver(CompressionType.GZIP)
        # Create a gzip file with an unusual name
        input_file = tmp_path / "data.bin"
        input_file.write_text("some content")
        compressed = archiver.compress_file(input_file)

        # Rename to an unknown suffix to trigger the .decompressed fallback
        renamed = tmp_path / "data.xyz"
        compressed.rename(renamed)

        # Decompress with explicit output path
        output = tmp_path / "output.txt"
        archiver.decompress_file(renamed, output)

    def test_decompress_zip_auto_output(self, tmp_path):
        """Cover auto output path for .zip suffix (line 2041)."""
        archiver = DataArchiver(CompressionType.ZIP)
        f = tmp_path / "item.txt"
        f.write_text("item data")
        archive_path = tmp_path / "auto.zip"
        archiver.create_archive([f], archive_path)

        # decompress_file with no output_path on .zip -> output_path = parent dir
        decompressed = archiver.decompress_file(archive_path)
        # For .zip, output_path becomes input_path.parent
        assert decompressed == archive_path.parent


class TestDataSchemaValidationEdgeCases:
    """Cover optional field with default (line 2097->2099)."""

    def test_optional_field_with_default_missing(self):
        schema = DataSchema(
            name="test",
            version="1.0",
            fields=[
                SchemaField(name="name", type="string", required=True),
                SchemaField(name="notes", type="string", required=False, default=""),
            ],
        )
        errors = schema.validate({"name": "test"})
        assert len(errors) == 0

    def test_required_field_with_default(self):
        """A required field with a non-None default should not error when missing."""
        schema = DataSchema(
            name="test",
            version="1.0",
            fields=[
                SchemaField(name="status", type="string", required=True, default="active"),
            ],
        )
        errors = schema.validate({})
        assert len(errors) == 0

    def test_validate_type_any(self):
        schema = DataSchema(
            name="test",
            version="1.0",
            fields=[SchemaField(name="data", type="any")],
        )
        errors = schema.validate({"data": [1, 2, 3]})
        assert len(errors) == 0

    def test_validate_float_accepts_int(self):
        schema = DataSchema(
            name="test",
            version="1.0",
            fields=[SchemaField(name="score", type="float")],
        )
        errors = schema.validate({"score": 42})
        assert len(errors) == 0

    def test_validate_bool_type(self):
        schema = DataSchema(
            name="test",
            version="1.0",
            fields=[SchemaField(name="flag", type="bool")],
        )
        errors = schema.validate({"flag": True})
        assert len(errors) == 0
        errors = schema.validate({"flag": "not a bool"})
        assert len(errors) == 1

    def test_validate_list_type(self):
        schema = DataSchema(
            name="test",
            version="1.0",
            fields=[SchemaField(name="items", type="list")],
        )
        errors = schema.validate({"items": [1, 2]})
        assert len(errors) == 0

    def test_validate_dict_type(self):
        schema = DataSchema(
            name="test",
            version="1.0",
            fields=[SchemaField(name="meta", type="dict")],
        )
        errors = schema.validate({"meta": {"k": "v"}})
        assert len(errors) == 0

    def test_validate_none_value_passes(self):
        schema = DataSchema(
            name="test",
            version="1.0",
            fields=[SchemaField(name="opt", type="string")],
        )
        errors = schema.validate({"opt": None})
        assert len(errors) == 0


class TestExportPipelineScalarInputs:
    """Cover pipeline steps with non-list (scalar) inputs."""

    def test_filter_scalar(self):
        pipeline = ExportPipeline().filter(lambda x: True)
        result = pipeline.execute({"key": "val"})
        assert result == {"key": "val"}

    def test_filter_scalar_false(self):
        pipeline = ExportPipeline().filter(lambda x: False)
        result = pipeline.execute({"key": "val"})
        assert result == []

    def test_transform_scalar(self):
        pipeline = ExportPipeline().transform(lambda x: {**x, "added": True})
        result = pipeline.execute({"key": "val"})
        assert result["added"] is True

    def test_select_scalar_dict(self):
        pipeline = ExportPipeline().select(["name"])
        result = pipeline.execute({"name": "test", "extra": "ignored"})
        assert "name" in result
        assert "extra" not in result

    def test_select_scalar_non_dict(self):
        pipeline = ExportPipeline().select(["name"])
        result = pipeline.execute(42)
        assert result == 42

    def test_rename_scalar_dict(self):
        pipeline = ExportPipeline().rename({"old": "new"})
        result = pipeline.execute({"old": "value"})
        assert "new" in result
        assert "old" not in result

    def test_rename_scalar_non_dict(self):
        pipeline = ExportPipeline().rename({"a": "b"})
        result = pipeline.execute(42)
        assert result == 42

    def test_sort_scalar(self):
        pipeline = ExportPipeline().sort("key")
        result = pipeline.execute({"key": 1})
        assert result == {"key": 1}

    def test_limit_scalar(self):
        pipeline = ExportPipeline().limit(5)
        result = pipeline.execute({"key": 1})
        assert result == {"key": 1}

    def test_configure(self):
        config = ExportConfig(pretty_print=False)
        pipeline = ExportPipeline().configure(config)
        assert pipeline._config.pretty_print is False


class TestExportPipelineExportTo:
    """Cover export_to with auto-detected formats (line 2317->2327)."""

    def test_export_to_csv(self, tmp_path):
        pipeline = ExportPipeline()
        data = [{"a": 1}, {"a": 2}]
        output = tmp_path / "output.csv"
        result = pipeline.export_to(data, output)
        assert result.exists()
        content = result.read_text()
        assert "a" in content

    def test_export_to_jsonl(self, tmp_path):
        pipeline = ExportPipeline()
        data = [{"a": 1}]
        output = tmp_path / "output.jsonl"
        result = pipeline.export_to(data, output)
        assert result.exists()

    def test_export_to_tsv(self, tmp_path):
        pipeline = ExportPipeline()
        data = [{"a": 1}]
        output = tmp_path / "output.tsv"
        result = pipeline.export_to(data, output)
        assert result.exists()

    def test_export_to_md(self, tmp_path):
        pipeline = ExportPipeline()
        data = [{"a": 1}]
        output = tmp_path / "output.md"
        result = pipeline.export_to(data, output)
        assert result.exists()

    def test_export_to_unknown_ext_defaults_json(self, tmp_path):
        pipeline = ExportPipeline()
        data = [{"a": 1}]
        output = tmp_path / "output.xyz"
        result = pipeline.export_to(data, output)
        assert result.exists()
        # Should default to JSON
        content = json.loads(result.read_text())
        assert isinstance(content, list)

    def test_export_to_explicit_format(self, tmp_path):
        pipeline = ExportPipeline()
        data = [{"a": 1}]
        output = tmp_path / "output.txt"
        result = pipeline.export_to(data, output, format=ExportFormat.MARKDOWN)
        assert result.exists()
        assert "|" in result.read_text()


class TestStreamExportNonJsonl:
    """Cover stream_export non-JSONL fallback (lines 2434-2437)."""

    def test_stream_export_json(self, tmp_path):
        data = [{"id": i} for i in range(5)]
        output = tmp_path / "out.json"
        count = stream_export(iter(data), output, ExportFormat.JSON)
        assert count == 5
        content = json.loads(output.read_text())
        assert len(content) == 5

    def test_stream_export_csv(self, tmp_path):
        data = [{"a": 1}, {"a": 2}]
        output = tmp_path / "out.csv"
        count = stream_export(iter(data), output, ExportFormat.CSV)
        assert count == 2


class TestCreateExportBundleExtended:
    """Cover create_export_bundle extra paths: dict data, schema inference types."""

    def test_bundle_with_dict_data(self, tmp_path):
        """Cover line 2496: JSON dict export via prepared[0]."""
        data = {"name": "experiment", "score": 0.95}
        result = create_export_bundle(
            data, tmp_path, "dict_bundle", formats=[ExportFormat.JSON], compress=False
        )
        assert result.exists()
        json_file = result / "dict_bundle.json"
        assert json_file.exists()
        content = json.loads(json_file.read_text())
        assert content["name"] == "experiment"

    def test_bundle_inferred_schema_types(self, tmp_path):
        """Cover schema inference with bool, float, list, dict (lines 2564-2572)."""
        data = [
            {
                "name": "test",
                "flag": True,
                "count": 42,
                "score": 3.14,
                "tags": ["a", "b"],
                "meta": {"k": "v"},
            }
        ]
        result = create_export_bundle(
            data,
            tmp_path,
            "typed_bundle",
            formats=[ExportFormat.JSON],
            compress=False,
            include_schema=True,
        )
        schema_file = result / "schema.json"
        assert schema_file.exists()
        schema = json.loads(schema_file.read_text())
        field_types = {f["name"]: f["type"] for f in schema["fields"]}
        assert field_types["name"] == "string"
        assert field_types["flag"] == "bool"
        assert field_types["count"] == "int"
        assert field_types["score"] == "float"
        assert field_types["tags"] == "list"
        assert field_types["meta"] == "dict"

    def test_bundle_no_schema(self, tmp_path):
        data = [{"a": 1}]
        result = create_export_bundle(
            data,
            tmp_path,
            "no_schema",
            formats=[ExportFormat.JSON],
            compress=False,
            include_schema=False,
        )
        assert not (result / "schema.json").exists()

    def test_bundle_non_deterministic(self, tmp_path):
        data = [{"a": 1}]
        result = create_export_bundle(
            data,
            tmp_path,
            "ndet_bundle",
            formats=[ExportFormat.JSON],
            compress=False,
            deterministic=False,
        )
        meta = json.loads((result / "metadata.json").read_text())
        # Non-deterministic should use actual time, not 1970 epoch
        assert meta["export_time"] != "1970-01-01T00:00:00"

    def test_bundle_with_custom_export_time(self, tmp_path):
        data = [{"a": 1}]
        result = create_export_bundle(
            data,
            tmp_path,
            "custom_time",
            formats=[ExportFormat.JSON],
            compress=False,
            export_time="2024-06-15T00:00:00",
        )
        meta = json.loads((result / "metadata.json").read_text())
        assert meta["export_time"] == "2024-06-15T00:00:00"

    def test_bundle_markdown_format(self, tmp_path):
        data = [{"col": "val"}]
        result = create_export_bundle(
            data,
            tmp_path,
            "md_bundle",
            formats=[ExportFormat.MARKDOWN],
            compress=False,
        )
        md_file = result / "md_bundle.md"
        assert md_file.exists()

    def test_bundle_tsv_format(self, tmp_path):
        data = [{"col": "val"}]
        result = create_export_bundle(
            data,
            tmp_path,
            "tsv_bundle",
            formats=[ExportFormat.TSV],
            compress=False,
        )
        tsv_file = result / "tsv_bundle.tsv"
        assert tsv_file.exists()

    def test_bundle_empty_data_no_schema(self, tmp_path):
        """Empty data + include_schema should not create schema."""
        data = []
        result = create_export_bundle(
            data,
            tmp_path,
            "empty",
            formats=[ExportFormat.JSON],
            compress=False,
            include_schema=True,
        )
        assert not (result / "schema.json").exists()


class TestGetExporterTSV:
    """Cover TSV exporter creation in get_exporter."""

    def test_get_tsv_exporter(self):
        exporter = get_exporter(ExportFormat.TSV)
        assert isinstance(exporter, CSVExporter)
        assert exporter.delimiter == "\t"

    def test_get_markdown_exporter(self):
        exporter = get_exporter(ExportFormat.MARKDOWN)
        assert isinstance(exporter, MarkdownExporter)

    def test_get_jsonl_exporter(self):
        exporter = get_exporter(ExportFormat.JSONL)
        assert isinstance(exporter, JSONLExporter)


class TestExportToFunctions:
    """Cover export_to_* convenience functions with kwargs."""

    def test_export_to_json_with_kwargs(self, tmp_path):
        output = tmp_path / "test.json"
        export_to_json({"k": "v"}, output, pretty_print=False)
        content = output.read_text()
        assert "\n" not in content.strip() or content.count("\n") <= 1

    def test_export_to_jsonl_with_kwargs(self, tmp_path):
        output = tmp_path / "test.jsonl"
        export_to_jsonl([{"a": 1}], output, encoding="utf-8")
        assert output.exists()

    def test_export_to_csv_with_kwargs(self, tmp_path):
        output = tmp_path / "test.csv"
        export_to_csv([{"a": 1}], output, null_value="NA")
        assert output.exists()

    def test_export_to_markdown_with_kwargs(self, tmp_path):
        output = tmp_path / "test.md"
        export_to_markdown([{"a": 1}], output, encoding="utf-8")
        assert output.exists()
