"""Tests for data export and serialization utilities."""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import pytest

from insideLLMs.export import (
    CSVExporter,
    CompressionType,
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


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_all_formats_exist(self):
        """Test that all formats are defined."""
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.JSONL.value == "jsonl"
        assert ExportFormat.CSV.value == "csv"
        assert ExportFormat.MARKDOWN.value == "markdown"


class TestCompressionType:
    """Tests for CompressionType enum."""

    def test_all_types_exist(self):
        """Test that all compression types are defined."""
        assert CompressionType.NONE.value == "none"
        assert CompressionType.GZIP.value == "gzip"
        assert CompressionType.ZIP.value == "zip"


class TestExportConfig:
    """Tests for ExportConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExportConfig()
        assert config.format == ExportFormat.JSON
        assert config.compression == CompressionType.NONE
        assert config.pretty_print is True
        assert config.encoding == "utf-8"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ExportConfig(
            format=ExportFormat.CSV,
            compression=CompressionType.GZIP,
            pretty_print=False,
        )
        assert config.format == ExportFormat.CSV
        assert config.compression == CompressionType.GZIP
        assert config.pretty_print is False


class TestExportMetadata:
    """Tests for ExportMetadata."""

    def test_default_values(self):
        """Test default metadata values."""
        metadata = ExportMetadata()
        assert metadata.version == "1.0"
        assert metadata.source == "insideLLMs"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = ExportMetadata(record_count=100, format="json")
        d = metadata.to_dict()

        assert d["record_count"] == 100
        assert d["format"] == "json"
        assert "export_time" in d


class TestSerialization:
    """Tests for serialization functions."""

    def test_serialize_basic_types(self):
        """Test serializing basic types."""
        config = ExportConfig()

        assert serialize_value("hello", config) == "hello"
        assert serialize_value(42, config) == 42
        assert serialize_value(3.14, config) == 3.14
        assert serialize_value(True, config) is True

    def test_serialize_none(self):
        """Test serializing None."""
        config = ExportConfig()
        assert serialize_value(None, config) is None

        config_csv = ExportConfig(format=ExportFormat.CSV, null_value="N/A")
        assert serialize_value(None, config_csv) == "N/A"

    def test_serialize_datetime(self):
        """Test serializing datetime."""
        config = ExportConfig(date_format="%Y-%m-%d")
        dt = datetime(2024, 1, 15)
        assert serialize_value(dt, config) == "2024-01-15"

    def test_serialize_enum(self):
        """Test serializing enum."""
        config = ExportConfig()
        assert serialize_value(ExportFormat.JSON, config) == "json"

    def test_serialize_dataclass(self):
        """Test serializing dataclass."""
        @dataclass
        class Sample:
            name: str
            value: int

        config = ExportConfig()
        obj = Sample(name="test", value=42)
        result = serialize_value(obj, config)

        assert result["name"] == "test"
        assert result["value"] == 42

    def test_serialize_record(self):
        """Test serializing a record."""
        config = ExportConfig()
        record = {"name": "test", "count": 5}
        result = serialize_record(record, config)

        assert result == {"name": "test", "count": 5}


class TestJSONExporter:
    """Tests for JSONExporter."""

    def test_export_dict(self):
        """Test exporting a dictionary."""
        exporter = JSONExporter()
        data = {"key": "value", "number": 42}

        result = exporter.export_string(data)
        parsed = json.loads(result)

        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_export_list(self):
        """Test exporting a list."""
        exporter = JSONExporter()
        data = [{"id": 1}, {"id": 2}]

        result = exporter.export_string(data)
        parsed = json.loads(result)

        assert len(parsed) == 2
        assert parsed[0]["id"] == 1

    def test_export_to_file(self, tmp_path):
        """Test exporting to file."""
        exporter = JSONExporter()
        data = {"test": "data"}
        output_path = tmp_path / "output.json"

        exporter.export(data, output_path)

        with open(output_path) as f:
            content = json.load(f)
        assert content["test"] == "data"

    def test_pretty_print_disabled(self):
        """Test disabling pretty print."""
        config = ExportConfig(pretty_print=False)
        exporter = JSONExporter(config)
        data = {"key": "value"}

        result = exporter.export_string(data)
        assert "\n" not in result


class TestJSONLExporter:
    """Tests for JSONLExporter."""

    def test_export_list(self):
        """Test exporting list to JSONL."""
        exporter = JSONLExporter()
        data = [{"id": 1}, {"id": 2}, {"id": 3}]

        result = exporter.export_string(data)
        lines = result.strip().split("\n")

        assert len(lines) == 3
        assert json.loads(lines[0])["id"] == 1
        assert json.loads(lines[2])["id"] == 3

    def test_export_to_file(self, tmp_path):
        """Test exporting to file."""
        exporter = JSONLExporter()
        data = [{"a": 1}, {"a": 2}]
        output_path = tmp_path / "output.jsonl"

        exporter.export(data, output_path)

        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_stream_export(self, tmp_path):
        """Test streaming export."""
        exporter = JSONLExporter()
        output_path = tmp_path / "stream.jsonl"

        def data_generator():
            for i in range(5):
                yield {"index": i}

        count = exporter.export_stream(data_generator(), output_path)

        assert count == 5
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 5


class TestCSVExporter:
    """Tests for CSVExporter."""

    def test_export_list(self):
        """Test exporting list to CSV."""
        exporter = CSVExporter()
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]

        result = exporter.export_string(data)
        lines = result.strip().split("\n")

        assert len(lines) == 3  # header + 2 rows
        assert "name" in lines[0]
        assert "Alice" in lines[1]

    def test_export_with_nested(self):
        """Test exporting with nested structures."""
        exporter = CSVExporter()
        data = [{"name": "Test", "tags": ["a", "b"]}]

        result = exporter.export_string(data)
        # CSV escapes quotes, so we just check the list is serialized
        assert "a" in result and "b" in result
        assert "Test" in result

    def test_export_to_file(self, tmp_path):
        """Test exporting to file."""
        exporter = CSVExporter()
        data = [{"col1": "val1"}]
        output_path = tmp_path / "output.csv"

        exporter.export(data, output_path)

        with open(output_path) as f:
            content = f.read()
        assert "col1" in content
        assert "val1" in content

    def test_tsv_delimiter(self):
        """Test TSV delimiter."""
        exporter = CSVExporter(delimiter="\t")
        data = [{"a": 1, "b": 2}]

        result = exporter.export_string(data)
        assert "\t" in result


class TestMarkdownExporter:
    """Tests for MarkdownExporter."""

    def test_export_list(self):
        """Test exporting list to Markdown."""
        exporter = MarkdownExporter()
        data = [{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}]

        result = exporter.export_string(data)
        lines = result.strip().split("\n")

        assert len(lines) == 4  # header + separator + 2 rows
        assert "|" in lines[0]
        assert "---" in lines[1]
        assert "Alice" in lines[2]

    def test_escape_pipes(self):
        """Test escaping pipe characters."""
        exporter = MarkdownExporter()
        data = [{"text": "a|b"}]

        result = exporter.export_string(data)
        assert "\\|" in result

    def test_export_to_file(self, tmp_path):
        """Test exporting to file."""
        exporter = MarkdownExporter()
        data = [{"col": "val"}]
        output_path = tmp_path / "output.md"

        exporter.export(data, output_path)

        with open(output_path) as f:
            content = f.read()
        assert "col" in content


class TestDataArchiver:
    """Tests for DataArchiver."""

    def test_compress_gzip(self, tmp_path):
        """Test GZIP compression."""
        archiver = DataArchiver(CompressionType.GZIP)
        input_file = tmp_path / "test.txt"
        input_file.write_text("Hello, World!")

        output_path = archiver.compress_file(input_file)

        assert output_path.exists()
        assert output_path.suffix == ".gz"

    def test_compress_zip(self, tmp_path):
        """Test ZIP compression."""
        archiver = DataArchiver(CompressionType.ZIP)
        input_file = tmp_path / "test.txt"
        input_file.write_text("Hello, World!")

        output_path = archiver.compress_file(input_file)

        assert output_path.exists()
        assert output_path.suffix == ".zip"

    def test_create_archive(self, tmp_path):
        """Test creating multi-file archive."""
        archiver = DataArchiver(CompressionType.ZIP)

        # Create test files
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        output_path = tmp_path / "archive.zip"
        archiver.create_archive([file1, file2], output_path)

        assert output_path.exists()

    def test_decompress_gzip(self, tmp_path):
        """Test GZIP decompression."""
        archiver = DataArchiver(CompressionType.GZIP)
        input_file = tmp_path / "test.txt"
        input_file.write_text("Test content")

        compressed = archiver.compress_file(input_file)
        input_file.unlink()  # Delete original

        decompressed = archiver.decompress_file(compressed)

        assert decompressed.exists()
        assert decompressed.read_text() == "Test content"


class TestDataSchema:
    """Tests for DataSchema."""

    def test_validate_valid_record(self):
        """Test validating a valid record."""
        schema = DataSchema(
            name="test",
            version="1.0",
            fields=[
                SchemaField(name="name", type="string"),
                SchemaField(name="count", type="int"),
            ],
        )

        errors = schema.validate({"name": "test", "count": 5})
        assert len(errors) == 0

    def test_validate_missing_required(self):
        """Test validation with missing required field."""
        schema = DataSchema(
            name="test",
            version="1.0",
            fields=[SchemaField(name="required_field", type="string", required=True)],
        )

        errors = schema.validate({})
        assert len(errors) == 1
        assert "Missing required" in errors[0]

    def test_validate_wrong_type(self):
        """Test validation with wrong type."""
        schema = DataSchema(
            name="test",
            version="1.0",
            fields=[SchemaField(name="number", type="int")],
        )

        errors = schema.validate({"number": "not a number"})
        assert len(errors) == 1
        assert "expected int" in errors[0]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        schema = DataSchema(
            name="test",
            version="1.0",
            fields=[SchemaField(name="field1", type="string")],
        )

        d = schema.to_dict()
        assert d["name"] == "test"
        assert len(d["fields"]) == 1


class TestExportPipeline:
    """Tests for ExportPipeline."""

    def test_filter(self):
        """Test filter step."""
        pipeline = ExportPipeline()
        pipeline.filter(lambda x: x.get("active", False))

        data = [{"id": 1, "active": True}, {"id": 2, "active": False}]
        result = pipeline.execute(data)

        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_transform(self):
        """Test transform step."""
        pipeline = ExportPipeline()
        pipeline.transform(lambda x: {**x, "doubled": x.get("value", 0) * 2})

        data = [{"value": 5}]
        result = pipeline.execute(data)

        assert result[0]["doubled"] == 10

    def test_select(self):
        """Test select step."""
        pipeline = ExportPipeline()
        pipeline.select(["name"])

        data = [{"name": "test", "extra": "ignored"}]
        result = pipeline.execute(data)

        assert "name" in result[0]
        assert "extra" not in result[0]

    def test_rename(self):
        """Test rename step."""
        pipeline = ExportPipeline()
        pipeline.rename({"old_name": "new_name"})

        data = [{"old_name": "value"}]
        result = pipeline.execute(data)

        assert "new_name" in result[0]
        assert "old_name" not in result[0]

    def test_sort(self):
        """Test sort step."""
        pipeline = ExportPipeline()
        pipeline.sort("score", reverse=True)

        data = [{"score": 5}, {"score": 10}, {"score": 3}]
        result = pipeline.execute(data)

        assert result[0]["score"] == 10
        assert result[2]["score"] == 3

    def test_limit(self):
        """Test limit step."""
        pipeline = ExportPipeline()
        pipeline.limit(2)

        data = [{"id": i} for i in range(10)]
        result = pipeline.execute(data)

        assert len(result) == 2

    def test_chaining(self):
        """Test chaining multiple steps."""
        pipeline = (
            ExportPipeline()
            .filter(lambda x: x.get("value", 0) > 0)
            .transform(lambda x: {**x, "squared": x["value"] ** 2})
            .select(["value", "squared"])
            .sort("value")
            .limit(3)
        )

        data = [{"value": -1}, {"value": 3}, {"value": 1}, {"value": 2}, {"value": 4}]
        result = pipeline.execute(data)

        assert len(result) == 3
        assert result[0]["value"] == 1
        assert result[0]["squared"] == 1
        assert result[2]["value"] == 3

    def test_export_to(self, tmp_path):
        """Test export_to method."""
        pipeline = ExportPipeline().filter(lambda x: x.get("include", False))

        data = [{"id": 1, "include": True}, {"id": 2, "include": False}]
        output_path = tmp_path / "output.json"

        pipeline.export_to(data, output_path)

        with open(output_path) as f:
            content = json.load(f)
        assert len(content) == 1


class TestGetExporter:
    """Tests for get_exporter function."""

    def test_get_json_exporter(self):
        """Test getting JSON exporter."""
        exporter = get_exporter(ExportFormat.JSON)
        assert isinstance(exporter, JSONExporter)

    def test_get_csv_exporter(self):
        """Test getting CSV exporter."""
        exporter = get_exporter(ExportFormat.CSV)
        assert isinstance(exporter, CSVExporter)

    def test_unsupported_format(self):
        """Test unsupported format."""
        with pytest.raises(ValueError):
            get_exporter(ExportFormat.PARQUET)


class TestQuickExportFunctions:
    """Tests for quick export functions."""

    def test_export_to_json(self, tmp_path):
        """Test export_to_json function."""
        data = {"key": "value"}
        output = tmp_path / "test.json"

        export_to_json(data, output)

        with open(output) as f:
            content = json.load(f)
        assert content["key"] == "value"

    def test_export_to_jsonl(self, tmp_path):
        """Test export_to_jsonl function."""
        data = [{"id": 1}, {"id": 2}]
        output = tmp_path / "test.jsonl"

        export_to_jsonl(data, output)

        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_export_to_csv(self, tmp_path):
        """Test export_to_csv function."""
        data = [{"name": "test"}]
        output = tmp_path / "test.csv"

        export_to_csv(data, output)

        with open(output) as f:
            content = f.read()
        assert "name" in content

    def test_export_to_markdown(self, tmp_path):
        """Test export_to_markdown function."""
        data = [{"col": "val"}]
        output = tmp_path / "test.md"

        export_to_markdown(data, output)

        with open(output) as f:
            content = f.read()
        assert "|" in content


class TestStreamExport:
    """Tests for stream_export function."""

    def test_stream_export_jsonl(self, tmp_path):
        """Test streaming to JSONL."""
        def generate_data():
            for i in range(100):
                yield {"index": i}

        output = tmp_path / "stream.jsonl"
        count = stream_export(generate_data(), output, ExportFormat.JSONL)

        assert count == 100
        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 100


class TestCreateExportBundle:
    """Tests for create_export_bundle function."""

    def test_create_bundle(self, tmp_path):
        """Test creating export bundle."""
        data = [{"name": "test", "value": 42}]

        result = create_export_bundle(
            data,
            tmp_path,
            "test_bundle",
            formats=[ExportFormat.JSON, ExportFormat.CSV],
            compress=False,
        )

        assert result.exists()
        assert (result / "test_bundle.json").exists()
        assert (result / "test_bundle.csv").exists()
        assert (result / "metadata.json").exists()
        assert (result / "schema.json").exists()

    def test_create_compressed_bundle(self, tmp_path):
        """Test creating compressed bundle."""
        data = [{"name": "test"}]

        result = create_export_bundle(
            data,
            tmp_path,
            "compressed_bundle",
            compress=True,
        )

        assert result.suffix == ".zip"
        assert result.exists()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_data(self):
        """Test exporting empty data."""
        exporter = JSONExporter()
        result = exporter.export_string([])
        assert json.loads(result) == []

    def test_unicode_data(self):
        """Test exporting unicode data."""
        exporter = JSONExporter()
        data = {"text": "Hello 世界! 🌍"}
        result = exporter.export_string(data)

        assert "世界" in result
        assert "🌍" in result

    def test_nested_structures(self):
        """Test exporting nested structures."""
        exporter = JSONExporter()
        data = {
            "level1": {
                "level2": {
                    "level3": "deep value"
                }
            }
        }

        result = exporter.export_string(data)
        parsed = json.loads(result)
        assert parsed["level1"]["level2"]["level3"] == "deep value"

    def test_special_characters_csv(self):
        """Test CSV with special characters."""
        exporter = CSVExporter()
        data = [{"text": 'Hello, "World"'}]

        result = exporter.export_string(data)
        assert "Hello" in result  # Should be properly escaped
