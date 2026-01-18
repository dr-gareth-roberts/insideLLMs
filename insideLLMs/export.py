"""
Data export and serialization utilities for LLM experiments.

Provides tools for:
- Multi-format export (JSON, JSONL, CSV, Parquet, Excel)
- Data compression and archiving
- Schema validation and versioning
- Streaming exports for large datasets
- Export pipeline composition
"""

import csv
import gzip
import io
import json
import shutil
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Optional,
    Protocol,
    TextIO,
    Union,
)


class ExportFormat(Enum):
    """Supported export formats."""

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"
    YAML = "yaml"
    PARQUET = "parquet"
    EXCEL = "excel"
    MARKDOWN = "markdown"


class CompressionType(Enum):
    """Supported compression types."""

    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"
    BZIP2 = "bz2"


@dataclass
class ExportConfig:
    """Configuration for data export."""

    format: ExportFormat = ExportFormat.JSON
    compression: CompressionType = CompressionType.NONE
    pretty_print: bool = True
    include_metadata: bool = True
    date_format: str = "%Y-%m-%dT%H:%M:%S"
    null_value: str = ""
    encoding: str = "utf-8"
    chunk_size: int = 1000


@dataclass
class ExportMetadata:
    """Metadata for exported data."""

    export_time: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    schema_version: str = "1.0"
    source: str = "insideLLMs"
    record_count: int = 0
    format: str = ""
    compression: str = "none"
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "export_time": self.export_time,
            "version": self.version,
            "schema_version": self.schema_version,
            "source": self.source,
            "record_count": self.record_count,
            "format": self.format,
            "compression": self.compression,
            "custom": self.custom,
        }


class Serializable(Protocol):
    """Protocol for serializable objects."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        ...


def serialize_value(value: Any, config: ExportConfig) -> Any:
    """Serialize a single value for export.

    Args:
        value: Value to serialize.
        config: Export configuration.

    Returns:
        Serialized value.
    """
    if value is None:
        return config.null_value if config.format in (ExportFormat.CSV, ExportFormat.TSV) else None

    if isinstance(value, datetime):
        return value.strftime(config.date_format)

    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)

    if hasattr(value, "to_dict"):
        return value.to_dict()

    if isinstance(value, (list, tuple)):
        return [serialize_value(v, config) for v in value]

    if isinstance(value, dict):
        return {k: serialize_value(v, config) for k, v in value.items()}

    return value


def serialize_record(record: Any, config: ExportConfig) -> dict[str, Any]:
    """Serialize a record for export.

    Args:
        record: Record to serialize.
        config: Export configuration.

    Returns:
        Serialized record as dictionary.
    """
    if isinstance(record, dict):
        data = record
    elif is_dataclass(record) and not isinstance(record, type):
        data = asdict(record)
    elif hasattr(record, "to_dict"):
        data = record.to_dict()
    elif hasattr(record, "__dict__"):
        data = record.__dict__
    else:
        data = {"value": record}

    return {k: serialize_value(v, config) for k, v in data.items()}


class Exporter(ABC):
    """Abstract base class for data exporters."""

    def __init__(self, config: Optional[ExportConfig] = None):
        """Initialize exporter.

        Args:
            config: Export configuration.
        """
        self.config = config or ExportConfig()

    @abstractmethod
    def export(self, data: Any, output: Union[str, Path, TextIO, BinaryIO]) -> None:
        """Export data to output.

        Args:
            data: Data to export.
            output: Output destination (path or file object).
        """
        pass

    @abstractmethod
    def export_string(self, data: Any) -> str:
        """Export data to string.

        Args:
            data: Data to export.

        Returns:
            Exported data as string.
        """
        pass

    def _prepare_data(self, data: Any) -> list[dict[str, Any]]:
        """Prepare data for export.

        Args:
            data: Input data.

        Returns:
            List of serialized records.
        """
        if isinstance(data, dict):
            return [serialize_record(data, self.config)]

        if isinstance(data, (list, tuple)):
            return [serialize_record(item, self.config) for item in data]

        if hasattr(data, "__iter__"):
            return [serialize_record(item, self.config) for item in data]

        return [serialize_record(data, self.config)]


class JSONExporter(Exporter):
    """Export data to JSON format."""

    def export(self, data: Any, output: Union[str, Path, TextIO, BinaryIO]) -> None:
        """Export data to JSON.

        Args:
            data: Data to export.
            output: Output destination.
        """
        content = self.export_string(data)

        if isinstance(output, (str, Path)):
            with open(output, "w", encoding=self.config.encoding) as f:
                f.write(content)
        else:
            output.write(content)

    def export_string(self, data: Any) -> str:
        """Export data to JSON string.

        Args:
            data: Data to export.

        Returns:
            JSON string.
        """
        prepared = self._prepare_data(data)

        # Unwrap single item
        if len(prepared) == 1 and isinstance(data, dict):
            prepared = prepared[0]

        indent = 2 if self.config.pretty_print else None
        return json.dumps(prepared, indent=indent, ensure_ascii=False, default=str)


class JSONLExporter(Exporter):
    """Export data to JSON Lines format."""

    def export(self, data: Any, output: Union[str, Path, TextIO, BinaryIO]) -> None:
        """Export data to JSONL.

        Args:
            data: Data to export.
            output: Output destination.
        """
        content = self.export_string(data)

        if isinstance(output, (str, Path)):
            with open(output, "w", encoding=self.config.encoding) as f:
                f.write(content)
        else:
            output.write(content)

    def export_string(self, data: Any) -> str:
        """Export data to JSONL string.

        Args:
            data: Data to export.

        Returns:
            JSONL string.
        """
        prepared = self._prepare_data(data)
        lines = [json.dumps(record, ensure_ascii=False, default=str) for record in prepared]
        return "\n".join(lines)

    def export_stream(self, data: Iterable[Any], output: Union[str, Path, TextIO]) -> int:
        """Export data as stream (memory efficient).

        Args:
            data: Iterable data source.
            output: Output destination.

        Returns:
            Number of records written.
        """
        count = 0

        def write_line(f, record):
            nonlocal count
            serialized = serialize_record(record, self.config)
            f.write(json.dumps(serialized, ensure_ascii=False, default=str))
            f.write("\n")
            count += 1

        if isinstance(output, (str, Path)):
            with open(output, "w", encoding=self.config.encoding) as f:
                for record in data:
                    write_line(f, record)
        else:
            for record in data:
                write_line(output, record)

        return count


class CSVExporter(Exporter):
    """Export data to CSV format."""

    def __init__(
        self,
        config: Optional[ExportConfig] = None,
        delimiter: str = ",",
        quoting: int = csv.QUOTE_MINIMAL,
    ):
        """Initialize CSV exporter.

        Args:
            config: Export configuration.
            delimiter: Field delimiter.
            quoting: CSV quoting style.
        """
        super().__init__(config)
        self.delimiter = delimiter
        self.quoting = quoting

    def export(self, data: Any, output: Union[str, Path, TextIO, BinaryIO]) -> None:
        """Export data to CSV.

        Args:
            data: Data to export.
            output: Output destination.
        """
        content = self.export_string(data)

        if isinstance(output, (str, Path)):
            with open(output, "w", encoding=self.config.encoding, newline="") as f:
                f.write(content)
        else:
            output.write(content)

    def export_string(self, data: Any) -> str:
        """Export data to CSV string.

        Args:
            data: Data to export.

        Returns:
            CSV string.
        """
        prepared = self._prepare_data(data)
        if not prepared:
            return ""

        output = io.StringIO()
        fieldnames = list(prepared[0].keys())

        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            delimiter=self.delimiter,
            quoting=self.quoting,
        )
        writer.writeheader()

        for record in prepared:
            # Flatten nested structures
            flat_record = self._flatten_record(record)
            writer.writerow(flat_record)

        return output.getvalue()

    def _flatten_record(self, record: dict[str, Any]) -> dict[str, str]:
        """Flatten nested record for CSV.

        Args:
            record: Record to flatten.

        Returns:
            Flattened record with string values.
        """
        flat = {}
        for key, value in record.items():
            if isinstance(value, (dict, list)):
                flat[key] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                flat[key] = self.config.null_value
            else:
                flat[key] = str(value)
        return flat


class MarkdownExporter(Exporter):
    """Export data to Markdown table format."""

    def export(self, data: Any, output: Union[str, Path, TextIO, BinaryIO]) -> None:
        """Export data to Markdown.

        Args:
            data: Data to export.
            output: Output destination.
        """
        content = self.export_string(data)

        if isinstance(output, (str, Path)):
            with open(output, "w", encoding=self.config.encoding) as f:
                f.write(content)
        else:
            output.write(content)

    def export_string(self, data: Any) -> str:
        """Export data to Markdown string.

        Args:
            data: Data to export.

        Returns:
            Markdown table string.
        """
        prepared = self._prepare_data(data)
        if not prepared:
            return ""

        # Get all column names
        columns = list(prepared[0].keys())

        # Build table
        lines = []

        # Header
        header = "| " + " | ".join(columns) + " |"
        lines.append(header)

        # Separator
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        lines.append(separator)

        # Rows
        for record in prepared:
            row_values = []
            for col in columns:
                value = record.get(col, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                elif value is None:
                    value = ""
                # Escape pipe characters
                value = str(value).replace("|", "\\|")
                row_values.append(value)
            row = "| " + " | ".join(row_values) + " |"
            lines.append(row)

        return "\n".join(lines)


class DataArchiver:
    """Create compressed archives of exported data."""

    def __init__(self, compression: CompressionType = CompressionType.ZIP):
        """Initialize archiver.

        Args:
            compression: Compression type to use.
        """
        self.compression = compression

    def compress_file(
        self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Compress a single file.

        Args:
            input_path: Path to input file.
            output_path: Path to output file (auto-generated if None).

        Returns:
            Path to compressed file.
        """
        input_path = Path(input_path)

        if output_path is None:
            ext = {
                CompressionType.GZIP: ".gz",
                CompressionType.ZIP: ".zip",
                CompressionType.BZIP2: ".bz2",
            }.get(self.compression, "")
            output_path = input_path.with_suffix(input_path.suffix + ext)
        else:
            output_path = Path(output_path)

        if self.compression == CompressionType.GZIP:
            with open(input_path, "rb") as f_in, gzip.open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        elif self.compression == CompressionType.ZIP:
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(input_path, input_path.name)

        elif self.compression == CompressionType.BZIP2:
            import bz2

            with open(input_path, "rb") as f_in, bz2.open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        return output_path

    def create_archive(
        self,
        files: list[Union[str, Path]],
        output_path: Union[str, Path],
        base_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Create archive from multiple files.

        Args:
            files: List of files to archive.
            output_path: Output archive path.
            base_path: Base path for relative file paths in archive.

        Returns:
            Path to archive.
        """
        output_path = Path(output_path)

        if self.compression == CompressionType.ZIP:
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in files:
                    file_path = Path(file_path)
                    arcname = file_path.relative_to(base_path) if base_path else file_path.name
                    zf.write(file_path, arcname)
        else:
            raise ValueError(f"Multi-file archive only supports ZIP, not {self.compression}")

        return output_path

    def decompress_file(
        self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Decompress a file.

        Args:
            input_path: Path to compressed file.
            output_path: Path to output file.

        Returns:
            Path to decompressed file.
        """
        input_path = Path(input_path)

        if output_path is None:
            # Remove compression extension
            if input_path.suffix in (".gz", ".bz2"):
                output_path = input_path.with_suffix("")
            elif input_path.suffix == ".zip":
                output_path = input_path.parent
            else:
                output_path = input_path.with_suffix(".decompressed")
        else:
            output_path = Path(output_path)

        if self.compression == CompressionType.GZIP or input_path.suffix == ".gz":
            with gzip.open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        elif self.compression == CompressionType.ZIP or input_path.suffix == ".zip":
            with zipfile.ZipFile(input_path, "r") as zf:
                zf.extractall(output_path)

        elif self.compression == CompressionType.BZIP2 or input_path.suffix == ".bz2":
            import bz2

            with bz2.open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        return output_path


@dataclass
class SchemaField:
    """Schema field definition."""

    name: str
    type: str
    required: bool = True
    default: Any = None
    description: str = ""


@dataclass
class DataSchema:
    """Schema definition for exported data."""

    name: str
    version: str
    fields: list[SchemaField]
    description: str = ""

    def validate(self, record: dict[str, Any]) -> list[str]:
        """Validate a record against this schema.

        Args:
            record: Record to validate.

        Returns:
            List of validation errors.
        """
        errors = []

        for field_def in self.fields:
            if field_def.name not in record:
                if field_def.required and field_def.default is None:
                    errors.append(f"Missing required field: {field_def.name}")
                continue

            value = record[field_def.name]

            # Type checking
            expected_type = {
                "string": str,
                "int": int,
                "float": (int, float),
                "bool": bool,
                "list": list,
                "dict": dict,
                "any": object,
            }.get(field_def.type, object)

            if value is not None and not isinstance(value, expected_type):
                errors.append(
                    f"Field {field_def.name}: expected {field_def.type}, got {type(value).__name__}"
                )

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "fields": [
                {
                    "name": f.name,
                    "type": f.type,
                    "required": f.required,
                    "default": f.default,
                    "description": f.description,
                }
                for f in self.fields
            ],
        }


class ExportPipeline:
    """Pipeline for composing export operations."""

    def __init__(self):
        """Initialize pipeline."""
        self._steps: list[Callable[[Any], Any]] = []
        self._config = ExportConfig()

    def configure(self, config: ExportConfig) -> "ExportPipeline":
        """Set export configuration.

        Args:
            config: Export configuration.

        Returns:
            Self for chaining.
        """
        self._config = config
        return self

    def filter(self, predicate: Callable[[Any], bool]) -> "ExportPipeline":
        """Add filter step.

        Args:
            predicate: Filter function.

        Returns:
            Self for chaining.
        """

        def filter_step(data):
            if isinstance(data, (list, tuple)):
                return [item for item in data if predicate(item)]
            return data if predicate(data) else []

        self._steps.append(filter_step)
        return self

    def transform(self, transformer: Callable[[Any], Any]) -> "ExportPipeline":
        """Add transform step.

        Args:
            transformer: Transform function.

        Returns:
            Self for chaining.
        """

        def transform_step(data):
            if isinstance(data, (list, tuple)):
                return [transformer(item) for item in data]
            return transformer(data)

        self._steps.append(transform_step)
        return self

    def select(self, fields: list[str]) -> "ExportPipeline":
        """Select specific fields.

        Args:
            fields: Fields to include.

        Returns:
            Self for chaining.
        """

        def select_step(data):
            def select_from_record(record):
                if isinstance(record, dict):
                    return {k: v for k, v in record.items() if k in fields}
                return record

            if isinstance(data, (list, tuple)):
                return [select_from_record(item) for item in data]
            return select_from_record(data)

        self._steps.append(select_step)
        return self

    def rename(self, mapping: dict[str, str]) -> "ExportPipeline":
        """Rename fields.

        Args:
            mapping: Old name to new name mapping.

        Returns:
            Self for chaining.
        """

        def rename_step(data):
            def rename_record(record):
                if isinstance(record, dict):
                    return {mapping.get(k, k): v for k, v in record.items()}
                return record

            if isinstance(data, (list, tuple)):
                return [rename_record(item) for item in data]
            return rename_record(data)

        self._steps.append(rename_step)
        return self

    def sort(self, key: str, reverse: bool = False) -> "ExportPipeline":
        """Sort data by field.

        Args:
            key: Field to sort by.
            reverse: Sort in descending order.

        Returns:
            Self for chaining.
        """

        def sort_step(data):
            if isinstance(data, (list, tuple)):
                return sorted(
                    data,
                    key=lambda x: x.get(key, "") if isinstance(x, dict) else x,
                    reverse=reverse,
                )
            return data

        self._steps.append(sort_step)
        return self

    def limit(self, n: int) -> "ExportPipeline":
        """Limit number of records.

        Args:
            n: Maximum records.

        Returns:
            Self for chaining.
        """

        def limit_step(data):
            if isinstance(data, (list, tuple)):
                return data[:n]
            return data

        self._steps.append(limit_step)
        return self

    def execute(self, data: Any) -> Any:
        """Execute pipeline on data.

        Args:
            data: Input data.

        Returns:
            Processed data.
        """
        result = data
        for step in self._steps:
            result = step(result)
        return result

    def export_to(
        self,
        data: Any,
        output: Union[str, Path],
        format: Optional[ExportFormat] = None,
    ) -> Path:
        """Execute pipeline and export to file.

        Args:
            data: Input data.
            output: Output path.
            format: Export format (auto-detect from extension if None).

        Returns:
            Output path.
        """
        processed = self.execute(data)
        output_path = Path(output)

        # Auto-detect format from extension
        if format is None:
            format = {
                ".json": ExportFormat.JSON,
                ".jsonl": ExportFormat.JSONL,
                ".csv": ExportFormat.CSV,
                ".tsv": ExportFormat.TSV,
                ".md": ExportFormat.MARKDOWN,
            }.get(output_path.suffix.lower(), ExportFormat.JSON)

        # Get appropriate exporter
        exporter = get_exporter(format, self._config)
        exporter.export(processed, output_path)

        return output_path


def get_exporter(format: ExportFormat, config: Optional[ExportConfig] = None) -> Exporter:
    """Get exporter for format.

    Args:
        format: Export format.
        config: Export configuration.

    Returns:
        Exporter instance.
    """
    exporters = {
        ExportFormat.JSON: JSONExporter,
        ExportFormat.JSONL: JSONLExporter,
        ExportFormat.CSV: CSVExporter,
        ExportFormat.TSV: lambda c: CSVExporter(c, delimiter="\t"),
        ExportFormat.MARKDOWN: MarkdownExporter,
    }

    exporter_class = exporters.get(format)
    if exporter_class is None:
        raise ValueError(f"Unsupported format: {format}")

    return exporter_class(config)


def export_to_json(data: Any, output: Union[str, Path, TextIO], **kwargs) -> None:
    """Quick export to JSON.

    Args:
        data: Data to export.
        output: Output destination.
        **kwargs: Additional options for ExportConfig.
    """
    config = ExportConfig(format=ExportFormat.JSON, **kwargs)
    exporter = JSONExporter(config)
    exporter.export(data, output)


def export_to_jsonl(data: Any, output: Union[str, Path, TextIO], **kwargs) -> None:
    """Quick export to JSONL.

    Args:
        data: Data to export.
        output: Output destination.
        **kwargs: Additional options for ExportConfig.
    """
    config = ExportConfig(format=ExportFormat.JSONL, **kwargs)
    exporter = JSONLExporter(config)
    exporter.export(data, output)


def export_to_csv(data: Any, output: Union[str, Path, TextIO], **kwargs) -> None:
    """Quick export to CSV.

    Args:
        data: Data to export.
        output: Output destination.
        **kwargs: Additional options for ExportConfig.
    """
    config = ExportConfig(format=ExportFormat.CSV, **kwargs)
    exporter = CSVExporter(config)
    exporter.export(data, output)


def export_to_markdown(data: Any, output: Union[str, Path, TextIO], **kwargs) -> None:
    """Quick export to Markdown table.

    Args:
        data: Data to export.
        output: Output destination.
        **kwargs: Additional options for ExportConfig.
    """
    config = ExportConfig(format=ExportFormat.MARKDOWN, **kwargs)
    exporter = MarkdownExporter(config)
    exporter.export(data, output)


def stream_export(
    data: Iterable[Any],
    output: Union[str, Path],
    format: ExportFormat = ExportFormat.JSONL,
    config: Optional[ExportConfig] = None,
) -> int:
    """Stream export for large datasets.

    Args:
        data: Iterable data source.
        output: Output path.
        format: Export format (JSONL recommended for streaming).
        config: Export configuration.

    Returns:
        Number of records written.
    """
    config = config or ExportConfig()

    if format == ExportFormat.JSONL:
        exporter = JSONLExporter(config)
        return exporter.export_stream(data, output)
    else:
        # Fall back to regular export (loads all into memory)
        exporter = get_exporter(format, config)
        data_list = list(data)
        exporter.export(data_list, output)
        return len(data_list)


def create_export_bundle(
    data: Any,
    output_dir: Union[str, Path],
    name: str,
    formats: Optional[list[ExportFormat]] = None,
    include_schema: bool = True,
    compress: bool = True,
) -> Path:
    """Create export bundle with multiple formats.

    Args:
        data: Data to export.
        output_dir: Output directory.
        name: Bundle name.
        formats: Formats to include (default: JSON, CSV).
        include_schema: Include schema file.
        compress: Create compressed archive.

    Returns:
        Path to bundle (archive or directory).
    """
    output_dir = Path(output_dir)
    bundle_dir = output_dir / name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    formats = formats or [ExportFormat.JSON, ExportFormat.CSV]
    config = ExportConfig()

    files = []

    # Export in each format
    for fmt in formats:
        ext = {
            ExportFormat.JSON: ".json",
            ExportFormat.JSONL: ".jsonl",
            ExportFormat.CSV: ".csv",
            ExportFormat.TSV: ".tsv",
            ExportFormat.MARKDOWN: ".md",
        }.get(fmt, ".txt")

        file_path = bundle_dir / f"{name}{ext}"
        exporter = get_exporter(fmt, config)
        exporter.export(data, file_path)
        files.append(file_path)

    # Add metadata
    prepared = JSONExporter(config)._prepare_data(data)
    metadata = ExportMetadata(
        record_count=len(prepared),
        format=",".join(f.value for f in formats),
    )
    metadata_path = bundle_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)
    files.append(metadata_path)

    # Add schema if requested
    if include_schema and prepared:
        schema_fields = []
        for key, value in prepared[0].items():
            field_type = "any"
            if isinstance(value, str):
                field_type = "string"
            elif isinstance(value, bool):
                field_type = "bool"
            elif isinstance(value, int):
                field_type = "int"
            elif isinstance(value, float):
                field_type = "float"
            elif isinstance(value, list):
                field_type = "list"
            elif isinstance(value, dict):
                field_type = "dict"

            schema_fields.append(SchemaField(name=key, type=field_type))

        schema = DataSchema(
            name=name,
            version="1.0",
            fields=schema_fields,
        )
        schema_path = bundle_dir / "schema.json"
        with open(schema_path, "w") as f:
            json.dump(schema.to_dict(), f, indent=2)
        files.append(schema_path)

    # Compress if requested
    if compress:
        archiver = DataArchiver(CompressionType.ZIP)
        archive_path = output_dir / f"{name}.zip"
        archiver.create_archive(files, archive_path, bundle_dir)
        shutil.rmtree(bundle_dir)
        return archive_path

    return bundle_dir
