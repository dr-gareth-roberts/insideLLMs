"""
Data export and serialization utilities for LLM experiments.

This module provides a comprehensive toolkit for exporting experimental data
from LLM analysis runs to various formats. It supports both batch and streaming
exports, data compression, schema validation, and flexible pipeline composition
for complex data transformation workflows.

Overview
--------
The export module is organized around several key concepts:

1. **Exporters**: Format-specific classes (JSONExporter, CSVExporter, etc.) that
   handle serialization to different file formats.

2. **Configuration**: The ExportConfig dataclass controls export behavior including
   formatting, encoding, and null value handling.

3. **Pipelines**: The ExportPipeline class enables fluent composition of filter,
   transform, and export operations.

4. **Archiving**: The DataArchiver class provides compression and archive creation
   capabilities for exported data.

5. **Schema Validation**: DataSchema and SchemaField classes enable validation
   of exported data against defined contracts.

Supported Formats
-----------------
- **JSON**: Standard JSON with optional pretty-printing
- **JSONL**: JSON Lines format for streaming and line-by-line processing
- **CSV**: Comma-separated values with configurable delimiters
- **TSV**: Tab-separated values (CSV variant)
- **Markdown**: Markdown tables for documentation and reports
- **YAML**: YAML format (requires PyYAML)
- **Parquet**: Columnar format for large datasets (requires pyarrow)
- **Excel**: Excel spreadsheets (requires openpyxl)

Examples
--------
Basic JSON export:

    >>> from insideLLMs.analysis.export import export_to_json
    >>> data = [
    ...     {"model": "gpt-4", "prompt": "Hello", "response": "Hi there!"},
    ...     {"model": "gpt-4", "prompt": "Goodbye", "response": "Farewell!"}
    ... ]
    >>> export_to_json(data, "/tmp/results.json")

Using an exporter directly with configuration:

    >>> from insideLLMs.analysis.export import JSONExporter, ExportConfig
    >>> config = ExportConfig(pretty_print=True, date_format="%Y-%m-%d")
    >>> exporter = JSONExporter(config)
    >>> json_string = exporter.export_string(data)
    >>> print(json_string[:50])
    [
      {
        "model": "gpt-4",
        "prompt": "Hello"

Export pipeline with transformations:

    >>> from insideLLMs.analysis.export import ExportPipeline, ExportFormat
    >>> pipeline = (
    ...     ExportPipeline()
    ...     .filter(lambda x: x.get("score", 0) > 0.8)
    ...     .select(["model", "prompt", "score"])
    ...     .sort("score", reverse=True)
    ...     .limit(100)
    ... )
    >>> results = [
    ...     {"model": "gpt-4", "prompt": "test1", "score": 0.9, "metadata": {}},
    ...     {"model": "gpt-4", "prompt": "test2", "score": 0.7, "metadata": {}},
    ...     {"model": "gpt-4", "prompt": "test3", "score": 0.95, "metadata": {}},
    ... ]
    >>> processed = pipeline.execute(results)
    >>> len(processed)
    2
    >>> processed[0]["score"]
    0.95

Streaming export for large datasets:

    >>> from insideLLMs.analysis.export import stream_export, ExportFormat
    >>> def generate_records():
    ...     for i in range(10000):
    ...         yield {"id": i, "value": i * 2}
    >>> count = stream_export(generate_records(), "/tmp/large_data.jsonl")
    >>> print(f"Exported {count} records")
    Exported 10000 records

Creating a compressed export bundle:

    >>> from insideLLMs.analysis.export import create_export_bundle, ExportFormat
    >>> data = [{"experiment": "test", "accuracy": 0.95}]
    >>> bundle_path = create_export_bundle(
    ...     data,
    ...     output_dir="/tmp/exports",
    ...     name="experiment_results",
    ...     formats=[ExportFormat.JSON, ExportFormat.CSV],
    ...     compress=True
    ... )
    >>> print(bundle_path)
    /tmp/exports/experiment_results.zip

Data compression with archiver:

    >>> from insideLLMs.analysis.export import DataArchiver, CompressionType
    >>> archiver = DataArchiver(CompressionType.GZIP)
    >>> compressed_path = archiver.compress_file("/tmp/results.json")
    >>> print(compressed_path)
    /tmp/results.json.gz

Schema validation before export:

    >>> from insideLLMs.analysis.export import DataSchema, SchemaField
    >>> schema = DataSchema(
    ...     name="ExperimentResult",
    ...     version="1.0",
    ...     fields=[
    ...         SchemaField(name="model", type="string", required=True),
    ...         SchemaField(name="score", type="float", required=True),
    ...         SchemaField(name="notes", type="string", required=False, default=""),
    ...     ]
    ... )
    >>> errors = schema.validate({"model": "gpt-4", "score": 0.95})
    >>> len(errors)
    0
    >>> errors = schema.validate({"model": "gpt-4"})  # Missing required field
    >>> "Missing required field: score" in errors
    True

Notes
-----
- For very large datasets (millions of records), prefer JSONL format with
  streaming export to avoid memory issues.
- The module automatically handles serialization of dataclasses, Enums,
  datetime objects, and objects implementing the Serializable protocol.
- Nested structures are automatically flattened for CSV/TSV formats using
  JSON serialization for complex values.
- Export bundles include metadata and optionally inferred schemas for
  documentation and reproducibility.

See Also
--------
insideLLMs.schemas.validator : Schema validation utilities
insideLLMs.schemas.registry : Schema registry for versioned schemas
insideLLMs.analysis.aggregation : Data aggregation before export
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

from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION


class ExportFormat(Enum):
    """Enumeration of supported export formats for data serialization.

    This enum defines all file formats that the export module can produce.
    Each format has different characteristics making it suitable for
    different use cases.

    Parameters
    ----------
    value : str
        The string identifier for the format, used in file extensions
        and format detection.

    Attributes
    ----------
    JSON : str
        Standard JSON format. Best for structured data that needs to be
        human-readable and parsed by other applications. Supports nested
        structures natively.
    JSONL : str
        JSON Lines format (newline-delimited JSON). Each line is a valid
        JSON object. Ideal for streaming large datasets and log files.
        Memory-efficient for processing.
    CSV : str
        Comma-separated values. Universal format for tabular data.
        Compatible with spreadsheet applications. Nested structures are
        JSON-serialized into cell values.
    TSV : str
        Tab-separated values. Similar to CSV but uses tabs as delimiters.
        Useful when data contains commas.
    YAML : str
        YAML Ain't Markup Language. Human-readable format good for
        configuration files. Requires PyYAML package.
    PARQUET : str
        Apache Parquet columnar format. Efficient for large analytical
        datasets. Supports compression and schema evolution. Requires
        pyarrow package.
    EXCEL : str
        Microsoft Excel format (.xlsx). Includes formatting support.
        Requires openpyxl package.
    MARKDOWN : str
        Markdown table format. Useful for documentation, README files,
        and rendering in Markdown-compatible viewers.

    Examples
    --------
    Using format enum for export selection:

        >>> from insideLLMs.analysis.export import ExportFormat, get_exporter
        >>> exporter = get_exporter(ExportFormat.JSON)
        >>> type(exporter).__name__
        'JSONExporter'

    Iterating over available formats:

        >>> from insideLLMs.analysis.export import ExportFormat
        >>> for fmt in ExportFormat:
        ...     print(f"{fmt.name}: {fmt.value}")
        JSON: json
        JSONL: jsonl
        CSV: csv
        TSV: tsv
        YAML: yaml
        PARQUET: parquet
        EXCEL: excel
        MARKDOWN: markdown

    Checking format from string:

        >>> from insideLLMs.analysis.export import ExportFormat
        >>> ExportFormat("json") == ExportFormat.JSON
        True
        >>> ExportFormat("csv").value
        'csv'

    See Also
    --------
    get_exporter : Factory function to create exporters for each format
    ExportConfig : Configuration options that apply to all formats
    """

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"
    YAML = "yaml"
    PARQUET = "parquet"
    EXCEL = "excel"
    MARKDOWN = "markdown"


class CompressionType(Enum):
    """Enumeration of supported compression algorithms for data archiving.

    This enum defines compression methods available through the DataArchiver
    class. Each compression type offers different trade-offs between
    compression ratio, speed, and compatibility.

    Parameters
    ----------
    value : str
        The string identifier for the compression type, typically matching
        the file extension used for compressed files.

    Attributes
    ----------
    NONE : str
        No compression. Data is stored as-is. Use when compression overhead
        is not worth the size reduction, or when data is already compressed.
    GZIP : str
        GNU zip compression. Widely supported, good balance of speed and
        compression ratio. Produces .gz files. Single-file compression only.
    ZIP : str
        ZIP archive format. Supports multiple files in a single archive.
        Universal compatibility across operating systems. Produces .zip files.
    BZIP2 : str
        bzip2 compression. Higher compression ratio than gzip but slower.
        Good for archival purposes. Produces .bz2 files.

    Examples
    --------
    Creating a gzip-compressed file:

        >>> from insideLLMs.analysis.export import DataArchiver, CompressionType
        >>> archiver = DataArchiver(CompressionType.GZIP)
        >>> # archiver.compress_file("data.json")  # Creates data.json.gz

    Creating a ZIP archive with multiple files:

        >>> from insideLLMs.analysis.export import DataArchiver, CompressionType
        >>> archiver = DataArchiver(CompressionType.ZIP)
        >>> # archiver.create_archive(
        >>> #     ["file1.json", "file2.json"],
        >>> #     "archive.zip"
        >>> # )

    Checking compression type from value:

        >>> from insideLLMs.analysis.export import CompressionType
        >>> CompressionType("gzip") == CompressionType.GZIP
        True
        >>> CompressionType.BZIP2.value
        'bz2'

    See Also
    --------
    DataArchiver : Class that uses these compression types
    create_export_bundle : Function that optionally compresses export bundles
    """

    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"
    BZIP2 = "bz2"


@dataclass
class ExportConfig:
    """Configuration settings for data export operations.

    This dataclass encapsulates all configuration options that control how
    data is serialized and formatted during export. It provides sensible
    defaults while allowing fine-grained customization.

    Parameters
    ----------
    format : ExportFormat, optional
        The target export format. Default is ExportFormat.JSON.
    compression : CompressionType, optional
        Compression to apply after export. Default is CompressionType.NONE.
    pretty_print : bool, optional
        Whether to format output with indentation for readability.
        Default is True. Set to False for compact output.
    include_metadata : bool, optional
        Whether to include export metadata (timestamp, version, etc.).
        Default is True.
    date_format : str, optional
        strftime format string for datetime serialization.
        Default is "%Y-%m-%dT%H:%M:%S" (ISO 8601 without timezone).
    null_value : str, optional
        String representation for None/null values in CSV/TSV formats.
        Default is empty string "".
    encoding : str, optional
        Character encoding for text output. Default is "utf-8".
    chunk_size : int, optional
        Number of records to process at a time for streaming exports.
        Default is 1000.

    Attributes
    ----------
    format : ExportFormat
        The configured export format.
    compression : CompressionType
        The configured compression type.
    pretty_print : bool
        Whether pretty-printing is enabled.
    include_metadata : bool
        Whether metadata inclusion is enabled.
    date_format : str
        The datetime format string.
    null_value : str
        The null value representation.
    encoding : str
        The character encoding.
    chunk_size : int
        The chunk size for streaming.

    Examples
    --------
    Default configuration:

        >>> from insideLLMs.analysis.export import ExportConfig
        >>> config = ExportConfig()
        >>> config.format
        <ExportFormat.JSON: 'json'>
        >>> config.pretty_print
        True

    Custom configuration for compact CSV:

        >>> from insideLLMs.analysis.export import ExportConfig, ExportFormat
        >>> config = ExportConfig(
        ...     format=ExportFormat.CSV,
        ...     pretty_print=False,
        ...     null_value="N/A",
        ...     encoding="utf-8"
        ... )
        >>> config.null_value
        'N/A'

    Configuration for date-heavy data:

        >>> from insideLLMs.analysis.export import ExportConfig
        >>> config = ExportConfig(
        ...     date_format="%Y-%m-%d %H:%M:%S",
        ...     include_metadata=True
        ... )
        >>> from datetime import datetime
        >>> dt = datetime(2024, 1, 15, 10, 30, 0)
        >>> dt.strftime(config.date_format)
        '2024-01-15 10:30:00'

    Using with an exporter:

        >>> from insideLLMs.analysis.export import ExportConfig, JSONExporter
        >>> config = ExportConfig(pretty_print=False)
        >>> exporter = JSONExporter(config)
        >>> result = exporter.export_string({"key": "value"})
        >>> result
        '{"key": "value"}'

    See Also
    --------
    ExportFormat : Available export formats
    CompressionType : Available compression types
    Exporter : Base class that uses this configuration
    """

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
    """Metadata container for exported data bundles.

    This dataclass captures metadata about an export operation including
    timestamps, version information, record counts, and custom fields.
    It is automatically generated when creating export bundles and can
    be used to track provenance and ensure reproducibility.

    Parameters
    ----------
    export_time : str, optional
        ISO 8601 timestamp of when the export was created.
        Default is generated at instantiation time.
    version : str, optional
        Version of the export format/structure. Default is "1.0".
    schema_version : str, optional
        Version of the data schema used. Default is DEFAULT_SCHEMA_VERSION
        from the schemas module.
    source : str, optional
        Identifier for the data source. Default is "insideLLMs".
    record_count : int, optional
        Number of records in the export. Default is 0.
    format : str, optional
        Comma-separated list of formats in the export. Default is "".
    compression : str, optional
        Compression type used. Default is "none".
    custom : dict[str, Any], optional
        Dictionary for arbitrary custom metadata. Default is empty dict.

    Attributes
    ----------
    export_time : str
        The export timestamp.
    version : str
        The export format version.
    schema_version : str
        The schema version.
    source : str
        The data source identifier.
    record_count : int
        The number of exported records.
    format : str
        The export format(s).
    compression : str
        The compression type.
    custom : dict[str, Any]
        Custom metadata fields.

    Examples
    --------
    Creating metadata with defaults:

        >>> from insideLLMs.analysis.export import ExportMetadata
        >>> metadata = ExportMetadata()
        >>> metadata.source
        'insideLLMs'
        >>> metadata.schema_version
        '1.0.1'

    Creating metadata with custom values:

        >>> from insideLLMs.analysis.export import ExportMetadata
        >>> metadata = ExportMetadata(
        ...     record_count=1500,
        ...     format="json,csv",
        ...     compression="gzip",
        ...     custom={"experiment_id": "exp-001", "model": "gpt-4"}
        ... )
        >>> metadata.record_count
        1500
        >>> metadata.custom["experiment_id"]
        'exp-001'

    Converting to dictionary for serialization:

        >>> from insideLLMs.analysis.export import ExportMetadata
        >>> metadata = ExportMetadata(record_count=100, format="json")
        >>> meta_dict = metadata.to_dict()
        >>> meta_dict["record_count"]
        100
        >>> "export_time" in meta_dict
        True

    Using in export bundle:

        >>> from insideLLMs.analysis.export import ExportMetadata
        >>> import json
        >>> metadata = ExportMetadata(
        ...     record_count=50,
        ...     format="json",
        ...     custom={"run_id": "run-123"}
        ... )
        >>> json_str = json.dumps(metadata.to_dict(), indent=2)
        >>> "run_id" in json_str
        True

    See Also
    --------
    create_export_bundle : Function that generates export metadata
    ExportConfig : Configuration that influences metadata generation
    """

    export_time: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    schema_version: str = DEFAULT_SCHEMA_VERSION
    source: str = "insideLLMs"
    record_count: int = 0
    format: str = ""
    compression: str = "none"
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all metadata fields with their current
            values. Suitable for JSON serialization.

        Examples
        --------
        Basic conversion:

            >>> from insideLLMs.analysis.export import ExportMetadata
            >>> metadata = ExportMetadata(record_count=42)
            >>> d = metadata.to_dict()
            >>> d["record_count"]
            42
            >>> isinstance(d, dict)
            True

        Serializing to JSON:

            >>> import json
            >>> from insideLLMs.analysis.export import ExportMetadata
            >>> metadata = ExportMetadata(format="csv")
            >>> json_str = json.dumps(metadata.to_dict())
            >>> '"format": "csv"' in json_str
            True
        """
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
    """Protocol defining the interface for serializable objects.

    This protocol specifies that any object implementing it must provide
    a `to_dict()` method that returns a dictionary representation. Objects
    conforming to this protocol can be automatically serialized by the
    export utilities.

    The protocol is used for structural subtyping (duck typing with type
    checking support). Any class with a matching `to_dict()` method is
    considered to implement this protocol, without explicit inheritance.

    Methods
    -------
    to_dict()
        Convert the object to a dictionary representation.

    Examples
    --------
    Creating a class that implements the protocol:

        >>> from insideLLMs.analysis.export import Serializable
        >>> class ExperimentResult:
        ...     def __init__(self, model: str, score: float):
        ...         self.model = model
        ...         self.score = score
        ...     def to_dict(self) -> dict:
        ...         return {"model": self.model, "score": self.score}
        >>> result = ExperimentResult("gpt-4", 0.95)
        >>> result.to_dict()
        {'model': 'gpt-4', 'score': 0.95}

    Using with serialize_record:

        >>> from insideLLMs.analysis.export import serialize_record, ExportConfig
        >>> class CustomData:
        ...     def to_dict(self):
        ...         return {"type": "custom", "value": 42}
        >>> config = ExportConfig()
        >>> serialize_record(CustomData(), config)
        {'type': 'custom', 'value': 42}

    Type checking example:

        >>> from insideLLMs.analysis.export import Serializable
        >>> def export_item(item: Serializable) -> dict:
        ...     return item.to_dict()
        >>> class MyData:
        ...     def to_dict(self):
        ...         return {"data": "value"}
        >>> export_item(MyData())  # Type checker accepts this
        {'data': 'value'}

    See Also
    --------
    serialize_record : Function that uses this protocol
    serialize_value : Function that handles Serializable objects
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert the object to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the object's data in a format suitable
            for JSON serialization.
        """
        ...


def serialize_value(value: Any, config: ExportConfig) -> Any:
    """Serialize a single value for export.

    This function handles the conversion of Python values to serializable
    representations based on the export configuration. It recursively
    processes nested structures and handles special types like datetime,
    Enum, dataclasses, and objects implementing the Serializable protocol.

    Parameters
    ----------
    value : Any
        The value to serialize. Can be any Python object including:
        - None (converted based on format)
        - datetime objects (formatted according to config.date_format)
        - Enum members (converted to their value)
        - dataclasses (converted to dict via asdict)
        - Serializable objects (converted via to_dict method)
        - lists/tuples (recursively serialized)
        - dicts (recursively serialized)
        - primitives (returned as-is)
    config : ExportConfig
        Export configuration controlling serialization behavior.

    Returns
    -------
    Any
        The serialized value. The return type depends on the input:
        - None values return config.null_value for CSV/TSV, else None
        - datetime returns a formatted string
        - Enum returns the enum's value attribute
        - dataclasses and Serializable objects return dicts
        - lists/tuples return lists of serialized values
        - dicts return dicts with serialized values
        - primitives return unchanged

    Examples
    --------
    Serializing None values:

        >>> from insideLLMs.analysis.export import serialize_value, ExportConfig, ExportFormat
        >>> config_json = ExportConfig(format=ExportFormat.JSON)
        >>> serialize_value(None, config_json) is None
        True
        >>> config_csv = ExportConfig(format=ExportFormat.CSV, null_value="N/A")
        >>> serialize_value(None, config_csv)
        'N/A'

    Serializing datetime:

        >>> from datetime import datetime
        >>> from insideLLMs.analysis.export import serialize_value, ExportConfig
        >>> config = ExportConfig(date_format="%Y-%m-%d")
        >>> dt = datetime(2024, 6, 15, 10, 30, 0)
        >>> serialize_value(dt, config)
        '2024-06-15'

    Serializing Enum:

        >>> from enum import Enum
        >>> from insideLLMs.analysis.export import serialize_value, ExportConfig
        >>> class Status(Enum):
        ...     SUCCESS = "success"
        ...     FAILURE = "failure"
        >>> serialize_value(Status.SUCCESS, ExportConfig())
        'success'

    Serializing nested structures:

        >>> from insideLLMs.analysis.export import serialize_value, ExportConfig
        >>> data = {
        ...     "results": [{"score": 0.9}, {"score": 0.8}],
        ...     "metadata": {"version": "1.0"}
        ... }
        >>> serialize_value(data, ExportConfig())
        {'results': [{'score': 0.9}, {'score': 0.8}], 'metadata': {'version': '1.0'}}

    Serializing dataclass:

        >>> from dataclasses import dataclass
        >>> from insideLLMs.analysis.export import serialize_value, ExportConfig
        >>> @dataclass
        ... class Result:
        ...     name: str
        ...     value: int
        >>> serialize_value(Result("test", 42), ExportConfig())
        {'name': 'test', 'value': 42}

    See Also
    --------
    serialize_record : Higher-level function for serializing complete records
    ExportConfig : Configuration that controls serialization behavior
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
    """Serialize a complete record for export.

    This function converts a record (which may be a dict, dataclass, object
    with to_dict method, or any object with __dict__) into a dictionary
    with all values serialized according to the export configuration.

    This is the primary entry point for serializing individual records before
    export. It first extracts the record's data as a dictionary, then applies
    serialize_value to each value.

    Parameters
    ----------
    record : Any
        The record to serialize. Supported types include:
        - dict: Used directly
        - dataclass instance: Converted via dataclasses.asdict
        - Object with to_dict method: Converted via that method
        - Object with __dict__: Uses the __dict__ attribute
        - Any other value: Wrapped as {"value": record}
    config : ExportConfig
        Export configuration controlling serialization behavior.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the serialized record data. All values
        are processed through serialize_value for consistent formatting.

    Examples
    --------
    Serializing a dictionary:

        >>> from insideLLMs.analysis.export import serialize_record, ExportConfig
        >>> record = {"name": "experiment-1", "score": 0.95, "tags": ["nlp", "gpt"]}
        >>> serialize_record(record, ExportConfig())
        {'name': 'experiment-1', 'score': 0.95, 'tags': ['nlp', 'gpt']}

    Serializing a dataclass:

        >>> from dataclasses import dataclass
        >>> from insideLLMs.analysis.export import serialize_record, ExportConfig
        >>> @dataclass
        ... class Experiment:
        ...     name: str
        ...     accuracy: float
        ...     iterations: int
        >>> exp = Experiment("test-run", 0.92, 100)
        >>> serialize_record(exp, ExportConfig())
        {'name': 'test-run', 'accuracy': 0.92, 'iterations': 100}

    Serializing an object with to_dict:

        >>> from insideLLMs.analysis.export import serialize_record, ExportConfig
        >>> class Result:
        ...     def __init__(self, status, value):
        ...         self.status = status
        ...         self.value = value
        ...     def to_dict(self):
        ...         return {"status": self.status, "value": self.value}
        >>> result = Result("success", 42)
        >>> serialize_record(result, ExportConfig())
        {'status': 'success', 'value': 42}

    Serializing a plain object:

        >>> from insideLLMs.analysis.export import serialize_record, ExportConfig
        >>> class SimpleData:
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = 20
        >>> serialize_record(SimpleData(), ExportConfig())
        {'x': 10, 'y': 20}

    Serializing a primitive value:

        >>> from insideLLMs.analysis.export import serialize_record, ExportConfig
        >>> serialize_record(42, ExportConfig())
        {'value': 42}
        >>> serialize_record("hello", ExportConfig())
        {'value': 'hello'}

    See Also
    --------
    serialize_value : Lower-level function for serializing individual values
    Exporter._prepare_data : Uses this function to prepare data for export
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
    """Abstract base class for data exporters.

    This class defines the interface that all format-specific exporters must
    implement. It provides common functionality for data preparation and
    configuration management while requiring subclasses to implement the
    actual serialization logic.

    Subclasses must implement:
    - export(): Write data to a file or file-like object
    - export_string(): Convert data to a string representation

    The base class provides:
    - Configuration management via ExportConfig
    - Data preparation/normalization via _prepare_data()

    Parameters
    ----------
    config : ExportConfig, optional
        Export configuration. If not provided, default configuration is used.

    Attributes
    ----------
    config : ExportConfig
        The export configuration used by this exporter.

    Examples
    --------
    Creating a custom exporter (subclass implementation):

        >>> from insideLLMs.analysis.export import Exporter, ExportConfig
        >>> from abc import ABC
        >>> class CustomExporter(Exporter):
        ...     def export(self, data, output):
        ...         content = self.export_string(data)
        ...         if isinstance(output, (str, type(None))):
        ...             with open(output, 'w') as f:
        ...                 f.write(content)
        ...         else:
        ...             output.write(content)
        ...     def export_string(self, data):
        ...         prepared = self._prepare_data(data)
        ...         return str(prepared)

    Using a built-in exporter:

        >>> from insideLLMs.analysis.export import JSONExporter, ExportConfig
        >>> config = ExportConfig(pretty_print=True)
        >>> exporter = JSONExporter(config)
        >>> result = exporter.export_string({"key": "value"})
        >>> print(result)
        {
          "key": "value"
        }

    Exporting to file:

        >>> from insideLLMs.analysis.export import JSONExporter
        >>> exporter = JSONExporter()
        >>> # exporter.export([{"a": 1}, {"a": 2}], "/tmp/data.json")

    See Also
    --------
    JSONExporter : JSON format exporter
    JSONLExporter : JSON Lines format exporter
    CSVExporter : CSV format exporter
    MarkdownExporter : Markdown table format exporter
    get_exporter : Factory function to get appropriate exporter
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        """Initialize the exporter with configuration.

        Parameters
        ----------
        config : ExportConfig, optional
            Export configuration controlling formatting, encoding, and other
            export options. If not provided, a default ExportConfig is created.

        Examples
        --------
        Default configuration:

            >>> from insideLLMs.analysis.export import JSONExporter
            >>> exporter = JSONExporter()
            >>> exporter.config.pretty_print
            True

        Custom configuration:

            >>> from insideLLMs.analysis.export import JSONExporter, ExportConfig
            >>> config = ExportConfig(pretty_print=False, encoding="utf-16")
            >>> exporter = JSONExporter(config)
            >>> exporter.config.encoding
            'utf-16'
        """
        self.config = config or ExportConfig()

    @abstractmethod
    def export(self, data: Any, output: Union[str, Path, TextIO, BinaryIO]) -> None:
        """Export data to an output destination.

        This method must be implemented by subclasses to handle the actual
        export process. It should serialize the data and write it to the
        specified output.

        Parameters
        ----------
        data : Any
            The data to export. Can be a single record, list of records,
            or any iterable. Records can be dicts, dataclasses, or objects
            with to_dict methods.
        output : Union[str, Path, TextIO, BinaryIO]
            The output destination. Can be:
            - str or Path: File path to write to
            - TextIO: Text file-like object for text formats
            - BinaryIO: Binary file-like object for binary formats

        Raises
        ------
        IOError
            If the file cannot be written.
        ValueError
            If the data cannot be serialized.

        Examples
        --------
        Export to file path:

            >>> from insideLLMs.analysis.export import JSONExporter
            >>> exporter = JSONExporter()
            >>> # exporter.export({"key": "value"}, "/tmp/output.json")

        Export to file object:

            >>> import io
            >>> from insideLLMs.analysis.export import JSONExporter
            >>> exporter = JSONExporter()
            >>> buffer = io.StringIO()
            >>> exporter.export({"key": "value"}, buffer)
            >>> buffer.getvalue()
            '{\\n  "key": "value"\\n}'
        """
        pass

    @abstractmethod
    def export_string(self, data: Any) -> str:
        """Export data to a string representation.

        This method must be implemented by subclasses to return the
        serialized data as a string. This is useful for in-memory
        processing or when the output destination is not a file.

        Parameters
        ----------
        data : Any
            The data to export. Can be a single record, list of records,
            or any iterable.

        Returns
        -------
        str
            The serialized data as a string in the exporter's format.

        Examples
        --------
        Get JSON string:

            >>> from insideLLMs.analysis.export import JSONExporter
            >>> exporter = JSONExporter()
            >>> result = exporter.export_string({"name": "test"})
            >>> '"name": "test"' in result
            True

        Get CSV string:

            >>> from insideLLMs.analysis.export import CSVExporter
            >>> exporter = CSVExporter()
            >>> result = exporter.export_string([{"a": 1, "b": 2}])
            >>> "a,b" in result
            True
        """
        pass

    def _prepare_data(self, data: Any) -> list[dict[str, Any]]:
        """Prepare and normalize data for export.

        This method converts input data into a consistent list of serialized
        dictionaries suitable for export. It handles various input types
        including single records, lists, and iterables.

        Parameters
        ----------
        data : Any
            Input data to prepare. Can be:
            - dict: Treated as a single record
            - list or tuple: Each item treated as a record
            - iterable: Each item treated as a record
            - other: Treated as a single record

        Returns
        -------
        list[dict[str, Any]]
            List of serialized record dictionaries ready for export.

        Examples
        --------
        Preparing a single dict:

            >>> from insideLLMs.analysis.export import JSONExporter
            >>> exporter = JSONExporter()
            >>> result = exporter._prepare_data({"key": "value"})
            >>> len(result)
            1
            >>> result[0]
            {'key': 'value'}

        Preparing a list of records:

            >>> from insideLLMs.analysis.export import JSONExporter
            >>> exporter = JSONExporter()
            >>> result = exporter._prepare_data([{"a": 1}, {"a": 2}])
            >>> len(result)
            2

        Preparing from generator:

            >>> from insideLLMs.analysis.export import JSONExporter
            >>> exporter = JSONExporter()
            >>> gen = ({"x": i} for i in range(3))
            >>> result = exporter._prepare_data(gen)
            >>> len(result)
            3
        """
        if isinstance(data, dict):
            return [serialize_record(data, self.config)]

        if isinstance(data, (list, tuple)):
            return [serialize_record(item, self.config) for item in data]

        if hasattr(data, "__iter__"):
            return [serialize_record(item, self.config) for item in data]

        return [serialize_record(data, self.config)]


class JSONExporter(Exporter):
    """Export data to JSON format.

    This exporter serializes data to standard JSON (JavaScript Object Notation)
    format. It supports pretty-printing for human readability and handles
    various Python types including dataclasses, Enums, and datetime objects.

    JSON is ideal for structured data interchange and is widely supported
    across programming languages and tools. However, for very large datasets,
    consider JSONLExporter for memory efficiency.

    Parameters
    ----------
    config : ExportConfig, optional
        Export configuration. Key settings for JSON export:
        - pretty_print: Enable indentation (default True)
        - encoding: Character encoding (default "utf-8")

    Attributes
    ----------
    config : ExportConfig
        The export configuration.

    Examples
    --------
    Basic export to file:

        >>> from insideLLMs.analysis.export import JSONExporter
        >>> exporter = JSONExporter()
        >>> data = [{"model": "gpt-4", "score": 0.95}]
        >>> # exporter.export(data, "/tmp/results.json")

    Export to string:

        >>> from insideLLMs.analysis.export import JSONExporter
        >>> exporter = JSONExporter()
        >>> result = exporter.export_string({"name": "test", "value": 42})
        >>> print(result)
        {
          "name": "test",
          "value": 42
        }

    Compact (non-pretty) output:

        >>> from insideLLMs.analysis.export import JSONExporter, ExportConfig
        >>> config = ExportConfig(pretty_print=False)
        >>> exporter = JSONExporter(config)
        >>> exporter.export_string({"a": 1, "b": 2})
        '{"a": 1, "b": 2}'

    Export list of records:

        >>> from insideLLMs.analysis.export import JSONExporter
        >>> exporter = JSONExporter()
        >>> data = [{"id": 1}, {"id": 2}, {"id": 3}]
        >>> result = exporter.export_string(data)
        >>> result.count('"id"')
        3

    Export to StringIO:

        >>> import io
        >>> from insideLLMs.analysis.export import JSONExporter
        >>> exporter = JSONExporter()
        >>> buffer = io.StringIO()
        >>> exporter.export({"key": "value"}, buffer)
        >>> buffer.getvalue()
        '{\\n  "key": "value"\\n}'

    See Also
    --------
    JSONLExporter : For streaming large datasets
    export_to_json : Convenience function for quick exports
    ExportFormat.JSON : The format enum value
    """

    def export(self, data: Any, output: Union[str, Path, TextIO, BinaryIO]) -> None:
        """Export data to JSON format.

        Serializes the data to JSON and writes it to the specified output
        destination. Single dict inputs are exported as JSON objects;
        lists are exported as JSON arrays.

        Parameters
        ----------
        data : Any
            Data to export. Can be a dict (exported as object), list
            (exported as array), or any serializable type.
        output : Union[str, Path, TextIO, BinaryIO]
            Output destination. File path or file-like object.

        Raises
        ------
        IOError
            If the file cannot be written.
        TypeError
            If the data contains non-serializable types.

        Examples
        --------
        Export dict to file:

            >>> from insideLLMs.analysis.export import JSONExporter
            >>> exporter = JSONExporter()
            >>> # exporter.export({"key": "value"}, "/tmp/out.json")

        Export to open file handle:

            >>> import io
            >>> from insideLLMs.analysis.export import JSONExporter
            >>> buffer = io.StringIO()
            >>> JSONExporter().export([{"a": 1}], buffer)
            >>> '"a": 1' in buffer.getvalue()
            True
        """
        content = self.export_string(data)

        if isinstance(output, (str, Path)):
            with open(output, "w", encoding=self.config.encoding) as f:
                f.write(content)
        else:
            output.write(content)

    def export_string(self, data: Any) -> str:
        """Export data to a JSON string.

        Serializes the data to a JSON-formatted string. Single dict inputs
        are returned as JSON objects; lists are returned as JSON arrays.
        Pretty-printing is controlled by the config.pretty_print setting.

        Parameters
        ----------
        data : Any
            Data to export. Supported types include dict, list, dataclass,
            and objects implementing to_dict().

        Returns
        -------
        str
            JSON-formatted string representation of the data.

        Examples
        --------
        Export single record:

            >>> from insideLLMs.analysis.export import JSONExporter
            >>> exporter = JSONExporter()
            >>> result = exporter.export_string({"name": "test"})
            >>> '"name": "test"' in result
            True

        Export list of records:

            >>> from insideLLMs.analysis.export import JSONExporter
            >>> exporter = JSONExporter()
            >>> result = exporter.export_string([{"x": 1}, {"x": 2}])
            >>> result.startswith('[')
            True

        Compact output:

            >>> from insideLLMs.analysis.export import JSONExporter, ExportConfig
            >>> exporter = JSONExporter(ExportConfig(pretty_print=False))
            >>> exporter.export_string({"a": 1})
            '{"a": 1}'

        Unicode handling:

            >>> from insideLLMs.analysis.export import JSONExporter
            >>> exporter = JSONExporter()
            >>> result = exporter.export_string({"text": "Hello World"})
            >>> "Hello" in result
            True
        """
        prepared = self._prepare_data(data)

        # Unwrap single item
        if len(prepared) == 1 and isinstance(data, dict):
            prepared = prepared[0]

        indent = 2 if self.config.pretty_print else None
        return json.dumps(prepared, indent=indent, ensure_ascii=False, default=str)


class JSONLExporter(Exporter):
    """Export data to JSON Lines (JSONL) format.

    JSON Lines is a text format where each line is a valid JSON value,
    typically a JSON object. This format is ideal for:
    - Streaming large datasets that don't fit in memory
    - Append-only log files
    - Line-by-line processing with tools like grep, awk, or jq
    - Parallel processing where each line is independent

    This exporter provides both batch export (via export/export_string)
    and streaming export (via export_stream) for memory-efficient
    processing of large datasets.

    Parameters
    ----------
    config : ExportConfig, optional
        Export configuration. Key settings:
        - encoding: Character encoding (default "utf-8")
        - chunk_size: Records per chunk for streaming (default 1000)

    Attributes
    ----------
    config : ExportConfig
        The export configuration.

    Examples
    --------
    Basic export to file:

        >>> from insideLLMs.analysis.export import JSONLExporter
        >>> exporter = JSONLExporter()
        >>> data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        >>> # exporter.export(data, "/tmp/users.jsonl")

    Export to string:

        >>> from insideLLMs.analysis.export import JSONLExporter
        >>> exporter = JSONLExporter()
        >>> data = [{"x": 1}, {"x": 2}]
        >>> result = exporter.export_string(data)
        >>> print(result)
        {"x": 1}
        {"x": 2}

    Streaming export for large datasets:

        >>> from insideLLMs.analysis.export import JSONLExporter
        >>> exporter = JSONLExporter()
        >>> def generate_records():
        ...     for i in range(1000):
        ...         yield {"id": i, "value": i * 2}
        >>> # count = exporter.export_stream(generate_records(), "/tmp/large.jsonl")

    Processing JSONL output line by line:

        >>> from insideLLMs.analysis.export import JSONLExporter
        >>> import json
        >>> exporter = JSONLExporter()
        >>> jsonl_str = exporter.export_string([{"a": 1}, {"a": 2}])
        >>> for line in jsonl_str.strip().split('\\n'):
        ...     record = json.loads(line)
        ...     print(record["a"])
        1
        2

    See Also
    --------
    JSONExporter : For standard JSON output
    export_to_jsonl : Convenience function for quick exports
    stream_export : High-level streaming export function
    """

    def export(self, data: Any, output: Union[str, Path, TextIO, BinaryIO]) -> None:
        """Export data to JSONL format.

        Each record in the data is serialized as a single JSON line.
        Lines are separated by newline characters.

        Parameters
        ----------
        data : Any
            Data to export. Should be a list/iterable of records.
            Each record becomes one JSON line.
        output : Union[str, Path, TextIO, BinaryIO]
            Output destination. File path or file-like object.

        Raises
        ------
        IOError
            If the file cannot be written.

        Examples
        --------
        Export to file:

            >>> from insideLLMs.analysis.export import JSONLExporter
            >>> exporter = JSONLExporter()
            >>> # exporter.export([{"a": 1}, {"a": 2}], "/tmp/data.jsonl")

        Export to StringIO:

            >>> import io
            >>> from insideLLMs.analysis.export import JSONLExporter
            >>> buffer = io.StringIO()
            >>> JSONLExporter().export([{"x": 1}], buffer)
            >>> buffer.getvalue()
            '{"x": 1}'
        """
        content = self.export_string(data)

        if isinstance(output, (str, Path)):
            with open(output, "w", encoding=self.config.encoding) as f:
                f.write(content)
        else:
            output.write(content)

    def export_string(self, data: Any) -> str:
        """Export data to a JSONL-formatted string.

        Each record in the input data is converted to a single-line JSON
        string. Records are joined with newline characters.

        Parameters
        ----------
        data : Any
            Data to export. Should be a list/iterable of records.

        Returns
        -------
        str
            JSONL-formatted string with one JSON object per line.

        Examples
        --------
        Export list of dicts:

            >>> from insideLLMs.analysis.export import JSONLExporter
            >>> exporter = JSONLExporter()
            >>> result = exporter.export_string([{"a": 1}, {"b": 2}])
            >>> lines = result.split('\\n')
            >>> len(lines)
            2

        Each line is valid JSON:

            >>> import json
            >>> from insideLLMs.analysis.export import JSONLExporter
            >>> result = JSONLExporter().export_string([{"key": "value"}])
            >>> json.loads(result)
            {'key': 'value'}
        """
        prepared = self._prepare_data(data)
        lines = [json.dumps(record, ensure_ascii=False, default=str) for record in prepared]
        return "\n".join(lines)

    def export_stream(self, data: Iterable[Any], output: Union[str, Path, TextIO]) -> int:
        """Export data as a stream for memory-efficient processing.

        Unlike export() which loads all data into memory, this method
        processes records one at a time from an iterable. This is essential
        for exporting large datasets that would otherwise exceed available
        memory.

        Parameters
        ----------
        data : Iterable[Any]
            Iterable data source (generator, iterator, or any iterable).
            Each item is serialized and written immediately.
        output : Union[str, Path, TextIO]
            Output destination. File path or text file-like object.

        Returns
        -------
        int
            Number of records written.

        Examples
        --------
        Stream from generator:

            >>> from insideLLMs.analysis.export import JSONLExporter
            >>> import io
            >>> exporter = JSONLExporter()
            >>> def gen():
            ...     for i in range(5):
            ...         yield {"num": i}
            >>> buffer = io.StringIO()
            >>> count = exporter.export_stream(gen(), buffer)
            >>> count
            5
            >>> len(buffer.getvalue().strip().split('\\n'))
            5

        Stream large dataset to file:

            >>> from insideLLMs.analysis.export import JSONLExporter
            >>> exporter = JSONLExporter()
            >>> def large_dataset():
            ...     for i in range(1000000):  # 1 million records
            ...         yield {"id": i, "data": f"item_{i}"}
            >>> # count = exporter.export_stream(large_dataset(), "/tmp/large.jsonl")
            >>> # print(f"Exported {count} records")

        Progress tracking with wrapper:

            >>> from insideLLMs.analysis.export import JSONLExporter
            >>> import io
            >>> def records_with_progress(items):
            ...     for i, item in enumerate(items):
            ...         if i % 1000 == 0:
            ...             pass  # print(f"Processing record {i}")
            ...         yield item
            >>> exporter = JSONLExporter()
            >>> data = [{"x": i} for i in range(100)]
            >>> buffer = io.StringIO()
            >>> count = exporter.export_stream(records_with_progress(data), buffer)
            >>> count
            100
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
    """Export data to CSV (Comma-Separated Values) format.

    CSV is a tabular data format widely supported by spreadsheet applications,
    databases, and data analysis tools. This exporter handles the conversion
    of structured records to CSV rows, including:
    - Automatic header generation from field names
    - Flattening of nested structures (serialized as JSON strings)
    - Configurable delimiters and quoting styles
    - Proper handling of null values

    For tab-separated output, use delimiter='\\t' or the TSV format enum.

    Parameters
    ----------
    config : ExportConfig, optional
        Export configuration. Key settings:
        - encoding: Character encoding (default "utf-8")
        - null_value: String for None values (default "")
    delimiter : str, optional
        Field delimiter character. Default is ",".
    quoting : int, optional
        CSV quoting style from csv module. Default is csv.QUOTE_MINIMAL.
        Options: QUOTE_ALL, QUOTE_MINIMAL, QUOTE_NONNUMERIC, QUOTE_NONE.

    Attributes
    ----------
    config : ExportConfig
        The export configuration.
    delimiter : str
        The field delimiter.
    quoting : int
        The CSV quoting style.

    Examples
    --------
    Basic export:

        >>> from insideLLMs.analysis.export import CSVExporter
        >>> exporter = CSVExporter()
        >>> data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        >>> result = exporter.export_string(data)
        >>> print(result)
        name,age
        Alice,30
        Bob,25
        <BLANKLINE>

    Tab-separated values:

        >>> from insideLLMs.analysis.export import CSVExporter
        >>> exporter = CSVExporter(delimiter='\\t')
        >>> result = exporter.export_string([{"a": 1, "b": 2}])
        >>> '\\t' in result
        True

    Handling nested data:

        >>> from insideLLMs.analysis.export import CSVExporter
        >>> exporter = CSVExporter()
        >>> data = [{"id": 1, "metadata": {"key": "value"}}]
        >>> result = exporter.export_string(data)
        >>> '{"key": "value"}' in result
        True

    Custom null value:

        >>> from insideLLMs.analysis.export import CSVExporter, ExportConfig
        >>> config = ExportConfig(null_value="N/A")
        >>> exporter = CSVExporter(config)
        >>> result = exporter.export_string([{"a": 1, "b": None}])
        >>> "N/A" in result
        True

    See Also
    --------
    export_to_csv : Convenience function for quick exports
    ExportFormat.CSV : The format enum value
    ExportFormat.TSV : For tab-separated output
    """

    def __init__(
        self,
        config: Optional[ExportConfig] = None,
        delimiter: str = ",",
        quoting: int = csv.QUOTE_MINIMAL,
    ):
        """Initialize CSV exporter with configuration.

        Parameters
        ----------
        config : ExportConfig, optional
            Export configuration. If not provided, defaults are used.
        delimiter : str, optional
            Field delimiter character. Default is "," for standard CSV.
            Use "\\t" for TSV format.
        quoting : int, optional
            CSV quoting constant from the csv module. Controls when fields
            are quoted in the output. Default is csv.QUOTE_MINIMAL.

        Examples
        --------
        Standard CSV:

            >>> from insideLLMs.analysis.export import CSVExporter
            >>> exporter = CSVExporter()
            >>> exporter.delimiter
            ','

        TSV format:

            >>> from insideLLMs.analysis.export import CSVExporter
            >>> exporter = CSVExporter(delimiter='\\t')
            >>> exporter.delimiter
            '\\t'

        Quote all fields:

            >>> import csv
            >>> from insideLLMs.analysis.export import CSVExporter
            >>> exporter = CSVExporter(quoting=csv.QUOTE_ALL)
            >>> result = exporter.export_string([{"a": "1"}])
            >>> '"a"' in result
            True
        """
        super().__init__(config)
        self.delimiter = delimiter
        self.quoting = quoting

    def export(self, data: Any, output: Union[str, Path, TextIO, BinaryIO]) -> None:
        """Export data to CSV format.

        Writes tabular CSV data to the specified output. The first row
        contains headers derived from the field names of the first record.

        Parameters
        ----------
        data : Any
            Data to export. Should be a list of records with consistent
            field names.
        output : Union[str, Path, TextIO, BinaryIO]
            Output destination. File path or file-like object.

        Raises
        ------
        IOError
            If the file cannot be written.

        Examples
        --------
        Export to file:

            >>> from insideLLMs.analysis.export import CSVExporter
            >>> exporter = CSVExporter()
            >>> # exporter.export([{"a": 1}], "/tmp/data.csv")

        Export to StringIO:

            >>> import io
            >>> from insideLLMs.analysis.export import CSVExporter
            >>> buffer = io.StringIO()
            >>> CSVExporter().export([{"x": 1}], buffer)
            >>> "x" in buffer.getvalue()
            True
        """
        content = self.export_string(data)

        if isinstance(output, (str, Path)):
            with open(output, "w", encoding=self.config.encoding, newline="") as f:
                f.write(content)
        else:
            output.write(content)

    def export_string(self, data: Any) -> str:
        """Export data to a CSV-formatted string.

        Converts the input data to a CSV string with headers and data rows.
        Nested structures (dicts, lists) are JSON-serialized into cell values.

        Parameters
        ----------
        data : Any
            Data to export. Should be a list of records.

        Returns
        -------
        str
            CSV-formatted string with headers and data rows.

        Examples
        --------
        Basic export:

            >>> from insideLLMs.analysis.export import CSVExporter
            >>> exporter = CSVExporter()
            >>> result = exporter.export_string([{"a": 1, "b": 2}])
            >>> "a,b" in result
            True
            >>> "1,2" in result
            True

        Empty data:

            >>> from insideLLMs.analysis.export import CSVExporter
            >>> CSVExporter().export_string([])
            ''

        Multiple records:

            >>> from insideLLMs.analysis.export import CSVExporter
            >>> data = [{"x": i} for i in range(3)]
            >>> result = CSVExporter().export_string(data)
            >>> result.count('\\n')
            3
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
        """Flatten a nested record for CSV output.

        Converts all values to strings suitable for CSV cells. Nested
        structures (dicts, lists) are JSON-serialized. None values are
        replaced with the configured null_value.

        Parameters
        ----------
        record : dict[str, Any]
            Record to flatten.

        Returns
        -------
        dict[str, str]
            Flattened record with all string values.

        Examples
        --------
        Flatten simple record:

            >>> from insideLLMs.analysis.export import CSVExporter
            >>> exporter = CSVExporter()
            >>> exporter._flatten_record({"a": 1, "b": "text"})
            {'a': '1', 'b': 'text'}

        Flatten nested structure:

            >>> from insideLLMs.analysis.export import CSVExporter
            >>> exporter = CSVExporter()
            >>> result = exporter._flatten_record({"data": {"x": 1}})
            >>> result["data"]
            '{"x": 1}'

        Handle None:

            >>> from insideLLMs.analysis.export import CSVExporter, ExportConfig
            >>> exporter = CSVExporter(ExportConfig(null_value="NULL"))
            >>> exporter._flatten_record({"a": None})
            {'a': 'NULL'}
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
    *,
    validate_output: bool = False,
    validate_schema_name: Optional[str] = None,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    validation_mode: str = "strict",
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
        schema_version=schema_version,
    )
    metadata_path = bundle_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        meta_dict = metadata.to_dict()

        if validate_output:
            from insideLLMs.schemas import OutputValidator, SchemaRegistry

            registry = SchemaRegistry()
            validator = OutputValidator(registry=registry)
            validator.validate(
                registry.EXPORT_METADATA,
                meta_dict,
                schema_version=schema_version,
                mode=validation_mode,  # type: ignore[arg-type]
            )

            if validate_schema_name and prepared:
                # Validate the actual payload if a schema is declared.
                if isinstance(prepared, list):
                    for item in prepared:
                        validator.validate(
                            validate_schema_name,
                            item,
                            schema_version=schema_version,
                            mode=validation_mode,  # type: ignore[arg-type]
                        )
                else:
                    validator.validate(
                        validate_schema_name,
                        prepared,
                        schema_version=schema_version,
                        mode=validation_mode,  # type: ignore[arg-type]
                    )

        json.dump(meta_dict, f, indent=2)
    files.append(metadata_path)

    # Add schema if requested
    if include_schema and prepared:
        if validate_schema_name:
            # Prefer the versioned, stable schema contract.
            from insideLLMs.schemas import SchemaRegistry

            registry = SchemaRegistry()
            schema = registry.get_json_schema(validate_schema_name, schema_version)
            schema_path = bundle_dir / "schema.json"
            with open(schema_path, "w") as f:
                json.dump(schema, f, indent=2)
            files.append(schema_path)
        else:
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
