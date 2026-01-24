"""
Compatibility shim for insideLLMs.analysis.export.

This module provides backward-compatible access to the data export and
serialization utilities from the ``insideLLMs.analysis.export`` module.
All public symbols are re-exported here to maintain API stability for
existing code that imports from ``insideLLMs.export``.

Overview
--------
The export module provides comprehensive tools for serializing and exporting
LLM experiment data in multiple formats. It supports:

* **Multi-format export**: JSON, JSONL, CSV, TSV, Markdown, and more
* **Data compression**: GZIP, ZIP, and BZIP2 compression
* **Schema validation**: Define and validate data schemas
* **Streaming exports**: Memory-efficient export for large datasets
* **Export pipelines**: Composable data transformation and export workflows
* **Export bundles**: Create multi-format archives with metadata and schemas

Re-exported Classes
-------------------
ExportFormat : Enum
    Enumeration of supported export formats (JSON, JSONL, CSV, TSV, YAML,
    PARQUET, EXCEL, MARKDOWN).

CompressionType : Enum
    Enumeration of supported compression types (NONE, GZIP, ZIP, BZIP2).

ExportConfig : dataclass
    Configuration options for data export operations including format,
    compression, encoding, and formatting preferences.

ExportMetadata : dataclass
    Metadata container for exported data including timestamps, versions,
    record counts, and custom fields.

Serializable : Protocol
    Protocol defining the interface for serializable objects (requires
    ``to_dict()`` method).

Exporter : ABC
    Abstract base class for all format-specific exporters.

JSONExporter : Exporter
    Exports data to JSON format with pretty-printing support.

JSONLExporter : Exporter
    Exports data to JSON Lines format with streaming support.

CSVExporter : Exporter
    Exports data to CSV/TSV format with configurable delimiters.

MarkdownExporter : Exporter
    Exports data to Markdown table format.

DataArchiver : class
    Creates and extracts compressed archives of exported data.

SchemaField : dataclass
    Definition of a single field in a data schema.

DataSchema : dataclass
    Complete schema definition for validating exported data.

ExportPipeline : class
    Fluent interface for composing data transformation and export operations.

Re-exported Functions
---------------------
serialize_value(value, config)
    Serialize a single value according to export configuration.

serialize_record(record, config)
    Serialize a complete record (dict, dataclass, or object) for export.

get_exporter(format, config)
    Factory function to get an appropriate exporter for a given format.

export_to_json(data, output, **kwargs)
    Quick export to JSON format.

export_to_jsonl(data, output, **kwargs)
    Quick export to JSON Lines format.

export_to_csv(data, output, **kwargs)
    Quick export to CSV format.

export_to_markdown(data, output, **kwargs)
    Quick export to Markdown table format.

stream_export(data, output, format, config)
    Memory-efficient streaming export for large iterables.

create_export_bundle(data, output_dir, name, ...)
    Create a complete export bundle with multiple formats, metadata, and schema.

Examples
--------
Basic JSON Export
~~~~~~~~~~~~~~~~~
Export experiment results to a JSON file:

>>> from insideLLMs.export import export_to_json
>>> results = [
...     {"model": "gpt-4", "accuracy": 0.95, "latency_ms": 120},
...     {"model": "claude-3", "accuracy": 0.93, "latency_ms": 85},
... ]
>>> export_to_json(results, "experiment_results.json", pretty_print=True)

Using Exporters Directly
~~~~~~~~~~~~~~~~~~~~~~~~
For more control, use exporter classes directly:

>>> from insideLLMs.export import JSONExporter, ExportConfig
>>> config = ExportConfig(
...     pretty_print=True,
...     date_format="%Y-%m-%d",
...     encoding="utf-8"
... )
>>> exporter = JSONExporter(config)
>>> exporter.export(results, "results.json")

Export to String
~~~~~~~~~~~~~~~~
Export data to a string for further processing:

>>> from insideLLMs.export import JSONLExporter
>>> exporter = JSONLExporter()
>>> jsonl_string = exporter.export_string(results)
>>> print(jsonl_string)
{"model": "gpt-4", "accuracy": 0.95, "latency_ms": 120}
{"model": "claude-3", "accuracy": 0.93, "latency_ms": 85}

CSV Export with Custom Delimiter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Export to CSV with tab delimiter (TSV):

>>> from insideLLMs.export import CSVExporter, ExportConfig
>>> config = ExportConfig(null_value="N/A")
>>> exporter = CSVExporter(config, delimiter="\\t")
>>> exporter.export(results, "results.tsv")

Markdown Table Export
~~~~~~~~~~~~~~~~~~~~~
Create a Markdown table for documentation:

>>> from insideLLMs.export import export_to_markdown
>>> export_to_markdown(results, "results.md")
>>> # Produces:
>>> # | model | accuracy | latency_ms |
>>> # | --- | --- | --- |
>>> # | gpt-4 | 0.95 | 120 |
>>> # | claude-3 | 0.93 | 85 |

Streaming Export for Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Export large datasets without loading everything into memory:

>>> from insideLLMs.export import JSONLExporter
>>> def generate_records():
...     for i in range(1000000):
...         yield {"id": i, "value": i * 2}
>>> exporter = JSONLExporter()
>>> count = exporter.export_stream(generate_records(), "large_dataset.jsonl")
>>> print(f"Exported {count} records")
Exported 1000000 records

Using Export Pipelines
~~~~~~~~~~~~~~~~~~~~~~
Build complex export workflows with the fluent pipeline API:

>>> from insideLLMs.export import ExportPipeline, ExportFormat
>>> pipeline = (
...     ExportPipeline()
...     .filter(lambda x: x.get("accuracy", 0) > 0.9)
...     .transform(lambda x: {**x, "accuracy_pct": x["accuracy"] * 100})
...     .select(["model", "accuracy_pct"])
...     .rename({"model": "Model Name", "accuracy_pct": "Accuracy (%)"})
...     .sort("Accuracy (%)", reverse=True)
...     .limit(10)
... )
>>> pipeline.export_to(results, "top_models.csv")

Creating Export Bundles
~~~~~~~~~~~~~~~~~~~~~~~
Create a complete export package with multiple formats:

>>> from insideLLMs.export import create_export_bundle, ExportFormat
>>> bundle_path = create_export_bundle(
...     data=results,
...     output_dir="./exports",
...     name="experiment_2024",
...     formats=[ExportFormat.JSON, ExportFormat.CSV, ExportFormat.MARKDOWN],
...     include_schema=True,
...     compress=True
... )
>>> print(f"Bundle created at: {bundle_path}")
Bundle created at: ./exports/experiment_2024.zip

Data Compression
~~~~~~~~~~~~~~~~
Compress exported files:

>>> from insideLLMs.export import DataArchiver, CompressionType
>>> archiver = DataArchiver(CompressionType.GZIP)
>>> compressed_path = archiver.compress_file("results.json")
>>> print(f"Compressed to: {compressed_path}")
Compressed to: results.json.gz

Decompress files:

>>> decompressed_path = archiver.decompress_file("results.json.gz")

Schema Validation
~~~~~~~~~~~~~~~~~
Define and validate data schemas:

>>> from insideLLMs.export import DataSchema, SchemaField
>>> schema = DataSchema(
...     name="ExperimentResult",
...     version="1.0",
...     description="Schema for experiment results",
...     fields=[
...         SchemaField(name="model", type="string", required=True),
...         SchemaField(name="accuracy", type="float", required=True),
...         SchemaField(name="latency_ms", type="int", required=False, default=0),
...     ]
... )
>>> errors = schema.validate({"model": "gpt-4", "accuracy": 0.95})
>>> if errors:
...     print(f"Validation errors: {errors}")
... else:
...     print("Record is valid")
Record is valid

Exporting Dataclasses
~~~~~~~~~~~~~~~~~~~~~
Dataclasses are automatically serialized:

>>> from dataclasses import dataclass
>>> from insideLLMs.export import export_to_json
>>> @dataclass
... class ExperimentResult:
...     model: str
...     accuracy: float
...     latency_ms: int
>>> results = [
...     ExperimentResult("gpt-4", 0.95, 120),
...     ExperimentResult("claude-3", 0.93, 85),
... ]
>>> export_to_json(results, "results.json")

Custom Serializable Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implement the ``to_dict()`` method for custom serialization:

>>> from insideLLMs.export import export_to_json
>>> class ModelMetrics:
...     def __init__(self, name, metrics):
...         self.name = name
...         self.metrics = metrics
...     def to_dict(self):
...         return {"name": self.name, **self.metrics}
>>> metrics = ModelMetrics("gpt-4", {"accuracy": 0.95, "f1": 0.94})
>>> export_to_json(metrics, "metrics.json")

Notes
-----
Migration Path
    This module is a compatibility shim. New code should prefer importing
    directly from ``insideLLMs.analysis.export`` for explicit dependency
    management. This shim will be maintained for backward compatibility.

Format Support
    Not all formats support all features. For example:

    - Streaming export is only efficient with JSONL format
    - CSV/TSV flattens nested structures to JSON strings
    - Markdown tables work best with simple, flat records

Performance Considerations
    For datasets with millions of records:

    - Use ``stream_export()`` with JSONL format
    - Use ``JSONLExporter.export_stream()`` for maximum control
    - Avoid loading entire datasets into memory

Thread Safety
    Exporter instances are not thread-safe. Create separate instances
    for concurrent export operations.

See Also
--------
insideLLMs.analysis.export : The canonical implementation module
insideLLMs.schemas : Schema validation and registry utilities
insideLLMs.analysis.metrics : Metric computation before export
"""

from insideLLMs.analysis.export import *  # noqa: F401,F403
