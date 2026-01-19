Output Schemas & Versioning
===========================

insideLLMs treats serialized outputs (results, harness records, exports) as a
**public contract**. To support reproducibility and API stability, outputs can be
validated against **versioned schemas**.

Schema versioning
-----------------

Schemas are versioned using **SemVer** (e.g. ``1.0.0``). A schema version is
embedded in emitted outputs as ``schema_version``.

Pydantic dependency
-------------------

Schema validation uses **Pydantic** and is optional. If you enable validation
without Pydantic installed, insideLLMs will raise an informative error.

Available schema names
----------------------

The following schema names are supported via ``insideLLMs.schemas.SchemaRegistry``:

- ``ProbeResult``: one probe execution result item
- ``RunnerOutput``: a batch wrapper for multiple ``ProbeResult`` items
- ``HarnessRecord``: a single JSONL record emitted by the harness
- ``HarnessSummary``: summary metadata written by the harness
- ``BenchmarkSummary``: aggregated benchmark output
- ``ComparisonReport``: model comparison output
- ``ExportMetadata``: metadata included in export bundles

Enable validation in Python
---------------------------

Validation is available on key boundaries (runner / results saving / exports).
Typical usage:

- Enable validation when running:
  ``run_experiment_from_config(..., validate_output=True)``
- Enable validation when saving:
  ``save_results_json(..., validate_output=True)``

CLI usage
---------

Run with schema validation enabled:

- ``insidellms run config.yaml --validate-output --schema-version 1.0.0``
- ``insidellms harness harness.yaml --validate-output --schema-version 1.0.0``

Validation modes:

- ``--validation-mode strict`` (default): fail on first mismatch
- ``--validation-mode warn``: emit warnings and continue

Dump JSON Schema
----------------

You can export JSON Schema for external tooling:

- ``insidellms schema list``
- ``insidellms schema dump --name RunnerOutput --version 1.0.0``

Validate existing files
-----------------------

Validate a JSON or JSONL file:

- ``insidellms schema validate --name ProbeResult --version 1.0.0 --input results.json``
- ``insidellms schema validate --name HarnessRecord --version 1.0.0 --input results.jsonl``

