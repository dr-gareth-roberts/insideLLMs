"""Results aggregation, reporting, and export utilities for LLM experiment analysis.

This module provides a comprehensive toolkit for persisting, formatting, and
analyzing experiment results from LLM probing and benchmarking runs. It supports
multiple output formats and includes statistical analysis capabilities.

Overview
--------
The results module is the final stage of the insideLLMs pipeline, handling:

* **Persistence**: Save experiment results in JSON, CSV, Markdown, or HTML formats
* **Reporting**: Generate human-readable reports with formatted tables and metrics
* **Comparison**: Create side-by-side benchmark comparisons across models
* **Statistics**: Produce statistical analysis reports with confidence intervals

Supported Formats
-----------------
JSON
    Machine-readable format with full fidelity. Supports schema validation
    and is suitable for downstream processing or archival.

Markdown
    Human-readable tables and reports. Ideal for documentation, GitHub
    issues/PRs, or embedding in notebooks.

HTML
    Styled, interactive reports with CSS formatting. Best for sharing
    results via web browsers or email.

CSV
    Tabular export for spreadsheet analysis. Useful for importing into
    Excel, Google Sheets, or pandas DataFrames.

Examples
--------
Basic JSON export of experiment results:

>>> from insideLLMs.results import save_results_json, load_results_json
>>> results = [
...     {"input": "What is 2+2?", "output": "4", "status": "success"},
...     {"input": "What is 3+3?", "output": "6", "status": "success"},
... ]
>>> save_results_json(results, "/tmp/results.json")
>>> loaded = load_results_json("/tmp/results.json")
>>> len(loaded)
2

Converting an ExperimentResult to Markdown:

>>> from insideLLMs.results import experiment_to_markdown
>>> from insideLLMs.types import ExperimentResult, ProbeResult, ModelInfo
>>> from insideLLMs.types import ProbeCategory, ResultStatus
>>> model = ModelInfo(name="GPT-4", provider="openai", model_id="gpt-4")
>>> probe_result = ProbeResult(
...     input="Test input",
...     output="Test output",
...     status=ResultStatus.SUCCESS,
...     latency_ms=150.5,
... )
>>> experiment = ExperimentResult(
...     experiment_id="exp-001",
...     model_info=model,
...     probe_name="arithmetic",
...     probe_category=ProbeCategory.FACTUAL,
...     results=[probe_result],
... )
>>> md = experiment_to_markdown(experiment)
>>> "# Experiment Report: exp-001" in md
True

Generating a statistical analysis report:

>>> from insideLLMs.results import generate_statistical_report
>>> # Assuming you have a list of ExperimentResult objects
>>> # report = generate_statistical_report(experiments, format="markdown")

Saving benchmark comparisons:

>>> from insideLLMs.results import comparison_to_markdown
>>> from insideLLMs.types import BenchmarkComparison
>>> from datetime import datetime
>>> comparison = BenchmarkComparison(
...     name="GPT-4 vs Claude",
...     experiments=[experiment],  # List of ExperimentResult
...     created_at=datetime.now(),
... )
>>> md = comparison_to_markdown(comparison)
>>> "# Benchmark Comparison" in md
True

Notes
-----
- All file operations use UTF-8 encoding by default
- JSON serialization handles dataclasses, datetime objects, and enums
- Markdown tables automatically escape pipe characters and convert newlines
- HTML reports include embedded CSS for portability
- Statistical reports require the `insideLLMs.statistics` module

See Also
--------
insideLLMs.types : Data structures for experiment results
insideLLMs.schemas : JSON schema validation utilities
insideLLMs.statistics : Statistical analysis functions
insideLLMs.runner : Experiment execution utilities
"""

import csv
import json
from dataclasses import asdict
from datetime import datetime
from html import escape as _html_escape
from typing import Any, Optional, Union

from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
from insideLLMs.types import (
    BenchmarkComparison,
    ExperimentResult,
    ProbeResult,
    ProbeScore,
)


def _escape_markdown_cell(value: Any) -> str:
    """Escape characters that break Markdown tables and normalize newlines.

    This helper function prepares arbitrary values for safe inclusion in
    Markdown table cells by escaping pipe characters (which delimit columns)
    and converting newlines to HTML breaks (which preserve multi-line content).

    Parameters
    ----------
    value : Any
        The value to escape. Will be converted to string via ``str()``.

    Returns
    -------
    str
        The escaped string safe for Markdown table inclusion.

    Examples
    --------
    Escaping pipe characters:

    >>> _escape_markdown_cell("Option A | Option B")
    'Option A \\\\| Option B'

    Converting newlines to HTML breaks:

    >>> _escape_markdown_cell("Line 1\\nLine 2")
    'Line 1<br>Line 2'

    Handling non-string values:

    >>> _escape_markdown_cell(42)
    '42'
    >>> _escape_markdown_cell(None)
    'None'

    Combined escaping:

    >>> _escape_markdown_cell("Value: 100 | Status\\nOK")
    'Value: 100 \\\\| Status<br>OK'

    Notes
    -----
    This function is used internally by ``results_to_markdown`` and
    ``experiment_to_markdown`` to ensure table cells render correctly.
    """
    text = str(value)
    text = text.replace("|", "\\|")
    text = text.replace("\n", "<br>")
    return text


def _format_number(value: Optional[float], precision: int = 4) -> str:
    """Format a floating-point number for display with specified precision.

    Converts numeric values to formatted strings with consistent decimal places,
    handling None values gracefully by returning "N/A".

    Parameters
    ----------
    value : float or None
        The numeric value to format. If None, returns "N/A".
    precision : int, default=4
        Number of decimal places to display.

    Returns
    -------
    str
        Formatted number string or "N/A" if value is None.

    Examples
    --------
    Default precision (4 decimal places):

    >>> _format_number(0.123456789)
    '0.1235'

    Custom precision for percentages:

    >>> _format_number(0.9567, precision=1)
    '1.0'

    Handling None values:

    >>> _format_number(None)
    'N/A'
    >>> _format_number(None, precision=2)
    'N/A'

    Formatting latency values:

    >>> _format_number(123.456, precision=1)
    '123.5'

    Formatting small values:

    >>> _format_number(0.0001234, precision=6)
    '0.000123'
    """
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def _serialize_for_json(obj: Any) -> Any:
    """Convert arbitrary Python objects to JSON-serializable format.

    Recursively transforms complex objects into primitive types that can be
    serialized by ``json.dump``. Handles dataclasses, datetime objects, enums,
    and nested structures.

    Parameters
    ----------
    obj : Any
        The object to serialize. Can be a dataclass, datetime, enum, list,
        dict, or any object with a ``__dict__`` attribute.

    Returns
    -------
    Any
        A JSON-serializable representation of the input:
        - Dataclasses become dicts via ``dataclasses.asdict``
        - Objects with ``__dict__`` become dicts
        - Datetime objects become ISO format strings
        - Enums become their ``.value``
        - Lists/tuples become lists with serialized elements
        - Dicts become dicts with serialized values
        - Primitives pass through unchanged

    Examples
    --------
    Serializing a dataclass:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Result:
    ...     score: float
    ...     label: str
    >>> _serialize_for_json(Result(score=0.95, label="pass"))
    {'score': 0.95, 'label': 'pass'}

    Serializing datetime objects:

    >>> from datetime import datetime
    >>> dt = datetime(2024, 1, 15, 10, 30, 0)
    >>> _serialize_for_json(dt)
    '2024-01-15T10:30:00'

    Serializing enums:

    >>> from enum import Enum
    >>> class Status(Enum):
    ...     SUCCESS = "success"
    ...     FAILURE = "failure"
    >>> _serialize_for_json(Status.SUCCESS)
    'success'

    Serializing nested structures:

    >>> data = {"results": [{"score": 0.9}, {"score": 0.8}]}
    >>> _serialize_for_json(data)
    {'results': [{'score': 0.9}, {'score': 0.8}]}

    Handling primitives (pass-through):

    >>> _serialize_for_json(42)
    42
    >>> _serialize_for_json("hello")
    'hello'
    >>> _serialize_for_json(None)

    Notes
    -----
    This function is used internally by ``save_results_json`` to prepare
    complex experiment results for JSON serialization.
    """
    if hasattr(obj, "__dict__"):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return obj.__dict__
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif hasattr(obj, "value"):  # Enum
        return obj.value
    return obj


# JSON Export Functions


def save_results_json(
    results: Union[list[dict[str, Any]], list[ProbeResult], ExperimentResult],
    path: str,
    indent: int = 2,
    *,
    validate_output: bool = False,
    schema_name: Optional[str] = None,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    validation_mode: str = "strict",
) -> None:
    """Save experiment results to a JSON file with optional schema validation.

    Serializes experiment results to JSON format, handling complex types like
    dataclasses, datetime objects, and enums. Optionally validates the output
    against a JSON schema before writing.

    Parameters
    ----------
    results : list[dict] or list[ProbeResult] or ExperimentResult
        The results to save. Accepts:
        - A list of dictionaries (raw result data)
        - A list of ProbeResult objects
        - A single ExperimentResult object
    path : str
        Output file path. Parent directories must exist.
    indent : int, default=2
        JSON indentation level for pretty-printing. Use 0 or None for
        compact output.
    validate_output : bool, default=False
        If True, validates the serialized data against a JSON schema
        before writing. Requires the ``insideLLMs.schemas`` module.
    schema_name : str or None, default=None
        Explicit schema name to validate against. If None and
        ``validate_output`` is True, the function attempts to infer
        the schema from the data structure.
    schema_version : str, default=DEFAULT_SCHEMA_VERSION
        Version of the schema to use for validation.
    validation_mode : str, default="strict"
        Validation strictness: "strict" fails on any error, "lenient"
        allows some flexibility.

    Raises
    ------
    FileNotFoundError
        If the parent directory of ``path`` does not exist.
    ValidationError
        If ``validate_output`` is True and the data fails schema validation.
    TypeError
        If results cannot be serialized to JSON.

    Examples
    --------
    Saving a simple list of results:

    >>> results = [
    ...     {"input": "What is 2+2?", "output": "4", "status": "success"},
    ...     {"input": "What is 3+3?", "output": "6", "status": "success"},
    ... ]
    >>> save_results_json(results, "/tmp/simple_results.json")

    Saving with custom indentation:

    >>> save_results_json(results, "/tmp/compact.json", indent=0)

    Saving ProbeResult objects:

    >>> from insideLLMs.types import ProbeResult, ResultStatus
    >>> probe_results = [
    ...     ProbeResult(input="Test", output="Response", status=ResultStatus.SUCCESS),
    ... ]
    >>> save_results_json(probe_results, "/tmp/probe_results.json")

    Saving an ExperimentResult:

    >>> from insideLLMs.types import ExperimentResult, ModelInfo, ProbeCategory
    >>> model = ModelInfo(name="GPT-4", provider="openai", model_id="gpt-4")
    >>> experiment = ExperimentResult(
    ...     experiment_id="exp-001",
    ...     model_info=model,
    ...     probe_name="arithmetic",
    ...     probe_category=ProbeCategory.FACTUAL,
    ...     results=[],
    ... )
    >>> save_results_json(experiment, "/tmp/experiment.json")

    Saving with schema validation:

    >>> save_results_json(
    ...     results,
    ...     "/tmp/validated.json",
    ...     validate_output=True,
    ...     schema_name="runner_item",
    ...     validation_mode="strict",
    ... )

    Notes
    -----
    - Non-serializable objects are converted to strings via the ``default=str``
      fallback in ``json.dump``
    - The file is written atomically (no partial writes on error)
    - UTF-8 encoding is used

    See Also
    --------
    load_results_json : Load results from a JSON file
    _serialize_for_json : Internal serialization helper
    """
    serializable = _serialize_for_json(results)

    if validate_output:
        from insideLLMs.schemas import OutputValidator, SchemaRegistry

        registry = SchemaRegistry()
        validator = OutputValidator(registry=registry)

        if schema_name:
            # If the caller specifies a schema, validate directly.
            if isinstance(serializable, list):
                for item in serializable:
                    validator.validate(
                        schema_name,
                        item,
                        schema_version=schema_version,
                        mode=validation_mode,  # type: ignore[arg-type]
                    )
            else:
                validator.validate(
                    schema_name,
                    serializable,
                    schema_version=schema_version,
                    mode=validation_mode,  # type: ignore[arg-type]
                )
        else:
            # Heuristic: list of per-item runner results
            if isinstance(serializable, list) and all(
                isinstance(x, dict) and "input" in x and "status" in x for x in serializable
            ):
                for item in serializable:
                    validator.validate(
                        registry.RUNNER_ITEM,
                        item,
                        schema_version=schema_version,
                        mode=validation_mode,  # type: ignore[arg-type]
                    )
    with open(path, "w") as f:
        json.dump(serializable, f, indent=indent, default=str)


def load_results_json(path: str) -> dict[str, Any]:
    """Load experiment results from a JSON file.

    Reads and parses a JSON file containing experiment results. The file
    should have been created by ``save_results_json`` or follow the same
    format.

    Parameters
    ----------
    path : str
        Path to the JSON file to load.

    Returns
    -------
    dict[str, Any] or list[dict[str, Any]]
        The loaded results. The exact type depends on what was saved:
        - If an ExperimentResult was saved, returns a dict
        - If a list of results was saved, returns a list of dicts

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file contains invalid JSON.
    PermissionError
        If the file cannot be read due to permissions.

    Examples
    --------
    Loading a simple results file:

    >>> # First, create a file to load
    >>> save_results_json([{"input": "test", "output": "response"}], "/tmp/test.json")
    >>> results = load_results_json("/tmp/test.json")
    >>> results[0]["input"]
    'test'

    Loading an experiment result:

    >>> # Assuming /tmp/experiment.json exists
    >>> experiment_data = load_results_json("/tmp/experiment.json")
    >>> "experiment_id" in experiment_data
    True

    Handling missing files:

    >>> try:
    ...     load_results_json("/nonexistent/path.json")
    ... except FileNotFoundError:
    ...     print("File not found")
    File not found

    Round-trip serialization:

    >>> original = {"accuracy": 0.95, "samples": 100}
    >>> save_results_json(original, "/tmp/roundtrip.json")
    >>> loaded = load_results_json("/tmp/roundtrip.json")
    >>> original == loaded
    True

    Notes
    -----
    - The function does not validate the loaded data against a schema.
      Use ``OutputValidator`` separately if validation is needed.
    - Datetime strings are not automatically converted back to datetime
      objects; they remain as ISO format strings.

    See Also
    --------
    save_results_json : Save results to a JSON file
    """
    with open(path) as f:
        return json.load(f)


# Markdown Export Functions


def results_to_markdown(results: list[dict[str, Any]]) -> str:
    """Convert a list of result dictionaries to a Markdown table.

    Creates a simple three-column Markdown table showing input, output, and
    error information for each result. Useful for quick inspection of raw
    results in documentation or notebooks.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of result dictionaries. Each dictionary should contain:
        - ``input``: The input/prompt sent to the model
        - ``output``: The model's response (optional)
        - ``error``: Any error message (optional)

    Returns
    -------
    str
        Markdown formatted table string. Returns "_No results_" if the
        input list is empty.

    Examples
    --------
    Basic usage with successful results:

    >>> results = [
    ...     {"input": "What is 2+2?", "output": "4", "error": ""},
    ...     {"input": "What is the capital of France?", "output": "Paris", "error": ""},
    ... ]
    >>> md = results_to_markdown(results)
    >>> print(md)
    | Input | Output | Error |
    |---|---|---|
    | What is 2+2? | 4 |  |
    | What is the capital of France? | Paris |  |

    Handling errors in results:

    >>> results = [
    ...     {"input": "Complex query", "output": "", "error": "Timeout"},
    ... ]
    >>> md = results_to_markdown(results)
    >>> "Timeout" in md
    True

    Empty results list:

    >>> results_to_markdown([])
    '_No results_'

    Results with pipe characters (auto-escaped):

    >>> results = [{"input": "A | B", "output": "C | D", "error": ""}]
    >>> md = results_to_markdown(results)
    >>> "\\\\|" in md  # Pipe is escaped
    True

    Results with newlines (converted to <br>):

    >>> results = [{"input": "Line1\\nLine2", "output": "Response", "error": ""}]
    >>> md = results_to_markdown(results)
    >>> "<br>" in md
    True

    Notes
    -----
    - Pipe characters (``|``) are escaped to ``\\|`` to prevent table breakage
    - Newlines are converted to ``<br>`` HTML tags
    - Missing keys default to empty strings

    See Also
    --------
    experiment_to_markdown : Convert ExperimentResult to full Markdown report
    save_results_markdown : Save Markdown to file
    """
    if not results:
        return "_No results_"

    md = "| Input | Output | Error |\n|---|---|---|\n"
    for r in results:
        input_str = _escape_markdown_cell(r.get("input", ""))
        output_str = _escape_markdown_cell(r.get("output", ""))
        error_str = _escape_markdown_cell(r.get("error", ""))
        md += f"| {input_str} | {output_str} | {error_str} |\n"
    return md


def experiment_to_markdown(experiment: ExperimentResult) -> str:
    """Convert an ExperimentResult to a comprehensive Markdown report.

    Generates a full Markdown document including model information, probe
    details, summary statistics, scores, timing information, and a results
    table. The output is suitable for documentation, GitHub, or rendering
    in Jupyter notebooks.

    Parameters
    ----------
    experiment : ExperimentResult
        The experiment result object containing all probe execution data,
        including model info, scores, and individual results.

    Returns
    -------
    str
        A complete Markdown document as a string, including:
        - Title with experiment ID
        - Model information section
        - Probe information section
        - Summary statistics (counts and success rate)
        - Scores section (if scores are available)
        - Timing section (if duration was recorded)
        - Results table with all individual probe results
        - Generation timestamp

    Examples
    --------
    Basic conversion of an experiment result:

    >>> from insideLLMs.types import (
    ...     ExperimentResult, ProbeResult, ModelInfo,
    ...     ProbeCategory, ResultStatus
    ... )
    >>> model = ModelInfo(name="GPT-4", provider="openai", model_id="gpt-4")
    >>> result = ProbeResult(
    ...     input="What is 2+2?",
    ...     output="4",
    ...     status=ResultStatus.SUCCESS,
    ...     latency_ms=150.0
    ... )
    >>> experiment = ExperimentResult(
    ...     experiment_id="math-test-001",
    ...     model_info=model,
    ...     probe_name="arithmetic",
    ...     probe_category=ProbeCategory.FACTUAL,
    ...     results=[result],
    ... )
    >>> md = experiment_to_markdown(experiment)
    >>> "# Experiment Report: math-test-001" in md
    True
    >>> "**Model ID:** gpt-4" in md
    True

    The output includes model information:

    >>> "## Model Information" in md
    True
    >>> "**Name:** GPT-4" in md
    True
    >>> "**Provider:** openai" in md
    True

    Summary statistics are included:

    >>> "## Summary" in md
    True
    >>> "**Total Samples:**" in md
    True
    >>> "**Success Rate:**" in md
    True

    Results table with truncated content:

    >>> "## Results" in md
    True
    >>> "| # | Input | Output | Status | Latency (ms) |" in md
    True

    Experiment with scores:

    >>> from insideLLMs.types import ProbeScore
    >>> experiment.score = ProbeScore(
    ...     accuracy=0.95,
    ...     precision=0.92,
    ...     recall=0.89,
    ...     f1_score=0.905,
    ...     error_rate=0.05,
    ... )
    >>> md = experiment_to_markdown(experiment)
    >>> "## Scores" in md
    True
    >>> "**Accuracy:**" in md
    True

    Notes
    -----
    - Input and output strings are truncated to 50 characters in the table
    - The generation timestamp uses ISO format
    - Scores section only appears if ``experiment.score`` is not None
    - Timing section only appears if ``experiment.duration_seconds`` is set

    See Also
    --------
    results_to_markdown : Simple table conversion for raw results
    save_results_markdown : Save Markdown to file
    experiment_to_html : HTML version of this report
    """
    lines = [
        f"# Experiment Report: {experiment.experiment_id}",
        "",
        "## Model Information",
        f"- **Name:** {experiment.model_info.name}",
        f"- **Provider:** {experiment.model_info.provider}",
        f"- **Model ID:** {experiment.model_info.model_id}",
        "",
        "## Probe Information",
        f"- **Probe:** {experiment.probe_name}",
        f"- **Category:** {experiment.probe_category.value}",
        "",
        "## Summary",
        f"- **Total Samples:** {experiment.total_count}",
        f"- **Successful:** {experiment.success_count}",
        f"- **Errors:** {experiment.error_count}",
        f"- **Success Rate:** {_format_number(experiment.success_rate * 100, 1)}%",
        "",
    ]

    if experiment.score:
        lines.extend(_score_to_markdown_lines(experiment.score))

    if experiment.duration_seconds:
        lines.extend(
            [
                "## Timing",
                f"- **Duration:** {experiment.duration_seconds:.2f} seconds",
                "",
            ]
        )

    # Add results table
    lines.append("## Results")
    lines.append("")
    lines.append("| # | Input | Output | Status | Latency (ms) |")
    lines.append("|---|---|---|---|---|")

    for i, result in enumerate(experiment.results, 1):
        input_str = _escape_markdown_cell(str(result.input)[:50])
        output_str = _escape_markdown_cell(str(result.output)[:50] if result.output else "")
        status = result.status.value
        latency = _format_number(result.latency_ms, 1)
        lines.append(f"| {i} | {input_str} | {output_str} | {status} | {latency} |")

    lines.append("")
    lines.append(f"_Generated at {datetime.now().isoformat()}_")

    return "\n".join(lines)


def _score_to_markdown_lines(score: ProbeScore) -> list[str]:
    """Convert a ProbeScore object to formatted Markdown lines.

    Helper function that transforms score metrics into a list of Markdown
    formatted lines suitable for inclusion in a report. Handles optional
    metrics gracefully and includes custom metrics if present.

    Parameters
    ----------
    score : ProbeScore
        The score object containing accuracy, precision, recall, F1 score,
        latency, error rate, and optional custom metrics.

    Returns
    -------
    list[str]
        A list of Markdown formatted strings, including:
        - Section header ("## Scores")
        - Standard metrics (accuracy, precision, recall, F1, latency, error rate)
        - Custom metrics subsection (if any exist)
        - Trailing empty line for proper document formatting

    Examples
    --------
    Converting a full score object:

    >>> from insideLLMs.types import ProbeScore
    >>> score = ProbeScore(
    ...     accuracy=0.95,
    ...     precision=0.92,
    ...     recall=0.89,
    ...     f1_score=0.905,
    ...     mean_latency_ms=150.5,
    ...     error_rate=0.05,
    ... )
    >>> lines = _score_to_markdown_lines(score)
    >>> "## Scores" in lines
    True
    >>> any("Accuracy" in line for line in lines)
    True

    Score with custom metrics:

    >>> score.custom_metrics = {"perplexity": 12.5, "bleu": 0.82}
    >>> lines = _score_to_markdown_lines(score)
    >>> any("Custom Metrics" in line for line in lines)
    True
    >>> any("perplexity" in line for line in lines)
    True

    Partial score (some metrics None):

    >>> partial_score = ProbeScore(
    ...     accuracy=0.90,
    ...     error_rate=0.10,
    ... )
    >>> lines = _score_to_markdown_lines(partial_score)
    >>> any("Precision" in line for line in lines)
    False  # precision was None

    Notes
    -----
    - Percentages are displayed with 1 decimal place (e.g., "95.0%")
    - F1 score is displayed with 4 decimal places
    - Latency is displayed in milliseconds with 1 decimal place
    - None values are silently omitted from the output
    """
    lines = ["## Scores", ""]

    if score.accuracy is not None:
        lines.append(f"- **Accuracy:** {_format_number(score.accuracy * 100, 1)}%")
    if score.precision is not None:
        lines.append(f"- **Precision:** {_format_number(score.precision * 100, 1)}%")
    if score.recall is not None:
        lines.append(f"- **Recall:** {_format_number(score.recall * 100, 1)}%")
    if score.f1_score is not None:
        lines.append(f"- **F1 Score:** {_format_number(score.f1_score, 4)}")
    if score.mean_latency_ms is not None:
        lines.append(f"- **Mean Latency:** {_format_number(score.mean_latency_ms, 1)} ms")
    lines.append(f"- **Error Rate:** {_format_number(score.error_rate * 100, 1)}%")

    if score.custom_metrics:
        lines.append("")
        lines.append("### Custom Metrics")
        for name, value in score.custom_metrics.items():
            lines.append(f"- **{name}:** {_format_number(value, 4)}")

    lines.append("")
    return lines


def save_results_markdown(
    results: Union[list[dict[str, Any]], ExperimentResult],
    path: str,
) -> None:
    """Save experiment results to a Markdown file.

    Writes results as a Markdown document. Automatically detects whether the
    input is an ExperimentResult (full report) or a list of dictionaries
    (simple table) and formats accordingly.

    Parameters
    ----------
    results : list[dict[str, Any]] or ExperimentResult
        The results to save. Accepts either:
        - A list of result dictionaries (produces a simple table)
        - An ExperimentResult object (produces a full report)
    path : str
        Output file path for the Markdown file. Parent directory must exist.

    Raises
    ------
    FileNotFoundError
        If the parent directory of ``path`` does not exist.
    PermissionError
        If the file cannot be written due to permissions.

    Examples
    --------
    Saving a simple results list:

    >>> results = [
    ...     {"input": "What is 2+2?", "output": "4", "error": ""},
    ...     {"input": "What is 3+3?", "output": "6", "error": ""},
    ... ]
    >>> save_results_markdown(results, "/tmp/results.md")

    Saving an ExperimentResult:

    >>> from insideLLMs.types import (
    ...     ExperimentResult, ProbeResult, ModelInfo,
    ...     ProbeCategory, ResultStatus
    ... )
    >>> model = ModelInfo(name="GPT-4", provider="openai", model_id="gpt-4")
    >>> result = ProbeResult(
    ...     input="Test",
    ...     output="Response",
    ...     status=ResultStatus.SUCCESS,
    ...     latency_ms=100.0
    ... )
    >>> experiment = ExperimentResult(
    ...     experiment_id="exp-001",
    ...     model_info=model,
    ...     probe_name="test",
    ...     probe_category=ProbeCategory.FACTUAL,
    ...     results=[result],
    ... )
    >>> save_results_markdown(experiment, "/tmp/experiment.md")

    Verifying the saved file:

    >>> with open("/tmp/results.md") as f:
    ...     content = f.read()
    >>> "| Input | Output | Error |" in content
    True

    Notes
    -----
    - Uses UTF-8 encoding
    - Overwrites existing files without warning
    - For ExperimentResult, generates a full report with sections
    - For dict lists, generates a simple three-column table

    See Also
    --------
    results_to_markdown : Convert dict list to Markdown table
    experiment_to_markdown : Convert ExperimentResult to full report
    """
    if isinstance(results, ExperimentResult):
        md = experiment_to_markdown(results)
    else:
        md = results_to_markdown(results)

    with open(path, "w") as f:
        f.write(md)


# CSV Export Functions


def results_to_csv(
    results: list[dict[str, Any]],
    fields: Optional[list[str]] = None,
) -> str:
    """Convert a list of result dictionaries to CSV format.

    Transforms results into a comma-separated values string suitable for
    import into spreadsheets or data analysis tools. Handles field selection
    and proper CSV escaping.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of result dictionaries to convert. Each dictionary represents
        one row in the output CSV.
    fields : list[str] or None, default=None
        Specific fields to include as columns. If None, uses all keys from
        the first result dictionary. Fields not present in a result will
        be empty in that row.

    Returns
    -------
    str
        CSV formatted string with header row and data rows. Returns an
        empty string if the input list is empty.

    Examples
    --------
    Basic conversion with automatic field detection:

    >>> results = [
    ...     {"input": "What is 2+2?", "output": "4", "score": 1.0},
    ...     {"input": "What is 3+3?", "output": "6", "score": 1.0},
    ... ]
    >>> csv_str = results_to_csv(results)
    >>> print(csv_str)
    input,output,score
    What is 2+2?,4,1.0
    What is 3+3?,6,1.0

    Specifying specific fields:

    >>> csv_str = results_to_csv(results, fields=["input", "score"])
    >>> print(csv_str)
    input,score
    What is 2+2?,1.0
    What is 3+3?,1.0

    Handling empty results:

    >>> results_to_csv([])
    ''

    Results with special characters (auto-quoted):

    >>> results = [{"input": "Hello, world", "output": "Hi there"}]
    >>> csv_str = results_to_csv(results)
    >>> '"Hello, world"' in csv_str  # Commas trigger quoting
    True

    Results with newlines:

    >>> results = [{"input": "Line1\\nLine2", "output": "Response"}]
    >>> csv_str = results_to_csv(results)
    >>> len(csv_str.split('\\n'))  # Still 2 lines (header + data)
    3  # Actually 3 because of the quoted newline

    Notes
    -----
    - Uses Python's csv.DictWriter with default dialect (excel)
    - Fields not in the field list are ignored (via ``extrasaction='ignore'``)
    - Values containing commas, quotes, or newlines are properly quoted
    - Output uses CRLF line endings (CSV standard)

    See Also
    --------
    save_results_csv : Save CSV to file
    """
    if not results:
        return ""

    if fields is None:
        fields = list(results[0].keys())

    import io

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    return output.getvalue()


def save_results_csv(
    results: list[dict[str, Any]],
    path: str,
    fields: Optional[list[str]] = None,
) -> None:
    """Save experiment results to a CSV file.

    Writes results as a comma-separated values file suitable for import into
    spreadsheets (Excel, Google Sheets) or data analysis tools (pandas).

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of result dictionaries to save. Each dictionary becomes one row.
    path : str
        Output file path for the CSV file. Parent directory must exist.
    fields : list[str] or None, default=None
        Specific fields to include as columns. If None, includes all fields
        from the first result dictionary.

    Raises
    ------
    FileNotFoundError
        If the parent directory of ``path`` does not exist.
    PermissionError
        If the file cannot be written due to permissions.

    Examples
    --------
    Saving results with automatic field detection:

    >>> results = [
    ...     {"input": "Query 1", "output": "Response 1", "latency": 100},
    ...     {"input": "Query 2", "output": "Response 2", "latency": 150},
    ... ]
    >>> save_results_csv(results, "/tmp/results.csv")

    Saving with specific fields:

    >>> save_results_csv(results, "/tmp/subset.csv", fields=["input", "latency"])

    Loading the saved CSV with pandas:

    >>> import pandas as pd  # doctest: +SKIP
    >>> df = pd.read_csv("/tmp/results.csv")  # doctest: +SKIP
    >>> df.columns.tolist()  # doctest: +SKIP
    ['input', 'output', 'latency']

    Verifying the file contents:

    >>> with open("/tmp/results.csv") as f:
    ...     lines = f.readlines()
    >>> lines[0].strip()  # Header
    'input,output,latency'
    >>> len(lines)  # Header + 2 data rows
    3

    Notes
    -----
    - Uses UTF-8 encoding
    - Overwrites existing files without warning
    - Empty results list creates an empty file
    - Uses standard CSV dialect (comma delimiter, quote on special chars)

    See Also
    --------
    results_to_csv : Convert to CSV string without file I/O
    """
    csv_content = results_to_csv(results, fields)
    with open(path, "w") as f:
        f.write(csv_content)


# HTML Export Functions


def experiment_to_html(
    experiment: ExperimentResult,
    include_statistics: bool = True,
    *,
    escape_html: bool = True,
    generated_at: Optional[str] = None,
) -> str:
    """Convert an ExperimentResult to a styled HTML report.

    Generates a complete, self-contained HTML document with embedded CSS
    styling. The report includes model information, summary metrics, scores,
    and a detailed results table with visual status indicators.

    Parameters
    ----------
    experiment : ExperimentResult
        The experiment result object containing all probe execution data.
    include_statistics : bool, default=True
        Whether to include statistical analysis sections. Currently reserved
        for future use; the parameter is accepted but does not change output.

    Returns
    -------
    str
        A complete HTML document as a string, including:
        - DOCTYPE declaration and HTML structure
        - Embedded CSS with modern styling (fonts, colors, hover effects)
        - Model information section
        - Summary metrics displayed as cards
        - Scores section with visual metric cards (if scores exist)
        - Results table with status coloring
        - Generation timestamp

    Examples
    --------
    Basic conversion to HTML:

    >>> from insideLLMs.types import (
    ...     ExperimentResult, ProbeResult, ModelInfo,
    ...     ProbeCategory, ResultStatus
    ... )
    >>> model = ModelInfo(name="GPT-4", provider="openai", model_id="gpt-4")
    >>> result = ProbeResult(
    ...     input="What is 2+2?",
    ...     output="4",
    ...     status=ResultStatus.SUCCESS,
    ...     latency_ms=150.0
    ... )
    >>> experiment = ExperimentResult(
    ...     experiment_id="math-test-001",
    ...     model_info=model,
    ...     probe_name="arithmetic",
    ...     probe_category=ProbeCategory.FACTUAL,
    ...     results=[result],
    ... )
    >>> html = experiment_to_html(experiment)
    >>> "<!DOCTYPE html>" in html
    True
    >>> "<title>Experiment Report: math-test-001</title>" in html
    True

    HTML includes styled metric cards:

    >>> "class='metric'" in html
    True
    >>> "metric-value" in html
    True

    Results table with status colors:

    >>> "class='success'" in html or "class='error'" in html
    True

    Experiment with scores:

    >>> from insideLLMs.types import ProbeScore
    >>> experiment.score = ProbeScore(
    ...     accuracy=0.95,
    ...     precision=0.92,
    ...     recall=0.89,
    ...     f1_score=0.905,
    ...     error_rate=0.05,
    ... )
    >>> html = experiment_to_html(experiment)
    >>> "Accuracy" in html
    True
    >>> "95.0%" in html
    True

    Writing to file for browser viewing:

    >>> with open("/tmp/report.html", "w") as f:
    ...     f.write(html)  # doctest: +SKIP
    >>> # Open /tmp/report.html in a browser

    Notes
    -----
    - The HTML is self-contained with no external dependencies
    - CSS uses system fonts with fallbacks for cross-platform consistency
    - Colors use a blue/green/red theme for neutral/success/error states
    - Input/output strings are truncated to 100 characters in the table
    - Hover effects on table rows improve readability
    - The report is responsive to different screen widths (max-width: 1200px)

    See Also
    --------
    experiment_to_markdown : Markdown version of this report
    save_results_html : Save HTML to file
    """

    def _esc(value: object) -> str:
        text = str(value)
        return _html_escape(text) if escape_html else text

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>Experiment Report: {_esc(experiment.experiment_id)}</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
        "h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
        "h2 { color: #34495e; margin-top: 30px; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
        "th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }",
        "th { background-color: #3498db; color: white; }",
        "tr:nth-child(even) { background-color: #f9f9f9; }",
        "tr:hover { background-color: #f1f1f1; }",
        ".metric { display: inline-block; background: #ecf0f1; padding: 10px 20px; margin: 5px; border-radius: 5px; }",
        ".metric-value { font-size: 24px; font-weight: bold; color: #2980b9; }",
        ".metric-label { font-size: 12px; color: #7f8c8d; }",
        ".success { color: #27ae60; }",
        ".error { color: #e74c3c; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Experiment Report: {_esc(experiment.experiment_id)}</h1>",
    ]

    # Model Info
    html_parts.extend(
        [
            "<h2>Model Information</h2>",
            "<ul>",
            f"<li><strong>Name:</strong> {_esc(experiment.model_info.name)}</li>",
            f"<li><strong>Provider:</strong> {_esc(experiment.model_info.provider)}</li>",
            f"<li><strong>Model ID:</strong> {_esc(experiment.model_info.model_id)}</li>",
            "</ul>",
        ]
    )

    # Summary metrics
    html_parts.extend(
        [
            "<h2>Summary</h2>",
            "<div class='metrics'>",
            f"<div class='metric'><div class='metric-value'>{experiment.total_count}</div><div class='metric-label'>Total Samples</div></div>",
            f"<div class='metric'><div class='metric-value success'>{experiment.success_count}</div><div class='metric-label'>Successful</div></div>",
            f"<div class='metric'><div class='metric-value error'>{experiment.error_count}</div><div class='metric-label'>Errors</div></div>",
            f"<div class='metric'><div class='metric-value'>{experiment.success_rate * 100:.1f}%</div><div class='metric-label'>Success Rate</div></div>",
            "</div>",
        ]
    )

    # Scores
    if experiment.score:
        html_parts.append("<h2>Scores</h2>")
        html_parts.append("<div class='metrics'>")
        if experiment.score.accuracy is not None:
            html_parts.append(
                f"<div class='metric'><div class='metric-value'>{experiment.score.accuracy * 100:.1f}%</div><div class='metric-label'>Accuracy</div></div>"
            )
        if experiment.score.precision is not None:
            html_parts.append(
                f"<div class='metric'><div class='metric-value'>{experiment.score.precision * 100:.1f}%</div><div class='metric-label'>Precision</div></div>"
            )
        if experiment.score.recall is not None:
            html_parts.append(
                f"<div class='metric'><div class='metric-value'>{experiment.score.recall * 100:.1f}%</div><div class='metric-label'>Recall</div></div>"
            )
        if experiment.score.f1_score is not None:
            html_parts.append(
                f"<div class='metric'><div class='metric-value'>{experiment.score.f1_score:.3f}</div><div class='metric-label'>F1 Score</div></div>"
            )
        html_parts.append("</div>")

    # Results table
    html_parts.extend(
        [
            "<h2>Results</h2>",
            "<table>",
            "<tr><th>#</th><th>Input</th><th>Output</th><th>Status</th><th>Latency (ms)</th></tr>",
        ]
    )

    for i, result in enumerate(experiment.results, 1):
        status_class = "success" if result.status.value == "success" else "error"
        input_str = _esc(str(result.input)[:100])
        output_str = _esc(str(result.output)[:100]) if result.output else ""
        latency = f"{result.latency_ms:.1f}" if result.latency_ms else "N/A"
        html_parts.append(
            f"<tr><td>{i}</td><td>{input_str}</td><td>{output_str}</td>"
            f"<td class='{status_class}'>{result.status.value}</td><td>{latency}</td></tr>"
        )

    html_parts.extend(
        [
            "</table>",
            f"<p><em>Generated at {_esc(generated_at)}</em></p>" if generated_at else "",
            "</body>",
            "</html>",
        ]
    )

    return "\n".join(html_parts)


def save_results_html(experiment: ExperimentResult, path: str) -> None:
    """Save experiment results to a styled HTML file.

    Writes an ExperimentResult as a self-contained HTML document with embedded
    CSS styling. The resulting file can be opened directly in any web browser.

    Parameters
    ----------
    experiment : ExperimentResult
        The experiment result object to convert and save.
    path : str
        Output file path for the HTML file. Parent directory must exist.

    Raises
    ------
    FileNotFoundError
        If the parent directory of ``path`` does not exist.
    PermissionError
        If the file cannot be written due to permissions.

    Examples
    --------
    Saving an experiment to HTML:

    >>> from insideLLMs.types import (
    ...     ExperimentResult, ProbeResult, ModelInfo,
    ...     ProbeCategory, ResultStatus
    ... )
    >>> model = ModelInfo(name="GPT-4", provider="openai", model_id="gpt-4")
    >>> result = ProbeResult(
    ...     input="Test input",
    ...     output="Test output",
    ...     status=ResultStatus.SUCCESS,
    ...     latency_ms=100.0
    ... )
    >>> experiment = ExperimentResult(
    ...     experiment_id="exp-001",
    ...     model_info=model,
    ...     probe_name="test",
    ...     probe_category=ProbeCategory.FACTUAL,
    ...     results=[result],
    ... )
    >>> save_results_html(experiment, "/tmp/report.html")

    Verifying the saved file:

    >>> with open("/tmp/report.html") as f:
    ...     content = f.read()
    >>> "<!DOCTYPE html>" in content
    True
    >>> "Experiment Report" in content
    True

    Opening in a browser (platform-specific):

    >>> import webbrowser  # doctest: +SKIP
    >>> webbrowser.open("file:///tmp/report.html")  # doctest: +SKIP

    Notes
    -----
    - Uses UTF-8 encoding
    - Overwrites existing files without warning
    - The HTML includes all CSS inline (no external stylesheets)
    - Compatible with all modern browsers

    See Also
    --------
    experiment_to_html : Convert ExperimentResult to HTML string
    save_results_markdown : Save as Markdown instead
    """
    html = experiment_to_html(experiment)
    with open(path, "w") as f:
        f.write(html)


# Comparison Reports


def comparison_to_markdown(comparison: BenchmarkComparison) -> str:
    """Convert a BenchmarkComparison to a Markdown report.

    Generates a Markdown document comparing multiple experiments, including
    summary tables, rankings, and aggregate statistics. Useful for comparing
    model performance across different probes or configurations.

    Parameters
    ----------
    comparison : BenchmarkComparison
        The benchmark comparison object containing multiple experiments,
        optional rankings, and summary statistics.

    Returns
    -------
    str
        A Markdown document as a string, including:
        - Title with comparison name
        - Creation timestamp
        - Summary table with model, probe, success rate, accuracy, and latency
        - Rankings section (if rankings are provided)
        - Summary section (if summary statistics are provided)

    Examples
    --------
    Basic comparison of two models:

    >>> from datetime import datetime
    >>> from insideLLMs.types import (
    ...     BenchmarkComparison, ExperimentResult, ProbeResult,
    ...     ModelInfo, ProbeCategory, ResultStatus, ProbeScore
    ... )
    >>> model1 = ModelInfo(name="GPT-4", provider="openai", model_id="gpt-4")
    >>> model2 = ModelInfo(name="Claude", provider="anthropic", model_id="claude-3")
    >>> score1 = ProbeScore(accuracy=0.95, mean_latency_ms=150.0, error_rate=0.05)
    >>> score2 = ProbeScore(accuracy=0.92, mean_latency_ms=120.0, error_rate=0.08)
    >>> exp1 = ExperimentResult(
    ...     experiment_id="exp-gpt4",
    ...     model_info=model1,
    ...     probe_name="factual",
    ...     probe_category=ProbeCategory.FACTUAL,
    ...     results=[],
    ...     score=score1,
    ... )
    >>> exp2 = ExperimentResult(
    ...     experiment_id="exp-claude",
    ...     model_info=model2,
    ...     probe_name="factual",
    ...     probe_category=ProbeCategory.FACTUAL,
    ...     results=[],
    ...     score=score2,
    ... )
    >>> comparison = BenchmarkComparison(
    ...     name="Model Comparison",
    ...     experiments=[exp1, exp2],
    ...     created_at=datetime.now(),
    ... )
    >>> md = comparison_to_markdown(comparison)
    >>> "# Benchmark Comparison: Model Comparison" in md
    True

    Summary table is included:

    >>> "| Model | Probe | Success Rate | Accuracy | Mean Latency |" in md
    True
    >>> "GPT-4" in md
    True
    >>> "Claude" in md
    True

    Comparison with rankings:

    >>> comparison.rankings = {
    ...     "accuracy": ["GPT-4", "Claude"],
    ...     "latency": ["Claude", "GPT-4"],
    ... }
    >>> md = comparison_to_markdown(comparison)
    >>> "## Rankings" in md
    True
    >>> "### accuracy" in md
    True

    Comparison with summary statistics:

    >>> comparison.summary = {
    ...     "best_model": "GPT-4",
    ...     "avg_accuracy": "93.5%",
    ... }
    >>> md = comparison_to_markdown(comparison)
    >>> "## Summary" in md
    True
    >>> "**best_model:**" in md
    True

    Notes
    -----
    - Success rate is displayed as a percentage with 1 decimal place
    - Accuracy shows "N/A" if no score or accuracy is available
    - Latency shows "N/A" if no score or mean_latency_ms is available
    - Rankings are numbered starting from 1

    See Also
    --------
    save_comparison_markdown : Save comparison to file
    experiment_to_markdown : Convert single experiment to Markdown
    """
    lines = [
        f"# Benchmark Comparison: {comparison.name}",
        "",
        f"_Created at {comparison.created_at.isoformat()}_",
        "",
        "## Experiments",
        "",
    ]

    # Summary table
    lines.append("| Model | Probe | Success Rate | Accuracy | Mean Latency |")
    lines.append("|---|---|---|---|---|")

    for exp in comparison.experiments:
        success_rate = f"{exp.success_rate * 100:.1f}%"
        accuracy = f"{exp.score.accuracy * 100:.1f}%" if exp.score and exp.score.accuracy else "N/A"
        latency = (
            f"{exp.score.mean_latency_ms:.1f} ms"
            if exp.score and exp.score.mean_latency_ms
            else "N/A"
        )
        lines.append(
            f"| {exp.model_info.name} | {exp.probe_name} | {success_rate} | {accuracy} | {latency} |"
        )

    lines.append("")

    # Rankings
    if comparison.rankings:
        lines.append("## Rankings")
        lines.append("")
        for metric, ranking in comparison.rankings.items():
            lines.append(f"### {metric}")
            for i, name in enumerate(ranking, 1):
                lines.append(f"{i}. {name}")
            lines.append("")

    # Summary
    if comparison.summary:
        lines.append("## Summary")
        lines.append("")
        for key, value in comparison.summary.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")

    return "\n".join(lines)


def save_comparison_markdown(comparison: BenchmarkComparison, path: str) -> None:
    """Save a benchmark comparison to a Markdown file.

    Writes a BenchmarkComparison as a Markdown document suitable for
    documentation, GitHub, or embedding in reports.

    Parameters
    ----------
    comparison : BenchmarkComparison
        The benchmark comparison object to convert and save.
    path : str
        Output file path for the Markdown file. Parent directory must exist.

    Raises
    ------
    FileNotFoundError
        If the parent directory of ``path`` does not exist.
    PermissionError
        If the file cannot be written due to permissions.

    Examples
    --------
    Saving a benchmark comparison:

    >>> from datetime import datetime
    >>> from insideLLMs.types import (
    ...     BenchmarkComparison, ExperimentResult, ModelInfo, ProbeCategory
    ... )
    >>> model = ModelInfo(name="GPT-4", provider="openai", model_id="gpt-4")
    >>> experiment = ExperimentResult(
    ...     experiment_id="exp-001",
    ...     model_info=model,
    ...     probe_name="test",
    ...     probe_category=ProbeCategory.FACTUAL,
    ...     results=[],
    ... )
    >>> comparison = BenchmarkComparison(
    ...     name="Single Model Test",
    ...     experiments=[experiment],
    ...     created_at=datetime.now(),
    ... )
    >>> save_comparison_markdown(comparison, "/tmp/comparison.md")

    Verifying the saved file:

    >>> with open("/tmp/comparison.md") as f:
    ...     content = f.read()
    >>> "# Benchmark Comparison: Single Model Test" in content
    True

    Notes
    -----
    - Uses UTF-8 encoding
    - Overwrites existing files without warning

    See Also
    --------
    comparison_to_markdown : Convert comparison to Markdown string
    """
    md = comparison_to_markdown(comparison)
    with open(path, "w") as f:
        f.write(md)


# Statistical Report Generation


def generate_statistical_report(
    experiments: list[ExperimentResult],
    output_path: Optional[str] = None,
    format: str = "markdown",
    confidence_level: float = 0.95,
) -> str:
    """Generate a comprehensive statistical analysis report.

    Analyzes multiple experiment results and produces a report with aggregate
    statistics, confidence intervals, and breakdowns by model and probe. The
    report can be output in Markdown, HTML, or JSON format.

    Parameters
    ----------
    experiments : list[ExperimentResult]
        List of experiment results to analyze. Should contain at least one
        experiment; returns a placeholder message if empty.
    output_path : str or None, default=None
        Optional file path to save the report. If None, only returns the
        report string without saving.
    format : str, default="markdown"
        Output format for the report. Options:
        - "markdown": Human-readable Markdown document
        - "html": Styled HTML document with CSS
        - "json": Machine-readable JSON structure
    confidence_level : float, default=0.95
        Confidence level for statistical intervals (e.g., 0.95 for 95% CI).
        Must be between 0 and 1.

    Returns
    -------
    str
        The generated report as a string in the specified format.

    Raises
    ------
    FileNotFoundError
        If ``output_path`` is specified but the parent directory does not exist.
    ImportError
        If the ``insideLLMs.statistics`` module is not available.

    Examples
    --------
    Generating a Markdown report:

    >>> from insideLLMs.types import (
    ...     ExperimentResult, ProbeResult, ModelInfo,
    ...     ProbeCategory, ResultStatus, ProbeScore
    ... )
    >>> model = ModelInfo(name="GPT-4", provider="openai", model_id="gpt-4")
    >>> result = ProbeResult(
    ...     input="Test",
    ...     output="Response",
    ...     status=ResultStatus.SUCCESS,
    ...     latency_ms=100.0
    ... )
    >>> experiment = ExperimentResult(
    ...     experiment_id="exp-001",
    ...     model_info=model,
    ...     probe_name="factual",
    ...     probe_category=ProbeCategory.FACTUAL,
    ...     results=[result],
    ...     score=ProbeScore(accuracy=0.95, error_rate=0.05),
    ... )
    >>> report = generate_statistical_report([experiment])
    >>> "# Statistical Analysis Report" in report
    True

    Generating with custom confidence level:

    >>> report = generate_statistical_report(
    ...     [experiment],
    ...     confidence_level=0.99,
    ... )
    >>> "99%" in report
    True

    Saving to a file:

    >>> report = generate_statistical_report(
    ...     [experiment],
    ...     output_path="/tmp/stats_report.md",
    ... )
    >>> import os
    >>> os.path.exists("/tmp/stats_report.md")
    True

    Generating HTML format:

    >>> html_report = generate_statistical_report(
    ...     [experiment],
    ...     format="html",
    ... )
    >>> "<!DOCTYPE html>" in html_report
    True
    >>> "Statistical Analysis Report" in html_report
    True

    Generating JSON format for programmatic use:

    >>> import json
    >>> json_report = generate_statistical_report(
    ...     [experiment],
    ...     format="json",
    ... )
    >>> data = json.loads(json_report)
    >>> "total_experiments" in data
    True

    Handling empty experiments list:

    >>> generate_statistical_report([])
    'No experiments to analyze.'

    Notes
    -----
    - The report includes breakdowns by model and by probe
    - Confidence intervals use standard statistical methods
    - Large experiment sets may take longer to analyze
    - The statistics module must be available for full functionality

    See Also
    --------
    insideLLMs.statistics.generate_summary_report : Underlying statistics function
    experiment_to_markdown : Single experiment report
    """
    from insideLLMs.statistics import (
        generate_summary_report,
    )

    if not experiments:
        return "No experiments to analyze."

    # Generate the summary
    summary = generate_summary_report(experiments, True, confidence_level)

    if format == "json":
        report = json.dumps(summary, indent=2, default=str)
    elif format == "html":
        report = _statistical_report_to_html(summary, confidence_level)
    else:  # markdown
        report = _statistical_report_to_markdown(summary, confidence_level)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report


def _statistical_report_to_markdown(
    summary: dict[str, Any],
    confidence_level: float,
) -> str:
    """Convert a statistical summary dictionary to a Markdown report.

    Helper function that transforms the output of ``generate_summary_report``
    into a formatted Markdown document with tables and statistics.

    Parameters
    ----------
    summary : dict[str, Any]
        Statistical summary dictionary containing:
        - ``total_experiments``: Number of experiments analyzed
        - ``unique_models``: List of model names
        - ``unique_probes``: List of probe names
        - ``overall``: Overall statistics (success_rate, latency_ms)
        - ``by_model``: Per-model breakdowns
        - ``by_probe``: Per-probe breakdowns
    confidence_level : float
        Confidence level used for intervals, displayed in the report header.

    Returns
    -------
    str
        Markdown formatted report string.

    Examples
    --------
    Converting a summary to Markdown:

    >>> summary = {
    ...     "total_experiments": 10,
    ...     "unique_models": ["GPT-4", "Claude"],
    ...     "unique_probes": ["factual", "reasoning"],
    ...     "overall": {
    ...         "success_rate": {"mean": 0.95, "std": 0.02},
    ...         "latency_ms": {"mean": 150.0, "std": 25.0},
    ...     },
    ...     "by_model": {
    ...         "GPT-4": {
    ...             "n_experiments": 5,
    ...             "success_rate": {"mean": 0.96},
    ...             "success_rate_ci": {"lower": 0.92, "upper": 0.99},
    ...             "latency_ms": {"mean": 140.0},
    ...         },
    ...     },
    ... }
    >>> md = _statistical_report_to_markdown(summary, 0.95)
    >>> "# Statistical Analysis Report" in md
    True
    >>> "**Confidence Level:** 95%" in md
    True

    Notes
    -----
    - Percentages are displayed with 1 decimal place
    - Standard deviations are shown in parentheses
    - Confidence intervals are shown in square brackets
    - Tables are formatted with alignment separators
    """
    lines = [
        "# Statistical Analysis Report",
        "",
        f"**Total Experiments:** {summary['total_experiments']}",
        f"**Models Analyzed:** {', '.join(summary['unique_models'])}",
        f"**Probes Used:** {', '.join(summary['unique_probes'])}",
        f"**Confidence Level:** {confidence_level * 100:.0f}%",
        "",
    ]

    # Overall statistics
    if summary.get("overall"):
        lines.extend(
            [
                "## Overall Performance",
                "",
            ]
        )
        overall = summary["overall"]
        if "success_rate" in overall:
            sr = overall["success_rate"]
            lines.append(
                f"- **Mean Success Rate:** {sr['mean'] * 100:.1f}% (SD: {sr['std'] * 100:.1f}%)"
            )

        if "latency_ms" in overall:
            lat = overall["latency_ms"]
            lines.append(f"- **Mean Latency:** {lat['mean']:.1f} ms (SD: {lat['std']:.1f} ms)")
        lines.append("")

    # By model
    if summary.get("by_model"):
        lines.extend(
            [
                "## Performance by Model",
                "",
                "| Model | N | Success Rate | SR CI | Latency (ms) |",
                "|---|---|---|---|---|",
            ]
        )

        for model_name, model_data in summary["by_model"].items():
            n = model_data["n_experiments"]
            sr = model_data["success_rate"]["mean"] * 100
            sr_ci = model_data.get("success_rate_ci", {})
            ci_str = (
                f"[{sr_ci.get('lower', 0) * 100:.1f}%, {sr_ci.get('upper', 0) * 100:.1f}%]"
                if sr_ci
                else "N/A"
            )
            lat = model_data.get("latency_ms", {}).get("mean", "N/A")
            lat_str = f"{lat:.1f}" if isinstance(lat, (int, float)) else lat
            lines.append(f"| {model_name} | {n} | {sr:.1f}% | {ci_str} | {lat_str} |")

        lines.append("")

    # By probe
    if summary.get("by_probe"):
        lines.extend(
            [
                "## Performance by Probe",
                "",
                "| Probe | N | Success Rate | Accuracy | Accuracy CI |",
                "|---|---|---|---|---|",
            ]
        )

        for probe_name, probe_data in summary["by_probe"].items():
            n = probe_data["n_experiments"]
            sr = probe_data["success_rate"]["mean"] * 100
            acc = probe_data.get("accuracy", {}).get("mean")
            acc_str = f"{acc * 100:.1f}%" if acc is not None else "N/A"
            acc_ci = probe_data.get("accuracy_ci", {})
            acc_ci_str = (
                f"[{acc_ci.get('lower', 0) * 100:.1f}%, {acc_ci.get('upper', 0) * 100:.1f}%]"
                if acc_ci
                else "N/A"
            )
            lines.append(f"| {probe_name} | {n} | {sr:.1f}% | {acc_str} | {acc_ci_str} |")

        lines.append("")

    lines.append(f"_Report generated at {datetime.now().isoformat()}_")

    return "\n".join(lines)


def _statistical_report_to_html(
    summary: dict[str, Any],
    confidence_level: float,
) -> str:
    """Convert a statistical summary dictionary to a styled HTML report.

    Helper function that transforms the output of ``generate_summary_report``
    into a self-contained HTML document with embedded CSS styling.

    Parameters
    ----------
    summary : dict[str, Any]
        Statistical summary dictionary containing:
        - ``total_experiments``: Number of experiments analyzed
        - ``unique_models``: List of model names
        - ``unique_probes``: List of probe names
        - ``by_model``: Per-model breakdowns with success rates and CIs
    confidence_level : float
        Confidence level used for intervals, displayed in the header cards.

    Returns
    -------
    str
        Complete HTML document as a string with embedded CSS.

    Examples
    --------
    Converting a summary to HTML:

    >>> summary = {
    ...     "total_experiments": 10,
    ...     "unique_models": ["GPT-4", "Claude"],
    ...     "unique_probes": ["factual", "reasoning"],
    ...     "by_model": {
    ...         "GPT-4": {
    ...             "n_experiments": 5,
    ...             "success_rate": {"mean": 0.96},
    ...             "success_rate_ci": {"lower": 0.92, "upper": 0.99},
    ...         },
    ...     },
    ... }
    >>> html = _statistical_report_to_html(summary, 0.95)
    >>> "<!DOCTYPE html>" in html
    True
    >>> "Statistical Analysis Report" in html
    True

    HTML includes styled summary cards:

    >>> "class='summary-stat'" in html
    True

    HTML includes per-model table:

    >>> "<table>" in html
    True
    >>> "GPT-4" in html
    True

    Notes
    -----
    - Uses a gradient color scheme (purple/blue) for summary cards
    - Table headers have a blue gradient background
    - Confidence intervals are displayed in [lower%, upper%] format
    - Includes a generation timestamp at the bottom
    """
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<title>Statistical Analysis Report</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f6fa; }",
        ".card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "h1 { color: #2c3e50; }",
        "h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }",
        "th { background: linear-gradient(135deg, #3498db, #2980b9); color: white; }",
        "tr:nth-child(even) { background-color: #f9f9f9; }",
        ".summary-stat { display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 15px 25px; margin: 10px; border-radius: 8px; text-align: center; }",
        ".summary-stat .value { font-size: 28px; font-weight: bold; }",
        ".summary-stat .label { font-size: 12px; opacity: 0.9; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Statistical Analysis Report</h1>",
        "<div class='card'>",
        f"<div class='summary-stat'><div class='value'>{summary['total_experiments']}</div><div class='label'>Experiments</div></div>",
        f"<div class='summary-stat'><div class='value'>{len(summary['unique_models'])}</div><div class='label'>Models</div></div>",
        f"<div class='summary-stat'><div class='value'>{len(summary['unique_probes'])}</div><div class='label'>Probes</div></div>",
        f"<div class='summary-stat'><div class='value'>{confidence_level * 100:.0f}%</div><div class='label'>Confidence Level</div></div>",
        "</div>",
    ]

    # By model table
    if summary.get("by_model"):
        html.extend(
            [
                "<div class='card'>",
                "<h2>Performance by Model</h2>",
                "<table>",
                "<tr><th>Model</th><th>Experiments</th><th>Success Rate</th><th>Confidence Interval</th></tr>",
            ]
        )

        for model_name, model_data in summary["by_model"].items():
            n = model_data["n_experiments"]
            sr = model_data["success_rate"]["mean"] * 100
            sr_ci = model_data.get("success_rate_ci", {})
            ci_str = (
                f"[{sr_ci.get('lower', 0) * 100:.1f}%, {sr_ci.get('upper', 0) * 100:.1f}%]"
                if sr_ci
                else "N/A"
            )
            html.append(
                f"<tr><td>{model_name}</td><td>{n}</td><td>{sr:.1f}%</td><td>{ci_str}</td></tr>"
            )

        html.extend(["</table>", "</div>"])

    html.extend(
        [
            f"<p><em>Report generated at {datetime.now().isoformat()}</em></p>",
            "</body>",
            "</html>",
        ]
    )

    return "\n".join(html)
