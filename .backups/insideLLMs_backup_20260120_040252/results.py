"""Results aggregation, reporting, and export utilities.

This module provides tools for:
- Saving results in various formats (JSON, Markdown, HTML, CSV)
- Generating reports with statistical analysis
- Visualizing experiment results
- Comparing multiple experiments
"""

import csv
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Optional, Union

from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION
from insideLLMs.types import (
    BenchmarkComparison,
    ExperimentResult,
    ProbeResult,
    ProbeScore,
)


def _escape_markdown_cell(value: Any) -> str:
    """Escape characters that break Markdown tables and normalize newlines."""
    text = str(value)
    text = text.replace("|", "\\|")
    text = text.replace("\n", "<br>")
    return text


def _format_number(value: Optional[float], precision: int = 4) -> str:
    """Format a number for display."""
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def _serialize_for_json(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
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
    """Save results to a JSON file.

    Args:
        results: Results to save (list of dicts, ProbeResults, or ExperimentResult).
        path: Output file path.
        indent: JSON indentation level.
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
    """Load results from a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Loaded results as dictionary.
    """
    with open(path) as f:
        return json.load(f)


# Markdown Export Functions


def results_to_markdown(results: list[dict[str, Any]]) -> str:
    """Convert results list to a Markdown table.

    Args:
        results: List of result dictionaries.

    Returns:
        Markdown formatted table.
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
    """Convert an ExperimentResult to a Markdown report.

    Args:
        experiment: The experiment result to convert.

    Returns:
        Markdown formatted report.
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
    """Convert ProbeScore to markdown lines."""
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
    """Save results as a Markdown file.

    Args:
        results: Results to save.
        path: Output file path.
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
    """Convert results to CSV format.

    Args:
        results: List of result dictionaries.
        fields: Specific fields to include (default: all fields from first result).

    Returns:
        CSV formatted string.
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
    """Save results as a CSV file.

    Args:
        results: Results to save.
        path: Output file path.
        fields: Specific fields to include.
    """
    csv_content = results_to_csv(results, fields)
    with open(path, "w") as f:
        f.write(csv_content)


# HTML Export Functions


def experiment_to_html(
    experiment: ExperimentResult,
    include_statistics: bool = True,
) -> str:
    """Convert an ExperimentResult to an HTML report.

    Args:
        experiment: The experiment result to convert.
        include_statistics: Whether to include statistical analysis.

    Returns:
        HTML formatted report.
    """
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>Experiment Report: {experiment.experiment_id}</title>",
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
        f"<h1>Experiment Report: {experiment.experiment_id}</h1>",
    ]

    # Model Info
    html_parts.extend(
        [
            "<h2>Model Information</h2>",
            "<ul>",
            f"<li><strong>Name:</strong> {experiment.model_info.name}</li>",
            f"<li><strong>Provider:</strong> {experiment.model_info.provider}</li>",
            f"<li><strong>Model ID:</strong> {experiment.model_info.model_id}</li>",
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
        input_str = str(result.input)[:100]
        output_str = str(result.output)[:100] if result.output else ""
        latency = f"{result.latency_ms:.1f}" if result.latency_ms else "N/A"
        html_parts.append(
            f"<tr><td>{i}</td><td>{input_str}</td><td>{output_str}</td>"
            f"<td class='{status_class}'>{result.status.value}</td><td>{latency}</td></tr>"
        )

    html_parts.extend(
        [
            "</table>",
            f"<p><em>Generated at {datetime.now().isoformat()}</em></p>",
            "</body>",
            "</html>",
        ]
    )

    return "\n".join(html_parts)


def save_results_html(experiment: ExperimentResult, path: str) -> None:
    """Save experiment results as an HTML file.

    Args:
        experiment: The experiment result to save.
        path: Output file path.
    """
    html = experiment_to_html(experiment)
    with open(path, "w") as f:
        f.write(html)


# Comparison Reports


def comparison_to_markdown(comparison: BenchmarkComparison) -> str:
    """Convert a BenchmarkComparison to a Markdown report.

    Args:
        comparison: The comparison to convert.

    Returns:
        Markdown formatted report.
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
    """Save comparison as a Markdown file.

    Args:
        comparison: The comparison to save.
        path: Output file path.
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
    """Generate a comprehensive statistical report.

    Args:
        experiments: List of experiment results to analyze.
        output_path: Optional path to save the report.
        format: Output format ("markdown", "html", "json").
        confidence_level: Confidence level for intervals.

    Returns:
        The generated report as a string.
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
    """Convert statistical summary to Markdown."""
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
    """Convert statistical summary to HTML."""
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
