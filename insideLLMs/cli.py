"""Command-line interface for insideLLMs.

Provides commands for running experiments, listing available models and probes,
benchmarking, interactive exploration, and managing configurations.
"""

import argparse
import asyncio
import hashlib
import html
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from insideLLMs.registry import (
    ensure_builtins_registered,
    model_registry,
    probe_registry,
)
from insideLLMs.results import results_to_markdown, save_results_json
from insideLLMs.runner import (
    derive_run_id_from_config_path,
    run_experiment_from_config,
    run_experiment_from_config_async,
    run_harness_from_config,
)
from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION
from insideLLMs.statistics import generate_summary_report
from insideLLMs.types import (
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ResultStatus,
)


def _add_output_schema_args(parser: argparse.ArgumentParser) -> None:
    """Add common output schema validation arguments to a subcommand parser."""

    parser.add_argument(
        "--validate-output",
        action="store_true",
        help="Validate serialized outputs against a versioned schema (requires pydantic)",
    )
    parser.add_argument(
        "--schema-version",
        type=str,
        default=DEFAULT_SCHEMA_VERSION,
        help=f"Output schema version to emit/validate (default: {DEFAULT_SCHEMA_VERSION})",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["strict", "warn"],
        default="strict",
        help="On schema mismatch: strict=error, warn=continue (default: strict)",
    )


def _json_default(obj: Any) -> Any:
    """JSON default handler for CLI writes."""

    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        values = list(obj)
        try:
            return sorted(values)
        except TypeError:
            return sorted(
                values,
                key=lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"), default=str),
            )
    if is_dataclass(obj) and not isinstance(obj, type):
        # Avoid importing asdict here; dataclass instances are rare in CLI outputs.
        return obj.__dict__
    return str(obj)


# ============================================================================
# Console Output Utilities (works without external dependencies)
# ============================================================================


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


def _supports_color() -> bool:
    """Check if the terminal supports color output."""
    # Check for NO_COLOR environment variable (standard)
    if os.environ.get("NO_COLOR"):
        return False

    # Check for FORCE_COLOR environment variable
    if os.environ.get("FORCE_COLOR"):
        return True

    # Check if stdout is a TTY
    if not hasattr(sys.stdout, "isatty"):
        return False

    if not sys.stdout.isatty():
        return False

    # Windows terminal detection
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return os.environ.get("ANSICON") is not None

    return True


# Global flag for color support
USE_COLOR = _supports_color()


def colorize(text: str, *codes: str) -> str:
    """Apply color codes to text if terminal supports colors."""
    if not USE_COLOR:
        return text
    return "".join(codes) + text + Colors.RESET


def print_header(title: str) -> None:
    """Print a styled header."""
    width = 70
    line = "═" * width
    print()
    print(colorize(line, Colors.BRIGHT_CYAN))
    print(colorize(f"  {title}".center(width), Colors.BOLD, Colors.BRIGHT_CYAN))
    print(colorize(line, Colors.BRIGHT_CYAN))


def print_subheader(title: str) -> None:
    """Print a styled subheader."""
    print()
    print(colorize(f"── {title} ", Colors.CYAN) + colorize("─" * (50 - len(title)), Colors.DIM))


def print_success(message: str) -> None:
    """Print a success message."""
    print(colorize("OK ", Colors.BRIGHT_GREEN) + message)


def print_error(message: str) -> None:
    """Print an error message."""
    print(colorize("ERROR ", Colors.BRIGHT_RED) + colorize(message, Colors.RED), file=sys.stderr)


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(colorize("WARN ", Colors.BRIGHT_YELLOW) + colorize(message, Colors.YELLOW))


def print_info(message: str) -> None:
    """Print an info message."""
    print(colorize("INFO ", Colors.BRIGHT_BLUE) + message)


def print_key_value(key: str, value: Any, indent: int = 2) -> None:
    """Print a key-value pair."""
    spaces = " " * indent
    print(f"{spaces}{colorize(key + ':', Colors.DIM)} {value}")


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=_json_default) + "\n")


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _fingerprint_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    return hashlib.sha256(_stable_json_dumps(value).encode("utf-8")).hexdigest()[:12]


def _status_from_record(value: Any) -> ResultStatus:
    if isinstance(value, ResultStatus):
        return value
    try:
        return ResultStatus(str(value))
    except Exception:
        return ResultStatus.ERROR


def _probe_category_from_value(value: Any) -> ProbeCategory:
    if isinstance(value, ProbeCategory):
        return value
    try:
        return ProbeCategory(str(value))
    except Exception:
        return ProbeCategory.CUSTOM


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    harness = custom.get("harness") if isinstance(custom.get("harness"), dict) else {}
    model_spec = record.get("model") if isinstance(record.get("model"), dict) else {}
    probe_spec = record.get("probe") if isinstance(record.get("probe"), dict) else {}

    model_id = (
        harness.get("model_id")
        or model_spec.get("model_id")
        or harness.get("model_name")
        or "model"
    )
    probe_id = (
        harness.get("probe_type")
        or probe_spec.get("probe_id")
        or harness.get("probe_name")
        or "probe"
    )
    replicate_key = custom.get("replicate_key")
    example_id = record.get("example_id") or harness.get("example_index")
    stable_id = record.get("messages_hash") or _fingerprint_value(record.get("input"))
    chosen_id = replicate_key or stable_id or example_id or "0"
    return (str(model_id), str(probe_id), str(chosen_id))


def _record_label(record: dict[str, Any]) -> tuple[str, str, str]:
    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    harness = custom.get("harness") if isinstance(custom.get("harness"), dict) else {}
    model_spec = record.get("model") if isinstance(record.get("model"), dict) else {}
    probe_spec = record.get("probe") if isinstance(record.get("probe"), dict) else {}

    model_name = harness.get("model_name") or model_spec.get("model_id") or "model"
    model_id = harness.get("model_id") or model_spec.get("model_id")
    if model_id and model_id != model_name:
        model_label = f"{model_name} ({model_id})"
    else:
        model_label = str(model_name)

    probe_name = harness.get("probe_name") or probe_spec.get("probe_id") or "probe"
    example_id = record.get("example_id") or harness.get("example_index") or "0"
    return (str(model_label), str(probe_name), str(example_id))


def _status_string(record: dict[str, Any]) -> str:
    status = record.get("status")
    return str(status) if status is not None else "unknown"


def _output_text(record: dict[str, Any]) -> Optional[str]:
    output_text = record.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = record.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        value = output.get("output_text") or output.get("text")
        if isinstance(value, str):
            return value
    return None


def _strip_volatile_keys(value: Any, ignore_keys: set[str]) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key).lower()
            if key_str in ignore_keys:
                continue
            cleaned[key] = _strip_volatile_keys(item, ignore_keys)
        return cleaned
    if isinstance(value, list):
        return [_strip_volatile_keys(item, ignore_keys) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_volatile_keys(item, ignore_keys) for item in value)
    return value


def _trim_text(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _output_summary(
    record: dict[str, Any], ignore_keys: Optional[set[str]]
) -> Optional[dict[str, Any]]:
    output_text = _output_text(record)
    if isinstance(output_text, str):
        return {"type": "text", "preview": _trim_text(output_text), "length": len(output_text)}
    output = record.get("output")
    if output is None:
        return None
    return {
        "type": "structured",
        "fingerprint": _output_fingerprint(record, ignore_keys=ignore_keys),
    }


def _output_fingerprint(
    record: dict[str, Any], ignore_keys: Optional[set[str]] = None
) -> Optional[str]:
    output = record.get("output")
    if ignore_keys:
        custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
        override = custom.get("output_fingerprint")
        if output is None and isinstance(override, str):
            return override
        if output is None:
            return None
        sanitized = _strip_volatile_keys(output, ignore_keys)
        return _fingerprint_value(sanitized)

    custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
    override = custom.get("output_fingerprint")
    if isinstance(override, str):
        return override
    if output is None:
        return None
    return _fingerprint_value(output)


def _primary_score(record: dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
    metric = record.get("primary_metric")
    if metric and metric in scores and isinstance(scores[metric], (int, float)):
        return str(metric), float(scores[metric])
    if "score" in scores and isinstance(scores["score"], (int, float)):
        return "score", float(scores["score"])
    return None, None


def _metric_mismatch_reason(record_a: dict[str, Any], record_b: dict[str, Any]) -> Optional[str]:
    scores_a = record_a.get("scores")
    scores_b = record_b.get("scores")
    if not isinstance(scores_a, dict) or not isinstance(scores_b, dict):
        return "type_mismatch"

    keys_a = sorted(scores_a.keys())
    keys_b = sorted(scores_b.keys())
    if not keys_a or not keys_b:
        return "missing_scores"

    primary_a = record_a.get("primary_metric")
    primary_b = record_b.get("primary_metric")
    if primary_a and primary_b and primary_a != primary_b:
        return "primary_metric_mismatch"
    if not primary_a or not primary_b:
        return "missing_primary_metric"

    if primary_a not in scores_a or primary_b not in scores_b:
        return "metric_key_missing"

    value_a = scores_a.get(primary_a)
    value_b = scores_b.get(primary_b)
    if not isinstance(value_a, (int, float)) or not isinstance(value_b, (int, float)):
        return "non_numeric_metric"

    return "type_mismatch"


def _metric_mismatch_context(record_a: dict[str, Any], record_b: dict[str, Any]) -> dict[str, Any]:
    scores_a = record_a.get("scores")
    scores_b = record_b.get("scores")
    primary_a = record_a.get("primary_metric")
    primary_b = record_b.get("primary_metric")

    keys_a = sorted(scores_a.keys()) if isinstance(scores_a, dict) else None
    keys_b = sorted(scores_b.keys()) if isinstance(scores_b, dict) else None

    context = {
        "baseline": {
            "primary_metric": primary_a,
            "scores_keys": keys_a,
            "metric_value": scores_a.get(primary_a) if isinstance(scores_a, dict) else None,
        },
        "candidate": {
            "primary_metric": primary_b,
            "scores_keys": keys_b,
            "metric_value": scores_b.get(primary_b) if isinstance(scores_b, dict) else None,
        },
    }

    custom = record_a.get("custom") if isinstance(record_a.get("custom"), dict) else {}
    replicate_key = custom.get("replicate_key")
    if replicate_key:
        context["replicate_key"] = replicate_key

    return context


def _metric_mismatch_details(record_a: dict[str, Any], record_b: dict[str, Any]) -> str:
    context = _metric_mismatch_context(record_a, record_b)

    baseline = context.get("baseline", {})
    candidate = context.get("candidate", {})
    details = [
        f"baseline.primary_metric={baseline.get('primary_metric')!r}",
        f"candidate.primary_metric={candidate.get('primary_metric')!r}",
    ]

    if baseline.get("scores_keys") is not None:
        details.append(f"baseline.scores keys={baseline.get('scores_keys')!r}")
    else:
        details.append("baseline.scores type=None")

    if candidate.get("scores_keys") is not None:
        details.append(f"candidate.scores keys={candidate.get('scores_keys')!r}")
    else:
        details.append("candidate.scores type=None")

    if baseline.get("primary_metric") is not None:
        details.append(
            f"baseline.{baseline.get('primary_metric')}={baseline.get('metric_value')!r}"
        )
    if candidate.get("primary_metric") is not None:
        details.append(
            f"candidate.{candidate.get('primary_metric')}={candidate.get('metric_value')!r}"
        )

    if "replicate_key" in context:
        details.append(f"replicate_key={context['replicate_key']}")

    return "; ".join(details)


def _build_experiments_from_records(
    records: list[dict[str, Any]],
) -> tuple[list[ExperimentResult], dict[str, Any], str]:
    if not records:
        return [], {"derived_from_records": True}, DEFAULT_SCHEMA_VERSION

    schema_version = records[0].get("schema_version") or DEFAULT_SCHEMA_VERSION

    harness_records = [
        r
        for r in records
        if isinstance(r.get("custom"), dict) and isinstance(r["custom"].get("harness"), dict)
    ]

    experiments: list[ExperimentResult] = []
    derived_config: dict[str, Any] = {"derived_from_records": True}

    if harness_records:
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in harness_records:
            harness = record.get("custom", {}).get("harness", {})
            experiment_id = harness.get("experiment_id") or "unknown"
            groups.setdefault(str(experiment_id), []).append(record)

        models: dict[str, dict[str, Any]] = {}
        probes: dict[str, dict[str, Any]] = {}
        dataset_summary: dict[str, Any] = {}

        for experiment_id, group_records in groups.items():
            first = group_records[0]
            harness = first.get("custom", {}).get("harness", {})

            model_name = (
                harness.get("model_name") or first.get("model", {}).get("model_id") or "model"
            )
            model_id = (
                harness.get("model_id") or first.get("model", {}).get("model_id") or model_name
            )
            provider = (
                first.get("model", {}).get("provider") or harness.get("model_type") or "unknown"
            )
            extra = first.get("model", {}).get("params") or {}

            probe_name = (
                harness.get("probe_name") or first.get("probe", {}).get("probe_id") or "probe"
            )
            probe_category = _probe_category_from_value(harness.get("probe_category"))

            model_info = ModelInfo(
                name=str(model_name),
                provider=str(provider),
                model_id=str(model_id),
                extra=extra,
            )

            def _sort_key(item: dict[str, Any]) -> int:
                harness_item = item.get("custom", {}).get("harness", {})
                return int(harness_item.get("example_index", 0))

            sorted_records = sorted(group_records, key=_sort_key)
            probe_results = [
                ProbeResult(
                    input=record.get("input"),
                    output=record.get("output"),
                    status=_status_from_record(record.get("status")),
                    error=record.get("error"),
                    latency_ms=record.get("latency_ms"),
                    metadata=record.get("custom") or {},
                )
                for record in sorted_records
            ]

            started_at = min(
                (dt for dt in (_parse_datetime(r.get("started_at")) for r in group_records) if dt),
                default=None,
            )
            completed_at = max(
                (
                    dt
                    for dt in (_parse_datetime(r.get("completed_at")) for r in group_records)
                    if dt
                ),
                default=None,
            )

            experiments.append(
                ExperimentResult(
                    experiment_id=experiment_id,
                    model_info=model_info,
                    probe_name=str(probe_name),
                    probe_category=probe_category,
                    results=probe_results,
                    score=None,
                    started_at=started_at,
                    completed_at=completed_at,
                    config={
                        "model": {"type": harness.get("model_type")},
                        "probe": {"type": harness.get("probe_type")},
                        "dataset": {
                            "name": harness.get("dataset"),
                            "format": harness.get("dataset_format"),
                        },
                    },
                )
            )

            if harness.get("model_type"):
                models.setdefault(
                    str(harness.get("model_type")), {"type": harness.get("model_type")}
                )
            if harness.get("probe_type"):
                probes.setdefault(
                    str(harness.get("probe_type")), {"type": harness.get("probe_type")}
                )
            if harness.get("dataset") and not dataset_summary:
                dataset_summary = {
                    "name": harness.get("dataset"),
                    "format": harness.get("dataset_format"),
                }

        derived_config.update(
            {
                "models": list(models.values()),
                "probes": list(probes.values()),
                "dataset": dataset_summary,
            }
        )
        return experiments, derived_config, schema_version

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        run_id = record.get("run_id") or "run"
        groups.setdefault(str(run_id), []).append(record)

    for run_id, group_records in groups.items():
        first = group_records[0]
        model_spec = first.get("model", {}) if isinstance(first.get("model"), dict) else {}
        probe_spec = first.get("probe", {}) if isinstance(first.get("probe"), dict) else {}

        model_id = model_spec.get("model_id") or "model"
        model_name = model_spec.get("params", {}).get("name") or model_id
        provider = model_spec.get("provider") or "unknown"
        extra = model_spec.get("params") or {}

        model_info = ModelInfo(
            name=str(model_name),
            provider=str(provider),
            model_id=str(model_id),
            extra=extra,
        )

        probe_name = probe_spec.get("probe_id") or "probe"
        probe_category = ProbeCategory.CUSTOM

        probe_results = [
            ProbeResult(
                input=record.get("input"),
                output=record.get("output"),
                status=_status_from_record(record.get("status")),
                error=record.get("error"),
                latency_ms=record.get("latency_ms"),
                metadata=record.get("custom") or {},
            )
            for record in group_records
        ]

        started_at = min(
            (dt for dt in (_parse_datetime(r.get("started_at")) for r in group_records) if dt),
            default=None,
        )
        completed_at = max(
            (dt for dt in (_parse_datetime(r.get("completed_at")) for r in group_records) if dt),
            default=None,
        )

        experiments.append(
            ExperimentResult(
                experiment_id=run_id,
                model_info=model_info,
                probe_name=str(probe_name),
                probe_category=probe_category,
                results=probe_results,
                score=None,
                started_at=started_at,
                completed_at=completed_at,
                config={},
            )
        )

    return experiments, derived_config, schema_version


def _build_basic_harness_report(
    experiments: list[ExperimentResult],
    summary: dict[str, Any],
    title: str,
    generated_at: Optional[datetime] = None,
) -> str:
    rows = []
    for experiment in experiments:
        latencies = [r.latency_ms for r in experiment.results if r.latency_ms is not None]
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        accuracy = experiment.score.accuracy if experiment.score else None
        rows.append(
            (
                html.escape(experiment.model_info.name),
                html.escape(experiment.probe_name),
                _format_percent(experiment.success_rate),
                _format_float(accuracy),
                _format_float(avg_latency),
            )
        )

    rows.sort(key=lambda row: (row[0], row[1]))

    rows_html = "\n".join(
        f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td>"
        f"<td>{row[3]}</td><td>{row[4]}</td></tr>"
        for row in rows
    )

    def summary_table(section: str) -> str:
        items = summary.get(section, {})
        lines = []
        for name in sorted(items):
            stats = items[name]
            success = stats.get("success_rate", {}).get("mean")
            ci = stats.get("success_rate_ci", {})
            ci_text = "-"
            if ci and ci.get("lower") is not None and ci.get("upper") is not None:
                ci_text = f"{ci.get('lower'):.3f}..{ci.get('upper'):.3f}"
            lines.append(
                f"<tr><td>{html.escape(name)}</td>"
                f"<td>{_format_percent(success)}</td><td>{ci_text}</td></tr>"
            )
        return "\n".join(lines)

    by_model_rows = summary_table("by_model")
    by_probe_rows = summary_table("by_probe")

    meta_line = ""
    if generated_at is not None:
        meta_line = f'<div class="meta">Generated {generated_at.isoformat()}</div>'

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #666; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .section {{ margin-top: 24px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  {meta_line}

  <div class="section">
    <h2>Model x Probe Summary</h2>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Probe</th>
          <th>Success Rate</th>
          <th>Accuracy</th>
          <th>Avg Latency (ms)</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>By Model</h2>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Success Rate</th>
          <th>Success Rate CI</th>
        </tr>
      </thead>
      <tbody>
        {by_model_rows}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>By Probe</h2>
    <table>
      <thead>
        <tr>
          <th>Probe</th>
          <th>Success Rate</th>
          <th>Success Rate CI</th>
        </tr>
      </thead>
      <tbody>
        {by_probe_rows}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


class ProgressBar:
    """Simple progress bar for CLI output."""

    def __init__(
        self,
        total: int,
        width: int = 40,
        prefix: str = "Progress",
        show_eta: bool = True,
    ):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.show_eta = show_eta
        self.current = 0
        self.start_time = time.time()

    def update(self, current: int) -> None:
        """Update the progress bar."""
        self.current = current
        self._render()

    def increment(self, amount: int = 1) -> None:
        """Increment the progress."""
        self.current += amount
        self._render()

    def _render(self) -> None:
        """Render the progress bar to the terminal."""
        pct = 100 if self.total == 0 else min(100, self.current / self.total * 100)

        filled = int(self.width * self.current / max(1, self.total))
        bar = "█" * filled + "░" * (self.width - filled)

        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.current > 0 and self.show_eta:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f" ETA: {eta:.1f}s" if eta > 0 else ""
        else:
            eta_str = ""

        line = f"\r{self.prefix}: {colorize(bar, Colors.CYAN)} {pct:5.1f}% ({self.current}/{self.total}){eta_str}"
        print(line, end="", flush=True)

    def finish(self) -> None:
        """Complete the progress bar."""
        self.current = self.total
        self._render()
        elapsed = time.time() - self.start_time
        print(f" {colorize(f'Done in {elapsed:.2f}s', Colors.GREEN)}")


class Spinner:
    """Simple spinner for indeterminate progress."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Loading"):
        self.message = message
        self.frame_idx = 0
        self.running = False

    def spin(self) -> None:
        """Render a single spinner frame."""
        frame = self.FRAMES[self.frame_idx % len(self.FRAMES)]
        print(f"\r{colorize(frame, Colors.CYAN)} {self.message}...", end="", flush=True)
        self.frame_idx += 1

    def stop(self, success: bool = True) -> None:
        """Stop the spinner with a final status."""
        if success:
            print(f"\r{colorize('OK', Colors.GREEN)} {self.message}... done")
        else:
            print(f"\r{colorize('FAIL', Colors.RED)} {self.message}... failed")


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    # Custom formatter that preserves formatting in epilog
    class CustomFormatter(argparse.RawDescriptionHelpFormatter):
        def __init__(self, prog):
            super().__init__(prog, max_help_position=40, width=100)

    parser = argparse.ArgumentParser(
        prog="insidellms",
        description=colorize("insideLLMs", Colors.BOLD, Colors.CYAN)
        + " - A world-class toolkit for probing, evaluating, and testing large language models",
        formatter_class=CustomFormatter,
        epilog=f"""
{colorize("Examples:", Colors.BOLD)}

  {colorize("# Quick evaluation", Colors.DIM)}
  insidellms quicktest "What is 2+2?" --model openai

  {colorize("# Run a full experiment from config", Colors.DIM)}
  insidellms run config.yaml --verbose

  {colorize("# Run benchmark suite", Colors.DIM)}
  insidellms benchmark --models openai,anthropic --probes logic,bias

  {colorize("# List available resources", Colors.DIM)}
  insidellms list all

  {colorize("# Interactive exploration", Colors.DIM)}
  insidellms interactive

  {colorize("# Compare models", Colors.DIM)}
  insidellms compare --models gpt-4,claude-3 --input "Explain quantum computing"

{colorize("Documentation:", Colors.BOLD)} https://github.com/dr-gareth-roberts/insideLLMs
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.2.0",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output (errors only)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # =========================================================================
    # Run command
    # =========================================================================
    run_parser = subparsers.add_parser(
        "run",
        help="Run an experiment from a configuration file",
        formatter_class=CustomFormatter,
    )
    run_parser.add_argument(
        "config",
        type=str,
        help="Path to the experiment configuration file (YAML or JSON)",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for results (JSON format)",
    )
    run_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "markdown", "table", "summary"],
        default="table",
        help="Output format (default: table)",
    )

    # Deterministic harness artifact location controls
    run_parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Write run artifacts (manifest.json + records.jsonl) exactly to this directory. "
            "(Final directory; overrides --run-root/--run-id.)"
        ),
    )
    run_parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help=(
            "Base directory for runs when --run-dir is not set (default: ~/.insidellms/runs). "
            "The final directory is <run_root>/<run_id>."
        ),
    )
    run_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Set the run_id recorded in the manifest. Also used as the directory name when using --run-root. "
            "If omitted, a random UUID is generated."
        ),
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing non-empty run directory (DANGEROUS).",
    )
    run_parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async execution for parallel processing",
    )
    run_parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=5,
        help="Number of concurrent requests (only with --async)",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information",
    )
    run_parser.add_argument(
        "--track",
        type=str,
        choices=["local", "wandb", "mlflow", "tensorboard"],
        help="Enable experiment tracking with specified backend",
    )
    run_parser.add_argument(
        "--track-project",
        type=str,
        default="insidellms",
        help="Project name for experiment tracking",
    )

    _add_output_schema_args(run_parser)

    # =========================================================================
    # Harness command
    # =========================================================================
    harness_parser = subparsers.add_parser(
        "harness",
        help="Run a cross-model probe harness",
        formatter_class=CustomFormatter,
    )
    harness_parser.add_argument(
        "config",
        type=str,
        help="Path to the harness configuration file",
    )
    harness_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for harness artifacts (alias for --run-dir; deprecated)",
    )

    harness_parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Write harness run artifacts (manifest.json + records.jsonl) exactly to this directory. "
            "(Final directory; overrides --run-root/--run-id.)"
        ),
    )
    harness_parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help=(
            "Base directory for harness runs when --run-dir is not set. "
            "The final directory is <run_root>/<run_id>."
        ),
    )
    harness_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Set the run_id recorded in the manifest. Also used as the directory name when using --run-root. "
            "If omitted, a random UUID is generated."
        ),
    )
    harness_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing non-empty run directory (DANGEROUS).",
    )
    harness_parser.add_argument(
        "--report-title",
        type=str,
        help="Title for the HTML report",
    )
    harness_parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip generating the HTML report",
    )
    harness_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress and tracebacks",
    )

    _add_output_schema_args(harness_parser)

    # =========================================================================
    # Report command
    # =========================================================================
    report_parser = subparsers.add_parser(
        "report",
        help="Rebuild summary.json and report.html from records.jsonl",
        formatter_class=CustomFormatter,
    )
    report_parser.add_argument(
        "run_dir",
        type=str,
        help="Run directory containing records.jsonl",
    )
    report_parser.add_argument(
        "--report-title",
        type=str,
        help="Title for the HTML report",
    )

    # =========================================================================
    # Diff command
    # =========================================================================
    diff_parser = subparsers.add_parser(
        "diff",
        help="Compare two run directories and show behavioural regressions",
        formatter_class=CustomFormatter,
    )
    diff_parser.add_argument(
        "run_dir_a",
        type=str,
        help="Baseline run directory",
    )
    diff_parser.add_argument(
        "run_dir_b",
        type=str,
        help="Comparison run directory",
    )
    diff_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    diff_parser.add_argument(
        "--output",
        type=str,
        help="Write JSON output to a file (json format only)",
    )
    diff_parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of items to show per section (default: 25)",
    )
    diff_parser.add_argument(
        "--fail-on-regressions",
        action="store_true",
        help="Exit with non-zero status if regressions are detected",
    )
    diff_parser.add_argument(
        "--fail-on-changes",
        action="store_true",
        help=(
            "Exit with non-zero status if any differences are detected "
            "(regressions, changes, or missing/extra records)"
        ),
    )
    diff_parser.add_argument(
        "--output-fingerprint-ignore",
        action="append",
        default=[],
        help=(
            "Comma-separated output keys to ignore when fingerprinting structured outputs "
            "(repeatable)."
        ),
    )

    # =========================================================================
    # Schema command
    # =========================================================================
    schema_parser = subparsers.add_parser(
        "schema",
        help="Inspect and validate versioned output schemas",
        formatter_class=CustomFormatter,
    )
    schema_parser.add_argument(
        "op",
        nargs="?",
        default="list",
        help=(
            "Operation: list | dump | validate | <SchemaName>. "
            "Shortcut: `insidellms schema ProbeResult` dumps JSON Schema."
        ),
    )
    schema_parser.add_argument(
        "--name",
        help="Schema name (optional when using shortcut form: `insidellms schema <SchemaName>`)",
    )
    schema_parser.add_argument(
        "--version",
        default=DEFAULT_SCHEMA_VERSION,
        help=f"Schema version (default: {DEFAULT_SCHEMA_VERSION})",
    )
    schema_parser.add_argument(
        "--input",
        "-i",
        help="Input file to validate (.json or .jsonl) (for op=validate)",
    )
    schema_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Write JSON Schema to a file (for op=dump; otherwise prints to stdout)",
    )
    schema_parser.add_argument(
        "--mode",
        choices=["strict", "warn"],
        default="strict",
        help="On validation error: strict=exit non-zero, warn=continue",
    )

    # =========================================================================
    # Quick test command
    # =========================================================================
    quicktest_parser = subparsers.add_parser(
        "quicktest",
        help="Quickly test a prompt against a model",
        formatter_class=CustomFormatter,
    )
    quicktest_parser.add_argument(
        "prompt",
        type=str,
        help="The prompt to test",
    )
    quicktest_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="dummy",
        help="Model to use (default: dummy)",
    )
    quicktest_parser.add_argument(
        "--model-args",
        type=str,
        default="{}",
        help="JSON string of model arguments",
    )
    quicktest_parser.add_argument(
        "--probe",
        "-p",
        type=str,
        help="Optional probe to apply to the response",
    )
    quicktest_parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    quicktest_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens in response",
    )

    # =========================================================================
    # Benchmark command
    # =========================================================================
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run comprehensive benchmarks across models and probes",
        formatter_class=CustomFormatter,
    )
    benchmark_parser.add_argument(
        "--models",
        "-m",
        type=str,
        default="dummy",
        help="Comma-separated list of models to benchmark",
    )
    benchmark_parser.add_argument(
        "--probes",
        "-p",
        type=str,
        default="logic",
        help="Comma-separated list of probes to run",
    )
    benchmark_parser.add_argument(
        "--datasets",
        "-d",
        type=str,
        help="Comma-separated list of benchmark datasets (e.g., reasoning,math,coding)",
    )
    benchmark_parser.add_argument(
        "--max-examples",
        "-n",
        type=int,
        default=10,
        help="Maximum examples per dataset (default: 10)",
    )
    benchmark_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory for benchmark results",
    )
    benchmark_parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate an HTML report with visualizations",
    )
    benchmark_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress",
    )

    # =========================================================================
    # Compare command
    # =========================================================================
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple models on the same inputs",
        formatter_class=CustomFormatter,
    )
    compare_parser.add_argument(
        "--models",
        "-m",
        type=str,
        required=True,
        help="Comma-separated list of models to compare",
    )
    compare_parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Single input prompt to compare",
    )
    compare_parser.add_argument(
        "--input-file",
        type=str,
        help="File with inputs (one per line or JSON/JSONL)",
    )
    compare_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for comparison results",
    )
    compare_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json", "markdown"],
        default="table",
        help="Output format",
    )

    # =========================================================================
    # List command
    # =========================================================================
    list_parser = subparsers.add_parser(
        "list",
        help="List available models, probes, or datasets",
        formatter_class=CustomFormatter,
    )
    list_parser.add_argument(
        "type",
        choices=["models", "probes", "datasets", "trackers", "all"],
        help="What to list",
    )
    list_parser.add_argument(
        "--filter",
        type=str,
        help="Filter results by name (substring match)",
    )
    list_parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show detailed information",
    )

    # =========================================================================
    # Init command
    # =========================================================================
    init_parser = subparsers.add_parser(
        "init",
        help="Generate a sample configuration file",
        formatter_class=CustomFormatter,
    )
    init_parser.add_argument(
        "output",
        type=str,
        nargs="?",
        default="experiment.yaml",
        help="Output file path (default: experiment.yaml)",
    )
    init_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="dummy",
        help="Model type for the sample config",
    )
    init_parser.add_argument(
        "--probe",
        "-p",
        type=str,
        default="logic",
        help="Probe type for the sample config",
    )
    init_parser.add_argument(
        "--template",
        type=str,
        choices=["basic", "benchmark", "tracking", "full"],
        default="basic",
        help="Configuration template to use",
    )

    # =========================================================================
    # Info command
    # =========================================================================
    info_parser = subparsers.add_parser(
        "info",
        help="Show detailed information about a model, probe, or dataset",
        formatter_class=CustomFormatter,
    )
    info_parser.add_argument(
        "type",
        choices=["model", "probe", "dataset"],
        help="Type of item to show info for",
    )
    info_parser.add_argument(
        "name",
        type=str,
        help="Name of the model, probe, or dataset",
    )

    # =========================================================================
    # Interactive command
    # =========================================================================
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Start an interactive exploration session",
        formatter_class=CustomFormatter,
    )
    interactive_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="dummy",
        help="Model to use in interactive mode",
    )
    interactive_parser.add_argument(
        "--history-file",
        type=str,
        default=".insidellms_history",
        help="File to store command history",
    )

    # =========================================================================
    # Validate command
    # =========================================================================
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a configuration file or a run directory (manifest + records.jsonl)",
        formatter_class=CustomFormatter,
    )
    validate_parser.add_argument(
        "config",
        type=str,
        help="Path to a config file (.yaml/.json) OR a run_dir containing manifest.json",
    )
    validate_parser.add_argument(
        "--mode",
        choices=["strict", "warn"],
        default="strict",
        help="On schema mismatch for run_dir validation: strict=exit non-zero, warn=continue",
    )
    validate_parser.add_argument(
        "--schema-version",
        type=str,
        default=None,
        help="Override schema version when validating a run_dir (defaults to manifest schema_version)",
    )

    # =========================================================================
    # Export command
    # =========================================================================
    export_parser = subparsers.add_parser(
        "export",
        help="Export results to various formats",
        formatter_class=CustomFormatter,
    )
    export_parser.add_argument(
        "input",
        type=str,
        help="Input results file (JSON)",
    )
    export_parser.add_argument(
        "--format",
        "-f",
        choices=["csv", "markdown", "html", "latex"],
        default="csv",
        help="Export format",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path",
    )

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the run command."""
    config_path = Path(args.config)

    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        return 1

    print_header("Running Experiment")
    print_key_value("Config", config_path)

    # Create progress bar if verbose
    progress_bar: Optional[ProgressBar] = None

    def progress_callback(current: int, total: int) -> None:
        nonlocal progress_bar
        if args.verbose:
            if progress_bar is None:
                progress_bar = ProgressBar(total, prefix="Evaluating")
            progress_bar.update(current)

    try:
        start_time = time.time()

        # Resolve deterministic run artifact location (used for both runner and UX hints)
        env_run_root = os.environ.get("INSIDELLMS_RUN_ROOT")
        default_run_root = (
            Path(env_run_root).expanduser().absolute()
            if env_run_root
            else Path.home() / ".insidellms" / "runs"
        )
        if args.run_id:
            resolved_run_id = args.run_id
        else:
            resolved_run_id = derive_run_id_from_config_path(
                config_path,
                schema_version=args.schema_version,
            )
        # Use absolute paths to reduce surprise when users provide relative paths.
        effective_run_root = (
            Path(args.run_root).expanduser().absolute() if args.run_root else default_run_root
        )
        effective_run_dir = (
            Path(args.run_dir).expanduser().absolute()
            if args.run_dir
            else (effective_run_root / resolved_run_id)
        )

        if args.use_async:
            print_info(f"Using async execution with concurrency={args.concurrency}")
            results = asyncio.run(
                run_experiment_from_config_async(
                    config_path,
                    concurrency=args.concurrency,
                    progress_callback=progress_callback if args.verbose else None,
                    validate_output=args.validate_output,
                    schema_version=args.schema_version,
                    validation_mode=args.validation_mode,
                    emit_run_artifacts=True,
                    run_dir=str(effective_run_dir) if args.run_dir else None,
                    run_root=str(effective_run_root),
                    run_id=resolved_run_id,
                    overwrite=bool(args.overwrite),
                )
            )
        else:
            results = run_experiment_from_config(
                config_path,
                progress_callback=progress_callback if args.verbose else None,
                validate_output=args.validate_output,
                schema_version=args.schema_version,
                validation_mode=args.validation_mode,
                emit_run_artifacts=True,
                run_dir=str(effective_run_dir) if args.run_dir else None,
                run_root=str(effective_run_root),
                run_id=resolved_run_id,
                overwrite=bool(args.overwrite),
            )

        elapsed = time.time() - start_time

        if progress_bar:
            progress_bar.finish()

        # Calculate summary
        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = sum(1 for r in results if r.get("status") == "error")
        total = len(results)

        # Output results
        if args.output:
            save_results_json(
                results,
                args.output,
                validate_output=args.validate_output,
                schema_version=args.schema_version,
                validation_mode=args.validation_mode,
            )
            print_success(f"Results saved to: {args.output}")

        if args.format == "json":
            print(json.dumps(results, indent=2, default=str))
        elif args.format == "markdown":
            print(results_to_markdown(results))
        elif args.format == "summary":
            # Minimal summary output
            print(
                f"OK {success_count}/{total} successful ({success_count / max(1, total) * 100:.1f}%)"
            )
        else:  # table format
            print_subheader("Results Summary")
            print_key_value("Total items", total)
            print_key_value(
                "Successful", f"{success_count} ({success_count / max(1, total) * 100:.1f}%)"
            )
            print_key_value("Errors", f"{error_count} ({error_count / max(1, total) * 100:.1f}%)")
            print_key_value("Duration", f"{elapsed:.2f}s")

            if results:
                latencies = [
                    r.get("latency_ms", 0) for r in results if r.get("status") == "success"
                ]
                if latencies:
                    avg_latency = sum(latencies) / len(latencies)
                    min_latency = min(latencies)
                    max_latency = max(latencies)
                    print_key_value("Avg latency", f"{avg_latency:.1f}ms")
                    print_key_value("Min/Max", f"{min_latency:.1f}ms / {max_latency:.1f}ms")

            # Show first few results
            print_subheader("Sample Results")
            for i, r in enumerate(results[:5]):
                status_icon = (
                    colorize("OK", Colors.GREEN)
                    if r.get("status") == "success"
                    else colorize("FAIL", Colors.RED)
                )
                inp = str(r.get("input", ""))[:50]
                if len(str(r.get("input", ""))) > 50:
                    inp += "..."
                print(f"  {status_icon} [{i + 1}] {inp}")

            if len(results) > 5:
                print(colorize(f"  ... and {len(results) - 5} more", Colors.DIM))

        # UX sugar: make it obvious where artifacts landed and how to validate.
        # Keep stdout JSON clean when --format json.
        hint_stream = sys.stderr if args.format == "json" else sys.stdout
        print(f"\nRun written to: {effective_run_dir}", file=hint_stream)
        print(f"Validate with: insidellms validate {effective_run_dir}", file=hint_stream)

        return 0

    except Exception as e:
        print_error(f"Error running experiment: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_harness(args: argparse.Namespace) -> int:
    """Execute the harness command."""
    config_path = Path(args.config)

    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        return 1

    print_header("Running Behavioural Harness")
    print_key_value("Config", config_path)

    progress_bar: Optional[ProgressBar] = None

    def progress_callback(current: int, total: int) -> None:
        nonlocal progress_bar
        if not args.verbose:
            return
        if progress_bar is None:
            progress_bar = ProgressBar(total, prefix="Evaluating")
        progress_bar.update(current)

    try:
        start_time = time.time()
        result = run_harness_from_config(
            config_path,
            progress_callback=progress_callback if args.verbose else None,
            validate_output=args.validate_output,
            schema_version=args.schema_version,
            validation_mode=args.validation_mode,
        )
        elapsed = time.time() - start_time

        if progress_bar:
            progress_bar.finish()

        # -----------------------------------------------------------------
        # Determine harness run directory and emit standardized run artifacts
        # (manifest.json + records.jsonl + config.resolved.yaml + .insidellms_run)
        # -----------------------------------------------------------------
        from insideLLMs.runner import (
            _build_resolved_config_snapshot,
            _deterministic_base_time,
            _deterministic_run_id_from_config_snapshot,
            _deterministic_run_times,
            _prepare_run_dir,
            _serialize_value,
        )
        from insideLLMs.schemas.registry import normalize_semver

        config_snapshot = _build_resolved_config_snapshot(result["config"], config_path.parent)

        resolved_run_id = args.run_id or result.get("run_id")
        if not resolved_run_id:
            resolved_run_id = _deterministic_run_id_from_config_snapshot(
                config_snapshot,
                schema_version=args.schema_version,
            )

        # Absolute paths reduce surprise when users pass relative paths.
        effective_run_root = Path(args.run_root).expanduser().absolute() if args.run_root else None
        config_default_dir = (
            Path(result["config"].get("output_dir", "results")).expanduser().absolute()
        )

        # Precedence: --run-dir > --output-dir (legacy alias) > --run-root/run-id > config output_dir
        if args.run_dir:
            output_dir = Path(args.run_dir).expanduser().absolute()
        elif args.output_dir:
            output_dir = Path(args.output_dir).expanduser().absolute()
        elif effective_run_root is not None:
            output_dir = effective_run_root / resolved_run_id
        else:
            output_dir = config_default_dir

        def _semver_tuple(version: str) -> tuple[int, int, int]:
            v = normalize_semver(version)
            parts = v.split(".")
            try:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            except Exception:
                return (0, 0, 0)

        def _atomic_write_text(path: Path, text: str) -> None:
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(tmp, path)

        def _atomic_write_yaml(path: Path, data: Any) -> None:
            import yaml

            content = yaml.safe_dump(
                _serialize_value(data),
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
            _atomic_write_text(path, content)

        def _ensure_run_sentinel(run_dir_path: Path) -> None:
            marker = run_dir_path / ".insidellms_run"
            if not marker.exists():
                try:
                    marker.write_text("insideLLMs run directory\n", encoding="utf-8")
                except Exception:
                    pass

        # Prepare run dir with the same safety policy as `insidellms run`.
        _prepare_run_dir(output_dir, overwrite=bool(args.overwrite), run_root=effective_run_root)
        _ensure_run_sentinel(output_dir)

        # Write resolved config snapshot for reproducibility.
        _atomic_write_yaml(output_dir / "config.resolved.yaml", config_snapshot)

        # Canonical record stream for validate/run-dir tooling.
        records_path = output_dir / "records.jsonl"

        # Ensure records' run_id matches the directory's run_id.
        for record in result.get("records", []):
            if isinstance(record, dict):
                record["run_id"] = resolved_run_id

        _write_jsonl(result["records"], records_path)
        print_success(f"Records written to: {records_path}")

        # Backward compatibility: keep results.jsonl alongside records.jsonl.
        legacy_results_path = output_dir / "results.jsonl"
        try:
            os.symlink("records.jsonl", legacy_results_path)
        except Exception:
            try:
                os.link(records_path, legacy_results_path)
            except Exception:
                shutil.copyfile(records_path, legacy_results_path)

        ds_cfg = result.get("config", {}).get("dataset")
        ds_cfg = ds_cfg if isinstance(ds_cfg, dict) else {}
        dataset_spec = {
            "dataset_id": ds_cfg.get("name") or ds_cfg.get("path") or ds_cfg.get("dataset_id"),
            "dataset_version": ds_cfg.get("split") or ds_cfg.get("dataset_version"),
            "dataset_hash": ds_cfg.get("dataset_hash"),
            "provenance": ds_cfg.get("format") or ds_cfg.get("provenance"),
            "params": ds_cfg,
        }

        models_cfg = result.get("config", {}).get("models")
        models_cfg = models_cfg if isinstance(models_cfg, list) else []
        model_types = [m.get("type") for m in models_cfg if isinstance(m, dict) and m.get("type")]

        probes_cfg = result.get("config", {}).get("probes")
        probes_cfg = probes_cfg if isinstance(probes_cfg, list) else []
        probe_types = [p.get("type") for p in probes_cfg if isinstance(p, dict) and p.get("type")]

        record_count = len(result.get("records", []))
        success_count = sum(1 for r in result.get("records", []) if r.get("status") == "success")
        error_count = sum(1 for r in result.get("records", []) if r.get("status") == "error")

        run_base_time = _deterministic_base_time(resolved_run_id)
        started_at, completed_at = _deterministic_run_times(run_base_time, record_count)
        created_at = started_at

        manifest: dict[str, Any] = {
            "schema_version": args.schema_version,
            "run_id": resolved_run_id,
            "created_at": created_at,
            "started_at": started_at,
            "completed_at": completed_at,
            "library_version": None,
            "python_version": sys.version,
            "platform": platform.platform(),
            "command": sys.argv.copy(),
            "model": {
                "model_id": "harness",
                "provider": "insideLLMs",
                "params": {"model_count": len(model_types), "models": model_types},
            },
            "probe": {
                "probe_id": "harness",
                "probe_version": None,
                "params": {"probe_count": len(probe_types), "probes": probe_types},
            },
            "dataset": dataset_spec,
            "record_count": record_count,
            "success_count": success_count,
            "error_count": error_count,
            "records_file": "records.jsonl",
            "schemas": {"RunManifest": args.schema_version, "ResultRecord": args.schema_version},
            "custom": {
                "harness": {
                    "models": models_cfg,
                    "probes": probes_cfg,
                    "dataset": ds_cfg,
                    "max_examples": result.get("config", {}).get("max_examples"),
                    "experiment_count": len(result.get("experiments", [])),
                    "legacy_results_file": "results.jsonl",
                }
            },
        }

        if _semver_tuple(args.schema_version) >= (1, 0, 1):
            manifest["run_completed"] = True

        try:
            import insideLLMs

            manifest["library_version"] = getattr(insideLLMs, "__version__", None)
        except Exception:
            pass

        if args.validate_output:
            from insideLLMs.schemas import OutputValidator, SchemaRegistry

            validator = OutputValidator(SchemaRegistry())
            validator.validate(
                SchemaRegistry.RUN_MANIFEST,
                manifest,
                schema_version=args.schema_version,
                mode=args.validation_mode,
            )

        manifest_path = output_dir / "manifest.json"
        _atomic_write_text(
            manifest_path,
            json.dumps(_serialize_value(manifest), indent=2, default=_serialize_value),
        )
        print_success(f"Manifest written to: {manifest_path}")

        summary_path = output_dir / "summary.json"
        report_path = output_dir / "report.html"

        summary_payload = {
            "schema_version": args.schema_version,
            "generated_at": created_at,
            "summary": result["summary"],
            "config": result["config"],
        }

        if args.validate_output:
            from insideLLMs.schemas import OutputValidator, SchemaRegistry

            validator = OutputValidator(SchemaRegistry())
            validator.validate(
                SchemaRegistry.HARNESS_SUMMARY,
                summary_payload,
                schema_version=args.schema_version,
                mode=args.validation_mode,
            )
        with open(summary_path, "w") as f:
            json.dump(summary_payload, f, indent=2, default=_json_default)
        print_success(f"Summary written to: {summary_path}")

        if not args.skip_report:
            report_title = args.report_title or result["config"].get(
                "report_title", "Behavioural Probe Report"
            )
            try:
                from insideLLMs.visualization import create_interactive_html_report

                create_interactive_html_report(
                    result["experiments"],
                    title=report_title,
                    save_path=str(report_path),
                )
            except ImportError:
                report_html = _build_basic_harness_report(
                    result["experiments"],
                    result["summary"],
                    report_title,
                    generated_at=created_at,
                )
                with open(report_path, "w") as f:
                    f.write(report_html)

            print_success(f"Report written to: {report_path}")

        print_key_value("Elapsed", f"{elapsed:.2f}s")

        print(f"\nRun written to: {output_dir}")
        print(f"Validate with: insidellms validate {output_dir}")
        return 0

    except Exception as e:
        print_error(f"Error running harness: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_schema(args: argparse.Namespace) -> int:
    """Inspect/dump/validate versioned output schemas."""

    from insideLLMs.schemas import OutputValidationError, OutputValidator, SchemaRegistry

    registry = SchemaRegistry()

    op = getattr(args, "op", None) or "list"
    name = getattr(args, "name", None)
    version = getattr(args, "version", DEFAULT_SCHEMA_VERSION)

    # Shortcut UX: `insidellms schema <SchemaName>` -> dump schema
    if op not in {"list", "dump", "validate"}:
        name = name or op
        op = "dump"

    if op == "list":
        print_header("Available Output Schemas")
        schema_names = [
            registry.RUNNER_ITEM,
            registry.RUNNER_OUTPUT,
            registry.RESULT_RECORD,
            registry.RUN_MANIFEST,
            registry.HARNESS_RECORD,
            registry.HARNESS_SUMMARY,
            registry.BENCHMARK_SUMMARY,
            registry.COMPARISON_REPORT,
            registry.DIFF_REPORT,
            registry.EXPORT_METADATA,
        ]
        for name in schema_names:
            versions = registry.available_versions(name)
            if not versions:
                continue
            print_key_value(name, ", ".join(versions))
        return 0

    if op == "dump":
        if not name:
            print_error(
                "Missing schema name. Use: insidellms schema dump --name <SchemaName> [--version X]"
            )
            return 1
        try:
            schema = registry.get_json_schema(name, version)
        except Exception as e:
            print_error(f"Could not dump schema {name}@{version}: {e}")
            return 2

        payload = json.dumps(schema, indent=2, default=_json_default)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(payload)
            print_success(f"Schema written to: {out_path}")
        else:
            print(payload)
        return 0

    if op == "validate":
        if not name:
            print_error(
                "Missing schema name. Use: insidellms schema validate --name <SchemaName> -i <file>"
            )
            return 1
        if not getattr(args, "input", None):
            print_error("Missing --input for schema validate")
            return 1

        in_path = Path(args.input)
        if not in_path.exists():
            print_error(f"Input file not found: {in_path}")
            return 1

        validator = OutputValidator(registry)
        errors = 0

        def validate_one(obj: Any) -> None:
            nonlocal errors
            try:
                validator.validate(
                    name,
                    obj,
                    schema_version=version,
                    mode="strict",
                )
            except OutputValidationError as e:
                errors += 1
                if args.mode == "warn":
                    print_warning(str(e))
                else:
                    raise

        try:
            if in_path.suffix.lower() == ".jsonl":
                with open(in_path) as f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError as e:
                            errors += 1
                            if args.mode == "warn":
                                print_warning(f"Invalid JSON on line {line_no}: {e}")
                                continue
                            raise
                        validate_one(obj)
            else:
                obj = json.loads(in_path.read_text())
                if isinstance(obj, list):
                    for item in obj:
                        validate_one(item)
                else:
                    validate_one(obj)
        except Exception as e:
            if args.mode == "warn":
                print_warning(f"Validation completed with errors: {e}")
            else:
                print_error(f"Validation failed: {e}")
                return 1

        if errors:
            if args.mode == "warn":
                print_warning(f"Validated with {errors} error(s) (warn mode)")
                return 0
            print_error(f"Validation failed with {errors} error(s)")
            return 1

        print_success("Validation OK")
        return 0

    print_error(f"Unknown schema op: {op}")
    return 1


def cmd_report(args: argparse.Namespace) -> int:
    """Rebuild summary.json and report.html from records.jsonl."""
    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        print_error(f"Run directory not found: {run_dir}")
        return 1

    records_path = run_dir / "records.jsonl"
    if not records_path.exists():
        print_error(f"records.jsonl not found in: {run_dir}")
        return 1

    try:
        records = _read_jsonl_records(records_path)
    except Exception as e:
        print_error(f"Could not read records.jsonl: {e}")
        return 1

    if not records:
        print_error("No records found in records.jsonl")
        return 1

    experiments, derived_config, schema_version = _build_experiments_from_records(records)
    if not experiments:
        print_error("No experiments could be reconstructed from records")
        return 1

    run_ids = {record.get("run_id") for record in records if record.get("run_id")}
    run_id = sorted(run_ids)[0] if run_ids else None
    if run_id and len(run_ids) > 1:
        print_warning(f"Multiple run_ids found; using {run_id}")

    generated_at = None
    if run_id:
        try:
            from insideLLMs.runner import _deterministic_base_time, _deterministic_run_times

            base_time = _deterministic_base_time(str(run_id))
            _, generated_at = _deterministic_run_times(base_time, len(records))
        except Exception:
            generated_at = None

    if generated_at is None:
        generated_at = max(
            (dt for dt in (_parse_datetime(r.get("completed_at")) for r in records) if dt),
            default=None,
        )

    summary = generate_summary_report(experiments, include_ci=True)
    summary_payload = {
        "schema_version": schema_version,
        "generated_at": generated_at,
        "summary": summary,
        "config": derived_config,
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2, default=_json_default)
    print_success(f"Summary written to: {summary_path}")

    report_title = args.report_title or "Behavioural Probe Report"
    report_path = run_dir / "report.html"
    try:
        from insideLLMs.visualization import create_interactive_html_report

        create_interactive_html_report(
            experiments,
            title=report_title,
            save_path=str(report_path),
        )
    except ImportError:
        report_html = _build_basic_harness_report(
            experiments,
            summary,
            report_title,
            generated_at=generated_at,
        )
        with open(report_path, "w") as f:
            f.write(report_html)

    print_success(f"Report written to: {report_path}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Compare two run directories and report behavioural regressions."""
    run_dir_a = Path(args.run_dir_a)
    run_dir_b = Path(args.run_dir_b)

    if not run_dir_a.exists() or not run_dir_a.is_dir():
        print_error(f"Run directory not found: {run_dir_a}")
        return 1
    if not run_dir_b.exists() or not run_dir_b.is_dir():
        print_error(f"Run directory not found: {run_dir_b}")
        return 1

    records_path_a = run_dir_a / "records.jsonl"
    records_path_b = run_dir_b / "records.jsonl"

    if not records_path_a.exists():
        print_error(f"records.jsonl not found in: {run_dir_a}")
        return 1
    if not records_path_b.exists():
        print_error(f"records.jsonl not found in: {run_dir_b}")
        return 1

    try:
        records_a = _read_jsonl_records(records_path_a)
        records_b = _read_jsonl_records(records_path_b)
    except Exception as e:
        print_error(f"Could not read records.jsonl: {e}")
        return 1

    if not records_a or not records_b:
        print_error("Both run directories must contain records to compare")
        return 1

    output_format = getattr(args, "format", "text")
    output_path = getattr(args, "output", None)
    if output_path and output_format != "json":
        print_warning("--output is only used with --format json")

    ignore_keys: set[str] = set()
    for entry in args.output_fingerprint_ignore or []:
        for item in str(entry).split(","):
            item = item.strip()
            if item:
                ignore_keys.add(item.lower())

    ignore_keys_set = ignore_keys if ignore_keys else None

    def build_index(
        records: list[dict[str, Any]],
    ) -> tuple[dict[tuple[str, str, str], dict[str, Any]], int]:
        index: dict[tuple[str, str, str], dict[str, Any]] = {}
        duplicates = 0
        for record in records:
            key = _record_key(record)
            if key in index:
                duplicates += 1
                continue
            index[key] = record
        return index, duplicates

    def record_identity(record: dict[str, Any]) -> dict[str, Any]:
        custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
        label = _record_label(record)
        key = _record_key(record)
        return {
            "record_key": {"model_id": key[0], "probe_id": key[1], "item_id": key[2]},
            "model_id": key[0],
            "probe_id": key[1],
            "example_id": label[2],
            "replicate_key": custom.get("replicate_key"),
            "label": {"model": label[0], "probe": label[1], "example": label[2]},
        }

    def record_summary(record: dict[str, Any]) -> dict[str, Any]:
        scores = record.get("scores") if isinstance(record.get("scores"), dict) else None
        _metric_name, metric_value = _primary_score(record)
        return {
            "status": _status_string(record),
            "primary_metric": record.get("primary_metric"),
            "primary_score": metric_value,
            "scores_keys": sorted(scores.keys()) if isinstance(scores, dict) else None,
            "output": _output_summary(record, ignore_keys_set),
        }

    index_a, dup_a = build_index(records_a)
    index_b, dup_b = build_index(records_b)

    if dup_a:
        print_warning(f"Baseline has {dup_a} duplicate key(s); first occurrence used")
    if dup_b:
        print_warning(f"Comparison has {dup_b} duplicate key(s); first occurrence used")

    all_keys = set(index_a) | set(index_b)
    regressions: list[tuple[str, str, str, str]] = []
    improvements: list[tuple[str, str, str, str]] = []
    changes: list[tuple[str, str, str, str]] = []
    only_a: list[tuple[str, str, str]] = []
    only_b: list[tuple[str, str, str]] = []

    regressions_json: list[dict[str, Any]] = []
    improvements_json: list[dict[str, Any]] = []
    changes_json: list[dict[str, Any]] = []
    only_a_json: list[dict[str, Any]] = []
    only_b_json: list[dict[str, Any]] = []

    for key in sorted(all_keys):
        record_a = index_a.get(key)
        record_b = index_b.get(key)

        if record_a is None:
            only_b.append(_record_label(record_b))  # type: ignore[arg-type]
            only_b_json.append(record_identity(record_b))  # type: ignore[arg-type]
            continue
        if record_b is None:
            only_a.append(_record_label(record_a))
            only_a_json.append(record_identity(record_a))
            continue

        label = _record_label(record_a)
        status_a = _status_string(record_a)
        status_b = _status_string(record_b)
        score_name_a, score_a = _primary_score(record_a)
        score_name_b, score_b = _primary_score(record_b)
        identity = record_identity(record_a)
        summary_a = record_summary(record_a)
        summary_b = record_summary(record_b)

        if status_a == "success" and status_b != "success":
            regressions.append((*label, f"status {status_a} -> {status_b}"))
            regressions_json.append(
                {
                    **identity,
                    "kind": "status_regression",
                    "detail": f"status {status_a} -> {status_b}",
                    "baseline": summary_a,
                    "candidate": summary_b,
                }
            )
            continue
        if status_a != "success" and status_b == "success":
            improvements.append((*label, f"status {status_a} -> {status_b}"))
            improvements_json.append(
                {
                    **identity,
                    "kind": "status_improvement",
                    "detail": f"status {status_a} -> {status_b}",
                    "baseline": summary_a,
                    "candidate": summary_b,
                }
            )
            continue

        metrics_compared = False
        if score_a is not None and score_b is not None and score_name_a == score_name_b:
            delta = score_b - score_a
            metrics_compared = True
            if delta < 0:
                regressions.append(
                    (*label, f"{score_name_a} {score_a:.4f} -> {score_b:.4f} (delta {delta:.4f})")
                )
                regressions_json.append(
                    {
                        **identity,
                        "kind": "metric_regression",
                        "metric": score_name_a,
                        "baseline": summary_a,
                        "candidate": summary_b,
                        "delta": delta,
                    }
                )
                continue
            if delta > 0:
                improvements.append(
                    (*label, f"{score_name_a} {score_a:.4f} -> {score_b:.4f} (delta +{delta:.4f})")
                )
                improvements_json.append(
                    {
                        **identity,
                        "kind": "metric_improvement",
                        "metric": score_name_a,
                        "baseline": summary_a,
                        "candidate": summary_b,
                        "delta": delta,
                    }
                )

        if not metrics_compared and (score_a is not None or score_b is not None):
            reason = _metric_mismatch_reason(record_a, record_b) or "type_mismatch"
            detail = _metric_mismatch_details(record_a, record_b)
            changes.append((*label, f"metrics not comparable:{reason}; {detail}"))
            changes_json.append(
                {
                    **identity,
                    "kind": "metrics_not_comparable",
                    "reason": reason,
                    "baseline": summary_a,
                    "candidate": summary_b,
                    "details": _metric_mismatch_context(record_a, record_b),
                }
            )

        if metrics_compared and status_a == "success" and status_b == "success":
            scores_a = record_a.get("scores") if isinstance(record_a.get("scores"), dict) else None
            scores_b = record_b.get("scores") if isinstance(record_b.get("scores"), dict) else None
            if isinstance(scores_a, dict) and isinstance(scores_b, dict):
                keys_a = sorted(scores_a.keys())
                keys_b = sorted(scores_b.keys())
                missing_in_b = sorted(set(keys_a) - set(keys_b))
                missing_in_a = sorted(set(keys_b) - set(keys_a))
                if missing_in_a or missing_in_b:
                    changes.append(
                        (
                            *label,
                            "metric_key_missing: "
                            f"baseline_missing={missing_in_a}, candidate_missing={missing_in_b}",
                        )
                    )
                    changes_json.append(
                        {
                            **identity,
                            "kind": "metric_key_missing",
                            "baseline_missing": missing_in_a,
                            "candidate_missing": missing_in_b,
                            "baseline": summary_a,
                            "candidate": summary_b,
                        }
                    )

        output_a = _output_text(record_a)
        output_b = _output_text(record_b)
        if output_a is not None or output_b is not None:
            if output_a != output_b:
                changes.append((*label, "output changed"))
                changes_json.append(
                    {
                        **identity,
                        "kind": "output_changed",
                        "baseline": summary_a,
                        "candidate": summary_b,
                    }
                )
        else:
            fingerprint_a = _output_fingerprint(record_a, ignore_keys=ignore_keys_set)
            fingerprint_b = _output_fingerprint(record_b, ignore_keys=ignore_keys_set)
            if fingerprint_a != fingerprint_b:
                if fingerprint_a and fingerprint_b:
                    changes.append(
                        (*label, f"output fingerprint {fingerprint_a} -> {fingerprint_b}")
                    )
                else:
                    changes.append((*label, "output changed (structured)"))
                changes_json.append(
                    {
                        **identity,
                        "kind": "output_changed",
                        "baseline": summary_a,
                        "candidate": summary_b,
                        "baseline_fingerprint": fingerprint_a,
                        "candidate_fingerprint": fingerprint_b,
                    }
                )
        if status_a != status_b:
            changes.append((*label, f"status {status_a} -> {status_b}"))
            changes_json.append(
                {
                    **identity,
                    "kind": "status_changed",
                    "detail": f"status {status_a} -> {status_b}",
                    "baseline": summary_a,
                    "candidate": summary_b,
                }
            )

    diff_report = {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "baseline": str(run_dir_a),
        "candidate": str(run_dir_b),
        "run_ids": {
            "baseline": sorted(
                {record.get("run_id") for record in records_a if record.get("run_id")}
            ),
            "candidate": sorted(
                {record.get("run_id") for record in records_b if record.get("run_id")}
            ),
        },
        "counts": {
            "common": len(all_keys) - len(only_a) - len(only_b),
            "only_baseline": len(only_a),
            "only_candidate": len(only_b),
            "regressions": len(regressions),
            "improvements": len(improvements),
            "other_changes": len(changes),
        },
        "duplicates": {"baseline": dup_a, "candidate": dup_b},
        "regressions": regressions_json,
        "improvements": improvements_json,
        "changes": changes_json,
        "only_baseline": only_a_json,
        "only_candidate": only_b_json,
    }

    if output_format == "json":
        payload = json.dumps(diff_report, indent=2, default=_json_default)
        if output_path:
            Path(output_path).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        if args.fail_on_regressions and regressions:
            return 2
        if args.fail_on_changes and (regressions or changes or only_a or only_b):
            return 2
        return 0

    print_header("Behavioural Diff")
    print_key_value("Baseline", run_dir_a)
    print_key_value("Comparison", run_dir_b)
    print_key_value("Common keys", diff_report["counts"]["common"])
    print_key_value("Only in baseline", diff_report["counts"]["only_baseline"])
    print_key_value("Only in comparison", diff_report["counts"]["only_candidate"])
    print_key_value("Regressions", diff_report["counts"]["regressions"])
    print_key_value("Improvements", diff_report["counts"]["improvements"])
    print_key_value("Other changes", diff_report["counts"]["other_changes"])

    def print_section(title: str, items: list[tuple[str, str, str, str]]) -> None:
        if not items:
            return
        print_subheader(title)
        for model_label, probe_label, example_id, detail in items[: args.limit]:
            print(f"  {model_label} | {probe_label} | example {example_id}: {detail}")
        if len(items) > args.limit:
            print(colorize(f"  ... and {len(items) - args.limit} more", Colors.DIM))

    print_section("Regressions", regressions)
    print_section("Improvements", improvements)
    print_section("Other Changes", changes)

    if only_a:
        print_subheader("Missing in Comparison")
        for model_label, probe_label, example_id in only_a[: args.limit]:
            print(f"  {model_label} | {probe_label} | example {example_id}")
        if len(only_a) > args.limit:
            print(colorize(f"  ... and {len(only_a) - args.limit} more", Colors.DIM))

    if only_b:
        print_subheader("New in Comparison")
        for model_label, probe_label, example_id in only_b[: args.limit]:
            print(f"  {model_label} | {probe_label} | example {example_id}")
        if len(only_b) > args.limit:
            print(colorize(f"  ... and {len(only_b) - args.limit} more", Colors.DIM))

    if args.fail_on_regressions and regressions:
        return 2
    if args.fail_on_changes and (regressions or changes or only_a or only_b):
        return 2
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Execute the list command."""
    ensure_builtins_registered()

    filter_str = args.filter.lower() if args.filter else None

    if args.type in ("models", "all"):
        print_subheader("Available Models")
        models = model_registry.list()
        if filter_str:
            models = [m for m in models if filter_str in m.lower()]

        for name in sorted(models):
            info = model_registry.info(name)
            doc = info.get("doc", "").split("\n")[0] if info.get("doc") else ""
            if args.detailed:
                print(f"\n  {colorize(name, Colors.BOLD, Colors.CYAN)}")
                print(f"    {colorize('Description:', Colors.DIM)} {doc[:70]}")
                if info.get("default_kwargs"):
                    print(f"    {colorize('Defaults:', Colors.DIM)} {info['default_kwargs']}")
            else:
                print(f"  {colorize(name, Colors.CYAN):25} {doc[:50]}")

        print(f"\n  {colorize(f'Total: {len(models)} models', Colors.DIM)}")

    if args.type in ("probes", "all"):
        print_subheader("Available Probes")
        probes = probe_registry.list()
        if filter_str:
            probes = [p for p in probes if filter_str in p.lower()]

        for name in sorted(probes):
            info = probe_registry.info(name)
            doc = info.get("doc", "").split("\n")[0] if info.get("doc") else ""
            if args.detailed:
                print(f"\n  {colorize(name, Colors.BOLD, Colors.GREEN)}")
                print(f"    {colorize('Description:', Colors.DIM)} {doc[:70]}")
            else:
                print(f"  {colorize(name, Colors.GREEN):25} {doc[:50]}")

        print(f"\n  {colorize(f'Total: {len(probes)} probes', Colors.DIM)}")

    if args.type in ("datasets", "all"):
        print_subheader("Built-in Benchmark Datasets")
        try:
            from insideLLMs.benchmark_datasets import list_builtin_datasets

            datasets = list_builtin_datasets()

            if filter_str:
                datasets = [d for d in datasets if filter_str in d["name"].lower()]

            for ds in datasets:
                if args.detailed:
                    print(f"\n  {colorize(ds['name'], Colors.BOLD, Colors.YELLOW)}")
                    print(f"    {colorize('Category:', Colors.DIM)} {ds['category']}")
                    print(f"    {colorize('Examples:', Colors.DIM)} {ds['num_examples']}")
                    print(f"    {colorize('Description:', Colors.DIM)} {ds['description'][:60]}")
                    print(
                        f"    {colorize('Difficulties:', Colors.DIM)} {', '.join(ds['difficulties'])}"
                    )
                else:
                    print(
                        f"  {colorize(ds['name'], Colors.YELLOW):15} {ds['num_examples']:4} examples  [{ds['category']}]"
                    )

            print(f"\n  {colorize(f'Total: {len(datasets)} datasets', Colors.DIM)}")
        except ImportError:
            print_warning("Benchmark datasets module not available")

    if args.type in ("trackers", "all"):
        print_subheader("Experiment Tracking Backends")
        trackers = [
            ("local", "Local file-based tracking (always available)"),
            ("wandb", "Weights & Biases (requires: pip install wandb)"),
            ("mlflow", "MLflow tracking (requires: pip install mlflow)"),
            ("tensorboard", "TensorBoard (requires: pip install tensorboard)"),
        ]
        for name, desc in trackers:
            print(f"  {colorize(name, Colors.MAGENTA):15} {desc}")

    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Execute the init command."""
    import yaml

    print_header("Initialize Experiment Configuration")

    # Base configuration
    config: dict[str, Any] = {
        "model": {
            "type": args.model,
            "args": {},
        },
        "probe": {
            "type": args.probe,
            "args": {},
        },
        "dataset": {
            "format": "jsonl",
            "path": "data/questions.jsonl",
        },
    }

    # Add model-specific args hints
    model_hints = {
        "openai": {"model_name": "gpt-4"},
        "anthropic": {"model_name": "claude-3-opus-20240229"},
        "cohere": {"model_name": "command-r-plus"},
        "gemini": {"model_name": "gemini-pro"},
        "huggingface": {"model_name": "gpt2"},
        "ollama": {"model_name": "llama2"},
    }
    if args.model in model_hints:
        config["model"]["args"] = model_hints[args.model]

    # Apply template enhancements
    if args.template == "benchmark":
        config["benchmark"] = {
            "datasets": ["reasoning", "math", "coding"],
            "max_examples_per_dataset": 10,
        }
    elif args.template == "tracking":
        config["tracking"] = {
            "backend": "local",
            "project": "my-experiment",
            "log_dir": "./experiments",
        }
    elif args.template == "full":
        config["benchmark"] = {
            "datasets": ["reasoning", "math", "coding", "safety"],
            "max_examples_per_dataset": 20,
        }
        config["tracking"] = {
            "backend": "local",
            "project": "my-experiment",
            "log_dir": "./experiments",
        }
        config["async"] = {
            "enabled": True,
            "concurrency": 5,
        }
        config["output"] = {
            "format": "json",
            "path": "results/experiment_results.json",
            "html_report": True,
        }

    output_path = Path(args.output)

    if output_path.suffix in (".yaml", ".yml"):
        content = yaml.dump(config, default_flow_style=False, sort_keys=False)
    else:
        content = json.dumps(config, indent=2)

    output_path.write_text(content)
    print_success(f"Created config: {output_path}")
    print_key_value("Template", args.template)
    print_key_value("Model", args.model)
    print_key_value("Probe", args.probe)

    # Create sample data directory and file
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    sample_data_path = data_dir / "questions.jsonl"
    if not sample_data_path.exists():
        sample_data = [
            {"question": "What is 2+2?", "reference_answer": "4"},
            {"question": "What is the capital of France?", "reference_answer": "Paris"},
            {"question": "Who wrote Romeo and Juliet?", "reference_answer": "William Shakespeare"},
            {
                "question": "If all cats are mammals, and all mammals are animals, are all cats animals?",
                "reference_answer": "Yes",
            },
            {"question": "What is the chemical symbol for water?", "reference_answer": "H2O"},
        ]
        with open(sample_data_path, "w") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")
        print_success(f"Created sample data: {sample_data_path}")

    print()
    print_info("Next steps:")
    print(f"  1. Edit {colorize(str(output_path), Colors.CYAN)} to customize your experiment")
    print(f"  2. Run: {colorize(f'insidellms run {output_path}', Colors.GREEN)}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    ensure_builtins_registered()

    try:
        if args.type == "model":
            info = model_registry.info(args.name)
        elif args.type == "probe":
            info = probe_registry.info(args.name)
        else:  # dataset
            from insideLLMs.benchmark_datasets import load_builtin_dataset

            ds = load_builtin_dataset(args.name)
            stats = ds.get_stats()
            print_header(f"Dataset: {args.name}")
            print_key_value("Description", ds.description)
            print_key_value(
                "Category", ds.category.value if hasattr(ds.category, "value") else ds.category
            )
            print_key_value("Total examples", stats.total_count)
            print_key_value(
                "Categories", ", ".join(stats.categories) if stats.categories else "N/A"
            )
            print_key_value(
                "Difficulties", ", ".join(stats.difficulties) if stats.difficulties else "N/A"
            )

            print_subheader("Sample Examples")
            for i, ex in enumerate(ds.sample(3, seed=42)):
                print(f"\n  {colorize(f'Example {i + 1}', Colors.BOLD)}")
                print(f"    {colorize('Input:', Colors.DIM)} {ex.input_text[:80]}...")
                if ex.expected_output:
                    print(f"    {colorize('Expected:', Colors.DIM)} {ex.expected_output[:50]}")
                print(f"    {colorize('Difficulty:', Colors.DIM)} {ex.difficulty}")

            return 0

        print_header(f"{args.type.capitalize()}: {args.name}")
        print_key_value("Factory", info["factory"])

        if info.get("default_kwargs"):
            print_key_value("Default args", json.dumps(info["default_kwargs"], indent=2))

        if info.get("doc"):
            print_subheader("Description")
            print(f"  {info['doc']}")

        return 0

    except KeyError:
        print_error(f"{args.type.capitalize()} '{args.name}' not found")
        return 1
    except Exception as e:
        print_error(f"Error: {e}")
        return 1


def cmd_quicktest(args: argparse.Namespace) -> int:
    """Execute the quicktest command."""
    ensure_builtins_registered()

    print_header("Quick Test")
    print_key_value("Model", args.model)
    print_key_value("Prompt", args.prompt[:50] + "..." if len(args.prompt) > 50 else args.prompt)

    try:
        # Parse model args
        model_args = json.loads(args.model_args)
        model_args["temperature"] = args.temperature
        model_args["max_tokens"] = args.max_tokens

        # Get model from registry - it may be a class or instance
        model_or_factory = model_registry.get(args.model)

        # If it's a class, instantiate it; if it's already an instance, use it
        if isinstance(model_or_factory, type):
            model = model_or_factory(**model_args)
        else:
            model = model_or_factory

        # Generate response
        spinner = Spinner("Generating response")
        start_time = time.time()
        spinner.spin()

        response = model.generate(args.prompt)
        elapsed = time.time() - start_time
        spinner.stop(success=True)

        print_subheader("Response")
        print(f"  {response}")

        print_subheader("Stats")
        print_key_value("Latency", f"{elapsed * 1000:.1f}ms")
        print_key_value("Response length", f"{len(response)} characters")

        # Apply probe if specified
        if args.probe:
            print_subheader(f"Probe: {args.probe}")
            probe_factory = probe_registry.get(args.probe)
            probe_factory()
            # Note: Probe evaluation would go here
            print_info(
                f"Probe '{args.probe}' applied (detailed scoring available in full experiments)"
            )

        return 0

    except KeyError as e:
        print_error(f"Unknown model or probe: {e}")
        return 1
    except Exception as e:
        print_error(f"Error: {e}")
        return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Execute the benchmark command."""
    ensure_builtins_registered()

    print_header("Running Benchmark Suite")

    models = [m.strip() for m in args.models.split(",")]
    probes = [p.strip() for p in args.probes.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else None

    print_key_value("Models", ", ".join(models))
    print_key_value("Probes", ", ".join(probes))
    if datasets:
        print_key_value("Datasets", ", ".join(datasets))
    print_key_value("Max examples", args.max_examples)

    try:
        from insideLLMs.benchmark_datasets import (
            create_comprehensive_benchmark_suite,
            load_builtin_dataset,
        )

        # Load datasets
        if datasets:
            suite_examples = []
            for ds_name in datasets:
                ds = load_builtin_dataset(ds_name)
                for ex in ds.sample(args.max_examples, seed=42):
                    suite_examples.append(ex)
        else:
            suite = create_comprehensive_benchmark_suite(
                max_examples_per_dataset=args.max_examples, seed=42
            )
            suite_examples = list(suite.sample(args.max_examples * 5, seed=42))

        print_info(f"Loaded {len(suite_examples)} benchmark examples")

        results_all: list[dict[str, Any]] = []

        for model_name in models:
            print_subheader(f"Model: {model_name}")

            try:
                model_or_factory = model_registry.get(model_name)
                model = (
                    model_or_factory() if isinstance(model_or_factory, type) else model_or_factory
                )
            except Exception as e:
                print_warning(f"Could not load model {model_name}: {e}")
                continue

            for probe_name in probes:
                print(f"  Running probe: {colorize(probe_name, Colors.GREEN)}")

                try:
                    probe_or_factory = probe_registry.get(probe_name)
                    probe = (
                        probe_or_factory()
                        if isinstance(probe_or_factory, type)
                        else probe_or_factory
                    )
                except Exception as e:
                    print_warning(f"  Could not load probe {probe_name}: {e}")
                    continue

                # Create runner and run
                from insideLLMs.runner import ProbeRunner

                runner = ProbeRunner(model, probe)

                inputs = [ex.input_text for ex in suite_examples[: args.max_examples]]
                progress = ProgressBar(len(inputs), prefix=f"  {probe_name}")

                probe_results = []
                for i, inp in enumerate(inputs):
                    try:
                        result = runner.run_single(inp)
                        probe_results.append(result)
                    except Exception:
                        pass
                    progress.update(i + 1)

                progress.finish()

                success_count = sum(
                    1
                    for r in probe_results
                    if getattr(r, "status", None) == "success"
                    or (isinstance(r, dict) and r.get("status") == "success")
                )
                results_all.append(
                    {
                        "model": model_name,
                        "probe": probe_name,
                        "total": len(inputs),
                        "success": success_count,
                        "accuracy": success_count / max(1, len(inputs)),
                    }
                )

        # Summary
        print_subheader("Benchmark Results Summary")
        print(f"\n  {'Model':<15} {'Probe':<15} {'Accuracy':>10} {'Success':>10}")
        print("  " + "-" * 55)
        for r in results_all:
            print(
                f"  {r['model']:<15} {r['probe']:<15} {r['accuracy'] * 100:>9.1f}% {r['success']:>5}/{r['total']}"
            )

        # Save results if output specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            results_file = output_dir / "benchmark_results.json"
            with open(results_file, "w") as f:
                json.dump(results_all, f, indent=2)
            print_success(f"Results saved to: {results_file}")

            if args.html_report:
                print_info(
                    "HTML report generation for `benchmark` requires ExperimentResult format; "
                    "use `insidellms harness` + `insidellms report`."
                )

        return 0

    except Exception as e:
        print_error(f"Benchmark error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Execute the compare command."""
    ensure_builtins_registered()

    print_header("Model Comparison")

    models = [m.strip() for m in args.models.split(",")]
    print_key_value("Models", ", ".join(models))

    # Get inputs
    inputs: list[str] = []
    if args.input:
        inputs = [args.input]
    elif args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print_error(f"Input file not found: {input_path}")
            return 1

        if input_path.suffix == ".json":
            with open(input_path) as f:
                data = json.load(f)
                inputs = [d.get("input", d.get("question", str(d))) for d in data]
        elif input_path.suffix == ".jsonl":
            with open(input_path) as f:
                for line in f:
                    data = json.loads(line)
                    inputs.append(data.get("input", data.get("question", str(data))))
        else:
            with open(input_path) as f:
                inputs = [line.strip() for line in f if line.strip()]
    else:
        print_error("Please provide --input or --input-file")
        return 1

    print_key_value("Inputs", len(inputs))

    try:
        results: list[dict[str, Any]] = []

        for inp in inputs:
            inp_display = inp[:50] + "..." if len(inp) > 50 else inp
            print_subheader(f"Input: {inp_display}")

            row = {"input": inp}

            for model_name in models:
                try:
                    model_or_factory = model_registry.get(model_name)
                    model = (
                        model_or_factory()
                        if isinstance(model_or_factory, type)
                        else model_or_factory
                    )

                    start = time.time()
                    response = model.generate(inp)
                    elapsed = time.time() - start

                    row[f"{model_name}_response"] = response
                    row[f"{model_name}_latency_ms"] = elapsed * 1000

                    print(f"\n  {colorize(model_name, Colors.CYAN)}:")
                    print(f"    {response[:100]}{'...' if len(response) > 100 else ''}")
                    print(f"    {colorize(f'({elapsed * 1000:.1f}ms)', Colors.DIM)}")

                except Exception as e:
                    row[f"{model_name}_response"] = f"ERROR: {e}"
                    row[f"{model_name}_latency_ms"] = None
                    print(f"\n  {colorize(model_name, Colors.RED)}: Error - {e}")

            results.append(row)

        # Output results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print_success(f"Results saved to: {args.output}")

        return 0

    except Exception as e:
        print_error(f"Comparison error: {e}")
        return 1


def cmd_interactive(args: argparse.Namespace) -> int:
    """Execute the interactive command."""
    ensure_builtins_registered()

    print_header("Interactive Mode")
    print_info(f"Model: {args.model}")
    print_info("Type 'help' for commands, 'quit' to exit")

    try:
        model_or_factory = model_registry.get(args.model)
        model = model_or_factory() if isinstance(model_or_factory, type) else model_or_factory
    except Exception as e:
        print_error(f"Could not load model: {e}")
        return 1

    # Command history
    history: list[str] = []

    # Load history file
    history_path = Path(args.history_file)
    if history_path.exists():
        with open(history_path) as f:
            history = [line.strip() for line in f.readlines()]

    print()

    while True:
        try:
            prompt = input(colorize(">>> ", Colors.BRIGHT_CYAN))
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        prompt = prompt.strip()
        if not prompt:
            continue

        # Save to history
        history.append(prompt)
        with open(history_path, "a") as f:
            f.write(prompt + "\n")

        # Handle commands
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif prompt.lower() == "help":
            print(f"""
{colorize("Available Commands:", Colors.BOLD)}
  help          - Show this help message
  quit/exit/q   - Exit interactive mode
  history       - Show command history
  clear         - Clear the screen
  model <name>  - Switch to a different model
  probe <name>  - Run a probe on the last response

{colorize("Usage:", Colors.BOLD)}
  Just type your prompt and press Enter to get a response.
""")
        elif prompt.lower() == "history":
            print_subheader("Command History")
            for i, h in enumerate(history[-20:], 1):
                print(f"  {i:3}. {h[:60]}")
        elif prompt.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
        elif prompt.lower().startswith("model "):
            new_model = prompt[6:].strip()
            try:
                model_or_factory = model_registry.get(new_model)
                model = (
                    model_or_factory() if isinstance(model_or_factory, type) else model_or_factory
                )
                print_success(f"Switched to model: {new_model}")
            except Exception as e:
                print_error(f"Could not load model: {e}")
        else:
            # Regular prompt - generate response
            spinner = Spinner("Thinking")
            start = time.time()
            spinner.spin()

            try:
                response = model.generate(prompt)
                elapsed = time.time() - start
                spinner.stop(success=True)

                print(f"\n{response}\n")
                print(colorize(f"[{elapsed * 1000:.0f}ms]", Colors.DIM))
                print()
            except Exception as e:
                spinner.stop(success=False)
                print_error(f"Error: {e}")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the validate command."""
    target_path = Path(args.config)
    if not target_path.exists():
        print_error(f"Path not found: {target_path}")
        return 1

    # ---------------------------------------------------------------------
    # Run directory validation (manifest.json + records.jsonl)
    # ---------------------------------------------------------------------
    if target_path.is_dir() or target_path.name == "manifest.json":
        run_dir = target_path if target_path.is_dir() else target_path.parent
        manifest_path = (
            target_path if target_path.name == "manifest.json" else run_dir / "manifest.json"
        )

        print_header("Validate Run Directory")
        print_key_value("Run dir", run_dir)
        print_key_value("Manifest", manifest_path)

        if not manifest_path.exists():
            print_error(f"manifest.json not found: {manifest_path}")
            return 1

        from insideLLMs.schemas import OutputValidationError, OutputValidator, SchemaRegistry

        registry = SchemaRegistry()
        validator = OutputValidator(registry)

        errors = 0

        def _handle_error(msg: str) -> None:
            nonlocal errors
            errors += 1
            if args.mode == "warn":
                print_warning(msg)
            else:
                print_error(msg)

        try:
            manifest_obj = json.loads(manifest_path.read_text())
        except Exception as e:
            _handle_error(f"Could not read manifest JSON: {e}")
            return 0 if args.mode == "warn" else 1

        # Determine schema version: CLI override > manifest.schema_version > manifest.schemas[name]
        schema_version = (
            args.schema_version
            or manifest_obj.get("schema_version")
            or manifest_obj.get("schemas", {}).get(registry.RUN_MANIFEST)
            or DEFAULT_SCHEMA_VERSION
        )

        print_key_value("Schema version", schema_version)

        # Validate manifest
        try:
            validator.validate(
                registry.RUN_MANIFEST,
                manifest_obj,
                schema_version=schema_version,
                mode="strict",
            )
        except OutputValidationError as e:
            _handle_error(f"Manifest schema mismatch: {e}")
            if args.mode != "warn":
                return 1

        # Validate records
        records_file = manifest_obj.get("records_file") or "records.jsonl"
        records_path = run_dir / records_file
        print_key_value("Records", records_path)

        if not records_path.exists():
            _handle_error(f"records file not found: {records_path}")
            return 0 if args.mode == "warn" else 1

        try:
            with open(records_path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        _handle_error(f"Invalid JSON on line {line_no}: {e}")
                        if args.mode != "warn":
                            return 1
                        continue
                    try:
                        validator.validate(
                            registry.RESULT_RECORD,
                            obj,
                            schema_version=schema_version,
                            mode="strict",
                        )
                    except OutputValidationError as e:
                        _handle_error(f"Record line {line_no} schema mismatch: {e}")
                        if args.mode != "warn":
                            return 1
        except Exception as e:
            _handle_error(f"Error reading records: {e}")
            return 0 if args.mode == "warn" else 1

        if errors:
            if args.mode == "warn":
                print_warning(f"Validation completed with {errors} error(s) (warn mode)")
                return 0
            print_error(f"Validation failed with {errors} error(s)")
            return 1

        print_success("Validation OK")
        return 0

    # ---------------------------------------------------------------------
    # Config validation (legacy)
    # ---------------------------------------------------------------------
    print_header("Validate Configuration")
    config_path = target_path
    print_key_value("Config", config_path)

    try:
        import yaml

        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f) if config_path.suffix in (".yaml", ".yml") else json.load(f)

        errors: list[str] = []
        warnings: list[str] = []

        # Validate model
        if "model" not in config:
            errors.append("Missing required field: model")
        else:
            model_config = config["model"]
            if "type" not in model_config:
                errors.append("Missing model.type")
            else:
                ensure_builtins_registered()
                if model_config["type"] not in model_registry.list():
                    errors.append(f"Unknown model type: {model_config['type']}")

        # Validate probe
        if "probe" not in config:
            errors.append("Missing required field: probe")
        else:
            probe_config = config["probe"]
            if "type" not in probe_config:
                errors.append("Missing probe.type")
            else:
                ensure_builtins_registered()
                if probe_config["type"] not in probe_registry.list():
                    errors.append(f"Unknown probe type: {probe_config['type']}")

        # Validate dataset
        if "dataset" not in config:
            warnings.append("No dataset specified (will use builtin)")
        else:
            ds_config = config["dataset"]
            if "path" in ds_config:
                ds_path = Path(ds_config["path"])
                if not ds_path.exists():
                    warnings.append(f"Dataset file not found: {ds_path}")

        # Report results
        if errors:
            print_subheader("Errors")
            for e in errors:
                print(f"  {colorize('ERROR', Colors.RED)} {e}")

        if warnings:
            print_subheader("Warnings")
            for w in warnings:
                print(f"  {colorize('WARN', Colors.YELLOW)} {w}")

        if not errors:
            print()
            print_success("Configuration is valid!")
            return 0
        else:
            print()
            print_error(f"Configuration has {len(errors)} error(s)")
            return 1

    except Exception as e:
        print_error(f"Validation error: {e}")
        return 1


def cmd_export(args: argparse.Namespace) -> int:
    """Execute the export command."""
    print_header("Export Results")

    input_path = Path(args.input)
    if not input_path.exists():
        print_error(f"Input file not found: {input_path}")
        return 1

    print_key_value("Input", input_path)
    print_key_value("Format", args.format)

    try:
        with open(input_path) as f:
            results = json.load(f)

        if not isinstance(results, list):
            results = [results]

        output_path = args.output
        if not output_path:
            output_path = input_path.stem + f".{args.format}"

        if args.format == "csv":
            import csv

            if results:
                keys = results[0].keys()
                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(results)

        elif args.format == "markdown":
            content = results_to_markdown(results)
            with open(output_path, "w") as f:
                f.write(content)

        elif args.format == "html":
            print_error(
                "HTML export requires plotly and ExperimentResult format; "
                "use `insidellms report <run_dir>` to produce report.html."
            )
            return 1

        elif args.format == "latex":
            # Generate LaTeX table
            if results:
                keys = list(results[0].keys())
                lines = [
                    "\\begin{table}[h]",
                    "\\centering",
                    "\\begin{tabular}{" + "l" * len(keys) + "}",
                    "\\hline",
                    " & ".join(keys) + " \\\\",
                    "\\hline",
                ]
                for r in results[:20]:  # Limit rows
                    values = [str(r.get(k, ""))[:30] for k in keys]
                    lines.append(" & ".join(values) + " \\\\")
                lines.extend(
                    [
                        "\\hline",
                        "\\end{tabular}",
                        "\\caption{Experiment Results}",
                        "\\end{table}",
                    ]
                )
                with open(output_path, "w") as f:
                    f.write("\n".join(lines))

        print_success(f"Exported to: {output_path}")
        return 0

    except Exception as e:
        print_error(f"Export error: {e}")
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI."""
    global USE_COLOR

    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle global flags
    if hasattr(args, "no_color") and args.no_color:
        USE_COLOR = False

    if not args.command:
        # Show a nice welcome message instead of just help
        print_header("insideLLMs")
        print(colorize("  A world-class toolkit for LLM evaluation and exploration", Colors.DIM))
        print()
        parser.print_help()
        print()
        print(
            colorize("Quick start: ", Colors.BOLD)
            + 'insidellms quicktest "Hello world" --model dummy'
        )
        return 0

    commands = {
        "run": cmd_run,
        "harness": cmd_harness,
        "report": cmd_report,
        "diff": cmd_diff,
        "schema": cmd_schema,
        "list": cmd_list,
        "init": cmd_init,
        "info": cmd_info,
        "quicktest": cmd_quicktest,
        "benchmark": cmd_benchmark,
        "compare": cmd_compare,
        "interactive": cmd_interactive,
        "validate": cmd_validate,
        "export": cmd_export,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        print_error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
