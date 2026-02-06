"""Report building utilities for the insideLLMs CLI."""

import html
from datetime import datetime
from typing import Any, Optional

from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION
from insideLLMs.types import (
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
)

from ._output import _format_float, _format_percent
from ._record_utils import _parse_datetime, _probe_category_from_value, _status_from_record


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

    run_groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        run_id = record.get("run_id") or "run"
        run_groups.setdefault(str(run_id), []).append(record)

    for run_id, group_records in run_groups.items():
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
