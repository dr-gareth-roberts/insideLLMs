"""Report building utilities for the insideLLMs CLI."""

import html
from datetime import datetime
from typing import Any, Mapping, Optional

from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION
from insideLLMs.types import (
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
)

from ._output import _format_float, _format_percent
from ._record_utils import _parse_datetime, _probe_category_from_value, _status_from_record


def _record_has_harness(record: dict[str, Any]) -> bool:
    custom = record.get("custom")
    return isinstance(custom, dict) and isinstance(custom.get("harness"), dict)


def _probe_result_from_record(record: dict[str, Any]) -> ProbeResult:
    """Rebuild a ProbeResult from persisted evidence only (no rescoring)."""
    scores_raw = record.get("scores")
    scores: dict[str, float] = {}
    if isinstance(scores_raw, dict):
        for key, value in scores_raw.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                scores[str(key)] = float(value)
    primary_metric = record.get("primary_metric")
    if primary_metric is not None:
        primary_metric = str(primary_metric)
    return ProbeResult(
        input=record.get("input"),
        output=record.get("output"),
        status=_status_from_record(record.get("status")),
        error=record.get("error"),
        latency_ms=record.get("latency_ms"),
        metadata=record.get("custom") if isinstance(record.get("custom"), dict) else {},
        scores=scores,
        primary_metric=primary_metric,
    )


def _aggregate_score_from_records(records: list[dict[str, Any]]) -> Optional[ProbeScore]:
    """Derive aggregate ProbeScore from persisted per-record scores only."""
    if not records:
        return None

    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    latencies: list[float] = []
    error_count = 0

    for record in records:
        # Match Probe.score(): error_rate is ERROR/total only; timeouts are
        # excluded (ProbeScore has no timeout_rate field — do not invent one).
        status = str(record.get("status") or "").lower()
        if status == "error":
            error_count += 1
        latency = record.get("latency_ms")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            latencies.append(float(latency))
        scores = record.get("scores")
        if not isinstance(scores, dict):
            continue
        for key, value in scores.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                metric_key = str(key)
                metric_sums[metric_key] = metric_sums.get(metric_key, 0.0) + float(value)
                metric_counts[metric_key] = metric_counts.get(metric_key, 0) + 1

    custom_metrics: dict[str, float | str] = {
        key: metric_sums[key] / metric_counts[key]
        for key in metric_sums
        if metric_counts.get(key, 0) > 0
    }

    def _pick(name: str) -> Optional[float]:
        value = custom_metrics.get(name)
        return float(value) if isinstance(value, (int, float)) else None

    # Always emit a score for non-empty records so timeout-only runs still get
    # error_rate=0.0 (matching Probe.score), not a missing aggregate.
    mean_latency = sum(latencies) / len(latencies) if latencies else None
    return ProbeScore(
        accuracy=_pick("accuracy"),
        precision=_pick("precision"),
        recall=_pick("recall"),
        f1_score=_pick("f1_score") if "f1_score" in custom_metrics else _pick("f1"),
        mean_latency_ms=mean_latency,
        error_rate=error_count / len(records),
        custom_metrics=custom_metrics,
    )


def _build_experiments_from_records(
    records: list[dict[str, Any]],
) -> tuple[list[ExperimentResult], dict[str, Any], str]:
    if not records:
        return [], {"derived_from_records": True}, DEFAULT_SCHEMA_VERSION

    schema_version = records[0].get("schema_version") or DEFAULT_SCHEMA_VERSION

    harness_records = [r for r in records if _record_has_harness(r)]
    non_harness_records = [r for r in records if not _record_has_harness(r)]

    experiments: list[ExperimentResult] = []
    derived_config: dict[str, Any] = {"derived_from_records": True}

    # Never drop canonical records. Prefer harness grouping when present; keep
    # any non-harness records via the run_id path (mixed metadata is allowed).
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
            model_spec = first.get("model", {}) if isinstance(first.get("model"), dict) else {}

            model_name = harness.get("model_name") or model_spec.get("model_id") or "model"
            model_id = harness.get("model_id") or model_spec.get("model_id") or model_name
            provider = model_spec.get("provider") or harness.get("model_type") or "unknown"
            extra = model_spec.get("params") if isinstance(model_spec.get("params"), dict) else {}

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
                try:
                    return int(harness_item.get("example_index", 0))
                except (TypeError, ValueError):
                    return 0

            sorted_records = sorted(group_records, key=_sort_key)
            probe_results = [_probe_result_from_record(record) for record in sorted_records]

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
                    score=_aggregate_score_from_records(sorted_records),
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

    # Run-id path for non-harness records (all records when none have harness).
    records_for_run_groups = non_harness_records if harness_records else records
    if records_for_run_groups:
        run_groups: dict[str, list[dict[str, Any]]] = {}
        for record in records_for_run_groups:
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

            probe_results = [_probe_result_from_record(record) for record in group_records]

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
                    experiment_id=run_id,
                    model_info=model_info,
                    probe_name=str(probe_name),
                    probe_category=probe_category,
                    results=probe_results,
                    score=_aggregate_score_from_records(group_records),
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
    *,
    run_health: Mapping[str, object] | None = None,
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

    health_banner = ""
    if run_health is not None:
        completion = run_health.get("run_completed")
        if completion is not True:
            state = "Incomplete run" if completion is False else "Run completion unknown"
            details: list[str] = []
            abort = run_health.get("abort")
            if isinstance(abort, Mapping) and abort.get("message") is not None:
                details.append(str(abort["message"]))
            health = run_health.get("health")
            if isinstance(health, Mapping):
                reasons = health.get("reasons")
                if isinstance(reasons, list):
                    details.extend(str(reason) for reason in reasons)
            detail_html = " ".join(html.escape(detail) for detail in details)
            health_banner = (
                f'<div class="incomplete-banner"><strong>{html.escape(state)}</strong>'
                f"<div>{detail_html}</div></div>"
            )

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
    .incomplete-banner {{ background: #fff3cd; border: 2px solid #b45309;
      padding: 12px; margin: 16px 0; color: #713f12; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  {meta_line}
  {health_banner}

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
