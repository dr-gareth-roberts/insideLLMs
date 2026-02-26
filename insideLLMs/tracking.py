"""Time-series run tracking for longitudinal regression detection.

Maintains a lightweight JSONL index of past runs so that ``insidellms trend``
can show score trajectories, detect threshold violations, and answer
"when did this metric start declining?" across an arbitrary number of runs.

The run index lives alongside run directories as ``run_index.jsonl``.
Each line records one completed run:

    {"run_id": "...", "timestamp": "...", "config": "...",
     "record_count": 12, "success_count": 10, "error_count": 2,
     "metrics": {"accuracy": 0.83, "error_rate": 0.17}}

Functions
---------
index_run : Append a completed run to the index.
load_index : Read all entries from a run index file.
compute_trends : Derive per-metric trajectories from indexed runs.
check_thresholds : Flag the first run where a metric crossed a threshold.
format_sparkline : Render a text sparkline for a metric series.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# -- index entry helpers ---------------------------------------------------


def _extract_metrics_from_records(records: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate key metrics from a list of result records."""
    total = len(records)
    if total == 0:
        return {}
    success = sum(1 for r in records if str(r.get("status", "")).lower() == "success")
    errors = sum(1 for r in records if str(r.get("status", "")).lower() in ("error", "timeout"))
    metrics: dict[str, float] = {
        "accuracy": success / total,
        "error_rate": errors / total,
    }

    # Average primary scores across records
    scores_sum: dict[str, float] = {}
    scores_count: dict[str, int] = {}
    for r in records:
        rec_scores = r.get("scores")
        if isinstance(rec_scores, dict):
            for k, v in rec_scores.items():
                if isinstance(v, (int, float)):
                    scores_sum[k] = scores_sum.get(k, 0.0) + float(v)
                    scores_count[k] = scores_count.get(k, 0) + 1
    for k in scores_sum:
        metrics[k] = scores_sum[k] / scores_count[k]

    return metrics


def _extract_metrics_from_manifest(manifest: dict[str, Any]) -> dict[str, float]:
    """Extract basic counts from a manifest dict and return metrics."""
    rc = manifest.get("record_count", 0)
    sc = manifest.get("success_count", 0)
    ec = manifest.get("error_count", 0)
    if rc == 0:
        return {}
    return {
        "accuracy": sc / rc,
        "error_rate": ec / rc,
    }


def _build_entry(
    run_id: str,
    timestamp: str,
    config: str,
    record_count: int,
    success_count: int,
    error_count: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "config": config,
        "record_count": record_count,
        "success_count": success_count,
        "error_count": error_count,
        "metrics": metrics,
    }


# -- public API ------------------------------------------------------------


def index_run(
    index_path: Path,
    run_dir: Path,
    *,
    config_label: str = "",
) -> dict[str, Any]:
    """Append a completed run to the index.

    Reads ``manifest.json`` and ``records.jsonl`` from *run_dir*, computes
    aggregate metrics, and appends one JSONL line to *index_path*.

    Returns the entry dict that was written.
    """
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "records.jsonl"

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    run_id = manifest.get("run_id", run_dir.name)
    timestamp = manifest.get("created_at") or datetime.utcnow().isoformat()

    records: list[dict[str, Any]] = []
    if records_path.exists():
        with open(records_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if records:
        metrics = _extract_metrics_from_records(records)
        record_count = len(records)
        success_count = sum(1 for r in records if str(r.get("status", "")).lower() == "success")
        error_count = sum(
            1 for r in records if str(r.get("status", "")).lower() in ("error", "timeout")
        )
    else:
        metrics = _extract_metrics_from_manifest(manifest)
        record_count = manifest.get("record_count", 0)
        success_count = manifest.get("success_count", 0)
        error_count = manifest.get("error_count", 0)

    entry = _build_entry(
        run_id=run_id,
        timestamp=str(timestamp),
        config=config_label or manifest.get("custom", {}).get("harness", {}).get("dataset", ""),
        record_count=record_count,
        success_count=success_count,
        error_count=error_count,
        metrics=metrics,
    )

    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")

    return entry


def load_index(index_path: Path) -> list[dict[str, Any]]:
    """Read all entries from *index_path*, ordered oldest-first."""
    entries: list[dict[str, Any]] = []
    if not index_path.exists():
        return entries
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_trends(
    entries: list[dict[str, Any]],
    metric: str = "accuracy",
) -> list[dict[str, Any]]:
    """Return per-run trend data for *metric*.

    Each item in the returned list has ``run_id``, ``timestamp``, ``value``,
    and ``delta`` (change from previous run, or ``None`` for the first).
    """
    trend: list[dict[str, Any]] = []
    prev: Optional[float] = None
    for entry in entries:
        value = entry.get("metrics", {}).get(metric)
        if value is None:
            continue
        delta = round(value - prev, 6) if prev is not None else None
        trend.append(
            {
                "run_id": entry["run_id"],
                "timestamp": entry.get("timestamp", ""),
                "value": value,
                "delta": delta,
            }
        )
        prev = value
    return trend


def check_thresholds(
    entries: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Check metrics against thresholds and return violations.

    *thresholds* maps metric names to minimum acceptable values, e.g.
    ``{"accuracy": 0.85}``.  Returns a list of dicts describing each
    violation: the run where the metric first dropped below threshold.
    """
    violations: list[dict[str, Any]] = []
    for metric_name, min_value in thresholds.items():
        was_above = True
        for entry in entries:
            value = entry.get("metrics", {}).get(metric_name)
            if value is None:
                continue
            if value < min_value and was_above:
                violations.append(
                    {
                        "metric": metric_name,
                        "threshold": min_value,
                        "value": value,
                        "run_id": entry["run_id"],
                        "timestamp": entry.get("timestamp", ""),
                    }
                )
                was_above = False
            elif value >= min_value:
                was_above = True
    return violations


_SPARK_CHARS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def format_sparkline(values: list[float]) -> str:
    """Render a text sparkline for a numeric series."""
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span == 0:
        return _SPARK_CHARS[4] * len(values)
    chars = []
    for v in values:
        idx = int((v - lo) / span * 7)
        idx = max(0, min(7, idx))
        chars.append(_SPARK_CHARS[idx + 1])
    return "".join(chars)
