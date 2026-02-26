"""Tests for insideLLMs/tracking.py — time-series run tracking."""

import json
from pathlib import Path

import pytest

from insideLLMs.tracking import (
    _build_entry,
    _extract_metrics_from_manifest,
    _extract_metrics_from_records,
    check_thresholds,
    compute_trends,
    format_sparkline,
    index_run,
    load_index,
)

# ---------------------------------------------------------------------------
# _extract_metrics helpers
# ---------------------------------------------------------------------------


class TestExtractMetricsFromRecords:
    def test_empty(self):
        assert _extract_metrics_from_records([]) == {}

    def test_all_success(self):
        records = [{"status": "success"}, {"status": "success"}]
        m = _extract_metrics_from_records(records)
        assert m["accuracy"] == 1.0
        assert m["error_rate"] == 0.0

    def test_mixed_status(self):
        records = [
            {"status": "success"},
            {"status": "error"},
            {"status": "success"},
            {"status": "timeout"},
        ]
        m = _extract_metrics_from_records(records)
        assert m["accuracy"] == 0.5
        assert m["error_rate"] == 0.5

    def test_aggregates_scores(self):
        records = [
            {"status": "success", "scores": {"f1": 0.8, "bleu": 0.6}},
            {"status": "success", "scores": {"f1": 0.9, "bleu": 0.4}},
        ]
        m = _extract_metrics_from_records(records)
        assert m["f1"] == pytest.approx(0.85)
        assert m["bleu"] == pytest.approx(0.5)


class TestExtractMetricsFromManifest:
    def test_empty(self):
        assert _extract_metrics_from_manifest({}) == {}
        assert _extract_metrics_from_manifest({"record_count": 0}) == {}

    def test_basic(self):
        m = _extract_metrics_from_manifest(
            {"record_count": 10, "success_count": 8, "error_count": 2}
        )
        assert m["accuracy"] == 0.8
        assert m["error_rate"] == 0.2


# ---------------------------------------------------------------------------
# _build_entry
# ---------------------------------------------------------------------------


class TestBuildEntry:
    def test_basic(self):
        e = _build_entry(
            run_id="r1",
            timestamp="2025-01-01T00:00:00",
            config="test.yaml",
            record_count=5,
            success_count=4,
            error_count=1,
            metrics={"accuracy": 0.8},
        )
        assert e["run_id"] == "r1"
        assert e["metrics"]["accuracy"] == 0.8
        assert e["record_count"] == 5


# ---------------------------------------------------------------------------
# index_run / load_index
# ---------------------------------------------------------------------------


class TestIndexRun:
    def test_index_from_records(self, tmp_path):
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        manifest = {"run_id": "run1", "created_at": "2025-01-01T00:00:00"}
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        records = [
            {"status": "success", "scores": {"f1": 0.9}},
            {"status": "success", "scores": {"f1": 0.8}},
            {"status": "error"},
        ]
        with open(run_dir / "records.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        index_path = tmp_path / "index.jsonl"
        entry = index_run(index_path, run_dir, config_label="test")
        assert entry["run_id"] == "run1"
        assert entry["record_count"] == 3
        assert entry["success_count"] == 2
        assert entry["error_count"] == 1
        assert entry["metrics"]["accuracy"] == pytest.approx(2 / 3)
        assert entry["metrics"]["f1"] == pytest.approx(0.85)

        # Verify it was written to the index file
        entries = load_index(index_path)
        assert len(entries) == 1
        assert entries[0]["run_id"] == "run1"

    def test_index_from_manifest_only(self, tmp_path):
        run_dir = tmp_path / "run2"
        run_dir.mkdir()
        manifest = {
            "run_id": "run2",
            "created_at": "2025-02-01T00:00:00",
            "record_count": 10,
            "success_count": 9,
            "error_count": 1,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        index_path = tmp_path / "index.jsonl"
        entry = index_run(index_path, run_dir)
        assert entry["run_id"] == "run2"
        assert entry["metrics"]["accuracy"] == 0.9

    def test_index_appends(self, tmp_path):
        index_path = tmp_path / "index.jsonl"
        for i in range(3):
            run_dir = tmp_path / f"run_{i}"
            run_dir.mkdir()
            manifest = {
                "run_id": f"run_{i}",
                "created_at": f"2025-01-0{i + 1}T00:00:00",
                "record_count": 10,
                "success_count": 10 - i,
                "error_count": i,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest))
            index_run(index_path, run_dir)

        entries = load_index(index_path)
        assert len(entries) == 3


class TestLoadIndex:
    def test_missing_file(self, tmp_path):
        assert load_index(tmp_path / "nope.jsonl") == []

    def test_empty_file(self, tmp_path):
        (tmp_path / "index.jsonl").write_text("")
        assert load_index(tmp_path / "index.jsonl") == []


# ---------------------------------------------------------------------------
# compute_trends
# ---------------------------------------------------------------------------


class TestComputeTrends:
    def _entries(self, values):
        return [
            {
                "run_id": f"r{i}",
                "timestamp": f"2025-01-0{i + 1}T00:00:00",
                "metrics": {"accuracy": v},
            }
            for i, v in enumerate(values)
        ]

    def test_basic(self):
        entries = self._entries([0.80, 0.85, 0.82])
        trend = compute_trends(entries, "accuracy")
        assert len(trend) == 3
        assert trend[0]["delta"] is None
        assert trend[1]["delta"] == pytest.approx(0.05)
        assert trend[2]["delta"] == pytest.approx(-0.03)

    def test_empty(self):
        assert compute_trends([], "accuracy") == []

    def test_missing_metric(self):
        entries = [{"run_id": "r0", "metrics": {"other": 1.0}}]
        assert compute_trends(entries, "accuracy") == []

    def test_single_entry(self):
        entries = self._entries([0.90])
        trend = compute_trends(entries, "accuracy")
        assert len(trend) == 1
        assert trend[0]["delta"] is None


# ---------------------------------------------------------------------------
# check_thresholds
# ---------------------------------------------------------------------------


class TestCheckThresholds:
    def _entries(self, values):
        return [
            {
                "run_id": f"r{i}",
                "timestamp": f"2025-01-0{i + 1}T00:00:00",
                "metrics": {"accuracy": v},
            }
            for i, v in enumerate(values)
        ]

    def test_no_violation(self):
        entries = self._entries([0.90, 0.91, 0.92])
        assert check_thresholds(entries, {"accuracy": 0.85}) == []

    def test_single_violation(self):
        entries = self._entries([0.90, 0.80, 0.75])
        violations = check_thresholds(entries, {"accuracy": 0.85})
        assert len(violations) == 1
        assert violations[0]["run_id"] == "r1"
        assert violations[0]["value"] == 0.80

    def test_recovery_and_second_violation(self):
        entries = self._entries([0.90, 0.80, 0.90, 0.80])
        violations = check_thresholds(entries, {"accuracy": 0.85})
        assert len(violations) == 2
        assert violations[0]["run_id"] == "r1"
        assert violations[1]["run_id"] == "r3"

    def test_empty(self):
        assert check_thresholds([], {"accuracy": 0.85}) == []


# ---------------------------------------------------------------------------
# format_sparkline
# ---------------------------------------------------------------------------


class TestFormatSparkline:
    def test_empty(self):
        assert format_sparkline([]) == ""

    def test_constant(self):
        result = format_sparkline([0.5, 0.5, 0.5])
        assert len(result) == 3

    def test_rising(self):
        result = format_sparkline([0.0, 0.5, 1.0])
        assert len(result) == 3
        assert result[0] < result[-1]  # first char < last char (unicode order)

    def test_single(self):
        result = format_sparkline([0.5])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# cmd_trend (CLI integration)
# ---------------------------------------------------------------------------


class TestCmdTrend:
    def _build_index(self, tmp_path, values):
        index_path = tmp_path / "index.jsonl"
        entries = []
        for i, v in enumerate(values):
            entries.append(
                {
                    "run_id": f"run_{i}",
                    "timestamp": f"2025-01-0{i + 1}T00:00:00",
                    "config": "test",
                    "record_count": 10,
                    "success_count": int(v * 10),
                    "error_count": 10 - int(v * 10),
                    "metrics": {"accuracy": v, "error_rate": 1 - v},
                }
            )
        with open(index_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        return index_path

    def test_trend_text_output(self, tmp_path):
        from insideLLMs.cli import main

        idx = self._build_index(tmp_path, [0.90, 0.85, 0.88])
        rc = main(["trend", "--index", str(idx)])
        assert rc == 0

    def test_trend_json_output(self, tmp_path, capsys):
        from insideLLMs.cli import main

        idx = self._build_index(tmp_path, [0.90, 0.85, 0.88])
        rc = main(["trend", "--index", str(idx), "--format", "json"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["metric"] == "accuracy"
        assert len(out["trend"]) == 3

    def test_trend_last_n(self, tmp_path, capsys):
        from insideLLMs.cli import main

        idx = self._build_index(tmp_path, [0.90, 0.85, 0.88, 0.92])
        rc = main(["trend", "--index", str(idx), "--last", "2", "--format", "json"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["runs"] == 2

    def test_trend_threshold_pass(self, tmp_path):
        from insideLLMs.cli import main

        idx = self._build_index(tmp_path, [0.90, 0.91])
        rc = main(["trend", "--index", str(idx), "--threshold", "0.85"])
        assert rc == 0

    def test_trend_threshold_fail(self, tmp_path):
        from insideLLMs.cli import main

        idx = self._build_index(tmp_path, [0.90, 0.80])
        rc = main(
            [
                "trend",
                "--index",
                str(idx),
                "--threshold",
                "0.85",
                "--fail-on-threshold",
            ]
        )
        assert rc == 2

    def test_trend_add_run(self, tmp_path):
        from insideLLMs.cli import main

        run_dir = tmp_path / "myrun"
        run_dir.mkdir()
        manifest = {
            "run_id": "myrun",
            "created_at": "2025-01-01T00:00:00",
            "record_count": 4,
            "success_count": 3,
            "error_count": 1,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        idx = tmp_path / "index.jsonl"

        rc = main(["trend", "--index", str(idx), "--add", str(run_dir)])
        assert rc == 0
        entries = load_index(idx)
        assert len(entries) == 1

    def test_trend_empty_index(self, tmp_path):
        from insideLLMs.cli import main

        idx = tmp_path / "empty.jsonl"
        rc = main(["trend", "--index", str(idx)])
        assert rc == 1

    def test_trend_custom_metric(self, tmp_path, capsys):
        from insideLLMs.cli import main

        idx = self._build_index(tmp_path, [0.90, 0.85])
        rc = main(
            [
                "trend",
                "--index",
                str(idx),
                "--metric",
                "error_rate",
                "--format",
                "json",
            ]
        )
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["metric"] == "error_rate"
        assert len(out["trend"]) == 2
