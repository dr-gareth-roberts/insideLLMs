"""Report rebuilding preserves authoritative run-completion evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from insideLLMs.cli import main


def _write_record(run_dir: Path, *, schema_version: str = "1.0.2") -> None:
    record = {
        "schema_version": schema_version,
        "run_id": "report-test",
        "input": "one",
        "output": "answer",
        "status": "success",
        "model": {"model_id": "dummy", "provider": "offline", "params": {}},
        "probe": {"probe_id": "logic"},
    }
    (run_dir / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def _write_summary(run_dir: Path, summary: object) -> None:
    payload = {
        "schema_version": "1.0.2",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "summary": summary,
        "config": {"old": True},
    }
    (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest(run_dir: Path, *, completed: bool, custom: dict | None = None) -> None:
    manifest = {
        "schema_version": "1.0.2",
        "run_id": "report-test",
        "run_completed": completed,
        "record_count": 1,
        "custom": custom or {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_actual_cli_failfast_rebuild_preserves_abort_and_expected_count(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [{"type": "dummy"}, {"type": "unknown-second-model"}],
                "probes": [{"type": "logic"}],
                "dataset": {"format": "inline", "data": [{"question": "One?"}]},
                "runner": {"stop_on_error": True},
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    assert main(["harness", str(config_path), "--run-dir", str(run_dir), "--quiet"]) == 1
    before = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest_before = (run_dir / "manifest.json").read_bytes()

    report_exit = main(["report", str(run_dir), "--quiet"])

    after = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert report_exit == 0
    assert after["summary"]["run_completed"] is False
    assert after["summary"]["abort"] == before["summary"]["abort"]
    assert after["summary"]["health"]["expected_count"] == 2
    assert (run_dir / "manifest.json").read_bytes() == manifest_before
    assert "incomplete" in (run_dir / "report.html").read_text(encoding="utf-8").lower()


def test_completed_manifest_marks_rebuilt_report_complete(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_record(run_dir)
    _write_manifest(
        run_dir,
        completed=True,
        custom={"health": {"healthy": True, "expected_count": 1, "record_count": 1}},
    )

    assert main(["report", str(run_dir), "--quiet"]) == 0

    summary = json.loads((run_dir / "summary.json").read_text())["summary"]
    assert summary["run_completed"] is True
    assert "<strong>incomplete run</strong>" not in (run_dir / "report.html").read_text().lower()


def test_secondary_diagnostics_and_escaped_abort_are_preserved(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_record(run_dir)
    diagnostics = [{"stage": "secondary", "message": "scorer <failed> & stopped"}]
    _write_summary(run_dir, {"run_completed": False, "secondary_diagnostics": diagnostics})
    _write_manifest(
        run_dir,
        completed=False,
        custom={
            "abort": {"code": "execution_error", "message": "provider <script>alert(1)</script>"},
            "health": {"healthy": False, "expected_count": 2, "record_count": 1},
        },
    )

    assert main(["report", str(run_dir), "--quiet"]) == 0

    payload = json.loads((run_dir / "summary.json").read_text())
    assert payload["summary"]["secondary_diagnostics"] == diagnostics
    html = (run_dir / "report.html").read_text()
    assert "provider &lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "provider <script>alert(1)</script>" not in html


def test_manifest_incomplete_wins_over_conflicting_old_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_record(run_dir)
    _write_summary(run_dir, {"run_completed": True, "health": {"healthy": True}})
    _write_manifest(
        run_dir,
        completed=False,
        custom={"health": {"healthy": False, "expected_count": 2, "record_count": 1}},
    )

    assert main(["report", str(run_dir), "--quiet"]) == 0

    summary = json.loads((run_dir / "summary.json").read_text())["summary"]
    assert summary["run_completed"] is False
    assert any("conflict" in warning.lower() for warning in summary["metadata_warnings"])


def test_partial_manifest_metadata_merges_summary_only_diagnostic_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_record(run_dir, schema_version="1.0.0")
    _write_summary(
        run_dir,
        {
            "run_completed": False,
            "abort": {
                "code": "old-code",
                "message": "old message",
                "secondary_diagnostics": [{"stage": "cleanup", "message": "disk warning"}],
            },
            "health": {"healthy": True, "expected_count": 2, "record_count": 1},
        },
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "custom": {
                    "abort": {"message": "stopped"},
                    "health": {"healthy": False},
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(["report", str(run_dir), "--quiet"]) == 0

    summary = json.loads((run_dir / "summary.json").read_text())["summary"]
    assert summary["run_completed"] is False
    assert summary["abort"] == {
        "code": "old-code",
        "message": "stopped",
        "secondary_diagnostics": [{"stage": "cleanup", "message": "disk warning"}],
    }
    assert summary["health"] == {"healthy": False, "expected_count": 2, "record_count": 1}
    assert any("health.healthy" in warning for warning in summary["metadata_warnings"])


def test_legacy_manifest_without_completion_preserves_summary_completion(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_record(run_dir, schema_version="1.0.0")
    _write_summary(run_dir, {"run_completed": True})
    (run_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0.0", "custom": {}}), encoding="utf-8"
    )

    assert main(["report", str(run_dir), "--quiet"]) == 0
    assert json.loads((run_dir / "summary.json").read_text())["summary"]["run_completed"] is True


def test_summary_only_abort_is_explicit_incomplete_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_record(run_dir)
    _write_summary(run_dir, {"abort": {"message": "summary-only stop"}})

    assert main(["report", str(run_dir), "--quiet"]) == 0

    summary = json.loads((run_dir / "summary.json").read_text())["summary"]
    assert summary["run_completed"] is False
    assert summary["abort"] == {"message": "summary-only stop"}


def test_missing_manifest_preserves_diagnostics_or_labels_completion_unknown(
    tmp_path: Path,
) -> None:
    with_summary = tmp_path / "with-summary"
    with_summary.mkdir()
    _write_record(with_summary)
    _write_summary(with_summary, {"run_completed": False, "abort": {"message": "old abort"}})
    assert main(["report", str(with_summary), "--quiet"]) == 0
    rebuilt = json.loads((with_summary / "summary.json").read_text())["summary"]
    assert rebuilt["run_completed"] is False
    assert rebuilt["abort"] == {"message": "old abort"}

    unknown = tmp_path / "unknown"
    unknown.mkdir()
    _write_record(unknown)
    assert main(["report", str(unknown), "--quiet"]) == 0
    rebuilt = json.loads((unknown / "summary.json").read_text())["summary"]
    assert rebuilt["run_completed"] is None
    assert "unknown" in (unknown / "report.html").read_text().lower()


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        (
            "manifest.json",
            {"custom": {"abort": {"secondary_diagnostics": ["invalid"]}}},
        ),
        (
            "summary.json",
            {"summary": {"abort": {"secondary_diagnostics": "invalid"}}},
        ),
        ("manifest.json", {"run_completed": "yes", "custom": {}}),
        (
            "manifest.json",
            {
                "run_completed": False,
                "custom": {"health": {"healthy": False, "expected_count": "two"}},
            },
        ),
        (
            "summary.json",
            {"schema_version": "1.0.2", "generated_at": "bad", "summary": [], "config": {}},
        ),
        (
            "summary.json",
            {
                "schema_version": "1.0.2",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "summary": {"run_completed": 0},
                "config": {},
            },
        ),
        (
            "summary.json",
            {
                "schema_version": "1.0.2",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "summary": {"metadata_warnings": "conflict"},
                "config": {},
            },
        ),
        (
            "summary.json",
            {
                "schema_version": "1.0.2",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "summary": {"secondary_diagnostics": ["not-an-object"]},
                "config": {},
            },
        ),
    ],
)
def test_malformed_authoritative_metadata_refuses_without_replacing_outputs(
    tmp_path: Path, filename: str, payload: object
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_record(run_dir)
    old_summary = json.dumps(
        {
            "schema_version": "1.0.2",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "summary": {"run_completed": False},
            "config": {},
        }
    ).encode()
    old_report = b"<html>old report</html>"
    (run_dir / "summary.json").write_bytes(old_summary)
    (run_dir / "report.html").write_bytes(old_report)
    (run_dir / filename).write_text(json.dumps(payload), encoding="utf-8")
    summary_before = (run_dir / "summary.json").read_bytes()

    assert main(["report", str(run_dir), "--quiet"]) == 1
    assert (run_dir / "summary.json").read_bytes() == summary_before
    assert (run_dir / "report.html").read_bytes() == old_report


def test_sealed_run_refusal_preserves_report_bytes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_record(run_dir)
    old_summary = b'{"old":"summary"}'
    old_report = b"<html>old report</html>"
    (run_dir / "summary.json").write_bytes(old_summary)
    (run_dir / "report.html").write_bytes(old_report)
    (run_dir / "integrity").mkdir()
    (run_dir / "integrity" / "bundle_id.txt").write_text("sealed", encoding="utf-8")

    assert main(["report", str(run_dir), "--quiet"]) == 1
    assert (run_dir / "summary.json").read_bytes() == old_summary
    assert (run_dir / "report.html").read_bytes() == old_report


@pytest.mark.parametrize("overlap", [False, True])
def test_rebuild_merges_diagnostic_arrays_without_loss_or_accumulation(tmp_path, overlap):
    _write_record(tmp_path)
    shared = {"stage": "scoring", "message": "original detail", "error_type": "ValueError"}
    summary_only = {"stage": "summary-write", "message": "disk detail"}
    manifest_only = {"stage": "scoring", "message": "different detail", "error_type": "OSError"}
    _write_summary(
        tmp_path,
        {"abort": {"message": "old message", "secondary_diagnostics": [shared, summary_only]}},
    )
    manifest_diagnostics = [shared, manifest_only] if overlap else [manifest_only]
    _write_manifest(
        tmp_path,
        completed=False,
        custom={
            "abort": {"message": "authoritative", "secondary_diagnostics": manifest_diagnostics}
        },
    )
    expected = manifest_diagnostics + ([summary_only] if overlap else [shared, summary_only])
    first_summary = None
    for _ in range(3):
        assert main(["report", str(tmp_path), "--quiet"]) == 0
        summary = json.loads((tmp_path / "summary.json").read_text())["summary"]
        assert summary["abort"]["secondary_diagnostics"] == expected
        assert summary["abort"]["message"] == "authoritative"
        if first_summary is None:
            first_summary = summary
        assert summary == first_summary


@pytest.mark.parametrize("failed_allocation", [1, 2])
def test_stage_allocation_failure_preserves_outputs_and_cleans_owned_stage(
    tmp_path, monkeypatch, capsys, failed_allocation
):
    from insideLLMs.cli.commands import report

    tmp_path = tmp_path / "run"
    tmp_path.mkdir()
    _write_record(tmp_path)
    _write_summary(tmp_path, {"run_completed": False})
    (tmp_path / "report.html").write_bytes(b"<html>old</html>")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    original = report._owned_stage_path
    allocations = 0

    def allocate(directory, suffix):
        nonlocal allocations
        allocations += 1
        if allocations == failed_allocation:
            raise OSError("allocation failed")
        return original(directory, suffix)

    monkeypatch.setattr(report, "_owned_stage_path", allocate)
    assert main(["report", str(tmp_path), "--quiet"]) == 1
    output = capsys.readouterr()
    assert "Could not rebuild report: allocation failed" in output.out + output.err
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
