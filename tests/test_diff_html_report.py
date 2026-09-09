"""Tests for the deterministic HTML diff renderer and the ``insidellms diff --html`` flag."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from insideLLMs.cli import main
from insideLLMs.cli._record_utils import _read_jsonl_records
from insideLLMs.runtime import diff_html_report
from insideLLMs.runtime.diff_html_report import render_diff_html
from insideLLMs.runtime.diffing import build_diff_computation, judge_diff_report

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "diff"

# Imports the renderer may use; anything else (analysis/, contrib/, matplotlib) is a boundary leak.
_ALLOWED_IMPORTS = {"__future__", "difflib", "html", "json", "typing"}


def _record(
    *,
    model_id: str = "m1",
    probe_id: str = "p1",
    example_id: str = "e1",
    status: str = "success",
    score: float = 0.9,
    output_text: str | None = None,
    custom: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "schema_version": "1.0.1",
        "run_id": "run-1",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:01+00:00",
        "model": {"model_id": model_id, "provider": "local", "params": {}},
        "probe": {"probe_id": probe_id, "probe_version": "1.0.0", "params": {}},
        "example_id": example_id,
        "status": status,
        "primary_metric": "score",
        "scores": {"score": score},
        "usage": {},
        "custom": custom if custom is not None else {},
    }
    if output_text is not None:
        rec["output_text"] = output_text
    return rec


def _report(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    return build_diff_computation(
        records_baseline=baseline,
        records_candidate=candidate,
        baseline_label="runs/baseline",
        candidate_label="runs/candidate",
    ).diff_report


@pytest.fixture(scope="module")
def example_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two real DummyModel runs of the offline diff example (Paris vs Lyon)."""
    root = tmp_path_factory.mktemp("html-runs")
    baseline = root / "baseline"
    candidate = root / "candidate"
    for config, run_dir in (("baseline.yaml", baseline), ("candidate.yaml", candidate)):
        rc = main(
            ["run", str(EXAMPLE_ROOT / config), "--format", "summary", "--run-dir", str(run_dir)]
        )
        assert rc == 0
    return baseline, candidate


@pytest.fixture(scope="module")
def example_report(example_runs: tuple[Path, Path]) -> dict[str, Any]:
    baseline, candidate = example_runs
    return build_diff_computation(
        records_baseline=_read_jsonl_records(baseline / "records.jsonl"),
        records_candidate=_read_jsonl_records(candidate / "records.jsonl"),
        baseline_label=str(baseline),
        candidate_label=str(candidate),
    ).diff_report


def test_renders_real_dummy_model_diff(example_report: dict[str, Any]) -> None:
    rendered = render_diff_html(example_report)

    assert rendered.startswith("<!DOCTYPE html>")
    assert "<title>Behavioural Diff Report</title>" in rendered
    assert "<h2>Regressions (1)</h2>" in rendered
    assert "<h2>Other Changes (1)</h2>" in rendered
    assert "metric_regression" in rendered
    assert "accuracy 1.0000 -&gt; 0.0000 (delta -1.0000)" in rendered
    assert "<del>Paris</del> <ins>Lyon</ins> is the capital of France." in rendered
    assert str(example_report["baseline"]) in rendered
    assert "Schema version" in rendered


def test_render_is_byte_identical_across_renders_and_input_order(
    example_report: dict[str, Any],
) -> None:
    first = render_diff_html(example_report)
    second = render_diff_html(example_report)
    round_tripped = render_diff_html(json.loads(json.dumps(example_report)))

    assert first == second
    assert first == round_tripped

    baseline = [_record(example_id=f"e{i}", output_text="a", score=0.9) for i in range(3)]
    candidate = [_record(example_id=f"e{i}", output_text="b", score=0.1) for i in range(3)]
    report = _report(baseline, candidate)
    shuffled = json.loads(json.dumps(report))
    for key in ("regressions", "changes"):
        shuffled[key] = list(reversed(shuffled[key]))
    shuffled = dict(reversed(list(shuffled.items())))

    assert render_diff_html(shuffled) == render_diff_html(report)


def test_escapes_script_in_outputs_and_markup_in_labels() -> None:
    report = _report(
        [_record(model_id="<b>m</b>", output_text="hello world")],
        [_record(model_id="<b>m</b>", output_text='<script>alert("x")</script> world')],
    )

    rendered = render_diff_html(report)

    assert "<script" not in rendered
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in rendered
    assert "<b>m</b>" not in rendered
    assert "&lt;b&gt;m&lt;/b&gt;" in rendered


def test_empty_and_absent_sections_render_without_error() -> None:
    identical = [_record(output_text="same")]
    no_diff = render_diff_html(_report(identical, identical))
    assert "No differences detected" in no_diff
    assert "<h2>" not in no_diff

    minimal = {"schema_version": "1.0.2", "baseline": "a", "candidate": "b"}
    rendered = render_diff_html(minimal)
    assert "<title>Behavioural Diff Report</title>" in rendered
    assert "No differences detected" in rendered

    with_empty_lists = dict(minimal, regressions=[], changes=[], only_baseline=[], counts={})
    assert render_diff_html(with_empty_lists) == render_diff_html(minimal)


def test_renders_trace_trajectory_and_one_sided_sections() -> None:
    baseline = [
        _record(
            example_id="shared",
            output_text="same",
            custom={
                "trace_fingerprint": "sha256:aaaa",
                "trace_violations": [],
                "tool_calls": [{"tool_name": "search", "arguments": {"q": "x"}}],
            },
        ),
        _record(example_id="only-a"),
    ]
    candidate = [
        _record(
            example_id="shared",
            output_text="same",
            custom={
                "trace_fingerprint": "sha256:bbbb",
                "trace_violations": [{"rule": "no-<pii>"}],
                "tool_calls": [{"tool_name": "search", "arguments": {"q": "y"}}],
            },
        ),
        _record(example_id="only-b"),
    ]

    rendered = render_diff_html(_report(baseline, candidate))

    assert "<h2>Trace Drifts (1)</h2>" in rendered
    assert "trace sha256:aaaa -&gt; sha256:bbbb" in rendered
    assert "<h2>Trace Violation Increases (1)</h2>" in rendered
    assert "violations 0 -&gt; 1" in rendered
    assert "no-&lt;pii&gt;" in rendered
    assert "<h2>Trajectory Drifts (1)</h2>" in rendered
    assert "Baseline trajectory" in rendered and "Candidate trajectory" in rendered
    assert "<h2>Missing in Candidate (1)</h2>" in rendered
    assert "<h2>New in Candidate (1)</h2>" in rendered
    assert "only-a" in rendered and "only-b" in rendered


def test_renders_judge_section_when_attached() -> None:
    report = _report([_record(output_text="hello")], [_record(output_text="world")])
    report["judge"] = judge_diff_report(report, policy="balanced").judge_report

    rendered = render_diff_html(report)

    assert "<h2>Judge Verdict</h2>" in rendered
    assert "Policy <strong>balanced</strong>" in rendered
    assert 'class="review">review</td>' in rendered
    assert "<h2>Judge Verdict</h2>" not in render_diff_html(_report([_record()], [_record()]))


def test_renderer_imports_only_stdlib_modules() -> None:
    tree = ast.parse(Path(diff_html_report.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])

    assert imported <= _ALLOWED_IMPORTS, sorted(imported - _ALLOWED_IMPORTS)


def test_cli_html_flag_writes_file_and_keeps_exit_code(
    example_runs: tuple[Path, Path], tmp_path: Path
) -> None:
    baseline, candidate = example_runs
    html_path = tmp_path / "diff.html"

    assert main(["diff", str(baseline), str(candidate)]) == 0
    assert main(["diff", str(baseline), str(candidate), "--html", str(html_path)]) == 0
    rendered = html_path.read_text(encoding="utf-8")
    assert rendered.startswith("<!DOCTYPE html>")
    assert "<del>Paris</del> <ins>Lyon</ins>" in rendered

    gated = ["diff", str(baseline), str(candidate), "--fail-on-regressions"]
    gated_html = tmp_path / "gated.html"
    assert main(gated) == 2
    assert main([*gated, "--html", str(gated_html)]) == 2
    assert gated_html.read_text(encoding="utf-8") == rendered

    json_path = tmp_path / "diff.json"
    json_html = tmp_path / "json-mode.html"
    rc = main(
        [
            "diff",
            str(baseline),
            str(candidate),
            "--format",
            "json",
            "--output",
            str(json_path),
            "--html",
            str(json_html),
        ]
    )
    assert rc == 0
    assert json.loads(json_path.read_text(encoding="utf-8"))["counts"]["regressions"] == 1
    assert json_html.read_text(encoding="utf-8") == rendered


def test_cli_html_includes_judge_section(example_runs: tuple[Path, Path], tmp_path: Path) -> None:
    baseline, candidate = example_runs
    html_path = tmp_path / "judged.html"

    rc = main(["diff", str(baseline), str(candidate), "--judge", "--html", str(html_path)])

    assert rc == 0
    assert "<h2>Judge Verdict</h2>" in html_path.read_text(encoding="utf-8")


def test_cli_html_unwritable_path_is_a_setup_error(
    example_runs: tuple[Path, Path], tmp_path: Path
) -> None:
    baseline, candidate = example_runs
    missing_dir = tmp_path / "does-not-exist" / "diff.html"

    rc = main(["diff", str(baseline), str(candidate), "--html", str(missing_dir)])

    assert rc == 1
    assert not missing_dir.exists()
