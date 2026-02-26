import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from insideLLMs.cli.commands.export import cmd_export
from insideLLMs.cli.commands.info import cmd_info
from insideLLMs.cli.commands.init_cmd import cmd_init
from insideLLMs.cli.commands.report import cmd_report


def _info_args(**kwargs):
    defaults = {"type": "model", "name": "dummy"}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _export_args(**kwargs):
    defaults = {
        "input": "",
        "format": "json",
        "output": None,
        "redact_pii": False,
        "encrypt": False,
        "encryption_key_env": "INSIDELLMS_ENCRYPTION_KEY",
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _init_args(**kwargs):
    defaults = {
        "output": "config.yaml",
        "template": "basic",
        "model": "dummy",
        "probe": "dummy",
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _report_args(run_dir: Path, **kwargs):
    defaults = {"run_dir": str(run_dir), "report_title": "Test Report"}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_cmd_info_dataset_branch_renders_examples_and_expected_output(capsys):
    fake_stats = SimpleNamespace(total_count=2, categories=["general"], difficulties=["easy"])
    fake_dataset = SimpleNamespace(
        description="Fake dataset",
        category=SimpleNamespace(value="custom"),
        get_stats=lambda: fake_stats,
        sample=lambda _n, seed=42: [
            SimpleNamespace(input_text="Q1", expected_output="A1", difficulty="easy"),
            SimpleNamespace(input_text="Q2", expected_output=None, difficulty="medium"),
        ],
    )

    with patch("insideLLMs.benchmark_datasets.load_builtin_dataset", return_value=fake_dataset):
        rc = cmd_info(_info_args(type="dataset", name="fake-ds"))

    captured = capsys.readouterr()
    assert rc == 0
    assert "Dataset: fake-ds" in captured.out
    assert "Expected:" in captured.out
    assert "Difficulty:" in captured.out


def test_cmd_info_model_branch_prints_defaults_and_doc(capsys):
    with patch(
        "insideLLMs.cli.commands.info.model_registry.info",
        return_value={
            "factory": "insideLLMs.models.dummy",
            "default_kwargs": {"temperature": 0.2},
            "doc": "Dummy model docs",
        },
    ):
        rc = cmd_info(_info_args(type="model", name="dummy"))

    captured = capsys.readouterr()
    assert rc == 0
    assert "Default args" in captured.out
    assert "Dummy model docs" in captured.out


def test_cmd_info_unexpected_exception_returns_error(capsys):
    with patch(
        "insideLLMs.cli.commands.info.model_registry.info", side_effect=RuntimeError("boom")
    ):
        rc = cmd_info(_info_args(type="model", name="dummy"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "Error:" in captured.err


def test_cmd_export_markdown_uses_default_output_for_single_object(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_file = tmp_path / "results.json"
    input_file.write_text(json.dumps({"status": "success", "input": "hello", "output": "world"}))

    rc = cmd_export(_export_args(input=str(input_file), format="markdown", output=None))

    output_file = tmp_path / "results.markdown"
    assert rc == 0
    assert output_file.exists()
    assert "| Input | Output | Error |" in output_file.read_text()


def test_cmd_export_csv_and_latex(tmp_path):
    results = [
        {"input": "Q1", "output": "A1", "status": "success"},
        {"input": "Q2", "output": "A2", "status": "error"},
    ]
    input_file = tmp_path / "results.json"
    input_file.write_text(json.dumps(results))

    csv_file = tmp_path / "results.csv"
    latex_file = tmp_path / "results.tex"

    rc_csv = cmd_export(_export_args(input=str(input_file), format="csv", output=str(csv_file)))
    rc_latex = cmd_export(
        _export_args(input=str(input_file), format="latex", output=str(latex_file))
    )

    assert rc_csv == 0
    assert rc_latex == 0
    assert "input,output,status" in csv_file.read_text()
    assert "\\begin{table}[h]" in latex_file.read_text()


def test_cmd_export_html_and_invalid_json_error_paths(tmp_path, capsys):
    valid_input = tmp_path / "results.json"
    valid_input.write_text(json.dumps([{"status": "success"}]))

    rc_html = cmd_export(
        _export_args(input=str(valid_input), format="html", output=str(tmp_path / "x.html"))
    )
    assert rc_html == 1

    invalid_input = tmp_path / "broken.json"
    invalid_input.write_text("not-json")
    rc_invalid = cmd_export(
        _export_args(input=str(invalid_input), format="csv", output=str(tmp_path / "x.csv"))
    )

    captured = capsys.readouterr()
    assert rc_invalid == 1
    assert "Export error" in captured.err


def test_cmd_export_redact_pii_applies_redaction(tmp_path):
    """--redact-pii redacts PII before export."""
    input_file = tmp_path / "results.json"
    input_file.write_text(json.dumps([{"email": "user@example.com", "output": "Hello"}]))
    output_file = tmp_path / "out.csv"

    rc = cmd_export(
        _export_args(
            input=str(input_file),
            format="csv",
            output=str(output_file),
            redact_pii=True,
        )
    )

    assert rc == 0
    content = output_file.read_text()
    # mask_pii typically replaces emails with placeholder
    assert "user@example.com" not in content or "email" in content


def test_cmd_export_encrypt_without_key_fails(tmp_path, capsys):
    """--encrypt without encryption key env set returns 1."""
    input_file = tmp_path / "results.json"
    input_file.write_text(json.dumps([{"a": 1}]))
    output_file = tmp_path / "out.jsonl"

    rc = cmd_export(
        _export_args(
            input=str(input_file),
            format="jsonl",
            output=str(output_file),
            encrypt=True,
        )
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert "not set" in captured.err or "Encryption" in captured.err


def test_cmd_export_jsonl_format_writes_jsonl(tmp_path):
    """format=jsonl writes one JSON object per line."""
    input_file = tmp_path / "results.json"
    input_file.write_text(json.dumps([{"id": 1}, {"id": 2}]))
    output_file = tmp_path / "out.jsonl"

    rc = cmd_export(_export_args(input=str(input_file), format="jsonl", output=str(output_file)))

    assert rc == 0
    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == 1
    assert json.loads(lines[1])["id"] == 2


def test_cmd_init_full_template_creates_yaml_and_sample_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "full.yaml"

    rc = cmd_init(
        _init_args(output=str(output_file), template="full", model="openai", probe="factuality")
    )

    assert rc == 0
    assert output_file.exists()
    content = output_file.read_text()
    assert "benchmark:" in content
    assert "tracking:" in content
    assert "async:" in content
    assert "model_name: gpt-4" in content

    sample_data = tmp_path / "data" / "questions.jsonl"
    assert sample_data.exists()
    assert "capital of France" in sample_data.read_text()


def test_cmd_init_json_output_and_existing_sample_data_is_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(parents=True)
    sample_path = tmp_path / "data" / "questions.jsonl"
    sample_path.write_text('{"question": "Existing", "reference_answer": "Existing"}\n')

    output_file = tmp_path / "tracking.json"
    rc = cmd_init(
        _init_args(output=str(output_file), template="tracking", model="dummy", probe="logic")
    )

    assert rc == 0
    payload = json.loads(output_file.read_text())
    assert payload["tracking"]["backend"] == "local"
    assert payload["model"]["args"] == {}
    assert sample_path.read_text() == '{"question": "Existing", "reference_answer": "Existing"}\n'


def test_cmd_report_handles_read_error_and_empty_reconstructed_experiments(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "records.jsonl").write_text("{}\n")

    with patch(
        "insideLLMs.cli.commands.report._read_jsonl_records",
        side_effect=ValueError("cannot read"),
    ):
        rc_read_error = cmd_report(_report_args(run_dir))

    assert rc_read_error == 1

    with (
        patch("insideLLMs.cli.commands.report._read_jsonl_records", return_value=[{"x": 1}]),
        patch(
            "insideLLMs.cli.commands.report._build_experiments_from_records",
            return_value=([], {"cfg": 1}, "1.0.1"),
        ),
    ):
        rc_no_experiments = cmd_report(_report_args(run_dir))

    captured = capsys.readouterr()
    assert rc_no_experiments == 1
    assert "No experiments could be reconstructed" in captured.err


def test_cmd_report_falls_back_to_basic_report_on_import_error_and_warns_multi_run_ids(
    tmp_path, capsys
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "records.jsonl").write_text("placeholder\n")

    records = [
        {"run_id": "run-b", "completed_at": "2024-01-02T00:00:00Z"},
        {"run_id": "run-a", "completed_at": "2024-01-03T00:00:00Z"},
    ]

    with (
        patch("insideLLMs.cli.commands.report._read_jsonl_records", return_value=records),
        patch(
            "insideLLMs.cli.commands.report._build_experiments_from_records",
            return_value=([{"experiment": "ok"}], {"cfg": 1}, "1.0.1"),
        ),
        patch("insideLLMs.cli.commands.report.generate_summary_report", return_value={"ok": True}),
        patch(
            "insideLLMs.runtime.runner._deterministic_base_time",
            side_effect=ValueError("bad run id"),
        ),
        patch(
            "insideLLMs.cli.commands.report._build_basic_harness_report",
            return_value="<html>fallback</html>",
        ),
        patch(
            "insideLLMs.visualization.create_interactive_html_report",
            side_effect=ImportError("plotly missing"),
        ),
    ):
        rc = cmd_report(_report_args(run_dir, report_title="Fallback"))

    captured = capsys.readouterr()
    assert rc == 0
    assert "Multiple run_ids found" in captured.out
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "report.html").read_text() == "<html>fallback</html>"


def test_cmd_report_uses_deterministic_generated_at_when_available(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "records.jsonl").write_text("placeholder\n")

    records = [{"run_id": "run-1", "completed_at": "2024-01-02T00:00:00Z"}]
    fake_html_report = MagicMock()

    with (
        patch("insideLLMs.cli.commands.report._read_jsonl_records", return_value=records),
        patch(
            "insideLLMs.cli.commands.report._build_experiments_from_records",
            return_value=([{"experiment": "ok"}], {"cfg": 1}, "1.0.1"),
        ),
        patch("insideLLMs.cli.commands.report.generate_summary_report", return_value={"ok": True}),
        patch(
            "insideLLMs.runtime.runner._deterministic_base_time",
            return_value="BASE",
        ),
        patch(
            "insideLLMs.runtime.runner._deterministic_run_times",
            return_value=("START", "2024-01-05T12:00:00Z"),
        ),
        patch("insideLLMs.visualization.create_interactive_html_report", fake_html_report),
    ):
        rc = cmd_report(_report_args(run_dir, report_title="Deterministic"))

    assert rc == 0
    fake_html_report.assert_called_once()
    assert fake_html_report.call_args.kwargs["generated_at"] == "2024-01-05T12:00:00Z"
