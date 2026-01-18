import json
from pathlib import Path

import yaml

from insideLLMs.results import results_to_markdown
from insideLLMs.runner import run_experiment_from_config
from insideLLMs.visualization import create_html_report


def test_run_experiment_from_config_uses_relative_dataset(tmp_path):
    dataset_path = tmp_path / "prompts.jsonl"
    # Store a JSON string so the probe receives a plain string prompt
    dataset_path.write_text(
        json.dumps("If A then B. A is true; what about B?") + "\n", encoding="utf-8"
    )

    config = {
        "model": {"type": "dummy", "args": {"name": "DummyModel"}},
        "probe": {"type": "logic", "args": {}},
        "dataset": {"format": "jsonl", "path": "prompts.jsonl"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    results = run_experiment_from_config(str(config_path))

    assert len(results) == 1
    assert "Solve this logic problem" in results[0]["output"]
    # Confirm relative dataset path was respected
    assert results[0]["input"] == "If A then B. A is true; what about B?"


def test_results_to_markdown_escapes_problematic_cells():
    sample = [
        {"input": "Hello|World", "output": "Line1\nLine2", "error": ""},
    ]
    md = results_to_markdown(sample)

    assert "Hello\\|World" in md
    assert "Line1<br>Line2" in md


def test_create_html_report_returns_path_and_writes_file(tmp_path):
    output_path = tmp_path / "report.html"
    results = [{"input": "sample", "output": "ok"}]

    returned_path = create_html_report(results, title="Test Report", save_path=output_path)

    assert Path(returned_path) == output_path
    contents = output_path.read_text(encoding="utf-8")
    assert "Test Report" in contents
    assert "sample" in contents
