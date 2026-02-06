"""Tests for insideLLMs.cli.commands.compare to increase coverage."""

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from insideLLMs.cli.commands.compare import cmd_compare


def _make_args(**kwargs):
    defaults = {
        "models": "dummy",
        "input": "Hello world",
        "input_file": None,
        "format": "table",
        "output": None,
        "verbose": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestCompareBasic:
    def test_no_models(self, capsys):
        rc = cmd_compare(_make_args(models=""))
        assert rc == 1

    def test_table_output(self, capsys):
        rc = cmd_compare(_make_args(models="dummy", input="test"))
        assert rc == 0

    def test_json_output_to_file(self, capsys, tmp_path):
        out = str(tmp_path / "compare.json")
        rc = cmd_compare(_make_args(models="dummy", input="test", format="json", output=out))
        assert rc == 0
        assert Path(out).exists()
        data = json.loads(Path(out).read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["input"] == "test"

    def test_markdown_output(self, capsys):
        rc = cmd_compare(_make_args(models="dummy", input="test", format="markdown"))
        captured = capsys.readouterr()
        assert rc == 0
        assert "|" in captured.out

    def test_markdown_output_to_file(self, capsys, tmp_path):
        out = str(tmp_path / "compare.md")
        rc = cmd_compare(
            _make_args(models="dummy", input="test", format="markdown", output=out)
        )
        assert rc == 0
        assert Path(out).exists()

    def test_table_output_to_file(self, capsys, tmp_path):
        out = str(tmp_path / "compare.json")
        rc = cmd_compare(_make_args(models="dummy", input="test", format="table", output=out))
        assert rc == 0
        assert Path(out).exists()

    def test_no_input(self, capsys):
        rc = cmd_compare(_make_args(input=None, input_file=None))
        assert rc == 1


class TestCompareInputFile:
    def test_json_list_input(self, capsys, tmp_path):
        input_file = tmp_path / "inputs.json"
        input_file.write_text(json.dumps(["Hello", "World"]))
        out = str(tmp_path / "out.json")
        rc = cmd_compare(
            _make_args(input=None, input_file=str(input_file), models="dummy", format="json", output=out)
        )
        assert rc == 0
        data = json.loads(Path(out).read_text())
        assert len(data) == 2

    def test_json_dict_input(self, capsys, tmp_path):
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps({"input": "Hello there"}))
        rc = cmd_compare(
            _make_args(input=None, input_file=str(input_file), models="dummy", format="table")
        )
        assert rc == 0

    def test_json_scalar_input(self, capsys, tmp_path):
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps("Just a string"))
        rc = cmd_compare(
            _make_args(input=None, input_file=str(input_file), models="dummy", format="table")
        )
        assert rc == 0

    def test_jsonl_input(self, capsys, tmp_path):
        input_file = tmp_path / "inputs.jsonl"
        input_file.write_text('{"input": "Hello"}\n{"input": "World"}\n')
        rc = cmd_compare(
            _make_args(input=None, input_file=str(input_file), models="dummy", format="table")
        )
        assert rc == 0

    def test_text_input(self, capsys, tmp_path):
        input_file = tmp_path / "inputs.txt"
        input_file.write_text("Hello\nWorld\n")
        rc = cmd_compare(
            _make_args(input=None, input_file=str(input_file), models="dummy", format="table")
        )
        assert rc == 0

    def test_missing_input_file(self, capsys):
        rc = cmd_compare(_make_args(input=None, input_file="/nonexistent/file.txt"))
        assert rc == 1

    def test_jsonl_invalid_json(self, capsys, tmp_path):
        input_file = tmp_path / "bad.jsonl"
        input_file.write_text("not json\n")
        rc = cmd_compare(_make_args(input=None, input_file=str(input_file), models="dummy"))
        assert rc == 1


class TestCompareModelErrors:
    def test_model_load_failure(self, capsys, tmp_path):
        out = str(tmp_path / "out.json")
        rc = cmd_compare(_make_args(models="nonexistent_model_xyz", input="test", format="json", output=out))
        assert rc == 0
        data = json.loads(Path(out).read_text())
        assert data[0]["models"][0]["error"] == "model_init_failed"

    def test_multiple_models_with_dummy(self, capsys, tmp_path):
        out = str(tmp_path / "out.json")
        rc = cmd_compare(
            _make_args(models="dummy,dummy", input="test", format="json", output=out)
        )
        assert rc == 0
        data = json.loads(Path(out).read_text())
        assert len(data[0]["models"]) == 2
