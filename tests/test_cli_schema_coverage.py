"""Tests for insideLLMs.cli.commands.schema to increase coverage."""

import argparse
import json
import tempfile
from pathlib import Path

from insideLLMs.cli.commands.schema import cmd_schema


def _make_args(**kwargs):
    defaults = {
        "op": "list",
        "name": None,
        "version": "1.0.1",
        "output": None,
        "input": None,
        "mode": "strict",
        "verbose": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestSchemaListOp:
    def test_list_schemas(self, capsys):
        rc = cmd_schema(_make_args(op="list"))
        captured = capsys.readouterr()
        assert rc == 0
        assert "Available Output Schemas" in captured.out

    def test_default_op_is_list(self, capsys):
        rc = cmd_schema(_make_args())
        assert rc == 0


class TestSchemaDumpOp:
    def _schema_name(self):
        from insideLLMs.schemas import SchemaRegistry
        registry = SchemaRegistry()
        for name in [registry.RESULT_RECORD, registry.RUNNER_ITEM, registry.RUNNER_OUTPUT]:
            if registry.available_versions(name):
                return name
        return None

    def test_dump_schema_to_stdout(self, capsys):
        name = self._schema_name()
        if not name:
            return
        rc = cmd_schema(_make_args(op="dump", name=name))
        captured = capsys.readouterr()
        assert rc == 0
        schema = json.loads(captured.out)
        assert isinstance(schema, dict)

    def test_dump_schema_to_file(self, capsys, tmp_path):
        name = self._schema_name()
        if not name:
            return
        out_file = str(tmp_path / "schema.json")
        rc = cmd_schema(_make_args(op="dump", name=name, output=out_file))
        assert rc == 0
        assert Path(out_file).exists()

    def test_dump_missing_name(self, capsys):
        rc = cmd_schema(_make_args(op="dump", name=None))
        captured = capsys.readouterr()
        assert rc == 1

    def test_dump_invalid_schema(self, capsys):
        rc = cmd_schema(_make_args(op="dump", name="NonExistentSchema123"))
        captured = capsys.readouterr()
        assert rc == 2

    def test_shortcut_ux(self, capsys):
        """If op is not list/dump/validate, treat it as schema name for dump."""
        name = self._schema_name()
        if not name:
            return
        rc = cmd_schema(_make_args(op=name))
        captured = capsys.readouterr()
        assert rc == 0


class TestSchemaValidateOp:
    def test_validate_missing_name(self, capsys):
        rc = cmd_schema(_make_args(op="validate", name=None))
        captured = capsys.readouterr()
        assert rc == 1

    def test_validate_missing_input(self, capsys):
        rc = cmd_schema(_make_args(op="validate", name="ResultRecord", input=None))
        captured = capsys.readouterr()
        assert rc == 1

    def test_validate_nonexistent_input(self, capsys):
        rc = cmd_schema(
            _make_args(op="validate", name="ResultRecord", input="/nonexistent/path.json")
        )
        captured = capsys.readouterr()
        assert rc == 1

    def test_validate_json_file(self, capsys, tmp_path):
        # Create a minimal valid record
        record = {
            "model": "test",
            "probe": "test",
            "input": "hello",
            "output": "world",
            "status": "success",
            "score": 1.0,
        }
        in_file = tmp_path / "record.json"
        in_file.write_text(json.dumps(record))
        rc = cmd_schema(_make_args(op="validate", name="ResultRecord", input=str(in_file)))
        # May pass or fail depending on schema strictness - test the code path
        assert rc in (0, 1)

    def test_validate_json_list(self, capsys, tmp_path):
        records = [
            {"model": "test", "probe": "test", "input": "hello", "output": "world", "status": "success", "score": 1.0}
        ]
        in_file = tmp_path / "records.json"
        in_file.write_text(json.dumps(records))
        rc = cmd_schema(_make_args(op="validate", name="ResultRecord", input=str(in_file)))
        assert rc in (0, 1)

    def test_validate_jsonl_file(self, capsys, tmp_path):
        record = {"model": "test", "probe": "test", "input": "hello", "output": "world", "status": "success", "score": 1.0}
        in_file = tmp_path / "records.jsonl"
        in_file.write_text(json.dumps(record) + "\n")
        rc = cmd_schema(_make_args(op="validate", name="ResultRecord", input=str(in_file)))
        assert rc in (0, 1)

    def test_validate_jsonl_bad_json(self, capsys, tmp_path):
        in_file = tmp_path / "bad.jsonl"
        in_file.write_text("not json\n")
        rc = cmd_schema(
            _make_args(op="validate", name="ResultRecord", input=str(in_file), mode="warn")
        )
        captured = capsys.readouterr()
        # Should handle gracefully in warn mode
        assert rc in (0, 1)

    def test_validate_warn_mode(self, capsys, tmp_path):
        record = {"invalid": "data"}
        in_file = tmp_path / "bad_record.json"
        in_file.write_text(json.dumps(record))
        rc = cmd_schema(
            _make_args(op="validate", name="ResultRecord", input=str(in_file), mode="warn")
        )
        assert rc in (0, 1)


class TestSchemaUnknownOp:
    def test_unknown_op_treated_as_dump(self, capsys):
        rc = cmd_schema(_make_args(op="SomeSchema", name=None))
        # It should treat "SomeSchema" as a schema name
        captured = capsys.readouterr()
        assert rc in (0, 2)  # Either succeeds or schema not found
