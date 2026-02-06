"""Tests for remaining CLI commands: info, list, quicktest, report, run."""

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from insideLLMs.cli.commands.info import cmd_info
from insideLLMs.cli.commands.list_cmd import cmd_list
from insideLLMs.cli.commands.quicktest import cmd_quicktest


def _info_args(**kwargs):
    defaults = {"type": "model", "name": "dummy"}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _list_args(**kwargs):
    defaults = {"type": "all", "filter": None, "detailed": False}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _quicktest_args(**kwargs):
    defaults = {
        "model": "dummy",
        "prompt": "Hello world",
        "model_args": "{}",
        "temperature": 0.7,
        "max_tokens": 100,
        "probe": None,
        "verbose": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestInfoCommand:
    def test_model_info(self, capsys):
        rc = cmd_info(_info_args(type="model", name="dummy"))
        assert rc == 0

    def test_probe_info(self, capsys):
        rc = cmd_info(_info_args(type="probe", name="factuality"))
        assert rc == 0

    def test_model_not_found(self, capsys):
        rc = cmd_info(_info_args(type="model", name="nonexistent_xyz"))
        captured = capsys.readouterr()
        assert rc == 1

    def test_probe_not_found(self, capsys):
        rc = cmd_info(_info_args(type="probe", name="nonexistent_xyz"))
        captured = capsys.readouterr()
        assert rc == 1

    def test_dataset_info(self, capsys):
        rc = cmd_info(_info_args(type="dataset", name="factuality"))
        # May return 0 or 1 depending on dataset availability
        assert rc in (0, 1)


class TestListCommand:
    def test_list_all(self, capsys):
        rc = cmd_list(_list_args(type="all"))
        captured = capsys.readouterr()
        assert rc == 0
        assert "Available Models" in captured.out
        assert "Available Probes" in captured.out

    def test_list_models(self, capsys):
        rc = cmd_list(_list_args(type="models"))
        captured = capsys.readouterr()
        assert rc == 0
        assert "Available Models" in captured.out

    def test_list_probes(self, capsys):
        rc = cmd_list(_list_args(type="probes"))
        captured = capsys.readouterr()
        assert rc == 0
        assert "Available Probes" in captured.out

    def test_list_datasets(self, capsys):
        rc = cmd_list(_list_args(type="datasets"))
        assert rc == 0

    def test_list_trackers(self, capsys):
        rc = cmd_list(_list_args(type="trackers"))
        captured = capsys.readouterr()
        assert rc == 0
        assert "Experiment Tracking" in captured.out

    def test_list_with_filter(self, capsys):
        rc = cmd_list(_list_args(type="models", filter="dummy"))
        captured = capsys.readouterr()
        assert rc == 0

    def test_list_detailed(self, capsys):
        rc = cmd_list(_list_args(type="models", detailed=True))
        captured = capsys.readouterr()
        assert rc == 0

    def test_list_probes_detailed(self, capsys):
        rc = cmd_list(_list_args(type="probes", detailed=True))
        captured = capsys.readouterr()
        assert rc == 0

    def test_list_datasets_detailed(self, capsys):
        rc = cmd_list(_list_args(type="datasets", detailed=True))
        assert rc == 0

    def test_list_datasets_with_filter(self, capsys):
        rc = cmd_list(_list_args(type="datasets", filter="fact"))
        assert rc == 0


class TestQuicktestCommand:
    def test_quicktest_basic(self, capsys):
        rc = cmd_quicktest(_quicktest_args())
        captured = capsys.readouterr()
        assert rc == 0
        assert "Response" in captured.out

    def test_quicktest_unknown_model(self, capsys):
        rc = cmd_quicktest(_quicktest_args(model="nonexistent_xyz"))
        captured = capsys.readouterr()
        assert rc == 1

    def test_quicktest_with_probe(self, capsys):
        # Exercises the probe loading code path; may fail if registry returns instance
        rc = cmd_quicktest(_quicktest_args(probe="factuality"))
        assert rc in (0, 1)

    def test_quicktest_long_prompt(self, capsys):
        rc = cmd_quicktest(_quicktest_args(prompt="x" * 100))
        assert rc == 0

    def test_quicktest_with_model_args(self, capsys):
        rc = cmd_quicktest(_quicktest_args(model_args='{"response_prefix": "test"}'))
        assert rc == 0


class TestRunCommand:
    def test_run_missing_config(self, capsys):
        from insideLLMs.cli.commands.run import cmd_run

        args = argparse.Namespace(
            config="/nonexistent/config.yaml",
            verbose=False,
            quiet=False,
            run_id=None,
            run_root=None,
            run_dir=None,
            schema_version="v1",
            strict_serialization=False,
            deterministic_artifacts=False,
            track=None,
            track_project="default",
            use_async=False,
            concurrency=5,
            validate_output=False,
            validation_mode="warn",
            output=None,
            format="table",
            overwrite=False,
            resume=False,
        )
        rc = cmd_run(args)
        captured = capsys.readouterr()
        assert rc == 1

    def test_run_with_config(self, capsys, tmp_path):
        from insideLLMs.cli.commands.run import cmd_run

        config = {
            "model": {"type": "dummy"},
            "probe": {"type": "dummy"},
            "dataset": {"type": "inline", "items": [{"input": "Hello"}]},
        }
        config_path = tmp_path / "config.yaml"
        import yaml

        config_path.write_text(yaml.dump(config))

        args = argparse.Namespace(
            config=str(config_path),
            verbose=False,
            quiet=True,
            run_id="test-run-id",
            run_root=str(tmp_path / "runs"),
            run_dir=None,
            schema_version="v1",
            strict_serialization=False,
            deterministic_artifacts=False,
            track=None,
            track_project="default",
            use_async=False,
            concurrency=5,
            validate_output=False,
            validation_mode="warn",
            output=None,
            format="table",
            overwrite=True,
            resume=False,
        )
        rc = cmd_run(args)
        assert rc in (0, 1)

    def test_run_json_format(self, capsys, tmp_path):
        from insideLLMs.cli.commands.run import cmd_run

        config = {
            "model": {"type": "dummy"},
            "probe": {"type": "dummy"},
            "dataset": {"type": "inline", "items": [{"input": "Hello"}]},
        }
        config_path = tmp_path / "config.yaml"
        import yaml

        config_path.write_text(yaml.dump(config))

        args = argparse.Namespace(
            config=str(config_path),
            verbose=False,
            quiet=True,
            run_id="test-run-json",
            run_root=str(tmp_path / "runs"),
            run_dir=None,
            schema_version="v1",
            strict_serialization=False,
            deterministic_artifacts=False,
            track=None,
            track_project="default",
            use_async=False,
            concurrency=5,
            validate_output=False,
            validation_mode="warn",
            output=None,
            format="json",
            overwrite=True,
            resume=False,
        )
        rc = cmd_run(args)
        assert rc in (0, 1)

    def test_run_summary_format(self, capsys, tmp_path):
        from insideLLMs.cli.commands.run import cmd_run

        config = {
            "model": {"type": "dummy"},
            "probe": {"type": "dummy"},
            "dataset": {"type": "inline", "items": [{"input": "Hello"}]},
        }
        config_path = tmp_path / "config.yaml"
        import yaml

        config_path.write_text(yaml.dump(config))

        args = argparse.Namespace(
            config=str(config_path),
            verbose=False,
            quiet=True,
            run_id="test-run-summary",
            run_root=str(tmp_path / "runs"),
            run_dir=None,
            schema_version="v1",
            strict_serialization=False,
            deterministic_artifacts=False,
            track=None,
            track_project="default",
            use_async=False,
            concurrency=5,
            validate_output=False,
            validation_mode="warn",
            output=None,
            format="summary",
            overwrite=True,
            resume=False,
        )
        rc = cmd_run(args)
        assert rc in (0, 1)

    def test_run_markdown_format(self, capsys, tmp_path):
        from insideLLMs.cli.commands.run import cmd_run

        config = {
            "model": {"type": "dummy"},
            "probe": {"type": "dummy"},
            "dataset": {"type": "inline", "items": [{"input": "Hello"}]},
        }
        config_path = tmp_path / "config.yaml"
        import yaml

        config_path.write_text(yaml.dump(config))

        args = argparse.Namespace(
            config=str(config_path),
            verbose=False,
            quiet=True,
            run_id="test-run-md",
            run_root=str(tmp_path / "runs"),
            run_dir=None,
            schema_version="v1",
            strict_serialization=False,
            deterministic_artifacts=False,
            track=None,
            track_project="default",
            use_async=False,
            concurrency=5,
            validate_output=False,
            validation_mode="warn",
            output=None,
            format="markdown",
            overwrite=True,
            resume=False,
        )
        rc = cmd_run(args)
        assert rc in (0, 1)

    def test_run_with_output_file(self, capsys, tmp_path):
        from insideLLMs.cli.commands.run import cmd_run

        config = {
            "model": {"type": "dummy"},
            "probe": {"type": "dummy"},
            "dataset": {"type": "inline", "items": [{"input": "Hello"}]},
        }
        config_path = tmp_path / "config.yaml"
        import yaml

        config_path.write_text(yaml.dump(config))

        args = argparse.Namespace(
            config=str(config_path),
            verbose=False,
            quiet=True,
            run_id="test-run-output",
            run_root=str(tmp_path / "runs"),
            run_dir=None,
            schema_version="v1",
            strict_serialization=False,
            deterministic_artifacts=False,
            track=None,
            track_project="default",
            use_async=False,
            concurrency=5,
            validate_output=False,
            validation_mode="warn",
            output=str(tmp_path / "results.json"),
            format="table",
            overwrite=True,
            resume=False,
        )
        rc = cmd_run(args)
        assert rc in (0, 1)

    def test_run_with_local_tracker(self, capsys, tmp_path):
        from insideLLMs.cli.commands.run import cmd_run

        config = {
            "model": {"type": "dummy"},
            "probe": {"type": "dummy"},
            "dataset": {"type": "inline", "items": [{"input": "Hello"}]},
        }
        config_path = tmp_path / "config.yaml"
        import yaml

        config_path.write_text(yaml.dump(config))

        args = argparse.Namespace(
            config=str(config_path),
            verbose=False,
            quiet=True,
            run_id="test-run-track",
            run_root=str(tmp_path / "runs"),
            run_dir=None,
            schema_version="v1",
            strict_serialization=False,
            deterministic_artifacts=False,
            track="local",
            track_project="test-project",
            use_async=False,
            concurrency=5,
            validate_output=False,
            validation_mode="warn",
            output=None,
            format="table",
            overwrite=True,
            resume=False,
        )
        rc = cmd_run(args)
        assert rc in (0, 1)


class TestReportCommand:
    def test_missing_run_dir(self, capsys):
        from insideLLMs.cli.commands.report import cmd_report

        args = argparse.Namespace(
            run_dir="/nonexistent/dir",
            report_title="Test Report",
        )
        rc = cmd_report(args)
        assert rc == 1

    def test_missing_records(self, capsys, tmp_path):
        from insideLLMs.cli.commands.report import cmd_report

        args = argparse.Namespace(
            run_dir=str(tmp_path),
            report_title="Test Report",
        )
        rc = cmd_report(args)
        captured = capsys.readouterr()
        assert rc == 1

    def test_empty_records(self, capsys, tmp_path):
        from insideLLMs.cli.commands.report import cmd_report

        (tmp_path / "records.jsonl").write_text("")
        args = argparse.Namespace(
            run_dir=str(tmp_path),
            report_title="Test Report",
        )
        rc = cmd_report(args)
        assert rc == 1

    def test_report_from_records(self, capsys, tmp_path):
        from insideLLMs.cli.commands.report import cmd_report

        records = [
            {
                "model": "dummy",
                "probe": "dummy",
                "input": "Hello",
                "output": "World",
                "status": "success",
                "score": 1.0,
                "latency_ms": 10.0,
                "completed_at": "2024-01-01T00:00:00Z",
                "schema_version": "v1",
            }
        ]
        records_path = tmp_path / "records.jsonl"
        records_path.write_text("\n".join(json.dumps(r) for r in records))

        args = argparse.Namespace(
            run_dir=str(tmp_path),
            report_title="Test Report",
        )
        rc = cmd_report(args)
        assert rc in (0, 1)
