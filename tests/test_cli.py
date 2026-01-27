"""Tests for the CLI module."""

import json
import tempfile
from pathlib import Path

import pytest


class TestColors:
    """Test the Colors class."""

    def test_all_color_codes_exist(self):
        """Test that all color codes are defined."""
        from insideLLMs.cli import Colors

        assert hasattr(Colors, "RESET")
        assert hasattr(Colors, "BOLD")
        assert hasattr(Colors, "DIM")
        assert hasattr(Colors, "RED")
        assert hasattr(Colors, "GREEN")
        assert hasattr(Colors, "YELLOW")
        assert hasattr(Colors, "BLUE")
        assert hasattr(Colors, "CYAN")
        assert hasattr(Colors, "MAGENTA")

    def test_color_codes_are_strings(self):
        """Test that color codes are ANSI escape sequences."""
        from insideLLMs.cli import Colors

        assert Colors.RESET == "\033[0m"
        assert Colors.BOLD == "\033[1m"
        assert Colors.RED == "\033[31m"
        assert Colors.GREEN == "\033[32m"


class TestColorize:
    """Test the colorize function."""

    def test_colorize_with_single_code(self):
        """Test colorize with a single color code."""
        from insideLLMs.cli import USE_COLOR, Colors, colorize

        if USE_COLOR:
            result = colorize("test", Colors.RED)
            assert Colors.RED in result
            assert Colors.RESET in result
            assert "test" in result

    def test_colorize_with_multiple_codes(self):
        """Test colorize with multiple color codes."""
        from insideLLMs.cli import USE_COLOR, Colors, colorize

        if USE_COLOR:
            result = colorize("test", Colors.BOLD, Colors.RED)
            assert Colors.BOLD in result
            assert Colors.RED in result
            assert "test" in result

    def test_colorize_without_color_support(self):
        """Test colorize when color is disabled."""
        import insideLLMs.cli as cli_module

        original = cli_module.USE_COLOR
        try:
            cli_module.USE_COLOR = False
            result = cli_module.colorize("test", cli_module.Colors.RED)
            assert result == "test"
        finally:
            cli_module.USE_COLOR = original


class TestProgressBar:
    """Test the ProgressBar class."""

    def test_create_progress_bar(self):
        """Test creating a progress bar."""
        from insideLLMs.cli import ProgressBar

        bar = ProgressBar(total=100, width=40, prefix="Test")
        assert bar.total == 100
        assert bar.width == 40
        assert bar.prefix == "Test"
        assert bar.current == 0

    def test_update_progress(self):
        """Test updating progress."""
        from insideLLMs.cli import ProgressBar

        bar = ProgressBar(total=100)
        bar.update(50)
        assert bar.current == 50

    def test_increment_progress(self):
        """Test incrementing progress."""
        from insideLLMs.cli import ProgressBar

        bar = ProgressBar(total=100)
        bar.increment()
        assert bar.current == 1
        bar.increment(5)
        assert bar.current == 6

    def test_finish_progress(self):
        """Test finishing progress."""
        from insideLLMs.cli import ProgressBar

        bar = ProgressBar(total=100)
        bar.update(50)
        bar.finish()
        assert bar.current == 100


class TestSpinner:
    """Test the Spinner class."""

    def test_create_spinner(self):
        """Test creating a spinner."""
        from insideLLMs.cli import Spinner

        spinner = Spinner(message="Loading")
        assert spinner.message == "Loading"
        assert spinner.frame_idx == 0
        assert not spinner.running

    def test_spinner_frames(self):
        """Test spinner has frames."""
        from insideLLMs.cli import Spinner

        assert len(Spinner.FRAMES) > 0


class TestCreateParser:
    """Test the argument parser creation."""

    def test_create_parser(self):
        """Test creating the argument parser."""
        from insideLLMs.cli import create_parser

        parser = create_parser()
        assert parser is not None
        assert parser.prog == "insidellms"

    def test_parser_has_all_commands(self):
        """Test parser has all expected commands."""
        from insideLLMs.cli import create_parser

        parser = create_parser()

        # Check that subparsers are set up
        args = parser.parse_args(["list", "all"])
        assert args.command == "list"
        assert args.type == "all"

    def test_run_command_args(self):
        """Test run command arguments."""
        from insideLLMs.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "config.yaml",
                "--verbose",
                "--async",
                "--run-dir",
                "./runs/run-1",
                "--run-root",
                "./runs",
                "--run-id",
                "run-1",
                "--overwrite",
                "--validate-output",
                "--schema-version",
                "1.0.0",
                "--validation-mode",
                "warn",
            ]
        )
        assert args.command == "run"
        assert args.config == "config.yaml"
        assert args.verbose is True
        assert args.use_async is True
        assert args.run_dir == "./runs/run-1"
        assert args.run_root == "./runs"
        assert args.run_id == "run-1"
        assert args.overwrite is True
        assert args.validate_output is True
        assert args.schema_version == "1.0.0"
        assert args.validation_mode == "warn"

    def test_harness_command_args(self):
        """Test harness command arguments."""
        from insideLLMs.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                "harness",
                "harness.yaml",
                "--output-dir",
                "./legacy_out",
                "--run-dir",
                "./runs/h-1",
                "--run-root",
                "./runs",
                "--run-id",
                "h-1",
                "--overwrite",
                "--skip-report",
                "--track",
                "local",
                "--track-project",
                "my-project",
                "--validate-output",
                "--schema-version",
                "1.0.0",
                "--validation-mode",
                "warn",
            ]
        )
        assert args.command == "harness"
        assert args.config == "harness.yaml"
        assert args.output_dir == "./legacy_out"
        assert args.run_dir == "./runs/h-1"
        assert args.run_root == "./runs"
        assert args.run_id == "h-1"
        assert args.overwrite is True
        assert args.skip_report is True
        assert args.track == "local"
        assert args.track_project == "my-project"
        assert args.validate_output is True
        assert args.schema_version == "1.0.0"
        assert args.validation_mode == "warn"


class TestCmdRun:
    """Test the run command end-to-end (including run_dir/run_root semantics)."""

    def _write_minimal_yaml_config(self, tmp_path: Path) -> Path:
        import yaml

        data_path = tmp_path / "data.jsonl"
        data_path.write_text('{"question": "Is 2+2=4?"}\n', encoding="utf-8")

        config_path = tmp_path / "config.yaml"
        config = {
            "model": {"type": "dummy"},
            "probe": {"type": "logic"},
            "dataset": {
                "path": str(data_path),
                "format": "jsonl",
                "input_field": "question",
            },
        }
        config_path.write_text(yaml.dump(config), encoding="utf-8")
        return config_path

    def test_run_with_run_dir_writes_artifacts(self, tmp_path, capsys):
        """--run-dir should be treated as the final directory for artifacts."""
        import importlib.util

        if importlib.util.find_spec("pydantic") is None:
            pytest.skip("pydantic not installed")

        from insideLLMs.cli import main

        config_path = self._write_minimal_yaml_config(tmp_path)
        run_dir = tmp_path / "my_run_dir"

        rc = main(
            [
                "run",
                str(config_path),
                "--format",
                "summary",
                "--run-dir",
                str(run_dir),
                "--run-id",
                "run-123",
            ]
        )
        assert rc == 0

        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "records.jsonl").exists()
        assert (run_dir / "config.resolved.yaml").exists()

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["run_id"] == "run-123"

        captured = capsys.readouterr()
        assert "Run written to:" in captured.out
        assert str(run_dir) in captured.out

    def test_run_with_run_root_and_run_id_forms_dir(self, tmp_path):
        """When --run-dir is not set, the directory should be <run_root>/<run_id>."""
        import importlib.util

        if importlib.util.find_spec("pydantic") is None:
            pytest.skip("pydantic not installed")

        from insideLLMs.cli import main

        config_path = self._write_minimal_yaml_config(tmp_path)
        run_root = tmp_path / "runs_root"
        run_id = "run-xyz"

        rc = main(
            [
                "run",
                str(config_path),
                "--format",
                "summary",
                "--run-root",
                str(run_root),
                "--run-id",
                run_id,
            ]
        )
        assert rc == 0

        run_dir = run_root / run_id
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "records.jsonl").exists()
        assert (run_dir / "config.resolved.yaml").exists()

    def test_run_overwrite_policy(self, tmp_path):
        """Existing non-empty run_dir should fail unless --overwrite is provided."""
        import importlib.util

        if importlib.util.find_spec("pydantic") is None:
            pytest.skip("pydantic not installed")

        from insideLLMs.cli import main

        config_path = self._write_minimal_yaml_config(tmp_path)
        run_dir = tmp_path / "existing_run"
        run_dir.mkdir(parents=True)
        (run_dir / "keep.txt").write_text("do not keep", encoding="utf-8")

        rc = main(["run", str(config_path), "--format", "summary", "--run-dir", str(run_dir)])
        assert rc == 1

        # With --overwrite but without a run sentinel, this should still refuse
        # (critical safety guard).
        rc = main(
            [
                "run",
                str(config_path),
                "--format",
                "summary",
                "--run-dir",
                str(run_dir),
                "--overwrite",
            ]
        )
        assert rc == 1

        # Add a sentinel marker to indicate this is an insideLLMs run directory.
        (run_dir / ".insidellms_run").write_text("marker\n", encoding="utf-8")

        rc = main(
            [
                "run",
                str(config_path),
                "--format",
                "summary",
                "--run-dir",
                str(run_dir),
                "--overwrite",
            ]
        )
        assert rc == 0

        assert not (run_dir / "keep.txt").exists()
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "records.jsonl").exists()
        assert (run_dir / "config.resolved.yaml").exists()

    def test_quicktest_command_args(self):
        """Test quicktest command arguments."""
        from insideLLMs.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["quicktest", "Hello", "-m", "dummy"])
        assert args.command == "quicktest"
        assert args.prompt == "Hello"
        assert args.model == "dummy"

    def test_benchmark_command_args(self):
        """Test benchmark command arguments."""
        from insideLLMs.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            ["benchmark", "--models", "dummy,openai", "--probes", "logic,bias", "-n", "5"]
        )
        assert args.command == "benchmark"
        assert args.models == "dummy,openai"
        assert args.probes == "logic,bias"
        assert args.max_examples == 5

    def test_compare_command_args(self):
        """Test compare command arguments."""
        from insideLLMs.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["compare", "--models", "dummy,openai", "--input", "Hello world"])
        assert args.command == "compare"
        assert args.models == "dummy,openai"
        assert args.input == "Hello world"

    def test_init_command_args(self):
        """Test init command arguments."""
        from insideLLMs.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["init", "test.yaml", "--template", "full"])
        assert args.command == "init"
        assert args.output == "test.yaml"
        assert args.template == "full"

    def test_info_command_args(self):
        """Test info command arguments."""
        from insideLLMs.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["info", "model", "dummy"])
        assert args.command == "info"
        assert args.type == "model"
        assert args.name == "dummy"

    def test_validate_command_args(self):
        """Test validate command arguments."""
        from insideLLMs.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["validate", "config.yaml"])
        assert args.command == "validate"
        assert args.config == "config.yaml"

    def test_schema_command_args(self):
        """Test schema command arguments."""
        from insideLLMs.cli import create_parser

        parser = create_parser()

        args = parser.parse_args(["schema", "list"])
        assert args.command == "schema"
        assert args.op == "list"

        args = parser.parse_args(["schema", "dump", "--name", "ProbeResult", "--version", "1.0.0"])
        assert args.command == "schema"
        assert args.op == "dump"
        assert args.name == "ProbeResult"
        assert args.version == "1.0.0"

        # Shortcut form: `insidellms schema <SchemaName>`
        args = parser.parse_args(["schema", "ProbeResult"])
        assert args.command == "schema"
        assert args.op == "ProbeResult"

    def test_export_command_args(self):
        """Test export command arguments."""
        from insideLLMs.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["export", "results.json", "-f", "csv", "-o", "out.csv"])
        assert args.command == "export"
        assert args.input == "results.json"
        assert args.format == "csv"
        assert args.output == "out.csv"


class TestCmdHarness:
    """Test the harness command emits a validate-able run directory."""

    def _write_minimal_harness_yaml_config(self, tmp_path: Path) -> Path:
        import yaml

        data_path = tmp_path / "data.jsonl"
        data_path.write_text(
            '{"question": "What is 2 + 2?"}\n{"question": "What is 3 + 3?"}\n',
            encoding="utf-8",
        )

        config_path = tmp_path / "harness.yaml"
        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(data_path), "input_field": "question"},
            "max_examples": 1,
        }
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return config_path

    def test_harness_emits_run_dir_and_validate_succeeds(self, tmp_path, capsys):
        import importlib.util

        if importlib.util.find_spec("pydantic") is None:
            pytest.skip("pydantic not installed")

        from insideLLMs.cli import main

        config_path = self._write_minimal_harness_yaml_config(tmp_path)
        run_dir = tmp_path / "harness_run"

        rc = main(
            [
                "harness",
                str(config_path),
                "--run-dir",
                str(run_dir),
                "--run-id",
                "harness-123",
            ]
        )
        assert rc == 0

        # Standardized run-dir artifacts
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "records.jsonl").exists()
        assert (run_dir / "config.resolved.yaml").exists()
        assert (run_dir / ".insidellms_run").exists()

        # Backward-compatible harness outputs
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "report.html").exists()
        assert (run_dir / "results.jsonl").exists()

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["run_id"] == "harness-123"
        assert manifest.get("records_file") == "records.jsonl"
        assert manifest.get("custom", {}).get("harness")

        # Records should be canonical ResultRecord lines with custom.harness nesting
        first_line = (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()[0]
        record = json.loads(first_line)
        assert record["run_id"] == "harness-123"
        assert record.get("custom", {}).get("harness")

        # Validate should succeed against manifest + records
        rc = main(["validate", str(run_dir)])
        assert rc == 0

        captured = capsys.readouterr()
        assert "Validate with:" in captured.out

    def test_harness_track_local_writes_tracking_artifacts(self, tmp_path):
        from insideLLMs.cli import main

        config_path = self._write_minimal_harness_yaml_config(tmp_path)
        run_dir = tmp_path / "harness_run"

        rc = main(
            [
                "harness",
                str(config_path),
                "--run-dir",
                str(run_dir),
                "--run-id",
                "harness-123",
                "--track",
                "local",
                "--track-project",
                "test-project",
            ]
        )
        assert rc == 0

        tracking_run_dir = tmp_path / "tracking" / "test-project" / "harness-123"
        assert tracking_run_dir.exists()
        assert (tracking_run_dir / "metadata.json").exists()
        assert (tracking_run_dir / "final_state.json").exists()
        assert (tracking_run_dir / "metrics.json").exists()
        assert (tracking_run_dir / "params.json").exists()
        assert (tracking_run_dir / "artifacts.json").exists()

        metrics = json.loads((tracking_run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert any("wall_time_seconds" in item.get("metrics", {}) for item in metrics)

        # Artifacts are copied to <tracking_run_dir>/artifacts/
        assert (tracking_run_dir / "artifacts" / "manifest.json").exists()
        assert (tracking_run_dir / "artifacts" / "records.jsonl").exists()

        # Tracking output must not pollute the standardized run directory.
        assert not (run_dir / "tracking").exists()


class TestMainFunction:
    """Test the main entry point."""

    def test_main_no_command_shows_help(self):
        """Test main with no command shows help."""
        from insideLLMs.cli import main

        result = main([])
        assert result == 0

    def test_main_with_version(self):
        """Test main with --version."""
        from insideLLMs.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0


class TestCmdList:
    """Test the list command."""

    def test_list_models(self, capsys):
        """Test listing models."""
        from insideLLMs.cli import main

        result = main(["list", "models"])
        assert result == 0

        captured = capsys.readouterr()
        assert "dummy" in captured.out.lower() or "model" in captured.out.lower()

    def test_list_probes(self, capsys):
        """Test listing probes."""
        from insideLLMs.cli import main

        result = main(["list", "probes"])
        assert result == 0

        captured = capsys.readouterr()
        assert "logic" in captured.out.lower() or "probe" in captured.out.lower()

    def test_list_datasets(self, capsys):
        """Test listing datasets."""
        from insideLLMs.cli import main

        result = main(["list", "datasets"])
        assert result == 0

        captured = capsys.readouterr()
        assert "dataset" in captured.out.lower() or "reasoning" in captured.out.lower()

    def test_list_all(self, capsys):
        """Test listing all resources."""
        from insideLLMs.cli import main

        result = main(["list", "all"])
        assert result == 0


class TestCmdInit:
    """Test the init command."""

    def test_init_creates_config(self):
        """Test init creates a configuration file."""
        from insideLLMs.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            result = main(["init", str(config_path)])
            assert result == 0
            assert config_path.exists()

    def test_init_basic_template(self):
        """Test init with basic template."""
        import yaml

        from insideLLMs.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            result = main(["init", str(config_path), "--template", "basic"])
            assert result == 0

            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert "model" in config
            assert "probe" in config
            assert "dataset" in config

    def test_init_full_template(self):
        """Test init with full template."""
        import yaml

        from insideLLMs.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            result = main(["init", str(config_path), "--template", "full"])
            assert result == 0

            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert "model" in config
            assert "probe" in config
            assert "benchmark" in config
            assert "tracking" in config


class TestCmdInfo:
    """Test the info command."""

    def test_info_model(self, capsys):
        """Test getting info about a model."""
        from insideLLMs.cli import main

        result = main(["info", "model", "dummy"])
        assert result == 0

        captured = capsys.readouterr()
        assert "dummy" in captured.out.lower()

    def test_info_probe(self, capsys):
        """Test getting info about a probe."""
        from insideLLMs.cli import main

        result = main(["info", "probe", "logic"])
        assert result == 0

        captured = capsys.readouterr()
        assert "logic" in captured.out.lower()

    def test_info_unknown_model(self, capsys):
        """Test getting info about an unknown model."""
        from insideLLMs.cli import main

        result = main(["info", "model", "nonexistent_model"])
        assert result == 1


class TestCmdQuicktest:
    """Test the quicktest command."""

    def test_quicktest_with_dummy_model(self, capsys):
        """Test quicktest with dummy model."""
        from insideLLMs.cli import main

        result = main(["quicktest", "What is 2+2?", "--model", "dummy"])
        assert result == 0

        captured = capsys.readouterr()
        assert "response" in captured.out.lower()


class TestCmdValidate:
    """Test the validate command."""

    def test_validate_valid_config(self):
        """Test validating a valid config."""
        import yaml

        from insideLLMs.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {
                "model": {"type": "dummy"},
                "probe": {"type": "logic"},
                "dataset": {"format": "jsonl", "path": "data.jsonl"},
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            result = main(["validate", str(config_path)])
            assert result == 0

    def test_validate_invalid_config(self):
        """Test validating an invalid config."""
        import yaml

        from insideLLMs.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {"invalid": "config"}  # Missing required fields
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            result = main(["validate", str(config_path)])
            assert result == 1

    def test_validate_missing_config(self):
        """Test validating a missing config file."""
        from insideLLMs.cli import main

        result = main(["validate", "nonexistent.yaml"])
        assert result == 1

    def test_validate_run_dir_ok(self, tmp_path):
        """Validate a run_dir containing manifest.json + records.jsonl."""
        import importlib.util

        if importlib.util.find_spec("pydantic") is None:
            pytest.skip("pydantic not installed")

        from insideLLMs.cli import main

        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)

        manifest = {
            "schema_version": "1.0.0",
            "run_id": "run-1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:00:01+00:00",
            "model": {"model_id": "dummy", "provider": "local", "params": {}},
            "probe": {"probe_id": "logic", "probe_version": "1.0.0", "params": {}},
            "dataset": {
                "dataset_id": "unit",
                "dataset_version": "1",
                "dataset_hash": None,
                "provenance": None,
                "params": {},
            },
            "record_count": 1,
            "success_count": 1,
            "error_count": 0,
            "records_file": "records.jsonl",
            "schemas": {"RunManifest": "1.0.0", "ResultRecord": "1.0.0"},
            "custom": {},
        }

        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        record = {
            "schema_version": "1.0.0",
            "run_id": "run-1",
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:00:01+00:00",
            "model": {"model_id": "dummy", "provider": "local", "params": {}},
            "probe": {"probe_id": "logic", "probe_version": "1.0.0", "params": {}},
            "example_id": "ex-1",
            "dataset": {
                "dataset_id": "unit",
                "dataset_version": "1",
                "dataset_hash": None,
                "provenance": None,
                "params": {},
            },
            "status": "success",
            "scores": {},
            "usage": {},
            "custom": {},
        }
        (run_dir / "records.jsonl").write_text(json.dumps(record) + "\n")

        assert main(["validate", str(run_dir)]) == 0

    def test_validate_run_dir_strict_and_warn(self, tmp_path):
        """Strict should fail on bad record; warn should still exit 0."""
        import importlib.util

        if importlib.util.find_spec("pydantic") is None:
            pytest.skip("pydantic not installed")

        from insideLLMs.cli import main

        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)

        manifest = {
            "schema_version": "1.0.0",
            "run_id": "run-1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:00:01+00:00",
            "model": {"model_id": "dummy", "provider": "local", "params": {}},
            "probe": {"probe_id": "logic", "probe_version": "1.0.0", "params": {}},
            "record_count": 1,
            "success_count": 1,
            "error_count": 0,
            "records_file": "records.jsonl",
            "schemas": {"RunManifest": "1.0.0", "ResultRecord": "1.0.0"},
            "custom": {},
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        bad_record = {
            "schema_version": "1.0.0",
            # missing run_id
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:00:01+00:00",
            "model": {"model_id": "dummy", "provider": "local", "params": {}},
            "probe": {"probe_id": "logic", "probe_version": "1.0.0", "params": {}},
            "example_id": "ex-1",
            "status": "success",
        }
        (run_dir / "records.jsonl").write_text(json.dumps(bad_record) + "\n")

        assert main(["validate", str(run_dir)]) == 1
        assert main(["validate", str(run_dir), "--mode", "warn"]) == 0


class TestCmdExport:
    """Test the export command."""

    def test_export_to_csv(self):
        """Test exporting results to CSV."""
        from insideLLMs.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input JSON
            input_path = Path(tmpdir) / "results.json"
            results = [
                {"input": "test1", "output": "result1", "status": "success"},
                {"input": "test2", "output": "result2", "status": "success"},
            ]
            with open(input_path, "w") as f:
                json.dump(results, f)

            output_path = Path(tmpdir) / "results.csv"
            result = main(["export", str(input_path), "-f", "csv", "-o", str(output_path)])
            assert result == 0
            assert output_path.exists()

    def test_export_to_markdown(self):
        """Test exporting results to Markdown."""
        from insideLLMs.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "results.json"
            results = [{"input": "test", "output": "result"}]
            with open(input_path, "w") as f:
                json.dump(results, f)

            output_path = Path(tmpdir) / "results.md"
            result = main(["export", str(input_path), "-f", "markdown", "-o", str(output_path)])
            assert result == 0

    def test_export_to_latex(self):
        """Test exporting results to LaTeX."""
        from insideLLMs.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "results.json"
            results = [{"input": "test", "output": "result"}]
            with open(input_path, "w") as f:
                json.dump(results, f)

            output_path = Path(tmpdir) / "results.tex"
            result = main(["export", str(input_path), "-f", "latex", "-o", str(output_path)])
            assert result == 0
            assert output_path.exists()

            # Check it's valid LaTeX
            content = output_path.read_text()
            assert "\\begin{table}" in content
            assert "\\end{table}" in content

    def test_export_missing_input(self):
        """Test exporting with missing input file."""
        from insideLLMs.cli import main

        result = main(["export", "nonexistent.json", "-f", "csv"])
        assert result == 1


class TestPrintFunctions:
    """Test the print utility functions."""

    def test_print_header(self, capsys):
        """Test print_header function."""
        from insideLLMs.cli import print_header

        print_header("Test Header")
        captured = capsys.readouterr()
        assert "Test Header" in captured.out

    def test_print_subheader(self, capsys):
        """Test print_subheader function."""
        from insideLLMs.cli import print_subheader

        print_subheader("Test Subheader")
        captured = capsys.readouterr()
        assert "Test Subheader" in captured.out

    def test_print_success(self, capsys):
        """Test print_success function."""
        from insideLLMs.cli import print_success

        print_success("Operation successful")
        captured = capsys.readouterr()
        assert "Operation successful" in captured.out

    def test_print_error(self, capsys):
        """Test print_error function."""
        from insideLLMs.cli import print_error

        print_error("Something went wrong")
        captured = capsys.readouterr()
        assert "Something went wrong" in captured.err

    def test_print_warning(self, capsys):
        """Test print_warning function."""
        from insideLLMs.cli import print_warning

        print_warning("Be careful")
        captured = capsys.readouterr()
        assert "Be careful" in captured.out

    def test_print_info(self, capsys):
        """Test print_info function."""
        from insideLLMs.cli import print_info

        print_info("Some information")
        captured = capsys.readouterr()
        assert "Some information" in captured.out

    def test_print_key_value(self, capsys):
        """Test print_key_value function."""
        from insideLLMs.cli import print_key_value

        print_key_value("Key", "Value")
        captured = capsys.readouterr()
        assert "Key" in captured.out
        assert "Value" in captured.out


class TestSupportsColor:
    """Test color support detection."""

    def test_no_color_env_disables_color(self):
        """Test NO_COLOR environment variable."""
        import os

        from insideLLMs.cli import _supports_color

        original = os.environ.get("NO_COLOR")
        try:
            os.environ["NO_COLOR"] = "1"
            assert _supports_color() is False
        finally:
            if original is None:
                os.environ.pop("NO_COLOR", None)
            else:
                os.environ["NO_COLOR"] = original

    def test_force_color_env_enables_color(self):
        """Test FORCE_COLOR environment variable."""
        import os

        from insideLLMs.cli import _supports_color

        original_no_color = os.environ.pop("NO_COLOR", None)
        original_force_color = os.environ.get("FORCE_COLOR")
        try:
            os.environ["FORCE_COLOR"] = "1"
            assert _supports_color() is True
        finally:
            if original_no_color is not None:
                os.environ["NO_COLOR"] = original_no_color
            if original_force_color is None:
                os.environ.pop("FORCE_COLOR", None)
            else:
                os.environ["FORCE_COLOR"] = original_force_color


class TestJsonDefault:
    """Test the _json_default function."""

    def test_datetime_serialization(self):
        """Test datetime is serialized to ISO format."""
        from datetime import datetime

        from insideLLMs.cli import _json_default

        dt = datetime(2026, 1, 21, 12, 0, 0)
        result = _json_default(dt)
        assert result == "2026-01-21T12:00:00"

    def test_enum_serialization(self):
        """Test enum is serialized to its value."""
        from enum import Enum

        from insideLLMs.cli import _json_default

        class Color(Enum):
            RED = "red"
            GREEN = "green"

        result = _json_default(Color.RED)
        assert result == "red"

    def test_path_serialization(self):
        """Test Path is serialized to string."""
        from insideLLMs.cli import _json_default

        result = _json_default(Path("/foo/bar"))
        assert result == "/foo/bar"

    def test_set_serialization(self):
        """Test set is serialized to sorted list."""
        from insideLLMs.cli import _json_default

        result = _json_default({3, 1, 2})
        assert result == [1, 2, 3]

    def test_frozenset_serialization(self):
        """Test frozenset is serialized to sorted list."""
        from insideLLMs.cli import _json_default

        result = _json_default(frozenset(["c", "a", "b"]))
        assert result == ["a", "b", "c"]

    def test_fallback_to_str(self):
        """Test fallback to str for unknown types."""
        from insideLLMs.cli import _json_default

        class CustomClass:
            def __str__(self):
                return "custom"

        result = _json_default(CustomClass())
        assert result == "custom"


class TestFormatFunctions:
    """Test formatting utility functions."""

    def test_format_percent_with_value(self):
        """Test _format_percent with a value."""
        from insideLLMs.cli import _format_percent

        assert _format_percent(0.5) == "50.0%"
        assert _format_percent(0.123) == "12.3%"
        assert _format_percent(1.0) == "100.0%"

    def test_format_percent_with_none(self):
        """Test _format_percent with None."""
        from insideLLMs.cli import _format_percent

        assert _format_percent(None) == "-"

    def test_format_float_with_value(self):
        """Test _format_float with a value."""
        from insideLLMs.cli import _format_float

        assert _format_float(0.5) == "0.500"
        assert _format_float(1.23456) == "1.235"

    def test_format_float_with_none(self):
        """Test _format_float with None."""
        from insideLLMs.cli import _format_float

        assert _format_float(None) == "-"


class TestParseDatetime:
    """Test _parse_datetime function."""

    def test_parse_datetime_object(self):
        """Test passing a datetime object."""
        from datetime import datetime

        from insideLLMs.cli import _parse_datetime

        dt = datetime(2026, 1, 21)
        result = _parse_datetime(dt)
        assert result == dt

    def test_parse_datetime_string(self):
        """Test parsing an ISO format string."""
        from datetime import datetime

        from insideLLMs.cli import _parse_datetime

        result = _parse_datetime("2026-01-21T12:00:00")
        assert result == datetime(2026, 1, 21, 12, 0, 0)

    def test_parse_datetime_invalid_string(self):
        """Test parsing an invalid string."""
        from insideLLMs.cli import _parse_datetime

        result = _parse_datetime("not a date")
        assert result is None

    def test_parse_datetime_non_string(self):
        """Test parsing a non-string, non-datetime value."""
        from insideLLMs.cli import _parse_datetime

        result = _parse_datetime(12345)
        assert result is None


class TestStableJsonDumps:
    """Test _stable_json_dumps function."""

    def test_stable_output(self):
        """Test that output is stable regardless of key order."""
        from insideLLMs.cli import _stable_json_dumps

        result1 = _stable_json_dumps({"b": 2, "a": 1})
        result2 = _stable_json_dumps({"a": 1, "b": 2})
        assert result1 == result2
        assert result1 == '{"a":1,"b":2}'


class TestFingerprintValue:
    """Test _fingerprint_value function."""

    def test_fingerprint_none(self):
        """Test fingerprint of None."""
        from insideLLMs.cli import _fingerprint_value

        assert _fingerprint_value(None) is None

    def test_fingerprint_string(self):
        """Test fingerprint of string is consistent."""
        from insideLLMs.cli import _fingerprint_value

        result1 = _fingerprint_value("test")
        result2 = _fingerprint_value("test")
        assert result1 == result2
        assert len(result1) == 12  # SHA256 truncated to 12 chars


class TestStatusFromRecord:
    """Test _status_from_record function."""

    def test_status_from_record_enum(self):
        """Test when value is already a ResultStatus."""
        from insideLLMs.cli import _status_from_record
        from insideLLMs.types import ResultStatus

        result = _status_from_record(ResultStatus.SUCCESS)
        assert result == ResultStatus.SUCCESS

    def test_status_from_record_string(self):
        """Test when value is a valid status string."""
        from insideLLMs.cli import _status_from_record
        from insideLLMs.types import ResultStatus

        result = _status_from_record("success")
        assert result == ResultStatus.SUCCESS

    def test_status_from_record_invalid(self):
        """Test when value is invalid returns ERROR."""
        from insideLLMs.cli import _status_from_record
        from insideLLMs.types import ResultStatus

        result = _status_from_record("invalid_status")
        assert result == ResultStatus.ERROR


class TestProbeCategoryFromValue:
    """Test _probe_category_from_value function."""

    def test_probe_category_enum(self):
        """Test when value is already a ProbeCategory."""
        from insideLLMs.cli import _probe_category_from_value
        from insideLLMs.types import ProbeCategory

        result = _probe_category_from_value(ProbeCategory.REASONING)
        assert result == ProbeCategory.REASONING

    def test_probe_category_string(self):
        """Test when value is a valid category string."""
        from insideLLMs.cli import _probe_category_from_value
        from insideLLMs.types import ProbeCategory

        result = _probe_category_from_value("reasoning")
        assert result == ProbeCategory.REASONING

    def test_probe_category_invalid(self):
        """Test when value is invalid returns CUSTOM."""
        from insideLLMs.cli import _probe_category_from_value
        from insideLLMs.types import ProbeCategory

        result = _probe_category_from_value("invalid_category")
        assert result == ProbeCategory.CUSTOM


class TestReadJsonlRecords:
    """Test _read_jsonl_records function."""

    def test_read_valid_jsonl(self):
        """Test reading a valid JSONL file."""
        from insideLLMs.cli import _read_jsonl_records

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": 1, "name": "test1"}\n')
            f.write('{"id": 2, "name": "test2"}\n')
            path = Path(f.name)

        try:
            records = _read_jsonl_records(path)
            assert len(records) == 2
            assert records[0]["id"] == 1
            assert records[1]["name"] == "test2"
        finally:
            path.unlink()

    def test_read_jsonl_with_empty_lines(self):
        """Test reading JSONL with empty lines."""
        from insideLLMs.cli import _read_jsonl_records

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": 1}\n')
            f.write("\n")  # empty line
            f.write('{"id": 2}\n')
            path = Path(f.name)

        try:
            records = _read_jsonl_records(path)
            assert len(records) == 2
        finally:
            path.unlink()

    def test_read_jsonl_invalid_json(self):
        """Test reading JSONL with invalid JSON raises ValueError."""
        from insideLLMs.cli import _read_jsonl_records

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": 1}\n')
            f.write("not valid json\n")
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid JSON on line 2"):
                _read_jsonl_records(path)
        finally:
            path.unlink()


class TestRecordKey:
    """Test _record_key function."""

    def test_record_key_with_harness(self):
        """Test record key extraction with harness metadata."""
        from insideLLMs.cli import _record_key

        record = {
            "custom": {
                "harness": {
                    "model_id": "gpt-4",
                    "probe_type": "logic",
                },
                "replicate_key": "rep-1",
            }
        }
        model_id, probe_id, example_id = _record_key(record)
        assert model_id == "gpt-4"
        assert probe_id == "logic"
        assert example_id == "rep-1"

    def test_record_key_with_model_probe_spec(self):
        """Test record key extraction from model/probe spec."""
        from insideLLMs.cli import _record_key

        record = {
            "model": {"model_id": "claude-3"},
            "probe": {"probe_id": "bias"},
            "example_id": "ex-1",
        }
        model_id, probe_id, example_id = _record_key(record)
        assert model_id == "claude-3"
        assert probe_id == "bias"
        assert example_id == "ex-1"


class TestRecordLabel:
    """Test _record_label function."""

    def test_record_label_with_harness(self):
        """Test record label extraction with harness metadata."""
        from insideLLMs.cli import _record_label

        record = {
            "custom": {
                "harness": {
                    "model_name": "GPT-4",
                    "model_id": "gpt-4-turbo",
                    "probe_name": "Logic Test",
                    "example_index": 5,
                }
            }
        }
        model_label, probe_name, example_id = _record_label(record)
        assert "GPT-4" in model_label
        assert "gpt-4-turbo" in model_label
        assert probe_name == "Logic Test"
        assert example_id == "5"


class TestStatusString:
    """Test _status_string function."""

    def test_status_string_with_status(self):
        """Test _status_string with status present."""
        from insideLLMs.cli import _status_string

        assert _status_string({"status": "success"}) == "success"

    def test_status_string_without_status(self):
        """Test _status_string without status."""
        from insideLLMs.cli import _status_string

        assert _status_string({}) == "unknown"


class TestOutputText:
    """Test _output_text function."""

    def test_output_text_direct(self):
        """Test output_text extraction from direct field."""
        from insideLLMs.cli import _output_text

        assert _output_text({"output_text": "Hello"}) == "Hello"

    def test_output_text_from_output_field(self):
        """Test output_text extraction from output field."""
        from insideLLMs.cli import _output_text

        assert _output_text({"output": "World"}) == "World"

    def test_output_text_from_nested(self):
        """Test output_text extraction from nested output dict."""
        from insideLLMs.cli import _output_text

        assert _output_text({"output": {"output_text": "Nested"}}) == "Nested"

    def test_output_text_none(self):
        """Test output_text returns None when not found."""
        from insideLLMs.cli import _output_text

        assert _output_text({}) is None


class TestStripVolatileKeys:
    """Test _strip_volatile_keys function."""

    def test_strip_keys_from_dict(self):
        """Test stripping keys from a dict."""
        from insideLLMs.cli import _strip_volatile_keys

        data = {"id": 1, "timestamp": "2026-01-21", "name": "test"}
        result = _strip_volatile_keys(data, {"timestamp"})
        assert result == {"id": 1, "name": "test"}

    def test_strip_keys_from_nested(self):
        """Test stripping keys from nested structure."""
        from insideLLMs.cli import _strip_volatile_keys

        data = {"outer": {"timestamp": "2026-01-21", "value": 42}}
        result = _strip_volatile_keys(data, {"timestamp"})
        assert result == {"outer": {"value": 42}}

    def test_strip_keys_from_list(self):
        """Test stripping keys from list of dicts."""
        from insideLLMs.cli import _strip_volatile_keys

        data = [{"id": 1, "timestamp": "a"}, {"id": 2, "timestamp": "b"}]
        result = _strip_volatile_keys(data, {"timestamp"})
        assert result == [{"id": 1}, {"id": 2}]


class TestTrimText:
    """Test _trim_text function."""

    def test_trim_short_text(self):
        """Test that short text is not trimmed."""
        from insideLLMs.cli import _trim_text

        assert _trim_text("short", limit=200) == "short"

    def test_trim_long_text(self):
        """Test that long text is trimmed with ellipsis."""
        from insideLLMs.cli import _trim_text

        result = _trim_text("a" * 300, limit=200)
        assert len(result) == 203  # 200 + "..."
        assert result.endswith("...")


class TestOutputSummary:
    """Test _output_summary function."""

    def test_output_summary_text(self):
        """Test output summary for text output."""
        from insideLLMs.cli import _output_summary

        record = {"output_text": "Hello world"}
        result = _output_summary(record, None)
        assert result["type"] == "text"
        assert result["preview"] == "Hello world"
        assert result["length"] == 11

    def test_output_summary_structured(self):
        """Test output summary for structured output."""
        from insideLLMs.cli import _output_summary

        record = {"output": {"key": "value"}}
        result = _output_summary(record, None)
        assert result["type"] == "structured"
        assert "fingerprint" in result

    def test_output_summary_none(self):
        """Test output summary when no output."""
        from insideLLMs.cli import _output_summary

        result = _output_summary({}, None)
        assert result is None


class TestPrimaryScore:
    """Test _primary_score function."""

    def test_primary_score_with_metric(self):
        """Test extracting primary score with primary_metric set."""
        from insideLLMs.cli import _primary_score

        record = {"primary_metric": "accuracy", "scores": {"accuracy": 0.95, "f1": 0.9}}
        metric, value = _primary_score(record)
        assert metric == "accuracy"
        assert value == 0.95

    def test_primary_score_default_score(self):
        """Test extracting default 'score' when no primary_metric."""
        from insideLLMs.cli import _primary_score

        record = {"scores": {"score": 0.8}}
        metric, value = _primary_score(record)
        assert metric == "score"
        assert value == 0.8

    def test_primary_score_none(self):
        """Test when no scores available."""
        from insideLLMs.cli import _primary_score

        metric, value = _primary_score({})
        assert metric is None
        assert value is None


class TestMetricMismatchReason:
    """Test _metric_mismatch_reason function."""

    def test_type_mismatch_non_dict(self):
        """Test when scores are not dicts."""
        from insideLLMs.cli import _metric_mismatch_reason

        result = _metric_mismatch_reason({"scores": "not dict"}, {"scores": {}})
        assert result == "type_mismatch"

    def test_missing_scores(self):
        """Test when scores are empty."""
        from insideLLMs.cli import _metric_mismatch_reason

        result = _metric_mismatch_reason({"scores": {}}, {"scores": {}})
        assert result == "missing_scores"

    def test_primary_metric_mismatch(self):
        """Test when primary metrics differ."""
        from insideLLMs.cli import _metric_mismatch_reason

        record_a = {"primary_metric": "accuracy", "scores": {"accuracy": 0.9}}
        record_b = {"primary_metric": "f1", "scores": {"f1": 0.85}}
        result = _metric_mismatch_reason(record_a, record_b)
        assert result == "primary_metric_mismatch"


class TestWriteJsonl:
    """Test _write_jsonl function."""

    def test_write_jsonl_basic(self):
        """Test writing JSONL file."""
        from insideLLMs.cli import _write_jsonl

        records = [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            _write_jsonl(records, output_path)

            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 2
            assert json.loads(lines[0]) == {"id": 1, "name": "test1"}
            assert json.loads(lines[1]) == {"id": 2, "name": "test2"}

    def test_write_jsonl_strict_rejects_non_serializable(self):
        """Strict mode should reject non-deterministic values."""
        from insideLLMs.cli import _write_jsonl

        class CustomObject:
            pass

        records = [{"bad": CustomObject()}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            with pytest.raises(ValueError, match="strict_serialization"):
                _write_jsonl(records, output_path, strict_serialization=True)


class TestCmdCompareJson:
    """Ensure compare JSON output is pipe-friendly."""

    def test_compare_json_stdout_is_clean(self, capsys):
        from insideLLMs.cli import main

        rc = main(["compare", "--models", "dummy", "--input", "Hello", "--format", "json"])
        assert rc == 0

        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        assert isinstance(payload, list)
        assert payload[0]["input"] == "Hello"
        assert payload[0]["models"][0]["model"] == "dummy"

    def test_compare_json_quiet_suppresses_status_output(self, capsys):
        from insideLLMs.cli import main

        rc = main(
            ["compare", "--models", "dummy", "--input", "Hello", "--format", "json", "--quiet"]
        )
        assert rc == 0

        captured = capsys.readouterr()
        json.loads(captured.out)
        assert captured.err == ""


class TestSpinnerMethods:
    """Test additional Spinner methods."""

    def test_spinner_spin(self, capsys):
        """Test spinner spin method."""
        from insideLLMs.cli import Spinner

        spinner = Spinner(message="Processing")
        spinner.spin()
        captured = capsys.readouterr()
        assert "Processing" in captured.out

    def test_spinner_stop_success(self, capsys):
        """Test spinner stop with success."""
        from insideLLMs.cli import Spinner

        spinner = Spinner(message="Processing")
        spinner.stop(success=True)
        captured = capsys.readouterr()
        assert "done" in captured.out

    def test_spinner_stop_failure(self, capsys):
        """Test spinner stop with failure."""
        from insideLLMs.cli import Spinner

        spinner = Spinner(message="Processing")
        spinner.stop(success=False)
        captured = capsys.readouterr()
        assert "failed" in captured.out


class TestProgressBarRender:
    """Test ProgressBar render methods."""

    def test_progress_bar_render(self, capsys):
        """Test progress bar rendering."""
        from insideLLMs.cli import ProgressBar

        bar = ProgressBar(total=10, show_eta=False)
        bar.update(5)
        captured = capsys.readouterr()
        assert "50.0%" in captured.out
        assert "5/10" in captured.out

    def test_progress_bar_zero_total(self, capsys):
        """Test progress bar with zero total."""
        from insideLLMs.cli import ProgressBar

        bar = ProgressBar(total=0, show_eta=False)
        bar.update(0)
        captured = capsys.readouterr()
        assert "100" in captured.out  # Should show 100%
