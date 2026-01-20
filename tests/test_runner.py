"""Tests for the runner module."""

import json
import tempfile
from pathlib import Path

import pytest


class TestProbeRunner:
    """Test the ProbeRunner class."""

    def test_create_runner(self):
        """Test creating a ProbeRunner."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        assert runner is not None
        assert runner.model == model
        assert runner.probe == probe

    def test_run_single_item(self):
        """Test running with a single input."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        result = runner.run(["Is the sky blue?"])
        assert len(result) == 1
        assert "input" in result[0]
        assert "output" in result[0] or "error" in result[0]

    def test_run_multiple(self):
        """Test running multiple probes."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        results = runner.run(["Question 1?", "Question 2?", "Question 3?"])
        assert len(results) == 3

    def test_run_result_structure(self):
        """Test that results have the correct structure."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        results = runner.run(["Test?"])
        assert len(results) == 1
        result = results[0]

        # Should have input
        assert "input" in result
        # Should have status
        assert "status" in result
        # Should have latency
        assert "latency_ms" in result


class TestRunExperimentFromConfig:
    """Test the run_experiment_from_config function."""

    def test_run_from_yaml_config(self):
        """Test running from YAML config."""
        import yaml

        from insideLLMs.runner import run_experiment_from_config

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data file
            data_path = Path(tmpdir) / "data.jsonl"
            with open(data_path, "w") as f:
                f.write('{"question": "Is 2+2=4?"}\n')
                f.write('{"question": "Is the sky green?"}\n')

            # Create config file
            config_path = Path(tmpdir) / "config.yaml"
            config = {
                "model": {"type": "dummy"},
                "probe": {"type": "logic"},
                "dataset": {
                    "path": str(data_path),
                    "format": "jsonl",
                    "input_field": "question",
                },
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            result = run_experiment_from_config(config_path)
            assert result is not None

    def test_run_from_json_config(self):
        """Test running from JSON config."""
        from insideLLMs.runner import run_experiment_from_config

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data file
            data_path = Path(tmpdir) / "data.jsonl"
            with open(data_path, "w") as f:
                f.write('{"question": "Test?"}\n')

            # Create config file
            config_path = Path(tmpdir) / "config.json"
            config = {
                "model": {"type": "dummy"},
                "probe": {"type": "logic"},
                "dataset": {
                    "path": str(data_path),
                    "format": "jsonl",
                    "input_field": "question",
                },
            }
            with open(config_path, "w") as f:
                json.dump(config, f)

            result = run_experiment_from_config(config_path)
            assert result is not None


class TestCreateExperimentResult:
    """Test the create_experiment_result function."""

    def test_create_result(self):
        """Test creating an experiment result."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import create_experiment_result

        model = DummyModel()
        probe = LogicProbe()

        results = [
            {
                "input": "test1",
                "output": "result1",
                "status": "success",
                "latency_ms": 100,
            },
            {
                "input": "test2",
                "output": "result2",
                "status": "success",
                "latency_ms": 150,
            },
        ]

        experiment = create_experiment_result(model, probe, results)

        assert experiment is not None
        assert experiment.model_info.name == model.name
        assert experiment.probe_name == probe.name
        assert len(experiment.results) == 2

    def test_create_result_with_errors(self):
        """Test creating an experiment result with errors."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import create_experiment_result
        from insideLLMs.types import ResultStatus

        model = DummyModel()
        probe = LogicProbe()

        results = [
            {"input": "test1", "output": "result1", "status": "success", "latency_ms": 100},
            {"input": "test2", "error": "Failed", "status": "error", "latency_ms": 50},
        ]

        experiment = create_experiment_result(model, probe, results)

        assert experiment is not None
        assert len(experiment.results) == 2
        # One success, one error
        success_count = sum(1 for r in experiment.results if r.status == ResultStatus.SUCCESS)
        assert success_count == 1


class TestLoadConfig:
    """Test config loading functions."""

    def test_load_yaml_config(self):
        """Test loading YAML config."""
        import yaml

        from insideLLMs.runner import load_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {
                "model": {"type": "dummy"},
                "probe": {"type": "logic"},
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            loaded = load_config(config_path)
            assert loaded["model"]["type"] == "dummy"
            assert loaded["probe"]["type"] == "logic"

    def test_load_json_config(self):
        """Test loading JSON config."""
        from insideLLMs.runner import load_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config = {
                "model": {"type": "dummy"},
                "probe": {"type": "logic"},
            }
            with open(config_path, "w") as f:
                json.dump(config, f)

            loaded = load_config(config_path)
            assert loaded["model"]["type"] == "dummy"
            assert loaded["probe"]["type"] == "logic"


class TestProgressCallback:
    """Test progress callback functionality."""

    def test_progress_callback_called(self):
        """Test that progress callback is called."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        progress_values = []

        def callback(current, total):
            progress_values.append((current, total))

        runner.run(
            ["Q1?", "Q2?", "Q3?"],
            progress_callback=callback,
        )

        # Callback should have been called
        assert len(progress_values) > 0
        # Final callback should be (3, 3)
        assert progress_values[-1] == (3, 3)


class TestAsyncRunner:
    """Test async runner functionality."""

    @pytest.mark.asyncio
    async def test_async_run(self):
        """Test async experiment execution."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import AsyncProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = AsyncProbeRunner(model, probe)

        results = await runner.run(["Test?", "Another?"])
        assert len(results) == 2


class TestRunProbe:
    """Test the run_probe convenience function."""

    def test_run_probe_basic(self):
        """Test basic run_probe usage."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import run_probe

        model = DummyModel()
        probe = LogicProbe()

        results = run_probe(model, probe, ["Test question?"])
        assert len(results) == 1
        assert "input" in results[0]

    def test_run_probe_multiple(self):
        """Test run_probe with multiple inputs."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import run_probe

        model = DummyModel()
        probe = LogicProbe()

        results = run_probe(model, probe, ["Q1?", "Q2?", "Q3?"])
        assert len(results) == 3


class TestEdgeCases:
    """Test edge cases in the runner."""

    def test_empty_inputs(self):
        """Test running with empty inputs."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner
        from insideLLMs.validation import ValidationError

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        # Empty prompt sets should now raise ValidationError
        with pytest.raises(ValidationError, match="cannot be empty"):
            runner.run([])

    def test_unicode_inputs(self):
        """Test running with Unicode inputs."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        results = runner.run(["日本語の質問?", "Вопрос на русском?"])
        assert len(results) == 2

    def test_very_long_input(self):
        """Test running with very long input."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        long_input = "a" * 10000 + "?"
        results = runner.run([long_input])
        assert len(results) == 1


class TestAsyncProbeRunner:
    """Test the AsyncProbeRunner class."""

    def test_create_async_runner(self):
        """Test creating an AsyncProbeRunner."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import AsyncProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = AsyncProbeRunner(model, probe)

        assert runner is not None
        assert runner.model == model
        assert runner.probe == probe

    @pytest.mark.asyncio
    async def test_async_runner_results(self):
        """Test async runner returns proper results."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import AsyncProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = AsyncProbeRunner(model, probe)

        results = await runner.run(["Test?"])
        assert len(results) == 1
        assert "input" in results[0]
        assert "status" in results[0]


class TestResourceCleanup:
    """Test that file handles are properly cleaned up, even on exceptions."""

    def test_proberunner_closes_file_on_success(self, tmp_path):
        """Test that ProbeRunner closes file handles after successful run."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        run_dir = tmp_path / "test_run"
        results = runner.run(
            ["Test?"],
            emit_run_artifacts=True,
            run_dir=run_dir,
        )

        # Verify records file exists and is closed (can be read)
        records_path = run_dir / "records.jsonl"
        assert records_path.exists()
        content = records_path.read_text()
        assert len(content) > 0

    def test_proberunner_closes_file_on_exception(self, tmp_path):
        """Test that ProbeRunner closes file handles even when probe raises."""
        from insideLLMs.models.base import Model
        from insideLLMs.probes.base import Probe
        from insideLLMs.runner import ProbeRunner

        class FailingModel(Model):
            """Model that always fails."""

            def __init__(self):
                super().__init__(name="failing-model")

            def generate(self, prompt, **kwargs):
                raise RuntimeError("Intentional failure")

        class SimpleProbe(Probe):
            """Simple probe that just calls generate."""

            def __init__(self):
                super().__init__(name="simple-probe")

            def run(self, model, item, **kwargs):
                return model.generate(item)

        model = FailingModel()
        probe = SimpleProbe()
        runner = ProbeRunner(model, probe)

        run_dir = tmp_path / "test_run_fail"
        # Should not raise - errors are captured
        results = runner.run(
            ["Test?"],
            emit_run_artifacts=True,
            run_dir=run_dir,
            stop_on_error=False,
        )

        # Verify records file exists and is closed (can be read)
        records_path = run_dir / "records.jsonl"
        assert records_path.exists()
        content = records_path.read_text()
        assert "error" in content.lower()

    @pytest.mark.asyncio
    async def test_asyncrunner_closes_file_on_success(self, tmp_path):
        """Test that AsyncProbeRunner closes file handles after successful run."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import AsyncProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = AsyncProbeRunner(model, probe)

        run_dir = tmp_path / "test_async_run"
        results = await runner.run(
            ["Test?"],
            emit_run_artifacts=True,
            run_dir=run_dir,
        )

        # Verify records file exists and is closed (can be read)
        records_path = run_dir / "records.jsonl"
        assert records_path.exists()
        content = records_path.read_text()
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_asyncrunner_closes_file_on_exception(self, tmp_path):
        """Test that AsyncProbeRunner closes file handles even when probe raises."""
        from insideLLMs.models.base import Model
        from insideLLMs.probes.base import Probe
        from insideLLMs.runner import AsyncProbeRunner

        class FailingModel(Model):
            """Model that always fails."""

            def __init__(self):
                super().__init__(name="failing-model")

            def generate(self, prompt, **kwargs):
                raise RuntimeError("Intentional failure")

        class SimpleProbe(Probe):
            """Simple probe that just calls generate."""

            def __init__(self):
                super().__init__(name="simple-probe")

            def run(self, model, item, **kwargs):
                return model.generate(item)

        model = FailingModel()
        probe = SimpleProbe()
        runner = AsyncProbeRunner(model, probe)

        run_dir = tmp_path / "test_async_run_fail"
        # Should not raise - errors are captured
        results = await runner.run(
            ["Test?"],
            emit_run_artifacts=True,
            run_dir=run_dir,
        )

        # Verify records file exists and is closed (can be read)
        records_path = run_dir / "records.jsonl"
        assert records_path.exists()
        content = records_path.read_text()
        assert "error" in content.lower()


class TestRunConfig:
    """Test the RunConfig dataclass."""

    def test_runconfig_defaults(self):
        """Test RunConfig has sensible defaults."""
        from insideLLMs.config_types import RunConfig

        config = RunConfig()
        assert config.stop_on_error is False
        assert config.validate_output is False
        assert config.emit_run_artifacts is True
        assert config.concurrency == 5
        assert config.store_messages is True
        assert config.validation_mode == "strict"

    def test_runconfig_custom_values(self):
        """Test RunConfig accepts custom values."""
        from insideLLMs.config_types import RunConfig

        config = RunConfig(
            stop_on_error=True,
            validate_output=True,
            concurrency=10,
            emit_run_artifacts=False,
        )
        assert config.stop_on_error is True
        assert config.validate_output is True
        assert config.concurrency == 10
        assert config.emit_run_artifacts is False

    def test_runconfig_validation_mode_invalid(self):
        """Test RunConfig rejects invalid validation_mode."""
        from insideLLMs.config_types import RunConfig

        with pytest.raises(ValueError, match="validation_mode"):
            RunConfig(validation_mode="invalid")

    def test_runconfig_concurrency_invalid(self):
        """Test RunConfig rejects invalid concurrency."""
        from insideLLMs.config_types import RunConfig

        with pytest.raises(ValueError, match="concurrency"):
            RunConfig(concurrency=0)

    def test_runconfig_from_kwargs(self):
        """Test RunConfig.from_kwargs() creates config from kwargs."""
        from insideLLMs.config_types import RunConfig

        config = RunConfig.from_kwargs(
            stop_on_error=True,
            concurrency=20,
        )
        assert config.stop_on_error is True
        assert config.concurrency == 20

    def test_runconfig_from_kwargs_ignores_unknown(self):
        """Test RunConfig.from_kwargs() ignores unknown kwargs with warning."""
        from insideLLMs.config_types import RunConfig

        with pytest.warns(UserWarning, match="Unknown RunConfig fields"):
            config = RunConfig.from_kwargs(
                stop_on_error=True,
                unknown_field="ignored",
            )
        assert config.stop_on_error is True

    def test_proberunner_with_runconfig(self, tmp_path):
        """Test ProbeRunner.run() accepts RunConfig."""
        from insideLLMs.config_types import RunConfig
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        config = RunConfig(
            emit_run_artifacts=True,
            run_dir=tmp_path / "config_run",
        )

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        results = runner.run(["Test?"], config=config)
        assert len(results) == 1
        assert (tmp_path / "config_run" / "records.jsonl").exists()

    def test_proberunner_kwargs_override_config(self, tmp_path):
        """Test that explicit kwargs override RunConfig values."""
        from insideLLMs.config_types import RunConfig
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        config = RunConfig(
            emit_run_artifacts=False,  # Config says no artifacts
        )

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        # But we override with explicit kwarg
        results = runner.run(
            ["Test?"],
            config=config,
            emit_run_artifacts=True,  # Override
            run_dir=tmp_path / "override_run",
        )
        assert len(results) == 1
        # Artifacts should exist because kwarg overrode config
        assert (tmp_path / "override_run" / "records.jsonl").exists()

    @pytest.mark.asyncio
    async def test_asyncrunner_with_runconfig(self, tmp_path):
        """Test AsyncProbeRunner.run() accepts RunConfig."""
        from insideLLMs.config_types import RunConfig
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import AsyncProbeRunner

        config = RunConfig(
            emit_run_artifacts=True,
            run_dir=tmp_path / "async_config_run",
            concurrency=2,
        )

        model = DummyModel()
        probe = LogicProbe()
        runner = AsyncProbeRunner(model, probe)

        results = await runner.run(["Test?"], config=config)
        assert len(results) == 1
        assert (tmp_path / "async_config_run" / "records.jsonl").exists()


class TestRunConfigBuilder:
    """Test RunConfigBuilder fluent API."""

    def test_builder_defaults(self):
        """Test builder creates config with defaults."""
        from insideLLMs.config_types import RunConfigBuilder

        config = RunConfigBuilder().build()
        assert config.stop_on_error is False
        assert config.validate_output is False
        assert config.emit_run_artifacts is True
        assert config.concurrency == 5

    def test_builder_with_validation(self):
        """Test builder with_validation method."""
        from insideLLMs.config_types import RunConfigBuilder

        config = (
            RunConfigBuilder()
            .with_validation(enabled=True, schema_version="1.0.0", mode="lenient")
            .build()
        )
        assert config.validate_output is True
        assert config.schema_version == "1.0.0"
        assert config.validation_mode == "lenient"

    def test_builder_with_artifacts(self, tmp_path):
        """Test builder with_artifacts method."""
        from insideLLMs.config_types import RunConfigBuilder

        config = (
            RunConfigBuilder()
            .with_artifacts(
                enabled=True,
                run_root=tmp_path,
                run_id="test-run",
                overwrite=True,
            )
            .build()
        )
        assert config.emit_run_artifacts is True
        assert config.run_root == tmp_path
        assert config.run_id == "test-run"
        assert config.overwrite is True

    def test_builder_with_concurrency(self):
        """Test builder with_concurrency method."""
        from insideLLMs.config_types import RunConfigBuilder

        config = RunConfigBuilder().with_concurrency(20).build()
        assert config.concurrency == 20

    def test_builder_with_error_handling(self):
        """Test builder with_error_handling method."""
        from insideLLMs.config_types import RunConfigBuilder

        config = RunConfigBuilder().with_error_handling(stop_on_error=True).build()
        assert config.stop_on_error is True

    def test_builder_with_dataset_info(self):
        """Test builder with_dataset_info method."""
        from insideLLMs.config_types import RunConfigBuilder

        info = {"name": "test-dataset", "version": "1.0"}
        config = RunConfigBuilder().with_dataset_info(info).build()
        assert config.dataset_info == info

    def test_builder_chaining(self, tmp_path):
        """Test builder method chaining."""
        from insideLLMs.config_types import RunConfigBuilder

        config = (
            RunConfigBuilder()
            .with_validation(enabled=True)
            .with_artifacts(run_root=tmp_path, run_id="chained-run")
            .with_concurrency(15)
            .with_error_handling(stop_on_error=True)
            .with_message_storage(enabled=False)
            .build()
        )
        assert config.validate_output is True
        assert config.run_root == tmp_path
        assert config.run_id == "chained-run"
        assert config.concurrency == 15
        assert config.stop_on_error is True
        assert config.store_messages is False

    def test_builder_validation_error(self):
        """Test builder validates on build."""
        from insideLLMs.config_types import RunConfigBuilder

        with pytest.raises(ValueError, match="concurrency"):
            RunConfigBuilder().with_concurrency(0).build()


class TestValidation:
    """Test input validation functionality."""

    def test_validate_prompt_valid(self):
        """Test validate_prompt with valid input."""
        from insideLLMs.validation import validate_prompt

        assert validate_prompt("Hello, world!") is True
        assert validate_prompt("A" * 1000) is True

    def test_validate_prompt_none(self):
        """Test validate_prompt rejects None."""
        from insideLLMs.validation import ValidationError, validate_prompt

        with pytest.raises(ValidationError, match="cannot be None"):
            validate_prompt(None)

    def test_validate_prompt_empty(self):
        """Test validate_prompt rejects empty strings by default."""
        from insideLLMs.validation import ValidationError, validate_prompt

        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_prompt("")

    def test_validate_prompt_empty_allowed(self):
        """Test validate_prompt allows empty when configured."""
        from insideLLMs.validation import validate_prompt

        assert validate_prompt("", allow_empty=True) is True

    def test_validate_prompt_wrong_type(self):
        """Test validate_prompt rejects non-string types."""
        from insideLLMs.validation import ValidationError, validate_prompt

        with pytest.raises(ValidationError, match="must be a string"):
            validate_prompt(123)

        with pytest.raises(ValidationError, match="must be a string"):
            validate_prompt(["list"])

    def test_validate_prompt_max_length(self):
        """Test validate_prompt enforces max_length."""
        from insideLLMs.validation import ValidationError, validate_prompt

        assert validate_prompt("short", max_length=100) is True

        with pytest.raises(ValidationError, match="too long"):
            validate_prompt("A" * 101, max_length=100)

    def test_validate_prompt_set_valid(self):
        """Test validate_prompt_set with valid input."""
        from insideLLMs.validation import validate_prompt_set

        assert validate_prompt_set(["Q1", "Q2"]) is True
        assert validate_prompt_set(["Single"]) is True
        # Dict items are allowed (structured prompts)
        assert validate_prompt_set([{"text": "Q1"}]) is True

    def test_validate_prompt_set_none(self):
        """Test validate_prompt_set rejects None."""
        from insideLLMs.validation import ValidationError, validate_prompt_set

        with pytest.raises(ValidationError, match="cannot be None"):
            validate_prompt_set(None)

    def test_validate_prompt_set_empty(self):
        """Test validate_prompt_set rejects empty sets by default."""
        from insideLLMs.validation import ValidationError, validate_prompt_set

        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_prompt_set([])

    def test_validate_prompt_set_wrong_type(self):
        """Test validate_prompt_set rejects non-list types."""
        from insideLLMs.validation import ValidationError, validate_prompt_set

        with pytest.raises(ValidationError, match="must be a list"):
            validate_prompt_set("not a list")

    def test_validate_positive_int(self):
        """Test validate_positive_int."""
        from insideLLMs.validation import ValidationError, validate_positive_int

        assert validate_positive_int(1) == 1
        assert validate_positive_int(100) == 100

        with pytest.raises(ValidationError, match="must be >= 1"):
            validate_positive_int(0)

        with pytest.raises(ValidationError, match="must be an integer"):
            validate_positive_int("5")

    def test_validate_choice(self):
        """Test validate_choice."""
        from insideLLMs.validation import ValidationError, validate_choice

        assert validate_choice("a", ["a", "b", "c"]) == "a"

        with pytest.raises(ValidationError, match="must be one of"):
            validate_choice("d", ["a", "b", "c"])

    def test_validation_error_message_format(self):
        """Test ValidationError message formatting."""
        from insideLLMs.validation import ValidationError

        error = ValidationError(
            "Test error",
            field="test_field",
            value="bad_value",
            suggestions=["Try this", "Or that"],
        )
        msg = str(error)
        assert "[test_field]" in msg
        assert "Test error" in msg
        assert "Suggestions:" in msg

    def test_proberunner_validates_prompt_set(self, tmp_path):
        """Test ProbeRunner validates prompt set before execution."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner
        from insideLLMs.validation import ValidationError

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        # None should fail
        with pytest.raises(ValidationError, match="cannot be None"):
            runner.run(None)

        # Non-list should fail
        with pytest.raises(ValidationError, match="must be a list"):
            runner.run("not a list")

    @pytest.mark.asyncio
    async def test_asyncrunner_validates_prompt_set(self, tmp_path):
        """Test AsyncProbeRunner validates prompt set before execution."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import AsyncProbeRunner
        from insideLLMs.validation import ValidationError

        model = DummyModel()
        probe = LogicProbe()
        runner = AsyncProbeRunner(model, probe)

        # None should fail
        with pytest.raises(ValidationError, match="cannot be None"):
            await runner.run(None)

        # Empty should fail
        with pytest.raises(ValidationError, match="cannot be empty"):
            await runner.run([])


class TestRunnerExecutionError:
    """Test enhanced error context in runner execution."""

    def test_runner_execution_error_attributes(self):
        """Test RunnerExecutionError has all expected attributes."""
        from insideLLMs.exceptions import RunnerExecutionError

        error = RunnerExecutionError(
            reason="Test failure",
            model_id="test-model",
            probe_id="test-probe",
            prompt="What is 2+2?",
            prompt_index=5,
            run_id="run-123",
            elapsed_seconds=1.5,
            original_error=ValueError("Original error"),
            suggestions=["Try this", "Try that"],
        )

        assert error.model_id == "test-model"
        assert error.probe_id == "test-probe"
        assert error.prompt == "What is 2+2?"
        assert error.prompt_index == 5
        assert error.run_id == "run-123"
        assert error.elapsed_seconds == 1.5
        assert isinstance(error.original_error, ValueError)
        assert len(error.suggestions) == 2

    def test_runner_execution_error_str_format(self):
        """Test RunnerExecutionError string formatting."""
        from insideLLMs.exceptions import RunnerExecutionError

        error = RunnerExecutionError(
            reason="API timeout",
            model_id="gpt-4",
            probe_id="LogicProbe",
            prompt="Test prompt",
            prompt_index=3,
            original_error=TimeoutError("Connection timed out"),
            suggestions=["Check network", "Retry later"],
        )

        error_str = str(error)
        assert "API timeout" in error_str
        assert "model=gpt-4" in error_str
        assert "probe=LogicProbe" in error_str
        assert "index=3" in error_str
        assert "Test prompt" in error_str
        assert "TimeoutError" in error_str
        assert "Check network" in error_str

    def test_stop_on_error_raises_runner_execution_error(self, tmp_path):
        """Test that stop_on_error=True raises RunnerExecutionError."""
        from insideLLMs.exceptions import RunnerExecutionError
        from insideLLMs.models.base import Model
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        class FailingModel(Model):
            def __init__(self):
                super().__init__(name="failing-model")

            def generate(self, prompt: str, **kwargs) -> str:
                raise ValueError("Simulated API failure")

            def info(self):
                return {"model_id": "failing-model"}

        model = FailingModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        with pytest.raises(RunnerExecutionError) as exc_info:
            runner.run(
                ["Test prompt"],
                stop_on_error=True,
                emit_run_artifacts=False,
            )

        error = exc_info.value
        assert error.model_id == "failing-model"
        assert error.prompt_index == 0
        assert "Simulated API failure" in str(error)
        assert isinstance(error.original_error, ValueError)

    def test_stop_on_error_false_does_not_raise(self, tmp_path):
        """Test that stop_on_error=False collects errors without raising."""
        from insideLLMs.models.base import Model
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        class FailingModel(Model):
            def __init__(self):
                super().__init__(name="failing-model")

            def generate(self, prompt: str, **kwargs) -> str:
                raise ValueError("Simulated failure")

            def info(self):
                return {"model_id": "failing-model"}

        model = FailingModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        # Should not raise, just collect errors
        results = runner.run(
            ["Test 1", "Test 2"],
            stop_on_error=False,
            emit_run_artifacts=False,
        )

        assert len(results) == 2
        assert all(r["status"] == "error" for r in results)
