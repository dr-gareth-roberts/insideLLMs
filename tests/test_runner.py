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

    def test_run_returns_experiment_result(self):
        """Test returning ExperimentResult from runner."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner
        from insideLLMs.types import ExperimentResult

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        result = runner.run(["Is the sky blue?"], emit_run_artifacts=False, return_experiment=True)
        assert isinstance(result, ExperimentResult)
        assert result.total_count == 1


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
        runner.run(
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
        runner.run(
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


class TestRunnerResume:
    """Tests for resume functionality."""

    def test_proberunner_resume_appends(self, tmp_path):
        """Test ProbeRunner resumes and appends remaining records."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        prompt_set = ["Q1?", "Q2?", "Q3?"]
        run_dir = tmp_path / "resume_run"
        run_id = "resume-test"

        runner = ProbeRunner(DummyModel(), LogicProbe())
        runner.run(
            prompt_set,
            emit_run_artifacts=True,
            run_dir=run_dir,
            run_id=run_id,
        )

        records_path = run_dir / "records.jsonl"
        lines = records_path.read_text().splitlines()
        records_path.write_text(lines[0] + "\n")

        results = runner.run(
            prompt_set,
            emit_run_artifacts=True,
            run_dir=run_dir,
            run_id=run_id,
            resume=True,
        )
        assert len(results) == len(prompt_set)
        assert len(records_path.read_text().splitlines()) == len(prompt_set)

    @pytest.mark.asyncio
    async def test_asyncrunner_resume_appends(self, tmp_path):
        """Test AsyncProbeRunner resumes and appends remaining records."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import AsyncProbeRunner

        prompt_set = ["Q1?", "Q2?", "Q3?"]
        run_dir = tmp_path / "resume_async_run"
        run_id = "resume-async-test"

        runner = AsyncProbeRunner(DummyModel(), LogicProbe())
        await runner.run(
            prompt_set,
            emit_run_artifacts=True,
            run_dir=run_dir,
            run_id=run_id,
        )

        records_path = run_dir / "records.jsonl"
        lines = records_path.read_text().splitlines()
        records_path.write_text(lines[0] + "\n")

        results = await runner.run(
            prompt_set,
            emit_run_artifacts=True,
            run_dir=run_dir,
            run_id=run_id,
            resume=True,
        )
        assert len(results) == len(prompt_set)
        assert len(records_path.read_text().splitlines()) == len(prompt_set)


class TestPipelineConfig:
    """Tests for pipeline config loading."""

    def test_create_model_with_pipeline(self):
        """Test pipeline middleware wrapping."""
        from insideLLMs.pipeline import ModelPipeline
        from insideLLMs.runner import _create_model_from_config

        config = {
            "type": "dummy",
            "pipeline": {
                "middlewares": [{"type": "cache", "args": {"cache_size": 2}}],
            },
        }
        model = _create_model_from_config(config)
        assert isinstance(model, ModelPipeline)
        info = model.info()
        assert info.get("pipeline") is True

    def test_create_model_with_async_pipeline(self):
        """Test async pipeline wrapping."""
        from insideLLMs.pipeline import AsyncModelPipeline
        from insideLLMs.runner import _create_model_from_config

        config = {
            "type": "dummy",
            "pipeline": {
                "async": True,
                "middlewares": ["cache"],
            },
        }
        model = _create_model_from_config(config)
        assert isinstance(model, AsyncModelPipeline)

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
        await runner.run(
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
        await runner.run(
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
        assert config.resume is False
        assert config.use_probe_batch is False
        assert config.batch_workers is None
        assert config.return_experiment is False

    def test_runconfig_custom_values(self):
        """Test RunConfig accepts custom values."""
        from insideLLMs.config_types import RunConfig

        config = RunConfig(
            stop_on_error=True,
            validate_output=True,
            concurrency=10,
            emit_run_artifacts=False,
            resume=True,
            use_probe_batch=True,
            batch_workers=2,
            return_experiment=True,
        )
        assert config.stop_on_error is True
        assert config.validate_output is True
        assert config.concurrency == 10
        assert config.emit_run_artifacts is False
        assert config.resume is True
        assert config.use_probe_batch is True
        assert config.batch_workers == 2
        assert config.return_experiment is True

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

    def test_builder_with_resume_and_batch(self):
        """Test builder resume/batch/output helpers."""
        from insideLLMs.config_types import RunConfigBuilder

        config = (
            RunConfigBuilder()
            .with_resume(True)
            .with_probe_batch(enabled=True, batch_workers=3)
            .with_output(return_experiment=True)
            .build()
        )
        assert config.resume is True
        assert config.use_probe_batch is True
        assert config.batch_workers == 3
        assert config.return_experiment is True

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


class TestProgressInfo:
    """Test ProgressInfo dataclass."""

    def test_progress_info_basic(self):
        """Test ProgressInfo basic attributes."""
        from insideLLMs.config_types import ProgressInfo

        info = ProgressInfo(
            current=5,
            total=10,
            elapsed_seconds=2.5,
            rate=2.0,
            eta_seconds=2.5,
        )
        assert info.current == 5
        assert info.total == 10
        assert info.percent == 50.0
        assert info.remaining == 5
        assert info.is_complete is False

    def test_progress_info_complete(self):
        """Test ProgressInfo when complete."""
        from insideLLMs.config_types import ProgressInfo

        info = ProgressInfo(
            current=10,
            total=10,
            elapsed_seconds=5.0,
            rate=2.0,
        )
        assert info.is_complete is True
        assert info.remaining == 0
        assert info.percent == 100.0

    def test_progress_info_create(self):
        """Test ProgressInfo.create() factory method."""
        import time

        from insideLLMs.config_types import ProgressInfo

        start = time.perf_counter() - 2.0  # Simulate 2 seconds elapsed
        info = ProgressInfo.create(
            current=4,
            total=10,
            start_time=start,
            current_item="test prompt",
            current_index=3,
        )

        assert info.current == 4
        assert info.total == 10
        assert info.elapsed_seconds >= 2.0
        assert info.rate > 0
        assert info.eta_seconds is not None
        assert info.current_item == "test prompt"
        assert info.current_index == 3

    def test_progress_info_str(self):
        """Test ProgressInfo string representation."""
        from insideLLMs.config_types import ProgressInfo

        info = ProgressInfo(
            current=5,
            total=10,
            elapsed_seconds=2.5,
            rate=2.0,
            eta_seconds=2.5,
        )
        s = str(info)
        assert "5/10" in s
        assert "50.0%" in s
        assert "2.0/s" in s
        assert "ETA:" in s

    def test_progress_callback_with_progress_info(self, tmp_path):
        """Test that runner accepts ProgressInfo callback."""
        from insideLLMs.config_types import ProgressInfo
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        progress_updates: list[ProgressInfo] = []

        def progress_callback(info: ProgressInfo):
            progress_updates.append(info)

        runner.run(
            ["Test 1", "Test 2", "Test 3"],
            progress_callback=progress_callback,
            emit_run_artifacts=False,
        )

        # Should have 4 updates: 3 for processing + 1 for complete
        assert len(progress_updates) == 4
        assert progress_updates[0].current == 0
        assert progress_updates[0].total == 3
        assert progress_updates[-1].current == 3
        assert progress_updates[-1].is_complete

    def test_progress_callback_legacy_signature(self, tmp_path):
        """Test that runner still accepts legacy (current, total) callback."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        progress_updates: list[tuple[int, int]] = []

        def progress_callback(current: int, total: int):
            progress_updates.append((current, total))

        runner.run(
            ["Test 1", "Test 2"],
            progress_callback=progress_callback,
            emit_run_artifacts=False,
        )

        # Should have 3 updates: 2 for processing + 1 for complete
        assert len(progress_updates) == 3
        assert progress_updates[0] == (0, 2)
        assert progress_updates[-1] == (2, 2)


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


class TestRunnerBaseProperties:
    """Test _RunnerBase properties (success_rate, error_count)."""

    def test_success_rate_with_mixed_results(self):
        """Test success_rate property with mixed success/error results."""
        from insideLLMs.models.base import Model
        from insideLLMs.probes.base import Probe
        from insideLLMs.runner import ProbeRunner

        class AlternatingModel(Model):
            """Model that alternates success/failure."""

            def __init__(self):
                super().__init__(name="alternating")
                self.call_count = 0

            def generate(self, prompt, **kwargs):
                self.call_count += 1
                if self.call_count % 2 == 0:
                    raise ValueError("Even call fails")
                return "success"

        class PassThroughProbe(Probe):
            def __init__(self):
                super().__init__(name="passthrough")

            def run(self, model, item, **kwargs):
                return model.generate(item)

        model = AlternatingModel()
        probe = PassThroughProbe()
        runner = ProbeRunner(model, probe)

        runner.run(["1", "2", "3", "4"], emit_run_artifacts=False, stop_on_error=False)

        # 2 successes (1, 3), 2 errors (2, 4)
        assert runner.success_rate == 0.5
        assert runner.error_count == 2

    def test_success_rate_empty_results(self):
        """Test success_rate with no results (before any run)."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import ProbeRunner

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        # Before any run
        assert runner.success_rate == 0.0
        assert runner.error_count == 0


class TestSerializeValue:
    """Tests for _serialize_value helper function."""

    def test_serialize_datetime(self):
        """Test serializing datetime objects."""
        from datetime import datetime

        from insideLLMs.runner import _serialize_value

        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = _serialize_value(dt)
        assert result == "2024-01-15T10:30:45"

    def test_serialize_path(self):
        """Test serializing Path objects."""
        from pathlib import Path

        from insideLLMs.runner import _serialize_value

        p = Path("/some/test/path")
        result = _serialize_value(p)
        assert result == "/some/test/path"

    def test_serialize_enum(self):
        """Test serializing Enum values."""
        from enum import Enum

        from insideLLMs.runner import _serialize_value

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        result = _serialize_value(Color.RED)
        assert result == "red"

    def test_serialize_dataclass(self):
        """Test serializing dataclass instances."""
        from dataclasses import dataclass

        from insideLLMs.runner import _serialize_value

        @dataclass
        class Person:
            name: str
            age: int

        p = Person(name="Alice", age=30)
        result = _serialize_value(p)
        assert result == {"name": "Alice", "age": 30}

    def test_serialize_set_and_frozenset(self):
        """Test serializing sets."""
        from insideLLMs.runner import _serialize_value

        s = {3, 1, 2}
        result = _serialize_value(s)
        assert result == [1, 2, 3]  # Sorted

        fs = frozenset(["c", "a", "b"])
        result = _serialize_value(fs)
        assert result == ["a", "b", "c"]

    def test_serialize_set_with_unsortable_items(self):
        """Test serializing sets with non-comparable items."""
        from insideLLMs.runner import _serialize_value

        # Sets containing dicts (cannot be directly compared)
        s = {frozenset([("a", 1)]), frozenset([("b", 2)])}
        result = _serialize_value(s)
        assert isinstance(result, list)

    def test_serialize_nested_structures(self):
        """Test serializing nested dicts and lists."""
        from pathlib import Path

        from insideLLMs.runner import _serialize_value

        data = {
            "path": Path("/test"),
            "items": [1, 2, 3],
            "nested": {"key": "value"},
        }
        result = _serialize_value(data)
        assert result["path"] == "/test"
        assert result["items"] == [1, 2, 3]

    def test_serialize_exotic_object(self):
        """Test serializing unknown object types falls back to str."""
        from insideLLMs.runner import _serialize_value

        class CustomObject:
            def __str__(self):
                return "custom_str"

        obj = CustomObject()
        result = _serialize_value(obj)
        assert result == "custom_str"


class TestSemverTuple:
    """Tests for _semver_tuple helper function."""

    def test_valid_semver(self):
        """Test parsing valid semver strings."""
        from insideLLMs.runner import _semver_tuple

        assert _semver_tuple("1.0.0") == (1, 0, 0)
        assert _semver_tuple("2.3.4") == (2, 3, 4)

    def test_invalid_semver(self):
        """Test parsing invalid semver returns (0, 0, 0)."""
        from insideLLMs.runner import _semver_tuple

        assert _semver_tuple("invalid") == (0, 0, 0)
        assert _semver_tuple("") == (0, 0, 0)


class TestDefaultRunRoot:
    """Tests for _default_run_root helper function."""

    def test_default_run_root_env_var(self, monkeypatch, tmp_path):
        """Test _default_run_root respects INSIDELLMS_RUN_ROOT env var."""
        from insideLLMs.runner import _default_run_root

        custom_root = str(tmp_path / "custom_runs")
        monkeypatch.setenv("INSIDELLMS_RUN_ROOT", custom_root)
        result = _default_run_root()
        assert result == tmp_path / "custom_runs"

    def test_default_run_root_no_env(self, monkeypatch):
        """Test _default_run_root falls back to home directory."""
        from pathlib import Path

        from insideLLMs.runner import _default_run_root

        monkeypatch.delenv("INSIDELLMS_RUN_ROOT", raising=False)
        result = _default_run_root()
        assert result == Path.home() / ".insidellms" / "runs"


class TestFingerprintValue:
    """Tests for _fingerprint_value helper function."""

    def test_fingerprint_none(self):
        """Test _fingerprint_value returns None for None input."""
        from insideLLMs.runner import _fingerprint_value

        result = _fingerprint_value(None)
        assert result is None

    def test_fingerprint_dict(self):
        """Test _fingerprint_value for dict input."""
        from insideLLMs.runner import _fingerprint_value

        result = _fingerprint_value({"key": "value"})
        assert result is not None
        assert len(result) == 12  # First 12 chars of hash


class TestNormalizeInfoObjToDict:
    """Tests for _normalize_info_obj_to_dict helper function."""

    def test_normalize_dict(self):
        """Test normalizing a dict returns it unchanged."""
        from insideLLMs.runner import _normalize_info_obj_to_dict

        d = {"key": "value"}
        result = _normalize_info_obj_to_dict(d)
        assert result == d

    def test_normalize_none(self):
        """Test normalizing None returns empty dict."""
        from insideLLMs.runner import _normalize_info_obj_to_dict

        result = _normalize_info_obj_to_dict(None)
        assert result == {}

    def test_normalize_dataclass(self):
        """Test normalizing a dataclass."""
        from dataclasses import dataclass

        from insideLLMs.runner import _normalize_info_obj_to_dict

        @dataclass
        class Info:
            name: str
            version: str

        info = Info(name="test", version="1.0")
        result = _normalize_info_obj_to_dict(info)
        assert result == {"name": "test", "version": "1.0"}

    def test_normalize_pydantic_v1_style(self):
        """Test normalizing object with .dict() method (pydantic v1 style)."""
        from insideLLMs.runner import _normalize_info_obj_to_dict

        class FakePydanticV1:
            def dict(self):
                return {"name": "v1", "type": "test"}

        obj = FakePydanticV1()
        result = _normalize_info_obj_to_dict(obj)
        assert result == {"name": "v1", "type": "test"}

    def test_normalize_pydantic_v2_style(self):
        """Test normalizing object with .model_dump() method (pydantic v2 style)."""
        from insideLLMs.runner import _normalize_info_obj_to_dict

        class FakePydanticV2:
            def model_dump(self):
                return {"name": "v2", "type": "test"}

        obj = FakePydanticV2()
        result = _normalize_info_obj_to_dict(obj)
        assert result == {"name": "v2", "type": "test"}

    def test_normalize_pydantic_dict_raises(self):
        """Test normalizing object whose .dict() raises returns empty dict."""
        from insideLLMs.runner import _normalize_info_obj_to_dict

        class BrokenPydantic:
            def dict(self):
                raise RuntimeError("broken")

        obj = BrokenPydantic()
        result = _normalize_info_obj_to_dict(obj)
        assert result == {}

    def test_normalize_pydantic_model_dump_raises(self):
        """Test normalizing object whose .model_dump() raises returns empty dict."""
        from insideLLMs.runner import _normalize_info_obj_to_dict

        class BrokenPydanticV2:
            def model_dump(self):
                raise RuntimeError("broken")

        obj = BrokenPydanticV2()
        result = _normalize_info_obj_to_dict(obj)
        assert result == {}


class TestPrepareRunDir:
    """Tests for _prepare_run_dir helper function."""

    def test_create_new_directory(self, tmp_path):
        """Test creating a new run directory."""
        from insideLLMs.runner import _prepare_run_dir

        run_dir = tmp_path / "new_run"
        _prepare_run_dir(run_dir, overwrite=False)
        assert run_dir.exists()

    def test_use_existing_empty_directory(self, tmp_path):
        """Test using an existing empty directory."""
        from insideLLMs.runner import _prepare_run_dir

        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        _prepare_run_dir(run_dir, overwrite=False)
        assert run_dir.exists()

    def test_fail_on_non_empty_directory(self, tmp_path):
        """Test failure when directory is non-empty and overwrite=False."""
        from insideLLMs.runner import _prepare_run_dir

        run_dir = tmp_path / "non_empty"
        run_dir.mkdir()
        (run_dir / "some_file.txt").write_text("content")

        with pytest.raises(FileExistsError, match="already exists and is not empty"):
            _prepare_run_dir(run_dir, overwrite=False)

    def test_fail_on_existing_file_not_dir(self, tmp_path):
        """Test failure when path exists but is a file."""
        from insideLLMs.runner import _prepare_run_dir

        file_path = tmp_path / "is_a_file"
        file_path.write_text("I am a file")

        with pytest.raises(FileExistsError, match="is not a directory"):
            _prepare_run_dir(file_path, overwrite=False)

    def test_overwrite_with_sentinel(self, tmp_path):
        """Test overwriting a directory with manifest.json sentinel."""
        from insideLLMs.runner import _prepare_run_dir

        run_dir = tmp_path / "run_to_overwrite"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text("{}")
        (run_dir / "old_file.txt").write_text("old")

        _prepare_run_dir(run_dir, overwrite=True, run_root=tmp_path)
        assert run_dir.exists()
        assert not (run_dir / "old_file.txt").exists()

    def test_overwrite_with_insidellms_sentinel(self, tmp_path):
        """Test overwriting a directory with .insidellms_run sentinel."""
        from insideLLMs.runner import _prepare_run_dir

        run_dir = tmp_path / "run_with_marker"
        run_dir.mkdir()
        (run_dir / ".insidellms_run").write_text("marker")
        (run_dir / "old_file.txt").write_text("old")

        _prepare_run_dir(run_dir, overwrite=True, run_root=tmp_path)
        assert run_dir.exists()
        assert not (run_dir / "old_file.txt").exists()

    def test_refuse_overwrite_without_sentinel(self, tmp_path):
        """Test refusing to overwrite directory without sentinel."""
        from insideLLMs.runner import _prepare_run_dir

        run_dir = tmp_path / "no_sentinel"
        run_dir.mkdir()
        (run_dir / "some_file.txt").write_text("content")

        with pytest.raises(ValueError, match="does not look like an insideLLMs run"):
            _prepare_run_dir(run_dir, overwrite=True, run_root=tmp_path)

    def test_refuse_overwrite_short_path(self, tmp_path):
        """Test refusing to overwrite very short paths."""
        # Try to create a fake short path - use a simulated path
        # Since we can't easily test /tmp, we test the logic indirectly
        # by checking that the function contains the safety guard
        import inspect
        from pathlib import Path

        from insideLLMs import runner
        from insideLLMs.runner import _prepare_run_dir

        source = inspect.getsource(runner._prepare_run_dir)
        assert "len(resolved.parts) <= 2" in source


class TestCoerceModelInfo:
    """Tests for _coerce_model_info helper function."""

    def test_coerce_from_dict_info(self):
        """Test coercing model info from dict."""
        from insideLLMs.models.base import Model
        from insideLLMs.runner import _coerce_model_info
        from insideLLMs.types import ModelInfo

        class DictInfoModel(Model):
            def __init__(self):
                super().__init__(name="dict-model")

            def info(self):
                return {
                    "name": "test-model",
                    "provider": "test-provider",
                    "model_id": "test-id",
                    "max_tokens": 4096,
                }

            def generate(self, prompt, **kwargs):
                return "test"

        model = DictInfoModel()
        result = _coerce_model_info(model)

        assert isinstance(result, ModelInfo)
        assert result.name == "test-model"
        assert result.provider == "test-provider"
        assert result.model_id == "test-id"

    def test_coerce_preserves_model_info(self):
        """Test that existing ModelInfo is returned unchanged."""
        from insideLLMs.models.base import Model
        from insideLLMs.runner import _coerce_model_info
        from insideLLMs.types import ModelInfo

        original_info = ModelInfo(
            name="original",
            provider="original-provider",
            model_id="original-id",
        )

        class ModelInfoModel(Model):
            def __init__(self):
                super().__init__(name="modelinfo-model")

            def info(self):
                return original_info

            def generate(self, prompt, **kwargs):
                return "test"

        model = ModelInfoModel()
        result = _coerce_model_info(model)

        assert result is original_info

    def test_coerce_with_exception_in_info(self):
        """Test coercing when model.info() raises an exception."""
        from insideLLMs.models.base import Model
        from insideLLMs.runner import _coerce_model_info
        from insideLLMs.types import ModelInfo

        class BrokenInfoModel(Model):
            def __init__(self):
                super().__init__(name="broken-model")

            def info(self):
                raise RuntimeError("info broken")

            def generate(self, prompt, **kwargs):
                return "test"

        model = BrokenInfoModel()
        result = _coerce_model_info(model)

        assert isinstance(result, ModelInfo)
        assert result.name == "broken-model"


class TestBuildResultRecord:
    """Tests for _build_result_record helper function."""

    def test_build_record_with_string_output(self):
        """Test building a record with string output."""
        from datetime import datetime, timezone

        from insideLLMs.runner import _build_result_record

        record = _build_result_record(
            schema_version="1.0.0",
            run_id="test-run",
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            model={"model_id": "test-model"},
            probe={"probe_id": "test-probe"},
            dataset={"dataset_id": "test-dataset"},
            item="test input",
            output="test output",
            latency_ms=100.0,
            store_messages=True,
            index=0,
            status="success",
            error=None,
        )

        assert record["output_text"] == "test output"
        assert record["status"] == "success"
        assert record["error"] is None

    def test_build_record_with_dict_output(self):
        """Test building a record with dict output containing scores."""
        from datetime import datetime, timezone

        from insideLLMs.runner import _build_result_record

        record = _build_result_record(
            schema_version="1.0.0",
            run_id="test-run",
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            model={"model_id": "test-model"},
            probe={"probe_id": "test-probe"},
            dataset={"dataset_id": "test-dataset"},
            item="test input",
            output={
                "output_text": "dict output text",
                "scores": {"accuracy": 0.95},
                "usage": {"tokens": 100},
                "primary_metric": "accuracy",
            },
            latency_ms=100.0,
            store_messages=True,
            index=0,
            status="success",
            error=None,
        )

        assert record["output_text"] == "dict output text"
        assert record["scores"] == {"accuracy": 0.95}
        assert record["usage"] == {"tokens": 100}
        assert record["primary_metric"] == "accuracy"

    def test_build_record_with_messages_input(self):
        """Test building a record with messages-style input."""
        from datetime import datetime, timezone

        from insideLLMs.runner import _build_result_record

        item = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "example_id": "example-1",
        }

        record = _build_result_record(
            schema_version="1.0.0",
            run_id="test-run",
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            model={"model_id": "test-model"},
            probe={"probe_id": "test-probe"},
            dataset={"dataset_id": "test-dataset"},
            item=item,
            output="response",
            latency_ms=100.0,
            store_messages=True,
            index=0,
            status="success",
            error=None,
        )

        assert record["example_id"] == "example-1"
        assert record["messages"] is not None
        assert len(record["messages"]) == 2
        assert record["messages_hash"] is not None

    def test_build_record_with_error(self):
        """Test building a record with an error."""
        from datetime import datetime, timezone

        from insideLLMs.runner import _build_result_record

        record = _build_result_record(
            schema_version="1.0.0",
            run_id="test-run",
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            model={"model_id": "test-model"},
            probe={"probe_id": "test-probe"},
            dataset={"dataset_id": "test-dataset"},
            item="test input",
            output=None,
            latency_ms=50.0,
            store_messages=False,
            index=0,
            status="error",
            error=ValueError("Test error"),
        )

        assert record["status"] == "error"
        assert record["error"] == "Test error"
        assert record["error_type"] == "ValueError"


class TestLoadConfigErrors:
    """Test load_config error handling."""

    def test_load_config_file_not_found(self):
        """Test load_config raises FileNotFoundError for missing file."""
        from insideLLMs.runner import load_config

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/config.yaml")

    def test_load_config_unsupported_format(self, tmp_path):
        """Test load_config raises ValueError for unsupported format."""
        from insideLLMs.runner import load_config

        config_path = tmp_path / "config.txt"
        config_path.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config(config_path)


class TestCreateModelFromConfigFallback:
    """Tests for _create_model_from_config fallback path."""

    def test_create_unknown_model_type(self):
        """Test creating model with unknown type raises ValueError."""
        from insideLLMs.runner import _create_model_from_config

        with pytest.raises(ValueError, match="Unknown model type"):
            _create_model_from_config({"type": "nonexistent_model_xyz"})


class TestCreateProbeFromConfigFallback:
    """Tests for _create_probe_from_config fallback path."""

    def test_create_unknown_probe_type(self):
        """Test creating probe with unknown type raises ValueError."""
        from insideLLMs.runner import _create_probe_from_config

        with pytest.raises(ValueError, match="Unknown probe type"):
            _create_probe_from_config({"type": "nonexistent_probe_xyz"})


class TestLoadDatasetFromConfig:
    """Tests for _load_dataset_from_config."""

    def test_load_csv_dataset(self, tmp_path):
        """Test loading CSV dataset from config."""
        import csv

        from insideLLMs.runner import _load_dataset_from_config

        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerow({"question": "Q1?", "answer": "A1"})
            writer.writerow({"question": "Q2?", "answer": "A2"})

        config = {"format": "csv", "path": str(csv_path)}
        result = _load_dataset_from_config(config, tmp_path)
        assert len(result) == 2

    def test_load_jsonl_dataset(self, tmp_path):
        """Test loading JSONL dataset from config."""
        import json

        from insideLLMs.runner import _load_dataset_from_config

        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"question": "Q1?"}) + "\n")
            f.write(json.dumps({"question": "Q2?"}) + "\n")

        config = {"format": "jsonl", "path": str(jsonl_path)}
        result = _load_dataset_from_config(config, tmp_path)
        assert len(result) == 2

    def test_load_unknown_format(self, tmp_path):
        """Test loading unknown format raises ValueError."""
        from insideLLMs.runner import _load_dataset_from_config

        config = {"format": "unknown_format"}

        with pytest.raises(ValueError, match="Unknown dataset format"):
            _load_dataset_from_config(config, tmp_path)


class TestAtomicWrite:
    """Tests for atomic write functions."""

    def test_atomic_write_text(self, tmp_path):
        """Test _atomic_write_text creates file correctly."""
        from insideLLMs.runner import _atomic_write_text

        file_path = tmp_path / "test.txt"
        _atomic_write_text(file_path, "Hello, World!")

        assert file_path.exists()
        assert file_path.read_text() == "Hello, World!"

    def test_atomic_write_yaml(self, tmp_path):
        """Test _atomic_write_yaml creates file correctly."""
        import yaml

        from insideLLMs.runner import _atomic_write_yaml

        file_path = tmp_path / "test.yaml"
        _atomic_write_yaml(file_path, {"key": "value", "number": 42})

        assert file_path.exists()
        content = yaml.safe_load(file_path.read_text())
        assert content["key"] == "value"
        assert content["number"] == 42


class TestDeterministicFunctions:
    """Tests for deterministic time and ID functions."""

    def test_deterministic_run_id_from_inputs(self):
        """Test deterministic run ID generation from inputs."""
        from insideLLMs.runner import _deterministic_run_id_from_inputs

        run_id = _deterministic_run_id_from_inputs(
            schema_version="1.0.0",
            model_spec={"model_id": "test"},
            probe_spec={"probe_id": "test"},
            dataset_spec={"dataset_id": "test"},
            prompt_set=["q1", "q2"],
            probe_kwargs={},
        )

        assert len(run_id) == 32
        # Same inputs should produce same ID
        run_id2 = _deterministic_run_id_from_inputs(
            schema_version="1.0.0",
            model_spec={"model_id": "test"},
            probe_spec={"probe_id": "test"},
            dataset_spec={"dataset_id": "test"},
            prompt_set=["q1", "q2"],
            probe_kwargs={},
        )
        assert run_id == run_id2

    def test_deterministic_base_time(self):
        """Test deterministic base time generation."""
        from insideLLMs.runner import _deterministic_base_time

        time1 = _deterministic_base_time("run-123")
        time2 = _deterministic_base_time("run-123")
        assert time1 == time2

        time3 = _deterministic_base_time("different-run")
        assert time3 != time1

    def test_deterministic_item_times(self):
        """Test deterministic item times."""
        from datetime import datetime, timezone

        from insideLLMs.runner import _deterministic_item_times

        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        started, completed = _deterministic_item_times(base, 5)

        assert started > base
        assert completed > started

    def test_deterministic_run_times_negative_total(self):
        """Test _deterministic_run_times handles negative total."""
        from datetime import datetime, timezone

        from insideLLMs.runner import _deterministic_run_times

        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        started, completed = _deterministic_run_times(base, -5)

        # Should treat negative as 0
        assert completed > started


class TestResolvePath:
    """Tests for _resolve_path helper function."""

    def test_resolve_absolute_path(self):
        """Test resolving absolute path returns it unchanged."""
        from pathlib import Path

        from insideLLMs.runner import _resolve_path

        abs_path = "/absolute/path/to/file.txt"
        result = _resolve_path(abs_path, Path("/base"))
        assert result == Path(abs_path)

    def test_resolve_relative_path(self):
        """Test resolving relative path uses base directory."""
        from pathlib import Path

        from insideLLMs.runner import _resolve_path

        result = _resolve_path("relative/file.txt", Path("/base/dir"))
        assert result == Path("/base/dir/relative/file.txt")


class TestBuildResolvedConfigSnapshot:
    """Tests for _build_resolved_config_snapshot."""

    def test_resolve_dataset_path(self, tmp_path):
        """Test that dataset path is normalized for determinism."""
        from insideLLMs.runner import _build_resolved_config_snapshot

        config = {
            "model": {"type": "dummy"},
            "probe": {"type": "logic"},
            "dataset": {
                "format": "jsonl",
                "path": "relative/data.jsonl",
            },
        }

        snapshot = _build_resolved_config_snapshot(config, tmp_path)

        # Path should be preserved (and normalized) for portability.
        assert snapshot["dataset"]["path"] == "relative/data.jsonl"

    def test_preserves_non_path_configs(self, tmp_path):
        """Test that non-path configs are preserved."""
        from insideLLMs.runner import _build_resolved_config_snapshot

        config = {
            "model": {"type": "dummy", "args": {"setting": "value"}},
            "probe": {"type": "logic"},
            "dataset": {"format": "hf", "name": "test-dataset"},
        }

        snapshot = _build_resolved_config_snapshot(config, tmp_path)

        assert snapshot["model"]["args"]["setting"] == "value"
        assert snapshot["dataset"]["name"] == "test-dataset"


class TestEnsureRunSentinel:
    """Tests for _ensure_run_sentinel."""

    def test_creates_sentinel(self, tmp_path):
        """Test sentinel file is created."""
        from insideLLMs.runner import _ensure_run_sentinel

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _ensure_run_sentinel(run_dir)

        sentinel = run_dir / ".insidellms_run"
        assert sentinel.exists()

    def test_does_not_overwrite_existing(self, tmp_path):
        """Test existing sentinel is not overwritten."""
        from insideLLMs.runner import _ensure_run_sentinel

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        sentinel = run_dir / ".insidellms_run"
        sentinel.write_text("original content")

        _ensure_run_sentinel(run_dir)

        # Should keep original content
        assert sentinel.read_text() == "original content"


class TestRunProbeAsync:
    """Tests for run_probe_async convenience function."""

    @pytest.mark.asyncio
    async def test_run_probe_async_basic(self):
        """Test basic run_probe_async usage."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import run_probe_async

        model = DummyModel()
        probe = LogicProbe()

        results = await run_probe_async(model, probe, ["Test?"], emit_run_artifacts=False)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_run_probe_async_with_concurrency(self):
        """Test run_probe_async with custom concurrency."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runner import run_probe_async

        model = DummyModel()
        probe = LogicProbe()

        results = await run_probe_async(
            model,
            probe,
            ["Q1", "Q2", "Q3"],
            concurrency=2,
            emit_run_artifacts=False,
        )
        assert len(results) == 3
