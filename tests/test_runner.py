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

        model = DummyModel()
        probe = LogicProbe()
        runner = ProbeRunner(model, probe)

        results = runner.run([])
        assert len(results) == 0

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
