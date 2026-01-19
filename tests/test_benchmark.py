"""Tests for the benchmark module."""

import json
import tempfile
from pathlib import Path

import pytest


class TestModelBenchmark:
    """Test the ModelBenchmark class."""

    def test_basic_creation(self):
        """Test creating a ModelBenchmark."""
        from insideLLMs.benchmark import ModelBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        models = [DummyModel(name="DummyModelA"), DummyModel(name="DummyModelB")]
        probe = LogicProbe()
        benchmark = ModelBenchmark(models, probe)

        assert benchmark is not None
        assert len(benchmark.models) == 2
        assert benchmark.probe == probe
        assert benchmark.name == "Model Benchmark"

    def test_with_custom_name(self):
        """Test ModelBenchmark with custom name."""
        from insideLLMs.benchmark import ModelBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        models = [DummyModel()]
        probe = LogicProbe()
        benchmark = ModelBenchmark(models, probe, name="Custom Benchmark")

        assert benchmark.name == "Custom Benchmark"

    def test_run_benchmark(self):
        """Test running a benchmark."""
        from insideLLMs.benchmark import ModelBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        models = [DummyModel()]
        probe = LogicProbe()
        benchmark = ModelBenchmark(models, probe)

        prompt_set = ["What is 2+2?", "Is the sky blue?"]
        results = benchmark.run(prompt_set)

        assert results is not None
        assert "name" in results
        assert "probe" in results
        assert "models" in results
        assert "timestamp" in results
        assert len(results["models"]) == 1

    def test_run_benchmark_multiple_models(self):
        """Test running a benchmark with multiple models."""
        from insideLLMs.benchmark import ModelBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        models = [DummyModel(name="DummyModelA"), DummyModel(name="DummyModelB")]
        probe = LogicProbe()
        benchmark = ModelBenchmark(models, probe)

        prompt_set = ["Test question?"]
        results = benchmark.run(prompt_set)

        assert len(results["models"]) == 2

    def test_benchmark_metrics(self):
        """Test that benchmark results contain metrics."""
        from insideLLMs.benchmark import ModelBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        models = [DummyModel()]
        probe = LogicProbe()
        benchmark = ModelBenchmark(models, probe)

        prompt_set = ["Test?"]
        results = benchmark.run(prompt_set)

        model_result = results["models"][0]
        assert "metrics" in model_result
        assert "total_time" in model_result["metrics"]
        assert "avg_time_per_item" in model_result["metrics"]
        assert "total_items" in model_result["metrics"]
        assert "success_rate" in model_result["metrics"]

    def test_save_results(self):
        """Test saving benchmark results to file."""
        from insideLLMs.benchmark import ModelBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        models = [DummyModel()]
        probe = LogicProbe()
        benchmark = ModelBenchmark(models, probe)

        prompt_set = ["Test?"]
        benchmark.run(prompt_set)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            benchmark.save_results(str(path))

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert "name" in data
            assert "models" in data

    def test_compare_models(self):
        """Test comparing models after benchmark."""
        from insideLLMs.benchmark import ModelBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        models = [DummyModel(name="DummyModelA"), DummyModel(name="DummyModelB")]
        probe = LogicProbe()
        benchmark = ModelBenchmark(models, probe)

        prompt_set = ["Test?"]
        benchmark.run(prompt_set)

        comparison = benchmark.compare_models()

        assert "name" in comparison
        assert "metrics" in comparison
        assert "rankings" in comparison
        assert "total_time" in comparison["metrics"]
        assert "success_rate" in comparison["metrics"]

    def test_compare_models_without_run(self):
        """Test that compare_models raises error without running benchmark first."""
        from insideLLMs.benchmark import ModelBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        models = [DummyModel()]
        probe = LogicProbe()
        benchmark = ModelBenchmark(models, probe)

        with pytest.raises(ValueError):
            benchmark.compare_models()


class TestProbeBenchmark:
    """Test the ProbeBenchmark class."""

    def test_basic_creation(self):
        """Test creating a ProbeBenchmark."""
        from insideLLMs.benchmark import ProbeBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import BiasProbe, LogicProbe

        model = DummyModel()
        probes = [LogicProbe(), BiasProbe()]
        benchmark = ProbeBenchmark(model, probes)

        assert benchmark is not None
        assert benchmark.model == model
        assert len(benchmark.probes) == 2
        assert benchmark.name == "Probe Benchmark"

    def test_with_custom_name(self):
        """Test ProbeBenchmark with custom name."""
        from insideLLMs.benchmark import ProbeBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        model = DummyModel()
        probes = [LogicProbe()]
        benchmark = ProbeBenchmark(model, probes, name="Custom Probe Benchmark")

        assert benchmark.name == "Custom Probe Benchmark"

    def test_run_benchmark(self):
        """Test running a probe benchmark."""
        from insideLLMs.benchmark import ProbeBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        model = DummyModel()
        probes = [LogicProbe()]
        benchmark = ProbeBenchmark(model, probes)

        prompt_set = ["What is 2+2?", "Is the sky blue?"]
        results = benchmark.run(prompt_set)

        assert results is not None
        assert "name" in results
        assert "model" in results
        assert "probes" in results
        assert "timestamp" in results
        assert len(results["probes"]) == 1

    def test_run_benchmark_multiple_probes(self):
        """Test running a benchmark with multiple probes."""
        from insideLLMs.benchmark import ProbeBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import BiasProbe, LogicProbe

        model = DummyModel()
        probes = [LogicProbe(), BiasProbe()]
        benchmark = ProbeBenchmark(model, probes)

        prompt_set = ["Test question?"]
        results = benchmark.run(prompt_set)

        assert len(results["probes"]) == 2

    def test_probe_benchmark_metrics(self):
        """Test that probe benchmark results contain metrics."""
        from insideLLMs.benchmark import ProbeBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        model = DummyModel()
        probes = [LogicProbe()]
        benchmark = ProbeBenchmark(model, probes)

        prompt_set = ["Test?"]
        results = benchmark.run(prompt_set)

        probe_result = results["probes"][0]
        assert "metrics" in probe_result
        assert "total_time" in probe_result["metrics"]
        assert "avg_time_per_item" in probe_result["metrics"]
        assert "total_items" in probe_result["metrics"]
        assert "success_rate" in probe_result["metrics"]

    def test_save_results(self):
        """Test saving probe benchmark results to file."""
        from insideLLMs.benchmark import ProbeBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        model = DummyModel()
        probes = [LogicProbe()]
        benchmark = ProbeBenchmark(model, probes)

        prompt_set = ["Test?"]
        benchmark.run(prompt_set)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            benchmark.save_results(str(path))

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert "name" in data
            assert "probes" in data


class TestBenchmarkEdgeCases:
    """Test edge cases in benchmarking."""

    def test_empty_prompt_set(self):
        """Test running benchmark with empty prompt set raises ValidationError."""
        import pytest

        from insideLLMs.benchmark import ModelBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.validation import ValidationError

        models = [DummyModel()]
        probe = LogicProbe()
        benchmark = ModelBenchmark(models, probe)

        # Empty prompt sets should now raise ValidationError
        with pytest.raises(ValidationError, match="cannot be empty"):
            benchmark.run([])

    def test_unicode_prompts(self):
        """Test benchmark with Unicode prompts."""
        from insideLLMs.benchmark import ModelBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        models = [DummyModel()]
        probe = LogicProbe()
        benchmark = ModelBenchmark(models, probe)

        prompt_set = ["日本語テスト", "Вопрос на русском?"]
        results = benchmark.run(prompt_set)

        assert results is not None
        assert results["models"][0]["metrics"]["total_items"] == 2

    def test_very_long_prompts(self):
        """Test benchmark with very long prompts."""
        from insideLLMs.benchmark import ModelBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        models = [DummyModel()]
        probe = LogicProbe()
        benchmark = ModelBenchmark(models, probe)

        prompt_set = ["a" * 10000 + "?"]
        results = benchmark.run(prompt_set)

        assert results is not None


class TestBenchmarkIntegration:
    """Integration tests for benchmarking."""

    def test_full_benchmark_workflow(self):
        """Test full benchmark workflow."""
        from insideLLMs.benchmark import ModelBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe

        # Create benchmark with distinctly named models
        models = [DummyModel(name="ModelA"), DummyModel(name="ModelB")]
        probe = LogicProbe()
        benchmark = ModelBenchmark(models, probe, name="Integration Test")

        # Run benchmark
        prompt_set = ["What is 1+1?", "What is 2+2?"]
        results = benchmark.run(prompt_set)

        # Check results
        assert results["name"] == "Integration Test"
        assert len(results["models"]) == 2

        # Compare models
        comparison = benchmark.compare_models()
        assert len(comparison["rankings"]["success_rate"]) == 2

        # Save results
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            benchmark.save_results(str(path))
            assert path.exists()

    def test_probe_benchmark_workflow(self):
        """Test full probe benchmark workflow."""
        from insideLLMs.benchmark import ProbeBenchmark
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import BiasProbe, LogicProbe

        # Create benchmark
        model = DummyModel()
        probes = [LogicProbe(), BiasProbe()]
        benchmark = ProbeBenchmark(model, probes, name="Probe Integration Test")

        # Run benchmark
        prompt_set = ["Test question?"]
        results = benchmark.run(prompt_set)

        # Check results
        assert results["name"] == "Probe Integration Test"
        assert len(results["probes"]) == 2

        # Save results
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            benchmark.save_results(str(path))
            assert path.exists()
