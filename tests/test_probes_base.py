"""Tests for insideLLMs/probes/base.py module."""

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from insideLLMs.probes.base import Probe, ProbeProtocol
from insideLLMs.types import ProbeCategory, ProbeScore, ResultStatus


class SimpleProbe(Probe[str]):
    """A simple probe implementation for testing."""

    default_category = ProbeCategory.LOGIC

    def run(self, model: Any, data: Any, **kwargs: Any) -> str:
        """Run the probe on the model."""
        return model.generate(data)

    def score(self, results: list) -> ProbeScore:
        """Score the probe results."""
        success_count = sum(1 for r in results if r.status == ResultStatus.SUCCESS)
        return ProbeScore(accuracy=success_count / len(results) if results else 0.0)


class TestProbeInitialization:
    """Tests for Probe base class initialization."""

    def test_init_with_name(self):
        """Test basic probe initialization with name."""
        probe = SimpleProbe(name="TestProbe")
        assert probe.name == "TestProbe"
        assert probe.category == ProbeCategory.LOGIC
        assert probe.description == "A simple probe implementation for testing."

    def test_init_with_custom_category(self):
        """Test probe initialization with custom category."""
        probe = SimpleProbe(name="TestProbe", category=ProbeCategory.BIAS)
        assert probe.category == ProbeCategory.BIAS

    def test_init_with_custom_description(self):
        """Test probe initialization with custom description."""
        probe = SimpleProbe(name="TestProbe", description="Custom description for testing")
        assert probe.description == "Custom description for testing"

    def test_init_inherits_default_category(self):
        """Test that subclass inherits default_category."""
        probe = SimpleProbe(name="Test")
        # SimpleProbe has default_category = ProbeCategory.LOGIC
        assert probe.category == ProbeCategory.LOGIC


class TestProbeRun:
    """Tests for Probe.run method."""

    def test_run_calls_model(self):
        """Test that run calls the model."""
        probe = SimpleProbe(name="Test")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="test output")

        result = probe.run(mock_model, "test input")

        mock_model.generate.assert_called_once_with("test input")
        assert result == "test output"

    def test_run_with_kwargs(self):
        """Test that run passes kwargs to model."""
        # Create a probe that uses kwargs

        class KwargsProbe(Probe[str]):
            def run(self, model, data, temperature=0.7, **kwargs):
                return model.generate(data, temperature=temperature)

        probe = KwargsProbe(name="Test")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="output")

        probe.run(mock_model, "input", temperature=0.5)
        mock_model.generate.assert_called_once_with("input", temperature=0.5)


class TestProbeRunBatch:
    """Tests for Probe.run_batch method."""

    def test_run_batch_success(self):
        """Test successful batch processing."""
        probe = SimpleProbe(name="Test")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(side_effect=["output1", "output2", "output3"])

        dataset = ["input1", "input2", "input3"]
        results = probe.run_batch(mock_model, dataset)

        assert len(results) == 3
        assert all(r.status == ResultStatus.SUCCESS for r in results)
        assert results[0].output == "output1"
        assert results[1].output == "output2"
        assert results[2].output == "output3"

    def test_run_batch_with_error(self):
        """Test batch processing with error."""
        probe = SimpleProbe(name="Test")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(
            side_effect=["output1", ValueError("Test error"), "output3"]
        )

        dataset = ["input1", "input2", "input3"]
        results = probe.run_batch(mock_model, dataset)

        assert len(results) == 3
        assert results[0].status == ResultStatus.SUCCESS
        assert results[1].status == ResultStatus.ERROR
        assert "ValueError" in results[1].error
        assert results[2].status == ResultStatus.SUCCESS

    def test_run_batch_with_timeout(self):
        """Test batch processing with timeout."""
        probe = SimpleProbe(name="Test")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(side_effect=TimeoutError("Request timed out"))

        dataset = ["input1"]
        results = probe.run_batch(mock_model, dataset)

        assert len(results) == 1
        assert results[0].status == ResultStatus.TIMEOUT
        assert "timed out" in results[0].error

    def test_run_batch_with_rate_limit(self):
        """Test batch processing with rate limit error."""
        probe = SimpleProbe(name="Test")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(side_effect=Exception("Rate limit exceeded (429)"))

        dataset = ["input1"]
        results = probe.run_batch(mock_model, dataset)

        assert len(results) == 1
        assert results[0].status == ResultStatus.RATE_LIMITED

    def test_run_batch_tracks_latency(self):
        """Test that batch processing tracks latency."""
        probe = SimpleProbe(name="Test")
        mock_model = MagicMock()

        def slow_generate(data):
            time.sleep(0.01)  # 10ms
            return "output"

        mock_model.generate = slow_generate

        dataset = ["input1"]
        results = probe.run_batch(mock_model, dataset)

        assert results[0].latency_ms >= 10  # At least 10ms

    def test_run_batch_parallel(self):
        """Test parallel batch processing."""
        probe = SimpleProbe(name="Test")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="output")

        dataset = ["input1", "input2", "input3", "input4"]
        results = probe.run_batch(mock_model, dataset, max_workers=2)

        assert len(results) == 4
        assert all(r.status == ResultStatus.SUCCESS for r in results)


class TestProbeScore:
    """Tests for Probe.score method."""

    def test_score_returns_probe_score(self):
        """Test that score returns ProbeScore."""
        probe = SimpleProbe(name="Test")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="output")

        results = probe.run_batch(mock_model, ["input1", "input2"])
        score = probe.score(results)

        assert isinstance(score, ProbeScore)
        assert score.accuracy == 1.0  # Both succeeded

    def test_score_with_failures(self):
        """Test score calculation with failures."""
        probe = SimpleProbe(name="Test")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(side_effect=["output", ValueError("Error")])

        results = probe.run_batch(mock_model, ["input1", "input2"])
        score = probe.score(results)

        assert score.accuracy == 0.5  # 1 of 2 succeeded

    def test_score_empty_results(self):
        """Test score with empty results."""
        probe = SimpleProbe(name="Test")
        score = probe.score([])
        assert score.accuracy == 0.0


class TestProbeProtocol:
    """Tests for ProbeProtocol."""

    def test_protocol_isinstance(self):
        """Test that Probe satisfies ProbeProtocol."""
        probe = SimpleProbe(name="Test")
        assert isinstance(probe, ProbeProtocol)

    def test_protocol_attributes(self):
        """Test protocol requires name and category attributes."""
        probe = SimpleProbe(name="Test", category=ProbeCategory.LOGIC)

        # ProbeProtocol requires name and category
        assert hasattr(probe, "name")
        assert hasattr(probe, "category")
        assert hasattr(probe, "run")


class TestProbeRepr:
    """Tests for Probe string representations."""

    def test_probe_has_name(self):
        """Test probe has accessible name."""
        probe = SimpleProbe(name="MyTestProbe")
        assert probe.name == "MyTestProbe"


class TestCustomProbeImplementation:
    """Tests for custom probe implementations."""

    def test_custom_probe_with_complex_output(self):
        """Test probe with complex output type."""

        @dataclass
        class ComplexOutput:
            text: str
            score: float

        class ComplexProbe(Probe[ComplexOutput]):
            def run(self, model, data, **kwargs):
                result = model.generate(data)
                return ComplexOutput(text=result, score=0.9)

        probe = ComplexProbe(name="Complex")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="test output")

        output = probe.run(mock_model, "test")
        assert isinstance(output, ComplexOutput)
        assert output.text == "test output"
        assert output.score == 0.9

    def test_probe_with_preprocessing(self):
        """Test probe that preprocesses input."""

        class PreprocessingProbe(Probe[str]):
            def run(self, model, data, **kwargs):
                processed = data.upper()
                return model.generate(processed)

        probe = PreprocessingProbe(name="Preprocess")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="output")

        probe.run(mock_model, "input")
        mock_model.generate.assert_called_once_with("INPUT")

    def test_probe_with_postprocessing(self):
        """Test probe that postprocesses output."""

        class PostprocessingProbe(Probe[str]):
            def run(self, model, data, **kwargs):
                result = model.generate(data)
                return result.lower()

        probe = PostprocessingProbe(name="Postprocess")
        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value="OUTPUT")

        result = probe.run(mock_model, "input")
        assert result == "output"


class TestProbeDefaultCategory:
    """Tests for default_category class attribute."""

    def test_default_category_override(self):
        """Test that subclass can override default_category."""

        class BiasProbe(Probe[str]):
            default_category = ProbeCategory.BIAS

            def run(self, model, data, **kwargs):
                return ""

        probe = BiasProbe(name="Test")
        assert probe.category == ProbeCategory.BIAS

    def test_default_category_custom(self):
        """Test default_category for CUSTOM category."""

        class CustomProbe(Probe[str]):
            default_category = ProbeCategory.CUSTOM

            def run(self, model, data, **kwargs):
                return ""

        probe = CustomProbe(name="Test")
        assert probe.category == ProbeCategory.CUSTOM
