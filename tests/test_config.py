"""Tests for configuration validation module."""

import json
import tempfile
from pathlib import Path

import pytest

from insideLLMs.config import (
    PYDANTIC_AVAILABLE,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    ModelProvider,
    ProbeConfig,
    ProbeType,
    RunnerConfig,
    create_example_config,
    load_config,
    load_config_from_json,
    save_config_to_json,
    validate_config,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_basic_creation(self):
        """Test basic ModelConfig creation."""
        config = ModelConfig(
            provider="openai" if not PYDANTIC_AVAILABLE else ModelProvider.OPENAI,
            model_id="gpt-4",
        )

        assert config.model_id == "gpt-4"
        assert config.temperature == 0.7  # Default

    def test_with_all_fields(self):
        """Test ModelConfig with all fields."""
        config = ModelConfig(
            provider="anthropic" if not PYDANTIC_AVAILABLE else ModelProvider.ANTHROPIC,
            model_id="claude-3-opus",
            name="Claude Opus",
            api_key_env="ANTHROPIC_API_KEY",
            temperature=0.5,
            max_tokens=2000,
            timeout=60.0,
            max_retries=5,
        )

        assert config.name == "Claude Opus"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Requires Pydantic")
    def test_temperature_validation(self):
        """Test temperature bounds validation."""
        # Valid temperature
        config = ModelConfig(provider=ModelProvider.DUMMY, model_id="test", temperature=1.5)
        assert config.temperature == 1.5

        # Invalid temperature (too high)
        with pytest.raises(Exception):  # ValidationError
            ModelConfig(provider=ModelProvider.DUMMY, model_id="test", temperature=3.0)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Requires Pydantic")
    def test_default_name_from_model_id(self):
        """Test that name defaults to model_id."""
        config = ModelConfig(provider=ModelProvider.OPENAI, model_id="gpt-4-turbo")
        assert config.name == "gpt-4-turbo"


class TestProbeConfig:
    """Tests for ProbeConfig."""

    def test_basic_creation(self):
        """Test basic ProbeConfig creation."""
        config = ProbeConfig(
            type="logic" if not PYDANTIC_AVAILABLE else ProbeType.LOGIC,
        )

        assert str(config.type) == "logic" or config.type == ProbeType.LOGIC

    def test_with_params(self):
        """Test ProbeConfig with parameters."""
        config = ProbeConfig(
            type="bias" if not PYDANTIC_AVAILABLE else ProbeType.BIAS,
            name="Bias Detection",
            params={"dimensions": ["gender", "age"]},
        )

        assert config.name == "Bias Detection"
        assert "dimensions" in config.params


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_file_source(self):
        """Test file source configuration."""
        config = DatasetConfig(
            source="file",
            path="data/test.jsonl",
        )

        assert config.source == "file"
        assert config.path == "data/test.jsonl"

    def test_inline_source(self):
        """Test inline source configuration."""
        data = [{"q": "1+1?", "a": "2"}]
        config = DatasetConfig(
            source="inline",
            data=data,
        )

        assert config.source == "inline"
        assert config.data == data

    def test_hf_source(self):
        """Test HuggingFace source configuration."""
        config = DatasetConfig(
            source="hf",
            name="squad",
            split="validation",
        )

        assert config.source == "hf"
        assert config.name == "squad"

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Requires Pydantic")
    def test_file_source_requires_path(self):
        """Test that file source requires path."""
        with pytest.raises(ValueError):
            DatasetConfig(source="file")

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Requires Pydantic")
    def test_inline_source_requires_data(self):
        """Test that inline source requires data."""
        with pytest.raises(ValueError):
            DatasetConfig(source="inline")


class TestRunnerConfig:
    """Tests for RunnerConfig."""

    def test_defaults(self):
        """Test default values."""
        config = RunnerConfig()

        assert config.concurrency == 1
        assert config.output_dir == "output"
        assert config.progress_bar is True

    def test_custom_values(self):
        """Test custom values."""
        config = RunnerConfig(
            concurrency=10,
            output_dir="results",
            output_formats=["json", "html"],
            verbose=True,
        )

        assert config.concurrency == 10
        assert config.output_dir == "results"
        assert "html" in config.output_formats


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_full_config(self):
        """Test complete experiment configuration."""
        config = ExperimentConfig(
            name="Test Experiment",
            model=ModelConfig(
                provider="openai" if not PYDANTIC_AVAILABLE else ModelProvider.OPENAI,
                model_id="gpt-4",
            ),
            probe=ProbeConfig(
                type="logic" if not PYDANTIC_AVAILABLE else ProbeType.LOGIC,
            ),
            dataset=DatasetConfig(
                source="inline",
                data=[{"q": "test"}],
            ),
        )

        assert config.name == "Test Experiment"
        assert config.model.model_id == "gpt-4"

    def test_with_runner(self):
        """Test experiment with custom runner."""
        config = ExperimentConfig(
            name="Test",
            model=ModelConfig(
                provider="dummy" if not PYDANTIC_AVAILABLE else ModelProvider.DUMMY,
                model_id="dummy-v1",
            ),
            probe=ProbeConfig(
                type="logic" if not PYDANTIC_AVAILABLE else ProbeType.LOGIC,
            ),
            dataset=DatasetConfig(source="inline", data=[{}]),
            runner=RunnerConfig(concurrency=5),
        )

        assert config.runner.concurrency == 5

    def test_with_tags_and_metadata(self):
        """Test experiment with tags and metadata."""
        config = ExperimentConfig(
            name="Test",
            model=ModelConfig(
                provider="dummy" if not PYDANTIC_AVAILABLE else ModelProvider.DUMMY,
                model_id="dummy-v1",
            ),
            probe=ProbeConfig(
                type="logic" if not PYDANTIC_AVAILABLE else ProbeType.LOGIC,
            ),
            dataset=DatasetConfig(source="inline", data=[{}]),
            tags=["test", "ci"],
            metadata={"version": "1.0"},
        )

        assert "test" in config.tags
        assert config.metadata["version"] == "1.0"


class TestConfigIO:
    """Tests for configuration I/O functions."""

    def test_save_and_load_json(self):
        """Test saving and loading JSON config."""
        config = create_example_config()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_config_to_json(config, path)
            loaded = load_config_from_json(path)

            assert loaded.name == config.name
            assert loaded.model.model_id == config.model.model_id
        finally:
            Path(path).unlink()

    def test_load_config_auto_detect_json(self):
        """Test auto-detection of JSON format."""
        config = create_example_config()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_config_to_json(config, path)
            loaded = load_config(path)

            assert loaded.name == config.name
        finally:
            Path(path).unlink()

    def test_load_config_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.json")

    def test_load_config_unsupported_format(self):
        """Test loading unsupported format."""
        with pytest.raises(ValueError):
            load_config("config.txt")


class TestCreateExampleConfig:
    """Tests for create_example_config function."""

    def test_creates_valid_config(self):
        """Test that example config is valid."""
        config = create_example_config()

        assert config.name == "Example Experiment"
        assert config.model is not None
        assert config.probe is not None
        assert config.dataset is not None
        assert config.runner is not None

    def test_example_config_has_inline_data(self):
        """Test that example config has inline data."""
        config = create_example_config()

        assert config.dataset.source == "inline"
        assert config.dataset.data is not None
        assert len(config.dataset.data) > 0


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_validate_dict(self):
        """Test validating a dictionary."""
        data = {
            "name": "Test",
            "model": {
                "provider": "dummy",
                "model_id": "dummy-v1",
            },
            "probe": {
                "type": "logic",
            },
            "dataset": {
                "source": "inline",
                "data": [{"q": "test"}],
            },
        }

        config = validate_config(data)

        assert config.name == "Test"
        assert config.model.model_id == "dummy-v1"

    def test_validate_existing_config(self):
        """Test validating an existing ExperimentConfig."""
        original = create_example_config()
        validated = validate_config(original)

        assert validated is original  # Should return same object


class TestModelDump:
    """Tests for model serialization."""

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Requires Pydantic")
    def test_model_dump(self):
        """Test Pydantic model_dump."""
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_id="gpt-4",
            temperature=0.8,
        )

        dumped = config.model_dump()

        assert isinstance(dumped, dict)
        assert dumped["model_id"] == "gpt-4"
        assert dumped["temperature"] == 0.8
