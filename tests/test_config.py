"""Tests for configuration validation module."""

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
    save_config_to_yaml,
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


class TestYamlIO:
    """Tests for YAML configuration I/O."""

    def test_load_yaml_config(self):
        """Test loading YAML configuration from manually created YAML."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        from insideLLMs.config import load_config_from_yaml

        # Create a simple YAML config manually (avoiding enum serialization issues)
        yaml_content = """
name: "YAML Test"
model:
  provider: "dummy"
  model_id: "dummy-v1"
  temperature: 0.7
probe:
  type: "logic"
dataset:
  source: "inline"
  data:
    - q: "test?"
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write(yaml_content)
            path = f.name

        try:
            loaded = load_config_from_yaml(path)
            assert loaded.name == "YAML Test"
            assert loaded.model.model_id == "dummy-v1"
        finally:
            Path(path).unlink()

    def test_load_config_auto_detect_yaml(self):
        """Test auto-detection of YAML format."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        yaml_content = """
name: "Auto Detect Test"
model:
  provider: "dummy"
  model_id: "test-model"
probe:
  type: "logic"
dataset:
  source: "inline"
  data:
    - q: "test"
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write(yaml_content)
            path = f.name

        try:
            loaded = load_config(path)
            assert loaded.name == "Auto Detect Test"
        finally:
            Path(path).unlink()

    def test_load_config_yml_extension(self):
        """Test loading .yml extension."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        yaml_content = """
name: "YML Extension Test"
model:
  provider: "dummy"
  model_id: "test"
probe:
  type: "logic"
dataset:
  source: "inline"
  data:
    - q: "test"
"""
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as f:
            f.write(yaml_content)
            path = f.name

        try:
            loaded = load_config(path)
            assert loaded.name == "YML Extension Test"
        finally:
            Path(path).unlink()


class TestRequirePydantic:
    """Tests for _require_pydantic function."""

    def test_require_pydantic_when_available(self):
        """Test _require_pydantic when Pydantic is available."""
        from insideLLMs.config import PYDANTIC_AVAILABLE, _require_pydantic

        if PYDANTIC_AVAILABLE:
            # Should not raise
            _require_pydantic()


class TestConfigFromDict:
    """Tests for creating configs from dicts."""

    def test_config_from_nested_dict(self):
        """Test creating config from nested dict."""
        data = {
            "name": "Nested Test",
            "model": {
                "provider": "dummy",
                "model_id": "dummy-v1",
                "temperature": 0.5,
                "max_tokens": 1000,
            },
            "probe": {
                "type": "logic",
                "name": "Logic Test",
                "params": {"difficulty": "hard"},
            },
            "dataset": {
                "source": "inline",
                "data": [{"q": "test?", "a": "answer"}],
            },
            "runner": {
                "concurrency": 4,
                "output_dir": "custom_output",
            },
        }

        config = validate_config(data)

        assert config.name == "Nested Test"
        assert config.model.temperature == 0.5
        assert config.model.max_tokens == 1000
        assert config.probe.params["difficulty"] == "hard"
        assert config.runner.concurrency == 4


class TestModelProviderEnum:
    """Tests for ModelProvider enum."""

    def test_provider_values(self):
        """Test ModelProvider enum values."""
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.HUGGINGFACE.value == "huggingface"
        assert ModelProvider.DUMMY.value == "dummy"
        assert ModelProvider.CUSTOM.value == "custom"


class TestProbeTypeEnum:
    """Tests for ProbeType enum."""

    def test_probe_type_values(self):
        """Test ProbeType enum values."""
        assert ProbeType.LOGIC.value == "logic"
        assert ProbeType.FACTUALITY.value == "factuality"
        assert ProbeType.BIAS.value == "bias"
        assert ProbeType.ATTACK.value == "attack"
        assert ProbeType.CUSTOM.value == "custom"


class TestModelConfigOptionalFields:
    """Tests for ModelConfig optional fields."""

    def test_api_base(self):
        """Test api_base field."""
        config = ModelConfig(
            provider="openai" if not PYDANTIC_AVAILABLE else ModelProvider.OPENAI,
            model_id="gpt-4",
            api_base="https://custom-endpoint.com",
        )
        assert config.api_base == "https://custom-endpoint.com"

    def test_extra_params(self):
        """Test extra_params field."""
        config = ModelConfig(
            provider="openai" if not PYDANTIC_AVAILABLE else ModelProvider.OPENAI,
            model_id="gpt-4",
            extra_params={"top_p": 0.9, "seed": 42},
        )
        assert config.extra_params["top_p"] == 0.9
        assert config.extra_params["seed"] == 42


class TestProbeConfigOptionalFields:
    """Tests for ProbeConfig optional fields."""

    def test_timeout_per_item(self):
        """Test timeout_per_item field."""
        config = ProbeConfig(
            type="logic" if not PYDANTIC_AVAILABLE else ProbeType.LOGIC,
            timeout_per_item=60.0,
        )
        assert config.timeout_per_item == 60.0

    def test_stop_on_error(self):
        """Test stop_on_error field."""
        config = ProbeConfig(
            type="logic" if not PYDANTIC_AVAILABLE else ProbeType.LOGIC,
            stop_on_error=True,
        )
        assert config.stop_on_error is True

    def test_description(self):
        """Test description field."""
        config = ProbeConfig(
            type="logic" if not PYDANTIC_AVAILABLE else ProbeType.LOGIC,
            description="Tests logical reasoning capabilities",
        )
        assert config.description == "Tests logical reasoning capabilities"


class TestDatasetConfigOptionalFields:
    """Tests for DatasetConfig optional fields."""

    def test_sample_size(self):
        """Test sample_size field."""
        config = DatasetConfig(
            source="inline",
            data=[{"q": "test"}],
            sample_size=10,
        )
        assert config.sample_size == 10

    def test_shuffle_and_seed(self):
        """Test shuffle and seed fields."""
        config = DatasetConfig(
            source="inline",
            data=[{"q": "test"}],
            shuffle=True,
            seed=42,
        )
        assert config.shuffle is True
        assert config.seed == 42


class TestRunnerConfigOptionalFields:
    """Tests for RunnerConfig optional fields."""

    def test_save_intermediate(self):
        """Test save_intermediate field."""
        config = RunnerConfig(save_intermediate=True)
        assert config.save_intermediate is True

    def test_cache_responses(self):
        """Test cache_responses field."""
        config = RunnerConfig(cache_responses=True)
        assert config.cache_responses is True


class TestExperimentConfigOptionalFields:
    """Tests for ExperimentConfig optional fields."""

    def test_description(self):
        """Test description field."""
        config = ExperimentConfig(
            name="Test",
            description="A test experiment",
            model=ModelConfig(
                provider="dummy" if not PYDANTIC_AVAILABLE else ModelProvider.DUMMY,
                model_id="dummy-v1",
            ),
            probe=ProbeConfig(
                type="logic" if not PYDANTIC_AVAILABLE else ProbeType.LOGIC,
            ),
            dataset=DatasetConfig(source="inline", data=[{}]),
        )
        assert config.description == "A test experiment"


class TestSaveConfigDeterminism:
    """Determinism checks for config serialization helpers."""

    def test_save_config_to_yaml_sorts_keys(self, tmp_path: Path):
        """YAML output should be stable across runs."""
        config = create_example_config()
        out_path = tmp_path / "config.yaml"

        save_config_to_yaml(config, out_path)
        content = out_path.read_text(encoding="utf-8")

        # Spot-check ordering to catch accidental non-determinism.
        assert "dataset:" in content
        assert "model:" in content
        assert content.index("dataset:") < content.index("model:")
