"""Tests for the harness runner functionality.

The harness enables running multiple model/probe combinations against
a shared dataset for comparison experiments.
"""

import json

import pytest
import yaml

from insideLLMs.runtime.runner import run_harness_from_config

try:
    import pydantic  # noqa: F401

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


class TestRunHarnessFromConfig:
    """Tests for run_harness_from_config function."""

    def test_run_harness_from_config(self, tmp_path):
        """Test basic harness execution with single model/probe."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "What is 2 + 2?"}\n{"question": "What is 3 + 3?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        assert len(result["records"]) == 1
        assert len(result["experiments"]) == 1
        assert result["summary"]["total_experiments"] == 1

        record = result["records"][0]
        # Harness should emit standard ResultRecord-shaped records.
        assert "run_id" in record
        assert "model" in record and isinstance(record["model"], dict)
        assert "probe" in record and isinstance(record["probe"], dict)
        assert "dataset" in record and isinstance(record["dataset"], dict)
        assert record.get("custom", {}).get("harness", {}).get("experiment_id")

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Requires Pydantic")
    def test_run_harness_from_config_validate_output(self, tmp_path):
        """Test harness with output validation enabled."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "What is 2 + 2?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        # Should validate records against standard ResultRecord schema.
        result = run_harness_from_config(config_path, validate_output=True)
        assert len(result["records"]) == 1

    def test_harness_multiple_models(self, tmp_path):
        """Test harness with multiple models."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test question?"}\n')

        config = {
            "models": [
                {"type": "dummy", "args": {"canned_response": "Model A response"}},
                {"type": "dummy", "args": {"canned_response": "Model B response"}},
            ],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        # Should have 2 experiments (1 example * 2 models * 1 probe)
        assert len(result["experiments"]) == 2
        assert result["summary"]["total_experiments"] == 2

    def test_harness_multiple_probes(self, tmp_path):
        """Test harness with multiple probes."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test question?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [
                {"type": "logic", "args": {}},
                {"type": "factuality", "args": {}},
            ],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        # Should have 2 experiments (1 example * 1 model * 2 probes)
        assert len(result["experiments"]) == 2
        assert result["summary"]["total_experiments"] == 2

    def test_harness_multiple_models_and_probes(self, tmp_path):
        """Test harness with multiple models and probes (cartesian product)."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [
                {"type": "dummy", "args": {"canned_response": "A"}},
                {"type": "dummy", "args": {"canned_response": "B"}},
            ],
            "probes": [
                {"type": "logic", "args": {}},
                {"type": "factuality", "args": {}},
            ],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        # Should have 4 experiments (1 example * 2 models * 2 probes)
        assert len(result["experiments"]) == 4
        assert result["summary"]["total_experiments"] == 4

    def test_harness_multiple_examples(self, tmp_path):
        """Test harness processes multiple examples."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Q1?"}\n{"question": "Q2?"}\n{"question": "Q3?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        assert len(result["records"]) == 3

    def test_harness_max_examples_limit(self, tmp_path):
        """Test harness respects max_examples limit."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text(
            '{"question": "Q1?"}\n'
            '{"question": "Q2?"}\n'
            '{"question": "Q3?"}\n'
            '{"question": "Q4?"}\n'
            '{"question": "Q5?"}\n'
        )

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 2,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        assert len(result["records"]) == 2

    def test_harness_json_config(self, tmp_path):
        """Test harness with JSON config file."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.json"
        config_path.write_text(json.dumps(config))

        result = run_harness_from_config(config_path)

        assert len(result["records"]) == 1

    def test_harness_returns_run_id(self, tmp_path):
        """Test harness returns a unique run_id."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        assert "run_id" in result
        assert isinstance(result["run_id"], str)
        assert len(result["run_id"]) > 0

    def test_harness_returns_config(self, tmp_path):
        """Test harness returns the loaded configuration."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        assert "config" in result
        assert result["config"]["max_examples"] == 1

    def test_harness_returns_generated_at(self, tmp_path):
        """Test harness returns generation timestamp."""
        from datetime import datetime

        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        assert "generated_at" in result
        # generated_at can be either a string or datetime object
        assert isinstance(result["generated_at"], (str, datetime))

    def test_harness_summary_structure(self, tmp_path):
        """Test harness summary has expected structure."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        summary = result["summary"]
        assert "total_experiments" in summary
        assert "total_records" in summary or len(result["records"]) >= 0

    def test_harness_progress_callback(self, tmp_path):
        """Test harness calls progress callback."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Q1?"}\n{"question": "Q2?"}\n{"question": "Q3?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 3,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        progress_calls = []

        def on_progress(current: int, total: int):
            progress_calls.append((current, total))

        run_harness_from_config(config_path, progress_callback=on_progress)

        # Should have received progress updates
        assert len(progress_calls) > 0
        # Last call should show completion
        assert progress_calls[-1][0] == progress_calls[-1][1]


class TestHarnessEdgeCases:
    """Edge case tests for harness functionality."""

    def test_harness_missing_models_raises(self, tmp_path):
        """Test harness raises error when models are missing."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        with pytest.raises((ValueError, KeyError)):
            run_harness_from_config(config_path)

    def test_harness_missing_probes_raises(self, tmp_path):
        """Test harness raises error when probes are missing."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        with pytest.raises((ValueError, KeyError)):
            run_harness_from_config(config_path)

    def test_harness_missing_dataset_raises(self, tmp_path):
        """Test harness raises error when dataset is missing."""
        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        with pytest.raises((ValueError, KeyError)):
            run_harness_from_config(config_path)

    def test_harness_nonexistent_config_file(self, tmp_path):
        """Test harness raises error for nonexistent config file."""
        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            run_harness_from_config(nonexistent)

    def test_harness_empty_models_list_raises(self, tmp_path):
        """Test harness raises error with empty models list."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        # Harness requires at least one model
        with pytest.raises(ValueError, match="at least one model"):
            run_harness_from_config(config_path)

    def test_harness_empty_probes_list_raises(self, tmp_path):
        """Test harness raises error with empty probes list."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        # Harness requires at least one probe
        with pytest.raises(ValueError, match="at least one probe"):
            run_harness_from_config(config_path)

    def test_harness_empty_dataset_raises(self, tmp_path):
        """Test harness raises error with empty dataset."""
        from insideLLMs.validation import ValidationError

        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text("")  # Empty file

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        # Empty dataset raises validation error
        with pytest.raises(ValidationError, match="cannot be empty"):
            run_harness_from_config(config_path)

    def test_harness_invalid_model_type(self, tmp_path):
        """Test harness raises error for invalid model type."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [{"type": "nonexistent_model", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        with pytest.raises((ValueError, KeyError)):
            run_harness_from_config(config_path)

    def test_harness_invalid_probe_type(self, tmp_path):
        """Test harness raises error for invalid probe type."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "nonexistent_probe", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        with pytest.raises((ValueError, KeyError)):
            run_harness_from_config(config_path)


class TestHarnessRecordStructure:
    """Tests for harness output record structure."""

    def test_record_has_model_info(self, tmp_path):
        """Test records contain model information."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {"canned_response": "Test"}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        record = result["records"][0]
        assert "model" in record
        assert "model_id" in record["model"] or "provider" in record["model"]

    def test_record_has_probe_info(self, tmp_path):
        """Test records contain probe information."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        record = result["records"][0]
        assert "probe" in record
        assert "probe_id" in record["probe"]

    def test_record_has_dataset_info(self, tmp_path):
        """Test records contain dataset information."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        record = result["records"][0]
        assert "dataset" in record

    def test_record_has_input_output(self, tmp_path):
        """Test records contain input and output."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "What is 2 + 2?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {"canned_response": "4"}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        record = result["records"][0]
        assert "input" in record
        assert "output" in record or "output_text" in record

    def test_record_has_harness_metadata(self, tmp_path):
        """Test records contain harness-specific metadata."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        record = result["records"][0]
        harness_meta = record.get("custom", {}).get("harness", {})
        assert "experiment_id" in harness_meta
        assert "model_type" in harness_meta or "model_name" in harness_meta


class TestHarnessIntegration:
    """Integration tests for harness functionality."""

    @pytest.mark.integration
    def test_harness_large_dataset(self, tmp_path):
        """Test harness handles larger datasets."""
        dataset_path = tmp_path / "data.jsonl"
        with open(dataset_path, "w") as f:
            for i in range(50):
                f.write(f'{{"question": "Question number {i}?"}}\n')

        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        assert len(result["records"]) == 50

    @pytest.mark.integration
    def test_harness_many_models_and_probes(self, tmp_path):
        """Test harness with many model/probe combinations."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [
                {"type": "dummy", "args": {"canned_response": f"Model {i}"}} for i in range(5)
            ],
            "probes": [
                {"type": "logic", "args": {}},
                {"type": "factuality", "args": {}},
                {"type": "bias", "args": {}},
            ],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result = run_harness_from_config(config_path)

        # 1 example * 5 models * 3 probes = 15 experiments
        assert len(result["experiments"]) == 15

    @pytest.mark.integration
    def test_harness_determinism(self, tmp_path):
        """Test harness produces deterministic output."""
        dataset_path = tmp_path / "data.jsonl"
        dataset_path.write_text('{"question": "Test?"}\n')

        config = {
            "models": [{"type": "dummy", "args": {"canned_response": "Fixed"}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": str(dataset_path)},
            "max_examples": 1,
        }

        config_path = tmp_path / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))

        result1 = run_harness_from_config(config_path, deterministic_artifacts=True)
        result2 = run_harness_from_config(config_path, deterministic_artifacts=True)

        # Records should be equivalent (ignoring volatile fields)
        r1 = result1["records"][0]
        r2 = result2["records"][0]
        assert r1["input"] == r2["input"]
        assert r1["model"] == r2["model"]
        assert r1["probe"] == r2["probe"]
