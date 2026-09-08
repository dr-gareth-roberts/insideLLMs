"""Accepted execution settings must have runtime meaning."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from insideLLMs.cli import main
from insideLLMs.config_schema import normalize_runtime_config
from insideLLMs.runtime.runner import run_experiment_from_config, run_harness_from_config


@pytest.mark.parametrize("harness", [False, True])
@pytest.mark.parametrize(
    ("component", "setting", "value"),
    [
        ("model", "generation", {"max_tokens": 7}),
        ("model", "probe_kwargs", {"max_tokens": 7}),
        ("model", "run_kwargs", {"max_tokens": 7}),
        ("model", "id", "candidate"),
        ("probe", "pipeline", {"middlewares": ["trace"]}),
        ("probe", "id", "logic_a"),
    ],
)
def test_unused_component_settings_fail_validation_and_runtime_before_execution(
    tmp_path: Path, harness: bool, component: str, setting: str, value: object
) -> None:
    config = {
        "model": {"type": "dummy"},
        "probe": {"type": "logic"},
        "dataset": {"format": "inline", "data": ["question"]},
    }
    config[component][setting] = value
    if harness:
        config["models"] = [config.pop("model")]
        config["probes"] = [config.pop("probe")]
    config_path = tmp_path / "unsupported.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["validate", str(config_path)]) == 1
    function = run_harness_from_config if harness else run_experiment_from_config
    with patch("insideLLMs.runtime._high_level._create_model_from_config") as create_model:
        with pytest.raises(ValueError, match=f"Unsupported {component} settings"):
            function(config_path)
        create_model.assert_not_called()


@pytest.mark.parametrize("format_name", ["inline", "jsonl", "csv"])
@pytest.mark.parametrize(
    "setting", ["sample_size", "shuffle", "seed", "input_field", "misspelled_option"]
)
def test_local_dataset_ignored_settings_fail_validation_and_runtime(
    tmp_path: Path, format_name: str, setting: str
) -> None:
    config = {
        "model": {"type": "dummy"},
        "probe": {"type": "logic"},
        "dataset": {
            "format": format_name,
            "data": ["first", "second"],
            "path": "data.jsonl",
            setting: 1,
        },
    }
    config_path = tmp_path / "unsupported.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["validate", str(config_path)]) == 1
    with pytest.raises(ValueError, match=f"Unsupported {format_name} dataset settings"):
        run_experiment_from_config(config_path)


def test_local_dataset_provenance_metadata_remains_supported() -> None:
    config = {
        "model": {"type": "dummy"},
        "probe": {"type": "logic", "generation": {"max_tokens": 7}},
        "dataset": {
            "format": "inline",
            "data": ["question"],
            "name": "my-data",
            "dataset_hash": "sha256:example",
            "dataset_version": "1",
            "provenance": "curated",
        },
    }
    assert normalize_runtime_config(config) == config


def test_huggingface_loader_extras_remain_supported() -> None:
    config = {
        "model": {"type": "dummy"},
        "probe": {"type": "logic"},
        "dataset": {"format": "hf", "name": "dataset", "split": "train", "revision": "abc"},
    }
    assert normalize_runtime_config(config) == config


def test_actual_legacy_builder_removes_inapplicable_hf_split_default() -> None:
    from insideLLMs.config import DatasetConfig, ExperimentConfig, ModelConfig, ProbeConfig

    config = ExperimentConfig(
        name="legacy",
        model=ModelConfig(provider="dummy", model_id="dummy"),
        probe=ProbeConfig(type="logic"),
        dataset=DatasetConfig(source="inline", data=[{"question": "Question"}]),
    )
    with pytest.warns(DeprecationWarning):
        normalized = config.to_runtime_config()
    assert normalized["dataset"]["format"] == "inline"
    assert normalized["dataset"]["data"] == [{"question": "Question"}]
    assert "split" not in normalized["dataset"]
