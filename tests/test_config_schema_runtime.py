"""Configuration contracts exercised through the same CLI a new user runs."""

import json
from pathlib import Path

import pytest
import yaml

from insideLLMs.cli import main
from insideLLMs.config import load_config as load_public_config
from insideLLMs.config_schema import RuntimeConfiguration, normalize_runtime_config
from insideLLMs.runtime._config_loader import load_config


@pytest.mark.parametrize("template", ["basic", "benchmark", "tracking", "full", "harness"])
@pytest.mark.parametrize("suffix", ["yaml", "json"])
def test_generated_template_validates_and_runs_from_unrelated_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, template: str, suffix: str
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "nested" / f"{template}.{suffix}"
    assert main(["init", str(config_path), "--template", template, "--quiet"]) == 0
    assert main(["validate", str(config_path)]) == 0
    assert isinstance(load_public_config(config_path), RuntimeConfiguration)
    config = load_config(config_path)
    assert config["config_version"] == "1"
    assert (config_path.parent / config["dataset"]["path"]).is_file()
    command = "harness" if template == "harness" else "run"
    run_dir = tmp_path / "run"
    options = ["--skip-report"] if command == "harness" else []
    assert main([command, str(config_path), "--run-dir", str(run_dir), *options]) == 0
    assert main(["validate", str(run_dir)]) == 0


@pytest.mark.parametrize("block", ["benchmark", "tracking", "async", "output"])
def test_unimplemented_blocks_rejected_by_both_loader_and_cli(tmp_path: Path, block: str) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"type": "dummy"},
                "probe": {"type": "logic"},
                "dataset": {"format": "inline", "data": []},
                block: {"enabled": True},
            }
        )
    )
    with pytest.raises(ValueError, match=block):
        load_config(config_path)
    assert main(["validate", str(config_path)]) == 1


def test_harness_dataset_resolves_relative_to_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path(__file__).resolve().parents[1] / "ci" / "harness.yaml"
    assert main(["validate", str(config_path)]) == 0


def test_legacy_public_builder_runs_after_explicit_conversion(tmp_path: Path) -> None:
    legacy = {
        "name": "old example",
        "model": {"provider": "dummy", "model_id": "demo"},
        "probe": {"type": "logic", "params": {}},
        "dataset": {"source": "inline", "data": [{"question": "2+2?"}]},
    }
    with pytest.warns(DeprecationWarning):
        converted = normalize_runtime_config(legacy)
    assert converted["model"] == {"type": "dummy", "args": {"name": "demo"}}
    assert converted["dataset"]["format"] == "inline"
    config_path = tmp_path / "legacy.json"
    config_path.write_text(json.dumps(legacy))
    with pytest.warns(DeprecationWarning):
        assert main(["run", str(config_path), "--run-dir", str(tmp_path / "run")]) == 0


def test_missing_dataset_is_validation_error(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"type": "dummy"},
                "probe": {"type": "logic"},
                "dataset": {"format": "jsonl", "path": "absent.jsonl"},
            }
        )
    )
    assert main(["validate", str(config_path)]) == 1


def test_unknown_config_version_rejected() -> None:
    with pytest.raises(ValueError, match="config_version"):
        normalize_runtime_config(
            {
                "config_version": "2",
                "model": {"type": "dummy"},
                "probe": {"type": "logic"},
                "dataset": {"format": "inline", "data": []},
            }
        )


def test_validation_errors_do_not_echo_secret_values() -> None:
    with pytest.raises(ValueError) as error:
        normalize_runtime_config(
            {
                "model": {"type": "dummy", "api_key": "SENTINEL_PRIVATE_KEY"},
                "probe": {"type": "logic"},
                "dataset": {"format": "inline", "data": []},
            }
        )
    assert "SENTINEL_PRIVATE_KEY" not in str(error.value)


def test_legacy_conversion_rejects_unconsumed_model_settings() -> None:
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="unsupported_setting"):
        normalize_runtime_config(
            {
                "model": {"provider": "dummy", "model_id": "demo", "unsupported_setting": True},
                "probe": {"type": "logic"},
                "dataset": {"source": "inline", "data": []},
            }
        )
