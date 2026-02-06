"""Additional branch coverage for runtime._config_loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from insideLLMs.registry import NotFoundError
from insideLLMs.runtime import _config_loader as cfg_loader


def test_load_config_rejects_non_mapping_top_level(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected a mapping at top level"):
        cfg_loader.load_config(path)


def test_build_resolved_config_snapshot_hashing_exception_and_absolute_path_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    cfg_no_hash = {
        "dataset": {
            "format": "csv",
            "path": "relative.csv",
        }
    }
    cfg_with_hash = {
        "dataset": {
            "format": "csv",
            "path": "/outside/base/data.csv",
            "dataset_hash": "sha256:abc",
        }
    }
    logged = {"called": False}

    def _boom_open(*args, **kwargs):
        raise RuntimeError("unexpected read failure")

    def _record_error(msg: str, **kwargs):
        _ = kwargs
        logged["called"] = "Unexpected error hashing dataset" in msg

    monkeypatch.setattr("builtins.open", _boom_open)
    monkeypatch.setattr(cfg_loader.logger, "error", _record_error)

    cfg_loader._build_resolved_config_snapshot(cfg_no_hash, tmp_path)
    assert logged["called"]

    snapshot = cfg_loader._build_resolved_config_snapshot(cfg_with_hash, tmp_path)
    assert snapshot["dataset"]["path"] == "data.csv"


def test_resolve_determinism_options_branches_and_validation_errors():
    strict, artifacts = cfg_loader._resolve_determinism_options(
        {"determinism": {"strict_serialization": False, "deterministic_artifacts": True}},
        strict_override=None,
        deterministic_artifacts_override=None,
    )
    assert strict is False
    assert artifacts is True

    strict2, artifacts2 = cfg_loader._resolve_determinism_options(
        {"determinism": {"strict_serialization": True}},
        strict_override=None,
        deterministic_artifacts_override=None,
    )
    assert strict2 is True
    assert artifacts2 is True

    strict3, artifacts3 = cfg_loader._resolve_determinism_options(
        {"determinism": {"strict_serialization": True, "deterministic_artifacts": None}},
        strict_override=False,
        deterministic_artifacts_override=False,
    )
    assert strict3 is False
    assert artifacts3 is False

    with pytest.raises(ValueError, match="strict_serialization must be a bool"):
        cfg_loader._resolve_determinism_options(
            {"determinism": {"strict_serialization": "yes"}},
            strict_override=None,
            deterministic_artifacts_override=None,
        )

    with pytest.raises(ValueError, match="deterministic_artifacts must be a bool"):
        cfg_loader._resolve_determinism_options(
            {"determinism": {"deterministic_artifacts": "yes"}},
            strict_override=None,
            deterministic_artifacts_override=None,
        )


def test_extract_probe_kwargs_non_dict_and_invalid_mapping():
    assert cfg_loader._extract_probe_kwargs_from_config(None) == {}

    with pytest.raises(ValueError, match="generation must be a mapping"):
        cfg_loader._extract_probe_kwargs_from_config({"generation": "bad"})


def test_create_middlewares_from_config_validation_branches():
    assert cfg_loader._create_middlewares_from_config(None) == []

    with pytest.raises(ValueError, match="must be a list"):
        cfg_loader._create_middlewares_from_config("not-a-list")

    with pytest.raises(ValueError, match="missing 'type'"):
        cfg_loader._create_middlewares_from_config([{"args": {"x": 1}}])

    with pytest.raises(ValueError, match="Unsupported middleware config entry"):
        cfg_loader._create_middlewares_from_config([123])

    with pytest.raises(ValueError, match="Unknown middleware type"):
        cfg_loader._create_middlewares_from_config([{"type": "nope"}])


def test_create_model_and_probe_fallback_paths(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        cfg_loader.model_registry,
        "get_factory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(NotFoundError("missing")),
    )
    monkeypatch.setattr(
        cfg_loader.probe_registry,
        "get_factory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(NotFoundError("missing")),
    )

    model = cfg_loader._create_model_from_config({"type": "dummy", "pipeline": {"middleware": []}})
    probe = cfg_loader._create_probe_from_config({"type": "logic"})

    assert model is not None
    assert probe is not None


def test_load_dataset_from_config_fallback_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        cfg_loader.dataset_registry,
        "get_factory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(NotFoundError("missing")),
    )

    from insideLLMs import dataset_utils

    monkeypatch.setattr(dataset_utils, "load_csv_dataset", lambda path: [f"csv:{path}"])
    monkeypatch.setattr(dataset_utils, "load_jsonl_dataset", lambda path: [f"jsonl:{path}"])
    monkeypatch.setattr(
        dataset_utils,
        "load_hf_dataset",
        lambda name, split="test", **kwargs: [f"hf:{name}:{split}:{sorted(kwargs.items())}"],
    )

    csv_data = cfg_loader._load_dataset_from_config(
        {"format": "csv", "path": "data.csv"}, tmp_path
    )
    jsonl_data = cfg_loader._load_dataset_from_config(
        {"format": "jsonl", "path": "data.jsonl"}, tmp_path
    )
    hf_data = cfg_loader._load_dataset_from_config(
        {"format": "hf", "name": "dataset-x", "split": "train", "revision": "abc"},
        tmp_path,
    )

    assert csv_data and csv_data[0].startswith("csv:")
    assert jsonl_data and jsonl_data[0].startswith("jsonl:")
    assert hf_data and hf_data[0].startswith("hf:dataset-x:train:")
