"""Branch-focused tests for runtime._high_level helpers."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from types import SimpleNamespace

import pytest

from insideLLMs._serialization import StrictSerializationError
from insideLLMs.runtime import _high_level as hl
from insideLLMs.types import ProbeResult, ResultStatus


class _DummyModel:
    model_id = "dummy-model"
    name = "dummy-model"

    def info(self):
        return {"model_id": "dummy-model", "provider": "dummy", "name": "dummy-model"}


class _DummyProbe:
    name = "dummy-probe"
    category = "custom-category"

    def score(self, results):
        _ = results
        return None


class _FakeRunner:
    def __init__(self, model, probe):
        _ = model, probe

    def run(self, dataset, **kwargs):
        _ = dataset, kwargs
        return [{"input": "q", "output": "a", "status": "success", "latency_ms": 1.0}]


def _minimal_harness_config(dataset_cfg: dict) -> dict:
    return {
        "models": [{"type": "dummy"}],
        "probes": [{"type": "logic"}],
        "dataset": dataset_cfg,
    }


def test_run_experiment_from_config_wraps_strict_serialization_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(hl, "load_config", lambda _: {"model": {}, "probe": {}, "dataset": {}})
    monkeypatch.setattr(hl, "_build_resolved_config_snapshot", lambda config, _: config)
    monkeypatch.setattr(hl, "_resolve_determinism_options", lambda *_args, **_kwargs: (True, True))
    monkeypatch.setattr(
        hl,
        "_deterministic_run_id_from_config_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StrictSerializationError("boom")),
    )

    with pytest.raises(ValueError, match="strict_serialization requires JSON-stable values"):
        hl.run_experiment_from_config("config.yaml", emit_run_artifacts=False)


@pytest.mark.asyncio
async def test_run_experiment_from_config_async_wraps_strict_serialization_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(hl, "load_config", lambda _: {"model": {}, "probe": {}, "dataset": {}})
    monkeypatch.setattr(hl, "_build_resolved_config_snapshot", lambda config, _: config)
    monkeypatch.setattr(hl, "_resolve_determinism_options", lambda *_args, **_kwargs: (True, True))
    monkeypatch.setattr(
        hl,
        "_deterministic_run_id_from_config_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StrictSerializationError("boom")),
    )

    with pytest.raises(ValueError, match="strict_serialization requires JSON-stable values"):
        await hl.run_experiment_from_config_async("config.yaml", emit_run_artifacts=False)


def test_run_harness_from_config_wraps_strict_serialization_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(hl, "load_config", lambda _: _minimal_harness_config({"format": "jsonl"}))
    monkeypatch.setattr(hl, "_build_resolved_config_snapshot", lambda config, _: config)
    monkeypatch.setattr(hl, "_resolve_determinism_options", lambda *_args, **_kwargs: (True, True))
    monkeypatch.setattr(
        hl,
        "_deterministic_run_id_from_config_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StrictSerializationError("boom")),
    )

    with pytest.raises(ValueError, match="strict_serialization requires JSON-stable values"):
        hl.run_harness_from_config("config.yaml")


def test_run_harness_dataset_version_composition_paths(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("insideLLMs.runtime._sync_runner.ProbeRunner", _FakeRunner)
    monkeypatch.setattr(hl, "_create_model_from_config", lambda *_args, **_kwargs: _DummyModel())
    monkeypatch.setattr(hl, "_create_probe_from_config", lambda *_args, **_kwargs: _DummyProbe())
    monkeypatch.setattr(hl, "_load_dataset_from_config", lambda *_args, **_kwargs: ["item"])
    monkeypatch.setattr(hl, "_build_resolved_config_snapshot", lambda config, _: config)
    monkeypatch.setattr(hl, "_resolve_determinism_options", lambda *_args, **_kwargs: (True, True))
    monkeypatch.setattr(hl, "_deterministic_run_id_from_config_snapshot", lambda *_args, **_kwargs: "run-1")
    monkeypatch.setattr(hl, "_deterministic_harness_experiment_id", lambda **_kwargs: "exp-1")
    monkeypatch.setattr(hl, "_deterministic_base_time", lambda *_args, **_kwargs: datetime(2025, 1, 1))
    monkeypatch.setattr(hl, "_deterministic_run_times", lambda *_args, **_kwargs: (datetime(2025, 1, 1), datetime(2025, 1, 1)))
    monkeypatch.setattr(hl, "_deterministic_item_times", lambda *_args, **_kwargs: (datetime(2025, 1, 1), datetime(2025, 1, 1)))
    monkeypatch.setattr(hl, "generate_summary_report", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(
        hl,
        "_build_result_record",
        lambda **kwargs: {"dataset": kwargs["dataset"], "custom": {}, "status": kwargs["status"]},
    )
    monkeypatch.setattr(
        hl,
        "create_experiment_result",
        lambda *args, **kwargs: SimpleNamespace(experiment_id="exp-1"),
    )

    monkeypatch.setattr(
        hl,
        "load_config",
        lambda _: _minimal_harness_config(
            {
                "format": "jsonl",
                "path": "data.jsonl",
                "revision": "r1",
                "split": "test",
            }
        ),
    )
    no_version = hl.run_harness_from_config("config.yaml")
    version_derived = no_version["records"][0]["dataset"]["dataset_version"]
    assert version_derived == "r1::test"

    monkeypatch.setattr(
        hl,
        "load_config",
        lambda _: _minimal_harness_config(
            {
                "format": "jsonl",
                "path": "data.jsonl",
                "version": "v1",
                "revision": "r1",
                "split": "train",
            }
        ),
    )
    with_version = hl.run_harness_from_config("config.yaml")
    version_composed = with_version["records"][0]["dataset"]["dataset_version"]
    assert version_composed == "v1@r1::train"


def test_create_experiment_result_status_category_and_strict_error_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    class _InvalidStatus(Enum):
        BAD = "not-a-status"

    probe = _DummyProbe()
    model = _DummyModel()
    output = hl.create_experiment_result(
        model=model,
        probe=probe,
        results=[{"input": "x", "output": "y", "status": _InvalidStatus.BAD}],
        experiment_id="exp-id",
    )
    assert output.results[0].status == ResultStatus.ERROR
    assert output.probe_category.value == "custom"

    probe_results = [ProbeResult(input="i", output="o", status=ResultStatus.SUCCESS, latency_ms=1.0)]
    passthrough = hl.create_experiment_result(model=model, probe=probe, results=probe_results)
    assert passthrough.results == probe_results
    assert passthrough.started_at is not None
    assert passthrough.completed_at is not None

    monkeypatch.setattr(
        hl,
        "_deterministic_hash",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StrictSerializationError("bad")),
    )
    with pytest.raises(ValueError, match="strict_serialization requires JSON-stable inputs"):
        hl.create_experiment_result(
            model=model,
            probe=probe,
            results=[{"input": "x", "output": "y", "status": "success"}],
            strict_serialization=True,
        )


def test_derive_run_id_from_config_path_wraps_strict_serialization_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(hl, "load_config", lambda _: {"dataset": {"format": "jsonl"}})
    monkeypatch.setattr(hl, "_build_resolved_config_snapshot", lambda config, _: config)
    monkeypatch.setattr(hl, "_resolve_determinism_options", lambda *_args, **_kwargs: (True, True))
    monkeypatch.setattr(
        hl,
        "_deterministic_run_id_from_config_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StrictSerializationError("bad")),
    )

    with pytest.raises(ValueError, match="strict_serialization requires JSON-stable values"):
        hl.derive_run_id_from_config_path("config.yaml")
