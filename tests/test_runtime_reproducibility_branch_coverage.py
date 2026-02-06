"""Additional branch coverage for runtime.reproducibility."""

from __future__ import annotations

import sys
import types

import pytest

from insideLLMs.runtime.reproducibility import (
    ConfigSnapshot,
    ConfigVersionManager,
    EnvironmentCapture,
    EnvironmentInfo,
    ExperimentCheckpointManager,
    ExperimentMetadata,
    ExperimentReplayManager,
    ExperimentSnapshot,
    ReplayResult,
    ReproducibilityChecker,
    SeedManager,
    SeedState,
    SnapshotFormat,
)


def _make_snapshot(snapshot_id: str = "snap-1") -> ExperimentSnapshot:
    return ExperimentSnapshot(
        snapshot_id=snapshot_id,
        metadata=ExperimentMetadata(experiment_id="exp-1", name="test-exp"),
        config=ConfigSnapshot(config={"a": 1}, config_hash="cfg123"),
        seed_state=SeedState(global_seed=42),
        environment=EnvironmentInfo(
            python_version="3.12",
            platform_info="test-platform",
            os_info="test-os",
        ),
        inputs={"in": 1},
        outputs={"out": 2},
    )


def test_replay_result_to_dict_and_missing_snapshot_verify():
    result = ReplayResult(success=True, original_hash="a", replay_hash="b", matches=False)
    as_dict = result.to_dict()
    assert as_dict["original_hash"] == "a"
    assert as_dict["matches"] is False

    manager = ExperimentReplayManager()
    missing = manager.verify_replay("missing", {"x": 1})
    assert missing.success is False
    assert missing.error == "Snapshot not found"


def test_seed_manager_numpy_import_error_branch(monkeypatch: pytest.MonkeyPatch):
    manager = SeedManager(global_seed=7)
    monkeypatch.setitem(sys.modules, "numpy", None)
    assert manager.set_numpy_seed() is False


def test_seed_manager_torch_and_tensorflow_success_paths(monkeypatch: pytest.MonkeyPatch):
    manager = SeedManager(global_seed=11)
    manual_calls: list[int] = []
    cuda_calls: list[int] = []
    tf_calls: list[int] = []

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def manual_seed_all(self, seed: int) -> None:
            cuda_calls.append(seed)

    fake_torch = types.SimpleNamespace(
        manual_seed=lambda seed: manual_calls.append(seed),
        cuda=FakeCuda(),
        get_rng_state=lambda: types.SimpleNamespace(
            numpy=lambda: types.SimpleNamespace(tobytes=lambda: b"state-bytes")
        ),
    )
    fake_tf = types.SimpleNamespace(
        random=types.SimpleNamespace(set_seed=lambda seed: tf_calls.append(seed))
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    assert manager.set_torch_seed() is True
    assert manager.set_tensorflow_seed() is True
    assert manual_calls == [11]
    assert cuda_calls == [11]
    assert tf_calls == [11]


def test_seed_manager_capture_state_handles_numpy_import_error(monkeypatch: pytest.MonkeyPatch):
    manager = SeedManager(global_seed=3)
    monkeypatch.setitem(sys.modules, "numpy", None)
    state = manager.get_state()
    assert state.global_seed == 3


def test_environment_capture_installed_packages_exception_path(monkeypatch: pytest.MonkeyPatch):
    capture = EnvironmentCapture()
    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata, "distributions", lambda: (_ for _ in ()).throw(Exception("boom"))
    )
    packages = capture._get_installed_packages()
    assert packages == {}


def test_environment_compare_env_var_differences_branch():
    capture = EnvironmentCapture()
    env1 = EnvironmentInfo(
        python_version="3.12",
        platform_info="p",
        os_info="o",
        env_vars={"A": "1", "B": "2"},
    )
    env2 = EnvironmentInfo(
        python_version="3.12",
        platform_info="p",
        os_info="o",
        env_vars={"A": "9", "C": "3"},
    )
    diff = capture.compare(env1, env2)
    assert "A" in diff["env_var_differences"]
    assert "B" in diff["env_var_differences"]
    assert "C" in diff["env_var_differences"]


def test_snapshot_save_yaml_success_and_yaml_import_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    snapshot = _make_snapshot("snap-yaml")

    def safe_dump(data, fh, **_kwargs):
        fh.write(f"snapshot_id: {data['snapshot_id']}\n")

    monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_dump=safe_dump))
    yaml_path = tmp_path / "snapshot.yaml"
    snapshot.save(str(yaml_path), format=SnapshotFormat.YAML)
    assert "snapshot_id: snap-yaml" in yaml_path.read_text(encoding="utf-8")

    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(RuntimeError, match="PyYAML not installed"):
        snapshot.save(str(tmp_path / "snapshot2.yaml"), format=SnapshotFormat.YAML)


def test_config_version_manager_get_current_none_and_nested_diff():
    manager = ConfigVersionManager()
    assert manager.get_current() is None

    manager.add_version("v1", {"a": 1, "nested": {"x": 1, "y": 2}})
    manager.add_version("v2", {"nested": {"x": 2}, "b": 3})
    diff = manager.diff("v1", "v2")

    assert diff["removed"]["a"] == 1
    assert diff["removed"]["nested.y"] == 2
    assert diff["added"]["b"] == 3
    assert diff["changed"]["nested.x"] == {"old": 1, "new": 2}


def test_replay_manager_load_snapshot_and_checkpoint_restore_none(tmp_path):
    snapshot = _make_snapshot("snap-load")
    json_path = tmp_path / "snap.json"
    snapshot.save(str(json_path), format=SnapshotFormat.JSON)

    replay_manager = ExperimentReplayManager()
    loaded = replay_manager.load_snapshot(str(json_path))
    assert loaded.snapshot_id == "snap-load"

    ckpt_manager = ExperimentCheckpointManager()
    assert ckpt_manager.restore_checkpoint("missing") is None


def test_reproducibility_checker_warning_for_unversioned_config():
    checker = ReproducibilityChecker()
    seed_manager = SeedManager(global_seed=99)
    seed_manager.set_python_seed()

    report = checker.check(snapshot=_make_snapshot("snap-check"), seed_manager=seed_manager)
    assert "Configuration not versioned" in report.warnings
    assert any("ConfigVersionManager" in rec for rec in report.recommendations)
