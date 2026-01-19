from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from insideLLMs.experiment_tracking import (
    ExperimentTracker,
    LocalFileTracker,
    MultiTracker,
    TrackingConfig,
    auto_track,
    create_tracker,
)


class CapturingTracker(ExperimentTracker):
    """A minimal in-memory tracker for testing the base API behaviour."""

    def __init__(self, config: TrackingConfig | None = None):
        super().__init__(config=config)
        self.started: list[dict[str, Any]] = []
        self.ended: list[str] = []
        self.metrics_calls: list[tuple[dict[str, float], int | None]] = []
        self.params_calls: list[dict[str, Any]] = []
        self.artifacts_calls: list[tuple[str, str | None, str | None]] = []

    def start_run(
        self,
        run_name: str | None = None,
        run_id: str | None = None,
        nested: bool = False,
    ) -> str:
        self._run_active = True
        self._run_id = run_id or "run-id"
        self.started.append({"run_name": run_name, "run_id": self._run_id, "nested": nested})
        return self._run_id

    def end_run(self, status: str = "finished") -> None:
        self._run_active = False
        self.ended.append(status)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics_calls.append((metrics, step))

    def log_params(self, params: dict[str, Any]) -> None:
        self.params_calls.append(params)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: str | None = None,
        artifact_type: str | None = None,
    ) -> None:
        self.artifacts_calls.append((artifact_path, artifact_name, artifact_type))


@dataclass(frozen=True)
class DummyModelInfo:
    name: str
    provider: str


@dataclass(frozen=True)
class DummyProbeCategory:
    value: str


@dataclass(frozen=True)
class DummyScore:
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    mean_latency_ms: float | None = None
    total_tokens: int | None = None
    error_rate: float | None = None


@dataclass(frozen=True)
class DummyExperimentResult:
    success_rate: float
    total_count: int
    success_count: int
    error_count: int
    score: DummyScore | None
    duration_seconds: float | None
    experiment_id: str
    model_info: DummyModelInfo
    probe_name: str
    probe_category: DummyProbeCategory


def test_tracking_config_defaults() -> None:
    cfg = TrackingConfig()
    assert cfg.project == "insideLLMs"
    assert cfg.experiment_name is None
    assert cfg.tags == []
    assert cfg.notes is None
    assert cfg.log_artifacts is True
    assert cfg.log_code is False
    assert cfg.auto_log_metrics is True


def test_experiment_tracker_context_manager_calls_start_and_end() -> None:
    tracker = CapturingTracker()
    assert tracker.started == []
    assert tracker.ended == []

    with tracker as t:
        assert t is tracker
        t.log_metrics({"x": 1.0})

    assert len(tracker.started) == 1
    assert tracker.ended == ["finished"]


def test_log_experiment_result_emits_expected_metrics_and_params() -> None:
    tracker = CapturingTracker()
    tracker.start_run(run_name="r")

    result = DummyExperimentResult(
        success_rate=0.75,
        total_count=4,
        success_count=3,
        error_count=1,
        score=DummyScore(
            accuracy=0.5,
            precision=None,
            recall=0.25,
            f1_score=0.33,
            mean_latency_ms=12.0,
            total_tokens=123,
            error_rate=0.1,
        ),
        duration_seconds=1.23,
        experiment_id="exp-1",
        model_info=DummyModelInfo(name="model-A", provider="provider-X"),
        probe_name="probe-foo",
        probe_category=DummyProbeCategory(value="behaviour"),
    )

    tracker.log_experiment_result(result, prefix="harness/")

    assert len(tracker.metrics_calls) == 1
    metrics, step = tracker.metrics_calls[0]
    assert step is None  # ExperimentTracker.log_experiment_result doesn't pass an explicit step

    # Basic counters
    assert metrics["harness/success_rate"] == 0.75
    assert metrics["harness/total_count"] == 4.0
    assert metrics["harness/success_count"] == 3.0
    assert metrics["harness/error_count"] == 1.0

    # Score-derived metrics (only non-None fields should appear)
    assert metrics["harness/accuracy"] == 0.5
    assert "harness/precision" not in metrics
    assert metrics["harness/recall"] == 0.25
    assert metrics["harness/f1_score"] == 0.33
    assert metrics["harness/mean_latency_ms"] == 12.0
    assert metrics["harness/total_tokens"] == 123.0
    assert metrics["harness/error_rate"] == 0.1

    # Duration
    assert metrics["harness/duration_seconds"] == 1.23

    assert len(tracker.params_calls) == 1
    params = tracker.params_calls[0]
    assert params["harness/experiment_id"] == "exp-1"
    assert params["harness/model_name"] == "model-A"
    assert params["harness/model_provider"] == "provider-X"
    assert params["harness/probe_name"] == "probe-foo"
    assert params["harness/probe_category"] == "behaviour"


def test_auto_track_logs_numeric_metrics_and_finishes() -> None:
    tracker = CapturingTracker()

    @auto_track(tracker)
    def f() -> dict[str, Any]:
        return {"accuracy": 0.9, "note": "not-numeric"}

    out = f()
    assert out["accuracy"] == 0.9

    # start + end called
    assert len(tracker.started) == 1
    assert tracker.ended == ["finished"]

    # metrics logged, but only numeric keys survive
    assert len(tracker.metrics_calls) == 1
    metrics, _ = tracker.metrics_calls[0]
    assert metrics == {"accuracy": 0.9}


def test_auto_track_marks_failed_on_exception() -> None:
    tracker = CapturingTracker()

    @auto_track(tracker)
    def boom() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        boom()

    assert len(tracker.started) == 1
    assert tracker.ended == ["failed"]


def test_multi_tracker_fans_out_calls() -> None:
    t1 = CapturingTracker()
    t2 = CapturingTracker()
    mt = MultiTracker([t1, t2])

    rid = mt.start_run(run_name="r")
    assert rid  # non-empty
    mt.log_metrics({"m": 1.0}, step=7)
    mt.log_params({"p": "x"})
    mt.log_artifact("some/path.txt", artifact_name="a", artifact_type="file")
    mt.end_run(status="finished")

    assert len(t1.started) == 1 and len(t2.started) == 1
    assert t1.metrics_calls == [({"m": 1.0}, 7)]
    assert t2.metrics_calls == [({"m": 1.0}, 7)]
    assert t1.params_calls == [{"p": "x"}]
    assert t2.params_calls == [{"p": "x"}]
    assert t1.artifacts_calls == [("some/path.txt", "a", "file")]
    assert t2.artifacts_calls == [("some/path.txt", "a", "file")]
    assert t1.ended == ["finished"]
    assert t2.ended == ["finished"]


def test_create_tracker_local_backend(tmp_path: Path) -> None:
    tracker = create_tracker("local", output_dir=str(tmp_path))
    assert isinstance(tracker, LocalFileTracker)


def test_create_tracker_unknown_backend_raises() -> None:
    with pytest.raises(ValueError):
        create_tracker("definitely-not-a-tracker")


def test_local_file_tracker_writes_expected_files(tmp_path: Path) -> None:
    cfg = TrackingConfig(project="proj-x", experiment_name="exp-x")
    tracker = LocalFileTracker(output_dir=str(tmp_path), config=cfg)

    run_id = tracker.start_run(run_name="run-name", run_id="run-123")
    assert run_id == "run-123"

    tracker.log_params({"alpha": 1, "beta": "two"})
    tracker.log_metrics({"loss": 0.5})
    tracker.log_metrics({"loss": 0.4})

    artifact_src = tmp_path / "artifact.txt"
    artifact_src.write_text("hello", encoding="utf-8")
    tracker.log_artifact(str(artifact_src), artifact_name="artifact.txt", artifact_type="data")

    tracker.end_run(status="finished")

    run_dir = tmp_path / "proj-x" / "run-123"
    assert run_dir.exists()

    # Expected files exist
    for name in [
        "metadata.json",
        "metrics.json",
        "params.json",
        "artifacts.json",
        "final_state.json",
    ]:
        assert (run_dir / name).exists(), f"missing {name}"

    # Validate stored content shape (don’t assert timestamps exactly)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "run-123"
    assert metadata["run_name"] == "run-name"
    assert metadata["project"] == "proj-x"
    assert "started_at" in metadata

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics) == 2
    assert metrics[0]["metrics"]["loss"] == 0.5
    assert metrics[1]["metrics"]["loss"] == 0.4

    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    assert params == {"alpha": 1, "beta": "two"}

    artifacts = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
    assert len(artifacts) == 1
    assert artifacts[0]["name"] == "artifact.txt"
    assert artifacts[0]["type"] == "data"
    assert artifacts[0]["original_path"] == str(artifact_src)

    final_state = json.loads((run_dir / "final_state.json").read_text(encoding="utf-8"))
    assert final_state["status"] == "finished"
    assert final_state["metrics_count"] == 2
    assert "ended_at" in final_state


def test_local_file_tracker_load_run_and_list_runs(tmp_path: Path) -> None:
    cfg = TrackingConfig(project="proj-y")
    tracker = LocalFileTracker(output_dir=str(tmp_path), config=cfg)

    tracker.start_run(run_id="a")
    tracker.log_metrics({"m": 1.0})
    tracker.end_run()

    tracker.start_run(run_id="b")
    tracker.log_metrics({"m": 2.0})
    tracker.end_run()

    runs = tracker.list_runs()
    assert set(runs) >= {"a", "b"}

    loaded = tracker.load_run("a")
    assert "metadata" in loaded
    assert "metrics" in loaded
    assert loaded["metadata"]["run_id"] == "a"
    assert loaded["metrics"][0]["metrics"]["m"] == 1.0
