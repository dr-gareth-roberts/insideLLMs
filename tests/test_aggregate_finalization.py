"""Aggregate failures are incomplete runs even when every model call succeeded."""

import json
from unittest.mock import patch

import pytest
import yaml

from insideLLMs.cli import main
from insideLLMs.config_types import RunConfig
from insideLLMs.exceptions import RunnerExecutionError
from insideLLMs.models import DummyModel
from insideLLMs.probes import LogicProbe
from insideLLMs.runtime._run_health import check_run_directory_health
from insideLLMs.runtime.runner import AsyncProbeRunner, ProbeRunner, run_harness_from_config


class FailingAggregateProbe(LogicProbe):
    def __init__(self):
        super().__init__()
        self.score_calls = 0

    def score(self, results):
        self.score_calls += 1
        raise ValueError("aggregate failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("schema_version", ["1.0.0", "1.0.1", "1.0.2"])
async def test_aggregate_failure_finalizes_incomplete_artifacts(
    tmp_path, asynchronous, schema_version
):
    failing_probe = FailingAggregateProbe()
    runner = (AsyncProbeRunner if asynchronous else ProbeRunner)(DummyModel(), failing_probe)
    run_dir = tmp_path / "run"
    with patch("insideLLMs.runtime._ultimate.run_ultimate_post_artifact") as publisher:
        with pytest.raises(RunnerExecutionError) as caught:
            result = runner.run(
                ["hello"],
                run_dir=run_dir,
                config=RunConfig(run_mode="ultimate"),
                schema_version=schema_version,
                validate_output=True,
            )
            if asynchronous:
                await result
    assert isinstance(caught.value.original_error, ValueError)
    assert len(caught.value.partial_results) == 1
    assert caught.value.partial_results[0]["status"] == "success"
    assert caught.value.partial_result["run_completed"] is False
    assert runner.last_experiment.score is None
    assert failing_probe.score_calls == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if schema_version != "1.0.0":
        assert manifest["run_completed"] is False
    assert manifest["custom"]["health"]["healthy"] is False
    health = check_run_directory_health(run_dir)
    assert health["healthy"] is False
    assert "Run did not complete" in health["reasons"]
    publisher.assert_not_called()
    assert main(["validate", str(run_dir), "--quiet"]) == 0


@pytest.mark.parametrize("failure_cell", [1, 2, 3, None])
def test_harness_scores_each_cell_once_and_retains_completed_cells(
    tmp_path, monkeypatch, failure_cell
):
    original_score = LogicProbe.score
    scored_cells = []

    def score(self, results):
        scored_cells.append(self)
        if len(scored_cells) == failure_cell:
            raise ValueError("aggregate failed")
        return original_score(self, results)

    monkeypatch.setattr(LogicProbe, "score", score)
    config_path = tmp_path / "harness.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [{"type": "dummy"}] * 3,
                "probes": [{"type": "logic"}],
                "dataset": {"format": "inline", "data": ["hello"]},
            }
        )
    )
    if failure_cell is None:
        result = run_harness_from_config(config_path)
        assert result["run_completed"] is True
        assert len(scored_cells) == 3
    else:
        with pytest.raises(RunnerExecutionError) as caught:
            run_harness_from_config(config_path)
        partial = caught.value.partial_result
        assert len(partial["experiments"]) == failure_cell
        assert len(partial["records"]) == failure_cell
        assert partial["experiments"][-1].score is None
        assert all(record["status"] == "success" for record in partial["records"])
        assert partial["run_completed"] is False
        assert len(scored_cells) == failure_cell
        scored_cells.clear()
        run_dir = tmp_path / "cli"
        assert (
            main(
                [
                    "harness",
                    str(config_path),
                    "--run-dir",
                    str(run_dir),
                    "--quiet",
                    "--skip-report",
                    "--validate-output",
                ]
            )
            == 1
        )
        manifest = json.loads((run_dir / "manifest.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())["summary"]
        assert manifest["run_completed"] is False
        assert manifest["record_count"] == failure_cell
        assert summary["run_completed"] is False
        assert main(["validate", str(run_dir), "--quiet"]) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failed_file", ["manifest.json", "summary.json"])
async def test_diagnostic_write_failure_retains_original_and_memory_payload(
    tmp_path, monkeypatch, asynchronous, failed_file
):
    import os
    from pathlib import Path

    original_replace = os.replace

    def replace(self, target):
        if Path(target).name == failed_file:
            raise OSError("diagnostic disk failure")
        return original_replace(self, target)

    monkeypatch.setattr(os, "replace", replace)
    probe = FailingAggregateProbe()
    runner = (AsyncProbeRunner if asynchronous else ProbeRunner)(DummyModel(), probe)
    with pytest.raises(RunnerExecutionError) as caught:
        result = runner.run(["hello"], run_dir=tmp_path / "run")
        if asynchronous:
            await result
    assert str(caught.value.original_error) == "aggregate failed"
    assert len(caught.value.partial_results) == 1
    assert caught.value.partial_result["run_completed"] is False
    assert caught.value.secondary_diagnostics[-1]["error_type"] == "OSError"
    assert caught.value.partial_result["summary"]["secondary_diagnostics"] == (
        caught.value.secondary_diagnostics
    )
    assert probe.score_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("schema_version", ["1.0.0", "1.0.1", "1.0.2"])
async def test_aggregate_failure_is_typed_without_artifacts(asynchronous, schema_version):
    probe = FailingAggregateProbe()
    runner = (AsyncProbeRunner if asynchronous else ProbeRunner)(DummyModel(), probe)
    progress = []
    with pytest.raises(RunnerExecutionError) as caught:
        result = runner.run(
            ["hello"],
            emit_run_artifacts=False,
            schema_version=schema_version,
            validate_output=True,
            progress_callback=lambda info: progress.append(info),
        )
        if asynchronous:
            await result
    assert caught.value.__cause__ is caught.value.original_error
    assert caught.value.partial_result["run_completed"] is False
    assert caught.value.partial_result["experiments"][0].score is None
    assert probe.score_calls == 1
    assert progress[-1].status == "aborted"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("schema_version", ["1.0.0", "1.0.1", "1.0.2"])
async def test_output_validation_failure_prevents_final_success(
    tmp_path, monkeypatch, asynchronous, schema_version
):
    from insideLLMs.schemas import OutputValidator, SchemaRegistry

    original_validate = OutputValidator.validate
    failure = ValueError("invalid runner output")

    def validate(self, schema_name, *args, **kwargs):
        if schema_name == SchemaRegistry.RUNNER_OUTPUT:
            raise failure
        return original_validate(self, schema_name, *args, **kwargs)

    monkeypatch.setattr(OutputValidator, "validate", validate)
    runner = (AsyncProbeRunner if asynchronous else ProbeRunner)(DummyModel(), LogicProbe())
    run_dir = tmp_path / "run"
    with patch("insideLLMs.runtime._ultimate.run_ultimate_post_artifact") as publisher:
        with pytest.raises(RunnerExecutionError) as caught:
            result = runner.run(
                ["hello"],
                run_dir=run_dir,
                validate_output=True,
                schema_version=schema_version,
                config=RunConfig(run_mode="ultimate"),
            )
            if asynchronous:
                await result
    assert caught.value.original_error is failure
    assert caught.value.partial_result["run_completed"] is False
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if schema_version != "1.0.0":
        assert manifest["run_completed"] is False
    health = check_run_directory_health(run_dir)
    assert health["healthy"] is False
    assert "Run did not complete" in health["reasons"]
    publisher.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("custom", [{}, {"abort": {}}, {"health": {"healthy": False}}])
async def test_legacy_completion_respects_explicit_failure_evidence(tmp_path, asynchronous, custom):
    tmp_path = tmp_path / "run"
    runner = (AsyncProbeRunner if asynchronous else ProbeRunner)(DummyModel(), LogicProbe())
    result = runner.run(["hello"], run_dir=tmp_path, schema_version="1.0.0", validate_output=True)
    if asynchronous:
        await result
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["completed_at"]
    manifest["custom"] = custom
    manifest_path.write_text(json.dumps(manifest))
    before = manifest_path.read_bytes()
    health = check_run_directory_health(tmp_path)
    assert health["healthy"] is (not custom)
    assert manifest_path.read_bytes() == before
