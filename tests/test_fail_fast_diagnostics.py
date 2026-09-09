"""Interrupted execution must retain usable, explicitly incomplete evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from insideLLMs.cli import main
from insideLLMs.exceptions import RunnerExecutionError
from insideLLMs.models import DummyModel
from insideLLMs.probes import LogicProbe
from insideLLMs.runtime.runner import AsyncProbeRunner, ProbeRunner, run_harness_from_config


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure_type", [RuntimeError, TimeoutError])
async def test_builtin_batch_preserves_exact_runtime_cause(
    tmp_path, monkeypatch, asynchronous, failure_type
):
    from dataclasses import asdict

    failure = failure_type("original batch cause")

    def generate(*args, **kwargs):
        raise failure

    monkeypatch.setattr(DummyModel, "generate", generate)
    probe = LogicProbe()
    direct = probe.run_batch(DummyModel(), [{"question": "One?"}])[0]
    assert direct.original_error is failure
    assert "original_error" not in asdict(direct)
    assert "_original_error" not in asdict(direct)
    from insideLLMs._serialization import serialize_value

    assert "_original_error" not in serialize_value(direct, strict=True)
    runner = (AsyncProbeRunner if asynchronous else ProbeRunner)(DummyModel(), probe)
    with pytest.raises(RunnerExecutionError) as caught:
        result = runner.run(
            [{"question": "One?"}],
            use_probe_batch=True,
            stop_on_error=True,
            run_dir=tmp_path / "run",
        )
        if asynchronous:
            await result
    assert caught.value.original_error is failure
    assert caught.value.__cause__ is failure
    assert main(["validate", str(tmp_path / "run"), "--quiet"]) == 0


def test_batch_evaluation_keeps_runtime_cause_without_serializing_it(monkeypatch):
    from dataclasses import asdict

    from insideLLMs.probes._scoring import evaluate_batch_result
    from insideLLMs.types import ProbeResult

    failure = ValueError("evaluation failed")

    def evaluate(*args, **kwargs):
        raise failure

    monkeypatch.setattr(LogicProbe, "evaluate_single", evaluate)
    result = evaluate_batch_result(
        LogicProbe(),
        ProbeResult(input="old", output="answer"),
        {"question": "Question?", "reference_answer": "answer"},
    )
    assert result.original_error is failure
    assert "original_error" not in asdict(result)
    normalized = evaluate_batch_result(LogicProbe(), result, {"question": "same"})
    assert normalized.original_error is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("schema_version", ["1.0.0", "1.0.1", "1.0.2"])
async def test_abort_scoring_failure_keeps_original_and_partial_summary(
    tmp_path, monkeypatch, asynchronous, schema_version
):
    failure = RuntimeError("provider failure")

    def generate(*args, **kwargs):
        raise failure

    def score(*args, **kwargs):
        raise ValueError("aggregate score unavailable")

    monkeypatch.setattr(DummyModel, "generate", generate)
    monkeypatch.setattr(LogicProbe, "score", score)
    runner = (AsyncProbeRunner if asynchronous else ProbeRunner)(DummyModel(), LogicProbe())
    run_dir = tmp_path / "run"
    with pytest.raises(RunnerExecutionError) as caught:
        result = runner.run(
            [{"question": "One?"}],
            stop_on_error=True,
            run_dir=run_dir,
            validate_output=True,
            schema_version=schema_version,
        )
        if asynchronous:
            await result
    assert caught.value.original_error is failure
    assert caught.value.__cause__ is failure
    assert runner.last_experiment.score is None
    summary = json.loads((run_dir / "summary.json").read_text())["summary"]
    assert summary["secondary_diagnostics"][0]["stage"] == "aggregate_scoring"
    assert summary["secondary_diagnostics"][0]["error_type"] == "ValueError"
    assert caught.value.partial_result["summary"] == summary
    assert main(["validate", str(run_dir), "--quiet"]) == 0


def test_successful_calls_raise_typed_aggregate_scoring_error(tmp_path, monkeypatch):
    def score(*args, **kwargs):
        raise ValueError("score failed on complete run")

    monkeypatch.setattr(LogicProbe, "score", score)
    with pytest.raises(RunnerExecutionError) as caught:
        ProbeRunner(DummyModel(), LogicProbe()).run(
            [{"question": "One?"}], run_dir=tmp_path / "run"
        )
    assert isinstance(caught.value.original_error, ValueError)
    assert str(caught.value.original_error) == "score failed on complete run"
    assert caught.value.partial_result["experiments"][0].score is None


def test_later_harness_abort_with_scoring_failure_preserves_prior_cells(tmp_path, monkeypatch):
    original_score = LogicProbe.score
    calls = []
    failure = RuntimeError("later provider error")

    def generate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 4:
            raise failure
        return "answer"

    def score(self, results):
        if len(calls) >= 4:
            raise ValueError("partial aggregate unavailable")
        return original_score(self, results)

    monkeypatch.setattr(DummyModel, "generate", generate)
    monkeypatch.setattr(LogicProbe, "score", score)
    run_dir = tmp_path / "run"
    assert (
        main(
            [
                "harness",
                str(_harness_config(tmp_path)),
                "--run-dir",
                str(run_dir),
                "--quiet",
                "--skip-report",
                "--validate-output",
            ]
        )
        == 1
    )
    assert len(calls) == 4
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["record_count"] == 4
    assert manifest["custom"]["abort"]["message"] == str(failure)
    summary = json.loads((run_dir / "summary.json").read_text())["summary"]
    assert summary["secondary_diagnostics"][0]["stage"] == "aggregate_scoring"


def _harness_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [{"type": "dummy"}, {"type": "dummy"}],
                "probes": [{"type": "logic"}],
                "dataset": {
                    "format": "inline",
                    "data": [
                        {"question": "One?"},
                        {"question": "Two?"},
                        {"question": "Three?"},
                    ],
                },
                "runner": {"stop_on_error": True},
            }
        ),
        encoding="utf-8",
    )
    return config_path


@pytest.mark.parametrize("failure_at", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("schema_version", ["1.0.0", "1.0.1", "1.0.2"])
def test_fail_fast_harness_retains_results_and_incomplete_manifest(
    tmp_path, monkeypatch, failure_at, schema_version
):
    calls = []

    def generate(self, prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == failure_at:
            raise RuntimeError("diagnostic provider failure")
        return "First response"

    monkeypatch.setattr(DummyModel, "generate", generate)
    config_path = _harness_config(tmp_path)
    run_dir = tmp_path / "run"

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
                "--schema-version",
                schema_version,
            ]
        )
        == 1
    )

    assert len(calls) == failure_at
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert [record["status"] for record in records] == ["success"] * (failure_at - 1) + ["error"]
    assert "diagnostic provider failure" in records[-1]["error"]
    if schema_version != "1.0.0":
        assert manifest["run_completed"] is False
    assert manifest["custom"]["abort"]["code"] == "execution_error"
    assert manifest["custom"]["health"]["healthy"] is False
    assert manifest["custom"]["health"]["expected_count"] == 6
    assert manifest["record_count"] == failure_at
    assert (run_dir / "summary.json").exists()
    assert main(["validate", str(run_dir), "--quiet"]) == 0


def test_harness_api_preserves_original_error_and_prior_cells(tmp_path, monkeypatch):
    failure = RuntimeError("original failure")
    calls = []

    def generate(self, prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 4:
            raise failure
        return "response"

    monkeypatch.setattr(DummyModel, "generate", generate)
    with pytest.raises(RunnerExecutionError) as caught:
        run_harness_from_config(_harness_config(tmp_path))

    assert caught.value.original_error is failure
    assert caught.value.__cause__ is failure
    partial = caught.value.partial_result
    assert partial is not None
    assert len(partial["records"]) == 4
    assert len(partial["experiments"]) == 2
    assert partial["run_completed"] is False


@pytest.mark.parametrize("batch", [False, True])
def test_sync_runner_preserves_every_completed_batch_result(tmp_path, monkeypatch, batch):
    calls = []

    def generate(self, prompt, **kwargs):
        calls.append(prompt)
        if "Two?" in prompt:
            raise RuntimeError("second item failed")
        return "response"

    monkeypatch.setattr(DummyModel, "generate", generate)
    runner = ProbeRunner(DummyModel(), LogicProbe())
    with pytest.raises(RunnerExecutionError) as caught:
        runner.run(
            [{"question": text} for text in ["One?", "Two?", "Three?"]],
            stop_on_error=True,
            use_probe_batch=batch,
            run_dir=tmp_path / "run",
        )

    # A submitted batch may have completed before its outcomes are observed.
    expected_count = 3 if batch else 2
    assert len(caught.value.partial_results) == expected_count
    assert len(calls) == expected_count
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["record_count"] == expected_count
    assert manifest["run_completed"] is False
    assert runner.last_experiment is not None
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["summary"]["run_completed"] is False
    assert caught.value.partial_result["summary"] == summary["summary"]
    assert main(["validate", str(tmp_path / "run"), "--quiet"]) == 0


@pytest.mark.asyncio
async def test_async_runner_finalizes_incomplete_manifest_before_raising(tmp_path, monkeypatch):
    calls = []

    def generate(self, prompt, **kwargs):
        calls.append(prompt)
        if "Two?" in prompt:
            raise RuntimeError("second item failed")
        return "response"

    monkeypatch.setattr(DummyModel, "generate", generate)
    runner = AsyncProbeRunner(DummyModel(), LogicProbe())
    with pytest.raises(RunnerExecutionError) as caught:
        await runner.run(
            [{"question": text} for text in ["One?", "Two?", "Three?"]],
            stop_on_error=True,
            concurrency=1,
            run_dir=tmp_path / "run",
        )

    assert len(calls) == 2
    # Undispatched items leave no placeholder (W7-0007); incompleteness is
    # explicit in the manifest instead.
    assert [item["status"] for item in caught.value.partial_results] == ["success", "error"]
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["run_completed"] is False
    assert manifest["custom"]["health"]["healthy"] is False
    assert "Expected 3 records, found 2" in manifest["custom"]["health"]["reasons"]
    assert runner.last_experiment is not None
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["summary"]["run_completed"] is False
    assert caught.value.partial_result["summary"] == summary["summary"]
    assert main(["validate", str(tmp_path / "run"), "--quiet"]) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_thrown_batch_error_finalizes_without_fabricating_outcomes(
    tmp_path, monkeypatch, asynchronous
):
    failure = RuntimeError("batch transport unavailable")

    def run_batch(self, model, inputs, **kwargs):
        raise failure

    monkeypatch.setattr(LogicProbe, "run_batch", run_batch)
    runner = (AsyncProbeRunner if asynchronous else ProbeRunner)(DummyModel(), LogicProbe())
    run_dir = tmp_path / "run"
    with pytest.raises(RunnerExecutionError) as caught:
        if asynchronous:
            await runner.run([{"question": "One?"}], use_probe_batch=True, run_dir=run_dir)
        else:
            runner.run([{"question": "One?"}], use_probe_batch=True, run_dir=run_dir)

    assert caught.value.original_error is failure
    assert caught.value.__cause__ is failure
    assert caught.value.partial_results == []
    assert (run_dir / "records.jsonl").read_text() == ""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_completed"] is False
    assert manifest["record_count"] == 0
    assert manifest["custom"]["abort"]["message"] == str(failure)
    assert (run_dir / "summary.json").exists()
    assert main(["validate", str(run_dir), "--quiet"]) == 0


def test_later_thrown_batch_error_keeps_completed_harness_cells(tmp_path, monkeypatch):
    original_batch = LogicProbe.run_batch
    batches = []

    def run_batch(self, model, inputs, **kwargs):
        batches.append(inputs)
        if len(batches) == 2:
            raise RuntimeError("second batch unavailable")
        return original_batch(self, model, inputs, **kwargs)

    monkeypatch.setattr(LogicProbe, "run_batch", run_batch)
    config_path = _harness_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["runner"]["use_probe_batch"] = True
    config_path.write_text(yaml.safe_dump(config))
    run_dir = tmp_path / "run"
    assert (
        main(["harness", str(config_path), "--run-dir", str(run_dir), "--quiet", "--skip-report"])
        == 1
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["record_count"] == 3
    assert manifest["run_completed"] is False
    assert manifest["custom"]["abort"]["message"] == "second batch unavailable"
    assert len((run_dir / "records.jsonl").read_text().splitlines()) == 3


def test_later_model_initialization_failure_retains_earlier_experiments(tmp_path, monkeypatch):
    original_init = DummyModel.__init__
    initialized = []

    def initialize(self, *args, **kwargs):
        initialized.append(self)
        if len(initialized) == 2:
            raise RuntimeError("second model unavailable")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(DummyModel, "__init__", initialize)
    config_path = _harness_config(tmp_path)
    run_dir = tmp_path / "run"
    assert (
        main(["harness", str(config_path), "--run-dir", str(run_dir), "--quiet", "--skip-report"])
        == 1
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["record_count"] == 3
    assert manifest["run_completed"] is False
    assert manifest["custom"]["health"]["healthy"] is False
    assert manifest["custom"]["abort"]["message"] == "second model unavailable"
    assert (run_dir / "summary.json").exists()
