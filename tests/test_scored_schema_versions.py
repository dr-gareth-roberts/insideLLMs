"""New scores must not retroactively change historical output contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from insideLLMs.models import DummyModel
from insideLLMs.probes.logic import LogicProbe
from insideLLMs.runtime.runner import ProbeRunner
from insideLLMs.schemas import OutputValidator, SchemaRegistry
from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION


@pytest.mark.parametrize("version", ["1.0.0", "1.0.1"])
def test_historical_runner_item_schema_rejects_new_top_level_score_fields(version: str) -> None:
    model = SchemaRegistry().get_model(SchemaRegistry.RUNNER_ITEM, version)
    assert "scores" not in model.model_fields
    assert "primary_metric" not in model.model_fields
    with pytest.raises(ValidationError, match="scores"):
        model(input="Question", status="success", scores={"accuracy": 1.0})


def test_default_runner_item_schema_accepts_scores_including_nested_batch_items() -> None:
    assert DEFAULT_SCHEMA_VERSION == "1.0.2"
    result = {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "input": "Question",
        "status": "success",
        "scores": {"accuracy": 1.0},
        "primary_metric": "accuracy",
    }
    validated = OutputValidator().validate(
        SchemaRegistry.RUNNER_OUTPUT,
        {"schema_version": DEFAULT_SCHEMA_VERSION, "results": [result]},
    )
    assert validated.results[0].scores == {"accuracy": 1.0}
    assert validated.results[0].primary_metric == "accuracy"


def test_benchmark_summary_uses_scored_nested_item_schema() -> None:
    from insideLLMs.contrib.benchmark import ModelBenchmark, ProbeBenchmark

    dataset = [{"question": "Capital?", "reference_answer": "Paris"}]
    model = DummyModel(canned_response="Paris")
    for benchmark in [ModelBenchmark([model], LogicProbe()), ProbeBenchmark(model, [LogicProbe()])]:
        payload = benchmark.run(dataset, emit_run_artifacts=False)
        OutputValidator().validate(SchemaRegistry.BENCHMARK_SUMMARY, payload)
        assert payload["schema_version"] == "1.0.2"


@pytest.mark.parametrize("version", ["1.0.0", "1.0.1"])
def test_migration_updates_nested_versions_without_fabricating_scores(version: str) -> None:
    original = {
        "schema_version": version,
        "results": [{"schema_version": version, "input": "Question", "status": "success"}],
    }
    migrated = SchemaRegistry().migrate(SchemaRegistry.RUNNER_OUTPUT, original, version, "1.0.2")
    assert original["schema_version"] == version
    assert original["results"][0]["schema_version"] == version
    assert migrated["schema_version"] == "1.0.2"
    assert migrated["results"][0]["schema_version"] == "1.0.2"
    assert "scores" not in migrated["results"][0]
    result = OutputValidator().validate(SchemaRegistry.RUNNER_OUTPUT, migrated)
    assert result.results[0].scores == {}
    assert result.results[0].primary_metric is None


@pytest.mark.parametrize("version", ["1.0.0", "1.0.1", "1.0.2"])
def test_explicit_old_version_preserves_record_scoring_without_schema_drift(tmp_path, version):
    run_dir = tmp_path / version
    runner = ProbeRunner(DummyModel(canned_response="Paris"), LogicProbe())
    dataset = [{"question": "Capital?", "reference_answer": "Paris"}]
    results = runner.run(dataset, run_dir=run_dir, schema_version=version, validate_output=True)
    record = json.loads((run_dir / "records.jsonl").read_text())
    assert record["scores"] == {"accuracy": 1.0}
    assert record["primary_metric"] == "accuracy"
    assert runner.last_experiment.score.accuracy == 1.0
    assert runner.last_experiment.results[0].scores == {"accuracy": 1.0}
    if version == "1.0.2":
        assert results[0]["scores"] == {"accuracy": 1.0}
    else:
        assert "scores" not in results[0]
        assert results[0]["metadata"]["scores"] == {"accuracy": 1.0}
    resumed = runner.run(
        dataset, run_dir=run_dir, schema_version=version, validate_output=True, resume=True
    )
    assert resumed == results
