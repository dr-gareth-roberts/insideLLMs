"""Behavioural scores must survive every execution path and drive CLI gates."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from insideLLMs.cli import main
from insideLLMs.models import DummyModel
from insideLLMs.probes.base import ScoredProbe
from insideLLMs.probes.instruction import InstructionFollowingProbe
from insideLLMs.probes.logic import LogicProbe
from insideLLMs.runtime.runner import (
    AsyncProbeRunner,
    ProbeRunner,
    run_harness_from_config,
)
from insideLLMs.types import ProbeResult, ResultStatus


class ExactMatchProbe(ScoredProbe[str]):
    def __init__(self, *, invalid_score: Any = None, fail_evaluation: bool = False) -> None:
        super().__init__(name="exact_match")
        self.evaluations = 0
        self.inputs: list[Any] = []
        self.invalid_score = invalid_score
        self.fail_evaluation = fail_evaluation

    def run(self, model: Any, data: Any, **kwargs: Any) -> str:
        self.inputs.append(data)
        return model.generate(str(data), **kwargs)

    def evaluate_single(self, model_output: str, reference: Any, input_data: Any) -> dict:
        self.evaluations += 1
        if self.fail_evaluation:
            raise ValueError("Evaluator unavailable")
        details = {"is_correct": model_output == reference}
        if self.invalid_score is not None:
            details["score"] = self.invalid_score
        return details


@pytest.mark.parametrize("asynchronous", [False, True])
def test_successful_repeat_runs_score_once_and_keep_identical_artifacts(
    tmp_path: Path, asynchronous: bool
) -> None:
    class CountingProbe(ExactMatchProbe):
        score_calls = 0

        def score(self, results):
            self.score_calls += 1
            return super().score(results)

    probe = CountingProbe()
    runner_class = AsyncProbeRunner if asynchronous else ProbeRunner
    runner = runner_class(DummyModel(canned_response="Paris"), probe)
    dataset = [{"question": "Capital?", "reference_answer": "Paris"}]
    for index in range(2):
        experiment = _run(
            runner,
            dataset,
            asynchronous=asynchronous,
            run_dir=tmp_path / str(index),
            return_experiment=True,
            strict_serialization=True,
            validate_output=True,
        )
        assert experiment.score.accuracy == 1.0
        assert probe.score_calls == index + 1
    for filename in ("records.jsonl", "manifest.json"):
        assert (tmp_path / "0" / filename).read_bytes() == (tmp_path / "1" / filename).read_bytes()


def _run(runner: Any, dataset: list[Any], *, asynchronous: bool, **kwargs: Any) -> Any:
    result = runner.run(dataset, **kwargs)
    return asyncio.run(result) if asynchronous else result


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_scores_survive_records_experiment_and_resume(
    tmp_path: Path, asynchronous: bool, batch: bool
) -> None:
    probe = ExactMatchProbe()
    model = DummyModel(canned_response="Paris")
    runner = AsyncProbeRunner(model, probe) if asynchronous else ProbeRunner(model, probe)
    dataset = [
        {
            "id": "correct",
            "question": "Capital of France?",
            "reference_answer": "Paris",
        },
        {
            "id": "wrong",
            "question": "Capital of Germany?",
            "reference_answer": "Berlin",
        },
    ]
    run_dir = tmp_path / "run"
    results = _run(
        runner,
        dataset,
        asynchronous=asynchronous,
        run_dir=run_dir,
        use_probe_batch=batch,
        validate_output=True,
    )

    assert probe.evaluations == 2
    assert all("reference_answer" not in item for item in probe.inputs)
    assert [result["scores"] for result in results] == [
        {"accuracy": 1.0},
        {"accuracy": 0.0},
    ]
    assert [result["primary_metric"] for result in results] == ["accuracy", "accuracy"]
    assert runner.last_experiment.score.accuracy == 0.5
    assert runner.last_experiment.results[0].scores == {"accuracy": 1.0}
    before = (run_dir / "records.jsonl").read_bytes()
    records = [json.loads(line) for line in before.splitlines()]
    assert records[0]["input"] == dataset[0]
    assert records[0]["output"] == "Paris"
    assert records[1]["custom"]["evaluation"]["is_correct"] is False

    resumed = _run(
        runner,
        dataset,
        asynchronous=asynchronous,
        run_dir=run_dir,
        use_probe_batch=batch,
        resume=True,
        validate_output=True,
    )
    assert probe.evaluations == 2
    assert resumed == results
    assert runner.last_experiment.score.accuracy == 0.5
    assert (run_dir / "records.jsonl").read_bytes() == before


def test_direct_probe_batch_scores_once_without_showing_reference_to_model() -> None:
    probe = ExactMatchProbe()
    results = probe.run_batch(
        DummyModel(canned_response="Paris"),
        [{"question": "Capital?", "reference_answer": "Paris"}],
    )
    assert results[0].scores == {"accuracy": 1.0}
    assert results[0].primary_metric == "accuracy"
    assert probe.evaluations == 1
    assert probe.inputs == [{"question": "Capital?"}]


def test_unlabelled_execution_does_not_claim_accuracy() -> None:
    probe = ExactMatchProbe()
    runner = ProbeRunner(DummyModel(), probe)
    results = runner.run([{"question": "Anything?"}], emit_run_artifacts=False)
    assert results[0]["status"] == "success"
    assert results[0]["scores"] == {}
    assert results[0]["primary_metric"] is None
    assert results[0]["metadata"]["evaluation"]["reason"] == "missing_reference"
    assert runner.last_experiment.score.accuracy is None
    assert probe.evaluations == 0


@pytest.mark.parametrize("asynchronous", [False, True])
def test_legacy_labelled_records_without_scores_cannot_resume(
    tmp_path: Path, asynchronous: bool
) -> None:
    probe = ExactMatchProbe()
    model = DummyModel(canned_response="Paris")
    runner = AsyncProbeRunner(model, probe) if asynchronous else ProbeRunner(model, probe)
    dataset = [{"question": "Capital?", "reference_answer": "Paris"}]
    run_dir = tmp_path / "legacy"
    _run(runner, dataset, asynchronous=asynchronous, run_dir=run_dir)
    record_path = run_dir / "records.jsonl"
    record = json.loads(record_path.read_text())
    record["scores"] = {}
    record["primary_metric"] = None
    record["custom"].pop("evaluation")
    record_path.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="without persisted evaluation"):
        _run(runner, dataset, asynchronous=asynchronous, run_dir=run_dir, resume=True)
    assert probe.evaluations == 1


@pytest.mark.parametrize(
    "scores", [{}, {"accuracy": None}, {"accuracy": float("nan")}, {"other": 1}]
)
def test_already_evaluated_batch_results_and_resume_reject_malformed_scores(scores: Any) -> None:
    from insideLLMs.probes._scoring import evaluate_batch_result, validate_scored_resume_record

    probe = ExactMatchProbe()
    item = {"question": "Capital?", "reference_answer": "Paris"}
    metadata = {"is_correct": True, "evaluation": {"status": "evaluated"}}
    result = ProbeResult(
        input=item, output="Paris", metadata=metadata, scores=scores, primary_metric="accuracy"
    )
    evaluated = evaluate_batch_result(probe, result, item)
    assert evaluated.status == ResultStatus.ERROR
    assert probe.evaluations == 0
    with pytest.raises(ValueError):
        validate_scored_resume_record(
            probe,
            {
                "input": item,
                "status": "success",
                "scores": scores,
                "primary_metric": "accuracy",
                "custom": {"evaluation": metadata},
            },
        )


def test_evaluation_credentials_are_scrubbed_before_serialization(tmp_path: Path) -> None:
    class CredentialProbe(ExactMatchProbe):
        def evaluate_single(self, model_output: str, reference: Any, input_data: Any) -> dict:
            self.live_details = {
                "is_correct": True,
                "judge_config": {"api_key": "SCORER_SENTINEL"},
            }
            return self.live_details

    probe = CredentialProbe()
    runner = ProbeRunner(DummyModel(canned_response="Paris"), probe)
    results = runner.run(
        [{"question": "Capital?", "reference_answer": "Paris"}], run_dir=tmp_path / "redacted"
    )
    assert probe.live_details["judge_config"]["api_key"] == "SCORER_SENTINEL"
    assert "SCORER_SENTINEL" not in json.dumps(results)
    for path in (tmp_path / "redacted").glob("*"):
        if path.is_file():
            assert "SCORER_SENTINEL" not in path.read_text()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_config_max_examples_caps_single_run_execution(tmp_path: Path, asynchronous: bool) -> None:
    import yaml

    from insideLLMs.runtime.runner import (
        run_experiment_from_config,
        run_experiment_from_config_async,
    )

    dataset = [
        {"question": "Capital?", "reference_answer": "Paris"},
        {"question": "Another capital?", "reference_answer": "Berlin"},
    ]
    (tmp_path / "dataset.jsonl").write_text("\n".join(json.dumps(item) for item in dataset))
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"type": "dummy", "args": {"canned_response": "Paris"}},
                "probe": {"type": "logic"},
                "dataset": {"path": "dataset.jsonl", "format": "jsonl"},
                "max_examples": 1,
            }
        )
    )
    run_dir = tmp_path / "capped"
    function = run_experiment_from_config_async if asynchronous else run_experiment_from_config
    result = function(config_path, run_dir=run_dir)
    results = asyncio.run(result) if asynchronous else result
    assert len(results) == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["record_count"] == 1
    assert manifest["custom"]["health"]["expected_count"] == 1


def test_explicit_scores_and_primary_metric_drive_diff_without_using_output_shape() -> None:
    class NumericProbe(ExactMatchProbe):
        def evaluate_single(self, model_output: str, reference: Any, input_data: Any) -> dict:
            return {
                "is_correct": True,
                "scores": {"quality": 0.75},
                "primary_metric": "quality",
            }

    runner = ProbeRunner(DummyModel(canned_response="raw response"), NumericProbe())
    result = runner.run(
        [{"question": "Question", "reference_answer": "reference"}], emit_run_artifacts=False
    )[0]
    assert result["output"] == "raw response"
    assert result["scores"] == {"quality": 0.75, "accuracy": 1.0}
    assert result["primary_metric"] == "quality"


def test_custom_batch_override_gets_no_held_out_reference() -> None:
    class CustomBatchProbe(ExactMatchProbe):
        def run_batch(self, model: Any, dataset: list[Any], **kwargs: Any) -> list[ProbeResult]:
            self.inputs.extend(dataset)
            return [ProbeResult(input=item, output="Paris") for item in dataset]

    probe = CustomBatchProbe()
    results = ProbeRunner(DummyModel(), probe).run(
        [{"question": "Capital?", "reference_answer": "Paris"}],
        emit_run_artifacts=False,
        use_probe_batch=True,
    )
    assert probe.inputs == [{"question": "Capital?"}]
    assert probe.evaluations == 1
    assert results[0]["scores"] == {"accuracy": 1.0}


def test_null_reference_is_an_explicit_reference_free_evaluation() -> None:
    probe = ExactMatchProbe()
    results = ProbeRunner(DummyModel(), probe).run(
        [{"question": "Anything?", "reference_answer": None}], emit_run_artifacts=False
    )
    assert results[0]["scores"] == {"accuracy": 0.0}
    assert probe.evaluations == 1


def test_reference_alias_and_conflicting_fields() -> None:
    probe = ExactMatchProbe()
    results = ProbeRunner(DummyModel(canned_response="Paris"), probe).run(
        [
            {"question": "Capital?", "reference": "Paris"},
            {"question": "Capital?", "reference": "Paris", "reference_answer": "Lyon"},
        ],
        emit_run_artifacts=False,
    )
    assert results[0]["scores"] == {"accuracy": 1.0}
    assert results[1]["status"] == "error"
    assert "only one" in results[1]["error"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_evaluator_failure_is_an_item_error(
    tmp_path: Path, asynchronous: bool, batch: bool
) -> None:
    probe = ExactMatchProbe(fail_evaluation=True)
    model = DummyModel()
    runner = AsyncProbeRunner(model, probe) if asynchronous else ProbeRunner(model, probe)
    results = _run(
        runner,
        [{"question": "Capital?", "reference_answer": "Paris"}],
        asynchronous=asynchronous,
        run_dir=tmp_path / "failed",
        use_probe_batch=batch,
    )
    assert results[0]["status"] == "error"
    assert "Evaluator unavailable" in results[0]["error"]
    assert runner.last_experiment.score.accuracy is None
    manifest = json.loads((tmp_path / "failed" / "manifest.json").read_text())
    assert manifest["custom"]["health"]["healthy"] is False


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), True, "1.0"])
def test_invalid_evaluation_metrics_are_errors(invalid_score: Any) -> None:
    probe = ExactMatchProbe(invalid_score=invalid_score)
    results = ProbeRunner(DummyModel(canned_response="Paris"), probe).run(
        [{"question": "Capital?", "reference_answer": "Paris"}],
        emit_run_artifacts=False,
    )
    assert results[0]["status"] == "error"
    assert "finite numbers" in results[0]["error"]


def test_probe_result_evaluations_preserve_raw_input_and_output() -> None:
    probe = InstructionFollowingProbe()
    item = {
        "task": "Reply with one word",
        "reference_answer": {"constraints": {"max_words": 1}},
    }
    results = ProbeRunner(DummyModel(canned_response="too many words"), probe).run(
        [item], emit_run_artifacts=False
    )
    assert results[0]["status"] == "success"
    assert results[0]["input"] == item
    assert results[0]["output"] == "too many words"
    assert results[0]["scores"]["accuracy"] == 0.0
    assert results[0]["primary_metric"] == "score"


def test_unrelated_metadata_does_not_count_as_incorrect_evaluation() -> None:
    score = LogicProbe().score(
        [
            ProbeResult(input="a", status=ResultStatus.SUCCESS, metadata={"is_correct": True}),
            ProbeResult(input="b", status=ResultStatus.SUCCESS, metadata={"trace": "present"}),
        ]
    )
    assert score.accuracy == 1.0


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_checked_in_cli_fixtures_gate_real_accuracy_regression(
    tmp_path: Path, asynchronous: bool, batch: bool
) -> None:
    fixture_dir = Path(__file__).resolve().parents[1] / "examples" / "diff"
    import yaml

    extra = ["--async"] if asynchronous else []
    for name, expected in [("baseline", 1.0), ("candidate", 0.0)]:
        config_path = fixture_dir / f"{name}.yaml"
        if batch:
            config = yaml.safe_load(config_path.read_text())
            config["dataset"]["path"] = str(fixture_dir / "dataset.jsonl")
            config["runner"] = {"use_probe_batch": True, "batch_workers": 2}
            config_path = tmp_path / f"{name}.yaml"
            config_path.write_text(yaml.safe_dump(config))
        assert main(["run", str(config_path), "--run-dir", str(tmp_path / name), *extra]) == 0
        record = json.loads((tmp_path / name / "records.jsonl").read_text())
        assert record["scores"] == {"accuracy": expected}
    assert (
        main(
            [
                "diff",
                str(tmp_path / "baseline"),
                str(tmp_path / "candidate"),
                "--fail-on-regressions",
            ]
        )
        == 2
    )


def test_harness_records_carry_evaluation_into_cli_gate(tmp_path: Path) -> None:
    import yaml

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps({"question": "Capital?", "reference_answer": "Paris"}))
    for name, response, expected in [
        ("baseline", "Paris", 1.0),
        ("candidate", "Lyon", 0.0),
    ]:
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "models": [{"type": "dummy", "args": {"canned_response": response}}],
                    "probes": [{"type": "logic"}],
                    "dataset": {"path": "dataset.jsonl", "format": "jsonl"},
                }
            )
        )
        payload = run_harness_from_config(config_path, validate_output=True)
        assert payload["records"][0]["scores"] == {"accuracy": expected}
        assert payload["experiments"][0].score.accuracy == expected
        assert main(["harness", str(config_path), "--run-dir", str(tmp_path / name)]) == 0
        record = json.loads((tmp_path / name / "records.jsonl").read_text())
        assert record["scores"] == {"accuracy": expected}
    assert (
        main(
            [
                "diff",
                str(tmp_path / "baseline"),
                str(tmp_path / "candidate"),
                "--fail-on-regressions",
            ]
        )
        == 2
    )
