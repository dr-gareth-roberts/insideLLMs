"""Persisted built-in probe outputs retain the same aggregate scoring contract."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from insideLLMs.probes.agent_probe import AgentProbe, AgentProbeResult
from insideLLMs.probes.attack import AttackProbe
from insideLLMs.probes.bias import BiasProbe
from insideLLMs.runtime.runner import AsyncProbeRunner, ProbeRunner
from insideLLMs.types import AttackResult, BiasResult, ProbeResult, ResultStatus


class OfflineAgentProbe(AgentProbe):
    def run_agent(self, model, prompt, tools, recorder, **kwargs):
        return model.generate(prompt)


class CountingModel:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls += 1
        return "I cannot comply with that request."


def _successful(output):
    return [ProbeResult(input="input", output=output, status=ResultStatus.SUCCESS)]


@pytest.mark.parametrize(
    ("probe", "typed_output", "metric", "expected"),
    [
        (
            AttackProbe(),
            AttackResult("attack", "blocked", "general", False, "low", []),
            "attack_success_rate",
            0.0,
        ),
        (
            BiasProbe(),
            BiasResult("a", "b", "good", "bad", "general", sentiment_diff=0.25, length_diff=4),
            "avg_sentiment_diff",
            0.25,
        ),
        (
            OfflineAgentProbe(name="offline"),
            AgentProbeResult("prompt", "answer", violations=[{"code": "blocked"}]),
            "violation_rate",
            1.0,
        ),
    ],
)
def test_builtin_scores_are_identical_for_typed_and_persisted_mapping_outputs(
    probe, typed_output, metric, expected
):
    typed_score = probe.score(_successful(typed_output))
    mapping_score = probe.score(_successful(asdict(typed_output)))

    assert typed_score == mapping_score
    assert mapping_score.custom_metrics[metric] == expected


@pytest.mark.parametrize(
    ("probe", "output"),
    [
        (AttackProbe(), {}),
        (AttackProbe(), {"attack_succeeded": "false", "severity": "low"}),
        (BiasProbe(), {}),
        (BiasProbe(), {"sentiment_diff": "0.1", "length_diff": 1}),
        (OfflineAgentProbe(name="offline"), {}),
        (OfflineAgentProbe(name="offline"), {"violations": "none"}),
    ],
)
def test_builtin_scores_reject_malformed_persisted_mapping_outputs(probe, output):
    with pytest.raises((KeyError, TypeError, ValueError)):
        probe.score(_successful(output))


def _probe_case(name: str):
    if name == "attack":
        return AttackProbe(), ["attack one"], 1
    if name == "bias":
        return BiasProbe(), [{"prompt_pairs": [["person a", "person b"]]}], 2
    return (
        OfflineAgentProbe(name="offline", trace_config={"contracts": {"enabled": False}}),
        [{"prompt": "agent one"}],
        1,
    )


def _run(runner, dataset, asynchronous: bool, **kwargs):
    result = runner.run(dataset, **kwargs)
    return asyncio.run(result) if asynchronous else result


def _is_saved_mapping_output(probe_name: str, output: Any) -> bool:
    return isinstance(output[0], dict) if probe_name == "bias" else isinstance(output, dict)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("probe_name", "schema_version"),
    [("attack", "1.0.0"), ("bias", "1.0.1"), ("agent", "1.0.2")],
)
def test_builtin_full_resume_scores_saved_mappings_without_model_calls(
    tmp_path: Path, asynchronous: bool, probe_name: str, schema_version: str
):
    probe, dataset, calls_per_input = _probe_case(probe_name)
    model = CountingModel()
    runner_class = AsyncProbeRunner if asynchronous else ProbeRunner
    runner = runner_class(model, probe)
    run_dir = tmp_path / f"{probe_name}-{asynchronous}"
    initial = _run(
        runner,
        dataset,
        asynchronous,
        run_dir=run_dir,
        schema_version=schema_version,
        validate_output=True,
    )
    initial_score = runner.last_experiment.score
    records_path = run_dir / "records.jsonl"
    records_before = records_path.read_bytes()

    resumed = _run(
        runner,
        dataset,
        asynchronous,
        run_dir=run_dir,
        schema_version=schema_version,
        validate_output=True,
        resume=True,
    )

    assert len(resumed) == len(initial)
    assert model.calls == calls_per_input
    assert runner.last_experiment.score == initial_score
    assert _is_saved_mapping_output(probe_name, runner.last_experiment.results[0].output)
    assert records_path.read_bytes() == records_before


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("probe_name", ["attack", "bias", "agent"])
def test_builtin_partial_resume_mixes_saved_and_live_outputs_and_runs_only_missing_inputs(
    tmp_path: Path, asynchronous: bool, probe_name: str
):
    probe, dataset, calls_per_input = _probe_case(probe_name)
    model = CountingModel()
    runner_class = AsyncProbeRunner if asynchronous else ProbeRunner
    runner = runner_class(model, probe)
    run_dir = tmp_path / f"{probe_name}-{asynchronous}"
    if probe_name == "bias":
        second = {"prompt_pairs": [["person c", "person d"]]}
    elif probe_name == "agent":
        second = {"prompt": "agent two"}
    else:
        second = "attack two"
    full_dataset = [*dataset, second]
    _run(runner, full_dataset, asynchronous, run_dir=run_dir)
    record_path = run_dir / "records.jsonl"
    first_record = record_path.read_text().splitlines()[0]
    record_path.write_text(first_record + "\n")

    results = _run(runner, full_dataset, asynchronous, run_dir=run_dir, resume=True)

    assert len(results) == 2
    assert model.calls == calls_per_input * 3
    assert _is_saved_mapping_output(probe_name, runner.last_experiment.results[0].output)
    assert not _is_saved_mapping_output(probe_name, runner.last_experiment.results[1].output)
