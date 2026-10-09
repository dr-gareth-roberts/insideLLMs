"""Locks for the product spine: smoke datasets, one score/spend envelope, and runner parity."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from insideLLMs.analysis.evaluation import EvaluationResult
from insideLLMs.benchmark_datasets import (
    SMOKE_SCALE,
    DatasetBuilder,
    get_all_builtin_datasets,
    list_builtin_datasets,
    load_builtin_dataset,
)
from insideLLMs.exceptions import APIError, structured_provider_error
from insideLLMs.inference.schemas import Spend
from insideLLMs.models import DummyModel
from insideLLMs.probes.base import Probe
from insideLLMs.runtime.runner import AsyncProbeRunner, ProbeRunner
from insideLLMs.types import ProbeResult, ResultStatus, ScoreSpendEnvelope

_ROOT = Path(__file__).resolve().parents[1]
_FROZEN_CONTRIB_EXPORTS = frozenset(
    {
        "APIKeyAuth",
        "AdversarialGenerator",
        "AdversarialType",
        "AgentConfig",
        "AgentExecutor",
        "AgentResult",
        "Annotation",
        "AnnotationCollector",
        "AnnotationWorkflow",
        "AppConfig",
        "ApprovalWorkflow",
        "BatchEndpoint",
        "ChainOfThoughtAgent",
        "ConsensusValidator",
        "DataAugmenter",
        "DeploymentApp",
        "DeploymentConfig",
        "EndpointConfig",
        "Feedback",
        "FeedbackCollector",
        "FeedbackType",
        "HITLConfig",
        "HITLSession",
        "HealthChecker",
        "HumanValidator",
        "IntentClassifier",
        "InteractiveSession",
        "MetricsCollector",
        "ModelBenchmark",
        "ModelEndpoint",
        "ModelPool",
        "Priority",
        "PriorityReviewQueue",
        "ProbeBenchmark",
        "ProbeEndpoint",
        "PromptVariator",
        "RateLimiter",
        "ReActAgent",
        "ReviewItem",
        "ReviewQueue",
        "ReviewStatus",
        "ReviewWorkflow",
        "Route",
        "RouteMatch",
        "RouterConfig",
        "RoutingStrategy",
        "SemanticRouter",
        "SimpleAgent",
        "SynthesisConfig",
        "SyntheticDataset",
        "TemplateGenerator",
        "Tool",
        "ToolRegistry",
        "VariationStrategy",
        "collect_feedback",
        "create_app",
        "create_calculator_tool",
        "create_hitl_session",
        "create_model_endpoint",
        "create_probe_endpoint",
        "create_react_agent",
        "create_router",
        "create_simple_agent",
        "generate_test_dataset",
        "quick_adversarial",
        "quick_agent_run",
        "quick_deploy",
        "quick_review",
        "quick_route",
        "quick_variations",
    }
)


class _FailingProbe(Probe[str]):
    def run(self, model, data, **kwargs):  # type: ignore[no-untyped-def]
        if data == "fail":
            raise APIError(
                "dummy",
                status_code=503,
                message="unavailable",
                response_body="secret-body",
            )
        return model.generate(str(data))


def _statuses(run_dir: Path) -> list[str]:
    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    return [record["status"] for record in records]


def test_builtin_datasets_stay_smoke_fixtures() -> None:
    datasets = get_all_builtin_datasets()
    assert datasets
    listed = {item["name"]: item for item in list_builtin_datasets()}
    assert set(listed) == set(datasets)
    for name, dataset in datasets.items():
        assert dataset.scale == SMOKE_SCALE
        assert listed[name]["scale"] == SMOKE_SCALE
        assert "smoke" in dataset.description.lower()
        assert dataset._examples
        assert all(example.metadata.get("scale") == SMOKE_SCALE for example in dataset)
        loaded = load_builtin_dataset(name)
        assert loaded.scale == SMOKE_SCALE

    custom = DatasetBuilder("custom").with_description("A caller-built set").build()
    assert custom.scale is None


def test_score_spend_envelope_is_shared() -> None:
    probe = ProbeResult(
        input="q",
        output="a",
        status=ResultStatus.SUCCESS,
        latency_ms=1500.0,
        scores={"accuracy": 1.0, "score": 0.5},
        primary_metric="score",
        metadata={"is_correct": True},
    )
    evaluation = EvaluationResult(
        score=0.5, passed=True, metric_name="score", details={"is_correct": True}
    )
    spend = Spend(calls=2, input_tokens=10, output_tokens=4, elapsed_seconds=1.5, cost=0.01)
    probe_envelope = probe.score_spend()
    eval_envelope = evaluation.to_envelope(spend.to_snapshot())

    assert isinstance(probe_envelope, ScoreSpendEnvelope)
    assert probe_envelope.to_dict()["score"] == 0.5
    assert probe_envelope.passed is True
    assert probe_envelope.spend.calls == 1
    assert probe_envelope.spend.elapsed_seconds == 1.5
    assert set(eval_envelope.to_dict()) == {"score", "passed", "metric_name", "details", "spend"}
    assert eval_envelope.spend.to_dict()["cost"] == 0.01
    assert eval_envelope.score == probe_envelope.score

    with pytest.raises(ValueError):
        ScoreSpendEnvelope(score=float("nan"), passed=None)


def test_structured_provider_error_omits_response_bodies() -> None:
    payload = structured_provider_error(
        APIError("dummy", status_code=503, message="unavailable", response_body="secret-body")
    )
    assert payload is not None
    assert payload["error_type"] == "APIError"
    assert payload["status_code"] == 503
    assert payload["model_id"] == "dummy"
    assert "response_body" not in payload
    assert "secret-body" not in json.dumps(payload)
    assert structured_provider_error(RuntimeError("plain")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_stop_on_error_records_match_and_keep_provider_error(tmp_path, asynchronous):
    runner = (AsyncProbeRunner if asynchronous else ProbeRunner)(
        DummyModel(), _FailingProbe(name="fail")
    )
    run_dir = tmp_path / ("async" if asynchronous else "sync")
    with pytest.raises(Exception):
        result = runner.run(
            ["ok", "fail", "later"],
            stop_on_error=True,
            run_dir=run_dir,
        )
        if asynchronous:
            await result

    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [record["status"] for record in records] == ["success", "error"]
    assert all(record["status"] != "skipped" for record in records)
    provider_error = records[1]["custom"]["provider_error"]
    assert provider_error["error_type"] == "APIError"
    assert provider_error["status_code"] == 503
    assert "response_body" not in provider_error
    assert "secret-body" not in json.dumps(records[1])


@pytest.mark.asyncio
async def test_sync_and_async_stop_on_error_write_the_same_statuses(tmp_path) -> None:
    sync_dir = tmp_path / "sync"
    async_dir = tmp_path / "async"

    with pytest.raises(Exception):
        ProbeRunner(DummyModel(), _FailingProbe(name="fail")).run(
            ["ok", "fail", "later"],
            stop_on_error=True,
            run_dir=sync_dir,
        )
    with pytest.raises(Exception):
        await AsyncProbeRunner(DummyModel(), _FailingProbe(name="fail")).run(
            ["ok", "fail", "later"],
            stop_on_error=True,
            run_dir=async_dir,
        )

    assert _statuses(sync_dir) == _statuses(async_dir) == ["success", "error"]
    sync_error = json.loads((sync_dir / "records.jsonl").read_text().splitlines()[1])
    async_error = json.loads((async_dir / "records.jsonl").read_text().splitlines()[1])
    assert sync_error["custom"]["provider_error"] == async_error["custom"]["provider_error"]


def test_root_lazy_api_does_not_grow_contrib_exports() -> None:
    tree = ast.parse((_ROOT / "insideLLMs" / "__init__.py").read_text(encoding="utf-8"))
    lazy: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "_LAZY_IMPORTS" for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        assert isinstance(value, dict)
        lazy = {str(name): str(module) for name, module in value.items()}
    contrib = {name for name, module in lazy.items() if module.startswith("insideLLMs.contrib")}
    assert contrib == _FROZEN_CONTRIB_EXPORTS
