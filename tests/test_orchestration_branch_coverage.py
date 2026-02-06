"""Additional branch coverage for orchestration helpers."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from insideLLMs.orchestration import (
    AsyncBatchRunner,
    BatchRunner,
    ExperimentBatchResult,
    ExperimentConfig,
    ExperimentGrid,
    ExperimentOrchestrator,
    ExperimentQueue,
    ExperimentRun,
    ExperimentStatus,
    MetricsCalculator,
    ModelExecutor,
    ProgressCallback,
)


def _cfg(exp_id: str, prompt: str = "p", model: str = "m") -> ExperimentConfig:
    return ExperimentConfig(id=exp_id, name=exp_id, prompt=prompt, model_id=model)


def test_experiment_run_duration_uses_timestamps_when_available():
    run = ExperimentRun(config=_cfg("e1"), latency_ms=123.0)
    run.started_at = datetime(2025, 1, 1, 0, 0, 0)
    run.completed_at = run.started_at + timedelta(seconds=2.5)
    assert run.duration_ms == 2500.0


def test_batch_result_empty_paths_and_model_filtering():
    runs = [
        ExperimentRun(config=_cfg("a", model="model-a"), status=ExperimentStatus.FAILED, error="x"),
        ExperimentRun(config=_cfg("b", model="model-b"), status=ExperimentStatus.PENDING),
    ]
    batch = ExperimentBatchResult(batch_id="b", runs=runs)
    assert batch.avg_latency_ms == 0.0
    assert batch.aggregate_metrics() == {}
    assert [r.config.id for r in batch.get_by_model("model-b")] == ["b"]


def test_protocol_placeholder_methods_are_callable():
    assert ProgressCallback.__call__(object(), 1, 2, None) is None
    assert ModelExecutor.__call__(object(), "prompt") is None
    assert MetricsCalculator.__call__(object(), "p", "r", _cfg("x")) is None


def test_experiment_queue_add_batch_empty_next_and_results_copy():
    queue = ExperimentQueue()
    queue.add_batch([_cfg("1"), _cfg("2")])
    assert queue.pending_count == 2
    first = queue.next()
    second = queue.next()
    assert first is not None and second is not None
    assert queue.next() is None

    queue.complete(ExperimentRun(config=first, status=ExperimentStatus.COMPLETED))
    results = queue.get_results()
    assert len(results) == 1
    results.append(ExperimentRun(config=second, status=ExperimentStatus.FAILED))
    assert len(queue.get_results()) == 1


def test_batch_runner_retry_and_parallel_mode_switch():
    calls = {"n": 0}

    def flaky_executor(prompt: str, **kwargs) -> str:
        _ = prompt, kwargs
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first call fails")
        return "ok"

    runner = BatchRunner(executor=flaky_executor, retry_on_failure=True, max_retries=1)
    result = runner.run_sequential([_cfg("retry")], batch_id="retry-batch")
    assert result.completed == 1
    assert calls["n"] == 2

    progress = []
    parallel_runner = BatchRunner(executor=lambda prompt, **_: f"resp:{prompt}", max_workers=2)
    parallel_runner.on_progress(lambda c, t, run: progress.append((c, t, run.config.id)))
    parallel_result = parallel_runner.run([_cfg("p1"), _cfg("p2")], parallel=True)
    assert parallel_result.completed == 2
    assert len(progress) == 2


@pytest.mark.asyncio
async def test_async_batch_runner_failure_metrics_and_progress_paths():
    async def executor(prompt: str, **kwargs) -> str:
        _ = kwargs
        if prompt == "boom":
            raise RuntimeError("broken")
        return f"ok:{prompt}"

    def metrics(prompt: str, response: str, config: ExperimentConfig) -> dict[str, float]:
        _ = prompt, config
        return {"response_len": float(len(response))}

    runner = AsyncBatchRunner(executor=executor, metrics_calculator=metrics, max_concurrent=1)
    progress = []
    chained = runner.on_progress(lambda c, t, current=None: progress.append((c, t)))
    assert chained is runner

    result = await runner.run([_cfg("ok1", prompt="ok"), _cfg("bad", prompt="boom")])
    assert len(result.runs) == 2
    assert result.failed == 1
    assert any(
        "response_len" in r.metrics for r in result.runs if r.status == ExperimentStatus.COMPLETED
    )
    assert progress == []


def test_experiment_grid_len_without_parameters_uses_single_param_count():
    grid = ExperimentGrid(base_prompt="base", model_ids=["m1", "m2"], parameters={})
    assert len(grid) == 2


def test_experiment_orchestrator_run_configs_get_batch_and_aggregate_report():
    orchestrator = ExperimentOrchestrator(executor=lambda prompt, **_: f"answer:{prompt}")
    orchestrator.set_metrics_calculator(lambda p, r, c: {"chars": float(len(r))})

    configs = [_cfg("c1", prompt="a"), _cfg("c2", prompt="bb")]
    result = orchestrator.run_configs(configs, parallel=False, batch_id="custom")

    assert result.batch_id == "custom"
    assert orchestrator.get_batch("custom") is result
    assert "custom" in orchestrator.list_batches()

    agg = orchestrator.aggregate_all()
    assert agg["chars"]["count"] == 2
    assert agg["chars"]["max"] >= agg["chars"]["min"]

    report = orchestrator.generate_report()
    assert "Aggregate Metrics" in report
    assert "**chars**" in report
