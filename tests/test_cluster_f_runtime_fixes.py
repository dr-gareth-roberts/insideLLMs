"""Regression tests for Cluster F runtime/diffing/reporting fixes."""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from insideLLMs.config_types import RunConfig
from insideLLMs.exceptions import RunnerExecutionError
from insideLLMs.models import DummyModel
from insideLLMs.runtime.diffing import (
    DiffGatePolicy,
    _outputs_differ,
    build_diff_computation,
    compute_diff_exit_code,
)
from insideLLMs.runtime.runner import AsyncProbeRunner, ProbeRunner
from insideLLMs.runtime.workflows import diff_run_dirs


class _EchoProbe:
    name = "echo-probe"

    def run(self, _model: Any, item: Any, **_kwargs: Any) -> Any:
        if isinstance(item, dict) and "messages" in item:
            return item["messages"][-1]["content"]
        return str(item)

    def run_batch(
        self,
        model: Any,
        items: list[Any],
        max_workers: int = 1,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> list[Any]:
        results = []
        for i, item in enumerate(items):
            if progress_callback:
                progress_callback(i + 1, len(items))
            results.append(self.run(model, item, **kwargs))
        return results


class _ShortBatchProbe(_EchoProbe):
    def run_batch(
        self,
        model: Any,
        items: list[Any],
        max_workers: int = 1,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> list[Any]:
        # Intentionally return fewer results than inputs.
        if not items:
            return []
        return [self.run(model, items[0], **kwargs)]


class _LongBatchProbe(_EchoProbe):
    def run_batch(
        self,
        model: Any,
        items: list[Any],
        max_workers: int = 1,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> list[Any]:
        base = [self.run(model, item, **kwargs) for item in items]
        return base + ["extra"]


def _prompt(text: str = "hi") -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}]}


# ---------------------------------------------------------------------------
# F1 — batch cardinality
# ---------------------------------------------------------------------------


def test_sync_batch_cardinality_too_few_raises_before_writes(tmp_path: Path) -> None:
    runner = ProbeRunner(DummyModel(), _ShortBatchProbe())
    run_dir = tmp_path / "batch-short"
    with pytest.raises(RunnerExecutionError, match="cardinality"):
        runner.run(
            [_prompt("a"), _prompt("b")],
            use_probe_batch=True,
            emit_run_artifacts=True,
            run_dir=run_dir,
            run_id="batch-short",
            overwrite=True,
            return_experiment=False,
        )
    records_path = run_dir / "records.jsonl"
    if records_path.exists():
        assert records_path.read_text(encoding="utf-8").strip() == ""


def test_sync_batch_cardinality_too_many_raises(tmp_path: Path) -> None:
    runner = ProbeRunner(DummyModel(), _LongBatchProbe())
    with pytest.raises(RunnerExecutionError, match="cardinality"):
        runner.run(
            [_prompt("a")],
            use_probe_batch=True,
            emit_run_artifacts=False,
            return_experiment=False,
        )


@pytest.mark.asyncio
async def test_async_batch_cardinality_too_few_raises(tmp_path: Path) -> None:
    runner = AsyncProbeRunner(DummyModel(), _ShortBatchProbe())
    with pytest.raises(RunnerExecutionError, match="cardinality"):
        await runner.run(
            [_prompt("a"), _prompt("b")],
            use_probe_batch=True,
            emit_run_artifacts=True,
            run_dir=tmp_path / "async-batch-short",
            run_id="async-batch-short",
            overwrite=True,
            return_experiment=False,
        )


# ---------------------------------------------------------------------------
# F2 — unsupported combos
# ---------------------------------------------------------------------------


def test_sync_rejects_batch_with_timeout() -> None:
    runner = ProbeRunner(DummyModel(), _EchoProbe())
    with pytest.raises(ValueError, match="use_probe_batch.*timeout"):
        runner.run(
            [_prompt()],
            config=RunConfig(use_probe_batch=True, timeout=1.0, emit_run_artifacts=False),
            return_experiment=False,
        )


@pytest.mark.asyncio
async def test_async_rejects_batch_with_timeout() -> None:
    runner = AsyncProbeRunner(DummyModel(), _EchoProbe())
    with pytest.raises(ValueError, match="use_probe_batch.*timeout"):
        await runner.run(
            [_prompt()],
            use_probe_batch=True,
            timeout=1.0,
            emit_run_artifacts=False,
            return_experiment=False,
        )


@pytest.mark.asyncio
async def test_async_rejects_ultimate_with_timeout() -> None:
    runner = AsyncProbeRunner(DummyModel(), _EchoProbe())
    with pytest.raises(ValueError, match="ultimate.*timeout"):
        await runner.run(
            [_prompt()],
            config=RunConfig(run_mode="ultimate", timeout=1.0, emit_run_artifacts=False),
            return_experiment=False,
        )


# ---------------------------------------------------------------------------
# F3 — validation failure does not cascade into empty artifacts
# ---------------------------------------------------------------------------


def test_sync_record_validation_failure_preserves_partial_artifacts(tmp_path: Path) -> None:
    """Schema/serialization failure routes through interruption finalizer."""
    from insideLLMs.schemas import OutputValidationError, OutputValidator, SchemaRegistry

    runner = ProbeRunner(DummyModel(), _EchoProbe())
    run_dir = tmp_path / "val-fail"
    record_attempts = {"n": 0}
    real_validate = OutputValidator.validate

    def flaky_validate(self, schema_name, data, *args, **kwargs):  # type: ignore[no-untyped-def]
        if schema_name == SchemaRegistry.RESULT_RECORD:
            record_attempts["n"] += 1
            if record_attempts["n"] == 1:
                raise OutputValidationError(
                    schema_name="ResultRecord",
                    schema_version="1.0.2",
                    errors=["forced record validation failure"],
                )
        return real_validate(self, schema_name, data, *args, **kwargs)

    with patch.object(OutputValidator, "validate", flaky_validate):
        with pytest.raises(RunnerExecutionError, match="validation or serialization"):
            runner.run(
                [_prompt("one"), _prompt("two")],
                emit_run_artifacts=True,
                run_dir=run_dir,
                run_id="val-fail",
                overwrite=True,
                validate_output=True,
                return_experiment=False,
            )

    # Manifest + summary should still exist via interruption finalizer path.
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "summary.json").is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("run_completed") is False


# ---------------------------------------------------------------------------
# F4 — zero common records under fail_on_regressions
# ---------------------------------------------------------------------------


def test_diff_exit_code_zero_common_with_nonempty_sides() -> None:
    baseline = [
        {
            "schema_version": "1.0.1",
            "run_id": "a",
            "model": {"model_id": "m1", "provider": "local", "params": {}},
            "probe": {"probe_id": "p1", "probe_version": "1.0.0", "params": {}},
            "example_id": "only-a",
            "status": "success",
            "primary_metric": "score",
            "scores": {"score": 0.9},
            "custom": {},
        }
    ]
    candidate = [
        {
            "schema_version": "1.0.1",
            "run_id": "b",
            "model": {"model_id": "m1", "provider": "local", "params": {}},
            "probe": {"probe_id": "p1", "probe_version": "1.0.0", "params": {}},
            "example_id": "only-b",
            "status": "success",
            "primary_metric": "score",
            "scores": {"score": 0.8},
            "custom": {},
        }
    ]
    computation = build_diff_computation(
        records_baseline=baseline,
        records_candidate=candidate,
        baseline_label="a",
        candidate_label="b",
    )
    assert computation.diff_report["counts"]["common"] == 0
    assert compute_diff_exit_code(computation) == 0
    assert compute_diff_exit_code(computation, DiffGatePolicy(fail_on_regressions=True)) == 1


# ---------------------------------------------------------------------------
# F5 — structured output metadata changes are differences
# ---------------------------------------------------------------------------


def test_outputs_differ_detects_structured_metadata_change() -> None:
    rec_a = {
        "output": {"text": "same", "metadata": {"version": 1}},
        "custom": {},
    }
    rec_b = {
        "output": {"text": "same", "metadata": {"version": 2}},
        "custom": {},
    }
    assert _outputs_differ(rec_a, rec_b, ignore_keys=None) is True

    # ignore_keys applied to whole structured value
    assert _outputs_differ(rec_a, rec_b, ignore_keys={"metadata"}) is False

    # string path unchanged
    assert (
        _outputs_differ({"output": "x", "custom": {}}, {"output": "x", "custom": {}}, None) is False
    )
    assert (
        _outputs_differ({"output": "x", "custom": {}}, {"output": "y", "custom": {}}, None) is True
    )


def test_build_diff_flags_structured_metadata_change() -> None:
    baseline = [
        {
            "schema_version": "1.0.1",
            "run_id": "a",
            "model": {"model_id": "m1", "provider": "local", "params": {}},
            "probe": {"probe_id": "p1", "probe_version": "1.0.0", "params": {}},
            "example_id": "e1",
            "status": "success",
            "primary_metric": "score",
            "scores": {"score": 0.9},
            "output": {"text": "same", "metadata": {"version": 1}},
            "custom": {},
        }
    ]
    candidate = [
        {
            **baseline[0],
            "run_id": "b",
            "output": {"text": "same", "metadata": {"version": 2}},
        }
    ]
    computation = build_diff_computation(
        records_baseline=baseline,
        records_candidate=candidate,
        baseline_label="a",
        candidate_label="b",
    )
    assert computation.has_differences is True


# ---------------------------------------------------------------------------
# F6 — ultimate dataset attestation keys
# ---------------------------------------------------------------------------


def test_ultimate_dataset_attestation_uses_normalized_keys(tmp_path: Path) -> None:
    from insideLLMs.attestations import parse_dsse_envelope
    from insideLLMs.runtime._ultimate import _build_attestations

    run_dir = tmp_path / "ult"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.2",
                "run_id": "ult-1",
                "record_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "records.jsonl").write_text("", encoding="utf-8")

    _build_attestations(
        run_dir,
        dataset_merkle_root=None,
        dataset_spec={"dataset_id": "ds-42", "dataset_version": "3.1.0"},
        config_snapshot={},
        insidellms_version="0.0.0-test",
    )
    att_path = run_dir / "attestations" / "02.dataset.dsse.json"
    assert att_path.is_file()
    envelope = json.loads(att_path.read_text(encoding="utf-8"))
    statement, _ = parse_dsse_envelope(envelope)
    predicate = statement.get("predicate", {})
    assert predicate.get("dataset_id") == "ds-42"
    assert predicate.get("dataset_version") == "3.1.0"
    # Absent merkle root must not be forced to empty string.
    assert "dataset_merkle_root" not in predicate


# ---------------------------------------------------------------------------
# F7 — workflows.diff_run_dirs trajectory drift flag
# ---------------------------------------------------------------------------


def test_diff_run_dirs_forwards_fail_on_trajectory_drift(tmp_path: Path) -> None:
    with patch("insideLLMs.cli.commands.diff.cmd_diff", return_value=5) as cmd_diff:
        code = diff_run_dirs(
            tmp_path / "a",
            tmp_path / "b",
            fail_on_trajectory_drift=True,
        )
    assert code == 5
    args = cmd_diff.call_args.args[0]
    assert args.fail_on_trajectory_drift is True


# ---------------------------------------------------------------------------
# F8 — eager heavy imports avoided
# ---------------------------------------------------------------------------


def test_runtime_import_does_not_load_visualization() -> None:
    # Snapshot and temporarily drop cached modules so we can observe import
    # side-effects cleanly. Every popped entry is restored afterwards: leaving
    # "matplotlib" popped while "matplotlib.pyplot" stays cached corrupts
    # plotting state for later tests in the same process.
    watched = (
        "insideLLMs.analysis.visualization",
        "pandas",
        "matplotlib",
        "seaborn",
    )
    saved = {name: sys.modules[name] for name in watched if name in sys.modules}
    doomed = [
        name
        for name in list(sys.modules)
        if name == "insideLLMs.analysis.visualization"
        or name.startswith("insideLLMs.analysis.visualization.")
        or name in {"pandas", "matplotlib", "seaborn"}
    ]
    for name in doomed:
        sys.modules.pop(name, None)

    try:
        # Importing interruptions / high_level via runtime must not pull visualization.
        importlib.invalidate_caches()
        # Ensure analysis package is loaded without visualization star-import.
        if "insideLLMs.analysis" in sys.modules:
            # Re-import analysis to apply lazy __getattr__ if needed
            pass

        t0 = time.perf_counter()
        import insideLLMs.runtime as runtime  # noqa: F401

        elapsed = time.perf_counter() - t0
        assert "insideLLMs.analysis.visualization" not in sys.modules
        # Soft bound: should be well under the old ~2.2s pandas pull on typical CI.
        assert elapsed < 5.0
    finally:
        # Drop anything imported during the probe, then restore the snapshot.
        for name in list(sys.modules):
            if (
                name == "insideLLMs.analysis.visualization"
                or name.startswith("insideLLMs.analysis.visualization.")
                or name in {"pandas", "matplotlib", "seaborn"}
            ):
                sys.modules.pop(name, None)
        sys.modules.update(saved)

    # Documented path still works.
    from insideLLMs.analysis import visualization as viz

    assert viz is not None


# ---------------------------------------------------------------------------
# F9 — report builder keeps all records and restores scores
# ---------------------------------------------------------------------------


def test_report_builder_keeps_mixed_harness_records_and_scores() -> None:
    from insideLLMs.cli._report_builder import _build_experiments_from_records

    harness_rec = {
        "schema_version": "1.0.2",
        "run_id": "run-mixed",
        "input": "q1",
        "output": "a1",
        "status": "success",
        "scores": {"accuracy": 1.0, "score": 0.9},
        "primary_metric": "accuracy",
        "model": {"model_id": "m1", "provider": "local", "params": {}},
        "probe": {"probe_id": "p1"},
        "custom": {
            "harness": {
                "experiment_id": "exp-1",
                "model_name": "M1",
                "model_id": "m1",
                "model_type": "dummy",
                "probe_name": "p1",
                "probe_type": "logic",
                "example_index": 0,
            }
        },
    }
    plain_rec = {
        "schema_version": "1.0.2",
        "run_id": "run-mixed",
        "input": "q2",
        "output": "a2",
        "status": "success",
        "scores": {"accuracy": 0.5},
        "primary_metric": "accuracy",
        "model": {"model_id": "m1", "provider": "local", "params": {}},
        "probe": {"probe_id": "p1"},
        "custom": {},
    }
    experiments, _config, _version = _build_experiments_from_records([harness_rec, plain_rec])
    total_results = sum(len(exp.results) for exp in experiments)
    assert total_results == 2

    # Per-record scores restored (no rescoring).
    all_probe_results = [r for exp in experiments for r in exp.results]
    scores_seen = {tuple(sorted((r.scores or {}).items())) for r in all_probe_results}
    assert (("accuracy", 1.0), ("score", 0.9)) in scores_seen or any(
        r.scores.get("accuracy") == 1.0 for r in all_probe_results
    )
    assert any(r.primary_metric == "accuracy" for r in all_probe_results)

    # Aggregate score derived from persisted evidence.
    scored = [exp for exp in experiments if exp.score is not None]
    assert scored
    assert scored[0].score is not None
    assert scored[0].score.accuracy is not None or "accuracy" in scored[0].score.custom_metrics


def test_report_builder_restores_scores_on_plain_records() -> None:
    from insideLLMs.cli._report_builder import _build_experiments_from_records

    records = [
        {
            "run_id": "r1",
            "input": "q",
            "output": "a",
            "status": "success",
            "scores": {"accuracy": 0.8},
            "primary_metric": "accuracy",
            "latency_ms": 12.0,
            "model": {"model_id": "m", "provider": "p", "params": {}},
            "probe": {"probe_id": "p"},
        }
    ]
    experiments, _, _ = _build_experiments_from_records(records)
    assert len(experiments) == 1
    exp = experiments[0]
    assert exp.results[0].scores["accuracy"] == 0.8
    assert exp.results[0].primary_metric == "accuracy"
    assert exp.score is not None
    assert exp.score.accuracy == pytest.approx(0.8)
    assert exp.score.mean_latency_ms == pytest.approx(12.0)


def test_report_builder_timeout_not_counted_in_error_rate() -> None:
    """Rebuilt error_rate matches Probe.score(): ERROR only, not TIMEOUT."""
    from insideLLMs.cli._report_builder import (
        _aggregate_score_from_records,
        _build_experiments_from_records,
    )
    from insideLLMs.probes.base import Probe
    from insideLLMs.types import ProbeResult, ResultStatus

    # Single-timeout run: base score.error_rate is 0.0 (timeout ≠ error).
    class _ScoreOnlyProbe(Probe[Any]):
        def run(self, model: Any, data: Any, **kwargs: Any) -> Any:
            return data

    timeout_only = [
        ProbeResult(input="q", output=None, status=ResultStatus.TIMEOUT, error="timed out")
    ]
    base_score = _ScoreOnlyProbe(name="score-only").score(timeout_only)
    assert base_score.error_rate == 0.0

    records = [
        {
            "run_id": "r-timeout",
            "input": "q",
            "output": None,
            "status": "timeout",
            "error": "timed out",
            "scores": {},
            "model": {"model_id": "m", "provider": "p", "params": {}},
            "probe": {"probe_id": "p"},
        }
    ]
    rebuilt = _aggregate_score_from_records(records)
    assert rebuilt is not None
    assert rebuilt.error_rate == 0.0
    assert rebuilt.error_rate == base_score.error_rate

    experiments, _, _ = _build_experiments_from_records(records)
    assert len(experiments) == 1
    assert experiments[0].score is not None
    assert experiments[0].score.error_rate == 0.0

    # Pure errors still contribute to error_rate.
    error_records = [
        {
            "run_id": "r-err",
            "input": "q",
            "status": "error",
            "error": "boom",
            "scores": {},
            "model": {"model_id": "m", "provider": "p", "params": {}},
            "probe": {"probe_id": "p"},
        },
        {
            "run_id": "r-err",
            "input": "q2",
            "status": "timeout",
            "error": "timed out",
            "scores": {},
            "model": {"model_id": "m", "provider": "p", "params": {}},
            "probe": {"probe_id": "p"},
        },
    ]
    mixed = _aggregate_score_from_records(error_records)
    assert mixed is not None
    assert mixed.error_rate == pytest.approx(0.5)


def test_analysis_star_import_exports_visualization_without_eager_deps() -> None:
    """Package import/__all__ stay lazy; star-import still lists viz public names.

    Runs in a subprocess so sys.modules surgery cannot corrupt later tests
    (matplotlib submodule cache, analysis.visualization identity, etc.).
    """
    import subprocess
    from textwrap import dedent

    probe = dedent(
        r"""
        import sys

        import insideLLMs.analysis as analysis

        # Package import and __all__ must not pull visualization/pandas.
        assert "insideLLMs.analysis.visualization" not in sys.modules
        assert "pandas" not in sys.modules
        assert "matplotlib" not in sys.modules

        public_names = list(analysis.__all__)
        for required in (
            "plot_accuracy_comparison",
            "text_bar_chart",
            "ExperimentExplorer",
            "create_interactive_html_report",
        ):
            assert required in public_names, required
        assert "insideLLMs.analysis.visualization" not in sys.modules
        assert "pandas" not in sys.modules

        # Attribute access still lazy-loads.
        text_bar = analysis.text_bar_chart
        assert callable(text_bar)
        assert "insideLLMs.analysis.visualization" in sys.modules

        # Fresh package view for star-import: drop only the viz module binding.
        sys.modules.pop("insideLLMs.analysis.visualization", None)
        analysis.__dict__.pop("visualization", None)
        for name in list(analysis.__dict__):
            if name in getattr(analysis, "_VISUALIZATION_EXPORTS", ()):
                analysis.__dict__.pop(name, None)

        ns: dict = {}
        exec("from insideLLMs.analysis import *", ns)
        assert "plot_accuracy_comparison" in ns
        assert "text_bar_chart" in ns
        assert "ExperimentExplorer" in ns
        assert callable(ns["text_bar_chart"])
        print("ok")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "ok" in completed.stdout
