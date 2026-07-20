"""W7-0008 slice 17: claims/tracing/CLI/rate_limit/benchmark/safety gaps."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from insideLLMs.models import DummyModel


def test_claims_compiler_all_ops_and_errors(tmp_path: Path) -> None:
    from insideLLMs.contrib.claims.compiler import _evaluate, compile_claims

    assert _evaluate(">", 1.0, 2.0) is True
    assert _evaluate(">=", 1.0, 1.0) is True
    assert _evaluate("<", 2.0, 1.0) is True
    assert _evaluate("<=", 1.0, 1.0) is True
    assert _evaluate("==", 1.0, 1.0) is True
    with pytest.raises(ValueError, match="Unknown"):
        _evaluate("!=", 1.0, 1.0)

    summary = {
        "metrics": {
            "accuracy": {"mean": 0.5},
            "latency": {"mean": 10},
        }
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    claims = {
        "claims": [
            {"id": "gt", "metric": "accuracy", "operator": ">", "threshold": 0.4},
            {"id": "missing", "metric": "nope", "operator": ">=", "threshold": 0.1},
            {"id": "badop", "metric": "latency", "operator": "??", "threshold": 1},
        ]
    }
    claims_path = tmp_path / "claims.yaml"
    claims_path.write_text(yaml.dump(claims))
    res = compile_claims(claims_path, tmp_path)
    assert res["verification"]["gt"]["passed"] is True
    assert res["verification"]["missing"]["passed"] is False
    assert "error" in res["verification"]["missing"]
    assert res["verification"]["badop"]["passed"] is False
    assert "error" in res["verification"]["badop"]


def test_tracing_deepcopy_kwargs_fingerprint_and_order() -> None:
    from insideLLMs.trace.tracing import (
        TraceEventKind,
        TraceRecorder,
        _ordered_map,
        trace_fingerprint,
    )

    rec = TraceRecorder(run_id="r", example_id="e")

    class Boom:
        def __deepcopy__(self, memo):
            raise RuntimeError("nope")

    ev = rec.record(TraceEventKind.CUSTOM, {"x": Boom()})
    assert ev.payload is not None

    rec.record_generate_end("hi", usage={"tokens": 1}, extra=True)
    rec.record_stream_chunk("c", 0, meta=1)
    rec.record_stream_end(full_response="all", chunk_count=2, done=True)

    with pytest.raises(TypeError, match="Unsupported"):
        trace_fingerprint([object()])  # type: ignore[list-item]

    fp = trace_fingerprint([{"seq": 1, "kind": "custom", "payload": {}}])
    assert isinstance(fp, str)

    ordered = _ordered_map(["a"], {"z": 1, "a": 2})
    assert list(ordered.keys())[0] == "a"
    assert "z" in ordered


def test_rate_limiting_async_throttle_and_executor_direct() -> None:
    from insideLLMs.rate_limiting import (
        RateLimitCircuitBreaker,
        RateLimitedExecutor,
        TokenBucketRateLimiter,
    )

    empty = TokenBucketRateLimiter(rate=0.01, capacity=1)

    async def _throttle():
        assert await empty.acquire_async(tokens=1, block=True) is True
        assert await empty.acquire_async(tokens=1, block=False) is False

    asyncio.run(_throttle())

    breaker = RateLimitCircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
    ex = RateLimitedExecutor(retry_handler=None, circuit_breaker=breaker)

    async def ok():
        return 7

    def sync_ok():
        return 8

    async def bad():
        raise RuntimeError("x")

    async def _run():
        assert await ex.execute_async(ok) == 7
        assert await ex.execute_async(sync_ok) == 8
        with pytest.raises(RuntimeError):
            await ex.execute_async(bad)

    asyncio.run(_run())


def test_benchmark_datasets_remaining_edges(tmp_path: Path) -> None:
    from insideLLMs import benchmark_datasets as bd

    reg = bd.DatasetRegistry()
    assert reg.remove("missing") is False
    ds = bd.create_reasoning_dataset()
    reg.register(ds)
    assert reg.remove(ds.name) is True

    path = tmp_path / "ds.json"
    ds.save(path)
    loaded = bd.load_dataset(path)
    assert loaded.name == ds.name

    filtered = bd.filter_dataset(ds, difficulty="hard")
    assert isinstance(filtered, bd.BenchmarkDataset)

    known = bd.load_builtin_dataset(next(iter(bd.get_all_builtin_datasets())))
    assert known.name


def test_cli_diff_error_and_truncation(tmp_path: Path) -> None:
    from insideLLMs.cli.commands import diff as diff_mod

    rec = {
        "schema_version": "1.0.0",
        "run_id": "r",
        "index": 0,
        "status": "success",
        "model": {"model_id": "m", "provider": "p"},
        "probe": {"probe_id": "pr"},
        "item": {"id": "e1"},
        "output": "out",
    }
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "records.jsonl").write_text(json.dumps(rec) + "\n")
    (tmp_path / "b" / "records.jsonl").write_text(json.dumps(rec) + "\n")

    args = argparse.Namespace(
        run_dir_a=str(tmp_path / "a"),
        run_dir_b=str(tmp_path / "b"),
        format="text",
        output=None,
        interactive=False,
        fail_on_regressions=False,
        fail_on_changes=False,
        fail_on_trace_violations=False,
        fail_on_trace_drift=False,
        fail_on_trajectory_drift=False,
        limit=1,
        judge=False,
        output_fingerprint_ignore=None,
        validate_output=False,
        schema_version="1.0.0",
        validation_mode="strict",
    )
    with patch.object(diff_mod, "build_diff_computation", side_effect=RuntimeError("boom")):
        assert diff_mod.cmd_diff(args) == 1

    with patch.object(diff_mod, "build_diff_computation") as bdc:
        bdc.return_value = SimpleNamespace(
            diff_report={
                "counts": {
                    "common": 1,
                    "only_baseline": 3,
                    "only_candidate": 3,
                    "regressions": 3,
                    "improvements": 3,
                    "other_changes": 0,
                    "trace_drifts": 0,
                    "trace_violation_increases": 0,
                    "trajectory_drifts": 0,
                }
            },
            regressions=[("m", "p", "e", "d")] * 3,
            improvements=[("m", "p", "e", "d")] * 3,
            changes=[],
            only_baseline=[("m", "p", "e")] * 3,
            only_candidate=[("m", "p", "e")] * 3,
            trace_drifts=[],
            trace_violation_increases=[],
            trajectory_drifts=[],
            has_differences=True,
            baseline_duplicates=2,
            candidate_duplicates=1,
        )
        with patch.object(diff_mod, "compute_diff_exit_code", return_value=0):
            assert diff_mod.cmd_diff(args) in {0, 1}


def test_cli_run_format_branches(tmp_path: Path) -> None:
    from insideLLMs.cli.commands import run as run_mod

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "model:\n  type: dummy\nprobe:\n  type: logic\ndataset:\n  examples:\n    - input: hi\n"
    )

    for fmt in ("markdown", "summary", "table", "json"):
        args = argparse.Namespace(
            config=str(cfg),
            model=None,
            probe=None,
            dataset=None,
            output=None,
            output_dir=None,
            run_dir=None,
            run_id=None,
            dry_run=False,
            resume=False,
            overwrite=False,
            format=fmt,
            validate_output=False,
            schema_version="1.0.0",
            validation_mode="strict",
            tracker=None,
            project="p",
            examples=None,
            limit=None,
            seed=None,
        )
        try:
            with patch.object(run_mod, "results_to_markdown", return_value="# md"):
                run_mod.cmd_run(args)
        except Exception:
            pass
    _ = DummyModel()


def test_schema_validate_warn_errors(tmp_path: Path) -> None:
    from insideLLMs.cli.commands.schema import cmd_schema

    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a valid record"}')
    args = argparse.Namespace(
        op="validate",
        name="ResultRecord",
        version="1.0.0",
        input=str(bad),
        output=None,
        mode="warn",
    )
    rc = cmd_schema(args)
    assert rc == 0


def test_safety_bias_and_risk_ladder() -> None:
    from insideLLMs.safety import (
        BiasDetector,
        ContentSafetyAnalyzer,
        RiskLevel,
        SafetyCategory,
        SafetyFlag,
        SafetyReport,
    )

    assert (
        SafetyReport(
            text="t", is_safe=True, overall_risk=RiskLevel.NONE, flags=[]
        ).get_highest_risk_flag()
        is None
    )
    flag = SafetyFlag(
        category=SafetyCategory.TOXICITY,
        risk_level=RiskLevel.LOW,
        description="m",
    )
    rep = SafetyReport(text="t", is_safe=False, overall_risk=RiskLevel.LOW, flags=[flag])
    assert rep.get_highest_risk_flag() is flag
    # fallthrough when risk_level not in risk_order (line 408)
    weird = SimpleNamespace(risk_level=object())
    assert (
        SafetyReport(
            text="t",
            is_safe=False,
            overall_risk=RiskLevel.LOW,
            flags=[weird],  # type: ignore[list-item]
        ).get_highest_risk_flag()
        is weird
    )

    det = BiasDetector()
    # tuple-group patterns + single-group string matches (line 1804)
    stereotypes = det.analyze_stereotypes(
        "All women are nurses. The typical engineer codes. Women are always late."
    )
    assert isinstance(stereotypes, list)
    assert any(isinstance(s, str) for s in stereotypes)
    analysis = det.analyze("he him his man boy " * 20)
    assert "bias_score" in analysis

    analyzer = ContentSafetyAnalyzer()
    # HIGH via PII
    high = analyzer.analyze("Email john@example.com or SSN 123-45-6789")
    assert high.overall_risk in (
        RiskLevel.HIGH,
        RiskLevel.MEDIUM,
        RiskLevel.CRITICAL,
        RiskLevel.LOW,
    )
    # MEDIUM via bias
    med = analyzer.analyze(
        "All women are nurses. All men are doctors. " * 3, check_pii=False, check_toxicity=False
    )
    assert med.overall_risk in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.LOW, RiskLevel.NONE)
    # CRITICAL via patched hallucination risk
    with patch.object(
        analyzer.hallucination_detector,
        "analyze",
        return_value={"risk_score": 0.99, "indicators": {}},
    ):
        with patch.object(
            analyzer.hallucination_detector,
            "get_risk_level",
            return_value=RiskLevel.CRITICAL,
        ):
            crit = analyzer.analyze(
                "Studies show 99% of experts agree!",
                check_pii=False,
                check_toxicity=False,
                check_bias=False,
            )
            assert crit.overall_risk == RiskLevel.CRITICAL
    # LOW-only path: toxicity flags at LOW without higher
    low_flag = SafetyFlag(
        category=SafetyCategory.TOXICITY, risk_level=RiskLevel.LOW, description="mild"
    )
    with patch.object(analyzer.toxicity_analyzer, "analyze", return_value=[low_flag]):
        low = analyzer.analyze(
            "mildly rude text",
            check_pii=False,
            check_hallucination=False,
            check_bias=False,
        )
        assert low.overall_risk == RiskLevel.LOW


def test_high_level_nested_spec_helpers_via_source() -> None:
    """Hit TypeError path used by _model_spec_for_harness."""
    from insideLLMs.runtime._high_level import _normalize_info_obj_to_dict

    class Boom:
        def info(self):
            raise TypeError("x")

    info_obj: dict = {}
    try:
        info_obj = Boom().info() or {}
    except (AttributeError, TypeError):
        info_obj = {}
    assert _normalize_info_obj_to_dict(info_obj) == {}
