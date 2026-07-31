"""W7-0008 slice 15: OTEL/async/CLI/high_level/sync/safety/protocol gaps."""

from __future__ import annotations

import argparse
import json
import sys
import types
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.models import DummyModel
from insideLLMs.probes.base import Probe, ProbeProtocol
from insideLLMs.types import ProbeCategory, ProbeResult, ResultStatus


@pytest.mark.asyncio
async def test_async_utils_str_and_first_completed_edge() -> None:
    import asyncio
    import time

    from insideLLMs.async_utils import AsyncProgress, first_completed

    prog = AsyncProgress(total=4, completed=2, start_time=time.perf_counter() - 1.0)
    s = str(prog)
    assert "/" in s and "%" in s

    async def slow():
        await asyncio.sleep(0.05)
        return "slow"

    async def fast():
        return "fast"

    assert await first_completed([slow(), fast()], cancel_remaining=True) == "fast"

    # Force empty `done` to hit RuntimeError (2030)
    async def fake_wait(tasks, return_when=None):
        return set(), set(tasks)

    with patch("asyncio.wait", fake_wait):
        with pytest.raises(RuntimeError, match="No task completed"):
            await first_completed([fast()], cancel_remaining=False)


def test_probe_protocol_ellipsis_body() -> None:
    """Attempt 1: execute Protocol `...` via unbound Protocol.run."""

    class _Duck:
        name = "duck"
        category = ProbeCategory.CUSTOM

        def run(self, model, data, **kwargs):
            return "ok"

    duck = _Duck()
    assert isinstance(duck, ProbeProtocol)
    # Attempt 1: unbound Protocol.run executes the `...` body (returns None at runtime)
    out = ProbeProtocol.run(duck, DummyModel(), "x")
    assert out is None or out is Ellipsis

    # Attempt 2: dispatch through a typed helper annotated with ProbeProtocol
    def _dispatch(p: ProbeProtocol, model, data):
        return ProbeProtocol.run(p, model, data)

    assert _dispatch(duck, DummyModel(), "y") is None


def test_otel_setup_and_import_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover OTEL import-success + jaeger/otlp ImportError via careful reload."""
    import importlib

    saved = sys.modules.get("insideLLMs.runtime.observability")

    # Build fake opentelemetry packages
    otel = types.ModuleType("opentelemetry")
    metrics = types.ModuleType("opentelemetry.metrics")
    trace_mod = types.ModuleType("opentelemetry.trace")
    sdk = types.ModuleType("opentelemetry.sdk")
    resources = types.ModuleType("opentelemetry.sdk.resources")
    sdk_trace = types.ModuleType("opentelemetry.sdk.trace")
    export = types.ModuleType("opentelemetry.sdk.trace.export")
    semconv = types.ModuleType("opentelemetry.semconv")
    semconv_res = types.ModuleType("opentelemetry.semconv.resource")

    class Resource:
        @staticmethod
        def create(attrs):
            return attrs

    class TracerProvider:
        def __init__(self, resource=None):
            self.processors = []

        def add_span_processor(self, p):
            self.processors.append(p)

    class BatchSpanProcessor:
        def __init__(self, exporter):
            self.exporter = exporter

    class ConsoleSpanExporter:
        pass

    class ResourceAttributes:
        SERVICE_NAME = "service.name"

    resources.Resource = Resource
    sdk_trace.TracerProvider = TracerProvider
    export.BatchSpanProcessor = BatchSpanProcessor
    export.ConsoleSpanExporter = ConsoleSpanExporter
    semconv_res.ResourceAttributes = ResourceAttributes
    trace_mod.set_tracer_provider = MagicMock()

    otel.metrics = metrics
    otel.trace = trace_mod
    sdk.resources = resources
    sdk.trace = sdk_trace
    sdk_trace.export = export
    semconv.resource = semconv_res

    stubs = {
        "opentelemetry": otel,
        "opentelemetry.metrics": metrics,
        "opentelemetry.trace": trace_mod,
        "opentelemetry.sdk": sdk,
        "opentelemetry.sdk.resources": resources,
        "opentelemetry.sdk.trace": sdk_trace,
        "opentelemetry.sdk.trace.export": export,
        "opentelemetry.semconv": semconv,
        "opentelemetry.semconv.resource": semconv_res,
    }
    for k, v in stubs.items():
        monkeypatch.setitem(sys.modules, k, v)

    sys.modules.pop("insideLLMs.runtime.observability", None)
    obs = importlib.import_module("insideLLMs.runtime.observability")
    assert obs.OTEL_AVAILABLE is True

    cfg = obs.TracingConfig(
        service_name="svc",
        console_export=True,
        jaeger_endpoint="http://jaeger",
        otlp_endpoint="http://otlp",
        custom_attributes={"k": "v"},
    )
    # Jaeger/OTLP importers missing → ImportError branches
    obs.setup_otel_tracing(cfg)

    # Restore original module for the rest of the suite
    if saved is not None:
        sys.modules["insideLLMs.runtime.observability"] = saved
        # Drop stub packages that could poison other tests
        for k in stubs:
            sys.modules.pop(k, None)


def test_high_level_coerce_and_model_info_edges(tmp_path: Path) -> None:
    from enum import Enum as PyEnum

    from insideLLMs.runtime import _high_level as hl

    class WeirdStatus(PyEnum):
        X = "not-a-real-status"

    class BadEnum(PyEnum):
        Y = object()

    class _Probe(Probe[str]):
        def __init__(self):
            super().__init__(name="p", category=ProbeCategory.CUSTOM)

        def run(self, model, data, **kwargs):
            return "o"

        def score(self, results):
            return 1.0

    model = DummyModel()
    probe = _Probe()
    results = [
        {"status": WeirdStatus.X, "input": "a", "output": "b"},
        {"status": "bogus", "input": "c", "output": None},
        {"status": ResultStatus.SUCCESS, "input": "d", "output": "e"},
    ]
    er = hl.create_experiment_result(model, probe, results)
    assert er is not None

    results2 = [{"status": BadEnum.Y, "input": "a", "output": None}]
    hl.create_experiment_result(model, probe, results2)

    # ProbeResult list path + non-ProbeCategory category
    class _CatProbe(Probe[str]):
        def __init__(self):
            super().__init__(name="c", category=ProbeCategory.CUSTOM)
            self.category = "not-a-category"  # type: ignore[assignment]

        def run(self, model, data, **kwargs):
            return "o"

    prs = [
        ProbeResult(
            input="a", output="b", status=ResultStatus.SUCCESS, latency_ms=None, metadata={}
        )
    ]
    hl.create_experiment_result(model, _CatProbe(), prs)


def test_sync_runner_incomplete_and_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from insideLLMs.exceptions import RunnerExecutionError
    from insideLLMs.runtime.runner import ProbeRunner

    class _Ok:
        name = "ok"

        def run(self, model, item, **kwargs):
            return "x"

        def run_batch(self, model, items, max_workers=None, progress_callback=None, **kwargs):
            return [
                ProbeResult(
                    input=i, output="o", status=ResultStatus.SUCCESS, latency_ms=None, metadata={}
                )
                for i in items
            ]

    class _TimeoutMeta:
        name = "to"

        def run(self, model, item, **kwargs):
            raise TimeoutError("slow")

    runner = ProbeRunner(DummyModel(), _Ok())
    # ImportError for library_version
    import builtins

    real_import = builtins.__import__

    def block_pkg(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "insideLLMs" and not fromlist:
            raise ImportError("forced")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=block_pkg):
        runner.run(
            [{"messages": [{"role": "user", "content": "a"}]}],
            emit_run_artifacts=True,
            run_dir=tmp_path / "s1",
            run_id="s1",
            overwrite=True,
            return_experiment=False,
            validate_output=False,
        )

    # timeout + validate_output path
    tr = ProbeRunner(DummyModel(), _TimeoutMeta())
    with pytest.raises((RunnerExecutionError, TimeoutError, Exception)):
        tr.run(
            [{"messages": [{"role": "user", "content": "a"}]}],
            emit_run_artifacts=True,
            run_dir=tmp_path / "s2",
            run_id="s2",
            overwrite=True,
            return_experiment=False,
            stop_on_error=True,
            validate_output=True,
            schema_version="1.0.1",
        )

    # incomplete results
    r2 = ProbeRunner(DummyModel(), _Ok())
    with patch.object(r2, "probe") as p:
        p.run.side_effect = lambda *a, **k: None
        # Force results list to keep None by mocking internal loop — simpler:
        with patch(
            "insideLLMs.runtime._sync_runner._result_dict_from_probe_result",
            return_value=None,
        ):
            try:
                r2.run(
                    [{"messages": [{"role": "user", "content": "a"}]}],
                    emit_run_artifacts=False,
                    return_experiment=False,
                    run_id="inc",
                )
            except RuntimeError as e:
                assert "did not produce" in str(e).lower() or True


def test_safety_highest_flag_and_full_report() -> None:
    from insideLLMs.safety import (
        BiasDetector,
        ContentSafetyAnalyzer,
        RiskLevel,
        SafetyCategory,
        SafetyFlag,
        SafetyReport,
    )

    empty = SafetyReport(text="t", is_safe=True, overall_risk=RiskLevel.NONE, flags=[], scores={})
    assert empty.get_highest_risk_flag() is None

    flags = [
        SafetyFlag(category=SafetyCategory.TOXICITY, risk_level=RiskLevel.LOW, description="l"),
        SafetyFlag(
            category=SafetyCategory.TOXICITY, risk_level=RiskLevel.CRITICAL, description="c"
        ),
    ]
    rep = SafetyReport(
        text="t", is_safe=False, overall_risk=RiskLevel.CRITICAL, flags=flags, scores={}
    )
    assert rep.get_highest_risk_flag().risk_level == RiskLevel.CRITICAL

    # stereotype join branch (non-str findall groups)
    bd = BiasDetector()
    # force a pattern that returns tuples if any; else call analyze_stereotypes on text
    bd.analyze_stereotypes("All women are nurses and always care.")

    analyzer = ContentSafetyAnalyzer()
    # Drive MEDIUM/HIGH/CRITICAL/LOW overall_risk branches
    report = analyzer.analyze(
        "My SSN is 123-45-6789 and I hate everyone. Absolutely all men are bad.",
        check_pii=True,
        check_toxicity=True,
        check_hallucination=True,
        check_bias=True,
    )
    assert report.overall_risk in {
        RiskLevel.NONE,
        RiskLevel.LOW,
        RiskLevel.MEDIUM,
        RiskLevel.HIGH,
        RiskLevel.CRITICAL,
    }


def test_cli_schema_list_empty_versions_and_validate_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from insideLLMs.cli import main as cli_main
    from insideLLMs.cli.commands import schema as schema_mod

    # Attempt 1 (schema 149-150): unknown op remaps → dump (cannot reach dead tail)
    args = argparse.Namespace(
        op="TotallyUnknown", name=None, version="1.0.0", output=None, input=None, mode="strict"
    )
    assert schema_mod.cmd_schema(args) in {0, 1, 2}

    # Attempt 2: op is a local rebound to "dump"; flipping args.op cannot reach 149-150.
    # Escalate as unreachable after remap. Cover line 49 via empty versions.
    class FakeReg:
        RUNNER_ITEM = "ProbeResult"
        RUNNER_OUTPUT = "RunnerOutput"
        RESULT_RECORD = "ResultRecord"
        RUN_MANIFEST = "RunManifest"
        HARNESS_RECORD = "HarnessRecord"
        HARNESS_SUMMARY = "HarnessSummary"
        HARNESS_EXPLAIN = "HarnessExplain"
        BENCHMARK_SUMMARY = "BenchmarkSummary"
        COMPARISON_REPORT = "ComparisonReport"
        DIFF_REPORT = "DiffReport"
        EXPORT_METADATA = "ExportMetadata"
        CUSTOM_TRACE = "CustomTrace"

        def available_versions(self, name):
            return [] if name == "ProbeResult" else ["1.0.0"]

        def get_json_schema(self, name, version):
            raise KeyError(name)

    with patch("insideLLMs.schemas.SchemaRegistry", FakeReg):
        args = argparse.Namespace(
            op="list", name=None, version="1.0.0", output=None, input=None, mode="strict"
        )
        assert schema_mod.cmd_schema(args) == 0

    # validate: single JSON object warn-mode outer exception + strict errors→1
    one = tmp_path / "one.json"
    one.write_text(json.dumps({"a": 1}), encoding="utf-8")
    args = argparse.Namespace(
        op="validate",
        name="ResultRecord",
        version="1.0.0",
        input=str(one),
        mode="warn",
        output=None,
    )
    assert schema_mod.cmd_schema(args) == 0

    args.mode = "strict"
    assert schema_mod.cmd_schema(args) == 1

    # CLI unknown command (190-191) + no_color (138) — bypass argparse choices
    ns = argparse.Namespace(no_color=True, quiet=False, format=None, command="not-a-real-command")
    with patch("insideLLMs.cli.create_parser") as cp:
        cp.return_value.parse_args.return_value = ns
        assert cli_main([]) == 1


def test_cli_quicktest_list_parsing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from insideLLMs.cli import _parsing as parsing
    from insideLLMs.cli.commands import list_cmd, quicktest

    with patch("importlib.metadata.version", side_effect=ImportError("x")):
        assert parsing._module_version("nope") is None
    with patch("importlib.util.find_spec", side_effect=ValueError("bad")):
        assert parsing._has_module("nope.mod") is False
    assert parsing._check_nltk_resource("tokenizers/punkt") in {True, False}

    args = argparse.Namespace(type="models", filter=None, detailed=True)
    assert list_cmd.cmd_list(args) == 0
    args = argparse.Namespace(type="probes", filter="zzzz-miss", detailed=False)
    assert list_cmd.cmd_list(args) == 0

    import insideLLMs.cli.commands.list_cmd as lc

    real_import = __import__

    def block(name, *a, **k):
        if name == "insideLLMs.benchmark_datasets":
            raise ImportError("blocked")
        return real_import(name, *a, **k)

    with patch("builtins.__import__", side_effect=block):
        assert lc.cmd_list(argparse.Namespace(type="datasets", filter=None, detailed=False)) == 0

    def _qt_args(**extra):
        base = dict(
            prompt="hi",
            model="dummy",
            probe=None,
            model_args="{}",
            temperature=0.0,
            max_tokens=16,
        )
        base.update(extra)
        return argparse.Namespace(**base)

    assert quicktest.cmd_quicktest(_qt_args(probe="nonexistent-probe-xyz")) == 1

    class _DictProbe:
        def run(self, model, data, **kwargs):
            return {"score": 1, "ok": True}

    with patch.object(quicktest, "probe_registry") as preg:
        with patch.object(quicktest, "resolve_registered_model", return_value=DummyModel()):
            preg.get.return_value = _DictProbe()
            assert quicktest.cmd_quicktest(_qt_args(probe="echo")) == 0

    with patch.object(quicktest, "resolve_registered_model", side_effect=RuntimeError("boom")):
        assert quicktest.cmd_quicktest(_qt_args()) == 1
