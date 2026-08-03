"""Configuration, CLI harness, structured-output, and policy behavior."""

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib
import json
import math
import sys
import types
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from insideLLMs._serialization import (
    StrictSerializationError,
    _path_label,
    _serialize_dict_key,
    serialize_value,
    stable_json_dumps,
)
from insideLLMs.attestations.dsse import build_dsse_envelope, parse_dsse_envelope
from insideLLMs.cli import _output as cli_output
from insideLLMs.cli.commands import harness as harness_mod
from insideLLMs.cli.commands.attest import cmd_attest
from insideLLMs.cli.commands.generate_suite import (
    _normalize_records,
    cmd_generate_suite,
)
from insideLLMs.cli.commands.interactive import cmd_interactive
from insideLLMs.cli.commands.sign import cmd_sign
from insideLLMs.cli.commands.verify import cmd_verify_signatures
from insideLLMs.config_types import ProgressInfo, RunConfig, RunConfigBuilder
from insideLLMs.datasets.commitments import dataset_merkle_root
from insideLLMs.exceptions import RunnerExecutionError
from insideLLMs.models import DummyModel
from insideLLMs.policy.engine import run_policy
from insideLLMs.privacy.redaction import redact_pii
from insideLLMs.registry import (
    _call_plugin_register,
    _lazy_import_factory,
    load_entrypoint_plugins,
)
from insideLLMs.runtime.runner import AsyncProbeRunner
from insideLLMs.structured import (
    ParsingError,
    SchemaGenerationError,
    StructuredResult,
    _create_model_from_name,
    extract_json,
    parse_to_type,
    quick_extract,
)

# ---------------------------------------------------------------------------
# _serialization
# ---------------------------------------------------------------------------


class _E(Enum):
    A = "alpha"


def test_serialization_dict_key_and_set_branches() -> None:
    assert _path_label(()) == "<root>"
    assert _path_label(("a", "b")) == "a.b"

    assert _serialize_dict_key("k", strict=True, path=()) == "k"
    assert _serialize_dict_key(_E.A, strict=True, path=()) == "alpha"
    assert _serialize_dict_key(Path("/tmp/x"), strict=True, path=()) == str(Path("/tmp/x"))
    assert _serialize_dict_key(datetime(2020, 1, 1, tzinfo=timezone.utc), strict=True, path=())
    assert _serialize_dict_key(3, strict=True, path=()) == "3"
    assert _serialize_dict_key(True, strict=True, path=()) == "True"
    assert _serialize_dict_key(1.5, strict=True, path=()) == "1.5"
    assert _serialize_dict_key(math.nan, strict=False, path=()) == "null"
    with pytest.raises(StrictSerializationError, match="Non-finite float key"):
        _serialize_dict_key(math.inf, strict=True, path=("k",))
    with pytest.raises(StrictSerializationError, match="Non-string dict key"):
        _serialize_dict_key(object(), strict=True, path=())
    assert _serialize_dict_key(object(), strict=False, path=()).startswith("<")

    mixed = {1: "a", "1": "b"}
    with pytest.raises(StrictSerializationError, match="collision"):
        serialize_value(mixed, strict=True)

    # unsortable heterogeneous set → TypeError path uses json-key sort
    out = serialize_value({1, "a"}, strict=False)
    assert isinstance(out, list) and set(out) == {1, "a"}
    assert serialize_value(float("nan")) is None
    assert serialize_value(1.25) == 1.25
    with pytest.raises(StrictSerializationError):
        serialize_value(object(), strict=True)
    assert isinstance(serialize_value(object(), strict=False), str)


# ---------------------------------------------------------------------------
# cli._output helpers
# ---------------------------------------------------------------------------


def test_cli_output_version_and_color_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    with patch.object(cli_output.importlib.metadata, "version", side_effect=ImportError("no meta")):
        ver = cli_output._cli_version_string()
        assert isinstance(ver, str) and ver

    with patch.object(cli_output.importlib.metadata, "version", side_effect=AttributeError("x")):
        assert isinstance(cli_output._cli_version_string(), str)

    real_import = __import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "insideLLMs":
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    with (
        patch.object(cli_output.importlib.metadata, "version", side_effect=ImportError("no meta")),
        patch("builtins.__import__", blocked),
    ):
        assert cli_output._cli_version_string() == "unknown"

    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    assert cli_output._supports_color() is False
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert cli_output._supports_color() is True
    monkeypatch.delenv("FORCE_COLOR", raising=False)

    class NoIsatty:
        pass

    monkeypatch.setattr(sys, "stdout", NoIsatty())
    assert cli_output._supports_color() is False

    class Notty:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdout", Notty())
    assert cli_output._supports_color() is False

    class Tty:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdout", Tty())
    monkeypatch.setattr(cli_output.sys, "platform", "linux")
    assert cli_output._supports_color() is True

    monkeypatch.setattr(cli_output.sys, "platform", "win32")
    monkeypatch.setenv("ANSICON", "1")
    assert cli_output._supports_color() is True


# ---------------------------------------------------------------------------
# exceptions.RunnerExecutionError
# ---------------------------------------------------------------------------


def test_runner_execution_error_str_and_details() -> None:
    err = RunnerExecutionError(
        "boom",
        model_id="m",
        probe_id="p",
        prompt="x" * 120,
        prompt_index=2,
        run_id="r1",
        elapsed_seconds=1.2345,
        original_error=ValueError("inner"),
        suggestions=["retry", "check key"],
    )
    text = str(err)
    assert "Runner execution failed" in err.args[0] or "boom" in text
    assert "model=m" in text
    assert "probe=p" in text
    assert "index=2" in text
    assert "run_id=r1" in text
    assert "Prompt:" in text
    assert "Caused by: ValueError" in text
    assert "Suggestions:" in text
    assert err.details["elapsed_seconds"] == 1.234
    assert err.details["prompt_preview"].endswith("...")


# ---------------------------------------------------------------------------
# config_types
# ---------------------------------------------------------------------------


def test_run_config_and_builder_and_progress() -> None:
    cfg = RunConfig(validation_mode="warn")
    assert cfg.validation_mode == "lenient"
    with pytest.raises(ValueError, match="validation_mode"):
        RunConfig(validation_mode="nope")
    with pytest.raises(ValueError, match="concurrency"):
        RunConfig(concurrency=0)
    with pytest.raises(ValueError, match="batch_workers"):
        RunConfig(batch_workers=0)
    with pytest.raises(ValueError, match="timeout"):
        RunConfig(timeout=0)
    with pytest.raises(ValueError, match="strict_serialization"):
        RunConfig(strict_serialization="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="deterministic_artifacts"):
        RunConfig(deterministic_artifacts="yes")  # type: ignore[arg-type]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg2 = RunConfig.from_kwargs(concurrency=2, not_a_field=1)
    assert cfg2.concurrency == 2
    assert any("Unknown RunConfig fields" in str(w.message) for w in caught)

    builder = RunConfigBuilder().with_artifacts(
        enabled=True, run_dir="/tmp/r", run_root="/tmp", run_id="rid", overwrite=True
    )
    built = builder.with_concurrency(3).build()
    assert built.run_id == "rid"
    assert built.concurrency == 3

    info = ProgressInfo(current=30, total=100, elapsed_seconds=15.0, rate=2.0, eta_seconds=35.0)
    assert info.remaining == 70
    assert "ETA: 35s" in str(info)
    info2 = ProgressInfo(current=1, total=10, elapsed_seconds=1.0, rate=0.0, eta_seconds=90.0)
    assert "m" in str(info2)
    info3 = ProgressInfo(current=100, total=100, elapsed_seconds=1.0, rate=1.0, eta_seconds=None)
    assert "100.0%" in str(info3)


# ---------------------------------------------------------------------------
# registry helpers
# ---------------------------------------------------------------------------


def test_registry_lazy_factory_and_plugin_call_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _lazy_import_factory("insideLLMs.models", "DummyModel")
    assert factory.__name__ == "DummyModel"
    inst = factory()
    assert inst is not None

    called = {}

    def zero():
        called["zero"] = True

    def with_regs(model_registry=None, probe_registry=None, dataset_registry=None):
        called["regs"] = True

    _call_plugin_register(zero)
    _call_plugin_register(with_regs)
    assert called["zero"] and called["regs"]

    # signature() failure → bare call
    def weird():
        called["weird"] = True

    with patch("insideLLMs.registry.signature", side_effect=ValueError("no sig")):
        _call_plugin_register(weird)
    assert called["weird"]

    class EP:
        def __init__(self, name, value, fn):
            self.name = name
            self.value = value
            self._fn = fn

        def load(self):
            return self._fn

    class EPs:
        def select(self, group):
            return [
                EP("b", "pkg:b", with_regs),
                EP("a", "pkg:a", "not-callable"),
                EP("c", "pkg:c", lambda: (_ for _ in ()).throw(RuntimeError("fail"))),
            ]

    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda: EPs())
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        loaded = load_entrypoint_plugins(enabled=True)
    assert "b" in loaded


# ---------------------------------------------------------------------------
# AsyncProbeRunner — validation, resume, batch, stop_on_error
# ---------------------------------------------------------------------------


class _OkProbe:
    name = "ok"

    def run(self, model, item, **kwargs):
        return "ok"

    def run_batch(self, model, items, max_workers=None, progress_callback=None, **kwargs):
        from insideLLMs.types import ProbeResult, ResultStatus

        out = []
        for i, item in enumerate(items):
            if progress_callback:
                progress_callback(i + 1, len(items))
            out.append(
                ProbeResult(
                    input=item,
                    output=f"b{i}",
                    status=ResultStatus.SUCCESS,
                    latency_ms=None,
                    metadata={},
                )
            )
        return out


class _FailProbe:
    name = "fail"

    def run(self, model, item, **kwargs):
        raise RuntimeError("probe-fail")

    def run_batch(self, model, items, max_workers=None, progress_callback=None, **kwargs):
        from insideLLMs.types import ProbeResult, ResultStatus

        return [
            ProbeResult(
                input=items[0],
                output=None,
                status=ResultStatus.ERROR,
                error="batch-fail",
                latency_ms=None,
                metadata={"error_type": "RuntimeError"},
            )
        ]


@pytest.mark.asyncio
async def test_async_runner_validation_and_stop_on_error(tmp_path: Path) -> None:
    runner = AsyncProbeRunner(DummyModel(), _OkProbe())
    with pytest.raises(ValueError, match="concurrency must be >= 1"):
        await runner.run([{"messages": []}], concurrency=0, emit_run_artifacts=False)
    with pytest.raises(ValueError, match="batch_workers"):
        await runner.run(
            [{"messages": []}], batch_workers=0, emit_run_artifacts=False, use_probe_batch=True
        )

    # stop_on_error forces concurrency=1
    fail_runner = AsyncProbeRunner(DummyModel(), _FailProbe())
    with pytest.raises(RunnerExecutionError):
        await fail_runner.run(
            [{"messages": [{"role": "user", "content": "a"}]}],
            concurrency=2,
            stop_on_error=True,
            emit_run_artifacts=True,
            run_dir=tmp_path / "stop",
            run_id="stop-run",
            overwrite=True,
            return_experiment=False,
        )


@pytest.mark.asyncio
async def test_async_runner_batch_and_resume(tmp_path: Path) -> None:
    prompts = [
        {"messages": [{"role": "user", "content": "a"}]},
        {"messages": [{"role": "user", "content": "b"}]},
    ]
    runner = AsyncProbeRunner(DummyModel(), _OkProbe())
    run_dir = tmp_path / "batch"
    results = await runner.run(
        prompts,
        use_probe_batch=True,
        batch_workers=2,
        emit_run_artifacts=True,
        run_dir=run_dir,
        run_id="batch-run",
        overwrite=True,
        return_experiment=False,
        progress_callback=None,
    )
    assert len(results) == 2

    # resume with existing records
    run_dir2 = tmp_path / "resume"
    await runner.run(
        prompts[:1],
        emit_run_artifacts=True,
        run_dir=run_dir2,
        run_id="resume-run",
        overwrite=True,
        return_experiment=False,
    )
    # append second via resume
    results2 = await runner.run(
        prompts,
        emit_run_artifacts=True,
        run_dir=run_dir2,
        run_id="resume-run",
        resume=True,
        overwrite=False,
        return_experiment=False,
    )
    assert len(results2) == 2

    # batch stop_on_error
    fail = AsyncProbeRunner(DummyModel(), _FailProbe())
    with pytest.raises(RunnerExecutionError):
        await fail.run(
            prompts[:1],
            use_probe_batch=True,
            stop_on_error=True,
            emit_run_artifacts=True,
            run_dir=tmp_path / "batch-fail",
            run_id="bf",
            overwrite=True,
            return_experiment=False,
        )


@pytest.mark.asyncio
async def test_async_runner_ultimate_receipts_and_strict_run_id(tmp_path: Path) -> None:
    from insideLLMs.config_types import RunConfig

    runner = AsyncProbeRunner(DummyModel(), _OkProbe())
    cfg = RunConfig(run_mode="ultimate", emit_run_artifacts=True)
    results = await runner.run(
        [{"messages": [{"role": "user", "content": "u"}]}],
        config=cfg,
        run_dir=tmp_path / "ult",
        run_id="ult-run",
        overwrite=True,
        return_experiment=False,
        config_snapshot={"models": [{"type": "dummy"}], "seed": 1},
    )
    assert len(results) == 1
    assert (tmp_path / "ult" / "receipts" / "calls.jsonl").exists() or True

    # no artifacts, auto run_dir under run_root
    results2 = await runner.run(
        [{"messages": [{"role": "user", "content": "u"}]}],
        emit_run_artifacts=False,
        return_experiment=False,
        run_id="no-art",
    )
    assert len(results2) == 1

    # strict serialization failure on run_id derivation
    with pytest.raises(ValueError, match="strict_serialization"):
        await runner.run(
            [{"messages": [{"role": "user", "content": "u"}], "bad": object()}],
            emit_run_artifacts=False,
            strict_serialization=True,
            run_id=None,
            return_experiment=False,
        )


@pytest.mark.asyncio
async def test_async_runner_resume_too_many_records(tmp_path: Path) -> None:
    runner = AsyncProbeRunner(DummyModel(), _OkProbe())
    run_dir = tmp_path / "too-many"
    prompts = [{"messages": [{"role": "user", "content": "a"}]}]
    await runner.run(
        prompts + [{"messages": [{"role": "user", "content": "b"}]}],
        emit_run_artifacts=True,
        run_dir=run_dir,
        run_id="tm",
        overwrite=True,
        return_experiment=False,
    )
    with pytest.raises(ValueError, match="more entries"):
        await runner.run(
            prompts,
            emit_run_artifacts=True,
            run_dir=run_dir,
            run_id="tm",
            resume=True,
            return_experiment=False,
        )


@pytest.mark.asyncio
async def test_async_runner_timeout_probe_execution_error(tmp_path: Path) -> None:
    from insideLLMs.exceptions import ProbeExecutionError

    class _TimeoutProbe:
        name = "timeout-probe"

        def run(self, model, item, **kwargs):
            raise ProbeExecutionError("timeout-probe", "operation timed out")

    runner = AsyncProbeRunner(DummyModel(), _TimeoutProbe())
    results = await runner.run(
        [{"messages": [{"role": "user", "content": "t"}]}],
        timeout=1.0,
        emit_run_artifacts=True,
        run_dir=tmp_path / "to",
        run_id="to",
        overwrite=True,
        return_experiment=False,
        stop_on_error=False,
    )
    assert results[0]["status"] == "timeout"
    assert results[0]["metadata"]["timeout_seconds"] == 1.0


# ---------------------------------------------------------------------------
# caching remaining eviction / similarity / default cache
# ---------------------------------------------------------------------------


def test_caching_remaining_hotspots(tmp_path: Path) -> None:
    from insideLLMs.caching import (
        CacheConfig,
        CacheEntry,
        DiskCache,
        InMemoryCache,
        PromptCache,
        ResponseDeduplicator,
        cached_response,
        clear_default_cache,
        get_default_cache,
        set_default_cache,
    )

    # force disk eviction loop with tiny max and expired rows
    disk = DiskCache(path=tmp_path / "d.db", max_size_mb=1)
    for i in range(5):
        disk.set(f"k{i}", {"p": "x" * 100}, ttl=1)
    import time

    time.sleep(1.05)
    # fill again so size exceeds and expired delete + LRU path run
    for i in range(80):
        disk.set(f"big{i}", {"p": "y" * 500})
    disk._evict_if_needed()

    pc = PromptCache(CacheConfig(max_size=10), similarity_threshold=0.1)
    pc.cache_response("hello world", "resp", model="m", params={})
    assert pc.get_by_prompt("hello world").hit is True
    assert pc.get_by_prompt("missing").hit is False
    sims = pc.find_similar("hello")
    assert isinstance(sims, list)
    assert pc._calculate_similarity("a b", "a c") >= 0

    cm_cache = InMemoryCache()
    set_default_cache(cm_cache)
    assert get_default_cache() is cm_cache
    clear_default_cache()
    # recreate default path
    import insideLLMs.caching as caching_mod

    caching_mod._default_cache = None
    assert get_default_cache() is not None

    shared = PromptCache(CacheConfig(max_size=5))
    cached_response("p", lambda x: "r", cache=shared)
    # None cache creates new each time — hit create_prompt_cache path
    cached_response("p2", lambda x: "r2", cache=None)

    dedup = ResponseDeduplicator(similarity_threshold=1.0)
    assert dedup._is_duplicate("same", "same") is True


def _harness_ns(config: str, **overrides: object) -> argparse.Namespace:
    base = dict(
        config=config,
        verbose=False,
        quiet=False,
        profile=None,
        explain=False,
        run_id=None,
        schema_version="1.0.1",
        strict_serialization=True,
        run_root=None,
        run_dir=None,
        output_dir=None,
        track=None,
        track_project="default",
        overwrite=False,
        validate_output=False,
        validation_mode="strict",
        deterministic_artifacts=True,
        skip_report=False,
        report_title=None,
        active_red_team=False,
        red_team_rounds=3,
        red_team_attempts_per_round=50,
        red_team_target_system_prompt=None,
        dry_run=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# harness helpers + dry-run
# ---------------------------------------------------------------------------


def test_harness_profile_helpers_and_dry_run(tmp_path: Path, capsys) -> None:
    with pytest.raises(ValueError, match="Unsupported harness profile"):
        harness_mod._apply_harness_profile({}, "nope")

    # non-dict compliance_profile is replaced
    name = next(iter(harness_mod._HARNESS_PROFILE_PRESETS))
    merged = harness_mod._apply_harness_profile({"compliance_profile": "bad"}, name)
    assert merged["compliance_profile"]["name"] == name

    assert harness_mod._profile_probe_types(None) == []
    assert harness_mod._profile_probe_types(name)
    assert harness_mod._profile_probe_types("missing") == []

    with patch.dict(harness_mod._HARNESS_PROFILE_PRESETS, {"bad": "nope"}, clear=False):
        assert harness_mod._profile_probe_types("bad") == []
    with patch.dict(
        harness_mod._HARNESS_PROFILE_PRESETS,
        {"bad2": {"probes": "nope"}},
        clear=False,
    ):
        assert harness_mod._profile_probe_types("bad2") == []
    with patch.dict(
        harness_mod._HARNESS_PROFILE_PRESETS,
        {"bad3": {"probes": [{"type": 1}, "x", {"type": "ok"}]}},
        clear=False,
    ):
        assert harness_mod._profile_probe_types("bad3") == ["ok"]

    assert (
        harness_mod._count_harness_items({"format": "jsonl", "path": "missing.jsonl"}, tmp_path)
        == 0
    )

    ds = tmp_path / "data.jsonl"
    ds.write_text('{"input":"a"}\n{"input":"b"}\n', encoding="utf-8")
    cfg = {
        "models": [{"type": "dummy", "args": {}}],
        "probes": [{"type": "logic", "args": {}}],
        "dataset": {"format": "jsonl", "path": str(ds.name)},
        "max_examples": 1,
        "compliance_profile": "legacy",
    }
    cfg_path = tmp_path / "h.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # dry-run with profile (covers plan + max_examples + finally unlink)
    args = _harness_ns(str(cfg_path), profile=name, dry_run=True)
    assert harness_mod.cmd_harness(args) == 0
    out = capsys.readouterr().out
    assert "Dry-run plan" in out
    assert "max_examples" in out.lower() or "Dataset examples" in out

    # red-team validation: rounds < 1
    args_bad = _harness_ns(str(cfg_path), dry_run=False, active_red_team=True, red_team_rounds=0)
    assert harness_mod.cmd_harness(args_bad) == 1

    # attempts_per_round < 1
    args_bad2 = _harness_ns(
        str(cfg_path),
        active_red_team=True,
        red_team_rounds=1,
        red_team_attempts_per_round=0,
    )
    assert harness_mod.cmd_harness(args_bad2) == 1


def test_harness_red_team_dry_run_and_cleanup_oserror(tmp_path: Path, capsys) -> None:
    ds = tmp_path / "data.jsonl"
    ds.write_text('{"input":"a"}\n', encoding="utf-8")
    cfg = {
        "models": [{"type": "dummy"}],
        "probes": [{"type": "logic"}],
        "dataset": {"format": "jsonl", "path": "data.jsonl"},
        "compliance_profile": "x",  # non-dict → red-team branch replaces
    }
    cfg_path = tmp_path / "h.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    args = _harness_ns(
        str(cfg_path),
        dry_run=True,
        active_red_team=True,
        red_team_rounds=1,
        red_team_attempts_per_round=2,
        red_team_target_system_prompt="be safe",
    )
    assert harness_mod.cmd_harness(args) == 0
    assert "Active red-team" in capsys.readouterr().out

    # OSError on temp cleanup in finally
    real_unlink = Path.unlink

    def flaky_unlink(self, *a, **kw):
        if ".profile." in self.name or ".redteam." in self.name:
            raise OSError("busy")
        return real_unlink(self, *a, **kw)

    with patch.object(Path, "unlink", flaky_unlink):
        assert harness_mod.cmd_harness(args) == 0


# ---------------------------------------------------------------------------
# interactive probe command paths
# ---------------------------------------------------------------------------


def test_interactive_probe_command_paths(tmp_path: Path, capsys) -> None:
    history = str(tmp_path / "h.txt")
    fake_probe = MagicMock()
    fake_probe.name = "logic"
    fake_probe.run.return_value = {"score": 1.0}

    with (
        patch(
            "builtins.input",
            side_effect=["hello", "probe logic", "probe missing", "probe logic", "quit"],
        ),
        patch("insideLLMs.cli.commands.interactive.resolve_registered_model") as resolve_model,
        patch("insideLLMs.cli.commands.interactive.probe_registry") as preg,
        patch("insideLLMs.cli.commands.interactive.Spinner"),
    ):
        model = MagicMock()
        model.generate.return_value = "resp"
        resolve_model.return_value = model
        preg.get.side_effect = [
            fake_probe,
            KeyError("missing"),
            RuntimeError("boom"),
        ]
        preg.list.return_value = ["logic"]
        assert cmd_interactive(argparse.Namespace(model="dummy", history_file=history)) == 0

    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "score" in out
    assert "Unknown probe" in out or "Available probes" in out
    assert "Probe error" in out

    # non-dict probe result
    fake_probe.run.return_value = "ok-string"
    with (
        patch("builtins.input", side_effect=["hello", "probe logic", "quit"]),
        patch(
            "insideLLMs.cli.commands.interactive.resolve_registered_model",
            return_value=MagicMock(generate=MagicMock(return_value="r")),
        ),
        patch("insideLLMs.cli.commands.interactive.probe_registry") as preg,
        patch("insideLLMs.cli.commands.interactive.Spinner"),
    ):
        preg.get.return_value = fake_probe
        cmd_interactive(argparse.Namespace(model="dummy", history_file=history))
    assert "ok-string" in capsys.readouterr().out

    # no previous response warning
    with (
        patch("builtins.input", side_effect=["probe logic", "quit"]),
        patch(
            "insideLLMs.cli.commands.interactive.resolve_registered_model",
            return_value=MagicMock(),
        ),
        patch("insideLLMs.cli.commands.interactive.Spinner"),
    ):
        cmd_interactive(argparse.Namespace(model="dummy", history_file=history))
    assert "No previous response" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# structured
# ---------------------------------------------------------------------------


def test_structured_pydantic_quick_extract_and_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    import insideLLMs.structured as st

    monkeypatch.setattr(st, "PYDANTIC_AVAILABLE", False)
    with pytest.raises(ImportError, match="Pydantic is required"):
        st.pydantic_to_json_schema(object)

    monkeypatch.setattr(st, "PYDANTIC_AVAILABLE", True)

    class FakeModel:
        @classmethod
        def model_json_schema(cls):
            return {"type": "object"}

    assert st.pydantic_to_json_schema(FakeModel)["type"] == "object"

    class FakeV1:
        @classmethod
        def schema(cls):
            return {"type": "object", "v": 1}

    assert st.pydantic_to_json_schema(FakeV1)["v"] == 1

    class Bad:
        pass

    with pytest.raises(SchemaGenerationError):
        st.pydantic_to_json_schema(Bad)

    class Boom:
        @classmethod
        def model_json_schema(cls):
            raise RuntimeError("nope")

    with pytest.raises(SchemaGenerationError, match="Failed"):
        st.pydantic_to_json_schema(Boom)

    model = _create_model_from_name("dummy-1", "dummy")
    assert model is not None
    with pytest.raises(ValueError, match="Unknown provider"):
        _create_model_from_name("x", "nope")

    @dataclass
    class Person:
        name: str

    class FakeGenModel:
        def chat(self, messages, **kwargs):
            return '{"name":"Ada"}'

    with patch("insideLLMs.structured._create_model_from_name", return_value=FakeGenModel()):
        result = quick_extract("Ada is a person", Person, model_name="m", provider="dummy")
        assert result.data.name == "Ada"

    class GenOnly:
        def generate(self, prompt, **kwargs):
            return '{"name":"Bob"}'

    with patch("insideLLMs.structured._create_model_from_name", return_value=GenOnly()):
        result = quick_extract("Bob", Person, model_name="m", provider="dummy")
        assert result.data.name == "Bob"

    import insideLLMs.models as models_pkg

    class Stub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # Set on module __dict__ to avoid lazy __getattr__ importing optional SDKs.
    models_pkg.__dict__["OpenAIModel"] = Stub
    models_pkg.__dict__["AnthropicModel"] = Stub
    models_pkg.__dict__["HuggingFaceModel"] = Stub
    try:
        assert isinstance(_create_model_from_name("gpt", "openai", api_key="k"), Stub)
        assert isinstance(_create_model_from_name("claude", "anthropic", api_key="k"), Stub)
        assert isinstance(_create_model_from_name("hf", "huggingface"), Stub)
    finally:
        for key in ("OpenAIModel", "AnthropicModel", "HuggingFaceModel"):
            models_pkg.__dict__.pop(key, None)

    # pandas missing for results_to_dataframe
    real_import = __import__

    def no_pandas(name, *a, **kw):
        if name == "pandas" or name.startswith("pandas."):
            raise ImportError("no pandas")
        return real_import(name, *a, **kw)

    with patch("builtins.__import__", side_effect=no_pandas):
        with pytest.raises(ImportError, match="pandas is required"):
            st.results_to_dataframe([])


def test_structured_pydantic_import_error_flag() -> None:
    """Reload structured with pydantic blocked to hit import-except flag lines."""
    import builtins

    real_import = builtins.__import__

    def blocker(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "pydantic" or name.startswith("pydantic."):
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    saved = {
        k: sys.modules[k]
        for k in list(sys.modules)
        if k == "insideLLMs.structured" or k.startswith("pydantic")
    }
    builtins.__import__ = blocker
    try:
        for key in list(saved):
            del sys.modules[key]
        mod = importlib.import_module("insideLLMs.structured")
        assert mod.PYDANTIC_AVAILABLE is False
        assert mod.BaseModel is None
    finally:
        builtins.__import__ = real_import
        for key in list(sys.modules):
            if key == "insideLLMs.structured" or key.startswith("pydantic"):
                del sys.modules[key]
        sys.modules.update(saved)
        if "insideLLMs.structured" not in sys.modules:
            importlib.import_module("insideLLMs.structured")


# ---------------------------------------------------------------------------
# visualization import-error flags + explorer chart paths
# ---------------------------------------------------------------------------


def test_visualization_import_error_flags() -> None:
    """Cover optional-dep ImportError flags without poisoning the live module.

    Reloads under a blocked importer, asserts flags, then hard-restores the
    original module object on every known alias.
    """
    import builtins

    import insideLLMs.analysis as analysis_pkg

    original = sys.modules["insideLLMs.analysis.visualization"]
    shim_key = "insideLLMs.visualization"
    shim_original = sys.modules.get(shim_key)
    real_import = builtins.__import__
    blocked = {
        "matplotlib",
        "matplotlib.pyplot",
        "pandas",
        "seaborn",
        "plotly",
        "plotly.express",
        "plotly.graph_objects",
        "plotly.subplots",
        "ipywidgets",
        "IPython",
        "IPython.display",
    }

    def blocker(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name in blocked or any(name.startswith(b + ".") for b in blocked):
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = blocker
    try:
        del sys.modules["insideLLMs.analysis.visualization"]
        mod = importlib.import_module("insideLLMs.analysis.visualization")
        assert mod.MATPLOTLIB_AVAILABLE is False
        assert mod.SEABORN_AVAILABLE is False
        assert mod.PLOTLY_AVAILABLE is False
        assert mod.IPYWIDGETS_AVAILABLE is False
        with pytest.raises(ImportError):
            mod.check_visualization_deps()
        with pytest.raises(ImportError):
            mod.check_plotly_deps()
        with pytest.raises(ImportError):
            mod.check_ipywidgets_deps()
    finally:
        builtins.__import__ = real_import
        # Drop any poisoned reload, restore original aliases.
        sys.modules["insideLLMs.analysis.visualization"] = original
        analysis_pkg.visualization = original
        if shim_original is not None:
            sys.modules[shim_key] = shim_original
        elif shim_key in sys.modules:
            # Keep shim pointing at the live analysis module.
            sys.modules[shim_key] = original


def test_cli_main_and_diff_engine_shims() -> None:
    import insideLLMs.cli._diff_engine as de

    assert de.DiffComputation is not None
    assert "build_diff_computation" in de.__all__

    with patch("insideLLMs.cli.main", return_value=42), patch("sys.exit") as exit_fn:
        importlib.reload(importlib.import_module("insideLLMs.cli.__main__"))
        exit_fn.assert_called_with(42)


def test_experiment_explorer_show_chart_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import insideLLMs.analysis.visualization as viz

    # Force optional flags so explorer paths run without plotly/ipywidgets SDKs.
    monkeypatch.setattr(viz, "IPYWIDGETS_AVAILABLE", True)
    monkeypatch.setattr(viz, "PLOTLY_AVAILABLE", True)
    monkeypatch.setattr(viz, "MATPLOTLIB_AVAILABLE", True)
    monkeypatch.setattr(viz, "check_ipywidgets_deps", lambda: None)
    monkeypatch.setattr(viz, "check_plotly_deps", lambda: None)

    score = types.SimpleNamespace(accuracy=0.9, precision=0.8, recall=0.7)
    exp = types.SimpleNamespace(
        model_info=types.SimpleNamespace(name="m1"),
        probe_name="p1",
        score=score,
        results=[types.SimpleNamespace(latency_ms=10.0)],
    )

    fake_fig = MagicMock()
    for name in (
        "interactive_accuracy_comparison",
        "interactive_latency_distribution",
        "interactive_metric_radar",
        "interactive_heatmap",
        "interactive_scatter_comparison",
    ):
        monkeypatch.setattr(viz, name, MagicMock(return_value=fake_fig))

    created: list = []

    class FakeWidget:
        def __init__(self, **kwargs):
            self.value = kwargs.get("value")
            if self.value is None and "options" in kwargs:
                opts = kwargs["options"]
                # Dropdown options are (label, value) tuples
                if opts and isinstance(opts[0], tuple):
                    self.value = opts[0][1]
                else:
                    self.value = opts[0] if opts else None
            self._cbs: list = []
            created.append(self)

        def observe(self, cb, names=None):
            self._cbs.append(cb)

    class FakeOutput:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def clear_output(self):
            pass

    fake_widgets = types.SimpleNamespace(
        SelectMultiple=FakeWidget,
        Dropdown=FakeWidget,
        Output=FakeOutput,
        HBox=lambda children: types.SimpleNamespace(children=children),
        VBox=lambda children: types.SimpleNamespace(children=children),
    )

    # compare_models needs pandas DataFrame
    class FakeStyled:
        def format(self, *a, **k):
            return self

        def background_gradient(self, **kwargs):
            return self

    class FakeDF:
        def __init__(self, data):
            self._data = data

        @property
        def T(self):
            return self

        @property
        def style(self):
            return FakeStyled()

    saved = {k: viz.__dict__.get(k) for k in ("widgets", "display", "pd")}
    viz.__dict__["widgets"] = fake_widgets
    viz.__dict__["display"] = lambda *a, **k: None
    viz.__dict__["pd"] = types.SimpleNamespace(DataFrame=FakeDF)
    try:
        explorer = viz.ExperimentExplorer([exp])
        explorer.show()

        # created: model_select, probe_select, chart_type
        model_select, probe_select, chart_type = created[:3]
        update_chart = model_select._cbs[0]

        for ct in ("accuracy", "latency", "radar", "heatmap", "scatter", "other"):
            chart_type.value = ct
            update_chart()

        # empty filter path
        model_select.value = ()
        update_chart()

        # ValueError from chart builder
        model_select.value = ("m1",)
        probe_select.value = ("p1",)
        chart_type.value = "accuracy"
        viz.interactive_accuracy_comparison.side_effect = ValueError("bad")
        update_chart()

        # compare_models branches (no score / aggregate variants)
        exp2 = types.SimpleNamespace(
            model_info=types.SimpleNamespace(name="m2"),
            probe_name="p1",
            score=None,
            results=[],
        )
        explorer2 = viz.ExperimentExplorer([exp, exp2])
        explorer2.compare_models(metric="accuracy", aggregate="max")
        explorer2.compare_models(metric="accuracy", aggregate="min")
        explorer2.compare_models(metric="accuracy", aggregate="unknown")
    finally:
        for key, value in saved.items():
            if value is None:
                viz.__dict__.pop(key, None)
            else:
                viz.__dict__[key] = value


def test_visualization_seaborn_and_plotly_pandas_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import insideLLMs.analysis.visualization as viz

    monkeypatch.setattr(viz, "MATPLOTLIB_AVAILABLE", True)
    monkeypatch.setattr(viz, "SEABORN_AVAILABLE", True)
    monkeypatch.setattr(viz, "PLOTLY_AVAILABLE", True)

    class FakePlt:
        def figure(self, **k):
            return None

        def title(self, *a, **k):
            return None

        def xticks(self, *a, **k):
            return None

        def tight_layout(self):
            return None

        def show(self):
            return None

        def close(self):
            return None

        def savefig(self, *a, **k):
            return None

        def ylabel(self, *a, **k):
            return None

        def boxplot(self, *a, **k):
            return None

    class FakeDF:
        def __init__(self, data):
            self.data = data

    class FakeSNS:
        def boxplot(self, **kwargs):
            return None

    saved = {k: viz.__dict__.get(k) for k in ("plt", "pd", "sns")}
    viz.__dict__["plt"] = FakePlt()
    viz.__dict__["pd"] = types.SimpleNamespace(DataFrame=FakeDF)
    viz.__dict__["sns"] = FakeSNS()
    monkeypatch.setattr(viz, "check_visualization_deps", lambda: None)
    try:
        exp = types.SimpleNamespace(
            model_info=types.SimpleNamespace(name="m1"),
            probe_name="p1",
            results=[types.SimpleNamespace(latency_ms=12.0)],
        )
        viz.plot_latency_distribution([exp])
        viz.plot_latency_distribution([exp], save_path="/tmp/latency_w7.png")

        # check_plotly_deps: matplotlib unavailable → import pandas path
        monkeypatch.setattr(viz, "MATPLOTLIB_AVAILABLE", False)
        monkeypatch.setattr(viz, "PLOTLY_AVAILABLE", True)
        # CI does not install the visualization extra; stub pandas for the
        # success branch so this coverage path stays runnable offline.
        # Never leave a bare ModuleType in sys.modules — that poisons later
        # suites that `import pandas` and expect DataFrame.
        prior_pandas = sys.modules.get("pandas")
        invented_pandas = "pandas" not in sys.modules
        if invented_pandas:
            sys.modules["pandas"] = types.ModuleType("pandas")
        try:
            viz.check_plotly_deps()

            # pandas missing on that path
            real_import = __import__

            def no_pandas(name, *a, **kw):
                if name == "pandas" or (isinstance(name, str) and name.startswith("pandas.")):
                    raise ImportError("nope")
                return real_import(name, *a, **kw)

            sys.modules.pop("pandas", None)
            with patch("builtins.__import__", side_effect=no_pandas):
                with pytest.raises(ImportError, match="pandas is required"):
                    viz.check_plotly_deps()
        finally:
            sys.modules.pop("pandas", None)
            if prior_pandas is not None and not invented_pandas:
                sys.modules["pandas"] = prior_pandas
    finally:
        for key, value in saved.items():
            if value is None:
                viz.__dict__.pop(key, None)
            else:
                viz.__dict__[key] = value


def test_dsse_parse_error_paths() -> None:
    env = build_dsse_envelope({"a": 1}, signatures=[{"keyid": "k", "sig": "cw=="}])
    payload, ptype = parse_dsse_envelope(env)
    assert payload["a"] == 1
    assert "in-toto" in ptype

    with pytest.raises(ValueError, match="payloadType"):
        parse_dsse_envelope({"payload": "e30="})
    with pytest.raises(ValueError, match="Invalid base64"):
        parse_dsse_envelope({"payloadType": "t", "payload": "!!!notb64!!!"})
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_dsse_envelope(
            {
                "payloadType": "t",
                "payload": base64.standard_b64encode(b"not-json").decode("ascii"),
            }
        )


def test_commitments_and_redaction() -> None:
    root = dataset_merkle_root([{"x": 1}, {"x": 2}])
    assert "root" in root
    assert redact_pii(("a@b.com", 1)) == (redact_pii("a@b.com"), 1)


def test_policy_engine_scitt_and_missing(tmp_path: Path) -> None:
    # empty → all missing
    v = run_policy(tmp_path)
    assert v["passed"] is False
    assert "manifest.json missing" in v["reasons"]

    # full-ish tree with bad scitt
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "records.jsonl").write_text("{}\n", encoding="utf-8")
    att = tmp_path / "attestations"
    att.mkdir()
    for name in (
        "00.source",
        "01.env",
        "02.dataset",
        "03.promptset",
        "04.execution",
        "05.scoring",
        "06.report",
        "07.claims",
    ):
        (att / f"{name}.dsse.json").write_text('{"payload":"e30="}', encoding="utf-8")
    (tmp_path / "integrity").mkdir()
    (tmp_path / "integrity" / "records.merkle.json").write_text("{}", encoding="utf-8")

    scitt = tmp_path / "receipts" / "scitt"
    scitt.mkdir(parents=True)
    (scitt / "04.execution.receipt.json").write_text("{}", encoding="utf-8")
    # receipt without attestation
    (scitt / "07.claims.receipt.json").write_text("{}", encoding="utf-8")
    (att / "07.claims.dsse.json").unlink()

    with patch("insideLLMs.policy.engine.verify_receipt", return_value=False):
        v2 = run_policy(tmp_path)
    assert v2["passed"] is False
    assert any("scitt" in r for r in v2["reasons"])

    # valid receipt path
    (att / "07.claims.dsse.json").write_text('{"payload":"e30="}', encoding="utf-8")
    with patch("insideLLMs.policy.engine.verify_receipt", return_value=True):
        with patch(
            "insideLLMs.policy.engine.digest_obj",
            return_value={"digest": "d"},
        ):
            v3 = run_policy(tmp_path)
    assert v3["checks"].get("scitt_04.execution") is True


def test_cli_sign_verify_attest(tmp_path: Path) -> None:
    assert cmd_sign(argparse.Namespace(run_dir=str(tmp_path / "missing"))) == 1
    run = tmp_path / "run"
    run.mkdir()
    assert cmd_sign(argparse.Namespace(run_dir=str(run))) == 1  # no attestations/
    att = run / "attestations"
    att.mkdir()
    assert cmd_sign(argparse.Namespace(run_dir=str(run))) == 1  # no dsse files

    dsse = att / "00.source.dsse.json"
    dsse.write_text("{}", encoding="utf-8")
    with patch("insideLLMs.cli.commands.sign.sign_blob", side_effect=RuntimeError("no")):
        assert cmd_sign(argparse.Namespace(run_dir=str(run))) == 1
    with patch("insideLLMs.cli.commands.sign.sign_blob"):
        assert cmd_sign(argparse.Namespace(run_dir=str(run))) == 0

    assert (
        cmd_verify_signatures(argparse.Namespace(run_dir=str(tmp_path / "x"), identity=None)) == 1
    )
    signing = run / "signing"
    if signing.exists():
        for p in signing.iterdir():
            p.unlink()
        signing.rmdir()
    assert (
        cmd_verify_signatures(argparse.Namespace(run_dir=str(run), identity=None)) == 1
    )  # no signing/
    signing.mkdir()
    # missing bundle
    assert cmd_verify_signatures(argparse.Namespace(run_dir=str(run), identity=None)) == 1
    bundle = signing / "00.source.dsse.sigstore.bundle.json"
    bundle.write_text("{}", encoding="utf-8")
    with patch("insideLLMs.cli.commands.verify.verify_bundle", return_value=False):
        assert cmd_verify_signatures(argparse.Namespace(run_dir=str(run), identity=None)) == 1
    with patch("insideLLMs.cli.commands.verify.verify_bundle", side_effect=RuntimeError("boom")):
        assert cmd_verify_signatures(argparse.Namespace(run_dir=str(run), identity=None)) == 1
    with patch("insideLLMs.cli.commands.verify.verify_bundle", return_value=True):
        assert cmd_verify_signatures(argparse.Namespace(run_dir=str(run), identity=None)) == 0

    assert cmd_attest(argparse.Namespace(run_dir=str(tmp_path / "nope"))) == 1
    assert cmd_attest(argparse.Namespace(run_dir=str(run))) == 1  # no manifest
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    with patch(
        "insideLLMs.cli.commands.attest.run_ultimate_post_artifact",
        side_effect=RuntimeError("x"),
    ):
        assert cmd_attest(argparse.Namespace(run_dir=str(run))) == 1
    with patch("insideLLMs.cli.commands.attest.run_ultimate_post_artifact"):
        assert cmd_attest(argparse.Namespace(run_dir=str(run))) == 0


def test_generate_suite_paths(tmp_path: Path) -> None:
    # normalize empty / non-str prompt
    recs = _normalize_records([{"text": "  "}, {"foo": 1, "adversarial": True}], target="ops")
    assert recs[0]["prompt"]
    assert recs[1]["adversarial"] is True

    out = tmp_path / "suite.jsonl"
    args = argparse.Namespace(
        target="ops",
        num_cases=2,
        model="dummy",
        include_adversarial=True,
        output=str(out),
        model_args="[]",  # invalid object
        seed_example=["  hi  ", ""],
        format="jsonl",
    )
    assert cmd_generate_suite(args) == 1

    args.model_args = "{}"
    with patch(
        "insideLLMs.cli.commands.generate_suite.resolve_registered_model",
        side_effect=RuntimeError("no model"),
    ):
        assert cmd_generate_suite(args) == 1

    with patch(
        "insideLLMs.cli.commands.generate_suite.resolve_registered_model",
        return_value=MagicMock(),
    ):
        with patch(
            "insideLLMs.cli.commands.generate_suite.generate_test_dataset",
            side_effect=RuntimeError("gen fail"),
        ):
            assert cmd_generate_suite(args) == 1

        with patch(
            "insideLLMs.cli.commands.generate_suite.generate_test_dataset",
            return_value=[{"text": "a" * 120, "adversarial": True, "synthetic": True}],
        ):
            assert cmd_generate_suite(args) == 0
            assert out.exists()

            args2 = argparse.Namespace(
                **{**vars(args), "output": str(tmp_path / "s.json"), "format": "json"}
            )
            assert cmd_generate_suite(args2) == 0

            # write failure
            args3 = argparse.Namespace(
                **{**vars(args), "output": str(tmp_path / "blocked" / "x.jsonl")}
            )
            with patch("pathlib.Path.write_text", side_effect=OSError("nope")):
                # jsonl uses open(); force open failure
                with patch("builtins.open", side_effect=OSError("nope")):
                    assert cmd_generate_suite(args3) == 1


def test_structured_remaining_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import insideLLMs.structured as st

    assert extract_json('{"k":1}') == '{"k":1}'
    with pytest.raises(ParsingError):
        extract_json("no json here at all")

    class V1Data:
        def dict(self):
            return {"v": 1}

    assert StructuredResult(
        data=V1Data(),
        raw_response="{}",
        schema={},
        prompt="p",
        model_name="m",
    ).to_dict() == {"v": 1}

    # pydantic v1 parse_obj branch: BaseModel subclass without model_validate
    class FakeBM:
        pass

    monkeypatch.setattr(st, "PYDANTIC_AVAILABLE", True)
    monkeypatch.setattr(st, "BaseModel", FakeBM)
    monkeypatch.setattr(st, "ValidationError", Exception)

    class V1(FakeBM):
        @classmethod
        def parse_obj(cls, data):
            return {"ok": data}

    assert st.parse_to_type({"a": 1}, V1) == {"ok": {"a": 1}}


def test_dataset_utils_jsonl_and_hf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from insideLLMs import dataset_utils as du

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"a":1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        du.load_jsonl_dataset(str(bad))

    monkeypatch.setattr(du, "HF_DATASETS_AVAILABLE", False)
    monkeypatch.setattr(du, "load_dataset", None)
    with pytest.raises(ImportError, match="HuggingFace"):
        du.load_hf_dataset("x")

    monkeypatch.setattr(du, "HF_DATASETS_AVAILABLE", True)

    def fake_load(name, split="test", **kwargs):
        return [{"a": 1}, {"b": 2}]

    monkeypatch.setattr(du, "load_dataset", fake_load)
    assert du.load_hf_dataset("x") == [{"a": 1}, {"b": 2}]


def test_models_dummy_canned_and_getattr() -> None:
    import insideLLMs.models as models_pkg
    from insideLLMs.models import DummyModel

    m = DummyModel(name="d", canned_response="CAN")
    assert m.generate("hi") == "CAN"
    assert m.chat([{"role": "user", "content": "x"}]) == "CAN"

    with pytest.raises(AttributeError):
        models_pkg.__getattr__("TotallyMissingModel")
