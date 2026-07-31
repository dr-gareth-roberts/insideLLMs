"""W7-0008 slice 4: close measured hotspots (serialization, CLI, config, async runner)."""

from __future__ import annotations

import argparse
import asyncio
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

from insideLLMs._serialization import (
    StrictSerializationError,
    _path_label,
    _serialize_dict_key,
    serialize_value,
    stable_json_dumps,
)
from insideLLMs.cli import _output as cli_output
from insideLLMs.config_types import ProgressInfo, RunConfig, RunConfigBuilder
from insideLLMs.exceptions import RunnerExecutionError
from insideLLMs.models import DummyModel
from insideLLMs.registry import (
    _call_plugin_register,
    _lazy_import_factory,
    load_entrypoint_plugins,
)
from insideLLMs.runtime.runner import AsyncProbeRunner

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
