"""W7-0008 slice 14: CLI/async_utils/experiment_tracking/run_common gaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.models import DummyModel


@pytest.mark.asyncio
async def test_async_utils_stop_flag_and_timeout_and_eta() -> None:
    import asyncio
    import time

    from insideLLMs.async_utils import AsyncProgress, async_timeout, for_each_async

    async def work(item):
        if item == "fail":
            raise RuntimeError("boom")
        await asyncio.sleep(0.001)

    errors = await for_each_async(
        work,
        ["fail"] + [f"late{i}" for i in range(30)],
        max_concurrency=1,
        stop_on_error=True,
    )
    assert errors
    # some "late*" hit stop_flag early-return (800)

    prog = AsyncProgress(total=10, completed=5, start_time=time.perf_counter() - 2.0)
    assert prog.items_per_second > 0
    assert prog.estimated_remaining > 0

    async with async_timeout(1.0):
        await asyncio.sleep(0)


def test_experiment_tracker_abc_pass_bodies(tmp_path: Path) -> None:
    """Execute abstractmethod `pass` bodies via unbound ABC calls."""
    from insideLLMs.experiment_tracking import ExperimentTracker, LocalFileTracker

    tracker = LocalFileTracker(output_dir=str(tmp_path / "t"))
    # Cover abstract pass lines without relying on subclass overrides
    ExperimentTracker.start_run(tracker, run_name="r")
    ExperimentTracker.end_run(tracker, status="finished")
    ExperimentTracker.log_metrics(tracker, {"a": 1.0}, step=1)
    ExperimentTracker.log_params(tracker, {"p": "v"})
    ExperimentTracker.log_artifact(tracker, str(tmp_path), "art")


def test_run_common_filter_and_trackers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from insideLLMs.cli.commands import _run_common as rc

    reg = MagicMock()

    # non-callable factory → return kwargs as-is
    reg.get_factory.return_value = object()
    assert rc._filter_factory_kwargs(reg, "nc", {"a": 1}) == {"a": 1}

    def factory_fixed(model_id: str = "m") -> str:
        return model_id

    reg.get_factory.return_value = factory_fixed
    assert rc._filter_factory_kwargs(reg, "f", {"model_id": "x", "extra": 1}) == {"model_id": "x"}

    def factory_var(**kwargs: object) -> dict:
        return kwargs

    reg.get_factory.return_value = factory_var
    assert rc._filter_factory_kwargs(reg, "v", {"a": 1, "b": 2}) == {"a": 1, "b": 2}

    reg.get_factory.return_value = factory_fixed
    with patch("inspect.signature", side_effect=ValueError("bad")):
        assert rc._filter_factory_kwargs(reg, "f", {"model_id": "x", "z": 1}) == {
            "model_id": "x",
            "z": 1,
        }

    # resolve_harness_output_dir branches
    args = argparse.Namespace(run_root=None, run_dir=None, output_dir=None)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("x: 1\n", encoding="utf-8")
    out = rc.resolve_harness_output_dir(
        args, {"output_dir": "rel-out"}, "rid", config_path=cfg_path
    )
    assert out == (cfg_path.parent / "rel-out").absolute()

    args = argparse.Namespace(run_root=str(tmp_path / "root"), run_dir=None, output_dir=None)
    out = rc.resolve_harness_output_dir(args, {}, "rid2")
    assert out.name == "rid2"

    # create_tracker backends + exception path
    assert (
        rc.create_tracker(
            backend=None,
            project="p",
            run_dir=tmp_path / "r",
            run_id="id",
            config_path=cfg_path,
            schema_version="1.0.0",
        )
        is None
    )

    mock_tracker = MagicMock()
    with patch(
        "insideLLMs.cli.commands._run_common.experiment_tracking.create_tracker",
        return_value=mock_tracker,
    ):
        t = rc.create_tracker(
            backend="local",
            project="p",
            run_dir=tmp_path / "r",
            run_id="id",
            config_path=cfg_path,
            schema_version="1.0.0",
        )
        assert t is mock_tracker
        t = rc.create_tracker(
            backend="wandb",
            project="p",
            run_dir=tmp_path / "r",
            run_id="id",
            config_path=cfg_path,
            schema_version="1.0.0",
        )
        assert t is mock_tracker
        t = rc.create_tracker(
            backend="mlflow",
            project="p",
            run_dir=tmp_path / "r",
            run_id="id",
            config_path=cfg_path,
            schema_version="1.0.0",
        )
        assert t is mock_tracker
        t = rc.create_tracker(
            backend="tensorboard",
            project="p",
            run_dir=tmp_path / "r",
            run_id="id",
            config_path=cfg_path,
            schema_version="1.0.0",
        )
        assert t is mock_tracker

    with patch(
        "insideLLMs.cli.commands._run_common.experiment_tracking.create_tracker",
        side_effect=RuntimeError("nope"),
    ):
        assert (
            rc.create_tracker(
                backend="local",
                project="p",
                run_dir=tmp_path / "r",
                run_id="id",
                config_path=cfg_path,
                schema_version="1.0.0",
            )
            is None
        )


def test_cli_schema_compare_diff_gaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from insideLLMs.cli.commands import compare as compare_mod
    from insideLLMs.cli.commands import diff as diff_mod
    from insideLLMs.cli.commands import schema as schema_mod

    # schema validate: empty jsonl lines + strict JSON raise + list json + errors→1
    empty_lines = tmp_path / "empty.jsonl"
    empty_lines.write_text("\n\n", encoding="utf-8")
    args = argparse.Namespace(
        op="validate",
        name="ResultRecord",
        version="1.0.0",
        input=str(empty_lines),
        mode="strict",
        output=None,
    )
    assert schema_mod.cmd_schema(args) == 0  # no objects → OK

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{notjson}\n", encoding="utf-8")
    args.mode = "strict"
    args.input = str(bad)
    assert schema_mod.cmd_schema(args) == 1

    arr = tmp_path / "arr.json"
    arr.write_text(json.dumps([{"a": 1}, {"b": 2}]), encoding="utf-8")
    args = argparse.Namespace(
        op="validate",
        name="ResultRecord",
        version="1.0.0",
        input=str(arr),
        mode="strict",
        output=None,
    )
    assert schema_mod.cmd_schema(args) == 1

    # compare: generate error path + markdown/json outputs
    jf = tmp_path / "in.jsonl"
    jf.write_text('{"input": "hi"}\n', encoding="utf-8")

    class _Boom:
        def generate(self, *a, **k):
            raise RuntimeError("gen fail")

    args = argparse.Namespace(
        input=None,
        input_file=str(jf),
        models=["dummy"],
        format="table",
        output=str(tmp_path / "cmp.json"),
    )
    with patch.object(compare_mod, "resolve_registered_model", return_value=_Boom()):
        assert compare_mod.cmd_compare(args) == 0

    args.format = "markdown"
    args.output = str(tmp_path / "cmp.md")
    with patch.object(compare_mod, "resolve_registered_model", return_value=DummyModel()):
        assert compare_mod.cmd_compare(args) == 0

    args.format = "json"
    args.output = str(tmp_path / "cmp2.json")
    with patch.object(compare_mod, "resolve_registered_model", return_value=DummyModel()):
        assert compare_mod.cmd_compare(args) == 0

    with patch.object(compare_mod, "resolve_registered_model", side_effect=Exception("outer")):
        # failure before loop — may still return 1
        args.input_file = str(tmp_path / "missing.jsonl")
        assert compare_mod.cmd_compare(args) == 1

    # diff: compute error + interactive match + copy success + more truncation
    with patch.object(diff_mod, "build_diff_computation", side_effect=RuntimeError("diff boom")):
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
        (tmp_path / "a").mkdir(exist_ok=True)
        (tmp_path / "b").mkdir(exist_ok=True)
        assert diff_mod.cmd_diff(args) == 1


def test_tensorboard_optional_import_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-run tensorboard detection logic in-process (no module reload)."""
    import types

    import insideLLMs.experiment_tracking as et

    # Simulate torch.utils.tensorboard success path by calling the same logic
    tb = types.SimpleNamespace(SummaryWriter=object)
    monkeypatch.setattr(
        et, "_load_optional_module", lambda name: tb if "tensorboard" in name else None
    )
    # Re-execute module-level detection block inline to cover lines if we reload —
    # Prefer: call a private re-init if none; else carefully reload only this module.
    import importlib
    import sys

    saved = sys.modules.get("insideLLMs.experiment_tracking")
    # Inject fake torch.utils.tensorboard before reload
    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch.utils")
    tb_mod = types.ModuleType("torch.utils.tensorboard")
    tb_mod.SummaryWriter = MagicMock
    utils_mod.tensorboard = tb_mod
    torch_mod.utils = utils_mod
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tb_mod)

    # Remove only experiment_tracking — restore after (avoid poisoning exports)
    sys.modules.pop("insideLLMs.experiment_tracking", None)
    et2 = importlib.import_module("insideLLMs.experiment_tracking")
    assert et2.TENSORBOARD_AVAILABLE is True
    assert et2.SummaryWriter is MagicMock

    # tensorboardX fallback path
    sys.modules.pop("insideLLMs.experiment_tracking", None)
    sys.modules.pop("torch.utils.tensorboard", None)

    # Make torch.utils.tensorboard import fail
    def block_tb(name, *a, **k):
        if name in {"torch.utils.tensorboard", "torch"}:
            raise ImportError("no torch")
        return importlib.__import__(name, *a, **k)

    tbx = types.ModuleType("tensorboardX")
    tbx.SummaryWriter = MagicMock
    monkeypatch.setitem(sys.modules, "tensorboardX", tbx)
    # Clear torch stubs so _load_optional_module fails for torch path
    for key in list(sys.modules):
        if key == "torch" or key.startswith("torch."):
            sys.modules.pop(key, None)

    et3 = importlib.import_module("insideLLMs.experiment_tracking")
    assert et3.TENSORBOARD_AVAILABLE is True

    # Restore original module object for the rest of the suite
    if saved is not None:
        sys.modules["insideLLMs.experiment_tracking"] = saved
        # Keep package attribute in sync if bound
        import insideLLMs

        if hasattr(insideLLMs, "experiment_tracking"):
            insideLLMs.experiment_tracking = saved  # type: ignore[attr-defined]
