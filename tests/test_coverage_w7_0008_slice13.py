"""W7-0008 slice 13: async_runner/CLI/local/probes/experiment_tracking gaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.config_types import RunConfig
from insideLLMs.exceptions import RunnerExecutionError
from insideLLMs.models import DummyModel
from insideLLMs.probes.base import Probe
from insideLLMs.runtime.runner import AsyncProbeRunner
from insideLLMs.types import ProbeCategory, ProbeResult, ResultStatus


class _OkProbe:
    name = "ok"

    def run(self, model, item, **kwargs):
        return "ok"

    def run_batch(self, model, items, max_workers=None, progress_callback=None, **kwargs):
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


class _MultiFailProbe:
    """First item fails so stop_on_error skips the rest (non-batch)."""

    name = "multi-fail"
    calls = 0

    def run(self, model, item, **kwargs):
        type(self).calls += 1
        if type(self).calls == 1:
            raise RuntimeError("first-fail")
        return "late"


@pytest.mark.asyncio
async def test_async_runner_stop_skip_validate_ultimate_snapshot(tmp_path: Path) -> None:
    _MultiFailProbe.calls = 0
    fail = AsyncProbeRunner(DummyModel(), _MultiFailProbe())
    with pytest.raises(RunnerExecutionError):
        await fail.run(
            [
                {"messages": [{"role": "user", "content": "a"}]},
                {"messages": [{"role": "user", "content": "b"}]},
            ],
            concurrency=1,
            stop_on_error=True,
            use_probe_batch=False,
            emit_run_artifacts=True,
            run_dir=tmp_path / "skip",
            run_id="skip-run",
            overwrite=True,
            return_experiment=False,
            validate_output=False,
        )

    # batch + progress_callback hits batch_progress (570) — legacy (cur, tot) only
    seen = []

    def cb(cur, tot):
        seen.append((cur, tot))

    runner = AsyncProbeRunner(DummyModel(), _OkProbe())
    await runner.run(
        [
            {"messages": [{"role": "user", "content": "a"}]},
            {"messages": [{"role": "user", "content": "b"}]},
        ],
        use_probe_batch=True,
        batch_workers=2,
        progress_callback=cb,
        emit_run_artifacts=True,
        run_dir=tmp_path / "prog",
        run_id="prog-run",
        overwrite=True,
        return_experiment=False,
        validate_output=True,
        schema_version="1.0.1",
    )
    assert seen

    # config_snapshot derives run_id (288); ultimate + validate (698/734)
    cfg = RunConfig(
        run_mode="ultimate",
        emit_run_artifacts=True,
        publish_oci_ref=None,
        scitt_service_url=None,
    )
    await runner.run(
        [{"messages": [{"role": "user", "content": "u"}]}],
        config=cfg,
        run_dir=tmp_path / "ult2",
        run_id=None,
        overwrite=True,
        return_experiment=False,
        config_snapshot={"models": [{"type": "dummy"}], "seed": 7},
        validate_output=True,
        schema_version="1.0.1",
    )

    # library_version ImportError branch during manifest write (694)
    import builtins

    real_import = builtins.__import__

    def block_pkg(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "insideLLMs" and not fromlist:
            raise ImportError("forced-pkg")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=block_pkg):
        await runner.run(
            [{"messages": [{"role": "user", "content": "u2"}]}],
            emit_run_artifacts=True,
            run_dir=tmp_path / "imp",
            run_id="imp",
            overwrite=True,
            return_experiment=False,
            validate_output=False,
        )

    # empty remaining on resume+batch (completed == len)
    run_dir = tmp_path / "full"
    prompts = [{"messages": [{"role": "user", "content": "x"}]}]
    await runner.run(
        prompts,
        emit_run_artifacts=True,
        run_dir=run_dir,
        run_id="full",
        overwrite=True,
        return_experiment=False,
    )
    await runner.run(
        prompts,
        use_probe_batch=True,
        emit_run_artifacts=True,
        run_dir=run_dir,
        run_id="full",
        resume=True,
        return_experiment=False,
    )


@pytest.mark.asyncio
async def test_async_runner_incomplete_results_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    runner = AsyncProbeRunner(DummyModel(), _OkProbe())

    async def fake_gather(*tasks, **kwargs):
        for t in tasks:
            if asyncio.iscoroutine(t):
                t.close()
            elif hasattr(t, "cancel"):
                t.cancel()
        return None

    monkeypatch.setattr("insideLLMs.runtime._async_runner.asyncio.gather", fake_gather)
    with pytest.raises(RuntimeError, match="did not produce results"):
        await runner.run(
            [{"messages": [{"role": "user", "content": "a"}]}],
            use_probe_batch=False,
            emit_run_artifacts=False,
            return_experiment=False,
            run_id="incomplete",
        )


def test_cli_diff_export_schema_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from insideLLMs.cli.commands import compare as compare_mod
    from insideLLMs.cli.commands import diff as diff_mod
    from insideLLMs.cli.commands import export as export_mod
    from insideLLMs.cli.commands import schema as schema_mod

    # --- schema ---
    args = argparse.Namespace(
        op="list",
        name=None,
        version="1.0.0",
        output=None,
        input=None,
        mode="strict",
    )
    assert schema_mod.cmd_schema(args) == 0

    args = argparse.Namespace(
        op="dump",
        name=None,
        version="1.0.0",
        output=None,
        input=None,
        mode="strict",
    )
    assert schema_mod.cmd_schema(args) == 1

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{not json\n{"a": 1}\n', encoding="utf-8")
    args = argparse.Namespace(
        op="validate",
        name="ResultRecord",
        version="1.0.0",
        input=str(bad),
        mode="warn",
        output=None,
    )
    # may fail name/schema — still exercises warn paths
    schema_mod.cmd_schema(args)

    # Unknown op is treated as dump shortcut → missing/unknown schema → 2
    args = argparse.Namespace(
        op="nope",
        name=None,
        version="1.0.0",
        input=None,
        mode="strict",
        output=None,
    )
    assert schema_mod.cmd_schema(args) == 2

    # --- export encrypt RuntimeError ---
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps([{"a": 1}]), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    monkeypatch.setenv("INSIDELLMS_ENCRYPTION_KEY", "dGVzdC1rZXktZm9yLWZlcm5ldC0xMjM0NTY=")
    args = argparse.Namespace(
        format="jsonl",
        output=str(out),
        encrypt=True,
        input=str(inp),
        redact_pii=False,
        encryption_key_env="INSIDELLMS_ENCRYPTION_KEY",
    )
    with patch(
        "insideLLMs.privacy.encryption.encrypt_jsonl",
        side_effect=RuntimeError("crypto boom"),
    ):
        assert export_mod.cmd_export(args) == 1

    # --- compare: empty jsonl line + model init fail ---
    jf = tmp_path / "in.jsonl"
    jf.write_text('\n{"input": "hi"}\n', encoding="utf-8")
    args = argparse.Namespace(
        input=None,
        input_file=str(jf),
        models=["missing-model"],
        format="table",
        output=None,
    )
    with patch.object(
        compare_mod,
        "resolve_registered_model",
        side_effect=RuntimeError("no model"),
    ):
        assert compare_mod.cmd_compare(args) == 0

    # --- diff: judge verdict non-dict + interactive copy empty ---
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()
    rec = {
        "schema_version": "1.0.0",
        "run_id": "r",
        "index": 0,
        "status": "success",
        "model": {"model_id": "m", "provider": "p"},
        "probe": {"probe_id": "pr"},
        "item": {"id": "e1"},
        "output": "x",
    }
    (run_a / "records.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    rec2 = dict(rec)
    rec2["output"] = "y"
    (run_b / "records.jsonl").write_text(json.dumps(rec2) + "\n", encoding="utf-8")

    diff_mod._print_judge_review(
        {
            "verdicts": [
                "skip",
                {
                    "decision": "review",
                    "reason": "r",
                    "label": {"model": "m", "probe": "p", "example": "e"},
                },
            ]
            + [{"decision": "ok", "reason": "x"} for _ in range(5)],
        },
        limit=2,
    )

    args = argparse.Namespace(
        run_dir_a=str(run_a),
        run_dir_b=str(run_b),
        format="text",
        output=None,
        interactive=True,
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
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with patch.object(diff_mod, "prompt_accept_snapshot", return_value=True):
        with patch.object(diff_mod, "copy_candidate_artifacts_to_baseline", return_value=[]):
            # may return 0/1 depending on diffs
            diff_mod.cmd_diff(args)


def test_local_model_and_vllm_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    import insideLLMs.models.local as local

    # LlamaCppModel
    llama_mod = MagicMock()
    llama_inst = MagicMock()
    llama_inst.return_value = {"choices": [{"text": "hi"}]}
    llama_inst.create_chat_completion.return_value = {"choices": [{"message": {"content": "chat"}}]}
    llama_mod.Llama = MagicMock(return_value=llama_inst)
    monkeypatch.setitem(__import__("sys").modules, "llama_cpp", llama_mod)

    # Avoid reload: patch import inside method via sys.modules only
    m = local.LlamaCppModel(model_path="/tmp/m.gguf")
    # force _model None and import path
    m._model = None
    with patch.dict("sys.modules", {"llama_cpp": llama_mod}):
        assert m.generate("p") == "hi"
        llama_inst.return_value = {"choices": []}
        assert m.generate("p") == ""
        llama_inst.create_chat_completion.return_value = {"choices": []}
        assert m.chat([{"role": "user", "content": "x"}]) == ""

    # ImportError path
    m2 = local.LlamaCppModel(model_path="/tmp/m.gguf")
    m2._model = None
    import builtins

    real_import = builtins.__import__

    def block(name, *a, **k):
        if name == "llama_cpp":
            raise ImportError("missing")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", block)
    with pytest.raises(ImportError, match="llama-cpp-python"):
        m2._get_model()
    monkeypatch.setattr(builtins, "__import__", real_import)

    # VLLMModel empty choices + openai import error
    vm = local.VLLMModel(model_name="m", base_url="http://localhost:8000")
    client = MagicMock()
    client.completions.create.return_value = SimpleNamespace(choices=[])
    client.chat.completions.create.return_value = SimpleNamespace(choices=[])
    vm._client = client
    assert vm.generate("p") == ""
    assert vm.chat([{"role": "user", "content": "x"}]) == ""

    vm._client = None

    def block_openai(name, *a, **k):
        if name == "openai" or name.startswith("openai"):
            raise ImportError("missing")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", block_openai)
    with pytest.raises(ImportError, match="openai"):
        vm._get_client()


def test_probe_run_batch_exception_status_mapping() -> None:
    class _BoomProbe(Probe[str]):
        category = ProbeCategory.CUSTOM

        def __init__(self):
            super().__init__(name="boom", category=ProbeCategory.CUSTOM)

        def run(self, model, data, **kwargs):
            raise TimeoutError("slow")

    class _RateProbe(Probe[str]):
        category = ProbeCategory.CUSTOM

        def __init__(self):
            super().__init__(name="rate", category=ProbeCategory.CUSTOM)

        def run(self, model, data, **kwargs):
            raise RuntimeError("rate limit 429")

    model = DummyModel()
    out = _BoomProbe().run_batch(model, ["a", "b"], max_workers=2)
    assert any(r.status == ResultStatus.TIMEOUT for r in out)

    out2 = _RateProbe().run_batch(model, ["a"], max_workers=2)
    assert out2[0].status == ResultStatus.RATE_LIMITED

    # Outer future.result() exception path (611–619)
    from concurrent import futures as cf

    class _Ok(Probe[str]):
        def __init__(self):
            super().__init__(name="ok", category=ProbeCategory.CUSTOM)

        def run(self, model, data, **kwargs):
            return "x"

    probe = _Ok()
    fut_timeout = cf.Future()
    fut_timeout.set_exception(TimeoutError("pool"))
    fut_rate = cf.Future()
    fut_rate.set_exception(RuntimeError("429 rate"))
    fut_err = cf.Future()
    fut_err.set_exception(ValueError("boom"))
    queue = [fut_timeout, fut_rate, fut_err]

    class _FakePool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, item):
            return queue.pop(0)

    with patch("concurrent.futures.ThreadPoolExecutor", _FakePool):
        with patch(
            "concurrent.futures.as_completed",
            lambda futs: list(futs.keys()) if isinstance(futs, dict) else list(futs),
        ):
            mapped = probe.run_batch(model, ["a", "b", "c"], max_workers=3)
    assert mapped[0].status == ResultStatus.TIMEOUT
    assert mapped[1].status == ResultStatus.RATE_LIMITED
    assert mapped[2].status == ResultStatus.ERROR


def test_experiment_tracking_wandb_mlflow_no_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    import insideLLMs.experiment_tracking as et

    # Patch in place — do not reload module
    monkeypatch.setattr(et, "WANDB_AVAILABLE", True)
    wb = MagicMock()
    run = MagicMock()
    type(run).id = property(lambda self: None)
    wb.init.return_value = run
    wb.finish = MagicMock()
    monkeypatch.setattr(et, "wandb", wb)

    tracker = et.WandBTracker(
        project="p", config=et.TrackingConfig(project="p", experiment_name="e")
    )
    monkeypatch.setattr(tracker, "_wandb", wb)
    with pytest.raises(RuntimeError, match="run id"):
        tracker.start_run("r1")

    monkeypatch.setattr(et, "MLFLOW_AVAILABLE", True)
    ml = MagicMock()
    info = SimpleNamespace(run_id=None, experiment_id=None)
    ml.start_run.return_value = SimpleNamespace(info=info)
    ml.set_experiment = MagicMock()
    monkeypatch.setattr(et, "mlflow", ml)
    mt = et.MLflowTracker(config=et.TrackingConfig(project="p", experiment_name="e"))
    monkeypatch.setattr(mt, "_mlflow", ml)
    with pytest.raises(RuntimeError, match="run id"):
        mt.start_run("r2")
