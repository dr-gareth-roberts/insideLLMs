"""W7-0008 slice24: close the last measured statement misses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_export_encrypt_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import insideLLMs.privacy.encryption as enc
    from insideLLMs.cli.commands import export as export_mod

    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text(json.dumps({"input": "a", "output": "b", "status": "success"}) + "\n")
    monkeypatch.setenv("INSIDELLMS_ENCRYPTION_KEY", "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY=")

    called = {}

    def _ok(path, key=None):
        called["path"] = str(path)

    monkeypatch.setattr(enc, "encrypt_jsonl", _ok)

    rc = export_mod.cmd_export(
        argparse.Namespace(
            input=str(inp),
            output=str(out),
            format="jsonl",
            encrypt=True,
            encryption_key_env="INSIDELLMS_ENCRYPTION_KEY",
            redact_pii=False,
        )
    )
    assert rc == 0
    assert called.get("path") == str(out)


def test_semantic_cache_redis_unavailable_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import insideLLMs.semantic_cache as sc

    monkeypatch.setattr(sc, "REDIS_AVAILABLE", False)
    with pytest.raises(ImportError, match="redis"):
        sc.RedisCache()
    with pytest.raises(ImportError, match="Redis support"):
        sc.SemanticCache(backend="redis")


def test_sync_runner_batch_validate_output(tmp_path: Path) -> None:
    from insideLLMs.models import DummyModel
    from insideLLMs.probes.logic import LogicProbe
    from insideLLMs.runtime._sync_runner import ProbeRunner

    runner = ProbeRunner(model=DummyModel(), probe=LogicProbe())
    runner.run(
        ["What is 2+2? Answer: 4"],
        run_dir=str(tmp_path / "run"),
        validate_output=True,
        schema_version="1.0.0",
        use_probe_batch=True,
        emit_run_artifacts=True,
        overwrite=True,
        store_messages=False,
    )


def test_run_cmd_logs_experiment_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from insideLLMs.cli.commands import run as run_mod
    from insideLLMs.types import (
        ExperimentResult,
        ModelInfo,
        ProbeCategory,
        ProbeResult,
        ResultStatus,
    )

    tracker = MagicMock()
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model: {type: dummy}\nprobe: {type: logic}\n")

    pr = ProbeResult(
        input="hi",
        output="hi",
        status=ResultStatus.SUCCESS,
        latency_ms=1.0,
    )
    exp = ExperimentResult(
        experiment_id="e1",
        model_info=ModelInfo(name="m", provider="dummy", model_id="m"),
        probe_name="logic",
        probe_category=ProbeCategory.LOGIC,
        results=[pr],
    )

    monkeypatch.setattr(run_mod, "create_tracker", lambda **_k: tracker)
    monkeypatch.setattr(run_mod, "run_experiment_from_config", lambda *a, **k: exp)
    monkeypatch.setattr(run_mod, "_prepare_run_dir", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(run_mod, "load_config", lambda *a, **k: {}, raising=False)
    monkeypatch.setattr(
        run_mod,
        "derive_run_id_from_config_path",
        lambda *a, **k: "rid",
        raising=False,
    )

    args = argparse.Namespace(
        config=str(cfg),
        use_async=False,
        async_mode=False,
        concurrency=1,
        timeout=None,
        stop_on_error=False,
        verbose=False,
        quiet=True,
        validate_output=False,
        schema_version="1.0.0",
        validation_mode="warn",
        run_dir=str(tmp_path / "run"),
        run_root=None,
        run_id="rid",
        overwrite=True,
        resume=False,
        strict_serialization=None,
        deterministic_artifacts=None,
        track="local",
        track_project="p",
        output=None,
        format="summary",
    )
    rc_code = run_mod.cmd_run(args)
    assert rc_code == 0
    tracker.log_experiment_result.assert_called_once_with(exp)
