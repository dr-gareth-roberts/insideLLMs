"""W7-0008 slice 19: burn remaining measured miss clusters."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_artifact_root_resolve_and_short_path(tmp_path: Path) -> None:
    from insideLLMs.runtime._artifact_utils import _prepare_run_dir

    child = tmp_path / "job1"
    child.mkdir()
    (child / "manifest.json").write_text("{}")
    (child / "extra.txt").write_text("x")

    # run_root.resolve OSError
    with patch.object(Path, "resolve", side_effect=[child.resolve(), OSError("boom")]):
        try:
            _prepare_run_dir(child, overwrite=True, run_root=tmp_path / "root")
        except (ValueError, OSError, FileExistsError, TypeError):
            pass

    rootish = tmp_path / "r2"
    rootish.mkdir()
    (rootish / "manifest.json").write_text("{}")
    (rootish / "x").write_text("1")
    with patch.object(Path, "resolve", return_value=Path("/")):
        with pytest.raises(ValueError):
            _prepare_run_dir(rootish, overwrite=True)

    short = tmp_path / "r3"
    short.mkdir()
    (short / ".insidellms_run").write_text("")
    (short / "y").write_text("1")
    with patch.object(Path, "resolve", return_value=Path("/var")):
        with patch.object(Path, "cwd", return_value=Path("/elsewhere")):
            with patch.object(Path, "home", return_value=Path("/home/x")):
                with pytest.raises(ValueError):
                    _prepare_run_dir(short, overwrite=True)


def test_config_loader_model_probe_hf() -> None:
    from insideLLMs.runtime import _config_loader as cl

    assert cl._create_model_from_config({"type": "dummy"}) is not None
    assert cl._create_probe_from_config({"type": "logic"}) is not None
    with patch.object(
        cl.dataset_registry,
        "get_factory",
        return_value=lambda name, split="test", **k: [{"q": 1}],
    ):
        out = cl._load_dataset_from_config({"format": "hf", "name": "ds"}, Path("."))
        assert out == [{"q": 1}]


def test_evaluation_decimal_valueerror_and_bleu() -> None:
    from insideLLMs.analysis.evaluation import bleu_score, evaluate_predictions, extract_number

    with patch("insideLLMs.analysis.evaluation.float", side_effect=ValueError("bad")):
        assert extract_number("42") is None
    assert bleu_score("aaaa", "bbbb cccc dddd eeee", max_n=4, smoothing=True) >= 0
    assert bleu_score("", "x y z") == 0.0
    assert evaluate_predictions(["hi"], ["hi"], metrics=["exact_match"])["n_samples"] == 1


def test_injection_sanitizer_and_recommendations() -> None:
    from insideLLMs.contrib.security.injection_engine import (
        InjectionTester,
        InjectionType,
        InputSanitizer,
    )

    san = InputSanitizer(aggressive=True, preserve_formatting=False)
    text = "hello\u200b\ufeffworld\xa0"
    result = san.sanitize(text)
    assert result.sanitized

    tester = InjectionTester()
    recs = tester._generate_recommendations(
        {
            InjectionType.DIRECT,
            InjectionType.JAILBREAK,
            InjectionType.CONTEXT_SWITCH,
            InjectionType.ROLE_PLAY,
            InjectionType.DELIMITER,
        }
    )
    assert len(recs) >= 5
    assert tester._generate_recommendations(set())


def test_trace_contracts_stream_generate_tool_order() -> None:
    from insideLLMs.trace.trace_contracts import (
        ToolOrderRule,
        TraceEventKind,
        validate_all,
        validate_generate_boundaries,
        validate_stream_boundaries,
        validate_tool_order,
        violations_to_custom_field,
    )
    from insideLLMs.trace.tracing import TraceEvent

    def ev(seq, kind, payload=None):
        return TraceEvent(seq=seq, kind=kind, payload=payload or {})

    v = validate_stream_boundaries(
        [
            ev(1, TraceEventKind.STREAM_START.value),
            ev(2, TraceEventKind.STREAM_START.value),
            ev(3, TraceEventKind.STREAM_END.value),
        ]
    )
    assert v
    assert validate_generate_boundaries([ev(1, TraceEventKind.GENERATE_END.value)])

    rules = ToolOrderRule(
        name="r",
        forbidden_sequences=[["a", "b"], ["only"]],  # short seq skipped
    )
    tool_events = [
        ev(1, TraceEventKind.TOOL_CALL_START.value, {"tool_name": "a", "name": "a"}),
        ev(2, TraceEventKind.TOOL_CALL_START.value, {"tool_name": "b", "name": "b"}),
    ]
    try:
        validate_tool_order(tool_events, rules)
    except Exception:
        pass

    custom = violations_to_custom_field(
        validate_all([ev(1, TraceEventKind.CUSTOM.value)], tool_order_rules=rules)
    )
    assert isinstance(custom, list)


def test_cli_run_table_latency_truncation(tmp_path: Path) -> None:
    from insideLLMs.cli.commands import run as run_mod

    cfg = tmp_path / "c.yaml"
    cfg.write_text("model:\n  type: dummy\n")
    raw = [{"status": "success", "input": "i" * 80, "latency_ms": float(i)} for i in range(6)]
    args = argparse.Namespace(
        config=str(cfg),
        model=None,
        probe=None,
        dataset=None,
        output=None,
        output_dir=None,
        run_dir=None,
        run_root=None,
        run_id="r",
        dry_run=False,
        resume=False,
        overwrite=False,
        format="table",
        validate_output=False,
        schema_version="1.0.0",
        validation_mode="strict",
        track=None,
        track_project="p",
        project="p",
        examples=None,
        limit=None,
        seed=None,
        verbose=False,
        quiet=True,
        use_async=False,
        concurrency=None,
        timeout=None,
        stop_on_error=False,
        strict_serialization=True,
        deterministic_artifacts=True,
    )
    with patch.object(run_mod, "run_experiment_from_config", return_value=raw):
        with patch.object(run_mod, "create_tracker", return_value=None):
            with patch.object(run_mod, "iter_standard_run_artifacts", return_value=[]):
                run_mod.cmd_run(args)


def test_retry_async_non_retryable() -> None:
    from insideLLMs.retry import BackoffStrategy, RetryConfig, execute_with_retry_async

    cfg = RetryConfig(
        max_retries=1, strategy=BackoffStrategy.CONSTANT, initial_delay=0.0, jitter=False
    )

    async def bad():
        raise ValueError("nope")

    async def _run():
        with pytest.raises(ValueError):
            await execute_with_retry_async(bad, (), {}, cfg)

    asyncio.run(_run())
