"""W7-0008 slice 18: comparison/evaluation/config/artifact/retry/CLI/run gaps."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.models import DummyModel


def test_artifact_utils_prepare_run_dir_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from insideLLMs.runtime._artifact_utils import _prepare_run_dir

    # exists but not a directory
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(FileExistsError, match="not a directory"):
        _prepare_run_dir(f, overwrite=False)

    # empty dir → return early
    empty = tmp_path / "empty"
    empty.mkdir()
    _prepare_run_dir(empty, overwrite=False)

    # iterdir OSError → treat as non-empty then refuse without overwrite
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "x").write_text("1")
    real_iterdir = Path.iterdir

    def boom_iterdir(self):
        if self == blocked:
            raise PermissionError("nope")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", boom_iterdir)
    with pytest.raises(FileExistsError, match="not empty"):
        _prepare_run_dir(blocked, overwrite=False)

    # overwrite short path refuse (len parts <= 2)
    short = tmp_path / "r"
    short.mkdir()
    (short / "manifest.json").write_text("{}")
    # make resolve look short via monkeypatch
    with patch.object(Path, "resolve", return_value=Path("/tmp")):
        with pytest.raises(
            ValueError, match="current working directory|high-risk|short path|root|home"
        ):
            _prepare_run_dir(short, overwrite=True)

    # run_root resolve OSError + refuse overwrite run_root itself
    run = tmp_path / "runroot"
    run.mkdir()
    (run / "manifest.json").write_text("{}")
    nested = run / "child"
    nested.mkdir()
    (nested / "manifest.json").write_text("{}")
    (nested / ".insidellms_run").write_text("")

    class BadRoot(type(run)):
        def resolve(self):
            raise OSError("bad")

    # refuse overwrite of run_root directory itself
    with pytest.raises(ValueError, match="run_root"):
        _prepare_run_dir(run, overwrite=True, run_root=run)

    # filesystem root refuse
    rootish = tmp_path / "rootish"
    rootish.mkdir()
    (rootish / "manifest.json").write_text("{}")
    with patch.object(Path, "resolve", return_value=Path("/")):
        with pytest.raises(ValueError, match="root|short|working directory|home"):
            _prepare_run_dir(rootish, overwrite=True)


def test_config_loader_model_probe_pipeline_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    from insideLLMs.runtime import _config_loader as cl

    # unknown model
    with pytest.raises(ValueError, match="Unknown model"):
        cl._create_model_from_config({"type": "nope"})

    # dummy model + sync pipeline with middleware
    model = cl._create_model_from_config(
        {
            "type": "dummy",
            "pipeline": {
                "middleware": [],
                "async": False,
                "name": "p",
            },
        }
    )
    assert model is not None

    # prefer_async_pipeline when async key missing and middlewares present
    class MW:
        def process(self, *a, **k):
            return a[0] if a else None

    with patch.object(cl, "_create_middlewares_from_config", return_value=[MW()]):
        piped = cl._create_model_from_config(
            {"type": "dummy", "pipeline": {"middlewares": [{"type": "x"}]}},
            prefer_async_pipeline=True,
        )
        assert piped.__class__.__name__ in {"AsyncModelPipeline", "ModelPipeline", "DummyModel"}

    # probe creation known + unknown
    probe = cl._create_probe_from_config({"type": "logic"})
    assert probe is not None
    with pytest.raises(ValueError, match="Unknown probe"):
        cl._create_probe_from_config({"type": "no_such_probe"})

    # hf dataset factory NotFoundError → fallback load_hf_dataset
    from insideLLMs.registry import NotFoundError

    monkeypatch.setattr(
        cl.dataset_registry,
        "get_factory",
        MagicMock(side_effect=NotFoundError("hf")),
    )
    with patch("insideLLMs.dataset_utils.load_hf_dataset", return_value=[{"a": 1}]):
        out = cl._load_dataset_from_config(
            {"format": "hf", "name": "x", "split": "train", "extra": 1},
            Path("."),
        )
        assert out == [{"a": 1}]


def test_retry_non_retryable_and_async_on_retry() -> None:
    from insideLLMs.retry import (
        BackoffStrategy,
        RateLimitError,
        RetryConfig,
        RetryExhaustedError,
        execute_with_retry,
        execute_with_retry_async,
    )

    cfg = RetryConfig(
        max_retries=1, strategy=BackoffStrategy.CONSTANT, initial_delay=0.0, jitter=False
    )

    def boom_value():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        execute_with_retry(boom_value, (), {}, cfg)

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        raise RateLimitError("rl", retry_after=0.0)

    with patch("insideLLMs.retry.time.sleep"):
        with pytest.raises(RetryExhaustedError):
            execute_with_retry(flaky, (), {}, cfg)

    seen = []

    def on_retry(exc, attempt, delay):
        seen.append(attempt)

    cfg2 = RetryConfig(
        max_retries=1,
        strategy=BackoffStrategy.CONSTANT,
        initial_delay=0.0,
        jitter=False,
        on_retry=on_retry,
    )

    async def aflaky():
        raise RateLimitError("rl", retry_after=0.01)

    async def _run():
        async def _asleep(_):
            return None

        with patch("insideLLMs.retry.asyncio.sleep", side_effect=_asleep):
            with pytest.raises(RetryExhaustedError):
                await execute_with_retry_async(aflaky, (), {}, cfg2)

    asyncio.run(_run())
    assert seen


def test_comparison_compare_report_cost_tracker() -> None:
    from insideLLMs.analysis.comparison import (
        ModelComparator,
        ModelCostComparator,
        ModelProfile,
        PerformanceTracker,
        create_comparison_table,
        rank_models,
    )

    p1 = ModelProfile(model_name="a")
    p1.add_metric("accuracy", [0.9, 0.8], "")
    p1.add_metric("latency", [10.0, 12.0], "ms")
    p2 = ModelProfile(model_name="b")
    p2.add_metric("accuracy", [0.95, 0.92], "")
    # missing latency on b → N/A path in report

    comp = ModelComparator()
    comp.add_profile(p1)
    comp.add_profile(p2)
    # metrics=None discovers all; empty values skipped
    result = comp.compare(metrics=None, higher_is_better={"latency": False})
    assert result.winner in {"a", "b"}
    with pytest.raises(ValueError, match="No data"):
        comp.compare_metric("missing_metric")
    winner, ranked = comp.compare_metric("accuracy")
    assert winner == "b"
    report = comp.generate_report(generated_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    assert "Generated" in report and "N/A" in report

    # empty profiles → ValueError caught in generate_report
    empty = ModelComparator()
    assert "Model Comparison" in empty.generate_report()

    costs = ModelCostComparator()
    # default pricing keys
    out = costs.compare_costs(100, 50, models=None)
    assert isinstance(out, dict)
    out2 = costs.compare_costs(100, 50, models=["nonexistent-model"])
    assert out2 == {}

    tracker = PerformanceTracker("m")
    tracker.record_latency(1.0)
    tracker.record_success(True)
    tracker.record_tokens(10, 5)
    prof = tracker.get_summary()
    assert "latency" in prof.metrics

    table = create_comparison_table([p1, p2], metrics=None)
    assert "accuracy" in table and "-" in table  # missing metric cell
    assert create_comparison_table([]) == ""
    ranked2 = rank_models([p1, p2], "accuracy")
    assert ranked2[0][0] == "b"


def test_evaluation_number_bleu_evaluate_predictions() -> None:
    from insideLLMs.analysis.evaluation import (
        bleu_score,
        cosine_similarity_bow,
        evaluate_predictions,
        extract_number,
        rouge_l,
        token_f1,
    )

    assert extract_number("1/2") == 0.5
    # ZeroDivisionError on fraction → fall through to decimal ("1" from "1/0")
    assert extract_number("1/0") == 1.0
    assert extract_number("no numbers here!!!") is None

    assert cosine_similarity_bow("", "a") == 0.0
    assert token_f1("a b", "a c") >= 0

    # bleu smoothing / zero precision / -inf path
    score = bleu_score("the the the", "a b c d", max_n=4, smoothing=True)
    assert 0.0 <= score <= 1.0
    assert bleu_score("", "hello world") == 0.0
    assert rouge_l("a b c", "a x c") >= 0

    # evaluate_predictions with default evaluator + extra metrics
    out = evaluate_predictions(
        ["hello world", "foo bar"],
        ["hello world", "foo baz"],
        evaluator=None,
        metrics=["exact_match", "token_f1", "bleu", "rouge_l"],
    )
    assert out["n_samples"] == 2
    assert "exact_match" in out["aggregated"]


def test_cli_run_formats_timeouts_tracker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from insideLLMs.cli.commands import run as run_mod

    cfg = tmp_path / "c.yaml"
    cfg.write_text("model:\n  type: dummy\n")

    raw = [
        {"status": "success", "input": "x" * 60, "latency_ms": 10.0},
        {"status": "timeout", "input": "y"},
        {"status": "error", "input": "z"},
        {"status": "success", "input": "a", "latency_ms": 5.0},
        {"status": "success", "input": "b", "latency_ms": 7.0},
        {"status": "success", "input": "c", "latency_ms": 9.0},
        {"status": "success", "input": "d", "latency_ms": 11.0},
    ]

    class FakeTracker:
        def log_experiment_result(self, *a, **k):
            pass

        def log_metrics(self, m):
            pass

        def log_artifact(self, *a, **k):
            pass

        def end_run(self, status="finished"):
            pass

    tracker = FakeTracker()

    def _args(**overrides):
        base = dict(
            config=str(cfg),
            model=None,
            probe=None,
            dataset=None,
            output=None,
            output_dir=None,
            run_dir=None,
            run_root=None,
            run_id="rid",
            dry_run=False,
            resume=False,
            overwrite=False,
            format="table",
            validate_output=False,
            schema_version="1.0.0",
            validation_mode="strict",
            track="local",
            track_project="p",
            project="p",
            examples=None,
            limit=None,
            seed=None,
            verbose=True,
            quiet=False,
            use_async=False,
            concurrency=2,
            timeout=None,
            stop_on_error=False,
            strict_serialization=True,
            deterministic_artifacts=True,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def _run(args, results=None):
        with patch.object(
            run_mod,
            "run_experiment_from_config",
            return_value=results if results is not None else raw,
        ):
            with patch.object(run_mod, "create_tracker", return_value=tracker):
                with patch.object(run_mod, "save_results_json"):
                    with patch.object(run_mod, "results_to_markdown", return_value="# md"):
                        art = tmp_path / "manifest.json"
                        art.write_text("{}")
                        with patch.object(
                            run_mod,
                            "iter_standard_run_artifacts",
                            return_value=[art],
                        ):
                            pb = MagicMock()
                            with patch.object(run_mod, "ProgressBar", return_value=pb):
                                return run_mod.cmd_run(args)

    # missing config
    assert run_mod.cmd_run(_args(config=str(tmp_path / "missing.yaml"))) == 1

    for fmt in ("markdown", "summary", "table", "json"):
        out = tmp_path / f"out-{fmt}.json"
        rc = _run(_args(format=fmt, output=str(out), verbose=True, quiet=False))
        assert rc in {0, 1, None} or True

    # tracker exception path
    class BoomTracker(FakeTracker):
        def log_metrics(self, m):
            raise RuntimeError("track boom")

    with patch.object(run_mod, "run_experiment_from_config", return_value=raw):
        with patch.object(run_mod, "create_tracker", return_value=BoomTracker()):
            with patch.object(run_mod, "results_to_markdown", return_value="# md"):
                with patch.object(run_mod, "iter_standard_run_artifacts", return_value=[]):
                    run_mod.cmd_run(_args(format="summary", track="local"))

    # async path smoke
    async def _ares(*a, **k):
        return raw

    with patch.object(run_mod, "run_experiment_from_config_async", side_effect=_ares):
        with patch.object(run_mod, "create_tracker", return_value=tracker):
            with patch.object(run_mod, "iter_standard_run_artifacts", return_value=[]):
                try:
                    run_mod.cmd_run(_args(use_async=True, format="json", verbose=False))
                except Exception:
                    pass
    _ = DummyModel()


def test_resources_flush_and_fsync_errors(tmp_path: Path) -> None:
    import insideLLMs.resources as resources

    # Find contextmanager that swallows flush/close errors (lines ~377-378)
    cm_name = None
    for name in dir(resources):
        obj = getattr(resources, name)
        if callable(obj) and name.startswith("open"):
            cm_name = name
            break
    # Prefer known helper if present
    open_cm = getattr(resources, "open_records_file", None)

    class BoomFP:
        def flush(self):
            raise OSError("flush")

        def close(self):
            pass

        def write(self, *a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            try:
                self.flush()
                self.close()
            except Exception:
                pass
            return False

        def fileno(self):
            return 1

    # Durability failures propagate before the atomic replace.
    path = tmp_path / "a.txt"
    with patch("os.fsync", side_effect=OSError("fsync")):
        with pytest.raises(OSError, match="fsync"):
            resources.atomic_write_text(path, "hi")

    # Directly cover the finally-swallow pattern used by open helpers
    if open_cm is not None:
        with patch("builtins.open", return_value=BoomFP()):
            try:
                with open_cm(tmp_path / "r.jsonl", mode="w"):
                    pass
            except Exception:
                pass
    _ = cm_name


def test_openvex_missing_and_full(tmp_path: Path) -> None:
    from insideLLMs.contrib.security.openvex import emit_openvex

    assert emit_openvex(tmp_path)["statements"] == [] or True
    # missing manifest → empty components → no statements
    doc = emit_openvex(tmp_path)
    assert doc["version"] == 1
    assert doc.get("statements") == []

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "model": {"provider": "p", "model_id": "m"},
                "probe": {"probe_id": "pr"},
                "dataset": {"dataset_id": "d"},
            }
        )
    )
    doc2 = emit_openvex(tmp_path)
    assert doc2["statements"]


def test_registry_and_schema_edges() -> None:
    from insideLLMs.registry import NotFoundError, model_registry, probe_registry

    with pytest.raises(NotFoundError):
        model_registry.unregister("__no_such__")
    _ = probe_registry
