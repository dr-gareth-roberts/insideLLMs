"""W7-0008 slice 3: interactive diff, optimize-prompt, omit-shrink modules."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.cli.commands.optimize_prompt import _parse_strategies, cmd_optimize_prompt
from insideLLMs.datasets.tuf_client import fetch_dataset
from insideLLMs.optimization import OptimizationStrategy
from insideLLMs.publish import oras as oras_mod
from insideLLMs.runtime.diffing_interactive import (
    _summary_display,
    build_interactive_review_lines,
    copy_candidate_artifacts_to_baseline,
    print_interactive_review,
    prompt_accept_snapshot,
)

# ---------------------------------------------------------------------------
# diffing_interactive
# ---------------------------------------------------------------------------


def test_diffing_interactive_full_coverage(tmp_path: Path, capsys) -> None:
    assert _summary_display(None) == "-"
    assert _summary_display({"output": "hello"}) == "hello"
    assert "accuracy=0.9" in _summary_display(
        {"status": "ok", "primary_metric": "accuracy", "primary_score": 0.9}
    )
    assert _summary_display({"status": "error"}) == "status=error"
    assert _summary_display({}) == "-"

    empty = build_interactive_review_lines({}, limit=5)
    assert empty == ["  No differences to review."]

    report = {
        "regressions": [
            {
                "label": {"model": "m", "probe": "p", "example": "e1"},
                "detail": "drop",
                "baseline": {"output": "a"},
                "candidate": {"output": "b"},
            },
            "skip",
            {
                "model_id": "m2",
                "probe_id": "p2",
                "example_id": "e2",
                "kind": "change",
                "baseline": {"status": "ok"},
                "candidate": {},
            },
        ],
        "improvements": [],
        "changes": [],
        "trace_drifts": [],
        "trace_violation_increases": [],
        "trajectory_drifts": [],
        "only_baseline": [
            {"label": {"model": "m", "probe": "p", "example": "old"}},
            "skip",
            {"model_id": "m3", "probe_id": "p3", "example_id": "e3"},
        ],
        "only_candidate": [
            {"label": {"model": "m", "probe": "p", "example": "new"}},
            "skip",
            {"model_id": "m4", "probe_id": "p4", "example_id": "e4"},
        ],
    }
    lines = build_interactive_review_lines(report, limit=1, dim_text=lambda s: f"DIM:{s}")
    assert any("DIM:" in ln for ln in lines)
    assert any("Missing in candidate" in ln for ln in lines)
    assert any("New in candidate" in ln for ln in lines)

    # no "more" truncation branches (limit covers all items)
    exact = build_interactive_review_lines(
        {
            "regressions": [
                {
                    "label": {"model": "m", "probe": "p", "example": "e"},
                    "detail": "d",
                    "baseline": {"output": "a"},
                    "candidate": {"output": "b"},
                }
            ],
            "improvements": [],
            "changes": [],
            "trace_drifts": [],
            "trace_violation_increases": [],
            "trajectory_drifts": [],
            "only_baseline": [{"model_id": "m", "probe_id": "p", "example_id": "e"}],
            "only_candidate": [{"model_id": "m", "probe_id": "p", "example_id": "e"}],
        },
        limit=10,
    )
    assert not any("more" in ln for ln in exact)
    # only regressions, no only_* lists
    assert build_interactive_review_lines(
        {
            "regressions": [
                {
                    "label": {"model": "m", "probe": "p", "example": "e"},
                    "detail": "d",
                    "baseline": {},
                    "candidate": {},
                }
            ],
            "improvements": [],
            "changes": [],
            "trace_drifts": [],
            "trace_violation_increases": [],
            "trajectory_drifts": [],
            "only_baseline": [],
            "only_candidate": [],
        },
        limit=5,
    )

    print_interactive_review(report, limit=2, emit_subheader=lambda t: print(f"## {t}"))
    assert "Interactive Snapshot Review" in capsys.readouterr().out
    print_interactive_review({"regressions": []}, limit=1)
    assert "No differences" in capsys.readouterr().out

    assert prompt_accept_snapshot(input_func=lambda _p: "yes") is True
    assert prompt_accept_snapshot(input_func=lambda _p: "n") is False

    def _eof(_p: str) -> str:
        raise EOFError

    assert prompt_accept_snapshot(input_func=_eof) is False

    base = tmp_path / "base"
    cand = tmp_path / "cand"
    cand.mkdir()
    (cand / "records.jsonl").write_text("{}\n", encoding="utf-8")
    (cand / "manifest.json").write_text("{}", encoding="utf-8")
    copied = copy_candidate_artifacts_to_baseline(base, cand)
    assert "records.jsonl" in copied
    assert (base / "records.jsonl").exists()


# ---------------------------------------------------------------------------
# optimize_prompt CLI
# ---------------------------------------------------------------------------


def test_optimize_prompt_cli_branches(tmp_path: Path, capsys, monkeypatch) -> None:
    assert _parse_strategies(None) is None
    parsed = _parse_strategies("compression,clarity")
    assert parsed is not None
    assert OptimizationStrategy.COMPRESSION in parsed
    with pytest.raises(ValueError, match="Unknown strategy"):
        _parse_strategies("not-a-real-strategy")

    # empty strategies string -> None
    assert _parse_strategies(" , , ") is None

    bad = argparse.Namespace(
        prompt=None,
        input_file=str(tmp_path / "missing.txt"),
        strategies=None,
        format="text",
        output=None,
        show_diff=False,
    )
    assert cmd_optimize_prompt(bad) == 1

    bad2 = argparse.Namespace(
        prompt="",
        input_file=None,
        strategies=None,
        format="text",
        output=None,
        show_diff=False,
    )
    assert cmd_optimize_prompt(bad2) == 1

    assert (
        cmd_optimize_prompt(
            argparse.Namespace(
                prompt="hello",
                input_file=None,
                strategies="bogus",
                format="text",
                output=None,
                show_diff=False,
            )
        )
        == 1
    )

    inp = tmp_path / "prompt.txt"
    inp.write_text("Please kindly explain AI briefly please.", encoding="utf-8")
    out_json = tmp_path / "report.json"
    rc = cmd_optimize_prompt(
        argparse.Namespace(
            prompt=None,
            input_file=str(inp),
            strategies=None,
            format="json",
            output=str(out_json),
            show_diff=False,
        )
    )
    assert rc == 0
    assert out_json.exists()

    rc = cmd_optimize_prompt(
        argparse.Namespace(
            prompt="Please kindly explain AI briefly please.",
            input_file=None,
            strategies=None,
            format="json",
            output=None,
            show_diff=False,
        )
    )
    assert rc == 0
    assert "{" in capsys.readouterr().out

    out_txt = tmp_path / "opt.txt"
    rc = cmd_optimize_prompt(
        argparse.Namespace(
            prompt="Please kindly explain AI briefly please. " * 3,
            input_file=None,
            strategies=None,
            format="text",
            output=str(out_txt),
            show_diff=True,
        )
    )
    assert rc == 0
    assert out_txt.exists()
    assert "Optimize Prompt" in capsys.readouterr().out

    # text path: no show_diff, no output file (still may print suggestions)
    rc = cmd_optimize_prompt(
        argparse.Namespace(
            prompt="Please kindly very carefully explain this please please.",
            input_file=None,
            strategies="compression",
            format="text",
            output=None,
            show_diff=False,
        )
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# tuf_client (omit shrink)
# ---------------------------------------------------------------------------


def test_tuf_client_mock_and_require_tuf() -> None:
    # Force ImportError path even when real `tuf` is installed.
    real_import = __import__

    def _block_tuf(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tuf" or name.startswith("tuf."):
            raise ImportError("blocked for coverage")
        return real_import(name, globals, locals, fromlist, level)

    import builtins

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(builtins, "__import__", _block_tuf)
        with pytest.raises(RuntimeError, match="tuf module not available"):
            fetch_dataset("ds", "1.0", allow_mock=False)

        path, proof = fetch_dataset("ds", "1.0", allow_mock=True, base_url="https://example.com")
        assert path.exists()
        assert proof["status"] == "mock-verified"
        assert proof["base_url"] == "https://example.com"

    # pretend tuf is available
    fake_tuf = types.ModuleType("tuf")
    fake_ng = types.ModuleType("tuf.ngclient")
    fake_ng.Updater = object
    sys.modules["tuf"] = fake_tuf
    sys.modules["tuf.ngclient"] = fake_ng
    try:
        path2, proof2 = fetch_dataset("ds", "2.0", allow_mock=False)
        assert proof2["status"] == "verified"
        assert proof2["method"] == "tuf.ngclient"
        assert path2.exists()
    finally:
        # Restore real modules if they were installed; otherwise clear fakes.
        sys.modules.pop("tuf", None)
        sys.modules.pop("tuf.ngclient", None)
        try:
            import tuf.ngclient  # noqa: F401
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# publish.oras (omit shrink) — mock client
# ---------------------------------------------------------------------------


def test_oras_import_success_flag() -> None:
    """Cover ORAS_AVAILABLE=True import branch (oras not installed in [dev])."""
    import importlib

    original = sys.modules["insideLLMs.publish.oras"]
    fake_client = types.ModuleType("oras.client")
    fake_oras = types.ModuleType("oras")
    fake_oras.client = fake_client
    sys.modules["oras"] = fake_oras
    sys.modules["oras.client"] = fake_client
    try:
        del sys.modules["insideLLMs.publish.oras"]
        reloaded = importlib.import_module("insideLLMs.publish.oras")
        assert reloaded.ORAS_AVAILABLE is True
    finally:
        sys.modules.pop("oras", None)
        sys.modules.pop("oras.client", None)
        sys.modules["insideLLMs.publish.oras"] = original
        import insideLLMs.publish as publish_pkg

        publish_pkg.oras = original


def test_oras_push_pull_verify_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oras_mod, "ORAS_AVAILABLE", False)
    monkeypatch.setattr(oras_mod, "oras_client", None)
    with pytest.raises(RuntimeError, match="oras library"):
        oras_mod.push_run_oci(tmp_path, "ref")

    class FakeClient:
        def __init__(self):
            self.pushed = None
            self.pulled = None

        def push(self, target, files):
            self.pushed = (target, files)

        def pull(self, target, outdir):
            self.pulled = (target, outdir)

    fake_mod = types.SimpleNamespace(OciClient=FakeClient)
    monkeypatch.setattr(oras_mod, "ORAS_AVAILABLE", True)
    monkeypatch.setattr(oras_mod, "oras_client", fake_mod)

    with pytest.raises(ValueError, match="does not exist"):
        oras_mod.push_run_oci(tmp_path / "missing", "r")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="empty"):
        oras_mod.push_run_oci(empty, "r")

    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    result = oras_mod.push_run_oci(run, "registry/run:tag")
    assert result.ref == "registry/run:tag"

    out = tmp_path / "pulled"
    # pull without verify
    pulled = oras_mod.pull_run_oci("registry/run:tag", out)
    assert pulled.path == out

    # verify missing files
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(ValueError, match="manifest.json missing"):
        oras_mod._verify_pulled_run(bare)

    (bare / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="records.jsonl missing"):
        oras_mod._verify_pulled_run(bare)

    (bare / "records.jsonl").write_text("{}\n", encoding="utf-8")
    oras_mod._verify_pulled_run(bare)
    with pytest.raises(ValueError, match="policy file"):
        oras_mod._verify_pulled_run(bare, policy_path="policy.yaml")
    (bare / "policy.yaml").write_text("rules: []\n", encoding="utf-8")
    oras_mod._verify_pulled_run(bare, policy_path="policy.yaml")

    # pull with verify=True against prepared outdir — pull clears? FakeClient doesn't write files
    # so stage files before verify by wrapping pull
    out2 = tmp_path / "pulled2"
    out2.mkdir()
    (out2 / "manifest.json").write_text("{}", encoding="utf-8")
    (out2 / "records.jsonl").write_text("{}\n", encoding="utf-8")

    class FakeClient2(FakeClient):
        def pull(self, target, outdir):
            # leave pre-staged files
            self.pulled = (target, outdir)

    monkeypatch.setattr(oras_mod, "oras_client", types.SimpleNamespace(OciClient=FakeClient2))
    assert oras_mod.pull_run_oci("r", out2, verify=True).path == out2


# ---------------------------------------------------------------------------
# openrouter (omit shrink) without openai SDK
# ---------------------------------------------------------------------------


def test_openrouter_without_openai_sdk() -> None:
    """Cover openrouter.py by temporarily stubbing its OpenAIModel base."""
    import importlib

    class FakeInfo:
        def __init__(self):
            self.provider = "openai"
            self.extra = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._base_url = kwargs.get("base_url")

        def info(self):
            return FakeInfo()

    saved_openai = sys.modules.get("insideLLMs.models.openai")
    saved_openrouter = sys.modules.get("insideLLMs.models.openrouter")
    fake_openai_mod = types.ModuleType("insideLLMs.models.openai")
    fake_openai_mod.OpenAIModel = FakeOpenAI
    sys.modules["insideLLMs.models.openai"] = fake_openai_mod
    sys.modules.pop("insideLLMs.models.openrouter", None)
    try:
        openrouter_mod = importlib.import_module("insideLLMs.models.openrouter")
        m = openrouter_mod.OpenRouterModel(api_key="k", extra_headers={"X": "1"})
        assert m.kwargs["api_key_env"] == "OPENROUTER_API_KEY"
        assert m.kwargs["default_headers"]["X"] == "1"
        assert m.info().provider == "openrouter"
        m2 = openrouter_mod.OpenRouterModel(api_key="k")
        assert "X-Title" in m2.kwargs["default_headers"]
    finally:
        if saved_openai is not None:
            sys.modules["insideLLMs.models.openai"] = saved_openai
        else:
            sys.modules.pop("insideLLMs.models.openai", None)
        if saved_openrouter is not None:
            sys.modules["insideLLMs.models.openrouter"] = saved_openrouter
        else:
            sys.modules.pop("insideLLMs.models.openrouter", None)
            # restore real openrouter against real openai if available
            if saved_openai is not None:
                importlib.import_module("insideLLMs.models.openrouter")


# ---------------------------------------------------------------------------
# integrations.langchain helpers (omit shrink) without LangChain installed
# ---------------------------------------------------------------------------


def test_langchain_helpers_without_deps() -> None:
    from insideLLMs.integrations import langchain as lc

    assert lc._message_content_to_text(None) == ""
    assert lc._message_content_to_text("hi") == "hi"
    assert lc._message_content_to_text(3) == "3"
    assert "a" in lc._message_content_to_text(["a", {"b": 1}])
    assert lc._message_content_to_text({"z": 1})

    class Bad:
        def __str__(self):
            return "bad"

    # non-json-serializable fallback via str after dumps fails — use a key that dumps can handle
    assert isinstance(lc._message_content_to_text(Bad()), str)

    class Msg:
        def __init__(self, typ, content, name=None):
            self.type = typ
            self.content = content
            self.name = name

    converted = lc._lc_messages_to_insidellms(
        [
            Msg("system", "s"),
            Msg("human", "h"),
            Msg("ai", "a"),
            Msg("tool", "t"),
            Msg("other", "o"),
        ]
    )
    assert [c["role"] for c in converted] == ["system", "user", "assistant", "assistant", "user"]

    prompt = lc._insidellms_messages_to_prompt(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}]
    )
    assert "USER: q" in prompt
    assert prompt.endswith("ASSISTANT:")

    def f(*, stop=None):
        return stop

    assert lc._call_with_stop(lambda: "ok") == "ok"
    assert lc._call_with_stop(f, stop=["END"]) == ["END"]

    def only_stop_sequences(*, stop_sequences=None):
        return stop_sequences

    assert lc._call_with_stop(only_stop_sequences, stop=["X"]) == ["X"]

    def no_stop():
        return "plain"

    assert lc._call_with_stop(no_stop, stop=["X"]) == "plain"

    # Force missing langchain_core even when the extra is installed.
    real_import = __import__

    def _block_lc(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "langchain_core" or name.startswith("langchain_core."):
            raise ImportError("blocked for coverage")
        return real_import(name, globals, locals, fromlist, level)

    import builtins

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(builtins, "__import__", _block_lc)
        with pytest.raises(lc.LangChainIntegrationError):
            lc.as_langchain_chat_model(MagicMock())
        with pytest.raises(lc.LangChainIntegrationError):
            lc.as_langchain_runnable(MagicMock())
