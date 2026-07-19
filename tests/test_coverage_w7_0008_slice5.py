"""W7-0008 slice 5: harness/interactive/structured/visualization measured gaps."""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from insideLLMs.cli.commands import harness as harness_mod
from insideLLMs.cli.commands.interactive import cmd_interactive
from insideLLMs.structured import (
    SchemaGenerationError,
    _create_model_from_name,
    quick_extract,
)


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
        viz.check_plotly_deps()

        # pandas missing on that path
        real_import = __import__

        def no_pandas(name, *a, **kw):
            if name == "pandas" or (isinstance(name, str) and name.startswith("pandas.")):
                raise ImportError("nope")
            return real_import(name, *a, **kw)

        with patch("builtins.__import__", side_effect=no_pandas):
            with pytest.raises(ImportError, match="pandas is required"):
                viz.check_plotly_deps()
    finally:
        for key, value in saved.items():
            if value is None:
                viz.__dict__.pop(key, None)
            else:
                viz.__dict__[key] = value
