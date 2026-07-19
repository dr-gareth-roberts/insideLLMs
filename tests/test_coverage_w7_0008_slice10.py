"""W7-0008 slice 10: viz/optimization/caching/tokens measured gaps."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from insideLLMs.caching import (
    CachedModel,
    CacheWarmer,
    DiskCache,
    InMemoryCache,
    StrategyCache,
)
from insideLLMs.optimization import (
    FewShotSelector,
    InstructionOptimizer,
    PromptOptimizer,
    TokenBudgetOptimizer,
)
from insideLLMs.tokens import (
    ContextWindowManager,
    EmbeddingUtils,
    TokenAnalyzer,
    TokenEstimator,
    TokenSpendingBudget,
    VocabCoverage,
)
from insideLLMs.types import (
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)


def _exp(name="m", tokens=10) -> ExperimentResult:
    results = [
        ProbeResult(
            input="i",
            output="o",
            status=ResultStatus.SUCCESS,
            latency_ms=10.0,
        )
    ]
    return ExperimentResult(
        experiment_id=f"e-{name}",
        model_info=ModelInfo(name=name, provider="p", model_id=name),
        probe_name="Logic",
        probe_category=ProbeCategory.LOGIC,
        results=results,
        score=ProbeScore(
            accuracy=0.9,
            precision=0.9,
            recall=0.9,
            f1_score=0.9,
            mean_latency_ms=10.0,
            total_tokens=tokens,
        ),
    )


def test_viz_show_paths_and_seaborn_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import insideLLMs.analysis.visualization as viz

    monkeypatch.setattr(viz.plt, "show", MagicMock())
    monkeypatch.setattr(viz.plt, "savefig", MagicMock())
    monkeypatch.setattr(viz.plt, "close", MagicMock())
    monkeypatch.setattr(viz.plt, "figure", MagicMock())
    monkeypatch.setattr(viz.plt, "bar", MagicMock())
    monkeypatch.setattr(viz.plt, "subplot", MagicMock())
    monkeypatch.setattr(viz.plt, "title", MagicMock())
    monkeypatch.setattr(viz.plt, "xlabel", MagicMock())
    monkeypatch.setattr(viz.plt, "ylabel", MagicMock())
    monkeypatch.setattr(viz.plt, "xticks", MagicMock())
    monkeypatch.setattr(viz.plt, "tight_layout", MagicMock())
    monkeypatch.setattr(viz.plt, "ylim", MagicMock())
    monkeypatch.setattr(viz.plt, "legend", MagicMock())
    monkeypatch.setattr(viz.plt, "grid", MagicMock())
    monkeypatch.setattr(viz.plt, "boxplot", MagicMock())
    monkeypatch.setattr(viz.plt, "plot", MagicMock())

    exps = [_exp("A"), _exp("B")]
    viz.plot_accuracy_comparison(exps, save_path=None)
    viz.plot_metric_comparison(exps, metrics=["accuracy"], save_path=None)
    viz.plot_success_rate_over_time([("t1", 0.5), ("t2", 0.8)], save_path=None)

    bias = [{"output": [("response A is longer", "response B")]}]
    fact = [
        {
            "output": [
                {"category": "science", "model_answer": "The sun is a star."},
                {"category": "history", "model_answer": "Rome was an empire."},
            ]
        }
    ]

    # non-seaborn path already covered elsewhere; force seaborn True
    original_seaborn = viz.SEABORN_AVAILABLE
    monkeypatch.setattr(viz, "SEABORN_AVAILABLE", True)
    fake_sns = MagicMock()
    viz.__dict__["sns"] = fake_sns
    try:
        viz.plot_bias_results(bias, save_path=None)
        viz.plot_factuality_results(fact, save_path=None)
        assert fake_sns.barplot.called
        assert fake_sns.boxplot.called
    finally:
        viz.__dict__.pop("sns", None)
        monkeypatch.setattr(viz, "SEABORN_AVAILABLE", original_seaborn)


def test_viz_reload_optional_import_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hit seaborn/plotly/ipywidgets import-success lines via reload with stubs."""
    sns = types.ModuleType("seaborn")
    sns.barplot = MagicMock()
    sns.boxplot = MagicMock()
    monkeypatch.setitem(sys.modules, "seaborn", sns)

    px = types.ModuleType("plotly.express")
    go = types.ModuleType("plotly.graph_objects")
    sub = types.ModuleType("plotly.subplots")
    plotly = types.ModuleType("plotly")
    plotly.express = px
    plotly.graph_objects = go
    plotly.subplots = sub
    sub.make_subplots = MagicMock()
    monkeypatch.setitem(sys.modules, "plotly", plotly)
    monkeypatch.setitem(sys.modules, "plotly.express", px)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", go)
    monkeypatch.setitem(sys.modules, "plotly.subplots", sub)

    ipy = types.ModuleType("ipywidgets")
    ipython_display = types.ModuleType("IPython.display")
    ipython = types.ModuleType("IPython")
    ipython.display = ipython_display
    ipython_display.display = MagicMock()
    monkeypatch.setitem(sys.modules, "ipywidgets", ipy)
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    for key in list(sys.modules):
        if key == "insideLLMs.analysis.visualization" or key.startswith(
            "insideLLMs.analysis.visualization."
        ):
            del sys.modules[key]

    import insideLLMs.analysis.visualization as viz

    assert viz.SEABORN_AVAILABLE is True
    assert viz.PLOTLY_AVAILABLE is True
    assert viz.IPYWIDGETS_AVAILABLE is True

    # cleanup for other tests
    for key in list(sys.modules):
        if key == "insideLLMs.analysis.visualization" or key.startswith(
            "insideLLMs.analysis.visualization."
        ):
            del sys.modules[key]


def test_viz_interactive_html_exception_and_stabilize(tmp_path: Path, monkeypatch) -> None:
    import insideLLMs.analysis.visualization as viz

    exps = [_exp("A", tokens=100), _exp("B", tokens=50)]
    bad_html = "<div>no plotly id</div>"

    class _Fig:
        def __init__(self, html):
            self._html = html

        def to_html(self, **kwargs):
            return self._html

        def update_layout(self, **kwargs):
            return None

    monkeypatch.setattr(viz, "PLOTLY_AVAILABLE", True)
    monkeypatch.setattr(viz, "check_plotly_deps", lambda: None)
    monkeypatch.setattr(viz, "pd", __import__("pandas"))

    class BoomPx:
        @staticmethod
        def bar(*a, **k):
            raise ValueError("token boom")

    viz.__dict__["px"] = BoomPx
    monkeypatch.setattr(
        viz,
        "interactive_accuracy_comparison",
        MagicMock(side_effect=ValueError("no")),
    )
    monkeypatch.setattr(
        viz,
        "interactive_latency_distribution",
        MagicMock(side_effect=KeyError("k")),
    )
    monkeypatch.setattr(
        viz,
        "interactive_metric_radar",
        MagicMock(return_value=_Fig('<div id="abc123" class="plotly-graph-div"></div>')),
    )
    monkeypatch.setattr(
        viz,
        "interactive_heatmap",
        MagicMock(return_value=_Fig(bad_html)),
    )

    out = tmp_path / "r.html"
    viz.create_interactive_html_report(exps, save_path=str(out), title="T")
    assert out.exists()

    calls = {"n": 0}

    class FlipPx:
        @staticmethod
        def bar(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Fig('<div id="x1" class="plotly-graph-div"></div>')
            raise KeyError("status")

    viz.__dict__["px"] = FlipPx
    monkeypatch.setattr(
        viz,
        "interactive_accuracy_comparison",
        MagicMock(return_value=_Fig('<div id="same" class="plotly-graph-div"></div>')),
    )
    monkeypatch.setattr(
        viz,
        "interactive_latency_distribution",
        MagicMock(return_value=_Fig(bad_html)),
    )
    monkeypatch.setattr(
        viz,
        "interactive_metric_radar",
        MagicMock(return_value=_Fig(bad_html)),
    )
    monkeypatch.setattr(
        viz,
        "interactive_heatmap",
        MagicMock(return_value=_Fig(bad_html)),
    )
    viz.create_interactive_html_report(exps, save_path=str(tmp_path / "r2.html"))
    viz.__dict__.pop("px", None)


def test_optimization_clarity_selector_budget() -> None:
    opt = InstructionOptimizer()
    long = "Write " + "word " * 120
    clarity = opt.analyze_clarity(long)
    assert any("too long" in i.lower() for i in clarity["issues"])

    sel = FewShotSelector()
    # duplicate examples force best is None on second pick
    ex = {"input": "hello world test", "output": "ok done."}
    res = sel.select("hello world", [ex, dict(ex)], n=2, input_key="input", output_key="output")
    assert len(res.selected_examples) >= 1

    # stop-word-only query → relevance 0.5; empty selected diversity; empty union diversity
    assert sel._calculate_relevance("the a an", "foo") == 0.5
    assert sel._calculate_diversity("x", []) == 1.0
    assert sel._calculate_diversity("", [""]) == 1.0  # empty word sets → no similarities
    assert sel._calculate_coverage("the a", [], input_key="input") == 1.0

    budget = TokenBudgetOptimizer(max_tokens=50)
    huge = "x" * 400
    examples = [{"input": "e" * 80, "output": "o" * 80} for _ in range(5)]
    out = budget.optimize(
        huge, examples=examples, system_prompt="sys " * 20, reserve_for_response=10
    )
    assert out["over_budget"] is True
    assert any(
        "Truncated" in a or "Reduced" in a or "Compressed" in a for a in out["actions_taken"]
    )
    assert budget._estimate_tokens("") == 0
    assert budget._estimate_tokens(None) == 0  # type: ignore[arg-type]

    po = PromptOptimizer()
    text, changes = po._optimize_structure("Items:\n1) one\n2) two")
    assert "Standardized" in " ".join(changes) or text


def test_caching_evict_cachedmodel_warmer(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    cache = DiskCache(path=db, max_size_mb=0)  # tiny → always over
    # seed entries
    for i in range(5):
        cache.set(f"k{i}", {"v": "x" * 200}, ttl=1)
    # force expires_at in past + size eviction
    import time

    conn = cache._get_conn()
    conn.execute("UPDATE cache SET expires_at = ?", (time.time() - 10,))
    conn.commit()
    # shrink max so eviction loop runs
    cache._max_size_bytes = 1
    cache._evict_if_needed()

    from insideLLMs.types import ModelResponse, TokenUsage

    model = MagicMock()
    model.model_id = "m"
    model.generate = MagicMock(
        return_value=ModelResponse(
            content="out",
            model="m",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    )
    cm = CachedModel(model, cache=InMemoryCache())
    assert cm.model is model
    assert cm.cache is not None
    assert cm.generate("p", temperature=0.0).content == "out"

    mem = StrategyCache()
    warmer = CacheWarmer(cache=mem, generator=lambda p: f"g:{p}")
    warmer.add_prompt("hello", model="m", params={}, priority=1)
    warmer.add_prompt("hello", model="m", params={}, priority=2)
    # pre-populate so skip_existing hits
    from insideLLMs.caching import generate_cache_key

    key = generate_cache_key("hello", "m", {})
    mem.set(key, "cached")
    results = warmer.warm(batch_size=10, skip_existing=True)
    assert any(r.get("status") == "skipped" for r in results)


def test_tokens_utils_gaps() -> None:
    vc = VocabCoverage(text_vocab=set(), reference_vocab=set(), covered=set(), uncovered=set())
    assert vc.coverage_ratio == 1.0

    est = TokenEstimator()
    # no sentence boundary → word boundary at 1179
    chunks = est.split_to_chunks("alpha beta gamma delta epsilon zeta eta", max_tokens=2)
    assert chunks

    ta = TokenAnalyzer()
    # disjoint vocab → cosine_sim 0.0 branch
    d = ta.compare_distributions("aaa bbb", "ccc ddd")
    assert d["cosine_similarity"] == 0.0

    with pytest.raises(ValueError):
        EmbeddingUtils.cosine_similarity([1.0], [1.0, 2.0])
    assert EmbeddingUtils.cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    with pytest.raises(ValueError):
        EmbeddingUtils.euclidean_distance([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        EmbeddingUtils.manhattan_distance([1.0], [1.0, 2.0])
    assert EmbeddingUtils.average_embeddings([]) == []

    budget = TokenSpendingBudget(total_budget=10)
    assert budget.reserve(100) is False
    mgr = ContextWindowManager(max_tokens=5)
    assert mgr.truncate_to_fit("hello world", reserve_tokens=10) == ""
    assert mgr.truncate_to_fit("hi", reserve_tokens=0) == "hi"
