"""Analysis, cache, schema, safety, and workflow behavior."""

from __future__ import annotations

import importlib
import json
import sys
import types
from enum import Enum
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
from insideLLMs.models import DummyModel
from insideLLMs.optimization import (
    FewShotSelector,
    InstructionOptimizer,
    PromptOptimizer,
    TokenBudgetOptimizer,
)
from insideLLMs.probes.bias import BiasProbe
from insideLLMs.retry import (
    BackoffStrategy,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
    RetryConfig,
    execute_with_retry,
)
from insideLLMs.runtime import _high_level as hl
from insideLLMs.runtime import workflows as wf
from insideLLMs.runtime._ultimate import (
    _load_normalized_receipts_for_merkle,
    run_ultimate_post_artifact,
)
from insideLLMs.safety import (
    BiasDetector,
    ContentSafetyAnalyzer,
    RiskLevel,
    SafetyCategory,
    SafetyFlag,
    SafetyHallucinationIndicatorDetector,
    SafetyReport,
)
from insideLLMs.schemas.registry import SchemaRegistry, semver_tuple
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

    # CI omits the visualization extra; bind fakes when matplotlib/pandas absent.
    class _FakeDF:
        def __init__(self, data, **kwargs):
            self._rows = data if isinstance(data, list) else []

        def __getitem__(self, key):
            return [row.get(key) for row in self._rows if isinstance(row, dict)]

        def sort_values(self, *args, **kwargs):
            return self

    saved = {k: viz.__dict__.get(k) for k in ("plt", "pd", "sns")}
    fake_plt = MagicMock()
    viz.__dict__["plt"] = fake_plt
    viz.__dict__["pd"] = types.SimpleNamespace(DataFrame=_FakeDF)
    monkeypatch.setattr(viz, "MATPLOTLIB_AVAILABLE", True)
    monkeypatch.setattr(viz, "check_visualization_deps", lambda: None)

    try:
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
        monkeypatch.setattr(viz, "SEABORN_AVAILABLE", True)
        fake_sns = MagicMock()
        viz.__dict__["sns"] = fake_sns
        viz.plot_bias_results(bias, save_path=None)
        viz.plot_factuality_results(fact, save_path=None)
        assert fake_sns.barplot.called
        assert fake_sns.boxplot.called
    finally:
        for key, value in saved.items():
            if value is None:
                viz.__dict__.pop(key, None)
            else:
                viz.__dict__[key] = value


def test_viz_reload_optional_import_success() -> None:
    """Hit seaborn/plotly/ipywidgets import-success lines via reload with stubs."""
    import insideLLMs.analysis as analysis_pkg

    original_viz = sys.modules.get("insideLLMs.analysis.visualization")
    saved_keys = (
        "seaborn",
        "plotly",
        "plotly.express",
        "plotly.graph_objects",
        "plotly.subplots",
        "ipywidgets",
        "IPython",
        "IPython.display",
        "insideLLMs.analysis.visualization",
    )
    saved = {k: sys.modules[k] for k in saved_keys if k in sys.modules}

    def _restore() -> None:
        for key in saved_keys:
            sys.modules.pop(key, None)
        sys.modules.update(saved)
        if original_viz is not None:
            sys.modules["insideLLMs.analysis.visualization"] = original_viz
            analysis_pkg.visualization = original_viz
        elif "insideLLMs.analysis.visualization" not in sys.modules:
            importlib.import_module("insideLLMs.analysis.visualization")

    try:
        sns = types.ModuleType("seaborn")
        sns.barplot = MagicMock()
        sns.boxplot = MagicMock()
        sys.modules["seaborn"] = sns

        px = types.ModuleType("plotly.express")
        go = types.ModuleType("plotly.graph_objects")
        sub = types.ModuleType("plotly.subplots")
        plotly = types.ModuleType("plotly")
        plotly.express = px
        plotly.graph_objects = go
        plotly.subplots = sub
        sub.make_subplots = MagicMock()
        sys.modules["plotly"] = plotly
        sys.modules["plotly.express"] = px
        sys.modules["plotly.graph_objects"] = go
        sys.modules["plotly.subplots"] = sub

        ipy = types.ModuleType("ipywidgets")
        ipython_display = types.ModuleType("IPython.display")
        ipython = types.ModuleType("IPython")
        ipython.display = ipython_display
        ipython_display.display = MagicMock()
        sys.modules["ipywidgets"] = ipy
        sys.modules["IPython"] = ipython
        sys.modules["IPython.display"] = ipython_display

        for key in list(sys.modules):
            if key == "insideLLMs.analysis.visualization" or key.startswith(
                "insideLLMs.analysis.visualization."
            ):
                del sys.modules[key]

        import insideLLMs.analysis.visualization as viz

        assert viz.SEABORN_AVAILABLE is True
        assert viz.PLOTLY_AVAILABLE is True
        assert viz.IPYWIDGETS_AVAILABLE is True
    finally:
        _restore()


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

    # Avoid requiring the visualization extra (pandas) in CI.
    # Use __dict__ — setattr fails when the optional import left `pd` unbound.
    class _FakeDF:
        def __init__(self, data, **kwargs):
            self.data = data

        def melt(self, **kwargs):
            return self

        def __getitem__(self, key):
            return []

    saved = {k: viz.__dict__.get(k) for k in ("pd", "px")}
    try:
        viz.__dict__["pd"] = types.SimpleNamespace(DataFrame=_FakeDF)

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
    finally:
        for key, value in saved.items():
            if value is None:
                viz.__dict__.pop(key, None)
            else:
                viz.__dict__[key] = value


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


def test_safety_report_and_detectors() -> None:
    report = SafetyReport(
        text="x",
        is_safe=True,
        overall_risk=RiskLevel.LOW,
        flags=[],
        scores={},
    )
    assert report.get_highest_risk_flag() is None

    flags = [
        SafetyFlag(
            category=SafetyCategory.PII_EXPOSURE,
            risk_level=RiskLevel.LOW,
            description="l",
            confidence=0.1,
        ),
        SafetyFlag(
            category=SafetyCategory.TOXICITY,
            risk_level=RiskLevel.HIGH,
            description="h",
            confidence=0.9,
        ),
    ]
    report2 = SafetyReport(
        text="x", is_safe=False, overall_risk=RiskLevel.HIGH, flags=flags, scores={}
    )
    assert report2.get_highest_risk_flag().risk_level == RiskLevel.HIGH

    hd = SafetyHallucinationIndicatorDetector()
    assert hd.get_risk_level({"risk_score": 0.1}) == RiskLevel.LOW
    assert hd.get_risk_level({"risk_score": 0.3}) == RiskLevel.MEDIUM
    assert hd.get_risk_level({"risk_score": 0.5}) == RiskLevel.HIGH
    assert hd.get_risk_level({"risk_score": 0.9}) == RiskLevel.CRITICAL

    bd = BiasDetector()
    text_stereo = "All women are naturally better at nursing than men."
    matches = bd.analyze_stereotypes(text_stereo)
    assert isinstance(matches, list)
    unbalanced = "He he he he he he he. " + text_stereo
    analysis = bd.analyze(unbalanced)
    assert "bias_score" in analysis

    sa = ContentSafetyAnalyzer()
    hall_text = (
        "Studies show that 97% of experts agree this unverified claim is definitely true "
        "according to research that proves it without any doubt whatsoever."
    )
    full = sa.analyze(
        hall_text + " " + unbalanced,
        check_toxicity=True,
        check_hallucination=True,
        check_bias=True,
    )
    assert full.overall_risk in (
        RiskLevel.NONE,
        RiskLevel.LOW,
        RiskLevel.MEDIUM,
        RiskLevel.HIGH,
        RiskLevel.CRITICAL,
    )
    low_only = SafetyReport(
        text="t",
        is_safe=True,
        overall_risk=RiskLevel.LOW,
        flags=[
            SafetyFlag(
                category=SafetyCategory.PII_EXPOSURE,
                risk_level=RiskLevel.LOW,
                description="l",
                confidence=0.2,
            )
        ],
        scores={},
    )
    assert low_only.get_highest_risk_flag().risk_level == RiskLevel.LOW


def test_high_level_coerce_and_create_experiment() -> None:
    from insideLLMs.models import DummyModel
    from insideLLMs.probes.logic import LogicProbe

    class WeirdStatus(Enum):
        X = "not-a-real-status"

    class BadEnum(Enum):
        Y = object()

    model = DummyModel()
    probe = LogicProbe()
    results = [
        {"input": "a", "output": "b", "status": WeirdStatus.X, "error": None},
        {"input": "c", "output": "d", "status": "success", "error": None},
        {"input": "e", "output": "f", "status": BadEnum.Y, "error": None},
    ]
    exp = hl.create_experiment_result(
        model=model,
        probe=probe,
        results=results,
        experiment_id=None,
    )
    assert exp.experiment_id
    assert all(isinstance(r, ProbeResult) for r in exp.results)

    pr = [ProbeResult(input="i", output="o", status=ResultStatus.SUCCESS)]
    exp2 = hl.create_experiment_result(model=model, probe=probe, results=pr)
    assert exp2.results

    exp3 = hl.create_experiment_result(model=model, probe=probe, results=[])
    assert exp3.experiment_id


@pytest.mark.asyncio
async def test_run_probe_async_wrapper() -> None:
    from insideLLMs.models import DummyModel
    from insideLLMs.probes.logic import LogicProbe

    out = await hl.run_probe_async(DummyModel(), LogicProbe(), ["hello"])
    assert out


def test_workflows_guards(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        wf._coerce_path("   ", name="x")
    with pytest.raises(ValueError, match="validation_mode"):
        wf.run_harness_to_dir(cfg, tmp_path / "r", validation_mode="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="track"):
        wf.run_harness_to_dir(cfg, tmp_path / "r", track="nope")
    with pytest.raises(FileNotFoundError):
        wf.run_harness_to_dir(tmp_path / "missing.yaml", tmp_path / "r")
    with pytest.raises(ValueError, match="output_format"):
        wf.diff_run_dirs(tmp_path, tmp_path, output_format="xml")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limit"):
        wf.diff_run_dirs(tmp_path, tmp_path, limit=0)


def test_semantic_cache_cosine_and_redis_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover cosine fallback + RedisCache(client=) without reloading the module.

    Reloading semantic_cache poisons other suites that hold RedisCache class
    references from the original module object (patch targets the new module).
    """
    import insideLLMs.semantic_cache as sc

    # numpy path (zero-norm + happy)
    fake_np = types.SimpleNamespace(
        array=lambda x: x,
        dot=lambda a, b: sum(i * j for i, j in zip(a, b)),
        linalg=types.SimpleNamespace(norm=lambda a: 0.0 if not any(a) else 1.0),
    )
    monkeypatch.setattr(sc, "NUMPY_AVAILABLE", True)
    monkeypatch.setattr(sc, "np", fake_np)
    assert sc.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert sc.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    # pure-python path
    monkeypatch.setattr(sc, "NUMPY_AVAILABLE", False)
    assert sc.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert sc.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    monkeypatch.setattr(sc, "REDIS_AVAILABLE", True)
    monkeypatch.setattr(sc, "redis", types.SimpleNamespace(Redis=MagicMock))
    client = MagicMock()
    cache = sc.RedisCache(client=client)
    assert cache._client is client
    # no-client path constructs redis.Redis(...)
    cache2 = sc.RedisCache()
    assert cache2._client is not None


def test_retry_else_backoff_and_circuit_half_open() -> None:
    cfg = RetryConfig(max_retries=0, strategy=BackoffStrategy.CONSTANT)
    # force unknown strategy via monkeypatch on instance
    cfg.strategy = "unknown"  # type: ignore[assignment]
    assert cfg.calculate_delay(1) >= 0

    cb = CircuitBreaker(
        name="t",
        config=CircuitBreakerConfig(failure_threshold=1, reset_timeout=0.01, half_open_max_calls=1),
    )
    # trip open
    with pytest.raises(RuntimeError):
        cb.execute(lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert cb._state == CircuitState.OPEN
    # force half-open
    cb._state = CircuitState.HALF_OPEN
    cb._half_open_calls = 0
    with pytest.raises(RuntimeError):
        cb.execute(lambda: (_ for _ in ()).throw(RuntimeError("y")))
    assert cb._state == CircuitState.OPEN

    cb2 = CircuitBreaker(
        name="t2",
        config=CircuitBreakerConfig(failure_threshold=5, reset_timeout=60, half_open_max_calls=1),
    )
    cb2._state = CircuitState.HALF_OPEN
    cb2._half_open_calls = 1
    with pytest.raises(CircuitBreakerOpen):
        cb2.execute(lambda: 1)


def test_schema_registry_edges() -> None:
    assert semver_tuple("x.y.z") == (0, 0, 0)

    reg = SchemaRegistry()
    with pytest.raises(KeyError):
        reg.get_model(reg.CUSTOM_TRACE, schema_version="bad@version")

    migrated = reg.migrate(
        reg.RUN_MANIFEST,
        {"schema_version": "1.0.0", "run_id": "r"},
        from_version="1.0.0",
        to_version="1.0.1",
        custom_migration=lambda d: {**d, "extra": 1},
    )
    assert migrated["schema_version"] == "1.0.1"
    assert migrated["run_completed"] is False
    assert migrated["extra"] == 1


def test_bias_probe_dict_pair_shapes() -> None:
    probe = BiasProbe()
    model = DummyModel()

    r1 = probe.run(model, {"pairs": [("a", "b")]})
    assert r1
    r2 = probe.run(model, {"prompt_pairs": [("c", "d")]})
    assert r2
    r3 = probe.run(model, {"prompt_a": "x", "prompt_b": "y"})
    assert r3
    r4 = probe.run(model, [{"a": "p", "b": "q"}])
    assert r4

    with pytest.raises(ValueError, match="prompt_a"):
        probe.run(model, {"foo": 1})
    with pytest.raises(ValueError, match="list of"):
        probe.run(model, "not-pairs")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pair dict"):
        probe.run(model, [{"prompt_a": "only"}])
    with pytest.raises(ValueError, match="2-item"):
        probe.run(model, [("only",)])


def test_ultimate_receipts_and_provided_roots(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts" / "calls.jsonl"
    receipts.parent.mkdir(parents=True)
    receipts.write_text(
        "\n"
        + json.dumps({"id": "1", "latency_ms": 12.0})
        + "\n\n"
        + json.dumps({"id": "2"})
        + "\n",
        encoding="utf-8",
    )
    loaded = _load_normalized_receipts_for_merkle(receipts)
    assert loaded[0]["latency_ms"] is None

    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_text(json.dumps({"a": 1}) + "\n", encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"run_id": "r"}), encoding="utf-8")
    run_ultimate_post_artifact(
        run,
        records_merkle_root="aa" * 32,
        receipts_merkle_root="bb" * 32,
        dataset_merkle_root="cc" * 32,
        promptset_merkle_root="dd" * 32,
        insidellms_version="0.0.0",
    )
    assert (run / "integrity" / "records.merkle.json").exists()
    assert (run / "integrity" / "dataset.merkle.json").exists()


def test_execute_with_retry_non_retryable() -> None:
    def boom():
        raise ValueError("no")

    with pytest.raises(ValueError):
        execute_with_retry(boom, (), {}, RetryConfig())
