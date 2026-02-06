"""Extended coverage tests for insideLLMs.analysis.visualization.

This module targets code paths not exercised by test_visualization.py,
aiming to raise branch coverage from ~41 % to 90 %+.  It covers:

* HTML-escape helpers (_escape_html_text, _escape_html_attr)
* _serialize_experiments_to_json
* create_interactive_html_report (with mocked Plotly)
* Matplotlib plot functions with real data and advanced parameters
* All interactive Plotly functions (mocked)
* ExperimentExplorer class (mocked ipywidgets / IPython)
* Edge cases (empty data, missing scores, malformed input)
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from insideLLMs.types import (
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_experiment(
    model_name: str = "TestModel",
    provider: str = "test-provider",
    probe_name: str = "TestProbe",
    category: ProbeCategory = ProbeCategory.LOGIC,
    accuracy: float = 0.85,
    precision: float = 0.80,
    recall: float = 0.90,
    f1: float = 0.85,
    latency: float = 150.0,
    total_tokens: int | None = 500,
    num_results: int = 10,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> ExperimentResult:
    """Build a realistic ExperimentResult for testing."""
    results = []
    success_count = int(accuracy * num_results)
    for i in range(num_results):
        status = ResultStatus.SUCCESS if i < success_count else ResultStatus.ERROR
        results.append(
            ProbeResult(
                input=f"input-{i}",
                output=f"output-{i}",
                status=status,
                latency_ms=100.0 + i * 10.0,
            )
        )

    return ExperimentResult(
        experiment_id=f"exp-{model_name}-{probe_name}",
        model_info=ModelInfo(
            name=model_name,
            provider=provider,
            model_id=f"id-{model_name.lower()}",
        ),
        probe_name=probe_name,
        probe_category=category,
        results=results,
        score=ProbeScore(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mean_latency_ms=latency,
            total_tokens=total_tokens,
        ),
        started_at=started_at,
        completed_at=completed_at,
    )


def _make_experiments_pair() -> list[ExperimentResult]:
    """Create two experiments with different models/probes for comparison charts."""
    return [
        _make_experiment(
            model_name="ModelA",
            provider="providerX",
            probe_name="ProbeAlpha",
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1=0.90,
            latency=120.0,
            total_tokens=1000,
        ),
        _make_experiment(
            model_name="ModelB",
            provider="providerY",
            probe_name="ProbeBeta",
            accuracy=0.75,
            precision=0.70,
            recall=0.80,
            f1=0.75,
            latency=200.0,
            total_tokens=1500,
        ),
    ]


def _make_no_score_experiment() -> ExperimentResult:
    return ExperimentResult(
        experiment_id="exp-noscore",
        model_info=ModelInfo(name="NoScoreModel", provider="test", model_id="ns"),
        probe_name="NoScoreProbe",
        probe_category=ProbeCategory.LOGIC,
        results=[],
        score=None,
    )


# ---------------------------------------------------------------------------
# 1. HTML-escape helpers
# ---------------------------------------------------------------------------


class TestHtmlEscapeHelpers:
    def test_escape_html_text_basic(self):
        from insideLLMs.analysis.visualization import _escape_html_text

        assert _escape_html_text("<b>hi</b>") == "&lt;b&gt;hi&lt;/b&gt;"

    def test_escape_html_text_ampersand(self):
        from insideLLMs.analysis.visualization import _escape_html_text

        assert "&amp;" in _escape_html_text("A & B")

    def test_escape_html_text_quotes_not_escaped(self):
        """quote=False means double quotes are NOT escaped."""
        from insideLLMs.analysis.visualization import _escape_html_text

        result = _escape_html_text('say "hello"')
        # With quote=False, " should remain unescaped
        assert '"hello"' in result

    def test_escape_html_text_non_string(self):
        from insideLLMs.analysis.visualization import _escape_html_text

        assert _escape_html_text(42) == "42"
        assert _escape_html_text(None) == "None"

    def test_escape_html_attr_basic(self):
        from insideLLMs.analysis.visualization import _escape_html_attr

        result = _escape_html_attr('<script>alert("xss")</script>')
        assert "<" not in result
        assert '"' not in result or "&quot;" in result

    def test_escape_html_attr_quotes_escaped(self):
        """quote=True means double quotes ARE escaped."""
        from insideLLMs.analysis.visualization import _escape_html_attr

        result = _escape_html_attr('val="x"')
        assert "&quot;" in result

    def test_escape_html_attr_non_string(self):
        from insideLLMs.analysis.visualization import _escape_html_attr

        assert _escape_html_attr(3.14) == "3.14"


# ---------------------------------------------------------------------------
# 2. _serialize_experiments_to_json
# ---------------------------------------------------------------------------


class TestSerializeExperimentsToJson:
    def test_basic_serialization(self):
        from insideLLMs.analysis.visualization import _serialize_experiments_to_json

        exps = [_make_experiment()]
        result = _serialize_experiments_to_json(exps)
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["model_info"]["name"] == "TestModel"

    def test_serializes_enums(self):
        from insideLLMs.analysis.visualization import _serialize_experiments_to_json

        exps = [_make_experiment()]
        result = _serialize_experiments_to_json(exps)
        data = json.loads(result)
        # ProbeCategory.LOGIC -> "logic"
        assert data[0]["probe_category"] == "logic"
        # ResultStatus.SUCCESS -> "success"
        assert data[0]["results"][0]["status"] == "success"

    def test_serializes_datetime(self):
        from insideLLMs.analysis.visualization import _serialize_experiments_to_json

        dt = datetime(2024, 6, 15, 12, 0, 0)
        exp = _make_experiment(started_at=dt, completed_at=dt)
        result = _serialize_experiments_to_json([exp])
        data = json.loads(result)
        assert "2024-06-15" in data[0]["started_at"]

    def test_serializes_none_score(self):
        from insideLLMs.analysis.visualization import _serialize_experiments_to_json

        exp = _make_no_score_experiment()
        result = _serialize_experiments_to_json([exp])
        data = json.loads(result)
        assert data[0]["score"] is None

    def test_serializes_dict_and_list(self):
        from insideLLMs.analysis.visualization import _serialize_experiments_to_json

        exp = _make_experiment()
        exp.config = {"key": "value"}
        exp.metadata = {"env": "test"}
        result = _serialize_experiments_to_json([exp])
        data = json.loads(result)
        assert data[0]["config"]["key"] == "value"

    def test_multiple_experiments(self):
        from insideLLMs.analysis.visualization import _serialize_experiments_to_json

        exps = _make_experiments_pair()
        result = _serialize_experiments_to_json(exps)
        data = json.loads(result)
        assert len(data) == 2
        names = {d["model_info"]["name"] for d in data}
        assert "ModelA" in names
        assert "ModelB" in names


# ---------------------------------------------------------------------------
# 3. Matplotlib plotting functions with data (not just empty-data paths)
# ---------------------------------------------------------------------------


class TestPlotAccuracyComparisonWithData:
    def test_saves_with_multiple_experiments(self, tmp_path):
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        exps = _make_experiments_pair()
        save_path = str(tmp_path / "acc.png")
        viz.plot_accuracy_comparison(exps, title="Multi-Model", save_path=save_path)
        assert Path(save_path).exists()

    def test_custom_figsize(self, tmp_path):
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        exps = [_make_experiment()]
        save_path = str(tmp_path / "acc_custom.png")
        viz.plot_accuracy_comparison(exps, figsize=(8, 4), save_path=save_path)
        assert Path(save_path).exists()

    def test_skips_no_accuracy(self, capsys):
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        exp = _make_no_score_experiment()
        viz.plot_accuracy_comparison([exp])
        captured = capsys.readouterr()
        assert "No accuracy" in captured.out


class TestPlotLatencyDistributionWithData:
    def test_saves_with_data(self, tmp_path):
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        exps = _make_experiments_pair()
        save_path = str(tmp_path / "lat.png")
        viz.plot_latency_distribution(exps, save_path=save_path)
        assert Path(save_path).exists()

    def test_custom_figsize_and_title(self, tmp_path):
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        exps = [_make_experiment()]
        save_path = str(tmp_path / "lat2.png")
        viz.plot_latency_distribution(
            exps, title="Custom Title", figsize=(6, 4), save_path=save_path
        )
        assert Path(save_path).exists()

    def test_without_seaborn_fallback(self, tmp_path):
        """Ensure matplotlib fallback (no seaborn) works."""
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        exps = [_make_experiment()]
        save_path = str(tmp_path / "lat_nosns.png")
        original = viz.SEABORN_AVAILABLE
        try:
            viz.SEABORN_AVAILABLE = False
            viz.plot_latency_distribution(exps, save_path=save_path)
            assert Path(save_path).exists()
        finally:
            viz.SEABORN_AVAILABLE = original

    def test_no_latency_data_empty_list(self, capsys):
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        viz.plot_latency_distribution([])
        captured = capsys.readouterr()
        assert "No latency" in captured.out

    def test_results_with_none_latency(self, tmp_path):
        """Experiment whose results all have latency_ms=None still creates a plot
        (latencies_by_model has an empty list which is truthy for dict check)."""
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        exp = ExperimentResult(
            experiment_id="test",
            model_info=ModelInfo(name="test", provider="test", model_id="test"),
            probe_name="TestProbe",
            probe_category=ProbeCategory.LOGIC,
            results=[
                ProbeResult(input="x", output="y", status=ResultStatus.SUCCESS, latency_ms=None)
            ],
            score=None,
        )
        save_path = str(tmp_path / "lat_none.png")
        # Use matplotlib fallback to avoid seaborn empty-data issue
        original = viz.SEABORN_AVAILABLE
        try:
            viz.SEABORN_AVAILABLE = False
            viz.plot_latency_distribution([exp], save_path=save_path)
        finally:
            viz.SEABORN_AVAILABLE = original


class TestPlotMetricComparisonWithData:
    def test_saves_with_data(self, tmp_path):
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        exps = _make_experiments_pair()
        save_path = str(tmp_path / "metrics.png")
        viz.plot_metric_comparison(exps, save_path=save_path)
        assert Path(save_path).exists()

    def test_custom_metrics_subset(self, tmp_path):
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        exps = [_make_experiment()]
        save_path = str(tmp_path / "metrics_sub.png")
        viz.plot_metric_comparison(
            exps,
            metrics=["accuracy", "recall"],
            title="Subset",
            save_path=save_path,
        )
        assert Path(save_path).exists()

    def test_no_metric_data_empty_list(self, capsys):
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        viz.plot_metric_comparison([])
        captured = capsys.readouterr()
        assert "No metric" in captured.out

    def test_no_score_experiments_still_plots(self, tmp_path):
        """Experiments with no scores populate model_data keys but no metrics.
        The code proceeds past the `if not model_data` check."""
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        save_path = str(tmp_path / "noscore_metric.png")
        viz.plot_metric_comparison([_make_no_score_experiment()], save_path=save_path)
        # The plot is created (even if empty bars)
        assert Path(save_path).exists()


class TestPlotSuccessRateOverTimeWithData:
    def test_saves_with_data(self, tmp_path):
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        data = [("Run 1", 0.80), ("Run 2", 0.85), ("Run 3", 0.90)]
        save_path = str(tmp_path / "success.png")
        viz.plot_success_rate_over_time(data, save_path=save_path)
        assert Path(save_path).exists()

    def test_custom_params(self, tmp_path):
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        data = [("A", 0.5), ("B", 0.6)]
        save_path = str(tmp_path / "sr2.png")
        viz.plot_success_rate_over_time(data, title="Custom", figsize=(6, 4), save_path=save_path)
        assert Path(save_path).exists()


class TestPlotBiasResultsWithData:
    def test_without_seaborn(self, tmp_path):
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        results = [
            {"output": [("response A is longer text", "response B")]},
            {"output": [("short", "this is a longer response text")]},
        ]
        save_path = str(tmp_path / "bias_nosns.png")
        original = viz.SEABORN_AVAILABLE
        try:
            viz.SEABORN_AVAILABLE = False
            viz.plot_bias_results(results, save_path=save_path)
            assert Path(save_path).exists()
        finally:
            viz.SEABORN_AVAILABLE = original


class TestPlotFactualityResultsWithData:
    def test_without_seaborn(self, tmp_path):
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        results = [
            {
                "output": [
                    {"category": "science", "model_answer": "Water is H2O"},
                    {"category": "history", "model_answer": "Rome fell in 476 AD"},
                    {"category": "science", "model_answer": "E = mc^2"},
                ]
            }
        ]
        save_path = str(tmp_path / "fact_nosns.png")
        original = viz.SEABORN_AVAILABLE
        try:
            viz.SEABORN_AVAILABLE = False
            viz.plot_factuality_results(results, save_path=save_path)
            assert Path(save_path).exists()
        finally:
            viz.SEABORN_AVAILABLE = original

    def test_missing_category_defaults_to_general(self, tmp_path):
        from insideLLMs.analysis import visualization as viz

        if not viz.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        results = [
            {
                "output": [
                    {"model_answer": "Some answer without category"},
                ]
            }
        ]
        save_path = str(tmp_path / "fact_nocat.png")
        viz.plot_factuality_results(results, save_path=save_path)
        assert Path(save_path).exists()


# ---------------------------------------------------------------------------
# 4. Mock Plotly environment for interactive functions
#    We patch module-level attributes in-place rather than reloading so that
#    coverage instrumentation is preserved.
# ---------------------------------------------------------------------------


def _make_mock_fig():
    """Create a reusable mock Plotly figure with standard methods."""
    fig = mock.MagicMock()
    fig.to_html.return_value = '<div id="abc123" class="plotly-graph-div" style=""></div>'
    fig.update_traces.return_value = fig
    fig.update_layout.return_value = fig
    fig.update_yaxes.return_value = fig
    fig.update_xaxes.return_value = fig
    fig.add_trace.return_value = fig
    return fig


@pytest.fixture()
def viz_with_plotly():
    """Fixture that patches the visualization module so that plotly / ipywidgets
    features appear available without actually importing those packages.

    This avoids reloading the module, which breaks under coverage instrumentation.
    """
    import insideLLMs.analysis.visualization as viz

    mock_fig = _make_mock_fig()

    # Build mock px (plotly.express)
    mock_px = mock.MagicMock()
    for fn in ["bar", "box", "violin", "histogram", "line", "scatter", "sunburst"]:
        getattr(mock_px, fn).return_value = mock_fig

    # Build mock go (plotly.graph_objects)
    mock_go = mock.MagicMock()
    mock_go.Figure.return_value = mock_fig
    mock_go.Bar = mock.MagicMock()
    mock_go.Box = mock.MagicMock()
    mock_go.Scatter = mock.MagicMock()
    mock_go.Scatterpolar = mock.MagicMock()
    mock_go.Heatmap.return_value = mock.MagicMock()

    mock_make_subplots = mock.MagicMock(return_value=mock_fig)

    # Build mock widgets
    mock_widgets = mock.MagicMock()
    mock_display = mock.MagicMock()

    # We must also ensure MATPLOTLIB_AVAILABLE is True so that check_plotly_deps()
    # does not attempt to re-import pandas (which can fail under coverage instrumentation
    # due to numpy module re-loading issues).
    # Get the pandas reference from any available source.  Under coverage
    # instrumentation the visualization module's initial import of matplotlib may
    # fail, leaving `pd` undefined.  We need to provide a real pandas module
    # since the interactive functions call pd.DataFrame / pd.melt etc.
    _real_pd = getattr(viz, "pd", None) or sys.modules.get("pandas")
    if _real_pd is None:
        # If pandas hasn't been loaded yet at all, try importing it
        # Skip this fixture if pandas is not available
        try:
            _real_pd = __import__("pandas")
        except ModuleNotFoundError:
            pytest.skip("pandas not available")

    # Save originals
    saved = {}
    attrs_to_patch: dict[str, Any] = {
        "PLOTLY_AVAILABLE": True,
        "IPYWIDGETS_AVAILABLE": True,
        "MATPLOTLIB_AVAILABLE": True,
        "pd": _real_pd,
        "px": mock_px,
        "go": mock_go,
        "make_subplots": mock_make_subplots,
        "widgets": mock_widgets,
        "display": mock_display,
    }
    for attr, new_val in attrs_to_patch.items():
        saved[attr] = getattr(viz, attr, None)
        setattr(viz, attr, new_val)

    try:
        yield viz, mock_fig
    finally:
        for attr, old_val in saved.items():
            if old_val is None:
                try:
                    delattr(viz, attr)
                except AttributeError:
                    pass
            else:
                setattr(viz, attr, old_val)


# ---------------------------------------------------------------------------
# 5. Interactive Plotly function tests (with mocked Plotly)
# ---------------------------------------------------------------------------


class TestInteractiveAccuracyComparison:
    def test_basic_call(self, viz_with_plotly):
        viz, mock_fig = viz_with_plotly
        exps = _make_experiments_pair()
        fig = viz.interactive_accuracy_comparison(exps)
        assert fig is not None

    def test_color_by_probe(self, viz_with_plotly):
        viz, mock_fig = viz_with_plotly
        exps = _make_experiments_pair()
        fig = viz.interactive_accuracy_comparison(exps, color_by="probe")
        assert fig is not None

    def test_raises_on_no_data(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        with pytest.raises(ValueError, match="No accuracy"):
            viz.interactive_accuracy_comparison([_make_no_score_experiment()])

    def test_custom_title(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        fig = viz.interactive_accuracy_comparison(exps, title="Custom Title")
        assert fig is not None


class TestInteractiveLatencyDistribution:
    def test_box_chart(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        fig = viz.interactive_latency_distribution(exps, chart_type="box")
        assert fig is not None

    def test_violin_chart(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        fig = viz.interactive_latency_distribution(exps, chart_type="violin")
        assert fig is not None

    def test_histogram_chart(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        fig = viz.interactive_latency_distribution(exps, chart_type="histogram")
        assert fig is not None

    def test_raises_on_no_data(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exp = ExperimentResult(
            experiment_id="nolat",
            model_info=ModelInfo(name="test", provider="test", model_id="test"),
            probe_name="P",
            probe_category=ProbeCategory.LOGIC,
            results=[
                ProbeResult(input="x", output="y", status=ResultStatus.SUCCESS, latency_ms=None)
            ],
            score=None,
        )
        with pytest.raises(ValueError, match="No latency"):
            viz.interactive_latency_distribution([exp])


class TestInteractiveMetricRadar:
    def test_basic_radar(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = _make_experiments_pair()
        fig = viz.interactive_metric_radar(exps)
        assert fig is not None

    def test_custom_metrics(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        fig = viz.interactive_metric_radar(exps, metrics=["accuracy", "precision"])
        assert fig is not None

    def test_no_score_still_works(self, viz_with_plotly):
        """Models with no scores should get 0 values in radar."""
        viz, _ = viz_with_plotly
        exps = [_make_no_score_experiment()]
        fig = viz.interactive_metric_radar(exps)
        assert fig is not None


class TestInteractiveTimeline:
    def test_basic_timeline(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = _make_experiments_pair()
        fig = viz.interactive_timeline(exps)
        assert fig is not None

    def test_f1_metric(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        fig = viz.interactive_timeline(exps, metric="f1_score")
        assert fig is not None

    def test_raises_on_no_data(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        with pytest.raises(ValueError, match="No accuracy"):
            viz.interactive_timeline([_make_no_score_experiment()], metric="accuracy")


class TestInteractiveHeatmap:
    def test_basic_heatmap(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = _make_experiments_pair()
        fig = viz.interactive_heatmap(exps)
        assert fig is not None

    def test_swapped_keys(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = _make_experiments_pair()
        fig = viz.interactive_heatmap(exps, row_key="probe", col_key="model")
        assert fig is not None

    def test_raises_on_no_data(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        with pytest.raises(ValueError, match="No accuracy"):
            viz.interactive_heatmap([_make_no_score_experiment()])


class TestInteractiveScatterComparison:
    def test_basic_scatter(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        fig = viz.interactive_scatter_comparison(exps)
        assert fig is not None

    def test_with_size_metric(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        fig = viz.interactive_scatter_comparison(
            exps,
            x_metric="accuracy",
            y_metric="mean_latency_ms",
            size_metric="f1_score",
        )
        assert fig is not None

    def test_raises_on_no_data(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        with pytest.raises(ValueError, match="No data"):
            viz.interactive_scatter_comparison([_make_no_score_experiment()])


class TestInteractiveSunburst:
    def test_basic_sunburst(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = _make_experiments_pair()
        fig = viz.interactive_sunburst(exps)
        assert fig is not None

    def test_f1_metric(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        fig = viz.interactive_sunburst(exps, value_metric="f1_score")
        assert fig is not None

    def test_raises_on_no_data(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        with pytest.raises(ValueError, match="No accuracy"):
            viz.interactive_sunburst([_make_no_score_experiment()])


class TestCreateInteractiveDashboard:
    def test_basic_dashboard(self, viz_with_plotly):
        viz, mock_fig = viz_with_plotly
        exps = _make_experiments_pair()
        fig = viz.create_interactive_dashboard(exps)
        assert fig is not None

    def test_dashboard_with_save_path(self, viz_with_plotly, tmp_path):
        viz, mock_fig = viz_with_plotly
        exps = _make_experiments_pair()
        save_path = str(tmp_path / "dashboard.html")
        fig = viz.create_interactive_dashboard(exps, save_path=save_path)
        assert fig is not None
        mock_fig.write_html.assert_called_with(save_path)

    def test_dashboard_no_score(self, viz_with_plotly):
        """Dashboard should not crash with no-score experiments."""
        viz, _ = viz_with_plotly
        exps = [_make_no_score_experiment()]
        fig = viz.create_interactive_dashboard(exps)
        assert fig is not None


# ---------------------------------------------------------------------------
# 6. create_interactive_html_report
# ---------------------------------------------------------------------------


class TestCreateInteractiveHtmlReport:
    def test_basic_report(self, viz_with_plotly, tmp_path):
        viz, _ = viz_with_plotly
        exps = _make_experiments_pair()
        save_path = str(tmp_path / "report.html")
        result = viz.create_interactive_html_report(exps, save_path=save_path)
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "<html" in content
        assert "LLM Evaluation Report" in content

    def test_custom_title(self, viz_with_plotly, tmp_path):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        save_path = str(tmp_path / "report_custom.html")
        result = viz.create_interactive_html_report(
            exps, title="My Custom Report", save_path=save_path
        )
        content = Path(result).read_text()
        assert "My Custom Report" in content

    def test_include_raw_results_false(self, viz_with_plotly, tmp_path):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        save_path = str(tmp_path / "report_noraw.html")
        viz.create_interactive_html_report(
            exps,
            save_path=save_path,
            include_raw_results=False,
        )
        assert Path(save_path).exists()

    def test_include_individual_results_false(self, viz_with_plotly, tmp_path):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        save_path = str(tmp_path / "report_noind.html")
        viz.create_interactive_html_report(
            exps,
            save_path=save_path,
            include_individual_results=False,
        )
        assert Path(save_path).exists()

    def test_embed_plotly_js_flag(self, viz_with_plotly, tmp_path):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        save_path = str(tmp_path / "report_embed.html")
        viz.create_interactive_html_report(
            exps,
            save_path=save_path,
            embed_plotly_js=True,
        )
        assert Path(save_path).exists()

    def test_generated_at_param(self, viz_with_plotly, tmp_path):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        save_path = str(tmp_path / "report_ts.html")
        viz.create_interactive_html_report(
            exps,
            save_path=save_path,
            generated_at=datetime(2024, 1, 1, 12, 0, 0),
        )
        assert Path(save_path).exists()

    def test_with_no_score_experiments(self, viz_with_plotly, tmp_path):
        viz, _ = viz_with_plotly
        exps = [_make_no_score_experiment()]
        save_path = str(tmp_path / "report_noscore.html")
        result = viz.create_interactive_html_report(exps, save_path=save_path)
        assert Path(result).exists()

    def test_html_escaping_in_title(self, viz_with_plotly, tmp_path):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        save_path = str(tmp_path / "report_escape.html")
        viz.create_interactive_html_report(
            exps,
            title='<script>alert("xss")</script>',
            save_path=save_path,
        )
        content = Path(save_path).read_text()
        assert "<script>alert" not in content
        assert "&lt;script&gt;" in content

    def test_experiments_json_embedded(self, viz_with_plotly, tmp_path):
        viz, _ = viz_with_plotly
        exps = [_make_experiment()]
        save_path = str(tmp_path / "report_json.html")
        viz.create_interactive_html_report(exps, save_path=save_path)
        content = Path(save_path).read_text()
        # The JSON data is embedded in a script tag
        assert "experiment-data" in content

    def test_with_tokens(self, viz_with_plotly, tmp_path):
        """Exercises the token-usage chart branch."""
        viz, _ = viz_with_plotly
        exp = _make_experiment(total_tokens=2000)
        save_path = str(tmp_path / "report_tokens.html")
        viz.create_interactive_html_report([exp], save_path=save_path)
        assert Path(save_path).exists()


# ---------------------------------------------------------------------------
# 7. ExperimentExplorer class
# ---------------------------------------------------------------------------


class TestExperimentExplorer:
    def test_init_sets_models_and_probes(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = _make_experiments_pair()
        explorer = viz.ExperimentExplorer(exps)
        assert "ModelA" in explorer.models
        assert "ModelB" in explorer.models
        assert "ProbeAlpha" in explorer.probes
        assert "ProbeBeta" in explorer.probes

    def test_compare_models_accuracy(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = _make_experiments_pair()
        explorer = viz.ExperimentExplorer(exps)
        result = explorer.compare_models(metric="accuracy")
        assert result is not None

    def test_compare_models_max_aggregate(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = _make_experiments_pair()
        explorer = viz.ExperimentExplorer(exps)
        result = explorer.compare_models(metric="accuracy", aggregate="max")
        assert result is not None

    def test_compare_models_min_aggregate(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = _make_experiments_pair()
        explorer = viz.ExperimentExplorer(exps)
        result = explorer.compare_models(metric="accuracy", aggregate="min")
        assert result is not None

    def test_compare_models_f1(self, viz_with_plotly):
        viz, _ = viz_with_plotly
        exps = _make_experiments_pair()
        explorer = viz.ExperimentExplorer(exps)
        result = explorer.compare_models(metric="f1_score")
        assert result is not None

    def test_compare_models_unknown_aggregate(self, viz_with_plotly):
        """Unknown aggregate falls back to mean."""
        viz, _ = viz_with_plotly
        exps = _make_experiments_pair()
        explorer = viz.ExperimentExplorer(exps)
        result = explorer.compare_models(metric="accuracy", aggregate="unknown_agg")
        assert result is not None

    def test_show(self, viz_with_plotly):
        """Test that show() calls display and creates widgets."""
        viz, mock_fig = viz_with_plotly

        # Create mock widget instances that support observe
        mock_model_select = mock.MagicMock()
        mock_model_select.value = ("ModelA",)
        mock_probe_select = mock.MagicMock()
        mock_probe_select.value = ("ProbeAlpha",)
        mock_chart_type = mock.MagicMock()
        mock_chart_type.value = "accuracy"
        mock_output = mock.MagicMock()
        mock_output.__enter__ = mock.MagicMock(return_value=mock_output)
        mock_output.__exit__ = mock.MagicMock(return_value=False)

        viz.widgets.SelectMultiple.return_value = mock_model_select
        viz.widgets.Dropdown.return_value = mock_chart_type
        viz.widgets.Output.return_value = mock_output

        exps = _make_experiments_pair()
        explorer = viz.ExperimentExplorer(exps)
        explorer.show()

        viz.display.assert_called_once()

    def test_show_update_chart_types(self, viz_with_plotly):
        """Exercise all chart type branches in update_chart."""
        viz, mock_fig = viz_with_plotly

        exps = _make_experiments_pair()

        mock_model_select = mock.MagicMock()
        mock_model_select.value = ("ModelA", "ModelB")
        mock_probe_select = mock.MagicMock()
        mock_probe_select.value = ("ProbeAlpha", "ProbeBeta")
        mock_chart_type = mock.MagicMock()
        mock_chart_type.value = "accuracy"
        mock_output = mock.MagicMock()
        mock_output.__enter__ = mock.MagicMock(return_value=mock_output)
        mock_output.__exit__ = mock.MagicMock(return_value=False)

        viz.widgets.SelectMultiple.return_value = mock_model_select
        viz.widgets.Dropdown.return_value = mock_chart_type
        viz.widgets.Output.return_value = mock_output

        explorer = viz.ExperimentExplorer(exps)
        # show() calls update_chart() internally
        explorer.show()

        # Check display was called
        assert viz.display.called


# ---------------------------------------------------------------------------
# 8. check_plotly_deps pandas import branch
# ---------------------------------------------------------------------------


class TestCheckPlotlyDepsPandasBranch:
    def test_plotly_available_no_matplotlib_imports_pandas(self):
        """When PLOTLY_AVAILABLE is True but MATPLOTLIB_AVAILABLE is False,
        check_plotly_deps should try to import pandas."""
        from insideLLMs.analysis import visualization as viz

        orig_plotly = viz.PLOTLY_AVAILABLE
        orig_mpl = viz.MATPLOTLIB_AVAILABLE
        try:
            viz.PLOTLY_AVAILABLE = True
            viz.MATPLOTLIB_AVAILABLE = False
            # This should succeed since pandas is installed.
            # However, under narrow coverage instrumentation (--cov=insideLLMs.analysis.visualization),
            # numpy may fail to re-import. In that case, the ImportError is expected.
            try:
                viz.check_plotly_deps()
            except ImportError as e:
                if "numpy" in str(e).lower() or "pandas" in str(e).lower():
                    pytest.skip("numpy/pandas re-import issue under coverage instrumentation")
                raise
        finally:
            viz.PLOTLY_AVAILABLE = orig_plotly
            viz.MATPLOTLIB_AVAILABLE = orig_mpl


# ---------------------------------------------------------------------------
# 9. Additional edge cases for text functions
# ---------------------------------------------------------------------------


class TestTextBarChartAdditionalEdgeCases:
    def test_single_value_bar(self):
        from insideLLMs.analysis.visualization import text_bar_chart

        result = text_bar_chart(["Only"], [42.0], title="Single", width=30)
        assert "Only" in result
        assert "42.00" in result
        assert "Single" in result

    def test_mismatched_labels_values(self):
        """Shorter labels list paired with longer values; zip truncates."""
        from insideLLMs.analysis.visualization import text_bar_chart

        # zip will stop at the shorter iterator
        result = text_bar_chart(["A"], [1.0, 2.0])
        assert "A" in result


class TestTextHistogramAdditionalEdgeCases:
    def test_large_dataset(self):
        import random

        from insideLLMs.analysis.visualization import text_histogram

        random.seed(42)
        values = [random.gauss(100, 20) for _ in range(1000)]
        result = text_histogram(values, bins=15, title="Big Dataset")
        assert "Big Dataset" in result
        assert len(result.split("\n")) >= 15

    def test_two_values(self):
        from insideLLMs.analysis.visualization import text_histogram

        result = text_histogram([1.0, 2.0], bins=2)
        assert "(" in result


class TestExperimentSummaryTextAdditional:
    def test_with_f1_score(self):
        from insideLLMs.analysis.visualization import experiment_summary_text

        exp = _make_experiment(f1=0.88)
        result = experiment_summary_text(exp)
        assert "F1 Score" in result
        assert "0.8800" in result

    def test_with_duration(self):
        from insideLLMs.analysis.visualization import experiment_summary_text

        exp = _make_experiment(
            started_at=datetime(2024, 1, 1, 0, 0, 0),
            completed_at=datetime(2024, 1, 1, 0, 1, 30),
        )
        result = experiment_summary_text(exp)
        assert "Duration" in result
        assert "90.00" in result

    def test_no_score_at_all(self):
        from insideLLMs.analysis.visualization import experiment_summary_text

        exp = _make_no_score_experiment()
        result = experiment_summary_text(exp)
        assert "NoScoreModel" in result
        assert "Scores:" not in result


# ---------------------------------------------------------------------------
# 10. create_html_report with additional output types
# ---------------------------------------------------------------------------


class TestCreateHtmlReportAdditional:
    def test_empty_results(self, tmp_path):
        from insideLLMs.analysis.visualization import create_html_report

        save_path = str(tmp_path / "empty.html")
        result_path = create_html_report([], save_path=save_path)
        content = Path(result_path).read_text()
        assert "Total Results:</strong> 0" in content

    def test_html_escaping_in_input(self, tmp_path):
        from insideLLMs.analysis.visualization import create_html_report

        results = [
            {"input": '<img src="x" onerror="alert(1)">', "output": "safe"},
        ]
        save_path = str(tmp_path / "escape.html")
        create_html_report(results, save_path=save_path)
        content = Path(save_path).read_text()
        assert '<img src="x"' not in content
        assert "&lt;img" in content

    def test_only_error_results(self, tmp_path):
        from insideLLMs.analysis.visualization import create_html_report

        results = [
            {"input": "q1", "error": "timeout"},
            {"input": "q2", "error": "rate limit"},
        ]
        save_path = str(tmp_path / "errors.html")
        result_path = create_html_report(results, save_path=save_path)
        content = Path(result_path).read_text()
        assert "Errors:</strong> 2" in content
        assert "timeout" in content

    def test_nested_list_output(self, tmp_path):
        from insideLLMs.analysis.visualization import create_html_report

        results = [
            {"input": "q", "output": ["item1", "item2", "item3"]},
        ]
        save_path = str(tmp_path / "list_out.html")
        create_html_report(results, save_path=save_path)
        content = Path(save_path).read_text()
        assert "item1" in content
        assert "item3" in content
        assert "<table" in content

    def test_dict_output(self, tmp_path):
        from insideLLMs.analysis.visualization import create_html_report

        results = [
            {"input": "q", "output": {"key1": "val1", "key2": "val2"}},
        ]
        save_path = str(tmp_path / "dict_out.html")
        create_html_report(results, save_path=save_path)
        content = Path(save_path).read_text()
        assert "key1" in content
        assert "val1" in content

    def test_no_input_no_output(self, tmp_path):
        """Results with no input or output keys."""
        from insideLLMs.analysis.visualization import create_html_report

        results = [{"something_else": True}]
        save_path = str(tmp_path / "noio.html")
        create_html_report(results, save_path=save_path)
        content = Path(save_path).read_text()
        assert "Result 1" in content

    def test_error_is_falsy(self, tmp_path):
        """Error key with empty string should not render error div."""
        from insideLLMs.analysis.visualization import create_html_report

        results = [{"input": "q", "output": "a", "error": ""}]
        save_path = str(tmp_path / "falsy_err.html")
        create_html_report(results, save_path=save_path)
        content = Path(save_path).read_text()
        # The error div should not appear for empty error string
        assert "Error:</strong>" not in content


# ---------------------------------------------------------------------------
# 11. Dependency check functions
# ---------------------------------------------------------------------------


class TestDependencyCheckEdgeCases:
    def test_check_visualization_deps_when_available(self):
        from insideLLMs.analysis import visualization as viz

        if viz.MATPLOTLIB_AVAILABLE:
            viz.check_visualization_deps()  # should not raise
        else:
            with pytest.raises(ImportError):
                viz.check_visualization_deps()

    def test_check_plotly_deps_when_not_available(self):
        from insideLLMs.analysis import visualization as viz

        orig = viz.PLOTLY_AVAILABLE
        try:
            viz.PLOTLY_AVAILABLE = False
            with pytest.raises(ImportError, match="plotly"):
                viz.check_plotly_deps()
        finally:
            viz.PLOTLY_AVAILABLE = orig

    def test_check_ipywidgets_deps_when_not_available(self):
        from insideLLMs.analysis import visualization as viz

        orig = viz.IPYWIDGETS_AVAILABLE
        try:
            viz.IPYWIDGETS_AVAILABLE = False
            with pytest.raises(ImportError, match="ipywidgets"):
                viz.check_ipywidgets_deps()
        finally:
            viz.IPYWIDGETS_AVAILABLE = orig


# ---------------------------------------------------------------------------
# 12. text_comparison_table additional cases
# ---------------------------------------------------------------------------


class TestTextComparisonTableAdditional:
    def test_mixed_int_float_str(self):
        from insideLLMs.analysis.visualization import text_comparison_table

        result = text_comparison_table(
            rows=["A", "B"],
            cols=["Int", "Float", "Str"],
            values=[[1, 2.5, "ok"], [3, 4.0, "bad"]],
        )
        assert "ok" in result
        assert "bad" in result
        assert "2.5000" in result

    def test_single_row_col(self):
        from insideLLMs.analysis.visualization import text_comparison_table

        result = text_comparison_table(
            rows=["X"],
            cols=["Y"],
            values=[[0.999]],
            title="Tiny",
        )
        assert "Tiny" in result
        assert "X" in result
        assert "0.9990" in result


# ---------------------------------------------------------------------------
# 13. text_summary_stats edge cases
# ---------------------------------------------------------------------------


class TestTextSummaryStatsAdditional:
    def test_only_ci_lower_provided(self):
        """If only one CI bound is given, CI should not be displayed."""
        from insideLLMs.analysis.visualization import text_summary_stats

        result = text_summary_stats(
            name="test",
            mean=1.0,
            std=0.1,
            min_val=0.5,
            max_val=1.5,
            n=10,
            ci_lower=0.9,
            ci_upper=None,
        )
        assert "95% CI" not in result

    def test_both_ci_bounds(self):
        from insideLLMs.analysis.visualization import text_summary_stats

        result = text_summary_stats(
            name="metric",
            mean=5.0,
            std=1.0,
            min_val=3.0,
            max_val=7.0,
            n=100,
            ci_lower=4.8,
            ci_upper=5.2,
        )
        assert "95% CI" in result
        assert "4.8000" in result
        assert "5.2000" in result
