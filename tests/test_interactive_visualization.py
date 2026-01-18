"""Tests for interactive visualization functions."""

from unittest.mock import patch

import pytest

from insideLLMs.types import (
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)
from insideLLMs.visualization import (
    IPYWIDGETS_AVAILABLE,
    PLOTLY_AVAILABLE,
    check_ipywidgets_deps,
    check_plotly_deps,
)


# Helper to create mock experiment results
def create_mock_experiment(
    model_name: str = "GPT-4",
    probe_name: str = "LogicProbe",
    accuracy: float = 0.85,
    precision: float = 0.82,
    recall: float = 0.88,
    f1_score: float = 0.85,
    latency_ms: float = 150.0,
    success_count: int = 85,
    error_count: int = 15,
    provider: str = "openai",
) -> ExperimentResult:
    """Create a mock experiment result for testing."""
    results = []
    for i in range(success_count):
        results.append(
            ProbeResult(
                input=f"test input {i}",
                output=f"test output {i}",
                status=ResultStatus.SUCCESS,
                latency_ms=latency_ms + (i % 50),
            )
        )
    for i in range(error_count):
        results.append(
            ProbeResult(
                input=f"error input {i}",
                output="",
                status=ResultStatus.ERROR,
                error="Test error",
            )
        )

    # ExperimentResult computes total_count, success_count, error_count,
    # and success_rate as properties from results list
    return ExperimentResult(
        experiment_id=f"exp_{model_name}_{probe_name}",
        model_info=ModelInfo(
            name=model_name,
            provider=provider,
            model_id=f"{provider}-{model_name.lower()}",
        ),
        probe_name=probe_name,
        probe_category=ProbeCategory.LOGIC,
        results=results,
        score=ProbeScore(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mean_latency_ms=latency_ms,
        ),
    )


@pytest.fixture
def mock_experiments():
    """Create a set of mock experiments for testing."""
    return [
        create_mock_experiment("GPT-4", "LogicProbe", 0.90, 0.88, 0.92, 0.90, 120.0),
        create_mock_experiment("GPT-4", "BiasProbe", 0.85, 0.82, 0.88, 0.85, 130.0),
        create_mock_experiment(
            "Claude-3", "LogicProbe", 0.92, 0.91, 0.93, 0.92, 100.0, provider="anthropic"
        ),
        create_mock_experiment(
            "Claude-3", "BiasProbe", 0.88, 0.86, 0.90, 0.88, 110.0, provider="anthropic"
        ),
        create_mock_experiment(
            "Llama-2", "LogicProbe", 0.78, 0.75, 0.81, 0.78, 80.0, provider="meta"
        ),
    ]


class TestDependencyChecks:
    """Tests for dependency checking functions."""

    def test_check_plotly_deps_raises_when_unavailable(self):
        """Test that check_plotly_deps raises ImportError when Plotly is not available."""
        with patch("insideLLMs.visualization.PLOTLY_AVAILABLE", False):
            with pytest.raises(ImportError, match="Plotly"):
                check_plotly_deps()

    def test_check_ipywidgets_deps_raises_when_unavailable(self):
        """Test that check_ipywidgets_deps raises ImportError when ipywidgets is not available."""
        with patch("insideLLMs.visualization.IPYWIDGETS_AVAILABLE", False):
            with pytest.raises(ImportError, match="ipywidgets"):
                check_ipywidgets_deps()


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestInteractiveAccuracyComparison:
    """Tests for interactive_accuracy_comparison function."""

    def test_creates_figure(self, mock_experiments):
        """Test that function creates a Plotly figure."""
        from insideLLMs.visualization import interactive_accuracy_comparison

        fig = interactive_accuracy_comparison(mock_experiments)
        assert fig is not None
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")

    def test_raises_on_empty_data(self):
        """Test that function raises ValueError on empty experiments."""
        from insideLLMs.visualization import interactive_accuracy_comparison

        empty_exp = create_mock_experiment()
        empty_exp.score = None

        with pytest.raises(ValueError, match="No accuracy data"):
            interactive_accuracy_comparison([empty_exp])

    def test_color_by_model(self, mock_experiments):
        """Test coloring by model."""
        from insideLLMs.visualization import interactive_accuracy_comparison

        fig = interactive_accuracy_comparison(mock_experiments, color_by="model")
        assert fig is not None

    def test_color_by_probe(self, mock_experiments):
        """Test coloring by probe."""
        from insideLLMs.visualization import interactive_accuracy_comparison

        fig = interactive_accuracy_comparison(mock_experiments, color_by="probe")
        assert fig is not None


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestInteractiveLatencyDistribution:
    """Tests for interactive_latency_distribution function."""

    def test_creates_box_plot(self, mock_experiments):
        """Test creating a box plot."""
        from insideLLMs.visualization import interactive_latency_distribution

        fig = interactive_latency_distribution(mock_experiments, chart_type="box")
        assert fig is not None

    def test_creates_violin_plot(self, mock_experiments):
        """Test creating a violin plot."""
        from insideLLMs.visualization import interactive_latency_distribution

        fig = interactive_latency_distribution(mock_experiments, chart_type="violin")
        assert fig is not None

    def test_creates_histogram(self, mock_experiments):
        """Test creating a histogram."""
        from insideLLMs.visualization import interactive_latency_distribution

        fig = interactive_latency_distribution(mock_experiments, chart_type="histogram")
        assert fig is not None

    def test_raises_on_no_latency_data(self):
        """Test that function raises when no latency data available."""
        from insideLLMs.visualization import interactive_latency_distribution

        exp = create_mock_experiment()
        for result in exp.results:
            result.latency_ms = None

        with pytest.raises(ValueError, match="No latency data"):
            interactive_latency_distribution([exp])


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestInteractiveMetricRadar:
    """Tests for interactive_metric_radar function."""

    def test_creates_radar_chart(self, mock_experiments):
        """Test creating a radar chart."""
        from insideLLMs.visualization import interactive_metric_radar

        fig = interactive_metric_radar(mock_experiments)
        assert fig is not None
        # Radar charts use Scatterpolar traces
        assert any("Scatterpolar" in str(type(trace)) for trace in fig.data)

    def test_custom_metrics(self, mock_experiments):
        """Test with custom metric selection."""
        from insideLLMs.visualization import interactive_metric_radar

        fig = interactive_metric_radar(mock_experiments, metrics=["accuracy", "precision"])
        assert fig is not None


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestInteractiveTimeline:
    """Tests for interactive_timeline function."""

    def test_creates_timeline(self, mock_experiments):
        """Test creating a timeline chart."""
        from insideLLMs.visualization import interactive_timeline

        fig = interactive_timeline(mock_experiments)
        assert fig is not None

    def test_custom_metric(self, mock_experiments):
        """Test with custom metric."""
        from insideLLMs.visualization import interactive_timeline

        fig = interactive_timeline(mock_experiments, metric="precision")
        assert fig is not None

    def test_raises_on_missing_metric(self, mock_experiments):
        """Test that function raises when metric not available."""
        from insideLLMs.visualization import interactive_timeline

        for exp in mock_experiments:
            exp.score = None

        with pytest.raises(ValueError):
            interactive_timeline(mock_experiments)


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestInteractiveHeatmap:
    """Tests for interactive_heatmap function."""

    def test_creates_heatmap(self, mock_experiments):
        """Test creating a heatmap."""
        from insideLLMs.visualization import interactive_heatmap

        fig = interactive_heatmap(mock_experiments)
        assert fig is not None
        # Heatmap should have Heatmap trace
        assert any("Heatmap" in str(type(trace)) for trace in fig.data)

    def test_row_col_configuration(self, mock_experiments):
        """Test with different row/col configurations."""
        from insideLLMs.visualization import interactive_heatmap

        fig = interactive_heatmap(mock_experiments, row_key="probe", col_key="model")
        assert fig is not None

    def test_custom_value_key(self, mock_experiments):
        """Test with custom value metric."""
        from insideLLMs.visualization import interactive_heatmap

        fig = interactive_heatmap(mock_experiments, value_key="precision")
        assert fig is not None


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestInteractiveScatterComparison:
    """Tests for interactive_scatter_comparison function."""

    def test_creates_scatter_plot(self, mock_experiments):
        """Test creating a scatter plot."""
        from insideLLMs.visualization import interactive_scatter_comparison

        fig = interactive_scatter_comparison(mock_experiments)
        assert fig is not None

    def test_custom_metrics(self, mock_experiments):
        """Test with custom x/y metrics."""
        from insideLLMs.visualization import interactive_scatter_comparison

        fig = interactive_scatter_comparison(
            mock_experiments,
            x_metric="precision",
            y_metric="recall",
        )
        assert fig is not None


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestInteractiveSunburst:
    """Tests for interactive_sunburst function."""

    def test_creates_sunburst(self, mock_experiments):
        """Test creating a sunburst chart."""
        from insideLLMs.visualization import interactive_sunburst

        fig = interactive_sunburst(mock_experiments)
        assert fig is not None

    def test_custom_metric(self, mock_experiments):
        """Test with custom value metric."""
        from insideLLMs.visualization import interactive_sunburst

        fig = interactive_sunburst(mock_experiments, value_metric="precision")
        assert fig is not None


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestCreateInteractiveDashboard:
    """Tests for create_interactive_dashboard function."""

    def test_creates_dashboard(self, mock_experiments):
        """Test creating a dashboard."""
        from insideLLMs.visualization import create_interactive_dashboard

        fig = create_interactive_dashboard(mock_experiments)
        assert fig is not None
        # Dashboard should have multiple subplots
        assert len(fig.data) > 0

    def test_saves_to_file(self, mock_experiments, tmp_path):
        """Test saving dashboard to HTML file."""
        from insideLLMs.visualization import create_interactive_dashboard

        save_path = tmp_path / "dashboard.html"
        create_interactive_dashboard(mock_experiments, save_path=str(save_path))

        assert save_path.exists()
        content = save_path.read_text()
        assert "plotly" in content.lower() or "chart" in content.lower()


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestCreateInteractiveHtmlReport:
    """Tests for create_interactive_html_report function."""

    def test_creates_report(self, mock_experiments, tmp_path):
        """Test creating an HTML report."""
        from insideLLMs.visualization import create_interactive_html_report

        save_path = tmp_path / "report.html"
        result_path = create_interactive_html_report(mock_experiments, save_path=str(save_path))

        assert save_path.exists()
        assert result_path == str(save_path)

        content = save_path.read_text()
        assert "LLM Evaluation Report" in content
        assert "Experiments" in content

    def test_includes_raw_results(self, mock_experiments, tmp_path):
        """Test including raw results table."""
        from insideLLMs.visualization import create_interactive_html_report

        save_path = tmp_path / "report_with_results.html"
        create_interactive_html_report(
            mock_experiments,
            save_path=str(save_path),
            include_raw_results=True,
        )

        content = save_path.read_text()
        assert "<table>" in content
        assert "GPT-4" in content or "Claude" in content

    def test_custom_title(self, mock_experiments, tmp_path):
        """Test with custom title."""
        from insideLLMs.visualization import create_interactive_html_report

        save_path = tmp_path / "custom_report.html"
        create_interactive_html_report(
            mock_experiments,
            title="My Custom Report",
            save_path=str(save_path),
        )

        content = save_path.read_text()
        assert "My Custom Report" in content


@pytest.mark.skipif(
    not (PLOTLY_AVAILABLE and IPYWIDGETS_AVAILABLE),
    reason="Plotly or ipywidgets not installed",
)
class TestExperimentExplorer:
    """Tests for ExperimentExplorer class."""

    def test_initialization(self, mock_experiments):
        """Test explorer initialization."""
        from insideLLMs.visualization import ExperimentExplorer

        explorer = ExperimentExplorer(mock_experiments)
        assert explorer.experiments == mock_experiments
        assert len(explorer.models) > 0
        assert len(explorer.probes) > 0

    def test_models_extracted(self, mock_experiments):
        """Test that models are correctly extracted."""
        from insideLLMs.visualization import ExperimentExplorer

        explorer = ExperimentExplorer(mock_experiments)
        assert "GPT-4" in explorer.models
        assert "Claude-3" in explorer.models

    def test_probes_extracted(self, mock_experiments):
        """Test that probes are correctly extracted."""
        from insideLLMs.visualization import ExperimentExplorer

        explorer = ExperimentExplorer(mock_experiments)
        assert "LogicProbe" in explorer.probes
        assert "BiasProbe" in explorer.probes

    def test_compare_models(self, mock_experiments):
        """Test compare_models method."""
        from insideLLMs.visualization import ExperimentExplorer

        explorer = ExperimentExplorer(mock_experiments)
        # This returns a styled DataFrame
        result = explorer.compare_models(metric="accuracy", aggregate="mean")
        assert result is not None


class TestEmptyExperiments:
    """Tests for handling empty experiment lists."""

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_interactive_accuracy_empty(self):
        """Test interactive_accuracy_comparison with empty list."""
        from insideLLMs.visualization import interactive_accuracy_comparison

        with pytest.raises(ValueError):
            interactive_accuracy_comparison([])

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_interactive_latency_empty(self):
        """Test interactive_latency_distribution with empty list."""
        from insideLLMs.visualization import interactive_latency_distribution

        with pytest.raises(ValueError):
            interactive_latency_distribution([])

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_interactive_heatmap_empty(self):
        """Test interactive_heatmap with empty list."""
        from insideLLMs.visualization import interactive_heatmap

        with pytest.raises(ValueError):
            interactive_heatmap([])
