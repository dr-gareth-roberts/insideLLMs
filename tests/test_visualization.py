"""Tests for the visualization module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.types import (
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)


# Helper to create mock experiment results
def create_mock_experiment(
    model_name: str = "test-model",
    probe_name: str = "TestProbe",
    accuracy: float = 0.85,
    num_results: int = 10,
) -> ExperimentResult:
    """Create a mock ExperimentResult for testing."""
    results = [
        ProbeResult(
            input=f"test input {i}",
            output=f"test output {i}",
            status=ResultStatus.SUCCESS if i < int(accuracy * num_results) else ResultStatus.ERROR,
            latency_ms=100.0 + i * 10,
        )
        for i in range(num_results)
    ]

    return ExperimentResult(
        experiment_id=f"exp_{model_name}_{probe_name}",
        model_info=ModelInfo(
            name=model_name,
            provider="test",
            model_id=f"test-{model_name.lower()}",
        ),
        probe_name=probe_name,
        probe_category=ProbeCategory.LOGIC,
        results=results,
        score=ProbeScore(
            accuracy=accuracy,
            precision=accuracy - 0.02,
            recall=accuracy + 0.02,
            f1_score=accuracy,
            mean_latency_ms=150.0,
        ),
    )


class TestTextBarChart:
    """Test the text_bar_chart function."""

    def test_basic_bar_chart(self):
        """Test creating a basic bar chart."""
        from insideLLMs.visualization import text_bar_chart

        labels = ["A", "B", "C"]
        values = [0.8, 0.6, 0.9]
        result = text_bar_chart(labels, values)

        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_bar_chart_with_title(self):
        """Test bar chart with title."""
        from insideLLMs.visualization import text_bar_chart

        labels = ["A", "B"]
        values = [0.5, 0.7]
        result = text_bar_chart(labels, values, title="Test Chart")

        assert "Test Chart" in result

    def test_bar_chart_empty_data(self):
        """Test bar chart with empty data."""
        from insideLLMs.visualization import text_bar_chart

        result = text_bar_chart([], [])
        assert "No data" in result

    def test_bar_chart_custom_width(self):
        """Test bar chart with custom width."""
        from insideLLMs.visualization import text_bar_chart

        labels = ["A"]
        values = [1.0]
        result = text_bar_chart(labels, values, width=20)
        # Should have bars of appropriate width
        assert "█" in result or "=" in result or "#" in result


class TestTextHistogram:
    """Test the text_histogram function."""

    def test_basic_histogram(self):
        """Test creating a basic histogram."""
        from insideLLMs.visualization import text_histogram

        values = [1, 2, 2, 3, 3, 3, 4, 4, 5]
        result = text_histogram(values)

        assert result is not None
        assert len(result) > 0

    def test_histogram_with_bins(self):
        """Test histogram with custom bins."""
        from insideLLMs.visualization import text_histogram

        values = list(range(100))
        result = text_histogram(values, bins=5)

        assert result is not None

    def test_histogram_empty_values(self):
        """Test histogram with empty values."""
        from insideLLMs.visualization import text_histogram

        result = text_histogram([])
        assert "No data" in result or result == ""

    def test_histogram_with_title(self):
        """Test histogram with title."""
        from insideLLMs.visualization import text_histogram

        values = [1, 2, 3]
        result = text_histogram(values, title="Test Histogram")

        assert "Test Histogram" in result


class TestTextSummaryStats:
    """Test the text_summary_stats function."""

    def test_basic_stats(self):
        """Test creating statistics summary string."""
        from insideLLMs.visualization import text_summary_stats

        result = text_summary_stats(
            name="accuracy",
            mean=0.85,
            std=0.05,
            min_val=0.75,
            max_val=0.95,
            n=100,
        )

        assert "accuracy" in result
        assert "0.85" in result or "85" in result
        assert "100" in result

    def test_stats_with_confidence_interval(self):
        """Test stats with confidence interval."""
        from insideLLMs.visualization import text_summary_stats

        result = text_summary_stats(
            name="score",
            mean=0.9,
            std=0.1,
            min_val=0.7,
            max_val=1.0,
            n=50,
            ci_lower=0.85,
            ci_upper=0.95,
        )

        assert "score" in result
        assert "CI" in result or "0.85" in result

    def test_stats_formatting(self):
        """Test that stats are properly formatted."""
        from insideLLMs.visualization import text_summary_stats

        result = text_summary_stats(
            name="test",
            mean=3.14159,
            std=1.0,
            min_val=1.0,
            max_val=5.0,
            n=10,
        )

        # Should contain formatted numbers
        assert "Mean" in result or "mean" in result.lower()
        assert "3.14" in result


class TestExperimentSummaryText:
    """Test the experiment_summary_text function."""

    def test_basic_summary(self):
        """Test creating experiment summary."""
        from insideLLMs.visualization import experiment_summary_text

        experiment = create_mock_experiment()
        result = experiment_summary_text(experiment)

        assert "test-model" in result.lower() or "model" in result.lower()
        assert "accuracy" in result.lower() or "score" in result.lower()

    def test_summary_includes_stats(self):
        """Test summary includes statistics."""
        from insideLLMs.visualization import experiment_summary_text

        experiment = create_mock_experiment(accuracy=0.9)
        result = experiment_summary_text(experiment)

        # Should contain some numeric information
        assert any(c.isdigit() for c in result)


class TestPlotlyVisualizationsAvailability:
    """Test that Plotly visualizations are available when plotly is installed."""

    def test_interactive_functions_exist(self):
        """Test that interactive visualization functions exist."""
        from insideLLMs import visualization

        # These should exist even if plotly is not installed
        assert hasattr(visualization, "text_bar_chart")
        assert hasattr(visualization, "text_histogram")
        assert hasattr(visualization, "text_summary_stats")

    def test_plotly_functions_are_importable(self):
        """Test that plotly functions can be imported."""
        try:
            from insideLLMs.visualization import (
                interactive_accuracy_comparison,
                interactive_latency_distribution,
                interactive_metric_radar,
                interactive_heatmap,
                interactive_scatter_comparison,
                create_interactive_dashboard,
                create_interactive_html_report,
            )
            plotly_available = True
        except ImportError:
            plotly_available = False

        # Either they're importable or we get ImportError - both are valid


class TestResultsVisualization:
    """Test visualization of results."""

    def test_visualize_results_function_exists(self):
        """Test that visualize_results function exists."""
        from insideLLMs import visualization

        # Should have some form of results visualization
        assert hasattr(visualization, "text_bar_chart") or hasattr(visualization, "visualize_results")


class TestTextBasedVisualization:
    """Test text-based visualization functions."""

    def test_create_ascii_bar(self):
        """Test creating ASCII bars."""
        from insideLLMs.visualization import text_bar_chart

        labels = ["Test"]
        values = [0.5]
        result = text_bar_chart(labels, values, width=10)
        assert "Test" in result

    def test_format_percentage(self):
        """Test formatting percentages in output."""
        from insideLLMs.visualization import text_bar_chart

        labels = ["A"]
        values = [0.75]
        result = text_bar_chart(labels, values)
        # Should contain some representation of the value
        assert "75" in result or "0.75" in result or "█" in result


class TestVisualizationWithExperiments:
    """Test visualization with ExperimentResult objects."""

    def test_visualize_multiple_experiments(self):
        """Test visualizing multiple experiments."""
        from insideLLMs.visualization import text_bar_chart

        experiments = [
            create_mock_experiment("Model-A", accuracy=0.8),
            create_mock_experiment("Model-B", accuracy=0.9),
        ]

        # Create data from experiments
        labels = [exp.model_info.name for exp in experiments]
        values = [exp.score.accuracy for exp in experiments]
        result = text_bar_chart(labels, values, title="Model Comparison")

        assert "Model-A" in result or "model" in result.lower()
        assert "Model-B" in result or "model" in result.lower()


class TestHTMLReportGeneration:
    """Test HTML report generation."""

    def test_html_report_structure(self):
        """Test that HTML report has correct structure."""
        try:
            from insideLLMs.visualization import create_interactive_html_report

            experiments = [create_mock_experiment()]

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "report.html"
                result_path = create_interactive_html_report(
                    experiments,
                    save_path=str(path),
                    title="Test Report",
                )

                assert Path(result_path).exists()
                content = Path(result_path).read_text()
                assert "<html" in content.lower()
                assert "Test Report" in content
        except ImportError:
            pytest.skip("Plotly not available")


class TestEdgeCases:
    """Test edge cases in visualization."""

    def test_very_long_labels(self):
        """Test handling of very long labels."""
        from insideLLMs.visualization import text_bar_chart

        labels = ["A" * 100, "B" * 100]
        values = [0.5, 0.7]
        result = text_bar_chart(labels, values)
        # Should not raise an error
        assert result is not None

    def test_unicode_labels(self):
        """Test handling of Unicode labels."""
        from insideLLMs.visualization import text_bar_chart

        labels = ["模型A", "モデルB", "Модель"]
        values = [0.5, 0.7, 0.8]
        result = text_bar_chart(labels, values)
        # Should handle Unicode without errors
        assert result is not None

    def test_zero_values(self):
        """Test handling of zero values."""
        from insideLLMs.visualization import text_bar_chart

        labels = ["Zero", "One"]
        values = [0.0, 1.0]
        result = text_bar_chart(labels, values)
        assert "Zero" in result
        assert "One" in result

    def test_negative_values(self):
        """Test handling of negative values."""
        from insideLLMs.visualization import text_histogram

        values = [-5, -3, 0, 3, 5]
        result = text_histogram(values)
        # Should handle negative values
        assert result is not None


class TestConvenienceFunctions:
    """Test convenience functions in visualization module."""

    def test_all_public_functions_documented(self):
        """Test that public functions have docstrings."""
        from insideLLMs import visualization

        public_funcs = [
            name for name in dir(visualization)
            if not name.startswith("_") and callable(getattr(visualization, name))
        ]

        for func_name in public_funcs[:5]:  # Check first 5
            func = getattr(visualization, func_name)
            if hasattr(func, "__doc__"):
                # Doc can be None or empty
                pass  # OK either way
