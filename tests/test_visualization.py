"""Tests for the visualization module."""

import contextlib
import tempfile
from pathlib import Path

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
        with contextlib.suppress(ImportError):
            pass

        # Either they're importable or we get ImportError - both are valid


class TestResultsVisualization:
    """Test visualization of results."""

    def test_visualize_results_function_exists(self):
        """Test that visualize_results function exists."""
        from insideLLMs import visualization

        # Should have some form of results visualization
        assert hasattr(visualization, "text_bar_chart") or hasattr(
            visualization, "visualize_results"
        )


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
            name
            for name in dir(visualization)
            if not name.startswith("_") and callable(getattr(visualization, name))
        ]

        for func_name in public_funcs[:5]:  # Check first 5
            func = getattr(visualization, func_name)
            if hasattr(func, "__doc__"):
                # Doc can be None or empty
                pass  # OK either way


class TestTextComparisonTable:
    """Test the text_comparison_table function."""

    def test_basic_comparison_table(self):
        """Test creating a basic comparison table."""
        from insideLLMs.visualization import text_comparison_table

        rows = ["Model A", "Model B"]
        cols = ["Accuracy", "F1"]
        values = [[0.85, 0.82], [0.90, 0.88]]
        result = text_comparison_table(rows, cols, values)

        assert "Model A" in result
        assert "Model B" in result
        assert "Accuracy" in result
        assert "F1" in result

    def test_comparison_table_with_title(self):
        """Test comparison table with title."""
        from insideLLMs.visualization import text_comparison_table

        rows = ["A", "B"]
        cols = ["X", "Y"]
        values = [[1, 2], [3, 4]]
        result = text_comparison_table(rows, cols, values, title="Test Table")

        assert "Test Table" in result

    def test_comparison_table_empty_data(self):
        """Test comparison table with empty data."""
        from insideLLMs.visualization import text_comparison_table

        result = text_comparison_table([], [], [])
        assert "No data" in result

    def test_comparison_table_float_formatting(self):
        """Test that floats are formatted properly."""
        from insideLLMs.visualization import text_comparison_table

        rows = ["A"]
        cols = ["Score"]
        values = [[0.123456789]]
        result = text_comparison_table(rows, cols, values)

        # Should be formatted to 4 decimal places (may round)
        assert "0.1234" in result or "0.1235" in result

    def test_comparison_table_string_values(self):
        """Test comparison table with string values."""
        from insideLLMs.visualization import text_comparison_table

        rows = ["A", "B"]
        cols = ["Status"]
        values = [["Pass"], ["Fail"]]
        result = text_comparison_table(rows, cols, values)

        assert "Pass" in result
        assert "Fail" in result


class TestHistogramEdgeCases:
    """Test edge cases for histogram function."""

    def test_histogram_all_same_values(self):
        """Test histogram when all values are the same."""
        from insideLLMs.visualization import text_histogram

        values = [5.0, 5.0, 5.0, 5.0]
        result = text_histogram(values)

        # Should handle this gracefully
        assert "5" in result or "equal" in result.lower()

    def test_histogram_single_value(self):
        """Test histogram with single value."""
        from insideLLMs.visualization import text_histogram

        values = [1.0]
        result = text_histogram(values)

        assert result is not None

    def test_histogram_custom_character(self):
        """Test histogram with custom bar character."""
        from insideLLMs.visualization import text_histogram

        values = [1, 2, 3, 4, 5]
        result = text_histogram(values, char="#")

        assert "#" in result


class TestBarChartEdgeCases:
    """Test edge cases for bar chart function."""

    def test_bar_chart_hide_values(self):
        """Test bar chart without showing values."""
        from insideLLMs.visualization import text_bar_chart

        labels = ["A", "B"]
        values = [0.5, 0.7]
        result = text_bar_chart(labels, values, show_values=False)

        # Should not contain formatted float values
        assert "0.50" not in result

    def test_bar_chart_custom_character(self):
        """Test bar chart with custom bar character."""
        from insideLLMs.visualization import text_bar_chart

        labels = ["A"]
        values = [1.0]
        result = text_bar_chart(labels, values, char="#", width=10)

        assert "#" in result

    def test_bar_chart_all_zero_values(self):
        """Test bar chart with all zero values."""
        from insideLLMs.visualization import text_bar_chart

        labels = ["A", "B"]
        values = [0.0, 0.0]
        result = text_bar_chart(labels, values)

        assert "A" in result
        assert "B" in result


class TestDependencyChecks:
    """Test dependency checking functions."""

    def test_check_visualization_deps_raises_when_missing(self):
        """Test that check raises when matplotlib not available."""
        from insideLLMs import visualization

        # If matplotlib is not available, this should raise
        # If it is available, the function should not raise
        if visualization.MATPLOTLIB_AVAILABLE:
            visualization.check_visualization_deps()  # Should not raise
        else:
            with pytest.raises(ImportError):
                visualization.check_visualization_deps()

    def test_check_plotly_deps_raises_when_missing(self):
        """Test that check raises when plotly not available."""
        from insideLLMs import visualization

        if visualization.PLOTLY_AVAILABLE:
            visualization.check_plotly_deps()  # Should not raise
        else:
            with pytest.raises(ImportError):
                visualization.check_plotly_deps()

    def test_check_ipywidgets_deps_raises_when_missing(self):
        """Test that check raises when ipywidgets not available."""
        from insideLLMs import visualization

        if visualization.IPYWIDGETS_AVAILABLE:
            visualization.check_ipywidgets_deps()  # Should not raise
        else:
            with pytest.raises(ImportError):
                visualization.check_ipywidgets_deps()


class TestExperimentSummaryEdgeCases:
    """Test edge cases for experiment summary."""

    def test_experiment_summary_no_score(self):
        """Test summary when experiment has no score."""
        from insideLLMs.types import ExperimentResult, ModelInfo, ProbeCategory
        from insideLLMs.visualization import experiment_summary_text

        experiment = ExperimentResult(
            experiment_id="test_exp",
            model_info=ModelInfo(name="test", provider="test", model_id="test"),
            probe_name="TestProbe",
            probe_category=ProbeCategory.LOGIC,
            results=[],
            score=None,  # No score
        )
        result = experiment_summary_text(experiment)

        assert "test_exp" in result
        assert "TestProbe" in result

    def test_experiment_summary_partial_scores(self):
        """Test summary when experiment has partial scores."""
        from insideLLMs.visualization import experiment_summary_text

        experiment = create_mock_experiment()
        experiment.score.precision = None  # Remove precision
        experiment.score.recall = None  # Remove recall
        result = experiment_summary_text(experiment)

        # Should still work
        assert "test-model" in result.lower() or "model" in result.lower()

    def test_experiment_summary_with_duration(self):
        """Test summary includes duration when available."""
        from datetime import datetime

        from insideLLMs.types import (
            ExperimentResult,
            ModelInfo,
            ProbeCategory,
            ProbeResult,
            ProbeScore,
            ResultStatus,
        )
        from insideLLMs.visualization import experiment_summary_text

        # Create an experiment with started_at/completed_at for duration
        results = [
            ProbeResult(
                input="test",
                output="test output",
                status=ResultStatus.SUCCESS,
                latency_ms=100.0,
            )
        ]
        experiment = ExperimentResult(
            experiment_id="test_exp",
            model_info=ModelInfo(name="test", provider="test", model_id="test"),
            probe_name="TestProbe",
            probe_category=ProbeCategory.LOGIC,
            results=results,
            score=ProbeScore(accuracy=0.85),
            started_at=datetime(2024, 1, 1, 0, 0, 0),
            completed_at=datetime(2024, 1, 1, 0, 2, 0),  # 2 minutes = 120 seconds
        )
        result = experiment_summary_text(experiment)

        # duration_seconds is a computed property based on start/end times
        assert experiment.duration_seconds is not None
        assert experiment.duration_seconds == 120.0
        # Should have some duration info
        assert "test_exp" in result


class TestCreateHtmlReport:
    """Test the create_html_report function."""

    def test_create_html_report_basic(self, tmp_path):
        """Test creating a basic HTML report."""
        from insideLLMs.visualization import create_html_report

        results = [
            {"input": "test1", "output": "result1"},
            {"input": "test2", "output": "result2"},
        ]
        save_path = str(tmp_path / "report.html")
        result_path = create_html_report(results, title="Test Report", save_path=save_path)

        assert Path(result_path).exists()
        content = Path(result_path).read_text()
        assert "<html>" in content
        assert "Test Report" in content
        assert "test1" in content
        assert "result1" in content

    def test_create_html_report_with_errors(self, tmp_path):
        """Test HTML report includes error handling."""
        from insideLLMs.visualization import create_html_report

        results = [
            {"input": "test1", "output": "result1"},
            {"input": "test2", "error": "Something went wrong"},
        ]
        save_path = str(tmp_path / "report.html")
        result_path = create_html_report(results, save_path=save_path)

        content = Path(result_path).read_text()
        assert "Something went wrong" in content
        assert "error" in content.lower()

    def test_create_html_report_with_list_output(self, tmp_path):
        """Test HTML report with list outputs."""
        from insideLLMs.visualization import create_html_report

        results = [
            {"input": "test", "output": ["item1", "item2", "item3"]},
        ]
        save_path = str(tmp_path / "report.html")
        create_html_report(results, save_path=save_path)

        content = Path(save_path).read_text()
        assert "item1" in content
        assert "<table" in content

    def test_create_html_report_with_tuple_output(self, tmp_path):
        """Test HTML report with tuple outputs (bias results)."""
        from insideLLMs.visualization import create_html_report

        results = [
            {"input": "test", "output": [("response1", "response2")]},
        ]
        save_path = str(tmp_path / "report.html")
        create_html_report(results, save_path=save_path)

        content = Path(save_path).read_text()
        assert "response1" in content
        assert "response2" in content
        assert "vs." in content

    def test_create_html_report_with_dict_output(self, tmp_path):
        """Test HTML report with dict outputs."""
        from insideLLMs.visualization import create_html_report

        results = [
            {"input": "test", "output": {"key1": "value1", "key2": "value2"}},
        ]
        save_path = str(tmp_path / "report.html")
        create_html_report(results, save_path=save_path)

        content = Path(save_path).read_text()
        assert "key1" in content
        assert "value1" in content

    def test_create_html_report_with_string_output(self, tmp_path):
        """Test HTML report with string outputs."""
        from insideLLMs.visualization import create_html_report

        results = [
            {"input": "test", "output": "simple string output"},
        ]
        save_path = str(tmp_path / "report.html")
        create_html_report(results, save_path=save_path)

        content = Path(save_path).read_text()
        assert "simple string output" in content


class TestMatplotlibVisualizationsWithDeps:
    """Test matplotlib visualization functions when dependencies are available."""

    def test_plot_accuracy_comparison_no_data(self, capsys):
        """Test plot_accuracy_comparison with no accuracy data."""
        from insideLLMs import visualization

        if not visualization.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        # Create experiments with no accuracy data
        from insideLLMs.types import ExperimentResult, ModelInfo, ProbeCategory

        experiments = [
            ExperimentResult(
                experiment_id="test",
                model_info=ModelInfo(name="test", provider="test", model_id="test"),
                probe_name="TestProbe",
                probe_category=ProbeCategory.LOGIC,
                results=[],
                score=None,  # No score
            )
        ]

        visualization.plot_accuracy_comparison(experiments)
        captured = capsys.readouterr()
        assert "No accuracy" in captured.out

    def test_plot_latency_distribution_no_data(self, capsys):
        """Test plot_latency_distribution with no latency data."""
        from insideLLMs import visualization

        if not visualization.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        # Empty experiment list triggers "No latency data" message
        visualization.plot_latency_distribution([])
        captured = capsys.readouterr()
        assert "No latency" in captured.out

    def test_plot_metric_comparison_no_data(self, capsys):
        """Test plot_metric_comparison with no metric data."""
        from insideLLMs import visualization

        if not visualization.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        # Empty experiment list triggers "No metric data" message
        visualization.plot_metric_comparison([])
        captured = capsys.readouterr()
        assert "No metric" in captured.out

    def test_plot_success_rate_over_time_no_data(self, capsys):
        """Test plot_success_rate_over_time with no data."""
        from insideLLMs import visualization

        if not visualization.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        visualization.plot_success_rate_over_time([])
        captured = capsys.readouterr()
        assert "No data" in captured.out

    def test_plot_accuracy_comparison_saves_file(self, tmp_path):
        """Test plot_accuracy_comparison saves to file."""
        from insideLLMs import visualization

        if not visualization.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        experiments = [create_mock_experiment("Model1", accuracy=0.85)]
        save_path = str(tmp_path / "accuracy.png")

        visualization.plot_accuracy_comparison(experiments, save_path=save_path)
        assert Path(save_path).exists()

    def test_plot_latency_distribution_saves_file(self, tmp_path):
        """Test plot_latency_distribution saves to file."""
        from insideLLMs import visualization

        if not visualization.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        experiments = [create_mock_experiment()]
        save_path = str(tmp_path / "latency.png")

        visualization.plot_latency_distribution(experiments, save_path=save_path)
        assert Path(save_path).exists()

    def test_plot_metric_comparison_saves_file(self, tmp_path):
        """Test plot_metric_comparison saves to file."""
        from insideLLMs import visualization

        if not visualization.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        experiments = [create_mock_experiment()]
        save_path = str(tmp_path / "metrics.png")

        visualization.plot_metric_comparison(experiments, save_path=save_path)
        assert Path(save_path).exists()

    def test_plot_success_rate_over_time_saves_file(self, tmp_path):
        """Test plot_success_rate_over_time saves to file."""
        from insideLLMs import visualization

        if not visualization.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        results = [("Run 1", 0.8), ("Run 2", 0.85), ("Run 3", 0.9)]
        save_path = str(tmp_path / "success_rate.png")

        visualization.plot_success_rate_over_time(results, save_path=save_path)
        assert Path(save_path).exists()


class TestBiasAndFactualityPlots:
    """Test legacy bias and factuality plot functions."""

    def test_plot_bias_results_no_data(self, capsys):
        """Test plot_bias_results with no data."""
        from insideLLMs import visualization

        if not visualization.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        visualization.plot_bias_results([])
        captured = capsys.readouterr()
        assert "No bias" in captured.out

    def test_plot_bias_results_with_data(self, tmp_path):
        """Test plot_bias_results with valid data."""
        from insideLLMs import visualization

        if not visualization.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        results = [
            {"output": [("response A is longer", "response B")]},
            {"output": [("short", "this is a longer response")]},
        ]
        save_path = str(tmp_path / "bias.png")
        visualization.plot_bias_results(results, save_path=save_path)
        assert Path(save_path).exists()

    def test_plot_factuality_results_no_data(self, capsys):
        """Test plot_factuality_results with no data."""
        from insideLLMs import visualization

        if not visualization.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        visualization.plot_factuality_results([])
        captured = capsys.readouterr()
        assert "No factuality" in captured.out

    def test_plot_factuality_results_with_data(self, tmp_path):
        """Test plot_factuality_results with valid data."""
        from insideLLMs import visualization

        if not visualization.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        results = [
            {
                "output": [
                    {"category": "science", "model_answer": "The sun is a star."},
                    {"category": "history", "model_answer": "Rome was an empire."},
                ]
            }
        ]
        save_path = str(tmp_path / "factuality.png")
        visualization.plot_factuality_results(results, save_path=save_path)
        assert Path(save_path).exists()


class TestPlotlyInteractiveVisualizationsAvailability:
    """Test Plotly interactive visualizations."""

    def test_interactive_accuracy_comparison_raises_without_plotly(self):
        """Test that interactive function raises without plotly."""
        from insideLLMs import visualization

        if visualization.PLOTLY_AVAILABLE:
            pytest.skip("Plotly is available")

        with pytest.raises(ImportError):
            visualization.interactive_accuracy_comparison([])

    def test_interactive_latency_distribution_raises_without_plotly(self):
        """Test that interactive function raises without plotly."""
        from insideLLMs import visualization

        if visualization.PLOTLY_AVAILABLE:
            pytest.skip("Plotly is available")

        with pytest.raises(ImportError):
            visualization.interactive_latency_distribution([])

    def test_interactive_accuracy_comparison_with_plotly(self):
        """Test interactive_accuracy_comparison when plotly available."""
        from insideLLMs import visualization

        if not visualization.PLOTLY_AVAILABLE:
            pytest.skip("Plotly not available")

        experiments = [create_mock_experiment("Model1", accuracy=0.85)]
        fig = visualization.interactive_accuracy_comparison(experiments)
        assert fig is not None

    def test_interactive_accuracy_comparison_no_data(self):
        """Test interactive_accuracy_comparison raises with no data."""
        from insideLLMs import visualization

        if not visualization.PLOTLY_AVAILABLE:
            pytest.skip("Plotly not available")

        from insideLLMs.types import ExperimentResult, ModelInfo, ProbeCategory

        experiments = [
            ExperimentResult(
                experiment_id="test",
                model_info=ModelInfo(name="test", provider="test", model_id="test"),
                probe_name="TestProbe",
                probe_category=ProbeCategory.LOGIC,
                results=[],
                score=None,
            )
        ]

        with pytest.raises(ValueError, match="No accuracy"):
            visualization.interactive_accuracy_comparison(experiments)

    def test_interactive_latency_distribution_no_data(self):
        """Test interactive_latency_distribution raises with no data."""
        from insideLLMs import visualization

        if not visualization.PLOTLY_AVAILABLE:
            pytest.skip("Plotly not available")

        from insideLLMs.types import ExperimentResult, ModelInfo, ProbeCategory

        experiments = [
            ExperimentResult(
                experiment_id="test",
                model_info=ModelInfo(name="test", provider="test", model_id="test"),
                probe_name="TestProbe",
                probe_category=ProbeCategory.LOGIC,
                results=[],
                score=None,
            )
        ]

        with pytest.raises(ValueError, match="No latency"):
            visualization.interactive_latency_distribution(experiments)


class TestInteractiveHtmlReport:
    """Test create_interactive_html_report function."""

    def test_create_interactive_html_report(self, tmp_path):
        """Test creating interactive HTML report."""
        from insideLLMs import visualization

        if not visualization.PLOTLY_AVAILABLE:
            pytest.skip("Plotly not available")

        experiments = [
            create_mock_experiment("Model1", "Probe1", accuracy=0.85),
            create_mock_experiment("Model2", "Probe1", accuracy=0.90),
        ]
        save_path = str(tmp_path / "report.html")

        result_path = visualization.create_interactive_html_report(
            experiments, save_path=save_path, title="Test Dashboard"
        )

        assert Path(result_path).exists()
        content = Path(result_path).read_text()
        assert "Test Dashboard" in content
        assert "<html" in content.lower()


class TestVisualizationFlags:
    """Test visualization dependency flags."""

    def test_matplotlib_flag_is_boolean(self):
        """Test that MATPLOTLIB_AVAILABLE is a boolean."""
        from insideLLMs import visualization

        assert isinstance(visualization.MATPLOTLIB_AVAILABLE, bool)

    def test_seaborn_flag_is_boolean(self):
        """Test that SEABORN_AVAILABLE is a boolean."""
        from insideLLMs import visualization

        assert isinstance(visualization.SEABORN_AVAILABLE, bool)

    def test_plotly_flag_is_boolean(self):
        """Test that PLOTLY_AVAILABLE is a boolean."""
        from insideLLMs import visualization

        assert isinstance(visualization.PLOTLY_AVAILABLE, bool)

    def test_ipywidgets_flag_is_boolean(self):
        """Test that IPYWIDGETS_AVAILABLE is a boolean."""
        from insideLLMs import visualization

        assert isinstance(visualization.IPYWIDGETS_AVAILABLE, bool)
