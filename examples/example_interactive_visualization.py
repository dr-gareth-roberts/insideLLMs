"""Example: Interactive Visualization

This example demonstrates how to use the interactive visualization
features powered by Plotly for exploring experiment results.

Note: Requires `pip install plotly pandas` for full functionality.
"""

import tempfile
from pathlib import Path

# Check for dependencies
try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Install with: pip install plotly")

from insideLLMs import (
    # Types for creating mock data
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)


def create_mock_experiments():
    """Create mock experiment results for demonstration."""
    experiments = []

    configs = [
        ("GPT-4", "openai", "LogicProbe", 0.92, 0.90, 0.94, 0.92, 120.0),
        ("GPT-4", "openai", "BiasProbe", 0.88, 0.86, 0.90, 0.88, 110.0),
        ("GPT-4", "openai", "FactualityProbe", 0.95, 0.93, 0.97, 0.95, 130.0),
        ("Claude-3", "anthropic", "LogicProbe", 0.90, 0.88, 0.92, 0.90, 100.0),
        ("Claude-3", "anthropic", "BiasProbe", 0.91, 0.89, 0.93, 0.91, 95.0),
        ("Claude-3", "anthropic", "FactualityProbe", 0.93, 0.91, 0.95, 0.93, 105.0),
        ("Llama-2", "meta", "LogicProbe", 0.78, 0.75, 0.81, 0.78, 80.0),
        ("Llama-2", "meta", "BiasProbe", 0.75, 0.72, 0.78, 0.75, 75.0),
        ("Llama-2", "meta", "FactualityProbe", 0.80, 0.77, 0.83, 0.80, 85.0),
    ]

    for model, provider, probe, acc, prec, rec, f1, latency in configs:
        results = []
        for i in range(100):
            status = ResultStatus.SUCCESS if i < int(acc * 100) else ResultStatus.ERROR
            results.append(
                ProbeResult(
                    input=f"test input {i}",
                    output=f"test output {i}" if status == ResultStatus.SUCCESS else "",
                    status=status,
                    latency_ms=latency + (i % 30) - 15,  # Add some variance
                    error="Test error" if status == ResultStatus.ERROR else None,
                )
            )

        experiments.append(
            ExperimentResult(
                experiment_id=f"exp_{model}_{probe}",
                model_info=ModelInfo(
                    name=model,
                    provider=provider,
                    model_id=f"{provider}-{model.lower()}",
                ),
                probe_name=probe,
                probe_category=ProbeCategory.LOGIC if "Logic" in probe else ProbeCategory.BIAS,
                results=results,
                score=ProbeScore(
                    accuracy=acc,
                    precision=prec,
                    recall=rec,
                    f1_score=f1,
                    mean_latency_ms=latency,
                ),
            )
        )

    return experiments


def interactive_accuracy_demo(experiments):
    """Demonstrate interactive accuracy comparison."""
    print("\n" + "=" * 60)
    print("Interactive Accuracy Comparison")
    print("=" * 60)

    from insideLLMs import interactive_accuracy_comparison

    # Create accuracy comparison chart
    fig = interactive_accuracy_comparison(experiments, color_by="model")
    print("Created accuracy comparison chart")
    print("  - Hover over bars for details")
    print("  - Click legend items to toggle models")

    # Save to HTML
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "accuracy_comparison.html"
        fig.write_html(str(path))
        print(f"  - Saved to: {path}")


def interactive_latency_demo(experiments):
    """Demonstrate interactive latency distribution."""
    print("\n" + "=" * 60)
    print("Interactive Latency Distribution")
    print("=" * 60)

    from insideLLMs import interactive_latency_distribution

    # Box plot
    fig = interactive_latency_distribution(experiments, chart_type="box")
    print("Created box plot of latency distribution")

    # Violin plot
    fig = interactive_latency_distribution(experiments, chart_type="violin")
    print("Created violin plot of latency distribution")

    # Histogram
    fig = interactive_latency_distribution(experiments, chart_type="histogram")
    print("Created histogram of latency distribution")


def interactive_radar_demo(experiments):
    """Demonstrate interactive radar chart."""
    print("\n" + "=" * 60)
    print("Interactive Metric Radar Chart")
    print("=" * 60)

    from insideLLMs import interactive_metric_radar

    fig = interactive_metric_radar(experiments)
    print("Created radar chart showing all metrics")
    print("  - Each model is a different polygon")
    print("  - Metrics shown: accuracy, precision, recall, f1_score")


def interactive_heatmap_demo(experiments):
    """Demonstrate interactive heatmap."""
    print("\n" + "=" * 60)
    print("Interactive Heatmap")
    print("=" * 60)

    from insideLLMs import interactive_heatmap

    fig = interactive_heatmap(experiments, row_key="model", col_key="probe")
    print("Created heatmap: models vs probes")
    print("  - Color intensity shows accuracy")
    print("  - Hover for exact values")


def interactive_scatter_demo(experiments):
    """Demonstrate interactive scatter comparison."""
    print("\n" + "=" * 60)
    print("Interactive Scatter Comparison")
    print("=" * 60)

    from insideLLMs import interactive_scatter_comparison

    fig = interactive_scatter_comparison(
        experiments,
        x_metric="accuracy",
        y_metric="mean_latency_ms",
    )
    print("Created scatter plot: accuracy vs latency")
    print("  - Each point is an experiment")
    print("  - Size can represent another metric")


def interactive_dashboard_demo(experiments):
    """Demonstrate interactive dashboard."""
    print("\n" + "=" * 60)
    print("Interactive Dashboard")
    print("=" * 60)

    from insideLLMs import create_interactive_dashboard

    fig = create_interactive_dashboard(experiments)
    print("Created comprehensive dashboard with:")
    print("  - Accuracy comparison bar chart")
    print("  - Latency distribution box plot")
    print("  - Metric radar chart")
    print("  - Performance heatmap")

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "dashboard.html"
        fig.write_html(str(path))
        print(f"  - Saved to: {path}")


def interactive_report_demo(experiments):
    """Demonstrate interactive HTML report generation."""
    print("\n" + "=" * 60)
    print("Interactive HTML Report")
    print("=" * 60)

    from insideLLMs import create_interactive_html_report

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "report.html"

        result_path = create_interactive_html_report(
            experiments,
            save_path=str(path),
            title="LLM Evaluation Report",
            include_raw_results=True,
        )

        print(f"Created comprehensive HTML report at: {result_path}")
        print("Report includes:")
        print("  - Summary statistics")
        print("  - Interactive charts")
        print("  - Raw results table")
        print("  - Model comparisons")


def show_text_visualizations():
    """Show text-based visualizations (no dependencies)."""
    print("\n" + "=" * 60)
    print("Text-Based Visualizations (No Dependencies)")
    print("=" * 60)

    from insideLLMs import (
        text_bar_chart,
        text_histogram,
        text_summary_stats,
        experiment_summary_text,
    )

    # Sample data
    data = {"GPT-4": 0.92, "Claude-3": 0.90, "Llama-2": 0.78}
    values = [100, 120, 110, 95, 105, 130, 115, 125, 108, 112]

    print("\nText Bar Chart:")
    print(text_bar_chart(data, title="Accuracy by Model"))

    print("\nText Histogram:")
    print(text_histogram(values, bins=5, title="Latency Distribution"))

    print("\nSummary Stats:")
    stats = text_summary_stats(values)
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Std Dev: {stats['std']:.2f}")


if __name__ == "__main__":
    if not PLOTLY_AVAILABLE:
        print("\n" + "=" * 60)
        print("Plotly not installed - showing text visualizations only")
        print("=" * 60)
        show_text_visualizations()
    else:
        experiments = create_mock_experiments()
        print(f"Created {len(experiments)} mock experiments for demonstration")

        interactive_accuracy_demo(experiments)
        interactive_latency_demo(experiments)
        interactive_radar_demo(experiments)
        interactive_heatmap_demo(experiments)
        interactive_scatter_demo(experiments)
        interactive_dashboard_demo(experiments)
        interactive_report_demo(experiments)
        show_text_visualizations()

    print("\n" + "=" * 60)
    print("Done! Run this script with Plotly installed to see all demos.")
    print("Install with: pip install plotly pandas")
    print("=" * 60)
