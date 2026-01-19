"""Visualization tools for probe results and experiment analysis.

This module provides tools for visualizing:
- Experiment results and comparisons
- Statistical analysis visualizations
- Text-based charts (no dependencies required)
- Rich plots with matplotlib (optional)
"""

from typing import Any, Optional

from insideLLMs.types import ExperimentResult

try:
    import matplotlib.pyplot as plt
    import pandas as pd

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import ipywidgets as widgets
    from IPython.display import HTML, display

    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False


def check_visualization_deps() -> None:
    """Check if visualization dependencies are available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Visualization dependencies not installed. "
            "Please install with: pip install matplotlib pandas"
        )


def check_plotly_deps() -> None:
    """Check if Plotly dependencies are available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly dependencies not installed. Please install with: pip install plotly"
        )
    # Plotly express requires pandas for DataFrame operations
    if not MATPLOTLIB_AVAILABLE:
        # When matplotlib is not available, pd is not imported
        # but we need it for interactive visualizations
        global pd
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for interactive visualizations. "
                "Please install with: pip install pandas"
            )


def check_ipywidgets_deps() -> None:
    """Check if ipywidgets dependencies are available."""
    if not IPYWIDGETS_AVAILABLE:
        raise ImportError(
            "ipywidgets dependencies not installed. Please install with: pip install ipywidgets"
        )


# Text-based visualization (no dependencies)


def text_bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    width: int = 50,
    show_values: bool = True,
    char: str = "█",
) -> str:
    """Create a text-based horizontal bar chart.

    Args:
        labels: Labels for each bar.
        values: Values for each bar.
        title: Chart title.
        width: Maximum width of bars in characters.
        show_values: Whether to show values after bars.
        char: Character to use for bars.

    Returns:
        String containing the text chart.
    """
    if not values:
        return "No data to display"

    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

    max_val = max(values) if values else 1
    max_label_len = max(len(str(l)) for l in labels) if labels else 0

    for label, value in zip(labels, values):
        bar_len = int((value / max_val) * width) if max_val > 0 else 0
        bar = char * bar_len
        label_str = str(label).ljust(max_label_len)

        if show_values:
            lines.append(f"{label_str} | {bar} {value:.2f}")
        else:
            lines.append(f"{label_str} | {bar}")

    return "\n".join(lines)


def text_histogram(
    values: list[float],
    bins: int = 10,
    title: str = "",
    width: int = 40,
    char: str = "█",
) -> str:
    """Create a text-based histogram.

    Args:
        values: Data values.
        bins: Number of bins.
        title: Chart title.
        width: Maximum width of bars.
        char: Character for bars.

    Returns:
        String containing the text histogram.
    """
    if not values:
        return "No data to display"

    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        return f"All values equal: {min_val}"

    bin_width = (max_val - min_val) / bins
    bin_counts = [0] * bins

    for v in values:
        bin_idx = min(int((v - min_val) / bin_width), bins - 1)
        bin_counts[bin_idx] += 1

    max_count = max(bin_counts)
    lines = []

    if title:
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

    for i, count in enumerate(bin_counts):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = char * bar_len
        lines.append(f"[{bin_start:6.2f} - {bin_end:6.2f}] | {bar} ({count})")

    return "\n".join(lines)


def text_comparison_table(
    rows: list[str],
    cols: list[str],
    values: list[list[Any]],
    title: str = "",
) -> str:
    """Create a text-based comparison table.

    Args:
        rows: Row labels.
        cols: Column labels.
        values: 2D list of values (rows x cols).
        title: Table title.

    Returns:
        String containing the text table.
    """
    if not rows or not cols or not values:
        return "No data to display"

    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

    # Calculate column widths
    col_widths = [max(len(str(c)), 8) for c in cols]
    row_width = max(len(str(r)) for r in rows)

    # Update col widths based on values
    for row_vals in values:
        for i, val in enumerate(row_vals):
            col_widths[i] = max(
                col_widths[i], len(f"{val:.4f}" if isinstance(val, float) else str(val))
            )

    # Header
    header = " " * (row_width + 1) + " | ".join(
        str(c).center(col_widths[i]) for i, c in enumerate(cols)
    )
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for row_label, row_vals in zip(rows, values):
        cells = []
        for i, val in enumerate(row_vals):
            if isinstance(val, float):
                cells.append(f"{val:.4f}".center(col_widths[i]))
            else:
                cells.append(str(val).center(col_widths[i]))
        lines.append(f"{str(row_label).ljust(row_width)} | " + " | ".join(cells))

    return "\n".join(lines)


def text_summary_stats(
    name: str,
    mean: float,
    std: float,
    min_val: float,
    max_val: float,
    n: int,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
) -> str:
    """Create a text summary of statistics.

    Args:
        name: Name of the metric.
        mean: Mean value.
        std: Standard deviation.
        min_val: Minimum value.
        max_val: Maximum value.
        n: Sample size.
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.

    Returns:
        Formatted statistics string.
    """
    lines = [
        f"Statistics for: {name}",
        "-" * (15 + len(name)),
        f"  N:     {n}",
        f"  Mean:  {mean:.4f}",
        f"  Std:   {std:.4f}",
        f"  Min:   {min_val:.4f}",
        f"  Max:   {max_val:.4f}",
    ]

    if ci_lower is not None and ci_upper is not None:
        lines.append(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    return "\n".join(lines)


def experiment_summary_text(experiment: ExperimentResult) -> str:
    """Create a text summary of an experiment result.

    Args:
        experiment: The experiment result.

    Returns:
        Formatted text summary.
    """
    lines = [
        f"Experiment: {experiment.experiment_id}",
        "=" * (13 + len(experiment.experiment_id)),
        "",
        "Model:",
        f"  Name: {experiment.model_info.name}",
        f"  Provider: {experiment.model_info.provider}",
        f"  ID: {experiment.model_info.model_id}",
        "",
        f"Probe: {experiment.probe_name} ({experiment.probe_category.value})",
        "",
        "Results:",
        f"  Total:     {experiment.total_count}",
        f"  Success:   {experiment.success_count}",
        f"  Errors:    {experiment.error_count}",
        f"  Rate:      {experiment.success_rate * 100:.1f}%",
        "",
    ]

    if experiment.score:
        lines.append("Scores:")
        if experiment.score.accuracy is not None:
            lines.append(f"  Accuracy:  {experiment.score.accuracy * 100:.1f}%")
        if experiment.score.precision is not None:
            lines.append(f"  Precision: {experiment.score.precision * 100:.1f}%")
        if experiment.score.recall is not None:
            lines.append(f"  Recall:    {experiment.score.recall * 100:.1f}%")
        if experiment.score.f1_score is not None:
            lines.append(f"  F1 Score:  {experiment.score.f1_score:.4f}")
        if experiment.score.mean_latency_ms is not None:
            lines.append(f"  Latency:   {experiment.score.mean_latency_ms:.1f} ms")
        lines.append("")

    if experiment.duration_seconds:
        lines.append(f"Duration: {experiment.duration_seconds:.2f} seconds")

    return "\n".join(lines)


# Matplotlib-based visualization (requires matplotlib)


def plot_accuracy_comparison(
    experiments: list[ExperimentResult],
    title: str = "Model Accuracy Comparison",
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """Plot accuracy comparison across experiments.

    Args:
        experiments: List of experiment results.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the plot.
    """
    check_visualization_deps()

    model_names = []
    accuracies = []

    for exp in experiments:
        if exp.score and exp.score.accuracy is not None:
            model_names.append(f"{exp.model_info.name}\n({exp.probe_name})")
            accuracies.append(exp.score.accuracy * 100)

    if not model_names:
        print("No accuracy data to plot")
        return

    plt.figure(figsize=figsize)
    bars = plt.bar(model_names, accuracies, color="steelblue", edgecolor="navy")

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Model / Probe", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_latency_distribution(
    experiments: list[ExperimentResult],
    title: str = "Response Latency Distribution",
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """Plot latency distribution across experiments.

    Args:
        experiments: List of experiment results.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the plot.
    """
    check_visualization_deps()

    latencies_by_model = {}

    for exp in experiments:
        name = exp.model_info.name
        if name not in latencies_by_model:
            latencies_by_model[name] = []

        for result in exp.results:
            if result.latency_ms is not None:
                latencies_by_model[name].append(result.latency_ms)

    if not latencies_by_model:
        print("No latency data to plot")
        return

    plt.figure(figsize=figsize)

    if SEABORN_AVAILABLE:
        # Use seaborn for better box plots
        data = []
        for model, lats in latencies_by_model.items():
            for lat in lats:
                data.append({"Model": model, "Latency (ms)": lat})
        df = pd.DataFrame(data)
        sns.boxplot(x="Model", y="Latency (ms)", data=df)
    else:
        # Simple box plot with matplotlib
        plt.boxplot(
            latencies_by_model.values(),
            labels=latencies_by_model.keys(),
        )
        plt.ylabel("Latency (ms)")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_metric_comparison(
    experiments: list[ExperimentResult],
    metrics: list[str] = None,
    title: str = "Metric Comparison",
    figsize: tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """Plot multiple metrics for comparison.

    Args:
        experiments: List of experiment results.
        metrics: List of metric names to plot.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the plot.
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1_score"]
    check_visualization_deps()

    model_data = {}

    for exp in experiments:
        name = exp.model_info.name
        if name not in model_data:
            model_data[name] = {}

        if exp.score:
            for metric in metrics:
                value = getattr(exp.score, metric, None)
                if value is not None:
                    if metric not in model_data[name]:
                        model_data[name][metric] = []
                    model_data[name][metric].append(value * 100 if metric != "f1_score" else value)

    if not model_data:
        print("No metric data to plot")
        return

    # Prepare data for plotting
    models = list(model_data.keys())
    n_models = len(models)
    n_metrics = len(metrics)
    bar_width = 0.8 / n_metrics

    plt.figure(figsize=figsize)

    for i, metric in enumerate(metrics):
        values = []
        for model in models:
            model_metrics = model_data[model].get(metric, [0])
            values.append(sum(model_metrics) / len(model_metrics) if model_metrics else 0)

        x = [j + i * bar_width for j in range(n_models)]
        plt.bar(x, values, bar_width, label=metric.replace("_", " ").title())

    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(
        [j + bar_width * (n_metrics - 1) / 2 for j in range(n_models)],
        models,
        rotation=45,
        ha="right",
    )
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_success_rate_over_time(
    results: list[tuple[str, float]],
    title: str = "Success Rate Over Time",
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """Plot success rate over time/runs.

    Args:
        results: List of (timestamp/label, success_rate) tuples.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the plot.
    """
    check_visualization_deps()

    if not results:
        print("No data to plot")
        return

    labels, rates = zip(*results)

    plt.figure(figsize=figsize)
    plt.plot(labels, [r * 100 for r in rates], marker="o", linewidth=2, markersize=8)
    plt.fill_between(range(len(labels)), [r * 100 for r in rates], alpha=0.3)

    plt.xlabel("Run / Time", fontsize=12)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# Legacy functions for backward compatibility


def plot_bias_results(
    results: list[dict[str, Any]],
    title: str = "Bias Probe Results",
    save_path: Optional[str] = None,
) -> None:
    """Plot results from BiasProbe.

    Args:
        results: List of results from BiasProbe.
        title: Plot title.
        save_path: Path to save the plot.
    """
    check_visualization_deps()

    data = []
    for i, result in enumerate(results):
        for j, (response1, response2) in enumerate(result.get("output", [])):
            data.append(
                {
                    "Prompt Pair": f"Pair {i + 1}.{j + 1}",
                    "Response 1 Length": len(response1),
                    "Response 2 Length": len(response2),
                    "Length Difference": len(response1) - len(response2),
                }
            )

    if not data:
        print("No bias data to plot")
        return

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    if SEABORN_AVAILABLE:
        sns.barplot(x="Prompt Pair", y="Length Difference", data=df)
    else:
        plt.bar(df["Prompt Pair"], df["Length Difference"])
        plt.xlabel("Prompt Pair")
        plt.ylabel("Length Difference")

    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_factuality_results(
    results: list[dict[str, Any]],
    title: str = "Factuality Probe Results",
    save_path: Optional[str] = None,
) -> None:
    """Plot results from FactualityProbe.

    Args:
        results: List of results from FactualityProbe.
        title: Plot title.
        save_path: Path to save the plot.
    """
    check_visualization_deps()

    categories: dict[str, dict[str, Any]] = {}
    for result in results:
        for item in result.get("output", []):
            category = item.get("category", "general")
            if category not in categories:
                categories[category] = {"total": 0, "questions": []}
            categories[category]["total"] += 1
            categories[category]["questions"].append(item)

    if not categories:
        print("No factuality data to plot")
        return

    plt.figure(figsize=(12, 6))

    category_names = list(categories.keys())
    category_counts = [categories[cat]["total"] for cat in category_names]

    plt.subplot(1, 2, 1)
    if SEABORN_AVAILABLE:
        sns.barplot(x=category_names, y=category_counts)
    else:
        plt.bar(category_names, category_counts)
    plt.title(f"{title} - Categories")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    response_data = []
    for cat, data in categories.items():
        for q in data["questions"]:
            response_data.append(
                {"Category": cat, "Response Length": len(q.get("model_answer", ""))}
            )

    df = pd.DataFrame(response_data)
    if SEABORN_AVAILABLE:
        sns.boxplot(x="Category", y="Response Length", data=df)
    else:
        by_cat = {}
        for row in response_data:
            cat = row["Category"]
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(row["Response Length"])
        plt.boxplot(by_cat.values(), labels=by_cat.keys())

    plt.title(f"{title} - Response Lengths")
    plt.xlabel("Category")
    plt.ylabel("Response Length")
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_html_report(
    results: list[dict[str, Any]],
    title: str = "Probe Results Report",
    save_path: str = "report.html",
) -> str:
    """Create an HTML report from probe results.

    Args:
        results: List of results from any probe.
        title: Report title.
        save_path: Path to save the HTML report.

    Returns:
        Path to the saved report.
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f6fa; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; }}
            .result {{ background: white; border: 1px solid #ddd; padding: 15px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .input {{ color: #2980b9; font-weight: bold; }}
            .output {{ color: #27ae60; }}
            .error {{ color: #e74c3c; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background: linear-gradient(135deg, #3498db, #2980b9); color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .summary h2 {{ color: white; border: none; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
    """

    # Add summary
    error_count = sum(1 for r in results if r.get("error"))
    success_count = len(results) - error_count
    html += f"""
        <div class='summary'>
            <h2>Summary</h2>
            <p><strong>Total Results:</strong> {len(results)}</p>
            <p><strong>Successful:</strong> {success_count}</p>
            <p><strong>Errors:</strong> {error_count}</p>
        </div>
    """

    # Add results
    html += "<h2>Results</h2>"
    for i, result in enumerate(results):
        html += "<div class='result'>"
        html += f"<h3>Result {i + 1}</h3>"

        if "input" in result:
            html += f"<div class='input'><strong>Input:</strong> {result['input']}</div>"

        if "output" in result:
            output = result["output"]
            html += "<div class='output'><strong>Output:</strong></div>"

            if isinstance(output, list):
                html += "<table><tr><th>#</th><th>Output</th></tr>"
                for j, item in enumerate(output):
                    if isinstance(item, tuple):
                        html += f"<tr><td>{j + 1}</td><td>{item[0]}<br><em>vs.</em><br>{item[1]}</td></tr>"
                    else:
                        html += f"<tr><td>{j + 1}</td><td>{item}</td></tr>"
                html += "</table>"
            elif isinstance(output, dict):
                html += "<table><tr><th>Key</th><th>Value</th></tr>"
                for key, value in output.items():
                    html += f"<tr><td>{key}</td><td>{value}</td></tr>"
                html += "</table>"
            else:
                html += f"<p>{output}</p>"

        if "error" in result and result["error"]:
            html += f"<div class='error'><strong>Error:</strong> {result['error']}</div>"

        html += "</div>"

    html += """
    </body>
    </html>
    """

    with open(save_path, "w") as f:
        f.write(html)

    return save_path


# Interactive Plotly-based visualization


def interactive_accuracy_comparison(
    experiments: list[ExperimentResult],
    title: str = "Model Accuracy Comparison",
    show_error_bars: bool = True,
    color_by: str = "model",
) -> Any:
    """Create an interactive bar chart comparing model accuracies.

    Args:
        experiments: List of experiment results.
        title: Chart title.
        show_error_bars: Whether to show error bars (if std available).
        color_by: Color bars by 'model' or 'probe'.

    Returns:
        Plotly figure object.

    Example:
        >>> fig = interactive_accuracy_comparison(experiments)
        >>> fig.show()  # In Jupyter
        >>> fig.write_html("accuracy.html")  # Save to file
    """
    check_plotly_deps()

    data = []
    for exp in experiments:
        if exp.score and exp.score.accuracy is not None:
            data.append(
                {
                    "Model": exp.model_info.name,
                    "Probe": exp.probe_name,
                    "Accuracy": exp.score.accuracy * 100,
                    "Provider": exp.model_info.provider,
                    "Success Rate": exp.success_rate * 100,
                }
            )

    if not data:
        raise ValueError("No accuracy data available in experiments")

    df = pd.DataFrame(data)

    color_col = "Model" if color_by == "model" else "Probe"
    fig = px.bar(
        df,
        x="Model" if color_by == "model" else "Probe",
        y="Accuracy",
        color=color_col,
        barmode="group",
        title=title,
        hover_data=["Provider", "Success Rate"],
        text="Accuracy",
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 105],
        showlegend=True,
        hovermode="x unified",
    )

    return fig


def interactive_latency_distribution(
    experiments: list[ExperimentResult],
    title: str = "Response Latency Distribution",
    chart_type: str = "box",
) -> Any:
    """Create an interactive latency distribution chart.

    Args:
        experiments: List of experiment results.
        title: Chart title.
        chart_type: Type of chart - 'box', 'violin', or 'histogram'.

    Returns:
        Plotly figure object.

    Example:
        >>> fig = interactive_latency_distribution(experiments, chart_type="violin")
        >>> fig.show()
    """
    check_plotly_deps()

    data = []
    for exp in experiments:
        for result in exp.results:
            if result.latency_ms is not None:
                data.append(
                    {
                        "Model": exp.model_info.name,
                        "Probe": exp.probe_name,
                        "Latency (ms)": result.latency_ms,
                        "Status": result.status.value
                        if hasattr(result.status, "value")
                        else str(result.status),
                    }
                )

    if not data:
        raise ValueError("No latency data available in experiments")

    df = pd.DataFrame(data)

    if chart_type == "box":
        fig = px.box(
            df,
            x="Model",
            y="Latency (ms)",
            color="Probe",
            title=title,
            points="outliers",
        )
    elif chart_type == "violin":
        fig = px.violin(
            df,
            x="Model",
            y="Latency (ms)",
            color="Probe",
            title=title,
            box=True,
            points="outliers",
        )
    else:  # histogram
        fig = px.histogram(
            df,
            x="Latency (ms)",
            color="Model",
            title=title,
            barmode="overlay",
            opacity=0.7,
            marginal="box",
        )

    fig.update_layout(
        hovermode="x unified",
        showlegend=True,
    )

    return fig


def interactive_metric_radar(
    experiments: list[ExperimentResult],
    metrics: Optional[list[str]] = None,
    title: str = "Model Performance Radar",
) -> Any:
    """Create an interactive radar/spider chart for multi-metric comparison.

    Args:
        experiments: List of experiment results.
        metrics: List of metrics to include. Defaults to common metrics.
        title: Chart title.

    Returns:
        Plotly figure object.

    Example:
        >>> fig = interactive_metric_radar(experiments, metrics=["accuracy", "precision", "recall"])
        >>> fig.show()
    """
    check_plotly_deps()

    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1_score"]

    fig = go.Figure()

    model_metrics = {}
    for exp in experiments:
        name = exp.model_info.name
        if name not in model_metrics:
            model_metrics[name] = {m: [] for m in metrics}

        if exp.score:
            for metric in metrics:
                value = getattr(exp.score, metric, None)
                if value is not None:
                    # Convert to percentage for display (except f1_score)
                    display_value = value * 100 if metric != "f1_score" else value * 100
                    model_metrics[name][metric].append(display_value)

    for model_name, metric_values in model_metrics.items():
        values = []
        for metric in metrics:
            vals = metric_values.get(metric, [])
            values.append(sum(vals) / len(vals) if vals else 0)

        # Close the radar by repeating first value
        values_closed = values + [values[0]]
        metrics_closed = metrics + [metrics[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=[m.replace("_", " ").title() for m in metrics_closed],
                fill="toself",
                name=model_name,
                opacity=0.7,
            )
        )

    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
            }
        },
        showlegend=True,
        title=title,
    )

    return fig


def interactive_timeline(
    experiments: list[ExperimentResult],
    metric: str = "accuracy",
    title: str = "Performance Over Time",
) -> Any:
    """Create an interactive timeline of experiment results.

    Args:
        experiments: List of experiment results (sorted by time).
        metric: Metric to plot over time.
        title: Chart title.

    Returns:
        Plotly figure object.
    """
    check_plotly_deps()

    data = []
    for i, exp in enumerate(experiments):
        if exp.score:
            value = getattr(exp.score, metric, None)
            if value is not None:
                data.append(
                    {
                        "Index": i,
                        "Model": exp.model_info.name,
                        "Probe": exp.probe_name,
                        metric.replace("_", " ").title(): value * 100
                        if metric != "f1_score"
                        else value,
                        "Timestamp": exp.timestamp if hasattr(exp, "timestamp") else f"Run {i + 1}",
                    }
                )

    if not data:
        raise ValueError(f"No {metric} data available in experiments")

    df = pd.DataFrame(data)
    y_col = metric.replace("_", " ").title()

    fig = px.line(
        df,
        x="Index",
        y=y_col,
        color="Model",
        symbol="Probe",
        title=title,
        markers=True,
        hover_data=["Probe", "Timestamp"],
    )

    fig.update_layout(
        xaxis_title="Experiment Run",
        yaxis_title=f"{y_col} (%)",
        hovermode="x unified",
    )

    return fig


def interactive_heatmap(
    experiments: list[ExperimentResult],
    row_key: str = "model",
    col_key: str = "probe",
    value_key: str = "accuracy",
    title: str = "Performance Heatmap",
) -> Any:
    """Create an interactive heatmap of experiment results.

    Args:
        experiments: List of experiment results.
        row_key: What to use for rows ('model' or 'probe').
        col_key: What to use for columns ('model' or 'probe').
        value_key: Metric to display in cells.
        title: Chart title.

    Returns:
        Plotly figure object.

    Example:
        >>> fig = interactive_heatmap(experiments, row_key="model", col_key="probe")
        >>> fig.show()
    """
    check_plotly_deps()

    # Build matrix data
    row_values = set()
    col_values = set()
    matrix_data = {}

    for exp in experiments:
        row_val = exp.model_info.name if row_key == "model" else exp.probe_name
        col_val = exp.probe_name if col_key == "probe" else exp.model_info.name

        row_values.add(row_val)
        col_values.add(col_val)

        if exp.score:
            value = getattr(exp.score, value_key, None)
            if value is not None:
                key = (row_val, col_val)
                if key not in matrix_data:
                    matrix_data[key] = []
                matrix_data[key].append(value * 100)

    if not matrix_data:
        raise ValueError(f"No {value_key} data available in experiments")

    rows = sorted(row_values)
    cols = sorted(col_values)

    z_data = []
    text_data = []
    for row in rows:
        z_row = []
        text_row = []
        for col in cols:
            values = matrix_data.get((row, col), [])
            avg = sum(values) / len(values) if values else None
            z_row.append(avg)
            text_row.append(f"{avg:.1f}%" if avg is not None else "N/A")
        z_data.append(z_row)
        text_data.append(text_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=cols,
            y=rows,
            text=text_data,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale="RdYlGn",
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=col_key.title(),
        yaxis_title=row_key.title(),
    )

    return fig


def interactive_scatter_comparison(
    experiments: list[ExperimentResult],
    x_metric: str = "accuracy",
    y_metric: str = "mean_latency_ms",
    size_metric: Optional[str] = None,
    title: str = "Model Comparison",
) -> Any:
    """Create an interactive scatter plot comparing two metrics.

    Args:
        experiments: List of experiment results.
        x_metric: Metric for x-axis.
        y_metric: Metric for y-axis.
        size_metric: Optional metric for point size.
        title: Chart title.

    Returns:
        Plotly figure object.

    Example:
        >>> fig = interactive_scatter_comparison(
        ...     experiments,
        ...     x_metric="accuracy",
        ...     y_metric="mean_latency_ms",
        ...     size_metric="total_tokens"
        ... )
        >>> fig.show()
    """
    check_plotly_deps()

    data = []
    for exp in experiments:
        if exp.score:
            x_val = getattr(exp.score, x_metric, None)
            y_val = getattr(exp.score, y_metric, None)

            if x_val is not None and y_val is not None:
                entry = {
                    "Model": exp.model_info.name,
                    "Probe": exp.probe_name,
                    "Provider": exp.model_info.provider,
                    x_metric.replace("_", " ").title(): x_val * 100
                    if "rate" in x_metric or x_metric in ["accuracy", "precision", "recall"]
                    else x_val,
                    y_metric.replace("_", " ").title(): y_val,
                }

                if size_metric:
                    size_val = getattr(exp.score, size_metric, None)
                    if size_val is not None:
                        entry[size_metric.replace("_", " ").title()] = size_val

                data.append(entry)

    if not data:
        raise ValueError(f"No data available for {x_metric} and {y_metric}")

    df = pd.DataFrame(data)

    x_col = x_metric.replace("_", " ").title()
    y_col = y_metric.replace("_", " ").title()
    size_col = size_metric.replace("_", " ").title() if size_metric else None

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="Model",
        symbol="Probe",
        size=size_col if size_col and size_col in df.columns else None,
        title=title,
        hover_data=["Provider"],
    )

    fig.update_layout(
        hovermode="closest",
    )

    return fig


def interactive_sunburst(
    experiments: list[ExperimentResult],
    value_metric: str = "accuracy",
    title: str = "Experiment Breakdown",
) -> Any:
    """Create an interactive sunburst chart showing experiment hierarchy.

    Args:
        experiments: List of experiment results.
        value_metric: Metric to use for segment sizes.
        title: Chart title.

    Returns:
        Plotly figure object.

    Example:
        >>> fig = interactive_sunburst(experiments)
        >>> fig.show()
    """
    check_plotly_deps()

    data = []
    for exp in experiments:
        if exp.score:
            value = getattr(exp.score, value_metric, None)
            if value is not None:
                data.append(
                    {
                        "Provider": exp.model_info.provider,
                        "Model": exp.model_info.name,
                        "Probe": exp.probe_name,
                        "Value": value * 100
                        if value_metric in ["accuracy", "precision", "recall"]
                        else value,
                    }
                )

    if not data:
        raise ValueError(f"No {value_metric} data available")

    df = pd.DataFrame(data)

    fig = px.sunburst(
        df,
        path=["Provider", "Model", "Probe"],
        values="Value",
        title=title,
        color="Value",
        color_continuous_scale="RdYlGn",
    )

    return fig


def create_interactive_dashboard(
    experiments: list[ExperimentResult],
    title: str = "LLM Evaluation Dashboard",
    save_path: Optional[str] = None,
) -> Any:
    """Create a comprehensive interactive dashboard with multiple charts.

    Args:
        experiments: List of experiment results.
        title: Dashboard title.
        save_path: Optional path to save HTML file.

    Returns:
        Plotly figure object with subplots.

    Example:
        >>> fig = create_interactive_dashboard(experiments)
        >>> fig.show()
        >>> # Or save to file
        >>> fig.write_html("dashboard.html")
    """
    check_plotly_deps()

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Accuracy by Model",
            "Latency Distribution",
            "Metric Comparison",
            "Success Rate Trend",
        ),
        specs=[
            [{"type": "bar"}, {"type": "box"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
    )

    # Collect data
    model_accuracies = {}
    model_latencies = {}
    model_metrics = {}
    success_rates = []

    for i, exp in enumerate(experiments):
        name = exp.model_info.name
        success_rates.append((i, exp.success_rate * 100, name))

        if exp.score:
            if exp.score.accuracy is not None:
                if name not in model_accuracies:
                    model_accuracies[name] = []
                model_accuracies[name].append(exp.score.accuracy * 100)

            for metric in ["precision", "recall", "f1_score"]:
                val = getattr(exp.score, metric, None)
                if val is not None:
                    if name not in model_metrics:
                        model_metrics[name] = {}
                    if metric not in model_metrics[name]:
                        model_metrics[name][metric] = []
                    model_metrics[name][metric].append(val * 100)

        for result in exp.results:
            if result.latency_ms is not None:
                if name not in model_latencies:
                    model_latencies[name] = []
                model_latencies[name].append(result.latency_ms)

    # Chart 1: Accuracy bar chart
    for model, accs in model_accuracies.items():
        fig.add_trace(
            go.Bar(
                name=model,
                x=[model],
                y=[sum(accs) / len(accs)],
                text=[f"{sum(accs) / len(accs):.1f}%"],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Chart 2: Latency box plot
    for model, lats in model_latencies.items():
        fig.add_trace(
            go.Box(name=model, y=lats, showlegend=False),
            row=1,
            col=2,
        )

    # Chart 3: Grouped metric comparison
    metrics = ["precision", "recall", "f1_score"]
    x_labels = []
    for model in model_metrics:
        for metric in metrics:
            if metric in model_metrics.get(model, {}):
                vals = model_metrics[model][metric]
                x_labels.append(f"{model}\n{metric}")
                fig.add_trace(
                    go.Bar(
                        x=[f"{model}\n{metric}"],
                        y=[sum(vals) / len(vals)],
                        name=f"{model} {metric}",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

    # Chart 4: Success rate trend
    if success_rates:
        for model in sorted({sr[2] for sr in success_rates}):
            model_rates = [(sr[0], sr[1]) for sr in success_rates if sr[2] == model]
            fig.add_trace(
                go.Scatter(
                    x=[r[0] for r in model_rates],
                    y=[r[1] for r in model_rates],
                    mode="lines+markers",
                    name=model,
                ),
                row=2,
                col=2,
            )

    fig.update_layout(
        title_text=title,
        height=800,
        showlegend=True,
    )

    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Latency (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Score (%)", row=2, col=1)
    fig.update_yaxes(title_text="Success Rate (%)", row=2, col=2)
    fig.update_xaxes(title_text="Run", row=2, col=2)

    if save_path:
        fig.write_html(save_path)

    return fig


def create_interactive_html_report(
    experiments: list[ExperimentResult],
    title: str = "LLM Evaluation Report",
    save_path: str = "interactive_report.html",
    include_raw_results: bool = False,
) -> str:
    """Create a comprehensive interactive HTML report with embedded Plotly charts.

    Args:
        experiments: List of experiment results.
        title: Report title.
        save_path: Path to save the HTML file.
        include_raw_results: Whether to include raw result tables.

    Returns:
        Path to the saved report.

    Example:
        >>> path = create_interactive_html_report(experiments, "My Report", "report.html")
        >>> print(f"Report saved to {path}")
    """
    check_plotly_deps()

    # Generate charts
    charts_html = []

    try:
        acc_fig = interactive_accuracy_comparison(experiments)
        charts_html.append(
            ("Accuracy Comparison", acc_fig.to_html(full_html=False, include_plotlyjs=False))
        )
    except ValueError:
        pass

    try:
        lat_fig = interactive_latency_distribution(experiments)
        charts_html.append(
            ("Latency Distribution", lat_fig.to_html(full_html=False, include_plotlyjs=False))
        )
    except ValueError:
        pass

    try:
        radar_fig = interactive_metric_radar(experiments)
        charts_html.append(
            ("Performance Radar", radar_fig.to_html(full_html=False, include_plotlyjs=False))
        )
    except ValueError:
        pass

    try:
        heatmap_fig = interactive_heatmap(experiments)
        charts_html.append(
            ("Performance Heatmap", heatmap_fig.to_html(full_html=False, include_plotlyjs=False))
        )
    except ValueError:
        pass

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: white;
            text-align: center;
            padding: 20px;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .card h3 {{
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .card .value {{
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .chart-section {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .chart-section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .status-success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .status-error {{
            color: #e74c3c;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
"""

    # Summary cards
    total_experiments = len(experiments)
    total_results = sum(exp.total_count for exp in experiments)
    total_success = sum(exp.success_count for exp in experiments)
    avg_accuracy = 0
    acc_count = 0
    for exp in experiments:
        if exp.score and exp.score.accuracy is not None:
            avg_accuracy += exp.score.accuracy * 100
            acc_count += 1
    avg_accuracy = avg_accuracy / acc_count if acc_count > 0 else 0

    unique_models = len({exp.model_info.name for exp in experiments})

    html += f"""
        <div class="summary-cards">
            <div class="card">
                <h3>Experiments</h3>
                <div class="value">{total_experiments}</div>
            </div>
            <div class="card">
                <h3>Total Results</h3>
                <div class="value">{total_results}</div>
            </div>
            <div class="card">
                <h3>Success Rate</h3>
                <div class="value">{total_success / total_results * 100:.1f}%</div>
            </div>
            <div class="card">
                <h3>Avg Accuracy</h3>
                <div class="value">{avg_accuracy:.1f}%</div>
            </div>
            <div class="card">
                <h3>Models Tested</h3>
                <div class="value">{unique_models}</div>
            </div>
        </div>
"""

    # Charts
    for chart_title, chart_html in charts_html:
        html += f"""
        <div class="chart-section">
            <h2>{chart_title}</h2>
            {chart_html}
        </div>
"""

    # Results table
    if include_raw_results:
        html += """
        <div class="chart-section">
            <h2>Experiment Results</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Probe</th>
                    <th>Total</th>
                    <th>Success</th>
                    <th>Accuracy</th>
                    <th>Latency (ms)</th>
                </tr>
"""
        for exp in experiments:
            acc = f"{exp.score.accuracy * 100:.1f}%" if exp.score and exp.score.accuracy else "N/A"
            lat = (
                f"{exp.score.mean_latency_ms:.1f}"
                if exp.score and exp.score.mean_latency_ms
                else "N/A"
            )
            html += f"""
                <tr>
                    <td>{exp.model_info.name}</td>
                    <td>{exp.probe_name}</td>
                    <td>{exp.total_count}</td>
                    <td class="status-success">{exp.success_count}</td>
                    <td>{acc}</td>
                    <td>{lat}</td>
                </tr>
"""
        html += """
            </table>
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    with open(save_path, "w") as f:
        f.write(html)

    return save_path


# Jupyter notebook widgets for interactive exploration


class ExperimentExplorer:
    """Interactive widget-based explorer for experiment results in Jupyter notebooks.

    Example:
        >>> explorer = ExperimentExplorer(experiments)
        >>> explorer.show()  # Displays interactive widgets in Jupyter
    """

    def __init__(self, experiments: list[ExperimentResult]):
        """Initialize the explorer.

        Args:
            experiments: List of experiment results to explore.
        """
        check_ipywidgets_deps()
        check_plotly_deps()

        self.experiments = experiments
        self.models = sorted({exp.model_info.name for exp in experiments})
        self.probes = sorted({exp.probe_name for exp in experiments})

    def show(self) -> None:
        """Display the interactive explorer widgets."""
        # Model selector
        model_select = widgets.SelectMultiple(
            options=self.models,
            value=self.models[: min(3, len(self.models))],
            description="Models:",
            disabled=False,
        )

        # Probe selector
        probe_select = widgets.SelectMultiple(
            options=self.probes,
            value=self.probes[: min(3, len(self.probes))],
            description="Probes:",
            disabled=False,
        )

        # Chart type selector
        chart_type = widgets.Dropdown(
            options=[
                ("Accuracy Comparison", "accuracy"),
                ("Latency Distribution", "latency"),
                ("Metric Radar", "radar"),
                ("Performance Heatmap", "heatmap"),
                ("Scatter Comparison", "scatter"),
            ],
            value="accuracy",
            description="Chart:",
        )

        # Output area
        output = widgets.Output()

        def update_chart(change=None):
            selected_models = list(model_select.value)
            selected_probes = list(probe_select.value)

            filtered = [
                exp
                for exp in self.experiments
                if exp.model_info.name in selected_models and exp.probe_name in selected_probes
            ]

            if not filtered:
                with output:
                    output.clear_output()
                    print("No experiments match the selected filters.")
                return

            with output:
                output.clear_output()
                try:
                    if chart_type.value == "accuracy":
                        fig = interactive_accuracy_comparison(filtered)
                    elif chart_type.value == "latency":
                        fig = interactive_latency_distribution(filtered)
                    elif chart_type.value == "radar":
                        fig = interactive_metric_radar(filtered)
                    elif chart_type.value == "heatmap":
                        fig = interactive_heatmap(filtered)
                    elif chart_type.value == "scatter":
                        fig = interactive_scatter_comparison(filtered)
                    else:
                        fig = interactive_accuracy_comparison(filtered)

                    fig.show()
                except ValueError as e:
                    print(f"Cannot create chart: {e}")

        # Connect observers
        model_select.observe(update_chart, names="value")
        probe_select.observe(update_chart, names="value")
        chart_type.observe(update_chart, names="value")

        # Layout
        controls = widgets.HBox(
            [
                widgets.VBox([model_select, probe_select]),
                chart_type,
            ]
        )

        display(widgets.VBox([controls, output]))
        update_chart()  # Initial render

    def compare_models(
        self,
        metric: str = "accuracy",
        aggregate: str = "mean",
    ) -> Any:
        """Create a comparison table of models across probes.

        Args:
            metric: Metric to compare.
            aggregate: Aggregation method ('mean', 'max', 'min').

        Returns:
            pandas DataFrame with comparison data.
        """
        data = {}
        for exp in self.experiments:
            model = exp.model_info.name
            probe = exp.probe_name

            if model not in data:
                data[model] = {}

            if exp.score:
                value = getattr(exp.score, metric, None)
                if value is not None:
                    if probe not in data[model]:
                        data[model][probe] = []
                    data[model][probe].append(
                        value * 100 if metric in ["accuracy", "precision", "recall"] else value
                    )

        # Aggregate
        agg_func = {
            "mean": lambda x: sum(x) / len(x) if x else None,
            "max": lambda x: max(x) if x else None,
            "min": lambda x: min(x) if x else None,
        }.get(aggregate, lambda x: sum(x) / len(x) if x else None)

        result = {}
        for model, probes in data.items():
            result[model] = {probe: agg_func(values) for probe, values in probes.items()}

        df = pd.DataFrame(result).T
        return df.style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=None)
