"""Visualization tools for probe results and experiment analysis.

This module provides a comprehensive suite of visualization tools for analyzing
and presenting LLM evaluation results from insideLLMs experiments. It supports
multiple output formats ranging from simple text-based charts (requiring no
external dependencies) to rich interactive dashboards using Plotly.

Overview
--------
The visualization module is organized into several categories:

**Text-Based Visualizations** (no dependencies):
    Simple ASCII charts that work in any terminal or console environment.
    Useful for quick inspection, logging, or environments without graphical
    capabilities.

**Matplotlib-Based Visualizations** (requires matplotlib, pandas):
    Static publication-quality plots for accuracy comparisons, latency
    distributions, metric comparisons, and time series analysis.

**Plotly Interactive Visualizations** (requires plotly, pandas):
    Rich interactive charts with hover tooltips, zooming, and export
    capabilities. Ideal for Jupyter notebooks and web-based reports.

**HTML Report Generation**:
    Self-contained HTML reports with embedded charts, filtering controls,
    and export functionality for sharing results.

**Jupyter Widgets** (requires ipywidgets):
    Interactive explorers for real-time filtering and chart type switching
    directly within Jupyter notebooks.

Dependency Availability
-----------------------
The module gracefully handles missing dependencies:

- ``MATPLOTLIB_AVAILABLE``: True if matplotlib and pandas are installed
- ``SEABORN_AVAILABLE``: True if seaborn is installed
- ``PLOTLY_AVAILABLE``: True if plotly is installed
- ``IPYWIDGETS_AVAILABLE``: True if ipywidgets and IPython are installed

Functions that require specific dependencies will raise ImportError with
helpful installation instructions if the dependencies are not available.

Examples
--------
Text-based visualization (no dependencies):

>>> from insideLLMs.analysis.visualization import text_bar_chart
>>> labels = ["GPT-4", "Claude-3", "Gemini"]
>>> values = [0.92, 0.89, 0.87]
>>> print(text_bar_chart(labels, values, title="Model Accuracy"))
Model Accuracy
==============
<BLANKLINE>
GPT-4    | ████████████████████████████████████████████████ 0.92
Claude-3 | ████████████████████████████████████████████ 0.89
Gemini   | ██████████████████████████████████████████ 0.87

Text-based histogram:

>>> from insideLLMs.analysis.visualization import text_histogram
>>> latencies = [100, 120, 95, 150, 110, 130, 105, 115, 140, 125]
>>> print(text_histogram(latencies, bins=5, title="Latency Distribution"))
Latency Distribution
====================
<BLANKLINE>
[ 95.00 - 106.00] | ████████████████████████████████████████ (3)
[106.00 - 117.00] | █████████████████████████████████████████████████████ (4)
[117.00 - 128.00] | █████████████████████████ (2)
[128.00 - 139.00] | █████████████ (1)
[139.00 - 150.00] | █████████████ (1)

Using matplotlib visualizations:

>>> from insideLLMs.analysis.visualization import plot_accuracy_comparison
>>> # Assuming 'experiments' is a list of ExperimentResult objects
>>> plot_accuracy_comparison(
...     experiments,
...     title="Q4 2024 Model Comparison",
...     figsize=(12, 6),
...     save_path="accuracy_comparison.png"
... )  # doctest: +SKIP

Creating interactive Plotly charts:

>>> from insideLLMs.analysis.visualization import interactive_accuracy_comparison
>>> fig = interactive_accuracy_comparison(experiments)  # doctest: +SKIP
>>> fig.show()  # Opens in browser or displays in Jupyter  # doctest: +SKIP
>>> fig.write_html("accuracy_chart.html")  # Save as standalone HTML  # doctest: +SKIP

Creating a comprehensive HTML report:

>>> from insideLLMs.analysis.visualization import create_interactive_html_report
>>> path = create_interactive_html_report(
...     experiments,
...     title="LLM Evaluation Report - January 2024",
...     save_path="evaluation_report.html",
...     include_individual_results=True
... )  # doctest: +SKIP
>>> print(f"Report saved to: {path}")  # doctest: +SKIP

Using the Jupyter widget explorer:

>>> from insideLLMs.analysis.visualization import ExperimentExplorer
>>> explorer = ExperimentExplorer(experiments)  # doctest: +SKIP
>>> explorer.show()  # Displays interactive widgets  # doctest: +SKIP
>>> # Or get a comparison DataFrame
>>> df = explorer.compare_models(metric="accuracy")  # doctest: +SKIP

Working with experiment summaries:

>>> from insideLLMs.analysis.visualization import experiment_summary_text
>>> summary = experiment_summary_text(experiments[0])  # doctest: +SKIP
>>> print(summary)  # doctest: +SKIP
Experiment: exp-2024-001
========================
<BLANKLINE>
Model:
  Name: GPT-4
  Provider: openai
  ID: gpt-4-turbo
<BLANKLINE>
Probe: factuality_basic (factuality)
<BLANKLINE>
Results:
  Total:     100
  Success:   98
  Errors:    2
  Rate:      98.0%

Notes
-----
- For production deployments, prefer the text-based or static matplotlib
  visualizations as they have fewer dependencies and are more portable.

- Interactive HTML reports are self-contained and can be shared via email
  or uploaded to web servers without requiring a Python environment.

- When using Jupyter notebooks, the interactive Plotly charts and widget
  explorer provide the best user experience for data exploration.

- All chart functions accept a ``save_path`` parameter for saving output
  to files instead of displaying directly.

See Also
--------
insideLLMs.types.ExperimentResult : Data structure for experiment results
insideLLMs.analysis.statistics : Statistical analysis functions
insideLLMs.export : Functions for exporting results to various formats
"""

import hashlib
import html
import re
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from insideLLMs.types import ExperimentResult


def _escape_html_text(value: object) -> str:
    return html.escape(str(value), quote=False)


def _escape_html_attr(value: object) -> str:
    return html.escape(str(value), quote=True)


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
    from IPython.display import display

    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False


def check_visualization_deps() -> None:
    """Check if matplotlib visualization dependencies are available.

    This function verifies that matplotlib and pandas are installed,
    which are required for static plot generation functions like
    ``plot_accuracy_comparison`` and ``plot_latency_distribution``.

    Raises
    ------
    ImportError
        If matplotlib or pandas is not installed, with installation
        instructions in the error message.

    Examples
    --------
    Checking dependencies before using matplotlib functions:

    >>> from insideLLMs.analysis.visualization import check_visualization_deps
    >>> try:
    ...     check_visualization_deps()
    ...     print("Dependencies available!")
    ... except ImportError as e:
    ...     print(f"Missing dependencies: {e}")  # doctest: +SKIP
    Dependencies available!

    See Also
    --------
    check_plotly_deps : Check for Plotly dependencies
    check_ipywidgets_deps : Check for Jupyter widget dependencies
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Visualization dependencies not installed. "
            "Please install with: pip install matplotlib pandas"
        )


def check_plotly_deps() -> None:
    """Check if Plotly dependencies are available.

    This function verifies that Plotly is installed for interactive
    chart generation. It also ensures pandas is available since Plotly
    Express requires DataFrame operations.

    If matplotlib is not installed (and thus pandas was not imported
    during module initialization), this function will attempt to import
    pandas directly.

    Raises
    ------
    ImportError
        If Plotly is not installed, or if pandas is not available
        for DataFrame operations.

    Examples
    --------
    Checking dependencies before creating interactive charts:

    >>> from insideLLMs.analysis.visualization import check_plotly_deps
    >>> try:
    ...     check_plotly_deps()
    ...     print("Plotly ready!")
    ... except ImportError as e:
    ...     print(f"Install missing: {e}")  # doctest: +SKIP
    Plotly ready!

    Typical usage within an interactive visualization function:

    >>> def my_interactive_chart(data):
    ...     check_plotly_deps()  # Raises if Plotly unavailable
    ...     # ... create chart with plotly
    ...     pass

    See Also
    --------
    check_visualization_deps : Check for matplotlib dependencies
    check_ipywidgets_deps : Check for Jupyter widget dependencies
    interactive_accuracy_comparison : Example function using Plotly
    """
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
    """Check if Jupyter ipywidgets dependencies are available.

    This function verifies that ipywidgets and IPython are installed,
    which are required for the interactive ``ExperimentExplorer`` widget
    class in Jupyter notebooks.

    Raises
    ------
    ImportError
        If ipywidgets or IPython display functions are not available.

    Examples
    --------
    Checking before creating Jupyter widgets:

    >>> from insideLLMs.analysis.visualization import check_ipywidgets_deps
    >>> try:
    ...     check_ipywidgets_deps()
    ...     print("Widgets available!")
    ... except ImportError:
    ...     print("Not in Jupyter or widgets not installed")  # doctest: +SKIP
    Widgets available!

    Notes
    -----
    This check is automatically performed when instantiating
    ``ExperimentExplorer``. You typically don't need to call it
    directly unless building custom widget-based visualizations.

    See Also
    --------
    check_visualization_deps : Check for matplotlib dependencies
    check_plotly_deps : Check for Plotly dependencies
    ExperimentExplorer : Interactive experiment exploration widget
    """
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

    Generates an ASCII bar chart suitable for terminal output, logging,
    or environments without graphical capabilities. Bars are scaled
    proportionally to the maximum value.

    Parameters
    ----------
    labels : list[str]
        Labels for each bar. These appear on the left side of the chart.
        Labels are left-padded to align uniformly.
    values : list[float]
        Numeric values for each bar. Must have the same length as labels.
        Values are used to determine bar lengths proportionally.
    title : str, optional
        Chart title displayed at the top. If provided, an underline of
        equal signs is added below it. Default is empty string (no title).
    width : int, optional
        Maximum width of bars in characters. The bar with the largest
        value will have this width. Default is 50.
    show_values : bool, optional
        Whether to display numeric values after each bar. Default is True.
    char : str, optional
        Character used to draw the bars. Default is "█" (full block).
        Other options include "#", "*", "=", or any single character.

    Returns
    -------
    str
        Multi-line string containing the complete text chart, ready
        for printing or logging.

    Examples
    --------
    Basic usage with model accuracy scores:

    >>> labels = ["GPT-4", "Claude-3", "Gemini"]
    >>> values = [92.5, 89.3, 87.1]
    >>> chart = text_bar_chart(labels, values, title="Model Accuracy (%)")
    >>> print(chart)
    Model Accuracy (%)
    ==================
    <BLANKLINE>
    GPT-4    | ██████████████████████████████████████████████████ 92.50
    Claude-3 | ████████████████████████████████████████████████ 89.30
    Gemini   | ███████████████████████████████████████████████ 87.10

    Compact chart without values:

    >>> chart = text_bar_chart(["A", "B", "C"], [10, 8, 6], show_values=False, width=20)
    >>> print(chart)
    A | ████████████████████
    B | ████████████████
    C | ████████████

    Using a different bar character:

    >>> chart = text_bar_chart(["Test1", "Test2"], [75, 50], char="=", width=30)
    >>> print(chart)
    Test1 | ============================== 75.00
    Test2 | ==================== 50.00

    Handling empty data:

    >>> text_bar_chart([], [])
    'No data to display'

    Comparing latency across providers:

    >>> providers = ["OpenAI", "Anthropic", "Google", "Meta"]
    >>> latencies = [150.5, 120.3, 180.2, 95.7]
    >>> chart = text_bar_chart(providers, latencies, title="API Latency (ms)")
    >>> print(chart)  # doctest: +NORMALIZE_WHITESPACE
    API Latency (ms)
    ================
    <BLANKLINE>
    OpenAI    | █████████████████████████████████████████ 150.50
    Anthropic | █████████████████████████████████ 120.30
    Google    | ██████████████████████████████████████████████████ 180.20
    Meta      | ██████████████████████████ 95.70

    Notes
    -----
    - Bar lengths are calculated proportionally: ``bar_len = (value / max_value) * width``
    - If all values are zero or the max value is zero, all bars will be empty
    - Labels are automatically padded to align the bars vertically

    See Also
    --------
    text_histogram : Create a text-based histogram
    text_comparison_table : Create a text-based comparison table
    text_summary_stats : Create a text summary of statistics
    """
    if not values:
        return "No data to display"

    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

    max_val = max(values) if values else 1
    max_label_len = max(len(str(label)) for label in labels) if labels else 0

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

    Generates an ASCII histogram showing the distribution of values
    across specified number of bins. Useful for visualizing latency
    distributions, token counts, or any continuous numeric data.

    Parameters
    ----------
    values : list[float]
        Numeric data values to bin and display. The function calculates
        bin edges automatically based on the min and max values.
    bins : int, optional
        Number of bins (bars) in the histogram. Default is 10.
        More bins provide finer granularity but may be harder to read.
    title : str, optional
        Chart title displayed at the top. Default is empty string.
    width : int, optional
        Maximum width of histogram bars in characters. Default is 40.
    char : str, optional
        Character used to draw the bars. Default is "█".

    Returns
    -------
    str
        Multi-line string containing the histogram with bin ranges,
        bars, and counts.

    Examples
    --------
    Basic latency distribution:

    >>> latencies = [100, 120, 95, 150, 110, 130, 105, 115, 140, 125]
    >>> hist = text_histogram(latencies, bins=5, title="Response Latency")
    >>> print(hist)  # doctest: +NORMALIZE_WHITESPACE
    Response Latency
    ================
    <BLANKLINE>
    [ 95.00 - 106.00] | ████████████████████████████████████████ (3)
    [106.00 - 117.00] | ████████████████████████████████████████████████████ (4)
    [117.00 - 128.00] | ██████████████████████████ (2)
    [128.00 - 139.00] | █████████████ (1)
    [139.00 - 150.00] | █████████████ (1)

    Token usage distribution with more bins:

    >>> tokens = [150, 200, 175, 225, 180, 195, 210, 165, 190, 185,
    ...           170, 205, 188, 192, 178]
    >>> hist = text_histogram(tokens, bins=8, title="Token Count Distribution")
    >>> len(hist.split("\\n")) > 5  # Multiple lines generated
    True

    Handling edge cases:

    >>> text_histogram([])
    'No data to display'

    >>> text_histogram([5, 5, 5, 5])  # All equal values
    'All values equal: 5'

    Narrow distribution:

    >>> hist = text_histogram([1.0, 1.1, 1.2, 1.0, 1.1], bins=3, width=20)
    >>> "[" in hist  # Contains bin ranges
    True

    Notes
    -----
    - Bin edges are calculated as: ``bin_width = (max - min) / bins``
    - Values equal to the maximum are placed in the last bin
    - Counts are shown in parentheses after each bar
    - If all values are identical, returns a special message

    See Also
    --------
    text_bar_chart : Create a text-based bar chart with labels
    plot_latency_distribution : Create matplotlib latency distribution plot
    interactive_latency_distribution : Create interactive Plotly distribution
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

    Generates a formatted ASCII table for comparing multiple items
    across multiple dimensions. Ideal for model-vs-probe comparisons
    or multi-metric summaries.

    Parameters
    ----------
    rows : list[str]
        Row labels (e.g., model names). Displayed on the left side.
    cols : list[str]
        Column labels (e.g., metric names or probe names).
        Displayed in the header row.
    values : list[list[Any]]
        2D list of values where ``values[i][j]`` is the value for
        row i and column j. Must have dimensions len(rows) x len(cols).
        Floats are formatted to 4 decimal places automatically.
    title : str, optional
        Table title displayed above the header. Default is empty.

    Returns
    -------
    str
        Multi-line string containing the formatted table with
        aligned columns and separator lines.

    Examples
    --------
    Comparing model performance across probes:

    >>> models = ["GPT-4", "Claude-3", "Gemini"]
    >>> probes = ["Factuality", "Logic", "Bias"]
    >>> scores = [
    ...     [0.92, 0.88, 0.85],  # GPT-4 scores
    ...     [0.89, 0.91, 0.87],  # Claude-3 scores
    ...     [0.85, 0.83, 0.82],  # Gemini scores
    ... ]
    >>> table = text_comparison_table(models, probes, scores, "Model Comparison")
    >>> print(table)  # doctest: +NORMALIZE_WHITESPACE
    Model Comparison
    ================
    <BLANKLINE>
              Factuality  |   Logic    |    Bias
    --------------------------------------------------
    GPT-4    |   0.9200   |   0.8800   |   0.8500
    Claude-3 |   0.8900   |   0.9100   |   0.8700
    Gemini   |   0.8500   |   0.8300   |   0.8200

    Mixed data types (strings and numbers):

    >>> rows = ["Test A", "Test B"]
    >>> cols = ["Status", "Score", "Notes"]
    >>> values = [
    ...     ["PASS", 95.5, "Good"],
    ...     ["FAIL", 45.2, "Needs work"],
    ... ]
    >>> table = text_comparison_table(rows, cols, values)
    >>> "PASS" in table
    True

    Handling empty data:

    >>> text_comparison_table([], [], [])
    'No data to display'

    Latency comparison across providers:

    >>> providers = ["OpenAI", "Anthropic"]
    >>> metrics = ["P50 (ms)", "P95 (ms)", "P99 (ms)"]
    >>> latencies = [
    ...     [120.5, 250.3, 480.1],
    ...     [95.2, 180.7, 320.5],
    ... ]
    >>> table = text_comparison_table(providers, metrics, latencies, "API Latency")
    >>> "OpenAI" in table and "P50" in table
    True

    Notes
    -----
    - Column widths are automatically calculated based on header and data
    - Float values are formatted with 4 decimal places
    - Columns are center-aligned for readability
    - Row labels are left-aligned

    See Also
    --------
    text_bar_chart : Create a text-based bar chart
    text_summary_stats : Create a text summary of statistics
    interactive_heatmap : Create an interactive heatmap comparison
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

    Generates a formatted text block showing key statistical measures
    for a metric. Useful for quick inspection of experiment results
    or for inclusion in logs and reports.

    Parameters
    ----------
    name : str
        Name or description of the metric being summarized.
        Appears in the header line.
    mean : float
        Arithmetic mean of the data.
    std : float
        Standard deviation of the data.
    min_val : float
        Minimum value in the data.
    max_val : float
        Maximum value in the data.
    n : int
        Sample size (number of data points).
    ci_lower : float, optional
        Lower bound of 95% confidence interval for the mean.
        If provided along with ci_upper, the CI is displayed.
    ci_upper : float, optional
        Upper bound of 95% confidence interval for the mean.
        Both ci_lower and ci_upper must be provided to display CI.

    Returns
    -------
    str
        Multi-line string with formatted statistics.

    Examples
    --------
    Basic statistics summary:

    >>> stats = text_summary_stats(
    ...     name="Response Latency (ms)",
    ...     mean=125.5,
    ...     std=30.2,
    ...     min_val=85.0,
    ...     max_val=220.0,
    ...     n=100
    ... )
    >>> print(stats)
    Statistics for: Response Latency (ms)
    --------------------------------------
      N:     100
      Mean:  125.5000
      Std:   30.2000
      Min:   85.0000
      Max:   220.0000

    With confidence interval:

    >>> stats = text_summary_stats(
    ...     name="Accuracy",
    ...     mean=0.892,
    ...     std=0.045,
    ...     min_val=0.78,
    ...     max_val=0.96,
    ...     n=50,
    ...     ci_lower=0.879,
    ...     ci_upper=0.905
    ... )
    >>> "95% CI:" in stats
    True
    >>> print(stats)
    Statistics for: Accuracy
    ------------------------
      N:     50
      Mean:  0.8920
      Std:   0.0450
      Min:   0.7800
      Max:   0.9600
      95% CI: [0.8790, 0.9050]

    Summarizing token usage:

    >>> stats = text_summary_stats(
    ...     name="Token Count",
    ...     mean=450.0,
    ...     std=120.0,
    ...     min_val=150.0,
    ...     max_val=800.0,
    ...     n=200
    ... )
    >>> "450.0000" in stats
    True

    Notes
    -----
    - All numeric values are formatted to 4 decimal places
    - The header underline length adjusts to the metric name
    - Confidence interval is only shown if both bounds are provided

    See Also
    --------
    text_bar_chart : Create a text-based bar chart
    text_histogram : Create a text-based histogram
    experiment_summary_text : Create a full experiment summary
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

    Generates a comprehensive human-readable summary of an experiment
    including model information, probe details, result counts, and
    scoring metrics. Ideal for logging, console output, or quick review.

    Parameters
    ----------
    experiment : ExperimentResult
        The experiment result object (ProbeExperimentResult) containing
        model info, probe details, individual results, and aggregate scores.

    Returns
    -------
    str
        Multi-line formatted text summary of the experiment.

    Examples
    --------
    Summarizing a completed experiment:

    >>> from insideLLMs.types import (
    ...     ProbeExperimentResult, ProbeResult, ProbeScore,
    ...     ModelInfo, ProbeCategory, ResultStatus
    ... )
    >>> model = ModelInfo(name="GPT-4", provider="openai", model_id="gpt-4-turbo")
    >>> results = [
    ...     ProbeResult("q1", "a1", ResultStatus.SUCCESS, latency_ms=100),
    ...     ProbeResult("q2", "a2", ResultStatus.SUCCESS, latency_ms=120),
    ...     ProbeResult("q3", error="timeout", status=ResultStatus.ERROR),
    ... ]
    >>> experiment = ProbeExperimentResult(
    ...     experiment_id="exp-2024-001",
    ...     model_info=model,
    ...     probe_name="factuality_basic",
    ...     probe_category=ProbeCategory.FACTUALITY,
    ...     results=results,
    ...     score=ProbeScore(accuracy=0.67, precision=0.65, recall=0.70, mean_latency_ms=110.0)
    ... )
    >>> summary = experiment_summary_text(experiment)
    >>> print(summary)  # doctest: +NORMALIZE_WHITESPACE
    Experiment: exp-2024-001
    ========================
    <BLANKLINE>
    Model:
      Name: GPT-4
      Provider: openai
      ID: gpt-4-turbo
    <BLANKLINE>
    Probe: factuality_basic (factuality)
    <BLANKLINE>
    Results:
      Total:     3
      Success:   2
      Errors:    1
      Rate:      66.7%
    <BLANKLINE>
    Scores:
      Accuracy:  67.0%
      Precision: 65.0%
      Recall:    70.0%
      Latency:   110.0 ms

    Experiment with F1 score:

    >>> experiment.score.f1_score = 0.673
    >>> summary = experiment_summary_text(experiment)
    >>> "F1 Score:" in summary
    True

    Experiment with duration:

    >>> from datetime import datetime, timedelta
    >>> experiment.started_at = datetime(2024, 1, 15, 10, 0, 0)
    >>> experiment.completed_at = datetime(2024, 1, 15, 10, 0, 45)
    >>> summary = experiment_summary_text(experiment)
    >>> "Duration:" in summary
    True

    Notes
    -----
    - Success rate is calculated as success_count / total_count
    - Only non-None score fields are displayed
    - Duration is only shown if both started_at and completed_at are set
    - Percentages are displayed with one decimal place

    See Also
    --------
    text_summary_stats : Create statistical summaries
    create_html_report : Generate an HTML report
    create_interactive_html_report : Generate an interactive HTML report
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
    """Plot accuracy comparison across experiments using matplotlib.

    Creates a bar chart comparing accuracy scores across different
    models and probes. Each bar represents one experiment with the
    model name and probe name as the label.

    Parameters
    ----------
    experiments : list[ExperimentResult]
        List of experiment results to compare. Each experiment should
        have a score with an accuracy value. Experiments without
        accuracy scores are silently skipped.
    title : str, optional
        Plot title displayed at the top. Default is "Model Accuracy Comparison".
    figsize : tuple[int, int], optional
        Figure size as (width, height) in inches. Default is (10, 6).
    save_path : str, optional
        If provided, saves the plot to this file path instead of
        displaying it. Supports PNG, PDF, SVG based on extension.
        Default is None (display plot).

    Raises
    ------
    ImportError
        If matplotlib or pandas is not installed.

    Examples
    --------
    Basic accuracy comparison:

    >>> from insideLLMs.analysis.visualization import plot_accuracy_comparison
    >>> # Assuming experiments is a list of ExperimentResult objects
    >>> plot_accuracy_comparison(experiments, title="Q4 2024 Results")  # doctest: +SKIP

    Save to file instead of displaying:

    >>> plot_accuracy_comparison(
    ...     experiments,
    ...     title="Model Comparison",
    ...     figsize=(12, 8),
    ...     save_path="accuracy_comparison.png"
    ... )  # doctest: +SKIP

    Creating a comparison report:

    >>> # Run multiple experiments
    >>> gpt4_results = run_probe(gpt4_client, factuality_probe)  # doctest: +SKIP
    >>> claude_results = run_probe(claude_client, factuality_probe)  # doctest: +SKIP
    >>> plot_accuracy_comparison(
    ...     [gpt4_results, claude_results],
    ...     title="GPT-4 vs Claude-3 Factuality",
    ...     save_path="reports/factuality.pdf"
    ... )  # doctest: +SKIP

    Notes
    -----
    - Bars are colored in steel blue with navy edges
    - Value labels are displayed above each bar
    - Y-axis is fixed to 0-105% range for consistent comparison
    - X-axis labels are rotated 45 degrees for readability
    - If no experiments have accuracy data, prints a message and returns

    See Also
    --------
    interactive_accuracy_comparison : Interactive Plotly version
    plot_metric_comparison : Compare multiple metrics
    plot_latency_distribution : Plot latency distributions
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
    """Plot latency distribution across experiments using matplotlib.

    Creates box plots showing the distribution of response latencies
    for each model. Useful for comparing response time variability
    and identifying outliers across different LLM providers.

    Parameters
    ----------
    experiments : list[ExperimentResult]
        List of experiment results containing individual result latencies.
        Latencies are extracted from each result's ``latency_ms`` field.
        Results without latency data are skipped.
    title : str, optional
        Plot title. Default is "Response Latency Distribution".
    figsize : tuple[int, int], optional
        Figure size as (width, height) in inches. Default is (10, 6).
    save_path : str, optional
        If provided, saves the plot to this file path instead of
        displaying it. Default is None (display plot).

    Raises
    ------
    ImportError
        If matplotlib or pandas is not installed.

    Examples
    --------
    Basic latency distribution:

    >>> from insideLLMs.analysis.visualization import plot_latency_distribution
    >>> plot_latency_distribution(experiments, title="API Response Times")  # doctest: +SKIP

    Comparing multiple providers:

    >>> openai_results = run_probe(openai_client, test_probe)  # doctest: +SKIP
    >>> anthropic_results = run_probe(anthropic_client, test_probe)  # doctest: +SKIP
    >>> google_results = run_probe(google_client, test_probe)  # doctest: +SKIP
    >>> plot_latency_distribution(
    ...     [openai_results, anthropic_results, google_results],
    ...     title="Provider Latency Comparison",
    ...     figsize=(12, 6),
    ...     save_path="latency_comparison.png"
    ... )  # doctest: +SKIP

    High-resolution export for publication:

    >>> plot_latency_distribution(
    ...     experiments,
    ...     title="Response Time Analysis",
    ...     figsize=(8, 5),
    ...     save_path="figures/latency.pdf"
    ... )  # doctest: +SKIP

    Notes
    -----
    - Uses seaborn for enhanced box plots if available, falls back to matplotlib
    - Latencies are grouped by model name
    - X-axis labels are rotated 45 degrees for readability
    - If no latency data is available, prints a message and returns
    - Box plots show median, quartiles, and outliers

    See Also
    --------
    interactive_latency_distribution : Interactive Plotly version
    text_histogram : Text-based latency histogram
    plot_accuracy_comparison : Compare accuracy scores
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
    """Plot multiple metrics for comparison using grouped bar charts.

    Creates a grouped bar chart comparing multiple metrics (accuracy,
    precision, recall, F1 score) across different models. Useful for
    comprehensive performance analysis.

    Parameters
    ----------
    experiments : list[ExperimentResult]
        List of experiment results to compare.
    metrics : list[str], optional
        List of metric names to include in the comparison.
        Valid options: "accuracy", "precision", "recall", "f1_score".
        Default is ["accuracy", "precision", "recall", "f1_score"].
    title : str, optional
        Plot title. Default is "Metric Comparison".
    figsize : tuple[int, int], optional
        Figure size as (width, height) in inches. Default is (12, 6).
    save_path : str, optional
        If provided, saves the plot to this file path.
        Default is None (display plot).

    Raises
    ------
    ImportError
        If matplotlib or pandas is not installed.

    Examples
    --------
    Compare all default metrics:

    >>> from insideLLMs.analysis.visualization import plot_metric_comparison
    >>> plot_metric_comparison(experiments, title="Full Metric Analysis")  # doctest: +SKIP

    Compare specific metrics only:

    >>> plot_metric_comparison(
    ...     experiments,
    ...     metrics=["accuracy", "f1_score"],
    ...     title="Accuracy vs F1 Comparison"
    ... )  # doctest: +SKIP

    Publication-ready comparison:

    >>> plot_metric_comparison(
    ...     experiments,
    ...     metrics=["precision", "recall"],
    ...     title="Precision-Recall Analysis",
    ...     figsize=(10, 5),
    ...     save_path="pr_analysis.pdf"
    ... )  # doctest: +SKIP

    Notes
    -----
    - Metrics are shown as grouped bars for each model
    - Percentages (accuracy, precision, recall) are displayed as 0-100
    - F1 score is displayed as-is (0.0-1.0 scale)
    - Multiple experiments for the same model are averaged
    - Legend shows metric names with underscores replaced by spaces

    See Also
    --------
    interactive_metric_radar : Interactive radar chart for metrics
    plot_accuracy_comparison : Simple accuracy comparison
    interactive_heatmap : Heatmap visualization of metrics
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
    """Plot success rate over time or across experiment runs.

    Creates a line chart showing how success rate varies over time
    or across sequential experiment runs. Useful for tracking
    performance trends and identifying regressions.

    Parameters
    ----------
    results : list[tuple[str, float]]
        List of (label, success_rate) tuples where label is a timestamp
        or run identifier string, and success_rate is a float between
        0.0 and 1.0 (will be converted to percentage).
    title : str, optional
        Plot title. Default is "Success Rate Over Time".
    figsize : tuple[int, int], optional
        Figure size as (width, height) in inches. Default is (10, 6).
    save_path : str, optional
        If provided, saves the plot to this file path.
        Default is None (display plot).

    Raises
    ------
    ImportError
        If matplotlib or pandas is not installed.

    Examples
    --------
    Tracking success rate over experiment runs:

    >>> from insideLLMs.analysis.visualization import plot_success_rate_over_time
    >>> results = [
    ...     ("Run 1", 0.85),
    ...     ("Run 2", 0.87),
    ...     ("Run 3", 0.92),
    ...     ("Run 4", 0.89),
    ...     ("Run 5", 0.94),
    ... ]
    >>> plot_success_rate_over_time(results, title="Weekly Success Rate")  # doctest: +SKIP

    Tracking by date:

    >>> results = [
    ...     ("2024-01-01", 0.82),
    ...     ("2024-01-08", 0.85),
    ...     ("2024-01-15", 0.88),
    ...     ("2024-01-22", 0.91),
    ... ]
    >>> plot_success_rate_over_time(
    ...     results,
    ...     title="January 2024 Performance",
    ...     save_path="january_trend.png"
    ... )  # doctest: +SKIP

    Monitoring API reliability:

    >>> results = [
    ...     ("Hour 1", 0.98),
    ...     ("Hour 2", 0.95),
    ...     ("Hour 3", 0.72),  # Incident
    ...     ("Hour 4", 0.99),
    ... ]
    >>> plot_success_rate_over_time(results, title="API Reliability")  # doctest: +SKIP

    Notes
    -----
    - Success rates are converted to percentages (0-100%)
    - Y-axis is fixed to 0-105% range for consistent scaling
    - Area under the line is shaded for visual clarity
    - Grid lines are displayed for easier reading
    - X-axis labels are rotated 45 degrees
    - If results list is empty, prints a message and returns

    See Also
    --------
    interactive_timeline : Interactive Plotly timeline
    plot_accuracy_comparison : Compare accuracy across models
    create_interactive_dashboard : Dashboard with success rate trends
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
    """Plot results from BiasProbe showing response length differences.

    This is a legacy function for visualizing bias probe results that
    compare responses to paired prompts (e.g., with different demographic
    attributes). It shows the difference in response lengths as an
    indicator of potential bias.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of result dictionaries from BiasProbe. Each dictionary should
        have an "output" key containing a list of (response1, response2)
        tuples from paired prompts.
    title : str, optional
        Plot title. Default is "Bias Probe Results".
    save_path : str, optional
        If provided, saves the plot to this file path.
        Default is None (display plot).

    Raises
    ------
    ImportError
        If matplotlib or pandas is not installed.

    Examples
    --------
    Visualizing bias probe results:

    >>> from insideLLMs.analysis.visualization import plot_bias_results
    >>> results = [
    ...     {"output": [("Response A", "Response B"), ("Response C", "Response D")]},
    ...     {"output": [("Short", "This is a much longer response")]},
    ... ]
    >>> plot_bias_results(results, title="Gender Bias Analysis")  # doctest: +SKIP

    Save to file:

    >>> plot_bias_results(
    ...     results,
    ...     title="Demographic Bias Test",
    ...     save_path="bias_analysis.png"
    ... )  # doctest: +SKIP

    Notes
    -----
    - Response length difference = len(response1) - len(response2)
    - Positive values indicate longer first responses
    - Uses seaborn for enhanced bar plots if available
    - This is a legacy function; consider using newer visualization methods

    See Also
    --------
    plot_factuality_results : Visualize factuality probe results
    create_html_report : Generate HTML reports from results
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
    """Plot results from FactualityProbe with category breakdown.

    This is a legacy function for visualizing factuality probe results.
    Creates a two-panel plot showing category distribution and response
    length statistics by category.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of result dictionaries from FactualityProbe. Each dictionary
        should have an "output" key containing a list of question/answer
        items with "category" and "model_answer" fields.
    title : str, optional
        Plot title prefix. Default is "Factuality Probe Results".
    save_path : str, optional
        If provided, saves the plot to this file path.
        Default is None (display plot).

    Raises
    ------
    ImportError
        If matplotlib or pandas is not installed.

    Examples
    --------
    Visualizing factuality results:

    >>> from insideLLMs.analysis.visualization import plot_factuality_results
    >>> results = [
    ...     {"output": [
    ...         {"category": "science", "model_answer": "The Earth orbits the Sun."},
    ...         {"category": "history", "model_answer": "World War II ended in 1945."},
    ...         {"category": "science", "model_answer": "Water is H2O."},
    ...     ]},
    ... ]
    >>> plot_factuality_results(results, title="Knowledge Test")  # doctest: +SKIP

    Save to file:

    >>> plot_factuality_results(
    ...     results,
    ...     title="Factuality Analysis",
    ...     save_path="factuality.png"
    ... )  # doctest: +SKIP

    Notes
    -----
    - Left panel shows bar chart of question counts by category
    - Right panel shows box plot of response lengths by category
    - Uses seaborn for enhanced plots if available
    - This is a legacy function; consider using newer visualization methods

    See Also
    --------
    plot_bias_results : Visualize bias probe results
    create_html_report : Generate HTML reports from results
    plot_accuracy_comparison : Modern accuracy visualization
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
    """Create a static HTML report from probe results.

    Generates a self-contained HTML file with styled presentation of
    probe results. This is a legacy function that creates simple static
    reports without interactive features.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of result dictionaries from any probe type. Each dictionary
        may contain:
        - "input": The input prompt or question
        - "output": The model's response (str, list, or dict)
        - "error": Any error message (optional)
    title : str, optional
        Report title displayed in the header. Default is "Probe Results Report".
    save_path : str, optional
        File path to save the HTML report. Default is "report.html".

    Returns
    -------
    str
        The path to the saved HTML report file.

    Examples
    --------
    Creating a basic report:

    >>> from insideLLMs.analysis.visualization import create_html_report
    >>> results = [
    ...     {"input": "What is 2+2?", "output": "4"},
    ...     {"input": "Capital of France?", "output": "Paris"},
    ... ]
    >>> path = create_html_report(results, title="Math & Geography Test")  # doctest: +SKIP
    >>> print(f"Report saved to: {path}")  # doctest: +SKIP

    Report with errors:

    >>> results = [
    ...     {"input": "Query 1", "output": "Success response"},
    ...     {"input": "Query 2", "error": "Timeout error"},
    ... ]
    >>> path = create_html_report(
    ...     results,
    ...     title="Error Analysis",
    ...     save_path="error_report.html"
    ... )  # doctest: +SKIP

    Report with structured output:

    >>> results = [
    ...     {"input": "Test prompt", "output": {"score": 0.95, "label": "positive"}},
    ...     {"input": "Another prompt", "output": [("A", "B"), ("C", "D")]},
    ... ]
    >>> path = create_html_report(results, title="Structured Results")  # doctest: +SKIP

    Notes
    -----
    - Report includes a summary section with success/error counts
    - Styling uses modern CSS with gradient headers
    - Structured outputs (dict/list) are rendered as tables
    - Errors are highlighted in red
    - For interactive reports, use ``create_interactive_html_report`` instead

    See Also
    --------
    create_interactive_html_report : Interactive HTML report with charts
    experiment_summary_text : Text-based experiment summary
    plot_accuracy_comparison : Visual accuracy comparison
    """
    title_html = _escape_html_text(title)
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title_html}</title>
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
        <h1>{title_html}</h1>
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
            html += (
                "<div class='input'><strong>Input:</strong> "
                f"{_escape_html_text(result['input'])}</div>"
            )

        if "output" in result:
            output = result["output"]
            html += "<div class='output'><strong>Output:</strong></div>"

            if isinstance(output, list):
                html += "<table><tr><th>#</th><th>Output</th></tr>"
                for j, item in enumerate(output):
                    if isinstance(item, tuple):
                        left = _escape_html_text(item[0])
                        right = _escape_html_text(item[1])
                        html += (
                            f"<tr><td>{j + 1}</td><td>{left}<br><em>vs.</em><br>{right}</td></tr>"
                        )
                    else:
                        html += f"<tr><td>{j + 1}</td><td>{_escape_html_text(item)}</td></tr>"
                html += "</table>"
            elif isinstance(output, dict):
                html += "<table><tr><th>Key</th><th>Value</th></tr>"
                for key, value in output.items():
                    html += (
                        f"<tr><td>{_escape_html_text(key)}</td>"
                        f"<td>{_escape_html_text(value)}</td></tr>"
                    )
                html += "</table>"
            else:
                html += f"<p>{_escape_html_text(output)}</p>"

        if "error" in result and result["error"]:
            html += (
                "<div class='error'><strong>Error:</strong> "
                f"{_escape_html_text(result['error'])}</div>"
            )

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

    Generates a Plotly bar chart with hover tooltips, zooming, and
    export capabilities. Ideal for Jupyter notebooks and web-based
    exploration of experiment results.

    Parameters
    ----------
    experiments : list[ExperimentResult]
        List of experiment results to compare. Each experiment should
        have a score with an accuracy value.
    title : str, optional
        Chart title. Default is "Model Accuracy Comparison".
    show_error_bars : bool, optional
        Whether to show error bars if standard deviation is available.
        Default is True. (Note: Currently not implemented, reserved
        for future enhancement.)
    color_by : str, optional
        How to color the bars. Options:
        - "model": Color by model name (default)
        - "probe": Color by probe name

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure object that can be displayed with ``.show()``
        or saved with ``.write_html()``.

    Raises
    ------
    ImportError
        If Plotly or pandas is not installed.
    ValueError
        If no experiments have accuracy data.

    Examples
    --------
    Display in Jupyter notebook:

    >>> from insideLLMs.analysis.visualization import interactive_accuracy_comparison
    >>> fig = interactive_accuracy_comparison(experiments)  # doctest: +SKIP
    >>> fig.show()  # doctest: +SKIP

    Save as standalone HTML:

    >>> fig = interactive_accuracy_comparison(
    ...     experiments,
    ...     title="Q4 2024 Accuracy Results"
    ... )  # doctest: +SKIP
    >>> fig.write_html("accuracy_comparison.html")  # doctest: +SKIP

    Color by probe instead of model:

    >>> fig = interactive_accuracy_comparison(
    ...     experiments,
    ...     title="Accuracy by Probe Type",
    ...     color_by="probe"
    ... )  # doctest: +SKIP
    >>> fig.show()  # doctest: +SKIP

    Embed in a web page:

    >>> fig = interactive_accuracy_comparison(experiments)  # doctest: +SKIP
    >>> html_div = fig.to_html(full_html=False, include_plotlyjs='cdn')  # doctest: +SKIP

    Notes
    -----
    - Hover tooltips show provider and success rate
    - Percentage labels are displayed above each bar
    - Y-axis is fixed to 0-105% for consistent comparison
    - Legend shows model or probe names based on color_by setting

    See Also
    --------
    plot_accuracy_comparison : Static matplotlib version
    interactive_metric_radar : Radar chart for multiple metrics
    interactive_heatmap : Heatmap of accuracy by model and probe
    create_interactive_dashboard : Full dashboard with multiple charts
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
                    # Convert to percentage for display
                    display_value = value * 100
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


def _serialize_experiments_to_json(experiments: list[ExperimentResult]) -> str:
    """Serialize experiments to JSON for embedding in HTML."""
    import json
    from datetime import datetime

    def serialize_value(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, "__dataclass_fields__"):
            return {k: serialize_value(getattr(obj, k)) for k in obj.__dataclass_fields__}
        if isinstance(obj, list):
            return [serialize_value(item) for item in obj]
        if isinstance(obj, dict):
            return {k: serialize_value(v) for k, v in obj.items()}
        return obj

    data = [serialize_value(exp) for exp in experiments]
    return json.dumps(data, indent=2, sort_keys=True)


def create_interactive_html_report(
    experiments: list[ExperimentResult],
    title: str = "LLM Evaluation Report",
    save_path: str = "interactive_report.html",
    include_raw_results: bool = True,
    include_individual_results: bool = True,
    embed_plotly_js: bool = False,
    generated_at: Optional[datetime] = None,
) -> str:
    """Create a comprehensive interactive HTML report with embedded Plotly charts.

    This creates a self-contained, shareable HTML report with:
    - Interactive Plotly charts (accuracy, latency, radar, heatmap, token usage)
    - Embedded JSON data for extraction
    - Export buttons (JSON, CSV download)
    - Dark/light mode toggle
    - Filtering controls (by model, probe, status)
    - Sortable, searchable results table
    - Expandable per-result details
    - Side-by-side model comparison view

    Args:
        experiments: List of experiment results.
        title: Report title.
        save_path: Path to save the HTML file.
        include_raw_results: Whether to include the results summary table.
        include_individual_results: Whether to include expandable individual results.
        embed_plotly_js: If True, embeds Plotly.js inline (larger file, fully offline).
                        If False, uses CDN (smaller file, requires internet).

    Returns:
        Path to the saved report.

    Example:
        >>> path = create_interactive_html_report(experiments, "My Report", "report.html")
        >>> print(f"Report saved to {path}")
    """
    check_plotly_deps()

    # Serialize experiment data for embedding
    experiments_json = _serialize_experiments_to_json(experiments)
    # Prevent `</script>` sequences from terminating the JSON script tag early.
    experiments_json_safe = experiments_json.replace("</", "<\\/")
    title_html = _escape_html_text(title)

    # Generate charts
    charts_html = []

    def _stable_plotly_div_id(title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "chart"
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:8]
        return f"chart-{slug}-{digest}"

    def _stabilize_plotly_div_id(html: str, title: str) -> str:
        match = re.search(r'<div id="([^"]+)" class="plotly-graph-div"', html)
        if not match:
            return html
        old_id = match.group(1)
        new_id = _stable_plotly_div_id(title)
        if old_id == new_id:
            return html
        return html.replace(old_id, new_id)

    def add_chart(fig_func, chart_title: str, chart_id: str) -> None:
        try:
            fig = fig_func(experiments)
            charts_html.append(
                (
                    chart_title,
                    chart_id,
                    _stabilize_plotly_div_id(
                        fig.to_html(full_html=False, include_plotlyjs=False),
                        chart_title,
                    ),
                )
            )
        except (ValueError, KeyError):
            pass

    add_chart(interactive_accuracy_comparison, "Accuracy Comparison", "accuracy")
    add_chart(interactive_latency_distribution, "Latency Distribution", "latency")
    add_chart(interactive_metric_radar, "Performance Radar", "radar")
    add_chart(interactive_heatmap, "Performance Heatmap", "heatmap")

    # Add token usage chart if data available
    try:
        token_data = []
        for exp in experiments:
            if exp.score and exp.score.total_tokens:
                token_data.append(
                    {
                        "Model": exp.model_info.name,
                        "Probe": exp.probe_name,
                        "Total Tokens": exp.score.total_tokens,
                    }
                )
        if token_data:
            df = pd.DataFrame(token_data)
            fig = px.bar(
                df,
                x="Model",
                y="Total Tokens",
                color="Probe",
                barmode="group",
                title="Token Usage by Model",
            )
            fig.update_layout(showlegend=True, hovermode="x unified")
            charts_html.append(
                (
                    "Token Usage",
                    "tokens",
                    _stabilize_plotly_div_id(
                        fig.to_html(full_html=False, include_plotlyjs=False),
                        "Token Usage",
                    ),
                )
            )
    except (ValueError, KeyError):
        pass

    # Add success/failure breakdown
    try:
        status_data = []
        for exp in experiments:
            status_data.append(
                {
                    "Model": exp.model_info.name,
                    "Probe": exp.probe_name,
                    "Success": exp.success_count,
                    "Error": exp.error_count,
                    "Other": exp.total_count - exp.success_count - exp.error_count,
                }
            )
        if status_data:
            df = pd.DataFrame(status_data)
            df_melted = df.melt(
                id_vars=["Model", "Probe"],
                value_vars=["Success", "Error", "Other"],
                var_name="Status",
                value_name="Count",
            )
            fig = px.bar(
                df_melted,
                x="Model",
                y="Count",
                color="Status",
                barmode="stack",
                title="Result Status Breakdown",
                color_discrete_map={
                    "Success": "#27ae60",
                    "Error": "#e74c3c",
                    "Other": "#95a5a6",
                },
            )
            charts_html.append(
                (
                    "Status Breakdown",
                    "status",
                    _stabilize_plotly_div_id(
                        fig.to_html(full_html=False, include_plotlyjs=False),
                        "Status Breakdown",
                    ),
                )
            )
    except (ValueError, KeyError):
        pass

    # Calculate summary statistics
    total_experiments = len(experiments)
    total_results = sum(exp.total_count for exp in experiments)
    total_success = sum(exp.success_count for exp in experiments)
    avg_accuracy = 0.0
    acc_count = 0
    total_tokens = 0
    avg_latency = 0.0
    lat_count = 0

    for exp in experiments:
        if exp.score:
            if exp.score.accuracy is not None:
                avg_accuracy += exp.score.accuracy * 100
                acc_count += 1
            if exp.score.total_tokens:
                total_tokens += exp.score.total_tokens
            if exp.score.mean_latency_ms is not None:
                avg_latency += exp.score.mean_latency_ms
                lat_count += 1

    avg_accuracy = avg_accuracy / acc_count if acc_count > 0 else 0
    avg_latency = avg_latency / lat_count if lat_count > 0 else 0
    success_rate = (total_success / total_results * 100) if total_results > 0 else 0

    unique_models = sorted({exp.model_info.name for exp in experiments})
    unique_probes = sorted({exp.probe_name for exp in experiments})
    unique_providers = sorted({exp.model_info.provider for exp in experiments})

    # Plotly JS source
    # Note: embed_plotly_js parameter is reserved for future use
    # Full embedding would make file very large (~3MB)
    # For now, we always use CDN
    _ = embed_plotly_js  # Silence unused parameter warning
    plotly_script = (
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" crossorigin="anonymous"></script>'
    )

    # Build the HTML
    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title_html}</title>
    {plotly_script}
    <style>
        :root {{
            --bg-primary: #f8fafc;
            --bg-secondary: #ffffff;
            --bg-tertiary: #f1f5f9;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --text-muted: #94a3b8;
            --border-color: #e2e8f0;
            --accent-primary: #6366f1;
            --accent-secondary: #8b5cf6;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            --success: #22c55e;
            --error: #ef4444;
            --warning: #f59e0b;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
            --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1);
            --radius-sm: 6px;
            --radius-md: 12px;
            --radius-lg: 16px;
        }}

        [data-theme="dark"] {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #64748b;
            --border-color: #334155;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.4);
            --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.5);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}

        .header {{
            background: var(--accent-gradient);
            padding: 24px 32px;
            color: white;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: var(--shadow-lg);
        }}

        .header-content {{
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}

        .header h1 {{
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0;
        }}

        .header-actions {{
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 18px;
            border-radius: var(--radius-sm);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            border: none;
            transition: all 0.2s ease;
        }}

        .btn-primary {{
            background: white;
            color: var(--accent-primary);
        }}

        .btn-primary:hover {{
            background: #f1f5f9;
            transform: translateY(-1px);
        }}

        .btn-secondary {{
            background: rgba(255,255,255,0.15);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
        }}

        .btn-secondary:hover {{
            background: rgba(255,255,255,0.25);
        }}

        .btn-icon {{
            width: 20px;
            height: 20px;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 32px;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }}

        .stat-card {{
            background: var(--bg-secondary);
            border-radius: var(--radius-md);
            padding: 24px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color);
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }}

        .stat-label {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }}

        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .stat-value.success {{ color: var(--success); }}
        .stat-value.error {{ color: var(--error); }}
        .stat-value.accent {{ color: var(--accent-primary); }}

        .section {{
            background: var(--bg-secondary);
            border-radius: var(--radius-lg);
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color);
        }}

        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 2px solid var(--border-color);
        }}

        .section-title {{
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .filters {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 24px;
            padding: 20px;
            background: var(--bg-tertiary);
            border-radius: var(--radius-md);
        }}

        .filter-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .filter-label {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            color: var(--text-muted);
        }}

        .filter-select, .filter-input {{
            padding: 10px 14px;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 14px;
            min-width: 160px;
        }}

        .filter-select:focus, .filter-input:focus {{
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }}

        .chart-tabs {{
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}

        .chart-tab {{
            padding: 10px 20px;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}

        .chart-tab:hover {{
            background: var(--bg-secondary);
            border-color: var(--accent-primary);
        }}

        .chart-tab.active {{
            background: var(--accent-gradient);
            color: white;
            border-color: transparent;
        }}

        .chart-container {{
            min-height: 400px;
        }}

        .chart-panel {{
            display: none;
        }}

        .chart-panel.active {{
            display: block;
        }}

        .table-container {{
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}

        th, td {{
            padding: 14px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background: var(--bg-tertiary);
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
            cursor: pointer;
            user-select: none;
            position: sticky;
            top: 0;
        }}

        th:hover {{
            background: var(--border-color);
        }}

        th .sort-icon {{
            margin-left: 6px;
            opacity: 0.5;
        }}

        th.sorted .sort-icon {{
            opacity: 1;
        }}

        tr:hover {{
            background: var(--bg-tertiary);
        }}

        .status-badge {{
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}

        .status-success {{
            background: rgba(34, 197, 94, 0.1);
            color: var(--success);
        }}

        .status-error {{
            background: rgba(239, 68, 68, 0.1);
            color: var(--error);
        }}

        .expandable-row {{
            cursor: pointer;
        }}

        .expand-icon {{
            transition: transform 0.2s ease;
            margin-right: 8px;
        }}

        .expandable-row.expanded .expand-icon {{
            transform: rotate(90deg);
        }}

        .detail-row {{
            display: none;
        }}

        .detail-row.visible {{
            display: table-row;
        }}

        .detail-content {{
            padding: 20px;
            background: var(--bg-tertiary);
        }}

        .detail-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}

        .detail-card {{
            background: var(--bg-secondary);
            padding: 16px;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-color);
        }}

        .detail-card h4 {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 12px;
        }}

        .result-item {{
            padding: 12px;
            margin-bottom: 8px;
            background: var(--bg-tertiary);
            border-radius: var(--radius-sm);
            font-size: 13px;
        }}

        .result-item:last-child {{
            margin-bottom: 0;
        }}

        .comparison-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 24px;
        }}

        .comparison-card {{
            background: var(--bg-tertiary);
            border-radius: var(--radius-md);
            padding: 20px;
        }}

        .comparison-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}

        .comparison-model {{
            font-size: 1.1rem;
            font-weight: 600;
        }}

        .comparison-metrics {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }}

        .comparison-metric {{
            text-align: center;
            padding: 12px;
            background: var(--bg-secondary);
            border-radius: var(--radius-sm);
        }}

        .comparison-metric-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--accent-primary);
        }}

        .comparison-metric-label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
        }}

        .pagination {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            margin-top: 20px;
        }}

        .pagination button {{
            padding: 8px 14px;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            background: var(--bg-secondary);
            color: var(--text-primary);
            cursor: pointer;
            font-size: 14px;
        }}

        .pagination button:hover:not(:disabled) {{
            background: var(--bg-tertiary);
            border-color: var(--accent-primary);
        }}

        .pagination button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}

        .pagination .page-info {{
            color: var(--text-secondary);
            font-size: 14px;
        }}

        .footer {{
            text-align: center;
            padding: 32px;
            color: var(--text-muted);
            font-size: 13px;
        }}

        .hidden {{
            display: none !important;
        }}

        @media (max-width: 768px) {{
            .header-content {{
                flex-direction: column;
                text-align: center;
            }}

            .container {{
                padding: 16px;
            }}

            .filters {{
                flex-direction: column;
            }}

            .filter-select, .filter-input {{
                width: 100%;
            }}
        }}

        /* Print styles */
        @media print {{
            .header {{
                position: static;
            }}

            .btn, .filters, .chart-tabs, .pagination {{
                display: none !important;
            }}

            .section {{
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <h1>{title_html}</h1>
            <div class="header-actions">
                <button class="btn btn-secondary" onclick="toggleTheme()" title="Toggle dark/light mode">
                    <svg class="btn-icon" id="theme-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="5"/>
                        <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
                    </svg>
                    <span id="theme-label">Dark Mode</span>
                </button>
                <button class="btn btn-secondary" onclick="downloadJSON()">
                    <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
                    </svg>
                    Export JSON
                </button>
                <button class="btn btn-primary" onclick="downloadCSV()">
                    <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
                    </svg>
                    Export CSV
                </button>
            </div>
        </div>
    </header>

    <main class="container">
        <!-- Summary Statistics -->
        <div class="summary-grid">
            <div class="stat-card">
                <div class="stat-label">Experiments</div>
                <div class="stat-value accent">{total_experiments}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Results</div>
                <div class="stat-value">{total_results}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Success Rate</div>
                <div class="stat-value success">{success_rate:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Accuracy</div>
                <div class="stat-value accent">{avg_accuracy:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Latency</div>
                <div class="stat-value">{avg_latency:.0f}ms</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Tokens</div>
                <div class="stat-value">{total_tokens:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Models</div>
                <div class="stat-value">{len(unique_models)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Probes</div>
                <div class="stat-value">{len(unique_probes)}</div>
            </div>
        </div>

        <!-- Filters -->
        <div class="filters">
            <div class="filter-group">
                <label class="filter-label">Model</label>
                <select class="filter-select" id="filter-model" onchange="applyFilters()">
                    <option value="">All Models</option>
                    {"".join(f'<option value="{_escape_html_attr(m)}">{_escape_html_text(m)}</option>' for m in unique_models)}
                </select>
            </div>
            <div class="filter-group">
                <label class="filter-label">Probe</label>
                <select class="filter-select" id="filter-probe" onchange="applyFilters()">
                    <option value="">All Probes</option>
                    {"".join(f'<option value="{_escape_html_attr(p)}">{_escape_html_text(p)}</option>' for p in unique_probes)}
                </select>
            </div>
            <div class="filter-group">
                <label class="filter-label">Provider</label>
                <select class="filter-select" id="filter-provider" onchange="applyFilters()">
                    <option value="">All Providers</option>
                    {"".join(f'<option value="{_escape_html_attr(p)}">{_escape_html_text(p)}</option>' for p in unique_providers)}
                </select>
            </div>
            <div class="filter-group">
                <label class="filter-label">Search</label>
                <input type="text" class="filter-input" id="filter-search"
                       placeholder="Search..." oninput="applyFilters()">
            </div>
            <div class="filter-group" style="justify-content: flex-end;">
                <button class="btn btn-secondary" onclick="resetFilters()"
                        style="background: var(--bg-secondary); color: var(--text-primary);">
                    Reset Filters
                </button>
            </div>
        </div>

        <!-- Charts Section -->
        <section class="section">
            <div class="section-header">
                <h2 class="section-title">Visualizations</h2>
            </div>
            <div class="chart-tabs">
"""

    # Add chart tabs
    for i, (chart_title, chart_id, _) in enumerate(charts_html):
        active = "active" if i == 0 else ""
        html += (
            f'                <button class="chart-tab {active}" '
            f"onclick=\"showChart('{chart_id}')\">{_escape_html_text(chart_title)}</button>\n"
        )

    html += """            </div>
            <div class="chart-container">
"""

    # Add chart panels
    for i, (chart_title, chart_id, chart_content) in enumerate(charts_html):
        active = "active" if i == 0 else ""
        html += f"""                <div class="chart-panel {active}" id="panel-{chart_id}">
                    {chart_content}
                </div>
"""

    html += """            </div>
        </section>

        <!-- Model Comparison Section -->
        <section class="section">
            <div class="section-header">
                <h2 class="section-title">Model Comparison</h2>
            </div>
            <div class="comparison-section">
"""

    # Add model comparison cards
    model_stats = {}
    for exp in experiments:
        model = exp.model_info.name
        if model not in model_stats:
            model_stats[model] = {
                "provider": exp.model_info.provider,
                "accuracy": [],
                "latency": [],
                "tokens": 0,
                "success": 0,
                "total": 0,
            }
        if exp.score:
            if exp.score.accuracy is not None:
                model_stats[model]["accuracy"].append(exp.score.accuracy * 100)
            if exp.score.mean_latency_ms is not None:
                model_stats[model]["latency"].append(exp.score.mean_latency_ms)
            if exp.score.total_tokens:
                model_stats[model]["tokens"] += exp.score.total_tokens
        model_stats[model]["success"] += exp.success_count
        model_stats[model]["total"] += exp.total_count

    for model, stats in model_stats.items():
        avg_acc = sum(stats["accuracy"]) / len(stats["accuracy"]) if stats["accuracy"] else 0
        avg_lat = sum(stats["latency"]) / len(stats["latency"]) if stats["latency"] else 0
        success_pct = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0

        model_attr = _escape_html_attr(model)
        model_text = _escape_html_text(model)
        provider_text = _escape_html_text(stats["provider"])

        html += f"""                <div class="comparison-card" data-model="{model_attr}">
                    <div class="comparison-header">
                        <span class="comparison-model">{model_text}</span>
                        <span class="status-badge status-success">{provider_text}</span>
                    </div>
                    <div class="comparison-metrics">
                        <div class="comparison-metric">
                            <div class="comparison-metric-value">{avg_acc:.1f}%</div>
                            <div class="comparison-metric-label">Accuracy</div>
                        </div>
                        <div class="comparison-metric">
                            <div class="comparison-metric-value">{avg_lat:.0f}ms</div>
                            <div class="comparison-metric-label">Avg Latency</div>
                        </div>
                        <div class="comparison-metric">
                            <div class="comparison-metric-value">{success_pct:.1f}%</div>
                            <div class="comparison-metric-label">Success Rate</div>
                        </div>
                        <div class="comparison-metric">
                            <div class="comparison-metric-value">{stats["tokens"]:,}</div>
                            <div class="comparison-metric-label">Total Tokens</div>
                        </div>
                    </div>
                </div>
"""

    html += """            </div>
        </section>
"""

    # Results table section
    if include_raw_results:
        html += """
        <!-- Results Table -->
        <section class="section">
            <div class="section-header">
                <h2 class="section-title">Experiment Results</h2>
            </div>
            <div class="table-container">
                <table id="results-table">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)">Model <span class="sort-icon">↕</span></th>
                            <th onclick="sortTable(1)">Probe <span class="sort-icon">↕</span></th>
                            <th onclick="sortTable(2)">Category <span class="sort-icon">↕</span></th>
                            <th onclick="sortTable(3)">Total <span class="sort-icon">↕</span></th>
                            <th onclick="sortTable(4)">Success <span class="sort-icon">↕</span></th>
                            <th onclick="sortTable(5)">Accuracy <span class="sort-icon">↕</span></th>
                            <th onclick="sortTable(6)">Latency <span class="sort-icon">↕</span></th>
                            <th onclick="sortTable(7)">Tokens <span class="sort-icon">↕</span></th>
                        </tr>
                    </thead>
                    <tbody>
"""

        for i, exp in enumerate(experiments):
            acc = f"{exp.score.accuracy * 100:.1f}%" if exp.score and exp.score.accuracy else "N/A"
            acc_val = exp.score.accuracy * 100 if exp.score and exp.score.accuracy else 0
            lat = (
                f"{exp.score.mean_latency_ms:.0f}ms"
                if exp.score and exp.score.mean_latency_ms
                else "N/A"
            )
            lat_val = exp.score.mean_latency_ms if exp.score and exp.score.mean_latency_ms else 0
            tokens = exp.score.total_tokens if exp.score and exp.score.total_tokens else 0
            category = (
                exp.probe_category.value
                if hasattr(exp.probe_category, "value")
                else str(exp.probe_category)
            )

            row_class = "expandable-row" if include_individual_results else ""
            model_name = exp.model_info.name
            probe_name = exp.probe_name
            provider_name = exp.model_info.provider
            model_attr = _escape_html_attr(model_name)
            probe_attr = _escape_html_attr(probe_name)
            provider_attr = _escape_html_attr(provider_name)
            model_text = _escape_html_text(model_name)
            probe_text = _escape_html_text(probe_name)
            provider_text = _escape_html_text(provider_name)
            category_text = _escape_html_text(category)
            experiment_id_text = _escape_html_text(exp.experiment_id)
            model_id_text = _escape_html_text(exp.model_info.model_id)
            html += f"""                        <tr class="{row_class}" data-row="{i}"
                            data-model="{model_attr}"
                            data-probe="{probe_attr}"
                            data-provider="{provider_attr}"
                            data-acc="{acc_val}" data-lat="{lat_val}">
                            <td><span class="expand-icon">▶</span>{model_text}</td>
                            <td>{probe_text}</td>
                            <td>{category_text}</td>
                            <td>{exp.total_count}</td>
                            <td><span class="status-badge status-success">{exp.success_count}</span></td>
                            <td>{acc}</td>
                            <td>{lat}</td>
                            <td>{tokens:,}</td>
                        </tr>
"""

            # Add expandable detail row
            if include_individual_results:
                html += f"""                        <tr class="detail-row" data-detail="{i}">
                            <td colspan="8">
                                <div class="detail-content">
                                    <div class="detail-grid">
                                        <div class="detail-card">
                                            <h4>Experiment Info</h4>
                                            <p><strong>ID:</strong> {experiment_id_text}</p>
                                            <p><strong>Provider:</strong> {provider_text}</p>
                                            <p><strong>Model ID:</strong> {model_id_text}</p>
                                        </div>
                                        <div class="detail-card">
                                            <h4>Metrics</h4>
                                            <p><strong>Precision:</strong> {f"{exp.score.precision:.3f}" if exp.score and exp.score.precision else "N/A"}</p>
                                            <p><strong>Recall:</strong> {f"{exp.score.recall:.3f}" if exp.score and exp.score.recall else "N/A"}</p>
                                            <p><strong>F1 Score:</strong> {f"{exp.score.f1_score:.3f}" if exp.score and exp.score.f1_score else "N/A"}</p>
                                            <p><strong>Error Rate:</strong> {f"{exp.score.error_rate:.1%}" if exp.score else "N/A"}</p>
                                        </div>
                                        <div class="detail-card">
                                            <h4>Sample Results ({min(5, len(exp.results))} of {len(exp.results)})</h4>
"""
                # Show first 5 results
                for j, result in enumerate(exp.results[:5]):
                    status_class = (
                        "status-success" if result.status.value == "success" else "status-error"
                    )
                    status_text = _escape_html_text(result.status.value)
                    input_preview = (
                        str(result.input)[:100] + "..."
                        if len(str(result.input)) > 100
                        else str(result.input)
                    )
                    input_preview_text = _escape_html_text(input_preview)
                    html += f"""                                            <div class="result-item">
                                                <span class="status-badge {status_class}">{status_text}</span>
                                                <span style="margin-left: 8px;">{input_preview_text}</span>
                                            </div>
"""
                html += """                                        </div>
                                    </div>
                                </div>
                            </td>
                        </tr>
"""

        html += """                    </tbody>
                </table>
            </div>
            <div class="pagination">
                <button onclick="prevPage()" id="prev-btn">← Previous</button>
                <span class="page-info" id="page-info">Page 1 of 1</span>
                <button onclick="nextPage()" id="next-btn">Next →</button>
            </div>
        </section>
"""

    # Footer
    if generated_at is None:
        generated_at = max(
            (exp.completed_at for exp in experiments if getattr(exp, "completed_at", None)),
            default=None,
        )
    generated_at_str = generated_at.strftime("%Y-%m-%d %H:%M:%S") if generated_at else "unknown"
    html += f"""
        <footer class="footer">
            <p>Generated by insideLLMs on {generated_at_str}</p>
            <p>Report contains {total_experiments} experiments across {len(unique_models)} models</p>
        </footer>
    </main>

    <!-- Embedded Data -->
    <script id="experiment-data" type="application/json">
{experiments_json_safe}
    </script>

    <script>
        // Global state
        let currentPage = 1;
        const rowsPerPage = 20;
        let sortColumn = -1;
        let sortAsc = true;
        let filteredRows = [];

        // Theme toggle
        function toggleTheme() {{
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            html.setAttribute('data-theme', newTheme);
            document.getElementById('theme-label').textContent =
                newTheme === 'light' ? 'Dark Mode' : 'Light Mode';
            localStorage.setItem('theme', newTheme);

            // Update Plotly charts for theme
            const plotlyDivs = document.querySelectorAll('.plotly-graph-div');
            plotlyDivs.forEach(div => {{
                const bgColor = newTheme === 'dark' ? '#1e293b' : '#ffffff';
                const textColor = newTheme === 'dark' ? '#f1f5f9' : '#1e293b';
                Plotly.relayout(div, {{
                    'paper_bgcolor': bgColor,
                    'plot_bgcolor': bgColor,
                    'font.color': textColor
                }});
            }});
        }}

        // Initialize theme from localStorage
        (function() {{
            const savedTheme = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', savedTheme);
            document.getElementById('theme-label').textContent =
                savedTheme === 'light' ? 'Dark Mode' : 'Light Mode';
        }})();

        // Chart tabs
        function showChart(chartId) {{
            document.querySelectorAll('.chart-tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.chart-panel').forEach(panel => panel.classList.remove('active'));

            event.target.classList.add('active');
            document.getElementById('panel-' + chartId).classList.add('active');

            // Trigger Plotly resize
            window.dispatchEvent(new Event('resize'));
        }}

        // Filtering
        function applyFilters() {{
            const modelFilter = document.getElementById('filter-model').value.toLowerCase();
            const probeFilter = document.getElementById('filter-probe').value.toLowerCase();
            const providerFilter = document.getElementById('filter-provider').value.toLowerCase();
            const searchFilter = document.getElementById('filter-search').value.toLowerCase();

            const rows = document.querySelectorAll('#results-table tbody tr[data-row]');
            filteredRows = [];

            rows.forEach(row => {{
                const model = row.dataset.model.toLowerCase();
                const probe = row.dataset.probe.toLowerCase();
                const provider = row.dataset.provider.toLowerCase();
                const text = row.textContent.toLowerCase();

                const matchModel = !modelFilter || model === modelFilter;
                const matchProbe = !probeFilter || probe === probeFilter;
                const matchProvider = !providerFilter || provider === providerFilter;
                const matchSearch = !searchFilter || text.includes(searchFilter);

                if (matchModel && matchProbe && matchProvider && matchSearch) {{
                    filteredRows.push(row);
                }}
            }});

            // Update comparison cards visibility
            document.querySelectorAll('.comparison-card').forEach(card => {{
                const model = card.dataset.model.toLowerCase();
                const show = !modelFilter || model === modelFilter;
                card.style.display = show ? 'block' : 'none';
            }});

            currentPage = 1;
            updatePagination();
        }}

        function resetFilters() {{
            document.getElementById('filter-model').value = '';
            document.getElementById('filter-probe').value = '';
            document.getElementById('filter-provider').value = '';
            document.getElementById('filter-search').value = '';
            applyFilters();
        }}

        // Pagination
        function updatePagination() {{
            const rows = filteredRows.length > 0 ? filteredRows :
                Array.from(document.querySelectorAll('#results-table tbody tr[data-row]'));

            const totalPages = Math.ceil(rows.length / rowsPerPage) || 1;
            const start = (currentPage - 1) * rowsPerPage;
            const end = start + rowsPerPage;

            // Hide all rows first
            document.querySelectorAll('#results-table tbody tr').forEach(row => {{
                row.classList.add('hidden');
            }});

            // Show filtered rows for current page
            rows.forEach((row, index) => {{
                if (index >= start && index < end) {{
                    row.classList.remove('hidden');
                    // Also show detail row if expanded
                    const detailRow = document.querySelector(`tr[data-detail="${{row.dataset.row}}"]`);
                    if (detailRow && row.classList.contains('expanded')) {{
                        detailRow.classList.remove('hidden');
                    }}
                }}
            }});

            document.getElementById('page-info').textContent = `Page ${{currentPage}} of ${{totalPages}}`;
            document.getElementById('prev-btn').disabled = currentPage === 1;
            document.getElementById('next-btn').disabled = currentPage === totalPages;
        }}

        function prevPage() {{
            if (currentPage > 1) {{
                currentPage--;
                updatePagination();
            }}
        }}

        function nextPage() {{
            const rows = filteredRows.length > 0 ? filteredRows :
                Array.from(document.querySelectorAll('#results-table tbody tr[data-row]'));
            const totalPages = Math.ceil(rows.length / rowsPerPage);
            if (currentPage < totalPages) {{
                currentPage++;
                updatePagination();
            }}
        }}

        // Sorting
        function sortTable(columnIndex) {{
            const table = document.getElementById('results-table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr[data-row]'));

            // Toggle sort direction
            if (sortColumn === columnIndex) {{
                sortAsc = !sortAsc;
            }} else {{
                sortColumn = columnIndex;
                sortAsc = true;
            }}

            // Update header styling
            table.querySelectorAll('th').forEach((th, i) => {{
                th.classList.toggle('sorted', i === columnIndex);
                const icon = th.querySelector('.sort-icon');
                if (icon) {{
                    icon.textContent = i === columnIndex ? (sortAsc ? '↑' : '↓') : '↕';
                }}
            }});

            // Sort rows
            rows.sort((a, b) => {{
                let aVal = a.cells[columnIndex].textContent.trim();
                let bVal = b.cells[columnIndex].textContent.trim();

                // Handle numeric values
                const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
                const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));

                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return sortAsc ? aNum - bNum : bNum - aNum;
                }}

                return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }});

            // Reorder rows in DOM
            rows.forEach(row => {{
                const detailRow = tbody.querySelector(`tr[data-detail="${{row.dataset.row}}"]`);
                tbody.appendChild(row);
                if (detailRow) {{
                    tbody.appendChild(detailRow);
                }}
            }});

            applyFilters();
        }}

        // Expandable rows
        document.querySelectorAll('.expandable-row').forEach(row => {{
            row.addEventListener('click', function() {{
                const rowIndex = this.dataset.row;
                const detailRow = document.querySelector(`tr[data-detail="${{rowIndex}}"]`);

                this.classList.toggle('expanded');
                detailRow.classList.toggle('visible');
                detailRow.classList.toggle('hidden', !detailRow.classList.contains('visible'));
            }});
        }});

        // Export functions
        function downloadJSON() {{
            const data = document.getElementById('experiment-data').textContent;
            const blob = new Blob([data], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'experiment_results.json';
            a.click();
            URL.revokeObjectURL(url);
        }}

        function downloadCSV() {{
            const data = JSON.parse(document.getElementById('experiment-data').textContent);
            const headers = ['Model', 'Provider', 'Probe', 'Category', 'Total', 'Success',
                           'Accuracy', 'Precision', 'Recall', 'F1', 'Latency_ms', 'Tokens'];

            let csv = headers.join(',') + '\\n';

            data.forEach(exp => {{
                const row = [
                    exp.model_info.name,
                    exp.model_info.provider,
                    exp.probe_name,
                    exp.probe_category,
                    exp.results ? exp.results.length : 0,
                    exp.results ? exp.results.filter(r => r.status === 'success').length : 0,
                    exp.score?.accuracy ?? '',
                    exp.score?.precision ?? '',
                    exp.score?.recall ?? '',
                    exp.score?.f1_score ?? '',
                    exp.score?.mean_latency_ms ?? '',
                    exp.score?.total_tokens ?? ''
                ].map(v => `"${{String(v).replace(/"/g, '""')}}"`);

                csv += row.join(',') + '\\n';
            }});

            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'experiment_results.csv';
            a.click();
            URL.revokeObjectURL(url);
        }}

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            filteredRows = Array.from(document.querySelectorAll('#results-table tbody tr[data-row]'));
            updatePagination();
        }});
    </script>
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
