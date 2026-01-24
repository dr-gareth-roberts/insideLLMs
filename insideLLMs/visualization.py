"""Compatibility shim for insideLLMs.analysis.visualization.

This module provides backward compatibility for code that imports from
``insideLLMs.visualization`` instead of the new canonical location at
``insideLLMs.analysis.visualization``.

Overview
--------
As part of a codebase reorganization, the visualization module was moved from
``insideLLMs.visualization`` to ``insideLLMs.analysis.visualization``. This
shim module ensures that existing code continues to work without modification
by transparently redirecting all imports to the new location.

When you import from this module, Python's module system is patched so that
``insideLLMs.visualization`` becomes an alias for the actual module at
``insideLLMs.analysis.visualization``. This means:

1. All functions, classes, and constants are available at both import paths
2. ``isinstance()`` and ``type()`` checks work identically
3. No performance penalty beyond the initial import
4. IDE autocompletion and type checking work normally

Migration Guide
---------------
While this shim will continue to work for backward compatibility, new code
should import directly from the canonical location:

**Old (deprecated but still works):**

>>> from insideLLMs.visualization import text_bar_chart  # doctest: +SKIP
>>> from insideLLMs.visualization import ExperimentExplorer  # doctest: +SKIP

**New (recommended):**

>>> from insideLLMs.analysis.visualization import text_bar_chart  # doctest: +SKIP
>>> from insideLLMs.analysis.visualization import ExperimentExplorer  # doctest: +SKIP

Available Components
--------------------
This module re-exports everything from ``insideLLMs.analysis.visualization``,
including:

**Dependency Availability Flags:**
    - ``MATPLOTLIB_AVAILABLE``: True if matplotlib and pandas are installed
    - ``SEABORN_AVAILABLE``: True if seaborn is installed
    - ``PLOTLY_AVAILABLE``: True if plotly is installed
    - ``IPYWIDGETS_AVAILABLE``: True if ipywidgets and IPython are installed

**Dependency Check Functions:**
    - ``check_visualization_deps()``: Check for matplotlib/pandas
    - ``check_plotly_deps()``: Check for plotly
    - ``check_ipywidgets_deps()``: Check for Jupyter widgets

**Text-Based Visualizations (no dependencies):**
    - ``text_bar_chart()``: ASCII horizontal bar charts
    - ``text_histogram()``: ASCII histogram of value distributions
    - ``text_comparison_table()``: ASCII comparison tables
    - ``text_summary_stats()``: Formatted statistical summaries
    - ``experiment_summary_text()``: Complete experiment summary

**Matplotlib Visualizations (requires matplotlib, pandas):**
    - ``plot_accuracy_comparison()``: Bar chart of model accuracies
    - ``plot_latency_distribution()``: Box plots of latency data
    - ``plot_metric_comparison()``: Grouped bar chart of multiple metrics
    - ``plot_success_rate_over_time()``: Line chart of success rates
    - ``plot_bias_results()``: Visualize BiasProbe results (legacy)
    - ``plot_factuality_results()``: Visualize FactualityProbe results (legacy)

**Plotly Interactive Visualizations (requires plotly, pandas):**
    - ``interactive_accuracy_comparison()``: Interactive accuracy bar chart
    - ``interactive_latency_distribution()``: Interactive box/violin/histogram
    - ``interactive_metric_radar()``: Spider/radar chart for multi-metric view
    - ``interactive_timeline()``: Performance metrics over time
    - ``interactive_heatmap()``: Model-probe performance heatmap
    - ``interactive_scatter_comparison()``: Two-metric scatter plot
    - ``interactive_sunburst()``: Hierarchical breakdown chart
    - ``create_interactive_dashboard()``: Multi-chart dashboard

**HTML Report Generation:**
    - ``create_html_report()``: Simple HTML report from probe results
    - ``create_interactive_html_report()``: Full-featured interactive report

**Jupyter Notebook Widgets (requires ipywidgets):**
    - ``ExperimentExplorer``: Interactive widget-based result explorer

Examples
--------
Quick text-based visualization (works everywhere):

>>> from insideLLMs.visualization import text_bar_chart
>>> labels = ["Model A", "Model B", "Model C"]
>>> scores = [0.95, 0.87, 0.92]
>>> chart = text_bar_chart(labels, scores, title="Accuracy Scores")
>>> print(chart)
Accuracy Scores
===============
<BLANKLINE>
Model A | ██████████████████████████████████████████████████ 0.95
Model B | █████████████████████████████████████████████ 0.87
Model C | ████████████████████████████████████████████████ 0.92

Creating a histogram of latency data:

>>> from insideLLMs.visualization import text_histogram
>>> latencies = [120, 145, 98, 132, 156, 110, 125, 140, 115, 138]
>>> hist = text_histogram(latencies, bins=4, title="API Latency (ms)")
>>> print(hist)  # doctest: +NORMALIZE_WHITESPACE
API Latency (ms)
================
<BLANKLINE>
[ 98.00 - 112.50] | ████████████████████████████████████████ (2)
[112.50 - 127.00] | ████████████████████████████████████████ (2)
[127.00 - 141.50] | ████████████████████████████████████████████████████████████ (3)
[141.50 - 156.00] | ████████████████████████████████████████████████████████████ (3)

Using matplotlib for publication-quality plots (requires matplotlib):

>>> from insideLLMs.visualization import plot_accuracy_comparison
>>> # experiments = [...]  # List of ExperimentResult objects
>>> plot_accuracy_comparison(
...     experiments,
...     title="Q1 2024 Model Comparison",
...     save_path="comparison.png"
... )  # doctest: +SKIP

Creating interactive Plotly charts (requires plotly):

>>> from insideLLMs.visualization import interactive_accuracy_comparison
>>> fig = interactive_accuracy_comparison(experiments)  # doctest: +SKIP
>>> fig.show()  # Opens in browser or Jupyter  # doctest: +SKIP

Generating a comprehensive HTML report:

>>> from insideLLMs.visualization import create_interactive_html_report
>>> path = create_interactive_html_report(
...     experiments,
...     title="Monthly Evaluation Report",
...     save_path="report.html"
... )  # doctest: +SKIP
>>> print(f"Report saved to: {path}")  # doctest: +SKIP

Using the Jupyter widget explorer:

>>> from insideLLMs.visualization import ExperimentExplorer
>>> explorer = ExperimentExplorer(experiments)  # doctest: +SKIP
>>> explorer.show()  # Interactive filtering and chart switching  # doctest: +SKIP

Notes
-----
**Why Use a Shim?**

Module shims like this one are a common pattern for maintaining backward
compatibility during codebase reorganization. They allow:

- Gradual migration of import statements across large codebases
- No breaking changes for users of the public API
- Clean separation of implementation from module structure
- Zero runtime overhead after initial import

**Implementation Details:**

This shim works by replacing itself in ``sys.modules`` with the target module.
After import, ``insideLLMs.visualization`` and ``insideLLMs.analysis.visualization``
are the exact same module object in memory:

>>> import insideLLMs.visualization as viz1  # doctest: +SKIP
>>> import insideLLMs.analysis.visualization as viz2  # doctest: +SKIP
>>> viz1 is viz2  # True - same object  # doctest: +SKIP
True

**Deprecation Timeline:**

This shim is provided for backward compatibility and is not deprecated.
Both import paths will continue to work indefinitely. However, documentation
and new examples will use the canonical path at ``insideLLMs.analysis.visualization``.

See Also
--------
insideLLMs.analysis.visualization : The canonical location for this module
insideLLMs.types.ExperimentResult : Data structure for experiment results
insideLLMs.analysis.statistics : Statistical analysis functions
insideLLMs.export : Functions for exporting results to various formats

References
----------
.. [1] Python Module Shim Pattern
       https://docs.python.org/3/reference/import.html#the-module-cache

.. [2] PEP 328 -- Imports: Multi-Line and Absolute/Relative
       https://peps.python.org/pep-0328/
"""

import sys

from insideLLMs.analysis import visualization as _visualization

# Replace this module in sys.modules with the actual visualization module.
# This makes `import insideLLMs.visualization` equivalent to
# `import insideLLMs.analysis.visualization` at the object level.
sys.modules[__name__] = _visualization
