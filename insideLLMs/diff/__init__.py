"""Diff reporting utilities for insideLLMs.

This module provides tools for comparing experiment runs and generating
visual reports of behavioral differences between baseline and candidate runs.

The main components are:

- ``generate_diff_html_report``: Create a self-contained HTML report showing
  side-by-side comparison of experiment results with filtering, search, and
  export capabilities.

Examples
--------
Generating an HTML diff report from diff data:

    >>> from insideLLMs.diff import generate_diff_html_report
    >>> diff_data = {
    ...     "schema_version": "1.0.0",
    ...     "baseline": "/path/to/baseline",
    ...     "candidate": "/path/to/candidate",
    ...     "counts": {"regressions": 2, "improvements": 1, "common": 100},
    ...     "regressions": [...],
    ...     "improvements": [...],
    ...     "changes": [...],
    ... }
    >>> generate_diff_html_report(
    ...     diff_data,
    ...     baseline_path="/path/to/baseline",
    ...     candidate_path="/path/to/candidate",
    ...     output_path="diff_report.html"
    ... )  # doctest: +SKIP

See Also
--------
insideLLMs.cli.cmd_diff : CLI command that generates diff data
insideLLMs.analysis.visualization : General visualization utilities
"""

from insideLLMs.diff.html_report import generate_diff_html_report

__all__ = [
    "generate_diff_html_report",
]
