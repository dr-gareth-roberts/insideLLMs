"""HTML diff report generation for insideLLMs.

This module generates self-contained HTML reports for comparing experiment runs,
showing side-by-side differences between baseline and candidate results with
interactive filtering, search, and export capabilities.

The generated reports are fully self-contained with all CSS and JavaScript inline,
requiring no external dependencies to view.

Examples
--------
Basic usage:

    >>> from insideLLMs.diff.html_report import generate_diff_html_report
    >>> diff_data = {...}  # From cmd_diff
    >>> generate_diff_html_report(
    ...     diff_data,
    ...     baseline_path="/runs/baseline",
    ...     candidate_path="/runs/candidate",
    ...     output_path="diff_report.html"
    ... )  # doctest: +SKIP
"""

import difflib
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

__all__ = ["generate_diff_html_report"]


def _escape_html(value: Any) -> str:
    """Escape a value for safe HTML display."""
    return html.escape(str(value), quote=False)


def _escape_attr(value: Any) -> str:
    """Escape a value for safe HTML attribute use."""
    return html.escape(str(value), quote=True)


def _generate_inline_diff(text_a: str, text_b: str) -> str:
    """Generate inline HTML diff between two text strings.

    Uses difflib to create a word-level diff with color-coded additions,
    deletions, and changes.

    Args:
        text_a: Baseline text.
        text_b: Candidate text.

    Returns:
        HTML string with diff highlighting.
    """
    if text_a == text_b:
        return f'<span class="diff-unchanged">{_escape_html(text_a)}</span>'

    # Split into words for word-level diff
    words_a = text_a.split() if text_a else []
    words_b = text_b.split() if text_b else []

    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    result_parts: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            text = " ".join(words_a[i1:i2])
            result_parts.append(f'<span class="diff-equal">{_escape_html(text)}</span>')
        elif tag == "delete":
            text = " ".join(words_a[i1:i2])
            result_parts.append(f'<span class="diff-delete">{_escape_html(text)}</span>')
        elif tag == "insert":
            text = " ".join(words_b[j1:j2])
            result_parts.append(f'<span class="diff-insert">{_escape_html(text)}</span>')
        elif tag == "replace":
            old_text = " ".join(words_a[i1:i2])
            new_text = " ".join(words_b[j1:j2])
            result_parts.append(f'<span class="diff-delete">{_escape_html(old_text)}</span>')
            result_parts.append(f'<span class="diff-insert">{_escape_html(new_text)}</span>')

    return " ".join(result_parts)


def _get_output_text(output: Any) -> str:
    """Extract text from an output field.

    Handles various output formats including strings, dicts with text/fingerprint.

    Args:
        output: The output field from a record summary.

    Returns:
        Text representation of the output.
    """
    if output is None:
        return "(no output)"
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        if "text" in output:
            return str(output["text"])
        if "fingerprint" in output:
            return f"[fingerprint: {output['fingerprint'][:16]}...]"
        return json.dumps(output, indent=2)
    return str(output)


def _truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to a maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def _format_score(score: Any) -> str:
    """Format a score value for display."""
    if score is None:
        return "-"
    if isinstance(score, float):
        return f"{score:.4f}"
    return str(score)


def _get_badge_class(kind: str) -> str:
    """Get CSS class for a change kind badge."""
    if "regression" in kind.lower():
        return "badge-regression"
    if "improvement" in kind.lower():
        return "badge-improvement"
    return "badge-change"


def _generate_item_html(item: dict[str, Any], category: str) -> str:
    """Generate HTML for a single diff item (regression/improvement/change).

    Args:
        item: Dict containing record_key, label, kind, detail, baseline, candidate.
        category: One of 'regression', 'improvement', 'change'.

    Returns:
        HTML string for the item card.
    """
    record_key = item.get("record_key", {})
    label = item.get("label", {})
    kind = item.get("kind", "unknown")
    detail = item.get("detail", "")
    baseline = item.get("baseline", {})
    candidate = item.get("candidate", {})

    model_id = record_key.get("model_id", "unknown")
    probe_id = record_key.get("probe_id", "unknown")
    item_id = record_key.get("item_id", "unknown")

    model_label = label.get("model", model_id)
    probe_label = label.get("probe", probe_id)
    example_label = label.get("example", item_id)

    badge_class = _get_badge_class(kind)

    # Get output texts for diff
    baseline_output = _get_output_text(baseline.get("output"))
    candidate_output = _get_output_text(candidate.get("output"))

    # Generate inline diff
    diff_html = _generate_inline_diff(baseline_output, candidate_output)

    return f'''
    <div class="diff-item" data-category="{_escape_attr(category)}"
         data-model="{_escape_attr(model_id)}" data-probe="{_escape_attr(probe_id)}"
         data-item="{_escape_attr(item_id)}">
        <div class="diff-item-header">
            <div class="diff-item-identity">
                <span class="badge {badge_class}">{_escape_html(kind)}</span>
                <span class="diff-model">{_escape_html(model_label)}</span>
                <span class="diff-separator">/</span>
                <span class="diff-probe">{_escape_html(probe_label)}</span>
                <span class="diff-separator">/</span>
                <span class="diff-example">#{_escape_html(example_label)}</span>
            </div>
            <button class="expand-btn" onclick="toggleExpand(this)">
                <svg class="expand-icon" viewBox="0 0 24 24" width="20" height="20">
                    <path fill="currentColor" d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6-1.41-1.41z"/>
                </svg>
            </button>
        </div>
        <div class="diff-item-detail">{_escape_html(detail)}</div>
        <div class="diff-item-expanded">
            <div class="comparison-grid">
                <div class="comparison-panel baseline-panel">
                    <div class="panel-header">
                        <span class="panel-label">Baseline</span>
                        <div class="panel-stats">
                            <span class="stat-badge">Status: {_escape_html(baseline.get("status", "-"))}</span>
                            <span class="stat-badge">Score: {_format_score(baseline.get("primary_score"))}</span>
                        </div>
                    </div>
                    <div class="panel-content">
                        <pre class="output-text">{_escape_html(_truncate_text(baseline_output, 1000))}</pre>
                    </div>
                </div>
                <div class="comparison-panel candidate-panel">
                    <div class="panel-header">
                        <span class="panel-label">Candidate</span>
                        <div class="panel-stats">
                            <span class="stat-badge">Status: {_escape_html(candidate.get("status", "-"))}</span>
                            <span class="stat-badge">Score: {_format_score(candidate.get("primary_score"))}</span>
                        </div>
                    </div>
                    <div class="panel-content">
                        <pre class="output-text">{_escape_html(_truncate_text(candidate_output, 1000))}</pre>
                    </div>
                </div>
            </div>
            <div class="diff-view">
                <div class="diff-view-header">Inline Diff</div>
                <div class="diff-content">{diff_html}</div>
            </div>
        </div>
    </div>
    '''


def _generate_only_item_html(item: dict[str, Any], side: str) -> str:
    """Generate HTML for items only in baseline or candidate.

    Args:
        item: Dict containing record_key, label, and record data.
        side: Either 'baseline' or 'candidate'.

    Returns:
        HTML string for the item card.
    """
    record_key = item.get("record_key", {})
    label = item.get("label", {})

    model_id = record_key.get("model_id", "unknown")
    probe_id = record_key.get("probe_id", "unknown")
    item_id = record_key.get("item_id", "unknown")

    model_label = label.get("model", model_id)
    probe_label = label.get("probe", probe_id)
    example_label = label.get("example", item_id)

    badge_class = "badge-missing-baseline" if side == "candidate" else "badge-missing-candidate"
    badge_text = f"Only in {side}"

    return f'''
    <div class="diff-item" data-category="only_{side}"
         data-model="{_escape_attr(model_id)}" data-probe="{_escape_attr(probe_id)}"
         data-item="{_escape_attr(item_id)}">
        <div class="diff-item-header">
            <div class="diff-item-identity">
                <span class="badge {badge_class}">{badge_text}</span>
                <span class="diff-model">{_escape_html(model_label)}</span>
                <span class="diff-separator">/</span>
                <span class="diff-probe">{_escape_html(probe_label)}</span>
                <span class="diff-separator">/</span>
                <span class="diff-example">#{_escape_html(example_label)}</span>
            </div>
        </div>
    </div>
    '''


def generate_diff_html_report(
    diff_data: dict[str, Any],
    baseline_path: str,
    candidate_path: str,
    output_path: str,
    title: Optional[str] = None,
    generated_at: Optional[datetime] = None,
) -> str:
    """Generate a self-contained HTML diff report.

    Creates a comprehensive HTML report showing side-by-side comparison of
    baseline vs candidate experiment results with interactive filtering,
    search, and export capabilities.

    Args:
        diff_data: Diff report data structure from cmd_diff containing:
            - schema_version: Version of the diff schema
            - baseline: Path to baseline run directory
            - candidate: Path to candidate run directory
            - counts: Dict with common, regressions, improvements, etc.
            - regressions: List of regression items
            - improvements: List of improvement items
            - changes: List of other change items
            - only_baseline: List of items only in baseline
            - only_candidate: List of items only in candidate
        baseline_path: Path to the baseline run directory (for display).
        candidate_path: Path to the candidate run directory (for display).
        output_path: Path where the HTML report will be written.
        title: Optional custom title for the report.
        generated_at: Optional datetime for report generation timestamp.

    Returns:
        The output_path where the report was written.

    Examples:
        >>> diff_data = {
        ...     "schema_version": "1.0.0",
        ...     "baseline": "/runs/baseline",
        ...     "candidate": "/runs/candidate",
        ...     "counts": {
        ...         "common": 100,
        ...         "regressions": 5,
        ...         "improvements": 3,
        ...         "other_changes": 2,
        ...         "only_baseline": 0,
        ...         "only_candidate": 1,
        ...     },
        ...     "regressions": [...],
        ...     "improvements": [...],
        ...     "changes": [...],
        ...     "only_baseline": [],
        ...     "only_candidate": [...],
        ... }
        >>> generate_diff_html_report(
        ...     diff_data,
        ...     "/runs/baseline",
        ...     "/runs/candidate",
        ...     "diff_report.html"
        ... )  # doctest: +SKIP
        'diff_report.html'
    """
    if generated_at is None:
        generated_at = datetime.now()

    if title is None:
        title = "Behavioral Diff Report"

    counts = diff_data.get("counts", {})
    regressions = diff_data.get("regressions", [])
    improvements = diff_data.get("improvements", [])
    changes = diff_data.get("changes", [])
    only_baseline = diff_data.get("only_baseline", [])
    only_candidate = diff_data.get("only_candidate", [])

    total_compared = counts.get("common", 0)
    regression_count = counts.get("regressions", len(regressions))
    improvement_count = counts.get("improvements", len(improvements))
    change_count = counts.get("other_changes", len(changes))
    only_baseline_count = counts.get("only_baseline", len(only_baseline))
    only_candidate_count = counts.get("only_candidate", len(only_candidate))
    unchanged_count = total_compared - regression_count - improvement_count - change_count

    # Collect unique models and probes for filters
    all_items = regressions + improvements + changes + only_baseline + only_candidate
    models: set[str] = set()
    probes: set[str] = set()
    for item in all_items:
        record_key = item.get("record_key", {})
        if record_key.get("model_id"):
            models.add(record_key["model_id"])
        if record_key.get("probe_id"):
            probes.add(record_key["probe_id"])

    # Generate item HTML
    regression_items_html = "\n".join(
        _generate_item_html(item, "regression") for item in regressions
    )
    improvement_items_html = "\n".join(
        _generate_item_html(item, "improvement") for item in improvements
    )
    change_items_html = "\n".join(_generate_item_html(item, "change") for item in changes)
    only_baseline_items_html = "\n".join(
        _generate_only_item_html(item, "baseline") for item in only_baseline
    )
    only_candidate_items_html = "\n".join(
        _generate_only_item_html(item, "candidate") for item in only_candidate
    )

    # Generate filter options
    model_options = "\n".join(
        f'<option value="{_escape_attr(m)}">{_escape_html(m)}</option>' for m in sorted(models)
    )
    probe_options = "\n".join(
        f'<option value="{_escape_attr(p)}">{_escape_html(p)}</option>' for p in sorted(probes)
    )

    # Escape paths for display
    baseline_display = _escape_html(baseline_path)
    candidate_display = _escape_html(candidate_path)
    title_html = _escape_html(title)
    generated_str = generated_at.strftime("%Y-%m-%d %H:%M:%S")

    # JSON data for export
    diff_json = json.dumps(diff_data, indent=2, default=str)
    diff_json_escaped = _escape_html(diff_json)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title_html}</title>
    <style>
        :root {{
            --bg-primary: #f8fafc;
            --bg-secondary: #ffffff;
            --bg-tertiary: #f1f5f9;
            --bg-code: #1e293b;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --text-muted: #94a3b8;
            --text-code: #e2e8f0;
            --border-color: #e2e8f0;
            --accent-primary: #6366f1;
            --accent-secondary: #8b5cf6;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            --success: #22c55e;
            --success-bg: rgba(34, 197, 94, 0.1);
            --error: #ef4444;
            --error-bg: rgba(239, 68, 68, 0.1);
            --warning: #f59e0b;
            --warning-bg: rgba(245, 158, 11, 0.1);
            --info: #3b82f6;
            --info-bg: rgba(59, 130, 246, 0.1);
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            --radius-sm: 6px;
            --radius-md: 12px;
            --radius-lg: 16px;
        }}

        [data-theme="dark"] {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --bg-code: #0f172a;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #64748b;
            --text-code: #e2e8f0;
            --border-color: #334155;
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
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

        /* Header */
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
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}

        .header h1 {{
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0;
        }}

        .header-subtitle {{
            font-size: 0.875rem;
            opacity: 0.9;
            margin-top: 4px;
        }}

        .header-actions {{
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .btn {{
            padding: 10px 18px;
            border-radius: var(--radius-sm);
            font-weight: 500;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s ease;
            border: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
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
            background: rgba(255, 255, 255, 0.15);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }}

        .btn-secondary:hover {{
            background: rgba(255, 255, 255, 0.25);
        }}

        .btn-icon {{
            width: 18px;
            height: 18px;
        }}

        /* Container */
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px;
        }}

        /* Summary Cards */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .stat-card {{
            background: var(--bg-secondary);
            border-radius: var(--radius-md);
            padding: 20px;
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

        .stat-value.regression {{ color: var(--error); }}
        .stat-value.improvement {{ color: var(--success); }}
        .stat-value.change {{ color: var(--warning); }}
        .stat-value.info {{ color: var(--info); }}

        /* Path Info */
        .path-info {{
            background: var(--bg-secondary);
            border-radius: var(--radius-md);
            padding: 16px 20px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
        }}

        .path-item {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .path-label {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            color: var(--text-muted);
        }}

        .path-value {{
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            font-size: 13px;
            color: var(--text-secondary);
            word-break: break-all;
        }}

        /* Filters */
        .filters {{
            background: var(--bg-secondary);
            border-radius: var(--radius-md);
            padding: 20px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
        }}

        .filter-row {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            align-items: flex-end;
        }}

        .filter-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            min-width: 150px;
        }}

        .filter-group.search {{
            flex: 1;
            min-width: 200px;
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
            background: var(--bg-tertiary);
            color: var(--text-primary);
            font-size: 14px;
            width: 100%;
        }}

        .filter-select:focus, .filter-input:focus {{
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }}

        /* Filter Tabs */
        .filter-tabs {{
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }}

        .filter-tab {{
            padding: 10px 20px;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            background: var(--bg-secondary);
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}

        .filter-tab:hover {{
            background: var(--bg-tertiary);
            border-color: var(--accent-primary);
        }}

        .filter-tab.active {{
            background: var(--accent-gradient);
            color: white;
            border-color: transparent;
        }}

        .filter-tab .count {{
            background: rgba(255, 255, 255, 0.2);
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
        }}

        .filter-tab:not(.active) .count {{
            background: var(--bg-tertiary);
            color: var(--text-muted);
        }}

        /* Diff Items */
        .diff-list {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .diff-section {{
            margin-bottom: 32px;
        }}

        .diff-section-header {{
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--border-color);
        }}

        .diff-item {{
            background: var(--bg-secondary);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
            overflow: hidden;
            transition: box-shadow 0.2s ease;
        }}

        .diff-item:hover {{
            box-shadow: var(--shadow-md);
        }}

        .diff-item-header {{
            padding: 16px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
        }}

        .diff-item-identity {{
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }}

        .diff-model {{
            font-weight: 600;
            color: var(--text-primary);
        }}

        .diff-probe {{
            color: var(--text-secondary);
        }}

        .diff-example {{
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
            color: var(--text-muted);
        }}

        .diff-separator {{
            color: var(--text-muted);
        }}

        .diff-item-detail {{
            padding: 12px 20px;
            font-size: 14px;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-color);
        }}

        .diff-item-expanded {{
            display: none;
            padding: 20px;
            background: var(--bg-primary);
        }}

        .diff-item.expanded .diff-item-expanded {{
            display: block;
        }}

        .expand-btn {{
            background: none;
            border: none;
            cursor: pointer;
            padding: 4px;
            color: var(--text-muted);
            transition: transform 0.2s ease;
        }}

        .diff-item.expanded .expand-btn {{
            transform: rotate(90deg);
        }}

        .expand-icon {{
            display: block;
        }}

        /* Badges */
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}

        .badge-regression {{
            background: var(--error-bg);
            color: var(--error);
        }}

        .badge-improvement {{
            background: var(--success-bg);
            color: var(--success);
        }}

        .badge-change {{
            background: var(--warning-bg);
            color: var(--warning);
        }}

        .badge-missing-baseline {{
            background: var(--info-bg);
            color: var(--info);
        }}

        .badge-missing-candidate {{
            background: var(--info-bg);
            color: var(--info);
        }}

        /* Comparison Grid */
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }}

        @media (max-width: 768px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .comparison-panel {{
            background: var(--bg-secondary);
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-color);
            overflow: hidden;
        }}

        .baseline-panel {{
            border-left: 3px solid var(--error);
        }}

        .candidate-panel {{
            border-left: 3px solid var(--success);
        }}

        .panel-header {{
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .panel-label {{
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            color: var(--text-secondary);
        }}

        .panel-stats {{
            display: flex;
            gap: 8px;
        }}

        .stat-badge {{
            font-size: 11px;
            padding: 2px 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            color: var(--text-muted);
        }}

        .panel-content {{
            padding: 16px;
        }}

        .output-text {{
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
            color: var(--text-primary);
            background: var(--bg-code);
            color: var(--text-code);
            padding: 12px;
            border-radius: var(--radius-sm);
            max-height: 300px;
            overflow-y: auto;
        }}

        /* Diff View */
        .diff-view {{
            background: var(--bg-secondary);
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-color);
            overflow: hidden;
        }}

        .diff-view-header {{
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            color: var(--text-secondary);
        }}

        .diff-content {{
            padding: 16px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
            line-height: 1.8;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        .diff-equal {{
            color: var(--text-primary);
        }}

        .diff-delete {{
            background: var(--error-bg);
            color: var(--error);
            text-decoration: line-through;
            padding: 2px 4px;
            border-radius: 3px;
        }}

        .diff-insert {{
            background: var(--success-bg);
            color: var(--success);
            padding: 2px 4px;
            border-radius: 3px;
        }}

        .diff-unchanged {{
            color: var(--text-muted);
        }}

        /* Empty State */
        .empty-state {{
            text-align: center;
            padding: 48px 24px;
            color: var(--text-muted);
        }}

        .empty-state-icon {{
            font-size: 48px;
            margin-bottom: 16px;
        }}

        .empty-state-title {{
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 32px;
            color: var(--text-muted);
            font-size: 13px;
            border-top: 1px solid var(--border-color);
            margin-top: 48px;
        }}

        /* Hidden Utility */
        .hidden {{
            display: none !important;
        }}

        /* JSON Modal */
        .modal-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.2s ease, visibility 0.2s ease;
        }}

        .modal-overlay.visible {{
            opacity: 1;
            visibility: visible;
        }}

        .modal {{
            background: var(--bg-secondary);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-lg);
            max-width: 800px;
            max-height: 80vh;
            width: 90%;
            display: flex;
            flex-direction: column;
        }}

        .modal-header {{
            padding: 20px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .modal-title {{
            font-size: 1.125rem;
            font-weight: 600;
        }}

        .modal-close {{
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: var(--text-muted);
            padding: 4px;
        }}

        .modal-body {{
            padding: 24px;
            overflow-y: auto;
            flex: 1;
        }}

        .modal-body pre {{
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 12px;
            background: var(--bg-code);
            color: var(--text-code);
            padding: 16px;
            border-radius: var(--radius-sm);
            overflow-x: auto;
            white-space: pre;
        }}

        .modal-footer {{
            padding: 16px 24px;
            border-top: 1px solid var(--border-color);
            display: flex;
            justify-content: flex-end;
            gap: 12px;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .header-content {{
                flex-direction: column;
                text-align: center;
            }}

            .container {{
                padding: 16px;
            }}

            .filter-row {{
                flex-direction: column;
            }}

            .filter-group {{
                width: 100%;
            }}

            .summary-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        /* Print Styles */
        @media print {{
            .header {{
                position: static;
                background: #6366f1;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}

            .filters, .header-actions, .expand-btn {{
                display: none !important;
            }}

            .diff-item-expanded {{
                display: block !important;
            }}
        }}
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <div>
                <h1>{title_html}</h1>
                <div class="header-subtitle">Generated: {generated_str}</div>
            </div>
            <div class="header-actions">
                <button class="btn btn-secondary" onclick="toggleTheme()">
                    <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="5"/>
                        <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
                    </svg>
                    Toggle Theme
                </button>
                <button class="btn btn-primary" onclick="showJsonModal()">
                    <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                        <line x1="16" y1="13" x2="8" y2="13"/>
                        <line x1="16" y1="17" x2="8" y2="17"/>
                        <polyline points="10 9 9 9 8 9"/>
                    </svg>
                    Export JSON
                </button>
            </div>
        </div>
    </header>

    <main class="container">
        <!-- Path Info -->
        <div class="path-info">
            <div class="path-item">
                <span class="path-label">Baseline</span>
                <span class="path-value">{baseline_display}</span>
            </div>
            <div class="path-item">
                <span class="path-label">Candidate</span>
                <span class="path-value">{candidate_display}</span>
            </div>
        </div>

        <!-- Summary Stats -->
        <div class="summary-grid">
            <div class="stat-card">
                <div class="stat-label">Compared</div>
                <div class="stat-value info">{total_compared}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Regressions</div>
                <div class="stat-value regression">{regression_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Improvements</div>
                <div class="stat-value improvement">{improvement_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Other Changes</div>
                <div class="stat-value change">{change_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Unchanged</div>
                <div class="stat-value">{unchanged_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Only Baseline</div>
                <div class="stat-value">{only_baseline_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Only Candidate</div>
                <div class="stat-value">{only_candidate_count}</div>
            </div>
        </div>

        <!-- Filters -->
        <div class="filters">
            <div class="filter-row">
                <div class="filter-group search">
                    <label class="filter-label">Search</label>
                    <input type="text" class="filter-input" id="searchInput"
                           placeholder="Search by model, probe, or example ID..."
                           oninput="applyFilters()">
                </div>
                <div class="filter-group">
                    <label class="filter-label">Model</label>
                    <select class="filter-select" id="modelFilter" onchange="applyFilters()">
                        <option value="">All Models</option>
                        {model_options}
                    </select>
                </div>
                <div class="filter-group">
                    <label class="filter-label">Probe</label>
                    <select class="filter-select" id="probeFilter" onchange="applyFilters()">
                        <option value="">All Probes</option>
                        {probe_options}
                    </select>
                </div>
                <div class="filter-group">
                    <button class="btn btn-secondary" onclick="resetFilters()"
                            style="background: var(--bg-tertiary); color: var(--text-primary); margin-top: auto;">
                        Reset
                    </button>
                </div>
            </div>
        </div>

        <!-- Category Tabs -->
        <div class="filter-tabs">
            <button class="filter-tab active" data-category="all" onclick="setCategory('all')">
                All <span class="count">{regression_count + improvement_count + change_count + only_baseline_count + only_candidate_count}</span>
            </button>
            <button class="filter-tab" data-category="regression" onclick="setCategory('regression')">
                Regressions <span class="count">{regression_count}</span>
            </button>
            <button class="filter-tab" data-category="improvement" onclick="setCategory('improvement')">
                Improvements <span class="count">{improvement_count}</span>
            </button>
            <button class="filter-tab" data-category="change" onclick="setCategory('change')">
                Changes <span class="count">{change_count}</span>
            </button>
            <button class="filter-tab" data-category="only_baseline" onclick="setCategory('only_baseline')">
                Only Baseline <span class="count">{only_baseline_count}</span>
            </button>
            <button class="filter-tab" data-category="only_candidate" onclick="setCategory('only_candidate')">
                Only Candidate <span class="count">{only_candidate_count}</span>
            </button>
        </div>

        <!-- Diff Items -->
        <div class="diff-list" id="diffList">
            {regression_items_html}
            {improvement_items_html}
            {change_items_html}
            {only_baseline_items_html}
            {only_candidate_items_html}
        </div>

        <div class="empty-state hidden" id="emptyState">
            <div class="empty-state-icon">&#128269;</div>
            <div class="empty-state-title">No results found</div>
            <div>Try adjusting your search or filter criteria</div>
        </div>
    </main>

    <footer class="footer">
        <p>Generated by insideLLMs Diff Report</p>
        <p>Schema Version: {_escape_html(diff_data.get("schema_version", "unknown"))}</p>
    </footer>

    <!-- JSON Modal -->
    <div class="modal-overlay" id="jsonModal">
        <div class="modal">
            <div class="modal-header">
                <span class="modal-title">Diff Data (JSON)</span>
                <button class="modal-close" onclick="hideJsonModal()">&times;</button>
            </div>
            <div class="modal-body">
                <pre id="jsonContent">{diff_json_escaped}</pre>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="copyJson()"
                        style="background: var(--bg-tertiary); color: var(--text-primary);">
                    Copy to Clipboard
                </button>
                <button class="btn btn-primary" onclick="downloadJson()"
                        style="background: var(--accent-primary); color: white;">
                    Download JSON
                </button>
            </div>
        </div>
    </div>

    <script>
        // State
        let currentCategory = 'all';
        let currentTheme = 'light';

        // Theme Toggle
        function toggleTheme() {{
            currentTheme = currentTheme === 'light' ? 'dark' : 'light';
            document.body.setAttribute('data-theme', currentTheme);
            localStorage.setItem('diff-report-theme', currentTheme);
        }}

        // Load saved theme
        (function() {{
            const saved = localStorage.getItem('diff-report-theme');
            if (saved) {{
                currentTheme = saved;
                document.body.setAttribute('data-theme', currentTheme);
            }}
        }})();

        // Category Filter
        function setCategory(category) {{
            currentCategory = category;
            document.querySelectorAll('.filter-tab').forEach(tab => {{
                tab.classList.toggle('active', tab.dataset.category === category);
            }});
            applyFilters();
        }}

        // Apply All Filters
        function applyFilters() {{
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            const modelFilter = document.getElementById('modelFilter').value;
            const probeFilter = document.getElementById('probeFilter').value;

            const items = document.querySelectorAll('.diff-item');
            let visibleCount = 0;

            items.forEach(item => {{
                const category = item.dataset.category;
                const model = item.dataset.model;
                const probe = item.dataset.probe;
                const itemId = item.dataset.item;

                // Category filter
                const categoryMatch = currentCategory === 'all' || category === currentCategory;

                // Model filter
                const modelMatch = !modelFilter || model === modelFilter;

                // Probe filter
                const probeMatch = !probeFilter || probe === probeFilter;

                // Search filter
                const searchText = `${{model}} ${{probe}} ${{itemId}}`.toLowerCase();
                const searchMatch = !searchTerm || searchText.includes(searchTerm);

                const visible = categoryMatch && modelMatch && probeMatch && searchMatch;
                item.classList.toggle('hidden', !visible);

                if (visible) visibleCount++;
            }});

            // Show/hide empty state
            document.getElementById('emptyState').classList.toggle('hidden', visibleCount > 0);
            document.getElementById('diffList').classList.toggle('hidden', visibleCount === 0);
        }}

        // Reset Filters
        function resetFilters() {{
            document.getElementById('searchInput').value = '';
            document.getElementById('modelFilter').value = '';
            document.getElementById('probeFilter').value = '';
            setCategory('all');
        }}

        // Expand/Collapse Items
        function toggleExpand(btn) {{
            const item = btn.closest('.diff-item');
            item.classList.toggle('expanded');
        }}

        // JSON Modal
        function showJsonModal() {{
            document.getElementById('jsonModal').classList.add('visible');
        }}

        function hideJsonModal() {{
            document.getElementById('jsonModal').classList.remove('visible');
        }}

        // Close modal on overlay click
        document.getElementById('jsonModal').addEventListener('click', function(e) {{
            if (e.target === this) {{
                hideJsonModal();
            }}
        }});

        // Close modal on Escape key
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                hideJsonModal();
            }}
        }});

        // Copy JSON to clipboard
        function copyJson() {{
            const json = document.getElementById('jsonContent').textContent;
            navigator.clipboard.writeText(json).then(() => {{
                alert('JSON copied to clipboard!');
            }}).catch(err => {{
                console.error('Failed to copy:', err);
            }});
        }}

        // Download JSON
        function downloadJson() {{
            const json = document.getElementById('jsonContent').textContent;
            const blob = new Blob([json], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'diff_report.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}

        // Diff data for programmatic access
        window.diffData = {diff_json};
    </script>
</body>
</html>"""

    # Write the output file
    output_file = Path(output_path)
    output_file.write_text(html_content, encoding="utf-8")

    return str(output_file)
