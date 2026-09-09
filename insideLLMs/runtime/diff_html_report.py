"""Deterministic, self-contained HTML rendering for DiffReport payloads.

Consumes the DiffReport JSON produced by :func:`insideLLMs.runtime.diffing.build_diff_computation`
(and written by ``insidellms diff --format json``) and renders a static page: summary counts,
one table per populated section, and collapsible before/after panels with a word-level inline
diff of the recorded output previews.

Constraints the renderer upholds:

- Deterministic: the same report renders byte-identical HTML. No timestamps, no generated ids,
  no JavaScript; every dict is iterated in a fixed or sorted order.
- Self-contained: inline CSS only, no external assets.
- Every untrusted string (labels, previews, error text, fingerprints) passes through
  :func:`html.escape` before reaching the page.
- Standard library only, so the module stays importable on core-only installs.
"""

from __future__ import annotations

import difflib
import html
import json
from typing import Any, Mapping, Sequence

__all__ = ["render_diff_html"]

_TITLE = "Behavioural Diff Report"
_PREVIEW_NOTE = (
    "Previews are the truncated output text recorded in the diff report, not the full output."
)

# (counts key, human label) in the order the CLI prints them.
_COUNT_LABELS: tuple[tuple[str, str], ...] = (
    ("common", "Compared"),
    ("only_baseline", "Only in baseline"),
    ("only_candidate", "Only in candidate"),
    ("regressions", "Regressions"),
    ("improvements", "Improvements"),
    ("other_changes", "Other changes"),
    ("trace_drifts", "Trace drifts"),
    ("trace_violation_increases", "Trace violation increases"),
    ("trajectory_drifts", "Trajectory drifts"),
)

# (report key, section heading, css tone) for sections whose entries are DiffChangeEntry.
_CHANGE_SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("regressions", "Regressions", "regression"),
    ("improvements", "Improvements", "improvement"),
    ("changes", "Other Changes", "change"),
    ("trace_drifts", "Trace Drifts", "trace"),
    ("trace_violation_increases", "Trace Violation Increases", "trace"),
    ("trajectory_drifts", "Trajectory Drifts", "trace"),
)

# (report key, section heading) for sections whose entries are DiffRecordIdentity only.
_IDENTITY_SECTIONS: tuple[tuple[str, str], ...] = (
    ("only_baseline", "Missing in Candidate"),
    ("only_candidate", "New in Candidate"),
)

_STYLE = """\
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
      margin: 24px; color: #222; background: #fff; line-height: 1.5; }
    h1 { margin-bottom: 4px; font-size: 1.5rem; }
    h2 { margin-top: 32px; font-size: 1.15rem; border-bottom: 1px solid #ddd;
      padding-bottom: 4px; }
    h3 { margin: 12px 0 4px; font-size: 0.9rem; color: #555; text-transform: uppercase;
      letter-spacing: 0.04em; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.92rem; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left;
      vertical-align: top; }
    th { background: #f5f5f5; font-weight: 600; }
    .meta td:first-child { width: 12em; color: #555; }
    .counts { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }
    .count { border: 1px solid #ddd; padding: 8px 12px; min-width: 9em; }
    .count .label { display: block; font-size: 0.75rem; color: #666;
      text-transform: uppercase; letter-spacing: 0.04em; }
    .count .value { font-size: 1.4rem; font-weight: 600; }
    .count.regression .value { color: #b91c1c; }
    .count.improvement .value { color: #15803d; }
    .count.change .value { color: #b45309; }
    .kind { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85rem;
      white-space: nowrap; }
    tr.regression td.kind { color: #b91c1c; }
    tr.improvement td.kind { color: #15803d; }
    tr.change td.kind { color: #b45309; }
    tr.trace td.kind { color: #4338ca; }
    td.expand { background: #fafafa; padding: 4px 8px; }
    details summary { cursor: pointer; color: #555; font-size: 0.85rem; }
    .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 8px 0; }
    .panel { border: 1px solid #ddd; }
    .panel .head { background: #f5f5f5; padding: 4px 8px; font-size: 0.8rem; color: #555;
      border-bottom: 1px solid #ddd; }
    pre { margin: 0; padding: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.82rem; white-space: pre-wrap; word-break: break-word; }
    .inline-diff { border: 1px solid #ddd; padding: 8px; font-family: ui-monospace,
      SFMono-Regular, Menlo, monospace; font-size: 0.82rem; white-space: pre-wrap;
      word-break: break-word; }
    del { background: #fee2e2; color: #991b1b; text-decoration: line-through; }
    ins { background: #dcfce7; color: #166534; text-decoration: none; }
    .note { color: #666; font-size: 0.8rem; margin: 4px 0; }
    .empty { color: #555; padding: 12px; border: 1px solid #ddd; }
    .breaking { color: #b91c1c; font-weight: 600; }
    .acceptable { color: #15803d; }
    .review { color: #b45309; }
    @media (max-width: 800px) { .panels { grid-template-columns: 1fr; } }
"""


def _esc(value: Any) -> str:
    """Escape any value for safe inclusion in HTML text or attributes."""
    return html.escape(str(value), quote=True)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _dump_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, default=str)


def _format_score(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _identity(entry: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the (model_id, probe_id, item_id) key an entry sorts by."""
    key = _as_dict(entry.get("record_key"))
    return (
        str(key.get("model_id") or entry.get("model_id") or ""),
        str(key.get("probe_id") or entry.get("probe_id") or ""),
        str(key.get("item_id") or entry.get("example_id") or ""),
    )


def _sort_key(entry: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (*_identity(entry), str(entry.get("kind") or ""))


def _labels(entry: Mapping[str, Any]) -> tuple[str, str, str]:
    """Human labels with a fall-back to the identity key when a label is absent."""
    label = _as_dict(entry.get("label"))
    model_id, probe_id, item_id = _identity(entry)
    return (
        str(label.get("model") or model_id),
        str(label.get("probe") or probe_id),
        str(label.get("example") or entry.get("example_id") or item_id),
    )


def _sorted_entries(report: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    entries = [item for item in _as_list(report.get(key)) if isinstance(item, Mapping)]
    return sorted((dict(item) for item in entries), key=_sort_key)


def _output_preview(summary: Mapping[str, Any]) -> tuple[str, bool]:
    """Return (display text, is_text) for a DiffRecordSummary's output."""
    output = _as_dict(summary.get("output"))
    if not output:
        return "(no output)", False
    if output.get("type") == "text":
        preview = output.get("preview")
        return (str(preview) if preview is not None else "", True)
    fingerprint = output.get("fingerprint")
    return (f"[structured output] fingerprint {fingerprint or 'absent'}", False)


def _inline_diff(before: str, after: str) -> str:
    """Word-level inline diff rendered with <del>/<ins>; every token is escaped."""
    if before == after:
        return _esc(before)
    words_a = before.split()
    words_b = after.split()
    matcher = difflib.SequenceMatcher(None, words_a, words_b, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old = _esc(" ".join(words_a[i1:i2]))
        new = _esc(" ".join(words_b[j1:j2]))
        if tag == "equal":
            parts.append(old)
        elif tag == "delete":
            parts.append(f"<del>{old}</del>")
        elif tag == "insert":
            parts.append(f"<ins>{new}</ins>")
        else:
            parts.append(f"<del>{old}</del> <ins>{new}</ins>")
    return " ".join(parts)


def _short_fingerprint(value: Any) -> str:
    return str(value)[:12] if value else "absent"


def _entry_detail(entry: Mapping[str, Any]) -> str:
    """Plain-text detail line derived per change kind, matching the CLI wording."""
    kind = str(entry.get("kind") or "")
    baseline = _as_dict(entry.get("baseline"))
    candidate = _as_dict(entry.get("candidate"))
    if kind in {"metric_regression", "metric_improvement"}:
        delta = entry.get("delta")
        delta_text = f"{float(delta):+.4f}" if isinstance(delta, (int, float)) else "?"
        return (
            f"{entry.get('metric') or 'score'} "
            f"{_format_score(baseline.get('primary_score'))} -> "
            f"{_format_score(candidate.get('primary_score'))} (delta {delta_text})"
        )
    if kind == "metrics_not_comparable":
        return f"metrics not comparable: {entry.get('reason') or 'unknown'}"
    if kind == "metric_key_missing":
        return (
            "metric keys missing: "
            f"baseline_missing={_as_list(entry.get('baseline_missing'))}, "
            f"candidate_missing={_as_list(entry.get('candidate_missing'))}"
        )
    if kind == "output_changed":
        _text_a, is_text_a = _output_preview(baseline)
        _text_b, is_text_b = _output_preview(candidate)
        fp_a = entry.get("baseline_fingerprint")
        fp_b = entry.get("candidate_fingerprint")
        if is_text_a or is_text_b:
            return "output changed"
        if fp_a and fp_b:
            return f"output fingerprint {fp_a} -> {fp_b}"
        return "output changed (structured)"
    if kind == "trace_drift":
        return (
            f"trace {_short_fingerprint(entry.get('baseline_trace_fingerprint'))} -> "
            f"{_short_fingerprint(entry.get('candidate_trace_fingerprint'))}"
        )
    if kind == "trace_violations_increased":
        return (
            f"violations {entry.get('baseline_violations', '?')} -> "
            f"{entry.get('candidate_violations', '?')}"
        )
    detail = entry.get("detail")
    return str(detail) if detail else (kind or "change")


def _entry_extras(entry: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(heading, preformatted text) blocks shown in the expandable row."""
    extras: list[tuple[str, str]] = []
    if entry.get("details") is not None:
        extras.append(("Metric context", _dump_json(entry.get("details"))))
    trace_a = entry.get("baseline_trace_fingerprint")
    trace_b = entry.get("candidate_trace_fingerprint")
    if trace_a is not None or trace_b is not None:
        extras.append(
            (
                "Trace fingerprints",
                f"baseline: {trace_a or 'absent'}\ncandidate: {trace_b or 'absent'}",
            )
        )
    if entry.get("candidate_violation_details"):
        extras.append(
            ("Candidate violations", _dump_json(entry.get("candidate_violation_details")))
        )
    for side in ("baseline", "candidate"):
        trajectory = entry.get(f"{side}_trajectory")
        if trajectory is not None:
            extras.append((f"{side.capitalize()} trajectory", _dump_json(trajectory)))
    return extras


def _side_cell(summary: Mapping[str, Any]) -> str:
    if not summary:
        return "-"
    metric = summary.get("primary_metric")
    score = _format_score(summary.get("primary_score"))
    score_text = f"{metric}={score}" if metric else score
    return f"{_esc(summary.get('status', '-'))}<br>{_esc(score_text)}"


def _panel(title: str, summary: Mapping[str, Any]) -> str:
    text, _is_text = _output_preview(summary)
    return f'<div class="panel"><div class="head">{_esc(title)}</div><pre>{_esc(text)}</pre></div>'


def _expanded_content(entry: Mapping[str, Any]) -> str:
    baseline = _as_dict(entry.get("baseline"))
    candidate = _as_dict(entry.get("candidate"))
    blocks: list[str] = []
    if baseline or candidate:
        blocks.append(
            f'<div class="panels">{_panel("Baseline", baseline)}{_panel("Candidate", candidate)}</div>'
        )
        text_a, is_text_a = _output_preview(baseline)
        text_b, is_text_b = _output_preview(candidate)
        if is_text_a and is_text_b:
            blocks.append("<h3>Inline diff</h3>")
            blocks.append(f'<div class="inline-diff">{_inline_diff(text_a, text_b)}</div>')
        blocks.append(f'<p class="note">{_esc(_PREVIEW_NOTE)}</p>')
    for heading, text in _entry_extras(entry):
        blocks.append(f"<h3>{_esc(heading)}</h3><pre>{_esc(text)}</pre>")
    return "".join(blocks)


def _change_rows(entry: Mapping[str, Any], tone: str) -> str:
    model, probe, example = _labels(entry)
    row = (
        f'<tr class="{tone}"><td>{_esc(model)}</td><td>{_esc(probe)}</td>'
        f"<td>{_esc(example)}</td>"
        f'<td class="kind">{_esc(entry.get("kind") or "change")}</td>'
        f"<td>{_esc(_entry_detail(entry))}</td>"
        f"<td>{_side_cell(_as_dict(entry.get('baseline')))}</td>"
        f"<td>{_side_cell(_as_dict(entry.get('candidate')))}</td></tr>"
    )
    expanded = _expanded_content(entry)
    if not expanded:
        return row
    return (
        f'{row}<tr><td class="expand" colspan="7"><details><summary>Before / after</summary>'
        f"{expanded}</details></td></tr>"
    )


def _change_section(title: str, tone: str, entries: Sequence[Mapping[str, Any]]) -> str:
    if not entries:
        return ""
    rows = "".join(_change_rows(entry, tone) for entry in entries)
    return (
        f"<h2>{_esc(title)} ({len(entries)})</h2><table><thead><tr>"
        "<th>Model</th><th>Probe</th><th>Example</th><th>Kind</th><th>Detail</th>"
        f"<th>Baseline</th><th>Candidate</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _identity_section(title: str, entries: Sequence[Mapping[str, Any]]) -> str:
    if not entries:
        return ""
    rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(*(_esc(v) for v in _labels(entry)))
        for entry in entries
    )
    return (
        f"<h2>{_esc(title)} ({len(entries)})</h2><table><thead><tr>"
        f"<th>Model</th><th>Probe</th><th>Example</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _count_tone(key: str) -> str:
    if key == "regressions":
        return "regression"
    if key == "improvements":
        return "improvement"
    if key in {"other_changes", "only_baseline", "only_candidate"}:
        return "change"
    return "neutral"


def _summary(report: Mapping[str, Any]) -> str:
    counts = _as_dict(report.get("counts"))
    ordered = [(key, label) for key, label in _COUNT_LABELS if key in counts]
    known = {key for key, _label in _COUNT_LABELS}
    ordered.extend((key, key.replace("_", " ")) for key in sorted(counts) if key not in known)
    cards = "".join(
        f'<div class="count {_count_tone(key)}"><span class="label">{_esc(label)}</span>'
        f'<span class="value">{_esc(counts.get(key, 0))}</span></div>'
        for key, label in ordered
    )
    run_ids = _as_dict(report.get("run_ids"))
    duplicates = _as_dict(report.get("duplicates"))
    meta_rows = [
        ("Baseline", report.get("baseline", "-")),
        ("Candidate", report.get("candidate", "-")),
        ("Baseline run ids", ", ".join(sorted(map(str, _as_list(run_ids.get("baseline"))))) or "-"),
        (
            "Candidate run ids",
            ", ".join(sorted(map(str, _as_list(run_ids.get("candidate"))))) or "-",
        ),
        (
            "Duplicate records",
            f"baseline {duplicates.get('baseline', 0)}, candidate {duplicates.get('candidate', 0)}",
        ),
        ("Schema version", report.get("schema_version", "-")),
    ]
    meta = "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in meta_rows)
    return f'<div class="counts">{cards}</div><table class="meta"><tbody>{meta}</tbody></table>'


def _judge_section(report: Mapping[str, Any]) -> str:
    judge = _as_dict(report.get("judge"))
    if not judge:
        return ""
    summary = _as_dict(judge.get("summary"))
    verdicts = sorted(
        (dict(item) for item in _as_list(judge.get("verdicts")) if isinstance(item, Mapping)),
        key=lambda item: (str(item.get("section") or ""), *_sort_key(item)),
    )
    header = (
        f"<p>Policy <strong>{_esc(judge.get('policy', 'strict'))}</strong>; breaking: "
        f"<strong>{'yes' if judge.get('breaking') else 'no'}</strong>; "
        f"breaking {_esc(summary.get('breaking', 0))}, review {_esc(summary.get('review', 0))}, "
        f"acceptable {_esc(summary.get('acceptable', 0))}.</p>"
    )
    if not verdicts:
        return f"<h2>Judge Verdict</h2>{header}<p class='empty'>No judged items.</p>"
    rows = []
    for item in verdicts:
        model, probe, example = _labels(item)
        decision = str(item.get("decision") or "review")
        rows.append(
            f'<tr><td class="{_esc(decision)}">{_esc(decision)}</td>'
            f"<td>{_esc(item.get('section') or '-')}</td><td>{_esc(model)}</td>"
            f"<td>{_esc(probe)}</td><td>{_esc(example)}</td>"
            f"<td>{_esc(item.get('detail') or item.get('kind') or 'change')}</td>"
            f"<td>{_esc(item.get('reason') or '')}</td></tr>"
        )
    return (
        f"<h2>Judge Verdict</h2>{header}<table><thead><tr><th>Decision</th><th>Section</th>"
        "<th>Model</th><th>Probe</th><th>Example</th><th>Detail</th><th>Reason</th></tr>"
        f"</thead><tbody>{''.join(rows)}</tbody></table>"
    )


def render_diff_html(report: Mapping[str, Any]) -> str:
    """Render a DiffReport payload as a deterministic, self-contained HTML document.

    Args:
        report: DiffReport JSON as produced by ``build_diff_computation`` or
            ``insidellms diff --format json``. Absent or empty sections render as nothing;
            an optional ``judge`` block (as attached by ``insidellms diff --judge``) renders
            as an extra section.

    Returns:
        The full HTML document as a string. Rendering the same report twice yields
        byte-identical output.
    """
    sections: list[str] = []
    for key, title, tone in _CHANGE_SECTIONS:
        sections.append(_change_section(title, tone, _sorted_entries(report, key)))
    for key, title in _IDENTITY_SECTIONS:
        sections.append(_identity_section(title, _sorted_entries(report, key)))
    body = "".join(sections)
    if not body:
        body = '<p class="empty">No differences detected between baseline and candidate.</p>'
    body += _judge_section(report)
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{_esc(_TITLE)}</title>\n<style>\n{_STYLE}</style>\n</head>\n<body>\n"
        f"<h1>{_esc(_TITLE)}</h1>\n{_summary(report)}\n{body}\n</body>\n</html>\n"
    )
