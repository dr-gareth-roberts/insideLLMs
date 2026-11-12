"""Interactive snapshot helpers for diff workflows."""

from __future__ import annotations

import os
import stat
import tempfile
import textwrap
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from insideLLMs.runtime._artifact_utils import _require_unsealed_run_directory

SNAPSHOT_CANDIDATE_FILES: tuple[str, ...] = (
    "records.jsonl",
    "results.jsonl",
    "manifest.json",
    "summary.json",
    "report.html",
    "config.resolved.yaml",
)


def _summary_display(summary: Mapping[str, Any] | None) -> str:
    """Produce compact text for interactive baseline/candidate review."""
    if not isinstance(summary, Mapping):
        return "-"

    output_value = summary.get("output")
    if output_value is not None:
        return str(output_value)

    status = summary.get("status")
    metric_name = summary.get("primary_metric")
    metric_value = summary.get("primary_score")
    if metric_name and metric_value is not None:
        return f"status={status}; {metric_name}={metric_value}"
    if status:
        return f"status={status}"
    return "-"


def _side_by_side_lines(
    baseline_text: str,
    candidate_text: str,
    *,
    width: int = 44,
) -> list[str]:
    left = textwrap.wrap(baseline_text, width=width) or [""]
    right = textwrap.wrap(candidate_text, width=width) or [""]
    lines = [
        "    baseline".ljust(width + 6) + "| candidate",
        "    " + "-" * width + "+" + "-" * (width + 1),
    ]
    for line_l, line_r in zip_longest(left, right, fillvalue=""):
        lines.append(f"    {line_l:<{width}} | {line_r}")
    return lines


def build_interactive_review_lines(
    diff_report: Mapping[str, Any],
    *,
    limit: int,
    width: int = 44,
    dim_text: Callable[[str], str] | None = None,
) -> list[str]:
    """Build a concise interactive review view for snapshot acceptance."""
    regressions = diff_report.get("regressions", [])
    improvements = diff_report.get("improvements", [])
    changes = diff_report.get("changes", [])
    trace_drifts = diff_report.get("trace_drifts", [])
    trace_violation_increases = diff_report.get("trace_violation_increases", [])
    trajectory_drifts = diff_report.get("trajectory_drifts", [])
    only_baseline = diff_report.get("only_baseline", [])
    only_candidate = diff_report.get("only_candidate", [])
    review_items = [
        *regressions,
        *improvements,
        *changes,
        *trace_drifts,
        *trace_violation_increases,
        *trajectory_drifts,
    ]
    dim = dim_text or (lambda s: s)

    lines: list[str] = []
    if not review_items and not only_baseline and not only_candidate:
        lines.append("  No differences to review.")
        return lines

    shown = 0
    for item in review_items[:limit]:
        if not isinstance(item, Mapping):
            continue
        label = item.get("label") if isinstance(item.get("label"), Mapping) else {}
        model_label = label.get("model", item.get("model_id", "unknown"))
        probe_label = label.get("probe", item.get("probe_id", "unknown"))
        example_label = label.get("example", item.get("example_id", "unknown"))
        detail = item.get("detail") or item.get("kind", "change")
        lines.append(f"  {model_label} | {probe_label} | example {example_label}: {detail}")
        baseline_text = _summary_display(item.get("baseline"))
        candidate_text = _summary_display(item.get("candidate"))
        lines.extend(_side_by_side_lines(baseline_text, candidate_text, width=width))
        shown += 1

    remaining = len(review_items) - shown
    if remaining > 0:
        lines.append(dim(f"  ... and {remaining} more reviewed changes"))

    if isinstance(only_baseline, list) and only_baseline:
        lines.append("  Missing in candidate:")
        for item in only_baseline[:limit]:
            if not isinstance(item, Mapping):
                continue
            label = item.get("label") if isinstance(item.get("label"), Mapping) else {}
            lines.append(
                f"    {label.get('model', item.get('model_id', 'unknown'))} | "
                f"{label.get('probe', item.get('probe_id', 'unknown'))} | "
                f"example {label.get('example', item.get('example_id', 'unknown'))}"
            )
        if len(only_baseline) > limit:
            lines.append(dim(f"    ... and {len(only_baseline) - limit} more"))

    if isinstance(only_candidate, list) and only_candidate:
        lines.append("  New in candidate:")
        for item in only_candidate[:limit]:
            if not isinstance(item, Mapping):
                continue
            label = item.get("label") if isinstance(item.get("label"), Mapping) else {}
            lines.append(
                f"    {label.get('model', item.get('model_id', 'unknown'))} | "
                f"{label.get('probe', item.get('probe_id', 'unknown'))} | "
                f"example {label.get('example', item.get('example_id', 'unknown'))}"
            )
        if len(only_candidate) > limit:
            lines.append(dim(f"    ... and {len(only_candidate) - limit} more"))

    return lines


def print_interactive_review(
    diff_report: Mapping[str, Any],
    *,
    limit: int,
    emit: Callable[[str], None] = print,
    emit_subheader: Callable[[str], None] | None = None,
    width: int = 44,
    dim_text: Callable[[str], str] | None = None,
) -> None:
    """Print interactive review lines using caller-provided emitters."""
    if emit_subheader is not None:
        emit_subheader("Interactive Snapshot Review")
    else:
        emit("Interactive Snapshot Review")
    for line in build_interactive_review_lines(
        diff_report,
        limit=limit,
        width=width,
        dim_text=dim_text,
    ):
        emit(line)


def prompt_accept_snapshot(
    *,
    input_func: Callable[[str], str] | None = None,
    prompt: str = "Accept new behavior as baseline? [y/N]: ",
) -> bool:
    """Ask whether to accept candidate outputs as the new baseline."""
    input_func = input_func or input
    try:
        response = input_func(prompt).strip().lower()
    except EOFError:
        return False
    return response in {"y", "yes"}


def _require_simple_artifact_name(artifact_name: str) -> str:
    """Reject path traversal / multi-component artifact names."""
    if not isinstance(artifact_name, str) or not artifact_name.strip():
        raise ValueError("artifact name must be a non-empty string")
    path = Path(artifact_name)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.parts[0] in {".", ".."}
        or path.name != artifact_name
    ):
        raise ValueError(
            f"artifact name must be a simple filename inside the run directory: {artifact_name!r}"
        )
    return artifact_name


def _supports_nofollow_dir_io() -> bool:
    """Return True when directory-relative O_NOFOLLOW opens are available."""
    return (
        all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"))
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
    )


def _require_nofollow_dir_io() -> None:
    """Fail closed when no-follow directory-relative I/O is unavailable."""
    if not _supports_nofollow_dir_io():
        raise OSError(
            "copy_candidate_artifacts_to_baseline requires no-follow "
            "directory-relative I/O (O_NOFOLLOW + dir_fd); refusing fallback opens"
        )


def _open_nofollow_regular(name: str, *, dir_fd: int) -> int:
    """Open a regular file without following a final-component symlink."""
    _require_nofollow_dir_io()
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise ValueError(f"Refusing non-regular source artifact: {name}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"Refusing non-regular source artifact: {name}")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _restore_baseline_publish_plan(
    publish_plan: list[tuple[str, Path, Path, Path | None]],
    *,
    published: int,
) -> None:
    """Restore destinations after a failed backup or publish phase.

    A backup may only be discarded after its restoration succeeds. If
    ``os.replace(backup, destination)`` fails, the ``.bak`` file is left in
    place so the original baseline content remains recoverable.
    """
    for index, (_name, _tmp, destination, backup) in enumerate(publish_plan):
        if backup is not None:
            try:
                os.replace(backup, destination)
            except OSError:
                # Keep the backup — it is the only recoverable copy.
                continue
            # replace consumed the backup path; ignore stray leftovers only after
            # a confirmed successful restore.
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass
        elif index < published:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass


def copy_candidate_artifacts_to_baseline(
    run_dir_baseline: Path,
    run_dir_candidate: Path,
    *,
    artifact_names: Sequence[str] = SNAPSHOT_CANDIDATE_FILES,
) -> list[str]:
    """Copy canonical candidate artifacts into the baseline run directory.

    All names and regular-file checks are validated before any baseline mutation.
    Sources are opened with directory-relative ``O_NOFOLLOW`` I/O; platforms that
    lack that support are refused rather than falling back to following opens.
    Present artifacts are staged to unique temps first; existing destinations are
    backed up before publish, and any backup- or publish-phase failure restores
    the prior baseline pair (including removing files that had no prior).
    """
    _require_nofollow_dir_io()
    run_dir_baseline = Path(run_dir_baseline)
    run_dir_candidate = Path(run_dir_candidate)
    _require_unsealed_run_directory(run_dir_baseline)
    run_dir_baseline.mkdir(parents=True, exist_ok=True)

    # Phase 1: validate every name and every present source/destination before
    # staging or publishing anything.
    planned: list[tuple[str, Path]] = []
    for raw_name in artifact_names:
        artifact_name = _require_simple_artifact_name(raw_name)
        source = run_dir_candidate / artifact_name
        destination = run_dir_baseline / artifact_name
        if not os.path.lexists(source):
            continue
        source_stat = source.lstat()
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"Refusing non-regular source artifact: {artifact_name}")
        if os.path.lexists(destination):
            dest_stat = destination.lstat()
            if not stat.S_ISREG(dest_stat.st_mode):
                raise ValueError(f"Refusing non-regular destination artifact: {artifact_name}")
        planned.append((artifact_name, destination))

    # Phase 2: stage all copies into unique same-directory temps using no-follow
    # opens against the candidate run directory.
    staged: list[tuple[str, Path, Path]] = []
    candidate_dir_fd: int | None = None
    # name, staged_tmp, destination, backup_or_None (None => destination was absent).
    publish_plan: list[tuple[str, Path, Path, Path | None]] = []
    published = 0
    try:
        candidate_dir_fd = os.open(
            str(run_dir_candidate),
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )

        for artifact_name, destination in planned:
            out_fd: int | None = None
            in_fd: int | None = None
            tmp_path: Path | None = None
            try:
                in_fd = _open_nofollow_regular(artifact_name, dir_fd=candidate_dir_fd)
                out_fd, tmp_name = tempfile.mkstemp(
                    prefix=f".{artifact_name}.",
                    suffix=".tmp",
                    dir=str(run_dir_baseline),
                )
                tmp_path = Path(tmp_name)
                with os.fdopen(out_fd, "wb") as out_handle:
                    out_fd = None
                    with os.fdopen(in_fd, "rb") as in_handle:
                        in_fd = None
                        while True:
                            chunk = in_handle.read(1024 * 1024)
                            if not chunk:
                                break
                            out_handle.write(chunk)
                    out_handle.flush()
                    os.fsync(out_handle.fileno())
                staged.append((artifact_name, tmp_path, destination))
                tmp_path = None
            finally:
                if in_fd is not None:
                    os.close(in_fd)
                if out_fd is not None:
                    os.close(out_fd)
                if tmp_path is not None:
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

        # Phase 3: back up existing destinations, then publish all staged copies.
        # Backup preparation is part of the same transaction as publish.
        for artifact_name, tmp_path, destination in staged:
            backup: Path | None = None
            if os.path.lexists(destination):
                dest_stat = destination.lstat()
                if not stat.S_ISREG(dest_stat.st_mode):
                    raise ValueError(f"Refusing non-regular destination artifact: {artifact_name}")
                bak_fd, bak_name = tempfile.mkstemp(
                    prefix=f".{artifact_name}.",
                    suffix=".bak",
                    dir=str(run_dir_baseline),
                )
                os.close(bak_fd)
                backup = Path(bak_name)
                try:
                    os.replace(destination, backup)
                except Exception:
                    try:
                        backup.unlink()
                    except OSError:
                        pass
                    raise
            # Record immediately so a later backup failure can restore this move.
            publish_plan.append((artifact_name, tmp_path, destination, backup))

        copied: list[str] = []
        for artifact_name, tmp_path, destination, _backup in publish_plan:
            os.replace(tmp_path, destination)
            published += 1
            copied.append(artifact_name)
        # Success: staged temps were consumed by replace; drop backups.
        staged = []
        for _name, _tmp, _dest, backup in publish_plan:
            if backup is not None:
                try:
                    backup.unlink()
                except OSError:
                    pass
        publish_plan = []
        return copied
    except Exception:
        _restore_baseline_publish_plan(publish_plan, published=published)
        # Do not delete leftover .bak files here: failed restores keep them as
        # the only recoverable baseline copies.
        raise
    finally:
        if candidate_dir_fd is not None:
            os.close(candidate_dir_fd)
        for _name, tmp_path, _destination in staged:
            try:
                tmp_path.unlink()
            except OSError:
                pass


__all__ = [
    "SNAPSHOT_CANDIDATE_FILES",
    "build_interactive_review_lines",
    "copy_candidate_artifacts_to_baseline",
    "print_interactive_review",
    "prompt_accept_snapshot",
]
