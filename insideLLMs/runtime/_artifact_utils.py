"""Artifact handling utilities for run directories and JSONL files.

This module provides utilities for managing run artifacts including:
- JSONL file reading and writing
- Run directory preparation and validation
- Resume record validation
- Atomic file writing
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import yaml

from insideLLMs._serialization import (
    StrictSerializationError,
)
from insideLLMs._serialization import (
    fingerprint_value as _fingerprint_value,
)
from insideLLMs._serialization import (
    serialize_value as _serialize_value,
)
from insideLLMs.runtime._result_utils import _record_index_from_record


def _default_run_root() -> Path:
    """Get the default root directory for run artifacts.

    Returns the directory where run artifacts (records.jsonl, manifest.json)
    are stored by default. Can be overridden via the INSIDELLMS_RUN_ROOT
    environment variable.

    Returns
    -------
    Path
        Default run root directory. Falls back to ~/.insidellms/runs.

    Examples
    --------
    Default location:

        >>> import os
        >>> if "INSIDELLMS_RUN_ROOT" not in os.environ:
        ...     root = _default_run_root()
        ...     str(root).endswith(".insidellms/runs")
        True

    With environment variable:

        >>> import os
        >>> os.environ["INSIDELLMS_RUN_ROOT"] = "/custom/runs"
        >>> _default_run_root()
        PosixPath('/custom/runs')
        >>> del os.environ["INSIDELLMS_RUN_ROOT"]

    See Also
    --------
    _prepare_run_dir : Create and validate a run directory.
    """
    env_root = os.environ.get("INSIDELLMS_RUN_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path.home() / ".insidellms" / "runs"


def _semver_tuple(version: Optional[str]) -> tuple[int, int, int]:
    """Convert a semver string to a tuple for comparison.

    Parses a semantic version string (e.g., "1.2.3") into a tuple of integers
    for numeric comparison. Delegates to insideLLMs.schemas.registry.semver_tuple.

    Parameters
    ----------
    version : Optional[str]
        Semantic version string in "major.minor.patch" format.

    Returns
    -------
    tuple[int, int, int]
        Tuple of (major, minor, patch) version numbers.

    Examples
    --------
    Parsing version strings:

        >>> _semver_tuple("1.0.0")
        (1, 0, 0)
        >>> _semver_tuple("2.3.4")
        (2, 3, 4)

    Version comparison:

        >>> _semver_tuple("1.0.1") >= (1, 0, 1)
        True
        >>> _semver_tuple("1.0.0") < (1, 0, 1)
        True

    See Also
    --------
    insideLLMs.schemas.registry.semver_tuple : The underlying implementation.
    """
    from insideLLMs.schemas.registry import semver_tuple

    return semver_tuple(version)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to a file atomically using a temp file and rename.

    Ensures that the file is either fully written or not modified at all,
    preventing partial writes from interruptions. Delegates to
    insideLLMs.resources.atomic_write_text.

    Parameters
    ----------
    path : Path
        Destination file path.
    text : str
        Text content to write.

    Examples
    --------
    Writing a manifest atomically:

        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path = Path(d) / "manifest.json"
        ...     _atomic_write_text(path, '{"key": "value"}')
        ...     path.read_text()
        '{"key": "value"}'

    Notes
    -----
    Uses a temporary file with rename to achieve atomicity. On POSIX systems,
    rename is atomic when source and destination are on the same filesystem.

    See Also
    --------
    _atomic_write_yaml : Atomic YAML file writing.
    insideLLMs.resources.atomic_write_text : The underlying implementation.
    """
    from insideLLMs.resources import atomic_write_text

    atomic_write_text(path, text)


def _atomic_write_yaml(path: Path, data: Any, *, strict_serialization: bool = False) -> None:
    """Write data to a YAML file atomically.

    Serializes data to YAML format and writes it atomically to ensure
    the file is either fully written or not modified. Used for writing
    config.resolved.yaml files for reproducibility.

    Parameters
    ----------
    path : Path
        Destination file path.
    data : Any
        Data to serialize as YAML. Passed through _serialize_value first.
    strict_serialization : bool, default False
        If True, fail on non-JSON-serializable values.

    Examples
    --------
    Writing configuration atomically:

        >>> from pathlib import Path
        >>> import tempfile
        >>> config = {"model": {"type": "openai", "args": {"model_name": "gpt-4"}}}
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path = Path(d) / "config.yaml"
        ...     _atomic_write_yaml(path, config)
        ...     "model:" in path.read_text()
        True

    See Also
    --------
    _atomic_write_text : Atomic text file writing.
    _serialize_value : Value serialization for YAML compatibility.
    """
    content = yaml.safe_dump(
        _serialize_value(data, strict=strict_serialization),
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
    )
    _atomic_write_text(path, content)


def _ensure_run_sentinel(run_dir_path: Path) -> None:
    """Create a marker file to identify run directories.

    Creates a .insidellms_run sentinel file in the run directory. This marker
    is used to verify that a directory is a legitimate insideLLMs run directory
    before allowing destructive operations like overwrite.

    Parameters
    ----------
    run_dir_path : Path
        Path to the run directory.

    Examples
    --------
    Creating a sentinel:

        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     run_dir = Path(d) / "test_run"
        ...     run_dir.mkdir()
        ...     _ensure_run_sentinel(run_dir)
        ...     (run_dir / ".insidellms_run").exists()
        True

    Notes
    -----
    This function never raises exceptions. If the sentinel cannot be created
    (e.g., due to permissions), the run continues without the marker.

    See Also
    --------
    _prepare_run_dir : Uses sentinel for overwrite safety.
    """
    marker = run_dir_path / ".insidellms_run"
    if not marker.exists():
        try:
            marker.write_text("insideLLMs run directory\n", encoding="utf-8")
        except (IOError, OSError):
            # Don't fail the run due to marker write issues.
            pass


def _truncate_incomplete_jsonl(path: Path) -> None:
    """Truncate a JSONL file to remove any incomplete final line.

    When a run is interrupted, the final line of records.jsonl may be
    incomplete (not terminated with a newline). This function removes
    any such incomplete line to ensure the file contains only valid
    JSON records for safe resume.

    Parameters
    ----------
    path : Path
        Path to the JSONL file to truncate. File must exist.

    Returns
    -------
    None
        The file is modified in place.

    Examples
    --------
    Truncating a file with incomplete final line:

        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as f:
        ...     f.write(b'{"line": 1}\\n{"line": 2}\\n{"incomplete":')
        ...     path = Path(f.name)
        >>> _truncate_incomplete_jsonl(path)
        >>> path.read_text()
        '{"line": 1}\\n{"line": 2}\\n'

    File already complete (no change):

        >>> with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as f:
        ...     f.write(b'{"line": 1}\\n{"line": 2}\\n')
        ...     path = Path(f.name)
        >>> _truncate_incomplete_jsonl(path)
        >>> path.read_text()
        '{"line": 1}\\n{"line": 2}\\n'

    Notes
    -----
    This function operates at the byte level for efficiency and handles
    files without any newlines by truncating to empty.

    See Also
    --------
    _read_jsonl_records : Read records with optional truncation.
    """
    data = path.read_bytes()
    if not data:
        return
    if data.endswith(b"\n"):
        return
    cutoff = data.rfind(b"\n")
    if cutoff == -1:
        path.write_bytes(b"")
        return
    path.write_bytes(data[: cutoff + 1])


def _read_jsonl_records(path: Path, *, truncate_incomplete: bool = False) -> list[dict[str, Any]]:
    """Read all records from a JSONL file.

    Parses a JSONL (JSON Lines) file and returns a list of record dictionaries.
    Optionally truncates incomplete final lines before reading, which is useful
    for resuming interrupted runs.

    Parameters
    ----------
    path : Path
        Path to the JSONL file. If the file doesn't exist, returns empty list.
    truncate_incomplete : bool, default False
        If True, truncate any incomplete final line before reading. This
        ensures safe parsing after an interrupted write operation.

    Returns
    -------
    list[dict[str, Any]]
        List of parsed record dictionaries. Empty lines are skipped.
        Non-dict JSON values are skipped.

    Raises
    ------
    ValueError
        If any non-empty line contains invalid JSON.

    Examples
    --------
    Reading a valid JSONL file:

        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        ...     f.write('{"id": 1, "status": "success"}\\n')
        ...     f.write('{"id": 2, "status": "error"}\\n')
        ...     path = Path(f.name)
        >>> records = _read_jsonl_records(path)
        >>> len(records)
        2
        >>> records[0]["id"]
        1

    Reading non-existent file:

        >>> _read_jsonl_records(Path("/nonexistent/file.jsonl"))
        []

    Reading with truncation (safe resume):

        >>> records = _read_jsonl_records(path, truncate_incomplete=True)

    Notes
    -----
    Empty lines are silently skipped. This allows JSONL files with trailing
    newlines or accidental blank lines to be parsed without error.

    See Also
    --------
    _truncate_incomplete_jsonl : Truncate incomplete final line.
    _build_result_record : Build records for writing to JSONL.
    """
    if not path.exists():
        return []
    if truncate_incomplete:
        _truncate_incomplete_jsonl(path)
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record on line {line_no} in {path}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def _validate_resume_record(
    record: dict[str, Any],
    *,
    expected_index: int,
    expected_item: Any,
    run_id: Optional[str],
    strict_serialization: bool = False,
) -> None:
    """Validate a stored record matches the expected prompt item.

    Ensures resume only proceeds when the existing records are a contiguous
    prefix and inputs match the current prompt_set. This prevents silently
    mixing results from different inputs.

    Parameters
    ----------
    record : dict[str, Any]
        The stored record to validate.
    expected_index : int
        The expected index of this record.
    expected_item : Any
        The expected input item for this record.
    run_id : Optional[str]
        The current run ID to check against.
    strict_serialization : bool, default False
        If True, require JSON-stable prompts for validation.

    Raises
    ------
    ValueError
        If the record doesn't match the expected index, run_id, or input.
    """
    record_index = _record_index_from_record(record, default=expected_index)
    if record_index != expected_index:
        raise ValueError("Existing records are not a contiguous prefix; cannot resume safely.")

    if run_id is not None:
        record_run_id = record.get("run_id")
        if record_run_id is not None and record_run_id != run_id:
            raise ValueError(
                f"Existing record run_id '{record_run_id}' does not match current run_id '{run_id}'."
            )

    try:
        expected_fp = _fingerprint_value(expected_item, strict=strict_serialization)
        record_fp = _fingerprint_value(record.get("input"), strict=strict_serialization)
    except StrictSerializationError as exc:
        raise ValueError(
            "strict_serialization requires JSON-stable prompts when validating resume records."
        ) from exc
    if expected_fp != record_fp:
        raise ValueError(
            f"Existing record input mismatch at index {expected_index}; cannot resume safely."
        )


def _prepare_run_dir(path: Path, *, overwrite: bool, run_root: Optional[Path] = None) -> None:
    """Prepare a directory for run artifact emission.

    Policy:
    - If directory doesn't exist: create it.
    - If directory exists and is empty: use it.
    - If directory exists and is non-empty: fail unless overwrite=True.
      If overwrite=True, remove and recreate.

    Parameters
    ----------
    path : Path
        Path to the run directory.
    overwrite : bool
        If True, allow overwriting existing non-empty directories.
    run_root : Optional[Path], default None
        The run root directory, used for safety checks.

    Raises
    ------
    FileExistsError
        If the path exists but is not a directory, or if it's non-empty
        and overwrite is False.
    ValueError
        If attempting to overwrite protected directories (cwd, home, root)
        or directories that don't look like insideLLMs runs.
    """
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"run_dir '{path}' exists and is not a directory")

        try:
            is_empty = not any(path.iterdir())
        except (OSError, PermissionError):
            is_empty = False

        if is_empty:
            return

        if not overwrite:
            raise FileExistsError(
                f"run_dir '{path}' already exists and is not empty. "
                "Use overwrite=True (or CLI --overwrite) to replace it."
            )

        resolved = path.resolve()

        # Critical safety guards for overwrite.
        cwd_resolved = Path.cwd().resolve()
        home_resolved = Path.home().resolve()
        if resolved == cwd_resolved:
            raise ValueError(f"Refusing to overwrite current working directory: '{resolved}'")
        if resolved == home_resolved:
            raise ValueError(f"Refusing to overwrite home directory: '{resolved}'")

        # Filesystem root (POSIX '/' or Windows drive root like 'C:\\').
        if resolved.anchor and Path(resolved.anchor) == resolved:
            raise ValueError(f"Refusing to overwrite filesystem root: '{resolved}'")

        if run_root is not None:
            try:
                run_root_resolved = run_root.resolve()
            except (OSError, ValueError):
                run_root_resolved = run_root
            if resolved == run_root_resolved:
                raise ValueError(f"Refusing to overwrite run_root directory itself: '{resolved}'")

        # Refuse very short paths (e.g. '/tmp', 'C:\\tmp') that are easy to fat-finger.
        if len(resolved.parts) <= 2:
            raise ValueError(f"Refusing to overwrite high-risk short path: '{resolved}'")

        # Sentinel requirement: only overwrite if this looks like an insideLLMs run dir.
        sentinel_ok = (
            (path / "manifest.json").exists()
            or (path / ".insidellms_run").exists()
            or (path / "records.jsonl").exists()
            or (path / "config.resolved.yaml").exists()
        )
        if not sentinel_ok:
            raise ValueError(
                f"Refusing to overwrite '{resolved}': directory is not empty and does not look like an "
                "insideLLMs run (missing manifest.json or .insidellms_run)."
            )

        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def _prepare_run_dir_for_resume(path: Path, *, run_root: Optional[Path] = None) -> None:
    """Prepare a directory for resumable run artifact emission.

    Policy:
    - If directory doesn't exist: create it.
    - If directory exists and is empty: use it.
    - If directory exists and is non-empty: require insideLLMs sentinel.

    Parameters
    ----------
    path : Path
        Path to the run directory.
    run_root : Optional[Path], default None
        The run root directory (unused, for API consistency).

    Raises
    ------
    FileExistsError
        If the path exists but is not a directory.
    ValueError
        If the directory is non-empty and doesn't look like an insideLLMs run.
    """
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"run_dir '{path}' exists and is not a directory")

        try:
            is_empty = not any(path.iterdir())
        except (OSError, PermissionError):
            is_empty = False

        if is_empty:
            return

        sentinel_ok = (path / "manifest.json").exists() or (path / ".insidellms_run").exists()
        if not sentinel_ok:
            raise ValueError(
                f"Refusing to resume in '{path}': directory is not empty and does not look like an "
                "insideLLMs run (missing manifest.json or .insidellms_run)."
            )
        return

    path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "_default_run_root",
    "_semver_tuple",
    "_atomic_write_text",
    "_atomic_write_yaml",
    "_ensure_run_sentinel",
    "_truncate_incomplete_jsonl",
    "_read_jsonl_records",
    "_validate_resume_record",
    "_prepare_run_dir",
    "_prepare_run_dir_for_resume",
]
