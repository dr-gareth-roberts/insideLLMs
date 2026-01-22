"""Resource management utilities for insideLLMs.

This module provides context managers and utilities for deterministic
resource cleanup, particularly for file handles and run directories.

The primary pattern uses ExitStack for managing multiple resources
with guaranteed cleanup even when exceptions occur.

Example:
    >>> from contextlib import ExitStack
    >>> from insideLLMs.resources import open_records_file, RunDirectoryManager
    >>>
    >>> with ExitStack() as stack:
    ...     run_mgr = stack.enter_context(RunDirectoryManager(run_dir))
    ...     records_fp = stack.enter_context(open_records_file(run_dir / "records.jsonl"))
    ...     # Work with files - cleanup is automatic
"""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import IO, Any, Iterator

import yaml


@contextmanager
def open_records_file(
    path: Path,
    mode: str = "x",
    encoding: str = "utf-8",
) -> Iterator[IO[str]]:
    """Open a records file with guaranteed cleanup.

    Args:
        path: Path to the records file.
        mode: File open mode. Default "x" for exclusive creation.
        encoding: File encoding. Default "utf-8".

    Yields:
        File handle for writing records.

    Raises:
        FileExistsError: If mode is "x" and file already exists.

    Example:
        >>> with open_records_file(Path("records.jsonl")) as fp:
        ...     fp.write('{"key": "value"}\\n')
    """
    fp = open(path, mode, encoding=encoding)
    try:
        yield fp
    finally:
        try:
            fp.flush()
            fp.close()
        except Exception:
            pass


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to a file atomically.

    Writes to a temporary file first, then atomically replaces the target.
    This prevents partial writes from corrupting the file.

    Args:
        path: Target file path.
        text: Text content to write.

    Example:
        >>> atomic_write_text(Path("config.json"), '{"key": "value"}')
    """
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)


def atomic_write_yaml(path: Path, data: Any, serializer: Any = None) -> None:
    """Write data to a YAML file atomically.

    Args:
        path: Target file path.
        data: Data to serialize as YAML.
        serializer: Optional function to serialize complex objects.
            If provided, will be applied to data before YAML dump.

    Example:
        >>> atomic_write_yaml(Path("config.yaml"), {"key": "value"})
    """
    if serializer is not None:
        data = serializer(data)
    content = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    atomic_write_text(path, content)


def ensure_run_sentinel(run_dir: Path) -> None:
    """Create a sentinel file marking a directory as an insideLLMs run.

    This marker is used to make --overwrite safer by only deleting
    directories that look like legitimate run directories.

    Args:
        run_dir: The run directory path.
    """
    marker = run_dir / ".insidellms_run"
    if not marker.exists():
        try:
            marker.write_text("insideLLMs run directory\n", encoding="utf-8")
        except Exception:
            pass


@contextmanager
def managed_run_directory(
    run_dir: Path,
    *,
    create: bool = True,
    sentinel: bool = True,
) -> Iterator[Path]:
    """Context manager for a run directory with optional setup.

    Args:
        run_dir: Path to the run directory.
        create: If True, create the directory if it doesn't exist.
        sentinel: If True, create the run sentinel file.

    Yields:
        The run directory path.

    Example:
        >>> with managed_run_directory(Path("runs/run_001")) as run_dir:
        ...     # Work with run_dir
        ...     pass
    """
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    if sentinel:
        ensure_run_sentinel(run_dir)
    yield run_dir


__all__ = [
    "open_records_file",
    "atomic_write_text",
    "atomic_write_yaml",
    "ensure_run_sentinel",
    "managed_run_directory",
    "ExitStack",
]
