"""Private regular-file snapshots using directory-relative, no-follow I/O.

Detected mutations fail closed; this is not a transactional filesystem snapshot.
"""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

_CHUNK_SIZE = 1024 * 1024
_REJECT_MESSAGE = (
    "Publication rejects symlinks and special files; remove or materialize aliases "
    "(including results.jsonl) in a separate export copy, preserving signed originals"
)


def _open_directory(path: str | Path, *, parent: int | None = None) -> int:
    if (
        not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"))
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
        or os.listdir not in os.supports_fd
    ):
        raise OSError("Evidence snapshots require no-follow directory-relative I/O")
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    except OSError as exc:
        raise OSError(f"{_REJECT_MESSAGE}: cannot open directory {path}") from exc


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)


@contextmanager
def _regular_file(name: str, parent: int) -> Iterator[BinaryIO]:
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{_REJECT_MESSAGE}: {name}")
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    except OSError as exc:
        raise OSError(f"{_REJECT_MESSAGE}: cannot open file {name}") from exc
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or _identity(before) != _identity(opened):
            raise ValueError(f"Artifact changed before copying: {name}")
        yield stream
        after = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (
            _identity(before) != _identity(os.fstat(stream.fileno()))
            or not stat.S_ISREG(after.st_mode)
            or _identity(before) != _identity(after)
        ):
            raise ValueError(f"Artifact changed while copying: {name}")


def _read_regular(name: str, parent: int) -> bytes:
    with _regular_file(name, parent) as stream:
        return stream.read()


def _copy_regular_tree(source: int, destination: Path) -> None:
    destination.mkdir(mode=0o700)
    for name in sorted(os.listdir(source)):
        before = os.stat(name, dir_fd=source, follow_symlinks=False)
        if stat.S_ISDIR(before.st_mode):
            child = _open_directory(name, parent=source)
            try:
                if _identity(before) != _identity(os.fstat(child)):
                    raise ValueError(f"Artifact directory changed before copying: {name}")
                _copy_regular_tree(child, destination / name)
            finally:
                os.close(child)
        elif stat.S_ISREG(before.st_mode):
            with _regular_file(name, source) as stream, (destination / name).open("xb") as out:
                while chunk := stream.read(_CHUNK_SIZE):
                    out.write(chunk)
        else:
            raise ValueError(f"{_REJECT_MESSAGE}: {name}")


@contextmanager
def regular_tree_snapshot(source: Path | str) -> Iterator[Path]:
    """Yield a private copy, kept alive through consumer success or failure."""
    with tempfile.TemporaryDirectory(prefix="insidellms-snapshot-") as root:
        snapshot = Path(root) / "run"
        descriptor = _open_directory(source)
        try:
            _copy_regular_tree(descriptor, snapshot)
        finally:
            os.close(descriptor)
        yield snapshot
