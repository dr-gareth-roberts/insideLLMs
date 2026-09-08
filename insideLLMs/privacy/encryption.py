"""Encrypt JSONL at rest."""

from __future__ import annotations

import logging
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

try:
    from cryptography.fernet import Fernet

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


logger = logging.getLogger(__name__)


def _validate_source_metadata(path: Path) -> None:
    try:
        path_metadata = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}") from None
    if stat.S_ISLNK(path_metadata.st_mode):
        raise ValueError("JSONL source must not be a symbolic link")
    if not stat.S_ISREG(path_metadata.st_mode):
        raise ValueError("JSONL source must be a regular file")


def _open_regular_source(path: Path) -> BinaryIO:
    """Open *path* without following a final-component symlink."""
    _validate_source_metadata(path)

    def no_follow_opener(name: str, flags: int) -> int:
        if os.name == "posix" and hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        return os.open(name, flags)

    source = open(path, "rb", opener=no_follow_opener)
    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
        source.close()
        raise ValueError("JSONL source must be a regular file")
    return source


def _transform_jsonl(path: Path, transform: Callable[[bytes], bytes]) -> None:
    """Transform nonblank lines through an exclusively owned sibling stage.

    Simultaneous in-place transforms of the same source require caller
    serialization. Unique staging is not a transactional concurrency guarantee
    against a malicious owner of the parent directory.
    """
    temporary_path: Path | None = None
    descriptor: int | None = None
    try:
        with _open_regular_source(path) as source:
            source_mode = stat.S_IMODE(os.fstat(source.fileno()).st_mode)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            temporary_path = Path(temporary_name)
            output = os.fdopen(descriptor, "wb")
            descriptor = None  # The stream now owns and closes the descriptor.
            with output:
                if hasattr(os, "fchmod"):
                    os.fchmod(output.fileno(), 0o600)
                for line in source:
                    if line.strip():
                        output.write(transform(line.strip()) + b"\n")
                if hasattr(os, "fchmod"):
                    os.fchmod(output.fileno(), source_mode)
                output.flush()
                os.fsync(output.fileno())

        os.replace(temporary_path, path)
        temporary_path = None
    except Exception:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to close owned JSONL staging descriptor (%s)",
                    type(cleanup_error).__name__,
                )
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to clean up owned JSONL staging file (%s)",
                    type(cleanup_error).__name__,
                )
        raise


def encrypt_jsonl(path: Path | str, *, key: bytes | None = None) -> None:
    """Encrypt a JSONL file in place line by line.

    Each line is encrypted separately so the file can still be processed
    line-by-line without loading the entire file into memory.
    """
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography library is required for encryption")

    if not key:
        raise ValueError("Encryption key is required")

    source_path = Path(path)
    _validate_source_metadata(source_path)
    fernet = Fernet(key)
    _transform_jsonl(source_path, fernet.encrypt)


def decrypt_jsonl(path: Path | str, *, key: bytes | None = None) -> None:
    """Decrypt a JSONL file in place line by line."""
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography library is required for decryption")

    if not key:
        raise ValueError("Decryption key is required")

    source_path = Path(path)
    _validate_source_metadata(source_path)
    fernet = Fernet(key)
    _transform_jsonl(source_path, fernet.decrypt)
