"""Encrypt JSONL at rest (stub)."""

from __future__ import annotations

from pathlib import Path


def encrypt_jsonl(path: Path | str, *, key: bytes | None = None) -> None:
    """Encrypt a JSONL file in place (stub)."""
    pass


def decrypt_jsonl(path: Path | str, *, key: bytes | None = None) -> None:
    """Decrypt a JSONL file (stub)."""
    pass
