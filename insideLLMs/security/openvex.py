"""OpenVEX emission from scan/context (stub)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def emit_openvex(run_dir: Path | str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate OpenVEX document (stub)."""
    return {"@context": "https://openvex.dev/ns", "statements": []}
