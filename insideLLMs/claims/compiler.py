"""Claims compiler: read claims.yaml, compute effects/CIs, emit claims.json + verification.json (stub)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def compile_claims(claims_yaml_path: Path | str, run_dir: Path | str) -> dict[str, Any]:
    """Read claims.yaml, compute using statistics, emit claims.json and verification.json (stub)."""
    return {"status": "stub", "verification": {}}
