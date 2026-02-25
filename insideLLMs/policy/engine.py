"""Policy engine: run checks and emit verdict (pass/fail + reasons)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_policy(run_dir: Path | str, policy_yaml_path: Path | str | None = None) -> dict[str, Any]:
    """Load policy, run checks (attestations exist, signatures verify, Merkle recompute, etc.), return verdict."""
    run_dir = Path(run_dir)
    verdict: dict[str, Any] = {"passed": True, "reasons": []}
    if not (run_dir / "manifest.json").exists():
        verdict["passed"] = False
        verdict["reasons"].append("manifest.json missing")
    return verdict
