"""ORAS/OCI: package run_dir as OCI artifact, push/pull (stub)."""

from __future__ import annotations

from pathlib import Path


def push_run_oci(run_dir: Path | str, ref: str) -> None:
    """Package run directory and push to OCI registry (stub)."""
    raise NotImplementedError("ORAS push not configured")


def pull_run_oci(ref: str, output_dir: Path | str, *, verify: bool = False, policy_path: str | None = None) -> Path:
    """Pull run artifact from OCI (stub)."""
    raise NotImplementedError("ORAS pull not configured")
