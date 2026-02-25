"""ORAS/OCI: package run_dir as OCI artifact, push/pull."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from oras import client as oras_client

    ORAS_AVAILABLE = True
except ImportError:
    ORAS_AVAILABLE = False
    oras_client = None  # type: ignore[assignment]


@dataclass
class PushResult:
    """Result of pushing a run bundle to OCI."""

    ref: str
    digest: str | None
    """OCI digest (e.g. sha256:...) if available from registry response."""


@dataclass
class PullResult:
    """Result of pulling a run bundle from OCI."""

    ref: str
    path: Path
    digest: str | None
    """OCI digest if available from manifest."""


def _require_oras() -> None:
    """Raise if ORAS is not available. Call before any push/pull."""
    if not ORAS_AVAILABLE or oras_client is None:
        raise RuntimeError(
            "oras library is required for OCI push/pull. Install with: pip install oras"
        )


def push_run_oci(run_dir: Path | str, ref: str) -> PushResult:
    """Package run directory and push to OCI registry.

    Returns
    -------
    PushResult
        ref, digest (digest may be None if registry does not return it).

    Raises
    ------
    RuntimeError
        If oras library is not installed.
    ValueError
        If run_dir does not exist or is empty.
    """
    _require_oras()
    run_path = Path(run_dir)
    if not run_path.exists() or not run_path.is_dir():
        raise ValueError(f"Run directory {run_dir} does not exist or is not a directory")

    oci_client = oras_client.OciClient()
    files_to_push = []
    for root, _, files in os.walk(run_path):
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(run_path)
            files_to_push.append(f"{file_path}:{rel_path}")

    if not files_to_push:
        raise ValueError(f"Run directory {run_dir} is empty")

    oci_client.push(target=ref, files=files_to_push)

    # oras-py push returns Response; digest may be in Location header.
    # For now return ref; digest can be populated by caller if needed.
    return PushResult(ref=ref, digest=None)


def pull_run_oci(
    ref: str,
    output_dir: Path | str,
    *,
    verify: bool = False,
    policy_path: str | None = None,
) -> PullResult:
    """Pull run artifact from OCI.

    Returns
    -------
    PullResult
        ref, path (pulled directory), digest (if available).

    Raises
    ------
    RuntimeError
        If oras library is not installed.
    ValueError
        If verify=True and required artifacts (manifest.json, records.jsonl) are missing.
    """
    _require_oras()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    oci_client = oras_client.OciClient()
    oci_client.pull(target=ref, outdir=str(out_path))

    if verify:
        _verify_pulled_run(out_path, policy_path=policy_path)

    return PullResult(ref=ref, path=out_path, digest=None)


def _verify_pulled_run(out_path: Path, *, policy_path: str | None = None) -> None:
    """Verify pulled run has expected layout and optionally policy file.

    Raises
    ------
    ValueError
        If manifest.json or records.jsonl is missing, or policy_path given but missing.
    """
    manifest = out_path / "manifest.json"
    records = out_path / "records.jsonl"
    if not manifest.exists():
        raise ValueError(f"Verification failed: manifest.json missing in {out_path}")
    if not records.exists():
        raise ValueError(f"Verification failed: records.jsonl missing in {out_path}")
    if policy_path is not None:
        policy_file = out_path / policy_path
        if not policy_file.exists():
            raise ValueError(
                f"Verification failed: policy file {policy_path} missing in {out_path}"
            )
