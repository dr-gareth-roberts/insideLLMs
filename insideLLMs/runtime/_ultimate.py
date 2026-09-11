"""Ultimate mode orchestration: integrity roots and attestations after a run.

Called by the runner after manifest (and optionally report) are written.
Computes Merkle roots over records, receipts, dataset, promptset; writes
integrity/*.merkle.json; builds and writes attestations 00–09 to attestations/.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Optional

from insideLLMs._artifact_snapshot import _open_directory, _regular_file, regular_tree_snapshot
from insideLLMs.attestations import build_dsse_envelope
from insideLLMs.attestations.steps.builders import (
    build_attestation_00_source,
    build_attestation_01_env,
    build_attestation_02_dataset,
    build_attestation_03_promptset,
    build_attestation_04_execution,
    build_attestation_05_scoring,
    build_attestation_06_report,
    build_attestation_07_claims,
    build_attestation_08_policy,
    build_attestation_09_publish,
)
from insideLLMs.crypto import (
    digest_obj,
    merkle_root_from_items,
    merkle_root_from_jsonl,
)
from insideLLMs.crypto.canonical import canonical_json_bytes, payload_identity
from insideLLMs.policy.engine import run_policy
from insideLLMs.publish.oras import push_run_oci
from insideLLMs.runtime._artifact_utils import _atomic_write_text
from insideLLMs.transparency.scitt_client import submit_statement


def _load_normalized_receipts_for_merkle(receipts_path: Path) -> list[dict[str, Any]]:
    """Load receipts JSONL and normalize volatile fields for deterministic Merkle roots."""
    normalized: list[dict[str, Any]] = []
    with open(receipts_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            receipt = json.loads(stripped)
            if isinstance(receipt, dict) and "latency_ms" in receipt:
                # Keep runtime latency in receipts file, but remove volatility from crypto commitments.
                receipt["latency_ms"] = None
            normalized.append(receipt)
    return normalized


def _build_attestations(
    run_dir: Path,
    *,
    manifest_digest: Optional[str] = None,
    records_merkle_root: Optional[str] = None,
    receipts_merkle_root: Optional[str] = None,
    dataset_merkle_root: Optional[str] = None,
    promptset_merkle_root: Optional[str] = None,
    prompt_set: Optional[list[Any]] = None,
    dataset_spec: Optional[dict[str, Any]] = None,
    config_snapshot: Optional[dict[str, Any]] = None,
    insidellms_version: Optional[str] = None,
    scitt_service_url: Optional[str] = None,
) -> None:
    """Build roots, stages 00–08 and structural policy in the caller's private tree."""
    run_dir = Path(run_dir)
    integrity_dir = run_dir / "integrity"
    attestations_dir = run_dir / "attestations"
    receipts_dir = run_dir / "receipts"
    integrity_dir.mkdir(parents=True, exist_ok=True)
    attestations_dir.mkdir(parents=True, exist_ok=True)

    # 1) Compute Merkle roots if not provided
    records_path = run_dir / "records.jsonl"
    records_bytes = records_path.read_bytes() if records_path.exists() else None
    receipts_path = receipts_dir / "calls.jsonl"
    if records_merkle_root is None and records_path.exists():
        merkle_result = merkle_root_from_jsonl(records_path)
        records_merkle_root = merkle_result["root"]
        _atomic_write_text(
            integrity_dir / "records.merkle.json",
            json.dumps(merkle_result, sort_keys=True, separators=(",", ":")) + "\n",
        )
    elif records_merkle_root is not None:
        _atomic_write_text(
            integrity_dir / "records.merkle.json",
            json.dumps(
                {
                    "root": records_merkle_root,
                    "count": None,
                    "algo": "sha256",
                    "canon_version": "canon_v1",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
        )

    if receipts_merkle_root is None and receipts_path.exists():
        normalized_receipts = _load_normalized_receipts_for_merkle(receipts_path)
        merkle_result = merkle_root_from_items(normalized_receipts)
        receipts_merkle_root = merkle_result["root"]
        _atomic_write_text(
            integrity_dir / "receipts.merkle.json",
            json.dumps(merkle_result, sort_keys=True, separators=(",", ":")) + "\n",
        )
    elif receipts_merkle_root is not None:
        _atomic_write_text(
            integrity_dir / "receipts.merkle.json",
            json.dumps(
                {
                    "root": receipts_merkle_root,
                    "count": None,
                    "algo": "sha256",
                    "canon_version": "canon_v1",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
        )

    if dataset_merkle_root is not None:
        _atomic_write_text(
            integrity_dir / "dataset.merkle.json",
            json.dumps(
                {"root": dataset_merkle_root, "algo": "sha256", "canon_version": "canon_v1"},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
        )
    if promptset_merkle_root is not None:
        _atomic_write_text(
            integrity_dir / "promptset.merkle.json",
            json.dumps(
                {"root": promptset_merkle_root, "algo": "sha256", "canon_version": "canon_v1"},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
        )

    # 2) Build attestations (minimal subjects; real impl would add file digests)
    manifest_path = run_dir / "manifest.json"
    manifest_digest_val = manifest_digest
    if manifest_digest_val is None and manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_digest_val = digest_obj(manifest_data, purpose="manifest")["digest"]

    subject_manifest = [{"name": "manifest.json", "digest": {"sha256": manifest_digest_val or ""}}]

    # Build and write attestations 00-07 first (policy checks these)
    steps_00_07 = [
        (
            "00.source",
            build_attestation_00_source(subject_manifest, insidellms_version=insidellms_version),
        ),
        ("01.env", build_attestation_01_env(subject_manifest)),
        (
            "02.dataset",
            build_attestation_02_dataset(
                subject_manifest,
                dataset_merkle_root=dataset_merkle_root or "",
                dataset_id=dataset_spec.get("name") if dataset_spec else None,
                dataset_version=dataset_spec.get("version") if dataset_spec else None,
            ),
        ),
        (
            "03.promptset",
            build_attestation_03_promptset(
                subject_manifest, promptset_merkle_root=promptset_merkle_root
            ),
        ),
        (
            "04.execution",
            build_attestation_04_execution(
                subject_manifest,
                manifest_digest=manifest_digest_val,
                records_digest=(
                    hashlib.sha256(records_bytes).hexdigest() if records_bytes is not None else None
                ),
                records_count=(
                    sum(bool(line.strip()) for line in records_bytes.splitlines())
                    if records_bytes is not None
                    else None
                ),
                records_merkle_root=records_merkle_root,
                receipts_merkle_root=receipts_merkle_root,
                runner_config_snapshot=config_snapshot,
            ),
        ),
        ("05.scoring", build_attestation_05_scoring(subject_manifest)),
        ("06.report", build_attestation_06_report(subject_manifest)),
        ("07.claims", build_attestation_07_claims(subject_manifest)),
    ]

    for name, statement in steps_00_07:
        envelope = build_dsse_envelope(statement)
        _atomic_write_text(
            attestations_dir / f"{name}.dsse.json",
            json.dumps(envelope, sort_keys=True, separators=(",", ":"), indent=2) + "\n",
        )

    # Submit critical attestations (04, 07) to SCITT when configured
    scitt_receipts_dir = receipts_dir / "scitt"
    if scitt_service_url:
        scitt_receipts_dir.mkdir(parents=True, exist_ok=True)
        for att_name in ("04.execution", "07.claims"):
            att_path = attestations_dir / f"{att_name}.dsse.json"
            if att_path.exists():
                envelope = json.loads(att_path.read_text(encoding="utf-8"))
                result = submit_statement(envelope, scitt_service_url)
                receipt_path = scitt_receipts_dir / f"{att_name}.receipt.json"
                _atomic_write_text(
                    receipt_path,
                    json.dumps(result, sort_keys=True, indent=2) + "\n",
                )

    # Run policy and persist verdict
    verdict = run_policy(run_dir)
    policy_dir = run_dir / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = policy_dir / "verdict.json"
    verdict_json = json.dumps(verdict, sort_keys=True, indent=2)
    _atomic_write_text(verdict_path, verdict_json + "\n")
    verdict_digest = digest_obj(verdict, purpose="policy_verdict")["digest"]
    policy_file_digest = digest_obj({"verdict": verdict}, purpose="policy_file")["digest"]

    # Policy is evidence in the payload, so finalize 08 before deriving its identity.
    statement_08 = build_attestation_08_policy(
        subject_manifest,
        policy_file_digest=policy_file_digest,
        verdict_digest=verdict_digest,
        passed=verdict["passed"],
        reasons=verdict["reasons"],
    )
    _atomic_write_text(
        attestations_dir / "08.policy.dsse.json",
        json.dumps(build_dsse_envelope(statement_08), sort_keys=True, indent=2) + "\n",
    )


# Fixed input selection is intentional: directory reuse must not publish stale
# reports, attestations, signatures, transparency or publication receipts.
_EXECUTION_INPUTS = (
    "manifest.json",
    "records.jsonl",
    "summary.json",
    "config.resolved.yaml",
    "receipts/calls.jsonl",
)
_MIGRATION = "Historical evidence conflicts; use a fresh export directory, preserving originals"


class PublicationError(RuntimeError):
    """A publication result, distinct from the completed execution's health."""

    def __init__(self, outcome: str, payload_id: str, receipt_path: Path, detail: str) -> None:
        super().__init__(f"Publication {outcome}: {detail}")
        self.outcome = outcome
        self.payload_id = payload_id
        self.receipt_path = receipt_path


def _destination_parent(root: Path, relative: str, *, create: bool) -> tuple[int, str]:
    """Open a contained destination parent without following directory aliases."""
    parts = Path(relative).parts
    descriptor = _open_directory(root)
    try:
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = _open_directory(part, parent=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def _same_bytes(path: Path, incoming: BinaryIO) -> bool:
    """Compare evidence exactly without loading an entire records file into memory."""
    with path.open("rb") as expected:
        while True:
            chunk = expected.read(1024 * 1024)
            if incoming.read(1024 * 1024) != chunk:
                return False
            if not chunk:
                return True


def _install_payload(payload: Path, source: Path, original: Path) -> None:
    """Check every conflict before adding evidence; existing bytes are never rewritten."""
    files = sorted(path for path in payload.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(payload).as_posix()
        previous = original / relative
        if previous.exists():
            if not previous.is_file():
                raise ValueError(f"{_MIGRATION}: {relative}")
            with previous.open("rb") as incoming:
                if not _same_bytes(path, incoming):
                    raise ValueError(f"{_MIGRATION}: {relative}")
    # Recheck live destinations before any write, using the same contained I/O
    # boundary as source snapshots. Concurrent writers still require caller coordination.
    for path in files:
        relative = path.relative_to(payload).as_posix()
        try:
            parent, name = _destination_parent(source, relative, create=False)
        except OSError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                continue
            raise
        try:
            try:
                with _regular_file(name, parent) as incoming:
                    if not _same_bytes(path, incoming):
                        raise ValueError(f"{_MIGRATION}: {relative}")
            except FileNotFoundError:
                continue
        finally:
            os.close(parent)
    for path in files:
        relative = path.relative_to(payload).as_posix()
        parent, name = _destination_parent(source, relative, create=True)
        try:
            try:
                descriptor = os.open(
                    name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent
                )
            except FileExistsError:
                with _regular_file(name, parent) as incoming:
                    if not _same_bytes(path, incoming):
                        raise ValueError(f"{_MIGRATION}: {relative}")
            else:
                with os.fdopen(descriptor, "wb") as output, path.open("rb") as incoming:
                    shutil.copyfileobj(incoming, output, length=1024 * 1024)
        finally:
            os.close(parent)


def _publish_payload(payload: Path, run_dir: Path, target: Optional[str], bundle_id: str) -> None:
    """Reserve a new receipt, perform at most one push, then record its outcome."""
    relative = "attestations/09.publish.dsse.json"
    if (run_dir / relative).exists():
        if target is None:
            return
        relative = f"publication/attempts/{uuid.uuid4().hex}.09.publish.dsse.json"
    parent, name = _destination_parent(run_dir, relative, create=True)
    try:
        receipt_fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
    finally:
        os.close(parent)
    outcome = "not_requested"
    digest: Optional[str] = None
    failure: Optional[Exception] = None
    with os.fdopen(receipt_fd, "w", encoding="utf-8") as output:
        if target:
            try:
                result = push_run_oci(payload, target)
                digest = result.digest
                outcome = "published" if digest else "unknown"
            except (TimeoutError, subprocess.TimeoutExpired) as exc:
                outcome, failure = "unknown", exc
            except Exception as exc:
                outcome, failure = "failed", exc
        statement = build_attestation_09_publish(
            [{"name": "integrity/bundle_identity.json", "digest": {"sha256": bundle_id}}],
            oci_ref=target,
            oci_digest=digest,
            payload_id=bundle_id,
            outcome=outcome,
        )
        output.write(json.dumps(build_dsse_envelope(statement), sort_keys=True, indent=2) + "\n")
    if outcome in {"failed", "unknown"}:
        detail = str(failure) if failure is not None else "registry returned no digest"
        raise PublicationError(outcome, bundle_id, run_dir / relative, detail) from failure


def run_ultimate_post_artifact(
    run_dir: Path,
    *,
    manifest_digest: Optional[str] = None,
    records_merkle_root: Optional[str] = None,
    receipts_merkle_root: Optional[str] = None,
    dataset_merkle_root: Optional[str] = None,
    promptset_merkle_root: Optional[str] = None,
    prompt_set: Optional[list[Any]] = None,
    dataset_spec: Optional[dict[str, Any]] = None,
    config_snapshot: Optional[dict[str, Any]] = None,
    insidellms_version: Optional[str] = None,
    publish_oci_ref: Optional[str] = None,
    scitt_service_url: Optional[str] = None,
) -> None:
    """Finalize an immutable v2 payload before optional structural-policy publication.

    Existing un-attested executions remain supported, including those without a
    summary. Conflicting historical evidence requires a fresh export directory.
    Failed/unknown remote writes raise PublicationError without rewriting execution
    health. Signed assurance remains the separate publish_verified_run API.
    """
    run_dir = Path(run_dir)
    with (
        regular_tree_snapshot(run_dir) as original,
        tempfile.TemporaryDirectory(prefix="insidellms-payload-") as temporary,
    ):
        payload = Path(temporary)
        for relative in _EXECUTION_INPUTS:
            source = original / relative
            if source.exists():
                if not source.is_file():
                    raise ValueError(f"Execution input must be a regular file: {relative}")
                destination = payload / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        _build_attestations(
            payload,
            manifest_digest=manifest_digest,
            records_merkle_root=records_merkle_root,
            receipts_merkle_root=receipts_merkle_root,
            dataset_merkle_root=dataset_merkle_root,
            promptset_merkle_root=promptset_merkle_root,
            prompt_set=prompt_set,
            dataset_spec=dataset_spec,
            config_snapshot=config_snapshot,
            insidellms_version=insidellms_version,
            scitt_service_url=scitt_service_url,
        )
        descriptor, bundle_id = payload_identity(payload)
        (payload / "integrity/bundle_identity.json").write_bytes(canonical_json_bytes(descriptor))
        (payload / "integrity/bundle_id.txt").write_text(bundle_id + "\n", encoding="utf-8")
        _install_payload(payload, run_dir, original)
        verdict = json.loads((payload / "policy/verdict.json").read_bytes())
        if publish_oci_ref and not verdict["passed"]:
            raise RuntimeError(
                "Publication blocked by failed policy: " + "; ".join(verdict["reasons"])
            )
        _publish_payload(payload, run_dir, publish_oci_ref, bundle_id)
