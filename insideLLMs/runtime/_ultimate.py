"""Ultimate mode orchestration: integrity roots and attestations after a run.

Called by the runner after manifest (and optionally report) are written.
Computes Merkle roots over records, receipts, dataset, promptset; writes
integrity/*.merkle.json; builds and writes attestations 00–09 to attestations/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

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
from insideLLMs.crypto import digest_obj, merkle_root_from_jsonl, run_bundle_id
from insideLLMs.policy.engine import run_policy
from insideLLMs.publish.oras import push_run_oci
from insideLLMs.runtime._artifact_utils import _atomic_write_text
from insideLLMs.transparency.scitt_client import submit_statement


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
    """Compute integrity roots, write attestations, and optionally run_bundle_id.

    Creates run_dir/integrity/*.merkle.json, run_dir/attestations/NN.*.dsse.json.
    If manifest_digest and roots are provided, writes integrity/bundle_id.txt.
    """
    run_dir = Path(run_dir)
    integrity_dir = run_dir / "integrity"
    attestations_dir = run_dir / "attestations"
    receipts_dir = run_dir / "receipts"
    integrity_dir.mkdir(parents=True, exist_ok=True)
    attestations_dir.mkdir(parents=True, exist_ok=True)

    # 1) Compute Merkle roots if not provided
    records_path = run_dir / "records.jsonl"
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
        merkle_result = merkle_root_from_jsonl(receipts_path)
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

    roots: dict[str, str] = {}
    if records_merkle_root:
        roots["records_merkle_root"] = records_merkle_root
    if receipts_merkle_root:
        roots["receipts_merkle_root"] = receipts_merkle_root
    if dataset_merkle_root:
        roots["dataset_merkle_root"] = dataset_merkle_root
    if promptset_merkle_root:
        roots["promptset_merkle_root"] = promptset_merkle_root

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
                records_merkle_root=records_merkle_root,
                receipts_merkle_root=receipts_merkle_root,
                runner_config_snapshot=config_snapshot,
            ),
        ),
        ("05.scoring", build_attestation_05_scoring(subject_manifest)),
        ("06.report", build_attestation_06_report(subject_manifest)),
        ("07.claims", build_attestation_07_claims(subject_manifest)),
    ]

    attestation_digests: list[str] = []
    for name, statement in steps_00_07:
        envelope = build_dsse_envelope(statement)
        digest = digest_obj(statement, purpose="attestation")["digest"]
        attestation_digests.append(digest)
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

    # Optionally push to OCI and get metadata for attestation 09
    oci_ref: Optional[str] = None
    oci_digest: Optional[str] = None
    if publish_oci_ref:
        push_result = push_run_oci(run_dir, publish_oci_ref)
        oci_ref = push_result.ref
        oci_digest = push_result.digest

    # Build and write attestations 08 (policy) and 09 (publish)
    statement_08 = build_attestation_08_policy(
        subject_manifest,
        policy_file_digest=policy_file_digest,
        verdict_digest=verdict_digest,
        passed=verdict["passed"],
        reasons=verdict["reasons"],
    )
    statement_09 = build_attestation_09_publish(
        subject_manifest, oci_ref=oci_ref, oci_digest=oci_digest
    )

    for name, statement in [("08.policy", statement_08), ("09.publish", statement_09)]:
        envelope = build_dsse_envelope(statement)
        digest = digest_obj(statement, purpose="attestation")["digest"]
        attestation_digests.append(digest)
        _atomic_write_text(
            attestations_dir / f"{name}.dsse.json",
            json.dumps(envelope, sort_keys=True, separators=(",", ":"), indent=2) + "\n",
        )

    # 3) Run bundle ID
    if manifest_digest_val and roots:
        bundle_id = run_bundle_id(manifest_digest_val, roots, attestation_digests)
        _atomic_write_text(integrity_dir / "bundle_id.txt", bundle_id + "\n")
