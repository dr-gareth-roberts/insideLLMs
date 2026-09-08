"""Caller-owned, fail-closed verification of detached signed run evidence."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from insideLLMs._artifact_snapshot import _open_directory, _read_regular, regular_tree_snapshot
from insideLLMs.attestations.statement import (
    PREDICATE_TYPE_EVAL_STEP,
    PREDICATE_TYPE_SLSA_PROVENANCE,
)
from insideLLMs.crypto import digest_obj
from insideLLMs.signing.cosign import verify_bundle

CORE_STAGES = (
    "00.source",
    "01.env",
    "02.dataset",
    "03.promptset",
    "04.execution",
    "05.scoring",
    "06.report",
    "07.claims",
)


def _read_artifact(run_dir: Path, *parts: str) -> bytes:
    """Open each component relative to an already-open directory, never links."""
    descriptor = _open_directory(run_dir)
    try:
        for component in parts[:-1]:
            child = _open_directory(component, parent=descriptor)
            os.close(descriptor)
            descriptor = child
        return _read_regular(parts[-1], descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class VerificationPolicy:
    """Exact signer constraints and an independently provisioned Sigstore root.

    The run directory never supplies trust configuration. SCITT authenticity
    is unsupported; requesting it always fails closed.
    """

    identity: str
    oidc_issuer: str
    trusted_root: Path
    required_stages: tuple[str, ...] = CORE_STAGES
    require_scitt: bool = False

    def __post_init__(self) -> None:
        for value in (self.identity, self.oidc_issuer):
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError("Exact nonempty identity and OIDC issuer are required")
        stages = tuple(self.required_stages)
        if "04.execution" not in stages or not set(stages).issubset(CORE_STAGES):
            raise ValueError("Required stages must include execution and be core stages")
        if len(stages) != len(set(stages)):
            raise ValueError("Required stages must be unique")
        object.__setattr__(self, "required_stages", stages)
        object.__setattr__(self, "trusted_root", Path(self.trusted_root))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _json_object(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _statement(raw: bytes, stage: str, manifest_digest: str) -> dict[str, Any]:
    envelope = _json_object(raw)
    if envelope.get("payloadType") != "application/vnd.in-toto+json":
        raise ValueError("Unsupported envelope payloadType")
    if not isinstance(envelope.get("signatures"), list):
        raise ValueError("Envelope signatures must be an array")
    statement = _json_object(base64.b64decode(envelope["payload"], validate=True))
    expected_type = (
        PREDICATE_TYPE_SLSA_PROVENANCE if stage == "04.execution" else PREDICATE_TYPE_EVAL_STEP
    )
    if statement.get("_type") != "https://in-toto.io/Statement/v1":
        raise ValueError("Unsupported statement type")
    if statement.get("predicateType") != expected_type:
        raise ValueError("Unexpected predicate type")
    if statement.get("subject") != [
        {"name": "manifest.json", "digest": {"sha256": manifest_digest}}
    ]:
        raise ValueError("Manifest subject commitment mismatch")
    predicate = statement.get("predicate")
    if not isinstance(predicate, dict) or predicate.get("step") != stage.split(".")[1]:
        raise ValueError("Predicate does not identify the required stage")
    return predicate


def _execution_binding(predicate: dict[str, Any], records: bytes, manifest: str) -> None:
    record_count = sum(1 for line in records.splitlines() if line.strip())
    for line in records.splitlines():
        if line.strip():
            _json_object(line)
    expected = [
        {"digest": {"sha256": hashlib.sha256(records).hexdigest()}, "uri": "records.jsonl"},
        {"digest": {"sha256": manifest}, "uri": "manifest.json"},
    ]
    if predicate.get("materials") != expected:
        raise ValueError("Execution manifest/records commitments missing or mismatched")
    count = predicate.get("records_count")
    if type(count) is not int or count != record_count:
        raise ValueError("Execution records count missing or mismatched")


def verify_policy(run_dir: Path | str, policy: VerificationPolicy) -> dict[str, Any]:
    """Verify required stages and exact signed bytes without rewriting originals.

    Results describe authenticity and artifact binding only, not whether the
    claimed execution occurred or its scientific conclusions are correct.
    """
    run_dir = Path(run_dir)
    verdict: dict[str, Any] = {
        "passed": False,
        "assurance": "cryptographic",
        "reasons": [],
        "checks": {},
        "scitt_authenticity": "unsupported",
    }
    checks = verdict["checks"]
    if policy.require_scitt:
        checks["scitt_authenticity"] = "unsupported"
        verdict["reasons"].append("SCITT authenticity verification is unsupported")
    try:
        manifest_bytes = _read_artifact(run_dir, "manifest.json")
        records = _read_artifact(run_dir, "records.jsonl")
        manifest = digest_obj(_json_object(manifest_bytes), purpose="manifest")["digest"]
        root_bytes = policy.trusted_root.read_bytes()
        _json_object(root_bytes)
    except (OSError, ValueError, TypeError) as exc:
        checks["artifacts_and_trust"] = "unavailable"
        verdict["reasons"].append(str(exc))
        return verdict
    checks["artifacts_and_trust"] = "passed"
    # Cosign and structural checks consume the same snapshot, closing the gap
    # between reading a payload and the verifier reopening its filename.
    with tempfile.TemporaryDirectory(prefix="insidellms-verify-") as temporary:
        snapshot = Path(temporary)
        trusted_root = snapshot / "trusted-root.json"
        trusted_root.write_bytes(root_bytes)
        for stage in policy.required_stages:
            try:
                blob_bytes = _read_artifact(run_dir, "attestations", f"{stage}.dsse.json")
                bundle_bytes = _read_artifact(
                    run_dir, "signing", f"{stage}.dsse.sigstore.bundle.json"
                )
                predicate = _statement(blob_bytes, stage, manifest)
                if stage == "04.execution":
                    _execution_binding(predicate, records, manifest)
                blob = snapshot / f"{stage}.dsse.json"
                bundle = snapshot / f"{stage}.bundle.json"
                blob.write_bytes(blob_bytes)
                bundle.write_bytes(bundle_bytes)
                if not verify_bundle(
                    blob,
                    bundle,
                    identity_constraints=policy.identity,
                    oidc_issuer=policy.oidc_issuer,
                    trusted_root=trusted_root,
                ):
                    raise ValueError("Signature, identity, issuer or trust verification failed")
                checks[stage] = "passed"
            except (OSError, subprocess.TimeoutExpired) as exc:
                checks[stage] = "unavailable"
                verdict["reasons"].append(f"{stage}: {exc}")
            except (ValueError, TypeError, KeyError) as exc:
                checks[stage] = "failed"
                verdict["reasons"].append(f"{stage}: {exc}")
    verdict["passed"] = all(value == "passed" for value in checks.values())
    return verdict


def publish_verified_run(run_dir: Path | str, reference: str, policy: VerificationPolicy) -> Any:
    """Publish only a private run snapshot which passed the caller's strict policy.

    Calling this function explicitly requests OCI publication. Verification
    alone never calls it. Symlinks and special files are rejected, including
    the optional legacy results.jsonl alias. No-follow directory-relative reads
    prevent link swapping from causing external files to be copied.
    """
    from insideLLMs.publish.oras import push_run_oci

    with regular_tree_snapshot(run_dir) as snapshot:
        verdict = verify_policy(snapshot, policy)
        if not verdict["passed"]:
            raise RuntimeError(
                "Publication blocked by failed policy: " + "; ".join(verdict["reasons"])
            )
        return push_run_oci(snapshot, reference)
