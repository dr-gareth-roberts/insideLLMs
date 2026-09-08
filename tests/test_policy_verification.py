"""Strict policy tests; mocked cosign results are not cryptographic proof."""

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from insideLLMs.policy.verification import VerificationPolicy, verify_policy
from insideLLMs.runtime import _ultimate


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text('{"run_completed": true}\n')
    (run / "records.jsonl").write_text('{"output": "ok"}\n')
    _ultimate.run_ultimate_post_artifact(run)
    (run / "signing").mkdir()
    for path in (run / "attestations").glob("*.dsse.json"):
        (run / "signing" / f"{path.stem}.sigstore.bundle.json").write_text("{}")
    root = tmp_path / "trusted-root.json"
    root.write_text("{}")
    policy = VerificationPolicy("builder@example.com", "https://issuer.example", root)
    verifier = Mock(return_value=True)
    monkeypatch.setattr("insideLLMs.policy.verification.verify_bundle", verifier)
    return run, policy, verifier


def test_bound_evidence_passes_and_forwards_exact_trust(evidence):
    run, policy, verifier = evidence
    assert verify_policy(run, policy)["passed"]
    assert verifier.call_count == 8
    assert verifier.call_args.kwargs["identity_constraints"] == policy.identity
    assert verifier.call_args.kwargs["oidc_issuer"] == policy.oidc_issuer


@pytest.mark.parametrize(
    "target",
    [
        "attestations/00.source.dsse.json",
        "signing/00.source.dsse.sigstore.bundle.json",
        "manifest.json",
        "records.jsonl",
    ],
)
def test_missing_required_evidence_fails(evidence, target):
    run, policy, _ = evidence
    (run / target).unlink()
    assert not verify_policy(run, policy)["passed"]


@pytest.mark.parametrize(
    "target", ["manifest.json", "records.jsonl", "attestations/04.execution.dsse.json"]
)
def test_tampering_fails(evidence, target):
    run, policy, _ = evidence
    (run / target).write_text('{"tampered": true}\n')
    assert not verify_policy(run, policy)["passed"]


@pytest.mark.parametrize(
    "error", [False, FileNotFoundError("cosign"), subprocess.TimeoutExpired("cosign", 30)]
)
def test_verifier_rejection_or_unavailability_fails(evidence, error):
    run, policy, verifier = evidence
    if error is False:
        verifier.return_value = False
    else:
        verifier.side_effect = error
    assert not verify_policy(run, policy)["passed"]


def test_scitt_authenticity_is_unsupported(evidence):
    run, policy, _ = evidence
    strict = VerificationPolicy(
        policy.identity, policy.oidc_issuer, policy.trusted_root, require_scitt=True
    )
    assert not verify_policy(run, strict)["passed"]


def test_policy_failure_blocks_automatic_publication(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "records.jsonl").write_text("{}\n")
    monkeypatch.setattr(
        _ultimate, "run_policy", lambda _: {"passed": False, "reasons": ["denied"], "checks": {}}
    )
    publisher = Mock()
    monkeypatch.setattr(_ultimate, "push_run_oci", publisher)
    with pytest.raises(RuntimeError, match="policy"):
        _ultimate.run_ultimate_post_artifact(tmp_path, publish_oci_ref="example.invalid/test")
    assert publisher.call_count == 0


@pytest.mark.parametrize("field", ["identity", "oidc_issuer"])
def test_wrong_identity_or_issuer_is_rejected_by_verifier(evidence, monkeypatch, field):
    from insideLLMs.signing import cosign

    run, policy, _ = evidence
    monkeypatch.setattr("insideLLMs.policy.verification.verify_bundle", cosign.verify_bundle)
    monkeypatch.setattr(cosign, "_cosign_path", lambda: Path("/test/cosign"))

    def verify_command(cmd, **kwargs):
        assert cmd[cmd.index("--trusted-root") + 1]
        identity = cmd[cmd.index("--certificate-identity") + 1]
        issuer = cmd[cmd.index("--certificate-oidc-issuer") + 1]
        accepted = identity == policy.identity and issuer == policy.oidc_issuer
        return subprocess.CompletedProcess(cmd, 0 if accepted else 1)

    monkeypatch.setattr(cosign.subprocess, "run", verify_command)
    values = {"identity": policy.identity, "oidc_issuer": policy.oidc_issuer}
    values[field] = "wrong@example.com"
    wrong = VerificationPolicy(**values, trusted_root=policy.trusted_root)
    assert not verify_policy(run, wrong)["passed"]


def test_legacy_execution_without_records_count_fails(evidence):
    run, policy, _ = evidence
    path = run / "attestations" / "04.execution.dsse.json"
    envelope = json.loads(path.read_text())
    statement = json.loads(base64.b64decode(envelope["payload"]))
    del statement["predicate"]["records_count"]
    envelope["payload"] = base64.b64encode(json.dumps(statement).encode()).decode()
    path.write_text(json.dumps(envelope))
    assert not verify_policy(run, policy)["passed"]


def test_verification_does_not_rewrite_signed_inputs(evidence):
    run, policy, _ = evidence
    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    assert verify_policy(run, policy)["passed"]
    assert before == {path: path.read_bytes() for path in before}


def test_records_whitespace_tampering_breaks_exact_byte_commitment(evidence):
    run, policy, _ = evidence
    path = run / "records.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    assert not verify_policy(run, policy)["passed"]


def test_structurally_valid_envelope_tampering_reaches_crypto_rejection(evidence):
    run, policy, verifier = evidence
    path = run / "attestations" / "00.source.dsse.json"
    original = path.read_bytes()
    path.write_bytes(original + b"\n")

    def reject_changed_bytes(blob, bundle, **kwargs):
        return blob.name != path.name or blob.read_bytes() == original

    verifier.side_effect = reject_changed_bytes
    verdict = verify_policy(run, policy)
    assert not verdict["passed"]
    assert verdict["checks"]["00.source"] == "failed"


def test_duplicate_envelope_fields_are_rejected(evidence):
    run, policy, _ = evidence
    path = run / "attestations" / "00.source.dsse.json"
    original = path.read_text()
    path.write_text('{"payload": "duplicate",' + original[1:])
    assert not verify_policy(run, policy)["passed"]


def test_missing_trusted_root_fails(evidence):
    run, policy, verifier = evidence
    policy.trusted_root.unlink()
    assert not verify_policy(run, policy)["passed"]
    verifier.assert_not_called()


def test_failed_strict_policy_never_publishes(evidence, monkeypatch):
    from insideLLMs.policy import publish_verified_run

    run, policy, verifier = evidence
    verifier.return_value = False
    publisher = Mock()
    monkeypatch.setattr("insideLLMs.publish.oras.push_run_oci", publisher)
    with pytest.raises(RuntimeError, match="policy"):
        publish_verified_run(run, "example.invalid/test", policy)
    assert publisher.call_count == 0


def test_successful_strict_publication_uses_verified_snapshot(evidence, monkeypatch):
    from insideLLMs.policy import publish_verified_run

    run, policy, _ = evidence
    expected = (run / "records.jsonl").read_bytes()
    published_snapshots = []

    def publish(snapshot, reference):
        published_snapshots.append(snapshot)
        (run / "records.jsonl").write_text("tampered")
        assert snapshot != run
        assert (snapshot / "records.jsonl").read_bytes() == expected
        return "published"

    monkeypatch.setattr("insideLLMs.publish.oras.push_run_oci", publish)
    assert publish_verified_run(run, "example.invalid/test", policy) == "published"
    assert published_snapshots and not published_snapshots[0].parent.exists()


@pytest.mark.parametrize("kind", ["file", "directory", "dangling", "fifo"])
def test_strict_publication_rejects_links_and_special_files(evidence, monkeypatch, kind):
    from insideLLMs.policy import publish_verified_run

    run, policy, _ = evidence
    external = run.parent / "external-secret"
    external.write_text("never read this secret")
    external_directory = run.parent / "external-directory"
    external_directory.mkdir()
    (external_directory / "secret").write_text("never read this either")
    link = run / "unrelated"
    if kind == "fifo":
        os.mkfifo(link)
    else:
        target = {
            "file": external,
            "directory": external_directory,
            "dangling": run.parent / "missing",
        }[kind]
        link.symlink_to(target, target_is_directory=kind == "directory")
    publisher = Mock()
    monkeypatch.setattr("insideLLMs.publish.oras.push_run_oci", publisher)
    with pytest.raises((OSError, ValueError)):
        publish_verified_run(run, "example.invalid/test", policy)
    assert publisher.call_count == 0


def test_strict_verification_rejects_symlinked_evidence(evidence):
    run, policy, _ = evidence
    target = run.parent / "external-records"
    records = run / "records.jsonl"
    records.rename(target)
    records.symlink_to(target)
    assert not verify_policy(run, policy)["passed"]


def test_publication_rejects_file_swapped_to_symlink_before_open(evidence, monkeypatch):
    from insideLLMs.policy import verification

    run, policy, _ = evidence
    extra = run / "unrelated"
    extra.write_text("allowed initial file")
    external = run.parent / "secret"
    external.write_text("must never be read")
    original_open = os.open

    def swap_before_open(name, flags, *args, **kwargs):
        if name == "unrelated":
            extra.unlink()
            extra.symlink_to(external)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_before_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {swap_before_open})
    publisher = Mock()
    monkeypatch.setattr("insideLLMs.publish.oras.push_run_oci", publisher)
    with pytest.raises(OSError):
        verification.publish_verified_run(run, "example.invalid/test", policy)
    assert publisher.call_count == 0


def test_cli_strict_verification_exit_status(evidence, capsys):
    from insideLLMs.cli import main

    run, policy, verifier = evidence
    argv = [
        "verify-policy",
        str(run),
        "--identity",
        policy.identity,
        "--oidc-issuer",
        policy.oidc_issuer,
        "--trusted-root",
        str(policy.trusted_root),
    ]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["assurance"] == "cryptographic"
    verifier.return_value = False
    assert main(argv) == 1


@pytest.mark.skipif(
    not shutil.which("cosign") or not os.environ.get("INSIDELLMS_COSIGN_TEST_FIXTURE"),
    reason="Real cosign verification requires cosign and an explicit offline signed fixture",
)
def test_external_cosign_signed_fixture():
    """Optional real gate: fixture root has run/, trusted-root.json and policy.json."""
    fixture = Path(os.environ["INSIDELLMS_COSIGN_TEST_FIXTURE"])
    expected = json.loads((fixture / "policy.json").read_text())
    policy = VerificationPolicy(
        expected["identity"], expected["oidc_issuer"], fixture / "trusted-root.json"
    )
    assert verify_policy(fixture / "run", policy)["passed"]
    wrong = VerificationPolicy("wrong@example.invalid", policy.oidc_issuer, policy.trusted_root)
    assert not verify_policy(fixture / "run", wrong)["passed"]
