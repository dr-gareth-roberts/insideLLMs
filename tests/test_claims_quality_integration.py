"""Opted-in metric acceptance gates Ultimate publication; fake pushes prove no authenticity."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from insideLLMs.attestations import parse_dsse_envelope
from insideLLMs.cli import main
from insideLLMs.config_types import RunConfig
from insideLLMs.contrib.claims.compiler import compile_claims
from insideLLMs.models import DummyModel
from insideLLMs.probes import LogicProbe
from insideLLMs.publish.oras import PushResult
from insideLLMs.runtime import _ultimate
from insideLLMs.runtime.runner import AsyncProbeRunner, ProbeRunner


def _execution(path: Path, *, accuracy: float = 0.95) -> Path:
    """Prepare an unsealed execution and compile its threshold evidence."""
    path.mkdir()
    (path / "manifest.json").write_text('{"run_id":"claims-test","run_completed":true}\n')
    (path / "records.jsonl").write_text('{"input":"q","output":"a"}\n')
    (path / "summary.json").write_text(
        json.dumps({"metrics": {"accuracy": {"mean": accuracy}}}), encoding="utf-8"
    )
    claims = {
        "claims": [{"id": "accuracy", "metric": "accuracy", "operator": ">=", "threshold": 0.9}]
    }
    (path / "claims.yaml").write_text(json.dumps(claims), encoding="utf-8")
    compile_claims(path / "claims.yaml", path)
    return path


def _tree_bytes(path: Path) -> dict[str, bytes]:
    """Capture exact payload bytes independently of production identity helpers."""
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def _verdict(path: Path) -> dict[str, Any]:
    """Read the persisted policy verdict."""
    return json.loads((path / "policy/verdict.json").read_bytes())


@pytest.mark.parametrize("evidence", ["failed", "missing", "malformed"])
@pytest.mark.parametrize("publish", [False, True])
def test_quality_rejection_preserves_evidence_and_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence: str, publish: bool
) -> None:
    """The requested gate rejects even when no OCI push was requested."""
    run_dir = _execution(tmp_path / "run", accuracy=0.8 if evidence == "failed" else 0.95)
    if evidence == "missing":
        (run_dir / "verification.json").unlink()
    elif evidence == "malformed":
        (run_dir / "verification.json").write_text('{"accuracy":{"passed":"true"}}')
    original = _tree_bytes(run_dir)
    push = Mock()
    monkeypatch.setattr(_ultimate, "push_run_oci", push)
    with pytest.raises(RuntimeError):
        _ultimate.run_ultimate_post_artifact(
            run_dir,
            require_claims_pass=True,
            publish_oci_ref="registry.example/claims:test" if publish else None,
        )
    push.assert_not_called()
    verdict = _verdict(run_dir)
    assert verdict["passed"] is False
    assert verdict["structural_passed"] is True
    assert verdict["claims_quality"]["passed"] is False
    policy_envelope = json.loads((run_dir / "attestations/08.policy.dsse.json").read_bytes())
    assert parse_dsse_envelope(policy_envelope)[0]["predicate"]["passed"] is False
    assert (run_dir / "integrity/bundle_identity.json").is_file()
    assert not (run_dir / "attestations/09.publish.dsse.json").exists()
    assert all((run_dir / name).read_bytes() == raw for name, raw in original.items())


def test_accepted_evidence_is_in_the_finalized_publication_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact accepted claim bytes are committed before the one requested push."""
    run_dir = _execution(tmp_path / "run")
    expected = {
        name: (run_dir / name).read_bytes() for name in ("claims.json", "verification.json")
    }
    captured = []

    def push(payload_dir: Path, ref: str) -> PushResult:
        assert payload_dir != run_dir
        payload = _tree_bytes(payload_dir)
        assert all(payload[name] == raw for name, raw in expected.items())
        assert all((run_dir / name).read_bytes() == raw for name, raw in payload.items())
        verdict = json.loads(payload["policy/verdict.json"])
        assert verdict["passed"] is True
        assert verdict["structural_passed"] is True
        assert verdict["claims_quality"]["passed"] is True
        assert verdict["signature_authenticity"] == "not_checked"
        descriptor = json.loads(payload["integrity/bundle_identity.json"])
        committed = {entry["path"]: entry["sha256"] for entry in descriptor["files"]}
        assert set(expected) <= set(committed)
        for name, digest in committed.items():
            assert digest == hashlib.sha256(payload[name]).hexdigest()
        canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        expected_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert payload["integrity/bundle_id.txt"] == (expected_id + "\n").encode("utf-8")
        assert "attestations/08.policy.dsse.json" in payload
        assert "attestations/09.publish.dsse.json" not in payload
        captured.append(payload)
        return PushResult(ref=ref, digest="sha256:fake-registry-receipt")

    monkeypatch.setattr(_ultimate, "push_run_oci", push)
    _ultimate.run_ultimate_post_artifact(
        run_dir, require_claims_pass=True, publish_oci_ref="registry.example/claims:test"
    )
    assert len(captured) == 1
    envelope = json.loads((run_dir / "attestations/09.publish.dsse.json").read_bytes())
    assert parse_dsse_envelope(envelope)[0]["predicate"]["outcome"] == "published"


def test_post_gate_source_mutation_does_not_change_admitted_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Push receives the accepted private snapshot even if the original later changes."""
    run_dir = _execution(tmp_path / "run")
    accepted = (run_dir / "verification.json").read_bytes()
    captured = []

    def push(payload_dir: Path, ref: str) -> PushResult:
        (run_dir / "verification.json").write_text('{"accuracy":{"passed":false}}')
        assert (payload_dir / "verification.json").read_bytes() == accepted
        assert _verdict(payload_dir)["claims_quality"]["passed"] is True
        captured.append(payload_dir)
        return PushResult(ref=ref, digest="sha256:fake-registry-receipt")

    monkeypatch.setattr(_ultimate, "push_run_oci", push)
    _ultimate.run_ultimate_post_artifact(
        run_dir, require_claims_pass=True, publish_oci_ref="registry.example/claims:test"
    )
    assert len(captured) == 1


@pytest.mark.parametrize("evidence", ["failed", "malformed"])
def test_disabled_gate_preserves_payload_selection_and_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence: str
) -> None:
    """Ignored claim files do not alter the original structural verdict or bundle ID."""
    captured = []

    def push(payload_dir: Path, ref: str) -> PushResult:
        captured.append(_tree_bytes(payload_dir))
        return PushResult(ref=ref, digest="sha256:fake-registry-receipt")

    monkeypatch.setattr(_ultimate, "push_run_oci", push)
    for name, gate_kwargs in (("without", {}), ("with", {"require_claims_pass": False})):
        run_dir = _execution(tmp_path / name)
        if name == "without":
            (run_dir / "claims.json").unlink()
            (run_dir / "verification.json").unlink()
        elif evidence == "failed":
            value = json.loads((run_dir / "verification.json").read_bytes())
            value["accuracy"]["passed"] = False
            (run_dir / "verification.json").write_text(json.dumps(value))
        else:
            (run_dir / "claims.json").write_bytes(b"\xff")
            (run_dir / "verification.json").write_bytes(b"{")
        _ultimate.run_ultimate_post_artifact(
            run_dir, publish_oci_ref="registry.example/claims:test", **gate_kwargs
        )
    assert len(captured) == 2
    assert captured[0] == captured[1]
    assert "claims.json" not in captured[0]
    assert "verification.json" not in captured[0]
    verdict = json.loads(captured[0]["policy/verdict.json"])
    assert verdict["passed"] is True
    assert "claims_quality" not in verdict


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sync_and_async_runners_enforce_the_configured_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    """Missing claims must block both real direct runner paths before publication."""
    runner_type = AsyncProbeRunner if asynchronous else ProbeRunner
    runner = runner_type(DummyModel(canned_response="4"), LogicProbe())
    run_dir = tmp_path / "run"
    config = RunConfig(
        run_mode="ultimate",
        run_dir=run_dir,
        require_claims_pass=True,
        publish_oci_ref="registry.example/claims:test",
    )
    push = Mock()
    monkeypatch.setattr(_ultimate, "push_run_oci", push)
    with pytest.raises(RuntimeError):
        result = runner.run([{"question": "2+2?", "reference_answer": "4"}], config=config)
        if asynchronous:
            asyncio.run(result)
    push.assert_not_called()
    verdict = _verdict(run_dir)
    assert verdict["structural_passed"] is True
    assert verdict["claims_quality"]["passed"] is False
    assert json.loads((run_dir / "manifest.json").read_bytes())["run_completed"] is True


@pytest.mark.parametrize("value", [None, "true", "false", 1, 0])
def test_run_config_rejects_ambiguous_claims_gate_values(value: Any) -> None:
    """Configuration cannot turn a requested gate into a truthiness convention."""
    with pytest.raises((TypeError, ValueError)):
        RunConfig(run_mode="ultimate", require_claims_pass=value)


@pytest.mark.parametrize("settings", [{"run_mode": "default"}, {"emit_run_artifacts": False}])
def test_run_config_rejects_a_gate_that_cannot_run(settings: dict[str, Any]) -> None:
    """The flag must not be silently ignored outside an artefact-producing Ultimate run."""
    options = {"run_mode": "ultimate", "require_claims_pass": True, **settings}
    with pytest.raises(ValueError):
        RunConfig(**options)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_runner_artifact_override_cannot_bypass_required_gate(
    tmp_path: Path, asynchronous: bool
) -> None:
    """Effective legacy kwargs cannot disable artefacts underneath an opted-in gate."""
    runner_type = AsyncProbeRunner if asynchronous else ProbeRunner
    runner = runner_type(DummyModel(canned_response="4"), LogicProbe())
    config = RunConfig(run_mode="ultimate", run_dir=tmp_path / "run", require_claims_pass=True)
    with pytest.raises(ValueError):
        result = runner.run(
            [{"question": "2+2?", "reference_answer": "4"}],
            config=config,
            emit_run_artifacts=False,
        )
        if asynchronous:
            asyncio.run(result)
    assert runner.last_experiment is None
    assert not (tmp_path / "run").exists()


@pytest.mark.contract
@pytest.mark.parametrize(
    ("accuracy", "enabled", "expected"), [(0.95, True, 0), (0.8, True, 1), (0.8, False, 0)]
)
def test_attest_cli_opt_in_controls_metric_acceptance_exit_status(
    tmp_path: Path, accuracy: float, enabled: bool, expected: int
) -> None:
    """The new flag exposes a usable gate while ordinary attest keeps its semantics."""
    run_dir = _execution(tmp_path / "run", accuracy=accuracy)
    args = ["attest", str(run_dir)]
    if enabled:
        args.append("--require-claims-pass")
    assert main(args) == expected
    verdict = _verdict(run_dir)
    if enabled:
        assert verdict["claims_quality"]["passed"] is (expected == 0)
    else:
        assert "claims_quality" not in verdict
        assert verdict["passed"] is True
